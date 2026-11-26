"""Brainstorming (docs/BRAINSTORM_NOCHE.md, ideas #37/#38):
Extiende el mecanismo ganador de g_09 (HGB causal por línea @15 → Escalón 7) en dos ejes inéditos:
  1) HGB_CTX_15 (6 cols = 5 líneas + suma): cada modelo intra-línea @15 (`d_gold_15`) deja de ver solo
     a sus 2 jugadores aislados; incorpora contexto de ambos Junglas (agresión temprana, XP/oro@15,
     margen, offrole, tags), Supports y señales de Escalón 6 (MATCHUPS, ROL_WR, KALMAN, TEMPO_OBJ).
  2) HGB_CONV_FIN (3 cols): modelos de árbol a nivel equipo estrictamente forward-chaining sobre
     objetivos post-15 de alto SNR:
       - d_xp_15 / 1000 (ventaja de niveles @15, sin ruido de shutdowns)
       - d_oro_fin / 2000 (diferencia final de oro)
       - conv_post15 = y - sigmoid(alpha * d_gold_15): quién remonta o sostiene tras el min 15
         (captura teamfight, escalado, visión y cierre sin confundir con fuerza @15).
Todo 100% causal (< t del bloque, solo Master+, ~val0) y antisimétrico por espejo exacto.
Evaluado sobre Escalón 7 oficial (V 0,6722), con placebo y chequeo ciego Diamond-V.
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.brainstorm_base import REPO, ajustar_predecir, cargar, comparar, evaluar, extras_referencia, mostrar  # noqa: E402
from historial.escalon9.brainstorm_g_01_foto1_wl import construir_extras  # noqa: E402

LANES = ["BOTTOM", "JUNGLE", "MIDDLE", "TOP", "UTILITY"]
IDX_L = {l: i for i, l in enumerate(LANES)}


def _lane_gold_15(D):
    """Extrae d_gold_15 por línea (NM, 5) en miles de oro desde full_dataset.csv sin alterar splits."""
    cache = REPO / "data" / "analysis" / "brainstorm" / "gemini_lane_g15.npz"
    if cache.exists():
        z = np.load(cache)
        if z["dg15"].shape[0] == D["NM"]:
            return z["dg15"], z["ok"]

    NM = D["NM"]
    pos = pd.Series(np.arange(NM, dtype=np.int32), index=D["Z"]["mids"].astype(str))
    df = pd.read_csv(
        REPO / "data" / "splits" / "train" / "full_dataset.csv",
        usecols=["match_id", "team_id", "lane", "gold_15"],
    )
    df["m"] = df["match_id"].astype(str).map(pos)
    df = df.dropna(subset=["m"]).copy()
    df["m"] = df["m"].astype(np.int32)
    df["l"] = df["lane"].map(IDX_L)
    df = df.dropna(subset=["l"]).copy()
    df["l"] = df["l"].astype(np.int32)
    df["e"] = (df["team_id"] == 200).astype(np.int32)

    G = np.zeros((NM, 2, 5), dtype=np.float32)
    CNT = np.zeros((NM, 2, 5), dtype=np.int8)
    m = df["m"].to_numpy()
    e = df["e"].to_numpy()
    l = df["l"].to_numpy()
    g = df["gold_15"].fillna(0.0).to_numpy(dtype=np.float32)
    np.add.at(G, (m, e, l), g)
    np.add.at(CNT, (m, e, l), 1)
    # Solo filas con 1 jugador por lado, duración >= 15 min y oro @15 válido
    dur_min = D["Z"]["D_dur_min"]
    ok = (CNT[:, 0, :] == 1) & (CNT[:, 1, :] == 1) & (G[:, 0, :] > 0) & (G[:, 1, :] > 0) & (dur_min >= 15.0)[:, None]
    dg15 = ((G[:, 0, :] - G[:, 1, :]) / 1000.0).astype(np.float32)
    dg15[~ok] = np.nan
    np.savez_compressed(cache, dg15=dg15, ok=ok)
    return dg15, ok


def _build_player_compact(D):
    """Tensor compacto (NM, 2, 5, Fp) con señales causales fuertes por jugador."""
    c = {k: i for i, k in enumerate(D["cols"])}
    use_cols = [
        "acum_kills_pm", "v20_kills_pm", "acum_deaths_pm", "v20_deaths_pm",
        "acum_damage_share_pct", "acum_gold_share_pct", "acum_kill_participation_pct",
        "acum_gold_pm", "v20_gold_pm", "acum_cs_pm", "acum_vision_pm",
        "acum_d_gold_10", "v20_d_gold_10", "acum_d_gold_15", "v20_d_gold_15",
        "acum_d_xp_15", "acum_d_cs_15", "wr_acum", "wr_v20",
        "logn_tot", "logn_champ", "estreno", "wr_champ", "share_champ", "share_lane", "offrole",
        "log_horas_desde", "ritmo", "tag_Fighter", "tag_Tank", "tag_Mage",
        "tag_Assassin", "tag_Marksman", "tag_Support", "st_attackrange",
        "perf_cs_10", "perf_gold_10", "perf_xp_10", "perf_damage_10", "perf_early_aggression_10",
        "perf_cs_15", "perf_gold_15", "perf_xp_15", "perf_damage_15",
    ]
    idx = [c[k] for k in use_cols]
    Xj_sub = D["Xj"][..., idx]  # (NM, 2, 5, 44)

    C = np.load(REPO / "data" / "analysis" / "brainstorm" / "margen_grupos_escala_sin_v.npz")
    MC_sum = C["MC"].mean(axis=-1, keepdims=True)  # (NM, 2, 5, 1)
    ML_sum = C["ML"].mean(axis=-1, keepdims=True)
    # Top 4 métricas de margen más directas (gold_10, gold_15, xp_15, kills_pm/cs_15 -> índices 0,1,2,3)
    MC_top = C["MC"][..., :5]
    ML_top = C["ML"][..., :5]
    A = np.load(REPO / "data" / "analysis" / "brainstorm" / "mmr_historico.npz")["A"][..., :2]
    P = np.concatenate([Xj_sub, MC_sum, ML_sum, MC_top, ML_top, A], axis=-1).astype(np.float32)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    return P, use_cols


def _forward_mirror_hgb(X_br, X_rb, y_tgt, valid_tgt, t, master, mask_fiv, tr_cut, params, qs=10):
    """Entrena HGB causal estrictamente forward-chaining (< chunk, solo Master+, fi_v) con simetría espejo."""
    NM = len(y_tgt)
    pred = np.zeros(NM, dtype=np.float32)

    fit_m = master & mask_fiv & (t < tr_cut)
    fit_idx = np.flatnonzero(fit_m)
    ord_fit = fit_idx[np.argsort(t[fit_idx], kind="mergesort")]
    chunks = np.array_split(ord_fit, qs)

    # Bloques 1..qs-1 predicen estrictamente con chunks < k
    hist = chunks[0]
    for k in range(1, qs):
        ch = chunks[k]
        t_lim = t[ch].min()
        ok_h = hist[valid_tgt[hist] & (t[hist] < t_lim)]
        if len(ok_h) >= 500:
            X_tr = np.vstack([X_br[ok_h], X_rb[ok_h]])
            y_tr = np.concatenate([y_tgt[ok_h], -y_tgt[ok_h]])
            mdl = HistGradientBoostingRegressor(**params, random_state=42)
            mdl.fit(X_tr, y_tr)
            pred[ch] = 0.5 * (mdl.predict(X_br[ch]) - mdl.predict(X_rb[ch]))
        hist = np.concatenate([hist, ch])

    # Modelo final con todo fit_m (< 16/09) para predecir V, Diamond-V y demás filas >= tr_cut
    ok_all = fit_idx[valid_tgt[fit_idx]]
    X_all = np.vstack([X_br[ok_all], X_rb[ok_all]])
    y_all = np.concatenate([y_tgt[ok_all], -y_tgt[ok_all]])
    mdl_fin = HistGradientBoostingRegressor(**params, random_state=42)
    mdl_fin.fit(X_all, y_all)

    post_m = ~mask_fiv
    pred[post_m] = 0.5 * (mdl_fin.predict(X_br[post_m]) - mdl_fin.predict(X_rb[post_m]))
    return np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def construir_hgb_ctx_y_post15(D):
    cache = REPO / "data" / "analysis" / "brainstorm" / "gemini_hgb_ctx_p15.npz"
    if cache.exists():
        z = np.load(cache)
        return z["HGB_CTX"], z["HGB_POST15"]

    t0 = time.time()
    NM, ms, t = D["NM"], D["master"], D["t"]
    mask_fiv = np.zeros(NM, dtype=bool)
    mask_fiv[D["fi_v"]] = True
    # Límite exacto anterior a V / Diamond-V (< 16/09)
    tr_cut = t[~mask_fiv].min()

    dg15, ok_lane = _lane_gold_15(D)
    P, p_cols = _build_player_compact(D)  # (NM, 2, 5, Fp)

    # Bloques globales causales de Escalón 6/7 ya existentes (antisimétricos Azul - Rojo)
    mpr = np.load(REPO / "data" / "analysis" / "brainstorm" / "gemini_mpr.npz")["MATCHUPS"]
    rol = np.load(REPO / "data" / "analysis" / "brainstorm" / "gemini_nuevas_v2.npz")["ROL_WR"]
    g11 = np.load(REPO / "data" / "analysis" / "brainstorm" / "gemini_g11.npz")
    f1 = construir_extras(D)["FOTO1_DIN"]
    G_glob = np.c_[mpr, rol, g11["KALMAN_OUT"], g11["KALMAN_MRG"], g11["TEMPO_OBJ"], g11["ECON_MID"], f1].astype(np.float32)

    # Resumen compacto de Jungla y Support para dar contexto intra-línea
    jg_b, jg_r = P[:, 0, 1, :], P[:, 1, 1, :]
    sup_b, sup_r = P[:, 0, 4, :], P[:, 1, 4, :]
    tier_m = np.nanmean(D["Z"]["tnum"].reshape(NM, 10), axis=1, keepdims=True).astype(np.float32)

    params_lane = dict(max_depth=5, min_samples_leaf=80, l2_regularization=10.0, max_iter=150, max_bins=128, learning_rate=0.05)

    HGB_CTX = np.zeros((NM, 6), dtype=np.float32)
    for l in range(5):
        tl = time.time()
        pb, pr = P[:, 0, l, :], P[:, 1, l, :]
        # Espejo exacto: al invertir Blue<->Red, G_glob cambia de signo y roles se intercambian
        extra_br = np.c_[G_glob, jg_b - jg_r, sup_b - sup_r, (pb[:, :20] - pr[:, :20]), tier_m]
        extra_rb = np.c_[-G_glob, jg_r - jg_b, sup_r - sup_b, (pr[:, :20] - pb[:, :20]), tier_m]
        X_br = np.c_[pb, pr, extra_br].astype(np.float32)
        X_rb = np.c_[pr, pb, extra_rb].astype(np.float32)
        HGB_CTX[:, l] = _forward_mirror_hgb(
            X_br, X_rb, dg15[:, l], ok_lane[:, l], t, ms, mask_fiv, tr_cut, params_lane, qs=7
        )
        print(f"  [HGB_CTX {LANES[l]:7s}] listo en {time.time() - tl:.1f}s", flush=True)
    HGB_CTX[:, 5] = HGB_CTX[:, :5].sum(axis=1)

    # --- Segunda familia: HGB equipo post-15 y XP@15 / Oro final ---
    # Representación por equipo simétrica-espejada: media de P[:, team] + max de P[:, team] + globales
    P_mean = P.mean(axis=2)  # (NM, 2, Fp)
    P_max = P.max(axis=2)
    X_eq_br = np.c_[P_mean[:, 0], P_mean[:, 1], P_max[:, 0] - P_max[:, 1], G_glob, tier_m].astype(np.float32)
    X_eq_rb = np.c_[P_mean[:, 1], P_mean[:, 0], P_max[:, 1] - P_max[:, 0], -G_glob, tier_m].astype(np.float32)

    params_eq = dict(max_depth=4, min_samples_leaf=100, l2_regularization=15.0, max_iter=150, max_bins=128, learning_rate=0.05)

    dur_min = D["Z"]["D_dur_min"]
    # 1) d_xp_15 / 1000
    dxp15 = (D["Z"]["D_d_xp_15"] / 1000.0).astype(np.float32)
    ok_xp15 = np.isfinite(dxp15) & (dur_min >= 15.0)
    pred_xp15 = _forward_mirror_hgb(X_eq_br, X_eq_rb, dxp15, ok_xp15, t, ms, mask_fiv, tr_cut, params_eq, qs=7)

    # 2) d_oro_fin / 2000
    dofin = (D["Z"]["D_d_oro_fin"] / 2000.0).astype(np.float32)
    ok_fin = np.isfinite(dofin)
    pred_ofin = _forward_mirror_hgb(X_eq_br, X_eq_rb, dofin, ok_fin, t, ms, mask_fiv, tr_cut, params_eq, qs=7)

    # 3) conv_post15 = (y - E[y | d_gold_15]) estrictamente causal por chunk
    dg15_tot = (D["Z"]["D_d_gold_15"] / 1000.0).astype(np.float32)
    ok_g15 = np.isfinite(dg15_tot) & (dur_min >= 15.0)
    y_bin = D["y"].astype(np.float32)
    pred_conv = np.zeros(NM, dtype=np.float32)

    fit_m = ms & mask_fiv & (t < tr_cut)
    fit_idx = np.flatnonzero(fit_m)
    ord_fit = fit_idx[np.argsort(t[fit_idx], kind="mergesort")]
    chunks = np.array_split(ord_fit, 7)
    hist = chunks[0]
    for k in range(1, 7):
        ch = chunks[k]
        t_lim = t[ch].min()
        ok_h = hist[ok_g15[hist] & (t[hist] < t_lim)]
        if len(ok_h) >= 500:
            lr_g = LogisticRegression(C=1.0, fit_intercept=False).fit(dg15_tot[ok_h][:, None], y_bin[ok_h])
            p_exp_h = expit(lr_g.coef_[0, 0] * dg15_tot[ok_h]).astype(np.float32)
            # Residuo antisimétrico exacto (sin intercepto: 1 - expit(z) == expit(-z))
            res_h = (y_bin[ok_h] - p_exp_h).astype(np.float32)
            X_tr = np.vstack([X_eq_br[ok_h], X_eq_rb[ok_h]])
            y_tr = np.concatenate([res_h, -res_h])
            mdl = HistGradientBoostingRegressor(**params_eq, random_state=42)
            mdl.fit(X_tr, y_tr)
            pred_conv[ch] = 0.5 * (mdl.predict(X_eq_br[ch]) - mdl.predict(X_eq_rb[ch]))
        hist = np.concatenate([hist, ch])

    ok_all = fit_idx[ok_g15[fit_idx]]
    lr_g = LogisticRegression(C=1.0, fit_intercept=False).fit(dg15_tot[ok_all][:, None], y_bin[ok_all])
    p_exp_all = expit(lr_g.coef_[0, 0] * dg15_tot[ok_all]).astype(np.float32)
    res_all = (y_bin[ok_all] - p_exp_all).astype(np.float32)
    mdl_c = HistGradientBoostingRegressor(**params_eq, random_state=42).fit(
        np.vstack([X_eq_br[ok_all], X_eq_rb[ok_all]]),
        np.concatenate([res_all, -res_all]),
    )
    post_m = ~mask_fiv
    pred_conv[post_m] = 0.5 * (mdl_c.predict(X_eq_br[post_m]) - mdl_c.predict(X_eq_rb[post_m]))
    pred_conv = np.nan_to_num(pred_conv, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    HGB_POST15 = np.c_[pred_xp15, pred_ofin, pred_conv].astype(np.float32)
    np.savez_compressed(cache, HGB_CTX=HGB_CTX, HGB_POST15=HGB_POST15)
    print(f"[+] Guardado {cache} en {time.time() - t0:.1f}s", flush=True)
    return HGB_CTX, HGB_POST15


if __name__ == "__main__":
    D = cargar()
    NM, ms = D["NM"], D["master"]
    X9 = extras_referencia(D, escalon=9)
    HGB_CTX, HGB_POST15 = construir_hgb_ctx_y_post15(D)

    # Reporte rápido en V sobre cuánto predicen los blancos físicos
    vi = np.flatnonzero(D["val"])
    print(f"[g_26 V] corr(d_xp_15, pred_xp15)={np.corrcoef(np.nan_to_num(D['Z']['D_d_xp_15'][vi]), HGB_POST15[vi, 0])[0,1]:.3f} | "
          f"corr(d_oro_fin, pred_ofin)={np.corrcoef(np.nan_to_num(D['Z']['D_d_oro_fin'][vi]), HGB_POST15[vi, 1])[0,1]:.3f}", flush=True)

    rng = np.random.default_rng(13)
    PL = np.c_[HGB_CTX, HGB_POST15][rng.permutation(NM)]

    def ml(X):
        return lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=X, extra_post=False, con_mapa=False)

    VAR = {
        "Escalón 9 oficial": ml(X9),
        "+ HGB_CTX_15 (6)": ml(np.c_[X9, HGB_CTX]),
        "+ HGB_POST15 (3)": ml(np.c_[X9, HGB_POST15]),
        "+ ambos (9)": ml(np.c_[X9, HGB_CTX, HGB_POST15]),
        "placebo": ml(np.c_[X9, PL]),
    }
    P = dict(Parallel(n_jobs=2, backend="threading")(delayed(lambda n: (n, evaluar(D, VAR[n])))(n) for n in VAR))
    mostrar(comparar(D, P, "Escalón 9 oficial"))

    di = np.flatnonzero(D["diam_v"])
    yd = D["y"][di]
    B = np.random.default_rng(1).integers(0, len(di), size=(300, len(di)))
    qd = P["Escalón 9 oficial"][0][di]
    for n in VAR:
        if n == "Escalón 9 oficial":
            continue
        pd_ = P[n][0][di]
        d = np.array([roc_auc_score(yd[b], pd_[b]) - roc_auc_score(yd[b], qd[b]) for b in B])
        print(f"  Diamond-V {n:20s}: {roc_auc_score(yd, pd_):.4f} (Δ {roc_auc_score(yd, pd_) - roc_auc_score(yd, qd):+.4f} [{np.percentile(d, 2.5):+.4f}; {np.percentile(d, 97.5):+.4f}])")
