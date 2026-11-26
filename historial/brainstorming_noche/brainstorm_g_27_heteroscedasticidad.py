"""Brainstorming (docs/BRAINSTORM_NOCHE.md, ideas #35/#36):
Sobre Escalón 7 oficial (V 0,6722, extras_referencia(D, escalon=7), sin mapa):
  1) RES_ECON (6 cols): eficiencia económica y conflicto de recursos en el equipo:
     - dmg_minus_gold_share (daño% - oro% por jugador: convierte econ en daño vs chupa oro)
     - sum_top4_gold_share (saturación: ¿cuatro carries compiten por las mismas oleadas?)
     - kp_minus_gold_share (participación en kills por unidad de oro)
     - weakside_eff (mejor eficiencia del jugador con menos oro)
  2) Z7_X_VOL (Heteroscedasticidad Probit/Logit causal + cola no lineal):
     - En todo modelo binario Probit/Logit, P = sigma(mu / sqrt(s0^2 + s_azul^2 + s_rojo^2)).
     - Las variables actuales son todas antisimétricas (Azul - Rojo), así que la incertidumbre
       simétrica del lobby se cancela en la resta lineal.
     - Calculamos z7_oof estrictamente forward-chaining (< chunk k) dentro de tr_m, y multiplicamos
       z7_oof * rasgo_simétrico_de_volatilidad_causal (total estrenos, inv veteranía mín., suma
       kills+agg_10, dispersión Kalman/nivel, suma DUR_EARLY_LATE, madrugada 0-6h) + cola cúbica/tanh.
     - Incluye ablaciones limpias y placebo con volatilidad permutada.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.brainstorm_base import REPO, ajustar_predecir, cargar, comparar, evaluar, extras_referencia, mostrar  # noqa: E402


def construir_res_econ(D):
    """Eficiencia de conversión oro→daño y conflicto de recursos por equipo (100% causal vía Xj)."""
    Xj = D["Xj"]  # (NM, 2, 5, 85)
    c = {k: i for i, k in enumerate(D["cols"])}

    gs_ac = Xj[..., c["acum_gold_share_pct"]]
    ds_ac = Xj[..., c["acum_damage_share_pct"]]
    kp_ac = Xj[..., c["acum_kill_participation_pct"]]
    gs_20 = Xj[..., c["v20_gold_share_pct"]]
    ds_20 = Xj[..., c["v20_damage_share_pct"]]

    dmg_over_gold = ds_ac - gs_ac
    dmg_over_gold_20 = ds_20 - gs_20
    kp_over_gold = kp_ac - gs_ac

    # Saturación de recursos: suma de los 4 roles con mayor gold_share (TOP, JUG, MID, BOT = [:4])
    sum_top4_gold = gs_ac[:, :, :4].sum(axis=2)
    sum_top4_gold_20 = gs_20[:, :, :4].sum(axis=2)
    # Weakside carry: eficiencia media de los 2 jugadores con menor gold_share histórico de cada equipo
    idx_low = np.argsort(gs_ac, axis=2)[:, :, :2]
    rows = np.arange(D["NM"])[:, None, None]
    teams = np.arange(2)[None, :, None]
    weak_eff = dmg_over_gold[rows, teams, idx_low].mean(axis=2)

    RES_ECON = np.c_[
        dmg_over_gold[:, 0].mean(1) - dmg_over_gold[:, 1].mean(1),
        dmg_over_gold_20[:, 0].mean(1) - dmg_over_gold_20[:, 1].mean(1),
        kp_over_gold[:, 0].mean(1) - kp_over_gold[:, 1].mean(1),
        dmg_over_gold[:, 0, 4] - dmg_over_gold[:, 1, 4],  # Support / weakside impact
        sum_top4_gold[:, 0] - sum_top4_gold[:, 1],
        sum_top4_gold_20[:, 0] - sum_top4_gold_20[:, 1],
        weak_eff[:, 0] - weak_eff[:, 1],
    ].astype(np.float32)
    return np.nan_to_num(RES_ECON, nan=0.0, posinf=0.0, neginf=0.0)


def construir_volatilidad_simetrica(D):
    """Rasgos simétricos de incertidumbre/varianza por partida (100% causales, sin Foto 2 ni futuro)."""
    NM = D["NM"]
    Xj = D["Xj"]
    c = {k: i for i, k in enumerate(D["cols"])}

    # 1) Total estrenos en la partida (0..10)
    estrenos = Xj[..., c["estreno"]].reshape(NM, 10).sum(1)
    # 2) Inv veteranía mínima (cuentas con menor historial en el lobby)
    logn_min = Xj[..., c["logn_tot"]].reshape(NM, 10).min(1)
    inv_vet = 1.0 / (1.0 + np.maximum(logn_min, 0.0))
    # 3) Caos / bloodbath: suma histórica de kills+deaths+early_agg_10 en el lobby
    kills = Xj[..., c["acum_kills_pm"]].reshape(NM, 10).mean(1)
    deaths = Xj[..., c["acum_deaths_pm"]].reshape(NM, 10).mean(1)
    agg10 = Xj[..., c["perf_early_aggression_10"]].reshape(NM, 10).mean(1)
    bloodbath = kills + deaths + 0.25 * agg10
    # 4) Dispersión de nivel histórico (lobby heterogéneo) y Kalman Elo spread
    A = np.load(REPO / "data" / "analysis" / "brainstorm" / "mmr_historico.npz")["A"][..., 0].reshape(NM, 10)
    niv_std = np.nanstd(A, axis=1)
    # 5) Hora madrugada (0-6h ART)
    hora = ((D["t"] / 3.6e6) - 3.0) % 24.0
    madrugada = ((hora >= 0.0) & (hora < 6.0)).astype(np.float32)
    # 6) Desbalance de tiempo de cierre promedio entre campeones (total scaling / fiesta tardía)
    g11 = np.load(REPO / "data" / "analysis" / "brainstorm" / "gemini_g11.npz")
    kal_abs = np.abs(g11["KALMAN_OUT"][:, 0])

    VOL = np.c_[estrenos, inv_vet, bloodbath, niv_std, madrugada, kal_abs].astype(np.float32)
    return np.nan_to_num(VOL, nan=0.0, posinf=0.0, neginf=0.0)


_Z_CACHE = {}


def _get_z_forward(D, X, tag, tr, te):
    key = (tag, int(tr[0]), int(tr[-1]), len(tr), int(te[0]), len(te))
    if key in _Z_CACHE:
        return _Z_CACHE[key]
    ms, t = D["master"], D["t"]
    tr_m = tr[ms[tr]]
    ord_idx = np.argsort(t[tr_m], kind="mergesort")
    tr_ord = tr_m[ord_idx]
    chunks = np.array_split(tr_ord, 6)

    z_oof = np.zeros(len(tr_ord), dtype=np.float32)
    pos_map = {idx: pos for pos, idx in enumerate(tr_ord)}
    hist = chunks[0]
    for k in range(1, len(chunks)):
        ch = chunks[k]
        p_ch = ajustar_predecir(D, hist, ch, extra=X, extra_post=False, con_mapa=False)
        z_ch = logit(np.clip(p_ch, 1e-6, 1.0 - 1e-6)).astype(np.float32)
        for idx_m, zv in zip(ch, z_ch):
            z_oof[pos_map[idx_m]] = zv
        hist = np.concatenate([hist, ch])

    p_te = ajustar_predecir(D, tr_m, te, extra=X, extra_post=False, con_mapa=False)
    z_te = logit(np.clip(p_te, 1e-6, 1.0 - 1e-6)).astype(np.float32)
    tr_fit = np.concatenate(chunks[1:])
    z_fit = z_oof[len(chunks[0]):]
    _Z_CACHE[key] = (tr_fit, z_fit, z_te)
    return _Z_CACHE[key]


def hacer_modelo_hetero(D, X, tag, VOL, usar_tail=True, usar_vol=True):
    """Calibrador Probit heteroscedástico de 2da etapa estrictamente forward-chaining dentro de tr_m."""
    y = D["y"]

    def fn(tr, te):
        tr_fit, z_fit, z_te = _get_z_forward(D, X, tag, tr, te)

        sc_v = StandardScaler().fit(VOL[tr_fit])
        V_fit = sc_v.transform(VOL[tr_fit]).astype(np.float32)
        V_te = sc_v.transform(VOL[te]).astype(np.float32)

        sc_z = StandardScaler().fit(z_fit[:, None])
        z_fit_s = sc_z.transform(z_fit[:, None]).ravel().astype(np.float32)
        z_te_s = sc_z.transform(z_te[:, None]).ravel().astype(np.float32)

        def build_feat(zs, zraw, Vs):
            cols = [zraw[:, None]]
            if usar_vol:
                cols.append((zs[:, None] * Vs))
            if usar_tail:
                # Término impar de curvatura/saturación de cola (antisimétrico)
                tail = (np.tanh(zraw) - 0.5 * zraw)[:, None]
                cols.append(tail)
            return np.hstack(cols).astype(np.float32)

        Xf = build_feat(z_fit_s, z_fit, V_fit)
        Xt = build_feat(z_te_s, z_te, V_te)

        sc2 = StandardScaler().fit(Xf)
        lr2 = LogisticRegression(C=0.05, max_iter=2000).fit(sc2.transform(Xf), y[tr_fit])
        return lr2.predict_proba(sc2.transform(Xt))[:, 1]

    return fn


if __name__ == "__main__":
    D = cargar()
    NM, ms = D["NM"], D["master"]
    X9 = extras_referencia(D, escalon=9)
    RES_ECON = construir_res_econ(D)
    X9_RE = np.c_[X9, RES_ECON].astype(np.float32)
    VOL = construir_volatilidad_simetrica(D)
    VOL_PL = VOL[np.random.default_rng(7).permutation(NM)]

    def ml(X):
        return lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=X, extra_post=False, con_mapa=False)

    VAR = {
        "Escalón 9 oficial": ml(X9),
        "+ RES_ECON (7)": ml(X9_RE),
        "+ Z9_X_VOL solo (6)": hacer_modelo_hetero(D, X9, "x9", VOL, usar_tail=False, usar_vol=True),
        "+ Z9_tail solo (1)": hacer_modelo_hetero(D, X9, "x9", VOL, usar_tail=True, usar_vol=False),
        "+ Z9_X_VOL + tail (7)": hacer_modelo_hetero(D, X9, "x9", VOL, usar_tail=True, usar_vol=True),
        "+ Z9_X_VOL + tail + RES_ECON": hacer_modelo_hetero(D, X9_RE, "x9re", VOL, usar_tail=True, usar_vol=True),
        "placebo VOL": hacer_modelo_hetero(D, X9, "x9", VOL_PL, usar_tail=False, usar_vol=True),
    }

    print("[g_27] Precalentando caché forward-chaining y evaluando sobre Escalón 9 (100% causal)...", flush=True)
    # Precomputar en secuencia para reutilizar _Z_CACHE sin colisiones y luego correr limpio
    P = {n: evaluar(D, fn) for n, fn in VAR.items()}
    mostrar(comparar(D, P, "Escalón 9 oficial"))

    vi = np.flatnonzero(D["val"])
    yv = D["y"][vi]
    print("\n--- Log-loss y bootstrap en V Master+ ---")
    ref_v = P["Escalón 9 oficial"][1]
    for n in VAR:
        pv = P[n][1]
        ll = log_loss(yv, pv)
        if n == "Escalón 9 oficial":
            print(f"  {n:30s}: V AUC {roc_auc_score(yv, pv):.4f} | LL {ll:.5f}")
        else:
            print(f"  {n:30s}: V AUC {roc_auc_score(yv, pv):.4f} | LL {ll:.5f} (ΔLL {ll - log_loss(yv, ref_v):+.5f})")

    di = np.flatnonzero(D["diam_v"])
    yd = D["y"][di]
    B = np.random.default_rng(1).integers(0, len(di), size=(300, len(di)))
    qd = P["Escalón 9 oficial"][0][di]
    print("\n--- Chequeo ciego Diamond-V (10.506 partidas) ---")
    for n in VAR:
        if n == "Escalón 9 oficial":
            continue
        pd_ = P[n][0][di]
        d = np.array([roc_auc_score(yd[b], pd_[b]) - roc_auc_score(yd[b], qd[b]) for b in B])
        print(f"  Diamond-V {n:30s}: {roc_auc_score(yd, pd_):.4f} (Δ {roc_auc_score(yd, pd_) - roc_auc_score(yd, qd):+.4f} [{np.percentile(d, 2.5):+.4f}; {np.percentile(d, 97.5):+.4f}])")
