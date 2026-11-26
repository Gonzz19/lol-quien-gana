"""Step 4 GPU (g_28): Reentrenamiento completo sobre Escalón 9 oficial (`V 0,6739`, `1.504` cols)
inyectando al encoder por jugador `P_jug` los canales causales que antes solo veía la logística:
  - `PER_JUG_ESC6[..., :16]` (`ROL_WR` 6 + `KALMAN_OUT/MRG/TEMPO` 10 de `gemini_g24.npz`)
  - `HGB @15` por línea con signo (`+HGB[:, :5]` Azul / `-HGB[:, :5]` Rojo de `gemini_hgb_lineas.npz`)
  - `MATCHUP_EXP_MARGIN @15` por línea con signo (`+MEM[:, 4:9]` Azul / `-MEM[:, 4:9]` Rojo de `gemini_g23.npz`)

Entrena estrictamente causal (`solo Master+`):
  1) `Ref9 oficial` (`C=0,05` exacto, `extras_referencia(D, escalon=9)`)
  2) `RedNeuralACWideDeep` (6 semillas RTX 3070, destilada OOF `@10/@15`)
  3) `RedAttnWideDeep` (5 semillas RTX 3070, destilada OOF `@10/@15`)
  4) `CatBoost GPU Residual v3` (`baseline = logit causal Escalón 9 oficial`)

Evalúa frente a **Escalón 9 oficial (`0,6739`)** y **Escalón 8 GPU (`0,6735`, `pred_g22_esc7.npz`)**
con mezclas prefijadas (`45/25/15/15`, `25/25/25/25`, `65/35`, `50/25/25`) y stacking causal en `A2..A4`,
incluyendo `V Master+`, `A2..A4`, LogLoss, ECE, cobertura, segmentos y chequeo ciego `Diamond-V`.
Guarda predicciones en `data/analysis/brainstorm/pred_g28_esc9.npz`.
"""
import gc
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
from pathlib import Path

import numpy as np
import torch
from catboost import CatBoostClassifier, Pool
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.brainstorm_base import (  # noqa: E402
    ajustar_predecir,
    cargar,
    comparar,
    extras_referencia,
    mostrar,
)
from historial.escalon9.brainstorm_g_01_foto1_wl import construir_extras  # noqa: E402
from historial.ensamble_en_vivo.brainstorm_g_05_gpu_master import cargar_foto1_jugador  # noqa: E402
from historial.escalon9.brainstorm_g_06_comp_poisson import construir_composicion  # noqa: E402
from historial.ensamble_en_vivo.brainstorm_g_22_red_y_catboost_esc6 import (  # noqa: E402
    BR_DIR,
    DEV,
    RedAttnWideDeep,
    RedNeuralACWideDeep,
    _ece,
    _logit,
    _sigm,
    construir_X_cb,
    construir_priv,
    entrenar_redes_fold,
    probs_maestro_oof,
)


def preparar_entradas_redes_esc9(D: dict, REF9: np.ndarray):
    """Construye `P_jug` enriquecido (207 pre + 12 post = 219 cols) y skip Wide Esc9 pre/post."""
    NM = D["NM"]
    F1 = cargar_foto1_jugador(D)                 # (NM, 2, 5, 4) post-t1
    COMP, J_CAMP = construir_composicion(D)      # (NM, 25), (NM, 2, 5, 14)

    A_mmr = np.load(BR_DIR / "mmr_historico.npz")["A"].astype(np.float32)
    with np.load(BR_DIR / "margen_grupos_escala_sin_v.npz") as z:
        MC, ML = z["MC"].astype(np.float32), z["ML"].astype(np.float32)
    with np.load(BR_DIR / "mc_ajuste.npz") as z:
        mc5, mc20, mcl = z["mc5"].astype(np.float32), z["mc20"].astype(np.float32), z["mcl"].astype(np.float32)
    H = np.load(BR_DIR / "conoce_matchup.npz")["H"].astype(np.float32)

    # Canales por jugador de Escalón 6..9 (todos estrictamente causales < t)
    PJ16 = np.load(BR_DIR / "gemini_g24.npz")["PER_JUG_ESC6"][..., :16].astype(np.float32)
    HGB = np.load(BR_DIR / "gemini_hgb_lineas.npz")["HGB"].astype(np.float32)
    MEM = np.load(BR_DIR / "gemini_g23.npz")["MATCHUP_EXP_MARGIN"].astype(np.float32)

    hgb_lane = np.stack([HGB[:, :5], -HGB[:, :5]], axis=1)[..., None].astype(np.float32)
    mem_lane = np.stack([MEM[:, 4:9], -MEM[:, 4:9]], axis=1)[..., None].astype(np.float32)
    post_p = np.broadcast_to(D["post"][:, None, None, None], (NM, 2, 5, 1)).astype(np.float32)

    # Últimas 12 cols = [D["J"](7), F1(4), post_flag(1)], compatible con d_pre_jug = shape[2] - 12
    P_jug = np.concatenate(
        [D["Xj"], A_mmr, MC, ML, mc5, mc20, mcl, H, J_CAMP, PJ16, hgb_lane, mem_lane, D["J"], F1, post_p],
        axis=-1,
    ).reshape(NM, 10, -1).astype(np.float32)
    np.nan_to_num(P_jug, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    F1_DIN = construir_extras(D)["FOTO1_DIN"].astype(np.float32)  # (NM, 4)
    # `_ref` layout en Escalón 9: mcml(150) + niv(3) + FOTO1_DIN(4) + escalon4..9
    REF_pre = np.c_[REF9[:, :153], REF9[:, 157:]].astype(np.float32)
    W_pre = np.c_[D["Xd"], REF_pre, COMP].astype(np.float32)
    W_post = np.c_[F1_DIN, (D["J"][:, 0] - D["J"][:, 1]).mean(1), (F1[:, 0] - F1[:, 1]).mean(1)].astype(np.float32)
    np.nan_to_num(W_pre, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(W_post, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    ids = D["Z"]["champ"].reshape(NM, 10)
    unicos = np.unique(ids)
    cmap = {int(c): i for i, c in enumerate(unicos)}
    camp_idx = np.vectorize(cmap.__getitem__, otypes=[np.int64])(ids)

    MC_ML_eq = np.c_[(MC[:, 0] - MC[:, 1]).reshape(NM, -1), (ML[:, 0] - ML[:, 1]).reshape(NM, -1)].astype(np.float32)
    F1_eq = np.c_[F1_DIN, (F1[:, 0] - F1[:, 1]).mean(1)].astype(np.float32)

    del A_mmr, MC, ML, mc5, mc20, mcl, H, J_CAMP, PJ16, hgb_lane, mem_lane
    gc.collect()
    return camp_idx, P_jug, W_pre, W_post, F1, COMP, len(unicos), MC_ML_eq, F1_eq, HGB, MEM


def cb_residual_v3_fold(D, X_pre, X_post, REF9, tr, te, dv_idx=None):
    """CatBoost GPU Residual v3 con baseline estrictamente causal = logit Escalón 9 oficial (C=0,05)."""
    t, y, post = D["t"], D["y"], D["post"]
    qs = np.quantile(t[tr], np.linspace(0, 1, 7)[1:-1])
    ch = np.searchsorted(qs, t[tr], side="right")
    ref_tr = np.zeros(len(tr), dtype=np.float32)
    draft_tr = np.zeros(len(tr), dtype=np.float32)

    for b in range(1, 6):
        itr, ite = tr[ch < b], tr[ch == b]
        p_ref = ajustar_predecir(D, itr, ite, extra=REF9, extra_post=False, con_mapa=False)
        ref_tr[ch == b] = _logit(p_ref)
        lr_d = LogisticRegression(C=0.05, max_iter=2000).fit(D["DRAFT"][itr], y[itr])
        draft_tr[ch == b] = _logit(lr_d.predict_proba(D["DRAFT"][ite])[:, 1])

    te_all = te if dv_idx is None else np.concatenate([te, dv_idx])
    p_ref_all = ajustar_predecir(D, tr, te_all, extra=REF9, extra_post=False, con_mapa=False)
    lr_dfull = LogisticRegression(C=0.05, max_iter=2000).fit(D["DRAFT"][tr], y[tr])
    draft_all = _logit(lr_dfull.predict_proba(D["DRAFT"][te_all])[:, 1]).astype(np.float32)

    def mat(idx, r_l, d_l):
        po = np.where(post[idx][:, None], X_post[idx], np.nan)
        return np.c_[r_l, d_l, np.abs(r_l), X_pre[idx], po, post[idx].astype(np.float32)].astype(np.float32)

    m_ok = ch >= 1
    tr_cb = tr[m_ok]
    X_tr = mat(tr_cb, ref_tr[m_ok], draft_tr[m_ok])
    ref_cb = ref_tr[m_ok].astype(np.float64)
    y_cb = y[tr_cb]

    pre_m, pos_m = ~post[tr_cb], post[tr_cb]
    if pos_m.sum() >= 1000:
        c_pre = np.quantile(t[tr_cb][pre_m], 0.88) if pre_m.any() else 0
        c_pos = np.quantile(t[tr_cb][pos_m], 0.80)
        is_es = np.where(post[tr_cb], t[tr_cb] >= c_pos, t[tr_cb] >= c_pre)
    else:
        is_es = t[tr_cb] >= np.quantile(t[tr_cb], 0.85)

    cb = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.02,
        depth=5,
        l2_leaf_reg=150.0,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=42,
        task_type="GPU" if DEV.type == "cuda" else "CPU",
        verbose=False,
        early_stopping_rounds=120,
    )
    cb.fit(
        Pool(X_tr[~is_es], y_cb[~is_es], baseline=ref_cb[~is_es]),
        eval_set=Pool(X_tr[is_es], y_cb[is_es], baseline=ref_cb[is_es]),
        use_best_model=True,
    )

    ref_all = _logit(p_ref_all).astype(np.float32)
    X_all = mat(te_all, ref_all, draft_all)
    p_all = cb.predict_proba(Pool(X_all, baseline=ref_all.astype(np.float64)))[:, 1].astype(np.float32)
    del cb
    gc.collect()
    if DEV.type == "cuda":
        torch.cuda.empty_cache()

    if dv_idx is None:
        return p_all
    nt = len(te)
    return p_all[:nt], p_all[nt:]


def _diag_reporte(D, P, p_dv_map, ref_key="Escalón 9 oficial"):
    y, val, diam_v = D["y"], D["val"], D["diam_v"]
    vi = np.flatnonzero(val)
    di = np.flatnonzero(diam_v)
    yv, yd = y[vi], y[di]

    B_v = np.random.default_rng(1).integers(0, len(yv), size=(300, len(yv)))
    B_d = np.random.default_rng(1).integers(0, len(yd), size=(300, len(yd)))
    qv_ref = P[ref_key][1]
    qd_ref = p_dv_map[ref_key]
    auc_v_ref = roc_auc_score(yv, qv_ref)
    auc_d_ref = roc_auc_score(yd, qd_ref)

    print("\n--- V Master+ (LogLoss / ECE / Cobertura / Bootstrap) ---", flush=True)
    for k, (po, pv) in P.items():
        auc_v = roc_auc_score(yv, pv)
        ll_v = log_loss(yv, pv)
        ece_v = _ece(yv, pv)
        d = np.array([roc_auc_score(yv[b], pv[b]) - roc_auc_score(yv[b], qv_ref[b]) for b in B_v])
        ok = (pv > 0.5) == yv
        o = np.argsort(-np.abs(pv - 0.5))
        cov = " ".join(f"{c:.0%}:{ok[o[:int(c * len(vi))]].mean():.1%}" for c in (0.05, 0.10, 0.20, 0.30, 0.50))
        print(f"  {k:34s}: AUC {auc_v:.4f} ({auc_v - auc_v_ref:+.4f} [{np.percentile(d, 2.5):+.4f}; {np.percentile(d, 97.5):+.4f}]) "
              f"| ac {ok.mean():.2%} | LL {ll_v:.5f} | ECE {ece_v:.4f} | {cov}", flush=True)

    print("\n--- Chequeo ciego Diamond-V (10.506 partidas) ---", flush=True)
    for k, pd_ in p_dv_map.items():
        auc_d = roc_auc_score(yd, pd_)
        ll_d = log_loss(yd, pd_)
        if k == ref_key:
            print(f"  {k:34s}: AUC {auc_d:.4f} | LL {ll_d:.5f}", flush=True)
        else:
            d = np.array([roc_auc_score(yd[b], pd_[b]) - roc_auc_score(yd[b], qd_ref[b]) for b in B_d])
            print(f"  {k:34s}: AUC {auc_d:.4f} ({auc_d - auc_d_ref:+.4f} [{np.percentile(d, 2.5):+.4f}; {np.percentile(d, 97.5):+.4f}]) | LL {ll_d:.5f}", flush=True)

    # Segmentos difíciles de V (`brainstorm_12`)
    Xj = D["Xj"]
    c = {name: i for i, name in enumerate(D["cols"])}
    est_sum = Xj[vi, ..., c["estreno"]].reshape(len(vi), 10).sum(1)
    n_prev = np.expm1(Xj[vi, ..., c["logn_tot"]].reshape(len(vi), 10))
    hora = ((D["t"][vi] / 3.6e6) - 3.0) % 24.0
    segs = {
        "sin_estrenos": est_sum == 0,
        "veteranos>=50": n_prev.min(1) >= 50,
        "18-24h": (hora >= 18.0) & (hora < 24.0),
        "madrugada_0-6h": (hora >= 0.0) & (hora < 6.0),
    }
    print("\n--- Segmentos difíciles V Master+ ---", flush=True)
    for sn, sm in segs.items():
        row = " | ".join(f"{k}={roc_auc_score(yv[sm], P[k][1][sm]):.4f}" for k in (ref_key, "Red NeuralAC Esc9+", "Red AttnWD Esc9+", "CatBoost GPU Res v3", "Mix 45/25/15/15 Esc9", "Mix 25/25/25/25 Esc9") if k in P)
        print(f"  {sn:15s} (n={int(sm.sum()):5d}): {row}", flush=True)


def main():
    t0 = time.time()
    D = cargar()
    NM, ms, fi, bl = D["NM"], D["master"], D["fi"], D["bloque"]
    REF9 = extras_referencia(D, escalon=9)
    print(f"[g_28] GPU={DEV} | REF9 shape={REF9.shape}", flush=True)

    camp_idx, P_jug, W_pre, W_post, F1, COMP, n_camp, MC_ML_eq, F1_eq, HGB, MEM = preparar_entradas_redes_esc9(D, REF9)
    PRIV = construir_priv(D)
    X_pre, X_post = construir_X_cb(D, F1, COMP, extra_opt=np.c_[HGB, MEM[:, 1:]].astype(np.float32))
    print(f"[g_28] P_jug={P_jug.shape} (pre={P_jug.shape[2]-12}+12post) | W_pre={W_pre.shape} | X_cb_pre={X_pre.shape}", flush=True)

    models = ("ref9", "nac", "att", "cb")
    po = {m: np.full(NM, np.nan, dtype=np.float32) for m in models}
    pv = {m: np.full(int(D["val"].sum()), np.nan, dtype=np.float32) for m in models}
    p_dv = {m: np.full(int(D["diam_v"].sum()), np.nan, dtype=np.float32) for m in models}

    seeds_nac = (11, 22, 33, 44, 55, 66)
    seeds_att = (101, 202, 303, 404, 505)

    for b in D["EV"]:
        tf = time.time()
        tr_b = fi[bl[fi] < b]
        tr_b = tr_b[ms[tr_b]]
        te_b = np.flatnonzero(bl == b)
        p_mae = probs_maestro_oof(D, PRIV, tr_b, MC_ML_eq, F1_eq, n_folds=4)

        po["ref9"][te_b] = ajustar_predecir(D, tr_b, te_b, extra=REF9, extra_post=False, con_mapa=False).astype(np.float32)
        po["nac"][te_b] = _sigm(entrenar_redes_fold(D, camp_idx, P_jug, W_pre, W_post, n_camp, p_mae, tr_b, te_b, seeds_nac, RedNeuralACWideDeep)).astype(np.float32)
        po["att"][te_b] = _sigm(entrenar_redes_fold(D, camp_idx, P_jug, W_pre, W_post, n_camp, p_mae, tr_b, te_b, seeds_att, RedAttnWideDeep)).astype(np.float32)
        po["cb"][te_b] = cb_residual_v3_fold(D, X_pre, X_post, REF9, tr_b, te_b)
        print(f"  [fold A{b + 1}] listo en {time.time() - tf:.1f}s | ref9={roc_auc_score(D['y'][te_b], po['ref9'][te_b]):.4f} "
              f"nac={roc_auc_score(D['y'][te_b], po['nac'][te_b]):.4f} "
              f"att={roc_auc_score(D['y'][te_b], po['att'][te_b]):.4f} "
              f"cb={roc_auc_score(D['y'][te_b], po['cb'][te_b]):.4f}", flush=True)

    # Fold V + Diamond-V (< 16/09, solo Master+)
    tf = time.time()
    tr_v = D["fi_v"][ms[D["fi_v"]]]
    vi = np.flatnonzero(D["val"])
    di = np.flatnonzero(D["diam_v"])
    p_mae_v = probs_maestro_oof(D, PRIV, tr_v, MC_ML_eq, F1_eq, n_folds=4)

    p_ref_vd = ajustar_predecir(D, tr_v, np.concatenate([vi, di]), extra=REF9, extra_post=False, con_mapa=False).astype(np.float32)
    pv["ref9"], p_dv["ref9"] = p_ref_vd[:len(vi)], p_ref_vd[len(vi):]

    zn_v, zn_d = entrenar_redes_fold(D, camp_idx, P_jug, W_pre, W_post, n_camp, p_mae_v, tr_v, vi, seeds_nac, RedNeuralACWideDeep, dv_idx=di)
    pv["nac"], p_dv["nac"] = _sigm(zn_v).astype(np.float32), _sigm(zn_d).astype(np.float32)

    za_v, za_d = entrenar_redes_fold(D, camp_idx, P_jug, W_pre, W_post, n_camp, p_mae_v, tr_v, vi, seeds_att, RedAttnWideDeep, dv_idx=di)
    pv["att"], p_dv["att"] = _sigm(za_v).astype(np.float32), _sigm(za_d).astype(np.float32)

    pv["cb"], p_dv["cb"] = cb_residual_v3_fold(D, X_pre, X_post, REF9, tr_v, vi, dv_idx=di)
    print(f"  [fold V+Diam] listo en {time.time() - tf:.1f}s (total {time.time() - t0:.1f}s)", flush=True)

    out_npz = BR_DIR / "pred_g28_esc9.npz"
    np.savez_compressed(
        out_npz,
        po_ref9=po["ref9"], pv_ref9=pv["ref9"], p_dv_ref9=p_dv["ref9"],
        po_nac=po["nac"], pv_nac=pv["nac"], p_dv_nac=p_dv["nac"],
        po_att=po["att"], pv_att=pv["att"], p_dv_att=p_dv["att"],
        po_cb=po["cb"], pv_cb=pv["cb"], p_dv_cb=p_dv["cb"],
    )
    print(f"[+] Guardado {out_npz}", flush=True)

    # Mezclas prefijadas en logits (cero ajuste sobre V/Diamond-V)
    def mix(ws):
        w_r, w_n, w_a, w_c = ws
        lo = w_r * _logit(po["ref9"]) + w_n * _logit(po["nac"]) + w_a * _logit(po["att"]) + w_c * _logit(po["cb"])
        lv = w_r * _logit(pv["ref9"]) + w_n * _logit(pv["nac"]) + w_a * _logit(pv["att"]) + w_c * _logit(pv["cb"])
        ld = w_r * _logit(p_dv["ref9"]) + w_n * _logit(p_dv["nac"]) + w_a * _logit(p_dv["att"]) + w_c * _logit(p_dv["cb"])
        return (_sigm(lo), _sigm(lv)), _sigm(ld)

    # Stacking causal entrenado únicamente en bloques A2..A4 Master+
    m_bl = np.isin(bl, D["EV"]) & ms
    Z_bl = np.c_[_logit(po["ref9"][m_bl]), _logit(po["nac"][m_bl]), _logit(po["att"][m_bl]), _logit(po["cb"][m_bl])]
    lr_st = LogisticRegression(C=0.5, fit_intercept=False, max_iter=1000).fit(Z_bl, D["y"][m_bl])
    ws_st = np.maximum(0.0, lr_st.coef_[0])
    if ws_st.sum() > 0:
        ws_st = ws_st / ws_st.sum()
    print(f"[g_28 Stacking A2..A4] pesos [Ref9, NAC, Att, CB] = {np.round(ws_st, 4)}", flush=True)

    m45, d45 = mix((0.45, 0.25, 0.15, 0.15))
    m25, d25 = mix((0.25, 0.25, 0.25, 0.25))
    m65, d65 = mix((0.65, 0.35, 0.00, 0.00))
    m50, d50 = mix((0.50, 0.25, 0.00, 0.25))
    mst, dst = mix(tuple(ws_st.tolist()))

    P = {
        "Escalón 9 oficial": (po["ref9"], pv["ref9"]),
        "Red NeuralAC Esc9+": (po["nac"], pv["nac"]),
        "Red AttnWD Esc9+": (po["att"], pv["att"]),
        "CatBoost GPU Res v3": (po["cb"], pv["cb"]),
        "Mix 65/35 Ref9+NAC": m65,
        "Mix 50/25/25 Ref9+NAC+CB": m50,
        "Mix 45/25/15/15 Esc9": m45,
        "Mix 25/25/25/25 Esc9": m25,
        "Mix Stack A2..A4 Esc9": mst,
    }
    p_dv_map = {
        "Escalón 9 oficial": p_dv["ref9"],
        "Red NeuralAC Esc9+": p_dv["nac"],
        "Red AttnWD Esc9+": p_dv["att"],
        "CatBoost GPU Res v3": p_dv["cb"],
        "Mix 65/35 Ref9+NAC": d65,
        "Mix 50/25/25 Ref9+NAC+CB": d50,
        "Mix 45/25/15/15 Esc9": d45,
        "Mix 25/25/25/25 Esc9": d25,
        "Mix Stack A2..A4 Esc9": dst,
    }
    mostrar(comparar(D, P, "Escalón 9 oficial"))
    _diag_reporte(D, P, p_dv_map, "Escalón 9 oficial")

    esc8_p = BR_DIR / "pred_g22_esc7.npz"
    if esc8_p.exists():
        z8 = np.load(esc8_p)
        l8_v = 0.45 * _logit(z8["pv_ref"]) + 0.25 * _logit(z8["pv_nac"]) + 0.15 * _logit(z8["pv_att"]) + 0.15 * _logit(z8["pv_cb"])
        l8_d = 0.45 * _logit(z8["p_dv_ref"]) + 0.25 * _logit(z8["p_dv_nac"]) + 0.15 * _logit(z8["p_dv_att"]) + 0.15 * _logit(z8["p_dv_cb"])
        p8_v, p8_d = _sigm(l8_v), _sigm(l8_d)
        yv, yd = D["y"][vi], D["y"][di]
        Bv = np.random.default_rng(1).integers(0, len(yv), size=(300, len(yv)))
        Bd = np.random.default_rng(1).integers(0, len(yd), size=(300, len(yd)))
        print("\n--- Comparación directa contra Escalón 8 vigente (`pred_g22_esc7.npz`, Mix 45/25/15/15) ---", flush=True)
        for k in ("Mix 45/25/15/15 Esc9", "Mix 25/25/25/25 Esc9", "Mix Stack A2..A4 Esc9"):
            pv_k, pd_k = P[k][1], p_dv_map[k]
            dv_b = np.array([roc_auc_score(yv[b], pv_k[b]) - roc_auc_score(yv[b], p8_v[b]) for b in Bv])
            dd_b = np.array([roc_auc_score(yd[b], pd_k[b]) - roc_auc_score(yd[b], p8_d[b]) for b in Bd])
            print(f"  {k:24s} vs Esc8: V Δ {roc_auc_score(yv, pv_k) - roc_auc_score(yv, p8_v):+.4f} "
                  f"[{np.percentile(dv_b, 2.5):+.4f}; {np.percentile(dv_b, 97.5):+.4f}] | "
                  f"Diam Δ {roc_auc_score(yd, pd_k) - roc_auc_score(yd, p8_d):+.4f} "
                  f"[{np.percentile(dd_b, 2.5):+.4f}; {np.percentile(dd_b, 97.5):+.4f}]", flush=True)


if __name__ == "__main__":
    main()
