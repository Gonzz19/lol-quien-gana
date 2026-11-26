"""Paso 3 (`docs/BRAINSTORM_NOCHE.md` + verificación lineal/GPU de `LK_AGGS`):
1. Compara en detalle `Escalón 9 oficial (0,6739)` vs **`Esc9 + LK_AGGS (4)` (`g_30`)** en V Master+, LogLoss, ECE,
   cobertura (5/10/20/30/50 %) y segmentos (`sin_estrenos`, `veteranos>=50`, `18-24h`, `madrugada`, `GM+` vs `Master`).
2. Verifica si reemplazar `Ref9` por **`Ref9 + LK_AGGS (4)`** en las **mezclas prefijadas de Escalón 10 (`pred_g28_esc9.npz`)**
   (`65/35`, `50/25/25`, `45/25/15/15`, `60/20/10/10`, `25/25/25/25`, SIN ajustar pesos en V ni Diamond-V)
   mejora tanto respecto a `Escalón 9 (0,6739)` como respecto a `Escalón 10 GPU (0,6749)` y `Esc9 + LK_AGGS (0,6749)`,
   en A2..A4, V Master+ y Diamond-V ciego.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.brainstorm_base import (  # noqa: E402
    ajustar_predecir,
    cargar,
    comparar,
    evaluar,
    extras_referencia,
    mostrar,
)

BR_DIR = REPO / "data" / "analysis" / "brainstorm"


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _sigm(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))


def _ece(y, p, n_bins=15):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    tot = 0.0
    for b0, b1 in zip(bins[:-1], bins[1:]):
        m = (p >= b0) & (p <= b1 if b1 == 1.0 else p < b1)
        if m.sum() > 0:
            tot += (m.sum() / len(y)) * abs(y[m].mean() - p[m].mean())
    return float(tot)


if __name__ == "__main__":
    D = cargar()
    NM, ms, y = D["NM"], D["master"], D["y"]
    X9 = extras_referencia(D, escalon=9)
    lk = np.load(BR_DIR / "gemini_g30_lk15.npz")["LK_AGGS"].astype(np.float32)
    X9_LK = np.c_[X9, lk].astype(np.float32)

    g28 = np.load(BR_DIR / "pred_g28_esc9.npz")
    po_ref9, pv_ref9, pd_ref9 = g28["po_ref9"], g28["pv_ref9"], g28["p_dv_ref9"]
    po_nac, pv_nac, pd_nac = g28["po_nac"], g28["pv_nac"], g28["p_dv_nac"]
    po_att, pv_att, pd_att = g28["po_att"], g28["pv_att"], g28["p_dv_att"]
    po_cb, pv_cb, pd_cb = g28["po_cb"], g28["pv_cb"], g28["p_dv_cb"]

    print("[g_33] Ajustando Ref9+LK_AGGS (4) en A2..A4 y V+Diamond-V...", flush=True)
    # Evaluamos OOF + V + Diamond-V de Ref9+LK_AGGS usando evaluar() para tener po completo + pv
    # Ojo: evaluar() predice todo te (`bl == b`) y en V usa solo val==1; por eso sacamos Diamond-V aparte limpio
    po_lk, pv_lk = evaluar(
        D,
        lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=X9_LK, extra_post=False, con_mapa=False),
    )
    tr_v = D["fi_v"][ms[D["fi_v"]]]
    vi = np.flatnonzero(D["val"])
    di = np.flatnonzero(D["diam_v"])
    p_vd_lk = ajustar_predecir(D, tr_v, np.concatenate([vi, di]), extra=X9_LK, extra_post=False, con_mapa=False)
    assert np.allclose(p_vd_lk[:len(vi)], pv_lk, atol=1e-6), "Desacuerdo entre evaluar y ajustar_predecir en V"
    pd_lk = p_vd_lk[len(vi):].astype(np.float32)

    # Chequeo consistencia Ref9 contra pred_g28_esc9
    p_vd_r9 = ajustar_predecir(D, tr_v, np.concatenate([vi, di]), extra=X9, extra_post=False, con_mapa=False)
    assert np.allclose(p_vd_r9[:len(vi)], pv_ref9, atol=1e-4) and np.allclose(p_vd_r9[len(vi):], pd_ref9, atol=1e-4)

    def mix(base_po, base_pv, base_pd, ws):
        w_r, w_n, w_a, w_c = ws
        lo = w_r * _logit(base_po) + w_n * _logit(po_nac) + w_a * _logit(po_att) + w_c * _logit(po_cb)
        lv = w_r * _logit(base_pv) + w_n * _logit(pv_nac) + w_a * _logit(pv_att) + w_c * _logit(pv_cb)
        ld = w_r * _logit(base_pd) + w_n * _logit(pd_nac) + w_a * _logit(pd_att) + w_c * _logit(pd_cb)
        return (_sigm(lo), _sigm(lv)), _sigm(ld)

    e10_50, e10_d50 = mix(po_ref9, pv_ref9, pd_ref9, (0.50, 0.25, 0.00, 0.25))
    e10_45, e10_d45 = mix(po_ref9, pv_ref9, pd_ref9, (0.45, 0.25, 0.15, 0.15))

    m50, d50 = mix(po_lk, pv_lk, pd_lk, (0.50, 0.25, 0.00, 0.25))
    m45, d45 = mix(po_lk, pv_lk, pd_lk, (0.45, 0.25, 0.15, 0.15))
    m60, d60 = mix(po_lk, pv_lk, pd_lk, (0.60, 0.20, 0.10, 0.10))
    m25, d25 = mix(po_lk, pv_lk, pd_lk, (0.25, 0.25, 0.25, 0.25))

    P = {
        "Escalón 9 oficial": (po_ref9, pv_ref9),
        "Escalón 10 (50/25/25)": e10_50,
        "Escalón 10 (45/25/15/15)": e10_45,
        "Ref9 + LK_AGGS (4) [lineal]": (po_lk, pv_lk),
        "Mix 50/25/25 Ref9_LK+NAC+CB": m50,
        "Mix 45/25/15/15 Ref9_LK+GPU": m45,
        "Mix 60/20/10/10 Ref9_LK+GPU": m60,
        "Mix 25/25/25/25 Ref9_LK+GPU": m25,
    }
    PD = {
        "Escalón 9 oficial": pd_ref9,
        "Escalón 10 (50/25/25)": e10_d50,
        "Escalón 10 (45/25/15/15)": e10_d45,
        "Ref9 + LK_AGGS (4) [lineal]": pd_lk,
        "Mix 50/25/25 Ref9_LK+NAC+CB": d50,
        "Mix 45/25/15/15 Ref9_LK+GPU": d45,
        "Mix 60/20/10/10 Ref9_LK+GPU": d60,
        "Mix 25/25/25/25 Ref9_LK+GPU": d25,
    }

    print("\n=== TABLA VS ESCALÓN 9 OFICIAL (V 0.6739) ===", flush=True)
    mostrar(comparar(D, P, "Escalón 9 oficial"))

    print("\n=== TABLA VS ESCALÓN 10 GPU 50/25/25 (V 0.6749) ===", flush=True)
    mostrar(comparar(D, P, "Escalón 10 (50/25/25)"))

    yv = y[vi]
    yd = y[di]
    Bv = np.random.default_rng(1).integers(0, len(yv), size=(300, len(yv)))
    Bd = np.random.default_rng(1).integers(0, len(yd), size=(300, len(yd)))

    print("\n--- Detalle V Master+ (LogLoss / ECE / Cobertura / vs Esc10) ---", flush=True)
    qv9 = pv_ref9
    qv10 = e10_50[1]
    for k, (_, pv) in P.items():
        auc_v = roc_auc_score(yv, pv)
        ll_v = log_loss(yv, pv)
        ece_v = _ece(yv, pv)
        ok = (pv > 0.5) == yv
        o = np.argsort(-np.abs(pv - 0.5))
        cov = " ".join(f"{c:.0%}:{ok[o[:int(c * len(vi))]].mean():.1%}" for c in (0.05, 0.10, 0.20, 0.50))
        d10 = np.array([roc_auc_score(yv[b], pv[b]) - roc_auc_score(yv[b], qv10[b]) for b in Bv])
        print(f"  {k:30s}: AUC {auc_v:.4f} (vs10 {auc_v - roc_auc_score(yv, qv10):+.4f} [{np.percentile(d10, 2.5):+.4f}; {np.percentile(d10, 97.5):+.4f}]) "
              f"| ac {ok.mean():.2%} | LL {ll_v:.5f} | ECE {ece_v:.4f} | {cov}", flush=True)

    print("\n--- Chequeo ciego Diamond-V (vs Esc9 y vs Esc10_45) ---", flush=True)
    qd9 = pd_ref9
    qd10 = e10_d45
    for k, pd_ in PD.items():
        auc_d = roc_auc_score(yd, pd_)
        ll_d = log_loss(yd, pd_)
        d9 = np.array([roc_auc_score(yd[b], pd_[b]) - roc_auc_score(yd[b], qd9[b]) for b in Bd])
        d10 = np.array([roc_auc_score(yd[b], pd_[b]) - roc_auc_score(yd[b], qd10[b]) for b in Bd])
        print(f"  {k:30s}: {auc_d:.4f} | vs9 {auc_d - roc_auc_score(yd, qd9):+.4f} [{np.percentile(d9, 2.5):+.4f}; {np.percentile(d9, 97.5):+.4f}] "
              f"| vs10_45 {auc_d - roc_auc_score(yd, qd10):+.4f} [{np.percentile(d10, 2.5):+.4f}; {np.percentile(d10, 97.5):+.4f}] | LL {ll_d:.5f}", flush=True)

    # Segmentos difíciles en V
    Xj = D["Xj"]
    c = {name: i for i, name in enumerate(D["cols"])}
    est_sum = Xj[vi, ..., c["estreno"]].reshape(len(vi), 10).sum(1)
    n_prev = np.expm1(Xj[vi, ..., c["logn_tot"]].reshape(len(vi), 10))
    hora = ((D["t"][vi] / 3.6e6) - 3.0) % 24.0
    tnum_v = D["Z"]["tnum"].reshape(NM, 10)[vi]
    segs = {
        "sin_estrenos": est_sum == 0,
        "veteranos>=50": n_prev.min(1) >= 50,
        "18-24h": (hora >= 18.0) & (hora < 24.0),
        "GM+ (tnum>=5)": tnum_v.mean(1) >= 5.0,
    }
    print("\n--- Segmentos V Master+ ---", flush=True)
    for sn, sm in segs.items():
        row = " | ".join(f"{k}={roc_auc_score(yv[sm], P[k][1][sm]):.4f}" for k in ("Escalón 9 oficial", "Escalón 10 (50/25/25)", "Ref9 + LK_AGGS (4) [lineal]", "Mix 50/25/25 Ref9_LK+NAC+CB", "Mix 45/25/15/15 Ref9_LK+GPU"))
        print(f"  {sn:15s} (n={int(sm.sum()):5d}): {row}", flush=True)
