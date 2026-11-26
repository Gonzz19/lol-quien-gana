"""Brainstorming (docs/BRAINSTORM_NOCHE.md, Step 2 post g_22):
Auditoría y ablación limpia del ensamble Escalón 8 candidato (`pred_g22_esc7.npz`):
  1) Mezclas sobre Escalón 7 PURO (`po_ref` sin L2 dif) + `RedNeuralAC` + `RedAttnWideDeep` + `CatBoost v2`,
     separando exactamente el aporte de L2 dif (`+0,0002`) del aporte no lineal real.
  2) Stacking / calibración estrictamente causal entrenado SOLO sobre bloques `A2..A4` (`< V`), evaluado
     ciegamente en `V Master+` y `Diamond-V` (AUC, acierto, log-loss, ECE, cobertura 5..50%, McNemar).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.brainstorm_base import REPO, cargar, comparar, mostrar  # noqa: E402

BR_DIR = REPO / "data" / "analysis" / "brainstorm"


def _logit(p):
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return logit(p)


def _ece(y, p, bins=10):
    dec = pd.qcut(p, bins, labels=False, duplicates="drop")
    g = pd.DataFrame({"p": p, "y": y, "d": dec}).groupby("d").agg(p=("p", "mean"), r=("y", "mean"), n=("y", "size"))
    return float((np.abs(g.p - g.r) * g.n).sum() / g.n.sum())


if __name__ == "__main__":
    D = cargar()
    z = np.load(BR_DIR / "pred_g22_esc7.npz")
    NM = D["NM"]
    vi = np.flatnonzero(D["val"])
    dv = np.flatnonzero(D["diam_v"])
    yv = D["y"][vi]
    ydv = D["y"][dv]

    po_ref, pv_ref, pdv_ref = z["po_ref"], z["pv_ref"], z["p_dv_ref"]
    po_log, pv_log, pdv_log = z["po_log"], z["pv_log"], z["p_dv_log"]
    po_nac, pv_nac, pdv_nac = z["po_nac"], z["pv_nac"], z["p_dv_nac"]
    po_att, pv_att, pdv_att = z["po_att"], z["pv_att"], z["p_dv_att"]
    po_cb, pv_cb, pdv_cb = z["po_cb"], z["pv_cb"], z["p_dv_cb"]

    def mix_w(weights, items_o, items_v, items_dv):
        zo = np.zeros(NM, dtype=np.float64)
        zv = np.zeros(len(vi), dtype=np.float64)
        zdv = np.zeros(len(dv), dtype=np.float64)
        for w, oi, vi_, dvi in zip(weights, items_o, items_v, items_dv):
            zo += w * _logit(np.nan_to_num(oi, nan=0.5))
            zv += w * _logit(vi_)
            zdv += w * _logit(dvi)
        return (expit(zo), expit(zv)), expit(zdv)

    P = {"Escalón 7 oficial (ref)": (po_ref, pv_ref)}
    DV = {}

    mixes = [
        ("Mix 65/35 Ref7+NAC (s/L2dif)", [0.65, 0.35], [po_ref, po_nac], [pv_ref, pv_nac], [pdv_ref, pdv_nac]),
        ("Mix 55/30/15 Ref7+NAC+Att (s/L2dif)", [0.55, 0.30, 0.15], [po_ref, po_nac, po_att], [pv_ref, pv_nac, pv_att], [pdv_ref, pdv_nac, pdv_att]),
        ("Mix 45/25/15/15 Ref7+NAC+Att+CB", [0.45, 0.25, 0.15, 0.15], [po_ref, po_nac, po_att, po_cb], [pv_ref, pv_nac, pv_att, pv_cb], [pdv_ref, pdv_nac, pdv_att, pdv_cb]),
        ("Mix 50/25/25 Ref7+NAC+CB", [0.50, 0.25, 0.25], [po_ref, po_nac, po_cb], [pv_ref, pv_nac, pv_cb], [pdv_ref, pdv_nac, pdv_cb]),
        ("Mix 45/25/15/15 LogL2+NAC+Att+CB", [0.45, 0.25, 0.15, 0.15], [po_log, po_nac, po_att, po_cb], [pv_log, pv_nac, pv_att, pv_cb], [pdv_log, pdv_nac, pdv_att, pdv_cb]),
        ("Mix 50/25/25 LogL2+NAC+CB", [0.50, 0.25, 0.25], [po_log, po_nac, po_cb], [pv_log, pv_nac, pv_cb], [pdv_log, pdv_nac, pdv_cb]),
    ]
    for nom, ws, io, iv, idv in mixes:
        (po_m, pv_m), pdv_m = mix_w(ws, io, iv, idv)
        P[nom] = (po_m, pv_m)
        DV[nom] = pdv_m

    # Stacking estrictamente causal ajustado SOLO en bloques A2..A4 (`EV`)
    m_ev = np.isin(D["bloque"], D["EV"]) & D["master"]
    Zo = np.c_[_logit(po_ref), _logit(po_nac), _logit(po_att), _logit(po_cb)]
    Zv = np.c_[_logit(pv_ref), _logit(pv_nac), _logit(pv_att), _logit(pv_cb)]
    Zdv = np.c_[_logit(pdv_ref), _logit(pdv_nac), _logit(pdv_att), _logit(pdv_cb)]
    lr_st = LogisticRegression(C=0.2, fit_intercept=False, max_iter=2000).fit(Zo[m_ev], D["y"][m_ev])
    print("Pesos Stacking causal (fit A2..A4 sin intercept, [Ref7, NAC, Att, CB]):",
          np.round(lr_st.coef_[0], 4), "suma=", round(float(lr_st.coef_[0].sum()), 4))
    po_st = np.full(NM, np.nan)
    po_st[m_ev] = lr_st.predict_proba(Zo[m_ev])[:, 1]
    pv_st = lr_st.predict_proba(Zv)[:, 1]
    pdv_st = lr_st.predict_proba(Zdv)[:, 1]
    P["Stack A2..A4 (no-int, [Ref7,NAC,Att,CB])"] = (po_st, pv_st)
    DV["Stack A2..A4 (no-int, [Ref7,NAC,Att,CB])"] = pdv_st

    # Versión con L2dif + NAC + CB ajustada en A2..A4 (sin Att)
    Zo2 = np.c_[_logit(po_log), _logit(po_nac), _logit(po_cb)]
    Zv2 = np.c_[_logit(pv_log), _logit(pv_nac), _logit(pv_cb)]
    Zdv2 = np.c_[_logit(pdv_log), _logit(pdv_nac), _logit(pdv_cb)]
    lr_st2 = LogisticRegression(C=0.2, fit_intercept=False, max_iter=2000).fit(Zo2[m_ev], D["y"][m_ev])
    print("Pesos Stacking causal (fit A2..A4 sin intercept, [LogL2, NAC, CB]):",
          np.round(lr_st2.coef_[0], 4), "suma=", round(float(lr_st2.coef_[0].sum()), 4))
    po_st2 = np.full(NM, np.nan)
    po_st2[m_ev] = lr_st2.predict_proba(Zo2[m_ev])[:, 1]
    P["Stack A2..A4 (no-int, [LogL2,NAC,CB])"] = (po_st2, lr_st2.predict_proba(Zv2)[:, 1])
    DV["Stack A2..A4 (no-int, [LogL2,NAC,CB])"] = lr_st2.predict_proba(Zdv2)[:, 1]

    mostrar(comparar(D, P, "Escalón 7 oficial (ref)"))

    print("\n--- Detalle en V Master+ (LL, ECE, Cobertura 5/10/20/30/50%) ---")
    for nom, (_, pv) in P.items():
        ok = (pv > 0.5) == yv
        o = np.argsort(-np.abs(pv - 0.5))
        cov = " ".join(f"{int(c*100)}%->{ok[o[:int(c*len(yv))]].mean():.1%}" for c in (0.05, 0.10, 0.20, 0.30, 0.50))
        print(f"  {nom:40s} AUC={roc_auc_score(yv, pv):.4f} ac={ok.mean():.2%} LL={log_loss(yv, pv):.5f} ECE={_ece(yv, pv):.4f} | {cov}")

    print("\n--- Chequeo ciego Diamond-V (10.506 partidas) ---")
    auc_ref_dv = roc_auc_score(ydv, pdv_ref)
    boots = np.random.default_rng(77).integers(0, len(ydv), size=(1500, len(ydv)))
    for nom, pdv in DV.items():
        auc_p = roc_auc_score(ydv, pdv)
        ds = np.array([roc_auc_score(ydv[b], pdv[b]) - roc_auc_score(ydv[b], pdv_ref[b]) for b in boots])
        print(f"  {nom:40s} AUC={auc_p:.4f} Δ={auc_p - auc_ref_dv:+.4f} [{np.percentile(ds, 2.5):+.4f}; {np.percentile(ds, 97.5):+.4f}] LL={log_loss(ydv, pdv):.5f}")
