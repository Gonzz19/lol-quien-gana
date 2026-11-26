"""Ablación limpia y verificación de `g_11` sobre el nuevo **Escalón 4b** (`extras_referencia(D, escalon=4, rol_wr=True)`):
1. Separa cada sub-bloque sin `CLUTCH_15`: `KALMAN_OUT`, `KALMAN_MRG`, `TEMPO_OBJ` (subdividido en
   `FB_FT_SWAP` = cols 0..4 vs `DUR_EARLY_LATE` = cols 5..9) y `ECON_MID`.
2. Evalúa `g11_sin_clutch` (`33` cols) y `g11_compacto` (`KALMAN_MRG + TEMPO_OBJ + ECON_MID`, `21` cols) s/Escalón 4b.
3. Corre **Placebo** (filas permutadas) + **Chequeo ciego en Diamond-V** (protocolo `brainstorm_17/28`).
4. Guarda `pred_g11.npz` para que Claude audite directo sin esperar ~10 min de recálculo.
"""
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.special import expit, logit
from sklearn.metrics import log_loss, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.brainstorm_base import ajustar_predecir, cargar, comparar, evaluar, extras_referencia, mostrar  # noqa: E402

CACHE_G11 = REPO / "data" / "analysis" / "brainstorm" / "gemini_g11.npz"
OUT_NPZ = REPO / "data" / "analysis" / "brainstorm" / "pred_g11.npz"


def main():
    t0 = time.time()
    D = cargar()
    NM, ms = D["NM"], D["master"]
    R4b = extras_referencia(D, escalon=4, rol_wr=True)
    G = np.load(CACHE_G11)

    K_OUT = G["KALMAN_OUT"].astype(np.float32)
    K_MRG = G["KALMAN_MRG"].astype(np.float32)
    T_OBJ = G["TEMPO_OBJ"].astype(np.float32)
    E_MID = G["ECON_MID"].astype(np.float32)

    FB_FT_SWAP = T_OBJ[:, :5]
    EARLY_LATE = T_OBJ[:, 5:]
    G11_NO_CL = np.c_[K_OUT, K_MRG, T_OBJ, E_MID].astype(np.float32)
    G11_COMP = np.c_[K_MRG, T_OBJ, E_MID].astype(np.float32)
    PL_G11 = G11_NO_CL[np.random.default_rng(7).permutation(NM)]

    VAR = {
        "ref: Escalón 4b (+ROL_WR)": R4b,
        "+ KALMAN_MRG (7)": np.c_[R4b, K_MRG],
        "+ KALMAN_OUT (12)": np.c_[R4b, K_OUT],
        "+ FB_FT_SWAP (5)": np.c_[R4b, FB_FT_SWAP],
        "+ DUR_EARLY_LATE (5)": np.c_[R4b, EARLY_LATE],
        "+ TEMPO_OBJ completo (10)": np.c_[R4b, T_OBJ],
        "+ ECON_MID (4)": np.c_[R4b, E_MID],
        "+ g11_comp (21, K_MRG+TEMPO+ECON)": np.c_[R4b, G11_COMP],
        "+ g11_puro (33, sin CLUTCH_15)": np.c_[R4b, G11_NO_CL],
        "placebo g11_puro": np.c_[R4b, PL_G11],
    }

    def run_one(nom, X):
        X32 = np.asarray(X, dtype=np.float32)
        return nom, evaluar(D, lambda tr, te, M=X32: ajustar_predecir(D, tr[ms[tr]], te, extra=M, extra_post=False))

    P = dict(Parallel(n_jobs=2, backend="threading")(delayed(run_one)(n, X) for n, X in VAR.items()))
    mostrar(comparar(D, P, "ref: Escalón 4b (+ROL_WR)"))

    # Métricas adicionales en V + mezcla exploratoria con Red NeuralAC orig (pred_g08)
    yv = D["y"][D["val"]]
    g08_path = REPO / "data" / "analysis" / "brainstorm" / "pred_g08.npz"
    if g08_path.exists():
        G8 = np.load(g08_path)
        p_red = (G8["po_red"], G8["pv_red"])
        eps = 1e-6
        L = lambda a: logit(np.clip(a, eps, 1.0 - eps))  # noqa: E731
        po_l, pv_l = P["+ g11_puro (33, sin CLUTCH_15)"]
        for w in (0.25, 0.35):
            P[f"Mix {1-w:.2f} g11_puro + {w:.2f} RedNAC(g08)"] = (
                expit((1.0 - w) * L(po_l) + w * L(p_red[0])),
                expit((1.0 - w) * L(pv_l) + w * L(p_red[1])),
            )
        mostrar(comparar(D, {k: P[k] for k in ["ref: Escalón 4b (+ROL_WR)", "+ g11_puro (33, sin CLUTCH_15)",
                                              "Mix 0.75 g11_puro + 0.25 RedNAC(g08)",
                                              "Mix 0.65 g11_puro + 0.35 RedNAC(g08)"]}, "ref: Escalón 4b (+ROL_WR)"))

    print("\nLogLoss V Master+:")
    for nom in ["ref: Escalón 4b (+ROL_WR)", "+ g11_comp (21, K_MRG+TEMPO+ECON)", "+ g11_puro (33, sin CLUTCH_15)",
                "Mix 0.65 g11_puro + 0.35 RedNAC(g08)"]:
        if nom in P:
            pv = P[nom][1]
            print(f"  {nom:40s} LL={log_loss(yv, np.clip(pv, 1e-6, 1-1e-6)):.4f}  AUC={roc_auc_score(yv, pv):.4f}")

    # Chequeo ciego en Diamond-V
    di = np.flatnonzero(D["diam_v"])
    yd = D["y"][di]
    B = np.random.default_rng(1).integers(0, len(di), size=(300, len(di)))
    b0 = P["ref: Escalón 4b (+ROL_WR)"][0][di]
    print(f"\nChequeo Diamond-V ({len(di):,} partidas, nunca usadas para decidir):")
    for nom in ["+ KALMAN_MRG (7)", "+ TEMPO_OBJ completo (10)", "+ ECON_MID (4)",
                "+ g11_comp (21, K_MRG+TEMPO+ECON)", "+ g11_puro (33, sin CLUTCH_15)", "placebo g11_puro"]:
        p = P[nom][0][di]
        d = np.array([roc_auc_score(yd[b], p[b]) - roc_auc_score(yd[b], b0[b]) for b in B])
        print(f"  {nom:38s} AUC {roc_auc_score(yd, p):.4f}  acierto {((p > .5) == yd).mean():.1%}  "
              f"Δ {roc_auc_score(yd, p) - roc_auc_score(yd, b0):+.4f} [{np.percentile(d, 2.5):+.4f}; {np.percentile(d, 97.5):+.4f}]")

    np.savez_compressed(
        OUT_NPZ,
        po_4b=P["ref: Escalón 4b (+ROL_WR)"][0], pv_4b=P["ref: Escalón 4b (+ROL_WR)"][1],
        po_g11_comp=P["+ g11_comp (21, K_MRG+TEMPO+ECON)"][0], pv_g11_comp=P["+ g11_comp (21, K_MRG+TEMPO+ECON)"][1],
        po_g11_puro=P["+ g11_puro (33, sin CLUTCH_15)"][0], pv_g11_puro=P["+ g11_puro (33, sin CLUTCH_15)"][1],
    )
    print(f"Listo y guardado en {OUT_NPZ} en {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
