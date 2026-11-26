"""Verificación de reproducibilidad por semillas para el Escalón 5 (`brainstorm_22`, pedido de Claude):
¿La Red NeuralAC Wide&Deep en GPU da el mismo salto con 3 semillas completamente nuevas `(101, 202, 303)`
respecto de `(0, 1, 2)` usadas en `pred_g08.npz`?

Evalúa:
1. `Red NeuralAC (semillas 101, 202, 303)` sola y en `Mix 65/35` / `Mix 50/50` con `Logística Escalón 4`.
2. `Red NeuralAC Destilada (alfa=0.5, semillas 101, 202, 303)` sola y en `Mix 65/35` / `Mix 45/30/25`.
3. Ensamble de las 6 semillas `(0, 1, 2, 101, 202, 303)` promedio en logits.
4. Chequeo honesto en `Diamond-V` (mismo protocolo de `brainstorm_22_verifica_escalon5.py`).
"""
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.brainstorm_base import cargar, comparar, evaluar, mostrar  # noqa: E402
from historial.ensamble_en_vivo.brainstorm_g_05_gpu_master import cargar_foto1_jugador, preparar_datos_red  # noqa: E402
from historial.escalon9.brainstorm_g_06_comp_poisson import construir_composicion  # noqa: E402
from historial.ensamble_en_vivo.brainstorm_g_08_destilacion_gpu import construir_priv, entrenar_red_destilada  # noqa: E402

OUT_NPZ = REPO / "data" / "analysis" / "brainstorm" / "pred_g15_semillas.npz"


def main():
    t0 = time.time()
    D = cargar()
    NM = D["NM"]
    C = np.load(REPO / "data" / "analysis" / "brainstorm" / "margen_grupos.npz")
    MC_raw, ML_raw = C["MC"], C["ML"]
    A_mmr = np.load(REPO / "data" / "analysis" / "brainstorm" / "mmr_historico.npz")["A"].astype(np.float32)
    NIV_EQ = (A_mmr[:, 0] - A_mmr[:, 1])[..., :3].mean(axis=1).astype(np.float32)
    MC_ML_NIV = np.c_[
        (MC_raw[:, 0] - MC_raw[:, 1]).reshape(NM, -1),
        (ML_raw[:, 0] - ML_raw[:, 1]).reshape(NM, -1),
        NIV_EQ,
    ].astype(np.float32)
    F1 = cargar_foto1_jugador(D)
    F1_eq = (F1[:, 0] - F1[:, 1]).mean(axis=1).astype(np.float32)
    PRIV = construir_priv(D)
    COMP, J_CAMP = construir_composicion(D)
    mpr_path = REPO / "data" / "analysis" / "brainstorm" / "gemini_mpr.npz"
    MATCHUPS = np.load(mpr_path)["MATCHUPS"].astype(np.float32)
    MC_ML_NAT = np.c_[MC_ML_NIV, MATCHUPS].astype(np.float32)

    camp_idx, P_jug_base, W_pre_base, d_post_eq, n_camp = preparar_datos_red(D, F1, MC_raw, ML_raw)
    P_jug = np.concatenate([A_mmr[:, :, :, :3].reshape(NM, 10, -1), J_CAMP.reshape(NM, 10, -1), P_jug_base], axis=-1).astype(np.float32)
    W_pre = np.c_[W_pre_base, NIV_EQ, MATCHUPS, COMP].astype(np.float32)

    g08 = np.load(REPO / "data" / "analysis" / "brainstorm" / "pred_g08.npz")
    p_log4 = (g08["po_log4"], g08["pv_log4"])
    p_red0 = (g08["po_red"], g08["pv_red"])
    p_dist0 = (g08["po_dist"], g08["pv_dist"])
    p_cb = (g08["po_cb"], g08["pv_cb"])

    nuevas = (101, 202, 303)
    print(f"Entrenando Red NeuralAC GPU alfa=1.0 con 3 semillas nuevas {nuevas} ({time.time()-t0:.1f}s)...", flush=True)
    p_red_new = evaluar(D, lambda tr, te: entrenar_red_destilada(
        D, camp_idx, P_jug, W_pre, d_post_eq, n_camp, PRIV, MC_ML_NAT, F1_eq, tr, te, alfa=1.0, seeds=nuevas
    ))

    print(f"Entrenando Red NeuralAC Dist alfa=0.5 con 3 semillas nuevas {nuevas} ({time.time()-t0:.1f}s)...", flush=True)
    p_dist_new = evaluar(D, lambda tr, te: entrenar_red_destilada(
        D, camp_idx, P_jug, W_pre, d_post_eq, n_camp, PRIV, MC_ML_NAT, F1_eq, tr, te, alfa=0.5, seeds=nuevas
    ))

    eps = 1e-6
    L = lambda x: logit(np.clip(x, eps, 1.0 - eps))  # noqa: E731
    mix = lambda items: tuple(expit(sum(w * L(p[k]) for w, p in items)) for k in (0, 1))  # noqa: E731

    p_red_6 = mix([(0.5, p_red0), (0.5, p_red_new)])
    p_dist_6 = mix([(0.5, p_dist0), (0.5, p_dist_new)])

    P = {
        "escalón 4 (logística)": p_log4,
        "Red NAC orig (0,1,2)": p_red0,
        "Red NAC nuevas (101,202,303)": p_red_new,
        "Red NAC 6 semillas": p_red_6,
        "Esc5 orig 65/35 (0,1,2)": mix([(0.65, p_log4), (0.35, p_red0)]),
        "Esc5 nuevas 65/35 (101,202,303)": mix([(0.65, p_log4), (0.35, p_red_new)]),
        "Esc5 nuevas 50/50 (101,202,303)": mix([(0.50, p_log4), (0.50, p_red_new)]),
        "Esc5 6 semillas (65/35)": mix([(0.65, p_log4), (0.35, p_red_6)]),
        "Red Dist orig (0,1,2)": p_dist0,
        "Red Dist nuevas (101,202,303)": p_dist_new,
        "Mix 45/30/25 Dist nuevas + CB": mix([(0.45, p_log4), (0.30, p_dist_new), (0.25, p_cb)]),
        "Mix 45/30/25 Dist 6sem + CB": mix([(0.45, p_log4), (0.30, p_dist_6), (0.25, p_cb)]),
    }
    mostrar(comparar(D, P, "escalón 4 (logística)"))

    di = np.flatnonzero(D["diam_v"])
    yv = D["y"][di]
    B = np.random.default_rng(1).integers(0, len(di), size=(300, len(di)))
    base = p_log4[0][di]
    print(f"\nChequeo Diamond-V ({len(di):,} partidas, nunca usadas para decidir):")
    for nom, (po, _) in P.items():
        p = po[di]
        d = np.array([roc_auc_score(yv[b], p[b]) - roc_auc_score(yv[b], base[b]) for b in B])
        print(f"  {nom:34s} AUC {roc_auc_score(yv, p):.4f}  acierto {((p > .5) == yv).mean():.1%}  "
              f"Δ {roc_auc_score(yv, p) - roc_auc_score(yv, base):+.4f} [{np.percentile(d, 2.5):+.4f}; {np.percentile(d, 97.5):+.4f}]")

    np.savez_compressed(
        OUT_NPZ,
        po_red_new=p_red_new[0], pv_red_new=p_red_new[1],
        po_dist_new=p_dist_new[0], pv_dist_new=p_dist_new[1],
        po_red_6=p_red_6[0], pv_red_6=p_red_6[1],
        po_dist_6=p_dist_6[0], pv_dist_6=p_dist_6[1],
    )
    print(f"Guardado en {OUT_NPZ} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
