"""Brainstorming (docs/BRAINSTORM_NOCHE.md): verificación (Claude) del escalón 5 propuesto por Gemini: mezcla de logits
65/35 de la logística del escalón 4 y la red NeuralAC en GPU (predicciones de `pred_g08.npz`, de brainstorm_g_08).

1. Recalcula AUC/acierto/Δ con `comparar` (V Master+ y sub-bloques Master+).
2. Chequeo honesto en las Diamond de la ventana de V (predichas por el modelo hacia adelante del sub-bloque A4,
   nunca usadas para decidir): ¿la mezcla también mejora ahí?
3. Sensibilidad al peso: 0,15 / 0,25 / 0,35 / 0,5 para la red.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.brainstorm_base import REPO, cargar, comparar, mostrar  # noqa: E402

D = cargar()
G = np.load(REPO / "data" / "analysis" / "brainstorm" / "pred_g08.npz")
L = (G["po_log4"], G["pv_log4"])
R = (G["po_red"], G["pv_red"])


def mezcla(w):
    f = lambda a, b: expit((1 - w) * logit(np.clip(a, 1e-6, 1 - 1e-6)) + w * logit(np.clip(b, 1e-6, 1 - 1e-6)))  # noqa: E731
    return (f(L[0], R[0]), f(L[1], R[1]))


P = {"escalón 4 (logística)": L, "red NeuralAC sola": R}
for w in (0.15, 0.25, 0.35, 0.5):
    P[f"mezcla red {w:.2f}"] = mezcla(w)
mostrar(comparar(D, P, "escalón 4 (logística)"))

di = np.flatnonzero(D["diam_v"])
y = D["y"][di]
B = np.random.default_rng(1).integers(0, len(di), size=(300, len(di)))
print(f"\nDiamond de la ventana de V ({len(di):,}, nunca usadas para decidir; modelo hacia adelante de A4):")
base = L[0][di]
for n, (po, _) in P.items():
    p = po[di]
    d = np.array([roc_auc_score(y[b], p[b]) - roc_auc_score(y[b], base[b]) for b in B])
    print(f"  {n:24s} AUC {roc_auc_score(y, p):.4f}  acierto {((p > .5) == y).mean():.1%}  Δ {roc_auc_score(y, p) - roc_auc_score(y, base):+.4f} "
          f"[{np.percentile(d, 2.5):+.4f}; {np.percentile(d, 97.5):+.4f}]")
