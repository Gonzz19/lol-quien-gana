"""Brainstorming (docs/BRAINSTORM_NOCHE.md): verificación (Claude) del escalón 10 de Gemini (ensamble sobre el escalón 9:
logística + red NeuralAC + red de atención + CatBoost residual, `pred_g28_esc9.npz`). Recalcula con `comparar`, prueba la
sensibilidad a los pesos (incluye pesos iguales, fijados sin mirar nada) y el chequeo en Diamond-V (`p_dv_*`)."""
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.brainstorm_base import REPO, cargar, comparar, mostrar  # noqa: E402

D = cargar()
z = np.load(REPO / "data" / "analysis" / "brainstorm" / "pred_g28_esc9.npz")
L = lambda a: logit(np.clip(a, 1e-6, 1 - 1e-6))  # noqa: E731
M = ("ref9", "nac", "att", "cb")


def mix(w):
    return tuple(expit(sum(wi * L(z[f"{k}_{m}"]) for wi, m in zip(w, M))) for k in ("po", "pv", "p_dv"))


W = {"escalón 9 (ref)": (1, 0, 0, 0), "65/35 ref+NAC": (.65, .35, 0, 0), "45/25/15/15 (Gemini)": (.45, .25, .15, .15),
     "pesos iguales 25/25/25/25": (.25, .25, .25, .25), "60/20/10/10": (.6, .2, .1, .1), "50/25/25 ref+NAC+CB": (.5, .25, 0, .25)}
P3 = {n: mix(w) for n, w in W.items()}
mostrar(comparar(D, {n: (p[0], p[1]) for n, p in P3.items()}, "escalón 9 (ref)"))
y = D["y"][D["diam_v"]]
assert len(y) == len(z["p_dv_ref9"])
B = np.random.default_rng(1).integers(0, len(y), size=(300, len(y)))
b0 = P3["escalón 9 (ref)"][2]
for n, p in P3.items():
    if n.startswith("escalón"):
        continue
    d = np.array([roc_auc_score(y[b], p[2][b]) - roc_auc_score(y[b], b0[b]) for b in B])
    print(f"Diamond-V {n:26s} Δ {roc_auc_score(y, p[2]) - roc_auc_score(y, b0):+.4f} [{np.percentile(d, 2.5):+.4f}; {np.percentile(d, 97.5):+.4f}]")
