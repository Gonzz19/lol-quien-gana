"""Brainstorming (docs/BRAINSTORM_NOCHE.md): ganancia total de la noche en V Master+, con IC.
Referencia de §9.S.51 (híbrido + mapa, entrenado con todo antes del 16/09) contra el candidato final (escalón 6 sin mapa,
`pred_final.npz`). ΔAUC y Δacierto con bootstrap pareado (B = 1000) y McNemar exacto para el acierto."""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.brainstorm_base import REPO, ajustar_predecir, cargar  # noqa: E402

D = cargar()
vi = np.flatnonzero(D["val"])
p0 = ajustar_predecir(D, D["fi_v"], vi)                    # §9.S.51: híbrido + mapa, todo el entrenamiento
F = np.load(REPO / "data" / "analysis" / "brainstorm" / "pred_final.npz")
assert (F["idx_v"] == vi).all()
p1, y = F["p_v"], D["y"][vi]
o0, o1 = (p0 > .5) == y, (p1 > .5) == y
B = np.random.default_rng(7).integers(0, len(vi), size=(1000, len(vi)))
da = np.array([roc_auc_score(y[b], p1[b]) - roc_auc_score(y[b], p0[b]) for b in B])
dc = np.array([o1[b].mean() - o0[b].mean() for b in B])
print(f"referencia §9.S.51: AUC {roc_auc_score(y, p0):.4f}, acierto {o0.mean():.2%}")
print(f"candidato final:    AUC {roc_auc_score(y, p1):.4f}, acierto {o1.mean():.2%}")
print(f"ΔAUC {roc_auc_score(y, p1) - roc_auc_score(y, p0):+.4f} [{np.percentile(da, 2.5):+.4f}; {np.percentile(da, 97.5):+.4f}]")
print(f"Δacierto {o1.mean() - o0.mean():+.2%} [{np.percentile(dc, 2.5):+.2%}; {np.percentile(dc, 97.5):+.2%}]")
b01, b10 = int((~o0 & o1).sum()), int((o0 & ~o1).sum())
print(f"McNemar: el final acierta y la referencia no en {b01}, al revés en {b10}; p = {binomtest(b01, b01 + b10).pvalue:.4f}")
