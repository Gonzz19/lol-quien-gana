"""Combinación de predicciones fuera de fold. Pesos ajustados sin el fold evaluado (anidado).

Uso: python mezcla.py tag1 tag2 [tag3 ...] [--l1]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import AQUI, cargar, evaluar, registrar  # noqa

tags = [t for t in sys.argv[1:] if not t.startswith("--")]
c = cargar()
e = c["e"]
dev, fold = ~c["caja"], c["fold"]
P = np.column_stack([np.load(AQUI / "preds" / f"{t}.npy") for t in tags])
z1 = e["z1"].to_numpy()


def pesos(filas):
    # Pesos no negativos que suman 1, minimizando el MAE (lo que mide el objetivo).
    f = lambda w: np.abs(z1[filas] - P[filas] @ (np.abs(w) / np.abs(w).sum())).mean()  # noqa: E731
    w = minimize(f, np.ones(len(tags)) / len(tags), method="Nelder-Mead", options={"maxiter": 2000}).x
    return np.abs(w) / np.abs(w).sum()


final = np.full(len(e), np.nan)
for k in range(5):
    w = pesos(dev & (fold != k))
    te = dev & (fold == k)
    final[te] = P[te] @ w
w_todos = pesos(dev)
registrar("mezcla(" + "+".join(tags) + ")", evaluar(e, pd.Series(final, index=e.index), e.index[dev]),
          "pesos " + ", ".join(f"{t} {x:.2f}" for t, x in zip(tags, w_todos)))
promedio = pd.Series(P.mean(1), index=e.index)
registrar("promedio(" + "+".join(tags) + ")", evaluar(e, promedio, e.index[dev]), "pesos iguales")
