"""Intervalos conformales Mondrian (por tier de la foto 2) sobre predicciones fuera de fold.

Para cada fold k, el cuantil del |error| sale de los otros folds del mismo tier; se mide la cobertura
y el ancho en el fold k. Uso: python conformal.py tag [--escala]
--escala: error normalizado por la dispersión entre expertos guardada en preds/<tag>_disp.npy (si existe).
"""
import sys
import numpy as np, pandas as pd
from comun import AQUI, cargar
tag = sys.argv[1]
c = cargar(); e = c["e"]; dev, fold = ~c["caja"], c["fold"]
p = np.load(AQUI / "preds" / f"{tag}.npy")
err = np.abs(e["z1"].to_numpy() - p)
tier = e["tier2"].to_numpy()
filas = []
for nivel in (0.8, 0.9):
    for t in ["DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER", "todos"]:
        cub, ancho = [], []
        for k in range(5):
            m_t = (tier == t) if t != "todos" else np.ones(len(e), bool)
            cal, te = dev & (fold != k) & m_t, dev & (fold == k) & m_t
            if te.sum() == 0:
                continue
            nq = cal.sum()
            q = np.quantile(err[cal], min(1, np.ceil((nq + 1) * nivel) / nq))
            cub.append((err[te] <= q).mean()); ancho.append(2 * q)
        filas.append({"nivel": nivel, "tier": t, "cobertura": np.mean(cub), "ancho_LP": np.mean(ancho)})
print(pd.DataFrame(filas).round(3).to_string(index=False))
