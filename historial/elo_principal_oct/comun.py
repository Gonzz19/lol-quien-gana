"""Carga del cache y métricas del objetivo (las 4 del notebook)."""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent


def cargar():
    with open(AQUI / "cache.pkl", "rb") as f:
        return pickle.load(f)


def evaluar(e: pd.DataFrame, pred: pd.Series, idx=None) -> dict:
    idx = e.index if idx is None else idx
    pred = pd.Series(pred, index=e.index) if not isinstance(pred, pd.Series) else pred
    err = e.loc[idx, "z1"] - pred[idx]
    ca = (e["z1"] - e["z2"])[idx]
    rho = [(len(g), g["z1"].corr(pred[g.index], method="spearman"))
           for _, g in e.loc[idx].groupby("tier2") if len(g) >= 30 and pred[g.index].nunique() > 1]
    return {"MAE": err.abs().mean(), "RMSE": np.sqrt((err ** 2).mean()),
            "R2": 1 - (err ** 2).sum() / ((ca - ca.mean()) ** 2).sum(),
            "rho": sum(n * r for n, r in rho) / sum(n for n, _ in rho) if rho else np.nan}


def linea(nombre: str, m: dict) -> str:
    return f"{nombre:<45} MAE {m['MAE']:6.1f} | RMSE {m['RMSE']:6.1f} | R2 {m['R2']:.3f} | rho {m['rho']:.3f}"


def registrar(nombre: str, m: dict, nota: str = "") -> None:
    """Agrega una fila al registro de experimentos (desarrollo, 5 folds)."""
    with open(AQUI / "registro.tsv", "a", encoding="utf-8") as f:
        f.write(f"{nombre}\t{m['MAE']:.1f}\t{m['RMSE']:.1f}\t{m['R2']:.3f}\t{m['rho']:.3f}\t{nota}\n")
    print(linea(nombre, m), nota, flush=True)
