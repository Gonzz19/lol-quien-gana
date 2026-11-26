"""Las 4 métricas del notebook (MAE, RMSE, R² del cambio, ρ dentro del rango) para las GRU de cada nivel."""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(r"c:/Users/DarkA/OneDrive/Escritorio/Facultad/3er año/ML/TPF/ProyectoFinalML")
sys.path.insert(0, str(ROOT))
from src.features.elo_reverso import cargar_train, evaluables  # noqa

d = cargar_train(ROOT / "data")
e = evaluables(d["foto1"], d["foto2"], d["partes"])
cambio = e["z1"] - e["z2"]


def evaluar(pred):
    err = e["z1"] - pred
    rho = [(len(g), g["z1"].corr(pred[g.index], method="spearman"))
           for _, g in e.groupby("tier2") if len(g) >= 30 and pred[g.index].nunique() > 1]
    return {"MAE": err.abs().mean(), "RMSE": np.sqrt((err ** 2).mean()),
            "R² del cambio": 1 - (err ** 2).sum() / ((cambio - cambio.mean()) ** 2).sum(),
            "ρ dentro del rango": sum(n * r for n, r in rho) / sum(n for n, _ in rho)}


filas = {}
for n in ("sin_mejoras", "simple", "completo"):
    p = pd.read_csv(ROOT / "data" / "analysis" / f"elo_gru_{n}_pred.csv").set_index("player_id")["pred_gru"]
    filas[f"GRU {n}"] = evaluar(p.reindex(e.index))
print(pd.DataFrame(filas).T.round(3).to_string())
