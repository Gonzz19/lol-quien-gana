"""Descomposición del error del mejor modelo actual (promedio GRU est_plus + GB decay2_800), desarrollo."""
import pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import AQUI, cargar, evaluar, linea  # noqa
c = cargar(); e = c["e"]; dev = ~c["caja"]
with open(AQUI / "seq.pkl", "rb") as fh:
    S = pickle.load(fh)
pred = pd.Series((np.load(AQUI / "preds/est_plus.npy") + np.load(AQUI / "preds/gb_decay2_800.npy")) / 2, index=e.index)
ed = e[dev]; err = (ed["z1"] - pred[dev]); sq = err ** 2
print(linea("promedio GRU + GB (desarrollo)", evaluar(e, pred, ed.index)))
est = S["est"].loc[ed.index]
grupo = np.select([(est["piso"] == 1) & (est["dias_fin"] * 30 > 14),
                   (ed["z1"] - ed["z2"]).abs() > 400, ed["tier2"].isin(["CHALLENGER", "GRANDMASTER"]),
                   ed["tier2"] == "MASTER"],
                  ["piso + inactivo al final", "|cambio| > 400 (resto)", "GM/Challenger (resto)", "Master (resto)"],
                  "Diamond (resto)")
df = pd.DataFrame({"g": grupo, "abs": err.abs(), "sq": sq})
t = df.groupby("g").agg(n=("abs", "size"), MAE=("abs", "mean"), RMSE=("sq", lambda s: np.sqrt(s.mean())), sq=("sq", "sum"))
t["% error²"] = 100 * t["sq"] / sq.sum(); t["% del MAE total"] = 100 * t["n"] * t["MAE"] / err.abs().sum()
print(t.drop(columns="sq").round(1).to_string())
print("\nSi el grupo 'piso + inactivo' tuviera el MAE del resto:",
      round(((err.abs()[grupo != "piso + inactivo al final"]).mean()), 1))
