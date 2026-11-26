"""¿De dónde viene el error? Por rango, por fracción de no observadas y por partidas entre fotos."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import cargar  # noqa

ROOT = Path(r"c:/Users/DarkA/OneDrive/Escritorio/Facultad/3er año/ML/TPF/ProyectoFinalML")
c = cargar()
e = c["e"]
oof = pd.read_csv(ROOT / "data" / "analysis" / "boost_completo_oof.csv", index_col="player_id").reindex(e.index)
pred = c["regla"] + oof["gb"]
err = e["z1"] - pred
frac = e["ocultas"] / e["dN"]
df = pd.DataFrame({"err": err, "abs": err.abs(), "sq": err ** 2, "tier": e["tier2"],
                   "frac_oc": pd.cut(frac, [-0.01, 0.02, 0.1, 0.3, 0.6, 1.0]),
                   "dN": pd.cut(e["dN"], [0, 10, 25, 50, 100, 200, 10_000]), "ocultas": e["ocultas"]})
tot_sq = df["sq"].sum()
for col in ("tier", "frac_oc", "dN"):
    g = df.groupby(col, observed=True).agg(n=("abs", "size"), MAE=("abs", "mean"),
                                           RMSE=("sq", lambda s: np.sqrt(s.mean())), sq=("sq", "sum"))
    g["% del error²"] = 100 * g["sq"] / tot_sq
    print(g.drop(columns="sq").round(1).to_string(), "\n")
# Sin ocultas: ¿cuánto error queda cuando vemos todas las partidas?
for lim in (0, 2, 5):
    m = e["ocultas"] <= lim
    print(f"ocultas <= {lim}: n={m.sum()}, MAE {df['abs'][m].mean():.1f}, RMSE {np.sqrt(df['sq'][m].mean()):.1f}")
print("cola: % del error² en el 5 % peor:", round(100 * df["sq"].nlargest(len(df) // 20).sum() / tot_sq, 1))
