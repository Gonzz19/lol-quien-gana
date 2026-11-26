"""Sumas de gap (lobby − LP del jugador en esa partida): ¿explican el residuo de la regla?"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import HuberRegressor, LinearRegression
sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import cargar, evaluar, linea  # noqa
c = cargar(); e, partes = c["e"], c["partes"]
dev = ~c["caja"]; fold = c["fold"]
p = partes[["player_id", "win"]].assign(gap=(c["lobp"]["lobby"] - c["lp_fila"]) / 100, lp=c["lp_fila"])
p = p[p["player_id"].isin(e.index)].dropna(subset=["gap"])
p["M"] = p["lp"] >= 400
F = {}
for nom, m in (("w_M", (p.win == 1) & p.M), ("l_M", (p.win != 1) & p.M), ("w_D", (p.win == 1) & ~p.M), ("l_D", (p.win != 1) & ~p.M)):
    F["S" + nom] = p.loc[m].groupby("player_id")["gap"].sum()
F = pd.DataFrame(F).reindex(e.index).fillna(0)
y = e["z1"] - c["regla"]
print(linea("regla sola (desarrollo)", evaluar(e, c["regla"], e.index[dev])))
for nombre, Mod in (("OLS", LinearRegression), ("Huber", lambda: HuberRegressor(epsilon=1.35, max_iter=500))):
    pred = pd.Series(np.nan, index=e.index)
    for k in range(5):
        tr, te = dev & (fold != k), dev & (fold == k)
        mod = Mod().fit(F[tr], y[tr]); pred[te] = mod.predict(F[te])
    print(linea(f"regla + {nombre}(sumas de gap)", evaluar(e, c["regla"] + pred, e.index[dev])),
          dict(zip(F.columns, np.round(Mod().fit(F[dev], y[dev]).coef_, 1))))
F.to_pickle(Path(__file__).resolve().parent / "sumas_gap.pkl")
