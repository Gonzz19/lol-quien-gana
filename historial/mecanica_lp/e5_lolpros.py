"""lolpros: ¿cuándo se reinicia W/L en la temporada 2026, con qué LP se arranca, y cómo es la subida?"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 200)
SP = Path(sys.argv[1]); V = SP / "vivo"
h = pd.read_pickle(V / "lolpros.pkl")
h["t"] = pd.to_datetime(h.created_at, utc=True)
h = h.dropna(subset=["wins", "losses", "tier"]).sort_values(["cuenta", "t"])
print("rango temporal:", h.t.min(), h.t.max(), "| filas", len(h), "| cuentas", h.cuenta.nunique())
print("temporadas:", h.groupby("season").t.agg(["min", "max", "size"]).to_string())
print("tier:", h.tier.value_counts().to_dict())
h["N"] = h.wins + h.losses
h["dN"] = h.groupby("cuenta").N.diff()
print("\nReinicios (W+L baja):")
print(h[h.dN < 0][["cuenta", "t", "season", "tier", "division", "league_points", "wins", "losses", "dN"]].to_string())

# Escala
ORD = {"70_challenger": "C", "60_grandmaster": "G", "20_master": "M"}
print("\nscore contra tier: ", h.groupby("tier").score.agg(["min", "max"]).to_string())
d = h[h.season == h.season.max()]
print("\nSeason más nueva:", h.season.max())
for c, g in d.groupby("cuenta"):
    g = g.sort_values("t")
    prim = g.iloc[0]; ult = g.iloc[-1]
    # primeros registros
    print(f"{c}: {len(g)} filas | inicio {prim.t.date()} {prim.tier} d{prim.division} {prim.league_points} LP W{prim.wins} L{prim.losses} score {prim.score} | fin {ult.t.date()} {ult.tier} {ult.league_points} W{ult.wins} L{ult.losses} score {ult.score}")
d2 = d.assign(net=d.wins - d.losses)
d2["z"] = d2.score - 4000 + 400  # score 4560 = Master 560 → z = 960 en la escala del proyecto (400 + LP)
print("\nprimer día de la temporada más nueva por cuenta (primeras 4 filas de 3 cuentas):")
for c, g in list(d2.groupby("cuenta"))[:3]:
    print(g.sort_values("t").head(6)[["t", "tier", "division", "league_points", "wins", "losses", "score"]].to_string())

# Cierre de temporada: en cada cuenta, Δscore contra K(W-L)
print("\nCierre por cuenta, temporada más nueva: z - z_ini contra 20·(W-L)/30·(W-L)")
filas = []
for c, g in d2.groupby("cuenta"):
    g = g.sort_values("t")
    if len(g) < 20:
        continue
    a, b = g.iloc[0], g.iloc[-1]
    filas.append(dict(cuenta=c, N=b.N - a.N, net=(b.wins - b.losses) - (a.wins - a.losses), dscore=b.score - a.score, por_partida=(b.score - a.score) / max(b.N - a.N, 1), score_ini=a.score, score_fin=b.score))
print(pd.DataFrame(filas).round(2).to_string())
