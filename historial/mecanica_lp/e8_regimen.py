"""¿Se ve en los datos que LAS no tuvo el reinicio del 29/04? En un servidor reiniciado el LP no puede
reflejar las W/L de enero a abril; en uno sin reinicio sí. Solo lee el estado actual del en vivo y las fotos."""
import contextlib
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 240); pd.set_option("display.max_columns", 40)
SP = Path(sys.argv[1]); V = SP / "vivo"
uri = f"file:{(V / 'lp_vivo.db').as_posix()}?mode=ro&immutable=1"
with contextlib.closing(sqlite3.connect(uri, uri=True, timeout=60)) as c:
    e = pd.read_sql_query("select platform, tier, lp, wins, losses, en_ladder from estado where en_ladder=1 and tier in ('MASTER','GRANDMASTER','CHALLENGER')", c)
e["N"] = e.wins + e.losses; e["net"] = e.wins - e.losses
print("=== Ladder Master+ actual por servidor ===")
print(e.groupby("platform").agg(jug=("lp", "size"), lp_med=("lp", "median"), lp_p90=("lp", lambda s: s.quantile(.9)), lp_max=("lp", "max"), N_med=("N", "median"), net_med=("net", "median")).round(0).to_string())
print(e.pivot_table(index="platform", columns="tier", values="lp", aggfunc="min").to_string())
print("\n=== Cuánto del LP sale de las W/L de TODA la temporada ===")
for p, g in e[e.N >= 100].groupby("platform"):
    b = np.polyfit(g.net, g.lp, 1)
    k = {"la2": 20, "br1": 30, "euw1": 30, "na1": 30}[p]
    # Pendiente en unidades de K: 1 = cada neta vale lo que dice K
    print(f"{p:5s} n={len(g):6d}  corr(LP, W-L)={np.corrcoef(g.lp, g.net)[0, 1]:+.3f}  pendiente={b[0]:6.2f} LP por neta  ({b[0] / k:.2f} K)  corr(LP, N)={np.corrcoef(g.lp, g.N)[0, 1]:+.3f}")
print("\nLP según W-L (mediana):")
e2 = e[e.N >= 100].assign(b=lambda x: pd.cut(x.net, [-500, -20, 0, 20, 40, 60, 100, 150, 1000]))
print(e2.pivot_table(index="b", columns="platform", values="lp", aggfunc="median", observed=True).round(0).to_string())
print("\nFoto 1 y foto 2 de LAS, cortes y máximos (LP sobre Master 0):")
f1 = pd.read_pickle(V / "foto1.pkl"); f2 = pd.read_pickle(V / "foto2.pkl")
for n, f in (("foto1", f1), ("foto2", f2)):
    x = f[f.tier.isin(["MASTER", "GRANDMASTER", "CHALLENGER"])]
    print(n, x.groupby("tier").league_points.agg(["min", "median", "max", "size"]).to_dict("index"))
