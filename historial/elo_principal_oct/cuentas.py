"""¿La foto 1 y la foto 2 de un jugador son de la misma cuenta? ¿Hay cambios de LP imposibles?"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import cargar  # noqa

ROOT = Path(r"c:/Users/DarkA/OneDrive/Escritorio/Facultad/3er año/ML/TPF/ProyectoFinalML")
c = cargar()
e = c["e"]
db = f"file:{(ROOT / 'data' / 'lol_las.db').resolve()}?mode=ro"
with sqlite3.connect(db, uri=True) as con:
    f1 = pd.read_sql_query("SELECT m.player_id, p.puuid, p.wins + p.losses AS n, p.updated_at "
                           "FROM players p JOIN player_id_map m ON m.puuid = p.puuid", con)
    f2 = pd.read_sql_query("SELECT m.player_id, s.puuid, s.wins + s.losses AS n FROM ladder_snapshots s "
                           "JOIN player_id_map m ON m.puuid = s.puuid WHERE s.snapshot_at = '2026-10-28 00:49:59'", con)
p1 = f1.sort_values("n").drop_duplicates("player_id", keep="last").set_index("player_id")
p2 = f2.sort_values("n").drop_duplicates("player_id", keep="last").set_index("player_id")
cuentas1 = f1.groupby("player_id").size()
cuentas2 = f2.groupby("player_id").size()
j = pd.DataFrame({"puuid1": p1["puuid"], "puuid2": p2["puuid"], "c1": cuentas1, "c2": cuentas2,
                  "upd": pd.to_datetime(p1["updated_at"])}).reindex(e.index)
distinta = j["puuid1"] != j["puuid2"]
print(f"jugadores evaluables: {len(e)}; con >1 cuenta en foto 1: {(j['c1'] > 1).sum()}, en foto 2: {(j['c2'] > 1).sum()}")
print(f"foto 1 y foto 2 de cuentas DISTINTAS: {distinta.sum()}")

oof = pd.read_csv(ROOT / "data" / "analysis" / "boost_completo_oof.csv", index_col="player_id").reindex(e.index)
err = e["z1"] - (c["regla"] + oof["gb"])
sq = err ** 2
print(f"  error de esos: MAE {err[distinta].abs().mean():.1f}, % del error² {100 * sq[distinta].sum() / sq.sum():.1f}")
print(f"  error del resto: MAE {err[~distinta].abs().mean():.1f}, RMSE {np.sqrt(sq[~distinta].mean()):.1f}")

# Cambio de LP imposible para la cantidad de partidas (≤ 35 LP por partida + 300 de decay)
imposible = (e["z1"] - e["z2"]).abs() > 35 * e["dN"].clip(lower=0) + 300
print(f"cambio imposible para sus partidas (misma cuenta): {(imposible & ~distinta).sum()}")
print("fecha de la foto 1 (updated_at):", j["upd"].min(), "→", j["upd"].max())
print(j["upd"].dt.date.value_counts().sort_index().tail(10).to_string())
print("dN negativo o raro:", (e["dN"] <= 0).sum(), "| obs > dN + 5:", (e["obs"] > e["dN"] + 5).sum())
peor = sq.nlargest(len(e) // 20).index
print("en el 5 % peor: cuentas distintas", distinta[peor].mean().round(3), "| obs > dN+5",
      (e.loc[peor, "obs"] > e.loc[peor, "dN"] + 5).mean().round(3))
print(e.loc[peor[:12], ["tier1", "z1", "tier2", "z2", "dN", "obs", "ocultas", "partidas1", "partidas2"]]
      .assign(err=err[peor[:12]].round(0), distinta=distinta[peor[:12]]).to_string())
