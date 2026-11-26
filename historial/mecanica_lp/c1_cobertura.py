"""Cuántas partidas faltan, por período y rango. Solo CUENTA filas antes de la foto 1: no lee
resultados, ni LP, ni nada del test más allá de cuántas partidas hay y de quién son."""
import contextlib
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(sys.argv[1])
uri = f"file:{(ROOT / 'data' / 'lol_las.db').resolve().as_posix()}?mode=ro"
FOTO2 = "2026-10-28 00:49:59"

with contextlib.closing(sqlite3.connect(uri, uri=True, timeout=60)) as c:
    t1_txt = c.execute("select max(updated_at) from players").fetchone()[0]
    upd = pd.read_sql_query("select substr(updated_at,1,10) d, count(*) n from players group by 1 order by 1", c)
    foto1 = pd.read_sql_query("select m.player_id, p.puuid, p.tier, p.wins, p.losses from players p join player_id_map m on m.puuid=p.puuid", c)
    foto2 = pd.read_sql_query("select m.player_id, s.tier, s.wins, s.losses, s.fuente from ladder_snapshots s join player_id_map m on m.puuid=s.puuid where s.snapshot_at=?", c, params=(FOTO2,))
    pm = pd.read_sql_query(
        "select mp.match_id, m.player_id, x.game_creation from match_participants mp "
        "join player_id_map m on m.puuid=mp.puuid join matches x on x.match_id=mp.match_id "
        "where mp.match_id like 'LA2_%'", c)
    win = pd.read_sql_query("select distinct window_key from crawl_window_matches", c)
    print("ventanas:", win.window_key.tolist())
    for k in win.window_key:
        print(k, c.execute("select count(*) from crawl_window_matches where window_key=?", (k,)).fetchone(),
              c.execute("select count(*) from crawl_window_players where window_key=?", (k,)).fetchone(),
              c.execute("select sum(n_ids), min(listed_at), max(listed_at) from crawl_window_players where window_key=?", (k,)).fetchone(),
              c.execute("select count(*) from crawl_window_matches w where window_key=? and match_id in (select match_id from matches)", (k,)).fetchone(),
              c.execute("select count(*) from crawl_window_matches w where window_key=? and match_id in (select match_id from discarded_matches)", (k,)).fetchone())

print("foto 1:", t1_txt, upd.tail(8).to_dict("records"))
t1 = int(pd.Timestamp(t1_txt, tz="UTC").value // 10**6)
t2 = int(pd.Timestamp(FOTO2, tz="UTC").value // 10**6)
print("primera/última partida:", pd.to_datetime(pm.game_creation.min(), unit="ms"), pd.to_datetime(pm.game_creation.max(), unit="ms"))
pm["periodo"] = np.where(pm.game_creation < t1, "antes", np.where(pm.game_creation < t2, "entre", "despues"))
m = pm.drop_duplicates("match_id")
print("partidas únicas LAS por período:", m.periodo.value_counts().to_dict(), "filas:", pm.periodo.value_counts().to_dict())
print("partidas por mes:", m.assign(mes=pd.to_datetime(m.game_creation, unit="ms").dt.strftime("%Y-%m")).groupby("mes").size().to_dict())

f1 = foto1.assign(N=foto1.wins + foto1.losses).sort_values("N").drop_duplicates("player_id", keep="last").set_index("player_id")
f2 = foto2[foto2.tier != "UNRANKED"].assign(N=lambda d: d.wins + d.losses).sort_values("N").drop_duplicates("player_id", keep="last").set_index("player_id")
obs = pm.groupby(["player_id", "periodo"]).size().unstack(fill_value=0).reindex(columns=["antes", "entre", "despues"], fill_value=0)

ORDEN = ["CHALLENGER", "GRANDMASTER", "MASTER", "DIAMOND"]
print("\n== ANTES de la foto 1 (por tier de la foto 1) ==")
a = f1[["tier", "N"]].join(obs["antes"].rename("n")).fillna({"n": 0})
a["oc"] = (a.N - a.n).clip(lower=0)
a["cob"] = a.n / a.N.where(a.N > 0)
print(a.groupby("tier").agg(jug=("N", "size"), N_tot=("N", "sum"), N_med=("N", "median"), obs=("n", "sum"), ocultas=("oc", "sum"), cob_med=("cob", "median"),
                            cob_p10=("cob", lambda s: s.quantile(.1)), cob_p90=("cob", lambda s: s.quantile(.9))).reindex(ORDEN).round(3).to_string())
print("totales: N", a.N.sum(), "obs", a.n.sum(), "ocultas", a.oc.sum(), "n>N en", (a.n > a.N).sum())

print("\n== ENTRE fotos (jugadores con las dos fotos, por tier de la foto 2) ==")
e = f1[["N"]].join(f2[["tier", "N"]], lsuffix="1", rsuffix="2", how="inner").join(obs["entre"].rename("n")).fillna({"n": 0})
e["dN"] = e.N2 - e.N1
e = e[e.dN > 0]
e["oc"] = (e.dN - e.n).clip(lower=0)
e["cob"] = (e.n / e.dN).clip(upper=1)
print(e.groupby("tier").agg(jug=("dN", "size"), dN=("dN", "sum"), dN_med=("dN", "median"), obs=("n", "sum"), ocultas=("oc", "sum"), cob_med=("cob", "median"),
                            cob_p10=("cob", lambda s: s.quantile(.1))).reindex(ORDEN).round(3).to_string())
print("totales: dN", e.dN.sum(), "obs", e.n.sum(), "ocultas", e.oc.sum())

# Solapamiento: cuántos jugadores del padrón hay por partida observada
padron = set(f1.index)
pm["padron"] = pm.player_id.isin(padron)
pp = pm.groupby(["match_id", "periodo"]).padron.sum().reset_index()
print("\njugadores del padrón por partida observada:", pp.groupby("periodo").padron.describe()[["mean", "50%"]].round(2).to_dict("index"))
# jugadores en el dataset y cuántos no tienen ninguna foto
todos = set(pm.player_id)
print("jugadores en partidas LAS:", len(todos), "con foto 1:", len(todos & set(f1.index)), "con foto 2 (rankeada):", len(todos & set(f2.index)),
      "con alguna:", len(todos & (set(f1.index) | set(f2.index))))
sin = pm[~pm.player_id.isin(set(f1.index) | set(f2.index))]
print("filas LAS de jugadores sin ninguna foto:", len(sin), f"{len(sin) / len(pm):.1%}", "por período", sin.periodo.value_counts().to_dict())
por_pl = pm.groupby("player_id").size()
print("filas de jugadores con foto 1:", pm.player_id.isin(set(f1.index)).mean().round(3), "con foto 2:", pm.player_id.isin(set(f2.index)).mean().round(3))
print("foto 2 fuente:", foto2.fuente.value_counts().to_dict(), "tier", foto2.tier.value_counts().to_dict())
