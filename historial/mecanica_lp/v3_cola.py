"""¿Qué son las victorias de +40/+60 y las derrotas de 0? Artefacto de réplicas o mecánica real."""
import contextlib
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
SP = Path(sys.argv[1]); ROOT = Path(sys.argv[2])
sys.path.insert(0, str(ROOT))
from src.features.rating import lp_absoluto  # noqa: E402

uri = f"file:{(SP / 'vivo' / 'lp_vivo.db').as_posix()}?mode=ro&immutable=1"
with contextlib.closing(sqlite3.connect(uri, uri=True, timeout=60)) as c:
    ev = pd.read_sql_query("select id, platform, puuid, t_antes, t_despues, tier_antes, rank_antes, lp_antes, wins_antes, losses_antes, tier_despues, rank_despues, lp_despues, wins_despues, losses_despues, tipo, estado, match_id from eventos where t_antes is not null", c)
    jug = pd.read_sql_query("select j.match_id, j.puuid, j.team_id, j.win, j.kills, j.deaths, j.assists, j.gold, j.cs, j.team_position, p.duracion_s, p.game_end, p.game_start from participantes j join partidas p on p.match_id=j.match_id", c)
for lado in ("antes", "despues"):
    ev[f"z_{lado}"] = [lp_absoluto(t, r, l) for t, r, l in zip(ev[f"tier_{lado}"], ev[f"rank_{lado}"], ev[f"lp_{lado}"])]
ev["dz"] = ev.z_despues - ev.z_antes
ev["dw"] = ev.wins_despues - ev.wins_antes
ev["dl"] = ev.losses_despues - ev.losses_antes
ev = ev.sort_values(["platform", "puuid", "t_despues", "id"])

las = ev[(ev.platform == "la2") & (ev.estado == "enlazada")]
print("=== LAS: valores exactos de ΔLP ===")
print("victorias:", las[las.dw == 1].dz.value_counts().sort_index().loc[lambda s: s >= 15].to_dict())
print("derrotas:", las[las.dl == 1].dz.value_counts().sort_index().loc[lambda s: s >= -40].to_dict())

# Evento anterior y siguiente del mismo jugador
g = ev.groupby(["platform", "puuid"])
for col in ("tipo", "dz", "t_despues", "z_despues", "dw", "dl", "estado"):
    ev[f"prev_{col}"] = g[col].shift(1)
    ev[f"next_{col}"] = g[col].shift(-1)
ev["min_desde_prev"] = (ev.t_despues - ev.prev_t_despues) / 60000
ev["min_hasta_next"] = (ev.next_t_despues - ev.t_despues) / 60000
ev["ventana"] = (ev.t_despues - ev.t_antes) / 60000

def clase(x):
    if x.tipo != "partida" or x.estado != "enlazada":
        return None
    if x.dw == 1:
        return "W normal" if 15 <= x.dz <= 27 else ("W alta" if x.dz > 27 else "W baja")
    return "L normal" if -27 <= x.dz <= -15 else ("L cero" if x.dz == 0 else ("L chica" if x.dz > -15 else "L grande"))

e = ev[(ev.platform == "la2") & (ev.tipo == "partida") & (ev.estado == "enlazada")].copy()
e["clase"] = [clase(x) for x in e.itertuples()]
print("\nclases LAS:", e.clase.value_counts().to_dict())

print("\n=== Evento inmediatamente anterior del mismo jugador (LAS) ===")
e["prev_cat"] = np.where(e.min_desde_prev.isna(), "ninguno",
                  np.where(e.min_desde_prev > 25, "lejos(>25min)", e.prev_tipo.fillna("") + np.where(e.prev_tipo == "sin_partida", ":" + e.prev_dz.round().astype("Int64").astype(str), "")))
top = e.groupby("clase").prev_cat.value_counts(normalize=True).groupby(level=0).head(6).round(3)
print(top.to_string())

print("\n=== Evento inmediatamente siguiente (LAS) ===")
e["next_cat"] = np.where(e.min_hasta_next.isna(), "ninguno",
                  np.where(e.min_hasta_next > 25, "lejos(>25min)", e.next_tipo.fillna("") + np.where(e.next_tipo == "sin_partida", ":" + e.next_dz.round().astype("Int64").astype(str), "")))
print(e.groupby("clase").next_cat.value_counts(normalize=True).groupby(level=0).head(6).round(3).to_string())

# Coherencia por partida
m = e.merge(jug[["match_id", "puuid", "team_id", "win", "kills", "deaths", "assists", "gold", "cs", "duracion_s", "team_position"]], on=["match_id", "puuid"], how="left")
print("\n=== Duración, rendimiento por clase (LAS) ===")
m["kda"] = (m.kills + m.assists) / m.deaths.clip(lower=1)
print(m.groupby("clase")[["duracion_s", "kda", "gold", "cs", "ventana", "z_antes"]].median().round(1).to_string())

por_lado = m.groupby(["match_id", "team_id", "win"]).agg(n=("dz", "size"), altos=("clase", lambda s: (s == "W alta").sum()), ceros=("clase", lambda s: (s == "L cero").sum()), chicas=("clase", lambda s: (s == "L chica").sum())).reset_index()
print("\n=== ¿Es un evento del equipo? victorias: cuántos 'W alta' por equipo con ≥3 medidos ===")
print(por_lado[(por_lado.win == 1) & (por_lado.n >= 3)].altos.value_counts().sort_index().to_dict())
print("derrotas: cuántos 'L cero' por equipo con ≥3 medidos:", por_lado[(por_lado.win == 0) & (por_lado.n >= 3)].ceros.value_counts().sort_index().to_dict())

# Por día/hora
m["dia"] = pd.to_datetime(m.t_despues, unit="ms").dt.strftime("%m-%d %H")
m["dia6"] = pd.to_datetime(m.t_despues, unit="ms").dt.floor("6h").dt.strftime("%m-%d %H")
t = m.pivot_table(index="dia6", columns="clase", values="dz", aggfunc="size", fill_value=0)
print("\n=== Por tramos de 6 h (LAS) ===\n", t.to_string())

# ¿Por jugador? ¿gap? ¿racha?
m = m.sort_values(["puuid", "t_despues"])
m["res_prev"] = m.groupby("puuid").dw.shift(1)
m["res_prev2"] = m.groupby("puuid").dw.shift(2)
w = m[m.dw == 1]
print("\nW alta según la anterior:", w.groupby("res_prev").clase.apply(lambda s: (s == "W alta").mean()).round(3).to_dict(),
      "| dos anteriores:", w.groupby(["res_prev", "res_prev2"]).clase.apply(lambda s: (s == "W alta").mean()).round(3).to_dict())
l = m[m.dl == 1]
print("L cero según la anterior:", l.groupby("res_prev").clase.apply(lambda s: (s == "L cero").mean()).round(3).to_dict())

# Sumas: dos eventos seguidos cerca que se cancelen. Cada W alta: ¿el salto previo fue negativo?
print("\n=== W alta: prev_dz cuando el anterior está a <25 min ===")
wa = e[(e.clase == "W alta") & (e.min_desde_prev < 25)]
print(wa.prev_dz.round().value_counts().head(10).to_dict(), "n", len(wa), "de", (e.clase == "W alta").sum())
print("L cero: prev_dz cuando el anterior está a <25 min")
lc = e[(e.clase == "L cero") & (e.min_desde_prev < 25)]
print(lc.prev_dz.round().value_counts().head(10).to_dict(), "n", len(lc), "de", (e.clase == "L cero").sum())
print("L cero: next_dz cuando el siguiente está a <25 min")
lc = e[(e.clase == "L cero") & (e.min_hasta_next < 25)]
print(lc.next_dz.round().value_counts().head(10).to_dict(), "n", len(lc))
print("W alta: next_dz a <25 min")
wa = e[(e.clase == "W alta") & (e.min_hasta_next < 25)]
print(wa.next_dz.round().value_counts().head(10).to_dict(), "n", len(wa))

print("\n=== Ejemplos de secuencias LAS alrededor de una W alta y una L cero ===")
cols = ["t_despues", "tipo", "estado", "lp_antes", "lp_despues", "tier_despues", "wins_antes", "losses_antes", "wins_despues", "losses_despues", "dz"]
for cl in ("W alta", "L cero"):
    for _, fila in e[e.clase == cl].sample(4, random_state=3).iterrows():
        h = ev[(ev.platform == "la2") & (ev.puuid == fila.puuid)]
        i = h.index.get_loc(fila.name)
        v = h.iloc[max(0, i - 3): i + 4][cols].copy()
        v["t_despues"] = pd.to_datetime(v.t_despues, unit="ms").dt.strftime("%d %H:%M")
        print(cl, fila.match_id); print(v.to_string(index=False))

ev.to_pickle(SP / "vivo" / "ev_todos.pkl"); m.to_pickle(SP / "vivo" / "m_las.pkl")
