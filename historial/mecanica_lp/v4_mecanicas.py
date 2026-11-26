"""¿Qué explica cada componente del ΔLP, con variables que TAMBIÉN existen en el dataset de LAS?"""
import contextlib
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 220)
SP = Path(sys.argv[1]); ROOT = Path(sys.argv[2])
sys.path.insert(0, str(ROOT))
from src.features.rating import lp_absoluto  # noqa: E402

uri = f"file:{(SP / 'vivo' / 'lp_vivo.db').as_posix()}?mode=ro&immutable=1"
with contextlib.closing(sqlite3.connect(uri, uri=True, timeout=60)) as c:
    d = pd.read_sql_query(
        "select e.platform, e.puuid, e.match_id, e.t_antes, e.t_despues, e.tier_antes, e.rank_antes, e.lp_antes, e.tier_despues, e.rank_despues, e.lp_despues, e.wins_antes, e.losses_antes "
        "from eventos e where e.estado='enlazada'", c)
    j = pd.read_sql_query("select j.match_id, j.puuid, j.team_id, j.win, j.champion, j.team_position, j.kills, j.deaths, j.assists, j.gold, j.cs, p.duracion_s, p.platform, p.game_end from participantes j join partidas p on p.match_id=j.match_id", c)
for lado in ("antes", "despues"):
    d[f"z_{lado}"] = [lp_absoluto(t, r, l) for t, r, l in zip(d[f"tier_{lado}"], d[f"rank_{lado}"], d[f"lp_{lado}"])]
d["dz"] = d.z_despues - d.z_antes
d = d.drop(columns="platform").merge(j, on=["match_id", "puuid"], how="left")

# Variables de equipo: AFK / abandono según oro y cs
j["gold_min"] = j.groupby(["match_id", "team_id"]).gold.transform("min")
j["gold_med"] = j.groupby(["match_id", "team_id"]).gold.transform("median")
j["kda_part"] = j.kills + j.assists
eq = j.groupby(["match_id", "team_id"]).agg(gold_min=("gold", "min"), gold_med=("gold", "median"), n_pos=("team_position", "nunique")).reset_index()
eq["ratio_min"] = eq.gold_min / eq.gold_med
d = d.merge(eq, on=["match_id", "team_id"], how="left")
d["soy_min"] = d.gold == d.gold_min
d["rival_ratio"] = d.merge(eq.assign(team_id=lambda x: 300 - x.team_id)[["match_id", "team_id", "ratio_min"]], on=["match_id", "team_id"], how="left", suffixes=("", "_riv"))["ratio_min_riv"].to_numpy()

# Rol habitual: moda en estos días
d["rol_jug"] = d.groupby(["platform", "puuid"]).team_position.transform(lambda s: s.mode().iat[0] if len(s.mode()) else None)
n_j = d.groupby(["platform", "puuid"]).dz.transform("size")
d["frac_rol"] = d.groupby(["platform", "puuid", "team_position"]).dz.transform("size") / n_j
d["fuera_rol"] = d.team_position != d.rol_jug

# LAS: componente por valor normalizado respecto de la mediana del servidor
base = {"la2": (20, -20.5), "br1": (30, -31), "euw1": (29, -32), "na1": (30, -31)}
d["mult"] = [dz / (base[p][0] if w == 1 else -base[p][1]) for dz, p, w in zip(d.dz, d.platform, d.win)]

print("=== Multiplicador de la ganancia (ΔLP / mediana del servidor), victorias ===")
w = d[d.win == 1]
print(pd.crosstab(w.platform, pd.cut(w.mult, [-5, .5, .8, 1.2, 1.4, 1.65, 1.8, 2.3, 5])).to_string())
print("=== Multiplicador, derrotas (positivo = pierde) ===")
l = d[d.win == 0]
print(pd.crosstab(l.platform, pd.cut(l.mult, [-5, -.01, .05, .3, .6, .8, 1.2, 1.4, 5])).to_string())

w = w.assign(doble=(w.mult > 1.65) & (w.mult < 2.3), triple=w.mult >= 2.3, tramo=(w.mult > 1.4) & (w.mult <= 1.65))
print("\n=== Doble (victoria) por rol y por servidor ===")
print(w.pivot_table(index="team_position", columns="platform", values="doble", aggfunc="mean").round(3).to_string())
print("\nDoble según fuera de rol (jugadores con ≥8 partidas enlazadas):")
ww = w[n_j.loc[w.index] >= 8]
print(ww.groupby(["platform", "fuera_rol"]).doble.agg(["mean", "size"]).round(3).to_string())
print("\nDoble según la fracción de partidas en ese rol:")
print(ww.groupby(pd.cut(ww.frac_rol, [0, .1, .2, .4, .6, .8, 1])).doble.agg(["mean", "size"]).round(3).to_string())

print("\n=== Persistencia por jugador del doble (jugadores con ≥8 victorias) ===")
pj = w.groupby(["platform", "puuid"]).doble.agg(["mean", "size"]); pj = pj[pj["size"] >= 8]
pbar = pj["mean"].mean()
print("n", len(pj), "media", round(pbar, 3), "var observada", round(pj["mean"].var(), 4), "var binomial esperada", round((pbar * (1 - pbar) / pj["size"]).mean(), 4))
print("cuantiles tasa:", pj["mean"].quantile([.1, .25, .5, .75, .9, .95, .99]).round(3).to_dict())

print("\n=== Doble contra rendimiento: oro relativo al equipo, kda ===")
w = w.assign(oro_rel=w.gold / w.gold_med, kp=(w.kills + w.assists))
print(w.groupby(pd.qcut(w.oro_rel, 5, duplicates="drop"), observed=True).doble.mean().round(3).to_dict())
print("por rival con AFK (ratio del oro mínimo rival):", w.groupby(pd.cut(w.rival_ratio, [0, .3, .5, .7, .85, 1.01]), observed=True).doble.agg(["mean", "size"]).round(3).to_dict("index"))
print("Por duración:", w.groupby(pd.cut(w.duracion_s, [0, 900, 1200, 1500, 1800, 2100, 2400, 4000]), observed=True).doble.mean().round(3).to_dict())
print("Por z de antes:", w.groupby(pd.cut(w.z_antes, [0, 400, 450, 500, 600, 800, 1000, 1300, 1800, 4000]), observed=True).doble.agg(["mean", "size"]).round(3).to_dict("index"))
print("Por N (W+L) del jugador:", w.assign(N=w.wins_antes + w.losses_antes).groupby(pd.cut(w.wins_antes + w.losses_antes, [0, 50, 100, 200, 300, 500, 800, 3000]), observed=True).doble.agg(["mean", "size"]).round(3).to_dict("index"))
print("Por campeón (top 8):", w.groupby("champion").doble.agg(["mean", "size"]).query("size>=100").sort_values("mean", ascending=False).head(8).round(3).to_dict("index"))

print("\n=== Derrota: equipos con oro mínimo bajo (posible AFK) y 0 LP ===")
l = l.assign(cero=l.dz == 0, soy_min=l.gold == l.gold_min)
for plat, x in l.groupby("platform"):
    print(plat, x.groupby(pd.cut(x.ratio_min, [0, .2, .3, .4, .5, .6, .7, .85, 1.01]), observed=True).cero.agg(["mean", "size"]).round(3).T.to_string())
print("\nLAS, derrotas en equipos con oro mínimo <0,4 del mediano: ΔLP del que tiene el mínimo y de los otros")
x = l[(l.platform == "la2") & (l.ratio_min < .4)]
print("el mínimo:", x[x.soy_min].dz.value_counts().head(8).to_dict(), "| los otros:", x[~x.soy_min].dz.value_counts().head(6).to_dict())
x = l[(l.platform == "la2") & (l.ratio_min >= .6)]
print("Equipos sin AFK: cero en", round(x.cero.mean(), 4), "z de esos ceros:", x[x.cero].z_antes.describe()[["25%", "50%", "75%"]].round(0).to_dict(), "en exacto z=400:", round((x[x.cero].z_antes == 400).mean(), 3))
# ¿Hay 0 de equipo sin AFK detectable? Por equipo
te = l.groupby(["platform", "match_id", "team_id"]).agg(n=("cero", "size"), ceros=("cero", "sum"), ratio=("ratio_min", "first"), dur=("duracion_s", "first")).reset_index()
te4 = te[te.ceros >= 3]
print("Equipos con ≥3 ceros: n", len(te4), "ratio_min cuantiles", te4.ratio.quantile([.1, .5, .9]).round(2).to_dict(), "duración mediana", te4.dur.median())
print("Equipos con ≥3 medidos y ratio<0,4: tasa de ≥3 ceros:", round((te[(te.n >= 3) & (te.ratio < .4)].ceros >= 3).mean(), 3), "n", ((te.n >= 3) & (te.ratio < .4)).sum(), "| ratio≥0,6:", round((te[(te.n >= 3) & (te.ratio >= .6)].ceros >= 3).mean(), 4))

print("\n=== Modo +30/-10 y demás en LAS ===")
la = d[d.platform == "la2"]
print("gana 30: z", la[(la.win == 1) & (la.dz == 30)].z_antes.describe()[["25%", "50%", "75%"]].round(0).to_dict(), "| pierde -10: z", la[(la.win == 0) & (la.dz == -10)].z_antes.describe()[["25%", "50%", "75%"]].round(0).to_dict())
# ¿Los mismos jugadores? secuencias
mix = la.groupby("puuid").agg(n=("dz", "size"), g30=("dz", lambda s: (s == 30).sum()), p10=("dz", lambda s: (s == -10).sum()), gan=("dz", lambda s: s[s > 0].median()), per=("dz", lambda s: s[s < 0].median()), z=("z_antes", "median"))
print(mix[(mix.g30 >= 2) | (mix.p10 >= 2)].sort_values("n", ascending=False).head(12).to_string())
print("Sumas exactas en jugador con +30:", mix[mix.g30 >= 2].shape[0], "con -10:", mix[mix.p10 >= 2].shape[0], "ambos:", mix[(mix.g30 >= 1) & (mix.p10 >= 1)].shape[0])

# Ganancia mediana por jugador contra gap propio
print("\n=== Ganancia y pérdida 'normal' de cada jugador (sin 0 ni doble) contra el gap, LAS ===")
g = d.groupby("match_id").z_antes
d["lobby"] = (g.transform("sum") - d.z_antes) / (g.transform("count") - 1).where(g.transform("count") > 1)
x = la.assign(lobby=d.loc[la.index, "lobby"])
nor = x[((x.win == 1) & (x.dz.between(10, 34))) | ((x.win == 0) & (x.dz.between(-30, -1)))]
pj = nor.groupby("puuid").agg(gap=("lobby", lambda s: s.mean()), z=("z_antes", "mean"), n=("dz", "size"))
pj["gap"] = pj.gap - pj.z
ga = nor[nor.win == 1].groupby("puuid").dz.mean().rename("g"); pe = nor[nor.win == 0].groupby("puuid").dz.mean().rename("p")
pj = pj.join(ga).join(pe).dropna(); pj = pj[pj.n >= 10]
print("jugadores", len(pj), "corr(gap,gana)", round(pj.gap.corr(pj.g), 3), "corr(gap,pierde)", round(pj.gap.corr(pj.p), 3), "corr(gana,pierde)", round(pj.g.corr(pj.p), 3))
print(pj.groupby(pd.cut(pj.gap, [-3000, -300, -100, 0, 100, 300, 3000]), observed=True)[["g", "p", "n"]].agg({"g": "mean", "p": "mean", "n": "size"}).round(2).to_string())
d.to_pickle(SP / "vivo" / "d_todos.pkl")
