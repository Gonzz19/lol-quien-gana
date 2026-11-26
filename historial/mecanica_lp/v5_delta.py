"""Estructura de δ en LAS y cruce con el dataset (solo entre fotos: nada anterior a la foto 1)."""
import contextlib
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
SP = Path(sys.argv[1]); ROOT = Path(sys.argv[2])
sys.path.insert(0, str(ROOT))
from src.features.rating import lp_absoluto  # noqa: E402

d = pd.read_pickle(SP / "vivo" / "d_todos.pkl")
la = d[d.platform == "la2"].copy().sort_values(["puuid", "t_despues"])

uri = f"file:{(ROOT / 'data' / 'lol_las.db').resolve().as_posix()}?mode=ro"
FOTO2 = "2026-10-28 00:49:59"
with contextlib.closing(sqlite3.connect(uri, uri=True, timeout=60)) as c:
    mapa = pd.read_sql_query("select puuid, player_id from player_id_map", c)
    f2 = pd.read_sql_query("select m.player_id, s.tier, s.rank, s.league_points, s.wins, s.losses from ladder_snapshots s join player_id_map m on m.puuid=s.puuid where s.snapshot_at=?", c, params=(FOTO2,))
f2["z2"] = [lp_absoluto(t, r, l) for t, r, l in zip(f2.tier, f2["rank"], f2.league_points)]
f2 = f2.assign(n=f2.wins + f2.losses).sort_values("n").drop_duplicates("player_id", keep="last").set_index("player_id")

print("eventos LAS con PUUID en player_id_map:", round(la.puuid.isin(set(mapa.puuid)).mean(), 3))
la = la.merge(mapa, on="puuid", how="left")

# Lobby completo: LP de antes si se midió, si no la foto 2 (2-4 días antes), para los 10 jugadores
uri_v = f"file:{(SP / 'vivo' / 'lp_vivo.db').as_posix()}?mode=ro&immutable=1"
with contextlib.closing(sqlite3.connect(uri_v, uri=True, timeout=60)) as c:
    jug = pd.read_sql_query("select j.match_id, j.puuid, j.team_id from participantes j join partidas p on p.match_id=j.match_id where p.platform='la2'", c)
jug = jug.merge(mapa, on="puuid", how="left").merge(f2[["z2"]], left_on="player_id", right_index=True, how="left")
jug = jug.merge(la[["match_id", "puuid", "z_antes"]], on=["match_id", "puuid"], how="left")
jug["z_lob"] = jug.z_antes.fillna(jug.z2)
print("jugadores de esas partidas con LP medido:", round(jug.z_antes.notna().mean(), 3), "| con LP medido o foto 2:", round(jug.z_lob.notna().mean(), 3))
g = jug.groupby("match_id").z_lob
jug["lobby"] = (g.transform("sum") - jug.z_lob.fillna(0)) / (g.transform("count") - jug.z_lob.notna().astype(int))
jug["lob_n"] = g.transform("count") - jug.z_lob.notna().astype(int)
la = la.merge(jug[["match_id", "puuid", "lobby", "lob_n"]], on=["match_id", "puuid"], how="left", suffixes=("_viejo", ""))
la["gap"] = la.lobby - la.z_antes

# δ por evento, sin mecánicas especiales
la["es_win"] = la.win == 1
la["delta"] = np.where(la.es_win, la.dz - 20, la.dz + 20)
norm = la[(la.es_win & la.dz.between(10, 31)) | (~la.es_win & la.dz.between(-30, -5))].copy()
print("\n=== δ = ΔLP ∓ 20 en LAS (partidas normales) ===")
print("distribución:", norm.delta.round().value_counts().sort_index().to_dict())
print("media por resultado:", norm.groupby("es_win").delta.mean().round(2).to_dict(), "| mediana:", norm.groupby("es_win").delta.median().to_dict())

# Persistencia de δ sin las mecánicas: ICC y autocorrelación, con el tiempo entre eventos
x = norm.copy(); x["n_j"] = x.groupby("puuid").delta.transform("size"); y = x[x.n_j >= 5]
gr = y.groupby("puuid").delta; n = gr.size(); k = len(n); N = n.sum()
msb = (n * (gr.mean() - y.delta.mean()) ** 2).sum() / (k - 1); msw = ((y.delta - gr.transform("mean")) ** 2).sum() / (N - k); n0 = (N - (n ** 2).sum() / N) / (k - 1)
vb = max((msb - msw) / n0, 0)
print(f"\nvar dentro del jugador {msw:.2f}  var entre jugadores {vb:.2f}  ICC {vb / (vb + msw):.3f}  (jugadores {k}, eventos {N})")
print("desvío de δ medio por jugador:", round(gr.mean().std(), 2), "cuantiles:", gr.mean().quantile([.01, .05, .25, .5, .75, .95, .99]).round(2).to_dict())
lag = y.assign(l1=gr.shift(1), l5=gr.shift(5)).dropna(subset=["l1"])
print("autocorr lag1:", round(lag.delta.corr(lag.l1), 3), "lag5:", round(lag.dropna(subset=["l5"]).delta.corr(lag.dropna(subset=["l5"]).l5), 3))
# ¿qué pasa con δ del mismo signo? victorias contra derrotas del mismo jugador
gg = y.pivot_table(index="puuid", columns="es_win", values="delta", aggfunc="mean").dropna()
print("corr(δ en victorias, δ en derrotas) por jugador:", round(gg[True].corr(gg[False]), 3), "dif media:", round((gg[True] - gg[False]).mean(), 2), "sd dif:", round((gg[True] - gg[False]).std(), 2))

print("\n=== δ contra el gap con lobby completo ===")
z = norm[norm.lob_n >= 6]
print(z.groupby(pd.cut(z.gap, [-5000, -800, -500, -300, -150, 0, 150, 300, 500, 800, 5000]), observed=True).delta.agg(["mean", "median", "std", "size"]).round(2).to_string())
pj = z.groupby("puuid").agg(delta=("delta", "mean"), gap=("gap", "mean"), lobby=("lobby", "mean"), z=("z_antes", "mean"), n=("delta", "size"))
pj = pj[pj.n >= 8]
print("por jugador (≥8): corr(δ, gap)", round(pj.delta.corr(pj.gap), 3), "corr(δ, lobby)", round(pj.delta.corr(pj.lobby), 3), "corr(δ, z)", round(pj.delta.corr(pj.z), 3), "n", len(pj))
print(pj.groupby(pd.cut(pj.gap, [-3000, -400, -200, -100, 0, 100, 200, 400, 800, 3000]), observed=True).delta.agg(["mean", "std", "size"]).round(2).to_string())

# W/L de temporada como otra pista de MMR
st = la.assign(wr=la.wins_antes / (la.wins_antes + la.losses_antes), N=la.wins_antes + la.losses_antes).groupby("puuid").agg(wr=("wr", "last"), N=("N", "last"), z=("z_antes", "last"))
pj = pj.join(st[["wr", "N"]])
print("corr(δ, wr temporada)", round(pj.delta.corr(pj.wr), 3), "corr(δ, N)", round(pj.delta.corr(pj.N), 3))

# Modelo simple por jugador: qué tanto de δ explican gap, z, wr. R² fuera de muestra.
from sklearn.ensemble import HistGradientBoostingRegressor as H
from sklearn.model_selection import cross_val_predict
X = pj[["gap", "lobby", "z", "wr", "N"]]; yy = pj.delta
pr = cross_val_predict(H(max_iter=150, learning_rate=.05, min_samples_leaf=15), X, yy, cv=5)
print(f"δ medio por jugador explicado por gap, lobby, z, wr, N: R² fuera de fold {1 - ((yy - pr) ** 2).sum() / ((yy - yy.mean()) ** 2).sum():.3f}  MAE {np.abs(yy - pr).mean():.2f} contra {np.abs(yy - yy.median()).mean():.2f} de la mediana")

# K según z: mediana de ganancia y pérdida normal, mediana de δ
print("\nMediana ganancia/pérdida por banda de z:")
print(norm.groupby([pd.cut(norm.z_antes, [0, 300, 400, 450, 500, 600, 800, 1000, 1300, 1800, 4000]), "es_win"], observed=True).dz.median().unstack().to_string())

# ZONA: derrotas en Diamond I, ¿piso?
print("\nCruces y pisos, LAS:")
print("L con z_antes=400 (Master 0):", la[(~la.es_win) & (la.z_antes == 400)].dz.value_counts().head(5).to_dict())
print("L con z_antes en (400,420):", la[(~la.es_win) & (la.z_antes > 400) & (la.z_antes < 420)].assign(zd=lambda v: v.z_despues).zd.value_counts().head(5).to_dict())
print("L en z 300-400:", la[(~la.es_win) & (la.z_antes.between(300, 399))].dz.value_counts().head(6).to_dict())
print("W en z 300-400:", la[(la.es_win) & (la.z_antes.between(300, 399))].dz.value_counts().head(6).to_dict())

# Historia de roles del dataset (entre fotos, sin target): frac del rol por jugador
t1 = None
print("\nleyendo roles del dataset entre fotos...")
with contextlib.closing(sqlite3.connect(uri, uri=True, timeout=60)) as c:
    t1_txt = c.execute("select max(updated_at) from players").fetchone()[0]
    t1 = int(pd.Timestamp(t1_txt, tz="UTC").value // 10**6)
    rol = pd.read_sql_query(
        "select mi.player_id, mp.individual_position pos, count(*) n from match_participants mp join matches x on x.match_id=mp.match_id "
        "join player_id_map mi on mi.puuid=mp.puuid where x.game_creation >= ? and mp.match_id like 'LA2_%' group by 1,2", c, params=(t1,))
rol["tot"] = rol.groupby("player_id").n.transform("sum")
rol["frac"] = rol.n / rol.tot
rol.to_pickle(SP / "vivo" / "roles.pkl")
la = la.merge(rol.rename(columns={"pos": "team_position", "frac": "frac_hist", "tot": "tot_hist"})[["player_id", "team_position", "frac_hist", "tot_hist"]], on=["player_id", "team_position"], how="left")
la["frac_hist"] = la.frac_hist.fillna(0)
tot = rol.groupby("player_id").tot.first()
la["tot_hist"] = la.player_id.map(tot)
la["doble"] = la.es_win & (la.dz >= 33)
w = la[la.es_win & (la.tot_hist >= 30)]
print("Victorias con ≥30 partidas de historial:", len(w), "de", la.es_win.sum())
print(w.groupby(pd.cut(w.frac_hist, [-.01, 0, .02, .05, .1, .2, .3, .5, .8, 1]), observed=True).doble.agg(["mean", "size"]).round(3).to_string())
print("Por rol y rol raro (frac<0,1):")
print(w.assign(raro=w.frac_hist < .1).pivot_table(index="team_position", columns="raro", values="doble", aggfunc=["mean", "size"]).round(3).to_string())
# ¿Y qué le pasa al perder en rol raro?
l = la[(~la.es_win) & (la.tot_hist >= 30)]
print("\nDerrotas según rareza: tasa de 0 y mediana de ΔLP")
print(l.groupby(pd.cut(l.frac_hist, [-.01, .02, .05, .1, .2, .5, 1]), observed=True).dz.agg([lambda s: (s == 0).mean(), "median", "size"]).round(3).to_string())
from sklearn.metrics import roc_auc_score
print("AUC del doble con la rareza del rol:", round(roc_auc_score(w.doble, -w.frac_hist), 3))
la.to_pickle(SP / "vivo" / "la_rico.pkl")
