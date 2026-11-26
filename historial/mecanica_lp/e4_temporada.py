"""1) ¿Cierra la temporada entera? z ≈ z0 + K(W-L) + c·N usando solo la foto (no toca partidas del test).
2) lolpros: reinicios de W/L y LP de arranque.
3) ¿Se detecta la derrota perdonada con datos de la partida?"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
SP = Path(sys.argv[1]); ROOT = Path(sys.argv[2]).resolve()
V = SP / "vivo"
f1 = pd.read_pickle(V / "foto1.pkl"); f2 = pd.read_pickle(V / "foto2.pkl")

print("=== 1. Cierre de temporada con la foto: z = z0 + K(W-L) + c·N (L1 por tramos) ===")
from sklearn.linear_model import QuantileRegressor
for nombre, f in (("foto 1 (04/09)", f1), ("foto 2 (28/10)", f2[f2.tier.isin(["CHALLENGER", "GRANDMASTER", "MASTER", "DIAMOND"])])):
    f = f[f.partidas >= 20].copy()
    f["net"] = f.wins - f.losses
    for tiers in (["DIAMOND"], ["MASTER"], ["GRANDMASTER", "CHALLENGER"], ["MASTER", "GRANDMASTER", "CHALLENGER"], ["DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER"]):
        g = f[f.tier.isin(tiers)]
        if len(g) > 6000:
            g = g.sample(6000, random_state=1)
        X = np.c_[g.net, g.partidas]
        q = QuantileRegressor(quantile=.5, alpha=0, solver="highs").fit(X, g.z)
        pred = q.predict(X)
        print(f"{nombre} {'+'.join(t[:2] for t in tiers):12s} n={len(g):6d}  z0={q.intercept_:7.1f}  K={q.coef_[0]:5.1f}  c={q.coef_[1]:5.2f}  MAE={np.abs(g.z - pred).mean():6.1f}  corr(z, W-L)={np.corrcoef(g.z, g.net)[0, 1]:.2f}  MAE sin modelo={np.abs(g.z - g.z.median()).mean():6.1f}")
    g = f[f.tier.isin(["DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER"])]
    print("  z - 20(W-L), por tramos de N (mediana):")
    print(g.assign(res=g.z - 20 * g.net).groupby(pd.cut(g.partidas, [20, 50, 100, 200, 300, 500, 800, 3000]), observed=True).res.agg(["median", lambda s: s.quantile(.1), lambda s: s.quantile(.9), "size"]).round(0).to_string())

print("\n=== 2. lolpros: reinicios y arranque ===")
crudos = glob.glob(str(ROOT / "data" / "external" / "lolpros" / "raw" / "*.json"))
print(len(crudos), "archivos")
ej = json.load(open(crudos[0], encoding="utf-8"))
print("tipo:", type(ej), "claves:", list(ej.keys())[:12] if isinstance(ej, dict) else (list(ej[0].keys()) if ej else None))
filas = []
for c in crudos:
    o = json.load(open(c, encoding="utf-8"))
    r = o.get("data", o) if isinstance(o, dict) else o
    if isinstance(r, dict):
        for v in r.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                r = v; break
    if not isinstance(r, list) or not r:
        continue
    d = pd.DataFrame(r)
    d["cuenta"] = Path(c).stem[:6]
    filas.append(d)
if filas:
    h = pd.concat(filas, ignore_index=True)
    print(h.columns.tolist()); print(h.head(3).to_string())
    h.to_pickle(V / "lolpros.pkl")

print("\n=== 3. Derrota perdonada: ¿se ve en los datos de la partida? ===")
d = pd.read_pickle(V / "d_todos.pkl")
from sklearn.ensemble import HistGradientBoostingClassifier as HC
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
import contextlib, sqlite3
uri = f"file:{(V / 'lp_vivo.db').as_posix()}?mode=ro&immutable=1"
with contextlib.closing(sqlite3.connect(uri, uri=True, timeout=60)) as c:
    j = pd.read_sql_query("select j.match_id, j.puuid, j.team_id, j.win, j.kills, j.deaths, j.assists, j.gold, j.cs, j.team_position, p.duracion_s, p.platform from participantes j join partidas p on p.match_id=j.match_id", c)
j["ka"] = j.kills + j.assists
eq = j.groupby(["match_id", "team_id"]).agg(win=("win", "first"), dur=("duracion_s", "first"), plat=("platform", "first"), gold_min=("gold", "min"), gold_med=("gold", "median"), gold_sum=("gold", "sum"),
                                         cs_min=("cs", "min"), d_max=("deaths", "max"), d_sum=("deaths", "sum"), ka_min=("ka", "min"), k_sum=("kills", "sum"), a_sum=("assists", "sum")).reset_index()
riv = eq[["match_id", "team_id", "gold_sum", "k_sum", "d_sum"]].assign(team_id=lambda x: 300 - x.team_id).rename(columns={"gold_sum": "gold_riv", "k_sum": "k_riv", "d_sum": "d_riv"})
eq = eq.merge(riv, on=["match_id", "team_id"])
eq["ratio_oro"] = eq.gold_min / eq.gold_med; eq["dif_oro_min"] = (eq.gold_sum - eq.gold_riv) / eq.dur * 60
eq["oro_min_rate"] = eq.gold_min / eq.dur * 60; eq["d_max_rate"] = eq.d_max / eq.dur * 60
lo = d[d.win == 0].assign(cero=lambda x: (x.dz == 0) & (x.z_antes != 400))
te = lo.groupby(["match_id", "team_id"]).agg(n=("cero", "size"), ceros=("cero", "sum")).reset_index()
te = te[te.n >= 2]; te["perdon"] = te.ceros >= (te.n * .6).clip(lower=2)
x = te.merge(eq, on=["match_id", "team_id"])
print("equipos perdedores con ≥2 medidos:", len(x), "perdonados:", x.perdon.mean().round(4), x.groupby("plat").perdon.mean().round(4).to_dict())
print("por duración:", x.groupby(pd.cut(x.dur, [0, 900, 1200, 1500, 1800, 2100, 4000]), observed=True).perdon.agg(["mean", "size"]).round(3).to_dict("index"))
cols = ["dur", "gold_min", "ratio_oro", "cs_min", "d_max", "d_max_rate", "d_sum", "ka_min", "k_sum", "a_sum", "dif_oro_min", "oro_min_rate"]
pr = cross_val_predict(HC(max_iter=200, learning_rate=.05, min_samples_leaf=20), x[cols], x.perdon, cv=GroupKFold(5), groups=x.match_id, method="predict_proba")[:, 1]
print("AUC perdón con duración, oro, kda, cs del equipo (fuera de fold):", round(roc_auc_score(x.perdon, pr), 3), "| precisión en el 5 % más probable:", round(x.perdon[pr >= np.quantile(pr, .95)].mean(), 3))

print("\n=== 4. Doble: ¿cuánto se identifica con rol + campeón del jugador? ===")
la = pd.read_pickle(V / "la_rico.pkl")
w = la[la.es_win & (la.tot_hist >= 30)].copy()
w = w.assign(rol=w.team_position.astype("category").cat.codes, z=w.z_antes, gap2=w.gap)
c2 = ["frac_hist", "rol", "z", "gap2", "tot_hist"]
pr = cross_val_predict(HC(max_iter=200, learning_rate=.05, min_samples_leaf=20), w[c2], w.doble, cv=GroupKFold(5), groups=w.puuid, method="predict_proba")[:, 1]
print("AUC doble con rareza, rol, z, gap, partidas (fuera de fold):", round(roc_auc_score(w.doble, pr), 3), "| precisión top 5 %:", round(w.doble[pr >= np.quantile(pr, .95)].mean(), 3), "| recall top 10 %:", round(w.doble[pr >= np.quantile(pr, .9)].sum() / w.doble.sum(), 3))
