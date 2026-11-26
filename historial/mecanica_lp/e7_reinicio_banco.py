"""1) ¿Hubo un reinicio de Master+ en lolpros? 2) ¿El decay en vivo sigue el banco de días o los 14 días sin jugar?
3) ¿El banco mejora la recorrida adelante entre fotos? 4) ¿Aegis protege al perder?"""
import contextlib
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 240); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
SP = Path(sys.argv[1]); ROOT = Path(sys.argv[2]).resolve(); sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(ROOT))
from mec import DIA, K, metricas, p_doble, parametros  # noqa: E402

V = SP / "vivo"
print("=== 1. lolpros: caídas de score sin partidas que las expliquen, temporada 16 ===")
h = pd.read_pickle(V / "lolpros.pkl")
h["t"] = pd.to_datetime(h.created_at, utc=True)
h = h[h.season == "season_16"].dropna(subset=["wins", "losses"]).sort_values(["cuenta", "t"])
h = h[h.score > 0]
h["N"] = h.wins + h.losses
for c, g in h.groupby("cuenta"):
    g = g.assign(ds=g.score.diff(), dN=g.N.diff(), net=(g.wins - g.losses).diff(), dias=g.t.diff().dt.total_seconds() / 86400)
    g["inexplicado"] = g.ds - 30 * g.net
    raro = g[(g.inexplicado.abs() > 300)]
    print(c, "filas", len(g), "| saltos inexplicados:", raro[["t", "tier", "league_points", "ds", "dN", "net", "dias", "inexplicado"]].assign(t=raro.t.dt.strftime("%m-%d")).to_dict("records")[:8])
por_mes = h.assign(mes=h.t.dt.strftime("%m")).groupby(["cuenta", "mes"]).agg(score=("score", "last"), N=("N", "last")).unstack(0)
print("\nscore a fin de cada mes:\n", por_mes["score"].round(0).to_string())

print("\n=== 2. Decay en vivo: ¿qué hizo en los 14 días previos cada jugador con -75? ===")
uri = f"file:{(V / 'lp_vivo.db').as_posix()}?mode=ro&immutable=1"
with contextlib.closing(sqlite3.connect(uri, uri=True, timeout=60)) as c:
    sp = pd.read_sql_query("select puuid, t_despues, tier_antes, lp_antes, tier_despues, rank_despues, lp_despues, wins_antes, losses_antes, wins_despues, losses_despues from eventos where platform='la2' and tipo='sin_partida' and t_antes is not null", c)
    est = pd.read_sql_query("select puuid, tier, lp, wins, losses from estado where platform='la2'", c)
sp["dlp"] = sp.lp_despues - sp.lp_antes
mapa_uri = f"file:{(ROOT / 'data' / 'lol_las.db').as_posix()}?mode=ro"
with contextlib.closing(sqlite3.connect(mapa_uri, uri=True, timeout=60)) as c:
    mapa = pd.read_sql_query("select puuid, player_id from player_id_map", c)
d75 = sp[(sp.dlp == -75) & sp.tier_antes.isin(["MASTER", "GRANDMASTER", "CHALLENGER"])].merge(mapa, on="puuid", how="left")
# rebotes: mismo jugador con +75 cerca
reb = sp[sp.dlp == 75].groupby("puuid").size()
print("eventos -75:", len(d75), "jugadores:", d75.puuid.nunique(), "| jugadores con algún +75:", d75.puuid.isin(reb.index).mean().round(3))
pa = pd.read_pickle(V / "partes.pkl")
t_ult = pa.game_creation.max()
ult = pa.groupby("player_id").game_creation.max()
d75 = d75.drop_duplicates("puuid")
d75["dias_sin_jugar_al_corte"] = (d75.t_despues - d75.player_id.map(ult)) / DIA
print("días entre su última partida del dataset y el decay observado:")
print(d75.dias_sin_jugar_al_corte.describe(percentiles=[.05, .25, .5, .75, .95]).round(1).to_dict(), "sin partidas en el dataset:", d75.dias_sin_jugar_al_corte.isna().mean().round(3))
print("con menos de 14 días (la regla 'de 14 días sin jugar' no lo explica):", round((d75.dias_sin_jugar_al_corte < 14).mean(), 3))

# Con banco: se reconstruye de las partidas entre fotos
def banco_al(hs, t_fin, ini=14.0, por=1.0, tope=14.0):
    """Banco de días al llegar a t_fin. Devuelve banco y días de decay acumulados."""
    b, t = ini, None
    dec = 0.0
    for x in hs:
        if t is not None:
            b -= (x - t) / DIA
            if b < 0:
                dec += -b; b = 0.0
        b = min(b + por, tope)
        t = x
    b -= (t_fin - t) / DIA
    if b < 0:
        dec += -b; b = 0.0
    return b, dec

horas = {j: g.game_creation.to_numpy() for j, g in pa[pa.player_id.isin(d75.player_id.dropna())].sort_values("game_creation").groupby("player_id")}
res = [banco_al(horas[j], t) if j in horas else (np.nan, np.nan) for j, t in zip(d75.player_id, d75.t_despues)]
d75["banco"], d75["dias_dec"] = zip(*res)
print("banco reconstruido al decaer (mediana, % con banco ≤ 1):", d75.banco.median(), round((d75.banco <= 1).mean(), 3), "| n", d75.banco.notna().sum())

# Control: todos los Master+ que NO decayeron, mismo cálculo
mas = est[est.tier.isin(["MASTER", "GRANDMASTER", "CHALLENGER"])].merge(mapa, on="puuid", how="left")
mas = mas[~mas.puuid.isin(d75.puuid) & mas.player_id.isin(pa.player_id.unique())]
t_ref = int(pd.Timestamp("2026-11-01 12:00", tz="UTC").value // 10**6)
hor2 = {j: g.game_creation.to_numpy() for j, g in pa[pa.player_id.isin(mas.player_id)].sort_values("game_creation").groupby("player_id")}
res = [banco_al(hor2[j], t_ref) for j in mas.player_id if j in hor2]
print("control sin decay: banco al 01/11 con solo partidas hasta el 26/10, mediana", round(np.median([r[0] for r in res]), 2), "% ≤ 1:", round(np.mean([r[0] <= 1 for r in res]), 3), "n", len(res))

print("\n=== 3. Recorrida adelante: decay '14 días sin jugar' contra banco ===")
P = parametros(V)
pl = pd.read_pickle(V / "partes_lobby.pkl"); pob = pd.read_pickle(V / "pob.pkl"); ts = pd.read_pickle(V / "ts.pkl")
roles = pd.read_pickle(V / "roles.pkl").set_index(["player_id", "pos"]).frac
t1, t2 = int(ts.t1), int(ts.t2)
KD = 18.75
ag = {j: (h_.win.to_numpy(), h_.game_creation.to_numpy(), h_.M.to_numpy(), h_.pos.to_numpy()) for j, h_ in pl[pl.player_id.isin(pob.index)].groupby("player_id", sort=False)}


def piso_decay(z, dias):
    """Aplica `dias` de decay con LP por día según rango, con los pisos."""
    d = int(np.floor(dias + 1e-9))
    for _ in range(d):
        if z >= 400:
            z = max(z - 75, 375.0)
        elif 275 < z < 400:
            z = max(z - 50, 275.0)
        else:
            break
    return z


def recorrer(j, z0, modo, tasa_oc=0.0, ini_m=14.0, ini_d=28.0):
    z = float(z0)
    b = ini_m if z >= 400 else ini_d
    if j not in ag:
        dias = (t2 - t1) / DIA
        if modo == "banco":
            return piso_decay(z, max(dias - b, 0))
        return piso_decay(z, max(dias - (14 if z >= 400 else 28), 0))
    win, tt, M, pos = ag[j]
    t_prev = t1
    for k in range(len(win) + 1):
        tk = tt[k] if k < len(win) else t2
        dias = (tk - t_prev) / DIA
        if modo == "hueco":
            z = piso_decay(z, max(dias - (14 if z >= 400 else 28), 0))
        elif modo == "banco":
            bb = b - dias
            if bb < 0:
                z = piso_decay(z, -bb); bb = 0.0
            b = bb
        t_prev = tk
        if k == len(win):
            break
        K_ = K if z >= 400 else KD
        d = float(np.interp(M[k] - z, P["gx"], P["dy"])) if not np.isnan(M[k]) else 0.0
        if win[k]:
            z += (K_ + d) * (1 + p_doble(P, pos[k], roles.get((j, pos[k]), 0.0)))
        else:
            q = P["perdon"]
            if z >= 400 and z - (K_ - d) * (1 - q) < 400:
                z = 400.0 if z > 400.5 else z - (1 - P["piso_quieto"]) * P["piso_baja"]
            else:
                z -= (K_ - d) * (1 - q)
        if modo == "banco":
            if z >= 400:
                b = min(b + 1.0 + tasa_oc, 14.0)
            else:
                b = min(b + 7.0 * (1 + tasa_oc), 28.0)
    return z


y = pob.z2.to_numpy(); zi = pob.z1.to_numpy()
grp = pob.tier2.map({"CHALLENGER": 0, "GRANDMASTER": 1, "MASTER": 2}).fillna(3).astype(int).to_numpy()
tasa = np.array([.03, .03, .13, 1.09])[grp]
preds = {"sin decay": [recorrer(j, z, "nada") for j, z in pob.z1.items()],
         "14 días sin jugar": [recorrer(j, z, "hueco") for j, z in pob.z1.items()],
         "banco (Riot)": [recorrer(j, z, "banco") for j, z in pob.z1.items()],
         "banco + ocultas por tasa": [recorrer(j, z, "banco", t) for (j, z), t in zip(pob.z1.items(), tasa)]}
preds = {k: np.array(v) for k, v in preds.items()}
print(pd.DataFrame({k: metricas(y, zi, v) for k, v in preds.items()}).T.round(3).to_string())
print(pd.DataFrame({k: {"C+GM": np.abs(v - y)[grp <= 1].mean(), "Master": np.abs(v - y)[grp == 2].mean(), "Diamond": np.abs(v - y)[grp == 3].mean(), "sesgo": (v - y).mean()} for k, v in preds.items()}).T.round(1).to_string())
pd.DataFrame(preds, index=pob.index).to_pickle(V / "pred_banco.pkl")

print("\n=== 4. Aegis al perder: tasa de 0 LP en rol raro, sin Master 0 y sin equipos con ≥2 en cero ===")
la = pd.read_pickle(V / "la_rico.pkl")
l = la[(~la.es_win) & (la.z_antes != 400) & (la.tot_hist >= 30)].copy()
l["c"] = l.groupby(["match_id", "team_id"]).dz.transform(lambda s: (s == 0).sum())
l["ind"] = (l.dz == 0) & (l.c == 1)
print(l.groupby(pd.cut(l.frac_hist, [-.01, .02, .05, .1, .2, .5, 1]), observed=True).ind.agg(["mean", "sum", "size"]).round(4).to_string())
print("por rol, rol raro:", l[l.frac_hist < .1].groupby("team_position").ind.agg(["mean", "size"]).round(3).to_dict("index"))
w = la[la.es_win & (la.tot_hist >= 30)]
print("doble en rol raro TOP/JG/SUP:", round(w[(w.frac_hist < .05) & w.team_position.isin(["TOP", "JUNGLE", "UTILITY"])].doble.mean(), 3))
# Doble y rendimiento: mastery >= C. Oro/KDA de los que NO cobran en rol raro
rr = w[(w.frac_hist < .05) & w.team_position.isin(["TOP", "JUNGLE", "UTILITY"])].copy()
rr["kda"] = (rr.kills + rr.assists) / rr.deaths.clip(lower=1)
print("rol raro: kda y oro según cobró el doble:", rr.groupby("doble")[["kda", "gold", "cs"]].median().round(1).to_dict("index"), "n", rr.doble.value_counts().to_dict())
