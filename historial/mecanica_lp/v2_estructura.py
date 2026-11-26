"""¿De qué está hecho el ΔLP por partida? Sobre lp_vivo.db, todos los servidores, foco en LAS."""
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
    d = pd.read_sql_query(
        "select e.id, e.match_id, e.platform, e.puuid, e.t_antes, e.t_despues, e.tier_antes, e.rank_antes, e.lp_antes, e.wins_antes, e.losses_antes,"
        " e.tier_despues, e.rank_despues, e.lp_despues, e.wins_despues, e.losses_despues, p.game_end, p.game_start, p.duracion_s, j.team_id, j.win"
        " from eventos e join partidas p on p.match_id=e.match_id join participantes j on j.match_id=e.match_id and j.puuid=e.puuid"
        " where e.estado='enlazada'", c)
    sp = pd.read_sql_query("select platform, puuid, t_antes, t_despues, tier_antes, rank_antes, lp_antes, tier_despues, rank_despues, lp_despues from eventos where tipo='sin_partida' and t_antes is not null", c)
    va = pd.read_sql_query("select platform, puuid, t_antes, t_despues, tier_antes, rank_antes, lp_antes, wins_antes, losses_antes, tier_despues, rank_despues, lp_despues, wins_despues, losses_despues from eventos where tipo='varias' and t_antes is not null", c)

for k, x in (("d", d), ("sp", sp), ("va", va)):
    for lado in ("antes", "despues"):
        x[f"z_{lado}"] = [lp_absoluto(t, r, l) for t, r, l in zip(x[f"tier_{lado}"], x[f"rank_{lado}"], x[f"lp_{lado}"])]
    x["dz"] = x.z_despues - x.z_antes

g = d.groupby("match_id")["z_antes"]
n_otros = g.transform("count") - 1
d["lobby"] = (g.transform("sum") - d.z_antes) / n_otros.where(n_otros > 0)
d["n_lobby"] = n_otros
d["gap"] = d.lobby - d.z_antes
d["signo"] = np.where(d.win == 1, 1, -1)
d["cruza_tier"] = d.tier_antes != d.tier_despues

print("=== ΔLP por servidor, victoria/derrota (cuantiles) ===")
print(d.groupby(["platform", "win"]).dz.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).round(1).to_string())
print("\ncasos con cruce de tier:", d.groupby("platform").cruza_tier.mean().round(4).to_dict())
print("signo contrario (ganó y bajó LP o perdió y subió):", d.groupby("platform").apply(lambda x: ((x.dz * x.signo) < 0).mean(), include_groups=False).round(4).to_dict())

# Regla por servidor y residuo
reg = d.groupby(["platform", "win"]).dz.transform("mean")
d["res"] = d.dz - reg
print("\n=== Residuo tras la regla (por servidor y resultado) ===")
print(d.groupby("platform").res.agg(["std", lambda s: s.abs().mean(), lambda s: s.abs().median()]).round(2).to_string())
print("kurtosis:", d.groupby("platform").res.apply(pd.Series.kurt).round(1).to_dict())
print("|res|>20:", d.groupby("platform").res.apply(lambda s: (s.abs() > 20).mean()).round(4).to_dict())

# Cola: ¿qué son los residuos enormes?
print("\n=== Cola en LAS: |res|>20 contra el resto ===")
l = d[d.platform == "la2"].copy()
c1 = l[l.res.abs() > 20]
print(c1[["tier_antes", "rank_antes", "lp_antes", "tier_despues", "rank_despues", "lp_despues", "win", "dz", "t_antes", "t_despues"]].assign(
    ventana_min=lambda x: (x.t_despues - x.t_antes) / 60000).drop(columns=["t_antes", "t_despues"]).sample(min(15, len(c1)), random_state=1).to_string())
print("ventana (min) cola/resto:", ((c1.t_despues - c1.t_antes) / 60000).median(), ((l.t_despues - l.t_antes) / 60000).median())

# Varianza entre jugadores (ICC) del residuo: limpio de cola y por servidor
print("\n=== ¿Es persistente por jugador? (residuo recortado a ±20) ===")
for plat, x in d.groupby("platform"):
    x = x[x.res.abs() <= 20].sort_values(["puuid", "game_end"])
    cnt = x.groupby("puuid").res.transform("size")
    y = x[cnt >= 5]
    grp = y.groupby("puuid").res
    n = grp.size()
    k = len(n)
    N = n.sum()
    msb = (n * (grp.mean() - y.res.mean()) ** 2).sum() / (k - 1)
    msw = ((y.res - grp.transform("mean")) ** 2).sum() / (N - k)
    n0 = (N - (n ** 2).sum() / N) / (k - 1)
    var_b = max((msb - msw) / n0, 0)
    ac = y.assign(lag=grp.shift(1)).dropna(subset=["lag"])
    # autocorrelación condicional al mismo signo: la que importa para la ganancia
    print(f"{plat}: jugadores {k:5d} eventos {N:6d}  var dentro {msw:6.1f}  var entre {var_b:6.2f}  ICC {var_b / (var_b + msw):.3f}  "
          f"autocorr lag1 {np.corrcoef(ac.res, ac.lag)[0, 1]:.3f}")

# ¿Ganancia y pérdida del mismo jugador se mueven juntas o al revés? (MMR>LP = +gana, -pierde)
print("\n=== Asimetría: ganancia media y pérdida media por jugador (≥5 de cada una) ===")
for plat, x in d.groupby("platform"):
    x = x[x.res.abs() <= 25]
    p = x.pivot_table(index="puuid", columns="win", values="dz", aggfunc=["mean", "size"])
    p.columns = ["L", "W", "nL", "nW"]
    p = p[(p.nL >= 5) & (p.nW >= 5)]
    print(f"{plat}: jugadores {len(p):5d}  media gana {p.W.mean():5.1f} (sd {p.W.std():4.1f})  media pierde {p.L.mean():6.1f} (sd {p.L.std():4.1f})  corr(gana, |pierde|) {np.corrcoef(p.W, -p.L)[0, 1]:+.2f}  deriva W+L sd {(p.W + p.L).std():4.1f}")

# Gap: ganancia según lobby - LP propio
print("\n=== Ganancia y pérdida según gap (lobby medido - LP propio) ===")
x = d[(d.n_lobby >= 3) & (d.res.abs() <= 30)].copy()
x["gap_b"] = pd.cut(x.gap, [-5000, -600, -300, -150, -50, 50, 150, 300, 600, 5000])
t = x.pivot_table(index="gap_b", columns=["platform", "win"], values="dz", aggfunc="mean", observed=True).round(1)
print(t.to_string())
print(x.pivot_table(index="gap_b", columns="platform", values="dz", aggfunc="size", observed=True).to_string())
for plat, y in x.groupby("platform"):
    for w in (0, 1):
        z = y[y.win == w]
        b = np.polyfit(z.gap, z.dz, 1)
        print(f"{plat} win={w}: pendiente por 100 LP de gap {b[0] * 100:+.2f}  R² {np.corrcoef(z.gap, z.dz)[0, 1] ** 2:.3f}  n={len(z)}")

# LP por tier en LAS
print("\n=== LAS, regla por z de antes (bandas) ===")
x = l.copy(); x = x[x.res.abs() <= 30]
x["z_b"] = pd.cut(x.z_antes, [0, 400, 500, 600, 800, 1000, 1300, 1800, 4000])
print(x.pivot_table(index="z_b", columns="win", values="dz", aggfunc=["mean", "std", "size"], observed=True).round(1).to_string())

print("\n=== Otros cambios ===")
print("sin_partida LAS: n", (sp.platform == "la2").sum(), "dz", sp[sp.platform == "la2"].dz.describe(percentiles=[.05, .5, .95]).round(1).to_dict())
print("valor más frecuente dz sin_partida:", sp[sp.platform == "la2"].dz.round().value_counts().head(8).to_dict())
print("varias LAS: dN", (va[va.platform == "la2"].wins_despues + va[va.platform == "la2"].losses_despues - va[va.platform == "la2"].wins_antes - va[va.platform == "la2"].losses_antes).describe().round(2).to_dict())
d.to_pickle(SP / "vivo" / "eventos.pkl")
