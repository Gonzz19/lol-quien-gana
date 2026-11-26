"""
¿Se puede mejorar el paso del mapa de LP (`src/features/lp_mecanica.py`, §9.S.43) con modelos
aprendidos? (PROJECT_SUMMARY.md §9.S.49)

El mapa avanza partida a partida con un paso mecánico: ±k·(1 ± p_doble) más δ(gap), con
gap = LP medio del lobby − LP propio como proxy de MMR − LP. Acá se predice el cambio real de LP
de cada partida (`dz`) con modelos entrenados en los eventos `enlazada` de `lp_vivo.db`, y se
comparan contra el paso mecánico ajustado con los mismos datos:

* *Paso a paso:* MAE del `dz` predicho, en jugadores no vistos (GroupKFold por jugador, 5 pliegues).
* *Acumulado:* tramos contiguos de ≥ 10 partidas de los jugadores de test, recorridos hacia
  adelante desde el primer LP real, con el paso de cada modelo. Es lo que hace el mapa causal.

Variables, todas disponibles en el mapa sobre `lol_las.db` en el momento del paso:
LP propio, resultado, LP del lobby (gap crudo y encogido), conocidos del lobby, rol, duración,
victorias/derrotas de la temporada antes de la partida (en el mapa: foto 1 + partidas
observadas) y racha y resultados recientes del jugador.

Modelos: paso mecánico (`ajustar_parametros_las` con los pliegues de ajuste), HGB con pérdida
absoluta solo con LAS, HGB con los 4 servidores (servidor como variable) y una corrección: HGB
sobre el residuo del paso mecánico.

    python scripts/analysis/mapa_lp_modelos.py --lp-vivo RUTA/lp_vivo.db   # ~3 min
"""
import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.features.lp_mecanica import (MASTER_BASE, ParametrosPaso, _paso_esperado_escalar,  # noqa: E402
                                      ajustar_parametros_las, encoger_gap, prob_doble)
from src.features.rating import lp_absoluto  # noqa: E402
from src.lp_vivo import agregar_lobby  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--lp-vivo", type=Path, required=True)
ap.add_argument("--min-tramo", type=int, default=10)
args = ap.parse_args()

SQL = ("SELECT e.platform, e.puuid, e.match_id, e.t_antes, e.t_despues, e.tier_antes, e.rank_antes, e.lp_antes, "
       "e.wins_antes, e.losses_antes, e.tier_despues, e.rank_despues, e.lp_despues, "
       "p.game_end, p.duracion_s, j.win, j.team_position "
       "FROM eventos e JOIN partidas p ON p.match_id = e.match_id "
       "JOIN participantes j ON j.match_id = e.match_id AND j.puuid = e.puuid "
       "WHERE e.tipo = 'partida' AND e.estado = 'enlazada'")
with sqlite3.connect(f"file:{args.lp_vivo.resolve().as_posix()}?mode=ro&immutable=1", uri=True) as con:
    ev = pd.read_sql_query(SQL, con)
for lado in ("antes", "despues"):
    ev[f"z_{lado}"] = [lp_absoluto(t, r, l) for t, r, l in zip(ev[f"tier_{lado}"], ev[f"rank_{lado}"], ev[f"lp_{lado}"])]
ev = ev.dropna(subset=["z_antes", "z_despues"])
ev["dz"] = ev.z_despues - ev.z_antes
ev["es_win"] = ev.win == 1
ev = pd.concat([agregar_lobby(g) for _, g in ev.groupby("platform")]).rename(
    columns={"lobby_lp": "lobby", "lobby_conocidos": "n_lob"})
ev = ev.sort_values(["platform", "puuid", "game_end"]).reset_index(drop=True)
print("eventos por servidor:", ev.platform.value_counts().to_dict(),
      f"| {pd.to_datetime(ev.game_end.min(), unit='ms'):%d/%m} → {pd.to_datetime(ev.game_end.max(), unit='ms'):%d/%m}")

# ---- variables disponibles en el mapa en el momento del paso
g = ev.groupby(["platform", "puuid"])
ev["n_prev"] = g.cumcount()
r = np.where(ev.es_win, 1.0, -1.0)
ev["r"] = r
for k in (3, 10):
    ev[f"neto_{k}"] = g["r"].transform(lambda s: s.shift(1).rolling(k, min_periods=1).sum()).fillna(0)
racha = []
for _, s in g["r"]:
    c, out = 0.0, []
    for v in s.to_numpy():
        out.append(c)
        c = c + v if np.sign(c) == np.sign(v) or c == 0 else v
    racha.extend(out)
ev["racha"] = racha
ev["temp_n"] = ev.wins_antes + ev.losses_antes
ev["temp_wr"] = (ev.wins_antes + 10) / (ev.temp_n + 20)
ev["gap"] = (ev.lobby - ev.z_antes).fillna(0)
ev["en_piso"] = (ev.z_antes == MASTER_BASE).astype(float)
ev["remake"] = (ev.duracion_s < 300).astype(float)
ev["rol"] = ev.team_position.fillna("").astype("category").cat.codes
ev["srv"] = ev.platform.astype("category").cat.codes
VARS = ["z_antes", "es_win", "gap", "n_lob", "rol", "duracion_s", "remake", "en_piso",
        "temp_n", "temp_wr", "neto_3", "neto_10", "racha"]

las = ev[ev.platform == "la2"].reset_index(drop=True)
todos = ev
print(f"LAS: {len(las):,} eventos de {las.puuid.nunique():,} jugadores")


def paso_mecanico(df, params):
    pdob = prob_doble(df.team_position.fillna("").to_numpy(), np.zeros(len(df)), params)
    return np.array([_paso_esperado_escalar(z, w, lob, nl, pd_, params) - z for z, w, lob, nl, pd_ in
                     zip(df.z_antes, df.es_win, df.lobby, df.n_lob.fillna(0), pdob)])


def hgb(loss="absolute_error"):
    return HistGradientBoostingRegressor(loss=loss, max_iter=400, learning_rate=0.05,
                                         max_leaf_nodes=31, min_samples_leaf=40, random_state=0)


PRED = {k: np.full(len(las), np.nan) for k in
        ("constante ±20", "paso mecánico", "HGB (LAS)", "HGB (4 servidores)", "mecánico + HGB del residuo",
         "HGB media (LAS)", "HGB media (4 servidores)", "mecánico + HGB media del residuo",
         "mecánico + corrección simétrica c(LP, gap)", "mecánico + c(LP, gap, resultado, rol, conocidos)")}
fold = np.full(len(las), -1)
for f, (tr, te) in enumerate(GroupKFold(5).split(las, groups=las.puuid)):
    fold[te] = f
    A, T = las.iloc[tr], las.iloc[te]
    params = ajustar_parametros_las(A.assign(lobby=A.lobby, n_lob=A.n_lob))
    PRED["constante ±20"][te] = np.where(T.es_win, 20.0, -20.0)
    m_te = paso_mecanico(T, params)
    PRED["paso mecánico"][te] = m_te
    PRED["HGB (LAS)"][te] = hgb().fit(A[VARS], A.dz).predict(T[VARS])
    jug_te = set(T.puuid)
    otros = todos[(todos.platform != "la2") | ~todos.puuid.isin(jug_te)]
    otros = otros[~((otros.platform == "la2") & otros.puuid.isin(jug_te))]
    PRED["HGB (4 servidores)"][te] = hgb().fit(otros[VARS + ["srv"]], otros.dz).predict(T[VARS + ["srv"]])
    m_tr = paso_mecanico(A, params)
    PRED["mecánico + HGB del residuo"][te] = m_te + hgb().fit(A[VARS], A.dz - m_tr).predict(T[VARS])
    PRED["HGB media (LAS)"][te] = hgb("squared_error").fit(A[VARS], A.dz).predict(T[VARS])
    PRED["HGB media (4 servidores)"][te] = hgb("squared_error").fit(otros[VARS + ["srv"]], otros.dz).predict(T[VARS + ["srv"]])
    PRED["mecánico + HGB media del residuo"][te] = m_te + hgb("squared_error").fit(A[VARS], A.dz - m_tr).predict(T[VARS])
    # la versión que entra en la grilla del mapa: un mismo corrimiento c para victoria y derrota,
    # función solo del LP propio y del gap encogido (lo que ve suavizar_en_grilla)
    ga, gt = encoger_gap(A.lobby, A.z_antes, A.n_lob.fillna(0), params), encoger_gap(T.lobby, T.z_antes, T.n_lob.fillna(0), params)
    c = hgb("squared_error").fit(np.c_[A.z_antes, ga], A.dz - m_tr)
    PRED["mecánico + corrección simétrica c(LP, gap)"][te] = m_te + c.predict(np.c_[T.z_antes, gt])
    VG = ["z_antes", "gap_enc", "es_win", "rol", "n_lob"]
    c2 = hgb("squared_error").fit(A.assign(gap_enc=ga)[VG], A.dz - m_tr)
    PRED["mecánico + c(LP, gap, resultado, rol, conocidos)"][te] = m_te + c2.predict(T.assign(gap_enc=gt)[VG])
    print(f"  pliegue {f} listo")

y = las.dz.to_numpy()
normal = ((las.es_win & las.dz.between(10, 31)) | (~las.es_win & las.dz.between(-30, -5))).to_numpy()
filas = []
for k, p in PRED.items():
    e = np.abs(p - y)
    filas.append({"modelo": k, "MAE": e.mean(), "RMSE": np.sqrt((e ** 2).mean()), "MedAE": np.median(e),
                  "MAE partidas normales": e[normal].mean(), "MAE fuera de lo normal": e[~normal].mean(),
                  "sesgo": (p - y).mean()})
print("\n=== Paso a paso (jugadores no vistos, LAS) ===")
print(pd.DataFrame(filas).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
print(f"partidas normales: {normal.mean():.1%}")

# ---- acumulado: tramos contiguos (sin huecos de partidas no enlazadas)
las["cont"] = las.groupby("puuid").z_antes.shift(-1).sub(las.z_despues).abs().lt(0.5)
filas = []
err = {k: [] for k in PRED}
for pu, s in las.groupby("puuid"):
    idx = s.index.to_numpy()
    corte = np.flatnonzero(~s.cont.to_numpy()[:-1]) + 1
    for seg in np.split(idx, corte):
        if len(seg) < args.min_tramo:
            continue
        real = np.r_[las.z_antes[seg].to_numpy(), las.z_despues[seg[-1]]]
        for k, p in PRED.items():
            camino = real[0] + np.r_[0, np.cumsum(p[seg])]
            err[k].append(np.abs(camino[1:] - real[1:]))
n_t = sum(1 for _ in err["paso mecánico"])
print(f"\n=== Acumulado: {n_t} tramos contiguos de ≥ {args.min_tramo} partidas (desde el primer LP real) ===")
for k, e in err.items():
    e = np.concatenate(e)
    print(f"  {k:44s} MAE {e.mean():6.2f}   RMSE {np.sqrt((e ** 2).mean()):6.2f}   p90 {np.percentile(e, 90):6.2f}")

# diferencia por tramo contra el paso mecánico (bootstrap sobre tramos)
base_t = np.array([e.mean() for e in err["paso mecánico"]])
rng = np.random.default_rng(0)
bt = rng.integers(0, len(base_t), size=(2000, len(base_t)))
for k_, e_ in err.items():
    if k_ == "paso mecánico":
        continue
    d = np.array([e.mean() for e in e_]) - base_t
    b = d[bt].mean(1)
    print(f"  Δ MAE por tramo vs mecánico, {k_:34s} {d.mean():+.2f} [{np.percentile(b, 2.5):+.2f}; {np.percentile(b, 97.5):+.2f}]")

m = hgb().fit(las[VARS], las.dz)
from sklearn.inspection import permutation_importance  # noqa: E402
imp = permutation_importance(m, las[VARS], las.dz, scoring="neg_mean_absolute_error", n_repeats=3, random_state=0)
print("\nImportancia por permutación (HGB LAS, aumento del MAE):")
print(pd.Series(imp.importances_mean, VARS).sort_values(ascending=False).round(3).to_string())
