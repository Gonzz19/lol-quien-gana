"""
Idea 43 del brainstorming (pedido del usuario, 2026-11-16): el modelo en vivo aplicado al **estado esperado**.

Antes de la partida no se conoce el estado al minuto 10/15/20, pero sí lo que cada jugador suele tener en esos
minutos. Se arma el estado esperado de cada línea con el promedio previo del jugador (tres versiones: toda su
historia, sus últimas 20 y su sesión del día) y se pasa por un modelo en vivo entrenado con los estados reales.
Las P(gana azul) resultantes al 10, 15 y 20 entran como columnas extra de la logística previa.

* Estado por línea y minuto: diferencia azul − rojo de oro, XP, CS, kills y muertes (en miles el oro y la XP).
* Promedios previos, todos causales (solo partidas anteriores del jugador):
  - historia: media expansiva;  - últimas 20: media móvil de las 20 anteriores;
  - sesión: media de las partidas anteriores del mismo día de juego (06:00 → 06:00 ART, como §9.S.45);
  sin partidas previas → la media del ajuste (la diferencia queda en 0).
* Modelo en vivo: logística o HGB, espejado (cada partida también con los lados invertidos) y antisimétrico al
  predecir: p(x) = (f(x) + 1 − f(−x)) / 2. Se entrena dentro de cada pliegue solo con las partidas de
  entrenamiento de ese pliegue que llegaron a ese minuto (sin fuga hacia los bloques ni hacia V).
* Variantes (contra el híbrido + mapa de §9.S.51, 0,653 en V Master+): las diferencias esperadas crudas
  (lineal, para medir cuánto es redundante) y el logit del modelo en vivo con logística y con HGB.
* Evaluación: `brainstorm_base.evaluar` / `comparar` (sub-bloques A2-A4 de Master+ y V Master+).

    python historial/ideas_posteriores/brainstorm_43_estado_esperado.py      # ~15 min
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.brainstorm_base import ajustar_predecir, cargar, comparar, evaluar, mostrar  # noqa: E402

LANES = ["BOTTOM", "JUNGLE", "MIDDLE", "TOP", "UTILITY"]      # el orden de embeddings_partidas.py
MINUTOS = (10, 15, 20)
STATS = ("gold", "xp", "cs", "kills", "deaths")
ESCALA = {"gold": 1000.0, "xp": 1000.0, "cs": 10.0, "kills": 1.0, "deaths": 1.0}
CACHE = REPO / "data" / "analysis" / "brainstorm" / "estado_esperado.npz"
T0 = time.time()
log = lambda s: print(f"[{time.time() - T0:5.0f}s] {s}", flush=True)  # noqa: E731


def construir(D):
    """Estados reales y esperados: arrays (partidas, minutos, líneas, stats) azul − rojo, en el orden de D."""
    if CACHE.exists():
        C = np.load(CACHE)
        return C["real"], {k: C[k] for k in ("historia", "ultimas20", "sesion")}
    cols = ["match_id", "game_creation", "game_duration", "player_id", "team_id", "lane", "win"] + \
           [f"{s}_{m}" for m in MINUTOS for s in STATS]
    df = pd.read_csv(REPO / "data/splits/train/full_dataset.csv", usecols=cols)
    df = df[df.lane.isin(LANES)]
    df = df[df.groupby(["match_id", "team_id"]).lane.transform("nunique") == 5]
    df = df[df.groupby("match_id").lane.transform("size") == 10]
    df = df[df.groupby("match_id").win.transform("sum") == 5]
    df = df.sort_values(["game_creation", "match_id", "team_id", "lane"]).reset_index(drop=True)
    mids = df.match_id.to_numpy()[::10]
    assert len(mids) == D["NM"] and (mids == D["Z"]["mids"]).all(), "el orden de las partidas no coincide con D"
    for m in MINUTOS:                                   # partidas que no llegaron al minuto: sin estado
        corto = df.game_duration <= 60 * m
        for s in STATS:
            df.loc[corto, f"{s}_{m}"] = np.nan
    vals = [f"{s}_{m}" for m in MINUTOS for s in STATS]
    X = df[vals].to_numpy(np.float64) / np.array([ESCALA[s] for m in MINUTOS for s in STATS])
    # promedios previos del jugador (causales): ordenar por jugador y fecha
    o = np.lexsort((df.game_creation.to_numpy(), df.player_id.to_numpy()))
    P = pd.DataFrame(X[o], columns=vals)
    P["pid"] = df.player_id.to_numpy()[o]
    dia = ((df.game_creation.to_numpy()[o] - 9 * 3.6e6) // 8.64e7).astype(np.int64)   # 06:00 ART = 09:00 UTC
    P["ses"] = P.pid.astype(str) + "_" + dia.astype(str)
    F = P[vals]

    def media_previa(clave):
        """media de las partidas anteriores dentro de cada grupo (sin contar la propia ni los NaN)"""
        hay, Fz = F.notna().astype(float), F.fillna(0.0)
        s = Fz.groupby(clave).cumsum() - Fz
        n = hay.groupby(clave).cumsum() - hay
        return s / n.replace(0, np.nan)

    previa = F.groupby(P.pid).shift(1)
    E = {
        "historia": media_previa(P.pid),
        "ultimas20": previa.groupby(P.pid).rolling(20, min_periods=1).mean().reset_index(level=0, drop=True)
                           .reindex(F.index),
        "sesion": media_previa(P.ses),
    }
    inv = np.empty_like(o)
    inv[o] = np.arange(len(o))
    def a_partidas(M):                                  # (filas) -> (partidas, minutos, líneas, stats) azul − rojo
        M = np.asarray(M, dtype=np.float32).reshape(D["NM"], 2, 5, len(MINUTOS), len(STATS))
        return (M[:, 0] - M[:, 1]).transpose(0, 2, 1, 3)
    real = a_partidas(X)
    esp = {}
    for k, M in E.items():
        M = M.to_numpy()[inv]
        mu = np.nanmean(M, axis=0)                      # sin partidas previas: la media (diferencia ≈ 0)
        M = np.where(np.isnan(M), mu, M)
        esp[k] = a_partidas(M)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, real=real, **esp)
    return real, esp


def vivo_predictor(tipo, Xr, Xe, y, tr, te):
    """Modelo en vivo de un minuto: se ajusta con los estados reales de `tr` y predice los esperados de tr y te."""
    ok = tr[~np.isnan(Xr[tr]).any(1)]
    Xa = np.r_[Xr[ok], -Xr[ok]]
    ya = np.r_[y[ok], 1 - y[ok]]
    if tipo == "logistica":
        m = LogisticRegression(C=1.0, max_iter=2000).fit(Xa, ya)
    else:
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                           min_samples_leaf=200, l2_regularization=1.0, random_state=0).fit(Xa, ya)
    def p(idx):
        a = m.predict_proba(Xe[idx])[:, 1]
        b = m.predict_proba(-Xe[idx])[:, 1]
        q = np.clip((a + 1 - b) / 2, 1e-4, 1 - 1e-4)
        return np.log(q / (1 - q))
    return p(tr), p(te)


def main():
    D = cargar()
    log(f"partidas {D['NM']:,}; V Master+ {int(D['val'].sum()):,}")
    real, esp = construir(D)
    log("estados reales y esperados listos")
    y = D["y"]
    plano = lambda A: A.reshape(len(A), -1)           # noqa: E731  (minutos × líneas × stats)
    por_min = lambda A, i: A[:, i].reshape(len(A), -1)  # noqa: E731

    def con_extra(fabricar):
        """predictor(tr, te) que arma columnas extra con `fabricar(tr, te)` y ajusta el híbrido + mapa."""
        def pred(tr, te):
            Etr, Ete = fabricar(tr, te)
            Z = np.zeros((D["NM"], Etr.shape[1]), np.float32)
            Z[tr], Z[te] = Etr, Ete
            return ajustar_predecir(D, tr, te, extra=Z, extra_post=False)
        return pred

    def crudo(agg):
        return lambda tr, te: (plano(esp[agg])[tr], plano(esp[agg])[te])

    def vivo(tipo, aggs):
        def fab(tr, te):
            a, b = [], []
            for agg in aggs:
                for i in range(len(MINUTOS)):
                    ptr, pte = vivo_predictor(tipo, por_min(real, i), por_min(esp[agg], i), y, tr, te)
                    a.append(ptr); b.append(pte)
            return np.c_[a].T, np.c_[b].T
        return fab

    P = {}
    variantes = [("Híbrido + mapa (referencia)", lambda tr, te: ajustar_predecir(D, tr, te))]
    for agg in ("historia", "ultimas20", "sesion"):
        variantes.append((f"+ estado esperado crudo ({agg})", con_extra(crudo(agg))))
    for agg in ("historia", "ultimas20", "sesion"):
        variantes.append((f"+ P en vivo, logística ({agg})", con_extra(vivo("logistica", [agg]))))
    for agg in ("historia", "ultimas20", "sesion"):
        variantes.append((f"+ P en vivo, HGB ({agg})", con_extra(vivo("hgb", [agg]))))
    variantes.append(("+ P en vivo, HGB (las tres)", con_extra(vivo("hgb", ["historia", "ultimas20", "sesion"]))))
    for nombre, pred in variantes:
        P[nombre] = evaluar(D, pred)
        log(f"listo: {nombre}")
    T = comparar(D, P, "Híbrido + mapa (referencia)")
    mostrar(T)
    T.to_csv(REPO / "data" / "analysis" / "brainstorm" / "estado_esperado.csv", index=False)


if __name__ == "__main__":
    main()
