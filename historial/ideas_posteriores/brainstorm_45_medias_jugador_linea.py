"""Brainstorming (PROJECT_SUMMARY.md §9.S.71): media por (jugador, línea) del estado a los 10/15/20 en sus partidas
anteriores, sumada al modelo previo a la partida (escalón 9, V Master+).

Por cada jugador y línea, con sus partidas ANTERIORES en esa línea (orden por `game_creation`, la actual no entra):
  * media de la diferencia contra el rival de línea de oro, XP, CS, daño, nivel, kills, muertes y asistencias al
    minuto 10, 15 y 20 (24 números), encogida hacia 0: suma / (n + K), K = 5;
  * media de oro, XP y CS propios (sin restar al rival) a 10/15/20, centrada en la media de la línea (9 números);
    la media de la línea se calcula solo con partidas anteriores al 16/09;
  * log(1 + n) de partidas previas en la línea.
El valor a 15 (20) solo cuenta si la partida duró más de 15 (20) minutos.

Al modelo entra azul − rojo por línea. Variantes sobre el escalón 9:
  A. oro/XP/CS contra el rival a 10/15/20, por línea (45 columnas)
  B. todo, por línea (170 columnas)
  C. todo, promediado en el equipo (34 columnas)

Escribe `data/analysis/brainstorm/medias_jugador_linea.npz` (medias por jugador y estado real por línea, que usa
`brainstorm_46_en_vivo_mejoras.py`) y `pred_esc9_oos.npz` (predicciones hacia adelante del escalón 9).

    python historial/ideas_posteriores/brainstorm_45_medias_jugador_linea.py     # ~15 min
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.brainstorm_base import ajustar_predecir, cargar, comparar, evaluar, extras_referencia, mostrar  # noqa: E402

CACHE = REPO / "data" / "analysis" / "brainstorm" / "medias_jugador_linea.npz"
PRED = REPO / "data" / "analysis" / "brainstorm" / "pred_esc9_oos.npz"
STATS = ["gold", "xp", "cs", "damage", "level", "kills", "deaths", "assists"]
MINS = (10, 15, 20)
ESCALA = {"gold": 1000.0, "xp": 1000.0, "damage": 1000.0, "cs": 10.0}
CRUDAS = ["gold", "xp", "cs"]
K = 5.0


def construir(D):
    """Devuelve (PJ, cols, ST): medias previas por jugador (NM, 2, 5, 34) y estado real azul − rojo (NM, 5, 8, 3)."""
    if CACHE.exists():
        c = np.load(CACHE, allow_pickle=True)
        return c["PJ"], list(c["cols"]), c["ST"]
    NM = D["NM"]
    cols = [f"{s}_{m}" for m in MINS for s in STATS]
    df = pd.read_csv(REPO / "data" / "splits" / "train" / "full_dataset.csv",
                     usecols=["match_id", "game_creation", "game_duration", "player_id", "team_id", "lane", "win"] + cols)
    df = df[df.lane.isin(["BOTTOM", "JUNGLE", "MIDDLE", "TOP", "UTILITY"])]
    df = df[df.groupby(["match_id", "team_id"]).lane.transform("nunique") == 5]
    df = df[df.groupby("match_id").lane.transform("size") == 10]
    df = df[df.groupby("match_id").win.transform("sum") == 5]
    df = df.sort_values(["game_creation", "match_id", "team_id", "lane"]).reset_index(drop=True)
    assert (df.match_id.to_numpy()[::10] == D["Z"]["mids"]).all()
    dur = df.game_duration.to_numpy(float)
    previo_v = np.repeat(~D["val0"], 10)                      # filas anteriores al 16/09: para la media de la línea
    lane_cod = np.tile(np.arange(5), NM * 2)
    grupo = df.groupby(["player_id", "lane"], sort=False)
    n_prev = grupo.cumcount().to_numpy(float)
    feats, nombres = [], []
    ST = np.full((NM, 5, len(STATS), len(MINS)), np.nan, np.float32)
    for k, m in enumerate(MINS):
        llego = dur > m * 60
        for j, s in enumerate(STATS):
            a = np.where(llego, df[f"{s}_{m}"].to_numpy(float), np.nan) / ESCALA.get(s, 1.0)
            r = a.reshape(NM, 2, 5)
            d = r[:, 0] - r[:, 1]
            ST[:, :, j, k] = d
            propio = np.stack([d, -d], axis=1).reshape(-1)      # diferencia vista por cada jugador

            def media_previa(v):
                ok = ~np.isnan(v)
                tmp = pd.DataFrame({"v": np.where(ok, v, 0.0), "n": ok.astype(float)})
                acum = tmp.groupby([df.player_id, df.lane], sort=False).cumsum().to_numpy()
                return (acum[:, 0] - tmp.v.to_numpy()) / (acum[:, 1] - tmp.n.to_numpy() + K)

            feats.append(media_previa(propio)); nombres.append(f"d_{s}_{m}")
            if s in CRUDAS:
                mu = np.array([np.nanmean(a[previo_v & (lane_cod == l)]) for l in range(5)])
                feats.append(media_previa(a - mu[lane_cod])); nombres.append(f"abs_{s}_{m}")
    feats.append(np.log1p(n_prev)); nombres.append("logn_linea")
    PJ = np.stack(feats, axis=1).astype(np.float32).reshape(NM, 2, 5, -1)
    np.savez_compressed(CACHE, PJ=PJ, cols=np.array(nombres), ST=ST, stats=np.array(STATS), mins=np.array(MINS))
    return PJ, nombres, ST


if __name__ == "__main__":
    D = cargar()
    ms, NM = D["master"], D["NM"]
    PJ, cols, _ = construir(D)
    dif = PJ[:, 0] - PJ[:, 1]                                   # (NM, 5, 34): azul − rojo por línea
    basicas = [i for i, c in enumerate(cols) if c.startswith("d_") and c.split("_")[1] in ("gold", "xp", "cs")]
    X9 = extras_referencia(D, escalon=9)
    m = lambda X: (lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=X, extra_post=False, con_mapa=False))  # noqa: E731
    VAR = {"escalón 9": m(X9),
           "A. + oro/XP/CS vs rival por línea (45)": m(np.c_[X9, dif[:, :, basicas].reshape(NM, -1)]),
           "B. + todo por línea (170)": m(np.c_[X9, dif.reshape(NM, -1)]),
           "C. + todo, media del equipo (34)": m(np.c_[X9, dif.mean(axis=1)])}
    P = dict(Parallel(n_jobs=2, backend="threading")(delayed(lambda n: (n, evaluar(D, VAR[n])))(n) for n in VAR))
    np.savez_compressed(PRED, po=P["escalón 9"][0], pv=P["escalón 9"][1])
    mostrar(comparar(D, P, "escalón 9"))
