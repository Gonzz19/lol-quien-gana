"""
¿Suma la huella mejorada al modelo previo a la partida? (pedido del usuario, 2026-11-18; sigue a
`huella_mejoras.py` y PROJECT_SUMMARY §10)

Las 85 variables por jugador del modelo base ya traen la huella progresiva de las 23 estadísticas (`perf_*`,
§9.S.23). Los pesos de la mejora 1 (Mahalanobis/LDA) son una transformación lineal de esas mismas variables, y una
logística ya aprende sus propios pesos: no pueden sumar. Lo único con información nueva son los **hábitos** de los
timelines completos (mejora 2). Acá se mide eso:

* Hábitos por partida (`data/analysis/huella/habitos.pkl`, de `huella_mejoras.py --parte 2`), como z dentro de
  (campeón, línea, resultado) con medias y desvíos solo de partidas anteriores a V.
* Hábito progresivo de cada jugador en cada partida: promedio de sus partidas **anteriores con timeline**, con peso
  $2^{-\\text{edad}/100}$ y encogido hacia 0, $h = \\sum w_j z_j / (\\sum w_j + 10)$ (igual que §9.S.23), más
  $\\log(1 + n)$. Estrictamente causal: la partida actual y las posteriores no entran.
* Azul − rojo por línea, encima del modelo base del brainstorming (draft + 85 variables + mapa de LP,
  `brainstorm_base.py`), con la regla de siempre: bloques A2-A4 por fecha y V Master+, IC por bootstrap pareado.

    python scripts/analysis/huella_previo.py      # ~5 min
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.brainstorm_base import ajustar_predecir, cargar, comparar, evaluar, mostrar  # noqa: E402
from scripts.analysis.huella_mejoras import HABITOS, SALIDA  # noqa: E402
from src.ganador import rutas  # noqa: E402

LANES = ["BOTTOM", "JUNGLE", "MIDDLE", "TOP", "UTILITY"]   # orden de las matrices (partida, equipo, línea)
VIDA, K = 100, 10


def habito_progresivo(D):
    H = pd.read_pickle(SALIDA / "habitos.pkl")
    mids = set(H["match_id"])
    partes = []
    for ch in pd.read_csv(rutas.TRAIN_CSV, chunksize=500_000,
                          usecols=["match_id", "participant_id", "player_id", "champion_name", "lane", "win",
                                   "game_creation"]):
        partes.append(ch[ch["match_id"].isin(mids)])
    F = pd.concat(partes).merge(H, on=["match_id", "participant_id"], how="inner")
    t_v = D["t"][D["val"]].min()                                     # inicio de V: nada de V en las medias
    ref = F[F["game_creation"] < t_v]
    g = ref.groupby(["champion_name", "lane", "win"])[HABITOS]
    mu, sd = g.mean(), g.std().replace(0, np.nan)
    n = g.size()
    ok = n[n >= 30].index
    clave = pd.MultiIndex.from_frame(F[["champion_name", "lane", "win"]])
    M, S = mu.reindex(clave).to_numpy(), sd.reindex(clave).to_numpy()
    Z = np.where(clave.isin(ok)[:, None], np.nan_to_num((F[HABITOS].to_numpy() - M) / S), 0.0)
    print(f"[*] hábitos de {F['match_id'].nunique():,} partidas, {F['player_id'].nunique():,} jugadores", flush=True)

    # eventos: actualizaciones (partidas con timeline) y consultas (cada jugador en cada partida del modelo)
    NM = D["NM"]
    jug = D["B"]["player"].reshape(-1)
    t_q = np.repeat(D["t"], 10)
    act = pd.DataFrame({"p": F["player_id"].astype(str).to_numpy(), "t": F["game_creation"].to_numpy(), "tipo": 1,
                        "i": np.arange(len(F))})
    con = pd.DataFrame({"p": jug, "t": t_q, "tipo": 0, "i": np.arange(len(jug))})
    E = pd.concat([act, con]).sort_values(["p", "t", "tipo"], kind="mergesort")   # consulta antes que su partida
    p_, tipo_, i_ = E["p"].to_numpy(), E["tipo"].to_numpy(), E["i"].to_numpy()
    out = np.zeros((len(jug), len(HABITOS)), np.float32)
    nn = np.zeros(len(jug), np.float32)
    dec = 2 ** (-1 / VIDA)
    s, w, prev = np.zeros(len(HABITOS)), 0.0, None
    for p, tp, i in zip(p_, tipo_, i_):
        if p != prev:
            s[:], w, prev = 0.0, 0.0, p
        if tp == 1:
            s = s * dec + Z[i]
            w = w * dec + 1.0
        else:
            out[i] = s / (w + K)
            nn[i] = w
    X = np.c_[out, np.log1p(nn)].reshape(NM, 2, 5, -1)
    cubre = (nn.reshape(NM, 10) > 0).mean(1)
    return (X[:, 0] - X[:, 1]).reshape(NM, -1).astype(np.float32), cubre


def main():
    D = cargar()
    Hx, cubre = habito_progresivo(D)
    print(f"[*] jugadores con historial de hábitos: V {cubre[D['val']].mean():.0%}, "
          f"ajuste {cubre[D['fit']].mean():.0%}", flush=True)
    P = {"base (draft + 85 variables + mapa)": evaluar(D, lambda tr, te: ajustar_predecir(D, tr, te)),
         "+ hábitos progresivos": evaluar(D, lambda tr, te: ajustar_predecir(D, tr, te, extra=Hx, extra_post=False))}
    T = comparar(D, P, "base (draft + 85 variables + mapa)")
    mostrar(T)
    T.to_csv(SALIDA / "huella_previo.csv", index=False)


if __name__ == "__main__":
    main()
