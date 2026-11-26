"""Fuerza de cada jugador por Bradley-Terry sobre los resultados (sin LP ni fotos: sin fuga).

P(gana azul) = σ(h + Σ s_azul − Σ s_rojo). Cada jugador se parte en épocas según el orden de
sus partidas (época 0 = las más viejas, cerca de la foto 1). La regularización C se elige por
log-loss de partidas guardadas (no mira el target). Escribe bt.pkl.
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import AQUI, cargar  # noqa

c = cargar()
p = c["partes"][["match_id", "team_id", "player_id", "win", "game_creation"]].copy()
t0 = time.time()


def ajustar(epocas: int, C: float, filas_ajuste=None):
    q = p.sort_values(["player_id", "game_creation"])
    rk = q.groupby("player_id").cumcount()
    n = q.groupby("player_id")["match_id"].transform("size")
    q["ep"] = np.minimum((epocas * rk / n).astype(int), epocas - 1)
    ent = q["player_id"].astype(str) + "#" + q["ep"].astype(str)
    codigos, uniq = pd.factorize(ent)
    mcod, muniq = pd.factorize(q["match_id"])
    signo = np.where(q["team_id"].to_numpy() == 100, 1.0, -1.0)
    X = sparse.csr_matrix((signo, (mcod, codigos)), shape=(len(muniq), len(uniq)))
    azul = q[q["team_id"] == 100].drop_duplicates("match_id").set_index("match_id")["win"]
    y = azul.reindex(muniq).to_numpy().astype(int)
    if filas_ajuste is None:
        filas_ajuste = np.ones(len(y), bool)
    m = LogisticRegression(C=C, max_iter=2000).fit(X[filas_ajuste], y[filas_ajuste])
    return m, X, y, uniq, muniq, q, codigos


# Elegir C con 20 % de partidas guardadas, por log-loss (no usa LP).
rng = np.random.default_rng(0)
for ep in (1, 3):
    for C in (0.03, 0.1, 0.3, 1.0):
        m, X, y, *_ = ajustar(ep, C, None if False else (rng0 := np.random.default_rng(1).random(134695) < 0.8))
        print(f"épocas {ep}, C {C}: log-loss guardadas {log_loss(y[~rng0], m.predict_proba(X[~rng0])[:, 1]):.4f} "
              f"({time.time() - t0:.0f} s)", flush=True)
