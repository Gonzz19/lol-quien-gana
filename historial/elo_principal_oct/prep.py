"""Objetivo de métricas: precalcula todo una vez y lo guarda en cache.pkl.

Protocolo anti-sobreajuste:
* caja cerrada: 15 % de jugadores por un hash distinto del de los folds; no se mira hasta el final.
* desarrollo: el 85 % restante, con 5 folds de `folds_por_jugador`.
* a, b de la regla se ajustan solo con desarrollo.
"""
import pickle
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"c:/Users/DarkA/OneDrive/Escritorio/Facultad/3er año/ML/TPF/ProyectoFinalML")
sys.path.insert(0, str(ROOT))
from src.features.elo_reverso import (ajustar_regla, al_reves, cargar_train, cortes, decay_total, evaluables,  # noqa
                                      folds_por_jugador, kalman_jugadores, lobby_por_fila, variables_estaticas)

AQUI = Path(__file__).resolve().parent
t0 = time.time()
d = cargar_train(ROOT / "data")
foto2, partes = d["foto2"], d["partes"]
e = evaluables(d["foto1"], foto2, partes)
caja = np.array([zlib.crc32(f"caja-{i}".encode()) % 100 < 15 for i in e.index])
fold = folds_por_jugador(e.index)
corte = cortes(foto2)
a, b = ajustar_regla(e[~caja])
z = foto2["z"]
regla = al_reves(partes, z, a, b, corte).reindex(e.index).fillna(e["z2"])
_, lp_fila = al_reves(partes, z, a, b, corte, devolver_camino=True)
lob2 = lobby_por_fila(partes, z)
lobp = lobby_por_fila(partes, lp_fila, por_fila=True)
km, kv = kalman_jugadores(e, partes, z, a, b, corte)
decay = decay_total(partes, d["t1"], d["t2"]).reindex(e.index).fillna(0.0)
X_simple = variables_estaticas(e, partes, z, regla).drop(columns="ocultas")
X_comp = (variables_estaticas(e, partes, z, regla, lob=lobp).drop(columns="ocultas")
          .assign(kal_media=km - e["z2"], kal_sd=np.sqrt(kv), decay=decay))
print(f"listo en {time.time() - t0:.0f} s; caja {caja.sum()} jugadores, desarrollo {(~caja).sum()}")
with open(AQUI / "cache.pkl", "wb") as f:
    pickle.dump({"e": e, "partes": partes, "foto2": foto2, "t1": d["t1"], "t2": d["t2"], "caja": caja,
                 "fold": fold, "corte": corte, "a": a, "b": b, "regla": regla, "lp_fila": lp_fila,
                 "lob2": lob2, "lobp": lobp, "km": km, "kv": kv, "decay": decay,
                 "X_simple": X_simple, "X_comp": X_comp}, f)
