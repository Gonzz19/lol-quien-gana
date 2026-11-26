"""¿Por qué el gradient boosting da 74,5 acá y 70,0 en el notebook commiteado? Prueba early stopping."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(r"c:/Users/DarkA/OneDrive/Escritorio/Facultad/3er año/ML/TPF/ProyectoFinalML")
sys.path.insert(0, str(ROOT))
from src.features.elo_reverso import (ajustar_regla, al_reves, cargar_train, cortes, evaluables,  # noqa
                                      kalman_jugadores, lobby_por_fila, variables_estaticas)

t0 = time.time()
d = cargar_train(ROOT / "data"); foto2, partes = d["foto2"], d["partes"]
e = evaluables(d["foto1"], foto2, partes); CORTE = cortes(foto2); A, B = ajustar_regla(e)
_, LP_FILA = al_reves(partes, foto2["z"], A, B, CORTE, devolver_camino=True)
LOB_PRED = lobby_por_fila(partes, LP_FILA, por_fila=True)
km, kv = kalman_jugadores(e, partes, foto2["z"], A, B, CORTE)
print(f"Kalman MAE {np.abs(e['z1'] - km).mean():.1f} ({time.time() - t0:.0f} s)")
base = al_reves(partes, foto2["z"], A, B, CORTE).reindex(e.index).fillna(e["z2"])
X = (variables_estaticas(e, partes, foto2["z"], base, lob=LOB_PRED).drop(columns="ocultas")
     .assign(kal_media=km - e["z2"], kal_sd=np.sqrt(kv)))
for es in ("auto", False, True):
    hgb = HistGradientBoostingRegressor(loss="absolute_error", max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                        min_samples_leaf=50, random_state=0, early_stopping=es).fit(X, e["z1"] - base)
    print(f"early_stopping={es}: MAE {np.abs(e['z1'] - base - hgb.predict(X)).mean():.1f}, iteraciones {hgb.n_iter_}")
