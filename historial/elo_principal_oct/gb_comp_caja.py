"""GB completo de referencia (el del pipeline antes del objetivo) entrenado con todo el desarrollo: preds de la caja."""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from comun import AQUI, cargar
c = cargar(); e = c["e"]; dev = ~c["caja"]; X = c["X_comp"]; y = e["z1"] - c["regla"]
gb = HistGradientBoostingRegressor(loss="absolute_error", max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                   min_samples_leaf=50, random_state=0, early_stopping=False).fit(X[dev], y[dev])
np.save(AQUI / "preds" / "gb_comp_caja.npy", (c["regla"] + gb.predict(X)).to_numpy())
print("ok")
