"""Gradient boosting para el objetivo: variables nuevas y regularización. 5 folds sobre desarrollo."""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import AQUI, cargar, evaluar, registrar  # noqa

c = cargar()
e = c["e"]
with open(AQUI / "seq.pkl", "rb") as fh:
    S = pickle.load(fh)
dev, fold = ~c["caja"], c["fold"]
P = S["PASO"]
# Agregados por jugador de las variables por partida (secuencias: más nueva → más vieja).
agg = {}
for nom in ("gap", "kda", "kp", "dmg", "oro", "muertes", "lobby2_rel"):
    j = P.index(nom)
    agg[f"m_{nom}"] = [s[:, j].mean() for s in S["seqs"]]
    agg[f"v_{nom}"] = [s[-10:, j].mean() for s in S["seqs"]]   # las 10 más viejas
jr = P.index("resultado")
agg["wr_viejas"] = [(s[-10:, jr] > 0).mean() for s in S["seqs"]]
agg["wr_nuevas"] = [(s[:10, jr] > 0).mean() for s in S["seqs"]]
jg = P.index("gap")
agg["suma_gap"] = [s[:, jg].sum() for s in S["seqs"]]
agg["suma_gap_w"] = [(s[:, jg] * (s[:, jr] > 0)).sum() for s in S["seqs"]]
A = pd.DataFrame(agg, index=e.index)
EST_N = S["est"][["piso", "dias_fin", "dias_ini", "hueco_max"]]
CONJ = {"comp": c["X_comp"], "comp+est": c["X_comp"].join(EST_N), "comp+est+agg": c["X_comp"].join(EST_N).join(A)}
y = e["z1"] - c["regla"]


def correr(tag, X, nota="", **kw):
    t0 = time.time()
    params = dict(loss="absolute_error", max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50,
                  random_state=0, early_stopping=False)
    params.update(kw)
    pred = pd.Series(np.nan, index=e.index)
    for k in range(5):
        tr, te = dev & (fold != k), dev & (fold == k)
        gb = HistGradientBoostingRegressor(**params).fit(X[tr], y[tr])
        pred[te] = gb.predict(X[te])
    final = c["regla"] + pred
    np.save(AQUI / "preds" / f"{tag}.npy", final.to_numpy())
    registrar(tag, evaluar(e, final, e.index[dev]), f"{nota} ({time.time() - t0:.0f} s)")


(AQUI / "preds").mkdir(exist_ok=True)
correr("gb_comp", CONJ["comp"], "GB completo de referencia")
correr("gb_comp_est", CONJ["comp+est"], "GB + piso/días sin jugar/hueco máx")
correr("gb_comp_est_agg", CONJ["comp+est+agg"], "GB + estáticas + agregados (rendimiento, gap, wr viejas/nuevas)")
X = CONJ["comp+est+agg"]
correr("gb_reg1", X, "regularizado: hojas 200, 15 nodos, l2 1, lr .03, 1500 it, ES interno",
       min_samples_leaf=200, max_leaf_nodes=15, l2_regularization=1.0, learning_rate=0.03, max_iter=1500,
       early_stopping=True, validation_fraction=0.1, n_iter_no_change=50)
correr("gb_reg2", X, "regularizado: hojas 100, 31 nodos, lr .03, 1500 it, ES interno",
       min_samples_leaf=100, learning_rate=0.03, max_iter=1500, early_stopping=True, validation_fraction=0.1,
       n_iter_no_change=50)
correr("gb_reg3", X, "regularizado + submuestreo de variables 0.5",
       min_samples_leaf=100, learning_rate=0.03, max_iter=1500, early_stopping=True, validation_fraction=0.1,
       n_iter_no_change=50, max_features=0.5)
correr("gb_l2", X, "pérdida cuadrática, regularizado", loss="squared_error", min_samples_leaf=100,
       learning_rate=0.03, max_iter=1500, early_stopping=True, validation_fraction=0.1, n_iter_no_change=50)
