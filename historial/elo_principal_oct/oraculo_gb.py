"""Techo: GB final + las ocultas exactas (información de la foto 1, solo para medir cuánto valen)."""
import pickle, sys; from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import AQUI, cargar, evaluar, linea
c = cargar(); e = c["e"]; S = pickle.load(open(AQUI / "seq.pkl", "rb")); est = S["est"]
dev, fold = ~c["caja"], c["fold"]
D = pd.DataFrame({"exc_fin": est["exc_fin"] * 30, "exc_ini": est["exc_ini"] * 30, "exc_medio": est["exc_medio"] * 30,
                  "n_huecos14": est["n_huecos14"] * 3, "partidas_antes_fin": est["antes_fin"] * 20,
                  "piso375": est["piso375"], "piso275": est["piso275"]})
X0 = c["X_comp"].join(est[["piso", "dias_fin", "dias_ini", "hueco_max"]]).join(D)
hW = e["dW"] - e["obs_w"]; hL = (e["dL"] - (e["obs"] - e["obs_w"])).clip(lower=0)
y = e["z1"] - c["regla"]
params = dict(loss="absolute_error", max_iter=800, learning_rate=0.03, max_leaf_nodes=31, min_samples_leaf=50, random_state=0, early_stopping=False)
for nombre, X in [("oráculo: + ocultas totales", X0.assign(oc=hW + hL)),
                  ("oráculo: + victorias y derrotas ocultas", X0.assign(hW=hW, hL=hL, neto=hW - hL))]:
    pred = pd.Series(np.nan, index=e.index)
    for k in range(5):
        tr, te = dev & (fold != k), dev & (fold == k)
        pred[te] = HistGradientBoostingRegressor(**params).fit(X[tr], y[tr]).predict(X[te])
    p = c["regla"] + pred
    print(linea(nombre, evaluar(e, p, e.index[dev])), {t: round(np.abs(e["z1"] - p)[dev & (e["tier2"] == t).to_numpy()].mean(), 1) for t in ["DIAMOND", "MASTER"]}, flush=True)
