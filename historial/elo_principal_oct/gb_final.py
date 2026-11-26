"""GB final (variables de gb_decay2_800): 5 folds en desarrollo y, con --caja, todo el desarrollo → caja cerrada."""
import pickle, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import AQUI, cargar, evaluar, registrar  # noqa
caja_modo = "--caja" in sys.argv
c = cargar(); e = c["e"]
with open(AQUI / "seq.pkl", "rb") as fh:
    S = pickle.load(fh)
dev, fold = ~c["caja"], c["fold"]
est = S["est"]
D = pd.DataFrame({"exc_fin": est["exc_fin"] * 30, "exc_ini": est["exc_ini"] * 30, "exc_medio": est["exc_medio"] * 30,
                  "n_huecos14": est["n_huecos14"] * 3, "partidas_antes_fin": est["antes_fin"] * 20,
                  "piso375": est["piso375"], "piso275": est["piso275"]})
X = c["X_comp"].join(est[["piso", "dias_fin", "dias_ini", "hueco_max"]]).join(D)
y = e["z1"] - c["regla"]
for tag, loss in (("gbF_l1", "absolute_error"), ("gbF_l2", "squared_error")):
    t0 = time.time()
    params = dict(loss=loss, max_iter=800, learning_rate=0.03, max_leaf_nodes=31, min_samples_leaf=50 if loss == "absolute_error" else 100,
                  random_state=0, early_stopping=False)
    pred = pd.Series(np.nan, index=e.index)
    if caja_modo:
        pred[~dev] = HistGradientBoostingRegressor(**params).fit(X[dev], y[dev]).predict(X[~dev]); idx = e.index[~dev]
    else:
        for k in range(5):
            tr, te = dev & (fold != k), dev & (fold == k)
            pred[te] = HistGradientBoostingRegressor(**params).fit(X[tr], y[tr]).predict(X[te])
        idx = e.index[dev]
    final = c["regla"] + pred
    np.save(AQUI / "preds" / f"{tag}{'_caja' if caja_modo else ''}.npy", final.to_numpy())
    registrar(("CAJA " if caja_modo else "") + tag, evaluar(e, final, idx), f"GB final, pérdida {loss} ({time.time() - t0:.0f} s)")
