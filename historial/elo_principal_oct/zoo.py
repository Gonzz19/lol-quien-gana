"""Zoológico de boostings sobre las variables del GB final (gbF), residuo de la regla, 5 folds del desarrollo.

Uso: python zoo.py modelo1 modelo2 ...   (ver MODELOS). Guarda preds/<modelo>.npy.
"""
import pickle, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import AQUI, cargar, evaluar, registrar  # noqa


def variables(c, S):
    e, est = c["e"], S["est"]
    D = pd.DataFrame({"exc_fin": est["exc_fin"] * 30, "exc_ini": est["exc_ini"] * 30, "exc_medio": est["exc_medio"] * 30,
                      "n_huecos14": est["n_huecos14"] * 3, "partidas_antes_fin": est["antes_fin"] * 20,
                      "piso375": est["piso375"], "piso275": est["piso275"]})
    return c["X_comp"].join(est[["piso", "dias_fin", "dias_ini", "hueco_max"]]).join(D)


def lgb(obj, **kw):
    import lightgbm as L
    p = dict(objective=obj, n_estimators=1000, learning_rate=0.03, num_leaves=31, min_child_samples=50,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, n_jobs=8)
    p.update(kw)
    return lambda: L.LGBMRegressor(**p)


def cat(loss, **kw):
    from catboost import CatBoostRegressor
    p = dict(loss_function=loss, iterations=2000, learning_rate=0.05, depth=6, l2_leaf_reg=3, verbose=0,
             thread_count=8, random_seed=0)
    p.update(kw)
    return lambda: CatBoostRegressor(**p)


def xgb(obj, **kw):
    import xgboost as X
    p = dict(objective=obj, n_estimators=1000, learning_rate=0.03, max_depth=6, min_child_weight=20,
             subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=8)
    p.update(kw)
    return lambda: X.XGBRegressor(**p)


class Bolsa:
    """Promedio de varias semillas (y opcionalmente submuestreo de filas/columnas) de un mismo modelo."""
    def __init__(self, fabricas):
        self.fabricas = fabricas

    def fit(self, X, y):
        self.m = [f().fit(X, y) for f in self.fabricas]
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.m], 0)


def bolsa(loss, semillas=5, **kw):
    return lambda: Bolsa([cat(loss, random_seed=s, **kw) for s in range(semillas)])


def arboles(tipo, **kw):
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    cls = ExtraTreesRegressor if tipo == "et" else RandomForestRegressor
    p = dict(n_estimators=500, min_samples_leaf=20, max_features=0.5, n_jobs=8, random_state=0)
    p.update(kw)
    return lambda: _SinNaN(cls(**p))


class _SinNaN:
    """Los bosques de sklearn no aceptan NaN: se rellenan con la mediana del entrenamiento."""
    def __init__(self, m):
        self.m = m

    def fit(self, X, y):
        self.med = X.median()
        self.m.fit(X.fillna(self.med), y)
        return self

    def predict(self, X):
        return self.m.predict(X.fillna(self.med))


MODELOS = {
    "et": (arboles("et"), "ExtraTrees 500, hoja 20, columnas .5"),
    "rf": (arboles("rf"), "RandomForest 500, hoja 20, columnas .5"),
    "cat_bag5": (bolsa("Huber:delta=60"), "CatBoost Huber 60, promedio de 5 semillas"),
    "cat_bag5_rsm": (bolsa("Huber:delta=60", rsm=0.7, bootstrap_type="Bernoulli", subsample=0.8),
                     "CatBoost Huber 60, 5 semillas, columnas .7 y filas .8"),
    "cat_bag5_h100": (bolsa("Huber:delta=100"), "CatBoost Huber 100, 5 semillas"),
    "lgb_l1": (lgb("l1"), "LightGBM L1, 1000 árboles, 31 hojas, bagging .8"),
    "lgb_huber": (lgb("huber", alpha=60.0), "LightGBM Huber δ=60 LP"),
    "lgb_l2": (lgb("l2"), "LightGBM L2"),
    "lgb_q25": (lgb("quantile", alpha=0.25), "LightGBM cuantil .25"),
    "lgb_q75": (lgb("quantile", alpha=0.75), "LightGBM cuantil .75"),
    "lgb_dart": (lgb("l1", boosting_type="dart", n_estimators=600, learning_rate=0.05), "LightGBM DART L1"),
    "cat_l1": (cat("MAE"), "CatBoost MAE, profundidad 6, 2000 it"),
    "cat_l1_d8": (cat("MAE", depth=8), "CatBoost MAE, profundidad 8"),
    "cat_rmse": (cat("RMSE"), "CatBoost RMSE"),
    "cat_huber": (cat("Huber:delta=60"), "CatBoost Huber δ=60"),
    "xgb_l1": (xgb("reg:absoluteerror"), "XGBoost L1, profundidad 6"),
    "cat_huber_d5": (cat("Huber:delta=60", depth=5), "CatBoost Huber 60, profundidad 5"),
    "cat_huber_d7": (cat("Huber:delta=60", depth=7), "CatBoost Huber 60, profundidad 7"),
    "cat_huber_lento": (cat("Huber:delta=60", iterations=5000, learning_rate=0.02), "CatBoost Huber 60, 5000 it lr .02"),
    "cat_huber30": (cat("Huber:delta=30"), "CatBoost Huber δ=30"),
    "cat_huber100": (cat("Huber:delta=100"), "CatBoost Huber δ=100"),
    "cat_huber_l2r10": (cat("Huber:delta=60", l2_leaf_reg=10), "CatBoost Huber 60, l2 10"),
    "cat_huber_s1": (cat("Huber:delta=60", random_seed=1), "CatBoost Huber 60, semilla 1"),
    "cat_huber_s2": (cat("Huber:delta=60", random_seed=2), "CatBoost Huber 60, semilla 2"),
    "cat_q50": (cat("Quantile:alpha=0.5"), "CatBoost cuantil .5"),
    "cat_logcosh": (cat("LogCosh"), "CatBoost LogCosh"),
    "xgb_pseudohuber": (xgb("reg:pseudohubererror", huber_slope=60.0), "XGBoost pseudo-Huber 60"),
}

if __name__ == "__main__":
    c = cargar(); e = c["e"]
    S = pickle.load(open(AQUI / "seq.pkl", "rb"))
    X = variables(c, S)
    y = e["z1"] - c["regla"]
    dev, fold = ~c["caja"], c["fold"]
    F = pickle.load(open(AQUI / "decay_cal.pkl", "rb"))
    DC = ["dcc_fin", "dcc_ini", "dcc_medio", "c375_fin", "c275_fin", "techo_fin", "c375_ini", "c275_ini", "techo_ini",
          "cens_medio", "oculto_fin", "oculto_ini"]
    X_base = X
    caja_modo = "--caja" in sys.argv
    for nombre in [a for a in sys.argv[1:] if not a.startswith("--")]:
        hacer, nota = MODELOS[nombre.split("+")[0].split("@")[0]]
        nota += "".join(f" + predicción anidada de {t}" for t in nombre.split("+an_")[1:])
        X = X_base.join(F[DC]) if "+dc" in nombre else X_base
        base = {"z2": e["z2"].astype(float), "km": c["km"]}.get(nombre.split("@")[1], None) if "@" in nombre else c["regla"]
        y = e["z1"] - base
        nota += f" (base {nombre.split('@')[1]})" if "@" in nombre else ""
        nota += " + decay calibrado" if nombre.endswith("+dc") else ""
        t0 = time.time()
        pred = pd.Series(np.nan, index=e.index)
        if caja_modo:   # todo el desarrollo → caja cerrada
            pred[~dev] = hacer().fit(X[dev], y[dev]).predict(X[~dev])
            final = base + pred
            np.save(AQUI / "preds" / f"{nombre}_caja.npy", final.to_numpy())
            print(f"{nombre}: caja lista ({time.time() - t0:.0f} s)", flush=True)
            continue
        anid = [np.load(AQUI / "preds" / f"anid_{t}.npy") for t in nombre.split("+an_")[1:]] if "+an_" in nombre else []
        for k in range(5):
            tr, te = dev & (fold != k), dev & (fold == k)
            Xk = X.assign(**{f"anid{i}": A[k] * 100 for i, A in enumerate(anid)})   # predicción anidada (sin fuga)
            pred[te] = hacer().fit(Xk[tr], y[tr]).predict(Xk[te])
        final = base + pred
        np.save(AQUI / "preds" / f"{nombre}.npy", final.to_numpy())
        registrar(nombre, evaluar(e, final, e.index[dev]), f"{nota} ({time.time() - t0:.0f} s)")
