"""NGBoost y GB (nivel completo) fuera de fold, 5 folds por jugador: métricas honestas y μ, σ para el stacking de la GRU."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from ngboost import NGBRegressor
from ngboost.distns import Normal
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(r"c:/Users/DarkA/OneDrive/Escritorio/Facultad/3er año/ML/TPF/ProyectoFinalML")
sys.path.insert(0, str(ROOT))
from src.features.elo_reverso import (ajustar_regla, al_reves, cargar_train, cortes, decay_total, evaluables,  # noqa
                                      kalman_jugadores, lobby_por_fila, variables_estaticas)

d = cargar_train(ROOT / "data"); foto2, partes = d["foto2"], d["partes"]
e = evaluables(d["foto1"], foto2, partes); CORTE = cortes(foto2); A, B = ajustar_regla(e)
regla = al_reves(partes, foto2["z"], A, B, CORTE).reindex(e.index).fillna(e["z2"])
_, LP_FILA = al_reves(partes, foto2["z"], A, B, CORTE, devolver_camino=True)
LOB_PRED = lobby_por_fila(partes, LP_FILA, por_fila=True)
km, kv = kalman_jugadores(e, partes, foto2["z"], A, B, CORTE)
DECAY = decay_total(partes, d["t1"], d["t2"]).reindex(e.index).fillna(0.0)
X = (variables_estaticas(e, partes, foto2["z"], regla, lob=LOB_PRED).drop(columns="ocultas")
     .assign(kal_media=km - e["z2"], kal_sd=np.sqrt(kv), decay=DECAY))
y = e["z1"] - regla
cambio = e["z1"] - e["z2"]
fold = np.random.default_rng(321).integers(0, 5, len(e))
mu, sigma, gbp = np.zeros(len(e)), np.zeros(len(e)), np.zeros(len(e))
for k in range(5):
    t0 = time.time()
    tr, te = fold != k, fold == k
    ngb = NGBRegressor(Dist=Normal, n_estimators=800, learning_rate=0.05, verbose=False, random_state=0,
                       Base=DecisionTreeRegressor(max_depth=5, min_samples_leaf=50)).fit(X[tr].fillna(-999), y[tr] / 100)
    dist = ngb.pred_dist(X[te].fillna(-999))
    mu[te], sigma[te] = 100 * dist.params["loc"], 100 * dist.params["scale"]
    gb = HistGradientBoostingRegressor(loss="absolute_error", max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                       min_samples_leaf=50, random_state=0, early_stopping=False).fit(X[tr], y[tr])
    gbp[te] = gb.predict(X[te])
    print(f"fold {k}: {time.time() - t0:.0f} s", flush=True)


def evaluar(pred):
    err = e["z1"] - pred
    rho = [(len(g), g["z1"].corr(pred[g.index], method="spearman"))
           for _, g in e.groupby("tier2") if len(g) >= 30]
    return {"MAE": err.abs().mean(), "RMSE": np.sqrt((err ** 2).mean()),
            "R² del cambio": 1 - (err ** 2).sum() / ((cambio - cambio.mean()) ** 2).sum(),
            "ρ dentro del rango": sum(n * r for n, r in rho) / sum(n for n, _ in rho),
            "cobertura IC 95 %": np.nan}


r = {"NGBoost OOF": evaluar(regla + mu), "GB OOF": evaluar(regla + gbp)}
r["NGBoost OOF"]["cobertura IC 95 %"] = 100 * (np.abs(y - mu) <= 1.96 * sigma).mean()
print(pd.DataFrame(r).T.round(3).to_string())
pd.DataFrame({"player_id": e.index, "ngb_mu": mu, "ngb_sigma": sigma, "gb": gbp, "fold": fold}).to_csv(
    ROOT / "data" / "analysis" / "boost_completo_oof.csv", index=False)
