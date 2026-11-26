"""GB: variables de decay más finas (hueco final, del medio, inicial; partidas antes del hueco final)."""
import pickle, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import AQUI, cargar, evaluar, registrar  # noqa
c = cargar(); e, partes = c["e"], c["partes"]
with open(AQUI / "seq.pkl", "rb") as fh:
    S = pickle.load(fh)
dev, fold = ~c["caja"], c["fold"]
DIA = 86_400_000
ps = partes[partes["player_id"].isin(e.index)].sort_values(["player_id", "game_creation"])
g = ps.groupby("player_id")["game_creation"]
fin = ((c["t2"] - g.max()) / DIA).reindex(e.index).fillna(55)
ini = ((g.min() - c["t1"]) / DIA).reindex(e.index).fillna(55)
huecos = g.diff() / DIA
medio = (huecos - 14).clip(lower=0).groupby(ps["player_id"]).sum().reindex(e.index).fillna(0)
n_huecos14 = (huecos > 14).groupby(ps["player_id"]).sum().reindex(e.index).fillna(0)
ult = g.max().reindex(e.index)
antes_fin = ps.assign(u=ps["player_id"].map(ult))
antes_fin = antes_fin[antes_fin["game_creation"] >= antes_fin["u"] - 14 * DIA].groupby("player_id").size().reindex(e.index).fillna(0)
D = pd.DataFrame({"exc_fin": (fin - 14).clip(lower=0), "exc_ini": (ini - 14).clip(lower=0), "exc_medio": medio,
                  "n_huecos14": n_huecos14, "partidas_antes_fin": antes_fin,
                  "piso375": (e["z2"] == 375).astype(float), "piso275": (e["z2"] == 275).astype(float)})
EST_N = S["est"][["piso", "dias_fin", "dias_ini", "hueco_max"]]
y = e["z1"] - c["regla"]
def correr(tag, X, nota="", **kw):
    t0 = time.time()
    params = dict(loss="absolute_error", max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50,
                  random_state=0, early_stopping=False); params.update(kw)
    pred = pd.Series(np.nan, index=e.index)
    for k in range(5):
        tr, te = dev & (fold != k), dev & (fold == k)
        pred[te] = HistGradientBoostingRegressor(**params).fit(X[tr], y[tr]).predict(X[te])
    final = c["regla"] + pred
    np.save(AQUI / "preds" / f"{tag}.npy", final.to_numpy())
    registrar(tag, evaluar(e, final, e.index[dev]), f"{nota} ({time.time() - t0:.0f} s)")
X1 = c["X_comp"].join(EST_N)
X2 = X1.join(D)
correr("gb_decay2", X2, "GB + piso + decay fino (excesos fin/ini/medio, banco, pisos 375/275)")
correr("gb_decay2_800", X2, "idem, 800 iteraciones lr .03", max_iter=800, learning_rate=0.03)
for s in (1, 2):
    correr(f"gb_decay2_s{s}", X2, f"idem, semilla {s} (ruido)", random_state=s)
