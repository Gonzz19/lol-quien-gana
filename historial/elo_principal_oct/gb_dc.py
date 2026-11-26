"""GB final + decay calibrado (decay_cal.pkl): como variables y/o como base."""
import pickle, sys, time
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from comun import AQUI, cargar, evaluar, registrar, linea
from zoo import variables
c = cargar(); e = c["e"]; S = pickle.load(open(AQUI / "seq.pkl", "rb"))
F = pickle.load(open(AQUI / "decay_cal.pkl", "rb"))
dev, fold = ~c["caja"], c["fold"]
# Devoluciones imposibles (LP previo > 1200) = jugó partidas que no vemos: no hubo decay.
for p in ("fin", "ini"):
    zdesp = e["z2"] if p == "fin" else c["regla"] + F["dc_fin"]
    F[f"dcc_{p}"] = np.where(zdesp + F[f"dc_{p}"] > 1200, 0, F[f"dc_{p}"])
    F[f"oculto_{p}"] = (zdesp + F[f"dc_{p}"] > 1200).astype(float)
F["dcc_medio"] = np.where(F["dc_medio"] > 800, 0, F["dc_medio"])
F["base_dcc"] = c["regla"] + F["dcc_fin"] + F["dcc_ini"] + F["dcc_medio"]
pickle.dump(F, open(AQUI / "decay_cal.pkl", "wb"))
for n, b in [("regla", c["regla"]), ("regla + decay calibrado", F["base_dcc"]), ("regla + decay viejo", c["regla"] + c["decay"])]:
    print(linea(n, evaluar(e, b, e.index[dev])))
X0 = variables(c, S)
nuevas = ["dcc_fin", "dcc_ini", "dcc_medio", "c375_fin", "c275_fin", "techo_fin", "c375_ini", "c275_ini", "techo_ini",
          "cens_medio", "oculto_fin", "oculto_ini"]
params = dict(loss="absolute_error", max_iter=800, learning_rate=0.03, max_leaf_nodes=31, min_samples_leaf=50, random_state=0, early_stopping=False)
for tag, X, base, nota in [
        ("gb_dc_var", X0.join(F[nuevas]), c["regla"], "gbF + decay calibrado como variables, base regla"),
        ("gb_dc_base", X0.join(F[nuevas]), F["base_dcc"], "gbF + decay calibrado, base regla + decay calibrado"),
        ("gb_dc_solo", X0.drop(columns=["decay"]).join(F[nuevas]), c["regla"], "gbF sin el decay viejo + calibrado")]:
    t0 = time.time()
    y = e["z1"] - base
    pred = pd.Series(np.nan, index=e.index)
    for k in range(5):
        tr, te = dev & (fold != k), dev & (fold == k)
        pred[te] = HistGradientBoostingRegressor(**params).fit(X[tr], y[tr]).predict(X[te])
    final = base + pred
    np.save(AQUI / "preds" / f"{tag}.npy", final.to_numpy())
    registrar(tag, evaluar(e, final, e.index[dev]), f"{nota} ({time.time() - t0:.0f} s)")
