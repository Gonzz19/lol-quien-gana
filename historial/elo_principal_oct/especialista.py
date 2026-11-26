"""¿Un CatBoost entrenado solo con un subgrupo (piso + inactivo, o sin partidas) le gana al global ahí?"""
import pickle, time
import numpy as np, pandas as pd
from comun import AQUI, cargar, evaluar, registrar
from zoo import MODELOS, variables
c = cargar(); e = c["e"]; S = pickle.load(open(AQUI / "seq.pkl", "rb")); est = S["est"]
X = variables(c, S); y = e["z1"] - c["regla"]
dev, fold = ~c["caja"], c["fold"]
glob = np.load(AQUI / "preds" / "cat_huber.npy")
grupos = {"piso+inactivo": (e["z2"].isin([275, 325, 375]) & (est["dias_fin"] * 30 > 14)).to_numpy(),
          "inactivo>14": (est["dias_fin"] * 30 > 14).to_numpy(),
          "pocas (<=5)": (e["obs"] <= 5).to_numpy()}
for nombre, g in grupos.items():
    t0 = time.time()
    p = glob.copy()
    for k in range(5):
        tr, te = dev & (fold != k) & g, dev & (fold == k) & g
        p[te] = c["regla"].to_numpy()[te] + MODELOS["cat_huber"][0]().fit(X[tr], y[tr]).predict(X[te])
    m = dev & g
    eg, ee = np.abs(e["z1"].to_numpy() - glob)[m].mean(), np.abs(e["z1"].to_numpy() - p)[m].mean()
    rg, re_ = np.sqrt(((e["z1"].to_numpy() - glob)[m] ** 2).mean()), np.sqrt(((e["z1"].to_numpy() - p)[m] ** 2).mean())
    print(f"{nombre}: n={m.sum()}  global MAE {eg:.1f} RMSE {rg:.1f} | especialista MAE {ee:.1f} RMSE {re_:.1f} ({time.time()-t0:.0f} s)", flush=True)
    np.save(AQUI / "preds" / f"esp_{nombre.split()[0].replace('+','_').replace('>','')}.npy", p)
