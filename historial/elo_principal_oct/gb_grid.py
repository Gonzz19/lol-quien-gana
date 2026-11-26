"""CatBoost Huber + variables del filtro de grilla (las de la corrida de cada fold externo)."""
import pickle, sys, time
import numpy as np, pandas as pd
from comun import AQUI, cargar, evaluar, registrar
from zoo import MODELOS, variables
args_ = [a for a in sys.argv[1:] if not a.startswith("--")]
tag_grid, modelo = args_[0], args_[1] if len(args_) > 1 else "cat_huber"
c = cargar(); e = c["e"]; S = pickle.load(open(AQUI / "seq.pkl", "rb"))
todas = [pickle.load(open(AQUI / f"{t}.pkl", "rb")) for t in tag_grid.split(",")]
X0 = variables(c, S); y = e["z1"] - c["regla"]
dev, fold = ~c["caja"], c["fold"]
t0 = time.time()
pred = pd.Series(np.nan, index=e.index)
caja_modo = "--caja" in sys.argv
for k in (["todo"] if caja_modo else range(5)):
    X = X0.copy()
    for j_, sal in enumerate(todas):
        g = sal[k]
        G = pd.DataFrame({"g_media": g["media"] - c["regla"], "g_mediana": g["mediana"] - c["regla"], "g_sd": g["sd"],
                          "g_p10": g["p10"] - c["regla"], "g_p90": g["p90"] - c["regla"], "g_pm": g["p_master"]})
        X = X.join(G.add_suffix(f"_{j_}"))
    for i_, t_ in enumerate(modelo.split("+an_")[1:]):     # predicciones anidadas (sin fuga) como variables
        A_ = np.load(AQUI / "preds" / f"anid_{t_}.npy")
        if k != "todo":
            X[f"anid{i_}"] = A_[k] * 100
        else:   # entrenar con todo el desarrollo: fuera de fold en dev, modelo de todo el desarrollo en la caja
            oof_ = np.choose(np.clip(fold, 0, 4), A_) * 100
            cj_ = np.load(AQUI / "preds" / f"{t_}_caja.npy") - c["regla"].to_numpy()
            X[f"anid{i_}"] = np.where(dev, oof_, cj_)
    tr, te = (dev, ~dev) if caja_modo else (dev & (fold != k), dev & (fold == k))
    pred[te] = MODELOS[modelo.split("+")[0]][0]().fit(X[tr], y[tr]).predict(X[te])
final = c["regla"] + pred
tag = f"{modelo}+{tag_grid}"
if caja_modo:
    np.save(AQUI / "preds" / f"{tag}_caja.npy", final.to_numpy()); print(f"{tag}: caja lista"); sys.exit()
np.save(AQUI / "preds" / f"{tag}.npy", final.to_numpy())
registrar(tag, evaluar(e, final, e.index[dev]), f"{modelo} + variables del filtro de grilla {tag_grid} ({time.time() - t0:.0f} s)")
