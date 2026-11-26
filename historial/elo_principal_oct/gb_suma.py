"""GB con objetivo agregado: LightGBM predice un delta por fila (cada partida observada + una fila de sesgo por
jugador con sus estáticas) y se entrena con la suma por jugador: pred_i = Σ_filas f(x), pérdida pseudo-Huber
sobre z1 − regla − pred_i. Gradiente propio: cada fila recibe el gradiente de la suma de su jugador.
5 folds del desarrollo. Uso: python gb_suma.py [--rondas N]
"""
import pickle, sys, time
import numpy as np, pandas as pd
import lightgbm as lgb
from comun import AQUI, cargar, evaluar, registrar
c = cargar(); e = c["e"]; S = pickle.load(open(AQUI / "seq.pkl", "rb")); est = S["est"]
dev, fold = ~c["caja"], c["fold"]
rondas = int(sys.argv[sys.argv.index("--rondas") + 1]) if "--rondas" in sys.argv else 600
P = ["resultado", "aliados", "rivales", "lobby", "gap", "lp_paso", "conocidos", "dias", "hueco", "racha"]
ip = [S["PASO"].index(x) for x in P]
EST = ["z2", "pred_regla", "partidas2", "wr_temporada", "obs", "obs_w", "kal_media", "kal_sd", "decay", "piso",
       "dias_fin", "dias_ini", "hueco_max", "lobby_viejo", "lobby_reciente", "exc_fin", "exc_ini", "piso375", "piso275"]
E = est[EST].to_numpy(np.float32)
filas, jug, orden = [], [], []
for i, s in enumerate(S["seqs"]):
    n_ = len(s) if e["obs"].iloc[i] > 0 else 0
    if n_:
        filas.append(np.hstack([s[:, ip], np.tile(E[i], (n_, 1)), np.zeros((n_, 1), np.float32),
                                (np.arange(n_) / max(n_, 1))[:, None].astype(np.float32)]))
        jug.append(np.full(n_, i)); orden.append(np.arange(n_))
    sesgo = np.hstack([np.zeros(len(ip), np.float32), E[i], [1.0], [0.0]]).astype(np.float32)[None, :]
    filas.append(sesgo); jug.append(np.array([i]))
X = np.vstack(filas); J = np.concatenate(jug)
print(f"{len(X)} filas", flush=True)
y = ((e["z1"] - c["regla"]).to_numpy() / 100).astype(np.float64)
DELTA = 1.0
MEDIA = "--media" in sys.argv


def entrenar(tr_j):
    m = np.isin(J, tr_j)
    Xt, Jt = X[m], J[m]
    _, inv = np.unique(Jt, return_inverse=True)
    yt = y[np.unique(Jt)]
    nj = np.bincount(inv)

    def objetivo(pred, _datos):   # lgb.train: (predicciones, Dataset)
        s = np.bincount(inv, weights=pred) / nj if MEDIA else np.bincount(inv, weights=pred)
        r = s - yt
        g = r / np.sqrt(1 + (r / DELTA) ** 2)                 # pseudo-Huber
        h = 1 / (1 + (r / DELTA) ** 2) ** 1.5
        if MEDIA:   # pred = promedio de las filas: grad g/n, hess h/n por fila
            return (g / nj)[inv], (h / nj)[inv]
        return g[inv], (h * nj)[inv]       # Newton para la suma si las filas del jugador se mueven juntas: Σ h m²

    d = lgb.Dataset(Xt, label=np.zeros(len(Xt)), free_raw_data=False)
    return lgb.train({"objective": objetivo, "learning_rate": 0.1, "num_leaves": 31, "min_data_in_leaf": 200,
                      "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1,
                      "num_threads": 8}, d, num_boost_round=rondas)


t0 = time.time()
pred = np.full(len(e), np.nan)
for k in range(5):
    tr_j = np.where(dev & (fold != k))[0]
    te_j = np.where(dev & (fold == k))[0]
    b = entrenar(tr_j)
    m = np.isin(J, te_j)
    pred[te_j] = np.bincount(J[m], weights=b.predict(X[m]), minlength=len(e))[te_j]
    if MEDIA:
        pred[te_j] /= np.bincount(J[m], minlength=len(e))[te_j]
    print(f"fold {k} ({time.time() - t0:.0f} s)", flush=True)
final = pd.Series(c["regla"].to_numpy() + 100 * pred, index=e.index)
tag = "gb_media" if MEDIA else "gb_suma"
np.save(AQUI / "preds" / f"{tag}.npy", final.to_numpy())
registrar(tag, evaluar(e, final, e.index[dev]),
          f"LightGBM con objetivo agregado (delta por partida + sesgo por jugador, pseudo-Huber), {rondas} rondas ({time.time() - t0:.0f} s)")
