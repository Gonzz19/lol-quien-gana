"""Regla aprendida lineal: z1 − z2 = Σ_partidas (α·x_victoria − β·x_derrota) + decay + ..., ajustada con
pérdida absoluta (QuantileRegressor 0,5) sobre sumas por jugador de interacciones resultado × contexto."""
import pickle, time
import numpy as np, pandas as pd
from sklearn.linear_model import QuantileRegressor
from comun import AQUI, cargar, evaluar, registrar
c = cargar(); e = c["e"]; S = pickle.load(open(AQUI / "seq.pkl", "rb")); est = S["est"]
F = pickle.load(open(AQUI / "decay_cal.pkl", "rb"))
dev, fold = ~c["caja"], c["fold"]
P = S["PASO"]; ir, il, ig, idias = P.index("resultado"), P.index("lp_paso"), P.index("gap"), P.index("dias")
gm, ch = c["corte"]
filas = []
for i, s in enumerate(S["seqs"]):
    w = (s[:, ir] > 0).astype(float); l = (s[:, ir] < 0).astype(float)
    if e["obs"].iloc[i] == 0:
        w[:] = 0; l[:] = 0
    lp = s[:, il] * 300 + e["z2"].iloc[i]
    g = np.where(lp >= ch, 0, np.where(lp >= gm, 1, np.where(lp >= 400, 2, 3)))
    r = {}
    for k in range(4):
        r[f"w{k}"] = (w * (g == k)).sum(); r[f"l{k}"] = (l * (g == k)).sum()
    r["w_gap"] = (w * s[:, ig]).sum(); r["l_gap"] = (l * s[:, ig]).sum()
    r["w_dias"] = (w * s[:, idias]).sum(); r["l_dias"] = (l * s[:, idias]).sum()
    filas.append(r)
X = pd.DataFrame(filas, index=e.index)
X = X.join(F[["dcc_fin", "dcc_ini", "dcc_medio", "c375_fin", "c275_fin", "techo_fin"]])
X = X.join(est[["partidas2", "obs", "g_ch", "g_gm", "g_ma", "piso375", "piso275", "exc_fin", "exc_ini", "lobby_reciente", "lobby_viejo"]])
X["sin_obs"] = (e["obs"] == 0).astype(float)
X["techo_fin"] = X["techo_fin"].clip(upper=1500)
X = (X - X[dev].mean()) / (X[dev].std() + 1e-9)
y = (e["z1"] - e["z2"]).to_numpy()
t0 = time.time()
pred = np.full(len(e), np.nan)
for k in range(5):
    tr, te = dev & (fold != k), dev & (fold == k)
    m = QuantileRegressor(quantile=0.5, alpha=1e-4, solver="highs").fit(X[tr], y[tr])
    pred[te] = m.predict(X[te])
final = e["z2"] + pred
np.save(AQUI / "preds" / "regla_lineal.npy", final.to_numpy())
registrar("regla_lineal", evaluar(e, final, e.index[dev]), f"regla aprendida lineal L1, {X.shape[1]} sumas por jugador ({time.time() - t0:.0f} s)")
print(pd.Series(m.coef_, index=X.columns).round(1).sort_values().to_string())
