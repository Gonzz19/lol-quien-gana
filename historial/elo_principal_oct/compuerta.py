"""Stacking con compuerta: una red chica mira el contexto del jugador y reparte pesos entre los expertos.

pred = Σ_i w_i(contexto) · experto_i, con w = softmax(red(contexto)). Con --correccion la red suma además
un término libre (en LP). Entrada: predicciones fuera de fold de preds/<tag>.npy. Validación: 5 folds del
desarrollo; dentro de cada fold externo, el número de épocas sale de un fold interno (anidado).
Con --gb, en vez de la red usa un HistGB L1 sobre (expertos relativos + contexto).

Uso: python compuerta.py tag1 tag2 ... [--correccion] [--gb] [--nombre X]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import AQUI, cargar, evaluar, registrar  # noqa
import pickle

banderas = {a for a in sys.argv[1:] if a.startswith("--")}
nombre = next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--nombre"), None)
ctx_tags = next((sys.argv[i + 1].split(",") for i, a in enumerate(sys.argv) if a == "--ctx"), [])
tags = [t for i, t in enumerate(sys.argv[1:], 1) if not t.startswith("--") and sys.argv[i - 1] not in ("--nombre", "--ctx")]
c = cargar()
e = c["e"]
with open(AQUI / "seq.pkl", "rb") as fh:
    S = pickle.load(fh)
dev, fold = ~c["caja"], c["fold"]
P = np.column_stack([np.load(AQUI / "preds" / f"{t}.npy") for t in tags]) / 100     # en divisiones
z1 = e["z1"].to_numpy() / 100
largos = np.array([len(s) for s in S["seqs"]])

# Contexto: lo que se sabe en la foto 2 (nada de la foto 1: ni dN ni ocultas).
est = S["est"]
ctx = pd.DataFrame({
    "log_obs": np.log1p(largos), "obs": est["obs"], "obs_w": est["obs_w"], "z2": est["z2"],
    "partidas2": est["partidas2"], "kal_sd": est["kal_sd"], "decay": est["decay"], "piso": est["piso"],
    "dias_fin": est["dias_fin"], "dias_ini": est["dias_ini"], "hueco_max": est["hueco_max"],
    "g_ch": est["g_ch"], "g_gm": est["g_gm"], "g_ma": est["g_ma"], "piso375": est["piso375"], "piso275": est["piso275"],
    "dispersion": P.std(1),
})
for i, t in enumerate(tags):
    ctx[f"desvio_{t}"] = P[:, i] - P.mean(1)
for t in ctx_tags:   # predicciones que solo dan contexto (p. ej. cuantiles)
    ctx[f"ctx_{t}"] = np.load(AQUI / "preds" / f"{t}.npy") / 100 - P.mean(1)
C = ctx.to_numpy(np.float32)
C = (C - C[dev].mean(0)) / (C[dev].std(0) + 1e-6)


class Compuerta(nn.Module):
    def __init__(self):
        super().__init__()
        self.red = nn.Sequential(nn.Linear(C.shape[1], 32), nn.ReLU(), nn.Linear(32, len(tags) + 1))

    def forward(self, x, p):
        o = self.red(x)
        w = o[:, :-1].softmax(1)
        corr = o[:, -1] if "--correccion" in banderas else 0
        return (w * p).sum(1) + corr, w


def entrenar(tr, va, epocas, semilla):
    torch.manual_seed(semilla)
    m = Compuerta()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3, weight_decay=1e-4)
    x, p, t = (torch.from_numpy(a[tr].astype(np.float32)) for a in (C, P, z1))
    hist = []
    for ep in range(epocas):
        m.train()
        for b in torch.randperm(len(tr)).split(1024):
            opt.zero_grad()
            r = m(x[b], p[b])[0] - t[b]
            (r.abs().mean() if "--huber" not in banderas else torch.nn.functional.huber_loss(r, torch.zeros_like(r), delta=1.0)).backward()
            opt.step()
        if va is not None:
            m.eval()
            with torch.no_grad():
                hist.append(np.abs(predecir(m, va)[0] - z1[va]).mean())
    return m, hist


def predecir(m, idx):
    with torch.no_grad():
        y, w = m(torch.from_numpy(C[idx]), torch.from_numpy(P[idx].astype(np.float32)))
    return y.numpy(), w.numpy()


final = np.full(len(e), np.nan)
W = np.full((len(e), len(tags)), np.nan)
if "--gb" in banderas:
    X = np.hstack([C, P - P.mean(1, keepdims=True)])
    for k in range(5):
        tr, te = dev & (fold != k), dev & (fold == k)
        gb = HistGradientBoostingRegressor(loss="absolute_error", max_iter=300, learning_rate=0.03, max_leaf_nodes=15,
                                           min_samples_leaf=200, random_state=0, early_stopping=False)
        final[te] = P[te].mean(1) + gb.fit(X[tr], (z1 - P.mean(1))[tr]).predict(X[te])
else:
    for k in range(5):
        tr = np.where(dev & (fold != k))[0]
        te = np.where(dev & (fold == k))[0]
        j = (k + 1) % 5
        itr, iva = np.where(dev & (fold != k) & (fold != j))[0], np.where(dev & (fold == j))[0]
        hist = np.mean([entrenar(itr, iva, 60, s)[1] for s in range(2)], 0)
        ep = int(np.argmin(hist)) + 1
        res = [predecir(entrenar(tr, None, ep, s)[0], te) for s in range(3)]
        final[te] = np.mean([r[0] for r in res], 0)
        W[te] = np.mean([r[1] for r in res], 0)
        print(f"fold {k}: {ep} épocas", flush=True)

if "--caja" in banderas:
    # Compuerta final: se entrena con todo el desarrollo (predicciones fuera de fold) y se aplica a las
    # predicciones de la caja (modelos entrenados con todo el desarrollo). Épocas: mediana de los folds.
    Pc = np.column_stack([np.load(AQUI / "preds" / f"{t}_caja.npy") for t in tags]) / 100
    eps = []
    for k in range(5):
        j = (k + 1) % 5
        itr, iva = np.where(dev & (fold != k) & (fold != j))[0], np.where(dev & (fold == j))[0]
        eps.append(int(np.argmin(np.mean([entrenar(itr, iva, 60, s)[1] for s in range(2)], 0))) + 1)
    ep = int(np.median(eps))
    caja = np.where(~dev)[0]
    # El contexto de la caja usa sus propias predicciones (dispersión y desvíos de los expertos).
    ctx_c = ctx.copy()
    ctx_c["dispersion"] = Pc.std(1)
    for i, t in enumerate(tags):
        ctx_c[f"desvio_{t}"] = Pc[:, i] - Pc.mean(1)
    for t in ctx_tags:
        ctx_c[f"ctx_{t}"] = np.load(AQUI / "preds" / f"{t}_caja.npy") / 100 - Pc.mean(1)
    Craw = ctx.to_numpy(np.float32)
    Cc = (ctx_c.to_numpy(np.float32) - Craw[dev].mean(0)) / (Craw[dev].std(0) + 1e-6)
    todo = np.where(dev)[0]
    res = []
    for s_ in range(3):
        m = entrenar(todo, None, ep, s_)[0]
        with torch.no_grad():
            yh, _ = m(torch.from_numpy(Cc[caja]), torch.from_numpy(Pc[caja].astype(np.float32)))
        res.append(yh.numpy())
    fc = np.full(len(e), np.nan); fc[caja] = np.mean(res, 0) * 100
    et = (nombre or "compuerta") + "_caja"
    np.save(AQUI / "preds" / f"{et}.npy", fc)
    print(f"{et}: caja lista ({ep} épocas)")   # las métricas de la caja las calcula evaluar_caja.py, una vez
    sys.exit()

etiqueta = nombre or ("compuerta" + ("_gb" if "--gb" in banderas else "") + ("_corr" if "--correccion" in banderas else ""))
np.save(AQUI / "preds" / f"{etiqueta}.npy", final * 100)
registrar(etiqueta + "(" + "+".join(tags) + ")", evaluar(e, pd.Series(final * 100, index=e.index), e.index[dev]),
          "contexto: obs, rango, días, piso, decay, kal_sd, dispersión")
if "--gb" not in banderas:
    # Cuánto le cree a cada experto según las partidas observadas.
    d = pd.DataFrame(W[dev], columns=tags)
    d["obs"] = pd.cut(largos[dev], [0, 5, 20, 60, 150, 10_000], labels=["1-5", "6-20", "21-60", "61-150", "150+"])
    d["tier"] = e["tier2"].to_numpy()[dev]
    print(d.groupby("obs", observed=True)[tags].mean().round(2).to_string())
    print(d.groupby("tier", observed=True)[tags].mean().round(2).to_string())
