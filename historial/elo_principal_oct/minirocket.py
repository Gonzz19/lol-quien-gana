"""MiniRocket multivariado de largo variable (en torch, GPU) + ridge con LOO, sobre el residuo de la regla.

Desarrollo: 5 folds sobre el 85 %; la caja cerrada no se toca. Los sesgos de los núcleos salen de las
secuencias del desarrollo sin mirar el objetivo (no supervisado). Guarda:
  preds/<tag>.npy      ridge solo (regla + ridge), fuera de fold
  preds/<tag>_gb.npy   GB final + la predicción del ridge como variable (anidado: el ridge que ve el GB
                       de cada fold nunca vio ese fold)
"""
import argparse
import itertools
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import AQUI, cargar, evaluar, registrar  # noqa

P0 = "resultado,aliados,rivales,lobby,conocidos,dias,hueco,azul"
E2 = ("z2,pred_regla,partidas2,wr_temporada,obs,obs_w,kal_media,kal_sd,decay,piso,dias_fin,dias_ini,hueco_max,"
      "g_ch,g_gm,g_ma,lobby_viejo,lobby_reciente,exc_fin,exc_ini,exc_medio,n_huecos14,antes_fin,piso375,piso275")
ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--paso", default=P0)
ap.add_argument("--est", default=E2)
ap.add_argument("--dil", default="1,2,4,8,16,32")
ap.add_argument("--nb", type=int, default=12, help="sesgos por núcleo y dilatación")
ap.add_argument("--pool", default="ppv", choices=["ppv", "ppv_mpv"], help="ppv_mpv suma la media de los positivos")
ap.add_argument("--sin-est", action="store_true", help="el ridge ve solo las secuencias")
ap.add_argument("--nota", default="")
args = ap.parse_args()
t0 = time.time()

c = cargar()
e = c["e"]
with open(AQUI / "seq.pkl", "rb") as fh:
    S = pickle.load(fh)
paso = args.paso.split(",")
ip = [S["PASO"].index(x) for x in paso]
seqs = [s[:, ip].astype(np.float32) for s in S["seqs"]]
largos = np.array([len(s) for s in seqs])
dev, fold = np.asarray(~c["caja"]), np.asarray(c["fold"])
regla = c["regla"].to_numpy()
y = (e["z1"].to_numpy() - regla) / 100
gpu = torch.device("cuda")
rng = np.random.default_rng(0)

# Canales estandarizados con las posiciones válidas del desarrollo.
todo = np.concatenate([seqs[i] for i in np.where(dev)[0]])
mu, sd = todo.mean(0), todo.std(0) + 1e-6
seqs = [(s - mu) / sd for s in seqs]

# 84 núcleos de largo 9: tres pesos 2 y seis −1 (suman 0). Cada combinación núcleo × dilatación
# mira un subconjunto aleatorio de canales, como en MiniRocket multivariado.
C = len(paso)
comb = list(itertools.combinations(range(9), 3))
base_w = -np.ones((84, 9), np.float32)
for k, cc in enumerate(comb):
    base_w[k, list(cc)] = 2
dils = [int(d) for d in args.dil.split(",")]
W = []
for d in dils:
    w = np.zeros((84, C, 9), np.float32)
    for k in range(84):
        n = int(2 ** rng.uniform(0, np.log2(min(C, 9) + 1)))
        for ch in rng.choice(C, size=min(n, C), replace=False):
            w[k, ch] = base_w[k]
    W.append(torch.from_numpy(w).to(gpu))
q = ((np.arange(1, args.nb + 1) * (np.sqrt(5) + 1) / 2) % 1).astype(np.float32)   # cuantiles de baja discrepancia


def lote(idx):
    lg = largos[idx]
    x = np.zeros((len(idx), lg.max(), C), np.float32)
    for j, k in enumerate(idx):
        x[j, :lg[j]] = seqs[k]
    x = torch.from_numpy(x).to(gpu).transpose(1, 2)
    m = torch.arange(x.shape[2], device=gpu)[None] < torch.from_numpy(lg).to(gpu)[:, None]
    return x, m


def convs(x):
    return [torch.nn.functional.conv1d(x, W[i], padding=4 * d, dilation=d) for i, d in enumerate(dils)]


# Sesgos: cuantiles de la salida de cada núcleo sobre 3000 secuencias del desarrollo (posiciones válidas).
muestra = rng.choice(np.where(dev & (largos >= 5))[0], 3000, replace=False)
acum = [[] for _ in dils]
for i in range(0, len(muestra), 256):
    x, m = lote(muestra[i:i + 256])
    for j, o in enumerate(convs(x)):
        acum[j].append(o.transpose(1, 2)[m])        # (posiciones, 84)
sesgos = [torch.quantile(torch.cat(a)[torch.randperm(sum(len(t) for t in a), device=gpu)[:120000]],
                         torch.from_numpy(q).to(gpu), dim=0).T.contiguous() for a in acum]   # (84, nb)

# Transformación de todos los jugadores (por lotes ordenados por largo).
orden = np.argsort(largos, kind="stable")
F = np.zeros((len(e), len(dils) * 84 * args.nb * (2 if args.pool == "ppv_mpv" else 1)), np.float32)
with torch.no_grad():
    for i in range(0, len(orden), 128):
        idx = orden[i:i + 128]
        x, m = lote(idx)
        L = m.sum(1, keepdim=True).float()
        partes = []
        for j, o in enumerate(convs(x)):
            pos = (o[..., None] > sesgos[j][None, :, None, :]) & m[:, None, :, None]    # (B, 84, L, nb)
            partes.append((pos.sum(2) / L[:, :, None]).flatten(1))
            if args.pool == "ppv_mpv":
                v = (o[..., None] - sesgos[j][None, :, None, :]).clamp(min=0) * pos
                partes.append((v.sum(2) / pos.sum(2).clamp(min=1)).flatten(1))
        F[idx] = torch.cat(partes, 1).cpu().numpy()
if not args.sin_est:
    F = np.hstack([F, S["est"][args.est.split(",")].to_numpy(np.float32)])
print(f"variables {F.shape[1]}  ({time.time() - t0:.0f} s)", flush=True)

alfas = torch.logspace(-1, 5, 25, dtype=torch.float64, device=gpu)


def ridge(tr, te):
    """Ridge estandarizado; alfa elegido por error dejando un jugador afuera (fórmula cerrada)."""
    X = torch.from_numpy(F[tr]).to(gpu, torch.float64)
    m_, s_ = X.mean(0), X.std(0) + 1e-6
    X = (X - m_) / s_
    yy = torch.from_numpy(y[tr]).to(gpu, torch.float64)
    ym = yy.mean()
    lam, V = torch.linalg.eigh(X.T @ X)
    XV = X @ V
    b = XV.T @ (yy - ym)
    XV2 = XV ** 2
    mejor = (np.inf, None)
    for a in alfas:
        inv = 1 / (lam + a)
        h = XV2 @ inv + 1 / len(tr)
        loo = ((yy - ym) - XV @ (b * inv)) / (1 - h)
        mejor = min(mejor, ((loo ** 2).mean().item(), a.item()), key=lambda t: t[0])
    coef = V @ (b / (lam + mejor[1]))
    Xte = (torch.from_numpy(F[te]).to(gpu, torch.float64) - m_) / s_
    return (Xte @ coef + ym).cpu().numpy(), mejor[1]


# 1) Ridge solo, fuera de fold.
oof = np.full(len(e), np.nan)
for k in range(5):
    tr, te = np.where(dev & (fold != k))[0], np.where(dev & (fold == k))[0]
    oof[te], a = ridge(tr, te)
    print(f"fold {k}: alfa {a:.0f}", flush=True)
final = pd.Series(regla + 100 * oof, index=e.index)
np.save(AQUI / "preds" / f"{args.tag}.npy", final.to_numpy())
registrar(args.tag, evaluar(e, final, e.index[dev]), f"MiniRocket {F.shape[1]} var + ridge. {args.nota} ({time.time() - t0:.0f} s)")

# 2) GB final + ridge como variable. Para el fold externo k, el ridge de las filas de entrenamiento sale
#    de ajustes sin {j, k}; el de las filas de prueba, del ridge sin k (oof).
interno = {k: np.full(len(e), np.nan) for k in range(5)}
for j, k in itertools.combinations(range(5), 2):
    tr = np.where(dev & (fold != j) & (fold != k))[0]
    fj, fk = np.where(dev & (fold == j))[0], np.where(dev & (fold == k))[0]
    p, _ = ridge(tr, np.concatenate([fj, fk]))
    interno[k][fj], interno[j][fk] = p[:len(fj)], p[len(fj):]
est = S["est"]
D = pd.DataFrame({"exc_fin": est["exc_fin"] * 30, "exc_ini": est["exc_ini"] * 30, "exc_medio": est["exc_medio"] * 30,
                  "n_huecos14": est["n_huecos14"] * 3, "partidas_antes_fin": est["antes_fin"] * 20,
                  "piso375": est["piso375"], "piso275": est["piso275"]})
X = c["X_comp"].join(est[["piso", "dias_fin", "dias_ini", "hueco_max"]]).join(D)
yr = e["z1"] - c["regla"]
params = dict(loss="absolute_error", max_iter=800, learning_rate=0.03, max_leaf_nodes=31, min_samples_leaf=50,
              random_state=0, early_stopping=False)
pred = pd.Series(np.nan, index=e.index)
for k in range(5):
    tr, te = dev & (fold != k), dev & (fold == k)
    Xk = X.assign(mr=np.where(te, oof, interno[k]) * 100)
    pred[te] = HistGradientBoostingRegressor(**params).fit(Xk[tr], yr[tr]).predict(Xk[te])
final = c["regla"] + pred
np.save(AQUI / "preds" / f"{args.tag}_gb.npy", final.to_numpy())
registrar(args.tag + "_gb", evaluar(e, final, e.index[dev]), f"GB final + ridge MiniRocket como variable ({time.time() - t0:.0f} s)")
