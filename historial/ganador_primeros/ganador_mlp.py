"""¿Con la huella progresiva y 160 mil partidas, conviene una MLP más grande que la logística?

Mismas variables que la mejor logística (comunes + huella progresiva + n), misma validación. Las redes se
entrenan espejadas, con parada temprana en el 15 % más nuevo del ajuste, y se promedian 3 semillas.
Curva de aprendizaje: las N partidas más nuevas del ajuste, para no cambiar lo reciente que es la huella.

    python historial/ganador_primeros/ganador_features_grande.py
    python historial/ganador_primeros/ganador_huella_progresiva.py
    python historial/ganador_primeros/ganador_mlp.py            # GPU, ~10-20 min
"""
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
A = REPO / "data" / "analysis"
d = pickle.load(open(A / "ganador_feats_grande.pkl", "rb"))
h = pickle.load(open(A / "ganador_huella_progresiva.pkl", "rb"))
feat, y, t, val, masks = d["feat"], d["y"], d["t"], d["val"], d["masks"]
COMUN = ["onehot", "experiencia", "cobertura", "dificultad", "matchups", "sinergias"]
X = np.hstack([feat[k] for k in COMUN] + [h["huella_prog"], h["huella_n"]]).astype(np.float32)
fit_idx = np.where(masks["C"])[0]
fit_idx = fit_idx[np.argsort(t[fit_idx])]
yv = y[val]
B = np.random.default_rng(0).integers(0, val.sum(), (500, val.sum()))
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{X.shape[1]} variables, dispositivo {dev}", flush=True)


def logistica(idx):
    sc = StandardScaler(with_mean=False).fit(X[idx])
    m = LogisticRegression(C=0.05, max_iter=5000).fit(sc.transform(X[idx]), y[idx])
    return m.predict_proba(sc.transform(X[val]))[:, 1]


def red(idx, capas, drop, wd, seed):
    torch.manual_seed(seed)
    corte = int(len(idx) * 0.85)
    tr, es = idx[:corte], idx[corte:]
    sc = StandardScaler(with_mean=False).fit(X[tr])
    Z = lambda i, sg=1.0: np.c_[sc.transform(X[i]) * sg, np.full(len(i), sg)]
    Xtr = torch.tensor(np.r_[Z(tr), Z(tr, -1.0)], dtype=torch.float32, device=dev)
    ytr = torch.tensor(np.r_[y[tr], 1 - y[tr]], dtype=torch.float32, device=dev)
    Xes = torch.tensor(Z(es), dtype=torch.float32, device=dev)
    Xv = torch.tensor(Z(np.where(val)[0]), dtype=torch.float32, device=dev)
    capas_ = []
    ent = Xtr.shape[1]
    for u in capas:
        capas_ += [torch.nn.Linear(ent, u), torch.nn.ReLU(), torch.nn.Dropout(drop)]
        ent = u
    net = torch.nn.Sequential(*capas_, torch.nn.Linear(ent, 1)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd)
    best, estado, pac = 1e9, None, 0
    for ep in range(200):
        net.train()
        for b in torch.randperm(len(Xtr), device=dev).split(1024):
            opt.zero_grad()
            torch.nn.functional.binary_cross_entropy_with_logits(net(Xtr[b]).squeeze(1), ytr[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            ll = log_loss(y[es], torch.sigmoid(net(Xes)).squeeze(1).cpu().numpy())
        if ll < best - 1e-5:
            best, estado, pac = ll, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            pac += 1
            if pac >= 10:
                break
    net.load_state_dict(estado)
    with torch.no_grad():
        return torch.sigmoid(net(Xv)).squeeze(1).cpu().numpy()


res = {}


def informe(nombre, p, n, seg, ref=None):
    res[nombre] = p
    txt = f"{nombre:44s} ajuste {n:>7,d}  acc {accuracy_score(yv, p > .5):.3f}  AUC {roc_auc_score(yv, p):.4f}  logloss {log_loss(yv, p):.4f}"
    if ref is not None:
        dif = [roc_auc_score(yv[b], p[b]) - roc_auc_score(yv[b], ref[b]) for b in B]
        txt += f"  vs logística {np.mean(dif):+.4f} [{np.quantile(dif, .025):+.4f}, {np.quantile(dif, .975):+.4f}]"
    print(txt + f"  ({seg:.0f} s)", flush=True)


print("\n=== Tamaños de red, todo el ajuste ===")
t0 = time.time()
ref = logistica(fit_idx)
informe("Logística C=0,05", ref, len(fit_idx), time.time() - t0)
ARQ = {"MLP 64 (la de hoy)": ((64,), .3, 1e-2), "MLP 256-128": ((256, 128), .3, 1e-2),
       "MLP 512-256-128": ((512, 256, 128), .4, 1e-2), "MLP 1024-512-256": ((1024, 512, 256), .5, 3e-2)}
for nombre, (capas, drop, wd) in ARQ.items():
    t0 = time.time()
    p = np.mean([red(fit_idx, capas, drop, wd, s) for s in (0, 1, 2)], axis=0)
    informe(nombre + ", 3 semillas", p, len(fit_idx), time.time() - t0, ref)
p = (ref + res["MLP 256-128, 3 semillas"]) / 2
informe("Promedio logística + MLP 256-128", p, len(fit_idx), 0, ref)

print("\n=== Curva de aprendizaje (las N partidas más nuevas del ajuste) ===")
for n in (27_000, 55_000, 110_000, len(fit_idx)):
    idx = fit_idx[-n:]
    t0 = time.time()
    pl = logistica(idx)
    informe(f"Logística, N={n:,}", pl, n, time.time() - t0)
    t0 = time.time()
    pr = np.mean([red(idx, (256, 128), .3, 1e-2, s) for s in (0, 1, 2)], axis=0)
    informe(f"MLP 256-128, N={n:,}", pr, n, time.time() - t0, pl)
pickle.dump(res, open(A / "ganador_mlp.pkl", "wb"))
