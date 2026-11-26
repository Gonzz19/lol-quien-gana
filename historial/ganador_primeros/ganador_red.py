"""Red chica espejada sin LP, con más partidas de ajuste. Logística como referencia.
A: ajuste 04/09-16/09 (como antes), huella con lo anterior al 04/09
B: ajuste 09/08-16/09, huella con el 40 % más viejo (antes del 09/08)
C: ajuste con todo el train antes del 16/09, sin huella
Validación: la misma de siempre (16/09-27/09)."""
import argparse, pickle, sys, time, warnings
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
warnings.filterwarnings("ignore")

from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--feats", default=str(REPO / "data" / "analysis" / "ganador_feats_grande.pkl"), help="salida de ganador_features_grande.py")
d = pickle.load(open(ap.parse_args().feats, "rb"))
feat, y, t, val, masks = d["feat"], d["y"], d["t"], d["val"], d["masks"]
COMUN = ["onehot", "experiencia", "cobertura", "dificultad", "matchups", "sinergias"]
CONFIG = {"A": COMUN + ["huella_foto"], "B": COMUN + ["huella_40"], "C": COMUN}
yv = y[val]
B = np.random.default_rng(0).integers(0, val.sum(), (200, val.sum()))
preds = {}

def informe(nombre, p, n_fit, seg):
    preds[nombre] = p
    lo, hi = np.quantile([roc_auc_score(yv[b], p[b]) for b in B], [0.025, 0.975])
    print(f"{nombre:34s} ajuste {n_fit:>7d}  acc {accuracy_score(yv, p > .5):.3f}  AUC {roc_auc_score(yv, p):.3f} "
          f"[{lo:.3f}-{hi:.3f}]  logloss {log_loss(yv, p):.4f}  ({seg:.0f}s)", flush=True)

def espejar(X, yy):
    return np.r_[np.c_[X, np.ones(len(X))], np.c_[-X, -np.ones(len(X))]], np.r_[yy, 1 - yy]

def mlp(Xtr, ytr, Xes, yes, Xv, seed):
    torch.manual_seed(seed)
    sc = StandardScaler(with_mean=False).fit(Xtr)
    T = lambda Z: torch.tensor(sc.transform(Z), dtype=torch.float32)
    A, Ae, Av = T(Xtr), T(Xes), T(Xv)
    yt = torch.tensor(ytr, dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(A.shape[1], 64), torch.nn.ReLU(), torch.nn.Dropout(0.3),
                              torch.nn.Linear(64, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    best, best_state, paciencia = 1e9, None, 0
    for ep in range(100):
        net.train()
        for b in torch.randperm(len(A)).split(512):
            opt.zero_grad()
            torch.nn.functional.binary_cross_entropy_with_logits(net(A[b]).squeeze(1), yt[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            ll = log_loss(yes, torch.sigmoid(net(Ae)).squeeze(1).numpy())
        if ll < best:
            best, best_state, paciencia = ll, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            paciencia += 1
            if paciencia >= 8:
                break
    net.load_state_dict(best_state)
    with torch.no_grad():
        return torch.sigmoid(net(Av)).squeeze(1).numpy()

for cfg, keys in CONFIG.items():
    X = np.hstack([feat[k] for k in keys])
    fit = masks[cfg]
    print(f"\n=== Config {cfg}: {', '.join(keys)} ===")
    t0 = time.time()
    sc = StandardScaler(with_mean=False).fit(X[fit])
    p = LogisticRegression(C=0.05, max_iter=5000).fit(sc.transform(X[fit]), y[fit]).predict_proba(sc.transform(X[val]))[:, 1]
    informe(f"{cfg}. logística", p, fit.sum(), time.time() - t0)
    idx = np.where(fit)[0]
    corte = np.quantile(t[idx], 0.85)                  # parada temprana: el 15 % más nuevo del ajuste
    tr, es = idx[t[idx] < corte], idx[t[idx] >= corte]
    Xtr, ytr = espejar(X[tr], y[tr])
    one = lambda Z: np.c_[Z, np.ones(len(Z))]
    ps = []
    for seed in (0, 1, 2):
        t0 = time.time()
        ps.append(mlp(Xtr, ytr, one(X[es]), y[es], one(X[val]), seed))
        informe(f"{cfg}. red espejada, semilla {seed}", ps[-1], fit.sum(), time.time() - t0)
    informe(f"{cfg}. red espejada, promedio 3 semillas", np.mean(ps, axis=0), fit.sum(), 0)

print("\nreferencia anterior (con LP, ajuste 26.934): logística 0,612 / 58,1 %; red espejada 0,607 / 57,5 % (media 3 semillas)")
