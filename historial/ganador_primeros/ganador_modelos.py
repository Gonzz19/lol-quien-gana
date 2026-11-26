"""Ablación de variables (logística) y comparación de modelos, con y sin espejado.
Lee data/analysis/ganador_feats.pkl de ganador_features.py (§9.S.23). Ajuste F (04/09-16/09), validación V (16/09-27/09).
Espejado: cada partida del ajuste se agrega con los lados invertidos (variables negadas,
resultado al revés) y una columna `lado` (+1 / -1). En validación se predice tal cual."""
import argparse, pickle, sys, time, warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
warnings.filterwarnings("ignore")

from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--feats", default=str(REPO / "data" / "analysis" / "ganador_feats.pkl"), help="salida de ganador_features.py")
d = pickle.load(open(ap.parse_args().feats, "rb"))
feat, y, fit, val, t = d["feat"], d["y"], d["fit"], d["val"], d["t"]
feat["onehot"] = feat["onehot"].toarray()
SIM = {"lp": [1]}          # columnas que no cambian de signo al espejar (indicador "falta LP")

def matriz(keys):
    X = np.hstack([feat[k] for k in keys])
    sim = np.zeros(X.shape[1], bool)
    off = 0
    for k in keys:
        for j in SIM.get(k, []):
            sim[off + j] = True
        off += feat[k].shape[1]
    return X, sim

def espejar(X, y, sim):
    Xm = np.where(sim, X, -X)
    return np.r_[np.c_[X, np.ones(len(X))], np.c_[Xm, -np.ones(len(X))]], np.r_[y, 1 - y]

def preparar(keys, espejo):
    X, sim = matriz(keys)
    Xf, yf = X[fit], y[fit]
    if espejo:
        Xf, yf = espejar(Xf, yf, sim)
        Xv = np.c_[X[val], np.ones(val.sum())]
    else:
        Xv = X[val]
    return Xf, yf, Xv

rng = np.random.default_rng(0)
B = rng.integers(0, val.sum(), (500, val.sum()))
yv = y[val]
res, preds = [], {}

def informe(nombre, p, seg):
    preds[nombre] = p
    lo, hi = np.quantile([roc_auc_score(yv[b], p[b]) for b in B[:200]], [0.025, 0.975])
    res.append(nombre)
    print(f"{nombre:46s} acc {accuracy_score(yv, p > .5):.3f}  AUC {roc_auc_score(yv, p):.3f} "
          f"[{lo:.3f}-{hi:.3f}]  logloss {log_loss(yv, p):.4f}  ({seg:.0f}s)", flush=True)

def logit(keys, espejo=False, C=0.05):
    t0 = time.time()
    Xf, yf, Xv = preparar(keys, espejo)
    sc = StandardScaler(with_mean=False).fit(Xf)   # sin centrar: mantiene la simetría del espejo
    m = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(Xf), yf)
    return m.predict_proba(sc.transform(Xv))[:, 1], time.time() - t0

BASE = ["onehot", "lp", "huella", "experiencia"]
print("=== Ablación con logística (sin espejar) ===")
for nombre, keys in [("base (el 0,610 de antes)", BASE),
                     ("base + cobertura", BASE + ["cobertura"]),
                     ("base + dificultad", BASE + ["dificultad"]),
                     ("base + matchups", BASE + ["matchups"]),
                     ("base + sinergias", BASE + ["sinergias"]),
                     ("base + matchups + sinergias", BASE + ["matchups", "sinergias"])]:
    informe(nombre, *logit(keys))
TODO = BASE + ["cobertura", "dificultad", "matchups", "sinergias"]
informe("todo", *logit(TODO))
informe("todo, espejado", *logit(TODO, espejo=True))
print("\nmatchups y sinergias solos:")
informe("  matchups solos", *logit(["matchups"]))
informe("  sinergias solas", *logit(["sinergias"]))

# ---------- modelos de árboles sobre 'todo'; parada temprana con el 15 % más nuevo del ajuste
print("\n=== Modelos sobre 'todo' ===")

def arbol(nombre, construir, espejo=True):
    t0 = time.time()
    X, sim = matriz(TODO)
    idx = np.where(fit)[0]
    corte = np.quantile(t[idx], 0.85)
    tr, es = idx[t[idx] < corte], idx[t[idx] >= corte]
    Xtr, ytr = (espejar(X[tr], y[tr], sim) if espejo else (X[tr], y[tr]))
    add = (lambda Z: np.c_[Z, np.ones(len(Z))]) if espejo else (lambda Z: Z)
    p = construir(Xtr, ytr, add(X[es]), y[es], add(X[val]))
    informe(nombre, p, time.time() - t0)

def hgb(Xtr, ytr, Xes, yes, Xv):
    m = HistGradientBoostingClassifier(max_iter=2000, learning_rate=0.03, max_leaf_nodes=15,
                                       min_samples_leaf=100, l2_regularization=1.0, early_stopping=True,
                                       validation_fraction=None, n_iter_no_change=100, random_state=0)
    # parada temprana con el bloque temporal: se pasa como validación vía warm start manual
    best, best_n = 1e9, 0
    m.set_params(early_stopping=False, max_iter=50, warm_start=True)
    for n in range(50, 2001, 50):
        m.set_params(max_iter=n).fit(Xtr, ytr)
        ll = log_loss(yes, m.predict_proba(Xes)[:, 1])
        if ll < best:
            best, best_n = ll, n
        elif n - best_n >= 300:
            break
    m.set_params(max_iter=best_n, warm_start=False).fit(Xtr, ytr)
    return m.predict_proba(Xv)[:, 1]

def lgbm(Xtr, ytr, Xes, yes, Xv):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.02, num_leaves=15, min_child_samples=100,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, verbose=-1)
    m.fit(Xtr, ytr, eval_set=[(Xes, yes)], callbacks=[lgb.early_stopping(200, verbose=False)])
    return m.predict_proba(Xv)[:, 1]

def xgbm(Xtr, ytr, Xes, yes, Xv):
    import xgboost as xgb
    m = xgb.XGBClassifier(n_estimators=5000, learning_rate=0.02, max_depth=4, min_child_weight=20,
                          subsample=0.8, colsample_bytree=0.5, reg_lambda=1.0, tree_method="hist",
                          early_stopping_rounds=200, eval_metric="logloss")
    m.fit(Xtr, ytr, eval_set=[(Xes, yes)], verbose=False)
    return m.predict_proba(Xv)[:, 1]

def cat(Xtr, ytr, Xes, yes, Xv):
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(iterations=5000, learning_rate=0.03, depth=6, l2_leaf_reg=5,
                           od_type="Iter", od_wait=200, verbose=False, random_seed=0, thread_count=-1)
    m.fit(Xtr, ytr, eval_set=(Xes, yes), use_best_model=True)
    return m.predict_proba(Xv)[:, 1]

def ngb(Xtr, ytr, Xes, yes, Xv):
    from ngboost import NGBClassifier
    from ngboost.distns import Bernoulli
    from sklearn.tree import DecisionTreeRegressor
    m = NGBClassifier(Dist=Bernoulli, n_estimators=1500, learning_rate=0.03, verbose=False,
                      Base=DecisionTreeRegressor(max_depth=3, min_samples_leaf=100), random_state=0)
    m.fit(Xtr, ytr, X_val=Xes, Y_val=yes, early_stopping_rounds=100)
    return m.predict_proba(Xv)[:, 1]

def mlp(Xtr, ytr, Xes, yes, Xv):
    import torch
    torch.manual_seed(0)
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

arbol("HistGradientBoosting, sin espejar", hgb, espejo=False)
arbol("HistGradientBoosting, espejado", hgb)
arbol("LightGBM, espejado", lgbm)
arbol("XGBoost, espejado", xgbm)
arbol("CatBoost, espejado", cat)
arbol("NGBoost, espejado", ngb)
arbol("Red chica (MLP 64), espejado", mlp)

print("\n=== Promedio de modelos ===")
for combo in [["todo, espejado", "CatBoost, espejado"],
              ["todo, espejado", "CatBoost, espejado", "LightGBM, espejado", "Red chica (MLP 64), espejado"]]:
    informe(" + ".join(c.split(",")[0] for c in combo), np.mean([preds[c] for c in combo], axis=0), 0)

print("\n=== Diferencia de AUC contra 'base' (bootstrap pareado, 500) ===")
for n in res[1:]:
    dd = np.array([roc_auc_score(yv[b], preds[n][b]) - roc_auc_score(yv[b], preds[res[0]][b]) for b in B])
    print(f"{n:46s} {dd.mean():+.4f}  IC95 [{np.quantile(dd, .025):+.4f}, {np.quantile(dd, .975):+.4f}]")
