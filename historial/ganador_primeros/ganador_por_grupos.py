"""¿Hay lugar para un MoE? Chequeo antes de construirlo.

Si el orden entre modelos cambia según el tipo de partida, una compuerta que pese a cada
experto según el caso puede ganar; si la logística gana igual en todos los grupos, no.
1. Ajusta logística, LightGBM y CatBoost (espejados) con la config pedida y lee las redes
   por jugador de ganador_red_jugador.py --guardar.
2. AUC de cada modelo por grupo de partida (descriptores simétricos entre lados: no cambian
   al invertir azul y rojo) y diferencia pareada contra la logística (bootstrap dentro del grupo).
3. Vista previa del stacking: V partida por fecha en dos mitades; se ajusta en una y se
   evalúa en la otra (las dos direcciones), sin compuerta y con compuerta por grupo.
Solo data/splits/train (V es la validación de siempre, 16/09-27/09)."""
import argparse, pickle, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss
warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--feats", default=str(REPO / "data" / "analysis" / "ganador_feats_grande.pkl"))
ap.add_argument("--redes", default=str(REPO / "data" / "analysis" / "ganador_red_jugador_pred.npz"))
ap.add_argument("--config", default="A")
ap.add_argument("--margen", help="salida de ganador_margen_duos.py: suma margen previo y duos (§9.S.24)")
ap.add_argument("--sin-huella", action="store_true")
args = ap.parse_args()
cfg = args.config

d = pickle.load(open(args.feats, "rb"))
feat, jug, mids, y, t, val, masks = d["feat"], d["jug"], d["mids"], d["y"], d["t"], d["val"], d["masks"]
HU = None if args.sin_huella else {"A": "huella_foto", "B": "huella_40", "C": None}[cfg]

MARGEN = None
if args.margen:
    MARGEN = pickle.load(open(args.margen, "rb"))
    assert (MARGEN["mids"] == d["mids"]).all(), "el pickle de margen no corresponde a estas partidas"
    AGREGADO = np.hstack([MARGEN["nuevo"][k] for k in ("winrate previo", "margen acum", "margen v20", "duos")])
fit, yv, vi = masks[cfg], y[val], np.where(val)[0]
X = np.hstack([feat[k] for k in ["onehot", "experiencia", "cobertura", "dificultad", "matchups", "sinergias"]
               + ([HU] if HU else [])] + ([AGREGADO] if MARGEN else [])).astype(np.float32)

# ---------- expertos
P = {}
t0 = time.time()
sc = StandardScaler(with_mean=False).fit(X[fit])
P["logística"] = LogisticRegression(C=0.05, max_iter=5000).fit(sc.transform(X[fit]), y[fit]).predict_proba(sc.transform(X[val]))[:, 1]
print(f"logística {time.time() - t0:.0f}s", flush=True)

idx = np.where(fit)[0]
corte = np.quantile(t[idx], 0.85)
tr, es = idx[t[idx] < corte], idx[t[idx] >= corte]
esp = lambda Z, yy: (np.r_[np.c_[Z, np.ones(len(Z))], np.c_[-Z, -np.ones(len(Z))]], np.r_[yy, 1 - yy])
one = lambda Z: np.c_[Z, np.ones(len(Z))]
Xtr, ytr = esp(X[tr], y[tr])
t0 = time.time()
m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.02, num_leaves=15, min_child_samples=200,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5, verbose=-1)
m.fit(Xtr, ytr, eval_set=[(one(X[es]), y[es])], callbacks=[lgb.early_stopping(150, verbose=False)])
P["LightGBM"] = m.predict_proba(one(X[val]))[:, 1]
print(f"LightGBM {time.time() - t0:.0f}s ({m.best_iteration_} árboles)", flush=True)
t0 = time.time()
m = CatBoostClassifier(iterations=3000, learning_rate=0.05, depth=6, l2_leaf_reg=10, verbose=0,
                       early_stopping_rounds=200, random_seed=0, train_dir=str(REPO / "catboost_info"))
m.fit(Xtr, ytr, eval_set=(one(X[es]), y[es]))
P["CatBoost"] = m.predict_proba(one(X[val]))[:, 1]
print(f"CatBoost {time.time() - t0:.0f}s ({m.get_best_iteration()} árboles)", flush=True)
del X, Xtr
if Path(args.redes).exists():
    for k, v in np.load(args.redes).items():
        if k.startswith(f"{cfg}_red_"):
            P[k[2:].replace("_", " ")] = v
else:
    print(f"(sin {args.redes}: correr ganador_red_jugador.py --guardar)")
P["promedio de todos"] = np.mean(list(P.values()), axis=0)

# ---------- descriptores simétricos de la partida (en V)
num = jug["num"][val]                                   # columnas: CAMP(6), logn_tot, DIF(3)
cob_min = np.expm1(num[:, :, 6].min(1))                 # partidas previas del jugador menos visto
estrenos = (num[:, :, 1] == 1).sum(1)
sin_hu = (jug["huella_foto"][val][:, :, -1] == 0).sum(1)
tier = pd.read_csv(REPO / "data" / "splits" / "match_split.csv", usecols=["match_id", "tier"]) \
         .set_index("match_id").tier.reindex(mids[val]).to_numpy()
q = np.quantile(cob_min, [1 / 3, 2 / 3])
GRUPOS = {
    "lobby": np.where(tier == "DIAMOND", "Diamond", "Master+"),
    "cobertura mínima": np.select([cob_min < q[0], cob_min < q[1]],
                                  [f"< {q[0]:.0f}", f"{q[0]:.0f}-{q[1]:.0f}"], f">= {q[1]:.0f}"),
    "estrenos en el lobby": np.select([estrenos == 0, estrenos == 1], ["0", "1"], "2+"),
    "sin huella en el lobby": np.select([sin_hu == 0, sin_hu <= 2], ["0", "1-2"], "3+"),
}

rng = np.random.default_rng(0)
nombres = list(P)
print(f"\n=== Config {cfg}: AUC por grupo (entre corchetes, diferencia contra la logística, IC95) ===")
print(f"{'':28s}{'n':>7s}  " + "  ".join(f"{n[:16]:>24s}" for n in nombres))
filas = [("todo V", np.ones(len(yv), bool))] + [(f"{g}: {v}", lab == v)
                                               for g, lab in GRUPOS.items() for v in sorted(set(lab))]
for nombre, sel in filas:
    ii = np.where(sel)[0]
    bs = [rng.choice(ii, len(ii)) for _ in range(200)]
    base = np.array([roc_auc_score(yv[b], P["logística"][b]) for b in bs])
    celdas = []
    for n in nombres:
        a = roc_auc_score(yv[ii], P[n][ii])
        if n == "logística":
            celdas.append(f"{a:.3f}{'':>18s}")
        else:
            dlo, dhi = np.quantile(np.array([roc_auc_score(yv[b], P[n][b]) for b in bs]) - base, [.025, .975])
            celdas.append(f"{a:.3f} [{dlo:+.3f};{dhi:+.3f}]")
    print(f"{nombre:28s}{len(ii):>7d}  " + "  ".join(f"{c:>24s}" for c in celdas), flush=True)

# ---------- vista previa del stacking (ajuste en una mitad de V, evaluación en la otra)
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
Z = np.c_[[logit(P[n]) for n in nombres if n != "promedio de todos"]].T
D = np.c_[[GRUPOS[g] == v for g in GRUPOS for v in sorted(set(GRUPOS[g]))]].T.astype(float)
compuerta = np.hstack([D] + [Z * D[:, [j]] for j in range(D.shape[1])])   # pesos de cada experto por grupo
mitad = np.median(t[val]) <= t[val]
print("\n=== Stacking: ajuste en una mitad de V, evaluación en la otra (AUC / log-loss) ===")
for a_nom, ajuste in (("1.ª -> 2.ª", ~mitad), ("2.ª -> 1.ª", mitad)):
    ev = ~ajuste
    res = {"logística sola": P["logística"][ev], "promedio simple": P["promedio de todos"][ev]}
    for nombre, M, C in (("stacking sin compuerta", Z, 1.0), ("stacking con compuerta", np.c_[Z, compuerta], 0.1)):
        res[nombre] = LogisticRegression(C=C, max_iter=5000).fit(M[ajuste], yv[ajuste]).predict_proba(M[ev])[:, 1]
    print(f"  {a_nom}: " + " | ".join(f"{k} {roc_auc_score(yv[ev], p):.4f} / {log_loss(yv[ev], p):.4f}"
                                      for k, p in res.items()), flush=True)
