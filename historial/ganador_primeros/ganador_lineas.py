"""Modelo por línea: quién gana cada línea, y su aporte al modelo del equipo (§9.S.36).

Idea del usuario (2026-11-13): un modelo por línea ayuda al puntaje global del equipo.
Para cada línea l, con información previa a la partida, se predice la diferencia de oro a los
15 entre el jugador azul y el rojo de esa línea (y = oro15_azul − oro15_rojo, en miles).
Entrada por línea: las variables por jugador de los dos (experiencia, margen previo, márgenes
por campeón y línea, Poisson, nivel del lobby) y el matchup de campeones de esa línea.
  ridge   lineal sobre (azul − rojo) y el matchup, espejado (sin constante: antisimétrico)
  HGB     árboles sobre (azul, rojo, matchup), espejado: cada partida entra también con los
          lados invertidos y el signo cambiado, y la predicción es (f(a, r) − f(r, a)) / 2
Sin leakage hacia el modelo del equipo: en el ajuste, cada partida se predice con un modelo que
no la vio (5 tramos por fecha, cada uno con los otros 4); V con todo el ajuste.
Las 5 predicciones y su suma entran a la logística con Poisson y nivel del lobby (0,6493).
Solo data/splits/train."""
import argparse, pickle, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, r2_score
warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
AN = REPO / "data/analysis"
ap = argparse.ArgumentParser()
ap.add_argument("--modelos", default="ridge,hgb")
ap.add_argument("--tramos", type=int, default=5)
args = ap.parse_args()
LANES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
t0 = time.time()

d = pickle.load(open(AN / "ganador_feats_grande.pkl", "rb"))
feat, jug, mids, y, t, val, fit = d["feat"], d["jug"], d["mids"], d["y"], d["t"], d["val"], d["masks"]["C"]
del d
M = pickle.load(open(AN / "ganador_margen_duos.pkl", "rb"))
G = pickle.load(open(AN / "ganador_margen_grupos.pkl", "rb"))
P = pickle.load(open(AN / "ganador_poisson.pkl", "rb"))
L = pickle.load(open(AN / "ganador_mmr.pkl", "rb"))
for o in (M, G, P, L):
    assert (o["mids"] == mids).all()
# por jugador: (partida, puesto = 5 * rojo + línea, variables)
J = np.concatenate([jug["num"], M["jug"], G["jug"], np.concatenate(list(P["jug"].values()), axis=2), L["jug"][:, :, :2]],
                   axis=2).astype(np.float32)
print(f"variables por jugador: {J.shape[2]}", flush=True)
base_X = np.hstack([feat[k] for k in ["onehot", "experiencia", "cobertura", "dificultad", "matchups", "sinergias"]]
                   + [M["nuevo"][k] for k in ("winrate previo", "margen acum", "margen v20", "duos")]
                   + list(G["nuevo"].values()) + [np.hstack(list(P["poisson"].values())), L["nuevo"]["nivel (foto)"]])
MT = feat["matchups"]
del feat, M, G, P, L, jug

# ---------- objetivo: diferencia de oro a los 15 por línea (azul − rojo), en miles
df = pd.read_csv(REPO / "data/splits/train/full_dataset.csv", usecols=["match_id", "team_id", "lane", "gold_15"])
df = df[df.match_id.isin(set(mids)) & df.lane.isin(LANES)]
w = df.pivot_table(index="match_id", columns=["team_id", "lane"], values="gold_15", aggfunc="first").reindex(mids)
Y = np.c_[[(w[(100, l)] - w[(200, l)]).to_numpy() / 1000 for l in LANES]].T          # (partidas, 5)
print(f"con oro a los 15 en las dos puntas: {np.isfinite(Y).all(1).mean():.1%} ({time.time() - t0:.0f}s)", flush=True)

idx, vi = np.where(fit)[0], np.where(val)[0]
tramo = np.searchsorted(np.quantile(t[idx], np.linspace(0, 1, args.tramos + 1)[1:-1]), t[idx], side="right")


def entradas(l, filas, modelo):
    a, r = J[filas, l], J[filas, 5 + l]
    m = MT[filas, l:l + 1]
    if modelo == "ridge":
        return np.c_[a - r, m], None
    return np.c_[a, r, m], np.c_[r, a, -m]


def ajustar_predecir(l, tr, te, modelo):
    ok = np.isfinite(Y[tr, l])
    tr = tr[ok]
    Xa, Xr = entradas(l, tr, modelo)
    yt = Y[tr, l]
    if modelo == "ridge":
        sc = StandardScaler().fit(np.r_[Xa, -Xa])
        f = Ridge(alpha=10.0, fit_intercept=False).fit(sc.transform(np.r_[Xa, -Xa]), np.r_[yt, -yt])
        Xp, _ = entradas(l, te, modelo)
        return f.predict(sc.transform(Xp))
    f = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                                      early_stopping=True, validation_fraction=0.1, random_state=0)
    f.fit(np.r_[Xa, Xr], np.r_[yt, -yt])
    pa, pr = entradas(l, te, modelo)
    return (f.predict(pa) - f.predict(pr)) / 2


PRED = {}
for modelo in args.modelos.split(","):
    pr = np.zeros((len(mids), 5))
    for l in range(5):
        for k in range(args.tramos):
            pr[idx[tramo == k], l] = ajustar_predecir(l, idx[tramo != k], idx[tramo == k], modelo)
        pr[vi, l] = ajustar_predecir(l, idx, vi, modelo)
    PRED[modelo] = pr
    print(f"\n=== {modelo} ({time.time() - t0:.0f}s): diferencia de oro a los 15 por línea, en V ===")
    for l, nom in enumerate(LANES):
        ok = np.isfinite(Y[vi, l])
        yy, pp = Y[vi, l][ok], pr[vi, l][ok]
        print(f"  {nom:8s} R² {r2_score(yy, pp):+.4f}  correlación {np.corrcoef(yy, pp)[0, 1]:.3f}  "
              f"AUC de 'azul gana la línea' {roc_auc_score(yy > 0, pp):.4f}  | ajuste fuera de tramo: correlación "
              f"{np.corrcoef(Y[idx, l][np.isfinite(Y[idx, l])], pr[idx, l][np.isfinite(Y[idx, l])])[0, 1]:.3f}", flush=True)
    yy = np.nansum(Y[vi], 1)
    print(f"  suma de las 5 contra la suma real: correlación {np.corrcoef(yy, pr[vi].sum(1))[0, 1]:.3f}; "
          f"AUC de la suma para ganar la partida {roc_auc_score(y[vi], pr[vi].sum(1)):.4f}")
pickle.dump({"mids": mids, "nuevo": {k: np.c_[v, v.sum(1)] for k, v in PRED.items()}}, open(AN / "ganador_lineas.pkl", "wb"),
            protocol=4)

# ---------- aporte al modelo del equipo
yv = y[val]
B = np.random.default_rng(0).integers(0, val.sum(), (200, val.sum()))


def ajustar(Xm):
    sc = StandardScaler(with_mean=False).fit(Xm[fit])
    return LogisticRegression(C=0.05, max_iter=5000).fit(sc.transform(Xm[fit]), y[fit]).predict_proba(sc.transform(Xm[val]))[:, 1]


def informe(nombre, p, ref=None):
    extra = ""
    if ref is not None:
        dd = np.array([roc_auc_score(yv[b], p[b]) - roc_auc_score(yv[b], ref[b]) for b in B])
        extra = f"  dif {roc_auc_score(yv, p) - roc_auc_score(yv, ref):+.4f} [{np.quantile(dd, .025):+.4f};{np.quantile(dd, .975):+.4f}]"
    print(f"  {nombre:30s} AUC {roc_auc_score(yv, p):.4f} acc {accuracy_score(yv, p > .5):.4f} logloss {log_loss(yv, p):.4f}{extra}", flush=True)

print("\n=== Modelo del equipo: logística con Poisson y nivel del lobby ===")
ref = ajustar(base_X); informe("base", ref)
for k, v in PRED.items():
    informe(f"+ líneas ({k})", ajustar(np.hstack([base_X, np.c_[v, v.sum(1)]])), ref)
if len(PRED) > 1:
    informe("+ líneas (todos)", ajustar(np.hstack([base_X] + [np.c_[v, v.sum(1)] for v in PRED.values()])), ref)
print("\nSolas (las 5 líneas y la suma, sin la base):")
for k, v in PRED.items():
    informe(f"líneas ({k})", ajustar(np.c_[v, v.sum(1)]))
print(f"total {time.time() - t0:.0f}s")
