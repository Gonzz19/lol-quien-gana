"""Rasgos condicionales del jugador y su cruce con el rival de línea (§9.S.35).

Idea del usuario (2026-11-13): "hay jugadores que si les das una kill antes del minuto 10 te
hacen la partida entera". Rasgos por jugador con sus partidas PREVIAS, encogidos hacia la
media de la población con k = 10 (src/features/conditional.py):
  adelante10   P(diferencia de oro con el rival a los 10 > +300)
  atras10      P(diferencia de oro a los 10 < −300)
  kill10       P(al menos una kill antes del 10)
  muere10      P(al menos una muerte antes del 10)          (el que "regala")
  conversion   media de (ganó − p̂) en las partidas en que iba > +500 de oro a los 15
  remontada    media de (ganó − p̂) en las partidas en que iba < −500 de oro a los 15
  largas/cortas media de (ganó − p̂) en partidas > 32 min / < 25 min (dependencia de duración)
  volatilidad  desvío de la diferencia de oro a los 15
con p̂ = P(gana | diferencia de oro a los 15, línea), una logística por línea ajustada con las
filas de ajuste: el rasgo mide cuánto gana MÁS de lo que su ventaja de oro predice.
Cruces con el rival directo de línea, antisimétricos (azul·rojo − rojo·azul, con los rasgos
centrados): conversión × muere10 del rival, adelante10 × atras10 del rival, kill10 × muere10
del rival, remontada × adelante10 del rival.
Antes de modelar: confiabilidad partidas pares contra impares (rasgo sin encoger, jugadores
con >= 40 partidas en el ajuste). Base: la logística con Poisson y nivel del lobby (0,6493).
Solo data/splits/train."""
import pickle, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.features.conditional import prior_conditional_mean
from src.features.margin import lane_diffs

AN = REPO / "data/analysis"
LANES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
K = 10
t0 = time.time()
df = pd.read_csv(REPO / "data/splits/train/full_dataset.csv",
                 usecols=["match_id", "game_creation", "game_duration", "player_id", "team_id", "lane", "win",
                          "kills_10", "deaths_10", "gold_10", "gold_15"])
df = df[df.lane.isin(LANES)]
df = df[df.groupby(["match_id", "team_id"]).lane.transform("nunique") == 5]
df = df[df.groupby("match_id").lane.transform("size") == 10].copy()
df = df.sort_values(["game_creation", "match_id", "team_id", "lane"]).reset_index(drop=True)
df = lane_diffs(df, ["gold_10", "gold_15"])

d = pickle.load(open(AN / "ganador_feats_grande.pkl", "rb"))
feat, mids, y, val, fit = d["feat"], d["mids"], d["y"], d["val"], d["masks"]["C"]
X = [feat[k] for k in ["onehot", "experiencia", "cobertura", "dificultad", "matchups", "sinergias"]]
del d, feat
M = pickle.load(open(AN / "ganador_margen_duos.pkl", "rb"))["nuevo"]
X += [M[k] for k in ("winrate previo", "margen acum", "margen v20", "duos")]
del M
X += list(pickle.load(open(AN / "ganador_margen_grupos.pkl", "rb"))["nuevo"].values())
P = pickle.load(open(AN / "ganador_poisson.pkl", "rb"))
X.append(np.hstack(list(P["poisson"].values())))
del P
X.append(pickle.load(open(AN / "ganador_mmr.pkl", "rb"))["nuevo"]["nivel (foto)"])
base_X = np.hstack(X)
del X
row = pd.Series(np.arange(len(mids)), index=mids).reindex(df.match_id).to_numpy()
assert not np.isnan(row).any()
row = row.astype(int)
fila_fit = fit[row]
lane_i = df.lane.map({l: i for i, l in enumerate(LANES)}).to_numpy()
s = np.where(df.team_id == 100, 1.0, -1.0)

# ---------- p̂(gana | oro a los 15, línea), ajustado con las filas de ajuste
g15 = df.d_gold_15
phat = pd.Series(np.nan, index=df.index)
for l in LANES:
    m = (df.lane == l) & g15.notna()
    lr = LogisticRegression().fit((g15[m & fila_fit] / 1000).to_frame(), df.win[m & fila_fit])
    phat[m] = lr.predict_proba((g15[m] / 1000).to_frame())[:, 1]
resid = df.win - phat
dur = df.game_duration / 60
g10 = df.d_gold_10
DEF = {   # nombre: (valor, condición)
    "adelante10": ((g10 > 300).astype(float).where(g10.notna()), pd.Series(True, index=df.index)),
    "atras10": ((g10 < -300).astype(float).where(g10.notna()), pd.Series(True, index=df.index)),
    "kill10": ((df.kills_10 >= 1).astype(float).where(df.kills_10.notna()), pd.Series(True, index=df.index)),
    "muere10": ((df.deaths_10 >= 1).astype(float).where(df.deaths_10.notna()), pd.Series(True, index=df.index)),
    "conversion": (resid, g15 > 500),
    "remontada": (resid, g15 < -500),
    "largas": (resid, dur > 32),
    "cortas": (resid, dur < 25),
    "oro15": (g15 / 1000, pd.Series(True, index=df.index)),
    "oro15_2": ((g15 / 1000) ** 2, pd.Series(True, index=df.index)),
}

# ---------- confiabilidad: pares contra impares (sin encoger), jugadores con >= 40 partidas de ajuste
print("=== Confiabilidad del rasgo: partidas pares contra impares del mismo jugador ===")
fa = df[fila_fit].copy()
fa["par"] = fa.groupby("player_id").cumcount() % 2
n_j = fa.groupby("player_id").size()
fa = fa[fa.player_id.isin(n_j[n_j >= 40].index)]
print(f"jugadores: {fa.player_id.nunique()}")
for nom, (v, c) in DEF.items():
    if nom == "oro15_2":
        continue
    vv, cc = v[fa.index], c[fa.index] & v[fa.index].notna()
    t = pd.DataFrame({"pj": fa.player_id, "par": fa.par, "v": vv.where(cc)}).groupby(["pj", "par"]).v.agg(["mean", "count"]).unstack()
    t = t[(t["count"] >= 5).all(axis=1)]
    r = np.corrcoef(t["mean"][0], t["mean"][1])[0, 1]
    print(f"  {nom:12s} r = {r:+.3f}  (jugadores con >= 5 partidas en la condición en cada mitad: {len(t)}; "
          f"Spearman-Brown con todas: {2 * r / (1 + r):+.3f})")
t = pd.DataFrame({"pj": fa.player_id, "par": fa.par, "v": g15[fa.index] / 1000}).groupby(["pj", "par"]).v.std().unstack().dropna()
r = np.corrcoef(t[0], t[1])[0, 1]
print(f"  {'volatilidad':12s} r = {r:+.3f}  (desvío de la diferencia de oro a los 15)")

# ---------- rasgos previos encogidos
R = {}
for nom, (v, c) in DEF.items():
    mu = float(v[c & fila_fit & v.notna()].mean())
    R[nom] = prior_conditional_mean(df, v, c, k=K, mu=mu).rasgo
R = pd.DataFrame(R)
R["volatilidad"] = np.sqrt((R.oro15_2 - R.oro15 ** 2).clip(lower=0))
R["duracion"] = R.largas - R.cortas
R = R.drop(columns=["oro15_2", "largas", "cortas", "oro15"])
Rc = R - R[fila_fit].mean()
print(f"\nrasgos ({time.time() - t0:.0f}s): {list(R.columns)}")


def por_linea(Xp):
    k = Xp.shape[1]
    out = np.zeros((len(mids), 6 * k))
    for li in range(5):
        sel = lane_i == li
        np.add.at(out[:, li * k:(li + 1) * k], row[sel], Xp[sel] * s[sel, None])
    np.add.at(out[:, 5 * k:], row, Xp * s[:, None])
    return out

# cruces con el rival de línea: f(azul, rojo) - f(rojo, azul), por línea y total
puesto = 5 * (s == -1).astype(int) + lane_i
T = np.zeros((len(mids), 10, Rc.shape[1]))
T[row, puesto] = Rc.to_numpy()
col = {c: i for i, c in enumerate(Rc.columns)}
CRUCES = [("conversion", "muere10"), ("adelante10", "atras10"), ("kill10", "muere10"), ("remontada", "adelante10")]
cr = []
for a, b in CRUCES:
    A, Bm = T[:, :, col[a]], T[:, :, col[b]]
    lin = A[:, :5] * Bm[:, 5:] - A[:, 5:] * Bm[:, :5]
    cr.append(np.c_[lin, lin.sum(1)])
NUEVO = {"rasgos": por_linea(Rc.to_numpy()), "cruces con el rival": np.hstack(cr)}
pickle.dump({"mids": mids, "nuevo": NUEVO, "jug": T.astype(np.float32), "columnas": list(Rc.columns)},
            open(AN / "ganador_snowball.pkl", "wb"), protocol=4)

# ---------- evaluación
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

print("\n=== Sobre la logística con Poisson y nivel del lobby ===")
ref = ajustar(base_X); informe("base", ref)
for k, Xn in NUEVO.items():
    informe(f"+ {k}", ajustar(np.hstack([base_X, Xn])), ref)
informe("+ rasgos y cruces", ajustar(np.hstack([base_X] + list(NUEVO.values()))), ref)
for i, nom in enumerate(Rc.columns):
    informe(f"+ solo {nom}", ajustar(np.hstack([base_X, NUEVO["rasgos"][:, [i + j * Rc.shape[1] for j in range(6)]]])), ref)
print("\nLas medidas solas (sin la base):")
for k, Xn in NUEVO.items():
    informe(k, ajustar(Xn))
print(f"total {time.time() - t0:.0f}s")
