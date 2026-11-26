"""Huella progresiva para quién gana: cada partida usa la huella de las partidas ANTERIORES del jugador,
y la logística recibe además con cuántas partidas se midió esa huella.

Con la huella congelada en una fecha, todo lo anterior se gasta en armarla y el ajuste queda en 26.934
partidas (PROJECT_SUMMARY.md §9.S.23, hallazgo 6). Acá la huella avanza con el jugador, así que se puede
ajustar con todo el train anterior a la validación sin que la huella envejezca.

Sin fuga: la huella de una partida solo usa partidas anteriores del jugador (la propia no). Las medias y
desvíos por campeón, línea y resultado de `residualize` salen solo de partidas anteriores a la validación.
`k` y la vida media se eligen con el 15 % más nuevo del ajuste; la validación se mira una vez por modelo.

Necesita la salida de `ganador_features_grande.py` (variables comunes, validación y máscaras) y lee
`data/splits/train/full_dataset.csv`. El test no se toca.

    python historial/ganador_primeros/ganador_features_grande.py
    python historial/ganador_primeros/ganador_huella_progresiva.py        # ~10 min
"""
import argparse
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.features.fingerprint import MIN_GROUP, STYLE_FEATURES, add_rates  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("--feats", default=str(REPO / "data" / "analysis" / "ganador_feats_grande.pkl"))
ap.add_argument("--salida", default=str(REPO / "data" / "analysis" / "ganador_huella_progresiva.pkl"))
args = ap.parse_args()

LANES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
COMUN = ["onehot", "experiencia", "cobertura", "dificultad", "matchups", "sinergias"]
KS = (0, 5, 10, 20)               # encogimiento: suma / (n + k)
VIDAS = (None, 30, 100)           # vida media en partidas del jugador; None = promedio simple
t0 = time.time()

# ---------- mismas partidas y mismo orden que ganador_features_grande.py
df = pd.read_csv(REPO / "data" / "splits" / "train" / "full_dataset.csv")
df = df[df.lane.isin(LANES)]
df = df[df.groupby(["match_id", "team_id"]).lane.transform("nunique") == 5]
df = df[df.groupby("match_id").lane.transform("size") == 10].copy()
df["s"] = np.where(df.team_id == 100, 1, -1)
df = df.sort_values(["game_creation", "match_id", "team_id", "lane"]).reset_index(drop=True)
mt = df.groupby("match_id").game_creation.first()
mids = mt.sort_values().index.to_numpy()

d = pickle.load(open(args.feats, "rb"))
feat, y, t, val, masks = d["feat"], d["y"], d["t"], d["val"], d["masks"]
assert np.array_equal(mt.reindex(mids).to_numpy(), t), "el orden de partidas no coincide con ganador_feats_grande.pkl"
mitad = t[val].min()                                      # inicio de la validación
print(f"{len(mids):,} partidas; validación desde {pd.to_datetime(mitad, unit='ms'):%d/%m %H:%M} ({val.sum():,})", flush=True)

# ---------- z de estilo con medias de grupo SOLO anteriores a la validación
df = add_rates(df)
F = STYLE_FEATURES
por = ["champion_name", "lane", "win"]
base = df[df.game_creation < mitad].dropna(subset=F)
g = base.groupby(por)[F]
est = g.mean().add_suffix("_m").join(g.std().add_suffix("_s")).join(g.size().rename("n_g"))
est = est[est.n_g >= MIN_GROUP]
df = df.join(est, on=por)
Z = (df[F].to_numpy() - df[[f + "_m" for f in F]].to_numpy()) / df[[f + "_s" for f in F]].replace(0, np.nan).to_numpy()
valida = np.isfinite(Z).all(axis=1)
Z[~valida] = 0.0
print(f"filas con z válida: {valida.mean():.1%}", flush=True)

# ---------- acumulados por jugador, solo partidas anteriores
orden = np.lexsort((df.match_id.to_numpy(), df.game_creation.to_numpy(), df.player_id.astype(str).to_numpy()))
pid = df.player_id.to_numpy()[orden]
Zo, vo = Z[orden], valida[orden].astype(float)
nuevo = np.r_[True, pid[1:] != pid[:-1]]
inicio = np.maximum.accumulate(np.where(nuevo, np.arange(len(pid)), 0))


def acumulado_previo(x):
    """Suma de las filas anteriores del mismo jugador (sin la actual)."""
    c = np.cumsum(x, axis=0)
    c0 = np.where(inicio[:, None] > 0, c[inicio - 1], 0) if x.ndim == 2 else np.where(inicio > 0, c[inicio - 1], 0)
    return c - x - c0


S = acumulado_previo(Zo * vo[:, None])
N = acumulado_previo(vo)


def ewm_previo(h):
    """Media exponencial de las partidas anteriores (pesos 2^(-edad/h)), con su n efectivo."""
    a = 0.5 ** (1.0 / h)
    num = np.zeros_like(Zo); den = np.zeros(len(pid))
    s, w = np.zeros(Zo.shape[1]), 0.0
    for i in range(len(pid)):
        if nuevo[i]:
            s[:] = 0.0; w = 0.0
        num[i], den[i] = s, w
        s = a * s + Zo[i] * vo[i]; w = a * w + vo[i]
    return num, den


def a_partida(X):
    """De filas en orden por jugador a filas en el orden original de df."""
    out = np.empty_like(X)
    out[orden] = X
    return out


df = df.sort_values(["match_id", "lane"])
fila = df.index.to_numpy()                                # posición en el orden original de cada fila ya reordenada
row = pd.Series(np.arange(len(mids)), index=mids)[df.match_id].to_numpy()
lane_i = df.lane.map({l: i for i, l in enumerate(LANES)}).to_numpy()
s = df.s.to_numpy()


def por_linea(X, total=True):
    dd = X.shape[1]
    out = np.zeros((len(mids), (6 if total else 5) * dd))
    for li in range(5):
        sel = lane_i == li
        np.add.at(out[:, li * dd:(li + 1) * dd], row[sel], X[sel] * s[sel, None])
    if total:
        np.add.at(out[:, 5 * dd:], row, X * s[:, None])
    return out


n_previas = a_partida(N)[fila]
feat["huella_n"] = por_linea(np.log1p(n_previas)[:, None])      # con cuántas partidas se midió la huella
huellas, filas_h = {}, {}
for vida in VIDAS:
    num, den = (S, N) if vida is None else ewm_previo(vida)
    filas_h[vida] = (num.astype(np.float32), den)
    for k in KS:
        huellas[(vida, k)] = por_linea(a_partida(num / (den[:, None] + k + 1e-9))[fila], total=False)
print(f"huellas listas ({time.time() - t0:.0f} s); partidas previas por fila: mediana {np.median(n_previas):.0f}", flush=True)

# ---------- modelos
yv = y[val]
B = np.random.default_rng(0).integers(0, val.sum(), (500, val.sum()))


def logistica(X, fit, pred):
    sc = StandardScaler(with_mean=False).fit(X[fit])
    m = LogisticRegression(C=0.05, max_iter=5000).fit(sc.transform(X[fit]), y[fit])
    return m.predict_proba(sc.transform(X[pred]))[:, 1]


def matriz(extra):
    return np.hstack([feat[k] for k in COMUN] + extra)


# Selección de k y vida media con el 15 % más nuevo del ajuste completo (sin mirar la validación)
fitC = masks["C"]
corte = np.quantile(t[fitC], 0.85)
sel_tr, sel_es = fitC & (t < corte), fitC & (t >= corte)
res_sel = {}
for clave, H in huellas.items():
    p = logistica(matriz([H, feat["huella_n"]]), sel_tr, sel_es)
    res_sel[clave] = roc_auc_score(y[sel_es], p)
    print(f"  selección: vida {str(clave[0]):>4s}, k {clave[1]:>2d} -> AUC {res_sel[clave]:.4f}", flush=True)
mejor = max(res_sel, key=res_sel.get)
print(f"elegida: vida media {mejor[0]}, k {mejor[1]}", flush=True)
H = huellas[mejor]
huella_fila = filas_h[mejor[0]][0] / (filas_h[mejor[0]][1][:, None] + mejor[1] + 1e-9)

modelos = {
    "Referencia: huella congelada al 04/09, ajuste 04/09-16/09": (matriz([feat["huella_foto"]]), masks["A"]),
    "Progresiva + n, mismo ajuste (04/09-16/09)": (matriz([H, feat["huella_n"]]), masks["A"]),
    "Progresiva sin n, ajuste con todo el train previo": (matriz([H]), fitC),
    "Progresiva + n, ajuste con todo el train previo": (matriz([H, feat["huella_n"]]), fitC),
    "Sin huella, ajuste con todo el train previo": (matriz([]), fitC),
}
preds = {}
for nombre, (X, fit) in modelos.items():
    t1_ = time.time()
    preds[nombre] = p = logistica(X, fit, val)
    lo, hi = np.quantile([roc_auc_score(yv[b], p[b]) for b in B], [0.025, 0.975])
    print(f"{nombre:58s} ajuste {fit.sum():>7,d}  acc {accuracy_score(yv, p > .5):.3f}  AUC {roc_auc_score(yv, p):.4f} "
          f"[{lo:.3f}-{hi:.3f}]  logloss {log_loss(yv, p):.4f}  ({time.time() - t1_:.0f} s)", flush=True)

ref = preds["Referencia: huella congelada al 04/09, ajuste 04/09-16/09"]
print("\nDiferencia de AUC contra la referencia (bootstrap pareado de la validación, IC 95 %):")
for nombre, p in preds.items():
    dif = [roc_auc_score(yv[b], p[b]) - roc_auc_score(yv[b], ref[b]) for b in B]
    print(f"  {nombre:58s} {np.mean(dif):+.4f} [{np.quantile(dif, .025):+.4f}, {np.quantile(dif, .975):+.4f}]")

# AUC según cuántas partidas previas tiene el lobby (¿la huella sirve más donde está bien medida?)
n_min = pd.Series(n_previas).groupby(row).min().reindex(range(len(mids))).to_numpy()
print("\nAUC en validación según el mínimo de partidas previas del lobby:")
mejor_nombre = "Progresiva + n, ajuste con todo el train previo"
for a, b in [(0, 10), (10, 30), (30, 60), (60, 100), (100, 10**6)]:
    m = (n_min[val] >= a) & (n_min[val] < b)
    if m.sum() > 500:
        print(f"  [{a},{b}): {m.sum():6,d} partidas  ref {roc_auc_score(yv[m], ref[m]):.3f}  progresiva {roc_auc_score(yv[m], preds[mejor_nombre][m]):.3f}")

pickle.dump({"preds": preds, "seleccion": res_sel, "mejor": mejor, "yv": yv,
             "huella_prog": H, "huella_n": feat["huella_n"],
             # por fila (jugador en una partida), para modelos que no restan azul - rojo
             "fila": {"match_id": df.match_id.to_numpy(), "team_id": df.team_id.to_numpy(), "lane": df.lane.to_numpy(),
                      "huella": a_partida(huella_fila)[fila], "n": n_previas}}, open(args.salida, "wb"))
print(f"\nlisto en {time.time() - t0:.0f} s -> {args.salida}")
