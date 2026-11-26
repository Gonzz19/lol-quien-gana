"""Cambio reciente del jugador y encaje del equipo (§9.S.30), sobre la mejor logística de
§9.S.29 (config C con margen, duos y margen por campeón / línea / últimas 5: 0,6446 en V).

Hipótesis (usuario, 2026-11-13): una cuenta boosteada o compartida se ve como un jugador que
se vuelve OTP y empieza a ganar; su historial deja de describir a quien juega.
Medidas por jugador (src/features/change.py), últimas `--ventana` partidas PREVIAS contra las
anteriores a esas:
  pool        d_pool (concentración del pool), otp_nuevo (peso del campeón actual)
  resultados  d_win, y la conjunción otp_nuevo+ * d_win+ ("se volvió OTP y empezó a ganar")
  huella      cambio_huella (distancia de estilo), y cambio_huella * d_win
  hábitos     cambio_horario, ritmo
Las distancias (huella, horario) no tienen signo y su valor esperado sin cambio depende de
cuántas partidas anteriores hay: se les resta la media de las filas de ajuste con la misma
cantidad de partidas previas (20 cuantiles). Sin las dos ventanas, la medida vale 0.
Encaje del equipo (por lado, con el margen acumulado de porción de oro y de daño de los 5):
|suma| y desvío entre los 5; la suma sola ya está en el bloque "margen acum".
Todo entra como azul - rojo. Solo data/splits/train."""
import argparse, pickle, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.features.fingerprint import add_rates, STYLE_FEATURES
from src.features.margin import lane_scale, prior_margin
from src.features.change import hour_shift, play_rate, pool_change, style_scale, style_shift, window_vs_before

ap = argparse.ArgumentParser()
ap.add_argument("--feats", default=str(REPO / "data/analysis/ganador_feats_grande.pkl"))
ap.add_argument("--margen", default=str(REPO / "data/analysis/ganador_margen_duos.pkl"))
ap.add_argument("--grupos", default=str(REPO / "data/analysis/ganador_margen_grupos.pkl"))
ap.add_argument("--salida", default=str(REPO / "data/analysis/ganador_cambio.pkl"))
ap.add_argument("--ventana", type=int, default=10)
ap.add_argument("--min-previas", type=int, default=5)
args = ap.parse_args()
W, MP = args.ventana, args.min_previas

LANES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
t0 = time.time()
df = pd.read_csv(REPO / "data/splits/train/full_dataset.csv")
df = df[df.lane.isin(LANES)]
df = df[df.groupby(["match_id", "team_id"]).lane.transform("nunique") == 5]
df = df[df.groupby("match_id").lane.transform("size") == 10].copy()
df = add_rates(df).sort_values(["game_creation", "match_id", "team_id", "lane"]).reset_index(drop=True)

d = pickle.load(open(args.feats, "rb"))
feat, mids, y, val, fit = d["feat"], d["mids"], d["y"], d["val"], d["masks"]["C"]
base = [feat[k] for k in ["onehot", "experiencia", "cobertura", "dificultad", "matchups", "sinergias"]]
del d, feat
M = pickle.load(open(args.margen, "rb"))["nuevo"]
base += [M[k] for k in ("winrate previo", "margen acum", "margen v20", "duos")]
del M
G = pickle.load(open(args.grupos, "rb"))["nuevo"]
base_X = np.hstack(base + list(G.values()))
del G, base
row = pd.Series(np.arange(len(mids)), index=mids).reindex(df.match_id).to_numpy()
assert not np.isnan(row).any()
row = row.astype(int)
fila_fit = fit[row]
lane_i = df.lane.map({l: i for i, l in enumerate(LANES)}).to_numpy()
s = np.where(df.team_id == 100, 1.0, -1.0)

# ---------- medidas de cambio (la escala del estilo sale solo de las filas de ajuste)
P = pool_change(df, W, MP)
V = window_vs_before(df, ["win"], W, MP)
H = style_shift(df, STYLE_FEATURES, W, MP, escala=style_scale(df[fila_fit], STYLE_FEATURES))
hor = hour_shift(df, W, MP)
rit = play_rate(df)
print(f"medidas de cambio ({time.time() - t0:.0f}s); con las dos ventanas: {P.d_pool.notna().mean():.1%} de las filas", flush=True)

n_prev = df.groupby("player_id", sort=False).cumcount().to_numpy()

def centrar(x):
    """Resta la media de las filas de ajuste con parecida cantidad de partidas previas."""
    ok = x.notna().to_numpy()
    cortes = np.unique(np.quantile(n_prev[ok & fila_fit], np.linspace(0, 1, 21)[1:-1]))
    b = np.searchsorted(cortes, n_prev)
    media = pd.Series(x.to_numpy()[ok & fila_fit]).groupby(b[ok & fila_fit]).mean()
    return (x - pd.Series(b, index=x.index).map(media)).fillna(0).to_numpy()

d_win = V.d_win.fillna(0).to_numpy()
d_pool, otp = P.d_pool.fillna(0).to_numpy(), P.otp_nuevo.fillna(0).to_numpy()
huella, horario = centrar(H.cambio_huella), centrar(hor)
J = {"pool": np.c_[d_pool, otp],
     "resultados": np.c_[d_win, np.clip(otp, 0, None) * np.clip(d_win, 0, None)],
     "huella": np.c_[huella, huella * d_win],
     "hábitos": np.c_[horario, rit.fillna(0).to_numpy()]}
COLS = ["d_pool", "otp_nuevo", "d_win", "otp_y_gana", "cambio_huella", "huella_x_d_win", "cambio_horario", "ritmo"]

print("\nDescripción por jugador-partida (filas con las dos ventanas, ajuste):")
desc = pd.DataFrame(np.hstack(list(J.values())), columns=COLS)[P.d_pool.notna().to_numpy() & fila_fit]
print(desc.describe().T[["mean", "std", "min", "50%", "max"]].round(3).to_string())
gana = df.win.to_numpy()[P.d_pool.notna().to_numpy() & fila_fit]
print("correlación con ganar la partida:", {c: round(float(np.corrcoef(desc[c], gana)[0, 1]), 4) for c in COLS})

def por_linea(X):
    k = X.shape[1]
    out = np.zeros((len(mids), 6 * k))
    for li in range(5):
        sel = lane_i == li
        np.add.at(out[:, li * k:(li + 1) * k], row[sel], X[sel] * s[sel, None])
    np.add.at(out[:, 5 * k:], row, X * s[:, None])
    return out

NUEVO = {k: por_linea(v) for k, v in J.items()}

# ---------- encaje del equipo: porción habitual de oro y de daño de los 5
SH = ["gold_share_pct", "damage_share_pct"]
A = prior_margin(df, SH, escala=lane_scale(df, SH)).filter(like="acum_").to_numpy()
puesto = 5 * (s == -1).astype(int) + lane_i
eq = np.zeros((len(mids), 10, 2))
eq[row, puesto] = A
az, ro = eq[:, :5], eq[:, 5:]
NUEVO["encaje"] = np.hstack([np.abs(az.sum(1)) - np.abs(ro.sum(1)), az.std(1) - ro.std(1)])

jug = np.zeros((len(mids), 10, len(COLS)), dtype=np.float32)
jug[row, puesto] = np.hstack(list(J.values()))
pickle.dump({"mids": mids, "nuevo": NUEVO, "jug": jug, "columnas": COLS, "ventana": W},
            open(args.salida, "wb"), protocol=4)

# ---------- evaluación
yv = y[val]
B = np.random.default_rng(0).integers(0, val.sum(), (200, val.sum()))

def ajustar(X):
    sc = StandardScaler(with_mean=False).fit(X[fit])
    return LogisticRegression(C=0.05, max_iter=5000).fit(sc.transform(X[fit]), y[fit]).predict_proba(sc.transform(X[val]))[:, 1]

def informe(nombre, p, ref=None):
    extra = ""
    if ref is not None:
        dif = np.array([roc_auc_score(yv[b], p[b]) - roc_auc_score(yv[b], ref[b]) for b in B])
        extra = f"  dif {roc_auc_score(yv, p) - roc_auc_score(yv, ref):+.4f} [{np.quantile(dif, .025):+.4f};{np.quantile(dif, .975):+.4f}]"
    print(f"  {nombre:34s} acc {accuracy_score(yv, p > .5):.4f}  AUC {roc_auc_score(yv, p):.4f}  logloss {log_loss(yv, p):.4f}{extra}", flush=True)

print(f"\n=== Config C con margen, duos y margen por grupo; ventana = {W}, mínimo de anteriores = {MP} ===")
ref = ajustar(base_X); informe("base", ref)
for k in NUEVO:
    informe(f"base + {k}", ajustar(np.hstack([base_X, NUEVO[k]])), ref)
informe("base + todo", ajustar(np.hstack([base_X] + list(NUEVO.values()))), ref)
print("\nLas medidas solas (sin la base):")
for k in NUEVO:
    informe(k, ajustar(NUEVO[k]))
print(f"total {time.time() - t0:.0f}s")
