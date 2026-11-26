"""¿La fuerza del campeón en el parche en curso ayuda a predecir quién gana? (§9.S.28)

Para cada fila (campeón c, línea l) y solo con partidas ANTERIORES:
    e_cl     winrate acumulado de (c, l) en toda la historia, encogido hacia 0,5 (lo que ya
             usan matchups y sinergias, ganador_features_grande.py);
    parche   desvío del winrate de (c, l) dentro del parche en curso:
             (G_p + K e_cl) / (N_p + K) - e_cl, con N_p y G_p las partidas y victorias previas
             del mismo parche: al empezar cada parche vale 0 y se mueve a medida que se juega;
    reciente lo mismo con un promedio con olvido exponencial por partidas de (c, l), que no
             sabe dónde empieza cada parche (control: ¿importa el corte del parche o solo lo nuevo?);
    tasa     cuántas veces se eligió (c, l) en el parche hasta ahora, sobre el total de partidas
             del parche hasta ahora (popularidad en el parche).
Entran azul - rojo por línea y total, sumadas a la logística de la config C con margen y duos
(0,636 en V). Solo data/splits/train."""
import argparse, pickle, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--feats", default=str(REPO / "data/analysis/ganador_feats_grande.pkl"))
ap.add_argument("--margen", default=str(REPO / "data/analysis/ganador_margen_duos.pkl"))
ap.add_argument("--k", type=float, default=50, help="encogimiento hacia el acumulado (partidas)")
ap.add_argument("--vida", type=float, default=300, help="vida media del olvido exponencial (partidas de c, l)")
args = ap.parse_args()

LANES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
df = pd.read_csv(REPO / "data/splits/train/full_dataset.csv",
                 usecols=["match_id", "game_creation", "game_version", "team_id", "win", "champion_id", "lane"])
df = df[df.lane.isin(LANES)]
df = df[df.groupby(["match_id", "team_id"]).lane.transform("nunique") == 5]
df = df[df.groupby("match_id").lane.transform("size") == 10].copy()
df = df.sort_values(["game_creation", "match_id", "team_id", "lane"]).reset_index(drop=True)
df["parche"] = df.game_version.str.split(".").str[:2].str.join(".")

cl = df.groupby(["champion_id", "lane"])
df["e_cl"] = ((cl.win.cumsum() - df.win) + 10) / (cl.cumcount() + 20)
cp = df.groupby(["champion_id", "lane", "parche"])
n_p, g_p = cp.cumcount(), cp.win.cumsum() - df.win
df["x_parche"] = (g_p + args.k * df.e_cl) / (n_p + args.k) - df.e_cl
# olvido exponencial por partidas de (c, l): S_t = a S_{t-1} + w, N_t = a N_{t-1} + 1, con valores previos
a = 0.5 ** (1 / args.vida)
orden_cl = cl.cumcount().to_numpy()
w = df.win.to_numpy(float)
S, N = np.zeros(len(df)), np.zeros(len(df))
ult_S, ult_N = {}, {}
for i, key in enumerate(zip(df.champion_id.to_numpy(), df.lane.to_numpy())):
    s0, n0 = ult_S.get(key, 0.0), ult_N.get(key, 0.0)
    S[i], N[i] = s0, n0                                  # solo lo anterior
    ult_S[key], ult_N[key] = a * s0 + w[i], a * n0 + 1
df["x_reciente"] = (S + args.k * df.e_cl) / (N + args.k) - df.e_cl
# popularidad dentro del parche: elecciones previas de (c, l) / partidas previas del parche
partidas_parche = df.groupby("parche").match_id.transform(lambda s: pd.factorize(s)[0])
df["x_tasa"] = n_p / (partidas_parche + 20)

d = pickle.load(open(args.feats, "rb"))
feat, mids, y, val, fit = d["feat"], d["mids"], d["y"], d["val"], d["masks"]["C"]
M = pickle.load(open(args.margen, "rb"))
row = pd.Series(np.arange(len(mids)), index=mids).reindex(df.match_id).to_numpy()
assert not np.isnan(row).any()
row = row.astype(int)
lane_i = df.lane.map({l: i for i, l in enumerate(LANES)}).to_numpy()
s = np.where(df.team_id == 100, 1.0, -1.0)

def por_linea(cols):
    X = df[cols].to_numpy()
    k = X.shape[1]
    out = np.zeros((len(mids), 6 * k))
    for li in range(5):
        sel = lane_i == li
        np.add.at(out[:, li * k:(li + 1) * k], row[sel], X[sel] * s[sel, None])
    np.add.at(out[:, 5 * k:], row, X * s[:, None])
    return out

NUEVO = {"parche": por_linea(["x_parche"]), "reciente": por_linea(["x_reciente"]), "tasa": por_linea(["x_tasa"]),
         "e_cl directo": por_linea(["e_cl"])}
base_X = np.hstack([feat[k] for k in ["onehot", "experiencia", "cobertura", "dificultad", "matchups", "sinergias"]]
                   + [M["nuevo"][k] for k in ("winrate previo", "margen acum", "margen v20", "duos")])
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
    print(f"  {nombre:36s} acc {accuracy_score(yv, p > .5):.3f}  AUC {roc_auc_score(yv, p):.4f}  logloss {log_loss(yv, p):.4f}{extra}", flush=True)

print(f"Config C con margen y duos; K = {args.k:g}, vida media = {args.vida:g} partidas")
base = ajustar(base_X); informe("base (margen + duos)", base)
for k in NUEVO:
    informe(f"base + {k}", ajustar(np.hstack([base_X, NUEVO[k]])), base)
informe("base + parche + tasa", ajustar(np.hstack([base_X, NUEVO["parche"], NUEVO["tasa"]])), base)
informe("base + parche + reciente + tasa", ajustar(np.hstack([base_X, NUEVO["parche"], NUEVO["reciente"], NUEVO["tasa"]])), base)
