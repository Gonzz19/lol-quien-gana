"""Margen previo afinado (§9.S.29): por (jugador, campeón), por (jugador, línea) y forma muy
reciente (últimas 5), sobre la logística de la config C con margen y duos (0,636 en V).
src/features/margin.py: prior_margin_by (desvío encogido hacia el margen general del jugador).
Escribe un pickle con los bloques (azul - rojo por línea y total, y por jugador para la red).
Solo data/splits/train."""
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
from src.features.fingerprint import add_rates
from src.features.margin import lane_diffs, lane_scale, prior_margin, prior_margin_by

ap = argparse.ArgumentParser()
ap.add_argument("--feats", default=str(REPO / "data/analysis/ganador_feats_grande.pkl"))
ap.add_argument("--margen", default=str(REPO / "data/analysis/ganador_margen_duos.pkl"))
ap.add_argument("--salida", default=str(REPO / "data/analysis/ganador_margen_grupos.pkl"))
ap.add_argument("--kg", type=float, default=10)
args = ap.parse_args()

LANES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
STATS = ["kills_pm", "assists_pm", "damage_pm", "damage_share_pct", "kill_participation_pct", "deaths_pm",
         "damage_taken_pm", "gold_pm", "cs_pm", "gold_share_pct", "vision_pm", "d_gold_10", "d_gold_15", "d_xp_15", "d_cs_15"]
t0 = time.time()
df = pd.read_csv(REPO / "data/splits/train/full_dataset.csv")
df = df[df.lane.isin(LANES)]
df = df[df.groupby(["match_id", "team_id"]).lane.transform("nunique") == 5]
df = df[df.groupby("match_id").lane.transform("size") == 10].copy()
df = add_rates(df).sort_values(["game_creation", "match_id", "team_id", "lane"]).reset_index(drop=True)
df = lane_diffs(df, ["gold_10", "gold_15", "xp_15", "cs_15"])
esc = lane_scale(df, STATS)
J = {"campeón": prior_margin_by(df, STATS, ["champion_id"], "camp", k_g=args.kg, escala=esc),
     "línea": prior_margin_by(df, STATS, ["lane"], "linea", k_g=args.kg, escala=esc),
     "últimas 5": prior_margin(df, STATS, window=5, escala=esc).filter(like="v5_")}
print(f"márgenes por grupo ({time.time() - t0:.0f}s)", flush=True)

d = pickle.load(open(args.feats, "rb"))
feat, mids, y, val, fit = d["feat"], d["mids"], d["y"], d["val"], d["masks"]["C"]
M = pickle.load(open(args.margen, "rb"))
row = pd.Series(np.arange(len(mids)), index=mids).reindex(df.match_id).to_numpy()
assert not np.isnan(row).any()
row = row.astype(int)
lane_i = df.lane.map({l: i for i, l in enumerate(LANES)}).to_numpy()
s = np.where(df.team_id == 100, 1.0, -1.0)

def por_linea(X):
    k = X.shape[1]
    out = np.zeros((len(mids), 6 * k))
    for li in range(5):
        sel = lane_i == li
        np.add.at(out[:, li * k:(li + 1) * k], row[sel], X[sel] * s[sel, None])
    np.add.at(out[:, 5 * k:], row, X * s[:, None])
    return out

NUEVO = {k: por_linea(v.to_numpy()) for k, v in J.items()}
jug = np.zeros((len(mids), 10, sum(v.shape[1] for v in J.values())), dtype=np.float32)
jug[row, 5 * (s == -1) + lane_i] = np.hstack([v.to_numpy() for v in J.values()])
pickle.dump({"mids": mids, "nuevo": NUEVO, "jug": jug}, open(args.salida, "wb"), protocol=4)

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
    print(f"  {nombre:34s} acc {accuracy_score(yv, p > .5):.4f}  AUC {roc_auc_score(yv, p):.4f}  logloss {log_loss(yv, p):.4f}{extra}", flush=True)

print(f"\n=== Config C con margen y duos; k_g = {args.kg:g} ===")
base = ajustar(base_X); informe("base", base)
for k in NUEVO:
    informe(f"base + margen por {k}", ajustar(np.hstack([base_X, NUEVO[k]])), base)
informe("base + los tres", ajustar(np.hstack([base_X] + list(NUEVO.values()))), base)
