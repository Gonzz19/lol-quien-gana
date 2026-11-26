"""Valor de arranque (prior) del margen para jugadores con poco historial (§9.S.35).

Hoy el margen previo es m = S / (N + 5): sin historial vale 0, como si el jugador fuera promedio.
Con un valor esperado mu según su nivel, el margen encogido hacia mu es
    m* = (S + k mu) / (N + k) = m + k mu / (N + k)
(src/features/margin.py: prior_shift). mu sale de una regresión lineal, por estadística, del
residuo r de cada partida (§9.S.24) contra el nivel del jugador ANTES de esa partida:
    LP propio de la foto del 04/09 (y si falta), MMR propio (nivel_acum, §9.S.31) y nivel del
    lobby de esa partida (media del LP de la foto de los otros 9).
La regresión se ajusta solo con las filas de ajuste (son 4 coeficientes por estadística).
Base: la logística con Poisson y nivel del lobby (experto del modelo de 2, 0,6493 en V).
Solo data/splits/train."""
import argparse, pickle, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.features.elo_reverso import cargar_foto
from src.features.fingerprint import add_rates
from src.features.lobby import prior_lobby_level
from src.features.margin import _residuo, lane_diffs, lane_scale, prior_shift

AN = REPO / "data/analysis"
ap = argparse.ArgumentParser()
ap.add_argument("--k", type=float, nargs="+", default=[5, 15])
ap.add_argument("--sin-lp-propio", action="store_true", help="mu sin el LP propio de la foto (solo información previa)")
ap.add_argument("--salida", default=str(AN / "ganador_prior.pkl"))
args = ap.parse_args()

LANES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
STATS = ["kills_pm", "assists_pm", "damage_pm", "damage_share_pct", "kill_participation_pct", "deaths_pm",
         "damage_taken_pm", "gold_pm", "cs_pm", "gold_share_pct", "vision_pm", "d_gold_10", "d_gold_15", "d_xp_15", "d_cs_15"]
t0 = time.time()
db = f"file:{(REPO / 'data/lol_las.db').as_posix()}?mode=ro"
foto1 = cargar_foto(db, "SELECT m.player_id, p.tier, p.rank, p.league_points, p.wins, p.losses "
                        "FROM players p JOIN player_id_map m ON m.puuid = p.puuid")
df = pd.read_csv(REPO / "data/splits/train/full_dataset.csv")
df = df[df.lane.isin(LANES)]
df = df[df.groupby(["match_id", "team_id"]).lane.transform("nunique") == 5]
df = df[df.groupby("match_id").lane.transform("size") == 10].copy()
df = add_rates(df).sort_values(["game_creation", "match_id", "team_id", "lane"]).reset_index(drop=True)
df = lane_diffs(df, ["gold_10", "gold_15", "xp_15", "cs_15"])
df["lp_foto"] = df.player_id.map(foto1.z)

d = pickle.load(open(AN / "ganador_feats_grande.pkl", "rb"))
feat, mids, y, t, val, fit = d["feat"], d["mids"], d["y"], d["t"], d["val"], d["masks"]["C"]
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
n_prev = df.groupby("player_id").cumcount().to_numpy()

# ---------- mu: margen esperado según el nivel antes de la partida
mu0 = float(prior_lobby_level(df[fila_fit], "lp_foto").nivel_lobby.mean())
niv = prior_lobby_level(df, "lp_foto", k=5, mu0=mu0)
Z = pd.DataFrame({"lp_propio": df.lp_foto.fillna(mu0), "sin_lp": df.lp_foto.isna().astype(float),
                  "mmr_propio": niv.nivel_acum, "nivel_partida": niv.nivel_lobby.fillna(mu0)})
Z[["lp_propio", "mmr_propio", "nivel_partida"]] = (Z[["lp_propio", "mmr_propio", "nivel_partida"]] - mu0) / 300
if args.sin_lp_propio:
    Z = Z.drop(columns=["lp_propio", "sin_lp"])
esc = lane_scale(df, STATS)
mu, r2 = {}, {}
for c in STATS:
    r, v = _residuo(df, c, 20, esc)
    ok = (v > 0).to_numpy() & fila_fit
    reg = LinearRegression().fit(Z[ok], r[ok])
    mu[c] = pd.Series(reg.predict(Z), index=df.index)
    r2[c] = reg.score(Z[ok], r[ok])
mu = pd.DataFrame(mu)
print(f"mu por estadística ({time.time() - t0:.0f}s): R² del residuo contra el nivel "
      + ", ".join(f"{c} {v:.3f}" for c, v in sorted(r2.items(), key=lambda x: -x[1])[:6]), flush=True)
print(f"desvío de mu entre filas: {mu.std().mean():.3f} (en desvíos de la estadística)")


def por_linea(Xp):
    k = Xp.shape[1]
    out = np.zeros((len(mids), 6 * k))
    for li in range(5):
        sel = lane_i == li
        np.add.at(out[:, li * k:(li + 1) * k], row[sel], Xp[sel] * s[sel, None])
    np.add.at(out[:, 5 * k:], row, Xp * s[:, None])
    return out

NUEVO = {f"prior k={k:g}": por_linea(prior_shift(df, STATS, mu, k_j=k).to_numpy()) for k in args.k}
pickle.dump({"mids": mids, "nuevo": NUEVO}, open(args.salida, "wb"), protocol=4)

# ---------- evaluación (y por cobertura del lobby)
yv = y[val]
B = np.random.default_rng(0).integers(0, val.sum(), (200, val.sum()))
nuevos = pd.Series(n_prev < 10).groupby(row).sum().reindex(range(len(mids))).to_numpy()[val]
t1 = t[val].min()
from src.features.elo_reverso import FOTO2  # noqa: F401  (solo para dejar claro de dónde salen las fotos)
fit_post = fit & (t >= np.datetime64("2026-09-04T22:51:22", "ms").astype(np.int64))


def ajustar(Xm, m=fit):
    sc = StandardScaler(with_mean=False).fit(Xm[m])
    return LogisticRegression(C=0.05, max_iter=5000).fit(sc.transform(Xm[m]), y[m]).predict_proba(sc.transform(Xm[val]))[:, 1]


def informe(nombre, p, ref):
    dd = np.array([roc_auc_score(yv[b], p[b]) - roc_auc_score(yv[b], ref[b]) for b in B])
    grupos = "  ".join(f"{g} {roc_auc_score(yv[sel], p[sel]):.4f}" for g, sel in
                       (("<10 previas: 0", nuevos == 0), ("1-2", (nuevos >= 1) & (nuevos <= 2)), ("3+", nuevos >= 3)))
    print(f"  {nombre:24s} AUC {roc_auc_score(yv, p):.4f} acc {accuracy_score(yv, p > .5):.4f} logloss {log_loss(yv, p):.4f}"
          f"  dif {roc_auc_score(yv, p) - roc_auc_score(yv, ref):+.4f} [{np.quantile(dd, .025):+.4f};{np.quantile(dd, .975):+.4f}]  | {grupos}", flush=True)

print("\n=== Logística con Poisson y nivel del lobby (experto del modelo de 2) ===")
ref = ajustar(base_X); informe("base", ref, ref)
for k, Xn in NUEVO.items():
    informe(f"+ {k}", ajustar(np.hstack([base_X, Xn])), ref)
print("\nAjustando solo con partidas posteriores a la foto:")
ref2 = ajustar(base_X, fit_post); informe("base", ref2, ref2)
for k, Xn in NUEVO.items():
    informe(f"+ {k}", ajustar(np.hstack([base_X, Xn]), fit_post), ref2)
print(f"total {time.time() - t0:.0f}s")
