"""MMR propio (§9.S.31): nivel del lobby previo y margen ajustado por el rival, sobre la mejor
logística de §9.S.29 (0,6446 en V). Pedido del usuario (2026-11-13), en vez de scrapear un
medidor de MMR: esas páginas estiman el MMR con el nivel de los lobbies, que ya tenemos.

  nivel (foto)     x = LP absoluto de la foto del padrón (04/09) de cada jugador; el nivel de
                   una partida para p es la media de x de los otros 9, y su MMR previo es la
                   media de eso en sus partidas anteriores (src/features/lobby.py).
                   OJO: para partidas anteriores a la foto, x es del futuro (de los otros, no
                   del jugador ni de la partida que se predice; V es toda posterior a la foto).
  nivel (rating)   lo mismo con x = mu_pre de TrueSkill (data/rating_partida.csv, §9.S.21):
                   solo información previa a cada partida.
  margen vs rival  estadísticas contra el rival de línea, sumándole el margen previo del rival
                   (src/features/margin.py: prior_margin_vs_rival).

Además escribe data/analysis/mmr_propio_jugadores.csv: el MMR propio de cada jugador al final
del train, con su LP en las dos fotos, para compararlo con páginas externas.
Solo data/splits/train."""
import argparse, pickle, sqlite3, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.features.elo_reverso import cargar_foto, FOTO2
from src.features.padron import verificar_foto1  # noqa: E402
from src.features.fingerprint import add_rates
from src.features.lobby import prior_lobby_level
from src.features.margin import lane_diffs, lane_scale, prior_margin_vs_rival

ap = argparse.ArgumentParser()
ap.add_argument("--feats", default=str(REPO / "data/analysis/ganador_feats_grande.pkl"))
ap.add_argument("--margen", default=str(REPO / "data/analysis/ganador_margen_duos.pkl"))
ap.add_argument("--grupos", default=str(REPO / "data/analysis/ganador_margen_grupos.pkl"))
ap.add_argument("--salida", default=str(REPO / "data/analysis/ganador_mmr.pkl"))
args = ap.parse_args()

LANES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
RIVAL = ["d_gold_10", "d_gold_15", "d_xp_15", "d_cs_15"]
t0 = time.time()
db = f"file:{(REPO / 'data/lol_las.db').as_posix()}?mode=ro"
foto1 = cargar_foto(db, "SELECT m.player_id, p.tier, p.rank, p.league_points, p.wins, p.losses "
                        "FROM players p JOIN player_id_map m ON m.puuid = p.puuid")
foto2 = cargar_foto(db, "SELECT m.player_id, s.tier, s.rank, s.league_points, s.wins, s.losses "
                        "FROM ladder_snapshots s JOIN player_id_map m ON m.puuid = s.puuid "
                        "WHERE s.snapshot_at = ?", (FOTO2,))
with sqlite3.connect(db, uri=True) as con:
    t1 = verificar_foto1(con)

df = pd.read_csv(REPO / "data/splits/train/full_dataset.csv")
df = df[df.lane.isin(LANES)]
df = df[df.groupby(["match_id", "team_id"]).lane.transform("nunique") == 5]
df = df[df.groupby("match_id").lane.transform("size") == 10].copy()
df = add_rates(df).sort_values(["game_creation", "match_id", "team_id", "lane"]).reset_index(drop=True)
df = lane_diffs(df, ["gold_10", "gold_15", "xp_15", "cs_15"])
df["lp_foto"] = df.player_id.map(foto1.z)
rt = pd.read_csv(REPO / "data/rating_partida.csv", usecols=["match_id", "player_id", "mu_pre"])
df["mu_pre"] = df.merge(rt.drop_duplicates(["match_id", "player_id"]), on=["match_id", "player_id"], how="left").mu_pre.to_numpy()
del rt
print(f"con LP de la foto: {df.lp_foto.notna().mean():.1%} de las filas; con mu_pre: {df.mu_pre.notna().mean():.1%}", flush=True)

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

# ---------- nivel del lobby previo (mu0 = media del nivel en las filas de ajuste)
N = {}
for nombre, col, escala in (("foto", "lp_foto", 100.0), ("rating", "mu_pre", 100.0)):
    mu0 = float(prior_lobby_level(df[fila_fit], col).nivel_lobby.mean())
    N[nombre] = prior_lobby_level(df, col, k=5, mu0=mu0, window=10)
    n = N[nombre]
    print(f"nivel ({nombre}): mu0 = {mu0:.0f}; lobby con nivel conocido {n.nivel_lobby.notna().mean():.1%}; "
          f"desvío entre filas de nivel_acum {n.nivel_acum.std():.0f}, dentro de la partida "
          f"{(n.nivel_acum - n.nivel_acum.groupby(df.match_id).transform('mean')).std():.0f}", flush=True)
R = prior_margin_vs_rival(df, RIVAL, escala=lane_scale(df, RIVAL))
print(f"medidas ({time.time() - t0:.0f}s)", flush=True)

def por_linea(X):
    k = X.shape[1]
    out = np.zeros((len(mids), 6 * k))
    for li in range(5):
        sel = lane_i == li
        np.add.at(out[:, li * k:(li + 1) * k], row[sel], X[sel] * s[sel, None])
    np.add.at(out[:, 5 * k:], row, X * s[:, None])
    return out

J = {"nivel (foto)": N["foto"][["nivel_acum", "nivel_v10"]].to_numpy() / 100,
     "nivel (rating)": N["rating"][["nivel_acum", "nivel_v10"]].to_numpy() / 100,
     "margen vs rival": R.to_numpy()}
NUEVO = {k: por_linea(v) for k, v in J.items()}
puesto = 5 * (s == -1).astype(int) + lane_i
jug = np.zeros((len(mids), 10, sum(v.shape[1] for v in J.values())), dtype=np.float32)
jug[row, puesto] = np.hstack(list(J.values()))
pickle.dump({"mids": mids, "nuevo": NUEVO, "jug": jug}, open(args.salida, "wb"), protocol=4)

# ---------- el MMR propio por jugador, al final del train, contra su LP real
g = df.assign(L=N["foto"].nivel_lobby).dropna(subset=["L"]).groupby("player_id")
tab = pd.DataFrame({"partidas_train": g.size(), "mmr_propio": g.L.mean(),
                    "mmr_propio_ult10": g.L.apply(lambda x: x.tail(10).mean()),
                    "ultima_partida": pd.to_datetime(g.game_creation.max(), unit="ms")})
tab["lp_foto1"] = foto1.z.reindex(tab.index)
tab["lp_foto2"] = foto2.z.reindex(tab.index)
tab.round(1).to_csv(REPO / "data/analysis/mmr_propio_jugadores.csv")
m = tab[tab.partidas_train >= 20]
print(f"\nMMR propio (media del LP de la foto 1 de los otros 9) contra el LP real, jugadores con >= 20 partidas ({len(m)}):")
for c in ("lp_foto1", "lp_foto2"):
    ok = m[c].notna()
    print(f"  {c}: n = {ok.sum()}, Spearman {spearmanr(m.mmr_propio[ok], m[c][ok])[0]:.3f} (últimas 10: "
          f"{spearmanr(m.mmr_propio_ult10[ok], m[c][ok])[0]:.3f}), MAE {np.abs(m.mmr_propio[ok] - m[c][ok]).mean():.0f} LP")
ok = m.lp_foto1.notna() & m.lp_foto2.notna()
print(f"  referencia, LP foto 1 contra LP foto 2: Spearman {spearmanr(m.lp_foto1[ok], m.lp_foto2[ok])[0]:.3f}")

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

print("\n=== Config C con margen, duos y margen por grupo ===")
ref = ajustar(base_X); informe("base", ref)
for k in NUEVO:
    informe(f"base + {k}", ajustar(np.hstack([base_X, NUEVO[k]])), ref)
informe("base + rating + rival", ajustar(np.hstack([base_X, NUEVO["nivel (rating)"], NUEVO["margen vs rival"]])), ref)
informe("base + todo", ajustar(np.hstack([base_X] + list(NUEVO.values()))), ref)
# el nivel de la foto, ajustando solo con partidas posteriores a la foto (sin LP del futuro)
t = pd.Series(df.game_creation.to_numpy(), index=df.match_id).groupby(level=0).first().reindex(mids).to_numpy()
fit_post = fit & (t >= t1)
def ajustar_post(X):
    sc = StandardScaler(with_mean=False).fit(X[fit_post])
    return LogisticRegression(C=0.05, max_iter=5000).fit(sc.transform(X[fit_post]), y[fit_post]).predict_proba(sc.transform(X[val]))[:, 1]
print(f"\nAjustando solo con partidas posteriores a la foto ({fit_post.sum()}):")
ref_post = ajustar_post(base_X); informe("base", ref_post)
informe("base + nivel (foto)", ajustar_post(np.hstack([base_X, NUEVO["nivel (foto)"]])), ref_post)
print("\nLas medidas solas (sin la base):")
for k in NUEVO:
    informe(k, ajustar(NUEVO[k]))
print(f"total {time.time() - t0:.0f}s")
