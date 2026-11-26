"""Modelo de ataque y defensa por jugador (Poisson) contra el margen previo (§9.S.25).

Tres medidas del equipo, cada una con su ataque y su defensa por jugador
(src/features/attack_defense.py): kills (Poisson, con los minutos como exposición), oro por
minuto y daño por minuto (normales, estandarizados). Los parámetros de cada partida salen de
un ajuste con los bloques temporales ANTERIORES (--bloques, por cuantiles de fecha).
Se suman a la logística de la config C (todo el train antes de V, sin huella) y se comparan
con el margen previo de ganador_margen_duos.py, en total y según la cobertura del lobby.
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
from src.features.attack_defense import block_scores

ap = argparse.ArgumentParser()
ap.add_argument("--feats", default=str(REPO / "data" / "analysis" / "ganador_feats_grande.pkl"))
ap.add_argument("--margen", default=str(REPO / "data" / "analysis" / "ganador_margen_duos.pkl"))
ap.add_argument("--salida", default=str(REPO / "data" / "analysis" / "ganador_poisson.pkl"))
ap.add_argument("--bloques", type=int, default=40)
args = ap.parse_args()

LANES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
d = pickle.load(open(args.feats, "rb"))
feat, jugf, mids, y, t, val, masks = d["feat"], d["jug"], d["mids"], d["y"], d["t"], d["val"], d["masks"]
M = pickle.load(open(args.margen, "rb"))
assert (M["mids"] == mids).all()

df = pd.read_csv(REPO / "data" / "splits" / "train" / "full_dataset.csv",
                 usecols=["match_id", "team_id", "lane", "player_id", "champion_id", "kills", "gold_earned",
                          "total_damage_dealt_to_champions", "game_duration"])
row = pd.Series(np.arange(len(mids)), index=mids).reindex(df.match_id).to_numpy()
df = df[~np.isnan(row)].copy()
row = row[~np.isnan(row)].astype(int)
puesto = (5 * (df.team_id == 200) + df.lane.map({l: i for i, l in enumerate(LANES)})).to_numpy()
n = len(mids)
jug = np.zeros((n, 10), int); jug[row, puesto] = df.player_id.astype("category").cat.codes.to_numpy()
cl = np.zeros((n, 10), int); cl[row, puesto] = (df.champion_id.astype(str) + df.lane).astype("category").cat.codes.to_numpy()
minutos = np.zeros(n); minutos[row] = df.game_duration.to_numpy() / 60

def equipo(col):
    out = np.zeros((n, 2)); np.add.at(out, (row, puesto // 5), df[col].to_numpy(float))
    return out
kills, oro, dano = equipo("kills"), equipo("gold_earned") / minutos[:, None], equipo("total_damage_dealt_to_champions") / minutos[:, None]
z = lambda a: (a - a.mean()) / a.std()                 # escala global (constantes, sin resultado)
bloque = np.searchsorted(np.quantile(t, np.linspace(0, 1, args.bloques + 1)[1:-1]), t, side="right")
print(f"{n} partidas, {jug.max() + 1} jugadores, {args.bloques} bloques; V cae en los bloques "
      f"{bloque[val].min()}-{bloque[val].max()}", flush=True)

MEDIDAS = {"kills": (kills, "poisson", (50.0, 300.0)), "oro": (z(oro), "gauss", (5.0, 25.0)),
           "daño": (z(dano), "gauss", (5.0, 25.0))}
P, PJ = {}, {}
for nombre, (v, familia, lams) in MEDIDAS.items():
    for lam in lams:
        t0 = time.time()
        s = block_scores(jug, cl, v[:, 0], v[:, 1], bloque, lam, 1.0, familia, minutos if familia == "poisson" else None)
        P[f"{nombre} λ={lam:g}"] = np.c_[s["ataque"], s["defensa"], s["eta"]]
        PJ[f"{nombre} λ={lam:g}"] = np.stack([s["ataque_jug"], s["defensa_jug"]], axis=2).astype(np.float32)
        print(f"  {nombre} λ={lam:g}: {time.time() - t0:.0f}s", flush=True)
# "jug": ataque y defensa de cada jugador, (partida, puesto, 2) por medida, para la red por jugador
pickle.dump({"mids": mids, "poisson": P, "jug": PJ, "bloque": bloque}, open(args.salida, "wb"), protocol=4)

# ---------- evaluación (config C)
fit, yv = masks["C"], y[val]
B = np.random.default_rng(0).integers(0, val.sum(), (200, val.sum()))
base_X = np.hstack([feat[k] for k in ["onehot", "experiencia", "cobertura", "dificultad", "matchups", "sinergias"]])
margen_X = np.hstack([M["nuevo"][k] for k in ("winrate previo", "margen acum", "margen v20", "duos")])

def ajustar(X):
    sc = StandardScaler(with_mean=False).fit(X[fit])
    return LogisticRegression(C=0.05, max_iter=5000).fit(sc.transform(X[fit]), y[fit]).predict_proba(sc.transform(X[val]))[:, 1]

def informe(nombre, p, ref=None, sel=None):
    ii = np.arange(len(yv)) if sel is None else np.where(sel)[0]
    extra = ""
    if ref is not None:
        bs = B if sel is None else [np.random.default_rng(k).choice(ii, len(ii)) for k in range(200)]
        dif = np.array([roc_auc_score(yv[b], p[b]) - roc_auc_score(yv[b], ref[b]) for b in bs])
        extra = f"  dif {roc_auc_score(yv[ii], p[ii]) - roc_auc_score(yv[ii], ref[ii]):+.4f} [{np.quantile(dif, .025):+.4f};{np.quantile(dif, .975):+.4f}]"
    print(f"  {nombre:40s} acc {accuracy_score(yv[ii], p[ii] > .5):.3f}  AUC {roc_auc_score(yv[ii], p[ii]):.4f}  "
          f"logloss {log_loss(yv[ii], p[ii]):.4f}{extra}", flush=True)

print("\n=== Config C: contra la base ===")
base = ajustar(base_X); informe("base", base)
for k in P:
    informe(f"base + {k}", ajustar(np.hstack([base_X, P[k]])), base)
mejores = [max((k for k in P if k.startswith(m)), key=lambda k: roc_auc_score(yv, ajustar(np.hstack([base_X, P[k]]))))
           for m in MEDIDAS]      # se elige λ mirando V: solo para describir, ver la nota del informe
todos = np.hstack([P[k] for k in P])
p_poi = ajustar(np.hstack([base_X, todos])); informe("base + las 3 medidas (todos los λ)", p_poi, base)
p_mar = ajustar(np.hstack([base_X, margen_X])); informe("base + margen + duos", p_mar, base)

print("\n=== Config C: contra base + margen + duos ===")
p_tod = ajustar(np.hstack([base_X, margen_X, todos])); informe("base + margen + duos + 3 medidas", p_tod, p_mar)
informe("solo las 3 medidas + margen (sin base)", ajustar(np.hstack([margen_X, todos])), p_mar)

num = jugf["num"][val]
cob = np.expm1(num[:, :, 6].min(1))
q = np.quantile(cob, [1 / 3, 2 / 3])
print("\n=== Por cobertura mínima del lobby (dif contra base + margen + duos) ===")
for nombre, sel in ((f"< {q[0]:.0f} partidas", cob < q[0]), (f"{q[0]:.0f}-{q[1]:.0f}", (cob >= q[0]) & (cob < q[1])),
                    (f">= {q[1]:.0f}", cob >= q[1])):
    print(f" cobertura {nombre} (n = {sel.sum()})")
    informe("base + margen + duos", p_mar, None, sel)
    informe("base + 3 medidas", p_poi, p_mar, sel)
    informe("base + margen + duos + 3 medidas", p_tod, p_mar, sel)
