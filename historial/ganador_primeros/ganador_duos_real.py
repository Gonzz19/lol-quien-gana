"""Duos reales contra coincidencias, como variables de quién gana (§9.S.29).

Para cada par del mismo equipo, solo con partidas ANTERIORES del train (todas las de 10
jugadores), se cuenta: pj (veces juntos), pc (veces enfrentados), gj (victorias juntos) y si
la partida anterior de los dos fue la misma y del mismo lado ("seguida") y cómo terminó.
Con el emparejamiento por roles, una coincidencia cae del mismo lado con P = 1/2 (§9.S.27),
así que pj >= 5 sin ningún cruce en contra es un duo casi seguro (1/32 por azar), y pj 3-4 es
dudoso. Variables por equipo, como azul - rojo y separadas Diamond / Master+:
    seguro    pares con pj >= 5 y pc = 0
    dudoso    pares con pj en 3-4 y pc = 0
    seguida   pares que vienen de jugar juntos la partida anterior (duo o reencolada)
    tras_perder  seguida y esa partida la perdieron
    record    registro conjunto del duo seguro: (gj + 5) / (pj + 10) - 0,5
Se comparan con la variable vieja de §9.S.24 sobre la logística de la config C con margen
(0,636 en V). Escribe un pickle con los bloques para la red y el stacking.
Solo data/splits/train."""
import argparse, itertools, pickle, warnings
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
ap.add_argument("--salida", default=str(REPO / "data/analysis/ganador_duos_real.pkl"))
args = ap.parse_args()

t = pd.read_csv(REPO / "data/splits/train/full_dataset.csv", usecols=["match_id", "game_creation", "player_id", "team_id", "win"])
t = t[t.groupby("match_id").player_id.transform("size") == 10]
t = t.sort_values(["game_creation", "match_id", "team_id", "player_id"])
pid = t.player_id.astype("category").cat.codes.to_numpy().reshape(-1, 10)
win = t.win.to_numpy().reshape(-1, 10)
assert (t.team_id.to_numpy().reshape(-1, 10)[:, :5] == 100).all()
m_ids = t.match_id.to_numpy()[::10]
n = len(pid)
last, prev_m = {}, np.full((n, 10), -1)
for m in range(n):
    for s in range(10):
        prev_m[m, s] = last.get(pid[m, s], -1)
        last[pid[m, s]] = m

partes = []
for i, j in itertools.combinations(range(10), 2):
    partes.append(pd.DataFrame({"m": np.arange(n, dtype=np.int32), "i": np.int8(i), "j": np.int8(j),
                                "a": np.minimum(pid[:, i], pid[:, j]), "b": np.maximum(pid[:, i], pid[:, j])}))
P = pd.concat(partes, ignore_index=True).sort_values(["m", "i", "j"], kind="stable").reset_index(drop=True)
P["junto"] = ((P.i < 5) == (P.j < 5)).astype(np.int8)
P["gj_now"] = (P.junto * win[P.m, P.i]).astype(np.int8)          # ganaron juntos en esta partida
g = P.groupby(["a", "b"], sort=False)
P["pj"] = g.junto.cumsum() - P.junto
P["pc"] = g.cumcount() - P.pj
P["gj"] = g.gj_now.cumsum() - P.gj_now
P["m_prev"] = g.m.shift().fillna(-1).astype(int)
P["junto_prev"] = g.junto.shift().fillna(0).astype(int)
P["gano_prev"] = g.gj_now.shift().fillna(0).astype(int)
P = P[P.junto == 1]
seguida = (prev_m[P.m, P.i] == P.m_prev) & (prev_m[P.m, P.j] == P.m_prev) & (P.junto_prev == 1)
seguro = (P.pj >= 5) & (P.pc == 0)
lado = np.where(P.i < 5, 1.0, -1.0)
cols = {"seguro": seguro, "dudoso": P.pj.between(3, 4) & (P.pc == 0), "seguida": seguida,
        "tras_perder": seguida & (P.gano_prev == 0),
        "record": np.where(seguro, (P.gj + 5) / (P.pj + 10) - 0.5, 0.0)}
por_m = pd.DataFrame({k: np.asarray(v, float) * lado for k, v in cols.items()}).groupby(P.m.to_numpy()).sum()
por_m = por_m.reindex(np.arange(n), fill_value=0)
por_m.index = m_ids

d = pickle.load(open(args.feats, "rb"))
feat, mids, y, val, fit = d["feat"], d["mids"], d["y"], d["val"], d["masks"]["C"]
M = pickle.load(open(args.margen, "rb"))
assert (M["mids"] == mids).all()
V = por_m.reindex(mids).fillna(0)
tier = pd.read_csv(REPO / "data/splits/match_split.csv", usecols=["match_id", "tier"]).set_index("match_id").tier.reindex(mids).to_numpy()
dia = (tier == "DIAMOND")[:, None]
cruzar = lambda nombres: np.hstack([V[nombres].to_numpy() * dia, V[nombres].to_numpy() * ~dia])
BLOQUES = {"seguro + dudoso": cruzar(["seguro", "dudoso"]), "seguida + tras perder": cruzar(["seguida", "tras_perder"]),
           "record": cruzar(["record"])}
BLOQUES["todo"] = np.hstack(list(BLOQUES.values()))
pickle.dump({"mids": mids, "bloques": BLOQUES, "crudo": V}, open(args.salida, "wb"), protocol=4)
print("filas de pares juntos:", len(P), "| por partida (media de |azul - rojo|):",
      {k: round(float(np.abs(V[k]).mean()), 3) for k in V})

base_X = np.hstack([feat[k] for k in ["onehot", "experiencia", "cobertura", "dificultad", "matchups", "sinergias"]]
                   + [M["nuevo"][k] for k in ("winrate previo", "margen acum", "margen v20")])
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
    print(f"  {nombre:40s} acc {accuracy_score(yv, p > .5):.4f}  AUC {roc_auc_score(yv, p):.4f}  logloss {log_loss(yv, p):.4f}{extra}", flush=True)

print("\n=== Config C con margen (sin duos) ===")
base = ajustar(base_X); informe("base (margen, sin duos)", base)
informe("base + duos viejos (§9.S.24)", ajustar(np.hstack([base_X, M["nuevo"]["duos"]])), base)
for k, X in BLOQUES.items():
    informe(f"base + {k}", ajustar(np.hstack([base_X, X])), base)
informe("base + duos viejos + todo lo nuevo", ajustar(np.hstack([base_X, M["nuevo"]["duos"], BLOQUES["todo"]])), base)
