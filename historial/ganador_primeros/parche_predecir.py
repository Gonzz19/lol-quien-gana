"""¿Se distingue el parche por el contenido de la partida, sin la fecha? (§9.S.28)

Idea del usuario: predecir el parche como información nueva. El parche es una función de la
fecha, así que la pregunta útil es qué parte del contenido lo delata, porque eso es lo que
cambia entre parches (el meta). Tres conjuntos de variables, por separado y juntos:
    campeones: cuántas veces aparece cada (campeón, línea) en la partida (los dos lados juntos);
               es lo único de los tres que se conoce ANTES de la partida;
    ítems:     cuántas veces aparece cada ítem en los inventarios finales;
    juego:     promedios de la partida (duración, kills, oro, daño, visión, @10/@15...).
Clasificador: logística multinomial; partición aleatoria estratificada 80/20 dentro de los
parches con >= MIN partidas (la pregunta es de distinguibilidad, no de pronóstico).
Después: qué campeones e ítems cambian más su tasa de elección entre parches seguidos.
Solo data/splits/train."""
import argparse, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--min", type=int, default=5000, help="partidas mínimas por parche")
ap.add_argument("--top", type=int, default=8)
args = ap.parse_args()

LANES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
ITEMS = [f"item{i}" for i in range(7)]
JUEGO = ["kills", "deaths", "assists", "total_damage_dealt_to_champions", "total_damage_taken", "gold_earned",
         "total_minions_killed", "neutral_minions_killed", "vision_score", "champ_level", "first_blood_kill",
         "first_tower_kill", "gold_10", "xp_10", "cs_10", "damage_10", "kills_10", "gold_15", "xp_15", "cs_15",
         "damage_15", "kills_15"]
df = pd.read_csv(REPO / "data/splits/train/full_dataset.csv",
                 usecols=["match_id", "game_duration", "game_version", "champion_id", "champion_name", "lane"] + ITEMS + JUEGO)
df["parche"] = df.game_version.str.split(".").str[:2].str.join(".")
cuenta = df.groupby("parche").match_id.nunique()
PARCHES = sorted(cuenta[cuenta >= args.min].index, key=lambda v: [int(x) for x in v.split(".")])
df = df[df.parche.isin(PARCHES) & df.lane.isin(LANES)]
mids = pd.Index(df.match_id.unique())
row = mids.get_indexer(df.match_id)
y = pd.Categorical(df.groupby("match_id").parche.first().reindex(mids), categories=PARCHES).codes
print(f"{len(mids)} partidas en {len(PARCHES)} parches: " + ", ".join(f"{p} {cuenta[p]}" for p in PARCHES))

def bolsa(cod, n_cols):
    return sparse.csr_matrix((np.ones(len(cod)), (row, cod)), shape=(len(mids), n_cols))

cl = pd.Categorical(df.champion_name + "|" + df.lane)
X_camp = bolsa(cl.codes, len(cl.categories))
it = df[ITEMS].to_numpy().ravel()
r_it = np.repeat(row, len(ITEMS))
ok = it > 0
it_cat = pd.Categorical(it[ok])
X_item = sparse.csr_matrix((np.ones(ok.sum()), (r_it[ok], it_cat.codes)), shape=(len(mids), len(it_cat.categories)))
g = df.assign(**{c: df[c] / (df.game_duration / 60) for c in JUEGO if c not in ("champ_level", "first_blood_kill", "first_tower_kill") and not c[-3:] in ("_10", "_15")})
X_juego = g.groupby("match_id")[JUEGO].mean().reindex(mids).fillna(0)
X_juego["duracion"] = df.groupby("match_id").game_duration.first().reindex(mids) / 60
X_juego = sparse.csr_matrix(StandardScaler().fit_transform(X_juego.to_numpy()))

tr, te = train_test_split(np.arange(len(mids)), test_size=0.2, stratify=y, random_state=0)
mayor = np.bincount(y[tr]).argmax()
print(f"\n=== Accuracy del parche (azar: clase mayoritaria {np.mean(y[te] == mayor):.3f}, balanceada {1 / len(PARCHES):.3f}) ===")
modelos = {}
for nombre, X in (("campeones (antes de la partida)", X_camp), ("ítems", X_item), ("juego", X_juego),
                  ("campeones + ítems", sparse.hstack([X_camp, X_item]).tocsr()),
                  ("todo", sparse.hstack([X_camp, X_item, X_juego]).tocsr())):
    X = MaxAbsScaler().fit_transform(X)
    m = LogisticRegression(C=1.0, max_iter=3000).fit(X[tr], y[tr])
    p = m.predict(X[te])
    modelos[nombre] = (m, p)
    vecino = np.mean(np.abs(p - y[te]) <= 1)
    print(f"  {nombre:32s} acc {accuracy_score(y[te], p):.3f}  balanceada {balanced_accuracy_score(y[te], p):.3f}  "
          f"a ±1 parche {vecino:.3f}", flush=True)

for nombre in ("campeones (antes de la partida)", "ítems"):
    print(f"\nMatriz de confusión, {nombre} (filas: parche real; % de la fila)")
    cm = confusion_matrix(y[te], modelos[nombre][1], normalize="true")
    print("        " + "".join(f"{p:>7s}" for p in PARCHES))
    for p, f in zip(PARCHES, cm):
        print(f"  {p:6s}" + "".join(f"{v:7.0%}" for v in f))

# ---------- qué cambia: tasa de elección por parche (por partida) y su salto entre parches seguidos
def movimientos(X, nombres, etiqueta):
    tasa = np.vstack([np.asarray(X[y == k].mean(0)).ravel() for k in range(len(PARCHES))])
    print(f"\n=== {etiqueta}: mayores cambios de tasa de aparición entre parches seguidos (por partida) ===")
    for k in range(1, len(PARCHES)):
        d = tasa[k] - tasa[k - 1]
        sube, baja = np.argsort(-d)[:args.top], np.argsort(d)[:args.top]
        print(f"  {PARCHES[k - 1]} -> {PARCHES[k]}  suben: " + ", ".join(f"{nombres[i]} {tasa[k - 1, i]:.2f}->{tasa[k, i]:.2f}" for i in sube))
        print(f"  {'':16s} bajan: " + ", ".join(f"{nombres[i]} {tasa[k - 1, i]:.2f}->{tasa[k, i]:.2f}" for i in baja))
    return tasa

movimientos(X_camp, list(cl.categories), "Campeones por línea")
movimientos(X_item, [str(int(v)) for v in it_cat.categories], "Ítems (id)")
