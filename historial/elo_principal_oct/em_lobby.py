"""Iteración tipo EM sobre los lobbies (una vuelta), sin fuga.

Para el fold externo k: el camino de LP de cada jugador del desarrollo se corrige con la predicción anidada
A[k] (que nunca vio el fold k): LP_nuevo(partida) = LP_regla(partida) + Δ·(posición desde la más nueva)/n,
con Δ = z1 predicho − regla. Se recalculan los lobbies (LP medio de los otros 9) y las variables del lobby,
y se entrena el modelo del zoo con folds ≠ k. Uso: python em_lobby.py anid_cat_huber cat_huber
"""
import pickle, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from comun import AQUI, cargar, evaluar, registrar
from zoo import MODELOS, variables
ROOT = Path(r"c:/Users/DarkA/OneDrive/Escritorio/Facultad/3er año/ML/TPF/ProyectoFinalML")
sys.path.insert(0, str(ROOT))
from src.features.elo_reverso import lobby_por_fila, variables_estaticas  # noqa
anid_tag, modelo = sys.argv[1], sys.argv[2]
c = cargar(); e, partes = c["e"], c["partes"]; S = pickle.load(open(AQUI / "seq.pkl", "rb"))
A = np.load(AQUI / "preds" / f"{anid_tag}.npy")
dev, fold = ~c["caja"], c["fold"]
X0 = variables(c, S); y = e["z1"] - c["regla"]
# Posición de cada partida desde la más nueva (1 = la más nueva) y cantidad de partidas por jugador.
ps = partes.sort_values(["player_id", "game_creation"], ascending=[True, False])
posicion = ps.groupby("player_id").cumcount().add(1).reindex(partes.index)
n_part = partes.groupby("player_id")["match_id"].transform("size")
frac = (posicion / n_part).to_numpy()
pos_e = pd.Series(np.arange(len(e)), index=e.index)
idx_e = partes["player_id"].map(pos_e)
t0 = time.time()
pred = pd.Series(np.nan, index=e.index)
cols = ["lobby", "lobby_viejo", "lobby_reciente", "gap"]
for k in range(5):
    delta = np.where(dev, A[k] * 100, 0.0)
    d_fila = np.where(idx_e.notna(), delta[idx_e.fillna(0).astype(int).to_numpy()], 0.0)
    lp_nuevo = c["lp_fila"].reindex(partes.index) + d_fila * frac
    lob = lobby_por_fila(partes, lp_nuevo, por_fila=True)
    V = variables_estaticas(e, partes, e["z2"], c["regla"], lob=lob)
    X = X0.copy()
    for col in cols:
        if col in X.columns:
            X[col] = V[col]
    if "--rel" in sys.argv:
        X["lobby_rel_pred"] = V["lobby"] - (c["regla"] + A[k] * 100)       # lobby contra el LP predicho
    tr, te = dev & (fold != k), dev & (fold == k)
    pred[te] = MODELOS[modelo][0]().fit(X[tr], y[tr]).predict(X[te])
    print(f"fold {k} ({time.time() - t0:.0f} s)", flush=True)
final = c["regla"] + pred
tag = f"{modelo}+em_{anid_tag}" + ("_rel" if "--rel" in sys.argv else "")
np.save(AQUI / "preds" / f"{tag}.npy", final.to_numpy())
registrar(tag, evaluar(e, final, e.index[dev]), f"{modelo} con lobbies recalculados con {anid_tag} (EM, 1 vuelta) ({time.time() - t0:.0f} s)")
