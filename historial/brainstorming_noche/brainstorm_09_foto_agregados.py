"""Idea 17 del brainstorming (docs/BRAINSTORM_NOCHE.md), sobre la referencia vigente (escalón 3, 0,6663):

* W/L de la foto 1 (dinámico causal, como `brainstorm_g_01_foto1_wl`) agregados por equipo con
  máximo, mínimo y desvío del winrate y del log de partidas, además de la media (que ya está).
* Por línea (5 × 2) en vez de solo la media.
* C de la logística (0,02 / 0,1) con las 157 columnas extra de la referencia.
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.brainstorm_base import REPO, ajustar_predecir, cargar, comparar, evaluar, extras_referencia, mostrar  # noqa: E402

D = cargar()
NM, ms, post = D["NM"], D["master"], D["post"]
REF = extras_referencia(D)

with sqlite3.connect(f"file:{REPO / 'data' / 'lol_las.db'}?mode=ro&immutable=1", uri=True) as con:
    foto = pd.read_sql_query("SELECT m.player_id, p.wins, p.losses FROM players p JOIN player_id_map m ON p.puuid = m.puuid",
                             con).drop_duplicates("player_id").set_index("player_id")
pl = D["B"]["player"].reshape(-1)
pr = np.repeat(post, 10)
al = foto.reindex(pl)
has = al.wins.notna().to_numpy() & pr
W0, L0 = al.wins.fillna(0).to_numpy(float), al.losses.fillna(0).to_numpy(float)
# partidas observadas desde la foto, solo anteriores (igual que Gemini)
win_slot = np.where(np.tile(np.r_[np.ones(5), np.zeros(5)].astype(bool), NM), np.repeat(D["y"], 10), 1 - np.repeat(D["y"], 10))
d = pd.DataFrame({"p": pl, "t": np.repeat(D["t"], 10), "m": np.repeat(np.arange(NM), 10), "w": win_slot, "pos": np.arange(NM * 10)})
s = d[pr].sort_values(["t", "m"], kind="mergesort")
dw = np.zeros(NM * 10)
dn = np.zeros(NM * 10)
dw[s.pos] = (s.groupby("p").w.cumsum() - s.w).to_numpy()
dn[s.pos] = s.groupby("p").cumcount().to_numpy()
W, L = W0 + dw, L0 + (dn - dw)
wr = np.where(has, (W + 10) / (W + L + 20) - 0.5, 0.0).reshape(NM, 2, 5)
ln = np.where(has, np.log1p(W + L) - 5.3, 0.0).reshape(NM, 2, 5)


def ag(a):
    return np.c_[a[:, 0].max(1) - a[:, 1].max(1), a[:, 0].min(1) - a[:, 1].min(1), a[:, 0].std(1) - a[:, 1].std(1)]


AGR = np.c_[ag(wr), ag(ln)].astype(np.float32)
LIN = np.c_[(wr[:, 0] - wr[:, 1]), (ln[:, 0] - ln[:, 1])].astype(np.float32)


def m(extra=None, C=0.05):
    X = REF if extra is None else np.c_[REF, extra]
    return lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=X, extra_post=False, C=C)


VAR = {"referencia (escalón 3)": m(), "+ W/L foto máx/mín/desvío": m(AGR), "+ W/L foto por línea": m(LIN),
       "C = 0,02": m(C=0.02), "C = 0,1": m(C=0.1)}
P = dict(Parallel(n_jobs=2, backend="threading")(delayed(lambda n: (n, evaluar(D, VAR[n])))(n) for n in VAR))
mostrar(comparar(D, P, "referencia (escalón 3)"))
