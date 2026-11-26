"""Idea 29 del brainstorming (docs/BRAINSTORM_NOCHE.md): draft jerárquico, sobre el escalón 4b.

El one-hot campeón × línea (851 columnas) se aprende con pocas partidas por par. Se suma un one-hot de campeón sin línea
(+1 azul / −1 rojo), para que las líneas compartan la fuerza del campeón (la L2 reparte entre los dos niveles)."""
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.brainstorm_base import ajustar_predecir, cargar, comparar, evaluar, extras_referencia, mostrar  # noqa: E402

D = cargar()
NM, ms = D["NM"], D["master"]
R = extras_referencia(D, escalon=4, rol_wr=True)
ids = D["Z"]["champ"].reshape(NM, 2, 5)
u, inv = np.unique(ids, return_inverse=True)
inv = inv.reshape(NM, 2, 5)
rows = np.repeat(np.arange(NM), 10)
vals = np.tile(np.r_[np.ones(5), -np.ones(5)], NM)
CH = sparse.csr_matrix((vals, (rows, inv.reshape(-1))), shape=(NM, len(u)))
D2 = dict(D, DRAFT=sparse.hstack([D["DRAFT"], CH]).tocsr())
m = lambda DD: (lambda tr, te: ajustar_predecir(DD, tr[ms[tr]], te, extra=R, extra_post=False))  # noqa: E731
VAR = {"escalón 4b": m(D), "+ campeón sin línea": m(D2)}
P = dict(Parallel(n_jobs=2, backend="threading")(delayed(lambda n: (n, evaluar(D, VAR[n])))(n) for n in VAR))
mostrar(comparar(D, P, "escalón 4b"))
