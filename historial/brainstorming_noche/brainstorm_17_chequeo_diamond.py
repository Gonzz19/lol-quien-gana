"""Brainstorming (docs/BRAINSTORM_NOCHE.md): chequeo honesto de la escalera de la noche en datos que NO se
usaron para decidir nada: las 10.506 partidas de Diamond de la ventana de V (16/09 → 27/09).

Todas las decisiones de la noche miraron V Master+ y los sub-bloques solo en Master+. Acá cada escalón
predice las partidas de Diamond de la ventana con el modelo entrenado antes del 16/09 (como V), y se mide
el Δ de cada escalón contra el anterior, con IC por bootstrap pareado. Si la mejora fuera selección sobre
V Master+, no tendría por qué aparecer acá. (Población distinta: el entrenamiento solo con Master+ puede
perjudicar a Diamond; los demás escalones agregan información.)
"""
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.brainstorm_base import REPO, ajustar_predecir, cargar, extras_referencia  # noqa: E402

D = cargar()
NM, ms = D["NM"], D["master"]
R4 = extras_referencia(D, escalon=4)
n_mcml = 150
MCML, NIV, W1, MAT = R4[:, :n_mcml], R4[:, n_mcml:n_mcml + 3], R4[:, n_mcml + 3:n_mcml + 7], R4[:, n_mcml + 7:]
assert MAT.shape[1] == 10 and W1.shape[1] == 4
di = np.flatnonzero(D["diam_v"])
tr = D["fi_v"]


def p(solo_master, *bl):
    X = np.hstack(bl).astype(np.float32) if bl else None
    t = tr[ms[tr]] if solo_master else tr
    return ajustar_predecir(D, t, di, extra=X, extra_post=False)


ESC = [("0. híbrido + mapa, todo el entrenamiento", (False,)),
       ("1. solo Master+", (True,)),
       ("1b. + margen campeón/línea (entrenando con todo)", (False, MCML)),
       ("2. + margen campeón/línea", (True, MCML)),
       ("3. + nivel histórico", (True, MCML, NIV)),
       ("4. + W/L foto 1", (True, MCML, NIV, W1)),
       ("5. + matchups", (True, MCML, NIV, W1, MAT))]
P = dict(Parallel(n_jobs=2, backend="threading")(delayed(lambda n, a: (n, p(*a)))(n, a) for n, a in ESC))
y = D["y"][di]
B = np.random.default_rng(1).integers(0, len(di), size=(300, len(di)))
print(f"Diamond de la ventana de V: {len(di):,} partidas (nunca usadas para decidir)")
prev = None
for n, _ in ESC:
    pp = P[n]
    s = f"{n:50s} AUC {roc_auc_score(y, pp):.4f}  acierto {((pp > .5) == y).mean():.1%}"
    ref = {"1b. + margen campeón/línea (entrenando con todo)": "0. híbrido + mapa, todo el entrenamiento"}.get(n, prev)
    if ref is not None:
        q = P[ref]
        d = np.array([roc_auc_score(y[b], pp[b]) - roc_auc_score(y[b], q[b]) for b in B])
        s += f"   Δ vs «{ref[:2]}» {roc_auc_score(y, pp) - roc_auc_score(y, q):+.4f} [{np.percentile(d, 2.5):+.4f}; {np.percentile(d, 97.5):+.4f}]"
    print(s)
    if not n.startswith("1b"):
        prev = n
