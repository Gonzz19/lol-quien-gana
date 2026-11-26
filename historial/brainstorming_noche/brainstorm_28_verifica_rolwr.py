"""Brainstorming (docs/BRAINSTORM_NOCHE.md): verificación (Claude) de ROL_WR de Gemini (brainstorm_g_10) sobre el escalón 4:
reproducción en V Master+ y bloques, placebo (ROL_WR de otra partida al azar) y chequeo en Diamond-V (nunca usado)."""
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.brainstorm_base import REPO, ajustar_predecir, cargar, comparar, evaluar, extras_referencia, mostrar  # noqa: E402

D = cargar()
NM, ms = D["NM"], D["master"]
R4 = extras_referencia(D, escalon=4)
RW = np.load(REPO / "data" / "analysis" / "brainstorm" / "gemini_nuevas_v2.npz")["ROL_WR"]
PL = RW[np.random.default_rng(0).permutation(NM)]
m = lambda X: (lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=X, extra_post=False))  # noqa: E731
VAR = {"escalón 4": m(R4), "+ ROL_WR": m(np.c_[R4, RW]), "placebo": m(np.c_[R4, PL])}
P = dict(Parallel(n_jobs=2, backend="threading")(delayed(lambda n: (n, evaluar(D, VAR[n])))(n) for n in VAR))
mostrar(comparar(D, P, "escalón 4"))
di = np.flatnonzero(D["diam_v"])
y = D["y"][di]
B = np.random.default_rng(1).integers(0, len(di), size=(300, len(di)))
b0 = P["escalón 4"][0][di]
for n in ("+ ROL_WR", "placebo"):
    p = P[n][0][di]
    d = np.array([roc_auc_score(y[b], p[b]) - roc_auc_score(y[b], b0[b]) for b in B])
    print(f"Diamond-V {n}: Δ {roc_auc_score(y, p) - roc_auc_score(y, b0):+.4f} [{np.percentile(d, 2.5):+.4f}; {np.percentile(d, 97.5):+.4f}]")
