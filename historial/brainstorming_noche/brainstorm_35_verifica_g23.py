"""Brainstorming (docs/BRAINSTORM_NOCHE.md): verificación (Claude) de g_23 de Gemini (MATCHUP_EXP_MARGIN, margen esperado por par de campeones,
caché `gemini_g23.npz`) sobre el escalón 7: reproducción, placebo y Diamond-V."""
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.brainstorm_base import REPO, ajustar_predecir, cargar, comparar, evaluar, extras_referencia, mostrar  # noqa: E402

D = cargar()
NM, ms = D["NM"], D["master"]
X6 = extras_referencia(D, escalon=7)
z = np.load(REPO / "data" / "analysis" / "brainstorm" / "gemini_g23.npz")
H = z["MATCHUP_EXP_MARGIN"]
PL = H[np.random.default_rng(0).permutation(NM)]
m = lambda X: (lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=X.astype(np.float32), extra_post=False, con_mapa=False))  # noqa: E731
VAR = {"escalón 7": m(X6), "+ MATCHUP_EXP_MARGIN": m(np.c_[X6, H]), "placebo": m(np.c_[X6, PL])}
P = dict(Parallel(n_jobs=2, backend="threading")(delayed(lambda n: (n, evaluar(D, VAR[n])))(n) for n in VAR))
mostrar(comparar(D, P, "escalón 7"))
di = np.flatnonzero(D["diam_v"]); y = D["y"][di]
B = np.random.default_rng(1).integers(0, len(di), size=(300, len(di)))
for n in ("+ MATCHUP_EXP_MARGIN", "placebo"):
    p, q = P[n][0][di], P["escalón 7"][0][di]
    d = np.array([roc_auc_score(y[b], p[b]) - roc_auc_score(y[b], q[b]) for b in B])
    print(f"Diamond-V {n}: Δ {roc_auc_score(y, p) - roc_auc_score(y, q):+.4f} [{np.percentile(d, 2.5):+.4f}; {np.percentile(d, 97.5):+.4f}]")
