"""Brainstorming (docs/BRAINSTORM_NOCHE.md): seguimiento del diagnóstico g_39 (el escalón 9 queda algo sobreconfiado en los
extremos en A2-A4 y C menor mejora los bloques). Se mide C = 0,02 y 0,01 en el escalón 9 sin mapa: bloques, V y Diamond-V."""
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.brainstorm_base import ajustar_predecir, cargar, comparar, evaluar, extras_referencia, mostrar  # noqa: E402

D = cargar()
ms = D["master"]
X = extras_referencia(D, escalon=9)
m = lambda C: (lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=X, extra_post=False, con_mapa=False, C=C))  # noqa: E731
VAR = {"escalón 9 (C = 0,05)": m(0.05), "C = 0,02": m(0.02), "C = 0,01": m(0.01)}
P = dict(Parallel(n_jobs=2, backend="threading")(delayed(lambda n: (n, evaluar(D, VAR[n])))(n) for n in VAR))
mostrar(comparar(D, P, "escalón 9 (C = 0,05)"))
yv = D["y"][D["val"]]
di = np.flatnonzero(D["diam_v"]); yd = D["y"][di]
B = np.random.default_rng(1).integers(0, len(di), size=(300, len(di)))
q = P["escalón 9 (C = 0,05)"][0][di]
for n, (po, pv) in P.items():
    p = po[di]
    d = np.array([roc_auc_score(yd[b], p[b]) - roc_auc_score(yd[b], q[b]) for b in B])
    print(f"{n:22s} log loss V {log_loss(yv, pv):.5f} | Diamond-V Δ {roc_auc_score(yd, p) - roc_auc_score(yd, q):+.4f} [{np.percentile(d, 2.5):+.4f}; {np.percentile(d, 97.5):+.4f}]")
