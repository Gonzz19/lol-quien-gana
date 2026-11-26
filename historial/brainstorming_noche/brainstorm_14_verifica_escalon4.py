"""Brainstorming (docs/BRAINSTORM_NOCHE.md): verificación independiente (Claude) del escalón 4 que
propone Gemini: escalón 3 + MATCHUPS (contracruces de campeones por línea y sinergias causales,
caché `gemini_mpr.npz` de `brainstorm_g_07_matchups_parche_rematch.py`).

Además del Δ global, se mira en qué lobbies suma (veteranos / sin estrenos, donde el escalón 3 rinde menos),
y una prueba de placebo: los MATCHUPS de otra partida al azar (misma distribución, sin relación con
esta) no deberían sumar nada.
"""
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.brainstorm_base import REPO, ajustar_predecir, cargar, comparar, evaluar, extras_referencia, mostrar  # noqa: E402

D = cargar()
NM, ms = D["NM"], D["master"]
REF = extras_referencia(D)
MAT = np.load(REPO / "data" / "analysis" / "brainstorm" / "gemini_mpr.npz")["MATCHUPS"]
PLAC = MAT[np.random.default_rng(0).permutation(NM)]


def m(extra=None):
    X = REF if extra is None else np.c_[REF, extra]
    return lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=X, extra_post=False)


VAR = {"escalón 3": m(), "escalón 4 (+ MATCHUPS)": m(MAT), "placebo (MATCHUPS de otra partida)": m(PLAC)}
P = dict(Parallel(n_jobs=2, backend="threading")(delayed(lambda n: (n, evaluar(D, VAR[n])))(n) for n in VAR))
mostrar(comparar(D, P, "escalón 3"))

vi = np.flatnonzero(D["val"])
y = D["y"][vi]
Xj, c = D["Xj"][vi], {k: i for i, k in enumerate(D["cols"])}
vet = np.expm1(Xj[..., c["logn_tot"]]).reshape(len(vi), 10).min(1) >= 50
sin_est = Xj[..., c["estreno"]].reshape(len(vi), 10).sum(1) == 0
p3, p4 = P["escalón 3"][1], P["escalón 4 (+ MATCHUPS)"][1]
for nom, mk in (("veteranos (mín ≥ 50)", vet), ("resto", ~vet), ("sin estrenos", sin_est), ("con estrenos", ~sin_est)):
    print(f"{nom:22s} n={mk.sum():5d}  AUC {roc_auc_score(y[mk], p3[mk]):.4f} → {roc_auc_score(y[mk], p4[mk]):.4f} "
          f"({roc_auc_score(y[mk], p4[mk]) - roc_auc_score(y[mk], p3[mk]):+.4f})")
