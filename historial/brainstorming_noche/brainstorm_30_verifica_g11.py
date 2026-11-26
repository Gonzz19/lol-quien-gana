"""Brainstorming (docs/BRAINSTORM_NOCHE.md): verificación (Claude) de g_11 de Gemini (Kalman-Elo causal con objetivo
mixto resultado/@15, tempo y objetivos, economía de mitad de partida; caché `gemini_g11.npz`).

Reproducción sobre el escalón 4 (todo g_11 y sin CLUTCH_15), sobre el 4b (+ ROL_WR), placebo (filas permutadas) y
chequeo en Diamond-V (nunca usado para decidir)."""
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
R4b = extras_referencia(D, escalon=4, rol_wr=True)
Gz = np.load(REPO / "data" / "analysis" / "brainstorm" / "gemini_g11.npz")
TODO = np.c_[Gz["KALMAN_OUT"], Gz["KALMAN_MRG"], Gz["CLUTCH_15"], Gz["TEMPO_OBJ"], Gz["ECON_MID"]]
SINC = np.c_[Gz["KALMAN_OUT"], Gz["KALMAN_MRG"], Gz["TEMPO_OBJ"], Gz["ECON_MID"]]
PLAC = SINC[np.random.default_rng(0).permutation(NM)]
m = lambda X: (lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=X.astype(np.float32), extra_post=False))  # noqa: E731
VAR = {"escalón 4": m(R4), "4 + g_11 todo": m(np.c_[R4, TODO]), "4 + g_11 sin CLUTCH": m(np.c_[R4, SINC]),
       "4 + placebo (g_11 sin CLUTCH permutado)": m(np.c_[R4, PLAC]),
       "4b": m(R4b), "4b + g_11 sin CLUTCH": m(np.c_[R4b, SINC])}
P = dict(Parallel(n_jobs=2, backend="threading")(delayed(lambda n: (n, evaluar(D, VAR[n])))(n) for n in VAR))
mostrar(comparar(D, {k: P[k] for k in list(VAR)[:4]}, "escalón 4"))
mostrar(comparar(D, {k: P[k] for k in ("4b", "4b + g_11 sin CLUTCH")}, "4b"))
di = np.flatnonzero(D["diam_v"])
y = D["y"][di]
B = np.random.default_rng(1).integers(0, len(di), size=(300, len(di)))
for n, ref in (("4 + g_11 todo", "escalón 4"), ("4 + g_11 sin CLUTCH", "escalón 4"), ("4 + placebo (g_11 sin CLUTCH permutado)", "escalón 4"),
               ("4b + g_11 sin CLUTCH", "4b")):
    p, q = P[n][0][di], P[ref][0][di]
    d = np.array([roc_auc_score(y[b], p[b]) - roc_auc_score(y[b], q[b]) for b in B])
    print(f"Diamond-V {n:42s} Δ {roc_auc_score(y, p) - roc_auc_score(y, q):+.4f} [{np.percentile(d, 2.5):+.4f}; {np.percentile(d, 97.5):+.4f}]")
np.savez_compressed(REPO / "data" / "analysis" / "brainstorm" / "pred_4b_g11.npz", po=P["4b + g_11 sin CLUTCH"][0], pv=P["4b + g_11 sin CLUTCH"][1])
