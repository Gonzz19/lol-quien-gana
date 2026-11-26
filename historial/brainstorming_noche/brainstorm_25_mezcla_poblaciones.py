"""Idea 26 del brainstorming (docs/BRAINSTORM_NOCHE.md): diversidad por población de entrenamiento.

Mezcla de logits (pesos fijos de antemano 0,8/0,2 y 0,65/0,35) del escalón 4 entrenado solo con Master+ y el mismo
modelo entrenado con todos los tiers. Además, la mezcla con la red de Gemini (escalón 5) para ver si suma encima."""
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.brainstorm_base import REPO, ajustar_predecir, cargar, comparar, evaluar, extras_referencia, mostrar  # noqa: E402

D = cargar()
ms = D["master"]
R4 = extras_referencia(D, escalon=4)
P = dict(Parallel(n_jobs=2, backend="threading")(delayed(lambda n, f: (n, evaluar(D, f)))(n, f) for n, f in (
    ("escalón 4 (solo Master+)", lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=R4, extra_post=False)),
    ("escalón 4 entrenado con todo", lambda tr, te: ajustar_predecir(D, tr, te, extra=R4, extra_post=False)))))
G = np.load(REPO / "data" / "analysis" / "brainstorm" / "pred_g08.npz")
RED = (G["po_red"], G["pv_red"])
L = lambda a: logit(np.clip(a, 1e-6, 1 - 1e-6))  # noqa: E731


def mix(ws, ps):
    return tuple(expit(sum(w * L(p[k]) for w, p in zip(ws, ps))) for k in (0, 1))


A, T = P["escalón 4 (solo Master+)"], P["escalón 4 entrenado con todo"]
P["mezcla 0,8 Master+ / 0,2 todo"] = mix((0.8, 0.2), (A, T))
P["mezcla 0,65 Master+ / 0,35 todo"] = mix((0.65, 0.35), (A, T))
P["escalón 5 (0,65 log + 0,35 red)"] = mix((0.65, 0.35), (A, RED))
P["0,5 log Master+ / 0,15 log todo / 0,35 red"] = mix((0.5, 0.15, 0.35), (A, T, RED))
mostrar(comparar(D, P, "escalón 4 (solo Master+)"))
