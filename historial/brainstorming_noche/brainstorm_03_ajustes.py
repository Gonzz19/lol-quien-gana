"""Brainstorming (docs/BRAINSTORM_NOCHE.md): ajustes del entrenamiento con solo Master+.

* C de la logística (0,02 / 0,05 / 0,1): con menos filas, la regularización óptima puede cambiar.
* Peso por antigüedad: w = exp(−(t_ref − t) / τ), τ = 15, 30 y 60 días (lo reciente rinde más, §9.S.48).
* Mapa por línea (5 × 7 + equipo) en vez de solo equipo.
"""
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.brainstorm_base import ajustar_predecir, cargar, comparar, evaluar, mostrar  # noqa: E402

D = cargar()
ms, t, J = D["master"], D["t"], D["J"]
LINEA = (J[:, 0] - J[:, 1]).reshape(D["NM"], -1).astype(np.float32)


def solo_m(**kw):
    return lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, **kw)


def reciente(tau):
    def f(tr, te):
        tr = tr[ms[tr]]
        w = np.exp(-(t[tr].max() - t[tr]) / (tau * 86_400_000))
        return ajustar_predecir(D, tr, te, peso=w / w.mean())
    return f


VAR = {
    "referencia: solo Master+ (C = 0,05)": solo_m(),
    "C = 0,02": solo_m(C=0.02),
    "C = 0,1": solo_m(C=0.1),
    "reciente τ = 15 d": reciente(15),
    "reciente τ = 30 d": reciente(30),
    "reciente τ = 60 d": reciente(60),
    "+ mapa por línea": solo_m(extra=LINEA, extra_post=True),
}
P = dict(Parallel(n_jobs=4, backend="threading")(delayed(lambda n: (n, evaluar(D, VAR[n])))(n) for n in VAR))
mostrar(comparar(D, P, "referencia: solo Master+ (C = 0,05)"))
