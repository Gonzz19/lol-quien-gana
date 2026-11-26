"""Ideas 2-4 del brainstorming (docs/BRAINSTORM_NOCHE.md), sobre el híbrido + mapa entrenado solo con Master+.

2. Asimetría de LP entre equipos (mapa causal, desde el 04/09): azul − rojo de la dispersión, el
   máximo y el mínimo del LP y del nivel del lobby de cada equipo.
3. Hora del día (ART) × diferencia de habilidad: franjas (0-6, 6-12, 12-18, 18-24) × diferencia de
   margen acumulado medio y × diferencia de LP del mapa.
4. Duos: par del mismo equipo con ≥ 3 partidas previas juntos y ninguna enfrentados (§9.S.24),
   calculado de forma causal (solo partidas anteriores); azul − rojo de la cantidad de duos.
"""
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.brainstorm_base import ajustar_predecir, cargar, comparar, evaluar, mostrar  # noqa: E402

D = cargar()
NM, J, t, post, ms = D["NM"], D["J"], D["t"], D["post"], D["master"]
cols = D["cols"]

# ---- 2. asimetría de LP (por equipo)
lp, niv = J[..., 3], J[..., 1]                      # centrados, 0 antes del 04/09


def por_eq(a, f):
    return f(a[:, 0], axis=1) - f(a[:, 1], axis=1)


ASIM = np.c_[por_eq(lp, np.std), por_eq(lp, np.max), por_eq(lp, np.min),
             por_eq(niv, np.std), por_eq(niv, np.max), por_eq(niv, np.min)].astype(np.float32)

# ---- 3. hora × diferencia de habilidad
hora = ((t / 3.6e6) - 3) % 24
franja = np.minimum((hora // 6).astype(int), 3)
iacum = [i for i, c in enumerate(cols) if c.startswith("acum_")]
marg = (D["Xj"][:, 0][:, :, iacum].mean(axis=(1, 2)) - D["Xj"][:, 1][:, :, iacum].mean(axis=(1, 2)))
dlp = lp[:, 0].mean(1) - lp[:, 1].mean(1)
HORA = np.column_stack([(franja == k) * x for k in range(4) for x in (marg, dlp)]).astype(np.float32)

# ---- 4. duos causales
pl = D["B"]["player"]
orden = np.argsort(t, kind="mergesort")
juntos, contra = defaultdict(int), defaultdict(int)
duos = np.zeros((NM, 2), np.float32)
for m in orden:
    eqs = [sorted(pl[m, 0]), sorted(pl[m, 1])]
    for s in (0, 1):
        duos[m, s] = sum(1 for a, b in combinations(eqs[s], 2) if juntos[(a, b)] >= 3 and contra[(a, b)] == 0)
    for s in (0, 1):
        for a, b in combinations(eqs[s], 2):
            juntos[(a, b)] += 1
    for a in eqs[0]:
        for b in eqs[1]:
            contra[(a, b) if a < b else (b, a)] += 1
DUO = np.c_[duos[:, 0] - duos[:, 1], (duos[:, 0] - duos[:, 1]) * ms].astype(np.float32)
print(f"partidas con algún duo: {(duos.sum(1) > 0).mean():.1%} (Master+ {(duos[ms].sum(1) > 0).mean():.1%})")


def solo_m(extra=None, extra_post=True):
    return lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=extra, extra_post=extra_post)


VAR = {
    "híbrido + mapa (todo el entrenamiento)": lambda tr, te: ajustar_predecir(D, tr, te),
    "referencia: solo Master+": solo_m(),
    "+ asimetría de LP": solo_m(ASIM, True),
    "+ hora × habilidad": solo_m(HORA, False),
    "+ duos": solo_m(DUO, False),
    "+ las tres": solo_m(np.c_[ASIM, HORA, DUO], False),
}
P = dict(Parallel(n_jobs=4, backend="threading")(delayed(lambda n: (n, evaluar(D, VAR[n])))(n) for n in VAR))
mostrar(comparar(D, P, "referencia: solo Master+"))
