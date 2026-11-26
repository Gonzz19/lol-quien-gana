"""Brainstorming (docs/BRAINSTORM_NOCHE.md), idea 21: poda del escalón 4 (0,6674).

Con margen por campeón/línea, nivel histórico, W/L de la foto y matchups adentro, ¿siguen sumando
el one-hot del draft (851 columnas), el perfil típico del campeón y Data Dragon, y el mapa de LP?
Sacar lo que sobra baja la varianza (y simplifica el modelo para la demo).
"""
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import scripts.analysis.brainstorm_base as bb  # noqa: E402
from scripts.analysis.brainstorm_base import ajustar_predecir, cargar, comparar, evaluar, extras_referencia, mostrar  # noqa: E402

D = cargar()
ms = D["master"]
REF = extras_referencia(D, escalon=4)
cols = D["cols"]
d = len(cols)
campeon = np.tile(np.array([c.startswith(("tag_", "info_", "st_", "perf_")) for c in cols]), 5)
D_sin_camp = dict(D, Xd=D["Xd"][:, ~campeon])


def m(DD=D, **kw):
    return lambda tr, te: ajustar_predecir(DD, tr[ms[tr]], te, extra=REF, extra_post=False, **kw)


VAR = {"escalón 4": m(), "sin one-hot del draft": m(con_draft=False), "sin perfil ni Data Dragon": m(D_sin_camp),
       "sin draft, perfil ni Data Dragon": m(D_sin_camp, con_draft=False), "sin el mapa de LP": m(con_mapa=False)}
P = dict(Parallel(n_jobs=2, backend="threading")(delayed(lambda n: (n, evaluar(D, VAR[n])))(n) for n in VAR))
mostrar(comparar(D, P, "escalón 4"))
