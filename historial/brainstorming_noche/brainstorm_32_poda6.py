"""Brainstorming (docs/BRAINSTORM_NOCHE.md): poda del escalón 6 sin mapa (candidato final). Se sacan las familias que
sobraban en la importancia (brainstorm_21): winrate previo, perfil típico del campeón, Data Dragon; y se mide en V y bloques."""
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.brainstorm_base import ajustar_predecir, cargar, comparar, evaluar, extras_referencia, mostrar  # noqa: E402

D = cargar()
ms = D["master"]
X = extras_referencia(D, escalon=6)
cols = D["cols"]
grupo = {"winrate previo": lambda c: c in ("wr_acum", "wr_v20"), "perfil": lambda c: c.startswith("perf_"),
         "Data Dragon": lambda c: c.startswith(("tag_", "info_", "st_"))}


def sin(*nombres):
    mk = np.tile(np.array([not any(grupo[n](c) for n in nombres) for c in cols]), 5)
    DD = dict(D, Xd=D["Xd"][:, mk])
    return lambda tr, te: ajustar_predecir(DD, tr[ms[tr]], te, extra=X, extra_post=False, con_mapa=False)


VAR = {"escalón 6 sin mapa": sin(), "sin winrate previo": sin("winrate previo"), "sin perfil": sin("perfil"),
       "sin winrate, perfil ni Data Dragon": sin("winrate previo", "perfil", "Data Dragon")}
P = dict(Parallel(n_jobs=2, backend="threading")(delayed(lambda n: (n, evaluar(D, VAR[n])))(n) for n in VAR))
mostrar(comparar(D, P, "escalón 6 sin mapa"))
