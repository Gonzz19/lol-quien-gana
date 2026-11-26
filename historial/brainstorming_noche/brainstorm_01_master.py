"""Idea 1 del brainstorming: ¿pesos específicos de Master+? (docs/BRAINSTORM_NOCHE.md)

Sobre el híbrido + mapa (§9.S.50-51), con la partición vigente:
* entrenar solo con partidas de Master+;
* ponderar más las de Master+ (×2, ×4) en el entrenamiento.

La elección o ponderación de filas de entrenamiento por tier usa el padrón (foto 1). Para las
partidas de entrenamiento anteriores al 04/09, ese tier es posterior a ellas, pero solo decide
qué filas entrenan, no entra como variable ni toca V, así que no infla V.
No se prueban interacciones variable × tier: con el tier del padrón como variable, las partidas
anteriores a la foto filtrarían el rango (§9.S.23).
"""
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.brainstorm_base import ajustar_predecir, cargar, comparar, evaluar, mostrar  # noqa: E402

D = cargar()
ms = D["master"]
print(f"Master+ en el entrenamiento de V: {ms[D['fi_v']].mean():.1%} de {len(D['fi_v']):,}")

VAR = {
    "referencia (híbrido + mapa)": lambda tr, te: ajustar_predecir(D, tr, te),
    "solo Master+ en el entrenamiento": lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te),
    "Master+ ×2": lambda tr, te: ajustar_predecir(D, tr, te, peso=np.where(ms[tr], 2.0, 1.0)),
    "Master+ ×4": lambda tr, te: ajustar_predecir(D, tr, te, peso=np.where(ms[tr], 4.0, 1.0)),
}
res = Parallel(n_jobs=4, backend="threading")(delayed(lambda n: (n, evaluar(D, VAR[n])))(n) for n in VAR)
P = dict(res)
mostrar(comparar(D, P, "referencia (híbrido + mapa)"))
