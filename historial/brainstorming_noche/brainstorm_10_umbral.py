"""Brainstorming (docs/BRAINSTORM_NOCHE.md): ¿con qué lobbies entrenar? Sobre la referencia vigente (escalón 3).

El tier medio del lobby (Diamond = 0, Master = 1, GM = 2, Chall = 3; padrón = foto 1) elige las filas
de entrenamiento (no es variable; no toca V). Se comparan:
* Master+ por mediana (≥ 6 de 10 Master+), la referencia;
* tier medio ≥ 0,3 / 0,5 / 0,7 / 0,9 (más o menos Diamond alto incluido);
* todo (sin filtro).
"""
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.brainstorm_base import ajustar_predecir, cargar, comparar, evaluar, extras_referencia, mostrar  # noqa: E402

D = cargar()
ms, tl = D["master"], np.nan_to_num(D["tier_lobby"], nan=-1)
REF = extras_referencia(D)


def filtro(mask):
    return lambda tr, te: ajustar_predecir(D, tr[mask[tr]], te, extra=REF, extra_post=False)


VAR = {"referencia: Master+ por mediana": filtro(ms)}
for u in (0.3, 0.5, 0.7, 0.9):
    VAR[f"tier medio ≥ {u}"] = filtro(tl >= u)
VAR["todo"] = filtro(np.ones(D["NM"], bool))
for n in list(VAR)[1:]:
    pass
fi_v = D["fi_v"]
print({n: f"{(m[fi_v]).mean():.1%}" for n, m in [("Master+", ms)] + [(f"≥{u}", tl >= u) for u in (0.3, 0.5, 0.7, 0.9)]})
P = dict(Parallel(n_jobs=2, backend="threading")(delayed(lambda n: (n, evaluar(D, VAR[n])))(n) for n in VAR))
mostrar(comparar(D, P, "referencia: Master+ por mediana"))
