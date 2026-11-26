"""Idea 23 del brainstorming (docs/BRAINSTORM_NOCHE.md): limpiar ruido de etiqueta en el ENTRENAMIENTO (sobre el escalón 4).

Partidas decididas por un remake, un AFK o una rendición muy temprana dicen poco de quién era mejor
antes de jugar. Se sacan del entrenamiento (V queda entera, como siempre):
* duración < 5 min (remakes), < 15 min, < 20 min con rendición (`D_sin_20`);
* o se ponderan por la duración (más peso a las partidas largas, que se "jugaron").
Elegir filas de entrenamiento con datos de esas mismas partidas no filtra nada hacia V.
"""
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.brainstorm_base import ajustar_predecir, cargar, comparar, evaluar, extras_referencia, mostrar  # noqa: E402

D = cargar()
ms = D["master"]
R4 = extras_referencia(D, escalon=4)
Zd = D["Z"]
dur = Zd["D_dur_min"]
sin20 = Zd["D_sin_20"].astype(bool)
print(f"entrenamiento Master+: <5 min {(dur < 5)[D['fi_v']].mean():.2%}, <15 min {(dur < 15)[D['fi_v']].mean():.2%}, "
      f"rendición <20 {sin20[D['fi_v']].mean():.2%}")


def m(ok=None, peso_dur=False):
    def f(tr, te):
        t = tr[ms[tr]]
        if ok is not None:
            t = t[ok[t]]
        w = None
        if peso_dur:
            w = np.clip(dur[t], 10, 35) / 25.0
        return ajustar_predecir(D, t, te, extra=R4, extra_post=False, peso=w)
    return f


VAR = {"escalón 4": m(), "sin remakes (< 5 min)": m(dur >= 5), "sin < 15 min": m(dur >= 15),
       "sin rendiciones antes del 20": m(~sin20 & (dur >= 5)), "peso por duración": m(peso_dur=True)}
P = dict(Parallel(n_jobs=2, backend="threading")(delayed(lambda n: (n, evaluar(D, VAR[n])))(n) for n in VAR))
mostrar(comparar(D, P, "escalón 4"))
