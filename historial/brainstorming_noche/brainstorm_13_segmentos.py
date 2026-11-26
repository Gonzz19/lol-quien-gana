"""Brainstorming (docs/BRAINSTORM_NOCHE.md), idea 18: modelos por segmento de lobby, sobre el escalón 3.

El análisis de errores (brainstorm_12) muestra que el modelo rinde menos en lobbies donde nadie
estrena campeón (AUC 0,6455) y donde todos tienen ≥ 50 partidas previas (0,6533). Ahí pierde su
señal principal (asimetría de experiencia). Se prueba entrenar un modelo por segmento (con las filas
de entrenamiento de ese segmento) y predecir cada partida con el modelo de su segmento:

* segmento A: ¿hay algún estreno en el lobby? (sí / no);
* segmento B: mínimo de partidas previas del lobby ≥ 50 (sí / no);
* mezcla: modelo global + modelo del segmento, promedio de logits 50/50 (fijo de antemano).
Los segmentos se definen con información previa a la partida (conteos de partidas anteriores).
"""
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.brainstorm_base import ajustar_predecir, cargar, comparar, evaluar, extras_referencia, mostrar  # noqa: E402

D = cargar()
NM, ms = D["NM"], D["master"]
REF = extras_referencia(D)
Xj, c = D["Xj"], {k: i for i, k in enumerate(D["cols"])}
con_estreno = Xj[..., c["estreno"]].reshape(NM, 10).sum(1) > 0
veterano = np.expm1(Xj[..., c["logn_tot"]]).reshape(NM, 10).min(1) >= 50


def glob(tr, te):
    return ajustar_predecir(D, tr[ms[tr]], te, extra=REF, extra_post=False)


def por_segmento(seg, mezcla=False):
    def f(tr, te):
        tr = tr[ms[tr]]
        p = np.empty(len(te))
        for v in (True, False):
            a, b = tr[seg[tr] == v], te[seg[te] == v]
            if len(b):
                p[seg[te] == v] = ajustar_predecir(D, a, b, extra=REF, extra_post=False)
        if mezcla:
            pg = glob(tr, te)
            lg = 0.5 * np.log(p / (1 - p)) + 0.5 * np.log(pg / (1 - pg))
            p = 1 / (1 + np.exp(-lg))
        return p
    return f


VAR = {"referencia (escalón 3)": glob,
       "por segmento: ¿hay estreno?": por_segmento(con_estreno),
       "por segmento: veteranos (mín ≥ 50)": por_segmento(veterano),
       "mezcla global + ¿hay estreno?": por_segmento(con_estreno, True),
       "mezcla global + veteranos": por_segmento(veterano, True)}
print(f"V Master+: con estreno {con_estreno[D['val']].mean():.1%}, veteranos {veterano[D['val']].mean():.1%}")
P = dict(Parallel(n_jobs=2, backend="threading")(delayed(lambda n: (n, evaluar(D, VAR[n])))(n) for n in VAR))
mostrar(comparar(D, P, "referencia (escalón 3)"))
