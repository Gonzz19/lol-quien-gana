"""Idea 27 del brainstorming (docs/BRAINSTORM_NOCHE.md): para la demo, ¿filtrar por acuerdo entre la logística (escalón 4)
y la red NeuralAC mejora el acierto en la fracción más segura respecto de ordenar por el ensamble (escalón 5)?
Desde `pred_g08.npz`, en V Master+."""
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.brainstorm_base import REPO, cargar  # noqa: E402

D = cargar()
G = np.load(REPO / "data" / "analysis" / "brainstorm" / "pred_g08.npz")
pl, pr = G["pv_log4"], G["pv_red"]
pe = expit(0.65 * logit(pl) + 0.35 * logit(pr))
y = D["y"][D["val"]]
n = len(y)
for nom, score in (("ensamble |p − 0,5|", np.abs(pe - 0.5)),
                   ("mínimo de las dos confianzas (acuerdo)", np.where((pl > .5) == (pr > .5), np.minimum(np.abs(pl - .5), np.abs(pr - .5)), -1)),
                   ("solo logística", np.abs(pl - 0.5))):
    o = np.argsort(-score)
    fila = []
    for c in (0.05, 0.1, 0.2, 0.3):
        s = o[:int(c * n)]
        fila.append(f"{c:.0%}→{((pe[s] > .5) == y[s]).mean():.1%}")
    print(f"{nom:42s} " + "  ".join(fila))
print(f"desacuerdo entre logística y red: {((pl > .5) != (pr > .5)).mean():.1%} de las partidas")
