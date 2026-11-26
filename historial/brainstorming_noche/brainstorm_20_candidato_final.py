"""Brainstorming (docs/BRAINSTORM_NOCHE.md): métricas del candidato final de la noche (escalón 4 sin el mapa de LP)
en V Master+: AUC, acierto, log loss, cobertura (acierto en la fracción más segura) y calibración."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.brainstorm_base import ajustar_predecir, cargar, extras_referencia  # noqa: E402

D = cargar()
ms = D["master"]
R4 = extras_referencia(D, escalon=int(sys.argv[1]) if len(sys.argv) > 1 else 4)
vi = np.flatnonzero(D["val"])
tr = D["fi_v"][ms[D["fi_v"]]]
y = D["y"][vi]
for nom, kw in (("con mapa", {}), ("sin mapa (candidato final)", {"con_mapa": False})):
    p = ajustar_predecir(D, tr, vi, extra=R4, extra_post=False, **kw)
    ok = (p > 0.5) == y
    o = np.argsort(-np.abs(p - 0.5))
    cov = "  ".join(f"{c:.0%}→{ok[o[:int(c * len(vi))]].mean():.1%}" for c in (0.05, 0.1, 0.2, 0.3, 0.5))
    dec = pd.qcut(p, 10, labels=False)
    g = pd.DataFrame({"p": p, "y": y, "d": dec}).groupby("d").agg(p=("p", "mean"), r=("y", "mean"), n=("y", "size"))
    ece = (np.abs(g.p - g.r) * g.n).sum() / g.n.sum()
    print(f"{nom}: AUC {roc_auc_score(y, p):.4f}  acierto {ok.mean():.1%}  log loss {log_loss(y, p):.4f}  ECE {ece:.4f}\n   cobertura: {cov}")
