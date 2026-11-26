"""Brainstorming (docs/BRAINSTORM_NOCHE.md): ¿cuánto mejora el modelo de quién gana al sumarle las predicciones del modelo
por línea (oro @15 de cada línea predicho antes de la partida, HGB causal de `brainstorm_g_09`, caché `gemini_hgb_lineas.npz`)?

Se mide en tres puntos, en V Master+ (y sub-bloques A2-A4):
  1. las predicciones de línea solas (5 líneas + suma);
  2. sobre el modelo base (draft + 85 variables por jugador, solo Master+): con y sin líneas;
  3. dentro del modelo final (escalón 9): sacándolas.
"""
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.brainstorm_base import REPO, ajustar_predecir, cargar, comparar, evaluar, extras_referencia, mostrar  # noqa: E402

D = cargar()
ms, y = D["master"], D["y"]
H = np.load(REPO / "data" / "analysis" / "brainstorm" / "gemini_hgb_lineas.npz")["HGB"]
X9 = extras_referencia(D, escalon=9)
i0 = extras_referencia(D, escalon=6).shape[1]
assert np.allclose(X9[:, i0:i0 + 6], H)
X9_sin = np.delete(X9, np.s_[i0:i0 + 6], axis=1)


def solo(tr, te):
    tr = tr[ms[tr]]
    mu, sd = H[tr].mean(0), H[tr].std(0)
    return LogisticRegression(C=1.0).fit((H[tr] - mu) / sd, y[tr]).predict_proba((H[te] - mu) / sd)[:, 1]


m = lambda X: (lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=X, extra_post=False, con_mapa=False))  # noqa: E731
VAR = {"líneas solas (6 columnas)": solo, "base (draft + jugadores)": m(None), "base + líneas": m(H),
       "final (escalón 9)": m(X9), "final sin líneas": m(X9_sin)}
P = dict(Parallel(n_jobs=2, backend="threading")(delayed(lambda n: (n, evaluar(D, VAR[n])))(n) for n in VAR))
print("--- contra el modelo base")
mostrar(comparar(D, {k: P[k] for k in ("base (draft + jugadores)", "líneas solas (6 columnas)", "base + líneas")}, "base (draft + jugadores)"))
print("--- contra el modelo final")
mostrar(comparar(D, {k: P[k] for k in ("final (escalón 9)", "final sin líneas")}, "final (escalón 9)"))
vi = np.flatnonzero(D["val"])
print("AUC de cada línea sola para el resultado de la partida (V Master+): " +
      "  ".join(f"{n} {roc_auc_score(y[vi], H[vi, k]):.3f}" for k, n in enumerate(("BOT", "JG", "MID", "TOP", "SUP", "suma"))))
