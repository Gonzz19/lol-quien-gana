"""Idea 8 del brainstorming (docs/BRAINSTORM_NOCHE.md): cobertura y calibración en V Master+ del
mejor modelo de la noche (híbrido + mapa + margen por campeón y línea, solo Master+).

* Acierto si solo se predice la fracción más segura (5, 10, 20, 30, 50, 100 %), con IC de Wilson.
* Calibración por deciles (ECE) y pendiente de recalibración.
* Acierto por tier del lobby (Master, GM+), por lado favorito y por cuántos días después de la foto.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.brainstorm_base import REPO, ajustar_predecir, cargar  # noqa: E402

D = cargar()
ms, NM = D["master"], D["NM"]
C = np.load(REPO / "data" / "analysis" / "brainstorm" / "margen_grupos.npz")
MCML = np.c_[(C["MC"][:, 0] - C["MC"][:, 1]).reshape(NM, -1), (C["ML"][:, 0] - C["ML"][:, 1]).reshape(NM, -1)]
vi = np.flatnonzero(D["val"])
tr = D["fi_v"][ms[D["fi_v"]]]
p = ajustar_predecir(D, tr, vi, extra=MCML, extra_post=False)
y = D["y"][vi]
ok = (p > 0.5) == y
conf = np.abs(p - 0.5)


def wilson(k, n, z=1.96):
    q = k / n
    c = (q + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(q * (1 - q) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


print(f"V Master+: {len(vi):,} partidas, acierto {ok.mean():.1%}")
filas = []
orden = np.argsort(-conf)
for cov in (0.05, 0.10, 0.20, 0.30, 0.50, 1.0):
    k = int(cov * len(vi))
    s = orden[:k]
    lo, hi = wilson(ok[s].sum(), k)
    filas.append({"cobertura": f"{cov:.0%}", "partidas": k, "umbral prob.": 0.5 + conf[s].min(),
                  "acierto": ok[s].mean(), "IC lo": lo, "IC hi": hi})
print(pd.DataFrame(filas).to_string(index=False, formatters={"acierto": "{:.1%}".format, "IC lo": "{:.1%}".format,
                                                             "IC hi": "{:.1%}".format, "umbral prob.": "{:.3f}".format}))
dec = pd.qcut(p, 10, labels=False)
cal = pd.DataFrame({"p": p, "y": y, "d": dec}).groupby("d").agg(p=("p", "mean"), real=("y", "mean"), n=("y", "size"))
print("\nCalibración por deciles:\n", cal.round(3).to_string())
print(f"ECE: {(np.abs(cal.p - cal.real) * cal.n).sum() / cal.n.sum():.4f}")
lg = np.log(p / (1 - p))
r = LogisticRegression(C=1e6).fit(lg[:, None], y)
print(f"recalibración: pendiente {r.coef_[0, 0]:.3f}, intercepto {r.intercept_[0]:+.3f}")
tl = D["tier_lobby"][vi]
for nom, m in (("lobby Master (media de tier < 1,5)", tl < 1.5), ("lobby GM+ (media ≥ 1,5)", tl >= 1.5)):
    print(f"{nom}: {m.sum():,} partidas, acierto {ok[m].mean():.1%}")
dias = (D["t"][vi] - D["B"]["t1"]) / 86_400_000
for a, b in ((11, 15), (15, 19), (19, 23)):
    m = (dias >= a) & (dias < b)
    print(f"días {a}-{b} después de la foto: {m.sum():,} partidas, acierto {ok[m].mean():.1%}")
fav_azul = p > 0.5
print(f"favorito azul: {fav_azul.mean():.1%} de las veces, acierto {ok[fav_azul].mean():.1%}; "
      f"favorito rojo: acierto {ok[~fav_azul].mean():.1%}")
