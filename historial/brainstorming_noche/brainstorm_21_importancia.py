"""Brainstorming (docs/BRAINSTORM_NOCHE.md): importancia por bloques del candidato final (escalón 4 sin mapa) en V Master+.

Se saca un bloque por vez y se mide ΔAUC y Δacierto en V (IC por bootstrap pareado). Bloques: los 5 agregados de la
noche (margen por campeón, por línea, nivel histórico, W/L de la foto, matchups), el one-hot del draft y las 9
familias de las 85 variables por jugador.
"""
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.brainstorm_base import ajustar_predecir, cargar, extras_referencia  # noqa: E402

D = cargar()
ms = D["master"]
R4 = extras_referencia(D, escalon=4)
SL = {"margen por campeón": slice(0, 75), "margen por línea": slice(75, 150), "nivel histórico": slice(150, 153),
      "W/L de la foto 1": slice(153, 157), "matchups": slice(157, 167)}
JC = {"logn_champ", "estreno", "wr_champ", "share_champ"}
FAM = {"margen acumulado": lambda c: c.startswith("acum_"), "margen últimas 20": lambda c: c.startswith("v20_"),
       "winrate previo": lambda c: c in ("wr_acum", "wr_v20"), "partidas, línea y rol": lambda c: c in ("logn_tot", "share_lane", "offrole"),
       "pool": lambda c: c in ("conc_rec", "conc_ant", "d_pool", "otp_nuevo"), "ritmo y descanso": lambda c: c in ("ritmo", "log_horas_desde"),
       "experiencia con el campeón": lambda c: c in JC, "Data Dragon": lambda c: c.startswith(("tag_", "info_", "st_")),
       "perfil del campeón": lambda c: c.startswith("perf_")}
fam = np.tile(np.array([next(f for f, g in FAM.items() if g(c)) for c in D["cols"]]), 5)
vi = np.flatnonzero(D["val"])
tr = D["fi_v"][ms[D["fi_v"]]]
y = D["y"][vi]


def corre(nombre):
    DD, X, kw = D, R4, {"con_mapa": False}
    if nombre in SL:
        X = np.delete(R4, np.r_[SL[nombre]], axis=1)
    elif nombre == "one-hot del draft":
        kw["con_draft"] = False
    elif nombre in FAM:
        DD = dict(D, Xd=D["Xd"][:, fam != nombre])
    return nombre, ajustar_predecir(DD, tr, vi, extra=X, extra_post=False, **kw)


nombres = ["completo"] + list(SL) + ["one-hot del draft"] + list(FAM)
P = dict(Parallel(n_jobs=2, backend="threading")(delayed(corre)(n) for n in nombres))
p0 = P["completo"]
B = np.random.default_rng(1).integers(0, len(vi), size=(200, len(vi)))
print(f"completo: AUC {roc_auc_score(y, p0):.4f}, acierto {((p0 > .5) == y).mean():.1%}")
filas = []
for n in nombres[1:]:
    p = P[n]
    d = np.array([roc_auc_score(y[b], p[b]) - roc_auc_score(y[b], p0[b]) for b in B])
    filas.append((roc_auc_score(y, p) - roc_auc_score(y, p0), np.percentile(d, 2.5), np.percentile(d, 97.5),
                  ((p > .5) == y).mean() - ((p0 > .5) == y).mean(), n))
for dl, lo, hi, da, n in sorted(filas):
    print(f"  sin {n:28s} ΔAUC {dl:+.4f} [{lo:+.4f}; {hi:+.4f}]   Δacierto {da:+.2%}")
