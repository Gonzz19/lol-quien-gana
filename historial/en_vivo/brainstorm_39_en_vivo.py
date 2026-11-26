"""Brainstorming (docs/BRAINSTORM_NOCHE.md): predicción en dos tiempos sobre el escalón 9 (V Master+).

Segunda etapa: logística sobre [logit previo del escalón 9, Δoro, Δxp, Δkills del equipo al minuto m] (m = 10, 15, 20).
Se ajusta con las predicciones hacia adelante de A2-A4 Master+ (el logit previo de esas partidas es fuera de muestra) y se
mide en V Master+. Solo partidas que llegaron al minuto m. También: solo el estado (sin el logit previo), y el acierto en
el 20 % de partidas más seguras antes de jugar según si el favorito va adelante en oro."""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.brainstorm_base import ajustar_predecir, cargar, evaluar, extras_referencia  # noqa: E402

D = cargar()
ms, y, Z = D["master"], D["y"], D["Z"]
X = extras_referencia(D, escalon=9)
po, pv = evaluar(D, lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=X, extra_post=False, con_mapa=False))
vi = np.flatnonzero(D["val"])
ai = np.flatnonzero((D["bloque"] >= 1) & ms)
lg = lambda p: np.log(p / (1 - p))  # noqa: E731
la, lv = lg(po[ai]), lg(pv)
print(f"antes de la partida (V Master+, {len(vi):,}): AUC {roc_auc_score(y[vi], pv):.4f}, acierto {((pv > .5) == y[vi]).mean():.1%}")
q80 = np.quantile(np.abs(lv), 0.8)
for m in (10, 15, 20):
    E = np.c_[Z[f"D_d_gold_{m}"], Z[f"D_d_xp_{m}"], Z[f"D_d_kills_{m}"]]
    oka, okv = ~np.isnan(E[ai]).any(1), ~np.isnan(E[vi]).any(1)
    mu, sd = E[ai][oka].mean(0), E[ai][oka].std(0)
    Ea, Ev = (E[ai][oka] - mu) / sd, (E[vi][okv] - mu) / sd
    ya, yv = y[ai][oka], y[vi][okv]
    p_s = LogisticRegression(C=1.0).fit(Ea, ya).predict_proba(Ev)[:, 1]
    p_c = LogisticRegression(C=1.0).fit(np.c_[Ea, la[oka]], ya).predict_proba(np.c_[Ev, lv[okv]])[:, 1]
    p0 = pv[okv]
    rng = np.random.default_rng(1)
    B = rng.integers(0, len(yv), size=(300, len(yv)))
    d = np.array([roc_auc_score(yv[b], p_c[b]) - roc_auc_score(yv[b], p_s[b]) for b in B])
    print(f"min {m} ({okv.sum():,} partidas): previo solo {roc_auc_score(yv, p0):.4f}/{((p0 > .5) == yv).mean():.1%} | estado solo "
          f"{roc_auc_score(yv, p_s):.4f}/{((p_s > .5) == yv).mean():.1%} | previo + estado {roc_auc_score(yv, p_c):.4f}/{((p_c > .5) == yv).mean():.1%} "
          f"(Δ vs estado {roc_auc_score(yv, p_c) - roc_auc_score(yv, p_s):+.4f} [{np.percentile(d, 2.5):+.4f}; {np.percentile(d, 97.5):+.4f}])")
    fac = np.abs(lv[okv]) >= q80
    fav = np.sign(lv[okv]); g = Z[f"D_d_gold_{m}"][vi][okv]
    adel = np.sign(g) == fav
    ok0 = (p0 > .5) == yv
    print(f"   20 % más seguro antes de jugar: favorito adelante en oro {adel[fac].mean():.1%} → acierto {ok0[fac & adel].mean():.1%}; "
          f"atrás → {ok0[fac & ~adel].mean():.1%}; con previo + estado, acierto en ese 20 %: {((p_c > .5) == yv)[fac].mean():.1%}")
    o = np.argsort(-np.abs(p_c - .5))
    print("   cobertura previo + estado: " + "  ".join(f"{c:.0%}→{((p_c > .5) == yv)[o[:int(c * len(yv))]].mean():.1%}" for c in (.1, .2, .5)))
