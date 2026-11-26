"""Brainstorming (docs/BRAINSTORM_NOCHE.md): segunda etapa en vivo POR LÍNEAS sobre el escalón 9 (V Master+).

Compara, al minuto 10 y 15 (ajuste con las predicciones hacia adelante de A2-A4 Master+, medición en V Master+):
  equipo:  [logit previo, Δoro, Δxp, Δkills del equipo]                       (brainstorm_39)
  líneas:  [logit previo, Δoro y Δxp de cada una de las 5 líneas, Δkills del equipo]
  líneas + sorpresa: además, Δoro de cada línea menos el oro @15 que predecía el modelo por línea antes de jugar (solo m = 15)
  HGB:     gradient boosting sobre las variables de "líneas" (no linealidades: una línea muy adelante, líneas cruzadas)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.brainstorm_base import ajustar_predecir, cargar, evaluar, extras_referencia  # noqa: E402

D = cargar()
ms, y, Z, NM = D["master"], D["y"], D["Z"], D["NM"]
X = extras_referencia(D, escalon=9)
po, pv = evaluar(D, lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=X, extra_post=False, con_mapa=False))
vi = np.flatnonzero(D["val"]); ai = np.flatnonzero((D["bloque"] >= 1) & ms)
lg = lambda p: np.log(p / (1 - p))  # noqa: E731
L = np.full(NM, np.nan); L[ai] = lg(po[ai]); L[vi] = lg(pv)

df = pd.read_csv(REPO / "data" / "splits" / "train" / "full_dataset.csv",
                 usecols=["match_id", "game_creation", "team_id", "lane", "win", "gold_10", "xp_10", "gold_15", "xp_15"])
df = df[df.lane.isin(["BOTTOM", "JUNGLE", "MIDDLE", "TOP", "UTILITY"])]
df = df[df.groupby(["match_id", "team_id"]).lane.transform("nunique") == 5]
df = df[df.groupby("match_id").lane.transform("size") == 10]
df = df[df.groupby("match_id").win.transform("sum") == 5]
df = df.sort_values(["game_creation", "match_id", "team_id", "lane"]).reset_index(drop=True)
assert (df.match_id.to_numpy()[::10] == Z["mids"]).all()
lane = lambda c: (lambda a: (a[:, 0] - a[:, 1]) / 1000.0)(df[c].to_numpy(float).reshape(NM, 2, 5))  # noqa: E731
HGBL = np.load(REPO / "data" / "analysis" / "brainstorm" / "gemini_hgb_lineas.npz")["HGB"][:, :5]   # oro @15 por línea predicho (miles)

for m in (10, 15):
    eq = np.c_[Z[f"D_d_gold_{m}"], Z[f"D_d_xp_{m}"], Z[f"D_d_kills_{m}"]]
    g, x = lane(f"gold_{m}"), lane(f"xp_{m}")
    V = {"equipo": np.c_[L, eq], "líneas": np.c_[L, g, x, Z[f"D_d_kills_{m}"]]}
    if True:  # el oro @15 esperado por línea también informa al minuto 10
        V["líneas + sorpresa"] = np.c_[V["líneas"], g - HGBL]
        EX = np.load(REPO / "data" / "analysis" / "brainstorm" / "hgb_lineas_extra.npz")   # brainstorm_42_xp_lineas.py
        esperado = [EX["xp_15"]] if m == 15 else [EX["gold_10"], EX["xp_10"]]
        V["líneas + esperado completo"] = np.c_[V["líneas + sorpresa"], *esperado]
    ok = ~np.isnan(np.c_[eq, g, x]).any(1)
    a, v = ai[ok[ai]], vi[ok[vi]]
    print(f"\nminuto {m}: ajuste {len(a):,}, V Master+ {len(v):,}")
    ref = None
    B = np.random.default_rng(1).integers(0, len(v), size=(300, len(v)))
    for n, F in V.items():
        mu, sd = F[a].mean(0), F[a].std(0) + 1e-9
        p = LogisticRegression(C=1.0, max_iter=2000).fit((F[a] - mu) / sd, y[a]).predict_proba((F[v] - mu) / sd)[:, 1]
        s = f"  {n:20s} AUC {roc_auc_score(y[v], p):.4f}  acierto {((p > .5) == y[v]).mean():.1%}"
        if ref is not None:
            d = np.array([roc_auc_score(y[v][b], p[b]) - roc_auc_score(y[v][b], ref[b]) for b in B])
            s += f"   Δ vs equipo {roc_auc_score(y[v], p) - roc_auc_score(y[v], ref):+.4f} [{np.percentile(d, 2.5):+.4f}; {np.percentile(d, 97.5):+.4f}]"
        else:
            ref = p
        print(s)
        if n == "líneas + sorpresa":      # cobertura y calibración del mejor modelo en vivo (para la demo)
            ok_, o = (p > .5) == y[v], np.argsort(-np.abs(p - .5))
            dec = pd.qcut(p, 10, labels=False)
            gq = pd.DataFrame({"p": p, "y": y[v], "d": dec}).groupby("d").agg(p=("p", "mean"), r=("y", "mean"), n=("y", "size"))
            print("     cobertura: " + "  ".join(f"{c:.0%}→{ok_[o[:int(c * len(v))]].mean():.1%}" for c in (.05, .1, .2, .3, .5)) +
                  f"   ECE {(np.abs(gq.p - gq.r) * gq.n).sum() / gq.n.sum():.4f}")
    F = V["líneas"]
    Fa, ya = np.r_[F[a], -F[a]], np.r_[y[a], 1 - y[a]]            # espejado: antisimétrico
    h = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=100,
                                       early_stopping=True, random_state=0).fit(Fa, ya)
    p = 0.5 * (h.predict_proba(F[v])[:, 1] + 1 - h.predict_proba(-F[v])[:, 1])
    d = np.array([roc_auc_score(y[v][b], p[b]) - roc_auc_score(y[v][b], ref[b]) for b in B])
    print(f"  {'HGB sobre líneas':20s} AUC {roc_auc_score(y[v], p):.4f}  acierto {((p > .5) == y[v]).mean():.1%}   Δ vs equipo "
          f"{roc_auc_score(y[v], p) - roc_auc_score(y[v], ref):+.4f} [{np.percentile(d, 2.5):+.4f}; {np.percentile(d, 97.5):+.4f}]")
    # ¿qué línea pesa más? logística solo con el oro por línea + previo
    F2 = np.c_[L, g]; mu, sd = F2[a].mean(0), F2[a].std(0)
    lr = LogisticRegression(C=1.0).fit((F2[a] - mu) / sd, y[a])
    print("  peso del oro por línea (por 1.000 de oro; BOT, JG, MID, TOP, SUP): " + "  ".join(f"{c / s_:.3f}" for c, s_ in zip(lr.coef_[0][1:], sd[1:])))
