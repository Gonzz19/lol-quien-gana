"""Brainstorming (PROJECT_SUMMARY.md §9.S.71): qué se le puede mejorar a la segunda etapa en vivo (minutos 10, 15 y 20).

Punto de partida: el mejor modelo de `brainstorm_40_en_vivo_lineas.py` ("vigente"): logística sobre el logit previo del
escalón 9, Δoro y ΔXP de cada línea, Δkills del equipo y lo que el modelo por línea esperaba antes de jugar. Se prueba,
de a un agregado por vez y acumulando:

  1. más estado por línea: CS, daño, nivel, kills, muertes y asistencias (el vigente solo mira oro y XP);
  2. trayectoria: el oro y la XP por línea de los minutos anteriores (a los 15, los del 10; a los 20, los del 10 y 15);
  3. medias del jugador en la línea (brainstorm_45): cuánto le saca al rival a esa altura en sus partidas anteriores;
  4. forma: logit previo × Δoro del equipo, mayor y menor ventaja de línea, desvío entre líneas, Δoro con raíz;
  5. ajustar con más partidas: sumar las Diamond de A2-A4 (el vigente ajusta solo con las Master+);
  6. mezcla con un HGB espejado (promedio de logits).

Ajuste con las predicciones hacia adelante de A2-A4, medición en V Master+ (IC por bootstrap pareado contra el vigente).
Réplica que no mira V: ajuste con A2-A3 y medición en A4 Master+. Solo partidas que llegaron al minuto.

    python historial/ideas_posteriores/brainstorm_45_medias_jugador_linea.py   # antes: deja las cachés
    python historial/en_vivo/brainstorm_46_en_vivo_mejoras.py        # ~3 min
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from historial.ideas_posteriores.brainstorm_45_medias_jugador_linea import MINS, PRED, STATS, construir  # noqa: E402
from scripts.analysis.brainstorm_base import cargar  # noqa: E402

BR = REPO / "data" / "analysis" / "brainstorm"
D = cargar()
ms, y, Z, NM, bl = D["master"], D["y"], D["Z"], D["NM"], D["bloque"]
PJ, cols, ST = construir(D)
pr = np.load(PRED)
lg = lambda p: np.log(p / (1 - p))  # noqa: E731
vi = np.flatnonzero(D["val"])
L = np.full(NM, np.nan); L[bl >= 1] = lg(pr["po"][bl >= 1]); L[vi] = lg(pr["pv"])
HGBL = np.load(BR / "gemini_hgb_lineas.npz")["HGB"][:, :5]
EX = np.load(BR / "hgb_lineas_extra.npz") if (BR / "hgb_lineas_extra.npz").exists() else None
dPJ = PJ[:, 0] - PJ[:, 1]
st = lambda s, m: ST[:, :, STATS.index(s), MINS.index(m)]       # noqa: E731  (NM, 5) azul − rojo por línea
pj = lambda c: dPJ[:, :, cols.index(c)]                          # noqa: E731


def lr_pred(F, a, v, C=1.0):
    mu, sd = F[a].mean(0), F[a].std(0) + 1e-9
    return LogisticRegression(C=C, max_iter=5000).fit((F[a] - mu) / sd, y[a]).predict_proba((F[v] - mu) / sd)[:, 1]


def hgb_pred(F, a, v):
    h = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=100,
                                       l2_regularization=1.0, early_stopping=True, random_state=0)
    h.fit(np.r_[F[a], -F[a]], np.r_[y[a], 1 - y[a]])
    return 0.5 * (h.predict_proba(F[v])[:, 1] + 1 - h.predict_proba(-F[v])[:, 1])


def fila(n, p, yv, ref, B):
    s = f"  {n:44s} AUC {roc_auc_score(yv, p):.4f}  acierto {((p > .5) == yv).mean():.1%}  log loss {log_loss(yv, p):.4f}"
    if ref is not None:
        d = np.array([roc_auc_score(yv[b], p[b]) - roc_auc_score(yv[b], ref[b]) for b in B])
        s += f"   Δ {roc_auc_score(yv, p) - roc_auc_score(yv, ref):+.4f} [{np.percentile(d, 2.5):+.4f}; {np.percentile(d, 97.5):+.4f}]"
    return s


for m in MINS:
    g, x, kt = st("gold", m), st("xp", m), st("kills", m).sum(1)
    ant = [k for k in MINS if k < m]
    V = {}
    V["equipo (brainstorm_39)"] = np.c_[L, g.sum(1), x.sum(1), kt]
    vig = [L, g, x, kt, g - HGBL]
    if EX is not None:
        vig += [EX["xp_15"]] if m == 15 else ([EX["gold_10"], EX["xp_10"]] if m == 10 else [EX["xp_15"], EX["gold_10"]])
    V["vigente (líneas + esperado)"] = np.c_[tuple(vig)]
    acum = [V["vigente (líneas + esperado)"]]
    acum.append(np.c_[tuple(st(s, m) for s in ("cs", "damage", "level", "kills", "deaths", "assists"))])
    V["1. + CS, daño, nivel y KDA por línea"] = np.c_[tuple(acum)]
    if ant:
        acum.append(np.c_[tuple(st(s, k) for k in ant for s in ("gold", "xp"))])
        V["2. + trayectoria (minutos anteriores)"] = np.c_[tuple(acum)]
    acum.append(np.c_[tuple(pj(f"d_{s}_{m}") for s in ("gold", "xp", "cs"))])
    V["3. + medias del jugador en la línea"] = np.c_[tuple(acum)]
    gs = g.sum(1)
    forma = np.c_[L * gs, g.max(1), g.min(1), g.std(1), np.sign(gs) * np.sqrt(np.abs(gs))]
    V["4. + forma (interacción y no linealidad)"] = np.c_[tuple(acum + [forma])]
    solo_medias = np.c_[V["vigente (líneas + esperado)"], acum[-1]]
    ok = ~np.isnan(np.c_[tuple(V.values())]).any(1)
    cortes = {"V Master+ (ajuste A2-A4 Master+)": (np.flatnonzero((bl >= 1) & ms & ok), vi[ok[vi]]),
              "A4 Master+ (ajuste A2-A3 Master+, no mira V)": (np.flatnonzero(np.isin(bl, (1, 2)) & ms & ok), np.flatnonzero((bl == 3) & ms & ok))}
    con_d = {"V Master+ (ajuste A2-A4 Master+)": np.flatnonzero((bl >= 1) & ok),
             "A4 Master+ (ajuste A2-A3 Master+, no mira V)": np.flatnonzero(np.isin(bl, (1, 2)) & ok)}
    print(f"\n================ minuto {m}")
    for nc, (a, v) in cortes.items():
        yv = y[v]
        B = np.random.default_rng(1).integers(0, len(v), size=(300, len(v)))
        print(f"--- {nc}: ajuste {len(a):,}, medición {len(v):,}")
        P = {n: lr_pred(F, a, v) for n, F in V.items()}
        ref = P["vigente (líneas + esperado)"]
        for n, p in P.items():
            print(fila(n, p, yv, None if n.startswith("vigente") else ref, B))
        print(fila("vigente + solo las medias del jugador", lr_pred(solo_medias, a, v), yv, ref, B))
        mejor = "1. + CS, daño, nivel y KDA por línea" if not ant else "2. + trayectoria (minutos anteriores)"
        F = V[mejor]
        print(fila(f"{mejor[:2]} con C = 0,1", lr_pred(F, a, v, C=0.1), yv, ref, B))
        ad = con_d[nc]
        p_d = lr_pred(F, ad, v)
        print(fila(f"5. {mejor[:2]} ajustado también con Diamond ({len(ad):,})", p_d, yv, ref, B))
        print(fila("5b. vigente ajustado también con Diamond", lr_pred(V["vigente (líneas + esperado)"], ad, v), yv, ref, B))
        p_h = hgb_pred(F, ad, v)
        print(fila("   HGB espejado solo (con Diamond)", p_h, yv, ref, B))
        p_mix = 1 / (1 + np.exp(-(0.7 * lg(p_d) + 0.3 * lg(np.clip(p_h, 1e-4, 1 - 1e-4)))))
        print(fila("6. mezcla 0,7 logística + 0,3 HGB", p_mix, yv, ref, B))
        if nc.startswith("V"):
            okv, o = (p_d > .5) == yv, np.argsort(-np.abs(p_d - .5))
            print("     cobertura del 5: " + "  ".join(f"{c:.0%}→{okv[o[:int(c * len(v))]].mean():.1%}" for c in (.05, .1, .2, .5)))
