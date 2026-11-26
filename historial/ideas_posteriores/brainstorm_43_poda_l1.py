"""Poda con L1 del modelo final de quién gana (escalón 9; PROJECT_SUMMARY.md §9.S.69).

Pregunta: ¿cuántas de las ~1.500 columnas de la logística final hacen falta? Se ajusta la misma matriz
(draft campeón × línea + 85 variables por jugador + extras del escalón 9, estandarizadas igual que en
`brainstorm_base.ajustar_predecir`) con penalización L1 para una grilla de C. Para cada C se mide:

* la logística L1 tal cual (coeficientes encogidos y varios en cero), y
* el reajuste: logística L2 (C = 0,05, la del modelo final) solo con las columnas que L1 dejó vivas.

Evaluación con la regla de la noche: sub-bloques A2-A4 hacia adelante (Master+) y V Master+, con Δ contra
la logística L2 completa e IC por bootstrap pareado en V. El C se elige mirando los bloques; V confirma.

    python historial/ideas_posteriores/brainstorm_43_poda_l1.py            # ~15 min
Salida: data/analysis/brainstorm/poda_l1.csv y poda_l1_columnas.csv (qué queda vivo en el ajuste de V).
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.brainstorm_base import cargar, extras_referencia  # noqa: E402

warnings.filterwarnings("ignore")
OUT = REPO / "data" / "analysis" / "brainstorm"
CS = [0.0005, 0.001, 0.002, 0.004, 0.008, 0.02, 0.05]     # C de la L1 (más chico = poda más fuerte)
C_L2 = 0.05

D = cargar()
X9 = extras_referencia(D, escalon=9)
# límites de cada bloque de extras, en el orden en que los apila extras_referencia
cortes = [extras_referencia(D, escalon=3).shape[1], extras_referencia(D, escalon=4).shape[1],
          extras_referencia(D, escalon=4, rol_wr=True).shape[1], extras_referencia(D, escalon=6).shape[1],
          extras_referencia(D, escalon=7).shape[1], X9.shape[1]]
NOMBRES = ["margen camp/línea + nivel hist. + W/L foto", "matchups", "winrate por rol", "Kalman + tempo + economía",
           "oro @15 predicho (HGB)", "margen esperado por par"]
n_draft, n_vars = D["DRAFT"].shape[1], D["Xd"].shape[1]
grupo = np.r_[np.zeros(n_draft, int), np.ones(n_vars, int),
              np.repeat(np.arange(2, 2 + len(cortes)), np.diff(np.r_[0, cortes]))]
GRUPOS = ["draft (campeón × línea)", "85 variables por jugador × 5 líneas"] + NOMBRES
y, master = D["y"], D["master"]


def matriz(tr, te):
    """Misma matriz que `ajustar_predecir(..., extra=X9, extra_post=False, con_mapa=False)`."""
    Xd = D["Xd"]
    mu, sd = Xd[tr].mean(0), Xd[tr].std(0) + 1e-6
    me, se = X9[tr].mean(0), X9[tr].std(0) + 1e-6
    arm = lambda f: sparse.hstack([D["DRAFT"][f], sparse.csr_matrix((Xd[f] - mu) / sd),  # noqa: E731
                                   sparse.csr_matrix((X9[f] - me) / se)]).tocsr().astype(np.float32)
    return arm(tr), arm(te)


def una_c(c, A, B, ytr):
    l1 = LogisticRegression(penalty="l1", solver="liblinear", C=c, tol=1e-4, max_iter=2000).fit(A, ytr)
    viva = np.flatnonzero(l1.coef_[0] != 0)
    p1 = l1.predict_proba(B)[:, 1]
    if len(viva) == 0:
        return c, viva, p1, np.full(B.shape[0], ytr.mean())
    l2 = LogisticRegression(C=C_L2, max_iter=4000).fit(A[:, viva], ytr)
    return c, viva, p1, l2.predict_proba(B[:, viva])[:, 1]


def particion(tr, te):
    A, B = matriz(tr, te)
    ref = LogisticRegression(C=C_L2, max_iter=4000).fit(A, y[tr]).predict_proba(B)[:, 1]
    res = Parallel(n_jobs=2, backend="threading")(delayed(una_c)(c, A, B, y[tr]) for c in CS)
    return ref, {c: (viva, p1, p2) for c, viva, p1, p2 in res}


t0 = time.time()
fi, bl = D["fi"], D["bloque"]
BLO = {}
for b in D["EV"]:
    tr = fi[bl[fi] < b]
    tr = tr[master[tr]]
    te = np.flatnonzero((bl == b) & master)
    BLO[b] = (te, *particion(tr, te))
    print(f"bloque A{b + 1}: {len(tr):,} → {len(te):,}  ({time.time() - t0:.0f} s)", flush=True)
tr_v = D["fi_v"][master[D["fi_v"]]]
vi = np.flatnonzero(D["val"])
ref_v, RV = particion(tr_v, vi)
yv = y[vi]
print(f"V: {len(tr_v):,} → {len(vi):,}  ({time.time() - t0:.0f} s)", flush=True)

BOOT = np.random.default_rng(1).integers(0, len(yv), size=(500, len(yv)))
auc_ref_boot = np.array([roc_auc_score(yv[s], ref_v[s]) for s in BOOT])


def fila(nombre, c, viva, pv, k):
    """k: 1 = L1 tal cual, 2 = reajuste L2 sobre las columnas vivas."""
    d_b = [roc_auc_score(y[te], r[c][k]) - roc_auc_score(y[te], ref) for te, ref, r in BLO.values()]
    d = np.array([roc_auc_score(yv[s], pv[s]) for s in BOOT]) - auc_ref_boot
    f = {"variante": nombre, "C L1": c, "columnas": len(viva)}
    for g, n in enumerate(GRUPOS):
        f[n] = int((grupo[viva] == g).sum())
    f.update({"AUC V": roc_auc_score(yv, pv), "acierto V": ((pv > 0.5) == yv).mean(),
              "Δ V": roc_auc_score(yv, pv) - roc_auc_score(yv, ref_v),
              "Δ V lo": np.percentile(d, 2.5), "Δ V hi": np.percentile(d, 97.5),
              "Δ bloques (media)": np.mean(d_b), "bloques que empeora": int(sum(x < 0 for x in d_b))})
    return f


filas =[{"variante": "L2 completa (modelo final)", "C L1": np.nan, "columnas": len(grupo),
          **{n: int((grupo == g).sum()) for g, n in enumerate(GRUPOS)},
          "AUC V": roc_auc_score(yv, ref_v), "acierto V": ((ref_v > 0.5) == yv).mean(), "Δ V": 0.0, "Δ V lo": 0.0,
          "Δ V hi": 0.0, "Δ bloques (media)": 0.0, "bloques que empeora": 0}]
for c in CS:
    viva, p1, p2 = RV[c]
    filas.append(fila("L1", c, viva, p1, 1))
    filas.append(fila("L1 → reajuste L2", c, viva, p2, 2))
T = pd.DataFrame(filas)
T.to_csv(OUT / "poda_l1.csv", index=False)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
m = T[["variante", "C L1", "columnas", "AUC V", "acierto V", "Δ V", "Δ V lo", "Δ V hi", "Δ bloques (media)", "bloques que empeora"]].copy()
for col in ("AUC V", "Δ V", "Δ V lo", "Δ V hi", "Δ bloques (media)"):
    m[col] = m[col].map(lambda x: f"{x:+.4f}" if col.startswith("Δ") else f"{x:.4f}")
m["acierto V"] = m["acierto V"].map(lambda x: f"{x:.2%}")
print(m.to_string(index=False))
print("\nColumnas vivas por grupo (ajuste de V):")
print(T[T.variante != "L1"][["C L1", "columnas"] + GRUPOS].to_string(index=False))

# qué variables por jugador sobreviven, por C (ajuste de V): nombre de la variable y línea
cols = D["cols"]
nv = len(cols)
det = []
for c in CS:
    viva = RV[c][0]
    v = viva[grupo[viva] == 1] - n_draft
    # Xd = (azul − rojo).reshape(NM, 5 líneas × nv variables): la variable es el índice módulo nv
    for nombre, n in pd.Series([cols[i % nv] for i in v]).value_counts().items():
        det.append({"C L1": c, "variable": nombre, "líneas vivas": int(n)})
pd.DataFrame(det).to_csv(OUT / "poda_l1_columnas.csv", index=False)
print(f"\n[+] {OUT / 'poda_l1.csv'}  ({time.time() - t0:.0f} s)")
