"""
Curva de aprendizaje del modelo de quién gana: cómo cambia el AUC en V según cuántas partidas se
usan para ajustarlo (PROJECT_SUMMARY.md §9.S.48).

Las variables de cada jugador (margen, experiencia, pool…) se calculan siempre con todo su
historial previo (caché de `embeddings_partidas.py`). Lo que varía es cuántas partidas del ajuste
entran a la logística:

* *recientes:* las N inmediatamente anteriores a V (lo que se haría en la práctica);
* *al azar:* N al azar dentro del ajuste (3 semillas), para separar cantidad de antigüedad.

Dos modelos, con la logística del EDA (C = 0,05, estandarización con las filas de ajuste):
el híbrido (draft campeón × línea + 85 variables por línea) y solo jugadores (sin el draft).
Además, con N = todo, se prueba C en {0,01; 0,05; 0,2} para ver si la regularización limita.

    python scripts/analysis/curva_aprendizaje.py     # ~10 min, 4 hilos
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.features.draft import draft_matrix_by_lane  # noqa: E402
ap = argparse.ArgumentParser()
ap.add_argument("--hilos", type=int, default=4)
ap.add_argument("--B", type=int, default=200)
args = ap.parse_args()
T0 = time.time()
SAL = REPO / "data" / "analysis" / "curva_aprendizaje"
SAL.mkdir(parents=True, exist_ok=True)

Z = np.load(REPO / "data" / "analysis" / "embeddings_partidas" / "variables.npz", allow_pickle=True)
Xj, y, val, champ, t = Z["Xj"], Z["y"], Z["val"], Z["champ"], Z["t"]
NM, fit = len(y), ~Z["val"]
Xd = (Xj[:, 0] - Xj[:, 1]).reshape(NM, -1).astype(np.float32)
DRAFT = draft_matrix_by_lane(champ)
fi = np.flatnonzero(fit)
fi = fi[np.argsort(t[fi])]                      # ajuste en orden temporal
vi = np.flatnonzero(val)
yv = y[vi]


def ajustar(filas, con_draft, C=0.05):
    mu, sd = Xd[filas].mean(0), Xd[filas].std(0) + 1e-6

    def X(f):
        D = sparse.csr_matrix((Xd[f] - mu) / sd)
        return sparse.hstack([DRAFT[f], D]).tocsr() if con_draft else D

    lr = LogisticRegression(C=C, max_iter=3000).fit(X(filas), y[filas])
    return lr.predict_proba(X(vi))[:, 1]


TAM = [2_000, 5_000, 10_000, 20_000, 40_000, 80_000, 120_000, len(fi)]
trabajos = []
for n in TAM:
    for modelo in ("híbrido", "solo jugadores"):
        trabajos.append(("recientes", n, 0, modelo, 0.05))
        if n < len(fi):
            for s in range(3):
                trabajos.append(("al azar", n, s, modelo, 0.05))
for C in (0.01, 0.2):
    for modelo in ("híbrido", "solo jugadores"):
        trabajos.append(("recientes", len(fi), 0, modelo, C))


def correr(tr):
    modo, n, s, modelo, C = tr
    filas = fi[-n:] if modo == "recientes" else np.sort(np.random.default_rng(s).choice(fi, n, replace=False))
    p = ajustar(filas, modelo == "híbrido", C)
    desde = pd.to_datetime(t[filas].min(), unit="ms")
    return {"modo": modo, "partidas": n, "semilla": s, "modelo": modelo, "C": C, "desde": f"{desde:%d/%m/%Y}",
            "AUC": roc_auc_score(yv, p), "acierto": ((p > 0.5) == yv).mean(), "log loss": log_loss(yv, p), "p": p}


res = Parallel(n_jobs=args.hilos, backend="threading")(delayed(correr)(tr) for tr in trabajos)
print(f"ajustes listos ({time.time() - T0:.0f}s)")

R = pd.DataFrame([{k: v for k, v in r.items() if k != "p"} for r in res])
R.to_csv(SAL / "resultado.tsv", sep="\t", index=False)

# IC del AUC (bootstrap) para la serie "recientes", C = 0,05, y Δ contra usar todo
BOOT = np.random.default_rng(1).integers(0, len(yv), size=(args.B, len(yv)))
P = {(r["modo"], r["partidas"], r["semilla"], r["modelo"], r["C"]): r["p"] for r in res}
filas, auc_ic = [], {}
for modelo in ("híbrido", "solo jugadores"):
    p_todo = P[("recientes", len(fi), 0, modelo, 0.05)]
    auc_ic[modelo] = []
    for n in TAM:
        p = P[("recientes", n, 0, modelo, 0.05)]
        aucs_b = np.array([roc_auc_score(yv[s], p[s]) for s in BOOT])
        auc_ic[modelo].append((float(np.percentile(aucs_b, 2.5)), float(np.percentile(aucs_b, 97.5))))
        d = np.array([roc_auc_score(yv[s], p[s]) - roc_auc_score(yv[s], p_todo[s]) for s in BOOT])
        az = R[(R.modo == "al azar") & (R.partidas == n) & (R.modelo == modelo)].AUC
        fila = R[(R.modo == "recientes") & (R.partidas == n) & (R.modelo == modelo) & (R.C == 0.05)].iloc[0]
        filas.append({"modelo": modelo, "partidas": n, "desde": fila.desde, "AUC recientes": fila.AUC,
                      "Δ vs todo": roc_auc_score(yv, p) - roc_auc_score(yv, p_todo),
                      "IC lo": np.percentile(d, 2.5), "IC hi": np.percentile(d, 97.5),
                      "acierto": fila.acierto,
                      "AUC al azar (media 3)": az.mean() if len(az) else np.nan,
                      "al azar (rango)": az.max() - az.min() if len(az) else np.nan})
T = pd.DataFrame(filas)
T.to_csv(SAL / "curva.tsv", sep="\t", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 20)
print(T.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nRegularización con todo el ajuste (recientes):")
print(R[(R.partidas == len(fi)) & (R.modo == "recientes")][["modelo", "C", "AUC", "acierto", "log loss"]]
      .sort_values(["modelo", "C"]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
fig, ax = plt.subplots(figsize=(8.4, 4.4))
col = {"híbrido": "#2a78d6", "solo jugadores": "#eb6834"}
for modelo in col:
    s = T[T.modelo == modelo]
    lo_m = [c[0] for c in auc_ic[modelo]]
    hi_m = [c[1] for c in auc_ic[modelo]]
    ax.plot(s.partidas, s["AUC recientes"], "-o", color=col[modelo], label=f"{modelo}: las más recientes (IC 95 % boot.)")
    ax.fill_between(s.partidas, lo_m, hi_m, color=col[modelo], alpha=0.14)
    ax.plot(s.partidas, s["AUC al azar (media 3)"], "--", color=col[modelo], label=f"{modelo}: al azar (media 3 semillas)")
ax.set_xscale("log")
ax.set_xlabel("partidas para ajustar el modelo (escala log)")
ax.set_ylabel(f"AUC en V (bandas: IC 95 % bootstrap, B = {args.B} partidas)")
ax.set_title("Curva de aprendizaje")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(SAL / "curva.png", dpi=130)
print(f"[+] {SAL} ({time.time() - T0:.0f}s)")
