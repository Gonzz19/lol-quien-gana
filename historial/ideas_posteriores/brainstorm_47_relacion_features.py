"""Brainstorming (PROJECT_SUMMARY.md §9.S.71): relación entre las variables del escalón 9 y poda por redundancia.

Se trabaja sobre las columnas densas del escalón 9 (todo menos el one-hot del draft, que queda siempre): las 85
variables por jugador × 5 líneas y los bloques extra (margen por campeón y línea, nivel histórico, W/L de la foto,
matchups, ROL_WR, Kalman, tempo, economía, modelo por línea, margen esperado por par). Todo se ajusta con las partidas
de entrenamiento de Master+ anteriores al 16/09 y se mide en V Master+.

  1. PCA: cuántas componentes hacen falta para el 50/80/90/95/99 % de la varianza, y AUC en V usando solo las
     primeras k componentes (arranca con todas y va cortando).
  2. Clustering jerárquico de VARIABLES (no K-Means): distancia 1 − |correlación|, enlace promedio. Se corta el árbol
     en cada vez menos grupos y cada grupo se reemplaza por una sola columna (la media de sus variables
     estandarizadas, con el signo alineado). AUC en V según la cantidad de grupos.
  3. Qué hay en cada grupo con un corte grueso: familias de variables que se agrupan, AUC del grupo solo y lo que
     cuesta sacarlo.

    python historial/ideas_posteriores/brainstorm_47_relacion_features.py     # ~25 min; tablas en data/analysis/brainstorm/relacion_*.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.brainstorm_base import cargar, extras_referencia  # noqa: E402

BR = REPO / "data" / "analysis" / "brainstorm"
LIN = ["BOT", "JG", "MID", "TOP", "SUP"]
D = cargar()
NM, y, ms = D["NM"], D["y"], D["master"]
tr = D["fi_v"][ms[D["fi_v"]]]
vi = np.flatnonzero(D["val"])

# ---- matriz densa y nombres (familia = bloque del que sale la columna)
X9 = extras_referencia(D, escalon=9)
anchos = [("margen jugador-campeón", 75), ("margen jugador-línea", 75), ("nivel histórico", 3)]
resto = X9.shape[1] - 150 - 3
g11 = np.load(BR / "gemini_g11.npz")
fijos = [("Kalman", g11["KALMAN_OUT"].shape[1] + g11["KALMAN_MRG"].shape[1]), ("tempo y objetivos", g11["TEMPO_OBJ"].shape[1]),
         ("economía 15→20", g11["ECON_MID"].shape[1]), ("modelo por línea", 6), ("margen esperado por par", 12)]
mat = np.load(BR / "gemini_mpr.npz")["MATCHUPS"].shape[1]
rol = np.load(BR / "gemini_nuevas_v2.npz")["ROL_WR"].shape[1]
foto = resto - mat - rol - sum(a for _, a in fijos)
anchos += [("W/L de la foto", foto), ("matchups", mat), ("ROL_WR", rol)] + fijos
assert sum(a for _, a in anchos) == X9.shape[1] and foto > 0


def familia_xj(c):
    if c.startswith(("tag_", "info_", "st_", "perf_")):
        return "campeón (estático y perfil)"
    if c.startswith(("acum_", "v20_")) or c in ("wr_acum", "wr_v20"):
        return "rendimiento del jugador"
    return "experiencia y hábitos"


nombres = [f"{c} [{LIN[l]}]" for l in range(5) for c in D["cols"]]
familias = [familia_xj(c) for l in range(5) for c in D["cols"]]
for n, a in anchos:
    nombres += [f"{n} #{i}" for i in range(a)]
    familias += [n] * a
Xall = np.c_[D["Xd"], X9].astype(np.float32)
mu, sd = Xall[tr].mean(0), Xall[tr].std(0)
vivas = sd > 1e-6
Xall, nombres, familias = (Xall[:, vivas] - mu[vivas]) / sd[vivas], np.array(nombres)[vivas], np.array(familias)[vivas]
P = Xall.shape[1]
DR = D["DRAFT"]
print(f"columnas densas: {P} (+ {DR.shape[1]} del draft); entrenamiento {len(tr):,}, V {len(vi):,}", flush=True)


def auc(F, con_draft=True):
    """Logística C = 0,05 sobre draft + F (F ya reducido; se reestandariza con el entrenamiento)."""
    m, s = F[tr].mean(0), F[tr].std(0) + 1e-6
    A = lambda i: sparse.hstack(([DR[i]] if con_draft else []) + [sparse.csr_matrix((F[i] - m) / s)]).tocsr()  # noqa: E731
    p = LogisticRegression(C=0.05, max_iter=4000).fit(A(tr), y[tr]).predict_proba(A(vi))[:, 1]
    return roc_auc_score(y[vi], p), ((p > .5) == y[vi]).mean(), p


a_ref, ac_ref, p_ref = auc(Xall)
print(f"referencia (todas): V AUC {a_ref:.4f} / {ac_ref:.1%}", flush=True)
BOOT = np.random.default_rng(1).integers(0, len(vi), size=(200, len(vi)))


def delta(p):
    d = np.array([roc_auc_score(y[vi][b], p[b]) - roc_auc_score(y[vi][b], p_ref[b]) for b in BOOT])
    return f"{np.percentile(d, 2.5):+.4f}; {np.percentile(d, 97.5):+.4f}"


# ---- 1. PCA
R = np.corrcoef(Xall[tr], rowvar=False)
w, Vec = np.linalg.eigh(R)
w, Vec = w[::-1].clip(0), Vec[:, ::-1]
ac = np.cumsum(w) / w.sum()
print("\n1. PCA sobre las columnas densas: componentes para explicar la varianza: " +
      "  ".join(f"{q:.0%}→{int(np.searchsorted(ac, q) + 1)}" for q in (.5, .8, .9, .95, .99)) +
      f"; dimensión efectiva (razón de participación) {w.sum() ** 2 / (w ** 2).sum():.0f}")
Z = Xall @ Vec.astype(np.float32)
f1 = []
for k in (P, 400, 300, 200, 150, 100, 60, 30, 15, 5):
    if k > P:
        continue
    a, c, p = auc(Z[:, :k])
    f1.append(dict(componentes=k, varianza=ac[k - 1], auc=a, acierto=c, delta=a - a_ref, ic=delta(p)))
    print(f"   primeras {k:4d} componentes ({ac[k - 1]:.1%} de la varianza): AUC {a:.4f} / {c:.1%}  Δ {a - a_ref:+.4f} [{f1[-1]['ic']}]", flush=True)
pd.DataFrame(f1).to_csv(BR / "relacion_pca.csv", index=False)
# ¿las componentes útiles son las de más varianza? AUC univariado de cada una de las 30 primeras
u = [abs(roc_auc_score(y[tr], Z[tr, j]) - .5) for j in range(min(60, P))]
print("   |AUC − 0,5| de cada componente sola (entrenamiento), las 10 mejores entre las 60 primeras: " +
      "  ".join(f"PC{j + 1} {u[j]:.3f}" for j in np.argsort(u)[::-1][:10]))

# ---- 2. clustering jerárquico de variables
dist = 1 - np.abs(R)
np.fill_diagonal(dist, 0)
Lk = linkage(squareform(np.clip((dist + dist.T) / 2, 0, None), checks=False), method="average")
sg1 = np.sign(R[:, :1].T)                                         # no se usa; el signo se alinea dentro de cada grupo


def reducir(k):
    et = fcluster(Lk, k, criterion="maxclust")
    cols = []
    for g in np.unique(et):
        idx = np.flatnonzero(et == g)
        s = np.sign(R[idx[0], idx]); s[s == 0] = 1                # alinear el signo con la primera variable del grupo
        cols.append((Xall[:, idx] * s).mean(1))
    return np.stack(cols, 1), et


f2 = []
print("\n2. clustering jerárquico de variables (1 − |correlación|, enlace promedio): cada grupo pasa a ser una columna")
for k in (500, 400, 300, 200, 150, 100, 60, 30, 15):
    if k >= P:
        continue
    F, et = reducir(k)
    a, c, p = auc(F)
    tam = np.bincount(et)[1:]
    f2.append(dict(grupos=F.shape[1], mayor=tam.max(), solos=int((tam == 1).sum()), auc=a, acierto=c, delta=a - a_ref, ic=delta(p)))
    print(f"   {F.shape[1]:4d} grupos (el mayor con {tam.max()} variables, {int((tam == 1).sum())} de una sola): AUC {a:.4f} / {c:.1%}  "
          f"Δ {a - a_ref:+.4f} [{f2[-1]['ic']}]", flush=True)
pd.DataFrame(f2).to_csv(BR / "relacion_clusters_poda.csv", index=False)

# ---- 3. qué hay en cada grupo (corte grueso) y cuánto pesa
K3 = 15
F, et = reducir(K3)
filas = []
print(f"\n3. corte en {K3} grupos: qué se agrupa, AUC del grupo solo (sin draft) y costo de sacarlo del modelo completo")
for j, g in enumerate(np.unique(et)):
    idx = np.flatnonzero(et == g)
    fam = pd.Series(familias[idx]).value_counts()
    rmed = np.abs(R[np.ix_(idx, idx)])[np.triu_indices(len(idx), 1)].mean() if len(idx) > 1 else 1.0
    a_solo = auc(Xall[:, idx], con_draft=False)[0]
    a_sin, _, p_sin = auc(np.delete(Xall, idx, axis=1))
    desc = "; ".join(f"{k} ({v})" for k, v in fam.head(3).items())
    ej = ", ".join(nombres[idx][:3])
    filas.append(dict(grupo=j + 1, variables=len(idx), corr_media=rmed, familias=desc, ejemplos=ej, auc_solo=a_solo,
                      costo_sacarlo=a_sin - a_ref, ic=delta(p_sin)))
    print(f"   G{j + 1:02d} ({len(idx):3d} vars, |r| medio {rmed:.2f}): {desc} | solo {a_solo:.4f} | sin él {a_sin - a_ref:+.4f} [{filas[-1]['ic']}]", flush=True)
pd.DataFrame(filas).to_csv(BR / "relacion_grupos.csv", index=False)
pd.DataFrame({"variable": nombres, "familia": familias, f"grupo_{K3}": et}).to_csv(BR / "relacion_variables.csv", index=False)
# correlación media entre familias (para ver qué bloques dicen lo mismo)
fams = sorted(set(familias))
M = pd.DataFrame([[np.abs(R[np.ix_(familias == a, familias == b)]).mean() for b in fams] for a in fams], index=fams, columns=fams)
M.to_csv(BR / "relacion_familias_corr.csv")
print("\n   |correlación| media entre familias (pares más altos, sin la diagonal):")
pares = [(M.iloc[i, j], fams[i], fams[j]) for i in range(len(fams)) for j in range(i + 1, len(fams))]
for r, a, b in sorted(pares, reverse=True)[:8]:
    print(f"     {r:.3f}  {a}  ↔  {b}")
