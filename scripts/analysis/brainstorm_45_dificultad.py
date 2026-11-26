"""
Idea 45 del brainstorming (pedido del usuario, 2026-11-16): ¿hay algo antes de la partida (clusters, PCA, un
espacio latente) que distinga las partidas difíciles de las fáciles?

"Difícil" no puede ser "el modelo duda": eso es circular (|logit| chico). La pregunta útil es si, **a igual
confianza**, hay partidas donde el modelo acierta más o menos de lo que promete. Se mide así:

* Predicción de referencia: híbrido + mapa, fuera de la muestra en los sub-bloques A2-A4 (descubrimiento) y en
  V Master+ (confirmación; el modelo de V se ajusta antes del 16/09). V no se usa para elegir nada.
* Contexto simétrico de la partida (no dice quién gana, describe el tipo de partida): media del lobby de las 85
  variables por jugador, su dispersión entre los 10, la magnitud |azul − rojo| por variable, el nivel (mapa de
  LP) medio y su dispersión, y el tier medio. Se estandariza con el ajuste.
* Exceso de acierto de un grupo = acierto observado − acierto esperado (media de max(p, 1 − p)). Si el modelo
  está calibrado y el grupo no aporta nada, el exceso es ≈ 0 en todos los grupos.
* Espacios: (1) PCA de 20 componentes (quintiles de las 5 primeras); (2) K-Means con k = 8 sobre la PCA;
  (3) un meta-modelo HGB que predice si el híbrido acierta a partir de |logit| + contexto, contra |logit| solo.
* Uso práctico: (a) cobertura selectiva (acierto en el 20/40/60 % de partidas más seguras según |logit| o según
  el meta-modelo); (b) recalibración heterogénea: logística de y sobre logit, logit × componentes y
  componentes, ajustada en los bloques y medida en V contra la referencia.

    python scripts/analysis/brainstorm_45_dificultad.py      # ~10 min
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.brainstorm_base import ajustar_predecir, cargar, evaluar  # noqa: E402

SALIDA = REPO / "data" / "analysis" / "brainstorm"
CACHE_P = SALIDA / "dificultad_p.npz"
T0 = time.time()
log = lambda s: print(f"[{time.time() - T0:5.0f}s] {s}", flush=True)  # noqa: E731
warnings.filterwarnings("ignore", category=RuntimeWarning)


def contexto(D):
    X = D["Xj"].reshape(D["NM"], 10, -1).astype(np.float32)         # (partidas, 10 jugadores, 85)
    media = X.mean(1)
    disp = X.std(1)
    dif = np.abs(D["Xj"][:, 0].mean(1) - D["Xj"][:, 1].mean(1))
    lp = D["B"]["lp_abs"].reshape(D["NM"], 10)
    extra = np.c_[np.nanmean(lp, 1), np.nanstd(lp, 1), D["tier_lobby"], np.nanstd(D["Z"]["tnum"], 1)]
    C = np.c_[media, disp, dif, extra].astype(np.float32)
    nombres = ([f"media_{c}" for c in D["cols"]] + [f"disp_{c}" for c in D["cols"]] + [f"dif_{c}" for c in D["cols"]]
               + ["lp_medio", "lp_disp", "tier_medio", "tier_disp"])
    fi = D["fi_v"]
    mu = np.nanmean(C[fi], 0)
    C = np.where(np.isnan(C), mu, C)
    sd = C[fi].std(0) + 1e-6
    return (C - C[fi].mean(0)) / sd, nombres


def tabla_grupos(g, p, y, idx, etiqueta):
    """acierto, esperado, exceso y pendiente de calibración por grupo (solo filas idx)."""
    filas = []
    for k in np.unique(g[idx]):
        m = idx[g[idx] == k]
        pk, yk = p[m], y[m]
        lg = np.log(pk / (1 - pk)).reshape(-1, 1)
        pend = LogisticRegression(C=1e4).fit(lg, yk).coef_[0, 0] if len(m) > 200 else np.nan
        ac, esp = ((pk > 0.5) == yk).mean(), np.maximum(pk, 1 - pk).mean()
        filas.append({"conjunto": etiqueta, "grupo": k, "partidas": len(m), "acierto": ac, "esperado": esp,
                      "exceso": ac - esp, "pendiente": pend, "AUC": roc_auc_score(yk, pk)})
    return pd.DataFrame(filas)


def main():
    D = cargar()
    y = D["y"]
    if CACHE_P.exists():
        C_ = np.load(CACHE_P)
        po, pv = C_["po"], C_["pv"]
    else:
        po, pv = evaluar(D, lambda tr, te: ajustar_predecir(D, tr, te))
        np.savez_compressed(CACHE_P, po=po, pv=pv)
    log("predicciones de referencia listas")
    B = np.flatnonzero(~np.isnan(po) & D["master"])                     # descubrimiento: sub-bloques Master+
    V = np.flatnonzero(D["val"])
    p = po.copy()
    p[V] = pv
    p = np.clip(p, 1e-4, 1 - 1e-4)
    lg = np.log(p / (1 - p))
    C, nombres = contexto(D)
    log(f"contexto: {C.shape[1]} variables; bloques {len(B):,}, V {len(V):,}")

    # 1) PCA ajustada con el ajuste de V (sin V)
    pca = PCA(20, random_state=0).fit(C[D["fi_v"]])
    Zp = pca.transform(C).astype(np.float32)
    log(f"PCA: varianza explicada 5 primeras {pca.explained_variance_ratio_[:5].round(3)}")
    out = []
    for j in range(5):
        cortes = np.quantile(Zp[B, j], [0.2, 0.4, 0.6, 0.8])
        q = np.digitize(Zp[:, j], cortes) + 1
        for nom, idx in (("bloques", B), ("V", V)):
            out.append(tabla_grupos(q, p, y, idx, f"PC{j + 1} {nom}").assign(espacio=f"PC{j + 1}"))
        top = np.argsort(-np.abs(pca.components_[j]))[:5]
        log(f"  PC{j + 1}: {', '.join(f'{nombres[i]} ({pca.components_[j, i]:+.2f})' for i in top)}")
    # 2) K-Means
    km = KMeans(8, n_init=5, random_state=0).fit(Zp[D["fi_v"]])
    g = km.predict(Zp)
    for nom, idx in (("bloques", B), ("V", V)):
        out.append(tabla_grupos(g, p, y, idx, f"K-Means {nom}").assign(espacio="K-Means"))
    T = pd.concat(out, ignore_index=True)
    T.to_csv(SALIDA / "dificultad_grupos.csv", index=False)
    # ¿el exceso por grupo se repite de bloques a V?
    rep = []
    for esp_, s in T.groupby("espacio"):
        b = s[s.conjunto.str.endswith("bloques")].set_index("grupo")
        v = s[s.conjunto.str.endswith(" V")].set_index("grupo")
        jj = b.index.intersection(v.index)
        rep.append({"espacio": esp_, "rango exceso bloques": b.exceso.max() - b.exceso.min(),
                    "rango exceso V": v.exceso.max() - v.exceso.min(),
                    "corr exceso bloques-V": np.corrcoef(b.exceso[jj], v.exceso[jj])[0, 1],
                    "rango acierto V": v.acierto.max() - v.acierto.min(),
                    "corr pendiente bloques-V": np.corrcoef(b.pendiente[jj], v.pendiente[jj])[0, 1]})
    R = pd.DataFrame(rep)
    print("\n¿El exceso de acierto por grupo se repite en V?")
    print(R.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print("\nK-Means en V:")
    print(T[T.conjunto == "K-Means V"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 3) meta-modelo: ¿acierta el híbrido? |logit| solo contra |logit| + contexto
    ok = ((p > 0.5) == y).astype(int)
    hgb = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,  # noqa: E731
                                                 min_samples_leaf=300, l2_regularization=1.0, random_state=0)
    a = np.abs(lg).reshape(-1, 1)
    meta = {"|logit| solo": a, "|logit| + PCA 20": np.c_[a, Zp], "|logit| + contexto completo": np.c_[a, C]}
    filas = []
    sc = {}
    for nom, X in meta.items():
        m = hgb().fit(X[B], ok[B])
        s = m.predict_proba(X[V])[:, 1]
        sc[nom] = s
        fila = {"meta-modelo": nom, "AUC de 'acierta' en V": roc_auc_score(ok[V], s)}
        for cov in (0.2, 0.4, 0.6):
            k = int(cov * len(V))
            fila[f"acierto en el {cov:.0%} más seguro"] = ok[V][np.argsort(-s)[:k]].mean()
        filas.append(fila)
    M = pd.DataFrame(filas)
    print("\nMeta-modelo (entrenado en los bloques, medido en V):")
    print(M.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    M.to_csv(SALIDA / "dificultad_meta.csv", index=False)

    # 4) recalibración heterogénea: y ~ logit + logit × PC + PC (bloques → V)
    filas = []
    for nom, Xf in (("referencia (logit)", lambda i: lg[i].reshape(-1, 1)),
                    ("logit + PC5 + logit × PC5", lambda i: np.c_[lg[i], Zp[i, :5], lg[i, None] * Zp[i, :5]]),
                    ("logit + PC20 + logit × PC20", lambda i: np.c_[lg[i], Zp[i], lg[i, None] * Zp[i]]),
                    ("logit + clusters + logit × clusters",
                     lambda i: np.c_[lg[i], np.eye(8)[g[i]], lg[i, None] * np.eye(8)[g[i]]])):
        r = LogisticRegression(C=1.0, max_iter=3000).fit(Xf(B), y[B])
        q = r.predict_proba(Xf(V))[:, 1]
        filas.append({"recalibración": nom, "AUC V": roc_auc_score(y[V], q), "acierto V": ((q > 0.5) == y[V]).mean()})
    Rc = pd.DataFrame(filas)
    print("\nRecalibración heterogénea (ajustada en bloques, medida en V):")
    print(Rc.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    Rc.to_csv(SALIDA / "dificultad_recalibracion.csv", index=False)
    log("listo")


if __name__ == "__main__":
    main()
