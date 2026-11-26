"""
K-Means de sklearn y e-means de FAISS con una misma interfaz (PROJECT_SUMMARY.md §9.S.53).

e-means (Bessa, Douze, Jégou et al., arXiv 2611.09501, nov. 2026; `faiss.contrib.e_means`
desde FAISS 1.16.0) no recorre todos los puntos en cada paso como Lloyd: guarda por
cluster una suma $S_j$ y un conteo $N_j$ y los actualiza por lotes con una media móvil
exponencial,

$$N_j \\leftarrow (1-\\alpha_t) N_j + \\alpha_t n_j, \\qquad S_j \\leftarrow (1-\\alpha_t) S_j + \\alpha_t s_j,
\\qquad c_j = S_j / N_j,$$

con $\\alpha_t$ en recocido coseno y tamaño de lote y $\\alpha_0$ fijados por leyes de escala
(sin ajuste más allá de $k$). Los clusters que se quedan sin uso se reviven partiendo uno
grande. Al final cada centroide es la media de los puntos que se le asignaron en la última
pasada.

Para comparar los dos métodos en igualdad de condiciones, las etiquetas son siempre las
del centroide más cercano al final (`assign` en e-means, `predict` en K-Means) y la SSE se
calcula con esas etiquetas.
"""

import time
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

METODOS = ("kmeans", "kmeans1", "emeans")   # kmeans1: un solo arranque, para comparar corrida contra corrida


@dataclass
class Resultado:
    etiquetas: np.ndarray
    centroides: np.ndarray
    sse: float
    segundos: float


def sse(X: np.ndarray, etiquetas: np.ndarray, centroides: np.ndarray) -> float:
    """Suma de distancias al cuadrado de cada punto a su centroide."""
    return float(((X - centroides[etiquetas]) ** 2).sum())


def ajustar(X: np.ndarray, k: int, metodo: str = "kmeans", seed: int = 0,
            n_init: int = 10, nepochs: float = 5) -> Resultado:
    """Agrupa `X` en `k` clusters con K-Means (`n_init` arranques k-means++; `kmeans1`: uno solo)
    o e-means (`nepochs` pasadas, un arranque)."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    t0 = time.perf_counter()
    if metodo in ("kmeans", "kmeans1"):
        km = KMeans(k, random_state=seed, n_init=n_init if metodo == "kmeans" else 1).fit(X)
        cent = km.cluster_centers_.astype(np.float32)
        etiq = km.predict(X)
    elif metodo == "emeans":
        from faiss.contrib.e_means import NumpyEmeans
        em = NumpyEmeans(X.shape[1], k, nepochs=nepochs, seed=seed)
        cent, _ = em.train(X)
        cent = np.asarray(cent, dtype=np.float32)
        _, etiq = em.assign(X)
        etiq = np.asarray(etiq, dtype=np.int64)
    else:
        raise ValueError(f"método desconocido: {metodo}")
    seg = time.perf_counter() - t0
    return Resultado(etiq, cent, sse(X, etiq, cent), seg)


def estabilidad(etiquetados: List[np.ndarray]) -> float:
    """ARI medio entre todos los pares de corridas con distinta semilla (1 = siempre la misma partición)."""
    pares = list(combinations(etiquetados, 2))
    if not pares:
        return float("nan")
    return float(np.mean([adjusted_rand_score(a, b) for a, b in pares]))


def comparar(X: np.ndarray, ks, semillas=range(10), n_init: int = 10, nepochs: float = 5,
             max_silhouette: int = 20_000, metodos=METODOS) -> List[Dict]:
    """Por cada $k$ y método: SSE (mejor y mediana entre semillas), silhouette de la mejor corrida,
    estabilidad entre semillas, tiempo medio y ARI de la mejor corrida de cada método contra la
    mejor de K-Means (`kmeans`, o el primer método de la lista)."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    rng = np.random.default_rng(0)
    sub = rng.choice(len(X), min(len(X), max_silhouette), replace=False)
    filas = []
    for k in ks:
        mejores = {}
        for metodo in metodos:
            rs = [ajustar(X, k, metodo, s, n_init, nepochs) for s in semillas]
            mejor = min(rs, key=lambda r: r.sse)
            mejores[metodo] = mejor
            usados = len(np.unique(mejor.etiquetas))
            sil = silhouette_score(X[sub], mejor.etiquetas[sub]) if usados > 1 else float("nan")
            filas.append({"k": k, "metodo": metodo, "sse_mejor": mejor.sse,
                          "sse_mediana": float(np.median([r.sse for r in rs])),
                          "silhouette": float(sil), "estabilidad": estabilidad([r.etiquetas for r in rs]),
                          "segundos": float(np.mean([r.segundos for r in rs])), "clusters_usados": usados})
        ref = mejores["kmeans"] if "kmeans" in mejores else mejores[metodos[0]]
        for f in filas[-len(metodos):]:
            f["ari_vs_kmeans"] = float(adjusted_rand_score(ref.etiquetas, mejores[f["metodo"]].etiquetas))
    return filas
