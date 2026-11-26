"""Tests de src/models/clustering.py: K-Means y e-means con la misma interfaz."""
import numpy as np
import pytest

from src.models import clustering as cl

pytest.importorskip("faiss.contrib.e_means")


def _blobs(n_por=300, k=4, d=6, sep=8.0, seed=0):
    rng = np.random.default_rng(seed)
    centros = rng.normal(scale=sep, size=(k, d))
    X = np.vstack([c + rng.normal(size=(n_por, d)) for c in centros])
    y = np.repeat(np.arange(k), n_por)
    return X.astype(np.float32), y


def test_sse_a_mano():
    X = np.array([[0, 0], [2, 0], [10, 10]], dtype=np.float32)
    cent = np.array([[1, 0], [10, 10]], dtype=np.float32)
    assert cl.sse(X, np.array([0, 0, 1]), cent) == pytest.approx(2.0)


@pytest.mark.parametrize("metodo", cl.METODOS)
def test_recupera_blobs_separados(metodo):
    """La mejor de 5 semillas recupera los grupos. Una sola corrida de e-means puede quedar en
    un mínimo local con k chico: arranca de k puntos al azar, sin k-means++."""
    from sklearn.metrics import adjusted_rand_score
    X, y = _blobs()
    r = min((cl.ajustar(X, 4, metodo, seed=s) for s in range(5)), key=lambda r: r.sse)
    assert r.etiquetas.shape == (len(X),)
    assert r.centroides.shape == (4, X.shape[1])
    assert adjusted_rand_score(y, r.etiquetas) > 0.95


def test_etiquetas_son_el_centroide_mas_cercano():
    X, _ = _blobs(seed=2)
    r = cl.ajustar(X, 4, "emeans", seed=0)
    d = ((X[:, None, :] - r.centroides[None]) ** 2).sum(-1)
    assert (d.argmin(1) == r.etiquetas).mean() > 0.999
    assert r.sse == pytest.approx(cl.sse(X, r.etiquetas, r.centroides), rel=1e-5)


def test_metodo_desconocido():
    with pytest.raises(ValueError):
        cl.ajustar(np.zeros((10, 2)), 2, "dbscan")


def test_estabilidad():
    a = np.array([0, 0, 1, 1])
    assert cl.estabilidad([a, 1 - a, a]) == pytest.approx(1.0)
    assert np.isnan(cl.estabilidad([a]))


def test_comparar_filas():
    X, _ = _blobs(n_por=100, k=3)
    filas = cl.comparar(X, ks=[2, 3], semillas=range(2), n_init=2)
    assert len(filas) == 2 * len(cl.METODOS)
    f3 = [f for f in filas if f["k"] == 3]
    assert {f["metodo"] for f in f3} == set(cl.METODOS)
    assert all(f["ari_vs_kmeans"] > 0.95 for f in f3)
    assert next(f for f in f3 if f["metodo"] == "kmeans")["ari_vs_kmeans"] == pytest.approx(1.0)
