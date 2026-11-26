"""Modelo de ataque y defensa por jugador (src/features/attack_defense.py)."""

import numpy as np
import pytest

from src.features.attack_defense import block_scores, design, fit


def _simular(n=4000, n_j=40, seed=0):
    """Partidas con ataque y defensa verdaderos por jugador; kills ~ Poisson."""
    rng = np.random.default_rng(seed)
    a, d = rng.normal(0, 0.3, n_j), rng.normal(0, 0.3, n_j)
    jug = np.array([rng.choice(n_j, 10, replace=False) for _ in range(n)])
    cl = rng.integers(0, 4, (n, 10))
    minutos = rng.uniform(20, 40, n)
    eta_az = np.log(0.8) + a[jug[:, :5]].sum(1) - d[jug[:, 5:]].sum(1)
    eta_ro = np.log(0.8) + a[jug[:, 5:]].sum(1) - d[jug[:, :5]].sum(1)
    return jug, cl, rng.poisson(minutos * np.exp(eta_az)), rng.poisson(minutos * np.exp(eta_ro)), minutos, a, d


def test_el_diseno_tiene_los_signos_correctos():
    jug = np.arange(10)[None, :]
    X = design(jug, np.zeros((1, 10), int), n_j=10, n_cl=1).toarray()
    azul, rojo = X[0], X[1]
    assert (azul[:5] == 1).all() and (azul[5:10] == 0).all()          # ataca el azul
    assert (azul[10:15] == 0).all() and (azul[15:20] == -1).all()     # defiende el rojo
    assert (rojo[5:10] == 1).all() and (rojo[10:15] == -1).all()
    assert azul[-2] == 1 and rojo[-2] == -1 and azul[-1] == rojo[-1] == 1


def test_recupera_ataque_y_defensa():
    jug, cl, ya, yr, minutos, a, d = _simular()
    X = design(jug, cl, 40, 4)
    y = np.empty(2 * len(jug)); y[0::2], y[1::2] = ya, yr
    pen = np.r_[np.full(80, 1.0), np.full(8, 1.0), 0, 0]
    th = fit(X, y, pen, "poisson", np.repeat(np.log(minutos), 2))
    assert np.corrcoef(th[:40], a)[0, 1] > 0.9
    assert np.corrcoef(th[40:80], d)[0, 1] > 0.9


def test_la_penalizacion_encoge_hacia_cero():
    jug, cl, ya, yr, minutos, _, _ = _simular(n=500)
    X = design(jug, cl, 40, 4)
    y = np.empty(1000); y[0::2], y[1::2] = ya, yr
    off = np.repeat(np.log(minutos), 2)
    norma = lambda lam: np.abs(fit(X, y, np.r_[np.full(80, lam), np.ones(8), 0, 0], "poisson", off)[:80]).mean()
    assert norma(100.0) < norma(1.0)


@pytest.mark.parametrize("familia", ["poisson", "gauss"])
def test_los_puntajes_no_miran_su_bloque_ni_los_posteriores(familia):
    jug, cl, ya, yr, minutos, _, _ = _simular(n=1200)
    bloque = np.arange(1200) // 200
    args = dict(bloque=bloque, lam_j=2.0, familia=familia, minutos=minutos if familia == "poisson" else None)
    antes = block_scores(jug, cl, ya.astype(float), yr.astype(float), **args)
    ya2, yr2 = ya.astype(float), yr.astype(float)
    ya2[bloque >= 3] = ya2[bloque >= 3] * 3 + 5          # cambia el bloque 3 y todo lo que sigue
    despues = block_scores(jug, cl, ya2, yr2, **args)
    for k in antes:
        assert (antes[k][bloque == 0] == 0).all()                     # sin historia, puntaje 0
        assert np.allclose(antes[k][bloque <= 3], despues[k][bloque <= 3], atol=1e-6)
        assert not np.allclose(antes[k][bloque >= 4], despues[k][bloque >= 4], atol=1e-6)


def test_el_puntaje_predice_al_que_gana_la_cuenta():
    jug, cl, ya, yr, minutos, _, _ = _simular(n=3000)
    s = block_scores(jug, cl, ya.astype(float), yr.astype(float), np.arange(3000) // 300, 2.0, minutos=minutos)
    m = np.arange(3000) >= 1500
    assert np.corrcoef(s["eta"][m], np.log1p(ya[m]) - np.log1p(yr[m]))[0, 1] > 0.5
