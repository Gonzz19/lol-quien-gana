"""Modelo de ataque y defensa por jugador, como los de Poisson del fútbol (Maher, Dixon-Coles).

Sin equipos fijos, la tasa de un equipo es la suma de sus cinco jugadores:

    eta_azul = mu + lado + sum_{i en azul} (a_i + A[campeón_i, línea_i])
                         - sum_{j en rojo} (d_j + D[campeón_j, línea_j])

    poisson:  y_azul ~ Poisson(minutos * exp(eta_azul))      (kills del equipo)
    gauss:    y_azul ~ Normal(eta_azul, 1)                   (p. ej. oro por minuto, estandarizado)

y lo mismo para el rojo con los papeles cambiados. a_i es el ataque del jugador y d_i su
defensa (más alto = el rival produce menos). Se estima todo junto por máxima verosimilitud
con penalización L2 (lam_j para jugadores, lam_cl para campeón-línea): un jugador con pocas
partidas queda cerca de 0 en vez de tomar un promedio ruidoso.

Sin leakage: `block_scores` parte las partidas en bloques temporales y a cada bloque le da
los parámetros ajustados SOLO con los bloques anteriores. Detalle en PROJECT_SUMMARY.md §9.S.25.
"""
from typing import Dict, Optional

import numpy as np
from scipy import sparse
from scipy.optimize import minimize


def design(jug: np.ndarray, cl: np.ndarray, n_j: int, n_cl: int) -> sparse.csr_matrix:
    """Matriz de diseño con dos filas por partida (ataca el azul, ataca el rojo).

    `jug` y `cl` son (partidas, 10) con códigos enteros; los puestos 0-4 son del azul.
    Columnas: ataque de jugadores, defensa de jugadores, ataque y defensa de campeón-línea,
    lado, constante.
    """
    n = len(jug)
    az, ro = slice(0, 5), slice(5, 10)
    filas, cols, vals = [], [], []
    for k, (ataca, defiende, lado) in enumerate(((az, ro, 1.0), (ro, az, -1.0))):
        f = np.repeat(2 * np.arange(n) + k, 5)
        for base, cod, signo, quien in ((0, jug, 1.0, ataca), (n_j, jug, -1.0, defiende),
                                        (2 * n_j, cl, 1.0, ataca), (2 * n_j + n_cl, cl, -1.0, defiende)):
            filas.append(f); cols.append(base + cod[:, quien].ravel()); vals.append(np.full(5 * n, signo))
        f1 = 2 * np.arange(n) + k
        filas += [f1, f1]
        cols += [np.full(n, 2 * n_j + 2 * n_cl), np.full(n, 2 * n_j + 2 * n_cl + 1)]
        vals += [np.full(n, lado), np.ones(n)]
    return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(filas), np.concatenate(cols))),
                             shape=(2 * n, 2 * n_j + 2 * n_cl + 2))


def fit(X: sparse.csr_matrix, y: np.ndarray, pen: np.ndarray, familia: str = "poisson",
        offset: Optional[np.ndarray] = None, theta0: Optional[np.ndarray] = None,
        maxiter: int = 200) -> np.ndarray:
    """Máxima verosimilitud penalizada: min  -loglik(theta) + 0,5 * sum(pen * theta^2)."""
    off = np.zeros(len(y)) if offset is None else offset
    Xt = X.T.tocsr()

    def f(th):
        eta = X @ th + off
        if familia == "poisson":
            mu = np.exp(np.clip(eta, -20, 20))
            perdida, res = (mu - y * eta).sum(), mu - y
        else:
            res = eta - y
            perdida = 0.5 * (res ** 2).sum()
        return perdida + 0.5 * (pen * th ** 2).sum(), Xt @ res + pen * th

    th0 = np.zeros(X.shape[1]) if theta0 is None else theta0
    return minimize(f, th0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter}).x


def block_scores(jug: np.ndarray, cl: np.ndarray, y_azul: np.ndarray, y_rojo: np.ndarray,
                 bloque: np.ndarray, lam_j: float, lam_cl: float = 1.0, familia: str = "poisson",
                 minutos: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Puntajes previos a cada partida, con parámetros ajustados en los bloques anteriores.

    `bloque` es un entero por partida, creciente en el tiempo. Devuelve por partida
    (azul - rojo): `ataque` (suma de a_i), `defensa` (suma de d_i) y `eta` (diferencia de
    tasas esperadas, con campeones y lado). También por jugador, (partidas, 10) en el orden
    de `jug`: `ataque_jug` (a_i) y `defensa_jug` (d_i), sin restar lados. El primer bloque
    queda en 0.
    """
    n, n_j, n_cl = len(jug), int(jug.max()) + 1, int(cl.max()) + 1
    X = design(jug, cl, n_j, n_cl)
    y = np.empty(2 * n); y[0::2], y[1::2] = y_azul, y_rojo
    off = None if minutos is None else np.repeat(np.log(minutos), 2)
    pen = np.r_[np.full(2 * n_j, lam_j), np.full(2 * n_cl, lam_cl), 0.0, 0.0]
    fila_bloque = np.repeat(bloque, 2)
    out = {k: np.zeros(n) for k in ("ataque", "defensa", "eta")}
    out.update({k: np.zeros((n, 10)) for k in ("ataque_jug", "defensa_jug")})
    th = None
    for b in np.unique(bloque)[1:]:
        antes = fila_bloque < b
        th = fit(X[antes], y[antes], pen, familia, None if off is None else off[antes], th)
        m = bloque == b
        a, d = th[:n_j][jug[m]], th[n_j:2 * n_j][jug[m]]
        out["ataque_jug"][m], out["defensa_jug"][m] = a, d
        out["ataque"][m] = a[:, :5].sum(1) - a[:, 5:].sum(1)
        out["defensa"][m] = d[:, :5].sum(1) - d[:, 5:].sum(1)
        eta = X[np.repeat(m, 2)] @ th
        out["eta"][m] = eta[0::2] - eta[1::2]
    return out
