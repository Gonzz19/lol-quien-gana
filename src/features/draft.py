"""Matrices del draft para las cachés con dos equipos y cinco posiciones."""

import numpy as np
from scipy import sparse


def draft_matrix_by_lane(champions: np.ndarray, fit: np.ndarray | None = None) -> sparse.csr_matrix:
    """Una columna por (campeón, línea), +1 azul y −1 rojo.

    Si se indica fit, el vocabulario se arma con esas filas y se omiten los
    pares que aparecen solo fuera del ajuste.
    """
    ids = np.asarray(champions, dtype=np.int64).reshape(-1, 2, 5)
    n_matches = len(ids)
    # Los códigos de campeón conservan el orden y evitan desbordar al añadir la línea.
    _, champion_codes = np.unique(ids.ravel(), return_inverse=True)
    pair_codes = champion_codes.reshape(ids.shape) * 5 + np.arange(5)
    vocabulary = np.unique(pair_codes if fit is None else pair_codes[fit])
    if not len(vocabulary):
        return sparse.csr_matrix((n_matches, 0), dtype=np.float64)

    pairs = pair_codes.ravel()
    cols = np.searchsorted(vocabulary, pairs)
    present = vocabulary[np.minimum(cols, len(vocabulary) - 1)] == pairs
    rows = np.repeat(np.arange(n_matches), 10)
    sides = np.tile(np.repeat([1.0, -1.0], 5), n_matches)
    return sparse.csr_matrix((sides[present], (rows[present], cols[present])),
                             shape=(n_matches, len(vocabulary)))
