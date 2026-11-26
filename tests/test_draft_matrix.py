"""Igualdad estructural de la matriz vectorizada con los scripts originales."""

import numpy as np
import pytest
from scipy import sparse

from src.features.draft import draft_matrix_by_lane


def _original(ids, fit):
    vocab = sorted({(int(champion), lane) for match in (ids if fit is None else ids[fit])
                    for team in match for lane, champion in enumerate(team)})
    cols_by_pair = {pair: i for i, pair in enumerate(vocab)}
    rows, cols, data = [], [], []
    for row, match in enumerate(ids):
        for sign, team in zip((1.0, -1.0), match):
            for lane, champion in enumerate(team):
                col = cols_by_pair.get((int(champion), lane))
                if col is not None:
                    rows.append(row)
                    cols.append(col)
                    data.append(sign)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(ids), len(vocab)), dtype=np.float64)


@pytest.mark.parametrize("fit", [None, np.array([True, False, True]), np.zeros(3, dtype=bool)])
def test_matriz_exacta_con_vocabulario_completo_o_de_ajuste(fit):
    ids = np.array([
        [[1, 79, 1, -1, 100], [1, 100, 1, -1, 79]],
        [[200, 111, 79, 111, 200], [-1, 5, 200, 1, 100]],
        [[100, 1, 100, 1, 1], [79, 1, 79, 1, 3]],
    ])
    expected = _original(ids, fit)
    actual = draft_matrix_by_lane(ids.reshape(3, 10), fit)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual.indptr, expected.indptr)
    np.testing.assert_array_equal(actual.indices, expected.indices)
    np.testing.assert_array_equal(actual.data, expected.data)
