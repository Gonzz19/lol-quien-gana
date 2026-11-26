import numpy as np
import pandas as pd
import pytest

from src.models.validacion_temporal import (bloques_por_fecha, comparar_por_bloques, decidir,
                                            predecir_por_bloques)


def test_bloques_ordenados_y_parejos():
    t = np.random.default_rng(0).permutation(1000)
    b = bloques_por_fecha(t, 5)
    assert set(b) == {0, 1, 2, 3, 4}
    assert np.bincount(b).tolist() == [200] * 5
    # orden temporal: todo el bloque b es anterior a todo el b + 1
    for k in range(4):
        assert t[b == k].max() < t[b == k + 1].min()


def test_v_master_plus():
    from src.models.validacion_temporal import particion_v_master
    nan = np.nan
    tnum = np.array([
        [1] * 6 + [0] * 4,          # 6 de 10 Master+: entra
        [1] * 5 + [0] * 5,          # 5 de 10 Master+ (regla oficial split.py): entra
        [3] * 10,                   # Challenger: entra
        [1] * 4 + [nan] * 6,        # solo 4 Master+ (< 5): no entra
        [nan] * 10,                 # sin dato: no entra
    ], dtype=float)
    val = np.array([True, True, True, True, False])
    vm, dv = particion_v_master(val, tnum)
    assert vm.tolist() == [True, True, True, False, False]
    assert dv.tolist() == [False, False, False, True, False]
    assert not (vm & dv).any()


def test_bloques_necesita_dos():
    with pytest.raises(ValueError):
        bloques_por_fecha(np.arange(10), 1)


def test_hacia_adelante_sin_futuro():
    """Cada bloque se predice con un modelo que solo vio partidas anteriores."""
    n = 500
    t = np.arange(n)
    b = bloques_por_fecha(t, 5)
    X = t[:, None].astype(float)
    y = np.arange(n) % 2
    vistos = []

    def ajustar(Xtr, ytr):
        vistos.append(Xtr[:, 0].max())
        return lambda Xte: np.where(Xte[:, 0] > vistos[-1], 0.5, np.nan)

    p = predecir_por_bloques(ajustar, X, y, b)
    assert np.isnan(p[b == 0]).all()
    assert (p[b >= 1] == 0.5).all()          # todo lo predicho es posterior a lo entrenado
    assert len(vistos) == 4


def test_comparar_detecta_mejora_y_decide():
    rng = np.random.default_rng(1)
    n = 8000
    b = np.repeat(np.arange(1, 5), n // 4)
    z = rng.normal(size=n)
    y = (z + rng.normal(size=n) > 0).astype(int)
    p_bueno = 1 / (1 + np.exp(-z))
    p_ruido = 1 / (1 + np.exp(-(0.3 * z + rng.normal(size=n))))
    tab = comparar_por_bloques(y, p_bueno, p_ruido, b, B=50)
    assert list(tab.bloque) == ["1", "2", "3", "4", "media"]
    assert (tab.delta > 0).all()
    assert decidir(tab) == "entra"
    assert decidir(comparar_por_bloques(y, p_ruido, p_bueno, b, B=50)) == "empeora"


def test_decidir_no_concluyente():
    filas = [{"bloque": str(k), "delta": d, "ic_lo": d - 0.01, "ic_hi": d + 0.01}
             for k, d in zip(range(1, 5), [0.002, -0.001, 0.001, -0.002])]
    filas.append({"bloque": "media", "delta": 0.0, "ic_lo": -0.001, "ic_hi": 0.001})
    assert decidir(pd.DataFrame(filas)) == "no concluyente"
    # media significativa pero mejora solo en 2 de 4: no entra
    filas[-1] = {"bloque": "media", "delta": 0.0005, "ic_lo": 0.0001, "ic_hi": 0.001}
    assert decidir(pd.DataFrame(filas)) == "no concluyente"
