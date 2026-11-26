"""Modelo previo de quién gana (src/ganador): funcional con datos sintéticos, no-fuga y caché."""
import numpy as np
import pytest
from scipy import sparse

from src.ganador import cache, modelo_previo


def _datos_sinteticos(n=3000, semilla=0):
    """D mínimo con señal conocida: gana azul si la variable 0 y el extra 0 lo favorecen."""
    rng = np.random.default_rng(semilla)
    Xd = rng.normal(size=(n, 6)).astype(np.float32)
    extras = rng.normal(size=(n, 3)).astype(np.float32)
    draft = sparse.csr_matrix(rng.choice([-1.0, 0.0, 1.0], size=(n, 8), p=[0.1, 0.8, 0.1]))
    logit = 1.2 * Xd[:, 0] + 0.8 * extras[:, 0]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    val = np.zeros(n, bool)
    val[-600:] = True                                   # V = las últimas 600 (como la ventana temporal)
    master = rng.random(n) < 0.8
    master[val] = True
    fi_v = np.flatnonzero(~val)
    D = dict(NM=n, Xd=Xd, DRAFT=draft, y=y, val=val, master=master, fi_v=fi_v)
    return D, extras


def test_aprende_la_senal_y_reporta_metricas():
    D, X = _datos_sinteticos()
    modelo, p, m = modelo_previo.evaluar_v(D, X)
    assert p.shape == (600,) and np.all((p > 0) & (p < 1))
    assert m["auc"] > 0.75
    assert m["acierto"] > m["linea_base"]
    assert m["n_columnas"] == 8 + 6 + 3
    assert m["n_entrenamiento"] == int(D["master"][D["fi_v"]].sum())
    assert 0 <= m["ece"] < 0.1
    assert set(m["cobertura"]) == set(modelo_previo.COBERTURAS)
    # las variables con señal llevan el coeficiente más grande de su bloque
    coef = modelo.lr.coef_[0]
    assert np.argmax(np.abs(coef[8:14])) == 0 and np.argmax(np.abs(coef[14:])) == 0


def test_entrena_solo_con_master_anterior_a_v():
    D, _ = _datos_sinteticos()
    tr = modelo_previo.filas_entrenamiento_v(D)
    assert not D["val"][tr].any()
    assert D["master"][tr].all()


def test_no_fuga_lo_que_pase_en_v_no_cambia_el_modelo():
    """Escalas y coeficientes salen solo del entrenamiento: alterar V (variables y resultado) no los mueve."""
    D, X = _datos_sinteticos()
    m1, p1, _ = modelo_previo.evaluar_v(D, X)
    D2, X2 = dict(D), X.copy()
    vi = np.flatnonzero(D["val"])
    D2["Xd"] = D["Xd"].copy()
    D2["Xd"][vi] *= 100.0
    X2[vi] += 50.0
    D2["y"] = D["y"].copy()
    D2["y"][vi] = 1 - D["y"][vi]
    m2 = modelo_previo.ajustar(D2, X2, modelo_previo.filas_entrenamiento_v(D2))
    np.testing.assert_array_equal(m1.mu_x, m2.mu_x)
    np.testing.assert_array_equal(m1.mu_e, m2.mu_e)
    np.testing.assert_allclose(m1.lr.coef_, m2.lr.coef_)
    # y la predicción de una partida no depende de las otras partidas de V
    p_sola = m1.predecir(D, X, vi[:1])
    np.testing.assert_allclose(p_sola, p1[:1])


def test_cache_recalcula_si_cambia_la_huella(tmp_path, monkeypatch):
    monkeypatch.setattr(cache.rutas, "CACHE", tmp_path)
    llamadas = []

    def construir():
        llamadas.append(1)
        return {"a": np.arange(3)}

    cache.obtener("g", construir, "h1")
    cache.obtener("g", construir, "h1")
    assert len(llamadas) == 1                           # misma huella: carga
    out = cache.obtener("g", construir, "h2")
    assert len(llamadas) == 2                           # huella nueva: recalcula
    np.testing.assert_array_equal(out["a"], np.arange(3))
    cache.obtener("g", construir, "h2", rehacer=True)
    assert len(llamadas) == 3


def test_huella_cambia_con_el_codigo_y_los_datos(tmp_path):
    f = tmp_path / "datos.csv"
    f.write_text("x")
    h1 = cache.huella(_datos_sinteticos, [f])
    assert h1 == cache.huella(_datos_sinteticos, [f])
    f.write_text("xy")
    assert cache.huella(_datos_sinteticos, [f]) != h1
    assert cache.huella(test_entrena_solo_con_master_anterior_a_v, [f]) != cache.huella(_datos_sinteticos, [f])
    assert cache.huella(_datos_sinteticos, [f], ["dep1"]) != cache.huella(_datos_sinteticos, [f], ["dep2"])


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
