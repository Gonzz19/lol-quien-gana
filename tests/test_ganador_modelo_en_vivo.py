"""Modelo en vivo de quién gana (src/ganador/modelo_en_vivo): funcional con datos sintéticos y no-fuga."""
import numpy as np
import pytest

from src.ganador import modelo_en_vivo as MV


def _insumos_sinteticos(n=4000, semilla=0):
    """Insumos con señal conocida: gana el que va arriba en oro; el previo L es débil antes de la foto (tramos 1-3)
    y fuerte después (A2-A4 y V), como el escalón 9."""
    rng = np.random.default_rng(semilla)
    chunk = np.sort(rng.integers(1, 9, size=n)).astype(np.int32)
    fuerza = rng.normal(size=n)                                          # lo que sabe el previo
    post = chunk >= MV.A2
    L = np.where(post, 1.0 * fuerza, 0.15 * fuerza + rng.normal(size=n)).astype(np.float64)
    tl = rng.normal(size=(n, 2, 5, 3, 8)).astype(np.float32)
    tl[:, :, :, :, MV.ST["gold"]] += np.cumsum(rng.normal(size=(n, 1, 1, 3)), axis=-1) * np.array([1, -1])[None, :, None, None]
    dg10 = (tl[:, 0, :, 0, MV.ST["gold"]] - tl[:, 1, :, 0, MV.ST["gold"]]).sum(1)
    logit = 0.4 * dg10 + 1.0 * fuerza
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.int8)
    I = dict(mids=np.array([f"LA2_{i}" for i in range(n)]), chunk=chunk, dur_sec=np.full(n, 1800.0, np.float32),
             tl_raw=tl, gold_resid=rng.normal(size=(n, 2, 5, 3)).astype(np.float32), L=L, y=y,
             master=np.ones(n, bool), val=chunk == MV.TV, diam_v=chunk == MV.TDV, post=post)
    for s in ("gold", "xp", "cs", "damage"):
        for m in MV.MINUTOS:
            I[f"exp_{s}_{m}"] = np.zeros((n, 5), np.float32)
    for m in (15, 20):
        I[f"tr_trans_gold_{m}"] = np.zeros((n, 5), np.float32)
    return I


def test_dos_etapas_aprende_y_devuelve_peso_al_previo_despues_de_la_foto():
    I = _insumos_sinteticos()
    v = np.flatnonzero(I["chunk"] == MV.TV)
    R = MV.ajustar_predecir(I, 10, v)
    y = I["y"][v].astype(int)
    m1, m2 = MV.metricas(y, MV._sigm(R["z1_fin"])), MV.metricas(y, MV._sigm(R["z2_fin"]))
    assert m1["auc"] > 0.7
    assert R["C1"] in MV.C_ETAPA1
    a, b, c = R["etapa2"]
    assert c > 0                                   # el previo pesa más en el régimen posterior a la foto
    assert m2["auc"] > m1["auc"]
    assert m2["acierto"] > m2["siempre_rojo"] - 0.5 and 0 <= m2["ece"] < 0.2
    # en A2 la etapa 2 es neutra (no tiene con qué ajustarse hacia adelante)
    n2 = len(R["bloques"][0])
    np.testing.assert_array_equal(R["z2_a24"][:n2], R["z1_a24"][:n2])


def test_el_modelo_principal_es_la_etapa_1():
    """Decisión de §9.S.78: la etapa 2 no pasa la regla de §9.S.47, así que `z_fin`/`z_a24` son la etapa 1."""
    I = _insumos_sinteticos(n=2500, semilla=3)
    v = np.flatnonzero(I["chunk"] == MV.TV)
    R = MV.ajustar_predecir(I, 10, v)
    assert MV.MODELO_PRINCIPAL == "etapa1"
    np.testing.assert_array_equal(R["z_fin"], R["z1_fin"])
    np.testing.assert_array_equal(R["z_a24"], R["z1_a24"])
    assert not np.array_equal(R["z_fin"], R["z2_fin"])    # la variante descartada se sigue calculando


def test_metricas_con_ic_y_linea_base_siempre_rojo():
    rng = np.random.default_rng(1)
    y = (rng.random(800) < 0.45).astype(int)
    p = np.clip(0.45 + 0.3 * (y - 0.45) + rng.normal(0, 0.1, 800), 0.01, 0.99)
    mt = MV.metricas_ic(y, p, B=50)
    assert mt["n"] == 800
    for k in ("auc", "acierto", "log_loss", "ece"):
        v, lo, hi = mt[k]
        assert lo <= v <= hi
    assert mt["siempre_rojo"][0] == pytest.approx((y == 0).mean())
    d, lo, hi = MV.delta_auc_ic(y, p, p, B=20)
    assert d == 0 and lo == 0 and hi == 0


def test_no_fuga_las_variables_del_minuto_no_leen_nada_posterior():
    """Cambiar el timeline y el oro residual de minutos posteriores, la duración ya cumplida y el resultado
    no cambia las variables al minuto m."""
    I = _insumos_sinteticos(n=300)
    filas = np.arange(300)
    for m, mi in ((10, 0), (15, 1)):
        X = MV.variables_minuto(I, m, filas)
        J = dict(I)
        J["tl_raw"] = I["tl_raw"].copy()
        J["tl_raw"][:, :, :, mi + 1:, :] += 1000.0
        J["gold_resid"] = I["gold_resid"].copy()
        J["gold_resid"][:, :, :, mi + 1:] -= 1000.0
        J["y"] = 1 - I["y"]
        np.testing.assert_array_equal(X, MV.variables_minuto(J, m, filas))
        # y una partida no depende de las demás
        np.testing.assert_array_equal(X[:5], MV.variables_minuto(I, m, filas[:5]))


def test_no_fuga_el_resultado_de_v_no_cambia_sus_predicciones():
    I = _insumos_sinteticos()
    v = np.flatnonzero(I["chunk"] == MV.TV)
    R = MV.ajustar_predecir(I, 15, v)
    J = dict(I)
    J["y"] = I["y"].copy()
    J["y"][v] = 1 - I["y"][v]
    np.testing.assert_allclose(R["z2_fin"], MV.ajustar_predecir(J, 15, v)["z2_fin"])


def test_pliegues_hacia_adelante_y_oro_residual_causal():
    rng = np.random.default_rng(3)
    n = 2000
    t = np.sort(rng.integers(0, 10**9, size=n))
    chunk = np.minimum(np.arange(n) // 250, 8).astype(np.int32)
    master = rng.random(n) < 0.8
    val0 = chunk >= MV.TV
    folds = MV.pliegues(chunk, t, master, val0)
    for te, tr in folds:
        assert master[tr].all() and t[tr].max() < t[te].min() and not val0[tr].any()
    tl = rng.normal(size=(n, 2, 5, 3, 8)).astype(np.float32)
    g = MV.oro_residual(tl, folds)
    tl2 = tl.copy()
    tl2[chunk >= 5] *= 3.0                              # cambiar el futuro (tramos 5+)
    g2 = MV.oro_residual(tl2, folds)
    antes = (chunk >= 1) & (chunk < 5)
    np.testing.assert_array_equal(g[antes], g2[antes])


def test_mids_se_verifican():
    MV.verificar_mids({"mids": np.array(["a", "b"])}, np.array(["a", "b"]), "x")
    with pytest.raises(ValueError):
        MV.verificar_mids({"mids": np.array(["b", "a"])}, np.array(["a", "b"]), "x")
    with pytest.raises(ValueError):
        MV.verificar_mids({"L": np.zeros(2)}, np.array(["a", "b"]), "x")
