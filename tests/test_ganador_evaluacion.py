"""Evaluación final (§9.S.80): bootstrap por bloques de días, ventanas, pendiente, semillas, modos y guardas."""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.ganador import evaluacion as EV
from src.ganador import rutas
from src.models.validacion_temporal import es_master_plus, es_master_plus_split, particion_v_master, rango_lobby

NAN = np.nan


@pytest.fixture(autouse=True)
def _modo_v():
    """Cada test deja el modo como estaba (v)."""
    yield
    rutas.configurar("v")


def _ms(texto):
    return int(datetime.strptime(texto, "%Y-%m-%d %H:%M").replace(tzinfo=rutas.ART).timestamp() * 1000)


# --------------------------------------------------------------------------------------------- métricas

def test_metricas_linea_base_siempre_rojo_y_ventaja():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.2, 0.3, 0.6, 0.7, 0.4])
    m = EV.metricas(y, p)
    assert m["siempre_rojo"] == pytest.approx(0.6)            # fijo de antemano, no "el lado más frecuente"
    assert m["acierto"] == pytest.approx(0.6)
    assert m["ventaja_rojo"] == pytest.approx(0.0)
    assert 0 < m["auc"] <= 1 and m["log_loss"] > 0 and m["ece"] >= 0


def test_dia_art_corta_a_medianoche_de_argentina():
    d = EV.dia_art(np.array([_ms("2026-09-26 23:59"), _ms("2026-09-27 00:00"), _ms("2026-09-27 23:59")]))
    assert d[0] + 1 == d[1] == d[2]
    assert EV.dia_art(np.array([rutas.corte_test_ms()]))[0] == d[1]


# --------------------------------------------------------------------------------------------- bootstrap por bloques

def _datos_por_dia(dias=20, por_dia=300, efecto_dia=1.5, semilla=0):
    """Señal real más un efecto común a cada día (las partidas de un día no son independientes)."""
    rng = np.random.default_rng(semilla)
    g = np.repeat(np.arange(dias), por_dia)
    x = rng.normal(size=len(g))
    u = rng.normal(scale=efecto_dia, size=dias)[g]             # lado favorecido ese día
    fuerza = rng.uniform(0.0, 2.0 * efecto_dia, size=dias)[g]  # qué tan bien anda el modelo ese día
    y = (rng.random(len(g)) < 1 / (1 + np.exp(-(fuerza * x + u)))).astype(int)
    p = 1 / (1 + np.exp(-x))
    return y, p, g


def test_bootstrap_bloques_es_reproducible_y_el_punto_es_la_metrica():
    y, p, g = _datos_por_dia()
    a = EV.bootstrap_bloques(y, p, g, B=200, semilla=3)
    b = EV.bootstrap_bloques(y, p, g, B=200, semilla=3)
    assert a == b
    assert a["auc"][0] == pytest.approx(EV.metricas(y, p)["auc"])
    assert a["n_grupos"] == 20 and a["n"] == len(y) and a["B"] == 200
    for k in EV.NOMBRES:
        assert a[k][1] <= a[k][0] <= a[k][2] or k == "ece"     # el ECE es sesgado: el punto puede quedar afuera


def test_bootstrap_remuestrea_grupos_enteros():
    """Cada réplica trae cada grupo completo (0, 1, 2... veces), nunca partidas sueltas de un grupo."""
    rng = np.random.default_rng(0)
    tam = {0: 5, 1: 7, 2: 11, 3: 13}
    g = np.concatenate([np.full(n, k) for k, n in tam.items()])
    y = rng.integers(0, 2, size=len(g))
    vistos = []

    def funcion(yy, pp):
        c = pd.Series(pp).value_counts()
        vistos.append(all(c[k] % tam[int(k)] == 0 for k in c.index))
        return {"x": float(len(yy))}

    EV.bootstrap_bloques(y, g.astype(float), g, B=50, semilla=1, funcion=funcion)
    assert len(vistos) == 51 and all(vistos)


def test_con_dias_dependientes_el_ic_por_bloques_es_mas_ancho_que_por_partidas():
    y, p, g = _datos_por_dia(efecto_dia=1.5)
    blo = EV.bootstrap_bloques(y, p, g, B=300, semilla=0)
    par = EV.bootstrap_partidas(y, p, B=300, semilla=0)
    ancho = lambda b, k: b[k][2] - b[k][1]  # noqa: E731
    assert ancho(blo, "acierto") > 1.5 * ancho(par, "acierto")
    assert ancho(blo, "log_loss") > 1.5 * ancho(par, "log_loss")


def test_bootstrap_rechaza_b_invalido():
    with pytest.raises(ValueError):
        EV.bootstrap_bloques(np.array([0, 1]), np.array([0.2, 0.8]), np.array([0, 1]), B=0)


# --------------------------------------------------------------------------------------------- ventanas y pendiente

def test_ventanas_en_orden_de_fecha_y_el_resto_chico_se_une():
    rng = np.random.default_rng(0)
    t = rng.permutation(np.arange(10_600))                    # desordenadas a propósito
    w = EV.ventanas(t, 2500)
    assert np.bincount(w).tolist() == [2500, 2500, 2500, 3100]  # 600 < 1250: se suman a la última
    assert (np.diff(w[np.argsort(t)]) >= 0).all()             # cronológicas
    w2 = EV.ventanas(np.arange(6000), 2500)
    assert np.bincount(w2).tolist() == [2500, 3500]           # 1000 < 1250: se une a la anterior
    assert np.bincount(EV.ventanas(np.arange(6300), 2500)).tolist() == [2500, 2500, 1300]


def _datos_que_se_degradan(pendiente, n=15_000, semilla=0):
    rng = np.random.default_rng(semilla)
    t = np.sort(rng.integers(0, 35 * 86_400_000, size=n))   # 5 semanas
    x = rng.normal(size=n)
    fuerza = 1.5 + pendiente * (t / (7 * 86_400_000))         # la señal cambia con el tiempo
    y = (rng.random(n) < 1 / (1 + np.exp(-fuerza * x))).astype(int)
    return y, 1 / (1 + np.exp(-x)), t


def test_pendiente_detecta_degradacion_y_no_inventa_una():
    y, p, t = _datos_que_se_degradan(-0.25)
    r = EV.pendiente_auc(y, p, t, 2500, B=200, semilla=0)
    assert r["n_ventanas"] == 6 and r["hi"] < 0 < -r["pendiente_semana"]
    y, p, t = _datos_que_se_degradan(0.0, semilla=1)
    r = EV.pendiente_auc(y, p, t, 2500, B=200, semilla=0)
    assert r["lo"] < 0 < r["hi"]


def test_metricas_por_ventana_tiene_ic_por_ventana():
    y, p, t = _datos_que_se_degradan(0.0)
    w = EV.metricas_por_ventana(y, p, t, 2500, B=100, semilla=0)
    assert len(w) == 6 and w["n"].sum() == len(y)
    assert ((w.auc_lo <= w.auc) & (w.auc <= w.auc_hi)).all()
    assert (w.desde.to_numpy()[1:] >= w.hasta.to_numpy()[:-1]).all()


# --------------------------------------------------------------------------------------------- semillas

def test_resumen_y_correlacion_de_semillas():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    base = rng.random(500)
    preds = {42: base, 0: np.clip(base + rng.normal(scale=0.01, size=500), 0.01, 0.99), 1: base.copy()}
    por = {s: EV.metricas(y, p) for s, p in preds.items()}
    r = EV.resumen_semillas(por, 42)
    assert r["auc"]["principal"] == por[42]["auc"] and r["auc"]["min"] <= r["auc"]["media"] <= r["auc"]["max"]
    c = EV.correlacion_predicciones(preds, 42)
    assert c["corr_con_principal"][1] == pytest.approx(1.0) and 0.99 < c["corr_con_principal"][0] < 1
    assert c["max_abs_dp_con_principal"][1] == 0.0


# --------------------------------------------------------------------------------------------- regla de Master+

def test_regla_del_split_cuenta_cinco_master_o_mas():
    tnum = np.array([
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],             # 5 Master: Master+ (la mediana de 10 da 0,5 → no)
        [1, 1, 1, 1, 0, 0, 0, 0, 0, NAN],           # 4: no
        [3, 3, 2, 2, 1, NAN, NAN, NAN, NAN, NAN],   # 5 con dato, todos Master+: sí
        [1, 1, 1, 1, NAN, NAN, NAN, NAN, NAN, NAN],  # mediana Master, pero 4 jugadores: no (la mediana dice sí)
    ], dtype=float)
    assert es_master_plus_split(tnum).tolist() == [True, False, True, False]
    assert es_master_plus(tnum).tolist() == es_master_plus_split(tnum).tolist()   # unificadas (merge de gonchi)
    assert rango_lobby(tnum).tolist() == [1, 0, 1, -1]   # 3: solo 4 GM o más
    v, d = particion_v_master(np.array([True, True, False, True]), tnum, "split")
    assert v.tolist() == [True, False, False, False] and d.tolist() == [False, True, False, True]
    with pytest.raises(ValueError):
        particion_v_master(np.ones(4, bool), tnum, "otra")


# --------------------------------------------------------------------------------------------- modos

def test_modos_fuentes_cache_regla_e_inicio():
    rutas.configurar("v")
    assert rutas.FUENTES == (rutas.TRAIN_CSV,) and rutas.CACHE == rutas.CACHE_V and rutas.regla_master() == "mediana"
    t = np.array([0, 10, 20, 30, 40])
    assert rutas.inicio_evaluacion(t, 10) == 25.0               # mediana de lo posterior a la foto
    rutas.configurar("simulacro")
    assert rutas.FUENTES == (rutas.TRAIN_CSV,) and rutas.CACHE == rutas.CACHE_SIMULACRO
    assert rutas.regla_master() == "split" and rutas.inicio_evaluacion(t, 10) == 25.0
    rutas.configurar("test")
    assert rutas.FUENTES == (rutas.TRAIN_CSV, rutas.TEST_CSV, rutas.DIAMANTE_DESPUES_CSV)
    assert rutas.CACHE == rutas.CACHE_TEST and rutas.regla_master() == "split"
    assert rutas.inicio_evaluacion(t, 10) == rutas.corte_test_ms() == _ms("2026-09-27 00:00")
    assert len({rutas.CACHE_V, rutas.CACHE_SIMULACRO, rutas.CACHE_TEST}) == 3   # el test no pisa la caché de V
    with pytest.raises(ValueError):
        rutas.configurar("otro")
    rutas.configurar("v")


def test_modo_test_sin_v_ajusta_hasta_el_16_09_como_v_aunque_haya_test():
    corte = _ms("2026-09-27 00:00")
    t_train = np.array([0, 10, 20, 30, 40])
    t = np.r_[t_train, corte, corte + 5, corte + 9]                # train + test (+ Diamond posterior)
    rutas.configurar("v")
    inicio_v = rutas.inicio_evaluacion(t_train, 10)
    rutas.configurar("test_sin_v")
    assert rutas.FUENTES == (rutas.TRAIN_CSV, rutas.TEST_CSV, rutas.DIAMANTE_DESPUES_CSV)
    assert rutas.CACHE == rutas.CACHE_TEST_SIN_V and rutas.regla_master() == "split"
    assert rutas.inicio_evaluacion(t, 10) == inicio_v == 25.0   # el test no mueve la mediana
    assert len({rutas.CACHE_V, rutas.CACHE_SIMULACRO, rutas.CACHE_TEST, rutas.CACHE_TEST_SIN_V}) == 4
    rutas.configurar("v")


def test_leer_partidas_concatena_las_fuentes(tmp_path, monkeypatch):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    pd.DataFrame({"match_id": ["x", "x"], "v": [1, 2], "w": [0, 0]}).to_csv(a, index=False)
    pd.DataFrame({"match_id": ["y"], "v": [3], "w": [0]}).to_csv(b, index=False)
    monkeypatch.setattr(rutas, "FUENTES", (a,))
    pd.testing.assert_frame_equal(rutas.leer_partidas(usecols=["match_id", "v"]),
                                  pd.read_csv(a, usecols=["match_id", "v"]))   # una fuente: lo mismo que antes
    monkeypatch.setattr(rutas, "FUENTES", (a, b))
    df = rutas.leer_partidas(usecols=["match_id", "v"])
    assert df["v"].tolist() == [1, 2, 3] and df.index.tolist() == [0, 1, 2]


# --------------------------------------------------------------------------------------------- guardas del script

def test_guardas_de_la_evaluacion_unica(tmp_path):
    from scripts.analysis import evaluacion_test as E
    log, proto = tmp_path / "LOG.md", tmp_path / "P.md"
    proto.write_text("x")
    assert E.problemas_previos(log, None, "", proto) == []
    assert len(E.problemas_previos(log, None, " M src/x.py", proto)) == 1     # árbol sucio
    log.write_text("ya corrió")
    assert len(E.problemas_previos(log, None, "", proto)) == 1                # una sola vez
    assert E.problemas_previos(log, "motivo aprobado", "", proto) == []       # excepción explícita
    assert len(E.problemas_previos(log, None, "", tmp_path / "no.md")) == 2


def test_hashes_del_protocolo_y_verificacion(tmp_path):
    from scripts.analysis import evaluacion_test as E
    f = tmp_path / "a.csv"
    f.write_bytes(b"abc")
    h = E.sha256(f)
    assert h == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    esperados = E.hashes_del_protocolo(f"texto\n<!-- sha256 match_split {h} -->\n<!-- sha256 test {'0' * 64} -->")
    assert esperados == {"match_split": h, "test": "0" * 64}
    r = E.verificar_hashes(esperados, {"match_split": f, "test": f})
    assert r["match_split"][1] and not r["test"][1]


def test_el_test_no_acepta_otras_semillas():
    from scripts.analysis import evaluacion_test as E
    with pytest.raises(SystemExit):
        E.main(["--semillas", "7"])
    with pytest.raises(SystemExit):
        E.main(["--sin-v", "--simulacro"])
    assert E.SALIDA_SIN_V != E.SALIDA and E.LOG_SIN_V != E.LOG
    assert E.SEMILLA_PRINCIPAL in E.SEMILLAS and len(E.SEMILLAS) == 5 and E.B >= 1000


def test_parciales_por_semilla_se_retoman_solo_con_la_misma_firma(tmp_path):
    """Un corte a mitad de la evaluación no pierde las semillas terminadas, y no se mezclan corridas distintas."""
    import importlib
    ET = importlib.import_module("scripts.analysis.evaluacion_test")
    firma = dict(commit="abc", modo="simulacro", hash_protocolo="h1")
    base = dict(mids=np.array(["LA2_1", "LA2_2"]), ev=np.array([0, 1]), dv=np.array([], dtype=int),
                tr=np.array([0]), t=np.array([1, 2], dtype=np.int64), y=np.array([1, 0], dtype=np.int8),
                rango=np.array([1, 2]), regla="split")
    pred = {"previo": np.array([0.4, 0.6]), "n_columnas": 1504, "vivo_10_n_entrenamiento": 100}
    ET.guardar_parcial(tmp_path, 42, firma, base, pred)
    b, p = ET.cargar_parcial(tmp_path, 42, firma)
    assert b["regla"] == "split" and p["n_columnas"] == 1504
    assert np.array_equal(b["mids"], base["mids"]) and np.array_equal(p["previo"], pred["previo"])
    assert ET.cargar_parcial(tmp_path, 42, firma | {"commit": "otro"}) is None
    assert ET.cargar_parcial(tmp_path, 0, firma) is None
