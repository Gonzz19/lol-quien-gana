"""Piezas del notebook de la entrega (`src/informe/`) que no necesitan los datos."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.informe import datos as D
from src.informe import estilo as E


def test_numeros_en_castellano():
    assert E.num(0.67267) == "0,6727"
    assert E.num(1234.5, 1) == "1.234,5"
    assert E.pct(0.6241) == "62,4 %"
    assert E.entero(187817) == "187.817"
    assert E.ic(0.6727, 0.6644, 0.6807) == "0,6727 [0,6644; 0,6807]"
    assert E.ic(0.624, 0.617, 0.632, d=1, porcentaje=True) == "62,4 % [61,7; 63,2]"


def test_auc_ponderado_coincide_con_sklearn_y_con_el_remuestreo():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 800).astype(float)
    p = np.clip(0.5 + 0.2 * (y - 0.5) + rng.normal(0, 0.2, 800), 0.001, 0.999)
    assert D.auc_ponderado(y, p, np.ones((1, 800)))[0] == pytest.approx(roc_auc_score(y, p))
    W = D.pesos_bootstrap(800, 5, semilla=3)
    for w, auc in zip(W, D.auc_ponderado(y, p, W)):
        idx = np.repeat(np.arange(800), w.astype(int))           # la réplica escrita como remuestreo explícito
        assert auc == pytest.approx(roc_auc_score(y[idx], p[idx]))


def test_resumen_trae_ic_que_contienen_al_valor():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 2000)
    p = np.clip(0.5 + 0.3 * (y - 0.5) + rng.normal(0, 0.25, 2000), 0.01, 0.99)
    r = D.resumen(y, p, b=200)
    for k in ("auc", "acierto", "log_loss", "siempre_rojo"):
        v, lo, hi = r[k]
        assert lo <= v <= hi


def test_sin_evaluacion_del_test_hay_aviso(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "TEST_FINAL", tmp_path / "no_existe")
    assert D.resultados_test() == {}
    from src.informe import figuras as F
    assert "Pendiente" in F.seccion_test().data
    assert "Pendiente" in F.seccion_robustez_test().data


def test_previo_test_lee_la_fila_principal(tmp_path, monkeypatch):
    """El recorrido y DraftRec toman del test la fila del previo con IC por días, y nada si no hay evaluación."""
    import pandas as pd
    from src.informe import figuras as F
    monkeypatch.setattr(D, "TEST_FINAL", tmp_path / "no_existe")
    assert F._previo_test() is None
    tmp = tmp_path / "test_final"
    tmp.mkdir()
    fila = dict(modelo="previo", minuto="", conjunto="evaluacion", n=1000, auc=0.66, auc_lo=0.65, auc_hi=0.67,
                acierto=0.61, acierto_lo=0.60, acierto_hi=0.62)
    pd.DataFrame([fila | {"ic": "partidas", "acierto": 0.5}, fila | {"ic": "dias"},
                  fila | {"ic": "dias", "conjunto": "evaluacion MASTER", "acierto": 0.5}]).to_csv(tmp / "metricas.csv",
                                                                                                    index=False)
    monkeypatch.setattr(D, "TEST_FINAL", tmp)
    assert F._previo_test() == {"acierto": (0.61, 0.60, 0.62), "auc": (0.66, 0.65, 0.67), "n": 1000}


def test_recorrido_y_draftrec_se_dibujan_con_y_sin_test(tmp_path, monkeypatch):
    from src.informe import figuras as F
    monkeypatch.setattr(E, "CARPETA_PNG", tmp_path)
    monkeypatch.setattr(F, "_previo_test", lambda: None)
    assert len(F.fig_recorrido_previo().axes[0].patches) == len(F.RECORRIDO_PREVIO)
    F.fig_draftrec()
    monkeypatch.setattr(F, "_previo_test", lambda: {"acierto": (0.61, 0.60, 0.62), "auc": (0.66, 0.65, 0.67), "n": 9})
    assert len(F.fig_recorrido_previo().axes[0].patches) == len(F.RECORRIDO_PREVIO) + 1
    F.fig_draftrec()
    assert {p.name for p in tmp_path.glob("*.png")} == {"04b_recorrido_previo.png", "04c_draftrec.png"}


def test_test_contra_v_con_y_sin_evaluacion(tmp_path, monkeypatch):
    """Sin evaluación del test, aviso; con ella, la figura toma la cohorte del minuto 20 de los dos conjuntos."""
    from src.informe import figuras as F
    monkeypatch.setattr(E, "CARPETA_PNG", tmp_path)
    monkeypatch.setattr(D, "TEST_FINAL", tmp_path / "no_existe")
    assert "Pendiente" in F.fig_test_vs_v().data
    if not (D.METRICAS_EN_VIVO.exists() and (D.DATA / "analysis" / "test_final" / "metricas.csv").exists()):
        return
    monkeypatch.setattr(D, "TEST_FINAL", D.DATA / "analysis" / "test_final")
    filas = F._filas_test_vs_v()
    assert [f[0].split()[0] for f in filas] == ["Antes", "Minuto", "Minuto", "Minuto"]
    assert all(f[2]["conjunto"] == "cohorte 20" and f[2]["ic"] == "dias" for f in filas)
    assert len({int(f[2]["n"]) for f in filas}) == 1      # las mismas partidas en los cuatro momentos
    F.fig_test_vs_v()
    assert (tmp_path / "04a_test_contra_v.png").exists()
