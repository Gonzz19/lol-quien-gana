"""Orquestación desatendida (scripts/collection/secuencia_fin_de_semana.py)."""

import pytest

from scripts.collection import secuencia_fin_de_semana as sec


@pytest.fixture(autouse=True)
def _log_aislado(monkeypatch, tmp_path):
    """Ningún test escribe en logs/secuencia_fin_de_semana.log, el log real de la corrida."""
    monkeypatch.setattr(sec, "LOG", tmp_path / "secuencia.log")


def test_repite_mientras_avanza_y_para_al_completar(monkeypatch):
    monkeypatch.setattr(sec.time, "sleep", lambda s: None)
    valores = iter([10, 6, 3, 0])
    corridas = []
    sec.repetir_mientras_avance(lambda: corridas.append(1), lambda: next(valores),
                                lambda n: n == 0, lambda: 999, "t")
    assert len(corridas) == 3


def test_corta_si_deja_de_avanzar(monkeypatch):
    monkeypatch.setattr(sec.time, "sleep", lambda s: None)
    corridas = []
    sec.repetir_mientras_avance(lambda: corridas.append(1), lambda: 7,
                                lambda n: n == 0, lambda: 999, "t")
    assert len(corridas) == sec.MAX_SIN_AVANCE


def test_no_arranca_sin_tiempo(monkeypatch):
    corridas = []
    sec.repetir_mientras_avance(lambda: corridas.append(1), lambda: 7,
                                lambda n: n == 0, lambda: 1, "t")
    assert corridas == []


def test_la_secuencia_y_el_crawler_cuentan_la_misma_ventana(temp_db, monkeypatch):
    import src.config as config
    from src.window_crawl import WindowCrawler, parse_window_date

    monkeypatch.setattr(config, "DEFAULT_PLATFORM", "br1")
    monkeypatch.setattr(sec, "DB", temp_db.db_path)
    window = WindowCrawler(None, temp_db, parse_window_date(sec.PRIMERA_VENTANA))
    with temp_db._get_connection() as conn:
        conn.executemany("INSERT INTO crawl_window_matches VALUES (?, ?)", [
            (window.key, mid) for mid in ["LA2_X", "BR1_1", "BR1_2", "BR1_3", "BR1_4"]])
        conn.execute("INSERT INTO matches (match_id) VALUES ('BR1_1')")
        conn.execute("INSERT INTO discarded_matches (match_id, reason) VALUES ('BR1_2', 'remake')")
        conn.commit()
    assert window.pending_match_ids() == ["BR1_3", "BR1_4"]
    assert window.status()["pendientes"] == sec.pendientes_ventana(sec.PRIMERA_VENTANA) == 2


def test_la_secuencia_corre_los_pasos_en_orden(monkeypatch, tmp_path):
    monkeypatch.setattr(sec, "LOG", tmp_path / "s.log")
    monkeypatch.setattr(sec.time, "sleep", lambda s: None)
    llamadas = []
    monkeypatch.setattr(sec, "run_py", lambda args, minutos: llamadas.append(args) or 0)
    pend = {"2026-09-01": iter([5, 0]), "2026-10-22": iter([0])}
    monkeypatch.setattr(sec, "pendientes_ventana", lambda d: next(pend.get(d, iter([])), 0))
    bajo = iter([4, 0])
    monkeypatch.setattr(sec, "master_bajo_50", lambda: next(bajo))
    # Reloj falso: cada consulta avanza 20 min (el log también consulta la hora).
    t = {"n": 0}

    class Reloj(sec.datetime):
        @classmethod
        def now(cls):
            t["n"] += 1
            return sec.datetime(2026, 10, 23, 10) + sec.timedelta(minutes=20 * t["n"])

    monkeypatch.setattr(sec, "datetime", Reloj)
    sec.main(["--hasta", "2026-10-25 12:00"])
    assert llamadas[0] == []                                   # primero: crawler normal (Master a 50)
    assert llamadas[1][:4] == ["--ventana-desde", "2026-09-01", "--fase", "descargar"]
    assert llamadas[2][:4] == ["--ventana-desde", "2026-10-22", "--fase", "todo"]
    # Entre ciclos del paso 3: relleno, GM+ primero y Master después.
    assert llamadas[3] == ["--tier", "CHALLENGER,GRANDMASTER", "--partidas-por-jugador", "300",
                           "--prioridad-lp", "0.7"]
    assert llamadas[4] == ["--tier", "MASTER", "--partidas-por-jugador", "300", "--prioridad-lp", "0.7"]
