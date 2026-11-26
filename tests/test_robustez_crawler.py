"""
Robustez del crawler y de la base (issue #9, PROJECT_SUMMARY.md §9.S.76).

* El cierre espera al escritor con tope y avisa si no terminó.
* El tier de una persona es el más alto en orden de liga, no en orden alfabético.
* Los errores de escritura tienen su propio contador en el reporte y el latido.
* Las partidas con un número de participantes distinto de 10 no se guardan.
* `export_to_csv` dice por qué no pudo leer `matchups_1v1.csv`.
* `run.py --db` apunta la corrida a una copia de la base.

Todo corre sobre bases SQLite temporales y clientes de Riot simulados: sin red.
"""

import sqlite3
from contextlib import closing
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.config import TIER_ORDER_LIGA
from src.database import PARTICIPANTES_POR_PARTIDA, sql_rango_tier, sql_tier_maximo
from tests.test_crawler import _client_for_roster, _mini_match
from tests.test_identity import _match10


# --------------------------------------------------------------------------- #
# Orden de liga en SQL                                                        #
# --------------------------------------------------------------------------- #
def _max_por_grupo(valores):
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE TABLE t (g INTEGER, tier TEXT)")
        conn.executemany("INSERT INTO t VALUES (1, ?)", [(v,) for v in valores])
        return conn.execute(f"SELECT {sql_tier_maximo('tier')} FROM t GROUP BY g").fetchone()[0]
    finally:
        conn.close()


def test_el_tier_maximo_sigue_el_orden_de_liga_y_no_el_alfabetico():
    # MAX de texto daría DIAMOND (D > C > M).
    assert _max_por_grupo(["DIAMOND", "CHALLENGER", "MASTER"]) == "CHALLENGER"
    assert _max_por_grupo(["GRANDMASTER", "MASTER", "DIAMOND"]) == "GRANDMASTER"
    assert _max_por_grupo(["EMERALD", "IRON", "PLATINUM"]) == "EMERALD"
    assert _max_por_grupo(["master", "diamond"]) == "MASTER"


def test_el_tier_maximo_ignora_desconocidos_y_conserva_el_valor_si_no_hay_otro():
    assert _max_por_grupo(["UNKNOWN", "MASTER", None]) == "MASTER"
    assert _max_por_grupo(["UNKNOWN", None]) == "UNKNOWN"
    assert _max_por_grupo([None]) is None


def test_el_rango_sql_respeta_el_orden_de_la_constante():
    conn = sqlite3.connect(":memory:")
    try:
        rangos = [conn.execute(f"SELECT {sql_rango_tier('?')}", (t,)).fetchone()[0]
                  for t in TIER_ORDER_LIGA]
    finally:
        conn.close()
    assert rangos == list(range(1, len(TIER_ORDER_LIGA) + 1))
    assert TIER_ORDER_LIGA.index("DIAMOND") < TIER_ORDER_LIGA.index("MASTER") \
        < TIER_ORDER_LIGA.index("GRANDMASTER") < TIER_ORDER_LIGA.index("CHALLENGER")


def _persona_con_dos_tiers(temp_db):
    """Una persona con dos PUUIDs en el padrón: uno DIAMOND y otro CHALLENGER.

    El Riot ID compartido no alcanza (dos PUUIDs del padrón quedan separados); el
    vínculo de alias legacy sí los une. En la base real esto no pasa hoy: es el
    caso que el orden de liga cubre si llega a pasar.
    """
    temp_db.save_players([
        {"puuid": "dual_p0", "summoner_id": "s0", "tier": "DIAMOND", "rank": "I",
         "league_points": 10, "wins": 0, "losses": 0},
        {"puuid": "dual_p1", "summoner_id": "s1", "tier": "CHALLENGER", "rank": "I",
         "league_points": 900, "wins": 0, "losses": 0},
    ])
    temp_db.record_riot_id_observations([("dual_p0", "Dual", "LAS")], project_index=None)
    temp_db.record_riot_id_observations([("dual_p1", "Dual", "LAS")], project_index=1)
    assert temp_db.record_puuid_alias("dual_p0", "dual_p1")
    temp_db.save_matches_and_timelines_batch([
        ("LA2_D1", _mini_match("LA2_D1", "dual_p0"), None, 0),
        ("LA2_D2", _mini_match("LA2_D2", "dual_p1"), None, 1),
    ])


def test_los_exports_toman_el_tier_mas_alto_de_la_persona(temp_db):
    from export_to_csv import export_full_dataset, export_players

    _persona_con_dos_tiers(temp_db)
    with tempfile.TemporaryDirectory() as tmp:
        full = export_full_dataset(temp_db.db_path, Path(tmp) / "full.csv")
        players = export_players(temp_db.db_path, Path(tmp) / "players.csv")
    assert set(full["player_tier"]) == {"CHALLENGER"}
    assert players["player_tier"].tolist() == ["CHALLENGER"]


def test_la_cobertura_cuenta_a_la_persona_en_su_tier_mas_alto(temp_db):
    _persona_con_dos_tiers(temp_db)
    temp_db.materialize_player_id_map()
    cov = temp_db.get_coverage_stats()
    assert cov["CHALLENGER"]["total"] == 1
    assert cov["DIAMOND"]["total"] == 0


# --------------------------------------------------------------------------- #
# Partidas con un número de participantes distinto de 10                      #
# --------------------------------------------------------------------------- #
@pytest.fixture
def db_estricta(temp_db):
    """`temp_db` con el control de 10 participantes encendido (el de producción)."""
    temp_db.participantes_esperados = PARTICIPANTES_POR_PARTIDA
    return temp_db


def test_por_defecto_la_base_exige_10_participantes():
    with tempfile.TemporaryDirectory() as tmp:
        from src.database import DatabaseManager
        assert DatabaseManager(Path(tmp) / "x.db").participantes_esperados == 10


@pytest.mark.parametrize("n", [9, 11, 0])
def test_el_lote_rechaza_partidas_sin_10_participantes_y_guarda_el_resto(db_estricta, n, capsys):
    rara = _match10("LA2_RARA", [f"r{i}" for i in range(n)])
    buena = _match10("LA2_BUENA", [f"b{i}" for i in range(10)])
    saved, _ = db_estricta.save_match_batch_with_counts(
        [("LA2_RARA", rara, None), ("LA2_BUENA", buena, None)])
    assert saved == 1
    assert db_estricta.partidas_rechazadas_participantes == 1
    assert "LA2_RARA rechazada" in capsys.readouterr().out
    with closing(sqlite3.connect(db_estricta.db_path)) as conn:
        assert conn.execute("SELECT match_id FROM matches").fetchall() == [("LA2_BUENA",)]
        assert conn.execute("SELECT COUNT(*) FROM match_participants").fetchone()[0] == 10
        # No se anota como descartada: un payload truncado se vuelve a pedir.
        assert conn.execute("SELECT COUNT(*) FROM discarded_matches").fetchone()[0] == 0
    assert db_estricta.filter_unprocessed_match_ids(["LA2_RARA"]) == ["LA2_RARA"]


def test_tambien_rechaza_10_filas_con_un_puuid_repetido(db_estricta):
    repetida = _match10("LA2_REP", [f"x{i}" for i in range(9)] + ["x0"])
    assert db_estricta.save_matches_and_timelines_batch([("LA2_REP", repetida, None)]) == 0
    assert db_estricta.partidas_rechazadas_participantes == 1


def test_el_guardado_individual_tambien_rechaza(db_estricta):
    assert db_estricta.save_match_and_participants(_match10("LA2_9", [f"n{i}" for i in range(9)])) is False
    assert db_estricta.save_match_and_participants(_match10("LA2_10", [f"m{i}" for i in range(10)])) is True
    assert db_estricta.get_total_matches() == 1
    assert db_estricta.partidas_rechazadas_participantes == 1


def test_el_reporte_cuenta_las_rechazadas_dentro_de_las_descartadas(db_estricta, monkeypatch, capsys):
    import src.collector as collector

    client = _client_for_roster(db_estricta)
    client.get_match_ids_by_puuid.return_value = ["LA2_OK", "LA2_CORTA"]
    detalles = {"LA2_OK": _match10("LA2_OK", ["p0"] + [f"o{i}" for i in range(9)]),
                "LA2_CORTA": _match10("LA2_CORTA", ["p0"] + [f"c{i}" for i in range(4)])}
    client.get_match_detail.side_effect = lambda mid: detalles[mid]
    monkeypatch.setattr(collector, "export_all_datasets", MagicMock())
    report = collector.DataCollector(client, db_estricta).collect_matches(depth_first=False, max_workers=1)
    assert report["matches_new"] == 1
    assert report["matches_rejected_participants"] == 1
    assert report["matches_discarded"] == 1
    assert report["write_errors"] == 0
    assert "rechazadas por no traer 10 participantes: 1" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Errores de escritura: contador propio                                       #
# --------------------------------------------------------------------------- #
def test_los_errores_de_escritura_van_a_su_propio_contador(temp_db, monkeypatch, capsys):
    import src.collector as collector

    client = _client_for_roster(temp_db)
    client.get_match_ids_by_puuid.return_value = ["LA2_E1", "LA2_E2"]
    client.get_match_detail.side_effect = lambda mid: _mini_match(mid, "p0")

    def falla(batch):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(temp_db, "save_match_batch_with_counts", falla)
    monkeypatch.setattr(collector, "export_all_datasets", MagicMock())
    report = collector.DataCollector(client, temp_db).collect_matches(depth_first=False, max_workers=1)
    assert report["write_errors"] == 2
    assert report["matches_discarded"] == 0
    assert report["matches_new"] == 0
    out = capsys.readouterr().out
    assert "errores de escritura en la BD (se reintentan): 2" in out
    assert "2 partidas sin guardar" in out


def test_el_latido_guarda_los_contadores_nuevos(temp_db):
    temp_db.log_crawler_metric({"session_id": "s", "write_errors": 3, "rejected_participants": 2})
    with closing(sqlite3.connect(temp_db.db_path)) as conn:
        assert conn.execute("SELECT write_errors, rejected_participants "
                            "FROM crawler_metrics_timeline").fetchall() == [(3, 2)]


def test_la_migracion_agrega_las_columnas_a_una_base_vieja(tmp_path):
    from src.database import DatabaseManager

    vieja = tmp_path / "vieja.db"
    with closing(sqlite3.connect(vieja)) as conn:
        conn.execute("CREATE TABLE crawler_metrics_timeline (id INTEGER PRIMARY KEY, session_id TEXT, "
                     "timestamp TIMESTAMP, elapsed_minutes REAL, active_keys INTEGER, workers INTEGER, "
                     "players_crawled INTEGER, total_matches_in_db INTEGER, new_matches_saved INTEGER, "
                     "duplicate_matches_skipped INTEGER, discarded_matches INTEGER, "
                     "net_matches_per_min REAL, requests_per_min REAL, c50_challenger INTEGER, "
                     "c50_grandmaster INTEGER, c50_master INTEGER, c50_total INTEGER, db_size_mb REAL)")
    DatabaseManager(vieja)
    DatabaseManager(vieja)  # idempotente
    with closing(sqlite3.connect(vieja)) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(crawler_metrics_timeline)")}
    assert {"write_errors", "rejected_participants"} <= cols


# --------------------------------------------------------------------------- #
# Cierre: el join del escritor tiene tope                                     #
# --------------------------------------------------------------------------- #
def test_si_el_escritor_no_termina_el_cierre_avisa_y_no_exporta(temp_db, monkeypatch, capsys):
    import src.collector as collector

    client = _client_for_roster(temp_db)
    client.get_match_ids_by_puuid.return_value = ["LA2_T1"]
    client.get_match_detail.side_effect = lambda mid: _mini_match(mid, "p0")
    liberar = threading.Event()
    save = temp_db.save_match_batch_with_counts

    def trabado(batch):
        liberar.wait(10)
        return save(batch)

    hilos = []

    class Escritor(threading.Thread):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            hilos.append(self)

    monkeypatch.setattr(temp_db, "save_match_batch_with_counts", trabado)
    monkeypatch.setattr(collector, "WRITER_JOIN_TIMEOUT_SECONDS", 0.3)
    monkeypatch.setattr(collector, "threading",
                        SimpleNamespace(Thread=Escritor, Lock=threading.Lock, Event=threading.Event))
    export = MagicMock()
    monkeypatch.setattr(collector, "export_all_datasets", export)
    materializar = MagicMock()
    monkeypatch.setattr(temp_db, "materialize_player_id_map", materializar)
    try:
        report = collector.DataCollector(client, temp_db).collect_matches(depth_first=False, max_workers=1)
    finally:
        liberar.set()
        for h in hilos:
            h.join(10)  # que suelte la base antes de borrar el directorio temporal
    assert report["writer_timed_out"] is True
    export.assert_not_called()
    materializar.assert_called_once()  # solo el refresco del arranque, no el del cierre
    out = capsys.readouterr().out
    assert "el escritor de la base no terminó" in out
    assert "Export de CSV omitido" in out


# --------------------------------------------------------------------------- #
# export_to_csv: el matchups ilegible se informa                              #
# --------------------------------------------------------------------------- #
def test_un_matchups_ilegible_se_informa_y_players_sale_igual(temp_db, capsys):
    from export_to_csv import export_players

    temp_db.save_players([{"puuid": "p0", "tier": "MASTER", "rank": "I"}])
    temp_db.save_matches_and_timelines_batch([("LA2_M", _mini_match("LA2_M", "p0"), None)])
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "matchups_1v1.csv").write_text("", encoding="utf-8")  # vacío: EmptyDataError
        df = export_players(temp_db.db_path, Path(tmp) / "players.csv")
    assert len(df) == 1
    out = capsys.readouterr().out
    assert "No se pudo leer 'matchups_1v1.csv'" in out and "EmptyDataError" in out


# --------------------------------------------------------------------------- #
# run.py --db                                                                 #
# --------------------------------------------------------------------------- #
def test_run_db_apunta_a_la_copia_y_rechaza_rutas_inexistentes(tmp_path, capsys):
    import run

    assert run.resolve_db_path(run.parse_args([])) == run.DEFAULT_DB_PATH
    assert run.resolve_db_path(run.parse_args(["--db", str(tmp_path / "no.db")])) is None
    copia = tmp_path / "copia.db"
    copia.touch()
    assert run.resolve_db_path(run.parse_args(["--db", str(copia)])) == copia.resolve()
    assert run.main(["--db", str(tmp_path / "no.db"), "--solo-validar"]) == 2
    assert run.main(["--db", str(copia), "--respaldar", "--solo-validar"]) == 2
    assert not (tmp_path / "no.db").exists()


def test_run_solo_validar_sobre_una_copia_no_escribe_fuera_de_su_carpeta(temp_db, tmp_path):
    import run

    copia_dir = tmp_path / "copia"
    copia_dir.mkdir()
    copia = copia_dir / "lol_las.db"
    temp_db.save_matches_and_timelines_batch([("LA2_V", _mini_match("LA2_V", "p0"), None)])
    with closing(sqlite3.connect(temp_db.db_path)) as src, closing(sqlite3.connect(copia)) as dst:
        src.backup(dst)
    assert run.main(["--db", str(copia), "--solo-validar"]) == 0
    # El cursor de la auditoría queda junto a la copia.
    assert (copia_dir / "validation_state.json").exists()
