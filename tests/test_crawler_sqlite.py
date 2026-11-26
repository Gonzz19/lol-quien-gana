"""Regresiones de SQLite y del monitor. Todas las bases son temporales."""

import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from scripts.ops import monitor_live
from src.database import DatabaseManager
from src.validation import open_readonly


def test_sqlite_aplica_cache_y_memoria_sin_perder_solo_lectura(temp_db):
    with temp_db._get_connection() as writer, closing(open_readonly(temp_db.db_path)) as reader:
        for conn in (writer, reader):
            assert conn.execute("PRAGMA cache_size").fetchone()[0] == -262144
            assert conn.execute("PRAGMA temp_store").fetchone()[0] == 2
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            reader.execute("DELETE FROM players")


def test_solo_migra_la_vista_si_cambio_su_definicion(temp_db):
    with temp_db._get_connection() as conn:
        expected_sql = conn.execute("SELECT sql FROM sqlite_master WHERE name='v_puuid_canonical'").fetchone()[0]
        initial_version = conn.execute("PRAGMA schema_version").fetchone()[0]
        DatabaseManager(temp_db.db_path)
        assert conn.execute("PRAGMA schema_version").fetchone()[0] == initial_version

        conn.execute("DROP VIEW v_puuid_canonical")
        conn.execute("CREATE VIEW v_puuid_canonical AS SELECT 'old' AS alias_puuid")
        DatabaseManager(temp_db.db_path)
        assert conn.execute("SELECT sql FROM sqlite_master WHERE name='v_puuid_canonical'").fetchone()[0] == expected_sql


def test_recuento_por_persona_separa_los_nombres_reutilizados(temp_db):
    temp_db.save_players([{"puuid": p, "tier": "MASTER", "rank": "I"} for p in ("p0a", "p0b")])
    for original, alias in (("p0a", "old_a"), ("p0b", "old_b")):
        temp_db.record_puuid_alias(original, alias, "Nombre", "LAS")
    with temp_db._get_connection() as conn:
        conn.executemany("INSERT INTO matches (match_id) VALUES (?)", [("LA2_1",), ("LA2_2",), ("LA2_3",)])
        conn.executemany("INSERT INTO match_participants (match_id, puuid) VALUES (?, ?)",
                         [("LA2_1", "p0a"), ("LA2_1", "old_a"), ("LA2_2", "old_a"), ("LA2_3", "old_b")])
        conn.commit()
    assert temp_db.get_player_match_count("p0a") == 3  # compatibilidad con una base sin mapa
    temp_db.materialize_player_id_map()
    assert temp_db.get_player_match_count("p0a") == 2  # deduplicación + rechazo de nombre compartido
    assert temp_db.get_player_match_count("p0b") == 1
    assert temp_db.get_player_match_count("desconocido") == 0


def test_monitor_refresca_por_persona_y_conserva_un_latido_de_cinco_minutos(temp_db, monkeypatch):
    temp_db.save_players([{"puuid": p, "tier": "MASTER", "rank": "I"} for p in ("p0", "p1")])
    with temp_db._get_connection() as conn:
        matches = [(f"LA2_{i}",) for i in range(50)]
        conn.executemany("INSERT INTO matches (match_id) VALUES (?)", matches)
        conn.executemany("INSERT INTO match_participants (match_id, puuid) VALUES (?, ?)",
                         [(mid, "p0" if i < 25 else "p1") for i, (mid,) in enumerate(matches)])
        conn.executemany("INSERT INTO player_id_map (puuid, player_id) VALUES (?, 'persona')", [("p0",), ("p1",)])
        then = datetime.now(timezone.utc) - timedelta(seconds=300)
        conn.execute("INSERT INTO crawler_metrics_timeline (timestamp, net_matches_per_min) VALUES (?, 10)",
                     (then.strftime("%Y-%m-%d %H:%M:%S"),))
        conn.commit()

    ticks = iter((100, 102, 160))
    monkeypatch.setattr(monitor_live, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    with closing(open_readonly(temp_db.db_path)) as conn:
        statements = []
        conn.set_trace_callback(statements.append)
        first = monitor_live.get_stats(conn)
        assert first["is_crawler_active"] and first["last_recorded_rate"] == 10
        assert first["tier_stats"]["MASTER"] == {"total": 1, "c50": 1, "c100": 0, "c200": 0, "c300": 0}
        assert first["tier_stats"] == temp_db.get_coverage_stats()

        statements.clear()
        with temp_db._get_connection() as writer:
            writer.execute("INSERT INTO matches (match_id) VALUES ('LA2_new')")
            writer.execute("INSERT INTO match_participants (match_id, puuid) VALUES ('LA2_new', 'p0')")
            writer.commit()
        fast = monitor_live.get_stats(conn, 50, 100, previous=first, db_path=temp_db.db_path)
        assert len(statements) == 2
        assert fast["total_matches"] == 51 and fast["total_participants"] == 50
        assert fast["new_in_session"] == 1 and fast["tier_stats"] == first["tier_stats"]
        refreshed = monitor_live.get_stats(conn, previous=fast, db_path=temp_db.db_path)
        assert refreshed["total_participants"] == 51
