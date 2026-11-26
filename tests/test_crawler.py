"""
Pruebas unitarias para la base de datos, deduplicación y componentes del crawler.
"""

import tempfile
import threading
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.database import DatabaseManager
from src.rate_limiter import RiotRateLimiter
from src.riot_client import RiotAPIClient


def test_player_save_and_update(temp_db):
    players = [
        {"puuid": "puuid_1", "summoner_id": "summ_1", "tier": "CHALLENGER", "rank": "I", "league_points": 500, "wins": 100, "losses": 50},
        {"puuid": "puuid_2", "summoner_id": "summ_2", "tier": "MASTER", "rank": "I", "league_points": 120, "wins": 60, "losses": 40}
    ]
    saved = temp_db.save_players(players)
    assert saved == 2

    # Verificar que se recuperan correctamente
    all_players = temp_db.get_all_players()
    assert len(all_players) == 2

    # Probar actualización (on conflict)
    updated_players = [
        {"puuid": "puuid_1", "summoner_id": "summ_1", "tier": "CHALLENGER", "rank": "I", "league_points": 550, "wins": 105, "losses": 50}
    ]
    temp_db.save_players(updated_players)
    all_players = temp_db.get_all_players()
    assert len(all_players) == 2
    assert all_players[0]["league_points"] == 550


def test_match_save_and_deduplication(temp_db):
    sample_match = {
        "metadata": {
            "matchId": "LA2_123456"
        },
        "info": {
            "gameCreation": 1690000000000,
            "gameDuration": 1800,
            "gameVersion": "14.1.1",
            "queueId": 420,
            "participants": [
                {
                    "puuid": "puuid_1",
                    "participantId": 1,
                    "teamId": 100,
                    "win": True,
                    "championId": 157,
                    "championName": "Yasuo",
                    "individualPosition": "MIDDLE",
                    "kills": 10,
                    "deaths": 2,
                    "assists": 5,
                    "totalDamageDealtToChampions": 25000,
                    "totalDamageTaken": 15000,
                    "goldEarned": 14000,
                    "totalMinionsKilled": 200,
                    "neutralMinionsKilled": 20,
                    "visionScore": 30,
                    "champLevel": 16,
                    "item0": 3031, "item1": 3046, "item2": 0, "item3": 0, "item4": 0, "item5": 0, "item6": 3340,
                    "firstBloodKill": True,
                    "firstTowerKill": False
                },
                {
                    "puuid": "puuid_2",
                    "participantId": 2,
                    "teamId": 200,
                    "win": False,
                    "championId": 84,
                    "championName": "Akali",
                    "individualPosition": "MIDDLE",
                    "kills": 2,
                    "deaths": 8,
                    "assists": 3,
                    "totalDamageDealtToChampions": 12000,
                    "totalDamageTaken": 18000,
                    "goldEarned": 9000,
                    "totalMinionsKilled": 150,
                    "neutralMinionsKilled": 5,
                    "visionScore": 15,
                    "champLevel": 13,
                    "item0": 3157, "item1": 0, "item2": 0, "item3": 0, "item4": 0, "item5": 0, "item6": 3340,
                    "firstBloodKill": False,
                    "firstTowerKill": False
                }
            ]
        }
    }

    # 1. Guardar por primera vez
    saved = temp_db.save_match_and_participants(sample_match)
    assert saved is True
    assert temp_db.is_match_saved("LA2_123456") is True

    # 2. Filtrado de partidas existentes (Deduplicación)
    unprocessed = temp_db.filter_unprocessed_match_ids(["LA2_123456", "LA2_999999"])
    assert unprocessed == ["LA2_999999"]

    # 3. Intentar guardar de nuevo (debe ser ignorado silenciosamente sin romper nada)
    saved_again = temp_db.save_match_and_participants(sample_match)
    assert saved_again is True

    summary = temp_db.get_database_summary()
    assert summary["total_matches"] == 1
    assert summary["total_participant_records"] == 2


def test_rate_limiter_slot():
    limiter = RiotRateLimiter(short_limit=5, short_window=0.5, long_limit=10, long_window=1.0)
    for _ in range(5):
        limiter.wait_for_slot()
    # Verifica que procesó las ranuras sin excepciones
    assert len(limiter.short_history) <= 5


def test_timeline_save_and_export(temp_db):
    match_id = "LA2_999999"
    # Insert dummy match
    sample_match = {
        "metadata": {"matchId": match_id},
        "info": {
            "gameCreation": 1690000000000,
            "gameDuration": 1800,
            "gameVersion": "14.1.1",
            "queueId": 420,
            "participants": [
                {"puuid": "puuid_1", "participantId": 1, "teamId": 100, "win": True, "championId": 1, "championName": "Annie", "individualPosition": "MIDDLE", "kills": 5, "deaths": 1, "assists": 4},
                {"puuid": "puuid_2", "participantId": 2, "teamId": 200, "win": False, "championId": 2, "championName": "Olaf", "individualPosition": "TOP", "kills": 1, "deaths": 5, "assists": 1}
            ]
        }
    }
    temp_db.save_match_and_participants(sample_match)

    dummy_timeline = {
        "info": {
            "frames": [
                {"events": [], "participantFrames": {str(i): {"totalGold": 500, "xp": 0, "level": 1, "minionsKilled": 0, "jungleMinionsKilled": 0, "damageStats": {"totalDamageDoneToChampions": 0}} for i in range(1, 11)}}
                for _ in range(25)
            ]
        }
    }
    # Modify min 10
    dummy_timeline["info"]["frames"][10]["participantFrames"]["1"]["totalGold"] = 3500
    dummy_timeline["info"]["frames"][10]["participantFrames"]["1"]["minionsKilled"] = 80
    dummy_timeline["info"]["frames"][10]["events"] = [{"type": "CHAMPION_KILL", "killerId": 1, "victimId": 2, "assistingParticipantIds": []}]

    saved = temp_db.save_match_timeline(match_id, dummy_timeline, {1: "puuid_1", 2: "puuid_2"})
    assert saved is True
    assert temp_db.is_timeline_saved(match_id) is True

    # Test export
    from export_to_csv import export_all_datasets
    with tempfile.TemporaryDirectory() as tmp_out:
        out_dir = Path(tmp_out)
        export_all_datasets(temp_db.db_path, out_dir)
        assert (out_dir / "full_dataset.csv").exists()
        assert (out_dir / "players.csv").exists()
        assert (out_dir / "champions_by_lane.csv").exists()
        assert (out_dir / "matchups_1v1.csv").exists()


def test_multi_key_client_and_rate_limiter():
    from src.riot_client import RiotAPIClient
    
    # 1. Probar inicialización con múltiples claves
    client = RiotAPIClient(api_keys=["RGAPI-TEST-KEY-1", "RGAPI-TEST-KEY-2"])
    assert client.num_active_keys == 2
    assert len(client.key_pool) == 2
    assert client.api_key == "RGAPI-TEST-KEY-1"

    # 2. Probar selección balanceada de clave
    selected_ctx1 = client._select_key_and_acquire()
    selected_ctx2 = client._select_key_and_acquire()
    # Con ambas claves libres (wait_time=0), deben ser consumidas secuencialmente
    assert selected_ctx1.key_index in (0, 1)
    assert selected_ctx2.key_index in (0, 1)

    # 3. Probar desactivación de una clave inválida
    selected_ctx1.is_active = False
    assert client.num_active_keys == 1
    next_ctx = client._select_key_and_acquire()
    assert next_ctx.key_index == 1


def test_tier_adaptive_thresholds(temp_db):
    from src.config import DEFAULT_TIER_MATCH_TARGETS, DYNAMIC_TIER_MATCH_TARGETS, TEMPORARY_DFS_TARGET
    
    # 1. Verificar configuración de umbrales dinámicos y temporales
    assert TEMPORARY_DFS_TARGET == 50
    assert DEFAULT_TIER_MATCH_TARGETS["CHALLENGER"] == 50
    assert DEFAULT_TIER_MATCH_TARGETS["DIAMOND"] == 50
    assert DYNAMIC_TIER_MATCH_TARGETS["CHALLENGER"] == 160
    assert DYNAMIC_TIER_MATCH_TARGETS["GRANDMASTER"] == 175
    assert DYNAMIC_TIER_MATCH_TARGETS["MASTER"] == 215
    assert DYNAMIC_TIER_MATCH_TARGETS["DIAMOND"] == 260

    # 2. Insertar jugadores de diferentes tiers
    players = [
        {"puuid": "p_chall", "summoner_id": "s1", "tier": "CHALLENGER", "rank": "I", "league_points": 800, "wins": 200, "losses": 100},
        {"puuid": "p_dia", "summoner_id": "s2", "tier": "DIAMOND", "rank": "II", "league_points": 50, "wins": 100, "losses": 90}
    ]
    temp_db.save_players(players)

    # 3. Probar conteo inicial
    assert temp_db.get_player_match_count("p_chall") == 0
    assert temp_db.get_player_match_count("p_dia") == 0

    # 4. Probar cálculo de partidas faltantes con el threshold temporal activo de 50
    assert max(0, DEFAULT_TIER_MATCH_TARGETS["CHALLENGER"] - temp_db.get_player_match_count("p_chall")) == 50
    assert max(0, DEFAULT_TIER_MATCH_TARGETS["DIAMOND"] - temp_db.get_player_match_count("p_dia")) == 50


def test_batch_save_matches_and_timelines(temp_db):
    match_1 = {
        "metadata": {"matchId": "LA2_BATCH_001"},
        "info": {
            "gameCreation": 1690000000000,
            "gameDuration": 1500,
            "gameVersion": "14.1.1",
            "queueId": 420,
            "participants": [{
                "puuid": "puuid_batch_1",
                "participantId": 1,
                "teamId": 100,
                "win": True,
                "championId": 81,
                "championName": "Ezreal",
                "individualPosition": "BOTTOM"
            }]
        }
    }
    match_2 = {
        "metadata": {"matchId": "LA2_BATCH_002"},
        "info": {
            "gameCreation": 1690000100000,
            "gameDuration": 1600,
            "gameVersion": "14.1.1",
            "queueId": 420,
            "participants": [{
                "puuid": "puuid_batch_2",
                "participantId": 1,
                "teamId": 200,
                "win": False,
                "championId": 222,
                "championName": "Jinx",
                "individualPosition": "BOTTOM"
            }]
        }
    }
    batch = [
        ("LA2_BATCH_001", match_1, None),
        ("LA2_BATCH_002", match_2, None)
    ]

    saved = temp_db.save_matches_and_timelines_batch(batch)
    assert saved == 2
    assert temp_db.is_match_saved("LA2_BATCH_001") is True
    assert temp_db.is_match_saved("LA2_BATCH_002") is True
    summary = temp_db.get_database_summary()
    assert summary["total_matches"] == 2


def test_concurrent_multithreaded_key_dispatch():
    from concurrent.futures import ThreadPoolExecutor

    client = RiotAPIClient(
        api_keys=["RGAPI-TEST-THREAD-1", "RGAPI-TEST-THREAD-2"],
        rate_limiter=RiotRateLimiter(short_limit=50, short_window=1.0, long_limit=100, long_window=1.0)
    )

    def worker():
        ctx = client._select_key_and_acquire()
        return ctx.key_index

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker) for _ in range(16)]
        results = [f.result() for f in futures]

    assert len(results) == 16
    assert all(r in (0, 1) for r in results)


def test_remake_discard_and_negative_cache(temp_db):
    remake_match = {
        "metadata": {"matchId": "LA2_REMAKE_123"},
        "info": {
            "gameCreation": 1690000000000,
            "gameDuration": 210,  # 3.5 minutos (remake <= 300s)
            "gameVersion": "14.1.1",
            "queueId": 420,
            "participants": [{
                "puuid": "puuid_remake_1",
                "participantId": 1,
                "teamId": 100,
                "win": False,
                "championId": 81,
                "championName": "Ezreal",
                "individualPosition": "BOTTOM"
            }]
        }
    }

    # 1. Probar que no se guarda en matches pero se guarda en discarded_matches
    saved = temp_db.save_matches_and_timelines_batch([("LA2_REMAKE_123", remake_match, None)])
    assert saved == 0
    assert temp_db.is_match_saved("LA2_REMAKE_123") is False

    summary = temp_db.get_database_summary()
    assert summary["total_matches"] == 0
    assert summary["total_discarded_matches"] == 1

    # 2. Probar que filter_unprocessed_match_ids filtra el remake con 0 requests adicionales
    unprocessed = temp_db.filter_unprocessed_match_ids(["LA2_REMAKE_123", "LA2_REAL_999"])
    assert unprocessed == ["LA2_REAL_999"]


# ---------------------------------------------------------------------------
# Sistema multi-proyecto: PUUIDs por API key unificados en identidad canónica
# ---------------------------------------------------------------------------

def test_canonical_id_format():
    assert DatabaseManager.canonical_id_for("Faker", "LAS1") == "faker#las1"
    assert DatabaseManager.canonical_id_for("  ShOwMaKeR ", " kr1 ") == "showmaker#kr1"


def test_register_projects_and_identity_crud(temp_db):
    idx = temp_db.register_api_projects(["KEY_PROJECT_0_ABCDEFGHIJ", "KEY_PROJECT_1_KLMNOPQRST"])
    assert idx == [0, 1]
    assert temp_db.get_num_projects() == 2

    c1 = temp_db.upsert_player_identity("TestPlayer", "LAS1", "puuid_P0_aaa", 0,
                                        tier="CHALLENGER", rank="I", league_points=900)
    c2 = temp_db.upsert_player_identity("testplayer", "las1", "puuid_P1_bbb", 1,
                                        tier="CHALLENGER", rank="I", league_points=900)
    assert c1 == c2 == "testplayer#las1"

    # Idempotencia: re-insertar no duplica
    temp_db.upsert_player_identity("TestPlayer", "LAS1", "puuid_P0_aaa", 0, tier="CHALLENGER")
    rows = temp_db.get_puuids_for_canonical("testplayer#las1")
    assert len(rows) == 2

    # Resolución O(1) por PUUID
    assert temp_db.resolve_canonical_id("puuid_P1_bbb") == "testplayer#las1"
    assert temp_db.resolve_canonical_id("puuid_desconocido") is None

    # Invariante: 2 PUUIDs == 2 proyectos
    cov = temp_db.get_identity_coverage()
    assert cov["num_projects"] == 2
    assert cov["total_identities"] == 1
    assert cov["complete"] == 1
    assert cov["incomplete"] == 0

    # Caché alias -> canónico (proyecto 0 es canónico; la vista incluye
    # también la fila identidad P0->P0)
    n = temp_db.reload_translation_cache()
    assert n == 2
    assert temp_db._translation_cache == {"puuid_P1_bbb": "puuid_P0_aaa",
                                                  "puuid_P0_aaa": "puuid_P0_aaa"}
    assert temp_db._canonical_puuid("puuid_P1_bbb") == "puuid_P0_aaa"
    assert temp_db._canonical_puuid("puuid_sin_mapear") == "puuid_sin_mapear"


def _mini_match(match_id, puuid, champion="Ahri"):
    return {
        "metadata": {"matchId": match_id},
        "info": {
            "gameCreation": 1690000000000,
            "gameDuration": 1800,
            "gameVersion": "16.17.1",
            "queueId": 420,
            "participants": [{
                "puuid": puuid, "participantId": 1, "teamId": 100, "win": True,
                "championId": 103, "championName": champion,
                "individualPosition": "MIDDLE", "kills": 5, "deaths": 2, "assists": 6
            }]
        }
    }


def test_batch_save_keeps_raw_puuid_and_records_its_project(temp_db):
    """La ingesta guarda el PUUID tal cual vino y anota el proyecto de la key que lo trajo.

    Contrato anterior: traducía al "canónico" al escribir, y ese canónico podía
    ser un PUUID de un proyecto sin key en el pool (§9.S.4). La persona se
    unifica en lectura; el PUUID guardado tiene que seguir siendo descifrable.
    """
    temp_db.register_api_projects(["KEY0_ABCDEFGHIJ", "KEY1_KLMNOPQRST"])
    temp_db.upsert_player_identity("Hero", "LA2", "puuid_canon", 0, tier="GRANDMASTER")
    temp_db.upsert_player_identity("Hero", "LA2", "puuid_alias", 1, tier="GRANDMASTER")
    temp_db.reload_translation_cache()

    saved = temp_db.save_matches_and_timelines_batch([
        ("LA2_TR_001", _mini_match("LA2_TR_001", "puuid_alias"), None, 1),
    ])
    assert saved == 1
    assert temp_db.get_participant_puuid_map("LA2_TR_001") == {1: "puuid_alias"}
    # La persona sigue siendo una sola: la partida cuenta desde cualquiera de sus PUUIDs.
    assert temp_db.get_player_match_count("puuid_canon") == 1
    with temp_db._get_connection() as conn:
        assert tuple(conn.execute("SELECT project_index, source FROM puuid_projects "
                                  "WHERE puuid = 'puuid_alias'").fetchone()) == (1, "exact")


def test_batch_save_without_served_project_records_no_label(temp_db):
    """Sin proyecto conocido de la key no se inventa etiqueta (ni se asume P0)."""
    temp_db.save_matches_and_timelines_batch([
        ("LA2_TR_002", _mini_match("LA2_TR_002", "puuid_x"), None),
    ])
    with temp_db._get_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM puuid_projects").fetchone()[0] == 0


def test_export_players_groups_by_canonical(temp_db):
    import tempfile as _tf
    temp_db.register_api_projects(["KEY0_ABCDEFGHIJ", "KEY1_KLMNOPQRST"])
    temp_db.upsert_player_identity("Duo", "LA2", "puuid_P0", 0, tier="CHALLENGER")
    temp_db.upsert_player_identity("Duo", "LA2", "puuid_P1", 1, tier="CHALLENGER")
    temp_db.save_players([
        {"puuid": "puuid_P0", "summoner_id": "s0", "tier": "CHALLENGER", "rank": "I",
         "league_points": 800, "wins": 10, "losses": 5},
        {"puuid": "puuid_P1", "summoner_id": "s1", "tier": "CHALLENGER", "rank": "I",
         "league_points": 800, "wins": 10, "losses": 5},
    ])
    temp_db.reload_translation_cache()
    # 2 partidas bajo P0 + 3 bajo P1 (alias) -> 5 unificadas bajo el canónico
    batch = [
        ("LA2_UN_001", _mini_match("LA2_UN_001", "puuid_P0"), None),
        ("LA2_UN_002", _mini_match("LA2_UN_002", "puuid_P0"), None),
        ("LA2_UN_003", _mini_match("LA2_UN_003", "puuid_P1"), None),
        ("LA2_UN_004", _mini_match("LA2_UN_004", "puuid_P1"), None),
        ("LA2_UN_005", _mini_match("LA2_UN_005", "puuid_P1"), None),
    ]
    assert temp_db.save_matches_and_timelines_batch(batch) == 5

    from export_to_csv import export_players
    with _tf.TemporaryDirectory() as tmp_out:
        out = Path(tmp_out) / "players.csv"
        df = export_players(temp_db.db_path, out)
        assert len(df) == 1
        row = df.iloc[0]
        assert row["games_recorded"] == 5
        assert row["player_id"] == "duo#la2"
        assert row["puuid_raw"] == "puuid_P0"   # representante determinista (MIN)
        assert row["num_puuids"] == 2
        # Fix 1 (Antigravity): main_lane + confidence_tier (5 partidas MIDDLE -> LOW)
        assert row["main_lane"] == "MIDDLE"
        assert str(row["confidence_tier"]) == "LOW"
        # Coherencia: SUM(games_recorded) == filas de participantes
        assert int(df["games_recorded"].sum()) == 5


def test_export_players_without_map_is_identity_trivial(temp_db):
    import tempfile as _tf
    temp_db.save_players([
        {"puuid": "solo_puuid", "summoner_id": "s", "tier": "MASTER", "rank": "I",
         "league_points": 100, "wins": 3, "losses": 2},
    ])
    temp_db.save_matches_and_timelines_batch(
        [("LA2_SO_001", _mini_match("LA2_SO_001", "solo_puuid"), None)])
    from export_to_csv import export_players
    with _tf.TemporaryDirectory() as tmp_out:
        df = export_players(temp_db.db_path, Path(tmp_out) / "players.csv")
        assert len(df) == 1
        assert df.iloc[0]["player_id"] == "solo_puuid"
        assert df.iloc[0]["num_puuids"] == 1
        # Fix 1: columnas presentes también en identidad trivial (1 partida -> UNRELIABLE)
        assert df.iloc[0]["main_lane"] == "MIDDLE"
        assert str(df.iloc[0]["confidence_tier"]) == "UNRELIABLE"


def test_roster_quotas_warn_only_never_blocks(temp_db):
    # Padrón vacío: debe reportar avisos, jamás lanzar excepción
    report = temp_db.check_roster_quotas()
    assert report["CHALLENGER"]["expected"] == 200
    assert report["GRANDMASTER"]["expected"] == 500
    assert report["MASTER"]["expected"] == 8259
    assert report["DIAMOND"]["expected"] == 31653
    assert report["CHALLENGER"]["ok"] is False

    # Padrón exacto: OK
    temp_db.save_players([
        {"puuid": f"ch_{i}", "summoner_id": f"s_{i}", "tier": "CHALLENGER",
         "rank": "I", "league_points": 500, "wins": 10, "losses": 5}
        for i in range(200)
    ])
    report = temp_db.check_roster_quotas()
    assert report["CHALLENGER"]["ok"] is True


def test_fingerprint_match_detects_rotation_vs_new_project():
    from scripts.identity.register_key_project import fingerprint_match
    stored = {0: {"a", "b", "c"}, 1: {"x", "y", "z"}}
    # Rotación: mismos PUUIDs del proyecto 0
    best, score = fingerprint_match({"a", "b", "c"}, stored)
    assert best == 0 and score == 1.0
    # Proyecto nuevo: sin intersección
    best, score = fingerprint_match({"m", "n", "o"}, stored)
    assert score == 0.0
    # Vacío: sin crash
    assert fingerprint_match(set(), stored) == (None, 0.0)


def test_canonical_deep_crawl_sums_across_projects(temp_db):
    temp_db.register_api_projects(["KEY0_ABCDEFGHIJ", "KEY1_KLMNOPQRST"])
    temp_db.upsert_player_identity("Deep", "LA2", "puuid_D0", 0, tier="CHALLENGER")
    temp_db.upsert_player_identity("Deep", "LA2", "puuid_D1", 1, tier="CHALLENGER")
    temp_db.save_players([
        {"puuid": "puuid_D0", "summoner_id": "s0", "tier": "CHALLENGER", "rank": "I",
         "league_points": 700, "wins": 1, "losses": 1},
        {"puuid": "puuid_D1", "summoner_id": "s1", "tier": "CHALLENGER", "rank": "I",
         "league_points": 700, "wins": 1, "losses": 1},
    ])
    temp_db.reload_translation_cache()
    temp_db.save_matches_and_timelines_batch([
        ("LA2_DC_001", _mini_match("LA2_DC_001", "puuid_D1"), None),
        ("LA2_DC_002", _mini_match("LA2_DC_002", "puuid_D1"), None),
    ])
    rows = temp_db.get_canonicals_for_deep_crawl(tier="CHALLENGER", min_matches_target=50,
                                                 only_incomplete=True)
    assert len(rows) == 1
    assert rows[0]["canonical_id"] == "deep#la2"
    assert rows[0]["puuid"] == "puuid_D0"  # representante canónico (proyecto 0)
    assert int(rows[0]["existing_recorded_games"]) == 2


def test_canonical_deep_crawl_falls_back_without_map(temp_db):
    temp_db.save_players([
        {"puuid": "legacy_1", "summoner_id": "s", "tier": "DIAMOND", "rank": "I",
         "league_points": 50, "wins": 1, "losses": 1},
    ])
    rows = temp_db.get_canonicals_for_deep_crawl(tier="DIAMOND", min_matches_target=50,
                                                 only_incomplete=True)
    assert len(rows) == 1
    assert rows[0]["puuid"] == "legacy_1"


def test_require_canonical_refuses_legacy_run(temp_db):
    from unittest.mock import MagicMock
    from src.collector import DataCollector
    mock_client = MagicMock()
    mock_client.num_active_keys = 2
    collector = DataCollector(riot_client=mock_client, db_manager=temp_db)
    # Mapa vacío + modo estricto: no recolecta nada (0 requests, 0 partidas)
    result = collector.collect_matches(tier_filter="CHALLENGER", require_canonical=True)
    assert result["total_matches"] == 0
    # Sin modo estricto: pasa el guard (llega a retornar resumen de jugadores vacíos)
    result = collector.collect_matches(tier_filter="CHALLENGER", require_canonical=False)
    assert result["total_matches"] == 0



def test_session_export_never_touches_the_repo_data_dir(temp_db):
    """Una sesión de recolección exporta junto a SU base, no al `data/` del repo.

    Regresión del 2026-09-08: el reporte de cierre llamaba
    `export_all_datasets(self.db.db_path, DATA_DIR)` con el DATA_DIR global, así
    que cualquier test que llegara al cierre de sesión volcaba una BD temporal
    encima de los 4 CSV reales del repositorio y los dejaba con solo la cabecera.
    """
    import threading
    from unittest.mock import MagicMock
    from src.collector import DataCollector
    from src.config import DATA_DIR

    nombres = ("full_dataset.csv", "players.csv", "champions_by_lane.csv", "matchups_1v1.csv")
    antes = {n: (DATA_DIR / n).stat().st_mtime_ns
             for n in nombres if (DATA_DIR / n).exists()}

    temp_db.save_players([
        {"puuid": f"p{i}", "summoner_id": "s", "tier": "DIAMOND", "rank": "I",
         "league_points": 0, "wins": 0, "losses": 0} for i in range(5)
    ])
    mock_client = MagicMock()
    mock_client.num_active_keys = 1
    mock_client.get_telemetry.return_value = {"total_requests": 0, "status_200": 0,
                                              "status_429": 0, "timeouts": 0,
                                              "http_errors": 0}
    collector = DataCollector(riot_client=mock_client, db_manager=temp_db)

    corte = threading.Event()
    corte.set()
    collector.collect_matches(tier_filter="DIAMOND", stop_event=corte)

    for n, mtime in antes.items():
        assert (DATA_DIR / n).stat().st_mtime_ns == mtime, (
            f"la sesión de recolección pisó el CSV real '{n}' del repositorio"
        )

    # Y sí volcó los 4 CSV junto a la BD temporal que realmente recolectó.
    destino = Path(temp_db.db_path).parent
    for n in nombres:
        assert (destino / n).exists()


def _participant(pid, puuid, name):
    return {"participantId": pid, "puuid": puuid, "teamId": 100 if pid <= 5 else 200,
            "win": pid <= 5, "championName": "Ahri", "championId": 103,
            "individualPosition": ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"][(pid - 1) % 5],
            "riotIdGameName": name, "riotIdTagline": "LAS"}


def test_session_refreshes_player_id_map_with_new_puuids(temp_db):
    """Al cerrar una sesión que guardó partidas, `player_id_map` incluye sus PUUIDs nuevos.

    Antes solo lo regeneraba `resolve_inverse_by_match.py`, así que tras un crawl
    `get_player_match_count()` no contaba las partidas de PUUIDs recién vistos.
    """
    from unittest.mock import MagicMock
    from src.collector import DataCollector

    temp_db.save_players([{"puuid": "padron_p0", "summoner_id": "s", "tier": "DIAMOND",
                           "rank": "I", "league_points": 0, "wins": 0, "losses": 0}])
    detail = {"metadata": {"matchId": "LA2_1"},
              "info": {"gameCreation": 1, "gameDuration": 1800, "gameVersion": "16.17",
                       "queueId": 420,
                       "participants": [_participant(1, "padron_p0", "Hero")]
                       + [_participant(i, f"nuevo_{i}", f"Rival{i}") for i in range(2, 11)]}}
    mock_client = MagicMock()
    mock_client.num_active_keys = 1
    mock_client.get_match_ids_by_puuid.side_effect = lambda **kw: ["LA2_1"] if kw.get("start", 0) == 0 else []
    mock_client.get_match_detail.return_value = detail
    # Timeline mínima: el export de cierre necesita los snapshots @10/@15/@20.
    frame = {"events": [], "participantFrames": {
        str(p): {"totalGold": 1000 * p, "xp": 0, "level": 1, "minionsKilled": 0,
                 "jungleMinionsKilled": 0, "damageStats": {}} for p in range(1, 11)}}
    mock_client.get_match_timeline.return_value = {"info": {"frames": [frame] * 21}}
    mock_client.get_telemetry.return_value = {"total_requests": 0, "status_200": 0,
                                              "status_429": 0, "timeouts": 0,
                                              "http_errors": 0}
    collector = DataCollector(riot_client=mock_client, db_manager=temp_db)
    collector.collect_matches(tier_filter="DIAMOND", matches_per_player=1)

    with temp_db._get_connection() as conn:
        mapped = {r[0] for r in conn.execute("SELECT puuid FROM main.player_id_map;")}
    assert {"padron_p0"} | {f"nuevo_{i}" for i in range(2, 11)} <= mapped


def test_la_cola_prioriza_a_los_de_menos_partidas(temp_db):
    """Primero los de menos partidas (sus partidas completan de rebote a los cercanos);
    los nunca visitados antes que los ya visitados; los de 0 partidas al final."""
    temp_db.save_players([{"puuid": p, "summoner_id": "s", "tier": "MASTER", "rank": "I"}
                          for p in ("lejos", "cerca", "visitado_cerca", "cero")])
    batch = []
    for puuid, n in (("lejos", 5), ("cerca", 45), ("visitado_cerca", 48)):
        batch += [(f"LA2_{puuid}_{i}", _mini_match(f"LA2_{puuid}_{i}", puuid), None) for i in range(n)]
    temp_db.save_matches_and_timelines_batch(batch)
    temp_db.mark_player_crawled("visitado_cerca")
    temp_db.materialize_player_id_map()
    orden = [r["players_puuid"] for r in temp_db.get_canonicals_for_deep_crawl(
        tier="MASTER", min_matches_target=50, only_incomplete=True)]
    assert orden == ["lejos", "cerca", "visitado_cerca", "cero"]


def test_el_crawler_refresca_el_mapa_al_arrancar(temp_db):
    """Si la corrida anterior se cortó sin refrescar player_id_map, el conteo arranca bien."""
    from unittest.mock import MagicMock
    from src.collector import DataCollector

    temp_db.save_players([{"puuid": "p", "summoner_id": "s", "tier": "MASTER", "rank": "I"}])
    temp_db.save_matches_and_timelines_batch([("LA2_X", _mini_match("LA2_X", "p"), None)])
    with temp_db._get_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM player_id_map").fetchone()[0] == 0
    mock = MagicMock(); mock.num_active_keys = 1
    mock.get_match_ids_by_puuid.return_value = []
    mock.get_telemetry.return_value = {"total_requests": 0, "status_200": 0, "status_429": 0,
                                       "timeouts": 0, "http_errors": 0}
    DataCollector(riot_client=mock, db_manager=temp_db).collect_matches(tier_filter="MASTER")
    with temp_db._get_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM player_id_map WHERE puuid = 'p'").fetchone()[0] == 1


def test_el_crawler_pagina_hacia_atras_hasta_completar_la_meta(temp_db):
    """Si las ~100 partidas más recientes ya están, sigue hacia atrás (antes se cortaba en 50).

    Además descarta partidas de otros servidores (BR1, NA1...).
    """
    from unittest.mock import MagicMock
    from src.collector import DataCollector

    temp_db.save_players([{"puuid": "p", "summoner_id": "s", "tier": "MASTER", "rank": "I"}])
    recientes = [f"LA2_R{i}" for i in range(100)]
    temp_db.save_matches_and_timelines_batch([(m, _mini_match(m, "p"), None) for m in recientes[:40]])
    temp_db.save_matches_and_timelines_batch([(m, _mini_match(m, "otro"), None) for m in recientes[40:]])
    viejas = ["BR1_X1"] + [f"LA2_V{i}" for i in range(30)]

    def pages(**kw):
        start = kw.get("start", 0)
        return recientes if start == 0 else (viejas if start == 100 else [])

    pedidas = []
    mock = MagicMock(); mock.num_active_keys = 1; mock.platform = "la2"
    mock.get_match_ids_by_puuid.side_effect = pages
    mock.get_match_detail.side_effect = lambda mid: (pedidas.append(mid), None)[1]
    mock.get_telemetry.return_value = {"total_requests": 0, "status_200": 0, "status_429": 0,
                                       "timeouts": 0, "http_errors": 0}
    DataCollector(riot_client=mock, db_manager=temp_db).collect_matches(tier_filter="MASTER")
    starts = [c.kwargs["start"] for c in mock.get_match_ids_by_puuid.call_args_list]
    assert starts[:2] == [0, 100]                          # llegó a la segunda página
    assert "BR1_X1" not in pedidas                         # otro servidor: no se baja
    assert sorted(pedidas) == sorted(f"LA2_V{i}" for i in range(10))  # justo las 10 que faltan


def _client_for_roster(temp_db, n=1):
    temp_db.save_players([{"puuid": f"p{i}", "tier": "MASTER", "rank": "I"} for i in range(n)])
    client = MagicMock()
    client.platform = "la2"
    client.num_active_keys = 1
    client.telemetry = {}
    client.get_telemetry.return_value = {}
    client.get_match_timeline.return_value = None
    return client


def test_latidos_reutilizan_cobertura_y_cuentan_jugadores(temp_db, monkeypatch):
    import src.collector as collector

    client = _client_for_roster(temp_db, n=7)
    known = ["LA2_A", "LA2_B"]
    temp_db.save_matches_and_timelines_batch([(mid, _mini_match(mid, "otro"), None) for mid in known])
    times = SimpleNamespace(now=0.0)
    monkeypatch.setattr(collector, "time",
                        SimpleNamespace(monotonic=lambda: times.now, time=lambda: times.now))

    def list_ids(**kwargs):
        times.now += 60
        return known

    client.get_match_ids_by_puuid.side_effect = list_ids
    coverage_calls = []
    coverage = temp_db.get_coverage_stats

    def coverage_traced():
        coverage_calls.append(times.now)
        return coverage()

    monkeypatch.setattr(temp_db, "get_coverage_stats", coverage_traced)
    monkeypatch.setattr(collector, "export_all_datasets", MagicMock())
    report = collector.DataCollector(client, temp_db).collect_matches(depth_first=False)
    assert coverage_calls == [60.0, 360.0, 420.0]
    assert report["coverage"] == report["database_summary"]["coverage"]
    assert report["matches_discovered"] == 14
    with temp_db._get_connection() as conn:
        rows = conn.execute("SELECT players_crawled, total_matches_in_db FROM crawler_metrics_timeline ORDER BY id")
        assert [tuple(row) for row in rows] == [(n, 2) for n in range(1, 8)]


def test_corte_por_tiempo_cosecha_listas_y_acota_la_gracia(temp_db, monkeypatch):
    import src.collector as collector

    client = _client_for_roster(temp_db)
    client.get_match_ids_by_puuid.return_value = [f"LA2_{i}" for i in range(5)]
    times = SimpleNamespace(now=0.0)
    monkeypatch.setattr(collector, "time",
                        SimpleNamespace(monotonic=lambda: times.now, time=lambda: times.now))
    monkeypatch.setattr(collector, "DOWNLOAD_GRACE_SECONDS", 0.2, raising=False)
    monkeypatch.setattr(collector, "export_all_datasets", MagicMock())
    submitted = []

    def result(mid):
        return mid, _mini_match(mid, "p0"), None, 0

    class Executor:
        def __init__(self, max_workers):
            pass

        def submit(self, worker, client, mid):
            fut = Future()
            if len(submitted) != 2:
                fut.set_running_or_notify_cancel()
            submitted.append((mid, fut))
            if len(submitted) == 4:
                times.now = 2.0
            return fut

        def shutdown(self, *, wait, cancel_futures):
            assert not wait and cancel_futures
            (mid0, done), (mid1, running), (_, pending), _ = submitted
            pending.cancel()
            done.set_result(result(mid0))
            threading.Timer(0.01, lambda: running.set_result(result(mid1))).start()

    monkeypatch.setattr(collector.concurrent.futures, "ThreadPoolExecutor", Executor)
    report = collector.DataCollector(client, temp_db).collect_matches(
        depth_first=False, max_workers=2, max_duration_seconds=1)
    assert len(submitted) == 4
    assert submitted[2][1].cancelled() and not submitted[3][1].done()
    assert report["matches_new"] == 2 and report["matches_discarded"] == 0
    assert temp_db.is_match_saved("LA2_0") and temp_db.is_match_saved("LA2_1")
    assert report["timelines_saved"] == 0


def test_el_cierre_recuerda_la_senal_si_el_pool_se_recupera(temp_db, monkeypatch):
    import src.collector as collector

    client = _client_for_roster(temp_db)
    stop = MagicMock(spec=threading.Event)
    stop.is_set.side_effect = [True, False]
    monkeypatch.setattr(collector, "export_all_datasets", MagicMock())
    report = collector.DataCollector(client, temp_db).collect_matches(depth_first=False, stop_event=stop)
    client.get_match_ids_by_puuid.assert_not_called()
    assert report["matches_new"] == 0


def test_el_mismo_id_se_reserva_mientras_esta_en_vuelo(temp_db, monkeypatch):
    import src.collector as collector

    client = _client_for_roster(temp_db, n=2)
    client.get_match_ids_by_puuid.side_effect = [["LA2_X", "LA2_X"], ["LA2_X"]]
    release = threading.Event()

    def load(mid):
        assert release.wait(3)
        return _mini_match(mid, "p0")

    client.get_match_detail.side_effect = load
    filter_db = temp_db.filter_unprocessed_match_ids
    page_calls = []

    def filter_page(ids):
        available = filter_db(ids)
        page_calls.append(ids)
        if len(page_calls) == 2:
            release.set()
        return available

    monkeypatch.setattr(temp_db, "filter_unprocessed_match_ids", filter_page)
    monkeypatch.setattr(collector, "export_all_datasets", MagicMock())
    report = collector.DataCollector(client, temp_db).collect_matches(
        depth_first=False, matches_per_player=2, max_workers=1)
    client.get_match_detail.assert_called_once_with("LA2_X")
    assert page_calls == [["LA2_X"], []]
    assert report["matches_new"] == 1 and report["matches_duplicate"] == 2


def test_reserva_dura_hasta_commit_y_el_export_espera_al_escritor(temp_db, monkeypatch):
    import src.collector as collector

    client = _client_for_roster(temp_db, n=2)
    writer_started = threading.Event()
    release = threading.Event()
    save = temp_db.save_match_batch_with_counts

    def delayed_save(batch):
        writer_started.set()
        assert release.wait(5)
        return save(batch)

    monkeypatch.setattr(temp_db, "save_match_batch_with_counts", delayed_save)

    class ImmediateExecutor:
        def __init__(self, max_workers):
            pass

        def submit(self, task, *args):
            fut = Future()
            fut.set_result(task(*args))
            return fut

        def shutdown(self, **kwargs):
            pass

    join_timeouts = []

    class WriterThread(threading.Thread):
        # El escritor queda trabado hasta el join del cierre: si el cierre no lo
        # esperara, el export vería 0 partidas. El join lleva tope (§9.S.76), pero
        # el tope alcanza de sobra para un escritor que avanza.
        def join(self, timeout=None):
            join_timeouts.append(timeout)
            release.set()
            super().join(timeout=timeout)

    monkeypatch.setattr(collector.concurrent.futures, "ThreadPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr(collector, "threading",
                        SimpleNamespace(Thread=WriterThread, Lock=threading.Lock, Event=threading.Event))
    ids = [f"LA2_{n}" for n in range(5)]

    def get_ids(**kwargs):
        if client.get_match_ids_by_puuid.call_count == 2:
            assert writer_started.wait(3)
        return ids

    client.get_match_ids_by_puuid.side_effect = get_ids
    client.get_match_detail.side_effect = lambda mid: _mini_match(mid, "p0")
    exported = []
    monkeypatch.setattr(collector, "export_all_datasets",
                        lambda *args: exported.append(temp_db.get_total_matches()))
    try:
        report = collector.DataCollector(client, temp_db).collect_matches(
            depth_first=False, matches_per_player=5, max_workers=5)
    finally:
        release.set()
    assert client.get_match_detail.call_count == 5
    assert report["matches_duplicate"] == 5
    assert report["matches_new"] == 5 and exported == [5]
    assert join_timeouts == [collector.WRITER_JOIN_TIMEOUT_SECONDS]
    assert report["writer_timed_out"] is False


def test_reporte_calcula_timelines_guardados_y_cobertura_de_la_sesion(temp_db, monkeypatch, capsys):
    import src.collector as collector

    client = _client_for_roster(temp_db)
    client.get_match_ids_by_puuid.return_value = ["LA2_CON", "LA2_SIN"]
    client.get_match_detail.side_effect = lambda mid: _mini_match(mid, "p0")
    client.get_match_timeline.side_effect = [{"info": {"frames": [{}]}}, None]
    monkeypatch.setattr(collector, "export_all_datasets", MagicMock())
    report = collector.DataCollector(client, temp_db).collect_matches(depth_first=False, max_workers=1)
    assert report["matches_new"] == 2
    assert report["timelines_saved"] == report["database_summary"]["total_matches_with_timeline"] == 1
    assert "Timelines guardadas: 1 (50.0%)" in capsys.readouterr().out
