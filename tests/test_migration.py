"""
Pruebas del sistema de migración legacy (vista unificada, alias, dedup, poda).
Todo en BD temporal; sin red ni BD viva.
"""

from pathlib import Path



from scripts.identity.migrate_dead_puuids import attribute_dead_puuid, migrate_player
from scripts.identity.register_key_project import probe_stored_decryptable
from src.database import DatabaseManager
from src.riot_client import RiotAPIClient


def test_attribute_dead_puuid_pure():
    detail = {"info": {"participants": [
        {"participantId": 1, "puuid": "AAA"},
        {"participantId": 3, "puuid": "FRESH"},
    ]}}
    assert attribute_dead_puuid(detail, "FRESH") == 3
    assert attribute_dead_puuid(detail, "ZZZ") is None
    assert attribute_dead_puuid({}, "FRESH") is None
    assert attribute_dead_puuid(detail, "") is None


def test_view_unifies_live_map_and_legacy_aliases(temp_db):
    temp_db.register_api_projects(["K0_ABCDEFGHIJ", "K1_KLMNOPQRST"])
    temp_db.upsert_player_identity("Live", "LA2", "LIVE_P0", 0, tier="CHALLENGER")
    temp_db.upsert_player_identity("Live", "LA2", "LIVE_P1", 1, tier="CHALLENGER")
    temp_db.record_puuid_alias("LIVE_P0", "DEAD_X", "Live", "LA2")
    n = temp_db.reload_translation_cache()
    assert temp_db._translation_cache["LIVE_P1"] == "LIVE_P0"
    assert temp_db._translation_cache["DEAD_X"] == "LIVE_P0"
    assert n == 3  # P1->P0, P0->P0 (identidad), DEAD_X->P0
    # canonical_id uniforme encadenado
    assert temp_db.resolve_canonical_id("DEAD_X") == "live#la2"
    assert temp_db.resolve_canonical_id("LIVE_P1") == "live#la2"
    # has_live_puuid distingue fila live de solo-alias
    assert temp_db.has_live_puuid("LIVE_P1") is True
    assert temp_db.has_live_puuid("DEAD_X") is False
    assert temp_db.has_live_puuid("nadie") is False


def test_record_alias_idempotent(temp_db):
    assert temp_db.record_puuid_alias("P", "A", "G", "T") is True
    assert temp_db.record_puuid_alias("P", "A", "G", "T") is True
    assert temp_db.record_puuid_alias("P", "P") is False
    assert temp_db.record_puuid_alias("", "A") is False
    rows = temp_db.get_player_aliases("P")
    assert len(rows) == 1 and rows[0]["alias_puuid"] == "A"


def _insert_participant(temp_db, match_id, pid, puuid):
    import sqlite3
    conn = sqlite3.connect(temp_db.db_path)
    try:
        conn.execute("INSERT OR IGNORE INTO matches (match_id, game_duration, queue_id)"
                     " VALUES (?, 1800, 420);", (match_id,))
        conn.execute("INSERT INTO match_participants (match_id, puuid, participant_id, team_id)"
                     " VALUES (?, ?, ?, 100);", (match_id, puuid, pid))
        conn.execute("INSERT OR REPLACE INTO match_timeline (match_id, participant_id, puuid)"
                     " VALUES (?, ?, ?);", (match_id, pid, puuid))
        conn.commit()
    finally:
        conn.close()


def test_dedupe_doubled_participants(temp_db):
    temp_db.record_puuid_alias("LIVE", "DEAD1", "G", "T")
    temp_db.record_puuid_alias("LIVE", "DEAD2", "G", "T")
    temp_db.reload_translation_cache()
    _insert_participant(temp_db, "M1", 4, "DEAD1")
    _insert_participant(temp_db, "M1", 4, "DEAD2")
    stats = temp_db.dedupe_doubled_participants()
    assert stats == {"groups": 1, "fixed": 1, "skipped": 0}
    import sqlite3
    conn = sqlite3.connect(temp_db.db_path)
    try:
        rows = conn.execute("SELECT puuid FROM match_participants WHERE match_id='M1'").fetchall()
        assert [r[0] for r in rows] == ["LIVE"]
        tl = conn.execute("SELECT puuid FROM match_timeline WHERE match_id='M1'").fetchone()[0]
        assert tl == "LIVE"
    finally:
        conn.close()


def test_dedupe_skips_conflicting_canonicals(temp_db):
    temp_db.record_puuid_alias("LIVE_A", "DEAD_A", "G", "T")
    temp_db.record_puuid_alias("LIVE_B", "DEAD_B", "G", "T")
    temp_db.reload_translation_cache()
    _insert_participant(temp_db, "M2", 1, "DEAD_A")
    _insert_participant(temp_db, "M2", 1, "DEAD_B")
    stats = temp_db.dedupe_doubled_participants()
    assert stats["fixed"] == 0 and stats["skipped"] == 1


def test_prune_stale_exact_replaces_partial_keeps(temp_db):
    dead = [{"puuid": f"dead_{i}", "summoner_id": f"s{i}", "tier": "CHALLENGER",
             "rank": "I", "league_points": 1, "wins": 1, "losses": 1} for i in range(200)]
    temp_db.save_players(dead)
    fresh = {f"fresh_{i}" for i in range(200)}
    temp_db.save_players([{"puuid": p, "summoner_id": "x", "tier": "CHALLENGER",
                           "rank": "I", "league_points": 1, "wins": 1, "losses": 1}
                          for p in fresh])
    rep = temp_db.prune_stale_tier_players({"CHALLENGER": fresh})
    assert rep["CHALLENGER"]["replaced"] is True
    assert rep["CHALLENGER"]["pruned"] == 200
    assert temp_db.check_roster_quotas()["CHALLENGER"]["actual"] == 200

    # Sync parcial (150 != 200): no se toca nada
    temp_db.save_players([{"puuid": "extra_dead", "summoner_id": "x", "tier": "GRANDMASTER",
                           "rank": "I", "league_points": 1, "wins": 1, "losses": 1}])
    rep = temp_db.prune_stale_tier_players({"GRANDMASTER": {f"g{i}" for i in range(150)}})
    assert rep["GRANDMASTER"]["replaced"] is False
    assert rep["GRANDMASTER"]["pruned"] == 0


def test_canonical_client_pins_configured_project_zero():
    client = RiotAPIClient(api_keys=["K0_ABCDEFGHIJ", "K1_KLMNOPQRST"], project_indices=[0, 1])
    from src.collector import DataCollector
    import tempfile as _tf
    with _tf.TemporaryDirectory() as tmp:
        db = DatabaseManager(Path(tmp) / "x.db")
        pinned = DataCollector(riot_client=client, db_manager=db)._canonical_client(0)
        assert len(pinned.key_pool) == 1
        assert pinned.key_pool[0].api_key == "K0_ABCDEFGHIJ"
        assert pinned.rate_limiter is client.key_pool[0].rate_limiter


class _CanonStub:
    regional_routing = "americas"

    def __init__(self, fresh, match_ids):
        self._fresh = fresh
        self._ids = match_ids

    def get_account_by_puuid(self, puuid):
        return ("Hero", "LA2")

    def get_match_ids_by_puuid(self, **kwargs):
        return list(self._ids)


class _PoolStub:
    def __init__(self, fresh):
        self._fresh = fresh

    def get_match_detail(self, match_id):
        parts = [{"participantId": i, "puuid": f"other_{i}"} for i in range(1, 11)]
        parts[2] = {"participantId": 3, "puuid": self._fresh}
        return {"info": {"participants": parts}}


class _ProbeStub:
    regional_routing = "americas"

    def __init__(self, payload):
        self._payload = payload

    def get_account_by_puuid(self, puuid):
        if not self._payload or not self._payload.get("gameName"):
            return None
        return self._payload.get("gameName"), self._payload.get("tagLine", "")


def _seed_match(temp_db, match_id, puuid_by_pid):
    import sqlite3
    conn = sqlite3.connect(temp_db.db_path)
    try:
        conn.execute("INSERT OR IGNORE INTO matches (match_id, game_duration, queue_id)"
                     " VALUES (?, 1800, 420);", (match_id,))
        for pid, pu in puuid_by_pid.items():
            conn.execute("INSERT INTO match_participants (match_id, puuid, participant_id, team_id)"
                         " VALUES (?, ?, ?, 100);", (match_id, pu, pid))
        conn.commit()
    finally:
        conn.close()


class _InversePoolStub:
    regional_routing = "americas"

    def __init__(self, details):
        self._details = details  # match_id -> detail DTO

    def get_match_detail(self, match_id):
        return self._details.get(match_id)

    def get_account_by_puuid(self, puuid):
        return None


def _dto(parts):
    # parts: [(pid, puuid, game, tag)]
    return {"info": {"participants": [
        {"participantId": pid, "puuid": pu, "riotIdGameName": g, "riotIdTagline": t}
        for pid, pu, g, t in parts]}}


def test_inverse_uncovered_and_greedy_order(temp_db):
    from scripts.identity.resolve_inverse_by_match import (
        build_coverage_index, greedy_match_order, uncovered_dead)
    _seed_match(temp_db, "G1", {1: "D1", 2: "D2", 3: "D3"})
    _seed_match(temp_db, "G2", {1: "D1", 2: "KNOWN"})
    temp_db.record_puuid_alias("LIVE_K", "KNOWN", "G", "T")
    unc = uncovered_dead(temp_db)
    assert unc == {"D1", "D2", "D3"}

    index = build_coverage_index(temp_db, unc)
    assert index == {"G1": {"D1", "D2", "D3"}, "G2": {"D1"}}

    order = list(greedy_match_order(index))
    # G1 primero (3 muertos). G2 solo aportaba D1, ya cubierto -> no se pide.
    assert order == [("G1", 3)]


def test_greedy_reescala_la_ganancia_en_vez_de_recalcular(temp_db):
    """La ganancia de una partida cae cuando otra ya cubrió parte de sus muertos."""
    from scripts.identity.resolve_inverse_by_match import greedy_match_order
    index = {
        "BIG": {"A", "B", "C"},
        "MID": {"C", "D"},      # tras BIG le queda solo D -> ganancia 1
        "SOLO": {"E", "F"},
    }
    order = list(greedy_match_order(index))
    assert order[0] == ("BIG", 3)
    assert ("SOLO", 2) in order
    assert ("MID", 1) in order
    # Cobertura total y sin pedir ninguna partida de más.
    assert order.index(("SOLO", 2)) < order.index(("MID", 1))
    assert sum(g for _m, g in order) == 6


def test_inverse_resolve_match_links_without_map_rows(temp_db):
    from scripts.identity.resolve_inverse_by_match import resolve_match
    _seed_match(temp_db, "M9", {1: "DEAD_1", 2: "DEAD_2"})
    pool = _InversePoolStub({"M9": _dto([(1, "FRESH_1", "Hero", "LA2"),
                                         (2, "FRESH_2", "Hero", "LA2")])})
    st = resolve_match(temp_db, pool, "M9")
    assert st == {"linked": 2, "already": 0, "no_identity": 0, "observed": 2}
    # Los riotId del payload quedan persistidos aunque no hicieran falta para vincular.
    with temp_db._get_connection() as conn:
        rows = conn.execute(
            "SELECT puuid, canonical_id FROM puuid_riot_ids ORDER BY puuid;").fetchall()
    assert [tuple(r) for r in rows] == [
        ("DEAD_1", "hero#la2"), ("DEAD_2", "hero#la2"),
        ("FRESH_1", "hero#la2"), ("FRESH_2", "hero#la2")
    ]
    # Links legacy sí, filas live NO (las completa register por roster)
    assert temp_db.resolve_canonical_id("DEAD_1") == "FRESH_1"  # primario live, sin canonical game#tag aún
    assert temp_db.has_live_puuid("FRESH_1") is False
    # Pero la vista agrupa por game#tag del alias para export/coverage
    import sqlite3 as _sq
    _c = _sq.connect(temp_db.db_path)
    try:
        cid = _c.execute("SELECT canonical_id FROM v_puuid_canonical "
                         "WHERE alias_puuid = 'DEAD_1'").fetchone()[0]
        assert cid == "hero#la2"
    finally:
        _c.close()
    # Segunda pasada: ya vinculados, 0 links nuevos
    st2 = resolve_match(temp_db, pool, "M9")
    assert st2["linked"] == 0 and st2["already"] == 2
    # Cobertura del resto baja a 0
    from scripts.identity.resolve_inverse_by_match import uncovered_dead
    assert uncovered_dead(temp_db) == set()


def test_migrate_player_links_and_then_known(temp_db):
    _insert_participant(temp_db, "MX", 3, "DEAD_HERO")
    entry = {"puuid": "FRESH_HERO", "tier": "CHALLENGER", "rank": "I",
             "league_points": 900, "wins": 5, "losses": 5, "summoner_id": "s"}
    out = migrate_player(temp_db, _CanonStub("FRESH_HERO", ["MX", "OTHER"]),
                         _PoolStub("FRESH_HERO"), entry)
    assert out == "linked"
    assert temp_db.resolve_canonical_id("DEAD_HERO") == "hero#la2"
    assert temp_db.resolve_canonical_id("FRESH_HERO") == "hero#la2"
    # Segunda pasada: ya conocido, sin requests
    out2 = migrate_player(temp_db, _CanonStub("FRESH_HERO", []),
                          _PoolStub("FRESH_HERO"), entry)
    assert out2 == "known"


def test_probe_stored_decryptable(temp_db):
    temp_db.save_players([{"puuid": "p", "summoner_id": "s", "tier": "CHALLENGER",
                           "rank": "I", "league_points": 1, "wins": 1, "losses": 1}])
    assert probe_stored_decryptable(temp_db, _ProbeStub(None)) is False
    assert probe_stored_decryptable(temp_db, _ProbeStub({"gameName": "A"})) is True


class _DoublesCanonStub(_CanonStub):
    pass


def test_only_doubles_resolves_and_dedupes(temp_db):
    from scripts.identity.migrate_dead_puuids import _run_only_doubles
    _insert_participant(temp_db, "MD", 2, "DEAD_A")
    _insert_participant(temp_db, "MD", 2, "DEAD_B")

    class Canon(_CanonStub):
        def __init__(self):
            super().__init__("FRESH_D", [])

    class PoolPid2:
        def get_match_detail(self, match_id):
            parts = [{"participantId": i, "puuid": f"other_{i}"} for i in range(1, 11)]
            parts[1] = {"participantId": 2, "puuid": "FRESH_D"}
            return {"info": {"participants": parts}}

    _run_only_doubles(temp_db, Canon(), PoolPid2(), dry_run=False)
    import sqlite3
    conn = sqlite3.connect(temp_db.db_path)
    try:
        rows = conn.execute("SELECT puuid FROM match_participants WHERE match_id='MD'").fetchall()
        assert [r[0] for r in rows] == ["FRESH_D"]
        assert temp_db.resolve_canonical_id("DEAD_A") == "hero#la2"
        assert temp_db.resolve_canonical_id("DEAD_B") == "hero#la2"
    finally:
        conn.close()
