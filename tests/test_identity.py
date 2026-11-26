"""
Tests de `src/features/identity.py` y del invariante de identidad en los exports.

El punto que cubren: ninguna columna sola es una clave de persona válida. El
PUUID se fragmenta entre proyectos de API; el `name#tag` se fragmenta ante
renombres. `player_id` une las dos capas y debe ser unívoco por humano.
"""

import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.features.identity import (
    UnionFind,
    build_player_id_map,
    materialize_player_id_map,
    materialize_for_database,
)


# --------------------------------------------------------------------------- #
# Union-find puro                                                              #
# --------------------------------------------------------------------------- #

def test_unionfind_transitividad():
    uf = UnionFind()
    uf.union("a", "b")
    uf.union("b", "c")
    uf.add("z")
    assert uf.find("a") == uf.find("c")
    assert uf.find("z") != uf.find("a")
    comps = {frozenset(v) for v in uf.components().values()}
    assert comps == {frozenset({"a", "b", "c"}), frozenset({"z"})}


# --------------------------------------------------------------------------- #
# build_player_id_map sobre una vista simulada                                 #
# --------------------------------------------------------------------------- #

def _conn_with_view(rows):
    """Conexión en memoria con `v_puuid_canonical` simulada."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE v_puuid_canonical "
                 "(alias_puuid TEXT, canonical_puuid TEXT, canonical_id TEXT);")
    conn.executemany("INSERT INTO v_puuid_canonical VALUES (?,?,?);", rows)
    return conn


def test_capa_1_une_alias_con_su_canonico_de_proyecto():
    # El mismo humano bajo dos PUUIDs, sin nombre conocido en ninguno.
    conn = _conn_with_view([("p_alias", "p_canon", None),
                            ("p_canon", "p_canon", None)])
    m = build_player_id_map(conn)
    assert m["p_alias"] == m["p_canon"]
    # Sin nombre disponible, la etiqueta es el PUUID mínimo del componente.
    assert m["p_alias"] == min("p_alias", "p_canon")


def test_capa_2_une_puuids_de_distintos_proyectos_por_nombre():
    # Caso real: 3 proyectos de API, 3 PUUIDs distintos, un solo humano.
    conn = _conn_with_view([("puuid_P0", "puuid_P0", "duo#la2"),
                            ("puuid_P1", "puuid_P1", "duo#la2"),
                            ("puuid_P2", "puuid_P2", "duo#la2")])
    m = build_player_id_map(conn)
    assert len({m["puuid_P0"], m["puuid_P1"], m["puuid_P2"]}) == 1
    assert m["puuid_P0"] == "duo#la2"


def test_renombre_no_parte_al_jugador():
    # Un mismo PUUID visto con dos nombres = renombre. Agrupar por `name#tag`
    # lo partiría en dos personas; el union-find lo mantiene unido.
    conn = _conn_with_view([("puuid_P0", "puuid_P0", "viejo#la2"),
                            ("puuid_P0", "puuid_P0", "nuevo#la2"),
                            ("puuid_P1", "puuid_P1", "nuevo#la2")])
    m = build_player_id_map(conn)
    assert m["puuid_P0"] == m["puuid_P1"]
    assert len(set(m.values())) == 1


def test_puuid_sin_aristas_queda_como_su_propio_componente():
    conn = _conn_with_view([])
    m = build_player_id_map(conn, extra_puuids=["huerfano_1", "huerfano_2"])
    assert m["huerfano_1"] == "huerfano_1"
    assert m["huerfano_2"] == "huerfano_2"


def test_etiqueta_es_determinista_y_prefiere_el_nombre():
    rows = [("pA", "pA", "zzz#la2"), ("pB", "pB", "aaa#la2"), ("pB", "pA", None)]
    primero = build_player_id_map(_conn_with_view(rows))
    segundo = build_player_id_map(_conn_with_view(list(reversed(rows))))
    assert primero == segundo
    # Prefiere un `name#tag` sobre un PUUID crudo, y entre nombres el menor.
    assert primero["pA"] == "aaa#la2"


def test_sin_vista_la_identidad_es_trivial_1a1():
    """Bases legacy sin `v_puuid_canonical` no deben romper el export."""
    conn = sqlite3.connect(":memory:")
    m = build_player_id_map(conn, extra_puuids=["solo"])
    assert m == {"solo": "solo"}


# --------------------------------------------------------------------------- #
# Invariante de los CSV exportados                                             #
# --------------------------------------------------------------------------- #

def _mini_match(match_id, puuid, champion="Ahri"):
    return {
        "metadata": {"matchId": match_id},
        "info": {
            "gameCreation": 1690000000000, "gameDuration": 1800,
            "gameVersion": "16.17.1", "queueId": 420,
            "participants": [{
                "puuid": puuid, "participantId": 1, "teamId": 100, "win": True,
                "championId": 103, "championName": champion,
                "individualPosition": "MIDDLE", "kills": 5, "deaths": 2, "assists": 6
            }]
        }
    }


def test_exports_no_exponen_columnas_de_identidad_ambiguas(temp_db):
    """
    Guardarraíl del footgun: agrupar por PUUID fragmenta al jugador. Los CSV no
    deben ofrecer una columna llamada `puuid` (que invita a agruparla) ni
    `canonical_id` (que se rompe ante renombres). Solo `player_id` es identidad.
    """
    from export_to_csv import export_full_dataset, export_players

    temp_db.save_players([
        {"puuid": "solo_puuid", "summoner_id": "s", "tier": "MASTER", "rank": "I",
         "league_points": 100, "wins": 3, "losses": 2},
    ])
    temp_db.save_matches_and_timelines_batch(
        [("LA2_ID_001", _mini_match("LA2_ID_001", "solo_puuid"), None)])

    with tempfile.TemporaryDirectory() as tmp:
        full = export_full_dataset(temp_db.db_path, Path(tmp) / "full_dataset.csv")
        players = export_players(temp_db.db_path, Path(tmp) / "players.csv")

    for df in (full, players):
        assert "player_id" in df.columns
        assert "puuid" not in df.columns
        assert "canonical_id" not in df.columns

    assert "puuid_raw" in full.columns
    # `players.csv` es un padrón: una fila por humano, sin excepción.
    assert players["player_id"].is_unique


def test_players_csv_unifica_puuids_de_varios_proyectos(temp_db):
    """Dos PUUIDs del mismo humano (2 proyectos) colapsan en una sola fila."""
    from export_to_csv import export_players

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
    temp_db.save_matches_and_timelines_batch([
        ("LA2_ID_010", _mini_match("LA2_ID_010", "puuid_P0"), None),
        ("LA2_ID_011", _mini_match("LA2_ID_011", "puuid_P1"), None),
    ])

    with tempfile.TemporaryDirectory() as tmp:
        df = export_players(temp_db.db_path, Path(tmp) / "players.csv")

    assert len(df) == 1
    assert df["player_id"].is_unique
    assert df.iloc[0]["player_id"] == "duo#la2"
    assert df.iloc[0]["num_puuids"] == 2
    assert int(df["games_recorded"].sum()) == 2


# --------------------------------------------------------------------------- #
# Ranura de proyecto explícita y observaciones de Riot ID                      #
# --------------------------------------------------------------------------- #

def test_ranura_de_proyecto_explicita_no_pisa_la_ajena(temp_db):
    """
    Con un pool que es un subconjunto de los proyectos históricos, registrar por
    posición sobrescribiría el `key_hint` de la ranura 0 ajena. La ranura
    explícita es la que evita esa pérdida.
    """
    temp_db.register_api_projects(["KEY_DEL_COMPANIERO_AAAA"])          # ocupa P0
    temp_db.register_api_projects(["KEY_PROPIA_NUEVA_BBBB"],
                                  project_indices=[3])                   # va a P3
    with temp_db._get_connection() as conn:
        rows = dict(conn.execute(
            "SELECT project_index, key_hint FROM api_projects;").fetchall())
    assert set(rows) == {0, 3}
    assert rows[0] == temp_db.mask_key_hint("KEY_DEL_COMPANIERO_AAAA")
    assert rows[3] == temp_db.mask_key_hint("KEY_PROPIA_NUEVA_BBBB")


def test_ranuras_explicitas_deben_coincidir_con_el_pool(temp_db):
    with pytest.raises(ValueError):
        temp_db.register_api_projects(["K1", "K2"], project_indices=[3])


def test_observaciones_de_riot_id_acumulan_y_abren_ventana(temp_db):
    temp_db.record_riot_id_observations([("P", "Hero", "LA2")],
                                        project_index=3, game_creation=1000)
    temp_db.record_riot_id_observations([("P", "Hero", "LA2")],
                                        project_index=3, game_creation=3000)
    with temp_db._get_connection() as conn:
        row = conn.execute(
            "SELECT observations, first_seen_game_creation, last_seen_game_creation "
            "FROM puuid_riot_ids WHERE puuid='P';").fetchone()
    assert tuple(row) == (2, 1000, 3000)


def test_renombre_queda_como_dos_filas_fechadas(temp_db):
    """
    El mismo PUUID con dos nombres es un renombre. Ambas filas se conservan con
    su ventana temporal: es la información que hoy no existe en ningún lado y la
    que permite ordenar cuál nombre es el vigente.
    """
    temp_db.record_riot_id_observations([("P", "Viejo", "LA2")], game_creation=1000)
    temp_db.record_riot_id_observations([("P", "Nuevo", "LA2")], game_creation=9000)
    with temp_db._get_connection() as conn:
        rows = conn.execute(
            "SELECT canonical_id, last_seen_game_creation FROM puuid_riot_ids "
            "WHERE puuid='P' ORDER BY last_seen_game_creation;").fetchall()
    assert [tuple(r) for r in rows] == [("viejo#la2", 1000), ("nuevo#la2", 9000)]


def test_observaciones_ignoran_filas_sin_nombre(temp_db):
    assert temp_db.record_riot_id_observations([("P", "", "LA2"), ("", "X", "LA2")]) == 0


def test_materialize_player_id_map_en_base_de_datos(temp_db):
    """Verifica que la tabla permanente player_id_map se cree y persista en SQLite."""
    temp_db.save_players([
        {"puuid": "p1", "summoner_id": "s1", "tier": "DIAMOND", "rank": "I", "league_points": 10},
        {"puuid": "p2", "summoner_id": "s2", "tier": "DIAMOND", "rank": "I", "league_points": 20},
    ])
    count = temp_db.materialize_player_id_map()
    assert count >= 2
    with temp_db._get_connection() as conn:
        rows = dict(conn.execute("SELECT puuid, player_id FROM player_id_map;").fetchall())
    assert "p1" in rows
    assert "p2" in rows


def test_capa_3_lee_puuid_riot_ids_y_une_entre_proyectos(temp_db):
    """`build_player_id_map` consume `puuid_riot_ids` (Capa 3) para unir PUUIDs entre proyectos."""
    temp_db.register_api_projects(["KEY0_ABCDEFGHIJ", "KEY3_QRSTUVWXYZ"], project_indices=[0, 3])
    temp_db.upsert_player_identity("Faker", "LAS", "puuid_P0", project_index=0, tier="CHALLENGER")
    temp_db.record_riot_id_observations([("puuid_P3", "Faker", "LAS")], project_index=3, game_creation=5000)

    with temp_db._get_connection() as conn:
        m = build_player_id_map(conn)
    assert m["puuid_P0"] == "faker#las"
    assert m["puuid_P3"] == "faker#las"


def test_invariante_mismo_proyecto_bloquea_reuso_de_nombre(temp_db):
    """Dos PUUIDs distintos del mismo proyecto que comparten nombre (reuso) NO se fusionan."""
    temp_db.save_players([
        {"puuid": "pix_cuenta_A_P0", "summoner_id": "sA", "tier": "DIAMOND", "rank": "II", "league_points": 75},
        {"puuid": "pix_cuenta_B_P0", "summoner_id": "sB", "tier": "DIAMOND", "rank": "IV", "league_points": 70},
    ])
    # Ambos tuvieron el mismo Riot ID en distintos momentos (ej. jugador a#taga)
    temp_db.record_puuid_alias("dead_A", "pix_cuenta_A_P0", "jugador a", "taga")
    temp_db.record_puuid_alias("dead_B", "pix_cuenta_B_P0", "jugador a", "taga")

    with temp_db._get_connection() as conn:
        m = build_player_id_map(conn)

    # Ambos componentes quedan separados y con player_id distintos (biyección componente <-> player_id)
    assert m["pix_cuenta_A_P0"] != m["pix_cuenta_B_P0"]
    assert m["pix_cuenta_A_P0"].startswith("jugador a#taga")
    assert m["pix_cuenta_B_P0"].startswith("jugador a#taga")


def test_export_players_incluye_partidas_unidas_por_capa_2_y_3(temp_db):
    """Un jugador del padrón (P0) con partidas bajo P3 unidas por nombre conserva todas sus partidas."""
    from export_to_csv import export_players

    temp_db.register_api_projects(["KEY0_ABCDEFGHIJ", "KEY3_QRSTUVWXYZ"], project_indices=[0, 3])
    temp_db.save_players([
        {"puuid": "padron_P0", "summoner_id": "s0", "tier": "MASTER", "rank": "I",
         "league_points": 250, "wins": 20, "losses": 15},
    ])
    temp_db.upsert_player_identity("Carry", "LAS", "padron_P0", project_index=0, tier="MASTER")
    # PUUID de proyecto 3 unido únicamente vía puuid_riot_ids (Capa 3), sin fila en player_puuid_map para P3
    temp_db.record_riot_id_observations([("partida_P3", "Carry", "LAS")], project_index=3, game_creation=2000)

    temp_db.save_matches_and_timelines_batch([
        ("LA2_C3_001", _mini_match("LA2_C3_001", "padron_P0"), None),
        ("LA2_C3_002", _mini_match("LA2_C3_002", "partida_P3"), None),
        ("LA2_C3_003", _mini_match("LA2_C3_003", "partida_P3"), None),
    ])

    with tempfile.TemporaryDirectory() as tmp:
        df = export_players(temp_db.db_path, Path(tmp) / "players.csv")

    assert len(df) == 1
    assert df.iloc[0]["player_id"] == "carry#las"
    assert int(df.iloc[0]["games_recorded"]) == 3


def test_ingesta_persiste_riotid_en_puuid_riot_ids(temp_db):
    """Guardar una partida extrae automáticamente riotIdGameName y riotIdTagline hacia puuid_riot_ids."""
    match_dto = {
        "metadata": {"matchId": "LA2_RIOTID_001"},
        "info": {
            "gameCreation": 1700000000000,
            "gameDuration": 1800,
            "gameVersion": "16.17.1",
            "queueId": 420,
            "participants": [{
                "puuid": "puuid_obs_1",
                "riotIdGameName": "ShowMaker",
                "riotIdTagline": "KR1",
                "participantId": 1,
                "teamId": 100,
                "win": True,
                "championId": 103,
                "championName": "Ahri",
                "individualPosition": "MIDDLE",
                "kills": 8, "deaths": 1, "assists": 7,
            }],
        },
    }
    assert temp_db.save_matches_and_timelines_batch([("LA2_RIOTID_001", match_dto, None)]) == 1
    with temp_db._get_connection() as conn:
        row = conn.execute(
            "SELECT puuid, canonical_id, game_name, tag_line, observations, "
            "first_seen_game_creation, last_seen_game_creation "
            "FROM puuid_riot_ids WHERE puuid = 'puuid_obs_1';"
        ).fetchone()
    assert tuple(row) == ("puuid_obs_1", "showmaker#kr1", "ShowMaker", "KR1", 1, 1700000000000, 1700000000000)


def test_canonical_client_respeta_project_indices_y_rechaza_sin_p0(temp_db):
    """`_canonical_client` busca la ranura 0 en `project_indices` y rechaza sincronizar si no hay P0."""
    from src.collector import DataCollector
    from src.riot_client import RiotAPIClient

    # Caso 1: P0 está en la posición 1 del pool (project_indices=[1, 0])
    client_reordered = RiotAPIClient(api_keys=["KEY_P1_AAAA", "KEY_P0_BBBB"])
    collector_ok = DataCollector(riot_client=client_reordered, db_manager=temp_db, project_indices=[1, 0])
    pinned = collector_ok._canonical_client(0)
    assert pinned is not None
    assert len(pinned.key_pool) == 1
    assert pinned.key_pool[0].api_key == "KEY_P0_BBBB"

    # Caso 2: Pool solo tiene P3 (project_indices=[3]) -> debe rechazar sincronizar para no borrar P0
    client_p3 = RiotAPIClient(api_keys=["KEY_P3_CCCC"])
    collector_p3 = DataCollector(riot_client=client_p3, db_manager=temp_db, project_indices=[3])
    assert collector_p3._canonical_client(0) is None
    assert collector_p3.sync_players() == 0




# --------------------------------------------------------------------------- #
# Auditoría de duplicados (scripts/ops/audit_duplicates.py)                   #
# --------------------------------------------------------------------------- #
def _match10(match_id, puuids):
    lanes = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
    return {"metadata": {"matchId": match_id},
            "info": {"gameCreation": 1, "gameDuration": 1800, "gameVersion": "16.20", "queueId": 420,
                     "participants": [{"participantId": i + 1, "puuid": p, "teamId": 100 if i < 5 else 200,
                                       "win": i < 5, "championName": "Ahri", "championId": 103,
                                       "individualPosition": lanes[i % 5],
                                       "riotIdGameName": f"n_{p}", "riotIdTagline": "LAS"}
                                      for i, p in enumerate(puuids)]}}


def _audit(db):
    from src.features.identity import attach_for_participants, audit_duplicates, suspicious_shared_names
    with db._get_connection() as conn:
        attach_for_participants(conn)
        return audit_duplicates(conn), suspicious_shared_names(conn)


def test_auditoria_limpia_da_todo_cero(temp_db):
    temp_db.save_matches_and_timelines_batch([("LA2_A1", _match10("LA2_A1", [f"p{i}" for i in range(10)]), None, 0)])
    res, sosp = _audit(temp_db)
    assert all(res[k] == 0 for k in ("padron_duplicado", "misma_persona_dos_veces_en_partida",
                                      "partidas_sin_10_personas", "personas_sin_nombre_con_partidas"))
    assert sosp == []


def test_auditoria_detecta_fusion_falsa(temp_db):
    # p0 y p1 comparten nombre (sin prueba de proyecto) y juegan la misma partida.
    temp_db.save_matches_and_timelines_batch([("LA2_A1", _match10("LA2_A1", [f"p{i}" for i in range(10)]), None)])
    temp_db.record_riot_id_observations([("p0", "Mismo", "LAS"), ("p1", "Mismo", "LAS")])
    res, _ = _audit(temp_db)
    assert res["misma_persona_dos_veces_en_partida"] == 1
    assert res["partidas_sin_10_personas"] == 1


def test_auditoria_detecta_padron_duplicado(temp_db):
    temp_db.save_players([{"puuid": "a", "summoner_id": "s", "tier": "MASTER", "rank": "I"},
                          {"puuid": "b", "summoner_id": "s", "tier": "MASTER", "rank": "I"}])
    temp_db.record_puuid_alias("a", "b", "Uno", "LAS")  # el padrón tiene a la misma persona dos veces
    res, _ = _audit(temp_db)
    assert res["padron_duplicado"] == 1


def test_auditoria_distingue_nombre_reusado_de_persona_partida(temp_db):
    temp_db.save_players([{"puuid": "cuenta_A", "summoner_id": "s", "tier": "MASTER", "rank": "I"},
                          {"puuid": "cuenta_B", "summoner_id": "s", "tier": "MASTER", "rank": "I"}])
    # Las dos son P0 (padrón) y usaron el mismo nombre: dos cuentas, separadas con prueba.
    temp_db.record_riot_id_observations([("cuenta_A", "Reusado", "LAS")], game_creation=1)
    temp_db.record_riot_id_observations([("cuenta_B", "Reusado", "LAS")], game_creation=2)
    res, sosp = _audit(temp_db)
    assert res["nombres_compartidos"] == 1 and sosp == []


# --------------------------------------------------------------------------- #
# Corrección de PUUIDs mal asignados en el mapa (fix_map_contradictions.py)   #
# --------------------------------------------------------------------------- #
def test_contradiccion_mapa_vs_alias_se_detecta_y_se_corrige(temp_db):
    from scripts.identity.fix_map_contradictions import apply_fix, classify, find_contradictions
    from src.features.identity import attach_for_participants

    # JugadorG (padrón, P0) y JugadorH (otra persona). El mapa de P2 le asignó a JugadorH
    # el PUUID de P2 de JugadorG, aunque la resolución por partida dice que es de JugadorG.
    temp_db.save_players([{"puuid": "kiyo_p0", "summoner_id": "s", "tier": "CHALLENGER", "rank": "I"},
                          {"puuid": "tono_p0", "summoner_id": "s", "tier": "MASTER", "rank": "I"}])
    temp_db.upsert_player_identity("JugadorG", "tagg", "kiyo_p0", project_index=0)
    temp_db.upsert_player_identity("jugadorh", "tagh", "tono_p0", project_index=0)
    temp_db.upsert_player_identity("jugadorh", "tagh", "kiyo_p2", project_index=2)   # fila errónea
    temp_db.record_puuid_alias("kiyo_fresco", "kiyo_p2", "JugadorG", "tagg")
    temp_db.save_matches_and_timelines_batch([("LA2_K1", _mini_match("LA2_K1", "kiyo_p2"), None)])

    with temp_db._get_connection() as conn:
        assert find_contradictions(conn) == [("kiyo_p2", "jugadorh#tagh", 2)]
        attach_for_participants(conn)
        person = dict(conn.execute("SELECT puuid, player_id FROM temp.player_id_map"))
    assert person["kiyo_p2"] == "jugadorh#tagh"                       # el error
    assert classify(person["kiyo_p2"], person["kiyo_p0"]) == "mal_asignado"

    apply_fix(temp_db, "kiyo_p2", "jugadorh#tagh", 2, "kiyo_p0", "JugadorG#tagg")
    with temp_db._get_connection() as conn:
        assert find_contradictions(conn) == []
        attach_for_participants(conn)
        person = dict(conn.execute("SELECT puuid, player_id FROM temp.player_id_map"))
    assert person["kiyo_p2"] == person["kiyo_p0"] == "jugadorg#tagg"
    assert person["tono_p0"] == "jugadorh#tagh"


def test_un_renombre_no_se_toca():
    from scripts.identity.fix_map_contradictions import classify
    assert classify("vladzukill#cr7", "vladzukill#cr7") == "renombre"
    assert classify("x#1", None) == "sin_dueno"


def test_puuid_que_representa_a_dos_personas_se_corrige_fila_por_fila(temp_db):
    from scripts.identity.fix_map_contradictions import fix_rows_for_puuid
    from src.features.identity import attach_for_participants

    temp_db.save_players([{"puuid": "kiyo_p0", "summoner_id": "s", "tier": "CHALLENGER", "rank": "I"},
                          {"puuid": "tono_p0", "summoner_id": "s", "tier": "MASTER", "rank": "I"}])
    temp_db.upsert_player_identity("JugadorG", "tagg", "kiyo_p0", project_index=0)
    temp_db.upsert_player_identity("jugadorh", "tagh", "tono_p0", project_index=0)
    temp_db.upsert_player_identity("jugadorh", "tagh", "mixto", project_index=2)
    # "mixto" quedó en una partida de JugadorH (legítima) y en una de JugadorG (reescrita).
    temp_db.save_matches_and_timelines_batch([("LA2_T", _mini_match("LA2_T", "mixto"), None),
                                              ("LA2_K", _mini_match("LA2_K", "mixto"), None)])

    class Riot:
        def get_match_detail(self, mid):
            real = "tono_p0" if mid == "LA2_T" else "kiyo_p0"
            name = ("jugadorh", "tagh") if mid == "LA2_T" else ("JugadorG", "tagg")
            return {"info": {"participants": [{"participantId": 1, "puuid": real,
                                               "riotIdGameName": name[0], "riotIdTagline": name[1]}]}}

    st = fix_rows_for_puuid(temp_db, Riot(), "mixto")
    assert (st["filas"], st["corregidas"]) == (2, 1)
    assert temp_db.get_participant_puuid_map("LA2_K") == {1: "kiyo_p0"}   # corregida
    assert temp_db.get_participant_puuid_map("LA2_T") == {1: "mixto"}     # la legítima no se toca


def test_contradiccion_por_el_lado_primario_tambien_se_detecta(temp_db):
    """Caso real jugadorg#tagg: el PUUID del mapa figura como primario (fresco) del alias."""
    from scripts.identity.fix_map_contradictions import find_contradictions
    temp_db.upsert_player_identity("jugadorh", "tagh", "kiyo_p2", project_index=2)
    temp_db.record_puuid_alias("kiyo_p2", "kiyo_viejo", "JugadorG", "tagg")   # kiyo_p2 es el primario
    with temp_db._get_connection() as conn:
        assert find_contradictions(conn, "alias") == []
        assert find_contradictions(conn, "primario") == [("kiyo_p2", "jugadorh#tagh", 2)]
        assert find_contradictions(conn) == [("kiyo_p2", "jugadorh#tagh", 2)]
