"""
Directorio de proyecto por PUUID y su uso en la recolección (PROJECT_SUMMARY.md §9.S.4).

Cada test fija una de las causas de los problemas de keys anteriores:
* la cola mandaba a Riot un PUUID de un proyecto sin key en el pool;
* al jugador no se lo marcaba como visitado (se marcaba con ese PUUID ajeno);
* la ingesta etiquetaba como P0 cualquier PUUID traducido;
* el proyecto de una key salía de su posición en una lista que podía venir filtrada.
"""

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.features.puuid_projects import exact_projects, infer_projects_from_matches
from src.riot_client import RiotAPIClient
from src.config import project_index_for_key


# --------------------------------------------------------------------------- #
# Deducción por partida                                                       #
# --------------------------------------------------------------------------- #
def test_deduce_el_proyecto_mayoritario_de_sus_partidas():
    rows = [("M1", "a"), ("M1", "b"), ("M1", "x"), ("M2", "c"), ("M2", "x")]
    out = infer_projects_from_matches(rows, exact={"a": 1, "b": 1, "c": 1})
    assert out == {"x": 1}


def test_no_deduce_si_la_pureza_no_alcanza():
    rows = [("M1", "a"), ("M1", "x"), ("M2", "b"), ("M2", "x")]
    out = infer_projects_from_matches(rows, exact={"a": 0, "b": 1}, min_purity=0.95)
    assert out == {}


def test_un_puuid_exacto_no_se_vota_a_si_mismo_ni_se_rededuce():
    rows = [("M1", "a"), ("M1", "b")]
    assert infer_projects_from_matches(rows, exact={"a": 0, "b": 1}) == {}


def test_la_etiqueta_0_de_puuid_riot_ids_no_cuenta_como_exacta(temp_db):
    temp_db.record_riot_id_observations([("contaminado", "X", "LAS")], project_index=0)
    temp_db.record_riot_id_observations([("p3_bueno", "Y", "LAS")], project_index=3)
    with temp_db._get_connection() as conn:
        exact = exact_projects(conn)
    assert "contaminado" not in exact
    assert exact["p3_bueno"] == 3


# --------------------------------------------------------------------------- #
# Directorio                                                                  #
# --------------------------------------------------------------------------- #
def _label(db, puuid):
    with db._get_connection() as conn:
        row = conn.execute("SELECT project_index, source FROM puuid_projects WHERE puuid = ?",
                           (puuid,)).fetchone()
    return tuple(row) if row else None


def test_lo_deducido_no_pisa_lo_exacto_y_lo_exacto_si_pisa_lo_deducido(temp_db):
    temp_db.record_puuid_projects([("p", 0)], source="exact")
    temp_db.record_puuid_projects([("p", 1)], source="inferred")
    assert _label(temp_db, "p") == (0, "exact")
    temp_db.record_puuid_projects([("q", 1)], source="inferred")
    temp_db.record_puuid_projects([("q", 0)], source="exact")
    assert _label(temp_db, "q") == (0, "exact")


def test_el_feedback_de_riot_confirma_y_corrige(temp_db):
    temp_db.record_puuid_projects([("bien", 1), ("mal", 1)], source="inferred")
    res = temp_db.apply_project_feedback([("bien", 1, True), ("mal", 1, False)])
    assert res == {"confirmados": 1, "corregidos": 1}
    assert _label(temp_db, "bien") == (1, "exact")
    assert _label(temp_db, "mal") is None


# --------------------------------------------------------------------------- #
# Proyecto de cada key                                                        #
# --------------------------------------------------------------------------- #
def test_el_proyecto_se_busca_por_la_key_no_por_la_posicion():
    env_keys, env_idx = ["KEY_P0_AAAAAAAA", "KEY_P1_BBBBBBBB"], [0, 1]
    # Si la key de P0 se venció y se descartó, la de P1 queda primera en la lista:
    # tiene que seguir siendo P1.
    assert project_index_for_key("KEY_P1_BBBBBBBB", env_keys, env_idx) == 1
    assert project_index_for_key("KEY_AJENA_CCCCCC", env_keys, env_idx) is None


def test_una_key_de_proyecto_desconocido_no_etiqueta_respuestas():
    client = RiotAPIClient(api_keys=["KEY_AJENA_DESCONOCIDA_123"])
    assert client.key_pool[0].project_known is False
    _fake_http(client, {})
    client.get_match_detail("LA2_1")
    assert client.last_served_project() is None


def test_refresh_por_proyecto_no_corre_las_ranuras():
    # El pool arrancó solo con P1 (la de P0 estaba vencida al inicio).
    client = RiotAPIClient(api_keys=["KEY_P1_VIEJA_XXXXXXX"], project_indices=[1])
    client.refresh_pool(["KEY_P0_NUEVA_YYYYYYY", "KEY_P1_NUEVA_ZZZZZZZ"], projects=[0, 1])
    assert [(c.project_index, c.api_key) for c in client.key_pool] == [(1, "KEY_P1_NUEVA_ZZZZZZZ")]


# --------------------------------------------------------------------------- #
# Ruteo de listas de partidas                                                 #
# --------------------------------------------------------------------------- #
class _Resp:
    def __init__(self, status, payload=None):
        self.status_code = status
        self.headers = {}
        self._payload = payload
        self.text = "Bad Request - Exception decrypting" if status == 400 else ""

    def json(self):
        return self._payload


def _fake_http(client, descifra):
    """`descifra` = {project: set(puuids que esa key puede descifrar)}. Registra cada llamada."""
    client.calls = []
    for ctx in client.key_pool:
        def get(url, params=None, timeout=None, _ctx=ctx):
            client.calls.append((_ctx.project_index, url))
            if "/by-puuid/" in url:
                puuid = url.split("/by-puuid/")[1].split("/")[0]
                ok = puuid in descifra.get(_ctx.project_index, set())
                return _Resp(200, ["LA2_1"]) if ok else _Resp(400)
            return _Resp(200, {"metadata": {"matchId": "LA2_1"}, "info": {}})
        ctx.session.get = get


def test_la_lista_va_solo_a_la_key_que_puede_descifrar_el_puuid():
    client = RiotAPIClient(api_keys=["KEY_P0_AAAAAAAA", "KEY_P1_BBBBBBBB"], project_indices=[0, 1])
    _fake_http(client, {0: {"pu_p0"}, 1: {"pu_p1"}})
    for _ in range(6):
        assert client.get_match_ids_by_puuid("pu_p0", count=20, candidates={0: "pu_p0"}) == ["LA2_1"]
    assert {proj for proj, _url in client.calls} == {0}
    assert client.telemetry["puuid_mismatch"] == 0


def test_etiqueta_falsa_se_informa_se_descarta_y_se_usa_la_otra():
    client = RiotAPIClient(api_keys=["KEY_P0_AAAAAAAA", "KEY_P1_BBBBBBBB"], project_indices=[0, 1])
    _fake_http(client, {0: {"pu_p0"}, 1: set()})  # el "P1" deducido era falso
    feedback = []
    client.project_feedback = lambda pu, pj, ok: feedback.append((pu, pj, ok))
    candidates = {0: "pu_p0", 1: "pu_falso"}
    client.project_wait_time = lambda pj: 0.0 if pj == 1 else 1.0  # P0 "ocupado": prueba P1 primero
    assert client.get_match_ids_by_puuid("pu_p0", count=20, candidates=candidates) == ["LA2_1"]
    assert ("pu_falso", 1, False) in feedback and ("pu_p0", 0, True) in feedback
    assert 1 not in candidates  # la página siguiente no lo vuelve a probar


def test_la_ingesta_sabe_que_proyecto_descargo_la_partida():
    # Con las dos keys libres gana la primera del pool: acá, la de P1.
    client = RiotAPIClient(api_keys=["KEY_P1_BBBBBBBB", "KEY_P0_AAAAAAAA"], project_indices=[1, 0])
    _fake_http(client, {})
    client.get_match_detail("LA2_1")
    assert client.last_served_project() == 1


# --------------------------------------------------------------------------- #
# Cola de recolección (caso real: jugadord#tagd)                          #
# --------------------------------------------------------------------------- #
def _padron_p0_aliased_a_p3(db):
    """Jugador del padrón cuyo PUUID de P0 quedó como alias de un PUUID de P3."""
    db.save_players([{"puuid": "kaneki_p0", "summoner_id": "s", "tier": "DIAMOND", "rank": "I",
                      "league_points": 0, "wins": 0, "losses": 0}])
    db.record_puuid_alias("kaneki_p3", "kaneki_p0", "jugadord", "tagd")
    db.record_riot_id_observations([("kaneki_p3", "jugadord", "tagd")], project_index=3)
    db.record_puuid_projects([("kaneki_p3", 3), ("kaneki_p1", 1)], source="exact")
    db.record_riot_id_observations([("kaneki_p1", "jugadord", "tagd")], project_index=1)


@pytest.mark.parametrize("con_player_id_map", [False, True])
def test_la_cola_manda_el_puuid_del_padron_y_los_candidatos_por_proyecto(temp_db, con_player_id_map):
    _padron_p0_aliased_a_p3(temp_db)
    if con_player_id_map:
        temp_db.materialize_player_id_map()
    rows = temp_db.get_canonicals_for_deep_crawl(tier="DIAMOND", min_matches_target=50,
                                                 only_incomplete=True)
    assert len(rows) == 1
    row = rows[0]
    assert row["puuid"] == "kaneki_p0"          # nunca el de P3
    assert row["players_puuid"] == "kaneki_p0"   # el que se marca como visitado
    assert row["candidates"][0] == "kaneki_p0"
    if con_player_id_map:
        # con el mapa por persona aparecen sus PUUIDs de otros proyectos
        assert row["candidates"].get(1) == "kaneki_p1"


def test_la_sesion_marca_al_jugador_y_no_pide_con_el_puuid_ajeno(temp_db):
    from src.collector import DataCollector

    _padron_p0_aliased_a_p3(temp_db)
    mock_client = MagicMock()
    mock_client.num_active_keys = 2
    mock_client.get_match_ids_by_puuid.return_value = []
    mock_client.get_telemetry.return_value = {"total_requests": 0, "status_200": 0,
                                              "status_429": 0, "timeouts": 0, "http_errors": 0}
    collector = DataCollector(riot_client=mock_client, db_manager=temp_db)
    collector.collect_matches(tier_filter="DIAMOND", matches_per_player=10)

    kwargs = mock_client.get_match_ids_by_puuid.call_args.kwargs
    assert kwargs["puuid"] == "kaneki_p0"
    # El de P3 puede viajar como candidato de su proyecto: el cliente descarta los
    # proyectos sin key viva, así que nunca se le manda a P0 ni a P1.
    assert kwargs["candidates"][0] == "kaneki_p0"
    assert kwargs["candidates"].get(3, "kaneki_p3") == "kaneki_p3"
    with temp_db._get_connection() as conn:
        visitado = conn.execute("SELECT last_crawled_at FROM players WHERE puuid = 'kaneki_p0'").fetchone()[0]
    assert visitado is not None


# --------------------------------------------------------------------------- #
# Export                                                                      #
# --------------------------------------------------------------------------- #
def test_full_dataset_da_tier_a_filas_guardadas_bajo_otro_proyecto(temp_db):
    from export_to_csv import export_full_dataset
    from tests.test_identity import _mini_match

    temp_db.save_players([{"puuid": "carry_p0", "summoner_id": "s", "tier": "MASTER", "rank": "I",
                           "league_points": 10, "wins": 0, "losses": 0}])
    temp_db.record_riot_id_observations([("carry_p0", "Carry", "LAS")], project_index=None)
    temp_db.record_riot_id_observations([("carry_p1", "Carry", "LAS")], project_index=1)
    temp_db.save_matches_and_timelines_batch([
        ("LA2_T1", _mini_match("LA2_T1", "carry_p1"), None, 1),
    ])
    with tempfile.TemporaryDirectory() as tmp:
        df = export_full_dataset(temp_db.db_path, Path(tmp) / "full.csv")
    assert df.loc[df.puuid_raw == "carry_p1", "player_tier"].tolist() == ["MASTER"]
