"""Fotos fechadas del rango: tabla `ladder_snapshots` y consulta por PUUID (§9.S.21)."""

import pytest

from src.riot_client import RiotAPIClient


def entrada(puuid, tier="MASTER", lp=100):
    return {"puuid": puuid, "tier": tier, "rank": "I", "league_points": lp, "wins": 10, "losses": 5}


def test_fotos_separadas_y_sin_tocar_players(temp_db):
    temp_db.save_ladder_snapshot("2026-09-04 22:51:22", [entrada("p1", lp=100)], 0, "ladder")
    temp_db.save_ladder_snapshot("2026-10-27 18:00:00", [entrada("p1", lp=300)], 0, "ladder")
    with temp_db._get_connection() as conn:
        filas = [tuple(r) for r in conn.execute("SELECT snapshot_at, league_points FROM ladder_snapshots ORDER BY 1")]
        players = conn.execute("SELECT COUNT(*) FROM players").fetchone()[0]
    assert filas == [("2026-09-04 22:51:22", 100), ("2026-10-27 18:00:00", 300)]
    assert players == 0


def test_refoto_idempotente_y_puuids_para_retomar(temp_db):
    s = "2026-10-27 18:00:00"
    temp_db.save_ladder_snapshot(s, [entrada("p1"), entrada("p2")], 0, "ladder")
    temp_db.save_ladder_snapshot(s, [entrada("p2", tier="UNRANKED", lp=0)], 1, "by_puuid")
    assert temp_db.ladder_snapshot_puuids(s) == {"p1", "p2"}
    with temp_db._get_connection() as conn:
        fila = conn.execute("SELECT tier, project_index, fuente FROM ladder_snapshots WHERE puuid='p2'").fetchone()
    assert tuple(fila) == ("UNRANKED", 1, "by_puuid")


def test_fuente_invalida(temp_db):
    with pytest.raises(ValueError):
        temp_db.save_ladder_snapshot("x", [entrada("p1")], 0, "otra")


def cliente_simulado(respuestas):
    """Cliente con `_request` reemplazado: `respuestas[project]` es lo que devuelve Riot."""
    c = RiotAPIClient(api_keys=["RGAPI-TEST-P0", "RGAPI-TEST-P1"], project_indices=[0, 1])
    llamadas = []

    def fake_request(url, params=None, max_retries=6, only_projects=None):
        (project,) = only_projects
        llamadas.append((project, url.rsplit("/", 1)[-1]))
        r = respuestas[project]
        c._tls.decrypt_failed = r == "decrypt"
        return None if r == "decrypt" else r
    c._request = fake_request
    return c, llamadas


def test_entry_soloq_y_unranked():
    c, _ = cliente_simulado({0: [{"queueType": "RANKED_FLEX_SR", "tier": "GOLD"},
                                 {"queueType": "RANKED_SOLO_5x5", "tier": "EMERALD", "rank": "II",
                                  "leaguePoints": 40, "wins": 30, "losses": 25}]})
    r = c.get_soloq_entry_by_puuid({0: "pp0"})
    assert (r["tier"], r["rank"], r["league_points"], r["project_index"]) == ("EMERALD", "II", 40, 0)

    c, _ = cliente_simulado({0: []})  # sin ranked solo: lista vacía, no falla
    assert c.get_soloq_entry_by_puuid({0: "pp0"})["tier"] == "UNRANKED"


def test_si_un_proyecto_no_descifra_prueba_el_otro():
    c, llamadas = cliente_simulado({0: "decrypt", 1: [{"queueType": "RANKED_SOLO_5x5", "tier": "MASTER",
                                                       "rank": "I", "leaguePoints": 5}]})
    avisos = []
    c.project_feedback = lambda puuid, project, ok: avisos.append((puuid, project, ok))
    r = c.get_soloq_entry_by_puuid({0: "pp0", 1: "pp1"})
    assert r["tier"] == "MASTER" and r["puuid"] == "pp1" and r["project_index"] == 1
    assert ("pp0", 0, False) in avisos and ("pp1", 1, True) in avisos
    assert {p for p, _ in llamadas} == {0, 1}


def test_sin_respuesta_devuelve_none():
    c, _ = cliente_simulado({0: "decrypt"})
    assert c.get_soloq_entry_by_puuid({0: "pp0"}) is None
