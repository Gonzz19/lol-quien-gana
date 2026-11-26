"""Lo previo a la partida que se extrae de la respuesta cruda de match-v5 (bans y rol pedido)."""
import gzip
import json

from scripts.collection.partidas_crudas import conectar, previo_de_partida


def _detalle():
    def jug(pid, equipo, pos, prefs):
        return {"participantId": pid, "teamId": equipo, "championId": 100 + pid, "teamPosition": pos,
                "positionAssignedByMatchmaking": pos, "selectedRolePreferences": prefs}
    return {"metadata": {"matchId": "LA2_1"},
            "info": {"teams": [{"teamId": 100, "bans": [{"championId": 30, "pickTurn": 2}, {"championId": -1, "pickTurn": 1},
                                                         {"championId": 25, "pickTurn": 3}]},
                               {"teamId": 200, "bans": []}],
                     "participants": [jug(1, 100, "TOP", "TOP.PRIMARY.TOP.JUNGLE"),
                                      jug(2, 100, "JUNGLE", "JUNGLE.AUTOFILL.BOTTOM.TOP.FILL"),
                                      jug(6, 200, "TOP", None)]}}


def test_bans_en_orden_de_turno_y_sin_los_vacios():
    previo = previo_de_partida(_detalle())
    assert previo["bans"] == {100: [30, 25], 200: []}


def test_rol_pedido_y_autofill():
    j1, j2, j6 = previo_de_partida(_detalle())["jugadores"]
    assert (j1["tipo_rol"], j1["autofill"], j1["pedidos"], j1["acepta_fill"]) == ("PRIMARY", False, ["TOP", "JUNGLE"], False)
    assert (j2["tipo_rol"], j2["autofill"], j2["pedidos"], j2["acepta_fill"]) == ("AUTOFILL", True, ["BOTTOM", "TOP"], True)
    assert j6["autofill"] is None and j6["pedidos"] == []      # sin el campo no se inventa un valor


def test_la_base_guarda_el_json_entero_comprimido(tmp_path):
    det = _detalle()
    with conectar(tmp_path / "crudas.db") as conn:
        conn.execute("INSERT INTO partidas VALUES (?, ?, ?, ?, ?, ?)",
                     ("LA2_1", "ok", 1, "16.18", 2, gzip.compress(json.dumps(det).encode("utf-8"))))
        conn.commit()
        blob = conn.execute("SELECT json_gz FROM partidas WHERE match_id = 'LA2_1'").fetchone()[0]
    assert json.loads(gzip.decompress(blob)) == det
