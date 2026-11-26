"""
Recolección por ventana de tiempo (src/window_crawl.py, run.py --ventana-desde).

Lo que fijan estos tests:
* todo jugador de los tiers elegidos se lista, aunque ya tenga 50+ partidas
  (el motivo de este modo: el crawler por meta salteaba a los maduros);
* Diamond queda afuera por defecto;
* las fechas viajan a Riot en segundos epoch, hora de Argentina;
* listar y descargar se pueden cortar y retomar sin repetir trabajo ni perder
  jugadores cuya lista falló;
* una partida que no baja queda pendiente, y lo descargado anota el proyecto
  exacto de sus PUUIDs.
"""

import argparse
import threading
from datetime import datetime, timezone

import pytest

from src.window_crawl import WindowCrawler, parse_window_date, window_key
from tests.test_identity import _mini_match

LANES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
D1 = parse_window_date("2026-10-01")
D2 = parse_window_date("2026-10-15")


class FakeClient:
    """Cliente sin red con la misma interfaz que usa WindowCrawler."""

    def __init__(self, ids_by_puuid, fail_lists=(), fail_details=(), project=1):
        self.ids_by_puuid = ids_by_puuid
        self.fail_lists = set(fail_lists)
        self.fail_details = set(fail_details)
        self.project = project
        self.num_active_keys = 2
        self.project_feedback = None
        self.list_calls, self.detail_calls = [], []
        self._tls = threading.local()

    def get_match_ids_by_puuid(self, puuid, count=100, start=0, start_time=None,
                               candidates=None, end_time=None, **kw):
        self.list_calls.append({"puuid": puuid, "start_time": start_time,
                                "end_time": end_time, "candidates": candidates})
        if puuid in self.fail_lists:
            self._tls.ok = False
            return []
        self._tls.ok = True
        if self.project_feedback:
            self.project_feedback(puuid, 0, True)
        return list(self.ids_by_puuid.get(puuid, []))

    def last_request_ok(self):
        return getattr(self._tls, "ok", False)

    def last_served_project(self):
        return self.project

    def get_match_detail(self, match_id):
        self.detail_calls.append(match_id)
        if match_id in self.fail_details:
            return None
        return {"metadata": {"matchId": match_id},
                "info": {"gameCreation": D1 * 1000, "gameDuration": 1800, "gameVersion": "16.20",
                         "queueId": 420, "participants": [
                             {"participantId": i, "puuid": f"{match_id}_p{i}", "teamId": 100 if i <= 5 else 200,
                              "win": i <= 5, "championName": "Ahri", "championId": 103,
                              "individualPosition": LANES[(i - 1) % 5],
                              "riotIdGameName": f"{match_id}_n{i}", "riotIdTagline": "LAS"}
                             for i in range(1, 11)]}}

    def get_match_timeline(self, match_id):
        frame = {"events": [], "participantFrames": {
            str(p): {"totalGold": 100, "xp": 0, "level": 1, "minionsKilled": 0,
                     "jungleMinionsKilled": 0, "damageStats": {}} for p in range(1, 11)}}
        return {"info": {"frames": [frame] * 21}}


def _padron(db):
    """GM con 50 partidas (ya 'maduro'), Master sin partidas y Diamond."""
    db.save_players([
        {"puuid": "gm_p0", "summoner_id": "s", "tier": "GRANDMASTER", "rank": "I", "league_points": 900},
        {"puuid": "ma_p0", "summoner_id": "s", "tier": "MASTER", "rank": "I", "league_points": 100},
        {"puuid": "di_p0", "summoner_id": "s", "tier": "DIAMOND", "rank": "I", "league_points": 50},
    ])
    db.save_matches_and_timelines_batch(
        [(f"LA2_OLD{i}", _mini_match(f"LA2_OLD{i}", "gm_p0"), None) for i in range(50)])
    db.materialize_player_id_map()


# --------------------------------------------------------------------------- #
# Fechas                                                                      #
# --------------------------------------------------------------------------- #
def test_la_fecha_es_medianoche_de_argentina_en_segundos():
    assert D1 == int(datetime(2026, 10, 1, 3, 0, tzinfo=timezone.utc).timestamp())
    assert window_key(D1, D2) == "2026-10-01..2026-10-15"
    assert window_key(D1, None) == "2026-10-01.."


def test_una_ventana_invertida_se_rechaza(temp_db):
    with pytest.raises(ValueError):
        WindowCrawler(FakeClient({}), temp_db, D2, D1)


def test_el_cliente_manda_starttime_y_endtime_en_segundos():
    from src.riot_client import RiotAPIClient

    client = RiotAPIClient(api_keys=["KEY_P0_AAAAAAAA"], project_indices=[0])
    sent = []

    class Resp:
        status_code, headers, text = 200, {}, ""

        def json(self):
            return []

    client.key_pool[0].session.get = lambda url, params=None, timeout=None: (sent.append(params) or Resp())
    client.get_match_ids_by_puuid("pu", count=20, start_time=D1, end_time=D2)
    assert sent[-1]["startTime"] == D1 and sent[-1]["endTime"] == D2
    client.get_match_ids_by_puuid("pu", count=20, start_time=D1)
    assert "endTime" not in sent[-1]
    assert client.last_request_ok() is True


# --------------------------------------------------------------------------- #
# Fase 1: listar                                                              #
# --------------------------------------------------------------------------- #
def test_lista_a_todo_master_plus_aunque_ya_tenga_50_y_deja_afuera_diamond(temp_db):
    _padron(temp_db)
    client = FakeClient({"gm_p0": ["LA2_A", "LA2_B"], "ma_p0": ["LA2_B", "LA2_C"], "di_p0": ["LA2_X"]})
    crawler = WindowCrawler(client, temp_db, D1, D2)
    st = crawler.list_phase()
    pedidos = {c["puuid"] for c in client.list_calls}
    assert pedidos == {"gm_p0", "ma_p0"}          # el GM con 50 partidas también
    assert all(c["start_time"] == D1 and c["end_time"] == D2 for c in client.list_calls)
    assert st["listados"] == 2
    assert sorted(crawler.pending_match_ids()) == ["LA2_A", "LA2_B", "LA2_C"]  # sin duplicar LA2_B


def test_el_listado_es_reanudable_y_reintenta_solo_los_fallidos(temp_db):
    _padron(temp_db)
    crawler = WindowCrawler(FakeClient({"gm_p0": ["LA2_A"], "ma_p0": ["LA2_C"]}, fail_lists={"ma_p0"}),
                            temp_db, D1, D2)
    st = crawler.list_phase()
    assert (st["listados"], st["fallidos"]) == (1, 1)

    retry = FakeClient({"gm_p0": ["LA2_A"], "ma_p0": ["LA2_C"]})
    crawler.client = retry
    crawler.list_phase()
    assert [c["puuid"] for c in retry.list_calls] == ["ma_p0"]
    assert sorted(crawler.pending_match_ids()) == ["LA2_A", "LA2_C"]


def test_el_listado_respeta_la_senal_de_corte(temp_db):
    _padron(temp_db)
    client = FakeClient({"gm_p0": ["LA2_A"]})
    corte = threading.Event()
    corte.set()
    st = WindowCrawler(client, temp_db, D1, D2).list_phase(stop_event=corte)
    assert st["cortado"] and client.list_calls == []


def test_el_feedback_de_proyecto_del_listado_se_guarda(temp_db):
    _padron(temp_db)
    WindowCrawler(FakeClient({"gm_p0": []}), temp_db, D1, D2).list_phase()
    with temp_db._get_connection() as conn:
        assert tuple(conn.execute("SELECT project_index, source FROM puuid_projects "
                                  "WHERE puuid = 'gm_p0'").fetchone()) == (0, "exact")


def test_el_estado_no_cuenta_lo_que_ya_esta_en_la_base(temp_db):
    _padron(temp_db)
    with temp_db._get_connection() as conn:
        conn.execute("INSERT INTO discarded_matches (match_id, reason) VALUES ('LA2_REMAKE', 'remake')")
        conn.commit()
    crawler = WindowCrawler(FakeClient({"gm_p0": ["LA2_OLD0", "LA2_REMAKE", "LA2_NUEVA"]}),
                            temp_db, D1, D2)
    crawler.list_phase()
    st = crawler.status()
    assert (st["ids_unicos"], st["ya_en_base"], st["pendientes"]) == (3, 2, 1)
    assert st["horas_estimadas"] >= 0


# --------------------------------------------------------------------------- #
# Fase 2: descargar                                                           #
# --------------------------------------------------------------------------- #
def _listado(db, ids, **kw):
    _padron(db)
    client = FakeClient({"gm_p0": ids}, **kw)
    crawler = WindowCrawler(client, db, D1, D2)
    crawler.list_phase()
    return crawler, client


def test_descarga_solo_lo_pendiente_y_anota_el_proyecto_exacto(temp_db):
    crawler, client = _listado(temp_db, ["LA2_N1", "LA2_N2", "LA2_OLD0"])
    st = crawler.download_phase(workers=2)
    assert st["matches_new"] == 2
    assert sorted(client.detail_calls) == ["LA2_N1", "LA2_N2"]   # LA2_OLD0 ya estaba
    with temp_db._get_connection() as conn:
        labels = conn.execute("SELECT project_index, source FROM puuid_projects "
                              "WHERE puuid LIKE 'LA2_N1_p%'").fetchall()
    assert len(labels) == 10 and all(tuple(r) == (1, "exact") for r in labels)
    # Reanudar no repite nada.
    client.detail_calls.clear()
    assert crawler.download_phase(workers=2)["matches_new"] == 0
    assert client.detail_calls == []


def test_una_partida_que_no_baja_queda_pendiente(temp_db):
    crawler, _ = _listado(temp_db, ["LA2_N1", "LA2_ROTA"], fail_details={"LA2_ROTA"})
    st = crawler.download_phase(workers=2)
    assert (st["matches_new"], st["fallidas"]) == (1, 1)
    assert crawler.pending_match_ids() == ["LA2_ROTA"]


def test_la_descarga_respeta_la_senal_de_corte(temp_db):
    crawler, client = _listado(temp_db, ["LA2_N1", "LA2_N2"])
    corte = threading.Event()
    corte.set()
    st = crawler.download_phase(stop_event=corte, workers=2)
    assert st["cortado"] and st["matches_new"] == 0 and client.detail_calls == []
    assert len(crawler.pending_match_ids()) == 2


def test_la_descarga_actualiza_player_id_map(temp_db):
    crawler, _ = _listado(temp_db, ["LA2_N1"])
    crawler.download_phase(workers=2)
    with temp_db._get_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM player_id_map WHERE puuid LIKE 'LA2_N1_p%'"
                            ).fetchone()[0] == 10


# --------------------------------------------------------------------------- #
# run.py                                                                      #
# --------------------------------------------------------------------------- #
def _args(**kw):
    base = dict(ventana_desde="2026-10-01", ventana_hasta="2026-10-15", fase="todo",
                ventana_tiers="CHALLENGER,GRANDMASTER,MASTER", workers=2)
    base.update(kw)
    return argparse.Namespace(**base)


def test_run_lista_y_descarga_de_punta_a_punta(temp_db):
    import run

    _padron(temp_db)
    client = FakeClient({"gm_p0": ["LA2_N1"], "ma_p0": ["LA2_N2"]})
    nuevas = run.ejecutar_ventana(_args(), client, temp_db, threading.Event(), None)
    assert nuevas == 2


def test_run_fase_listar_no_descarga(temp_db):
    import run

    _padron(temp_db)
    client = FakeClient({"gm_p0": ["LA2_N1"]})
    assert run.ejecutar_ventana(_args(fase="listar"), client, temp_db, threading.Event(), None) == 0
    assert client.detail_calls == []


@pytest.mark.parametrize("desde,hasta", [("2026-10-15", "2026-10-01"), ("01/10/2026", None)])
def test_run_rechaza_fechas_invalidas_sin_tocar_la_red(desde, hasta):
    import run

    argv = ["--ventana-desde", desde] + (["--ventana-hasta", hasta] if hasta else [])
    assert run.main(argv) == 2


def test_el_listado_es_concurrente(temp_db):
    """Con varios hilos la latencia de cada request no se suma: 8 jugadores x 0,2 s."""
    import time

    _padron(temp_db)
    temp_db.save_players([{"puuid": f"m{i}", "summoner_id": "s", "tier": "MASTER", "rank": "I",
                           "league_points": 1} for i in range(6)])
    temp_db.materialize_player_id_map()

    class Lento(FakeClient):
        def get_match_ids_by_puuid(self, puuid, **kw):
            time.sleep(0.2)
            return super().get_match_ids_by_puuid(puuid, **kw)

    client = Lento({})
    t0 = time.monotonic()
    st = WindowCrawler(client, temp_db, D1, D2).list_phase(workers=8)
    elapsed = time.monotonic() - t0
    assert st["listados"] == 8          # GM + Master original + 6 Master nuevos
    assert elapsed < 0.2 * 8 / 2        # secuencial serían 1,6 s


# --------------------------------------------------------------------------- #
# Prueba de que nadie quedó partido: cada partida listada contiene al jugador #
# --------------------------------------------------------------------------- #
class ConJugador(FakeClient):
    """El DTO incluye al jugador listado como participante 1 bajo `pu_en_partida`."""

    def __init__(self, ids_by_puuid, pu_en_partida, nombre, **kw):
        super().__init__(ids_by_puuid, **kw)
        self.pu, self.nombre = pu_en_partida, nombre

    def get_match_detail(self, match_id):
        d = super().get_match_detail(match_id)
        p = d["info"]["participants"][0]
        p["puuid"], p["riotIdGameName"] = self.pu, self.nombre
        return d


def _auditoria(db):
    from src.features.identity import attach_for_participants, audit_duplicates
    with db._get_connection() as conn:
        attach_for_participants(conn)
        return audit_duplicates(conn)["listado_ausente_en_su_partida"]


def test_jugador_presente_en_sus_partidas_da_cero(temp_db):
    _padron(temp_db)
    temp_db.record_riot_id_observations([("ma_p0", "Master1", "LAS")])
    c = ConJugador({"ma_p0": ["LA2_N1"]}, pu_en_partida="ma_p1_nuevo", nombre="Master1")
    WindowCrawler(c, temp_db, D1, D2).list_phase()
    WindowCrawler(c, temp_db, D1, D2).download_phase(workers=2)
    assert _auditoria(temp_db) == 0  # su PUUID nuevo se unió por nombre


def test_jugador_partido_tras_renombre_se_detecta(temp_db):
    _padron(temp_db)
    temp_db.record_riot_id_observations([("ma_p0", "NombreViejo", "LAS")])
    # Riot lista la partida para ma_p0, pero llega bajo un PUUID desconocido con otro nombre.
    c = ConJugador({"ma_p0": ["LA2_N1"]}, pu_en_partida="ma_p1_nuevo", nombre="NombreNuevo")
    WindowCrawler(c, temp_db, D1, D2).list_phase()
    WindowCrawler(c, temp_db, D1, D2).download_phase(workers=2)
    assert _auditoria(temp_db) == 1


def test_el_chequeo_sobrevive_a_un_cambio_de_etiqueta(temp_db):
    """La etiqueta player_id puede cambiar al recalcular el mapa; la prueba no se apoya en ella."""
    _padron(temp_db)
    temp_db.record_riot_id_observations([("ma_p0", "NombreViejo", "LAS")])
    c = ConJugador({"ma_p0": ["LA2_N1"]}, pu_en_partida="ma_p1_nuevo", nombre="NombreNuevo")
    WindowCrawler(c, temp_db, D1, D2).list_phase()
    WindowCrawler(c, temp_db, D1, D2).download_phase(workers=2)
    temp_db.record_riot_id_observations([("ma_p0", "Aaa", "LAS")])  # nueva etiqueta mínima
    temp_db.materialize_player_id_map()
    assert _auditoria(temp_db) == 1


def test_relist_vuelve_a_listar_y_registra(temp_db):
    _padron(temp_db)
    c = FakeClient({"gm_p0": ["LA2_A"], "ma_p0": ["LA2_B"]})
    crawler = WindowCrawler(c, temp_db, D1, D2)
    crawler.list_phase()
    with temp_db._get_connection() as conn:
        conn.execute("DELETE FROM crawl_window_listings"); conn.commit()
    gm_pid = next(r["canonical_id"] for r in crawler.roster() if r["players_puuid"] == "gm_p0")
    crawler.relist([gm_pid])
    c.list_calls.clear()
    crawler.list_phase()
    assert [x["puuid"] for x in c.list_calls] == ["gm_p0"]
    with temp_db._get_connection() as conn:
        assert [tuple(r) for r in conn.execute("SELECT players_puuid, match_id FROM crawl_window_listings")] == [("gm_p0", "LA2_A")]


def test_el_crawler_normal_no_toma_diamond_por_defecto():
    import run
    assert run.parse_args([]).tier == "CHALLENGER,GRANDMASTER,MASTER"
    assert run.parse_args(["--ventana-desde", "2026-09-01"]).ventana_tiers == "CHALLENGER,GRANDMASTER,MASTER"
    assert run.parse_args(["--tier", "DIAMOND"]).tier == "DIAMOND"


def test_solo_se_descargan_partidas_del_servidor_del_proyecto(temp_db):
    crawler, client = _listado(temp_db, ["LA2_N1", "BR1_N2", "NA1_N3"])
    assert crawler.pending_match_ids() == ["LA2_N1"]
    crawler.download_phase(workers=2)
    assert client.detail_calls == ["LA2_N1"]
