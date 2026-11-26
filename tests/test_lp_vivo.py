"""Tests del LP por partida en vivo (`src/lp_vivo.py`)."""

from contextlib import contextmanager

import pandas as pd
import pytest

from src.lp_vivo import (MAX_INTENTOS, AlmacenLP, agregar_lobby, ciclo_ladder, clasificar, comparar, elegir_partida, resolver_evento,
                         resumir_partida)

T0 = 1_000_000_000


def jug(puuid="a", lp=100, w=10, l=10, tier="MASTER", rank="I"):
    return {"puuid": puuid, "tier": tier, "rank": rank, "league_points": lp, "wins": w, "losses": l}


def detalle(match_id="LA2_1", fin=T0 + 100_000, duracion=1800, ganador="a", perdedor="b"):
    return {"metadata": {"matchId": match_id},
            "info": {"platformId": "LA2", "queueId": 420, "gameVersion": "16.20", "gameCreation": fin - 1_900_000,
                     "gameStartTimestamp": fin - duracion * 1000, "gameEndTimestamp": fin, "gameDuration": duracion,
                     "participants": [{"puuid": ganador, "teamId": 100, "win": True, "championName": "Ahri"},
                                      {"puuid": perdedor, "teamId": 200, "win": False, "championName": "Lux"}]}}


class Cliente:
    """Doble de `RiotAPIClient`: ligas fijas por tier y partidas por jugador."""

    def __init__(self, ligas=None, por_puuid=None, ids=None, detalles=None, ok=True):
        self.ligas, self.por_puuid, self.ok = ligas or {}, por_puuid or {}, ok
        self.ids, self.detalles = ids or {}, detalles or {}
        self.pedidos_detalle = 0
        self.pedidos_ids = 0

    def get_apex_league(self, tier):
        return self.ligas.get(tier, [])

    def last_request_ok(self):
        return self.ok

    def get_soloq_entry_by_puuid(self, opciones):
        puuid = next(iter(opciones.values()))
        return self.por_puuid.get(puuid)

    def get_match_ids_by_puuid(self, puuid, count, start_time):
        self.pedidos_ids += 1
        return self.ids.get(puuid, [])

    def get_match_detail(self, match_id):
        self.pedidos_detalle += 1
        return self.detalles.get(match_id)


@pytest.fixture
def almacen(tmp_path):
    return AlmacenLP(tmp_path / "lp.db")


def eventos(almacen):
    with almacen._conn() as c:
        return [dict(f) for f in c.execute("SELECT * FROM eventos ORDER BY id").fetchall()]


def reloj(*tiempos):
    it = iter(tiempos)
    return lambda: next(it)


def test_lobby_excluye_al_propio_jugador_y_no_cruza_partidas():
    d = pd.DataFrame({"match_id": ["m", "m", "m", "n"], "z_antes": [100.0, 200.0, 600.0, 900.0]})
    r = agregar_lobby(d)
    assert r["lobby_lp"].tolist()[:3] == [400.0, 350.0, 150.0]
    assert r["lobby_conocidos"].tolist()[:3] == [2, 2, 2]
    assert pd.isna(r["lobby_lp"].iloc[3])   # solo él medido en esa partida


def test_replica_atrasada_no_genera_partida_falsa(almacen):
    previo = {"a": jug("a", lp=118, w=11)}
    viejo = Cliente({"MASTER": [jug("a", lp=100, w=10)]})
    t, _ = ciclo_ladder(viejo, almacen, "la2", previo, T0, ["MASTER"], 1, reloj(T0 + 1, T0 + 2))
    ciclo_ladder(Cliente({"MASTER": [jug("a", lp=118, w=11)]}), almacen, "la2", previo, t, ["MASTER"], 1,
                 reloj(T0 + 3, T0 + 4))
    assert eventos(almacen) == [] and previo["a"]["league_points"] == 118


def test_baja_grande_de_w_mas_l_es_reinicio_y_no_atraso(almacen):
    previo = {"a": jug("a", w=100)}
    ciclo_ladder(Cliente({"MASTER": [jug("a", lp=0, w=1, l=0)]}), almacen, "la2", previo, T0, ["MASTER"], 1,
                 reloj(T0 + 1, T0 + 2))
    assert eventos(almacen)[0]["tipo"] == "reinicio"


def test_corrido_por_el_tope_sin_jugar_no_genera_evento_ni_se_reconsulta(almacen):
    previo = {"a": jug("a"), "b": jug("b")}
    c = Cliente({"MASTER": [jug("b")]}, por_puuid={"a": jug("a")})
    ciclo_ladder(c, almacen, "la2", previo, T0, ["MASTER"], 1, reloj(T0 + 1, T0 + 2))
    assert eventos(almacen) == [] and previo["a"]["en_ladder"] is False


def test_un_jugador_no_enlaza_dos_eventos_a_la_misma_partida(almacen):
    previo = {"a": jug("a")}
    ciclo_ladder(Cliente({"MASTER": [jug("a", lp=118, w=11)]}), almacen, "la2", previo, T0, ["MASTER"], 1,
                 reloj(T0 + 180_000, T0 + 181_000))
    ciclo_ladder(Cliente({"MASTER": [jug("a", lp=140, w=12)]}), almacen, "la2", previo, T0 + 180_000, ["MASTER"], 1,
                 reloj(T0 + 300_000, T0 + 301_000))
    c = Cliente(ids={"a": ["LA2_1"]}, detalles={"LA2_1": detalle()})
    resolver_evento(c, almacen, almacen.pendientes("la2", T0 + 400_000)[0], T0 + 400_000)
    assert [e["estado"] for e in eventos(almacen)].count("enlazada") == 1


@pytest.mark.parametrize("despues, esperado", [
    (jug(lp=120, w=11), "partida"), (jug(lp=90, l=11), "partida"), (jug(lp=130, w=12), "varias"),
    (jug(lp=25), "sin_partida"), (jug(lp=0, w=0, l=0), "reinicio")])
def test_clasificar_por_la_resta_de_w_mas_l(despues, esperado):
    assert clasificar(jug(), despues) == esperado


def test_clasificar_sin_lectura_previa_no_es_partida():
    assert clasificar(None, jug()) == "entro"


def test_comparar_ignora_a_los_que_no_cambiaron_y_detecta_salidas():
    previo = {"a": jug("a"), "b": jug("b"), "c": {**jug("c"), "en_ladder": False}}
    cambios, salieron = comparar(previo, {"a": jug("a"), "d": jug("d")})
    assert [(x is None, y["puuid"]) for x, y in cambios] == [(True, "d")]
    assert salieron == ["b"]   # c ya estaba afuera: no se lo vuelve a consultar


def test_elegir_partida_exige_una_sola_en_la_ventana_y_descarta_remakes():
    p = {"duracion_s": 1800, "game_end": 1000}
    remake = {"duracion_s": 200, "game_end": 1000}
    assert elegir_partida([p, remake], 900, 1100) is p
    assert elegir_partida([p, dict(p)], 900, 1100) is None
    assert elegir_partida([{"duracion_s": 1800, "game_end": 10_000_000}], 900, 1100) is None


def test_primera_lectura_no_genera_eventos(almacen):
    previo = {}
    c = Cliente({"MASTER": [jug("a")]})
    t, r = ciclo_ladder(c, almacen, "la2", previo, None, ["MASTER"], 1, reloj(T0, T0 + 1))
    assert t == T0 and r["jugadores"] == 1 and eventos(almacen) == []
    assert almacen.cargar_estado("la2")[0]["a"]["league_points"] == 100


def test_lectura_fallida_no_marca_salidas_ni_avanza_el_tiempo(almacen):
    previo = {"a": jug("a")}
    c = Cliente({"MASTER": []}, por_puuid={"a": jug("a", lp=0, tier="DIAMOND")})
    t, r = ciclo_ladder(c, almacen, "la2", previo, T0, ["MASTER"], 1, reloj(T0 + 5))
    assert r["fallo"] and t == T0 and eventos(almacen) == []


def test_una_partida_guarda_lp_de_antes_y_despues_y_queda_pendiente(almacen):
    previo = {"a": jug("a")}
    c = Cliente({"MASTER": [jug("a", lp=118, w=11)]})
    ciclo_ladder(c, almacen, "la2", previo, T0, ["MASTER"], 1, reloj(T0 + 180_000, T0 + 181_000))
    ev = eventos(almacen)[0]
    assert (ev["tipo"], ev["estado"], ev["lp_antes"], ev["lp_despues"]) == ("partida", "pendiente", 100, 118)
    assert (ev["t_antes"], ev["t_despues"]) == (T0, T0 + 181_000)


def test_el_que_baja_a_diamond_se_mide_por_puuid(almacen):
    previo = {"a": jug("a", lp=5), "b": jug("b")}
    c = Cliente({"MASTER": [jug("b")]}, por_puuid={"a": {**jug("a", lp=80, l=11, tier="DIAMOND"), "project_index": 1}})
    ciclo_ladder(c, almacen, "la2", previo, T0, ["MASTER"], 1, reloj(T0 + 1, T0 + 2))
    ev = eventos(almacen)[0]
    assert (ev["tipo"], ev["tier_despues"], ev["lp_despues"]) == ("partida", "DIAMOND", 80)
    assert previo["a"]["en_ladder"] is False


def test_una_partida_enlaza_a_todos_con_un_solo_pedido(almacen):
    previo = {"a": jug("a"), "b": jug("b")}
    ladder = Cliente({"MASTER": [jug("a", lp=118, w=11), jug("b", lp=85, l=11)]})
    ciclo_ladder(ladder, almacen, "la2", previo, T0, ["MASTER"], 1, reloj(T0 + 180_000, T0 + 181_000))
    pend = almacen.pendientes("la2", T0 + 200_000)
    c = Cliente(ids={"a": ["LA2_1"]}, detalles={"LA2_1": detalle()})
    resolver_evento(c, almacen, pend[0], T0 + 200_000)
    assert [(e["estado"], e["match_id"]) for e in eventos(almacen)] == [("enlazada", "LA2_1")] * 2
    assert c.pedidos_detalle == 1 and almacen.pendientes("la2", T0 + 200_000) == []


def test_resultado_contrario_al_w_l_queda_inconsistente(almacen):
    previo = {"a": jug("a")}
    ciclo_ladder(Cliente({"MASTER": [jug("a", lp=118, w=11)]}), almacen, "la2", previo, T0, ["MASTER"], 1,
                 reloj(T0 + 180_000, T0 + 181_000))
    c = Cliente(ids={"a": ["LA2_1"]}, detalles={"LA2_1": detalle(ganador="b", perdedor="a")})
    resolver_evento(c, almacen, almacen.pendientes("la2", T0 + 200_000)[0], T0 + 200_000)
    assert eventos(almacen)[0]["estado"] == "inconsistente"


def test_sin_partida_reintenta_y_termina_sin_enlace(almacen):
    previo = {"a": jug("a")}
    ciclo_ladder(Cliente({"MASTER": [jug("a", lp=118, w=11)]}), almacen, "la2", previo, T0, ["MASTER"], 1,
                 reloj(T0 + 180_000, T0 + 181_000))
    ev = almacen.pendientes("la2", T0 + 200_000)[0]
    resolver_evento(Cliente(), almacen, ev, T0 + 200_000)
    assert almacen.pendientes("la2", T0 + 200_000) == []   # espera antes de reintentar
    for _ in range(MAX_INTENTOS):
        resolver_evento(Cliente(), almacen, ev, T0 + 200_000)
    assert eventos(almacen)[0]["estado"] == "sin_enlace"


def test_partida_ya_guardada_no_se_vuelve_a_pedir(almacen):
    almacen.guardar_partida(*resumir_partida(detalle()))
    previo = {"a": jug("a")}
    ciclo_ladder(Cliente({"MASTER": [jug("a", lp=118, w=11)]}), almacen, "la2", previo, T0, ["MASTER"], 1,
                 reloj(T0 + 180_000, T0 + 181_000))
    c = Cliente(ids={"a": ["LA2_1"]})
    resolver_evento(c, almacen, almacen.pendientes("la2", T0 + 200_000)[0], T0 + 200_000)
    assert c.pedidos_detalle == 0 and eventos(almacen)[0]["estado"] == "enlazada"


def test_retomar_compara_contra_la_ultima_lectura_guardada(almacen):
    previo = {}
    ciclo_ladder(Cliente({"MASTER": [jug("a")]}), almacen, "la2", previo, None, ["MASTER"], 1, reloj(T0, T0 + 1))
    previo, t = almacen.cargar_estado("la2")
    assert t == T0
    ciclo_ladder(Cliente({"MASTER": [jug("a", lp=118, w=11)]}), almacen, "la2", previo, t, ["MASTER"], 1,
                 reloj(T0 + 180_000, T0 + 181_000))
    assert eventos(almacen)[0]["lp_antes"] == 100


def test_evento_con_dos_partidas_validas_queda_pendiente(almacen):
    prev = {"a": jug("a"), "b": jug("b")}
    ladder = Cliente({"MASTER": [jug("a", lp=118, w=11), jug("b", lp=85, l=11)]})
    ahora = T0 + 200_000
    ciclo_ladder(ladder, almacen, "la2", prev, T0, ["MASTER"], 1, reloj(T0 + 180_000, T0 + 181_000))
    ids = ["LA2_1", "LA2_2"]
    cliente = Cliente(ids={"a": ids}, detalles={
        "LA2_1": detalle(), "LA2_2": detalle("LA2_2", fin=T0 + 150_000, perdedor="c"),
    })
    resolver_evento(cliente, almacen, almacen.pendientes("la2", ahora)[0], ahora)
    a, b = eventos(almacen)
    assert (a["estado"], a["match_id"], a["intentos"]) == ("pendiente", None, 1)
    assert a["proximo_intento"] > ahora
    assert (b["estado"], b["match_id"]) == ("enlazada", "LA2_1")


def test_alternativa_ya_en_la_base_tambien_hace_ambiguo_el_evento(almacen):
    almacen.guardar_partida(*resumir_partida(detalle("LA2_2", fin=T0 + 140_000)))
    ahora = T0 + 200_000
    prev = {"a": jug("a")}
    ciclo_ladder(Cliente({"MASTER": [jug("a", w=11)]}), almacen, "la2", prev, T0,
                 ["MASTER"], 1, reloj(T0 + 180_000, T0 + 181_000))
    cliente = Cliente(ids={"a": ["LA2_1"]}, detalles={"LA2_1": detalle()})
    resolver_evento(cliente, almacen, almacen.pendientes("la2", ahora)[0], ahora)
    assert eventos(almacen)[0]["estado"] == "pendiente"


def test_evento_reintenta_si_falta_un_detalle(almacen):
    prev = {"a": jug("a")}
    ahora = T0 + 200_000
    ciclo_ladder(Cliente({"MASTER": [jug("a", w=11)]}), almacen, "la2", prev, T0,
                 ["MASTER"], 1, reloj(T0 + 180_000, T0 + 181_000))
    cliente = Cliente(ids={"a": ["LA2_1", "LA2_falta"]}, detalles={"LA2_1": detalle()})
    resolver_evento(cliente, almacen, almacen.pendientes("la2", ahora)[0], ahora)
    assert (eventos(almacen)[0]["estado"], eventos(almacen)[0]["intentos"]) == ("pendiente", 1)


def test_enlace_en_lote_agrupa_los_selects_de_diez_jugadores(almacen, monkeypatch):
    match = detalle()
    ids = [str(n) for n in range(10)]
    match["info"]["participants"] = [{"puuid": puuid, "win": n < 5} for n, puuid in enumerate(ids)]
    partida, jugadores = resumir_partida(match)
    almacen.guardar_partida(partida, jugadores)
    prev = {p: jug(p) for p in ids}
    ladder = Cliente({"MASTER": [jug(p, w=10 + int(n < 5), l=10 + int(n >= 5)) for n, p in enumerate(ids)]})
    ciclo_ladder(ladder, almacen, "la2", prev, T0, ["MASTER"], 1,
                 reloj(T0 + 180_000, T0 + 181_000))
    statements = []
    connect = almacen._conn

    @contextmanager
    def traced():
        with connect() as conn:
            conn.set_trace_callback(statements.append)
            yield conn

    monkeypatch.setattr(almacen, "_conn", traced)
    assert almacen.enlazar_participantes(partida) == 10
    reads = [sql for sql in statements if sql.lstrip().startswith(("SELECT", "WITH"))]
    assert len(reads) == 1
    assert {e["estado"] for e in eventos(almacen)} == {"enlazada"}
