"""
Pruebas del motor de validación incremental y del planificador oportunista.

Todo en BD temporal o en memoria: sin red y sin tocar `data/lol_las.db`.
"""

import sqlite3
import time

import pytest

import run
from src.validation import (
    ERROR,
    INFO,
    IncrementalValidator,
    MatchFacts,
    ValidationState,
    build_facts,
    evaluate,
    global_checks,
    open_readonly,
)


# --------------------------------------------------------------------------
# Constructores de filas crudas (las mismas tuplas que devuelve el SELECT)
# --------------------------------------------------------------------------
def participante(match_id, pid, team, win, position="TOP", champion_id=1,
                 champion_name="Aatrox", kills=1, deaths=1, assists=1, gold=10000):
    return (match_id, pid, team, win, champion_id, champion_name, position,
            kills, deaths, assists, gold)


def partida_sana(match_id="LA2_1", duration=1800, queue_id=420, con_timeline=True,
                 gold_20=5000):
    """Una partida 5v5 impecable: 10 participantes, roles biyectivos, timeline completa."""
    posiciones = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
    participantes = []
    pid = 1
    for team in (100, 200):
        for pos in posiciones:
            participantes.append(participante(match_id, pid, team, 1 if team == 100 else 0, pos))
            pid += 1
    timeline = [(match_id, 3000, gold_20) for _ in range(10)] if con_timeline else []
    return [(match_id, duration, queue_id)], participantes, timeline


def facts_de(match_rows, participants, timeline):
    facts = build_facts(match_rows, participants, timeline)
    assert len(facts) == 1
    return facts[0]


# --------------------------------------------------------------------------
# build_facts: la agregación
# --------------------------------------------------------------------------
def test_build_facts_agrega_equipos_y_timeline():
    f = facts_de(*partida_sana())
    assert (f.n_participants, f.n_blue, f.n_red, f.n_wins) == (10, 5, 5, 5)
    assert f.n_distinct_participant_ids == 10
    assert f.n_timeline_rows == 10
    assert f.n_position_duplicated == 0
    assert f.n_gold20_null == 0


def test_build_facts_ignora_filas_de_otras_partidas():
    m, p, t = partida_sana("LA2_1")
    p.append(participante("LA2_OTRA", 1, 100, 1))
    t.append(("LA2_OTRA", 3000, 5000))
    f = facts_de(m, p, t)
    assert f.n_participants == 10 and f.n_timeline_rows == 10


def test_build_facts_cuenta_participantes_en_choque_de_posicion():
    m, p, _ = partida_sana()
    p[1] = participante("LA2_1", 2, 100, 1, "TOP")      # segundo TOP azul
    p[2] = participante("LA2_1", 3, 100, 1, "TOP")      # tercer TOP azul
    f = facts_de(m, p, [])
    # El choque involucra a 3 participantes, no a 1 "choque".
    assert f.n_position_duplicated == 3


def test_build_facts_marca_snapshots_nulos():
    m, p, _ = partida_sana()
    f = facts_de(m, p, [("LA2_1", None, None) for _ in range(10)])
    assert f.n_gold10_null == 10 and f.n_gold20_null == 10


# --------------------------------------------------------------------------
# evaluate: la doctrina
# --------------------------------------------------------------------------
def test_partida_sana_no_dispara_nada():
    assert evaluate(facts_de(*partida_sana())) == []


def test_detecta_ganador_incoherente():
    m, p, t = partida_sana()
    p[0] = participante("LA2_1", 1, 100, 0, "TOP")  # el azul gana con solo 4 wins
    assert "ganador_incoherente" in evaluate(facts_de(m, p, t))


def test_detecta_equipos_desbalanceados():
    m, p, t = partida_sana()
    p[9] = participante("LA2_1", 10, 100, 0, "UTILITY")  # 6 azules, 4 rojos
    hallazgos = evaluate(facts_de(m, p, t))
    assert "equipos_desbalanceados" in hallazgos


def test_participantes_incompletos_absorbe_los_derivados():
    """Con roster incompleto, equipos y ganador son consecuencia, no hallazgos aparte."""
    m, p, t = partida_sana()
    hallazgos = evaluate(facts_de(m, p[:9], t))
    assert "participantes_incompletos" in hallazgos
    assert "equipos_desbalanceados" not in hallazgos
    assert "ganador_incoherente" not in hallazgos


def test_detecta_participant_id_duplicado_y_campeon_invalido():
    m, p, t = partida_sana()
    p[1] = participante("LA2_1", 1, 100, 1, "JUNGLE", champion_id=0, champion_name="")
    hallazgos = evaluate(facts_de(m, p, t))
    assert "participant_id_duplicado" in hallazgos
    assert "campeon_invalido" in hallazgos


def test_detecta_stats_negativas():
    m, p, t = partida_sana()
    p[0] = participante("LA2_1", 1, 100, 1, "TOP", kills=-3)
    assert "stats_negativas" in evaluate(facts_de(m, p, t))


def test_remake_y_su_timeline_desperdiciada():
    m, p, t = partida_sana(duration=120)
    hallazgos = evaluate(facts_de(m, p, t))
    assert "remake_no_descartado" in hallazgos
    assert "timeline_de_remake" in hallazgos
    # Un remake sin timeline no reclama la timeline faltante.
    assert "timeline_faltante" not in evaluate(facts_de(*partida_sana(duration=120,
                                                                     con_timeline=False)))


def test_timeline_faltante_e_incompleta():
    m, p, _ = partida_sana(con_timeline=False)
    assert "timeline_faltante" in evaluate(facts_de(m, p, []))
    assert "timeline_incompleto" in evaluate(facts_de(m, p, [("LA2_1", 3000, 5000)] * 7))


def test_snapshot20_solo_se_exige_pasados_los_21_minutos():
    """Rendirse al minuto 19 no es un error de dataset; a los 30 sí falta el @20."""
    m, p, _ = partida_sana(duration=1140)  # 19 min
    sin_20 = [("LA2_1", 3000, None)] * 10
    assert "snapshot20_ausente" not in evaluate(facts_de(m, p, sin_20))

    m_larga, p_larga, _ = partida_sana(duration=1800)
    assert "snapshot20_ausente" in evaluate(facts_de(m_larga, p_larga, sin_20))


def test_duracion_invalida_no_arrastra_chequeos_de_timeline():
    m, p, t = partida_sana(duration=None)
    hallazgos = evaluate(facts_de(m, p, t))
    assert hallazgos == ["duracion_invalida"]


def test_cola_no_soloq():
    assert "cola_no_soloq" in evaluate(facts_de(*partida_sana(queue_id=440)))


def test_posiciones_vacias_no_cuentan_como_choque():
    m, p, t = partida_sana()
    for i in (0, 1, 2):
        p[i] = participante("LA2_1", i + 1, 100, 1, "Invalid")
    f = facts_de(m, p, t)
    assert f.n_position_empty == 3 and f.n_position_duplicated == 0
    hallazgos = evaluate(f)
    assert "posicion_vacia" in hallazgos and "posicion_duplicada" not in hallazgos


# --------------------------------------------------------------------------
# IncrementalValidator sobre BD real (temporal)
# --------------------------------------------------------------------------
def insertar(db, match_rows, participants, timeline=()):
    with db._get_connection() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO matches (match_id, game_duration, queue_id) VALUES (?,?,?);",
            match_rows)
        conn.executemany(
            "INSERT OR REPLACE INTO match_participants "
            "(match_id, puuid, participant_id, team_id, win, champion_id, champion_name, "
            " individual_position, kills, deaths, assists, gold_earned) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?);",
            [(r[0], f"PUUID_{r[0]}_{r[1]}", r[1], r[2], r[3], r[4], r[5], r[6],
              r[7], r[8], r[9], r[10]) for r in participants])
        if timeline:
            conn.executemany(
                "INSERT OR REPLACE INTO match_timeline (match_id, participant_id, gold_10, gold_20) "
                "VALUES (?,?,?,?);",
                [(t[0], i + 1, t[1], t[2]) for i, t in enumerate(timeline)])
        conn.commit()


def test_validador_recorre_la_bd_y_acumula_hallazgos(temp_db, tmp_path):
    m, p, t = partida_sana("LA2_1")
    insertar(temp_db, m, p, t)
    m2, p2, _ = partida_sana("LA2_2", duration=100)
    insertar(temp_db, m2, p2)

    v = IncrementalValidator(temp_db.db_path, state_path=tmp_path / "estado.json", slice_size=10)
    assert v.pending() == 2
    assert v.run_slice() == 2
    assert v.run_slice() == 0            # al día
    assert v.state.matches_validated == 2
    assert v.state.findings["remake_no_descartado"]["samples"] == ["LA2_2"]
    v.close()


def test_cursor_es_incremental_y_no_reaudita(temp_db, tmp_path):
    insertar(temp_db, *partida_sana("LA2_1"))
    estado = tmp_path / "estado.json"
    v = IncrementalValidator(temp_db.db_path, state_path=estado, slice_size=10)
    v.run_until_done()
    primera = v.state.matches_validated
    v.close()

    # Llega una partida nueva mientras el crawler sigue: solo esa se audita.
    insertar(temp_db, *partida_sana("LA2_2", duration=100))
    v2 = IncrementalValidator(temp_db.db_path, state_path=estado, slice_size=10)
    assert v2.pending() == 1
    assert v2.run_until_done() == 1
    assert v2.state.matches_validated == primera + 1
    v2.close()


def test_rebanadas_pequenas_cubren_todo_el_backlog(temp_db, tmp_path):
    for i in range(7):
        insertar(temp_db, *partida_sana(f"LA2_{i}"))
    v = IncrementalValidator(temp_db.db_path, state_path=tmp_path / "e.json", slice_size=2)
    assert v.run_until_done() == 7
    assert v.state.slices == 4  # 2 + 2 + 2 + 1
    v.close()


def test_reset_borra_cursor_y_hallazgos(temp_db, tmp_path):
    insertar(temp_db, *partida_sana("LA2_1", duration=100))
    estado = tmp_path / "e.json"
    v = IncrementalValidator(temp_db.db_path, state_path=estado, slice_size=10)
    v.run_until_done()
    assert v.state.findings
    v.reset()
    assert v.state.last_rowid == 0 and v.state.findings == {}
    assert IncrementalValidator(temp_db.db_path, state_path=estado).state.last_rowid == 0
    v.close()


def test_estado_sobrevive_al_disco_y_descarta_versiones_viejas():
    st = ValidationState(last_rowid=42, matches_validated=10)
    st.record("cola_no_soloq", "LA2_9")
    ida = ValidationState.from_json(st.to_json())
    assert ida.last_rowid == 42 and ida.findings["cola_no_soloq"]["count"] == 1
    # Un estado de otra versión se descarta entero en vez de leerse mal.
    assert ValidationState.from_json({"version": 999, "last_rowid": 5}).last_rowid == 0


def test_severidades_y_conteo_de_errores(temp_db, tmp_path):
    m, p, t = partida_sana("LA2_1")
    p[0] = participante("LA2_1", 1, 100, 0, "JUNGLE")  # rompe ganador y duplica JUNGLE
    insertar(temp_db, m, p, t)
    v = IncrementalValidator(temp_db.db_path, state_path=tmp_path / "e.json")
    v.run_until_done()
    grouped = v.findings_by_severity()
    assert [slug for slug, _, _ in grouped[ERROR]] == ["ganador_incoherente"]
    assert [slug for slug, _, _ in grouped[INFO]] == ["posicion_duplicada"]
    assert v.error_count() == 1
    v.close()


def test_la_validacion_no_puede_escribir(temp_db, tmp_path):
    insertar(temp_db, *partida_sana("LA2_1"))
    conn = open_readonly(temp_db.db_path)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM matches;")
    conn.close()


def test_global_checks_detecta_huerfanos(temp_db):
    insertar(temp_db, *partida_sana("LA2_1"))
    # Huérfano deliberado: se apagan las claves foráneas para poder crearlo.
    raw = sqlite3.connect(temp_db.db_path)
    raw.execute("PRAGMA foreign_keys = OFF;")
    raw.execute("INSERT INTO match_participants (match_id, puuid, participant_id) "
                "VALUES ('LA2_FANTASMA', 'PUUID_X', 1);")
    raw.execute("INSERT INTO discarded_matches (match_id, reason) VALUES ('LA2_1', 'remake');")
    raw.commit()
    raw.close()

    conn = open_readonly(temp_db.db_path)
    try:
        res = global_checks(conn)
    finally:
        conn.close()
    assert res["participaciones_huerfanas"] == 1
    assert res["descartadas_pero_guardadas"] == 1
    assert res["timelines_huerfanas"] == 0
    assert res["partidas_sin_participantes"] == 0


# --------------------------------------------------------------------------
# Planificador oportunista (run.py)
# --------------------------------------------------------------------------
class _LimiterFalso:
    def __init__(self, wait, activo=True):
        self.wait = wait
        # `activo` simula que esa clave viene pidiendo: historial no vacio.
        self.long_history = [1.0] if activo else []

    def get_wait_time(self):
        return self.wait


class _KeyFalsa:
    def __init__(self, wait, is_active=True, pidiendo=True):
        self.rate_limiter = _LimiterFalso(wait, activo=pidiendo)
        self.is_active = is_active


class _ClienteFalso:
    def __init__(self, *waits):
        self.key_pool = list(waits)


def _worker(client, **kwargs):
    return run.OpportunisticValidator(validator=None, client=client, **kwargs)


def test_sin_stall_no_se_valida():
    """Si alguna key tiene ranura libre, la prioridad es la red: presupuesto cero."""
    w = _worker(_ClienteFalso(_KeyFalsa(0.0), _KeyFalsa(3.0)))
    w._last_pass = time.time()
    presupuesto, de_stall = w._budget()
    assert presupuesto == 0.0 and de_stall is False


def test_el_stall_del_pool_se_convierte_en_presupuesto():
    w = _worker(_ClienteFalso(_KeyFalsa(1.5), _KeyFalsa(3.0)), max_budget=5.0)
    presupuesto, de_stall = w._budget()
    assert de_stall is True and presupuesto == pytest.approx(1.45)


def test_el_presupuesto_esta_acotado():
    w = _worker(_ClienteFalso(_KeyFalsa(120.0)), max_budget=5.0)
    presupuesto, _ = w._budget()
    assert presupuesto == 5.0


def test_las_keys_inactivas_no_cuentan_como_ranura_libre():
    w = _worker(_ClienteFalso(_KeyFalsa(0.0, is_active=False), _KeyFalsa(2.0)))
    presupuesto, de_stall = w._budget()
    assert de_stall is True and presupuesto > 0


def test_sin_crawler_el_presupuesto_es_maximo():
    w = _worker(None, max_budget=4.0)
    assert w._budget() == (4.0, False)


def test_sin_stall_prolongado_se_valida_igual():
    w = _worker(_ClienteFalso(_KeyFalsa(0.0)), idle_every=10.0, idle_budget=0.7)
    w._last_pass = time.time() - 30.0
    presupuesto, de_stall = w._budget()
    assert de_stall is False and presupuesto == 0.7


def test_resolve_keys_descarta_placeholders():
    keys = run.resolve_keys("RGAPI-real-1, RGAPI-TU-CLAVE-AQUI ,RGAPI-real-2,")
    assert keys == ["RGAPI-real-1", "RGAPI-real-2"]


def test_solo_validar_y_sin_validar_se_excluyen():
    assert run.main(["--solo-validar", "--sin-validar"]) == 2


def test_el_pool_ocioso_libera_toda_la_maquina():
    """La exportacion de los 4 CSV son minutos sin red: tiempo gratis, no espera."""
    w = _worker(_ClienteFalso(_KeyFalsa(0.0, pidiendo=False)), max_budget=5.0)
    w._last_pass = time.time()
    presupuesto, de_stall = w._budget()
    assert presupuesto == 5.0 and de_stall is False


def test_un_pool_activo_sin_stall_sigue_sin_presupuesto():
    w = _worker(_ClienteFalso(_KeyFalsa(0.0, pidiendo=True)))
    w._last_pass = time.time()
    assert w._budget() == (0.0, False)
