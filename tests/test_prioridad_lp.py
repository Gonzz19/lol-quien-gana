"""Orden suave de la cola por LP y menos partidas (src/collector.py: priorizar_por_lp)."""

import pytest

from src.collector import priorizar_por_lp


def _p(nombre, tier, lp, juegos):
    return {"canonical_id": nombre, "tier": tier, "league_points": lp, "existing_recorded_games": juegos}


def test_con_peso_alto_va_primero_el_lp_alto():
    cola = [_p("bajo", "MASTER", 10, 60), _p("alto", "MASTER", 800, 60), _p("medio", "MASTER", 300, 60)]
    assert [p["canonical_id"] for p in priorizar_por_lp(cola, 0.7)] == ["alto", "medio", "bajo"]


def test_es_suave_un_master_bajo_con_muy_pocas_partidas_puede_adelantarse():
    cola = ([_p(f"alto{i}", "MASTER", 900 - i, 280) for i in range(5)]
            + [_p("bajo_vacio", "MASTER", 5, 0)] + [_p(f"medio{i}", "MASTER", 400 - i, 250) for i in range(5)])
    orden = [p["canonical_id"] for p in priorizar_por_lp(cola, 0.7)]
    assert len(orden) == len(cola)                      # nadie sale de la cola
    assert orden.index("bajo_vacio") < orden.index("medio4")  # pocas partidas lo adelanta
    assert orden[0] == "alto0"


def test_el_tier_pesa_mas_que_el_lp():
    cola = [_p("master900", "MASTER", 900, 50), _p("gm0", "GRANDMASTER", 0, 50)]
    assert priorizar_por_lp(cola, 1.0)[0]["canonical_id"] == "gm0"


def test_peso_cero_es_solo_menos_partidas():
    cola = [_p("a", "MASTER", 900, 200), _p("b", "MASTER", 0, 10)]
    assert priorizar_por_lp(cola, 0.0)[0]["canonical_id"] == "b"


def test_peso_invalido():
    with pytest.raises(ValueError):
        priorizar_por_lp([_p("a", "MASTER", 1, 1), _p("b", "MASTER", 2, 2)], 1.5)
