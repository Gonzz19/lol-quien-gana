"""Tests del rating TrueSkill por jugador y partida (src/features/rating.py)."""

import math

import numpy as np
import pandas as pd
import pytest

from src.features.rating import (ParametrosRating, actualizar, correr, lp_absoluto, medir, metricas,
                                 ocultas_por_partida, prob_azul)


def partida(match_id, t, azul, rojo, gana_azul):
    filas = [{"match_id": match_id, "game_creation": t, "player_id": p, "team_id": 100, "win": int(gana_azul)}
             for p in azul]
    filas += [{"match_id": match_id, "game_creation": t, "player_id": p, "team_id": 200,
               "win": int(not gana_azul)} for p in rojo]
    return filas


AZ = ["a1", "a2", "a3", "a4", "a5"]
RO = ["r1", "r2", "r3", "r4", "r5"]


def test_lp_absoluto_escala():
    assert lp_absoluto("DIAMOND", "IV", 0) == 0
    assert lp_absoluto("DIAMOND", "I", 99) == 399
    assert lp_absoluto("MASTER", "I", 0) == 400
    assert lp_absoluto("CHALLENGER", "I", 1500) == 1900
    assert lp_absoluto("EMERALD", "I", 50) == -50
    assert lp_absoluto("EMERALD", "IV", 0) == -400
    assert lp_absoluto("PLATINUM", "I", 0) == -500
    assert math.isnan(lp_absoluto("UNRANKED", None, 0))


def test_prob_simetrica_e_invariante_a_desplazamiento():
    mu = [100.0] * 5 + [100.0] * 5
    s2 = [50.0 ** 2] * 10
    assert prob_azul(mu, s2, range(5), range(5, 10), 200) == pytest.approx(0.5)
    mu2 = [m + 1000 for m in mu]
    mu2[0] += 300
    mu[0] += 300
    assert prob_azul(mu, s2, range(5), range(5, 10), 200) == pytest.approx(
        prob_azul(mu2, s2, range(5), range(5, 10), 200))
    # La ventaja roja baja la probabilidad azul con habilidades iguales.
    assert prob_azul([0.0] * 10, s2, range(5), range(5, 10), 200, ventaja_roja=50) < 0.5


def test_actualizar_sube_ganadores_baja_perdedores_y_reduce_sigma():
    mu = [0.0] * 10
    s2 = [300.0 ** 2] * 10
    actualizar(mu, s2, list(range(5)), list(range(5, 10)), 200)
    assert all(m > 0 for m in mu[:5]) and all(m < 0 for m in mu[5:])
    assert sum(mu) == pytest.approx(0.0, abs=1e-9)  # con sigmas iguales, suma cero
    assert all(s < 300.0 ** 2 for s in s2)


def test_sorpresa_mueve_mas_que_resultado_esperado():
    def cambio(mu_gan):
        mu = [mu_gan] * 5 + [0.0] * 5
        s2 = [200.0 ** 2] * 10
        actualizar(mu, s2, list(range(5)), list(range(5, 10)), 200)
        return mu[0] - mu_gan
    assert cambio(-200.0) > cambio(0.0) > cambio(200.0) > 0


def test_medicion_acerca_a_la_foto_y_reduce_sigma():
    mu, s2 = [0.0, 0.0], [400.0 ** 2, 50.0 ** 2]
    medir(mu, s2, [0, 1], [800.0, 800.0], sigma_foto=100.0)
    assert 0 < mu[1] < mu[0] < 800  # el incierto se mueve más
    assert s2[0] < 400.0 ** 2 and s2[1] < 50.0 ** 2


def test_rating_previo_no_depende_del_resultado_propio():
    base = partida("m1", 1, AZ, RO, True) + partida("m2", 2, AZ, RO, True)
    alterna = partida("m1", 1, AZ, RO, True) + partida("m2", 2, AZ, RO, False)
    p = ParametrosRating()
    f1 = correr(pd.DataFrame(base), p)["filas"].set_index(["match_id", "player_id"])
    f2 = correr(pd.DataFrame(alterna), p)["filas"].set_index(["match_id", "player_id"])
    pd.testing.assert_series_equal(f1.loc["m2", "mu_pre"], f2.loc["m2", "mu_pre"])
    assert (f1.loc["m2", "mu_pre"][AZ] > 0).all()  # m1 ya se reflejó


def test_foto_se_aplica_en_su_fecha_y_no_antes():
    datos = pd.DataFrame(partida("m1", 1, AZ, RO, True) + partida("m2", 10, AZ, RO, True))
    foto = pd.DataFrame({"player_id": ["a1"], "z": [2000.0]})
    f = correr(datos, ParametrosRating(), foto=foto, t_foto_ms=5)["filas"].set_index(["match_id", "player_id"])
    assert f.loc[("m1", "a1"), "mu_pre"] == 0.0  # antes de la foto: prior
    assert f.loc[("m2", "a1"), "mu_pre"] > 1000.0  # después: corregido hacia la foto


def test_partidas_ocultas_agrandan_la_incertidumbre():
    datos = pd.DataFrame(partida("m1", 1, AZ, RO, True) + partida("m2", 2, AZ, RO, True))
    p = ParametrosRating(tau=20.0)
    sin = correr(datos, p, foto=pd.DataFrame({"player_id": [], "z": []}), t_foto_ms=100)
    con = correr(datos, p, foto=pd.DataFrame({"player_id": ["zz"], "z": [0.0]}), t_foto_ms=100,
                 ocultas=pd.Series({"a1": 5.0}))
    s_sin = sin["filas"].set_index(["match_id", "player_id"]).loc[("m2", "a1"), "sigma_pre"]
    s_con = con["filas"].set_index(["match_id", "player_id"]).loc[("m2", "a1"), "sigma_pre"]
    assert s_con > s_sin


def test_partida_sin_ganador_no_actualiza():
    filas = partida("m1", 1, AZ, RO, True)
    for f in filas:
        f["win"] = 0
    r = correr(pd.DataFrame(filas + partida("m2", 2, AZ, RO, True)), ParametrosRating())
    f = r["filas"].set_index(["match_id", "player_id"])
    assert (f.loc["m2", "mu_pre"] == 0).all()
    assert r["partidas"].set_index("match_id").loc["m1", "gana_azul"] == -1


def test_ocultas_por_partida():
    partes = pd.DataFrame({"player_id": ["a"] * 4 + ["b"] * 10, "game_creation": [5] * 14})
    foto = pd.DataFrame({"player_id": ["a", "b", "c"], "partidas": [20, 8, 30]})
    h = ocultas_por_partida(partes, foto, desde_ms=0, hasta_ms=10, tope=30)
    assert h["a"] == pytest.approx(4.0)   # (20 − 4) / 4
    assert h["b"] == 0.0                  # vimos más de las que dice la foto
    assert "c" not in h.index             # sin partidas observadas


def test_metricas_constante():
    partidas = pd.DataFrame({"p_azul_pre": [0.5] * 4, "gana_azul": [1, 0, 1, -1]})
    m = metricas(partidas)
    assert m["n"] == 3
    assert m["log_loss"] == pytest.approx(np.log(2))
