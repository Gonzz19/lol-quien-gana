"""Tests para el modelo mecánico del paso de LP y el mapa por tramos (`src/features/lp_mecanica.py`)."""

import numpy as np
import pandas as pd
import pytest

from src.features.elo_reverso import DIA_MS
from src.features.lp_mecanica import (
    ParametrosPaso,
    _pullback_lineal,
    _pushforward_lineal,
    ajustar_k_diamond,
    ajustar_parametros_las,
    avanzar_banco,
    decaer_dias,
    delta_de,
    encoger_gap,
    estimar_mapa_tramo,
    filtrar_rebotes_sin_partida,
    prob_doble,
    recargar_banco,
    recorrer_adelante,
    suavizar_en_grilla,
    suavizar_lobby_jugador,
)


def _partida(mid, t, azul, rojo, gana_azul):
    filas = []
    for pid in azul:
        filas.append({"match_id": mid, "game_creation": t, "player_id": pid, "team_id": 100,
                      "win": int(gana_azul), "pos": "MIDDLE"})
    for pid in rojo:
        filas.append({"match_id": mid, "game_creation": t, "player_id": pid, "team_id": 200,
                      "win": int(not gana_azul), "pos": "MIDDLE"})
    return filas


def test_encoger_gap_y_delta_monotona():
    p = ParametrosPaso(c_lobby=4.0)
    assert encoger_gap(np.nan, 400.0, 5.0, p) == 0.0
    assert encoger_gap(600.0, 400.0, 0.0, p) == 0.0
    assert encoger_gap(600.0, 400.0, 4.0, p) == pytest.approx(100.0)
    assert encoger_gap(600.0, 400.0, 36.0, p) == pytest.approx(180.0)
    # δ(gap) debe ser no decreciente y acotada
    gaps = np.array([-800.0, -200.0, 0.0, 200.0, 800.0])
    ds = delta_de(gaps, p)
    assert np.all(np.diff(ds) >= -1e-12)
    assert ds[0] == p.delta_nodos[0] and ds[-1] == p.delta_nodos[-1]


def test_decaer_dias_y_banco_master_y_diamond():
    # Master 500 con banco inicial 14: a los 14 días no pierde nada; a los 16 pierde 150 -> 375 (piso D1 75)
    z, b = avanzar_banco(500.0, 14.0, 14.0)
    assert z == pytest.approx(500.0) and b == pytest.approx(0.0)
    z2, b2 = avanzar_banco(z, b, 2.0)
    assert z2 == pytest.approx(375.0) and b2 == pytest.approx(0.0)
    # Si sigue otro día debajo de 400 consume decay de Diamond hasta piso 275
    assert decaer_dias(375.0, 3) == pytest.approx(275.0)
    assert decaer_dias(275.0, 10) == pytest.approx(275.0)
    # Recarga: 1 día en Master+ (tope 14), 7 en Diamond (tope 28)
    assert recargar_banco(450.0, 13.5) == pytest.approx(14.0)
    assert recargar_banco(350.0, 25.0) == pytest.approx(28.0)


def test_filtrar_rebotes_sin_partida_quita_atrasos_y_conserva_decay_real():
    ev = pd.DataFrame([
        # Jugador a: decay real -75
        {"id": 1, "puuid": "a", "t_despues": 10, "tipo": "sin_partida", "z_antes": 500.0, "z_despues": 425.0,
         "wins_antes": 10, "losses_antes": 10, "wins_despues": 10, "losses_despues": 10},
        # Jugador b: rebote -75 / +75 con mismo W+L
        {"id": 2, "puuid": "b", "t_despues": 10, "tipo": "sin_partida", "z_antes": 500.0, "z_despues": 425.0,
         "wins_antes": 20, "losses_antes": 20, "wins_despues": 20, "losses_despues": 20},
        {"id": 3, "puuid": "b", "t_despues": 12, "tipo": "sin_partida", "z_antes": 425.0, "z_despues": 500.0,
         "wins_antes": 20, "losses_antes": 20, "wins_despues": 20, "losses_despues": 20},
    ])
    limpio = filtrar_rebotes_sin_partida(ev)
    assert limpio["id"].tolist() == [1]


def test_ajustar_parametros_las_y_k_diamond():
    # Recupera K_diamond sintético
    ev_f = pd.DataFrame({
        "z1": [100.0, 150.0, 200.0],
        "z2": [100.0 + 18.0 * 10, 150.0 - 18.0 * 5, 200.0 + 18.0 * 2],
        "dW": [20, 10, 12],
        "dL": [10, 15, 10],
        "dN": [30, 25, 22],
    })
    assert ajustar_k_diamond(ev_f) == pytest.approx(18.0)

    # Eventos LAS sintéticos: dobles en TOP raro, perdón en derrota 0, piso Master 0 y δ(gap)
    filas = []
    for i in range(40):
        # 4 jugadores activos con gap positivo (+200) ganan +25 y pierden -15 (δ = +5)
        pu = f"p{i // 10}"
        es_w = i % 2 == 0
        filas.append({
            "puuid": pu, "z_antes": 500.0, "z_despues": 525.0 if es_w else 485.0,
            "dz": 25.0 if es_w else -15.0, "es_win": es_w, "team_position": "MIDDLE",
            "lobby": 700.0, "n_lob": 8,
        })
    for i in range(10):
        # Jugador con gap negativo (-200) gana +18 y pierde -22 (δ = -2)
        es_w = i % 2 == 0
        filas.append({
            "puuid": "p_neg", "z_antes": 500.0, "z_despues": 518.0 if es_w else 478.0,
            "dz": 18.0 if es_w else -22.0, "es_win": es_w, "team_position": "MIDDLE",
            "lobby": 300.0, "n_lob": 8,
        })
    # Derrota perdonada fuera de 400 (1 de 21) y doble en TOP raro
    filas.append({"puuid": "p4", "z_antes": 500.0, "z_despues": 500.0, "dz": 0.0, "es_win": False,
                  "team_position": "MIDDLE", "lobby": 500.0, "n_lob": 8})
    filas.append({"puuid": "p4", "z_antes": 400.0, "z_despues": 400.0, "dz": 0.0, "es_win": False,
                  "team_position": "MIDDLE", "lobby": 400.0, "n_lob": 8})
    filas.append({"puuid": "p4", "z_antes": 400.0, "z_despues": 350.0, "dz": -50.0, "es_win": False,
                  "team_position": "MIDDLE", "lobby": 400.0, "n_lob": 8})
    filas.append({"puuid": "p0", "z_antes": 500.0, "z_despues": 540.0, "dz": 40.0, "es_win": True,
                  "team_position": "TOP", "lobby": 500.0, "n_lob": 8})
    params = ajustar_parametros_las(pd.DataFrame(filas), k_diamond=18.0, min_eventos_delta=5)
    assert params.k_diamond == 18.0
    assert params.piso_quieto == pytest.approx(0.5)
    assert params.piso_baja == pytest.approx(50.0)
    assert delta_de(200.0, params) == pytest.approx(5.0, abs=0.5)
    assert prob_doble(["TOP"], [0.01], params)[0] > 0.0


def test_recorrer_adelante_piso_decay_y_saldo():
    p = ParametrosPaso(perdon=0.0, piso_quieto=1.0, doble_base=0.0, delta_nodos=(0.0, 0.0), gap_nodos=(-100.0, 100.0))
    # Desde 405 pierde (queda clavado en Master 0 = 400), luego 15 días sin jugar -> decae 1 día a 375
    t0 = 0
    ts = [1 * DIA_MS]
    zf, zpre = recorrer_adelante([False], ts, [np.nan], [0.0], ["MIDDLE"], [1.0], 405.0, t0, 16 * DIA_MS, p)
    assert zpre[0] == pytest.approx(405.0)
    assert zf == pytest.approx(375.0)

    # Modo con saldo W/L conocido: 1 vista (W) + 2 ocultas (1W, 1L) desde 500 en 2 días -> 520
    zf2, _ = recorrer_adelante([True], [1 * DIA_MS], [np.nan], [0.0], ["MIDDLE"], [1.0],
                               500.0, 0, 2 * DIA_MS, p, dW_total=2, dL_total=1)
    assert zf2 == pytest.approx(520.0)


def test_operadores_grilla_son_adjuntos_exactos():
    rng = np.random.default_rng(1)
    a = rng.uniform(0.0, 1.0, 81)
    b = rng.uniform(0.0, 1.0, 81)
    dest = np.linspace(-100.0, 900.0, 81) + rng.normal(0.0, 15.0, 81)
    lhs = float(np.dot(_pushforward_lineal(a, dest, 0.0, 10.0), b))
    rhs = float(np.dot(a, _pullback_lineal(b, dest, 0.0, 10.0)))
    assert lhs == pytest.approx(rhs, rel=1e-10)


def test_suavizar_en_grilla_puente_causal_y_casos_extremos():
    p = ParametrosPaso(perdon=0.0, piso_quieto=0.5, piso_baja=50.0, doble_base=0.0,
                       gap_nodos=(-500.0, 500.0), delta_nodos=(-2.0, 8.0), sigma_paso=2.0)
    ts = [1 * DIA_MS, 2 * DIA_MS]
    lob = [np.nan, np.nan]
    nl = [0.0, 0.0]
    pos = ["MIDDLE", "MIDDLE"]
    fr = [1.0, 1.0]

    # Puente exacto 500 -> +20 -> +20 -> 540
    res = suavizar_en_grilla([True, True], ts, lob, nl, pos, fr, 500.0, 0.0, 540.0, 0.0, 0, 3 * DIA_MS, p, step=10.0)
    assert res["z_media"] == pytest.approx([500.0, 520.0], abs=3.0)
    assert res["z_fin"] == pytest.approx(540.0)

    # Escalador imposible con K=20 puro (500 -> 1W 1L -> 700): no produce NaN y respeta p10 <= mediana <= p90
    esc = suavizar_en_grilla([True, False], ts, lob, nl, pos, fr, 500.0, 0.0, 700.0, 0.0, 0, 3 * DIA_MS, p, step=10.0)
    assert np.all(np.isfinite(esc["z_media"]))
    assert np.all(esc["z_p10"] <= esc["z_mediana"]) and np.all(esc["z_mediana"] <= esc["z_p90"])

    # Invariancia causal (z2=None): cambiar resultado actual o agregar lobbies futuros no cambia las estimaciones anteriores
    c1 = suavizar_en_grilla([True, False], ts, [500.0, 900.0], [9.0, 9.0], pos, fr, 500.0, 0.0, None, 0.0, 0, 3 * DIA_MS, p, tau_lobby=0.5, step=10.0)
    c2 = suavizar_en_grilla([False, False, True, True], [1 * DIA_MS, 2 * DIA_MS, 3 * DIA_MS, 4 * DIA_MS],
                            [500.0, 900.0, 1100.0, 1200.0], [9.0] * 4, pos * 2, fr * 2, 500.0, 0.0, None, 0.0, 0, 5 * DIA_MS, p, tau_lobby=0.5, step=10.0)
    assert c2["z_media"][0] == pytest.approx(c1["z_media"][0], abs=1e-9)
    assert c2["z_media"][1] != pytest.approx(c1["z_media"][1], abs=1e-3)


def test_suavizar_lobby_jugador_causal_no_mira_futuro():
    lob = [400.0, 800.0, 1000.0]
    nl = [9.0, 9.0, 9.0]
    m_c, n_c = suavizar_lobby_jugador(lob, nl, modo="causal", ventana=31)
    assert m_c[0] == pytest.approx(400.0) and n_c[0] == pytest.approx(1.0)
    assert m_c[1] == pytest.approx(600.0) and n_c[1] == pytest.approx(2.0)


def test_estimar_mapa_tramo_causal_vs_descriptivo():
    p = ParametrosPaso(perdon=0.0, doble_base=0.0, gap_nodos=(-500.0, 500.0), delta_nodos=(0.0, 0.0), sigma_paso=2.0)
    partes = pd.DataFrame(
        _partida("m1", 1 * DIA_MS, ["p1", "p2"], ["p3", "p4"], True)
        + _partida("m2", 2 * DIA_MS, ["p1", "p2"], ["p3", "p4"], True)
    )
    f1 = pd.DataFrame({"z": [500.0, 500.0, 500.0], "wins": [10, 10, 10], "losses": [10, 10, 10]},
                      index=["p1", "p2", "p3"])
    f2_a = pd.DataFrame({"z": [540.0, 540.0], "wins": [12, 12], "losses": [10, 10]}, index=["p1", "p3"])
    f2_b = pd.DataFrame({"z": [900.0, 100.0], "wins": [50, 10], "losses": [10, 50]}, index=["p1", "p3"])

    # Modo causal ignora por completo foto2
    c_a = estimar_mapa_tramo(partes, f1, f2_a, 0, 3 * DIA_MS, p, modo="causal")
    c_b = estimar_mapa_tramo(partes, f1, f2_b, 0, 3 * DIA_MS, p, modo="causal")
    pd.testing.assert_frame_equal(c_a, c_b)
    assert set(c_a["metodo"].unique()) == {"causal_desde_foto1", "causal_lobby"}

    # Modo causal rechaza foto1 si hay partidas anteriores a t1
    with pytest.raises(ValueError, match="fuga"):
        estimar_mapa_tramo(partes, f1, None, 2 * DIA_MS, 3 * DIA_MS, p, modo="causal")

    # Modo descriptivo usa ambos extremos y asigna el método correcto a cada tramo
    anclas = pd.Series({"p4": 500.0})
    desc = estimar_mapa_tramo(partes, f1, f2_a, 0, 3 * DIA_MS, p, anclas_z1=anclas, modo="descriptivo")
    assert desc.loc[partes["player_id"] == "p1", "metodo"].iloc[0] == "puente_grilla"
    assert desc.loc[partes["player_id"] == "p2", "metodo"].iloc[0] == "adelante_foto1"
    assert desc.loc[partes["player_id"] == "p4", "metodo"].iloc[0] == "ancla_principal_grilla"
    p1_vals = desc.loc[partes["player_id"] == "p1", "z_media"].to_numpy()
    assert p1_vals[0] == pytest.approx(500.0, abs=4.0) and p1_vals[1] == pytest.approx(520.0, abs=4.0)

    # Modo causal es invariante a agregar una partida futura m3
    partes_ext = pd.concat([partes, pd.DataFrame(_partida("m3", 3 * DIA_MS, ["p1", "p2"], ["p3", "p4"], False))], ignore_index=True)
    c_ext = estimar_mapa_tramo(partes_ext, f1, None, 0, 4 * DIA_MS, p, modo="causal")
    pd.testing.assert_frame_equal(c_a.loc[partes["match_id"] == "m1"], c_ext.loc[partes_ext["match_id"] == "m1"])

    # jugadores_objetivo devuelve el mismo resultado para los seleccionados y marca omitidos al resto
    sub = estimar_mapa_tramo(partes, f1, f2_a, 0, 3 * DIA_MS, p, anclas_z1=anclas,
                             modo="descriptivo", n_pasadas=2, jugadores_objetivo=["p1"])
    pd.testing.assert_frame_equal(sub.loc[partes["player_id"] == "p1"], desc.loc[partes["player_id"] == "p1"])
    assert set(sub.loc[partes["player_id"] != "p1", "metodo"].unique()) == {"omitido"}


def test_causal_lobby_sin_fuga_hacia_atras():
    """En modo causal un jugador sin foto no hereda el lobby de partidas futuras por bfill."""
    f1 = pd.DataFrame(
        {"z": [600.0, 600.0, 600.0], "wins": [10, 10, 10], "losses": [10, 10, 10]},
        index=["p1", "p2", "p3"],
    )
    filas = (
        _partida("m0", 1 * DIA_MS, ["p4", "p5"], ["p6", "p7"], True)
        + _partida("m1", 2 * DIA_MS, ["p4", "p1"], ["p2", "p3"], True)
    )
    partes = pd.DataFrame(filas)
    p = ParametrosPaso()

    solo_m0 = estimar_mapa_tramo(
        partes.loc[partes["match_id"] == "m0"], f1, None, 0, 3 * DIA_MS, p, modo="causal"
    )
    con_m1 = estimar_mapa_tramo(partes, f1, None, 0, 3 * DIA_MS, p, modo="causal")

    fila_m0 = con_m1.loc[(partes["match_id"] == "m0") & (partes["player_id"] == "p4")].iloc[0]
    fila_m1 = con_m1.loc[(partes["match_id"] == "m1") & (partes["player_id"] == "p4")].iloc[0]
    assert pd.isna(fila_m0["z_media"])
    assert fila_m0["metodo"] == "sin_dato"
    assert fila_m1["metodo"] == "causal_lobby"
    assert np.isfinite(fila_m1["z_media"])
    pd.testing.assert_frame_equal(solo_m0, con_m1.loc[partes["match_id"] == "m0"])

