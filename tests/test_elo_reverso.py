import numpy as np
import pandas as pd
import pytest

from src.features.elo_reverso import (adelante, ajustar_regla, al_reves, calibrar_lobby, calibrar_lobby_por_grupo,
                                      camino, decaimiento,
                                      decay_total, division_de, grupo_de,
                                      folds_por_jugador, kalman_al_reves, kalman_jugadores, lobby_por_fila, sin_deriva,
                                      rango_de)

CORTE = (600.0, 900.0)   # Grandmaster desde 600, Challenger desde 900


def _partida(mid, t, azul, rojo, gana_azul):
    filas = []
    for pid in azul:
        filas.append({"match_id": mid, "game_creation": t, "player_id": pid, "team_id": 100, "win": int(gana_azul)})
    for pid in rojo:
        filas.append({"match_id": mid, "game_creation": t, "player_id": pid, "team_id": 200, "win": int(not gana_azul)})
    return filas


def test_grupo_rango_division():
    assert [grupo_de(z, CORTE) for z in (950, 700, 450, 120, -50)] == [0, 1, 2, 3, 3]
    assert list(rango_de([950, 700, 450, 120, -50], CORTE)) == ["CHALLENGER", "GRANDMASTER", "MASTER",
                                                                "DIAMOND", "DEBAJO"]
    assert list(division_de([120, 399, 450], CORTE)) == ["1", "3", "MASTER"]


def test_ajustar_regla_recupera_a_b():
    rng = np.random.default_rng(0)
    filas = []
    for t, (a, b) in zip(["CHALLENGER", "GRANDMASTER", "MASTER", "DIAMOND"], [(25, 24), (23, 22), (20, 19), (16, 16)]):
        dw, dl = rng.integers(10, 80, 200), rng.integers(10, 80, 200)
        z1 = rng.uniform(0, 300, 200)
        filas.append(pd.DataFrame({"tier2": t, "dW": dw, "dL": dl, "dN": dw + dl, "z1": z1,
                                   "z2": z1 + a * dw - b * dl}))
    a, b = ajustar_regla(pd.concat(filas, ignore_index=True))
    assert a[0] == pytest.approx(25) and b[0] == pytest.approx(24)
    assert a[3] == pytest.approx(16) and b[3] == pytest.approx(16)


def test_al_reves_regla_deshace_por_rango():
    # p1 gana dos partidas siendo Diamond: hacia atrás se le restan 2a de Diamond.
    partes = pd.DataFrame(_partida("m1", 1, ["p1"], ["p2"], True) + _partida("m2", 2, ["p1"], ["p2"], True))
    a = {0: 30.0, 1: 25.0, 2: 20.0, 3: 10.0}
    b = {0: 29.0, 1: 24.0, 2: 19.0, 3: 9.0}
    z = pd.Series({"p1": 200.0, "p2": 410.0})
    out = al_reves(partes, z, a, b, CORTE)
    assert out["p1"] == pytest.approx(180.0)
    # p2 pierde siendo Master (410) → 429; ya es Master, pierde otra vez → 448.
    assert out["p2"] == pytest.approx(448.0)


def test_al_reves_usa_el_rango_del_paso():
    # Cruza el corte de Master al deshacer: la segunda derrota usa b de Master.
    partes = pd.DataFrame(_partida("m1", 1, ["p1"], ["p2"], False) + _partida("m2", 2, ["p1"], ["p2"], False))
    a = {0: 30.0, 1: 25.0, 2: 20.0, 3: 10.0}
    b = {0: 29.0, 1: 24.0, 2: 19.0, 3: 9.0}
    out = al_reves(partes, pd.Series({"p1": 395.0, "p2": 300.0}), a, b, CORTE)
    assert out["p1"] == pytest.approx(395.0 + 9.0 + 19.0)


def test_al_reves_relleno_sin_foto():
    partes = pd.DataFrame(_partida("m1", 1, ["p1", "p3"], ["p2", "p4"], True))
    a = {k: 10.0 for k in range(4)}
    b = {k: 10.0 for k in range(4)}
    out = al_reves(partes, pd.Series({"p1": 100.0, "p2": 300.0}), a, b, CORTE)
    assert out["p3"] == pytest.approx(200.0 - 10.0)   # arranca en la media de los que tienen foto
    assert out["p4"] == pytest.approx(200.0 + 10.0)


def test_al_reves_elo_con_beta_grande_es_la_regla():
    partes = pd.DataFrame(_partida("m1", 1, ["p1"], ["p2"], True))
    a = {k: 10.0 for k in range(4)}
    b = {k: 10.0 for k in range(4)}
    z = pd.Series({"p1": 100.0, "p2": 300.0})
    regla = al_reves(partes, z, a, b, CORTE)
    elo = al_reves(partes, z, a, b, CORTE, modo="elo", beta=1e9)
    assert elo["p1"] == pytest.approx(regla["p1"])
    # Con β chico, el no favorito que gana "sorprende": se le resta más.
    assert al_reves(partes, z, a, b, CORTE, modo="elo", beta=50)["p1"] < regla["p1"]


def test_lobby_por_fila():
    partes = pd.DataFrame(_partida("m1", 1, ["p1", "p2"], ["p3", "p4"], True))
    z = pd.Series({"p1": 100.0, "p2": 200.0, "p3": 400.0})
    lob = lobby_por_fila(partes, z)
    fila = lob.iloc[0]   # p1
    assert fila["aliados"] == pytest.approx(200.0)
    assert fila["rivales"] == pytest.approx(400.0)
    assert fila["lobby"] == pytest.approx(300.0)
    assert fila["conocidos"] == 2
    assert lob.iloc[3]["aliados"] == pytest.approx(400.0)


def test_camino_anclado_en_las_dos_fotos():
    a = b = {k: 20.0 for k in range(4)}
    p = pd.DataFrame({"game_creation": [3, 1, 2], "win": [1, 1, 0]})
    assert camino(p, 500, 480, a, b, CORTE) == pytest.approx([480, 500, 480, 500])   # la regla sola
    assert camino(p, 500, 450, a, b, CORTE) == pytest.approx([450, 480, 470, 500])   # residuo repartido
    assert camino(p.iloc[:0], 500, 450, a, b, CORTE) == pytest.approx([500])


def test_sin_deriva_es_suma_cero():
    a, b = sin_deriva({0: 25.0, 3: 16.0}, {0: 23.0, 3: 16.0})
    assert a == b == {0: 24.0, 3: 16.0}


def test_kalman_sin_mediciones_es_la_regla():
    a = b = {k: 20.0 for k in range(4)}
    win, sin_lobby = np.array([1, 0, 1]), np.full(3, np.nan)   # de la más nueva a la más vieja
    m, v = kalman_al_reves(win, sin_lobby, 500, a, b, CORTE, (0.0, 1.0, 100.0), q=25)
    assert m == pytest.approx([480, 500, 480, 500])
    assert v == pytest.approx([75, 50, 25, 0])
    # Partidas no observadas: con winrate 50 % y a = b no mueven la media, suman varianza.
    m2, v2 = kalman_al_reves(win, sin_lobby, 500, a, b, CORTE, (0.0, 1.0, 100.0), q=25, ocultas_por_partida=1.0)
    assert m2 == pytest.approx(m) and v2[0] == pytest.approx(75 + 3 * (0.25 * 40 ** 2 + 25))


def test_kalman_la_medicion_tira_hacia_el_lobby():
    a = b = {k: 20.0 for k in range(4)}
    win = np.array([1, 1])
    m, v = kalman_al_reves(win, np.array([np.nan, 300.0]), 500, a, b, CORTE, (0.0, 1.0, 50.0), q=25)
    assert 300 < m[0] < 460 and v[0] < 50 and m[-1] == 500 and v[-1] == 0


def test_calibrar_lobby():
    z = pd.Series([0.0, 100, 200, 300], index=list("abcd"))
    c0, c1, R = calibrar_lobby(pd.Series([50.0, 130, 210, np.nan], index=list("abcd")), z)
    assert (c0, c1, R) == pytest.approx((50, 0.8, 0), abs=1e-6)


def test_al_reves_descarta_no_observadas_en_proporcion():
    # 1:6 → cada partida observada que se deshace arrastra 6 no observadas, con el
    # cambio esperado según el winrate total: 6 · (0,75 · 20 − 0,25 · 20) = 60 LP.
    partes = pd.DataFrame(_partida("m1", 1, ["p1"], ["p2"], True))
    a = b = {k: 20.0 for k in range(4)}
    z = pd.Series({"p1": 300.0, "p2": 300.0})
    out = al_reves(partes, z, a, b, CORTE, no_observadas=pd.Series({"p1": 6.0, "p2": 0.0}),
                   wr=pd.Series({"p1": 0.75}))
    assert out["p1"] == pytest.approx(300 - 20 - 60)
    assert out["p2"] == pytest.approx(320)   # sin no observadas y winrate 0,5 por defecto
    p = pd.DataFrame({"game_creation": [1], "win": [1]})
    assert camino(p, 300, 220, a, b, CORTE, no_observadas=6.0, wr=0.75) == pytest.approx([220, 300])
    assert camino(p, 300, 0, a, b, CORTE, no_observadas={k: 6.0 for k in range(4)}, wr=0.75)[0] == pytest.approx(0)


def test_adelante_desde_la_foto_1():
    a = b = {k: 20.0 for k in range(4)}
    p = pd.DataFrame({"game_creation": [2, 1, 3], "win": [0, 1, 1]})   # en orden: gana, pierde, gana
    assert adelante(p, 300, a, b, CORTE) == pytest.approx([300, 320, 300, 320])
    assert adelante(p, 300, a, b, CORTE, no_observadas=1.0, wr=0.75)[1] == pytest.approx(300 + 20 + 10)


def test_al_reves_ajusta_a_b_por_winrate_y_lobby():
    # δ = k_wr (wr − 0,5) + k_lobby (lobby − z_foto)/100: gana más y pierde menos quien
    # tiene el MMR (winrate, lobby) por encima de su LP.
    partes = pd.DataFrame(_partida("m1", 1, ["p1"], ["p2"], True))
    a = b = {k: 20.0 for k in range(4)}
    z = pd.Series({"p1": 300.0, "p2": 300.0})
    wr = pd.Series({"p1": 0.75, "p2": 0.75})
    out = al_reves(partes, z, a, b, CORTE, wr=wr, k_wr=20)
    assert out["p1"] == pytest.approx(300 - 25) and out["p2"] == pytest.approx(300 + 15)
    lobby = pd.Series([500.0, 100.0], index=partes.index)   # p1 juega con gente de 500, p2 con gente de 100
    out = al_reves(partes, z, a, b, CORTE, lobby=lobby, k_lobby=1)
    assert out["p1"] == pytest.approx(300 - 22) and out["p2"] == pytest.approx(300 + 22)


def test_decaimiento_y_decay_total():
    dia = 86_400_000
    assert decaimiento(14 * dia) == 0 and decaimiento(20 * dia) == 450 and decaimiento(20.9 * dia) == 450
    partes = pd.DataFrame({"player_id": ["p", "p"], "game_creation": [2 * dia, 30 * dia]})
    # foto 1 en 0, foto 2 en 50 días: huecos de 2, 28 y 20 días → 0 + 14·75 + 6·75.
    assert decay_total(partes, 0, 50 * dia)["p"] == pytest.approx(75 * (14 + 6))


def test_al_reves_devuelve_el_decay_de_master():
    dia = 86_400_000
    a = b = {k: 20.0 for k in range(4)}
    partes = pd.DataFrame(_partida("m1", 10 * dia, ["p1"], ["p2"], True))
    z = pd.Series({"p1": 500.0, "p2": 200.0})   # p1 Master, p2 Diamond (no decae)
    out = al_reves(partes, z, a, b, (2000.0, 3000.0), fotos=(9 * dia, 30 * dia))
    # 20 días sin jugar hasta la foto 2: p1 tenía 500 + 450 antes del hueco; deshace la victoria.
    assert out["p1"] == pytest.approx(500 + 450 - 20)
    assert out["p2"] == pytest.approx(200 + 20)
    p = pd.DataFrame({"game_creation": [10 * dia], "win": [1]})
    assert camino(p, 500, 930, a, b, (2000.0, 3000.0), fotos=(9 * dia, 30 * dia)) == pytest.approx([930, 500])
    m, _ = kalman_al_reves(np.array([1]), np.array([np.nan]), 500, a, b, (2000.0, 3000.0), (0.0, 1.0, 100.0),
                           tiempos=np.array([10 * dia]), fotos=(9 * dia, 30 * dia))
    assert m[0] == pytest.approx(930)


def test_al_reves_devuelve_el_lp_de_cada_partida():
    # p1 gana m2 (la más nueva) y pierde m1: antes de m2 tenía 500 − 20, antes de m1 480 + 20.
    partes = pd.DataFrame(_partida("m1", 1, ["p1"], ["p2"], False) + _partida("m2", 2, ["p1"], ["p2"], True))
    a = b = {k: 20.0 for k in range(4)}
    fin, lp = al_reves(partes, pd.Series({"p1": 500.0, "p2": 500.0}), a, b, (2000.0, 3000.0), devolver_camino=True)
    assert fin["p1"] == pytest.approx(500)
    assert list(lp.index) == list(partes.index)
    assert lp[(partes["match_id"] == "m2") & (partes["player_id"] == "p1")].item() == pytest.approx(480)
    assert lp[(partes["match_id"] == "m1") & (partes["player_id"] == "p1")].item() == pytest.approx(500)


def test_lobby_por_fila_con_lp_por_partida():
    partes = pd.DataFrame(_partida("m1", 1, ["p1", "p2"], ["p3", "p4"], True))
    lp = pd.Series([100.0, 200.0, 300.0, 400.0], index=partes.index)   # LP de cada uno en esa partida
    out = lobby_por_fila(partes, lp, por_fila=True)
    assert out["aliados"].iloc[0] == pytest.approx(200) and out["rivales"].iloc[0] == pytest.approx(350)
    assert out["lobby"].iloc[0] == pytest.approx(300)


def test_calibrar_lobby_por_grupo_y_kalman_con_dict():
    z = pd.Series([100.0, 200, 300, 700, 800, 900], index=list("abcdef"))
    lobby = pd.Series([150.0, 250, 350, 600, 650, 700], index=list("abcdef"))
    med = calibrar_lobby_por_grupo(pd.concat([lobby] * 20), z, (650.0, 850.0))
    assert set(med) == {0, 1, 2, 3}
    a = b = {k: 20.0 for k in range(4)}
    m, v = kalman_al_reves(np.array([1]), np.array([np.nan]), 500, a, b, CORTE, med, q={k: 9.0 for k in range(4)})
    assert m == pytest.approx([480, 500]) and v == pytest.approx([9, 0])


def test_folds_por_jugador_fijos_y_repartidos():
    ids = [f"p{i}" for i in range(2000)]
    f = folds_por_jugador(ids)
    assert set(f) == {0, 1, 2, 3, 4}
    assert list(folds_por_jugador(ids[::-1])) == list(f[::-1])   # no depende del orden
    assert np.bincount(f).min() > 300


def test_kalman_jugadores_sin_partidas_queda_en_la_foto():
    partes = pd.DataFrame(_partida("m1", 1, ["p1", "p2"], ["p3", "p4"], True)
                          + _partida("m2", 2, ["p1", "p2"], ["p3", "p4"], False))
    z = pd.Series({"p1": 100.0, "p2": 150.0, "p3": 200.0, "p4": 250.0, "p5": 300.0})
    e = pd.DataFrame({"z2": z, "obs": [2, 2, 2, 2, 0], "obs_w": [1, 1, 1, 1, 0], "ocultas": 0.0})
    a = b = {k: 20.0 for k in range(4)}
    m, v = kalman_jugadores(e, partes, z, a, b, CORTE)
    assert m["p5"] == 300.0 and np.isnan(v["p5"])
    assert m["p1"] == pytest.approx(100.0, abs=10) and v["p1"] >= 0   # con 4 jugadores el lobby es exacto: R = 0
