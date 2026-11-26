"""Residuo por pliegue del 2-fold de `huella_patrones_cuenta.py` (PROJECT_SUMMARY §9.S.77 punto 5, §10.7)."""

import numpy as np
import pandas as pd
import pytest

from scripts.analysis import huella_mejoras as hm
from scripts.analysis import huella_patrones_cuenta as hpc

FEATS = hpc.FEATS
BY = ["champion_name", "lane", "win"]


def _crudo(n_players=40, n_games=60, seed=0):
    """Partidas sintéticas con las 41 columnas: 2 campeones, una línea, gana y pierde alternado."""
    rng = np.random.default_rng(seed)
    filas = []
    for p in range(n_players):
        estilo = rng.normal(0, 1, len(FEATS))
        for g in range(n_games):
            x = estilo + rng.normal(0, 1, len(FEATS))
            filas.append({"player_id": f"p{p:02d}", "champion_name": ("Ahri", "Zed")[g % 2], "lane": "MIDDLE",
                          "win": (g // 2) % 2, "game_creation": g, **dict(zip(FEATS, x))})
    return pd.DataFrame(filas)


def _por_jugador(d, pids):
    return d[d["player_id"].isin(pids)].sort_values(["player_id", "game_creation"]).reset_index(drop=True)


def _sin_fuga(residuo):
    """Cambiar las partidas de un jugador analizado no mueve nada de lo que su pliegue usa para los demás.

    Para cada pliegue: ni las filas con que se entrena su red ni las filas con que proyecta a los otros jugadores
    analizados pueden depender de un jugador que ese pliegue proyecta (es lo que la red "nunca vio").
    """
    crudo = _crudo()
    df1, pl1 = hpc.datos_2fold(residuo, crudo=crudo)
    for i, (_, entrena, proyecta) in enumerate(pl1):
        q = sorted(proyecta)[0]
        crudo2 = crudo.copy()
        crudo2.loc[crudo2["player_id"] == q, hm.STYLE_FEATURES] += 100.0
        df2, pl2 = hpc.datos_2fold(residuo, crudo=crudo2)
        assert pl2[i][1] == entrena and pl2[i][2] == proyecta
        pd.testing.assert_frame_equal(_por_jugador(pl1[i][0], entrena), _por_jugador(pl2[i][0], entrena))
        otros = proyecta - {q}
        pd.testing.assert_frame_equal(_por_jugador(df1, otros), _por_jugador(df2, otros))


def test_residuo_por_pliegue_no_filtra_a_los_jugadores_analizados():
    _sin_fuga("pliegue")


def test_residuo_global_si_filtra():
    # la versión anterior normalizaba con todos: el mismo cambio se filtra (es la fuga que corrige el pliegue)
    with pytest.raises(AssertionError):
        _sin_fuga("global")


def test_datos_2fold_cada_jugador_con_la_normalizacion_de_su_pliegue():
    crudo = _crudo()
    df, pliegues = hpc.datos_2fold("pliegue", crudo=crudo)
    (r1, ajuste, evalu), (r2, entrena2, proyecta2) = pliegues
    assert entrena2 == evalu and proyecta2 == ajuste and not (ajuste & evalu)
    assert ajuste | evalu == set(crudo["player_id"]) == set(df["player_id"])
    assert len(df) == len(crudo)                       # todos los grupos llegan a 30 con cada mitad

    # un jugador de evaluación se analiza con media y desvío de los de ajuste, y viceversa
    for entrena, proyecta in ((ajuste, evalu), (evalu, ajuste)):
        fit = crudo[crudo["player_id"].isin(entrena)]
        mu, sd = fit.groupby(BY)[FEATS].mean(), fit.groupby(BY)[FEATS].std()
        p = sorted(proyecta)[0]
        fila = crudo[crudo["player_id"] == p].sort_values("game_creation").iloc[0]
        got = df[(df["player_id"] == p) & (df["game_creation"] == fila["game_creation"])].iloc[0]
        clave = tuple(fila[BY])
        assert np.isclose(got["kills_pm"], (fila["kills_pm"] - mu.loc[clave, "kills_pm"]) / sd.loc[clave, "kills_pm"])

    # con "global" los dos pliegues comparten una normalización y la partición es la misma
    dg, pg = hpc.datos_2fold("global", crudo=crudo)
    assert pg[0][1] == ajuste and pg[0][2] == evalu and pg[0][0] is pg[1][0]
    with pytest.raises(ValueError):
        hpc.datos_2fold("otro", crudo=crudo)


def test_redes_2fold_proyecta_a_todos_con_la_red_del_otro_pliegue():
    pytest.importorskip("torch")
    crudo = _crudo(n_players=20, n_games=50)
    df, emb = hpc.redes_2fold("pliegue", pasos=3, crudo=crudo, cache=False)
    assert set(emb) == set(df["player_id"])
    ajuste, evalu = hm.partir_jugadores(crudo["player_id"].unique())
    f_eval = {id(emb[p]) for p in evalu}
    f_ajus = {id(emb[p]) for p in ajuste}
    assert len(f_eval) == 1 and len(f_ajus) == 1 and f_eval != f_ajus
    E = emb[sorted(evalu)[0]](df[FEATS].to_numpy(np.float32)[:4])
    assert E.shape == (4, 32) and np.allclose(np.linalg.norm(E, axis=1), 1, atol=1e-5)


def test_ic_auxiliares():
    lo, hi = hpc.ic_wilson(8, 5843)
    assert 0 < lo < 8 / 5843 < hi < 0.01
    m, lo, hi = hpc.ic_media(np.arange(100.0))
    assert lo < m == 49.5 < hi
    r, lo, hi = hpc.ic_corr(np.arange(50.0), np.arange(50.0) + np.random.default_rng(0).normal(0, 5, 50))
    assert lo < r < hi <= 1
