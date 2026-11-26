"""Residuo de la huella con hábitos sin fuga (`huella_niveles.datos_timeline`, PROJECT_SUMMARY §9.S.77)."""

import numpy as np
import pandas as pd

from scripts.analysis import huella_mejoras as hm
from scripts.analysis import huella_niveles as hn

FEATS = hm.STYLE_FEATURES + hm.HABITOS


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
    df = pd.DataFrame(filas)
    df["l1_q"] = 1.0                                     # hábito constante: debe valer 0, no tirar la fila
    return df


def test_datos_timeline_normaliza_solo_con_los_jugadores_de_ajuste():
    crudo = _crudo()
    resid, ajuste, evalu = hn.datos_timeline("ajuste", crudo=crudo)
    assert ajuste and evalu and not (ajuste & evalu)
    assert ajuste | evalu == set(crudo["player_id"])
    assert set(resid["player_id"]) == set(crudo["player_id"])       # los de evaluación siguen, normalizados
    assert (resid["l1_q"] == 0).all()

    # media y desvío de cada grupo salen solo de las filas de ajuste
    by = ["champion_name", "lane", "win"]
    fit = crudo[crudo["player_id"].isin(ajuste)]
    mu, sd = fit.groupby(by)[FEATS].mean(), fit.groupby(by)[FEATS].std()
    p = sorted(evalu)[0]
    fila = crudo[crudo["player_id"] == p].iloc[0]
    clave = tuple(fila[by])
    f = "kills_pm"
    assert np.isclose(resid.loc[fila.name, f], (fila[f] - mu.loc[clave, f]) / sd.loc[clave, f])

    # cambiar las partidas de un jugador de evaluación no mueve el residuo de los de ajuste
    crudo2 = crudo.copy()
    crudo2.loc[crudo2["player_id"] == p, hm.STYLE_FEATURES] += 100.0
    resid2, ajuste2, _ = hn.datos_timeline("ajuste", crudo=crudo2)
    assert ajuste2 == ajuste
    en = resid["player_id"].isin(ajuste)
    pd.testing.assert_frame_equal(resid[en], resid2[resid2["player_id"].isin(ajuste)])

    # con "global" (versión anterior) el mismo cambio sí se filtra a los de ajuste: es la fuga corregida
    g1, _, _ = hm.cargar_timeline("global", crudo=crudo)
    g2, _, _ = hm.cargar_timeline("global", crudo=crudo2)
    assert not np.allclose(g1.loc[g1["player_id"].isin(ajuste), f], g2.loc[g2["player_id"].isin(ajuste), f])


def test_tamano_minimo_del_grupo_se_cuenta_en_el_ajuste():
    crudo = _crudo(n_players=40, n_games=8)          # 2 grupos por jugador * 2 partidas; ~20 de ajuste => ~40
    _, ajuste, _ = hm.cargar_timeline("ajuste", crudo=crudo)
    n_fit = crudo[crudo["player_id"].isin(ajuste)].groupby(["champion_name", "lane", "win"]).size()
    n_todos = crudo.groupby(["champion_name", "lane", "win"]).size()
    umbral = int(n_fit.max()) + 1
    assert (n_todos >= umbral).all()                 # con todos alcanzaría, con el ajuste no
    assert hm.residuo_habitos(crudo, FEATS, ajuste=ajuste, min_group=umbral).empty
    assert len(hm.residuo_habitos(crudo, FEATS, min_group=umbral)) == len(crudo)
