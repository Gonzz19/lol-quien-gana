"""Huella de estilo, nivel 1 (src/features/fingerprint.py, PROJECT_SUMMARY.md §10)."""

import numpy as np
import pandas as pd

from src.features.fingerprint import (evaluate, hard_negative_mask, keep_min_tier, make_halves,
                                      player_variance_share, profiles, residualize, row_auc)

FEATS = ["f1", "f2", "f3"]
CHAMPS = ["Ahri", "Zed", "Lux", "Garen", "Jinx", "Thresh"]


def _games(n_players=60, n_games=60, style=1.0, champ_effect=3.0, own_champs=False, seed=0):
    """Partidas sintéticas: x = efecto del campeón + estilo propio * `style` + ruido.

    Con `own_champs`, cada jugador juega solo 2 campeones propios (su pool lo delata).
    """
    rng = np.random.default_rng(seed)
    ceff = {c: rng.normal(0, champ_effect, len(FEATS)) for c in CHAMPS}
    rows = []
    for p in range(n_players):
        s = rng.normal(0, 1, len(FEATS)) * style
        pool = [CHAMPS[p % 6], CHAMPS[(p + 1) % 6]] if own_champs else CHAMPS
        for g in range(n_games):
            c = pool[rng.integers(len(pool))]
            x = ceff[c] + s + rng.normal(0, 1, len(FEATS))
            rows.append({"player_id": f"p{p}", "champion_name": c, "lane": "MIDDLE", "win": g % 2,
                         "game_creation": g, **dict(zip(FEATS, x))})
    return pd.DataFrame(rows)


def _auc(df, mode, k=15):
    halves = make_halves(df, k=k, mode=mode, seed=1)
    _, A, B = profiles(df, halves, FEATS)
    return evaluate(A, B, n_perm=200, n_boot=50)


def test_con_huella_se_reconoce_al_jugador():
    df = residualize(_games(style=1.0), FEATS, min_group=10)
    res = _auc(df, "aleatoria")
    assert res["auc"] > 0.9 and res["p_valor"] < 0.01


def test_sin_huella_da_azar():
    df = residualize(_games(style=0.0), FEATS, min_group=10)
    res = _auc(df, "aleatoria")
    assert abs(res["auc"] - 0.5) < 0.06 and res["p_valor"] > 0.01


def test_residualizar_quita_el_efecto_del_campeon():
    # Sin estilo propio pero con pools distintos: crudo, el campeón "reconoce" al
    # jugador; residualizado, no queda nada.
    raw = _games(style=0.0, own_champs=True)
    assert _auc(raw, "aleatoria")["auc"] > 0.75
    res = residualize(raw, FEATS, min_group=10)
    assert abs(_auc(res, "aleatoria")["auc"] - 0.5) < 0.06


def test_residualizar_con_ajuste_usa_solo_esos_jugadores():
    raw = _games(style=1.0)
    ajuste = {f"p{p}" for p in range(30)}
    res = residualize(raw, FEATS, min_group=10, ajuste=ajuste)
    # los de evaluación siguen en la salida, normalizados con media y desvío del ajuste
    assert set(res["player_id"]) == set(raw["player_id"])
    fit = raw[raw["player_id"].isin(ajuste)]
    mu = fit.groupby(["champion_name", "lane", "win"])[FEATS].mean()
    sd = fit.groupby(["champion_name", "lane", "win"])[FEATS].std()
    fila = raw[raw["player_id"] == "p45"].iloc[0]
    clave = (fila["champion_name"], fila["lane"], fila["win"])
    esperado = (fila[FEATS].astype(float) - mu.loc[clave]) / sd.loc[clave]
    np.testing.assert_allclose(res.loc[fila.name, FEATS].astype(float), esperado.to_numpy(float))
    # cambiar las partidas de un jugador de evaluación no mueve a los de ajuste
    raw2 = raw.copy()
    raw2.loc[raw2["player_id"] == "p45", FEATS] += 100.0
    res2 = residualize(raw2, FEATS, min_group=10, ajuste=ajuste)
    en_ajuste = res["player_id"].isin(ajuste)
    pd.testing.assert_frame_equal(res[en_ajuste], res2[res2["player_id"].isin(ajuste)])
    # el tamaño mínimo se cuenta solo en el ajuste
    assert residualize(raw, FEATS, min_group=10_000, ajuste=ajuste).empty


def test_campeones_distintos_son_disjuntos_y_la_huella_sobrevive():
    df = residualize(_games(style=1.0), FEATS, min_group=10)
    halves = make_halves(df, k=15, mode="campeones_distintos", seed=3)
    assert halves
    for h in halves.values():
        assert set(df.loc[h["A"], "champion_name"]).isdisjoint(df.loc[h["B"], "champion_name"])
        assert len(h["A"]) == len(h["B"]) == 15
    assert _auc(df, "campeones_distintos")["auc"] > 0.85


def test_temporal_toma_las_mas_viejas_y_las_mas_nuevas():
    df = _games(n_players=2, n_games=40)
    halves = make_halves(df, k=10, mode="temporal")
    h = halves["p0"]
    assert df.loc[h["A"], "game_creation"].max() < df.loc[h["B"], "game_creation"].min()


def test_row_auc_y_mascara_de_rivales_dificiles():
    D = np.array([[0.0, 1.0, 2.0], [3.0, 1.0, 0.5], [1.0, 1.0, 1.0]])
    np.testing.assert_allclose(row_auc(D), [1.0, 0.5, 0.5])
    info = pd.DataFrame({"player_tier": ["MASTER", "MASTER", "DIAMOND"],
                         "main_lane": ["MIDDLE", "MIDDLE", "MIDDLE"]}, index=["a", "b", "c"])
    m = hard_negative_mask(["a", "b", "c"], info)
    assert m[0, 1] and not m[0, 2]
    # La fila 1 sin máscara empata 0,5 por "c" (otro tier); comparada solo contra "a", 1,0.
    np.testing.assert_allclose(row_auc(D, m)[:2], [1.0, 1.0])


def test_reparto_de_varianza_por_jugador():
    # x = estilo (varianza 1) + ruido (varianza 1): la mitad es del jugador.
    df = _games(n_players=200, n_games=60, style=1.0, champ_effect=0.0)
    share = player_variance_share(df, FEATS, min_games=40)
    assert np.allclose(share, 0.5, atol=0.1)
    # Sin estilo, con 10 partidas por jugador: sin descontar el ruido de las medias
    # daría ~0,09 (sigma_w / n sobre el total).
    sin = player_variance_share(_games(n_players=400, n_games=10, style=0.0, champ_effect=0.0),
                                FEATS, min_games=10)
    assert np.allclose(sin, 0.0, atol=0.04)


def test_keep_min_tier_filtra_por_jugador_con_la_moda_del_padron():
    df = pd.DataFrame({
        "player_id":   ["a", "a", "a", "b", "b", "c", "c", "d"],
        "player_tier": ["MASTER", "MASTER", "DIAMOND", "DIAMOND", "DIAMOND", "CHALLENGER", None, None],
    })
    out = keep_min_tier(df, "MASTER")
    # a: moda MASTER (se queda con sus 3 filas, también la que dice DIAMOND); b: DIAMOND afuera;
    # c: CHALLENGER aunque una fila no tenga tier; d: sin padrón, afuera.
    assert list(out["player_id"]) == ["a", "a", "a", "c", "c"]
    assert set(keep_min_tier(df, "DIAMOND")["player_id"]) == {"a", "b", "c"}
