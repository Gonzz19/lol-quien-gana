"""Margen previo por jugador (src/features/margin.py): valores y ausencia de leakage."""

import numpy as np
import pandas as pd
import pytest

from src.features.margin import ORDEN, lane_diffs, lane_scale, prior_margin, prior_margin_by, prior_winrate

LANES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]


def _datos(n_partidas=60, seed=0):
    """Partidas de 10 jugadores tomados de un grupo de 25, con algunos faltantes en `x`."""
    rng = np.random.default_rng(seed)
    filas = []
    for m in range(n_partidas):
        jug = rng.choice(25, 10, replace=False)
        gana = rng.integers(0, 2)
        for i, p in enumerate(jug):
            team = 100 if i < 5 else 200
            filas.append({"match_id": f"M{m:03d}", "game_creation": 1000 * m, "team_id": team,
                          "lane": LANES[i % 5], "player_id": f"P{p}", "champion_id": int(rng.integers(0, 6)),
                          "win": int(gana == (team == 100)), "x": rng.normal(), "z": rng.normal()})
    df = pd.DataFrame(filas).sort_values(ORDEN).reset_index(drop=True)
    df.loc[rng.choice(len(df), 15, replace=False), "x"] = np.nan
    return df


def _fuerza_bruta(df, c, k_cl, k_j, window, escala):
    """Lo mismo que prior_margin, fila por fila y mirando solo filas anteriores."""
    mu, sd = escala[c]
    r = np.zeros(len(df))
    for k in range(len(df)):
        f = df.iloc[k]
        antes = df.iloc[:k]
        cl = antes[(antes.champion_id == f.champion_id) & (antes.lane == f.lane)][c].dropna()
        media = (cl.sum() + k_cl * mu[f.lane]) / (len(cl) + k_cl)
        r[k] = 0.0 if np.isnan(f[c]) else (f[c] - media) / sd[f.lane]
    acum, vent = np.zeros(len(df)), np.zeros(len(df))
    for k in range(len(df)):
        prev = np.where((df.player_id.to_numpy()[:k] == df.player_id.iloc[k]))[0]
        valido = df[c].notna().to_numpy()
        acum[k] = r[prev].sum() / (valido[prev].sum() + k_j)
        ult = prev[-window:]
        vent[k] = r[ult].sum() / (valido[ult].sum() + k_j)
    return acum, vent


def test_coincide_con_la_fuerza_bruta():
    df = _datos()
    escala = lane_scale(df, ["x"])
    out = prior_margin(df, ["x"], k_cl=3, k_j=2, window=4, escala=escala)
    acum, vent = _fuerza_bruta(df, "x", 3, 2, 4, escala)
    assert np.allclose(out["acum_x"], acum)
    assert np.allclose(out["v4_x"], vent)


def test_la_primera_partida_de_cada_jugador_vale_cero():
    df = _datos()
    out = prior_margin(df, ["x", "z"])
    primera = ~df.player_id.duplicated()
    assert (out[primera] == 0).all().all()


def test_no_mira_la_partida_actual_ni_las_posteriores():
    """Cambiar el dato propio y todo lo que viene después no mueve el puntaje de una fila."""
    df = _datos()
    escala = lane_scale(df, ["x", "z"])
    antes = prior_margin(df, ["x", "z"], escala=escala)
    wr_antes = prior_winrate(df)
    corte = 300                                    # límite de partida: 30 partidas x 10 filas
    mut = df.copy()
    mut.loc[corte:, ["x", "z"]] = mut.loc[corte:, ["x", "z"]] * 50 + 7
    mut.loc[corte:, "win"] = 1 - mut.loc[corte:, "win"]
    despues = prior_margin(mut, ["x", "z"], escala=escala)
    partida = df.match_id == df.match_id.iloc[corte]
    assert np.allclose(antes[: corte], despues[: corte])
    assert np.allclose(antes[partida], despues[partida])          # la primera partida mutada tampoco
    assert np.allclose(wr_antes[partida], prior_winrate(mut)[partida])
    assert not np.allclose(antes[corte + 10:], despues[corte + 10:])   # las posteriores sí cambian


def test_winrate_previo():
    df = _datos()
    out = prior_winrate(df, window=3, k=10)
    p = df.player_id.iloc[-1]
    w = df[df.player_id == p].win.to_numpy()
    fila = out[df.player_id == p].iloc[-1]
    assert fila.wr_acum == pytest.approx((w[:-1].sum() + 5) / (len(w) - 1 + 10) - 0.5)
    assert fila.wr_v3 == pytest.approx((w[-4:-1].sum() + 5) / (3 + 10) - 0.5)


def test_diferencia_contra_el_rival_de_linea():
    df = lane_diffs(_datos(), ["x", "z"])
    m = df[df.match_id == "M005"]
    top = m[m.lane == "TOP"].set_index("team_id")
    assert top.loc[100, "d_z"] == pytest.approx(top.loc[100, "z"] - top.loc[200, "z"])
    assert top.loc[200, "d_z"] == pytest.approx(-top.loc[100, "d_z"])
    falta = df[df.x.isna()].iloc[0]
    par = df[(df.match_id == falta.match_id) & (df.lane == falta.lane)]
    assert par.d_x.isna().all()


def test_exige_orden_temporal():
    with pytest.raises(ValueError):
        prior_margin(_datos().iloc[::-1], ["x"])


def test_margen_por_grupo_coincide_con_la_fuerza_bruta():
    """m_pg - m_p, con m_pg = (S_pg + k_g m_p) / (N_pg + k_g), solo con filas anteriores."""
    df = _datos()
    escala = lane_scale(df, ["x"])
    out = prior_margin_by(df, ["x"], ["champion_id"], "camp", k_cl=3, k_j=2, k_g=4, escala=escala)
    acum, _ = _fuerza_bruta(df, "x", 3, 2, 4, escala)
    mu, sd = escala["x"]
    r = np.zeros(len(df))
    for k in range(len(df)):
        f, antes = df.iloc[k], df.iloc[:k]
        cl = antes[(antes.champion_id == f.champion_id) & (antes.lane == f.lane)]["x"].dropna()
        media = (cl.sum() + 3 * mu[f.lane]) / (len(cl) + 3)
        r[k] = 0.0 if np.isnan(f.x) else (f.x - media) / sd[f.lane]
    valido = df.x.notna().to_numpy()
    for k in range(len(df)):
        prev = np.where((df.player_id.to_numpy()[:k] == df.player_id.iloc[k])
                        & (df.champion_id.to_numpy()[:k] == df.champion_id.iloc[k]))[0]
        esperado = (r[prev].sum() + 4 * acum[k]) / (valido[prev].sum() + 4) - acum[k]
        assert out["camp_x"].iloc[k] == pytest.approx(esperado)


def test_margen_por_grupo_no_mira_la_partida_actual_ni_las_posteriores():
    df = _datos()
    escala = lane_scale(df, ["x", "z"])
    antes = prior_margin_by(df, ["x", "z"], ["lane"], "linea", escala=escala)
    corte = 300
    mut = df.copy()
    mut.loc[corte:, ["x", "z"]] = mut.loc[corte:, ["x", "z"]] * 50 + 7
    despues = prior_margin_by(mut, ["x", "z"], ["lane"], "linea", escala=escala)
    partida = df.match_id == df.match_id.iloc[corte]
    assert np.allclose(antes[:corte], despues[:corte])
    assert np.allclose(antes[partida], despues[partida])
    assert not np.allclose(antes[corte + 10:], despues[corte + 10:])


def test_prior_shift_es_el_margen_encogido_hacia_el_prior():
    """m + pri = (S + k mu) / (N + k): con mu = 0 vuelve a prior_margin; mu entra con peso k / (N + k)."""
    from src.features.margin import prior_shift
    df = _datos()
    esc = lane_scale(df, ["x"])
    rng = np.random.default_rng(3)
    mu = pd.DataFrame({"x": rng.normal(size=len(df))}, index=df.index)
    m = prior_margin(df, ["x"], k_cl=4, k_j=3, window=5, escala=esc)
    pri = prior_shift(df, ["x"], mu, k_j=3, window=5)
    pid, val = df.player_id.to_numpy(), df.x.notna().to_numpy()
    for k in range(len(df)):
        prev = np.where(pid[:k] == pid[k])[0]
        n, n5 = val[prev].sum(), val[prev[-5:]].sum()
        assert pri.pri_acum_x.iloc[k] == pytest.approx(3 * mu.x.iloc[k] / (n + 3))
        assert pri.pri_v5_x.iloc[k] == pytest.approx(3 * mu.x.iloc[k] / (n5 + 3))
    # mu = 0: no cambia nada
    cero = prior_shift(df, ["x"], mu * 0, k_j=3, window=5)
    assert (cero.to_numpy() == 0).all()
    # la partida actual y las futuras no cambian el término de las filas anteriores
    otro = df.copy()
    otro.loc[otro.index >= 300, "x"] = np.nan
    pd.testing.assert_frame_equal(prior_shift(otro, ["x"], mu, 3, 5).iloc[:300], pri.iloc[:300])
