"""Nivel del lobby previo (src/features/lobby.py) y margen ajustado por rival: valores y leakage."""

import numpy as np
import pandas as pd
import pytest

from src.features.lobby import prior_lobby_level
from src.features.margin import ORDEN, lane_scale, prior_margin_vs_rival

LANES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]


def _datos(n_partidas=60, seed=0):
    """Partidas de 10 jugadores de un grupo de 25; solo 9 tienen nivel, así hay lobbies sin L."""
    rng = np.random.default_rng(seed)
    nivel = {p: (float(rng.normal(300, 200)) if p < 9 else np.nan) for p in range(25)}
    filas = []
    for m in range(n_partidas):
        jug = rng.choice(25, 10, replace=False)
        gana = rng.integers(0, 2)
        for i, p in enumerate(jug):
            team = 100 if i < 5 else 200
            filas.append({"match_id": f"M{m:03d}", "game_creation": 1000 * m, "team_id": team,
                          "lane": LANES[i % 5], "player_id": f"P{p}", "champion_id": int(rng.integers(0, 6)),
                          "win": int(gana == (team == 100)), "lp": nivel[p], "x": rng.normal()})
    df = pd.DataFrame(filas).sort_values(ORDEN).reset_index(drop=True)
    df.loc[rng.choice(len(df), 15, replace=False), "x"] = np.nan
    return df


def test_nivel_coincide_con_la_fuerza_bruta():
    df = _datos()
    got = prior_lobby_level(df, "lp", k=3, mu0=250.0, window=4, min_otros=4)
    L = np.full(len(df), np.nan)
    for i in range(len(df)):
        otros = df[(df.match_id == df.match_id.iloc[i]) & (df.player_id != df.player_id.iloc[i])].lp.dropna()
        if len(otros) >= 4:
            L[i] = otros.mean()
    assert 0.2 < np.isnan(L).mean() < 0.8                      # hay lobbies con y sin nivel
    np.testing.assert_allclose(got.nivel_lobby.to_numpy(), L, equal_nan=True)
    for i in range(len(df)):
        prev = np.where(df.player_id.to_numpy()[:i] == df.player_id.iloc[i])[0]
        prev = prev[~np.isnan(L[prev])]
        assert got.nivel_acum.iloc[i] == pytest.approx((L[prev].sum() + 3 * 250.0) / (len(prev) + 3))
        assert got.nivel_v4.iloc[i] == pytest.approx((L[prev[-4:]].sum() + 3 * 250.0) / (len(prev[-4:]) + 3))
        assert got.n_nivel.iloc[i] == len(prev)


def test_nivel_no_usa_el_valor_propio_ni_las_partidas_futuras():
    df = _datos()
    a = prior_lobby_level(df, "lp", mu0=250.0)
    otro = df.copy()
    otro.loc[otro.player_id == "P1", "lp"] = 5000.0            # el nivel propio no entra en el suyo
    b = prior_lobby_level(otro, "lp", mu0=250.0)
    mio = (df.player_id == "P1").to_numpy()
    pd.testing.assert_frame_equal(a[mio], b[mio])
    assert not a[~mio].equals(b[~mio])
    corte = 40 * 10
    tarde = df.copy()
    tarde.loc[tarde.index >= corte, "lp"] = 0.0                # partida 40 en adelante
    c = prior_lobby_level(tarde, "lp", mu0=250.0)
    cols = ["nivel_acum", "nivel_v10", "n_nivel"]
    pd.testing.assert_frame_equal(a[cols].iloc[:corte + 10], c[cols].iloc[:corte + 10])
    assert not a[cols].iloc[corte + 10:].equals(c[cols].iloc[corte + 10:])


def _rival_fuerza_bruta(df, c, k_cl, k_j, window, escala):
    mu, sd = escala[c]
    n = len(df)
    pid, val = df.player_id.to_numpy(), df[c].notna().to_numpy()
    r = np.zeros(n)
    for k in range(n):
        f, antes = df.iloc[k], df.iloc[:k]
        cl = antes[(antes.champion_id == f.champion_id) & (antes.lane == f.lane)][c].dropna()
        media = (cl.sum() + k_cl * mu[f.lane]) / (len(cl) + k_cl)
        r[k] = 0.0 if np.isnan(f[c]) else (f[c] - media) / sd[f.lane]
    previas = [np.where(pid[:k] == pid[k])[0] for k in range(n)]
    m = np.array([r[p].sum() / (val[p].sum() + k_j) for p in previas])
    r2 = r.copy()
    for k in range(n):
        riv = np.where(((df.match_id == df.match_id.iloc[k]) & (df.lane == df.lane.iloc[k])).to_numpy() & (pid != pid[k]))[0]
        if len(riv) == 1 and val[k]:
            r2[k] += m[riv[0]]
    acum = np.array([r2[p].sum() / (val[p].sum() + k_j) for p in previas])
    vent = np.array([r2[p[-window:]].sum() / (val[p[-window:]].sum() + k_j) for p in previas])
    return acum, vent


def test_margen_contra_rival_coincide_con_la_fuerza_bruta():
    df = _datos()
    esc = lane_scale(df, ["x"])
    got = prior_margin_vs_rival(df, ["x"], k_cl=3, k_j=2, window=4, escala=esc)
    acum, vent = _rival_fuerza_bruta(df, "x", 3, 2, 4, esc)
    np.testing.assert_allclose(got.adj_acum_x.to_numpy(), acum, atol=1e-10)
    np.testing.assert_allclose(got.adj_v4_x.to_numpy(), vent, atol=1e-10)


def test_margen_contra_rival_no_usa_la_partida_actual_ni_las_futuras():
    df = _datos()
    esc = lane_scale(df, ["x"])
    corte = 40 * 10
    otro = df.copy()
    otro.loc[otro.index >= corte, "x"] = 9.0
    a = prior_margin_vs_rival(df, ["x"], escala=esc)
    b = prior_margin_vs_rival(otro, ["x"], escala=esc)
    pd.testing.assert_frame_equal(a.iloc[:corte + 10], b.iloc[:corte + 10])
    assert not a.iloc[corte + 10:].equals(b.iloc[corte + 10:])
