"""Rasgos condicionales (src/features/conditional.py): valores y ausencia de leakage."""

import numpy as np
import pandas as pd
import pytest

from src.features.conditional import prior_conditional_mean
from src.features.margin import ORDEN

LANES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]


def _datos(n_partidas=60, seed=0):
    rng = np.random.default_rng(seed)
    filas = []
    for m in range(n_partidas):
        jug = rng.choice(20, 10, replace=False)
        for i, p in enumerate(jug):
            filas.append({"match_id": f"M{m:03d}", "game_creation": 1000 * m, "team_id": 100 if i < 5 else 200,
                          "lane": LANES[i % 5], "player_id": f"P{p}", "v": rng.normal(), "c": rng.random() < 0.4})
    df = pd.DataFrame(filas).sort_values(ORDEN).reset_index(drop=True)
    df.loc[rng.choice(len(df), 20, replace=False), "v"] = np.nan
    return df


def test_coincide_con_la_fuerza_bruta():
    df = _datos()
    got = prior_conditional_mean(df, df.v, df.c, k=4, mu=0.3)
    pid = df.player_id.to_numpy()
    for i in range(len(df)):
        prev = df.iloc[:i][(pid[:i] == pid[i])]
        sel = prev[prev.c & prev.v.notna()]
        assert got.rasgo.iloc[i] == pytest.approx((sel.v.sum() + 4 * 0.3) / (len(sel) + 4))
        assert got.n.iloc[i] == len(sel)


def test_no_usa_la_partida_actual_ni_las_futuras():
    df = _datos()
    a = prior_conditional_mean(df, df.v, df.c)
    otro = df.copy()
    tarde = otro.index >= 400
    otro.loc[tarde, "v"] = 50.0
    otro.loc[tarde, "c"] = True
    b = prior_conditional_mean(otro, otro.v, otro.c)
    pd.testing.assert_frame_equal(a.iloc[:400], b.iloc[:400])
    assert not a.iloc[400:].equals(b.iloc[400:])


def test_exige_orden_temporal():
    df = _datos().iloc[::-1]
    with pytest.raises(ValueError):
        prior_conditional_mean(df, df.v, df.c)
