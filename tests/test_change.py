"""Cambio reciente del jugador (src/features/change.py): valores y ausencia de leakage."""

import numpy as np
import pandas as pd
import pytest

from src.features.change import (hour_shift, play_rate, pool_change, style_scale, style_shift,
                                 window_vs_before)
from src.features.margin import ORDEN

LANES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
W, MP = 4, 3


def _datos(n_partidas=90, seed=0):
    """Partidas de 10 jugadores tomados de un grupo de 14, cada ~7 h, con faltantes en `x`."""
    rng = np.random.default_rng(seed)
    filas = []
    for m in range(n_partidas):
        jug = rng.choice(14, 10, replace=False)
        gana = rng.integers(0, 2)
        t = int(m * 7 * 3.6e6 + rng.integers(0, 3.6e6))
        for i, p in enumerate(jug):
            team = 100 if i < 5 else 200
            filas.append({"match_id": f"M{m:03d}", "game_creation": t, "team_id": team,
                          "lane": LANES[i % 5], "player_id": f"P{p}", "champion_id": int(rng.integers(0, 4)),
                          "win": int(gana == (team == 100)), "x": rng.normal(), "z": rng.normal()})
    df = pd.DataFrame(filas).sort_values(ORDEN).reset_index(drop=True)
    df.loc[rng.choice(len(df), 20, replace=False), "x"] = np.nan
    return df


def _previas(df, k):
    """Filas anteriores del mismo jugador: (recientes, anteriores a esas)."""
    antes = df.iloc[:k]
    prev = antes[antes.player_id == df.player_id.iloc[k]]
    return prev.iloc[-W:], prev.iloc[:-W]


def _conc(champs):
    n = len(champs)
    return sum(a == b for i, a in enumerate(champs) for j, b in enumerate(champs) if i != j) / (n * (n - 1))


def test_pool_coincide_con_la_fuerza_bruta():
    df = _datos()
    got = pool_change(df, W, MP)
    for k in range(len(df)):
        rec, ant = _previas(df, k)
        if len(rec) < W or len(ant) < MP:
            assert got.iloc[k].isna().all()
            continue
        c = df.champion_id.iloc[k]
        a, b = _conc(rec.champion_id.tolist()), _conc(ant.champion_id.tolist())
        esperado = [a, b, a - b, (rec.champion_id == c).mean() - (ant.champion_id == c).mean()]
        assert got.iloc[k].to_numpy() == pytest.approx(esperado)
    assert got.d_pool.notna().sum() > 100


def test_ventanas_coinciden_con_la_fuerza_bruta():
    df = _datos()
    got = window_vs_before(df, ["win"], W, MP)
    for k in range(len(df)):
        rec, ant = _previas(df, k)
        if len(rec) < W or len(ant) < MP:
            assert np.isnan(got.d_win.iloc[k])
        else:
            assert got.d_win.iloc[k] == pytest.approx(rec.win.mean() - ant.win.mean())


def test_ventanas_con_faltantes_exigen_la_ventana_completa():
    df = _datos()
    got = window_vs_before(df, ["x"], W, MP)
    for k in range(len(df)):
        rec, ant = _previas(df, k)
        completo = rec.x.notna().sum() >= W and ant.x.notna().sum() >= MP
        assert np.isnan(got.d_x.iloc[k]) != completo
        if completo:
            assert got.d_x.iloc[k] == pytest.approx(rec.x.mean() - ant.x.mean())


def test_huella_coincide_con_la_fuerza_bruta():
    df = _datos().drop(columns="x").assign(x=lambda d: d.z * 2 + 1)      # sin faltantes
    esc = style_scale(df, ["x", "z"])
    got = style_shift(df, ["x", "z"], W, MP, k_cl=3, escala=esc)
    r = {}
    for c in ("x", "z"):
        mu, sd = esc[c]
        r[c] = np.zeros(len(df))
        for k in range(len(df)):
            f = df.iloc[k]
            antes = df.iloc[:k]
            g = antes[(antes.champion_id == f.champion_id) & (antes.lane == f.lane) & (antes.win == f.win)][c]
            r[c][k] = (f[c] - (g.sum() + 3 * mu[(f.lane, f.win)]) / (len(g) + 3)) / sd[(f.lane, f.win)]
    for k in range(len(df)):
        rec, ant = _previas(df, k)
        if len(rec) < W or len(ant) < MP:
            assert np.isnan(got.cambio_huella.iloc[k])
            continue
        z2 = [((r[c][rec.index].mean() - r[c][ant.index].mean()) / np.sqrt(1 / len(rec) + 1 / len(ant))) ** 2
              for c in ("x", "z")]
        assert got.cambio_huella.iloc[k] == pytest.approx(np.mean(z2))
        assert got.n_ant.iloc[k] == len(ant)


def test_horario_coincide_con_la_fuerza_bruta():
    df = _datos()
    got = hour_shift(df, W, MP, bins=6, utc_offset=-3)
    franja = (((df.game_creation / 3.6e6 - 3) % 24) // 4).astype(int)
    for k in range(len(df)):
        rec, ant = _previas(df, k)
        if len(rec) < W or len(ant) < MP:
            assert np.isnan(got.iloc[k])
            continue
        p = (np.bincount(franja[rec.index], minlength=6) + 0.5) / (len(rec) + 3)
        q = (np.bincount(franja[ant.index], minlength=6) + 0.5) / (len(ant) + 3)
        assert got.iloc[k] == pytest.approx(0.5 * np.abs(p - q).sum())


def test_horario_distingue_un_cambio_de_franja():
    """Un jugador que pasa de jugar a las 15 h a jugar a las 4 h queda lejos de sí mismo."""
    dia = 8.64e7
    def jugador(pid, horas):
        return [{"match_id": f"{pid}{i}", "game_creation": int(i * dia + (h + 3) * 3.6e6), "team_id": 100,
                 "lane": "TOP", "player_id": pid, "champion_id": 1, "win": 1} for i, h in enumerate(horas)]
    df = pd.DataFrame(jugador("igual", [15] * 30) + jugador("cambia", [15] * 20 + [4] * 10))
    df = df.sort_values(ORDEN).reset_index(drop=True)
    tv = hour_shift(df, window=10, min_prev=5)
    ultima = df.groupby("player_id").tail(1)
    assert tv[ultima.index[ultima.player_id == "igual"][0]] < 0.1       # no es 0 por el suavizado
    assert tv[ultima.index[ultima.player_id == "cambia"][0]] > 0.5


def test_ritmo_coincide_con_la_fuerza_bruta():
    df = _datos()
    got = play_rate(df, horas=48, min_prev=5, min_dias=3)
    for k in range(len(df)):
        antes = df.iloc[:k]
        prev = antes[antes.player_id == df.player_id.iloc[k]]
        t = df.game_creation.iloc[k]
        if len(prev) < 5 or t - prev.game_creation.iloc[0] < 3 * 8.64e7:
            assert np.isnan(got.iloc[k])
            continue
        rec = (prev.game_creation >= t - 48 * 3.6e6).sum()
        hab = len(prev) * 48 * 3.6e6 / (t - prev.game_creation.iloc[0])
        assert got.iloc[k] == pytest.approx(np.log((rec + 1) / (hab + 1)))
    assert got.notna().sum() > 100


def _todo(df):
    esc = style_scale(_datos(), ["x", "z"])          # escala fija: no depende de lo que se toque
    return pd.concat([pool_change(df, W, MP), window_vs_before(df, ["win", "x"], W, MP),
                      style_shift(df, ["x", "z"], W, MP, escala=esc), hour_shift(df, W, MP),
                      play_rate(df, min_prev=5, min_dias=3)], axis=1)


def test_no_usa_la_partida_actual_ni_las_futuras():
    """Cambiar todo desde una partida en adelante no mueve ninguna medida de las filas anteriores,
    ni las de esa misma partida (salvo `otp_nuevo`, que depende del campeón elegido: es el draft)."""
    df = _datos()
    corte = 60 * 10                                   # primera fila de la partida 60
    otro = df.copy()
    rng = np.random.default_rng(1)
    tarde = otro.index >= corte
    otro.loc[tarde, "win"] = 1 - otro.loc[tarde, "win"]
    otro.loc[tarde, ["x", "z"]] = rng.normal(size=(tarde.sum(), 2)) * 5
    otro.loc[otro.index >= corte + 10, "champion_id"] = 3
    a, b = _todo(df), _todo(otro)
    pd.testing.assert_frame_equal(a.iloc[:corte + 10], b.iloc[:corte + 10])
    assert not a.iloc[corte + 10:].equals(b.iloc[corte + 10:])


def test_exige_orden_temporal():
    df = _datos().iloc[::-1]
    for f in (pool_change, hour_shift, play_rate):
        with pytest.raises(ValueError):
            f(df)
    with pytest.raises(ValueError):
        window_vs_before(df, ["win"])
