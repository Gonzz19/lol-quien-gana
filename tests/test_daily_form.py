"""Forma del día por jugador y partida (src/features/daily_form.py)."""

import pandas as pd
import pytest

from src.features.daily_form import baseline_winrate, daily_form

H = 3_600_000
T0 = 1_788_231_600_000  # 2026-09-01 00:00 ART (martes)


def _parts(rows):
    """rows: [(match_id, player_id, win, horas desde T0)]; partidas de 30 min."""
    return pd.DataFrame([{"match_id": m, "player_id": p, "win": w,
                          "game_creation": T0 + int(h * H), "game_duration": 1800}
                         for m, p, w, h in rows])


def _row(df, match_id, player="P"):
    return df[(df.match_id == match_id) & (df.player_id == player)].iloc[0]


def test_el_dia_de_juego_va_de_8_a_8():
    df = daily_form(_parts([("A", "P", 1, 3), ("B", "P", 1, 7.9), ("C", "P", 1, 8), ("D", "P", 1, 30)]))
    # 03:00 y 07:54 del martes son del lunes; 08:00 arranca el martes; 06:00 del miércoles es del martes.
    assert list(df.dia_juego) == ["2026-08-31", "2026-08-31", "2026-09-01", "2026-09-01"]


def test_cuenta_solo_las_partidas_previas_del_mismo_dia():
    df = daily_form(_parts([("A", "P", 1, 10), ("B", "P", 0, 11), ("C", "P", 1, 12), ("D", "P", 0, 40)]))
    c = _row(df, "C")
    assert (c.n_previas_dia, c.ganadas_previas_dia, c.perdidas_previas_dia) == (2, 1, 1)
    assert _row(df, "D").n_previas_dia == 0                      # día nuevo


def test_sin_leakage_el_resultado_propio_no_entra_en_su_indice():
    base = _parts([("A", "P", 1, 10), ("B", "P", 0, 11), ("C", "P", 1, 12)])
    flip = base.copy()
    flip.loc[flip.match_id == "C", "win"] = 0
    wr = pd.Series({"P": 0.5})
    assert _row(daily_form(base, wr), "C").indice_forma_dia == _row(daily_form(flip, wr), "C").indice_forma_dia


def test_indice_cero_sin_previas_y_formula():
    wr = pd.Series({"P": 0.5})
    df = daily_form(_parts([("A", "P", 1, 10), ("B", "P", 1, 11), ("C", "P", 1, 12)]), wr, c=3)
    assert _row(df, "A").indice_forma_dia == 0
    # C: 2 ganadas previas de 2 -> (2 + 3*0.5)/(2+3) - 0.5 = 0.2
    assert _row(df, "C").indice_forma_dia == pytest.approx(0.2)


def test_racha_con_signo():
    df = daily_form(_parts([("A", "P", 1, 10), ("B", "P", 1, 11), ("C", "P", 0, 12),
                            ("D", "P", 0, 13), ("E", "P", 0, 14), ("F", "P", 1, 15)]))
    assert list(df.racha) == [0, 1, 2, -1, -2, -3]


def test_jugadores_separados_y_descanso():
    df = daily_form(_parts([("A", "P", 1, 10), ("A", "Q", 0, 10), ("B", "P", 0, 12), ("C", "P", 0, 40)]))
    assert _row(df, "A", "Q").n_previas_dia == 0
    assert _row(df, "B").minutos_desde_ultima == pytest.approx(90.0)   # 12:00 - (10:00 + 30 min)
    assert not _row(df, "B").primera_del_dia
    # Primera del día (otro día de juego): sin descanso, marcada aparte.
    assert _row(df, "C").primera_del_dia and pd.isna(_row(df, "C").minutos_desde_ultima)
    assert _row(df, "A").primera_del_dia and pd.isna(_row(df, "A").minutos_desde_ultima)


def test_winrate_habitual_sale_del_train_y_se_suaviza():
    # P: 1 sola partida ganada; Q: 0 de 3. Global del train = 1/4.
    train = _parts([("A", "P", 1, 1), ("B", "Q", 0, 2), ("C", "Q", 0, 3), ("D", "Q", 0, 4)])
    wr = baseline_winrate(train, k=10)
    assert wr["P"] == pytest.approx((1 + 10 * 0.25) / 11)    # no 1.0
    assert wr["Q"] == pytest.approx((0 + 10 * 0.25) / 13)    # no 0.0


def test_dia_completo_solo_dentro_de_la_ventana():
    parts = _parts([("A", "P", 1, 10), ("B", "P", 1, 34), ("Z", "P", 1, -40), ("X", "Q", 1, 10)])
    expected = pd.DataFrame({"player_id": ["P", "P", "Q", "Q"], "match_id": ["A", "B", "X", "FALTA"]})
    df = daily_form(parts, expected=expected, complete_days=("2026-09-01", "2026-09-02"))
    assert _row(df, "A").dia_completo is True                   # P tiene todo lo listado
    assert _row(df, "Z").dia_completo is None                   # día anterior a la ventana: desconocido
    assert _row(df, "X", "Q").dia_completo is False             # a Q le falta una partida listada
