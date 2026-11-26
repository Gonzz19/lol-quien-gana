"""Cambio reciente del jugador: sus últimas `window` partidas PREVIAS contra las anteriores a esas.

La pregunta es si el historial de la cuenta sigue describiendo a quien la juega (cuenta
boosteada o compartida, jugador que se volvió OTP, cambio de hábitos). Cuatro medidas, todas
con partidas anteriores a la fila:

* `pool_change`: concentración del pool de campeones y peso del campeón actual.
* `window_vs_before`: media reciente y media anterior de cualquier columna (p. ej. `win`).
* `style_shift`: distancia de estilo entre las dos ventanas, descontando campeón, línea y
  resultado (la huella de `fingerprint.residualize`, pero con medias previas).
* `hour_shift` y `play_rate`: horario y ritmo de juego.

Con n_rec y n_ant las partidas de cada ventana, una medida vale NaN si n_rec < `window` o
n_ant < `min_prev`: sin las dos ventanas no hay cambio que medir.

`df` tiene que venir ordenado por `margin.ORDEN`, una fila por jugador-partida.
Detalle en PROJECT_SUMMARY.md §9.S.30.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.features.margin import ORDEN, _previo, _ultimas


def _ordenado(df: pd.DataFrame) -> None:
    if not df[ORDEN[0]].is_monotonic_increasing:
        raise ValueError("df tiene que estar ordenado por game_creation")


def _filas_por_jugador(df: pd.DataFrame):
    """Posiciones de las filas de cada jugador, en orden temporal."""
    return df.groupby("player_id", sort=False).indices.values()


def pool_change(df: pd.DataFrame, window: int = 10, min_prev: int = 5) -> pd.DataFrame:
    """Concentración del pool reciente contra la anterior, y peso del campeón de la fila.

    Concentración = probabilidad de que dos partidas distintas de la ventana sean con el
    mismo campeón, (sum_c n_c^2 - n) / (n (n - 1)): a diferencia del índice de Herfindahl
    sum_c (n_c / n)^2, no depende del tamaño de la ventana.

    Devuelve `conc_rec`, `conc_ant`, `d_pool` = conc_rec - conc_ant, y `otp_nuevo` =
    (fracción de la ventana reciente con el campeón de la fila) - (lo mismo en la anterior).
    """
    _ordenado(df)
    if window < 2 or min_prev < 2:
        raise ValueError("window y min_prev tienen que ser >= 2")
    champ = df["champion_id"].to_numpy()
    out = np.full((len(df), 4), np.nan)
    for idx in _filas_por_jugador(df):
        cs = champ[idx].tolist()
        rec, ant = {}, {}
        ss_rec = ss_ant = n_ant = 0
        for j, (i, c) in enumerate(zip(idx.tolist(), cs)):
            if j >= window and n_ant >= min_prev:
                a = (ss_rec - window) / (window * (window - 1))
                b = (ss_ant - n_ant) / (n_ant * (n_ant - 1))
                out[i] = a, b, a - b, rec.get(c, 0) / window - ant.get(c, 0) / n_ant
            k = rec.get(c, 0); ss_rec += 2 * k + 1; rec[c] = k + 1
            if j >= window:                       # la más vieja de la ventana pasa a "anterior"
                o = cs[j - window]
                k = rec[o]; ss_rec -= 2 * k - 1; rec[o] = k - 1
                k = ant.get(o, 0); ss_ant += 2 * k + 1; ant[o] = k + 1
                n_ant += 1
    return pd.DataFrame(out, index=df.index, columns=["conc_rec", "conc_ant", "d_pool", "otp_nuevo"])


def _ventanas(r: pd.Series, v: pd.Series, pj: pd.Series, window: int):
    """Sumas y conteos de la ventana reciente y de la anterior (r ya sin faltantes)."""
    S, N = _previo(r, pj), _previo(v, pj)
    s_rec, n_rec = _ultimas(S, pj, window), _ultimas(N, pj, window)
    return s_rec, n_rec, S - s_rec, N - n_rec


def window_vs_before(df: pd.DataFrame, cols: List[str], window: int = 10, min_prev: int = 5) -> pd.DataFrame:
    """`rec_<col>`, `ant_<col>` (medias de cada ventana) y `d_<col>` = rec - ant."""
    _ordenado(df)
    pj = df["player_id"]
    out = {}
    for c in cols:
        s_rec, n_rec, s_ant, n_ant = _ventanas(df[c].fillna(0).astype(float), df[c].notna().astype(float), pj, window)
        ok = (n_rec >= window) & (n_ant >= min_prev)
        out[f"rec_{c}"] = (s_rec / n_rec).where(ok)
        out[f"ant_{c}"] = (s_ant / n_ant).where(ok)
        out[f"d_{c}"] = out[f"rec_{c}"] - out[f"ant_{c}"]
    return pd.DataFrame(out, index=df.index)


def style_scale(df: pd.DataFrame, stats: List[str]) -> Dict[str, Tuple[pd.Series, pd.Series]]:
    """Media y desvío de cada estadística por (línea, resultado)."""
    g = df.groupby(["lane", "win"])
    return {c: (g[c].mean(), g[c].std()) for c in stats}


def style_shift(df: pd.DataFrame, stats: List[str], window: int = 10, min_prev: int = 5,
                k_cl: float = 20, escala: Optional[dict] = None) -> pd.DataFrame:
    """Distancia de estilo entre la ventana reciente y la anterior.

    Para cada estadística x, con g = (campeón, línea, resultado):
        r = (x - media_previa(g)) / desvío(línea, resultado)
        z = (media_rec(r) - media_ant(r)) / sqrt(1 / n_rec + 1 / n_ant)
    y `cambio_huella` = media de z^2 sobre las estadísticas con las dos ventanas completas.
    La media previa de g se encoge hacia la de (línea, resultado) con peso `k_cl`.
    Descontar el resultado saca la fuerza: queda cómo juega, no cuánto gana.
    Devuelve también `n_ant` (el sesgo de la distancia depende de él).
    """
    _ordenado(df)
    escala = escala or style_scale(df, stats)
    pj = df["player_id"]
    g = [df["champion_id"], df["lane"], df["win"]]
    lw = pd.MultiIndex.from_frame(df[["lane", "win"]])
    z2 = []
    for c in stats:
        mu = pd.Series(escala[c][0].reindex(lw).to_numpy(), index=df.index)
        sd = pd.Series(escala[c][1].reindex(lw).to_numpy(), index=df.index)
        x, v = df[c], df[c].notna().astype(float)
        media_g = (_previo(x.fillna(0), g) + k_cl * mu) / (_previo(v, g) + k_cl)
        r = ((x - media_g) / sd).fillna(0)
        s_rec, n_rec, s_ant, n_ant = _ventanas(r, v, pj, window)
        ok = (n_rec >= window) & (n_ant >= min_prev)
        z = (s_rec / n_rec - s_ant / n_ant) / np.sqrt(1 / n_rec + 1 / n_ant)
        z2.append((z ** 2).where(ok))
    n_prev = df.groupby("player_id", sort=False).cumcount()
    return pd.DataFrame({"cambio_huella": pd.concat(z2, axis=1).mean(axis=1),
                         "n_ant": (n_prev - window).clip(lower=0)}, index=df.index)


def hour_shift(df: pd.DataFrame, window: int = 10, min_prev: int = 5, bins: int = 6,
               utc_offset: float = -3) -> pd.Series:
    """Distancia de variación total entre el horario de juego reciente y el anterior.

    La hora local (`utc_offset`, -3 = Argentina) se parte en `bins` franjas iguales;
    p_rec y p_ant son las proporciones de cada ventana con suavizado de 0,5 por franja, y
    la medida es 0,5 * sum |p_rec - p_ant| (0 = mismo horario, 1 = horarios disjuntos).
    """
    _ordenado(df)
    pj = df["player_id"]
    hora = ((df["game_creation"] / 3.6e6 + utc_offset) % 24).to_numpy()
    franja = np.minimum((hora * bins / 24).astype(int), bins - 1)
    uno = pd.Series(1.0, index=df.index)
    tv = 0.0
    for b in range(bins):
        s_rec, n_rec, s_ant, n_ant = _ventanas(pd.Series((franja == b).astype(float), index=df.index), uno, pj, window)
        tv = tv + ((s_rec + 0.5) / (n_rec + 0.5 * bins) - (s_ant + 0.5) / (n_ant + 0.5 * bins)).abs()
    return (0.5 * tv).where((n_rec >= window) & (n_ant >= min_prev)).rename("cambio_horario")


def play_rate(df: pd.DataFrame, horas: float = 48, min_prev: int = 10, min_dias: float = 7) -> pd.Series:
    """Ritmo de juego: partidas previas en las últimas `horas` contra lo habitual del jugador.

    Con n las partidas previas, t0 la primera y t la actual: habitual = n * horas / (t - t0),
    y la medida es log((recientes + 1) / (habitual + 1)). NaN con menos de `min_prev` partidas
    previas o menos de `min_dias` días de historial.
    """
    _ordenado(df)
    t = df["game_creation"].to_numpy(dtype=float)
    out = np.full(len(df), np.nan)
    ms = horas * 3.6e6
    for idx in _filas_por_jugador(df):
        ts = t[idx]
        n = np.arange(len(ts))
        rec = n - np.searchsorted(ts, ts - ms, side="left")
        span = ts - ts[0]
        ok = (n >= min_prev) & (span >= min_dias * 8.64e7)
        hab = np.divide(n * ms, span, out=np.zeros(len(ts)), where=span > 0)
        out[idx] = np.where(ok, np.log((rec + 1) / (hab + 1)), np.nan)
    return pd.Series(out, index=df.index, name="ritmo")
