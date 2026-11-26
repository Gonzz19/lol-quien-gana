"""Nivel del lobby previo: un MMR propio, estimado con quién te empareja el juego.

El emparejamiento junta jugadores de MMR parecido. Entonces, si `x` es una medida de nivel de
cada jugador (LP de una foto del padrón, o un rating previo a la partida), el nivel de una
partida para el jugador p es la media de `x` de los OTROS 9:

    L(p, partida) = (sum_{q != p} x_q) / (cantidad de q != p con x conocido)

y su MMR estimado antes de una partida es la media de L en sus partidas ANTERIORES, encogida
hacia `mu0` con peso `k`:

    nivel_acum = (sum L previos + k * mu0) / (n + k)

`nivel_v<window>` usa solo las últimas `window` partidas previas con L conocido.
`nivel_lobby` es el L de la partida actual: se conoce antes de jugar si `x` también.
El propio `x` del jugador nunca entra en su nivel.

`df` ordenado por `margin.ORDEN`, una fila por jugador-partida. PROJECT_SUMMARY.md §9.S.31.
"""
from typing import Optional

import pandas as pd

from src.features.margin import ORDEN, _previo


def prior_lobby_level(df: pd.DataFrame, col: str, k: float = 5, mu0: Optional[float] = None,
                      window: int = 10, min_otros: int = 3) -> pd.DataFrame:
    """`nivel_lobby`, `nivel_acum`, `nivel_v<window>` y `n_nivel` (partidas previas con L conocido).

    `nivel_lobby` es NaN si menos de `min_otros` de los otros jugadores tienen `col`.
    `mu0` por defecto es la media de `nivel_lobby` en `df` (una constante de escala).
    """
    if not df[ORDEN[0]].is_monotonic_increasing:
        raise ValueError("df tiene que estar ordenado por game_creation")
    x, v = df[col].fillna(0).astype(float), df[col].notna().astype(float)
    m = df["match_id"]
    otros = v.groupby(m, sort=False).transform("sum") - v
    L = ((x.groupby(m, sort=False).transform("sum") - x) / otros).where(otros >= min_otros)
    if mu0 is None:
        mu0 = float(L.mean())
    pj = df["player_id"]
    ok = L.notna()
    S, N = _previo(L.fillna(0), pj), _previo(ok.astype(float), pj)
    # ventana: las últimas `window` partidas previas CON L conocido (no las últimas `window` filas)
    def ventana(valores: pd.Series) -> pd.Series:
        c = valores[ok].groupby(pj[ok], sort=False).cumsum()                 # incluye la fila propia
        w = (c - c.groupby(pj[ok], sort=False).shift(window).fillna(0)).reindex(df.index)
        antes = w.groupby(pj, sort=False).shift(1)                          # la fila anterior del jugador
        return antes.groupby(pj, sort=False).ffill().fillna(0)

    s_w, n_w = ventana(L), ventana(ok.astype(float))
    return pd.DataFrame({
        "nivel_lobby": L,
        "nivel_acum": (S + k * mu0) / (N + k),
        f"nivel_v{window}": (s_w + k * mu0) / (n_w + k),
        "n_nivel": N,
    }, index=df.index)
