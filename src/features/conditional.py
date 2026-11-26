"""Rasgos condicionales del jugador: su media de algo, solo en las partidas previas en que pasó otra cosa.

Ejemplo (idea del usuario, 2026-11-13): "hay jugadores que si les das una kill antes del minuto
10 te hacen la partida entera". Eso es la media de (ganó − lo esperado) en las partidas previas
en las que iba adelante: cómo convierte una ventaja. Con valor v y condición c por fila:

    rasgo = (sum_{previas con c} v + k mu) / (#previas con c + k)

encogido hacia `mu` (la media de la población) con peso `k`: con pocas partidas en la
condición, el rasgo queda cerca de la media. Toda suma es `cumsum - fila propia`, así que la
partida que se predice nunca entra. `df` ordenado por `margin.ORDEN`. PROJECT_SUMMARY.md §9.S.35.
"""
import pandas as pd

from src.features.margin import ORDEN, _previo


def prior_conditional_mean(df: pd.DataFrame, value: pd.Series, cond: pd.Series, k: float = 10,
                           mu: float = 0.0) -> pd.DataFrame:
    """`rasgo` (media previa encogida de `value` donde `cond`) y `n` (partidas previas con `cond`).

    Filas donde `value` es NaN no cuentan. `cond` booleano, mismo índice que `df`.
    """
    if not df[ORDEN[0]].is_monotonic_increasing:
        raise ValueError("df tiene que estar ordenado por game_creation")
    pj = df["player_id"]
    w = (cond.astype(bool) & value.notna()).astype(float)
    S = _previo(value.fillna(0) * w, pj)
    N = _previo(w, pj)
    return pd.DataFrame({"rasgo": (S + k * mu) / (N + k), "n": N}, index=df.index)
