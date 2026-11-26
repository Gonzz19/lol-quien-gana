"""
Forma del día por jugador y partida (PROJECT_SUMMARY.md §9.S.16).

Para cada fila jugador-partida, cómo le venía yendo a ese jugador **en ese día de
juego, antes de esa partida**.

* **Día de juego:** de 08:00 a 08:00 hora de Argentina
  (`dia_juego = fecha(inicio − 8 h)`): una partida a las 03:00 del martes es de la
  sesión del lunes.
* **Sin leakage del resultado propio:** todo se calcula con las partidas
  anteriores del mismo día. La partida actual nunca entra en su propio índice.
* **Índice resumen** (winrate del día suavizado hacia el nivel habitual del
  jugador, centrado en ese nivel):

  $$\\text{indice} = \\frac{G + C \\cdot WR_j}{n + C} - WR_j$$

  con $G$ y $n$ las ganadas y jugadas previas del día. Vale 0 sin partidas previas
  y tiende a "cuánto mejor o peor que lo habitual" al acumular partidas. $WR_j$
  tiene que salir **solo del train** (`baseline_winrate`), para no meter el test
  por la puerta de atrás.
"""

from typing import Optional

import numpy as np
import pandas as pd

DAY_START_HOUR = 8
C_DEFAULT = 3.0


K_BASELINE = 10.0


def baseline_winrate(parts_train: pd.DataFrame, k: float = K_BASELINE) -> pd.Series:
    """Winrate habitual por `player_id`, calculado con las partidas de train.

    Suavizado hacia el winrate global del train con $k$ partidas "virtuales":
    $(W + k \cdot \bar{WR}) / (N + k)$. Sin esto, un jugador con 1 o 2 partidas en el
    train tendría winrate habitual de 0 o 1, y su índice del día no tendría sentido.
    """
    glob = float(parts_train["win"].mean()) if len(parts_train) else 0.5
    g = parts_train.groupby("player_id")["win"].agg(["sum", "size"])
    return (g["sum"] + k * glob) / (g["size"] + k)


def daily_form(parts: pd.DataFrame, baseline: Optional[pd.Series] = None,
               c: float = C_DEFAULT, day_start_hour: int = DAY_START_HOUR,
               expected: Optional[pd.DataFrame] = None,
               complete_days: Optional[tuple] = None) -> pd.DataFrame:
    """Columnas de forma del día para cada fila de `parts`.

    `parts`: `match_id`, `player_id`, `win`, `game_creation` (ms), `game_duration` (s).
    `baseline`: winrate habitual por `player_id` (del train). Faltante → winrate
    global del baseline (o 0,5).
    `expected` + `complete_days`: opcionales. `expected` son las partidas que Riot
    listó para cada jugador (`player_id`, `match_id`) y `complete_days` =
    `(primer_dia, ultimo_dia)` de juego cubiertos enteros por ese listado. Con
    ellos, `dia_completo` es True si el jugador tiene bajadas todas sus partidas
    listadas, False si le falta alguna, y NaN (desconocido) fuera de esos días o
    para jugadores sin listado: ahí no hay forma de saber cuántas jugó.
    """
    df = parts[["match_id", "player_id", "win", "game_creation", "game_duration"]].copy()
    inicio = pd.to_datetime(df["game_creation"], unit="ms") - pd.Timedelta(hours=3)
    df["dia_juego"] = (inicio - pd.Timedelta(hours=day_start_hour)).dt.strftime("%Y-%m-%d")
    df = df.sort_values(["player_id", "game_creation", "match_id"], kind="mergesort")

    keys = [df["player_id"], df["dia_juego"]]
    g = df.groupby(keys, sort=False)
    df["n_previas_dia"] = g.cumcount()
    df["ganadas_previas_dia"] = g["win"].cumsum() - df["win"]
    df["perdidas_previas_dia"] = df["n_previas_dia"] - df["ganadas_previas_dia"]

    # Racha con signo al llegar a la partida (+3 = viene de 3 victorias seguidas ese día).
    prev = g["win"].shift(1)
    first = prev.isna()
    cambio = first | (prev != g["win"].shift(2)) | g["win"].shift(2).isna()
    run = cambio.cumsum()
    largo = df.groupby(run).cumcount() + 1
    df["racha"] = np.where(first, 0, np.where(prev == 1, largo, -largo)).astype(int)

    # Descanso desde el fin de su partida anterior del mismo día; vacío en la primera
    # del día (no hay "anterior" en la sesión), marcada aparte con `primera_del_dia`.
    fin_prev = (df["game_creation"] + df["game_duration"] * 1000).groupby(keys).shift(1)
    df["primera_del_dia"] = df["n_previas_dia"] == 0
    df["minutos_desde_ultima"] = ((df["game_creation"] - fin_prev) / 60000).round(1)

    wr_global = float(baseline.mean()) if baseline is not None and len(baseline) else 0.5
    wr = (df["player_id"].map(baseline) if baseline is not None else pd.Series(np.nan, index=df.index))
    wr = wr.fillna(wr_global)
    df["winrate_habitual"] = wr.round(4)
    df["indice_forma_dia"] = ((df["ganadas_previas_dia"] + c * wr) / (df["n_previas_dia"] + c) - wr).round(4)

    con_completitud = expected is not None and complete_days is not None
    if con_completitud:
        presentes = set(zip(df["match_id"], df["player_id"]))
        # Una partida listada y no bajada no tiene fecha (no se sabe en qué día cae),
        # así que el jugador entero queda sin días garantizados hasta bajarla.
        incompletos = {p for m, p in zip(expected["match_id"], expected["player_id"])
                       if (m, p) not in presentes}
        con_listado = set(expected["player_id"])
        desde, hasta = complete_days
        dentro = (df["dia_juego"] >= desde) & (df["dia_juego"] <= hasta) & df["player_id"].isin(con_listado)
        completo = ~df["player_id"].isin(incompletos)
        df["dia_completo"] = pd.Series(np.where(dentro, completo, None), index=df.index, dtype=object)
    cols = ["match_id", "player_id", "dia_juego", "n_previas_dia", "ganadas_previas_dia",
            "perdidas_previas_dia", "racha", "primera_del_dia", "minutos_desde_ultima", "winrate_habitual",
            "indice_forma_dia"] + (["dia_completo"] if con_completitud else [])
    return df[cols].reset_index(drop=True)
