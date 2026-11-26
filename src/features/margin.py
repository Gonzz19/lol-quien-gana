"""Margen previo por jugador: por cuánto le fue bien en sus partidas ANTERIORES.

A diferencia de la huella (`fingerprint.residualize`), acá se descuenta el campeón y la
línea pero NO el resultado: la idea es medir fuerza, no estilo. Para cada estadística x:

    r = (x - media_previa(campeón, línea)) / desvío(línea)
    puntaje antes de la partida = S / (N + k_j)

con S y N la suma de r y la cantidad de partidas previas del jugador (todas, o las últimas
`window`). La media del campeón en la línea también usa solo partidas anteriores, encogida
hacia la media de la línea con peso `k_cl`.

Sin leakage por construcción: toda suma es `cumsum - valor propio`. La media y el desvío por
línea se pueden pasar ya calculados (`escala`); si no, salen de todo `df` (son una escala).
Detalle en PROJECT_SUMMARY.md §9.S.24.
"""
from typing import Dict, List, Optional, Tuple

import pandas as pd

ORDEN = ["game_creation", "match_id", "team_id", "lane"]


def _previo(s: pd.Series, claves) -> pd.Series:
    """Suma de las filas anteriores del mismo grupo (sin la fila actual)."""
    return s.groupby(claves, sort=False).cumsum() - s


def _ultimas(S: pd.Series, jugador: pd.Series, window: int) -> pd.Series:
    """De una suma previa acumulada, la suma de las últimas `window` filas previas."""
    return S - S.groupby(jugador, sort=False).shift(window).fillna(0)


def lane_diffs(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Agrega `d_<col>`: el valor propio menos el del rival directo de línea.

    Queda vacío si falta alguno de los dos.
    """
    g = df.groupby(["match_id", "lane"])
    return df.assign(**{"d_" + c: 2 * df[c] - g[c].transform("sum", min_count=2) for c in cols})


def lane_scale(df: pd.DataFrame, stats: List[str]) -> Dict[str, Tuple[pd.Series, pd.Series]]:
    """Media y desvío por línea de cada estadística."""
    g = df.groupby("lane")
    return {c: (g[c].mean(), g[c].std()) for c in stats}


def _residuo(df: pd.DataFrame, c: str, k_cl: float, escala: dict) -> Tuple[pd.Series, pd.Series]:
    """r de cada fila (0 si falta el dato) y el indicador de dato presente."""
    cl = [df["champion_id"], df["lane"]]
    x, v = df[c], df[c].notna().astype(float)
    mu, sd = df["lane"].map(escala[c][0]), df["lane"].map(escala[c][1])
    media_cl = (_previo(x.fillna(0), cl) + k_cl * mu) / (_previo(v, cl) + k_cl)
    return ((x - media_cl) / sd).fillna(0), v


def prior_margin(df: pd.DataFrame, stats: List[str], k_cl: float = 20, k_j: float = 5,
                 window: int = 20, escala: Optional[dict] = None) -> pd.DataFrame:
    """Puntajes de margen previos a cada fila de `df` (una fila por jugador-partida).

    `df` tiene que venir ordenado por `ORDEN`. Devuelve, con el mismo índice,
    `acum_<stat>` y `v<window>_<stat>` por estadística.
    """
    if not df[ORDEN[0]].is_monotonic_increasing:
        raise ValueError("df tiene que estar ordenado por game_creation")
    escala = escala or lane_scale(df, stats)
    pj = df["player_id"]
    out = {}
    for c in stats:
        r, v = _residuo(df, c, k_cl, escala)
        S, N = _previo(r, pj), _previo(v, pj)
        out[f"acum_{c}"] = S / (N + k_j)
        out[f"v{window}_{c}"] = _ultimas(S, pj, window) / (_ultimas(N, pj, window) + k_j)
    return pd.DataFrame(out, index=df.index)


def prior_margin_by(df: pd.DataFrame, stats: List[str], keys: List[str], nombre: str,
                    k_cl: float = 20, k_j: float = 5, k_g: float = 10,
                    escala: Optional[dict] = None) -> pd.DataFrame:
    """Margen previo del jugador dentro de un grupo (p. ej. su campeón o su línea), como desvío.

    Con m_p el margen acumulado del jugador (el `acum_` de `prior_margin`) y S_pg, N_pg las
    sumas previas del jugador en el grupo `keys`:
        m_pg = (S_pg + k_g m_p) / (N_pg + k_g),    devuelve  `<nombre>_<stat>` = m_pg - m_p
    Sin partidas previas en el grupo vale 0: no agrega nada al margen general.
    """
    if not df[ORDEN[0]].is_monotonic_increasing:
        raise ValueError("df tiene que estar ordenado por game_creation")
    escala = escala or lane_scale(df, stats)
    pj = df["player_id"]
    grupo = [pj] + [df[k] for k in keys]
    out = {}
    for c in stats:
        r, v = _residuo(df, c, k_cl, escala)
        m_p = _previo(r, pj) / (_previo(v, pj) + k_j)
        out[f"{nombre}_{c}"] = (_previo(r, grupo) + k_g * m_p) / (_previo(v, grupo) + k_g) - m_p
    return pd.DataFrame(out, index=df.index)


def prior_margin_vs_rival(df: pd.DataFrame, stats: List[str], k_cl: float = 20, k_j: float = 5,
                          window: int = 20, escala: Optional[dict] = None) -> pd.DataFrame:
    """Margen previo ajustado por la fuerza del rival directo de línea.

    Con r el residuo de `prior_margin` y m = S / (N + k_j) el margen acumulado previo:
        r2 = r + m_rival
    donde m_rival es el margen previo, en la misma estadística, del rival de línea de esa
    partida: sacarle +1 a un rival que suele sacar +1 vale +2. El puntaje es la media previa
    de r2 (`adj_acum_<stat>`, `adj_v<window>_<stat>`), encogida igual que en `prior_margin`.
    Pensado para las estadísticas contra el rival (`d_*` de `lane_diffs`). Si en la partida no
    hay exactamente un rival en la línea, r2 = r.
    """
    if not df[ORDEN[0]].is_monotonic_increasing:
        raise ValueError("df tiene que estar ordenado por game_creation")
    escala = escala or lane_scale(df, stats)
    pj = df["player_id"]
    ml = [df["match_id"], df["lane"]]
    par = df.groupby(["match_id", "lane"], sort=False)["player_id"].transform("size") == 2
    out = {}
    for c in stats:
        r, v = _residuo(df, c, k_cl, escala)
        m = _previo(r, pj) / (_previo(v, pj) + k_j)
        m_rival = (m.groupby(ml, sort=False).transform("sum") - m).where(par, 0.0)
        r2 = r + m_rival * v
        S, N = _previo(r2, pj), _previo(v, pj)
        out[f"adj_acum_{c}"] = S / (N + k_j)
        out[f"adj_v{window}_{c}"] = _ultimas(S, pj, window) / (_ultimas(N, pj, window) + k_j)
    return pd.DataFrame(out, index=df.index)


def prior_shift(df: pd.DataFrame, stats: List[str], mu: pd.DataFrame, k_j: float = 5,
                window: int = 20) -> pd.DataFrame:
    """Lo que cambia el margen previo si se encoge hacia un prior `mu` en vez de hacia 0.

    `prior_margin` da m = S / (N + k_j). Con un valor esperado mu (por fila y estadística,
    p. ej. según el nivel del jugador), el margen encogido hacia él es
        m* = (S + k_j mu) / (N + k_j) = m + k_j mu / (N + k_j),
    así que alcanza con sumar `pri_acum_<stat>` = k_j mu / (N + k_j) y lo mismo con las
    últimas `window` (`pri_v<window>_<stat>`). Pesa mucho con pocas partidas previas y se
    apaga con muchas. N cuenta solo las partidas previas del jugador con la estadística.
    `mu` tiene que estar disponible antes de la partida; columnas = `stats`.
    """
    if not df[ORDEN[0]].is_monotonic_increasing:
        raise ValueError("df tiene que estar ordenado por game_creation")
    pj = df["player_id"]
    out = {}
    for c in stats:
        N = _previo(df[c].notna().astype(float), pj)
        out[f"pri_acum_{c}"] = k_j * mu[c] / (N + k_j)
        out[f"pri_v{window}_{c}"] = k_j * mu[c] / (_ultimas(N, pj, window) + k_j)
    return pd.DataFrame(out, index=df.index)


def prior_winrate(df: pd.DataFrame, window: int = 20, k: float = 10) -> pd.DataFrame:
    """Winrate previo del jugador, encogido hacia 0,5 y centrado: acumulado y últimas `window`."""
    pj = df["player_id"]
    w = df["win"].astype(float)
    W, N = _previo(w, pj), _previo(pd.Series(1.0, index=df.index), pj)
    return pd.DataFrame({
        "wr_acum": (W + k / 2) / (N + k) - 0.5,
        f"wr_v{window}": (_ultimas(W, pj, window) + k / 2) / (_ultimas(N, pj, window) + k) - 0.5,
    }, index=df.index)
