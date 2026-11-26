"""
Elo "al revés": reconstruir el LP de un jugador hacia atrás desde una foto del
ranking (PROJECT_SUMMARY.md §9.S.21, "Informe de modelos al revés").

Se parte del LP de una foto (en LP absolutos, ver `rating.lp_absoluto`), se
recorren las partidas de la más nueva a la más vieja y en cada una se deshace el
cambio de LP que el modelo dice que produjo.

* **Regla al revés.** Deshace $+a$ por victoria y $-b$ por derrota, con $a, b$
  del rango en que está el jugador en ese paso. $a, b$ salen de las restas de dos
  fotos por mínimos cuadrados: $\\Delta LP \\approx a\\,\\Delta W - b\\,\\Delta L$.
* **Elo al revés.** Igual, pero multiplica el cambio por la sorpresa del resultado,
  $2(1 - \\Phi(d))$, con $d$ la diferencia de LP entre equipos (congelado en la
  foto) sobre $\\sqrt{10}\\,\\beta$. Con $\\beta \\to \\infty$ vuelve a ser la regla.

Split del problema: train = foto 1 (04/09) → foto 2 (28/10); test = antes del
04/09. `cargar_train` nunca devuelve una partida anterior a la foto 1.
"""

import sqlite3
import zlib
from math import sqrt
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.features.padron import verificar_foto1
from src.features.rating import _Phi, lp_absoluto

FOTO2 = "2026-10-28 00:49:59"
GRUPOS = ["CHALLENGER", "GRANDMASTER", "MASTER", "DIAMOND"]
AZUL = 100
DIA_MS = 86_400_000
DECAY_LP_DIA = 75        # Master+: LP que se pierden por día sin jugar...
DECAY_GRACIA_DIAS = 14   # ...después de 14 días sin jugar (regla dada por el usuario)
NIVELES = ["sin_mejoras", "simple", "completo"]   # mejoras acumuladas de los modelos que aprenden
K_FOLDS = 5


def cargar_foto(db_uri: str, sql: str, params: tuple = ()) -> pd.DataFrame:
    """Una foto por `player_id` con `z` (LP absoluto) y `partidas` (W + L del split).

    Una persona con varias cuentas queda con la cuenta con más partidas.
    """
    with sqlite3.connect(db_uri, uri=True) as con:
        f = pd.read_sql_query(sql, con, params=params)
    f["z"] = [lp_absoluto(t, r, l) for t, r, l in zip(f["tier"], f["rank"], f["league_points"])]
    f["partidas"] = f["wins"] + f["losses"]
    return f.sort_values("partidas").drop_duplicates("player_id", keep="last").set_index("player_id")


def cargar_train(data_dir: Path, foto2: str = FOTO2) -> dict:
    """Fotos y partidas del train (foto 1 ≤ partida < foto 2), filtradas al leer."""
    data_dir = Path(data_dir)
    db = f"file:{(data_dir / 'lol_las.db').resolve()}?mode=ro"
    foto1 = cargar_foto(db, "SELECT m.player_id, p.tier, p.rank, p.league_points, p.wins, p.losses "
                            "FROM players p JOIN player_id_map m ON m.puuid = p.puuid")
    with sqlite3.connect(db, uri=True) as con:
        t1 = verificar_foto1(con)           # foto 1 congelada (§9.S.67)
    t2 = int(pd.Timestamp(foto2, tz="UTC").value // 10**6)
    f2 = cargar_foto(db, "SELECT m.player_id, s.tier, s.rank, s.league_points, s.wins, s.losses, s.fuente "
                         "FROM ladder_snapshots s JOIN player_id_map m ON m.puuid = s.puuid "
                         "WHERE s.snapshot_at = ?", (foto2,))
    partes = []
    for ch in pd.read_csv(data_dir / "full_dataset.csv", chunksize=500_000,
                          usecols=["match_id", "game_creation", "player_id", "team_id", "win"]):
        partes.append(ch[ch["match_id"].str.startswith("LA2_") & (ch["game_creation"] >= t1)
                         & (ch["game_creation"] < t2)])
    partes = pd.concat(partes, ignore_index=True)
    assert partes["game_creation"].min() >= t1, "se coló una partida del test"
    return {"foto1": foto1, "foto2": f2, "partes": partes, "t1": t1, "t2": t2}


def evaluables(foto1: pd.DataFrame, foto2: pd.DataFrame, partes: pd.DataFrame) -> pd.DataFrame:
    """Jugadores con las dos fotos y al menos una partida entre ellas según Riot.

    Columnas con sufijo 1 (foto 1, el target) y 2 (foto 2, el punto de partida),
    `obs`/`obs_w` (partidas y victorias observadas), `dN`/`dW`/`dL` (restas de Riot)
    y `ocultas` (partidas según Riot que no tenemos).
    """
    e = foto1[["tier", "z", "wins", "losses", "partidas"]].join(
        foto2[["tier", "rank", "z", "wins", "losses", "partidas"]], how="inner", lsuffix="1", rsuffix="2")
    obs = partes.groupby("player_id")["win"].agg(obs="size", obs_w="sum")
    e = e.join(obs, how="left").fillna({"obs": 0, "obs_w": 0})
    e["dN"] = e["partidas2"] - e["partidas1"]
    e["dW"], e["dL"] = e["wins2"] - e["wins1"], e["losses2"] - e["losses1"]
    e = e[e["dN"] > 0].dropna(subset=["z1", "z2"]).copy()
    e["ocultas"] = (e["dN"] - e["obs"]).clip(lower=0)
    return e


def cortes(foto: pd.DataFrame) -> Tuple[float, float]:
    """LP absoluto mínimo de Grandmaster y de Challenger en una foto."""
    return (float(foto.loc[foto["tier"] == "GRANDMASTER", "z"].min()),
            float(foto.loc[foto["tier"] == "CHALLENGER", "z"].min()))


def grupo_de(z: float, corte: Tuple[float, float]) -> int:
    """Índice en GRUPOS: 0 Challenger, 1 Grandmaster, 2 Master, 3 Diamond o menos."""
    gm, ch = corte
    return 0 if z >= ch else 1 if z >= gm else 2 if z >= 400 else 3


def rango_de(z, corte: Tuple[float, float]) -> np.ndarray:
    gm, ch = corte
    z = np.asarray(z, dtype=float)
    return np.where(z >= ch, "CHALLENGER", np.where(z >= gm, "GRANDMASTER", np.where(
        z >= 400, "MASTER", np.where(z >= 0, "DIAMOND", "DEBAJO"))))


def division_de(z, corte: Tuple[float, float]) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return np.where(z >= 400, rango_de(z, corte), np.floor(z / 100).astype(int).astype(str))


def ajustar_regla(e: pd.DataFrame, min_partidas: int = 10,
                  decay: pd.Series = None) -> Tuple[Dict[int, float], Dict[int, float]]:
    """$a, b$ por grupo con $\\Delta LP = z_2 - z_1 \\approx a\\,\\Delta W - b\\,\\Delta L$.

    Usa jugadores con al menos `min_partidas` entre fotos y restas no negativas; el
    grupo sale del rango en la foto 2 (Diamond agrupa todo lo que no es Master+).
    """
    fit = e[(e["dN"] >= min_partidas) & (e["dW"] >= 0) & (e["dL"] >= 0)]
    a, b = {}, {}
    for k, t in enumerate(GRUPOS):
        g = fit[fit["tier2"] == t] if t != "DIAMOND" else fit[~fit["tier2"].isin(GRUPOS[:3])]
        y = g["z2"] - g["z1"]
        if decay is not None:   # el LP perdido por decay no sale de las partidas
            y = y + decay.reindex(g.index).fillna(0.0).where((g["z1"] >= 400) | (g["z2"] >= 400), 0.0)
        (a[k], b[k]), *_ = np.linalg.lstsq(np.c_[g["dW"], -g["dL"]], y.to_numpy(), rcond=None)
        a[k], b[k] = float(a[k]), float(b[k])
    return a, b


def sin_deriva(a: Dict[int, float], b: Dict[int, float]) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Regla de suma cero: $a = b = (a + b) / 2$ en cada grupo.

    Con $a > b$ (Master+), un jugador de winrate 50 % gana $(a - b)/2$ LP por partida:
    la deriva de LP. Esta versión la saca para medir cuánto aporta.
    """
    m = {k: (a[k] + b[k]) / 2 for k in a}
    return dict(m), dict(m)


def decaimiento(ms_sin_jugar: float, lp_dia: float = None) -> float:
    """LP que pierde un Master+ por un hueco sin jugar: `lp_dia` (75 por defecto) por día
    completo pasados los 14."""
    lp_dia = DECAY_LP_DIA if lp_dia is None else lp_dia
    return lp_dia * max(0, int(ms_sin_jugar // DIA_MS) - DECAY_GRACIA_DIAS)


def decay_total(partes: pd.DataFrame, t1: int, t2: int) -> pd.Series:
    """LP de decay entre fotos por `player_id`, si hubiera estado en Master+ todo el tiempo.

    Suma `decaimiento` de los huecos foto 1 → primera partida, entre partidas y última
    partida → foto 2. Las partidas no observadas pueden cortar un hueco que acá no se
    ve: en Master+ vemos ~96 % de las partidas.
    """
    t = partes[["player_id", "game_creation"]].sort_values(["player_id", "game_creation"])
    huecos = t.groupby("player_id")["game_creation"].diff()
    total = huecos.map(lambda h: decaimiento(h) if h == h else 0.0).groupby(t["player_id"]).sum()
    g = t.groupby("player_id")["game_creation"]
    total += (g.min() - t1).map(decaimiento) + (t2 - g.max()).map(decaimiento)
    return total


def cambio_oculto(h: float, wr: float, a: float, b: float) -> float:
    """LP esperado que movieron `h` partidas no observadas, con winrate `wr`: $h\,(wr\,a - (1 - wr)\,b)$."""
    return h * (wr * a - (1 - wr) * b)


def _tasa(no_observadas, g: int, i=None) -> float:
    if no_observadas is None:
        return 0.0
    if isinstance(no_observadas, dict):
        return float(no_observadas[g])
    return float(no_observadas if i is None else no_observadas[i])


def al_reves(partes: pd.DataFrame, z_inicio: pd.Series, a: Dict[int, float], b: Dict[int, float],
             corte: Tuple[float, float], modo: str = "regla", beta: float = None,
             no_observadas=None, wr: pd.Series = None, lobby: pd.Series = None,
             k_wr: float = 0.0, k_lobby: float = 0.0, fotos: Tuple[int, int] = None,
             devolver_camino: bool = False):
    """LP reconstruido al llegar a la partida más vieja, por `player_id`.

    `z_inicio`: LP absoluto en la foto de partida (NaN o ausente = sin foto). Un
    jugador sin foto arranca con el LP medio de los que sí la tienen en su primera
    partida (la más nueva) y se recorre igual. `modo`: "regla" o "elo" (con `beta`).

    `no_observadas`: partidas no observadas por cada observada, como dict por grupo o
    Series por `player_id`. Por cada partida observada que se deshace se deshacen
    también esas, con su cambio esperado según `wr` (winrate total por jugador;
    0,5 si falta): si la proporción es 1:6, cada paso descarta 1 + 6 partidas.

    Ajuste por partida (Riot da más LP por victoria y quita menos por derrota cuando el
    MMR está por encima del LP): en cada partida $a' = a + \delta$ y $b' = b - \delta$,
    con $\delta = k_{wr}\,(wr - 0{,}5) + k_{lobby}\,(\mathrm{lobby} - z_0)/100$. `lobby` es
    el LP medio de los otros 9 de esa partida (índice de `partes`, de `lobby_por_fila`)
    y $z_0$ el LP del jugador en la foto, congelado: con el LP del paso, la recorrida al
    revés amplifica el error partida a partida y diverge. $a', b'$ no bajan de 1.

    `fotos` = (t1, t2) en ms: aplica el decay de Master+ (`decaimiento`) a cada hueco
    sin jugar (foto 2 → partida más nueva, entre partidas, partida más vieja → foto 1).
    Hacia atrás se le devuelve el LP perdido si el jugador está en Master+ después del
    hueco.

    `devolver_camino`: devuelve además el LP de cada jugador al jugar cada partida (antes
    de la partida), como Series con el índice de `partes`.
    """
    if lobby is not None:
        partes = partes.assign(_lobby=lobby.reindex(partes.index))
    df = partes.sort_values(["game_creation", "match_id", "team_id"], ascending=[False, False, True],
                            kind="mergesort")
    orig = df.index.to_numpy()
    df = df.reset_index(drop=True)
    lp_fila = np.full(len(df), np.nan)
    jug, idx_np = np.unique(df["player_id"].to_numpy(dtype=str), return_inverse=True)
    mu = [float(x) for x in z_inicio.reindex(jug).to_numpy(dtype=float)]
    z0 = list(mu)   # fuerza congelada en la foto, para la sorpresa del Elo
    wr_j = (wr.reindex(jug).fillna(0.5).to_numpy(float) if wr is not None else np.full(len(jug), 0.5)).tolist()
    h_j = (no_observadas.reindex(jug).fillna(0.0).to_numpy(float).tolist()
           if isinstance(no_observadas, pd.Series) else no_observadas)
    idx, win = idx_np.tolist(), df["win"].to_numpy().tolist()
    lob = df["_lobby"].to_numpy(float).tolist() if lobby is not None else None
    tiempo = df["game_creation"].to_numpy().tolist()
    ultimo = [fotos[1]] * len(jug) if fotos is not None else None   # hora del evento más nuevo ya deshecho
    _, ini = np.unique(df["match_id"].to_numpy(dtype=str), return_index=True)
    ini = np.sort(ini)
    for s, t in zip(ini.tolist(), np.r_[ini[1:], len(df)].tolist()):
        ids = [idx[r] for r in range(s, t)]
        conocidos = [mu[i] for i in ids if mu[i] == mu[i]]
        relleno = sum(conocidos) / len(conocidos) if conocidos else 0.0
        for i in ids:
            if mu[i] != mu[i]:
                mu[i] = z0[i] = relleno
        if ultimo is not None:
            for r in range(s, t):
                i = idx[r]
                if mu[i] >= 400:
                    mu[i] += decaimiento(ultimo[i] - tiempo[r])
                ultimo[i] = tiempo[r]
        gan = [r for r in range(s, t) if win[r] == 1]
        per = [r for r in range(s, t) if win[r] != 1]
        if not gan or not per:
            for r in range(s, t):
                lp_fila[r] = mu[idx[r]]
            continue
        k = 1.0
        if modo == "elo":
            k = 2 * (1 - _Phi((sum(z0[idx[r]] for r in gan) - sum(z0[idx[r]] for r in per)) / (sqrt(10) * beta)))
        for r in gan + per:
            i = idx[r]
            g = grupo_de(mu[i], corte)
            d = k_wr * (wr_j[i] - 0.5)
            if lob is not None and lob[r] == lob[r]:
                d += k_lobby * (lob[r] - z0[i]) / 100
            oculto = cambio_oculto(_tasa(h_j, g, i), wr_j[i], a[g], b[g])
            if win[r] == 1:
                mu[i] -= k * max(a[g] + d, 1.0) + oculto
            else:
                mu[i] += k * max(b[g] - d, 1.0) - oculto
            lp_fila[r] = mu[i]
    if ultimo is not None:
        for i in range(len(jug)):
            if mu[i] >= 400:
                mu[i] += decaimiento(ultimo[i] - fotos[0])
    if devolver_camino:
        return pd.Series(mu, index=jug), pd.Series(lp_fila, index=orig).reindex(partes.index)
    return pd.Series(mu, index=jug)


def camino(partidas: pd.DataFrame, z2: float, z1_pred: float, a: Dict[int, float], b: Dict[int, float],
           corte: Tuple[float, float], no_observadas=0.0, wr: float = 0.5,
           fotos: Tuple[int, int] = None) -> np.ndarray:
    """LP antes de cada partida de un jugador, anclado en la foto 2 y en `z1_pred`.

    `partidas`: las del jugador (`game_creation`, `win`). Se deshace la regla de la
    más nueva a la más vieja y la diferencia entre `z1_pred` y donde termina la regla
    se reparte en proporción a las partidas deshechas: el camino arranca en
    `z1_pred` y llega a `z2`. Devuelve el LP antes de cada partida, de la más vieja a
    la más nueva, seguido de `z2`. `no_observadas` y `wr` como en `al_reves` (un número
    o un dict por grupo). `fotos` = (t1, t2): decay de Master+ como en `al_reves`.
    """
    p = partidas.sort_values("game_creation", ascending=False)
    win, ts = p["win"].to_numpy(), p["game_creation"].to_numpy()
    z, antes = float(z2), []
    ultimo = fotos[1] if fotos is not None else None
    for w, t in zip(win, ts):
        if ultimo is not None:
            z += decaimiento(ultimo - t) if z >= 400 else 0.0
            ultimo = t
        g = grupo_de(z, corte)
        z = (z - a[g] if w == 1 else z + b[g]) - cambio_oculto(_tasa(no_observadas, g), wr, a[g], b[g])
        antes.append(z)
    antes = np.array(antes[::-1] + [float(z2)])
    if len(win):
        inicial = decaimiento(ultimo - fotos[0]) if ultimo is not None and antes[0] >= 400 else 0.0
        antes = antes + (z1_pred - inicial - antes[0]) * (1 - np.arange(len(antes)) / len(win))
    return antes


def adelante(partidas: pd.DataFrame, z1: float, a: Dict[int, float], b: Dict[int, float],
             corte: Tuple[float, float], no_observadas=0.0, wr: float = 0.5) -> np.ndarray:
    """LP antes de cada partida avanzando desde la foto 1 (para quien no tiene foto 2).

    Suma $+a$ por victoria y $-b$ por derrota, más el cambio esperado de las no
    observadas que acompañan a cada partida (como en `al_reves`). Devuelve el LP antes
    de cada partida, de la más vieja a la más nueva, seguido del LP después de la última.
    """
    win = partidas.sort_values("game_creation")["win"].to_numpy()
    z, antes = float(z1), [float(z1)]
    for w in win:
        g = grupo_de(z, corte)
        z = (z + a[g] if w == 1 else z - b[g]) + cambio_oculto(_tasa(no_observadas, g), wr, a[g], b[g])
        antes.append(z)
    return np.array(antes)


def calibrar_lobby(lobby: pd.Series, z2: pd.Series) -> Tuple[float, float, float]:
    """Medición del suavizador: lobby de una partida $\\approx c_0 + c_1 z + \\varepsilon$, $\\varepsilon \\sim N(0, R)$.

    `lobby`: el de las partidas más nuevas de cada jugador (índice `player_id`), donde
    su LP es casi el de la foto 2. Devuelve $(c_0, c_1, R)$.
    """
    zz = z2.reindex(lobby.index)
    ok = lobby.notna() & zz.notna()
    z = zz[ok].to_numpy(float)
    c1, c0 = np.polyfit(z, lobby[ok].to_numpy(float), 1)
    return float(c0), float(c1), float(np.var(lobby[ok].to_numpy(float) - (c0 + c1 * z)))


def calibrar_lobby_por_grupo(lobby: pd.Series, z2: pd.Series, corte: Tuple[float, float],
                             escala_R: Dict[int, float] = None) -> Dict[int, Tuple[float, float, float]]:
    """`calibrar_lobby` por grupo de rango (curva del lobby por tramos, no una sola recta).

    `escala_R` multiplica el ruido de cada grupo (para elegirlo por verosimilitud).
    """
    zz = z2.reindex(lobby.index)
    g = zz.map(lambda z: grupo_de(z, corte) if z == z else -1)
    out = {}
    for k in range(len(GRUPOS)):
        m = g == k
        out[k] = calibrar_lobby(lobby[m], z2) if m.sum() >= 50 else calibrar_lobby(lobby, z2)
        if escala_R:
            out[k] = (out[k][0], out[k][1], out[k][2] * escala_R.get(k, 1.0))
    return out


def kalman_al_reves(win: np.ndarray, lobby: np.ndarray, z2: float, a: Dict[int, float], b: Dict[int, float],
                    corte: Tuple[float, float], medicion: Tuple[float, float, float], q: float = 25.0,
                    ocultas_por_partida=0.0, wr: float = 0.5, tiempos: np.ndarray = None,
                    fotos: Tuple[int, int] = None, decay_lp_dia: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """Suavizador de Kalman (Rauch-Tung-Striebel) del LP, recorriendo al revés desde la foto 2.

    Estado: el LP antes de cada partida. `win` y `lobby` van de la partida más nueva a
    la más vieja. Se arranca en `z2` con varianza 0 y en cada partida:

    * **observada:** se deshace su resultado ($-a$ si ganó, $+b$ si perdió), más un
      ruido de proceso `q`;
    * **no observadas:** `ocultas_por_partida` partidas sin resultado conocido (número
      o dict por grupo). Cada una mueve la media en $-(p\\,a - (1 - p)\\,b)$, con $p$ =
      `wr`, y suma $p(1 - p)(a + b)^2 + q$ de varianza;
    * **medición:** el lobby de la partida, con `medicion` = $(c_0, c_1, R)$ de
      `calibrar_lobby` (NaN = sin medición).

    `medicion` puede ser un dict por grupo (`calibrar_lobby_por_grupo`): la curva del
    lobby por tramos según el rango del paso. `q` también puede ser un dict por grupo.

    Con `tiempos` (hora de cada partida, en el mismo orden) y `fotos` = (t1, t2), el
    decay de Master+ (`decay_lp_dia` por día) entra como control determinístico en cada
    hueco sin jugar. El de la
    partida más vieja a la foto 1 no está en la salida: sumarlo con `decaimiento`.

    Devuelve la media y la varianza suavizadas del LP antes de cada partida, de la más
    vieja a la más nueva, seguidas de `z2` (varianza 0): el mismo formato que `camino`.
    """
    n = len(win)
    xp, Pp, xf, Pf = (np.zeros(n + 1) for _ in range(4))
    xf[0], x, P = z2, float(z2), 0.0
    ultimo = fotos[1] if fotos is not None and tiempos is not None else None
    for k in range(n):
        if ultimo is not None:
            x += decaimiento(ultimo - tiempos[k], decay_lp_dia) if x >= 400 else 0.0
            ultimo = tiempos[k]
        g = grupo_de(x, corte)
        c0, c1, R = medicion[g] if isinstance(medicion, dict) else medicion
        qg = q[g] if isinstance(q, dict) else q
        h = ocultas_por_partida[g] if isinstance(ocultas_por_partida, dict) else float(ocultas_por_partida)
        x -= a[g] if win[k] == 1 else -b[g]
        x -= h * (wr * a[g] - (1 - wr) * b[g])
        P += qg + h * (wr * (1 - wr) * (a[g] + b[g]) ** 2 + qg)
        xp[k + 1], Pp[k + 1] = x, P
        if lobby[k] == lobby[k]:
            K = P * c1 / (c1 * c1 * P + R)
            x += K * (lobby[k] - c0 - c1 * x)
            P -= K * c1 * P
        xf[k + 1], Pf[k + 1] = x, P
    xs, Ps = xf.copy(), Pf.copy()
    for k in range(n - 1, -1, -1):
        J = Pf[k] / Pp[k + 1] if Pp[k + 1] > 0 else 0.0
        xs[k] = xf[k] + J * (xs[k + 1] - xp[k + 1])
        Ps[k] = Pf[k] + J * J * (Ps[k + 1] - Pp[k + 1])
    return xs[::-1].copy(), Ps[::-1].copy()


def kalman_jugadores(e: pd.DataFrame, partes: pd.DataFrame, z: pd.Series, a: Dict[int, float],
                     b: Dict[int, float], corte: Tuple[float, float], q: float = 25.0) -> Tuple[pd.Series, pd.Series]:
    """Media y varianza del Kalman en la foto 1 por jugador de `e` (el "Kalman original" de §9.S.21).

    Medición: el lobby de cada partida con el LP de los otros 9 en la foto `z`, una sola
    recta calibrada con las 3 partidas más nuevas de cada jugador. No observadas: la tasa
    de su rango en `e` (no observadas por observada; existe en el test). Winrate: el
    observado, suavizado. Un jugador sin partidas observadas queda en `z2` con varianza NaN.
    """
    lob = lobby_por_fila(partes, z)["lobby"]
    pk = partes[["player_id", "game_creation", "win"]].assign(lobby=lob)
    pk = pk[pk["player_id"].isin(e.index)].sort_values(["player_id", "game_creation"], ascending=[True, False])
    seq = {pid: (g["win"].to_numpy(), g["lobby"].to_numpy()) for pid, g in pk.groupby("player_id", sort=False)}
    medicion = calibrar_lobby(pk.groupby("player_id").head(3).set_index("player_id")["lobby"], e["z2"])
    g2 = e["z2"].map(lambda v: grupo_de(v, corte))
    tasa = (e.groupby(g2)["ocultas"].sum() / e.groupby(g2)["obs"].sum()).to_dict()
    wr = (e["obs_w"] + 10) / (e["obs"] + 20)
    media, var = {}, {}
    for pid in e.index:
        if pid not in seq:
            media[pid], var[pid] = e.at[pid, "z2"], np.nan
            continue
        m, v = kalman_al_reves(*seq[pid], e.at[pid, "z2"], a, b, corte, medicion, q=q,
                               ocultas_por_partida=tasa, wr=wr[pid])
        media[pid], var[pid] = m[0], v[0]
    return pd.Series(media).reindex(e.index), pd.Series(var).reindex(e.index)


def lobby_por_fila(partes: pd.DataFrame, z: pd.Series, por_fila: bool = False) -> pd.DataFrame:
    """Por jugador-partida: LP medio (en la foto `z`) de aliados, rivales y de los 9 otros.

    También `conocidos`: cuántos de los otros 9 tienen foto. NaN si no hay ninguno.
    """
    p = partes[["match_id", "team_id", "player_id"]].copy()
    # por_fila: `z` trae el LP de cada jugador en esa partida (índice de `partes`), no su foto.
    zi = z.reindex(partes.index) if por_fila else p["player_id"].map(z)
    k = zi.notna().astype(float)
    zf = zi.fillna(0.0)
    tmp = pd.DataFrame({"m": p["match_id"], "t": p["team_id"], "zf": zf, "k": k})
    eq = tmp.groupby(["m", "t"])[["zf", "k"]].transform("sum")
    tot = tmp.groupby("m")[["zf", "k"]].transform("sum")
    al_s, al_n = eq["zf"] - zf, eq["k"] - k
    en_s, en_n = tot["zf"] - eq["zf"], tot["k"] - eq["k"]
    return pd.DataFrame({"aliados": al_s / al_n.replace(0, np.nan), "rivales": en_s / en_n.replace(0, np.nan),
                         "lobby": (al_s + en_s) / (al_n + en_n).replace(0, np.nan), "conocidos": al_n + en_n},
                        index=partes.index)


def variables_estaticas(e: pd.DataFrame, partes: pd.DataFrame, z: pd.Series, pred_regla: pd.Series,
                        lob: pd.DataFrame = None) -> pd.DataFrame:
    """Variables por jugador para corregir el residuo de la regla (todas existen en el test).

    `ocultas` usa el W/L del extremo target: en el test solo se calcula hasta el
    inicio del split.
    """
    lob = lobby_por_fila(partes, z) if lob is None else lob   # `lob`: p. ej. con el LP predicho por partida
    p = partes[["player_id", "game_creation"]].assign(lobby=lob["lobby"]).sort_values("game_creation")
    gl = p.groupby("player_id")["lobby"]
    X = pd.DataFrame({"z2": e["z2"], "pred_regla": pred_regla, "obs": e["obs"], "obs_w": e["obs_w"],
                      "wr_obs": (e["obs_w"] + 10) / (e["obs"] + 20), "lobby": gl.mean().reindex(e.index),
                      "lobby_viejo": gl.apply(lambda s: s.head(10).mean()).reindex(e.index),
                      "lobby_reciente": gl.apply(lambda s: s.tail(10).mean()).reindex(e.index),
                      "partidas2": e["partidas2"], "wr_temporada": e["wins2"] / e["partidas2"],
                      "ocultas": e["ocultas"]})
    X["gap"] = X["lobby"] - X["z2"]
    return X


def folds_por_jugador(ids, k: int = K_FOLDS) -> np.ndarray:
    """Fold de cada jugador (0..k-1), fijo por un hash de su id: el mismo en todos los scripts."""
    return np.array([zlib.crc32(str(i).encode()) % k for i in ids])


def matrices_por_nivel(d: dict, e: pd.DataFrame, a: Dict[int, float], b: Dict[int, float],
                       corte: Tuple[float, float]) -> Dict[str, Tuple[pd.Series, pd.DataFrame]]:
    """Por nivel de `NIVELES`, la base ($\\hat z_1$ sin aprender) y las variables de los boostings.

    * `sin_mejoras`: base $z_2$ (se aprende el cambio); z2, W + L del split y partidas y
      victorias observadas.
    * `simple`: base la regla al revés (con deriva); `variables_estaticas` con el lobby de
      la foto 2, sin `ocultas`.
    * `completo`: como `simple`, pero el lobby con el LP de cada uno de los otros 9 en esa
      partida (camino de la regla), más la media y el desvío del Kalman original y el decay
      total de Master+.

    `d`: lo que devuelve `cargar_train`. Ninguna variable usa el target.
    """
    partes, z = d["partes"], d["foto2"]["z"]
    regla = al_reves(partes, z, a, b, corte).reindex(e.index).fillna(e["z2"])
    _, lp_fila = al_reves(partes, z, a, b, corte, devolver_camino=True)
    km, kv = kalman_jugadores(e, partes, z, a, b, corte)
    decay = decay_total(partes, d["t1"], d["t2"]).reindex(e.index).fillna(0.0)
    completo = (variables_estaticas(e, partes, z, regla, lob=lobby_por_fila(partes, lp_fila, por_fila=True))
                .drop(columns="ocultas").assign(kal_media=km - e["z2"], kal_sd=np.sqrt(kv), decay=decay))
    return {"sin_mejoras": (e["z2"].astype(float), e[["z2", "partidas2", "obs", "obs_w"]].astype(float)),
            "simple": (regla, variables_estaticas(e, partes, z, regla).drop(columns="ocultas")),
            "completo": (regla, completo)}
