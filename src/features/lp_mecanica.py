"""Modelo mecánico del paso de LP y reconstrucción por tramos (PROJECT_SUMMARY.md §9.S.40).

En LAS (`la2`), el cambio de LP de una partida es casi determinista por partes:

    Δz = m * (K + δ_i)        si gana (m = 2 con Aegis of Valor, 1 si no)
       = -(K - δ_i)           si pierde (0 si hay devolución por equipo)

con pisos en Master 0 (63 % queda en 400 al perder) y decay por banco de días
(Master+: 1 día por partida, tope 14, −75/día hasta 375; Diamond: 7/partida,
tope 28, −50/día hasta 275).

Componentes exportados:
* `ParametrosPaso`, `cargar_eventos_vivo_las`, `ajustar_parametros_las`,
  `ajustar_k_diamond`: ajuste con LAS (`enlazada`), sin mezclar los otros
  servidores reiniciados el 29/04 (BR/NA/EUW tienen ±30).
* `decaer_dias`, `avanzar_banco`, `recargar_banco`: decay exacto por banco.
* `recorrer_adelante`: trayectoria causal hacia adelante (o con saldo W/L
  conocido entre fotos), encogiendo el lobby según la incertidumbre.
* `suavizar_en_grilla`: filtro/suavizador 1D forward–backward sin invertir la
  dinámica (operadores adjuntos en grilla de LP), con bandas `[z_p10, z_p90]`.
* `estimar_mapa_tramo`: reconstrucción por tramos en dos pasadas (recalcula el
  lobby de los otros 9 jugadores con las estimaciones de la pasada 1), con
  `modo="descriptivo"` o `modo="causal"`.
"""

from __future__ import annotations

import contextlib
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.features.elo_reverso import DIA_MS
from src.features.rating import MASTER_BASE, lp_absoluto
from src.lp_vivo import agregar_lobby

BANDAS_RAREZA = (-0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 1.01)
PISO_MASTER_DECAY = 375.0
PISO_DIAMOND_DECAY = 275.0
TOPE_BANCO_MASTER = 14.0
TOPE_BANCO_DIAMOND = 28.0
DECAY_MASTER_DIA = 75.0
DECAY_DIAMOND_DIA = 50.0


@dataclass(frozen=True)
class ParametrosPaso:
    """Parámetros del paso de LP en LAS."""

    k_master: float = 20.0
    k_diamond: float = 18.75
    perdon: float = 0.045
    piso_quieto: float = 0.63
    piso_baja: float = 50.0
    doble_base: float = 0.067
    doble_por_rol: Dict[str, Tuple[float, ...]] = field(default_factory=dict)
    gap_nodos: Tuple[float, ...] = (-600.0, -300.0, -150.0, -50.0, 50.0, 150.0, 300.0, 500.0)
    delta_nodos: Tuple[float, ...] = (-1.5, -1.2, -0.9, -0.3, 0.5, 2.7, 5.1, 9.6)
    c_lobby: float = 0.5      # peso EB en escala n_eff = sum(n_lob/9) (~4.5 co-jugadores)
    sigma_paso: float = 2.5   # difusión base por partida observada en la grilla (LP)


def delta_de(gap, params: ParametrosPaso):
    """Ajuste δ(gap) acotado e interpolado por tramos sobre `params.gap_nodos`."""
    return np.interp(gap, params.gap_nodos, params.delta_nodos)


def encoger_gap(lobby, z, n_obs_lobby, params: ParametrosPaso):
    """Gap `lobby - z` encogido a 0 cuando hay pocos conocidos o cobertura baja.

    Al ser δ(gap) convexa, el ruido en el lobby sesga el paso hacia arriba;
    encoger por `n_eff / (n_eff + c_lobby)` evita que explote con pocas muestras.
    """
    lob = np.asarray(lobby, dtype=float)
    zv = np.asarray(z, dtype=float)
    if n_obs_lobby is None:
        peso = np.where(np.isfinite(lob), 1.0, 0.0)
    else:
        n_eff = np.maximum(np.asarray(n_obs_lobby, dtype=float), 0.0)
        valido = np.isfinite(lob) & (n_eff > 0.0)
        denom = np.where(valido, n_eff + params.c_lobby, 1.0)
        peso = np.where(valido, np.where(denom > 0.0, n_eff / denom, 0.0), 0.0)
    raw = np.where(np.isfinite(lob), lob - zv, 0.0)
    out = peso * raw
    return float(out) if np.ndim(lobby) == 0 and np.ndim(z) == 0 else out


def prob_doble(pos: Sequence[str], frac_rol: Sequence[float], params: ParametrosPaso) -> np.ndarray:
    """Probabilidad de victoria doble (Aegis) por rol y fracción de partidas en ese rol."""
    pos_arr = np.asarray(pos, dtype=object)
    frac_arr = np.nan_to_num(np.asarray(frac_rol, dtype=float), nan=0.0)
    bins = np.digitize(frac_arr, BANDAS_RAREZA[1:-1], right=True)
    out = np.full(len(pos_arr), params.doble_base, dtype=float)
    for rol, tabla in params.doble_por_rol.items():
        m = pos_arr == rol
        if m.any():
            arr = np.asarray(tabla, dtype=float)
            idx = np.clip(bins[m], 0, len(arr) - 1)
            vals = arr[idx]
            out[m] = np.where(np.isfinite(vals), vals, params.doble_base)
    return out


def decaer_dias(z, dias_enteros: int):
    """Aplica `dias_enteros` días completos de decay sobre `z` (escalar o vector).

    Master+ (z >= 400): −75 LP/día hasta 375 (Diamond I 75 LP).
    Diamond (275 < z < 400): −50 LP/día hasta 275 (Diamond II 75 LP).
    """
    d = int(dias_enteros)
    escalar = np.ndim(z) == 0
    zv = np.atleast_1d(np.asarray(z, dtype=float)).copy()
    if d <= 0:
        return float(zv[0]) if escalar else zv
    m = zv >= MASTER_BASE
    usados = np.zeros_like(zv)
    if m.any():
        pasos = np.minimum(np.ceil((zv[m] - PISO_MASTER_DECAY) / DECAY_MASTER_DIA), d)
        zv[m] = np.maximum(zv[m] - DECAY_MASTER_DIA * pasos, PISO_MASTER_DECAY)
        usados[m] = pasos
    r = (zv < MASTER_BASE) & (zv > PISO_DIAMOND_DECAY) & (usados < d)
    if r.any():
        pasos = np.minimum(np.ceil((zv[r] - PISO_DIAMOND_DECAY) / DECAY_DIAMOND_DIA), d - usados[r])
        zv[r] = np.maximum(zv[r] - DECAY_DIAMOND_DIA * pasos, PISO_DIAMOND_DECAY)
    return float(zv[0]) if escalar else zv


def avanzar_banco(z: float, banco: float, dias: float) -> Tuple[float, float]:
    """Consume `dias` del banco de días; si cae debajo de 0, aplica días enteros de decay."""
    b = banco - max(float(dias), 0.0)
    if b >= 0.0:
        return float(z), b
    d = int(np.floor(-b + 1e-9))
    if d > 0:
        z = decaer_dias(z, d)
        b = min(b + d, 0.0)
    return float(z), float(b)


def recargar_banco(z: float, banco: float, partidas_equiv: float = 1.0) -> float:
    """Recarga el banco tras jugar partidas (1 día en Master+, 7 días en Diamond)."""
    b0 = max(float(banco), 0.0)
    if z >= MASTER_BASE:
        return min(b0 + 1.0 * partidas_equiv, TOPE_BANCO_MASTER)
    return min(b0 + 7.0 * partidas_equiv, TOPE_BANCO_DIAMOND)


def filtrar_rebotes_sin_partida(ev: pd.DataFrame) -> pd.DataFrame:
    """Filtra lecturas atrasadas del ladder en eventos `sin_partida` (rebotes +75/−75 con mismo W+L).

    En `lp_vivo.db`, el 89 % de los jugadores con decay muestra rebotes a LP previo y vuelta
    cuando el endpoint devuelve una réplica vieja sin cambio en W + L.
    """
    if ev.empty:
        return ev.copy()
    df = ev.sort_values(["puuid", "t_despues", "id"] if "id" in ev.columns else ["puuid", "t_despues"]).copy()
    dz = df["z_despues"] - df["z_antes"]
    dwl = (df["wins_despues"] + df["losses_despues"]) - (df["wins_antes"] + df["losses_antes"])
    es_sp = (df["tipo"] == "sin_partida") & (dwl == 0)
    # Un rebote positivo sin partidas ni cambio de W+L es siempre artefacto de réplica; además
    # el par (-d, +d) adyacente del mismo jugador se cancela entre sí.
    next_dz = dz.groupby(df["puuid"], sort=False).shift(-1)
    prev_dz = dz.groupby(df["puuid"], sort=False).shift(1)
    next_sp = es_sp.groupby(df["puuid"], sort=False).shift(-1, fill_value=False)
    prev_sp = es_sp.groupby(df["puuid"], sort=False).shift(1, fill_value=False)
    rebote = es_sp & ((dz > 0) | (next_sp & (dz + next_dz == 0)) | (prev_sp & (dz + prev_dz == 0)))
    return df.loc[~rebote].copy()


def cargar_eventos_vivo_las(lp_vivo_db: Path) -> pd.DataFrame:
    """Lee los eventos `enlazada` de LAS (`la2`) de `lp_vivo.db` en modo solo lectura."""
    uri = f"file:{Path(lp_vivo_db).resolve().as_posix()}?mode=ro"
    sql = (
        "SELECT e.id, e.puuid, e.match_id, e.t_antes, e.t_despues, "
        "e.tier_antes, e.rank_antes, e.lp_antes, e.wins_antes, e.losses_antes, "
        "e.tier_despues, e.rank_despues, e.lp_despues, e.wins_despues, e.losses_despues, "
        "p.game_end, p.duracion_s, j.team_id, j.win, j.team_position, j.gold "
        "FROM eventos e "
        "JOIN partidas p ON p.match_id = e.match_id "
        "JOIN participantes j ON j.match_id = e.match_id AND j.puuid = e.puuid "
        "WHERE e.platform = 'la2' AND e.tipo = 'partida' AND e.estado = 'enlazada'"
    )
    with contextlib.closing(sqlite3.connect(uri, uri=True, timeout=60)) as con:
        df = pd.read_sql_query(sql, con)
    for lado in ("antes", "despues"):
        df[f"z_{lado}"] = [
            lp_absoluto(t, r, l)
            for t, r, l in zip(df[f"tier_{lado}"], df[f"rank_{lado}"], df[f"lp_{lado}"])
        ]
    df["dz"] = df["z_despues"] - df["z_antes"]
    df["es_win"] = df["win"] == 1
    return agregar_lobby(df).rename(columns={"lobby_lp": "lobby", "lobby_conocidos": "n_lob"})


def ajustar_k_diamond(evaluables_df: pd.DataFrame, k_min: float = 10.0, k_max: float = 30.0) -> float:
    """Ajusta `k_diamond` por L1 sobre jugadores que arrancaron y terminaron en Diamond."""
    dia = evaluables_df[
        (evaluables_df["z1"] < 380) & (evaluables_df["z2"] < 380) & (evaluables_df["z1"] >= 0)
        & (evaluables_df["dW"] >= 0) & (evaluables_df["dL"] >= 0) & (evaluables_df["dN"] >= 10)
    ]
    if dia.empty:
        return ParametrosPaso.k_diamond
    grilla = np.arange(k_min, k_max + 0.01, 0.25)
    net = (dia["dW"] - dia["dL"]).to_numpy(float)
    dz = (dia["z2"] - dia["z1"]).to_numpy(float)
    errores = [np.abs(dz - k * net).mean() for k in grilla]
    return float(grilla[int(np.argmin(errores))])


def ajustar_parametros_las(
    eventos_las: pd.DataFrame,
    roles_hist: Optional[pd.DataFrame] = None,
    k_diamond: float = ParametrosPaso.k_diamond,
    min_lobby_conocidos: int = 5,
    min_eventos_delta: int = 8,
) -> ParametrosPaso:
    """Estima los parámetros del paso usando únicamente eventos `enlazada` de LAS.

    `roles_hist`: DataFrame opcional con `(puuid, team_position, frac_hist, tot_hist)` del
    historial previo; si no se pasa, calcula `frac_hist` dentro de `eventos_las` para
    jugadores con al menos 8 partidas.
    """
    df = eventos_las.copy()
    if roles_hist is not None and not roles_hist.empty:
        df = df.merge(
            roles_hist[["puuid", "team_position", "frac_hist", "tot_hist"]],
            on=["puuid", "team_position"],
            how="left",
        )
    else:
        tot = df.groupby("puuid")["dz"].transform("size")
        df["tot_hist"] = tot
        df["frac_hist"] = df.groupby(["puuid", "team_position"])["dz"].transform("size") / tot.clip(lower=1)
    df["frac_hist"] = df["frac_hist"].fillna(0.0)

    l = df[~df["es_win"]]
    fuera_piso = l[l["z_antes"] != MASTER_BASE]
    perdon = float((fuera_piso["dz"] == 0).mean()) if len(fuera_piso) else ParametrosPaso.perdon
    en_piso = l[l["z_antes"] == MASTER_BASE]
    piso_quieto = float((en_piso["dz"] == 0).mean()) if len(en_piso) else ParametrosPaso.piso_quieto
    bajan = en_piso[en_piso["dz"] < 0]["dz"]
    piso_baja = float(-bajan.mean()) if len(bajan) else ParametrosPaso.piso_baja

    w = df[df["es_win"]].copy()
    w["doble"] = w["dz"] >= 33.0
    doble_base = float(w["doble"].mean()) if len(w) else ParametrosPaso.doble_base
    doble_por_rol: Dict[str, Tuple[float, ...]] = {}
    w_rico = w[w["tot_hist"] >= 20].copy()
    if not w_rico.empty:
        w_rico["bf"] = pd.cut(w_rico["frac_hist"], BANDAS_RAREZA, labels=False)
        gr = w_rico.groupby(["team_position", "bf"], observed=False)["doble"]
        sums = gr.sum().unstack(fill_value=0.0)
        cnts = gr.count().unstack(fill_value=0.0)
        n_bandas = len(BANDAS_RAREZA) - 1
        c_suav = 10.0
        for rol in sums.index:
            s = sums.loc[rol].reindex(range(n_bandas), fill_value=0.0).to_numpy(float)
            n = cnts.loc[rol].reindex(range(n_bandas), fill_value=0.0).to_numpy(float)
            fila = (s + c_suav * doble_base) / (n + c_suav)
            doble_por_rol[str(rol)] = tuple(float(v) for v in fila)

    norm = df[
        (df["n_lob"] >= min_lobby_conocidos)
        & ((df["es_win"] & df["dz"].between(10, 31)) | (~df["es_win"] & df["dz"].between(-30, -5)))
    ].copy()
    norm["delta"] = np.where(norm["es_win"], norm["dz"] - 20.0, norm["dz"] + 20.0)
    norm["gap"] = norm["lobby"] - norm["z_antes"]
    pj = norm.groupby("puuid").agg(d=("delta", "mean"), gap=("gap", "mean"), n=("delta", "size"))
    pj = pj[pj["n"] >= min_eventos_delta]
    cortes = (-3000.0, -400.0, -200.0, -100.0, 0.0, 100.0, 200.0, 400.0, 650.0, 3000.0)
    med = pj.groupby(pd.cut(pj["gap"], cortes), observed=False).agg(g=("gap", "mean"), d=("d", "mean")).dropna()
    if len(med) >= 2:
        gap_nodos = tuple(float(v) for v in med["g"].to_numpy())
        delta_mon = np.minimum(np.maximum.accumulate(med["d"].to_numpy()), 10.0)
        delta_nodos = tuple(float(v) for v in delta_mon)
    else:
        gap_nodos = ParametrosPaso.gap_nodos
        delta_nodos = ParametrosPaso.delta_nodos

    return ParametrosPaso(
        k_master=20.0,
        k_diamond=float(k_diamond),
        perdon=perdon,
        piso_quieto=piso_quieto,
        piso_baja=piso_baja,
        doble_base=doble_base,
        doble_por_rol=doble_por_rol,
        gap_nodos=gap_nodos,
        delta_nodos=delta_nodos,
    )


def _paso_esperado_escalar(
    z: float,
    win: bool,
    lobby: float,
    n_lob: float,
    p_doble_k: float,
    params: ParametrosPaso,
) -> float:
    k_ = params.k_master if z >= MASTER_BASE else params.k_diamond
    gap = encoger_gap(lobby, z, n_lob, params)
    d = float(delta_de(gap, params))
    if win:
        return z + (k_ + d) * (1.0 + p_doble_k)
    q = params.perdon
    dest = z - (k_ - d) * (1.0 - q)
    if z > MASTER_BASE + 0.5 and dest < MASTER_BASE:
        return MASTER_BASE
    if MASTER_BASE <= z <= MASTER_BASE + 0.5:
        return z - (1.0 - params.piso_quieto) * params.piso_baja
    return dest


def recorrer_adelante(
    win: Sequence[bool],
    tiempos_ms: Sequence[int],
    lobby: Sequence[float],
    n_lob: Sequence[float],
    pos: Sequence[str],
    frac_rol: Sequence[float],
    z0: float,
    t_ini: int,
    t_fin: int,
    params: ParametrosPaso,
    dW_total: Optional[int] = None,
    dL_total: Optional[int] = None,
    tasa_oc: float = 0.0,
    p_oc: float = 0.5,
) -> Tuple[float, np.ndarray]:
    """Recorre hacia adelante las partidas de un jugador y devuelve `(z_final, z_pre)`.

    Si `dW_total` y `dL_total` se pasan (entre dos fotos), reparte el saldo real de las
    partidas no observadas proporcionalmente; si no, usa `tasa_oc` y `p_oc` (modo causal).
    """
    w_arr = np.asarray(win, dtype=bool)
    t_arr = np.asarray(tiempos_ms, dtype=np.int64)
    lob_arr = np.asarray(lobby, dtype=float)
    nl_arr = np.asarray(n_lob, dtype=float)
    pd_arr = prob_doble(pos, frac_rol, params) if len(w_arr) else np.array([], dtype=float)

    z = float(z0)
    banco = TOPE_BANCO_MASTER if z >= MASTER_BASE else TOPE_BANCO_DIAMOND
    n = len(w_arr)

    if n == 0:
        if dW_total is not None and dL_total is not None and (dW_total + dL_total) > 0:
            oc_tot = max(int(dW_total) + int(dL_total), 0)
            k_ = params.k_master if z >= MASTER_BASE else params.k_diamond
            z += int(dW_total) * k_ * (1.0 + params.doble_base) - int(dL_total) * k_ * (1.0 - params.perdon)
            banco = recargar_banco(z, banco, oc_tot)
        z, _ = avanzar_banco(z, banco, (t_fin - t_ini) / DIA_MS)
        return z, np.array([], dtype=float)

    if dW_total is not None and dL_total is not None:
        obs_w = int(w_arr.sum())
        obs_l = n - obs_w
        oc_w_p = max(int(dW_total) - obs_w, 0) / n
        oc_l_p = max(int(dL_total) - obs_l, 0) / n
    else:
        oc_w_p = max(float(tasa_oc), 0.0) * p_oc
        oc_l_p = max(float(tasa_oc), 0.0) * (1.0 - p_oc)

    z_pre = np.empty(n, dtype=float)
    t_prev = int(t_ini)
    for k in range(n):
        z, banco = avanzar_banco(z, banco, (int(t_arr[k]) - t_prev) / DIA_MS)
        t_prev = int(t_arr[k])
        z_pre[k] = z
        z = _paso_esperado_escalar(z, bool(w_arr[k]), lob_arr[k], nl_arr[k], float(pd_arr[k]), params)
        if oc_w_p + oc_l_p > 0.0:
            k_ = params.k_master if z >= MASTER_BASE else params.k_diamond
            d = float(delta_de(encoger_gap(lob_arr[k], z, nl_arr[k], params), params))
            z += oc_w_p * (k_ + d) * (1.0 + params.doble_base) - oc_l_p * (k_ - d) * (1.0 - params.perdon)
        banco = recargar_banco(z, banco, 1.0 + oc_w_p + oc_l_p)

    z, _ = avanzar_banco(z, banco, (int(t_fin) - t_prev) / DIA_MS)
    return z, z_pre


def _masita_en_grilla(grid: np.ndarray, z: Optional[float], sigma: float) -> np.ndarray:
    """Distribución inicial o terminal sobre la grilla 1D."""
    n = len(grid)
    if z is None or not np.isfinite(z):
        return np.full(n, 1.0 / n, dtype=float)
    if sigma <= 0.0:
        out = np.zeros(n, dtype=float)
        step = grid[1] - grid[0]
        x = float(np.clip((z - grid[0]) / step, 0.0, n - 1.0))
        i = min(int(np.floor(x)), n - 2)
        w = x - i
        out[i] += 1.0 - w
        out[i + 1] += w
        return out
    out = np.exp(-0.5 * ((grid - z) / sigma) ** 2)
    s = out.sum()
    return out / s if s > 0.0 else np.full(n, 1.0 / n, dtype=float)


def _pushforward_lineal(dist: np.ndarray, dest: np.ndarray, z0: float, step: float) -> np.ndarray:
    """Empuja una densidad 1D a `dest` (adjunto exacto de la interpolación lineal)."""
    n = len(dist)
    x = np.clip((dest - z0) / step, 0.0, n - 1.0 - 1e-9)
    idx = np.floor(x).astype(int)
    w = x - idx
    out = np.bincount(idx, weights=dist * (1.0 - w), minlength=n)
    out += np.bincount(idx + 1, weights=dist * w, minlength=n)
    return out


def _pullback_lineal(beta: np.ndarray, dest: np.ndarray, z0: float, step: float) -> np.ndarray:
    """Retrocede `beta` evaluándola en `dest` (transpuesta exacta de `_pushforward_lineal`)."""
    n = len(beta)
    x = np.clip((dest - z0) / step, 0.0, n - 1.0 - 1e-9)
    idx = np.floor(x).astype(int)
    w = x - idx
    return beta[idx] * (1.0 - w) + beta[idx + 1] * w


def _difundir_gauss(dist: np.ndarray, sigma: float, step: float) -> np.ndarray:
    """Convolución gaussiana simétrica en 1D (conserva simetría entre forward y backward)."""
    if sigma <= 0.0:
        return dist
    sig_idx = sigma / step
    if sig_idx < 0.15:
        return dist
    radio = max(1, int(np.ceil(3.5 * sig_idx)))
    kx = np.arange(-radio, radio + 1, dtype=float)
    kernel = np.exp(-0.5 * (kx / sig_idx) ** 2)
    kernel /= kernel.sum()
    return np.convolve(dist, kernel, mode="same")


def _lik_lobby(grid: np.ndarray, lobby: float, n_lob: float, params: ParametrosPaso, tau: float,
               sigma_lob: float = 130.0) -> np.ndarray:
    """Verosimilitud Student-t suave del lobby observado (solo cuando `tau > 0` y `lobby` válido)."""
    if tau <= 0.0 or not np.isfinite(lobby) or n_lob <= 0:
        return np.ones_like(grid)
    peso = float(n_lob) / (float(n_lob) + params.c_lobby)
    nu = 4.0
    r = (lobby - grid) / sigma_lob
    lik = (1.0 + (r * r) / nu) ** (-(nu + 1.0) / 2.0)
    return lik ** (tau * peso)


def _resumen_posterior(grid: np.ndarray, post: np.ndarray) -> Tuple[float, float, float, float]:
    """Media, mediana, p10 y p90 de una densidad discreta normalizada."""
    s = post.sum()
    if s <= 0.0 or not np.isfinite(s):
        return float("nan"), float("nan"), float("nan"), float("nan")
    p = post / s
    media = float(p @ grid)
    cdf = np.cumsum(p)
    p10 = float(grid[min(int(np.searchsorted(cdf, 0.10)), len(grid) - 1)])
    med = float(grid[min(int(np.searchsorted(cdf, 0.50)), len(grid) - 1)])
    p90 = float(grid[min(int(np.searchsorted(cdf, 0.90)), len(grid) - 1)])
    return media, med, p10, p90


def suavizar_en_grilla(
    win: Sequence[bool],
    tiempos_ms: Sequence[int],
    lobby: Sequence[float],
    n_lob: Sequence[float],
    pos: Sequence[str],
    frac_rol: Sequence[float],
    z1: Optional[float],
    sigma1: float,
    z2: Optional[float],
    sigma2: float,
    t_ini: int,
    t_fin: int,
    params: ParametrosPaso,
    dW_total: Optional[int] = None,
    dL_total: Optional[int] = None,
    tasa_oc: float = 0.0,
    p_oc: float = 0.5,
    tau_lobby: float = 0.0,
    z_min: float = -400.0,
    z_max: float = 3400.0,
    step: float = 10.0,
) -> dict:
    """Suavizado forward–backward en grilla 1D sin invertir la dinámica.

    * Si `z2 is None`, actúa como filtro causal hacia adelante (no ve el futuro).
    * Si `z1` y `z2` están anclados, combina ambos extremos con los operadores adjuntos
      exactos del paso mecánico, devolviendo `z_media`, `z_mediana`, `z_p10`, `z_p90` antes
      de cada partida (más `z_ini` y `z_fin`).
    """
    grid = np.arange(z_min, z_max + step * 0.5, step, dtype=float)
    z0_grid = grid[0]
    w_arr = np.asarray(win, dtype=bool)
    t_arr = np.asarray(tiempos_ms, dtype=np.int64)
    lob_arr = np.asarray(lobby, dtype=float)
    nl_arr = np.asarray(n_lob, dtype=float)
    pd_arr = prob_doble(pos, frac_rol, params) if len(w_arr) else np.array([], dtype=float)
    n = len(w_arr)

    if n > 0 and dW_total is not None and dL_total is not None:
        obs_w = int(w_arr.sum())
        obs_l = n - obs_w
        oc_w_p = max(int(dW_total) - obs_w, 0) / n
        oc_l_p = max(int(dL_total) - obs_l, 0) / n
    else:
        oc_w_p = max(float(tasa_oc), 0.0) * p_oc
        oc_l_p = max(float(tasa_oc), 0.0) * (1.0 - p_oc)
    h_oc = oc_w_p + oc_l_p
    k_grid = np.where(grid >= MASTER_BASE, params.k_master, params.k_diamond)

    # Seguimiento escalar de banco de días para Master+ y Diamond a lo largo de los huecos.
    z_ref = z1 if (z1 is not None and np.isfinite(z1)) else (z2 if z2 is not None and np.isfinite(z2) else 400.0)
    b_m = TOPE_BANCO_MASTER
    b_d = TOPE_BANCO_DIAMOND

    def _dias_decay_gap(dt_dias: float) -> Tuple[int, int]:
        nonlocal b_m, b_d
        b_m -= dt_dias
        b_d -= dt_dias
        dm = max(0, int(np.floor(-b_m + 1e-9))) if b_m < 0 else 0
        dd = max(0, int(np.floor(-b_d + 1e-9))) if b_d < 0 else 0
        if dm > 0:
            b_m = min(b_m + dm, 0.0)
        if dd > 0:
            b_d = min(b_d + dd, 0.0)
        return dm, dd

    def _aplicar_gap_op(dm: int, dd: int, pi_oc: float):
        dest_m = decaer_dias(grid, dm)
        dest_d = decaer_dias(grid, dd)
        dest = np.where(grid >= MASTER_BASE, dest_m, dest_d)

        def fwd(a):
            mov = _pushforward_lineal(a, dest, z0_grid, step)
            return (1.0 - pi_oc) * mov + pi_oc * a if pi_oc > 0.0 else mov

        def bwd(b):
            mov = _pullback_lineal(b, dest, z0_grid, step)
            return (1.0 - pi_oc) * mov + pi_oc * b if pi_oc > 0.0 else mov

        return fwd, bwd

    # Precomputamos operadores de transición por paso para garantizar adjunción exacta.
    gap_ops = []
    match_ops = []
    lik_lob = []
    t_prev = int(t_ini)
    pi_oc_gap = min(0.7, h_oc / (1.0 + h_oc)) if h_oc > 0.0 else 0.0

    for k in range(n):
        dt_dias = max((int(t_arr[k]) - t_prev) / DIA_MS, 0.0)
        t_prev = int(t_arr[k])
        dm, dd = _dias_decay_gap(dt_dias)
        gap_ops.append(_aplicar_gap_op(dm, dd, pi_oc_gap if dt_dias > 14.0 else 0.0))
        lik_lob.append(_lik_lobby(grid, lob_arr[k], nl_arr[k], params, tau_lobby))

        gap_grid = encoger_gap(lob_arr[k], grid, nl_arr[k], params)
        d = delta_de(gap_grid, params)
        oc_drift = oc_w_p * (k_grid + d) * (1.0 + params.doble_base) - oc_l_p * (k_grid - d) * (1.0 - params.perdon)
        sigma_k = float(np.sqrt(params.sigma_paso ** 2 + h_oc * 0.25 * (2.0 * 20.0) ** 2 + max(step * 0.5, 1.0) ** 2))

        if w_arr[k]:
            p_dob = float(pd_arr[k])
            d1 = grid + (k_grid + d) + oc_drift
            d2 = grid + 2.0 * (k_grid + d) + oc_drift
            ramas = ((1.0 - p_dob, d1), (p_dob, d2))
        else:
            q = params.perdon
            caida = grid - (k_grid - d) + oc_drift
            caida = np.where((grid > MASTER_BASE + 0.5) & (caida < MASTER_BASE), MASTER_BASE, caida)
            en_0 = np.abs(grid - MASTER_BASE) < step * 0.5
            caida_piso = np.where(en_0, grid - params.piso_baja + oc_drift, caida)
            quieto_piso = np.where(en_0, grid + oc_drift, caida)
            ramas = (
                ((1.0 - q) * (1.0 - params.piso_quieto), caida_piso),
                ((1.0 - q) * params.piso_quieto, quieto_piso),
                (q, grid + oc_drift),
            )
        match_ops.append((ramas, sigma_k))
        b_m = min(max(b_m, 0.0) + 1.0 * (1.0 + h_oc), TOPE_BANCO_MASTER)
        b_d = min(max(b_d, 0.0) + 7.0 * (1.0 + h_oc), TOPE_BANCO_DIAMOND)

    dt_fin = max((int(t_fin) - t_prev) / DIA_MS, 0.0)
    dm_fin, dd_fin = _dias_decay_gap(dt_fin)
    gap_fin = _aplicar_gap_op(dm_fin, dd_fin, pi_oc_gap if dt_fin > 14.0 else 0.0)

    # Pasada hacia adelante (α)
    alpha_pre_match = np.empty((n, len(grid)), dtype=float)
    a = _masita_en_grilla(grid, z1, sigma1)
    alpha_ini = a.copy()
    for k in range(n):
        a = gap_ops[k][0](a)
        a = a * lik_lob[k]
        s = a.sum()
        a = a / s if s > 0.0 else alpha_ini.copy()
        alpha_pre_match[k] = a
        ramas, sigma_k = match_ops[k]
        a_next = np.zeros_like(grid)
        for prob, dest in ramas:
            if prob > 0.0:
                a_next += prob * _pushforward_lineal(a, dest, z0_grid, step)
        a = _difundir_gauss(a_next, sigma_k, step)
        s = a.sum()
        if s > 0.0:
            a = a / s
    a_fin = gap_fin[0](a)
    s_fin = a_fin.sum()
    if s_fin > 0.0:
        a_fin = a_fin / s_fin

    # Si hay z2, pasada hacia atrás (β) y combinación forward-backward; si no, filtro causal puro.
    z_media = np.empty(n, dtype=float)
    z_mediana = np.empty(n, dtype=float)
    z_p10 = np.empty(n, dtype=float)
    z_p90 = np.empty(n, dtype=float)
    if z2 is not None and np.isfinite(z2):
        beta_ini = _masita_en_grilla(grid, z2, sigma2)
        beta_fin = beta_ini.copy()
        b = gap_fin[1](beta_ini)
        s_b = b.sum()
        if s_b > 0.0:
            b = b / s_b

        beta_pre_match = np.empty((n, len(grid)), dtype=float)
        for k in range(n - 1, -1, -1):
            ramas, sigma_k = match_ops[k]
            b_dif = _difundir_gauss(b, sigma_k, step)
            b_k = np.zeros_like(grid)
            for prob, dest in ramas:
                if prob > 0.0:
                    b_k += prob * _pullback_lineal(b_dif, dest, z0_grid, step)
            beta_pre_match[k] = b_k
            b_prev = gap_ops[k][1](b_k * lik_lob[k])
            s_b = b_prev.sum()
            b = b_prev / s_b if s_b > 0.0 else np.ones_like(grid)

        for k in range(n):
            post = alpha_pre_match[k] * beta_pre_match[k]
            if post.sum() <= 1e-20:
                # Si las puntas son incompatibles con la trayectoria, promediar suavizado sin colapsar a NaN
                post = 0.5 * alpha_pre_match[k] + 0.5 * (beta_pre_match[k] / max(beta_pre_match[k].sum(), 1e-300))
            z_media[k], z_mediana[k], z_p10[k], z_p90[k] = _resumen_posterior(grid, post)

        post_ini = alpha_ini * b
        if post_ini.sum() <= 1e-20:
            post_ini = alpha_ini
        post_fin = a_fin * beta_fin
        if post_fin.sum() <= 1e-20:
            post_fin = beta_fin
    else:
        for k in range(n):
            z_media[k], z_mediana[k], z_p10[k], z_p90[k] = _resumen_posterior(grid, alpha_pre_match[k])
        post_ini = alpha_ini
        post_fin = a_fin

    m_ini, _, _, _ = _resumen_posterior(grid, post_ini)
    m_fin, _, p10_fin, p90_fin = _resumen_posterior(grid, post_fin)
    return {
        "z_ini": m_ini if (z1 is None or not np.isfinite(z1) or sigma1 > 0) else float(z1),
        "z_fin": m_fin if (z2 is None or not np.isfinite(z2) or sigma2 > 0) else float(z2),
        "z_fin_p10": p10_fin,
        "z_fin_p90": p90_fin,
        "z_media": z_media,
        "z_mediana": z_mediana,
        "z_p10": z_p10,
        "z_p90": z_p90,
        "z_ref": float(z_ref),
    }


def _serie_z_inicial(
    partes: pd.DataFrame,
    z1: pd.Series,
    z2: Optional[pd.Series],
    t1: int,
    t2: int,
    modo: str,
) -> pd.Series:
    """Serie inicial de LP por fila para construir lobbies antes de la primera pasada."""
    if modo == "causal" or z2 is None:
        return partes["player_id"].map(z1)
    tf = np.clip((partes["game_creation"].to_numpy(float) - t1) / max(t2 - t1, 1), 0.0, 1.0)
    a = partes["player_id"].map(z1).to_numpy(float)
    b = partes["player_id"].map(z2).to_numpy(float)
    val = a + (b - a) * tf
    val = np.where(np.isfinite(val), val, np.where(np.isfinite(a), a, b))
    return pd.Series(val, index=partes.index)


def _lobby_inicial_tramo(
    partes: pd.DataFrame,
    z1: pd.Series,
    z2: Optional[pd.Series],
    t1: int,
    t2: int,
    modo: str,
) -> Tuple[pd.Series, pd.Series]:
    """LP de lobby por fila (de los otros 9 jugadores) para la pasada 1."""
    zt = _serie_z_inicial(partes, z1, z2, t1, t2, modo)
    return _lobby_de_serie(partes, zt)


def _lobby_de_serie(partes: pd.DataFrame, z_fila: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Calcula `(lobby, n_lob)` de los otros jugadores de cada partida usando una Serie alineada a `partes`."""
    ok = z_fila.notna().astype(float)
    zf = z_fila.fillna(0.0).astype(float)
    m = partes["match_id"]
    sum_tot = zf.groupby(m, sort=False).transform("sum") - zf
    cnt_tot = ok.groupby(m, sort=False).transform("sum") - ok
    lobby = (sum_tot / cnt_tot.where(cnt_tot > 0)).reindex(partes.index)
    return lobby, cnt_tot.reindex(partes.index)


def suavizar_lobby_jugador(
    lobby: Sequence[float],
    n_lob: Sequence[float],
    modo: str = "descriptivo",
    ventana: int = 31,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estima el MMR local del jugador suavizando el lobby entre partidas vecinas.

    Devuelve `(M, n_eff)` donde `M` es la media móvil del lobby y `n_eff` suma las
    fracciones de lobby observado (`n_lob / 9`) en la ventana, para encoger cuando hay
    pocas partidas o pocos jugadores medidos. En `modo="causal"` solo usa partidas `<= k`.
    """
    lob_arr = np.asarray(lobby, dtype=float)
    nl_arr = np.nan_to_num(np.asarray(n_lob, dtype=float), nan=0.0)
    pesos = np.where(np.isfinite(lob_arr), np.clip(nl_arr / 9.0, 0.0, 1.0), 0.0)
    num = pd.Series(np.where(np.isfinite(lob_arr), lob_arr * pesos, 0.0))
    den = pd.Series(pesos)
    centrado = modo == "descriptivo"
    s_num = num.rolling(ventana, center=centrado, min_periods=1).sum().to_numpy(float)
    s_den = den.rolling(ventana, center=centrado, min_periods=1).sum().to_numpy(float)
    M = np.where(s_den > 0.0, s_num / np.maximum(s_den, 1e-12), np.nan)
    return M, s_den


def _metodo_jugador(tiene_z1: bool, tiene_z2: bool, tiene_ancla: bool, modo: str) -> str:
    if modo == "causal":
        return "causal_desde_foto1" if tiene_z1 else "causal_lobby"
    if tiene_z1 and tiene_z2:
        return "puente_grilla"
    if tiene_z1:
        return "adelante_foto1"
    if tiene_z2 and not tiene_ancla:
        return "atras_foto2"
    if tiene_ancla:
        return "ancla_principal_grilla"
    return "lobby"


def estimar_mapa_tramo(
    partes: pd.DataFrame,
    foto1: Optional[pd.DataFrame],
    foto2: Optional[pd.DataFrame],
    t1: int,
    t2: int,
    params: ParametrosPaso,
    anclas_z1: Optional[pd.Series] = None,
    sigma_ancla_z1: float = 100.0,
    modo: str = "descriptivo",
    n_pasadas: int = 2,
    tasa_oc_defecto: float = 0.0,
    step: float = 10.0,
    jugadores_objetivo: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Construye el mapa básico de LP partida a partida sobre un tramo `[t1, t2)`.

    `partes`: columnas `match_id`, `game_creation`, `player_id`, `win` (y opcional `pos`).
    `foto1`, `foto2`: DataFrames indexados por `player_id` con `z`, `wins`, `losses` (opcionales).
    `anclas_z1`: predicciones de foto 1 (p. ej. del modelo principal) para jugadores sin foto 1 real.
    `modo`: `"descriptivo"` (usa ambas fotos y suavizado bidireccional) o `"causal"` (ignora `foto2`
    y saldo futuro; solo usa información previa a cada partida).
    `jugadores_objetivo`: subconjunto opcional de `player_id` a suavizar; mantiene al resto
    de los participantes solo para construir los lobbies de referencia.

    Devuelve DataFrame alineado al índice de `partes` con `z_media`, `z_mediana`, `z_p10`, `z_p90`, `metodo`.
    """
    if modo not in ("descriptivo", "causal"):
        raise ValueError("modo debe ser 'descriptivo' o 'causal'")

    df = partes.sort_values(["player_id", "game_creation", "match_id"], kind="mergesort").copy()
    if modo == "causal" and foto1 is not None and not df.empty and (df["game_creation"] < t1).any():
        raise ValueError("modo='causal' no admite foto1 para partidas anteriores a t1 (fuga del futuro)")
    if "pos" not in df.columns:
        df["pos"] = "UNKNOWN"
    if modo == "causal":
        prev_tot = df.groupby("player_id", sort=False).cumcount()
        prev_rol = df.groupby(["player_id", "pos"], sort=False).cumcount()
        df["frac_rol"] = np.where(prev_tot > 0, prev_rol / np.maximum(prev_tot, 1), 0.2)
    else:
        cnt_j = df.groupby("player_id")["win"].transform("size")
        df["frac_rol"] = df.groupby(["player_id", "pos"])["win"].transform("size") / cnt_j.clip(lower=1)

    z1_ser = foto1["z"].dropna() if (foto1 is not None and "z" in foto1.columns) else pd.Series(dtype=float)
    z2_ser = (
        foto2.loc[foto2["z"].notna(), "z"]
        if (modo == "descriptivo" and foto2 is not None and "z" in foto2.columns)
        else pd.Series(dtype=float)
    )
    ancla_ser = anclas_z1.dropna() if (modo == "descriptivo" and anclas_z1 is not None) else pd.Series(dtype=float)
    z1_lobby = z1_ser.combine_first(ancla_ser) if not ancla_ser.empty else z1_ser

    # Saldo exacto (dW, dL) solo si tenemos ambas fotos reales y estamos en modo descriptivo
    dw_map, dl_map = {}, {}
    if modo == "descriptivo" and foto1 is not None and foto2 is not None:
        if {"wins", "losses"}.issubset(foto1.columns) and {"wins", "losses"}.issubset(foto2.columns):
            comunes = foto1.index.intersection(foto2.index)
            dw = foto2.loc[comunes, "wins"] - foto1.loc[comunes, "wins"]
            dl = foto2.loc[comunes, "losses"] - foto1.loc[comunes, "losses"]
            ok_saldo = (dw >= 0) & (dl >= 0)
            dw_map = dw[ok_saldo].to_dict()
            dl_map = dl[ok_saldo].to_dict()

    z_base = _serie_z_inicial(df, z1_lobby, z2_ser if not z2_ser.empty else None, t1, t2, modo)
    lob, n_lob = _lobby_de_serie(df, z_base)

    obj_set = set(jugadores_objetivo) if jugadores_objetivo is not None else None
    indices_por_pid = df.groupby("player_id", sort=False).indices
    grupos = [
        (pid, np.asarray(pos_idx, dtype=int))
        for pid, pos_idx in indices_por_pid.items()
        if obj_set is None or pid in obj_set
    ]

    win_all = df["win"].to_numpy(bool)
    ts_all = df["game_creation"].to_numpy(np.int64)
    pos_all = df["pos"].to_numpy(object)
    frac_all = df["frac_rol"].to_numpy(float)
    n_rows = len(df)
    z_out = np.full(n_rows, np.nan, dtype=float)
    med_out = np.full(n_rows, np.nan, dtype=float)
    p10_out = np.full(n_rows, np.nan, dtype=float)
    p90_out = np.full(n_rows, np.nan, dtype=float)
    met_out = np.full(n_rows, "lobby" if obj_set is None else "omitido", dtype=object)

    pasadas = 1 if modo == "causal" else max(int(n_pasadas), 1)
    for pasada in range(pasadas):
        if pasada > 0:
            z_ser_lob = pd.Series(z_out, index=df.index)
            if obj_set is not None:
                z_ser_lob = z_ser_lob.fillna(z_base)
            lob, n_lob = _lobby_de_serie(df, z_ser_lob)
        lob_arr = lob.to_numpy(float)
        nlob_arr = n_lob.to_numpy(float)

        for pid, idx_arr in grupos:
            z1_val = z1_ser.get(pid, np.nan)
            z2_val = z2_ser.get(pid, np.nan)
            tiene_z1 = np.isfinite(z1_val)
            tiene_z2 = np.isfinite(z2_val)
            tiene_ancla = pid in ancla_ser.index
            met = _metodo_jugador(tiene_z1, tiene_z2, tiene_ancla, modo)
            if pasada == 0:
                met_out[idx_arr] = met

            if not tiene_z1 and not tiene_z2 and not tiene_ancla:
                # Sin fotos: estimación por lobby observado (sin bfill en modo causal)
                l_vals = lob_arr[idx_arr]
                if np.isfinite(l_vals).any():
                    if modo == "causal":
                        ll, _ = suavizar_lobby_jugador(l_vals, nlob_arr[idx_arr], modo="causal")
                    else:
                        ll = pd.Series(l_vals).ffill().bfill().to_numpy(float)
                    z_out[idx_arr] = ll
                    med_out[idx_arr] = ll
                    p10_out[idx_arr] = ll - 150.0
                    p90_out[idx_arr] = ll + 150.0
                    if pasada == 0:
                        met_out[idx_arr] = np.where(np.isfinite(ll), met, "sin_dato")
                elif pasada == 0:
                    met_out[idx_arr] = "sin_dato"
                continue

            z1_in = float(z1_val) if tiene_z1 else (float(ancla_ser[pid]) if tiene_ancla else None)
            sig1 = 0.0 if tiene_z1 else (sigma_ancla_z1 if tiene_ancla else 0.0)
            z2_in = float(z2_val) if tiene_z2 else None
            sig2 = 0.0 if tiene_z2 else 0.0
            dw_t = int(dw_map[pid]) if pid in dw_map else None
            dl_t = int(dl_map[pid]) if pid in dl_map else None
            tau_l = 0.0 if (tiene_z1 and tiene_z2) else 0.5
            m_lob, n_eff = suavizar_lobby_jugador(lob_arr[idx_arr], nlob_arr[idx_arr], modo=modo)

            res = suavizar_en_grilla(
                win=win_all[idx_arr],
                tiempos_ms=ts_all[idx_arr],
                lobby=m_lob,
                n_lob=n_eff,
                pos=pos_all[idx_arr],
                frac_rol=frac_all[idx_arr],
                z1=z1_in,
                sigma1=sig1,
                z2=z2_in,
                sigma2=sig2,
                t_ini=t1,
                t_fin=t2,
                params=params,
                dW_total=dw_t,
                dL_total=dl_t,
                tasa_oc=tasa_oc_defecto if dw_t is None else 0.0,
                tau_lobby=tau_l,
                step=step,
            )
            z_out[idx_arr] = res["z_media"]
            med_out[idx_arr] = res["z_mediana"]
            p10_out[idx_arr] = res["z_p10"]
            p90_out[idx_arr] = res["z_p90"]

    out = pd.DataFrame(
        {
            "z_media": z_out,
            "z_mediana": med_out,
            "z_p10": p10_out,
            "z_p90": p90_out,
            "metodo": met_out,
        },
        index=df.index,
    )
    return out.reindex(partes.index)
