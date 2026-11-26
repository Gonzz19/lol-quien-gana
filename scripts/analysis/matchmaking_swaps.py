"""Experimento contrafáctico de reemparejamiento (swaps) en Master+ (LAS).

Evalúa cuánto se puede acercar a 50-50 la probabilidad pre-partida de victoria con el MISMO
grupo de 10 jugadores reacomodándolos entre equipos y roles, midiendo el costo de rol por
partida (D = sum_i d_i) y por jugador (d_i, Delta d_i), la frontera "parejo vs. costo de rol",
y el sesgo de optimización (mitades disjuntas del entrenamiento).

Usa exactamente la definición de 320 columnas sin draft de `calib_columnas_sin_draft.json`
(entrenada solo con partidas Master+ anteriores al 16/09, logística C=0,05) y verifica la
reproducción exacta de las predicciones reales (max |Delta p|). Como chequeo secundario,
evalúa los intercambios en la misma línea (2^5 = 32) bajo el modelo escalón 9 con draft
(1.504 columnas), llevando cada jugador su campeón.

Uso:
    python -u scripts/analysis/matchmaking_swaps.py
    python -u scripts/analysis/matchmaking_swaps.py --muestra 500
    python -u scripts/analysis/matchmaking_swaps.py --n-train-sample 5000

Salidas (en corrida completa):
    - data/analysis/matchmaking/swaps_resultados.parquet
    - data/analysis/matchmaking/swaps_jugadores.parquet
    - data/analysis/matchmaking/REPORTE_swaps.md
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import closing, contextmanager
from datetime import datetime
import gc
import json
import math
from pathlib import Path
import sqlite3
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit, logit
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.analysis.brainstorm_base import ajustar_predecir, cargar, extras_referencia  # noqa: E402
from src.features.fingerprint import add_rates  # noqa: E402
from src.features.lobby import prior_lobby_level  # noqa: E402
from src.features.margin import ORDEN, _previo, _residuo, lane_diffs, lane_scale, prior_margin, prior_margin_by, prior_winrate  # noqa: E402
from src.features.padron import leer_foto1, verificar_foto1  # noqa: E402
from src.features.rating import lp_absoluto  # noqa: E402

ANALISIS = REPO / "data" / "analysis"
CACHES = ANALISIS / "brainstorm"
SALIDA = ANALISIS / "matchmaking"

ROLES = ("BOTTOM", "JUNGLE", "MIDDLE", "TOP", "UTILITY")
ROLE_TO_IDX = {r: i for i, r in enumerate(ROLES)}
STATS = (
    "kills_pm", "assists_pm", "damage_pm", "damage_share_pct", "kill_participation_pct",
    "deaths_pm", "damage_taken_pm", "gold_pm", "cs_pm", "gold_share_pct", "vision_pm",
    "d_gold_10", "d_gold_15", "d_xp_15", "d_cs_15",
)
XJ_HISTORIAL = (
    "wr_acum", "wr_v20", "logn_tot", "share_lane", "offrole",
    "log_horas_desde", "ritmo", "conc_rec", "conc_ant", "d_pool",
)
ROL_WR_NAMES = (
    "wr_linea_media", "exceso_wr_linea_media", "exceso_wr_campeon_linea_media",
    "logn_linea_media", "estreno_linea_media", "caida_autofill_media", "autofill_media",
    "wr_linea_minimo", "exceso_wr_linea_minimo", "exceso_wr_campeon_linea_minimo",
)
TEMPO_NAMES = (
    "primera_sangre", "primera_sangre_jungla_soporte", "primera_torre",
    "primera_torre_bot_top", "cambio_trinket", "duracion", "sesgo_temprano_tardio_jugador",
    "sesgo_campeon_linea_media", "sesgo_campeon_linea_maximo", "interaccion_jugador_campeon",
)
EXTRAS_SPEC = (
    ("MC", tuple(f"{rol}/{s}" for rol in ROLES for s in STATS), ()),
    ("ML", tuple(f"{rol}/{s}" for rol in ROLES for s in STATS), tuple(range(75))),
    ("MMR_HIST", ("acumulado", "ultimas_10", "logn"), (0, 1, 2)),
    ("FOTO1_DIN", ("winrate", "logn", "victorias_netas", "master_0_lp"), (0, 1, 2, 3)),
    ("MATCHUPS", tuple(map(str, range(10))), ()),
    ("ROL_WR", ROL_WR_NAMES, (0, 1, 3, 4, 5, 6, 7, 8)),
    ("KALMAN_OUT", ("media", "minimo", "maximo", "desvio", "media_dos_mayores", "forma",
                    "nivel_rivales", *ROLES), tuple(range(12))),
    ("KALMAN_MRG", ("media", "minimo", "maximo", "forma", "incertidumbre", "exceso_sobre_lobby",
                    "brecha_lobby_absoluta"), tuple(range(7))),
    ("TEMPO_OBJ", TEMPO_NAMES, tuple(range(7))),
    ("ECON_MID", ("crecimiento_oro_media", "crecimiento_oro_maximo", "eficiencia_media",
                  "eficiencia_bot_mid"), (0, 1, 2, 3)),
    ("HGB", (*ROLES, "suma"), ()),
    ("MATCHUP_EXP_MARGIN", tuple(map(str, range(12))), ()),
)

# 32 combinaciones de intercambio entre equipos en la misma línea (mask=0 es asignación real)
SWAPS_32 = np.zeros((32, 10), dtype=np.int32)
SIGNS_32 = np.ones((32, 5), dtype=np.float64)
for _m in range(32):
    for _l in range(5):
        if (_m >> _l) & 1:
            SWAPS_32[_m, _l] = 5 + _l
            SWAPS_32[_m, 5 + _l] = _l
            SIGNS_32[_m, _l] = -1.0
        else:
            SWAPS_32[_m, _l] = _l
            SWAPS_32[_m, 5 + _l] = 5 + _l

ROLE_REAL = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=np.int32)
TEAM_REAL = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int32)
EPS_LOGIT = 1e-12
TOL_D = 1e-7

ESCENARIOS_PRINCIPALES = (
    "a_misma_linea",
    "b_rol_t20_d_le",
    "b_rol_t20_sin_tope",
    "b_rol_t35_d_le",
    "b_rol_t35_sin_tope",
    "ab_1swap_t20_d_le",
    "ab_1swap_t20_sin_tope",
    "ab_1swap_t35_d_le",
    "ab_1swap_t35_sin_tope",
    "ab_exh_t20_d_le",
    "ab_exh_t20_sin_tope",
    "ab_exh_t35_d_le",
    "ab_exh_t35_sin_tope",
    "ab_pareto_t20",
    "ab_minD_t20",
)

ESCENARIOS_JUGADORES = (
    "a_misma_linea",
    "b_rol_t20_d_le",
    "b_rol_t20_sin_tope",
    "b_rol_t35_d_le",
    "b_rol_t35_sin_tope",
    "ab_exh_t20_d_le",
    "ab_exh_t20_sin_tope",
    "ab_exh_t35_d_le",
    "ab_exh_t35_sin_tope",
    "ab_pareto_t20",
    "ab_minD_t20",
)

ETIQUETAS_ESC = {
    "real": "Real (Riot)",
    "a_misma_linea": "(a) Misma línea (32)",
    "b_rol_t20_d_le": "(b) 1-swap rol τ=0,20 · D≤D_real",
    "b_rol_t20_sin_tope": "(b) 1-swap rol τ=0,20 · sin tope",
    "b_rol_t35_d_le": "(b) 1-swap rol τ=0,35 · D≤D_real",
    "b_rol_t35_sin_tope": "(b) 1-swap rol τ=0,35 · sin tope",
    "ab_1swap_t20_d_le": "(a+b 1-swap) τ=0,20 · D≤D_real",
    "ab_1swap_t20_sin_tope": "(a+b 1-swap) τ=0,20 · sin tope",
    "ab_1swap_t35_d_le": "(a+b 1-swap) τ=0,35 · D≤D_real",
    "ab_1swap_t35_sin_tope": "(a+b 1-swap) τ=0,35 · sin tope",
    "ab_exh_t20_d_le": "(a+b exhaustivo) τ=0,20 · D≤D_real",
    "ab_exh_t20_sin_tope": "(a+b exhaustivo) τ=0,20 · sin tope",
    "ab_exh_t35_d_le": "(a+b exhaustivo) τ=0,35 · D≤D_real",
    "ab_exh_t35_sin_tope": "(a+b exhaustivo) τ=0,35 · sin tope",
    "ab_pareto_t20": "(a+b exh.) τ=0,20 · Δd_i≤0 todos",
    "ab_minD_t20": "(a+b exh.) τ=0,20 · min D luego |p-0,5|",
}

FRONTERA_DELTAS = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50, float("inf"))


@contextmanager
def reservar_recurso(etiqueta: str = "neon-swaps"):
    SALIDA.mkdir(parents=True, exist_ok=True)
    marca = SALIDA / "RECURSO_EN_USO.txt"
    while True:
        try:
            archivo = marca.open("x", encoding="utf-8")
        except FileExistsError:
            print(f"Esperando la liberación de {marca.name}...", flush=True)
            time.sleep(15)
        else:
            break
    try:
        with archivo:
            archivo.write(f"{etiqueta} {datetime.now().astimezone().isoformat()}\n")
        yield
    finally:
        if marca.exists():
            marca.unlink()


def aplicar_platt(p: np.ndarray, a: float, b: float) -> np.ndarray:
    z = logit(np.clip(p, EPS_LOGIT, 1.0 - EPS_LOGIT))
    return expit(a + b * z)


def num(x: float, dec: int = 4) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.{dec}f}".replace(".", ",")


def pct(x: float, dec: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{100.0 * x:.{dec}f} %".replace(".", ",")


def pp(x: float, dec: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{100.0 * x:.{dec}f}".replace(".", ",")


def cuenta(n: int) -> str:
    return f"{int(n):,}".replace(",", ".")


def cargar_esquema_sin_draft(D: dict, extras_all: np.ndarray) -> tuple[list[str], list[str], list[int], dict[str, slice], list[str]]:
    margen = tuple(c for s in STATS for c in (f"acum_{s}", f"v20_{s}"))
    conservar = set(margen + XJ_HISTORIAL)
    columnas = list(map(str, D["cols"]))
    base = [c for c in columnas if c in conservar]
    assert len(base) == 40, f"Esperadas 40 cols base en Xj, encontradas {len(base)}"

    keep = []
    grupos = {}
    inicio = 0
    for clave, nombres, elegidos in EXTRAS_SPEC:
        grupos[clave] = slice(inicio, inicio + len(nombres))
        keep.extend(inicio + i for i in elegidos)
        inicio += len(nombres)
    assert inicio == extras_all.shape[1], "Ancho de extras_referencia inesperado."

    orden_cols = (
        [f"Xj/{rol}/{c}" for rol in ROLES for c in base]
        + [f"{g}/{nombres[i]}" for g, nombres, elegidos in EXTRAS_SPEC for i in elegidos]
    )
    assert len(orden_cols) == 320

    ruta_sol = SALIDA / "calib_columnas_sin_draft.json"
    if ruta_sol.is_file():
        sol_json = json.loads(ruta_sol.read_text(encoding="utf-8"))
        assert sol_json["orden_columnas_modelo"] == orden_cols, "Desajuste de columnas frente a Sol!"
        assert sol_json["Xj_conservadas_por_linea"] == base
    return columnas, base, keep, grupos, orden_cols


def _query_cumsum_por_linea(
    pid_all: np.ndarray,
    lane_code_all: np.ndarray,
    values_list: list[np.ndarray],
    eval_pids: np.ndarray,
    eval_match_starts: np.ndarray,
) -> list[np.ndarray]:
    """Para cada partida en eval_match_starts y sus 10 jugadores (eval_pids: N_eval x 10),
    devuelve las sumas acumuladas estrictamente previas (< match_start) en cada una de las 5 líneas.
    Resultado: lista paralela a values_list, cada elemento shape (N_eval, 10, 5), float64.
    """
    n_rows = len(pid_all)
    base = np.int64(n_rows + 1)
    n_eval = len(eval_match_starts)
    q_pid = eval_pids.reshape(-1).astype(np.int64)
    q_row = np.repeat(eval_match_starts.astype(np.int64), 10)
    q_key = q_pid * base + q_row

    outs = [np.zeros((n_eval * 10, 5), dtype=np.float64) for _ in values_list]
    row_idx = np.arange(n_rows, dtype=np.int64)

    for l_idx in range(5):
        mask_l = lane_code_all == l_idx
        p_l = pid_all[mask_l].astype(np.int64)
        r_l = row_idx[mask_l]
        # Ordenar por (pid, row_idx): mergesort preserva orden creciente de row_idx dentro de cada pid
        ord_l = np.argsort(p_l, kind="mergesort")
        p_s = p_l[ord_l]
        r_s = r_l[ord_l]
        keys_l = p_s * base + r_s

        # Posición de la última aparición del jugador en línea l_idx con row < match_start
        pos = np.searchsorted(keys_l, q_key, side="left") - 1
        valid = (pos >= 0) & (p_s[np.maximum(pos, 0)] == q_pid)
        v_pos = pos[valid]

        # Bordes por jugador para cumsum segmentado sin bucle Python
        starts = np.flatnonzero(np.r_[True, p_s[1:] != p_s[:-1]])
        grp_id = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(p_s)]))

        for vi, arr in enumerate(values_list):
            vals_s = arr[mask_l][ord_l].astype(np.float64)
            cs = np.cumsum(vals_s)
            # Restar el cumsum acumulado justo antes del inicio de cada grupo de jugador
            prev_base = np.where(starts > 0, cs[np.maximum(starts - 1, 0)], 0.0)
            cs_grp = cs - prev_base[grp_id]
            outs[vi][valid, l_idx] = cs_grp[v_pos]

    return [o.reshape(n_eval, 10, 5) for o in outs]


def construir_historial_y_tensores(
    D: dict,
    corte: float,
    columnas: list[str],
    base: list[str],
    eval_idx: np.ndarray,
    v_idx_in_eval: np.ndarray,
    extraer_draft: bool = True,
) -> dict:
    """Reconstruye Xd y ML sin fuga de V (exactamente como Sol) y extrae TODOS los
    tensores causales por jugador y por (jugador, línea) para `eval_idx`.
    """
    t0 = time.perf_counter()
    nm = D["NM"]
    uso = [
        *ORDEN, "player_id", "win", "champion_id", "game_duration",
        "kills", "deaths", "assists", "kill_participation_pct", "damage_share_pct",
        "gold_share_pct", "total_damage_dealt_to_champions", "total_damage_taken",
        "gold_earned", "total_minions_killed", "neutral_minions_killed", "vision_score",
        "gold_10", "gold_15", "gold_20", "xp_15", "cs_15",
        "first_blood_kill", "first_tower_kill", "item6",
    ]
    print("[1/5] Leyendo full_dataset.csv y filtrando 5v5...", flush=True)
    df = pd.read_csv(
        REPO / "data" / "splits" / "train" / "full_dataset.csv",
        usecols=uso,
        dtype={"lane": "category", "match_id": "category", "player_id": "category"},
    )
    df = df[df["lane"].isin(ROLES)]
    df = df[df.groupby(["match_id", "team_id"], observed=True)["lane"].transform("nunique") == 5]
    por_partida = df.groupby("match_id", observed=True)["win"]
    df = df[(por_partida.transform("size") == 10) & (por_partida.transform("sum") == 5)]
    df = df.sort_values(ORDEN).reset_index(drop=True)
    assert np.array_equal(df["match_id"].to_numpy()[::10], D["Z"]["mids"]), "Desajuste mids CSV vs caché."
    df = lane_diffs(add_rates(df), ["gold_10", "gold_15", "xp_15", "cs_15"])

    antes = df["game_creation"] < corte
    escala = lane_scale(df.loc[antes, ["lane", *STATS]], list(STATS))

    # Códigos enteros rápidos para búsquedas causales por (player_id, lane)
    pid_codes = df["player_id"].cat.codes.to_numpy(np.int64)
    lane_codes = df["lane"].astype(str).map(ROLE_TO_IDX).to_numpy(np.int8)
    eval_match_starts = eval_idx.astype(np.int64) * 10
    eval_rows = (eval_match_starts[:, None] + np.arange(10, dtype=np.int64)[None, :]).reshape(-1)
    eval_pids = pid_codes[eval_rows].reshape(len(eval_idx), 10)

    # Verificar que los 10 jugadores de cada partida en eval son distintos
    for k in range(len(eval_idx)):
        if len(set(eval_pids[k].tolist())) != 10:
            raise ValueError(f"Partida con player_id repetido en eval_idx[{k}].")

    # 1. Xj base (40 cols por jugador) y ML (15 stats por jugador y por línea nueva)
    print("[2/5] Recalculando márgenes (Xj + ML) y tensores por (jugador, línea)...", flush=True)
    base_idx_in_xj = [columnas.index(c) for c in base]
    xj_eval = D["Xj"][eval_idx][..., base_idx_in_xj].reshape(len(eval_idx), 10, len(base)).astype(np.float64).copy()
    xd_sin = D["Xd"].reshape(nm, 5, -1)[..., base_idx_in_xj].astype(np.float32).copy()
    ml_diff_all = np.empty((nm, 5, len(STATS)), dtype=np.float32)
    ml_eval = np.empty((len(eval_idx), 10, 5, len(STATS)), dtype=np.float64)

    pj = df["player_id"]
    ones = pd.Series(1.0, index=df.index)
    win_f = df["win"].to_numpy(np.float64)
    n_tot_all = _previo(ones, pj).to_numpy(np.float64)
    wr_all = prior_winrate(df, window=20, k=10)
    wr_acum_all = wr_all["wr_acum"].to_numpy(np.float64)

    # Conteo y victorias previas por (jugador, línea) para las 5 líneas posibles
    n_pl_eval, w_pl_eval = _query_cumsum_por_linea(
        pid_codes, lane_codes, [np.ones(len(df), dtype=np.float64), win_f], eval_pids, eval_match_starts
    )
    n_tot_eval = n_tot_all[eval_rows].reshape(len(eval_idx), 10)
    wr_acum_eval = wr_acum_all[eval_rows].reshape(len(eval_idx), 10)

    # Chequeo secundario con draft: exceso de winrate (player, champ, lane) para las partidas de V
    wr_champ_lane_exc_v = None
    if extraer_draft:
        v_global = eval_idx[v_idx_in_eval]
        v_rows = (v_global.astype(np.int64)[:, None] * 10 + np.arange(10, dtype=np.int64)[None, :]).reshape(-1)
        pl_key = [df["player_id"], df["lane"]]
        n_pl_real = _previo(ones, pl_key).to_numpy(np.float64)
        w_pl_real = _previo(pd.Series(win_f, index=df.index), pl_key).to_numpy(np.float64)
        wr_lane_real = (w_pl_real + 8.0 * (0.5 + wr_acum_all)) / (n_pl_real + 8.0) - 0.5
        pcl_key = [df["player_id"], df["champion_id"], df["lane"]]
        n_pcl = _previo(ones, pcl_key).to_numpy(np.float64)
        w_pcl = _previo(pd.Series(win_f, index=df.index), pcl_key).to_numpy(np.float64)
        wcle = (w_pcl + 6.0 * (0.5 + wr_lane_real)) / (n_pcl + 6.0) - 0.5 - wr_lane_real
        wr_champ_lane_exc_v = wcle[v_rows].reshape(len(v_global), 10)
        del n_pl_real, w_pl_real, wr_lane_real, n_pcl, w_pcl, wcle

    for si, stat in enumerate(STATS):
        # Misma aritmética exacta que prior_margin y prior_margin_by con k_cl=20, k_j=5, k_g=10, window=20
        r_ser, v_ser = _residuo(df, stat, 20.0, escala)
        S_p = _previo(r_ser, pj)
        N_p = _previo(v_ser, pj)
        m_acum = S_p / (N_p + 5.0)
        S_20 = S_p - S_p.groupby(pj, sort=False).shift(20).fillna(0.0)
        N_20 = N_p - N_p.groupby(pj, sort=False).shift(20).fillna(0.0)
        m_v20 = S_20 / (N_20 + 5.0)

        i_ac = base.index(f"acum_{stat}")
        i_v20 = base.index(f"v20_{stat}")
        m_ac_np = m_acum.to_numpy(np.float32)
        m_v20_np = m_v20.to_numpy(np.float32)
        xd_sin[..., i_ac] = m_ac_np.reshape(nm, 2, 5)[:, 0] - m_ac_np.reshape(nm, 2, 5)[:, 1]
        xd_sin[..., i_v20] = m_v20_np.reshape(nm, 2, 5)[:, 0] - m_v20_np.reshape(nm, 2, 5)[:, 1]
        xj_eval[..., i_ac] = m_ac_np[eval_rows].reshape(len(eval_idx), 10)
        xj_eval[..., i_v20] = m_v20_np[eval_rows].reshape(len(eval_idx), 10)

        # ML sobre toda la base (para entrenamiento)
        grupo_pl = [pj, df["lane"]]
        S_pl_real = _previo(r_ser, grupo_pl)
        N_pl_real = _previo(v_ser, grupo_pl)
        ml_real = ((S_pl_real + 10.0 * m_acum) / (N_pl_real + 10.0) - m_acum).to_numpy(np.float32).reshape(nm, 2, 5)
        ml_diff_all[..., si] = ml_real[:, 0] - ml_real[:, 1]

        # ML contrafáctico para TODAS las 5 líneas en eval_idx
        S_pl_5, N_pl_5 = _query_cumsum_por_linea(
            pid_codes, lane_codes, [r_ser.to_numpy(np.float64), v_ser.to_numpy(np.float64)],
            eval_pids, eval_match_starts,
        )
        m_p_ev = m_acum.to_numpy(np.float64)[eval_rows].reshape(len(eval_idx), 10, 1)
        ml_eval[..., si] = (S_pl_5 + 10.0 * m_p_ev) / (N_pl_5 + 10.0) - m_p_ev

    # Recalcular log_horas_desde exactamente como Sol
    horas = (df["game_creation"] - df.groupby("player_id", observed=True)["game_creation"].shift()) / 3.6e6
    log_horas = np.log1p(horas)
    med_h = float(log_horas[antes].median())
    lh_np = log_horas.fillna(med_h).to_numpy(np.float32)
    i_lh = base.index("log_horas_desde")
    xd_sin[..., i_lh] = lh_np.reshape(nm, 2, 5)[:, 0] - lh_np.reshape(nm, 2, 5)[:, 1]
    xj_eval[..., i_lh] = lh_np[eval_rows].reshape(len(eval_idx), 10)

    # Verificar que share_lane y offrole recalculados coinciden con xj_eval en la línea real
    idx_sl = base.index("share_lane")
    idx_off = base.index("offrole")
    share_lane_5 = (n_pl_eval + 1.0) / (n_tot_eval[:, :, None] + 2.0)
    offrole_5 = ((n_tot_eval[:, :, None] >= 10.0) & (n_pl_eval / np.maximum(1.0, n_tot_eval[:, :, None]) < 0.2)).astype(np.float64)
    sl_real_chk = share_lane_5[:, np.arange(10), ROLE_REAL]
    off_real_chk = offrole_5[:, np.arange(10), ROLE_REAL]
    err_sl = float(np.max(np.abs(sl_real_chk - xj_eval[..., idx_sl])))
    err_off = float(np.max(np.abs(off_real_chk - xj_eval[..., idx_off])))
    assert err_sl < 1e-6 and err_off < 1e-6, f"Desajuste share_lane/offrole: {err_sl}, {err_off}"

    # ROL_WR contrafáctico (por jugador y línea nueva)
    wr_lane_5 = (w_pl_eval + 8.0 * (0.5 + wr_acum_eval[:, :, None])) / (n_pl_eval + 8.0) - 0.5
    wr_lane_diff_5 = wr_lane_5 - wr_acum_eval[:, :, None]
    logn_lane_5 = np.log1p(n_pl_eval)
    estreno_lane_5 = (n_pl_eval == 0.0).astype(np.float64)
    share_g10_5 = n_pl_eval / np.maximum(1.0, n_tot_eval[:, :, None])
    autofill_drop_5 = (1.0 - share_g10_5) * wr_lane_diff_5
    autofill_flag_5 = ((n_tot_eval[:, :, None] >= 15.0) & (share_g10_5 < 0.20)).astype(np.float64) * np.minimum(0.0, wr_lane_diff_5)
    rol_wr_lin_5 = np.stack(
        [wr_lane_5, wr_lane_diff_5, logn_lane_5, estreno_lane_5, autofill_drop_5, autofill_flag_5],
        axis=-1,
    )  # (N_eval, 10, 5, 6)

    # Extraer columnas crudas para g11 y g23 antes de liberar df
    fb_arr = df["first_blood_kill"].fillna(0).to_numpy(np.float32).reshape(nm, 10)
    ft_arr = df["first_tower_kill"].fillna(0).to_numpy(np.float32).reshape(nm, 10)
    gs_raw = df["gold_share_pct"].to_numpy(np.float32).reshape(nm, 10)
    ds_raw = df["damage_share_pct"].to_numpy(np.float32).reshape(nm, 10)
    if float(np.nanmedian(gs_raw)) > 1.5:
        gs_raw = gs_raw / 100.0
        ds_raw = ds_raw / 100.0
    gs_arr = np.nan_to_num(gs_raw, nan=0.20)
    ds_arr = np.nan_to_num(ds_raw, nan=0.20)
    g15_p = df["gold_15"].to_numpy(np.float32).reshape(nm, 10)
    g20_p = df["gold_20"].to_numpy(np.float32).reshape(nm, 10)
    it6_arr = df["item6"].fillna(3340).to_numpy(np.int32).reshape(nm, 10)

    # Para chequeo secundario con draft: diferencias @10/@15 de g23
    d4_g23 = None
    if extraer_draft:
        dur_m = df["game_duration"].to_numpy(np.float64).reshape(nm, 2, 5)[:, 0, 0]
        fit0 = ~D["val0"]

        def _diff_mat(col: str, min_dur: float) -> np.ndarray:
            a = df[col].to_numpy(np.float64).reshape(nm, 2, 5)
            d = a[:, 0, :] - a[:, 1, :]
            ok = (dur_m >= min_dur)[:, None] & np.isfinite(d)
            sc = float(np.nanstd(d[fit0[:, None] & ok])) + 1e-6
            return np.where(ok, d / sc, np.nan)

        d4_g23 = [
            _diff_mat("gold_15", 900.0),
            _diff_mat("xp_15", 900.0),
            _diff_mat("cs_15", 900.0),
            _diff_mat("gold_10", 600.0),
        ]

    del df, pj, ones, n_tot_all, wr_all, wr_acum_all, pid_codes, lane_codes
    gc.collect()

    # 2. Nivel histórico sin ventana (exactamente como Sol)
    print("[3/5] Recalculando MMR_HIST (sin ventana) y FOTO1_DIN por jugador...", flush=True)
    uri = (REPO / "data" / "lol_las.db").resolve().as_uri() + "?mode=ro&immutable=1"
    with closing(sqlite3.connect(uri, uri=True)) as con:
        verificar_foto1(con)
        foto_niv = leer_foto1(con, "p.tier, p.rank, p.league_points, p.wins, p.losses")
        foto_f1 = pd.read_sql_query(
            "SELECT m.player_id, p.tier, p.league_points, p.wins, p.losses "
            "FROM players p JOIN player_id_map m ON p.puuid = m.puuid",
            con,
        ).drop_duplicates("player_id")

    foto_niv["z"] = [lp_absoluto(t, r, l) for t, r, l in zip(foto_niv.tier, foto_niv["rank"], foto_niv.league_points)]
    foto_niv = foto_niv.assign(partidas=foto_niv.wins + foto_niv.losses).sort_values("partidas")
    x_map = foto_niv.drop_duplicates("player_id", keep="last").set_index("player_id")["z"]
    df_niv = pd.DataFrame({
        "match_id": np.repeat(np.arange(nm), 10),
        "game_creation": np.repeat(D["t"], 10),
        "player_id": pd.Categorical(D["B"]["player"].reshape(-1)),
    })
    df_niv["x"] = x_map.reindex(df_niv.player_id.cat.categories).to_numpy()[df_niv.player_id.cat.codes]
    df_niv["player_id"] = df_niv.player_id.cat.codes
    mu_niv = float(df_niv.loc[df_niv.game_creation < corte, "x"].mean())
    nivel = prior_lobby_level(df_niv, "x", mu0=mu_niv)
    mmr_all_jug = np.column_stack((
        nivel.nivel_acum - mu_niv,
        nivel.nivel_v10 - mu_niv,
        np.log1p(nivel.n_nivel),
    )).astype(np.float32).reshape(nm, 2, 5, 3)
    mmr_diff_all = (mmr_all_jug[:, 0] - mmr_all_jug[:, 1]).mean(axis=1)
    mmr_eval = mmr_all_jug[eval_idx].reshape(len(eval_idx), 10, 3).astype(np.float64)
    del df_niv, nivel, mmr_all_jug

    # 3. FOTO1_DIN por jugador (idéntico a brainstorm_g_01_foto1_wl.py)
    pl_flat = D["B"]["player"].reshape(-1)
    pr_flat = np.repeat(D["post"], 10)
    f_map = foto_f1.set_index("player_id")
    has_f1 = pd.Series(pl_flat).isin(f_map.index).to_numpy() & pr_flat
    aligned = f_map.reindex(pl_flat)
    W0 = aligned["wins"].fillna(0.0).to_numpy(np.float64)
    L0 = aligned["losses"].fillna(0.0).to_numpy(np.float64)
    zero_m = (
        has_f1
        & (aligned["tier"].fillna("").to_numpy() == "MASTER")
        & (aligned["league_points"].fillna(-1).to_numpy() == 0)
    ).astype(np.float32)
    y_jug = np.repeat(D["y"], 10).astype(np.float64)
    es_azul = np.tile(np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=bool), nm)
    win_slot = np.where(es_azul, y_jug, 1.0 - y_jug)
    df_post = pd.DataFrame({
        "_pos": np.arange(nm * 10),
        "player_id": pl_flat,
        "t": np.repeat(D["t"], 10),
        "m": np.repeat(np.arange(nm), 10),
        "post": pr_flat,
        "win": win_slot,
    })
    sub_p = df_post[df_post["post"]].sort_values(["t", "m"], kind="mergesort").copy()
    sub_p["dw_prev"] = sub_p.groupby("player_id", sort=False)["win"].cumsum() - sub_p["win"]
    sub_p["dn_prev"] = sub_p.groupby("player_id", sort=False).cumcount().astype(np.float64)
    sub_p["dl_prev"] = sub_p["dn_prev"] - sub_p["dw_prev"]
    dw_all = np.zeros(nm * 10, dtype=np.float64)
    dl_all = np.zeros(nm * 10, dtype=np.float64)
    dw_all[sub_p["_pos"].to_numpy()] = sub_p["dw_prev"].to_numpy()
    dl_all[sub_p["_pos"].to_numpy()] = sub_p["dl_prev"].to_numpy()
    Wt = W0 + dw_all
    Lt = L0 + dl_all
    tot_t = Wt + Lt
    f1_jug_all = np.stack([
        np.where(has_f1, (Wt + 10.0 * 0.5) / (tot_t + 20.0) - 0.5, 0.0).astype(np.float32).reshape(nm, 10),
        np.where(has_f1, np.log1p(tot_t) - 5.3, 0.0).astype(np.float32).reshape(nm, 10),
        np.where(has_f1, Wt - Lt, 0.0).astype(np.float32).reshape(nm, 10),
        zero_m.reshape(nm, 10),
    ], axis=-1)
    f1_eval = f1_jug_all[eval_idx].astype(np.float64)
    del df_post, sub_p, f1_jug_all, aligned

    # 4. Pasada causal g11 para extraer estados por jugador en eval_idx
    print("[4/5] Extrayendo estados causales Kalman / Tempo / Econ por jugador...", flush=True)
    g11_eval = _extraer_g11_por_jugador(
        D, eval_idx, fb_arr, ft_arr, gs_arr, ds_arr, g15_p, g20_p, it6_arr
    )
    del fb_arr, ft_arr, gs_arr, ds_arr, g15_p, g20_p, it6_arr

    # 5. Pasadas rápidas para el chequeo secundario CON DRAFT (solo V Master+)
    draft_eval = None
    if extraer_draft:
        print("[5/5] Extrayendo sinergias causales y duelo @10/@15 para chequeo con draft...", flush=True)
        v_global = eval_idx[v_idx_in_eval]
        syn_orig, syn_cross = _extraer_sinergias_v(D, v_global)
        exp_m_v = _extraer_matchup_exp_v(D, v_global, d4_g23)
        draft_eval = {
            "syn_orig": syn_orig,
            "syn_cross": syn_cross,
            "exp_m_v": exp_m_v,
            "wr_champ_lane_exc_v": wr_champ_lane_exc_v,
            "cbias_v": g11_eval["cbias"][v_idx_in_eval],
            "pbias_v": g11_eval["pbias"][v_idx_in_eval],
        }
        del d4_g23

    print(f"    Tensores contrafácticos construidos en {time.perf_counter() - t0:.1f} s.", flush=True)
    return {
        "xd_sin": xd_sin.reshape(nm, -1),
        "ml_diff_all": ml_diff_all.reshape(nm, -1),
        "mmr_diff_all": mmr_diff_all,
        "xj_eval": xj_eval,           # (N_eval, 10, 40)
        "share_lane_5": share_lane_5, # (N_eval, 10, 5)
        "offrole_5": offrole_5,       # (N_eval, 10, 5)
        "ml_eval": ml_eval,           # (N_eval, 10, 5, 15)
        "mmr_eval": mmr_eval,         # (N_eval, 10, 3)
        "f1_eval": f1_eval,           # (N_eval, 10, 4)
        "rol_wr_lin_5": rol_wr_lin_5, # (N_eval, 10, 5, 6)
        "wr_lane_5": wr_lane_5,       # (N_eval, 10, 5)
        "wr_lane_diff_5": wr_lane_diff_5,  # (N_eval, 10, 5)
        "g11_eval": g11_eval,
        "draft_eval": draft_eval,
    }


def _extraer_g11_por_jugador(
    D: dict,
    eval_idx: np.ndarray,
    fb_arr: np.ndarray,
    ft_arr: np.ndarray,
    gs_arr: np.ndarray,
    ds_arr: np.ndarray,
    g15_p: np.ndarray,
    g20_p: np.ndarray,
    it6_arr: np.ndarray,
) -> dict[str, np.ndarray]:
    """Replica exactamente el bucle de brainstorm_g_11_elo_clutch_tempo.py y captura los
    valores causales previos por jugador en las partidas de `eval_idx`.
    """
    nm = D["NM"]
    Z = D["Z"]
    mids = Z["mids"].astype(str)
    t_arr = D["t"].astype(np.int64)
    y_arr = D["y"].astype(np.int8)
    val0 = D["val0"].astype(bool)
    post = D["post"].astype(bool)
    champ = Z["champ"].reshape(nm, 10)
    dur_min = Z["D_dur_min"].astype(np.float32)
    dg15 = Z["D_d_gold_15"].astype(np.float32)
    dxp15 = Z["D_d_xp_15"].astype(np.float32)
    dk15 = Z["D_d_kills_15"].astype(np.float32)

    players = D["B"]["player"].reshape(nm, 10).astype(str)
    uniq_p, p_inv = np.unique(players.reshape(-1), return_inverse=True)
    pid_mat = p_inv.reshape(nm, 10).tolist()
    n_p = len(uniq_p)

    cl_pairs = champ * 5 + np.tile(np.arange(5), 2)[None, :]
    uniq_cl, cl_inv = np.unique(cl_pairs.reshape(-1), return_inverse=True)
    cl_mat = cl_inv.reshape(nm, 10).tolist()
    n_cl = len(uniq_cl)

    swap_arr = (it6_arr != 3340).astype(np.float32)
    dmg_gold_eff = (ds_arr - gs_arr).astype(np.float32)
    surge_20_15 = (g20_p - g15_p).astype(np.float32)
    surge_valid = np.isfinite(surge_20_15[:, :5]) & np.isfinite(surge_20_15[:, 5:])
    surge_diff = np.where(surge_valid, (surge_20_15[:, :5] - surge_20_15[:, 5:]) / 1000.0, 0.0)

    m_fit15 = (~val0) & (~post) & np.isfinite(dg15) & np.isfinite(dxp15) & np.isfinite(dk15) & (dur_min >= 15.0)
    X15 = np.c_[
        np.nan_to_num(dg15, 0.0) / 1000.0,
        np.nan_to_num(dxp15, 0.0) / 1000.0,
        np.nan_to_num(dk15, 0.0),
    ]
    lr15 = LogisticRegression(C=1.0, max_iter=500).fit(X15[m_fit15], y_arr[m_fit15])
    p_exp15_all = lr15.predict_proba(X15)[:, 1].astype(np.float64)
    valid15 = np.isfinite(dg15) & (dur_min >= 15.0)

    eval_map = {int(m): pos for pos, m in enumerate(eval_idx)}
    n_ev = len(eval_idx)
    out = {
        k: np.zeros((n_ev, 10), dtype=np.float64)
        for k in (
            "r_elo", "r_slow", "r_opp", "r_mrg", "r_mrg_slow", "inv_sqrt_cnt",
            "fbv", "ftv", "swv", "durv", "pbias", "cbias", "srv", "dgev",
        )
    }

    r_elo = [0.0] * n_p
    r_slow = [0.0] * n_p
    r_opp = [0.0] * n_p
    r_mrg = [0.0] * n_p
    r_mrg_slow = [0.0] * n_p
    cnt_p = [0] * n_p

    fb_sum = [0.0] * n_p
    ft_sum = [0.0] * n_p
    sw_sum = [0.0] * n_p
    dur_sum = [0.0] * n_p
    ew_cnt = [0.0] * n_p
    wr_e_sum = [0.0] * n_p
    wr_e_n = [0.0] * n_p
    wr_l_sum = [0.0] * n_p
    wr_l_n = [0.0] * n_p
    sr_sum = [0.0] * n_p
    sr_w = [0.0] * n_p
    dg_sum = [0.0] * n_p
    cl_e_win = [0.0] * n_cl
    cl_e_n = [0.0] * n_cl
    cl_l_win = [0.0] * n_cl
    cl_l_n = [0.0] * n_cl

    dec20 = math.exp(-math.log(2.0) / 20.0)
    dec25 = math.exp(-math.log(2.0) / 25.0)
    dec30 = math.exp(-math.log(2.0) / 30.0)

    order = np.lexsort((mids, t_arr))
    fb_l = fb_arr.tolist()
    ft_l = ft_arr.tolist()
    sw_l = swap_arr.tolist()
    dge_l = dmg_gold_eff.tolist()
    sd_l = surge_diff.tolist()
    sv_l = surge_valid.tolist()

    for m in order:
        mi = int(m)
        ps = pid_mat[mi]
        cls = cl_mat[mi]
        yb = float(y_arr[mi])
        dm = float(dur_min[mi])
        v15 = bool(valid15[mi])
        p15 = float(p_exp15_all[mi])

        rb = [r_elo[ps[i]] for i in range(5)]
        rr = [r_elo[ps[5 + i]] for i in range(5)]
        rmb = [r_mrg[ps[i]] for i in range(5)]
        rmr = [r_mrg[ps[5 + i]] for i in range(5)]

        pos = eval_map.get(mi)
        if pos is not None:
            for slot in range(10):
                p = ps[slot]
                cl = cls[slot]
                out["r_elo"][pos, slot] = r_elo[p]
                out["r_slow"][pos, slot] = r_slow[p]
                out["r_opp"][pos, slot] = r_opp[p]
                out["r_mrg"][pos, slot] = r_mrg[p]
                out["r_mrg_slow"][pos, slot] = r_mrg_slow[p]
                out["inv_sqrt_cnt"][pos, slot] = 1.0 / math.sqrt(1.0 + cnt_p[p])
                out["fbv"][pos, slot] = (fb_sum[p] + 12.0 * 0.10) / (ew_cnt[p] + 12.0)
                out["ftv"][pos, slot] = (ft_sum[p] + 12.0 * 0.10) / (ew_cnt[p] + 12.0)
                out["swv"][pos, slot] = (sw_sum[p] + 10.0 * 0.79) / (ew_cnt[p] + 10.0)
                out["durv"][pos, slot] = (dur_sum[p] + 10.0 * 28.0) / (ew_cnt[p] + 10.0) - 28.0
                out["pbias"][pos, slot] = (
                    ((wr_l_sum[p] + 6.0 * 0.5) / (wr_l_n[p] + 12.0))
                    - ((wr_e_sum[p] + 6.0 * 0.5) / (wr_e_n[p] + 12.0))
                )
                out["cbias"][pos, slot] = (
                    ((cl_l_win[cl] + 10.0 * 0.5) / (cl_l_n[cl] + 20.0))
                    - ((cl_e_win[cl] + 10.0 * 0.5) / (cl_e_n[cl] + 20.0))
                )
                out["srv"][pos, slot] = sr_sum[p] / (sr_w[p] + 10.0)
                out["dgev"][pos, slot] = dg_sum[p] / (ew_cnt[p] + 10.0)

        mb = sum(rb) * 0.2
        mr = sum(rr) * 0.2
        diff_elo = (mb - mr) / 200.0 - 0.105
        p_hat = 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, diff_elo))))
        err_out = yb - p_hat

        m_rmb = sum(rmb) * 0.2
        m_rmr = sum(rmr) * 0.2
        diff_mrg = (m_rmb - m_rmr) / 200.0 - 0.105
        p_hat_m = 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, diff_mrg))))
        mrg_target = 0.70 * yb + 0.30 * (p15 if v15 else yb)
        err_mrg = mrg_target - p_hat_m

        lobby_elo = (mb + mr) * 0.5
        lobby_mrg = (m_rmb + m_rmr) * 0.5
        fb_row = fb_l[mi]
        ft_row = ft_l[mi]
        sw_row = sw_l[mi]
        dge_row = dge_l[mi]
        sd_row = sd_l[mi]
        sv_row = sv_l[mi]

        for slot in range(10):
            p = ps[slot]
            cl = cls[slot]
            sgn = 1.0 if slot < 5 else -1.0
            win_p = yb if slot < 5 else (1.0 - yb)
            n = cnt_p[p]
            lr_w = 32.0 / math.sqrt(1.0 + n / 25.0)
            lr_pull = 0.22 if n < 10 else 0.10 / math.sqrt(1.0 + (n - 10) / 35.0)
            other9_e = (10.0 * lobby_elo - r_elo[p]) / 9.0
            other9_m = (10.0 * lobby_mrg - r_mrg[p]) / 9.0

            new_e = r_elo[p] + lr_w * sgn * err_out + lr_pull * (other9_e - r_elo[p])
            new_m = r_mrg[p] + lr_w * sgn * err_mrg + lr_pull * (other9_m - r_mrg[p])
            new_o = r_opp[p] + (0.28 if n < 10 else 0.12 / math.sqrt(1.0 + (n - 10) / 30.0)) * (other9_e - r_opp[p])

            r_elo[p] = new_e
            r_slow[p] = 0.94 * r_slow[p] + 0.06 * new_e
            r_mrg[p] = new_m
            r_mrg_slow[p] = 0.94 * r_mrg_slow[p] + 0.06 * new_m
            r_opp[p] = new_o
            cnt_p[p] = n + 1

            fb_sum[p] = fb_sum[p] * dec25 + fb_row[slot]
            ft_sum[p] = ft_sum[p] * dec25 + ft_row[slot]
            sw_sum[p] = sw_sum[p] * dec25 + sw_row[slot]
            dur_sum[p] = dur_sum[p] * dec25 + dm
            dg_sum[p] = dg_sum[p] * dec25 + dge_row[slot]
            ew_cnt[p] = ew_cnt[p] * dec25 + 1.0

            if dm < 26.0:
                wr_e_sum[p] = wr_e_sum[p] * dec30 + win_p
                wr_e_n[p] = wr_e_n[p] * dec30 + 1.0
                cl_e_win[cl] += win_p
                cl_e_n[cl] += 1.0
            elif dm >= 29.0:
                wr_l_sum[p] = wr_l_sum[p] * dec30 + win_p
                wr_l_n[p] = wr_l_n[p] * dec30 + 1.0
                cl_l_win[cl] += win_p
                cl_l_n[cl] += 1.0

            lane = slot % 5
            if sv_row[lane]:
                sr_sum[p] = sr_sum[p] * dec20 + sd_row[lane] * sgn
                sr_w[p] = sr_w[p] * dec20 + 1.0
    return out


def _extraer_sinergias_v(D: dict, v_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calcula exactamente las 3 sinergias aliadas (BOT+SUP, TOP+JG, MID+JG) tanto para
    los pares originales (s[l1] == s[l2]) como para los pares cruzados (s[l1] != s[l2]) en `v_idx`.
    """
    nm = D["NM"]
    t = D["t"]
    y = D["y"].astype(float)
    champ = D["Z"]["champ"].reshape(nm, 2, 5).astype(int)
    ord_idx = np.lexsort((np.arange(nm), t))

    g_cl = defaultdict(float)
    n_cl = defaultdict(int)
    g_syn = defaultdict(float)
    n_syn = defaultdict(int)
    SYN_PAIRS = ((0, 4), (3, 1), (2, 1))

    v_map = {int(m): i for i, m in enumerate(v_idx)}
    syn_orig = np.zeros((len(v_idx), 3), dtype=np.float64)
    syn_cross = np.zeros((len(v_idx), 3), dtype=np.float64)

    for m in ord_idx:
        mi = int(m)
        ym = y[mi]
        pos = v_map.get(mi)
        if pos is not None:
            wr_side = [[(g_cl[(int(champ[mi, s, l]), l)] + 15.0) / (n_cl[(int(champ[mi, s, l]), l)] + 30.0)
                        for l in range(5)] for s in (0, 1)]
            for idx_p, (l1, l2) in enumerate(SYN_PAIRS):
                # Original: (azul l1, azul l2) vs (rojo l1, rojo l2)
                vals_o = [0.0, 0.0]
                for s in (0, 1):
                    c1, c2 = int(champ[mi, s, l1]), int(champ[mi, s, l2])
                    ks = (idx_p, c1, c2) if c1 <= c2 else (idx_p, c2, c1)
                    pe = min(0.9, max(0.1, 0.5 + (wr_side[s][l1] - 0.5) + (wr_side[s][l2] - 0.5)))
                    vals_o[s] = (g_syn[ks] + 30.0 * pe) / (n_syn[ks] + 30.0) - pe
                syn_orig[pos, idx_p] = vals_o[0] - vals_o[1]
                # Cruzado: (azul l1, rojo l2) vs (rojo l1, azul l2)
                vals_c = [0.0, 0.0]
                for s in (0, 1):
                    c1, c2 = int(champ[mi, s, l1]), int(champ[mi, 1 - s, l2])
                    ks = (idx_p, c1, c2) if c1 <= c2 else (idx_p, c2, c1)
                    pe = min(0.9, max(0.1, 0.5 + (wr_side[s][l1] - 0.5) + (wr_side[1 - s][l2] - 0.5)))
                    vals_c[s] = (g_syn[ks] + 30.0 * pe) / (n_syn[ks] + 30.0) - pe
                syn_cross[pos, idx_p] = vals_c[0] - vals_c[1]

        for s in (0, 1):
            ws = ym if s == 0 else (1.0 - ym)
            for l in range(5):
                kcl = (int(champ[mi, s, l]), l)
                g_cl[kcl] += ws
                n_cl[kcl] += 1
            for idx_p, (l1, l2) in enumerate(SYN_PAIRS):
                c1, c2 = int(champ[mi, s, l1]), int(champ[mi, s, l2])
                ks = (idx_p, c1, c2) if c1 <= c2 else (idx_p, c2, c1)
                g_syn[ks] += ws
                n_syn[ks] += 1
    return syn_orig, syn_cross


def _extraer_matchup_exp_v(D: dict, v_idx: np.ndarray, d4: list[np.ndarray]) -> np.ndarray:
    """Extrae `exp_m[len(v_idx), 4, 5]` causal exacto (idéntico a brainstorm_g_23)."""
    nm = D["NM"]
    mids = D["Z"]["mids"].astype(str)
    t_arr = D["t"].astype(np.int64)
    champ = D["Z"]["champ"].reshape(nm, 2, 5).astype(np.int64)
    cl_mat = (champ * 5 + np.arange(5)[None, None, :]).tolist()
    ch_l = champ.tolist()
    d_lists = [m.tolist() for m in d4]

    cl_sum = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    cl_cnt = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    pair_sum = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    pair_cnt = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    k_m, k_cl = 12.0, 20.0

    v_map = {int(m): i for i, m in enumerate(v_idx)}
    exp_v = np.zeros((len(v_idx), 4, 5), dtype=np.float64)
    order = np.lexsort((mids, t_arr))

    for m in order:
        mi = int(m)
        cb_row, cr_row = ch_l[mi][0], ch_l[mi][1]
        clb_row, clr_row = cl_mat[mi][0], cl_mat[mi][1]
        pos = v_map.get(mi)
        if pos is not None:
            for l in range(5):
                cb, cr = cb_row[l], cr_row[l]
                if cb == cr:
                    continue
                sgn = 1.0 if cb < cr else -1.0
                pk = (cb, cr, l) if cb < cr else (cr, cb, l)
                ps, pc = pair_sum[pk], pair_cnt[pk]
                cbs, cbc = cl_sum[clb_row[l]], cl_cnt[clb_row[l]]
                crs, crc = cl_sum[clr_row[l]], cl_cnt[clr_row[l]]
                for s_i in range(4):
                    prior_dif = cbs[s_i] / (cbc[s_i] + k_cl) - crs[s_i] / (crc[s_i] + k_cl)
                    exp_v[pos, s_i, l] = (sgn * ps[s_i] + k_m * prior_dif) / (pc[s_i] + k_m)

        for s_i in range(4):
            obs_row = d_lists[s_i][mi]
            for l in range(5):
                obs = obs_row[l]
                if not math.isfinite(obs):
                    continue
                cb, cr = cb_row[l], cr_row[l]
                cl_sum[clb_row[l]][s_i] += obs
                cl_cnt[clb_row[l]][s_i] += 1.0
                cl_sum[clr_row[l]][s_i] -= obs
                cl_cnt[clr_row[l]][s_i] += 1.0
                if cb != cr:
                    sgn = 1.0 if cb < cr else -1.0
                    pk = (cb, cr, l) if cb < cr else (cr, cb, l)
                    pair_sum[pk][s_i] += sgn * obs
                    pair_cnt[pk][s_i] += 1.0
    return exp_v


def ajustar_modelo_sin_draft(X_all: np.ndarray, y: np.ndarray, tr_idx: np.ndarray, C: float = 0.05) -> dict:
    """Ajusta logística estandarizada (idéntica a brainstorm_base.ajustar_predecir con con_draft=False)
    y pliega la estandarización en pesos efectivos (B0, W_eff) en espacio crudo.
    """
    mu = X_all[tr_idx].mean(axis=0).astype(np.float64)
    sd = X_all[tr_idx].std(axis=0).astype(np.float64) + 1e-6
    X_tr_s = sparse.csr_matrix(((X_all[tr_idx] - mu.astype(np.float32)) / sd.astype(np.float32)).astype(np.float32))
    lr = LogisticRegression(C=C, max_iter=4000).fit(X_tr_s, y[tr_idx])
    coef_std = lr.coef_[0].astype(np.float64)
    intercept_std = float(lr.intercept_[0])
    W_eff = coef_std / sd
    B0 = intercept_std - float(np.dot(W_eff, mu))
    return {
        "lr": lr,
        "mu": mu,
        "sd": sd,
        "coef_std": coef_std,
        "intercept_std": intercept_std,
        "W": W_eff,
        "B0": B0,
    }


def precomputar_modelo_eval(modelo: dict, T: dict, base: list[str]) -> tuple[float, np.ndarray, np.ndarray]:
    """Dado un modelo sin draft (B0, W de 320 cols) y los tensores T para `eval_idx`,
    factoriza las 310 columnas lineales en `C_lin[N_eval, 10, 5]` y devuelve `(B0, C_lin, w_nl[10])`.
    """
    W = modelo["W"]
    B0 = modelo["B0"]
    n_ev = T["xj_eval"].shape[0]
    C_lin = np.zeros((n_ev, 10, 5), dtype=np.float64)

    idx_sl = base.index("share_lane")
    idx_off = base.index("offrole")

    # 1. Xj (cols 0..199 = 5 roles x 40 vars)
    W_xj = W[0:200].reshape(5, 40)
    mask_solo_jug = np.ones(40, dtype=bool)
    mask_solo_jug[idx_sl] = False
    mask_solo_jug[idx_off] = False
    # Producto punto de las 38 features solo-jugador contra W_xj[:, mask_solo_jug] -> (n_ev, 10, 5)
    C_lin += np.einsum("ijf,lf->ijl", T["xj_eval"][..., mask_solo_jug], W_xj[:, mask_solo_jug])
    # + share_lane y offrole recalculados por línea nueva
    C_lin += T["share_lane_5"] * W_xj[:, idx_sl][None, None, :]
    C_lin += T["offrole_5"] * W_xj[:, idx_off][None, None, :]

    # 2. ML (cols 200..274 = 5 roles x 15 stats)
    W_ml = W[200:275].reshape(5, 15)
    C_lin += np.einsum("ijls,ls->ijl", T["ml_eval"], W_ml)

    # 3. MMR_HIST (cols 275..277, media por equipo -> factor 0.2, independiente de línea)
    mmr_c = (0.2 * (T["mmr_eval"] @ W[275:278]))[:, :, None]
    C_lin += mmr_c

    # 4. FOTO1_DIN (cols 278..281, media por equipo -> factor 0.2, independiente de línea)
    f1_c = (0.2 * (T["f1_eval"] @ W[278:282]))[:, :, None]
    C_lin += f1_c

    # 5. ROL_WR medias (cols 282..287 -> factor 0.2, dependiente de (jugador, línea))
    C_lin += 0.2 * np.einsum("ijlf,f->ijl", T["rol_wr_lin_5"], W[282:288])

    # 6. KALMAN_OUT lineales (cols 290, 295, 296, 297..301)
    g11 = T["g11_eval"]
    r_elo_s = g11["r_elo"] / 100.0
    r_forma_s = (g11["r_elo"] - g11["r_slow"]) / 100.0
    r_opp_s = g11["r_opp"] / 100.0
    C_lin += (0.2 * (W[290] * r_elo_s + W[295] * r_forma_s + W[296] * r_opp_s))[:, :, None]
    C_lin += r_elo_s[:, :, None] * W[297:302][None, None, :]

    # 7. KALMAN_MRG lineales (cols 302, 305, 306, 307)
    r_mrg_s = g11["r_mrg"] / 100.0
    r_mrg_f_s = (g11["r_mrg"] - g11["r_mrg_slow"]) / 100.0
    r_exc_s = (g11["r_elo"] - g11["r_opp"]) / 100.0
    C_lin += (0.2 * (W[302] * r_mrg_s + W[305] * r_mrg_f_s + W[306] * g11["inv_sqrt_cnt"] + W[307] * r_exc_s))[:, :, None]

    # 8. TEMPO_OBJ (cols 309..315, TODAS lineales)
    C_lin += (
        W[309] * g11["fbv"]
        + W[311] * g11["ftv"]
        + W[313] * g11["swv"]
        + 0.2 * W[314] * g11["durv"]
        + 0.2 * W[315] * g11["pbias"]
    )[:, :, None]
    # Col 310: fbv en JG (1) + SUP (4)
    C_lin[:, :, 1] += W[310] * g11["fbv"]
    C_lin[:, :, 4] += W[310] * g11["fbv"]
    # Col 312: ftv en BOT (0) + TOP (3)
    C_lin[:, :, 0] += W[312] * g11["ftv"]
    C_lin[:, :, 3] += W[312] * g11["ftv"]

    # 9. ECON_MID lineales (cols 316, 318, 319)
    C_lin += (0.2 * (W[316] * g11["srv"] + W[318] * g11["dgev"]))[:, :, None]
    # Col 319: dgev en BOT (0) + MID (2)
    C_lin[:, :, 0] += W[319] * g11["dgev"]
    C_lin[:, :, 2] += W[319] * g11["dgev"]

    # 10 pesos de columnas NO LINEALES:
    # [0] ROL_WR/wr_linea_minimo (288), [1] ROL_WR/exceso_wr_linea_minimo (289),
    # [2] KALMAN_OUT/minimo (291), [3] KALMAN_OUT/maximo (292),
    # [4] KALMAN_OUT/desvio (293), [5] KALMAN_OUT/media_dos_mayores (294),
    # [6] KALMAN_MRG/minimo (303), [7] KALMAN_MRG/maximo (304),
    # [8] KALMAN_MRG/brecha_lobby_absoluta (308), [9] ECON_MID/crecimiento_oro_maximo (317)
    w_nl = np.array([
        W[288], W[289], W[291], W[292], W[293], W[294], W[303], W[304], W[308], W[317]
    ], dtype=np.float64)
    return B0, C_lin, w_nl


def evaluar_configs_partida(
    cfgs: np.ndarray,
    B0: float,
    C_lin_m: np.ndarray,
    w_nl: np.ndarray,
    wr_lane_m: np.ndarray,
    wr_lane_diff_m: np.ndarray,
    r_elo_m: np.ndarray,
    r_mrg_m: np.ndarray,
    srv_m: np.ndarray,
) -> np.ndarray:
    """Evalúa `p_azul` exacta para K configuraciones `cfgs` (K, 10) de una partida."""
    B = cfgs[:, :5]
    R = cfgs[:, 5:]
    L = np.arange(5)[None, :]

    z = B0 + C_lin_m[B, L].sum(axis=1) - C_lin_m[R, L].sum(axis=1)

    # ROL_WR mínimos
    z += w_nl[0] * (wr_lane_m[B, L].min(axis=1) - wr_lane_m[R, L].min(axis=1))
    z += w_nl[1] * (wr_lane_diff_m[B, L].min(axis=1) - wr_lane_diff_m[R, L].min(axis=1))

    # KALMAN_OUT min, max, std, top2
    rb = r_elo_m[B]
    rr = r_elo_m[R]
    rb_s = np.sort(rb, axis=1)
    rr_s = np.sort(rr, axis=1)
    z += w_nl[2] * ((rb_s[:, 0] - rr_s[:, 0]) * 0.01)
    z += w_nl[3] * ((rb_s[:, 4] - rr_s[:, 4]) * 0.01)
    z += w_nl[4] * ((rb.std(axis=1) - rr.std(axis=1)) * 0.01)
    z += w_nl[5] * (((rb_s[:, 3] + rb_s[:, 4]) - (rr_s[:, 3] + rr_s[:, 4])) * 0.005)

    # KALMAN_MRG min, max, |mb - mr|
    rmb = r_mrg_m[B]
    rmr = r_mrg_m[R]
    z += w_nl[6] * ((rmb.min(axis=1) - rmr.min(axis=1)) * 0.01)
    z += w_nl[7] * ((rmb.max(axis=1) - rmr.max(axis=1)) * 0.01)
    z += w_nl[8] * (np.abs(rb.mean(axis=1) - rr.mean(axis=1)) * 0.01)

    # ECON_MID max
    z += w_nl[9] * (srv_m[B].max(axis=1) - srv_m[R].max(axis=1))
    return expit(z)


def generar_configs_partida(pis_m: np.ndarray, tau_min: float = 0.20) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Genera TODAS las configuraciones válidas (equipo, rol) donde cada jugador conserva su
    rol real o pasa a un rol nuevo con pi_i(r) >= tau_min.

    Devuelve:
        cfgs: (K, 10) int32, donde cfgs[0] es siempre la asignación real [0..9].
        n_cambios_rol: (K,) int32, cantidad de jugadores cuyo rol difiere del real (0, 2, 3, ...).
        min_pi_changed: (K,) float64, mínimo pi_i(r_nuevo) entre quienes cambian de rol (1.0 si nadie cambia).
        es_solo_b: (K,) bool, True si es la real o un único 2-swap de roles desde la real (los otros 8 sin tocar).
    """
    allow = pis_m >= tau_min
    allow[np.arange(10), ROLE_REAL] = True

    opts = [np.flatnonzero(allow[i]).tolist() for i in range(10)]
    order = sorted(range(10), key=lambda i: len(opts[i]))
    cap = [2, 2, 2, 2, 2]
    assign = [0] * 10
    particiones = []

    def bt(pos: int):
        if pos == 10:
            particiones.append(tuple(assign))
            return
        u = order[pos]
        for r in opts[u]:
            if cap[r] > 0:
                cap[r] -= 1
                assign[u] = r
                bt(pos + 1)
                cap[r] += 1

    bt(0)

    # Poner la partición real primero para garantizar que cfgs[0] == [0..9]
    real_part = tuple(ROLE_REAL.tolist())
    particiones_ord = [real_part] + [p for p in particiones if p != real_part]

    all_cfgs = []
    all_n_cr = []
    all_min_pi = []
    all_solo_b = []

    for part in particiones_ord:
        # Agrupar los 2 jugadores asignados a cada rol l=0..4
        by_role = [[] for _ in range(5)]
        n_cr = 0
        min_pi = 1.0
        changed_players = []
        for i, r in enumerate(part):
            by_role[r].append(i)
            if r != ROLE_REAL[i]:
                n_cr += 1
                changed_players.append(i)
                v_pi = float(pis_m[i, r])
                if v_pi < min_pi:
                    min_pi = v_pi

        # Base de 10 asientos para esta partición de roles
        base_cfg = np.empty(10, dtype=np.int32)
        if n_cr == 0:
            base_cfg[:] = np.arange(10, dtype=np.int32)
        else:
            for r in range(5):
                u, v = by_role[r]
                # Para consistencia estética, si uno ya estaba en azul en ese rol, dejarlo primero
                if u < 5 and ROLE_REAL[u] == r:
                    base_cfg[r] = u
                    base_cfg[5 + r] = v
                elif v < 5 and ROLE_REAL[v] == r:
                    base_cfg[r] = v
                    base_cfg[5 + r] = u
                else:
                    base_cfg[r] = u
                    base_cfg[5 + r] = v

        # Expandir las 32 orientaciones de equipo (azul/rojo por línea)
        cfgs_p = base_cfg[SWAPS_32]  # (32, 10)
        all_cfgs.append(cfgs_p)
        all_n_cr.append(np.full(32, n_cr, dtype=np.int32))
        all_min_pi.append(np.full(32, min_pi, dtype=np.float64))

        solo_b_p = np.zeros(32, dtype=bool)
        if n_cr == 0:
            solo_b_p[0] = True
        elif n_cr == 2:
            # La única configuración de (b) puro para el par (i, j) que intercambia roles
            # es intercambiar exactamente los asientos i y j en [0..9]
            i0, j0 = changed_players
            target_b = np.arange(10, dtype=np.int32)
            target_b[i0], target_b[j0] = j0, i0
            matches = np.all(cfgs_p == target_b[None, :], axis=1)
            solo_b_p |= matches
        all_solo_b.append(solo_b_p)

    return (
        np.vstack(all_cfgs),
        np.concatenate(all_n_cr),
        np.concatenate(all_min_pi),
        np.concatenate(all_solo_b),
    )


def cargar_perfil_roles_alineado(mids: np.ndarray, players_10: np.ndarray) -> dict[str, np.ndarray]:
    """Lee perfil_roles.parquet y alinea exactamente a (N, 10) con `players_10`."""
    cols_pr = [
        "match_id", "player_id", "n_prev", "rol_principal", "d", "autofill",
        "pi_BOTTOM", "pi_JUNGLE", "pi_MIDDLE", "pi_TOP", "pi_UTILITY",
    ]
    pr = pd.read_parquet(SALIDA / "perfil_roles.parquet", columns=cols_pr)
    pr = pr[pr["match_id"].isin(set(mids.tolist()))]
    keys = pd.DataFrame({
        "match_id": np.repeat(mids, 10),
        "player_id": players_10.reshape(-1),
    })
    m = keys.merge(pr, on=["match_id", "player_id"], how="left")
    assert m["rol_principal"].isna().sum() == 0, "Faltan jugadores en perfil_roles.parquet!"

    n = len(mids)
    pis = m[[f"pi_{r}" for r in ROLES]].to_numpy(np.float64).reshape(n, 10, 5)
    rol_p_str = m["rol_principal"].to_numpy(str).reshape(n, 10)
    rol_p_idx = np.vectorize(ROLE_TO_IDX.__getitem__, otypes=[np.int32])(rol_p_str)
    pi_max = pis[np.arange(n)[:, None], np.arange(10)[None, :], rol_p_idx]
    d_mat = np.maximum(0.0, pi_max[:, :, None] - pis)  # (n, 10, 5)
    d_real_parq = m["d"].to_numpy(np.float64).reshape(n, 10)
    err_d = float(np.max(np.abs(d_mat[:, np.arange(10), ROLE_REAL] - d_real_parq)))
    assert err_d < 1e-5, f"Desajuste d vs perfil_roles: {err_d}"
    return {
        "pis": pis,
        "rol_p_idx": rol_p_idx,
        "d_mat": d_mat,
        "d_real": d_mat[:, np.arange(10), ROLE_REAL],
        "n_prev": m["n_prev"].to_numpy(np.int32).reshape(n, 10),
        "autofill": m["autofill"].to_numpy(bool).reshape(n, 10),
    }


def evaluar_chequeo_con_draft(
    D: dict,
    extras_all: np.ndarray,
    tr: np.ndarray,
    tr_a: np.ndarray,
    tr_b: np.ndarray,
    v_idx: np.ndarray,
    T: dict,
    v_idx_in_eval: np.ndarray,
) -> dict:
    """Chequeo secundario: modelo escalón 9 CON DRAFT (1.504 columnas) evaluado
    exactamente sobre los 2^5 = 32 intercambios en la misma línea (cada jugador se lleva su campeón).
    Incluye validación contra calib_v_escalon9.parquet y sesgo de optimización A <-> B.
    """
    print("[Draft] Ajustando escalón 9 completo y mitades A/B para chequeo secundario...", flush=True)
    nm = D["NM"]
    n_v = len(v_idx)
    d_draft = D["DRAFT"].shape[1]
    d_xd = D["Xd"].shape[1]

    def _fit_e9(tr_sub: np.ndarray) -> dict:
        mu_xd = D["Xd"][tr_sub].mean(0).astype(np.float64)
        sd_xd = D["Xd"][tr_sub].std(0).astype(np.float64) + 1e-6
        mu_ex = extras_all[tr_sub].mean(0).astype(np.float64)
        sd_ex = extras_all[tr_sub].std(0).astype(np.float64) + 1e-6
        Mtr = sparse.hstack([
            D["DRAFT"][tr_sub],
            sparse.csr_matrix(((D["Xd"][tr_sub] - mu_xd.astype(np.float32)) / sd_xd.astype(np.float32)).astype(np.float32)),
            sparse.csr_matrix(((extras_all[tr_sub] - mu_ex.astype(np.float32)) / sd_ex.astype(np.float32)).astype(np.float32)),
        ]).tocsr()
        lr = LogisticRegression(C=0.05, max_iter=4000).fit(Mtr, D["y"][tr_sub])
        c = lr.coef_[0].astype(np.float64)
        w_draft = c[:d_draft]
        w_xd = c[d_draft:d_draft + d_xd] / sd_xd
        w_ex = c[d_draft + d_xd:] / sd_ex
        B0 = float(lr.intercept_[0]) - float(np.dot(w_xd, mu_xd) + np.dot(w_ex, mu_ex))
        return {"lr": lr, "B0": B0, "w_draft": w_draft, "w_xd": w_xd, "w_ex": w_ex,
                "mu_xd": mu_xd, "sd_xd": sd_xd, "mu_ex": mu_ex, "sd_ex": sd_ex}

    m_full = _fit_e9(tr)
    m_a = _fit_e9(tr_a)
    m_b = _fit_e9(tr_b)

    # Precomputar ingredientes por partida de V para evaluación exacta sobre las 32 orientaciones
    # 1. Contribución por línea del draft: (n_v, 5)
    champ_v = D["Z"]["champ"][v_idx].reshape(n_v, 2, 5)
    _, champion_codes = np.unique(D["Z"]["champ"].ravel(), return_inverse=True)
    pair_codes = champion_codes.reshape(nm, 2, 5) * 5 + np.arange(5)[None, None, :]
    vocab = np.unique(pair_codes)
    assert len(vocab) == d_draft
    p_v = pair_codes[v_idx]  # (n_v, 2, 5)
    col_b = np.searchsorted(vocab, p_v[:, 0, :])
    col_r = np.searchsorted(vocab, p_v[:, 1, :])

    # Slices exactos en extras_all (228 cols)
    # MC(0:75), ML(75:150), MMR(150:153), F1(153:157), MATCHUPS(157:167),
    # ROL_WR(167:177), KALMAN_OUT(177:189), KALMAN_MRG(189:196),
    # TEMPO_OBJ(196:206), ECON_MID(206:210), HGB(210:216), MATCHUP_EXP(216:228)
    xd_v = D["Xd"][v_idx].reshape(n_v, 5, 85).astype(np.float64)
    mc_v = extras_all[v_idx, 0:75].reshape(n_v, 5, 15).astype(np.float64)
    ml_v = extras_all[v_idx, 75:150].reshape(n_v, 5, 15).astype(np.float64)
    mat_cnt_v = extras_all[v_idx, 157:162].astype(np.float64)  # 5 counters
    hgb_v = extras_all[v_idx, 210:215].astype(np.float64)      # 5 líneas HGB

    # Tensores de jugador en V (índice v_idx_in_eval dentro de eval_idx)
    # En escalón 9 original, MMR_HIST viene de mmr_historico.npz (no de nivel_sin_ventana)
    mmr_v = np.load(CACHES / "mmr_historico.npz")["A"][v_idx, :, :, :3].reshape(n_v, 10, 3).astype(np.float64)
    f1_v = T["f1_eval"][v_idx_in_eval]         # (n_v, 10, 4)
    rwr_v_lin = T["rol_wr_lin_5"][v_idx_in_eval][:, np.arange(10), ROLE_REAL, :]  # (n_v, 10, 6)
    wrl_v = T["wr_lane_5"][v_idx_in_eval][:, np.arange(10), ROLE_REAL]            # (n_v, 10)
    wrd_v = T["wr_lane_diff_5"][v_idx_in_eval][:, np.arange(10), ROLE_REAL]       # (n_v, 10)
    de = T["draft_eval"]
    wcle_v = de["wr_champ_lane_exc_v"]         # (n_v, 10)
    syn_o = de["syn_orig"]                     # (n_v, 3)
    syn_c = de["syn_cross"]                    # (n_v, 3)
    exp_m = de["exp_m_v"]                      # (n_v, 4, 5)
    cbias_v = de["cbias_v"]                    # (n_v, 10)
    pbias_v = de["pbias_v"]                    # (n_v, 10)
    g11 = {k: v[v_idx_in_eval] for k, v in T["g11_eval"].items()}

    # Verificar que syn_orig coincide con MATCHUPS[:, 6:9]
    err_syn = float(np.max(np.abs(syn_o - extras_all[v_idx, 163:166])))
    assert err_syn < 1e-5, f"Desajuste sinergias: {err_syn}"

    def _eval_32_e9(mod: dict) -> np.ndarray:
        """Devuelve p_32 de shape (n_v, 32) exacta para el modelo escalón 9."""
        B0 = mod["B0"]
        wd, wxd, wex = mod["w_draft"], mod["w_xd"], mod["w_ex"]
        # Contribución por línea s_l in {+1, -1} -> lin_c shape (n_v, 5)
        lin_c = wd[col_b] - wd[col_r]
        lin_c += np.einsum("nlf,lf->nl", xd_v, wxd.reshape(5, 85))
        lin_c += np.einsum("nlf,lf->nl", mc_v, wex[0:75].reshape(5, 15))
        lin_c += np.einsum("nlf,lf->nl", ml_v, wex[75:150].reshape(5, 15))
        # MMR_HIST (150:153) y FOTO1_DIN (153:157)
        lin_c += 0.2 * np.einsum("nlf,f->nl", mmr_v[:, :5, :] - mmr_v[:, 5:, :], wex[150:153])
        lin_c += 0.2 * np.einsum("nlf,f->nl", f1_v[:, :5, :] - f1_v[:, 5:, :], wex[153:157])
        # MATCHUPS counters (157:162) + sum (162)
        lin_c += mat_cnt_v * (wex[157:162] + wex[162])[None, :]
        # ROL_WR medias (cols 0,1,2,3,4,5,6 -> ex 167:174)
        w_rwr_m = np.array([wex[167], wex[168], wex[170], wex[171], wex[172], wex[173]])
        lin_c += 0.2 * np.einsum("nlf,f->nl", rwr_v_lin[:, :5, :] - rwr_v_lin[:, 5:, :], w_rwr_m)
        lin_c += 0.2 * wex[169] * (wcle_v[:, :5] - wcle_v[:, 5:])
        # KALMAN_OUT lineales (ex 177, 182, 183, 184:189)
        re = g11["r_elo"] * 0.01
        rf = (g11["r_elo"] - g11["r_slow"]) * 0.01
        ro = g11["r_opp"] * 0.01
        lin_c += 0.2 * (wex[177] * (re[:, :5] - re[:, 5:]) + wex[182] * (rf[:, :5] - rf[:, 5:]) + wex[183] * (ro[:, :5] - ro[:, 5:]))
        lin_c += (re[:, :5] - re[:, 5:]) * wex[184:189][None, :]
        # KALMAN_MRG lineales (ex 189, 192, 193, 194)
        rm = g11["r_mrg"] * 0.01
        rmf = (g11["r_mrg"] - g11["r_mrg_slow"]) * 0.01
        rexc = (g11["r_elo"] - g11["r_opp"]) * 0.01
        lin_c += 0.2 * (
            wex[189] * (rm[:, :5] - rm[:, 5:])
            + wex[192] * (rmf[:, :5] - rmf[:, 5:])
            + wex[193] * (g11["inv_sqrt_cnt"][:, :5] - g11["inv_sqrt_cnt"][:, 5:])
            + wex[194] * (rexc[:, :5] - rexc[:, 5:])
        )
        # TEMPO_OBJ lineales (ex 196..203 y 205; ex 204 es max(cbias))
        lin_c += wex[196] * (g11["fbv"][:, :5] - g11["fbv"][:, 5:])
        lin_c[:, 1] += wex[197] * (g11["fbv"][:, 1] - g11["fbv"][:, 6])
        lin_c[:, 4] += wex[197] * (g11["fbv"][:, 4] - g11["fbv"][:, 9])
        lin_c += wex[198] * (g11["ftv"][:, :5] - g11["ftv"][:, 5:])
        lin_c[:, 0] += wex[199] * (g11["ftv"][:, 0] - g11["ftv"][:, 5])
        lin_c[:, 3] += wex[199] * (g11["ftv"][:, 3] - g11["ftv"][:, 8])
        lin_c += wex[200] * (g11["swv"][:, :5] - g11["swv"][:, 5:])
        lin_c += 0.2 * wex[201] * (g11["durv"][:, :5] - g11["durv"][:, 5:])
        lin_c += 0.2 * wex[202] * (pbias_v[:, :5] - pbias_v[:, 5:])
        lin_c += 0.2 * wex[203] * (cbias_v[:, :5] - cbias_v[:, 5:])
        lin_c += 0.2 * wex[205] * ((pbias_v[:, :5] * cbias_v[:, :5]) - (pbias_v[:, 5:] * cbias_v[:, 5:]))
        # ECON_MID lineales (ex 206, 208, 209)
        lin_c += 0.2 * wex[206] * (g11["srv"][:, :5] - g11["srv"][:, 5:])
        lin_c += 0.2 * wex[208] * (g11["dgev"][:, :5] - g11["dgev"][:, 5:])
        lin_c[:, 0] += wex[209] * (g11["dgev"][:, 0] - g11["dgev"][:, 5])
        lin_c[:, 2] += wex[209] * (g11["dgev"][:, 2] - g11["dgev"][:, 7])
        # HGB (ex 210:215 + suma 215)
        lin_c += hgb_v * (wex[210:215] + wex[215])[None, :]
        # MATCHUP_EXP_MARGIN (ex 216:228)
        for s_i in range(4):
            lin_c += 0.2 * wex[216 + s_i] * exp_m[:, s_i, :]
        lin_c += exp_m[:, 0, :] * wex[220:225][None, :]
        lin_c[:, 2] += wex[225] * exp_m[:, 1, 2]
        lin_c[:, 1] += wex[226] * exp_m[:, 1, 1]
        lin_c[:, 3] += wex[227] * exp_m[:, 1, 3]

        # Contribución lineal sobre las 32 orientaciones: (n_v, 32)
        Z32 = B0 + lin_c @ SIGNS_32.T

        # Sinergias MATCHUPS (ex 163:166 + sum 166)
        SYN_PAIRS = ((0, 4), (3, 1), (2, 1))
        for idx_p, (l1, l2) in enumerate(SYN_PAIRS):
            w_s = wex[163 + idx_p] + wex[166]
            s1 = SIGNS_32[:, l1]  # (32,)
            same = s1 == SIGNS_32[:, l2]
            Z32 += w_s * np.where(same[None, :], syn_o[:, idx_p:idx_p + 1] * s1[None, :],
                                  syn_c[:, idx_p:idx_p + 1] * s1[None, :])

        # Términos NO lineales sobre las 32 orientaciones (vectorizado por orientación k=0..31)
        for k in range(32):
            b_idx = SWAPS_32[k, :5]
            r_idx = SWAPS_32[k, 5:]
            # ROL_WR mínimos (ex 174, 175, 176)
            Z32[:, k] += wex[174] * (wrl_v[:, b_idx].min(1) - wrl_v[:, r_idx].min(1))
            Z32[:, k] += wex[175] * (wrd_v[:, b_idx].min(1) - wrd_v[:, r_idx].min(1))
            Z32[:, k] += wex[176] * (wcle_v[:, b_idx].min(1) - wcle_v[:, r_idx].min(1))
            # KALMAN_OUT min(178), max(179), std(180), top2(181)
            rb = g11["r_elo"][:, b_idx]
            rr = g11["r_elo"][:, r_idx]
            rbs = np.sort(rb, axis=1)
            rrs = np.sort(rr, axis=1)
            Z32[:, k] += wex[178] * ((rbs[:, 0] - rrs[:, 0]) * 0.01)
            Z32[:, k] += wex[179] * ((rbs[:, 4] - rrs[:, 4]) * 0.01)
            Z32[:, k] += wex[180] * ((rb.std(axis=1) - rr.std(axis=1)) * 0.01)
            Z32[:, k] += wex[181] * (((rbs[:, 3] + rbs[:, 4]) - (rrs[:, 3] + rrs[:, 4])) * 0.005)
            # KALMAN_MRG min(190), max(191), brecha(195)
            rmb = g11["r_mrg"][:, b_idx]
            rmr = g11["r_mrg"][:, r_idx]
            Z32[:, k] += wex[190] * ((rmb.min(1) - rmr.min(1)) * 0.01)
            Z32[:, k] += wex[191] * ((rmb.max(1) - rmr.max(1)) * 0.01)
            Z32[:, k] += wex[195] * (np.abs(rb.mean(1) - rr.mean(1)) * 0.01)
            # TEMPO_OBJ max cbias (204) y ECON_MID max srv (207)
            Z32[:, k] += wex[204] * (cbias_v[:, b_idx].max(1) - cbias_v[:, r_idx].max(1))
            Z32[:, k] += wex[207] * (g11["srv"][:, b_idx].max(1) - g11["srv"][:, r_idx].max(1))

        return expit(Z32)

    p32_full = _eval_32_e9(m_full)
    p32_a = _eval_32_e9(m_a)
    p32_b = _eval_32_e9(m_b)

    # Validar contra calib_v_escalon9.parquet (o predict_proba directo)
    Mte = sparse.hstack([
        D["DRAFT"][v_idx],
        sparse.csr_matrix(((D["Xd"][v_idx] - m_full["mu_xd"].astype(np.float32)) / m_full["sd_xd"].astype(np.float32)).astype(np.float32)),
        sparse.csr_matrix(((extras_all[v_idx] - m_full["mu_ex"].astype(np.float32)) / m_full["sd_ex"].astype(np.float32)).astype(np.float32)),
    ]).tocsr()
    p_ref_direct = m_full["lr"].predict_proba(Mte)[:, 1]
    err_direct = float(np.max(np.abs(p32_full[:, 0] - p_ref_direct)))

    err_sol = np.nan
    ruta_sol_e9 = SALIDA / "calib_v_escalon9.parquet"
    if ruta_sol_e9.is_file():
        df_e9 = pd.read_parquet(ruta_sol_e9).set_index("match_id").reindex(D["Z"]["mids"][v_idx])
        err_sol = float(np.max(np.abs(p32_full[:, 0] - df_e9["p_azul"].to_numpy())))

    best_full = np.argmin(np.abs(p32_full - 0.5), axis=1)
    best_a = np.argmin(np.abs(p32_a - 0.5), axis=1)
    best_b = np.argmin(np.abs(p32_b - 0.5), axis=1)
    rows_ar = np.arange(n_v)

    y_v = D["y"][v_idx]
    return {
        "p_real": p32_full[:, 0],
        "p_best": p32_full[rows_ar, best_full],
        "p_a_real": p32_a[:, 0],
        "p_b_real": p32_b[:, 0],
        "p_a_eleg": p32_a[rows_ar, best_a],
        "p_b_indep": p32_b[rows_ar, best_a],
        "p_b_eleg": p32_b[rows_ar, best_b],
        "p_a_indep": p32_a[rows_ar, best_b],
        "err_direct": err_direct,
        "err_sol": err_sol,
        "auc_full": float(roc_auc_score(y_v, p32_full[:, 0])),
        "acc_full": float(((p32_full[:, 0] > 0.5) == y_v).mean()),
        "auc_a": float(roc_auc_score(y_v, p32_a[:, 0])),
        "acc_a": float(((p32_a[:, 0] > 0.5) == y_v).mean()),
        "auc_b": float(roc_auc_score(y_v, p32_b[:, 0])),
        "acc_b": float(((p32_b[:, 0] > 0.5) == y_v).mean()),
    }


def correr_experimento_conjunto(
    nombre_conj: str,
    mids: np.ndarray,
    y_conj: np.ndarray,
    players_10: np.ndarray,
    roles_info: dict,
    T_sub: dict,
    B0_full: float, C_lin_full: np.ndarray, w_nl_full: np.ndarray,
    B0_a: float, C_lin_a: np.ndarray, w_nl_a: np.ndarray,
    B0_b: float, C_lin_b: np.ndarray, w_nl_b: np.ndarray,
    guardar_jugadores: bool = True,
) -> dict:
    """Corre la búsqueda exhaustiva para cada partida del conjunto (V Master+ o muestra train Master+)."""
    n = len(mids)
    t0 = time.perf_counter()
    pis = roles_info["pis"]
    d_mat = roles_info["d_mat"]
    d_real_vec = roles_info["d_real"]
    rol_p_idx = roles_info["rol_p_idx"]
    n_prev = roles_info["n_prev"]
    autofill_real = roles_info["autofill"]
    fuera_mat = (np.arange(5)[None, None, :] != rol_p_idx[:, :, None])  # (n, 10, 5)

    wrl = T_sub["wr_lane_5"]
    wrd = T_sub["wr_lane_diff_5"]
    r_elo = T_sub["g11_eval"]["r_elo"]
    r_mrg = T_sub["g11_eval"]["r_mrg"]
    srv = T_sub["g11_eval"]["srv"]

    # Contenedores por partida
    res = {
        "match_id": mids,
        "y": y_conj.astype(np.int8),
        "p_real": np.zeros(n, dtype=np.float64),
        "p_a_real": np.zeros(n, dtype=np.float64),
        "p_b_real": np.zeros(n, dtype=np.float64),
        "D_real": d_real_vec.sum(axis=1),
        "D_azul_real": d_real_vec[:, :5].sum(axis=1),
        "D_rojo_real": d_real_vec[:, 5:].sum(axis=1),
        "n_fuera_real": fuera_mat[:, np.arange(10), ROLE_REAL].sum(axis=1).astype(np.int32),
        "n_autofill_real": autofill_real.sum(axis=1).astype(np.int32),
        "n_cands_t20": np.zeros(n, dtype=np.int32),
        "n_cands_t35": np.zeros(n, dtype=np.int32),
    }
    for esc in ESCENARIOS_PRINCIPALES:
        res[f"p_{esc}"] = np.zeros(n, dtype=np.float64)
        res[f"D_{esc}"] = np.zeros(n, dtype=np.float64)
        res[f"n_fuera_{esc}"] = np.zeros(n, dtype=np.int32)
        res[f"n_cambios_rol_{esc}"] = np.zeros(n, dtype=np.int32)
        res[f"n_cambios_eq_{esc}"] = np.zeros(n, dtype=np.int32)
        res[f"p_a_eleg_{esc}"] = np.zeros(n, dtype=np.float64)
        res[f"p_b_indep_{esc}"] = np.zeros(n, dtype=np.float64)
        res[f"p_b_eleg_{esc}"] = np.zeros(n, dtype=np.float64)
        res[f"p_a_indep_{esc}"] = np.zeros(n, dtype=np.float64)

    # Frontera parejo vs costo de rol (τ=0,20 y τ=0,35)
    frontera = {}
    for tau_tag in ("t20", "t35"):
        for dmax in FRONTERA_DELTAS:
            k_f = f"{tau_tag}_dmax_{'inf' if math.isinf(dmax) else f'{dmax:.2f}'}"
            frontera[k_f] = {
                "p": np.zeros(n, dtype=np.float64),
                "p_a_eleg": np.zeros(n, dtype=np.float64),
                "p_b_indep": np.zeros(n, dtype=np.float64),
                "p_b_eleg": np.zeros(n, dtype=np.float64),
                "p_a_indep": np.zeros(n, dtype=np.float64),
                "D": np.zeros(n, dtype=np.float64),
                "n_fuera": np.zeros(n, dtype=np.int32),
                "d_jug_gt03": np.zeros(n, dtype=np.int32),
                "n_mejora": np.zeros(n, dtype=np.int32),
                "n_empeora": np.zeros(n, dtype=np.int32),
            }

    # Contenedor compacto para swaps_jugadores.parquet (si guardar_jugadores)
    # Guardamos para cada escenario en ESCENARIOS_JUGADORES la mejor configuración `best_cfg[esc][m, 10]`
    best_cfgs = {esc: np.zeros((n, 10), dtype=np.int8) for esc in ESCENARIOS_JUGADORES} if guardar_jugadores else {}

    L_row = np.arange(5)[None, :]
    for m in range(n):
        cfgs, n_cr, min_pi, es_solo_b = generar_configs_partida(pis[m], tau_min=0.20)
        K = len(cfgs)
        B = cfgs[:, :5]
        R = cfgs[:, 5:]

        # Evaluar los 3 modelos (full, mitad A, mitad B) sobre todas las K configuraciones
        p_full = evaluar_configs_partida(cfgs, B0_full, C_lin_full[m], w_nl_full, wrl[m], wrd[m], r_elo[m], r_mrg[m], srv[m])
        p_a = evaluar_configs_partida(cfgs, B0_a, C_lin_a[m], w_nl_a, wrl[m], wrd[m], r_elo[m], r_mrg[m], srv[m])
        p_b = evaluar_configs_partida(cfgs, B0_b, C_lin_b[m], w_nl_b, wrl[m], wrd[m], r_elo[m], r_mrg[m], srv[m])

        res["p_real"][m] = p_full[0]
        res["p_a_real"][m] = p_a[0]
        res["p_b_real"][m] = p_b[0]

        # Costo de rol y desvíos por jugador para cada una de las K configuraciones
        d_B = d_mat[m][B, L_row]
        d_R = d_mat[m][R, L_row]
        D_cfg = d_B.sum(axis=1) + d_R.sum(axis=1)
        D_real_m = D_cfg[0]
        nf_cfg = fuera_mat[m][B, L_row].sum(axis=1) + fuera_mat[m][R, L_row].sum(axis=1)
        d_gt03_cfg = (d_B > 0.30).sum(axis=1) + (d_R > 0.30).sum(axis=1)
        dd_B = d_B - d_real_vec[m][B]
        dd_R = d_R - d_real_vec[m][R]
        max_dd_cfg = np.maximum(dd_B.max(axis=1), dd_R.max(axis=1))
        n_mej_cfg = (dd_B < -TOL_D).sum(axis=1) + (dd_R < -TOL_D).sum(axis=1)
        n_emp_cfg = (dd_B > TOL_D).sum(axis=1) + (dd_R > TOL_D).sum(axis=1)

        # Cuántos jugadores cambian de equipo: en B (azul=0) los de TEAM_REAL==1; en R (rojo=1) los de TEAM_REAL==0
        n_ceq_cfg = (TEAM_REAL[B] == 1).sum(axis=1) + (TEAM_REAL[R] == 0).sum(axis=1)

        # Máscaras booleanas por escenario
        m_a = n_cr == 0
        m_t20 = min_pi >= 0.20 - 1e-12
        m_t35 = min_pi >= 0.35 - 1e-12
        m_d_le = D_cfg <= D_real_m + TOL_D

        res["n_cands_t20"][m] = int(m_t20.sum())
        res["n_cands_t35"][m] = int(m_t35.sum())

        masks = {
            "a_misma_linea": m_a,
            "b_rol_t20_d_le": es_solo_b & m_t20 & m_d_le,
            "b_rol_t20_sin_tope": es_solo_b & m_t20,
            "b_rol_t35_d_le": es_solo_b & m_t35 & m_d_le,
            "b_rol_t35_sin_tope": es_solo_b & m_t35,
            "ab_1swap_t20_d_le": (n_cr <= 2) & m_t20 & m_d_le,
            "ab_1swap_t20_sin_tope": (n_cr <= 2) & m_t20,
            "ab_1swap_t35_d_le": (n_cr <= 2) & m_t35 & m_d_le,
            "ab_1swap_t35_sin_tope": (n_cr <= 2) & m_t35,
            "ab_exh_t20_d_le": m_t20 & m_d_le,
            "ab_exh_t20_sin_tope": m_t20,
            "ab_exh_t35_d_le": m_t35 & m_d_le,
            "ab_exh_t35_sin_tope": m_t35,
            "ab_pareto_t20": m_t20 & (max_dd_cfg <= TOL_D),
            "ab_minD_t20": m_t20 & (D_cfg <= D_cfg[m_t20].min() + TOL_D),
        }

        dist_full = np.abs(p_full - 0.5)
        dist_a = np.abs(p_a - 0.5)
        dist_b = np.abs(p_b - 0.5)

        for esc, msk in masks.items():
            idxs = np.flatnonzero(msk)
            # Desempate determinista por menor D si empatan en |p - 0.5|
            # (usamos lexsort: clave primaria dist, secundaria D)
            i_full = idxs[int(np.lexsort((D_cfg[idxs], dist_full[idxs]))[0])]
            i_a = idxs[int(np.lexsort((D_cfg[idxs], dist_a[idxs]))[0])]
            i_b = idxs[int(np.lexsort((D_cfg[idxs], dist_b[idxs]))[0])]

            res[f"p_{esc}"][m] = p_full[i_full]
            res[f"D_{esc}"][m] = D_cfg[i_full]
            res[f"n_fuera_{esc}"][m] = nf_cfg[i_full]
            res[f"n_cambios_rol_{esc}"][m] = n_cr[i_full]
            res[f"n_cambios_eq_{esc}"][m] = n_ceq_cfg[i_full]
            res[f"p_a_eleg_{esc}"][m] = p_a[i_a]
            res[f"p_b_indep_{esc}"][m] = p_b[i_a]
            res[f"p_b_eleg_{esc}"][m] = p_b[i_b]
            res[f"p_a_indep_{esc}"][m] = p_a[i_b]
            if guardar_jugadores and esc in best_cfgs:
                best_cfgs[esc][m] = cfgs[i_full].astype(np.int8)

        # Frontera parejo vs costo de rol (con evaluación independiente A ↔ B)
        for tau_tag, m_tau in (("t20", m_t20), ("t35", m_t35)):
            for dmax in FRONTERA_DELTAS:
                k_f = f"{tau_tag}_dmax_{'inf' if math.isinf(dmax) else f'{dmax:.2f}'}"
                msk_f = m_tau if math.isinf(dmax) else (m_tau & (D_cfg <= D_real_m + dmax + TOL_D))
                idxs_f = np.flatnonzero(msk_f)
                i_f = idxs_f[int(np.lexsort((D_cfg[idxs_f], dist_full[idxs_f]))[0])]
                i_fa = idxs_f[int(np.lexsort((D_cfg[idxs_f], dist_a[idxs_f]))[0])]
                i_fb = idxs_f[int(np.lexsort((D_cfg[idxs_f], dist_b[idxs_f]))[0])]
                frontera[k_f]["p"][m] = p_full[i_f]
                frontera[k_f]["p_a_eleg"][m] = p_a[i_fa]
                frontera[k_f]["p_b_indep"][m] = p_b[i_fa]
                frontera[k_f]["p_b_eleg"][m] = p_b[i_fb]
                frontera[k_f]["p_a_indep"][m] = p_a[i_fb]
                frontera[k_f]["D"][m] = D_cfg[i_f]
                frontera[k_f]["n_fuera"][m] = nf_cfg[i_f]
                frontera[k_f]["d_jug_gt03"][m] = d_gt03_cfg[i_f]
                frontera[k_f]["n_mejora"][m] = n_mej_cfg[i_f]
                frontera[k_f]["n_empeora"][m] = n_emp_cfg[i_f]

        if (m + 1) % 4000 == 0 or (m + 1) == n:
            print(f"    [{nombre_conj}] {m + 1:,}/{n:,} partidas ({time.perf_counter() - t0:.1f} s)", flush=True)

    # Armar tabla de jugadores si se solicitó
    df_jug = None
    stats_jug = {}
    if guardar_jugadores:
        # Estadísticas de Riot (real) por jugador
        d_r_flat = d_real_vec.reshape(-1)
        nf_r_flat = fuera_mat[:, np.arange(10), ROLE_REAL].reshape(-1)
        stats_jug["real"] = {
            "d_mean": float(d_r_flat.mean()),
            "d_p50": float(np.percentile(d_r_flat, 50)),
            "d_p90": float(np.percentile(d_r_flat, 90)),
            "d_p99": float(np.percentile(d_r_flat, 99)),
            "pct_d_gt03": float((d_r_flat > 0.30).mean()),
            "pct_fuera": float(nf_r_flat.mean()),
            "dd_mean": 0.0, "dd_p10": 0.0, "dd_p50": 0.0, "dd_p90": 0.0, "dd_p99": 0.0,
            "pct_mejora_d": 0.0, "pct_empeora_d": 0.0, "pct_cambio_rol": 0.0, "pct_cambio_eq": 0.0,
        }

        partes_jug = []
        mid_rep = np.repeat(mids, 10)
        pid_flat = players_10.reshape(-1)
        rol_p_flat = np.array(ROLES)[rol_p_idx.reshape(-1)]
        n_prev_flat = n_prev.reshape(-1)
        eq_real_flat = np.tile(np.where(TEAM_REAL == 0, "BLUE", "RED"), n)
        lin_real_flat = np.tile(np.array(ROLES)[ROLE_REAL], n)
        af_real_flat = autofill_real.reshape(-1)
        roles_arr = np.array(ROLES)

        for esc in ESCENARIOS_JUGADORES:
            bc = best_cfgs[esc]  # (n, 10): seat s -> player i
            # Invertir permutación para obtener por cada jugador i su asiento nuevo (equipo_nuevo, linea_nueva)
            seat_of_player = np.argsort(bc, axis=1)  # (n, 10): player i -> seat s in 0..9
            eq_new_idx = (seat_of_player >= 5).astype(np.int32)
            lin_new_idx = (seat_of_player % 5).astype(np.int32)

            d_new = d_mat[np.arange(n)[:, None], np.arange(10)[None, :], lin_new_idx].reshape(-1)
            dd = d_new - d_r_flat
            fuera_new = (lin_new_idx != rol_p_idx).reshape(-1)
            cambio_eq = (eq_new_idx != TEAM_REAL[None, :]).reshape(-1)
            cambio_rol = (lin_new_idx != ROLE_REAL[None, :]).reshape(-1)

            stats_jug[esc] = {
                "d_mean": float(d_new.mean()),
                "d_p50": float(np.percentile(d_new, 50)),
                "d_p90": float(np.percentile(d_new, 90)),
                "d_p99": float(np.percentile(d_new, 99)),
                "pct_d_gt03": float((d_new > 0.30).mean()),
                "pct_fuera": float(fuera_new.mean()),
                "dd_mean": float(dd.mean()),
                "dd_p10": float(np.percentile(dd, 10)),
                "dd_p50": float(np.percentile(dd, 50)),
                "dd_p90": float(np.percentile(dd, 90)),
                "dd_p99": float(np.percentile(dd, 99)),
                "pct_mejora_d": float((dd < -1e-7).mean()),
                "pct_empeora_d": float((dd > 1e-7).mean()),
                "pct_cambio_rol": float(cambio_rol.mean()),
                "pct_cambio_eq": float(cambio_eq.mean()),
            }

            partes_jug.append(pd.DataFrame({
                "match_id": mid_rep,
                "escenario": esc,
                "player_id": pid_flat,
                "rol_principal": rol_p_flat,
                "n_prev": n_prev_flat,
                "equipo_real": eq_real_flat,
                "linea_real": lin_real_flat,
                "d_real": d_r_flat.astype(np.float32),
                "autofill_real": af_real_flat,
                "equipo_nuevo": np.where(eq_new_idx.reshape(-1) == 0, "BLUE", "RED"),
                "linea_nueva": roles_arr[lin_new_idx.reshape(-1)],
                "d_nuevo": d_new.astype(np.float32),
                "delta_d": dd.astype(np.float32),
                "cambio_equipo": cambio_eq,
                "cambio_rol": cambio_rol,
            }))

        df_jug = pd.concat(partes_jug, ignore_index=True)

    return {
        "df_res": pd.DataFrame(res),
        "df_jug": df_jug,
        "stats_jug": stats_jug,
        "frontera": frontera,
        "best_cfgs": best_cfgs,
    }


def resumen_escenario(p_real: np.ndarray, p_esc: np.ndarray, D_esc: np.ndarray, D_real: np.ndarray, nf_esc: np.ndarray) -> dict:
    dist_r = np.abs(p_real - 0.5)
    dist_e = np.abs(p_esc - 0.5)
    mej = dist_r - dist_e
    return {
        "dist_mean": float(dist_e.mean()),
        "dist_p50": float(np.percentile(dist_e, 50)),
        "dist_p75": float(np.percentile(dist_e, 75)),
        "dist_p90": float(np.percentile(dist_e, 90)),
        "dist_p95": float(np.percentile(dist_e, 95)),
        "mej_p50": float(np.percentile(mej, 50)),
        "mej_mean": float(mej.mean()),
        "pct_45_55": float((dist_e <= 0.05).mean()),
        "pct_48_52": float((dist_e <= 0.02).mean()),
        "D_mean": float(D_esc.mean()),
        "D_p50": float(np.percentile(D_esc, 50)),
        "D_p90": float(np.percentile(D_esc, 90)),
        "dD_mean": float((D_esc - D_real).mean()),
        "nf_mean": float(nf_esc.mean()),
    }


def seleccionar_tres_ejemplos(
    df_res: pd.DataFrame,
    best_cfgs: dict[str, np.ndarray],
    players_10: np.ndarray,
    roles_info: dict,
) -> list[dict]:
    """Elige 3 partidas representativas de V Master+ para ilustrar los mecanismos:
    1) Caso A: partida muy despareja (|p_real - 0.5| >= 0.14) resuelta casi a 50-50 solo con (a) misma línea (ΔD = 0).
    2) Caso B: partida con autofill/desvío alto donde (a+b) τ=0,20 con D≤D_real mejora |p-0,5| y además REDUCE D.
    3) Caso C: partida donde (a+b) τ=0,20 sin tope logra acercar mucho más a 50-50 que (a), pero aumentando D (trade-off real).
    """
    p_r = df_res["p_real"].to_numpy()
    d_r = np.abs(p_r - 0.5)
    D_r = df_res["D_real"].to_numpy()
    p_a = df_res["p_a_misma_linea"].to_numpy()
    p_ab_le = df_res["p_ab_exh_t20_d_le"].to_numpy()
    D_ab_le = df_res["D_ab_exh_t20_d_le"].to_numpy()
    p_ab_st = df_res["p_ab_exh_t20_sin_tope"].to_numpy()
    D_ab_st = df_res["D_ab_exh_t20_sin_tope"].to_numpy()
    nc_le = df_res["n_cambios_rol_ab_exh_t20_d_le"].to_numpy()
    nc_st = df_res["n_cambios_rol_ab_exh_t20_sin_tope"].to_numpy()

    # Ejemplo 1: muy despareja, (a) la deja en [0.49, 0.51] cambiando 1 o 2 líneas
    cand1 = np.flatnonzero((d_r >= 0.14) & (np.abs(p_a - 0.5) <= 0.01) & (df_res["n_cambios_eq_a_misma_linea"].to_numpy() <= 4))
    if len(cand1) == 0:
        cand1 = np.flatnonzero(d_r >= 0.10)
    idx1 = int(cand1[np.argmax(d_r[cand1] - np.abs(p_a[cand1] - 0.5))])

    # Ejemplo 2: (a+b) D<=D_real reduce D claramente (ΔD <= -0.25) y mejora mucho |p-0,5| más que (a)
    cand2 = np.flatnonzero(
        (d_r >= 0.10) & (nc_le >= 2) & ((D_ab_le - D_r) <= -0.25)
        & (np.abs(p_ab_le - 0.5) <= 0.015) & (np.abs(p_a - 0.5) >= 0.025)
    )
    if len(cand2) == 0:
        cand2 = np.flatnonzero((nc_le >= 2) & ((D_ab_le - D_r) < -0.05))
    if len(cand2) == 0:
        cand2 = np.flatnonzero(nc_le >= 2)
    idx2 = int(cand2[np.argmax((d_r[cand2] - np.abs(p_ab_le[cand2] - 0.5)) - 0.3 * (D_ab_le[cand2] - D_r[cand2]))]) if len(cand2) else (idx1 + 1) % len(df_res)

    # Ejemplo 3: (a+b) sin tope supera claramente a (a) (en >= 3 pp) a costa de aumentar D (ΔD >= 0.15)
    cand3 = np.flatnonzero(
        (d_r >= 0.09) & (nc_st >= 2) & ((D_ab_st - D_r) >= 0.15)
        & (np.abs(p_a - 0.5) - np.abs(p_ab_st - 0.5) >= 0.03)
        & (np.abs(p_ab_st - 0.5) <= 0.015)
    )
    if len(cand3) == 0:
        cand3 = np.flatnonzero((nc_st >= 2) & ((D_ab_st - D_r) > 0.05))
    cand3 = cand3[(cand3 != idx1) & (cand3 != idx2)]
    idx3 = int(cand3[np.argmax(np.abs(p_a[cand3] - 0.5) - np.abs(p_ab_st[cand3] - 0.5))]) if len(cand3) else (idx2 + 1) % len(df_res)

    ejemplos = []
    specs = [
        ("Ejemplo 1 — Equilibrio sin tocar roles `(a) Misma línea`", idx1, "a_misma_linea"),
        ("Ejemplo 2 — Mejor equilibrio y MENOR desvío de rol `(a+b) τ=0,20, D ≤ D_real`", idx2, "ab_exh_t20_d_le"),
        ("Ejemplo 3 — Trade-off explícito: más parejo a costa de subir el desvío `(a+b) τ=0,20, sin tope`", idx3, "ab_exh_t20_sin_tope"),
    ]
    for titulo, idx, esc in specs:
        bc = best_cfgs[esc][idx]
        seat_of_p = np.argsort(bc)
        filas = []
        for i in range(10):
            s_new = int(seat_of_p[i])
            lin_new = s_new % 5
            eq_new = "BLUE" if s_new < 5 else "RED"
            eq_old = "BLUE" if i < 5 else "RED"
            lin_old = i % 5
            d_old = float(roles_info["d_real"][idx, i])
            d_new = float(roles_info["d_mat"][idx, i, lin_new])
            rp = ROLES[int(roles_info["rol_p_idx"][idx, i])]
            pi_new = float(roles_info["pis"][idx, i, lin_new])
            filas.append({
                "player_id": str(players_10[idx, i]),
                "rol_principal": rp,
                "n_prev": int(roles_info["n_prev"][idx, i]),
                "eq_real": eq_old,
                "lin_real": ROLES[lin_old],
                "d_real": d_old,
                "eq_nuevo": eq_new,
                "lin_nuevo": ROLES[lin_new],
                "pi_nuevo": pi_new,
                "d_nuevo": d_new,
                "delta_d": d_new - d_old,
            })
        ejemplos.append({
            "titulo": titulo,
            "match_id": str(df_res["match_id"].iloc[idx]),
            "escenario": esc,
            "y": int(df_res["y"].iloc[idx]),
            "p_real": float(p_r[idx]),
            "p_a": float(p_a[idx]),
            "p_nuevo": float(df_res[f"p_{esc}"].iloc[idx]),
            "D_real": float(D_r[idx]),
            "D_nuevo": float(df_res[f"D_{esc}"].iloc[idx]),
            "filas": filas,
        })
    return ejemplos


def escribir_reporte(
    info_modelos: dict,
    validacion: dict,
    pesos_bloques: list[dict],
    out_v: dict,
    out_tr: dict | None,
    draft_res: dict | None,
    mapa_platt: dict | None,
    ejemplos: list[dict],
    duracion_s: float,
):
    df_v = out_v["df_res"]
    p_r = df_v["p_real"].to_numpy()
    D_r = df_v["D_real"].to_numpy()
    nf_r = df_v["n_fuera_real"].to_numpy()
    af_r = df_v["n_autofill_real"].to_numpy()
    n_v = len(df_v)

    # Correlaciones |p-0,5| vs D_real / autofill
    dist_r = np.abs(p_r - 0.5)
    r_p_D, p_p_D = pearsonr(dist_r, D_r)
    rho_D, ps_D = spearmanr(dist_r, D_r)
    r_p_af, _ = pearsonr(dist_r, af_r)
    rho_af, _ = spearmanr(dist_r, af_r)
    r_p_nf, _ = pearsonr(dist_r, nf_r)
    diff_D_eq = df_v["D_azul_real"].to_numpy() - df_v["D_rojo_real"].to_numpy()
    r_dir, _ = pearsonr(p_r - 0.5, diff_D_eq)

    lines = [
        "# Reemparejamiento contrafáctico (Swaps) en Master+ — LAS",
        "",
        f"- **Fecha de generación:** {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %z')}",
        f"- **Tiempo de ejecución:** {duracion_s:.1f} s",
        f"- **Población analizada:** estrictamente **Master+** (`particion_v_master` / máscara `D['master']`; nada de Diamond).",
        f"  - Conjunto principal: **V Master+** ({cuenta(n_v)} partidas, 16/09 → 27/09).",
    ]
    if out_tr is not None:
        lines.append(f"  - Muestra de verificación en entrenamiento: **{cuenta(len(out_tr['df_res']))}** partidas Master+ anteriores al 16/09 (aleatoria estratificada con semilla fija).")
    lines.extend([
        f"- **Modelo pre-partida principal:** logística sin draft ni columnas de campeón (`C = 0,05`), **320 columnas** exactas idénticas a `calib_columnas_sin_draft.json` de Sol, entrenada solo con {cuenta(info_modelos['n_tr'])} partidas Master+ anteriores al 16/09.",
        f"  - En V Master+: **AUC {num(info_modelos['auc_full'])} / acierto {pct(info_modelos['acc_full'])}** (log loss {num(info_modelos['ll_full'])}).",
        f"  - Mitades disjuntas del entrenamiento ({cuenta(info_modelos['n_tr_a'])} y {cuenta(info_modelos['n_tr_b'])} partidas, partición intercalada cronológica): "
        f"Mitad A **AUC {num(info_modelos['auc_a'])} / acierto {pct(info_modelos['acc_a'])}**; "
        f"Mitad B **AUC {num(info_modelos['auc_b'])} / acierto {pct(info_modelos['acc_b'])}**.",
    ])
    if draft_res is not None:
        lines.append(
            f"- **Chequeo secundario con draft (escalón 9, 1.504 columnas):** "
            f"en V Master+ **AUC {num(draft_res['auc_full'])} / acierto {pct(draft_res['acc_full'])}** "
            f"(mitades A: {num(draft_res['auc_a'])} / {pct(draft_res['acc_a'])}; B: {num(draft_res['auc_b'])} / {pct(draft_res['acc_b'])})."
        )

    lines.extend([
        "",
        "> **Alcance (§9.S.67 y §9.S.77).** Es una **simulación bajo el modelo**: las configuraciones nuevas no se "
        "jugaron y su probabilidad es la que asigna la logística, no un resultado observado. **No restringe dúos ni "
        "premades** (la base no los registra): un reacomodo puede separar a dos jugadores que entraron juntos a la cola. "
        "La posición es `individual_position`, que aproxima el rol asignado; no se conoce el rol pedido. "
        "La variante principal es la que **protege a cada jugador** (`ab_pareto_t20`, Δd_i ≤ 0 para los 10): el tope "
        "D ≤ D_real solo protege el total y deja que algunos jugadores empeoren su rol (sección 4, columna «% mejora / empeora d»).",
        "",
        "---",
        "",
        "## 1. Contrafáctico de variables y validación exacta",
        "",
        "### Coincidencia con Sol y exactitud del contrafáctico",
        "",
        f"- **Esquema de columnas:** coincide 100 % con `data/analysis/matchmaking/calib_columnas_sin_draft.json` (320 columnas en el mismo orden, escalas de `Xj`/`ML`, imputación de `log_horas_desde` y prior de `MMR_HIST` congelados antes del cuantil temporal 90 % de `~val0`).",
        f"- **Columnas aproximadas en la variante sin draft:** **0 de 320**. Todas las 320 columnas —incluidas las 93 columnas que dependen de `(jugador, línea)` (`Xj/share_lane`, `Xj/offrole`, `ML` × 75, `ROL_WR` × 8) y las 10 columnas no lineales por equipo (`min`, `max`, `std`, `top2`, `|mb−mr|`)— se recalculan de forma **100 % exacta y causal** para cualquier asignación de los 10 jugadores a `(equipo, línea)`.",
        f"- **Validación sobre la asignación real en V Master+ ({cuenta(n_v)} partidas):**",
        f"  - Contrafáctico vs. `lr.predict_proba` propio: **máx |Δp| = {validacion['err_own']:.3e}**.",
        f"  - Contrafáctico vs. `calib_v_sin_draft.parquet` de Sol: **máx |Δp| = {validacion['err_sol']:.3e}**.",
    ])
    if draft_res is not None:
        lines.append(
            f"  - Chequeo secundario con draft (1.504 cols) vs. `lr.predict_proba` propio: **máx |Δp| = {draft_res['err_direct']:.3e}**; "
            f"vs. `calib_v_escalon9.parquet` de Sol: **máx |Δp| = {draft_res['err_sol']:.3e}** (0 columnas aproximadas bajo swaps en la misma línea)."
        )

    lines.extend([
        "",
        "### Tratamiento contrafáctico y peso por bloque en el modelo sin draft",
        "",
        "Como todas las variables entran estandarizadas a la logística, `|coef|` equivale directamente a `|β|·sd` en escala de log-odds.",
        "",
        "| Bloque | Cols | Tipo contrafáctico | Recálculo en línea/equipo nuevo | Suma |coef|·sd | Máx |coef|·sd |",
        "|---|---:|---|---|---:|---:|",
    ])
    for b in pesos_bloques:
        lines.append(
            f"| `{b['bloque']}` | {b['n_cols']} | {b['tipo']} | {b['recalculo']} | {num(b['sum_abs'], 3)} | {num(b['max_abs'], 4)} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Espacio de búsqueda y movimientos permitidos",
        "",
        "En Master+, los jugadores tienen perfiles de rol muy concentrados. Sobre todas las "
        f"{cuenta(n_v)} partidas de V Master+, la enumeración **100 % exhaustiva** de todas las asignaciones de "
        "los 10 jugadores a 2 equipos × 5 líneas donde cada jugador conserva su rol real o pasa a un rol con "
        "$\\pi_i(r_{\\text{nuevo}}) \\ge \\tau$ resulta completamente manejable (no se requiere heurística greedy/beam):",
        "",
        f"- **$\\tau = 0,20$:** media **{num(df_v['n_cands_t20'].mean(), 1)}** configuraciones válidas por partida "
        f"(mediana {cuenta(int(np.median(df_v['n_cands_t20'])))}, p95 {cuenta(int(np.percentile(df_v['n_cands_t20'], 95)))}, "
        f"máximo {cuenta(int(df_v['n_cands_t20'].max()))}; equivale a media {num(df_v['n_cands_t20'].mean() / 32.0, 2)} particiones de roles × 32 orientaciones de equipo).",
        f"- **$\\tau = 0,35$:** media **{num(df_v['n_cands_t35'].mean(), 1)}** configuraciones válidas por partida "
        f"(mediana {cuenta(int(np.median(df_v['n_cands_t35'])))}, p95 {cuenta(int(np.percentile(df_v['n_cands_t35'], 95)))}, "
        f"máximo {cuenta(int(df_v['n_cands_t35'].max()))}; equivale a media {num(df_v['n_cands_t35'].mean() / 32.0, 2)} particiones de roles × 32 orientaciones de equipo).",
        "- Se evalúan las 32 orientaciones de equipo completas en cada partición de roles porque el lado azul/rojo tiene efecto propio (intercepto distinto de 0 y término par `KALMAN_MRG/brecha_lobby_absoluta`).",
        "",
        "---",
        "",
        "## 3. Resultados principales en V Master+ (probabilidad cruda)",
        "",
        "| Escenario | |p−0,5| media (pp) | |p−0,5| med / p90 (pp) | Mejora med / media (pp) | p∈[0,45;0,55] | p∈[0,48;0,52] | D media (ΔD) | Fuera rol princ. /10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    m_real = resumen_escenario(p_r, p_r, D_r, D_r, nf_r)
    lines.append(
        f"| **{ETIQUETAS_ESC['real']}** | {pp(m_real['dist_mean'])} | {pp(m_real['dist_p50'])} / {pp(m_real['dist_p90'])} | "
        f"— | {pct(m_real['pct_45_55'])} | {pct(m_real['pct_48_52'])} | {num(m_real['D_mean'], 3)} (0,000) | {num(m_real['nf_mean'], 2)} |"
    )
    for esc in ESCENARIOS_PRINCIPALES:
        m_e = resumen_escenario(p_r, df_v[f"p_{esc}"].to_numpy(), df_v[f"D_{esc}"].to_numpy(), D_r, df_v[f"n_fuera_{esc}"].to_numpy())
        dD_str = f"{m_e['dD_mean']:+.3f}".replace(".", ",")
        lines.append(
            f"| {ETIQUETAS_ESC[esc]} | {pp(m_e['dist_mean'])} | {pp(m_e['dist_p50'])} / {pp(m_e['dist_p90'])} | "
            f"{pp(m_e['mej_p50'])} / {pp(m_e['mej_mean'])} | {pct(m_e['pct_45_55'])} | {pct(m_e['pct_48_52'])} | "
            f"{num(m_e['D_mean'], 3)} ({dD_str}) | {num(m_e['nf_mean'], 2)} |"
        )

    # Tabla recalibrada con Platt de Sol (si existe)
    if mapa_platt is not None and "sin_draft" in mapa_platt:
        mp = mapa_platt["sin_draft"]
        a_s, b_s = mp["intercepto"], mp["pendiente"]
        p_r_cal = aplicar_platt(p_r, a_s, b_s)
        lines.extend([
            "",
            "### Resultados con recalibración Platt de Sol (`calib_mapa.json`)",
            "",
            f"Parámetros de Sol para la variante sin draft: $a = {num(a_s, 4)}$, $b = {num(b_s, 4)}$ "
            f"(`preferido_pre_v = {mp['preferido_pre_v']}`; Sol concluyó que la probabilidad original ya está mejor calibrada en V, "
            "por lo que estos valores sirven como control de sensibilidad).",
            "",
            "| Escenario (Platt) | |q−0,5| media (pp) | |q−0,5| med / p90 (pp) | Mejora med (pp) | q∈[0,45;0,55] | q∈[0,48;0,52] |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        mr_c = resumen_escenario(p_r_cal, p_r_cal, D_r, D_r, nf_r)
        lines.append(
            f"| **{ETIQUETAS_ESC['real']}** | {pp(mr_c['dist_mean'])} | {pp(mr_c['dist_p50'])} / {pp(mr_c['dist_p90'])} | "
            f"— | {pct(mr_c['pct_45_55'])} | {pct(mr_c['pct_48_52'])} |"
        )
        for esc in ("a_misma_linea", "b_rol_t20_d_le", "b_rol_t20_sin_tope", "b_rol_t35_d_le", "b_rol_t35_sin_tope",
                    "ab_exh_t20_d_le", "ab_exh_t20_sin_tope", "ab_exh_t35_d_le", "ab_exh_t35_sin_tope"):
            q_e = aplicar_platt(df_v[f"p_{esc}"].to_numpy(), a_s, b_s)
            mc = resumen_escenario(p_r_cal, q_e, df_v[f"D_{esc}"].to_numpy(), D_r, df_v[f"n_fuera_{esc}"].to_numpy())
            lines.append(
                f"| {ETIQUETAS_ESC[esc]} | {pp(mc['dist_mean'])} | {pp(mc['dist_p50'])} / {pp(mc['dist_p90'])} | "
                f"{pp(mc['mej_p50'])} | {pct(mc['pct_45_55'])} | {pct(mc['pct_48_52'])} |"
            )

    # 4. Costo de rol por jugador (d_nuevo y Delta d)
    lines.extend([
        "",
        "---",
        "",
        "## 4. Costo de rol por jugador ($d_{\\text{nuevo}}$ y $\\Delta d$ en V Master+)",
        "",
        f"Sobre las **{cuenta(n_v * 10)}** participaciones jugador-partida de V Master+, con $d_i(\\text{{config}}) = \\pi_i(\\text{{rol principal}}) - \\pi_i(\\text{{línea asignada}})$ usando los $\\pi_i$ causales de `perfil_roles.parquet`:",
        "",
        "| Escenario | d media | d p90 | d p99 | % d > 0,30 | % fuera rol princ. | Δd media | Δd p90 / p99 | % mejora / empeora d | % cambia rol / eq |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    sj = out_v["stats_jug"]
    for esc in ("real", *ESCENARIOS_JUGADORES):
        s = sj[esc]
        dd_m = f"{s['dd_mean']:+.4f}".replace(".", ",")
        lines.append(
            f"| {ETIQUETAS_ESC[esc]} | {num(s['d_mean'], 4)} | {num(s['d_p90'], 4)} | {num(s['d_p99'], 4)} | "
            f"{pct(s['pct_d_gt03'])} | {pct(s['pct_fuera'])} | {dd_m} | {num(s['dd_p90'], 3)} / {num(s['dd_p99'], 3)} | "
            f"{pct(s['pct_mejora_d'], 1)} / {pct(s['pct_empeora_d'], 1)} | {pct(s['pct_cambio_rol'], 1)} / {pct(s['pct_cambio_eq'], 1)} |"
        )

    # 5. Frontera parejo vs costo de rol
    lines.extend([
        "",
        "---",
        "",
        "## 5. Frontera «parejo vs. costo de rol» (V Master+)",
        "",
        "Evolución de $|p - 0,5|$ al relajar el tope máximo permitido de $\\Delta D = D_{\\text{nuevo}} - D_{\\text{real}}$ en la búsqueda exhaustiva `(a+b)`:",
        "",
        "| Umbral τ | Tope de costo de rol | |p−0,5| media (pp) | |p−0,5| med (pp) | p∈[0,45;0,55] | p∈[0,48;0,52] | D media (ΔD) | Fuera rol /10 | % jug d>0,30 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for tau_tag, tau_lbl in (("t20", "τ = 0,20"), ("t35", "τ = 0,35")):
        if tau_tag == "t20":
            for extra_k, extra_lbl in (("ab_minD_t20", "Minimizar D primero"), ("ab_pareto_t20", "Pareto jug (Δd_i ≤ 0)")):
                me = resumen_escenario(p_r, df_v[f"p_{extra_k}"].to_numpy(), df_v[f"D_{extra_k}"].to_numpy(), D_r, df_v[f"n_fuera_{extra_k}"].to_numpy())
                dD_s = f"{me['dD_mean']:+.3f}".replace(".", ",")
                lines.append(
                    f"| {tau_lbl} | {extra_lbl} | {pp(me['dist_mean'])} | {pp(me['dist_p50'])} | "
                    f"{pct(me['pct_45_55'])} | {pct(me['pct_48_52'])} | {num(me['D_mean'], 3)} ({dD_s}) | "
                    f"{num(me['nf_mean'], 2)} | {pct(sj[extra_k]['pct_d_gt03'])} |"
                )
        for dmax in FRONTERA_DELTAS:
            k_f = f"{tau_tag}_dmax_{'inf' if math.isinf(dmax) else f'{dmax:.2f}'}"
            fr = out_v["frontera"][k_f]
            lbl_f = "Sin tope (ΔD libre)" if math.isinf(dmax) else f"ΔD ≤ +{num(dmax, 2)}" if dmax > 0 else "ΔD ≤ 0,00 (D ≤ D_real)"
            me = resumen_escenario(p_r, fr["p"], fr["D"], D_r, fr["n_fuera"])
            dD_s = f"{me['dD_mean']:+.3f}".replace(".", ",")
            pct_gt = float(fr["d_jug_gt03"].sum() / (n_v * 10.0))
            lines.append(
                f"| {tau_lbl} | {lbl_f} | {pp(me['dist_mean'])} | {pp(me['dist_p50'])} | "
                f"{pct(me['pct_45_55'])} | {pct(me['pct_48_52'])} | {num(me['D_mean'], 3)} ({dD_s}) | "
                f"{num(me['nf_mean'], 2)} | {pct(pct_gt)} |"
            )

    # 6. Desbalance vs autofill
    lines.extend([
        "",
        "---",
        "",
        "## 6. ¿Las partidas más desparejas tienen más autofill?",
        "",
        f"- Correlación $|p_{{\\text{{real}}}} - 0,5|$ vs. $D_{{\\text{{real}}}}$: Pearson $r = {num(r_p_D, 4)}$ ($p = {p_p_D:.2e}$), Spearman $\\rho = {num(rho_D, 4)}$ ($p = {ps_D:.2e}$).",
        f"- Correlación $|p_{{\\text{{real}}}} - 0,5|$ vs. cantidad de autofills ($\pi < 0,15, n_{{\\text{{prev}}}} \\ge 20$): Pearson $r = {num(r_p_af, 4)}$, Spearman $\\rho = {num(rho_af, 4)}$.",
        f"- Correlación $|p_{{\\text{{real}}}} - 0,5|$ vs. jugadores fuera de su rol principal: Pearson $r = {num(r_p_nf, 4)}$.",
        f"- Correlación direccional $(p_{{\\text{{azul}}}} - 0,5)$ vs. $(D_{{\\text{{azul}}}} - D_{{\\text{{rojo}}}})$: Pearson $r = {num(r_dir, 4)}$ (signo negativo esperado: el equipo con mayor desvío de rol tiene menor probabilidad de ganar).",
        "",
        "### Estratificación por nivel de desbalance real ($|p_{\\text{real}} - 0,5|$)",
        "",
        "| Tramo de |p_real − 0,5| | Partidas | % | D_real media | Autofills / partida | Fuera rol princ. /10 | |p−0,5| tras (a) (pp) | |p−0,5| tras (a+b) τ=0,20 D≤D_real (pp) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    bins = [(0.0, 0.05, "[0, 5) pp"), (0.05, 0.10, "[5, 10) pp"), (0.10, 0.15, "[10, 15) pp"), (0.15, 1.0, "≥ 15 pp")]
    p_a_v = df_v["p_a_misma_linea"].to_numpy()
    p_ab_v = df_v["p_ab_exh_t20_d_le"].to_numpy()
    for lo, hi, lbl in bins:
        m_b = (dist_r >= lo) & (dist_r < hi)
        lines.append(
            f"| {lbl} | {cuenta(int(m_b.sum()))} | {pct(m_b.mean(), 1)} | {num(D_r[m_b].mean(), 3)} | "
            f"{num(af_r[m_b].mean(), 2)} | {num(nf_r[m_b].mean(), 2)} | "
            f"{pp(np.abs(p_a_v[m_b] - 0.5).mean())} | {pp(np.abs(p_ab_v[m_b] - 0.5).mean())} |"
        )

    lines.extend([
        "",
        "### Estratificación por cantidad de autofills en la partida",
        "",
        "| Autofills en partida | Partidas | % | |p_real − 0,5| media (pp) | |p_real − 0,5| med (pp) | D_real media |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for k_af, lbl_af in ((0, "0"), (1, "1"), (2, "2"), (3, "3+")):
        m_k = (af_r == k_af) if k_af < 3 else (af_r >= 3)
        lines.append(
            f"| {lbl_af} | {cuenta(int(m_k.sum()))} | {pct(m_k.mean(), 1)} | "
            f"{pp(dist_r[m_k].mean())} | {pp(np.percentile(dist_r[m_k], 50))} | {num(D_r[m_k].mean(), 3)} |"
        )

    # 7. Sesgo de optimización
    lines.extend([
        "",
        "---",
        "",
        "## 7. Sesgo de optimización (mitades disjuntas A ↔ B)",
        "",
        "Se ajustaron dos logísticas independientes con mitades disjuntas del entrenamiento Master+ "
        f"(partición cronológica intercalada: {cuenta(info_modelos['n_tr_a'])} y {cuenta(info_modelos['n_tr_b'])} partidas). "
        "En cada partida de V Master+, un modelo elige la mejor asignación y el otro la evalúa de forma independiente (y viceversa; "
        "se promedian ambos sentidos). Nota: cada mitad dispone de $N/2$ datos, por lo que su varianza de estimación es "
        "aproximadamente el doble que la del modelo completo; el sesgo aquí medido actúa como una cota conservadora del error "
        "de estimación de coeficientes (aunque no captura factores no observados fuera de las 320 features).",
        "",
        "| Escenario | |p−0,5| real med / media (pp) | Elegido med / media (pp) | Eval. indep. med / media (pp) | Sesgo media (pp) | Mejora retenida | Indep. [0,45;0,55] | Indep. [0,48;0,52] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    d_ab_real = 0.5 * (np.abs(df_v["p_a_real"].to_numpy() - 0.5) + np.abs(df_v["p_b_real"].to_numpy() - 0.5))
    real_m = float(d_ab_real.mean())
    real_med = float(0.5 * (np.percentile(np.abs(df_v["p_a_real"].to_numpy() - 0.5), 50)
                            + np.percentile(np.abs(df_v["p_b_real"].to_numpy() - 0.5), 50)))
    for esc in ESCENARIOS_PRINCIPALES:
        d_el_a = np.abs(df_v[f"p_a_eleg_{esc}"].to_numpy() - 0.5)
        d_el_b = np.abs(df_v[f"p_b_eleg_{esc}"].to_numpy() - 0.5)
        d_in_a = np.abs(df_v[f"p_b_indep_{esc}"].to_numpy() - 0.5)
        d_in_b = np.abs(df_v[f"p_a_indep_{esc}"].to_numpy() - 0.5)
        el_m = float(0.5 * (d_el_a.mean() + d_el_b.mean()))
        el_med = float(0.5 * (np.percentile(d_el_a, 50) + np.percentile(d_el_b, 50)))
        in_m = float(0.5 * (d_in_a.mean() + d_in_b.mean()))
        in_med = float(0.5 * (np.percentile(d_in_a, 50) + np.percentile(d_in_b, 50)))
        sesgo = in_m - el_m
        ret = (real_m - in_m) / (real_m - el_m) if (real_m - el_m) > 1e-9 else 0.0
        p45 = 0.5 * ((d_in_a <= 0.05).mean() + (d_in_b <= 0.05).mean())
        p48 = 0.5 * ((d_in_a <= 0.02).mean() + (d_in_b <= 0.02).mean())
        lines.append(
            f"| {ETIQUETAS_ESC[esc]} | {pp(real_med)} / {pp(real_m)} | {pp(el_med)} / {pp(el_m)} | "
            f"{pp(in_med)} / {pp(in_m)} | +{pp(sesgo)} | {pct(ret, 1)} | {pct(p45)} | {pct(p48)} |"
        )

    # 8. Chequeo secundario CON DRAFT
    if draft_res is not None:
        pr_d = draft_res["p_real"]
        pb_d = draft_res["p_best"]
        dr_d = np.abs(pr_d - 0.5)
        db_d = np.abs(pb_d - 0.5)
        d_el_d = 0.5 * (np.abs(draft_res["p_a_eleg"] - 0.5) + np.abs(draft_res["p_b_eleg"] - 0.5))
        d_in_da = np.abs(draft_res["p_b_indep"] - 0.5)
        d_in_db = np.abs(draft_res["p_a_indep"] - 0.5)
        d_in_d = 0.5 * (d_in_da + d_in_db)
        d_re_d = 0.5 * (np.abs(draft_res["p_a_real"] - 0.5) + np.abs(draft_res["p_b_real"] - 0.5))
        ret_d = (d_re_d.mean() - d_in_d.mean()) / (d_re_d.mean() - d_el_d.mean())

        lines.extend([
            "",
            "---",
            "",
            "## 8. Chequeo secundario: escalón 9 CON DRAFT (solo misma línea)",
            "",
            "Cada jugador se lleva su campeón al intercambiar equipo dentro de la misma línea ($2^5 = 32$ configuraciones, $\\Delta D = 0$). "
            "Se recalculan exactamente las 1.504 columnas, incluidas las sinergias aliadas cruzadas y los duelos @10/@15.",
            "",
            "| Variante (escalón 9 con draft) | |p−0,5| media (pp) | |p−0,5| med / p90 (pp) | Mejora med (pp) | p∈[0,45;0,55] | p∈[0,48;0,52] |",
            "|---|---:|---:|---:|---:|---:|",
            f"| Real (cruda) | {pp(dr_d.mean())} | {pp(np.percentile(dr_d, 50))} / {pp(np.percentile(dr_d, 90))} | — | {pct((dr_d <= 0.05).mean())} | {pct((dr_d <= 0.02).mean())} |",
            f"| (a) Misma línea (cruda) | {pp(db_d.mean())} | {pp(np.percentile(db_d, 50))} / {pp(np.percentile(db_d, 90))} | {pp(np.percentile(dr_d - db_d, 50))} | {pct((db_d <= 0.05).mean())} | {pct((db_d <= 0.02).mean())} |",
        ])
        if mapa_platt is not None and "escalon9" in mapa_platt:
            m9 = mapa_platt["escalon9"]
            qr_d = aplicar_platt(pr_d, m9["intercepto"], m9["pendiente"])
            qb_d = aplicar_platt(pb_d, m9["intercepto"], m9["pendiente"])
            qdr, qdb = np.abs(qr_d - 0.5), np.abs(qb_d - 0.5)
            lines.extend([
                f"| Real (Platt Sol: a={num(m9['intercepto'])}, b={num(m9['pendiente'])}) | {pp(qdr.mean())} | {pp(np.percentile(qdr, 50))} / {pp(np.percentile(qdr, 90))} | — | {pct((qdr <= 0.05).mean())} | {pct((qdr <= 0.02).mean())} |",
                f"| (a) Misma línea (Platt Sol) | {pp(qdb.mean())} | {pp(np.percentile(qdb, 50))} / {pp(np.percentile(qdb, 90))} | {pp(np.percentile(qdr - qdb, 50))} | {pct((qdb <= 0.05).mean())} | {pct((qdb <= 0.02).mean())} |",
            ])
        lines.extend([
            "",
            f"- **Sesgo de optimización con draft `(a)` (mitades A ↔ B):** |p−0,5| real **{pp(d_re_d.mean())} pp** → "
            f"elegido **{pp(d_el_d.mean())} pp** → evaluado de forma independiente **{pp(d_in_d.mean())} pp** "
            f"(sesgo **+{pp(d_in_d.mean() - d_el_d.mean())} pp**, mejora retenida **{pct(ret_d, 1)}**, "
            f"indep. en [0,45;0,55]: **{pct(0.5 * ((d_in_da <= 0.05).mean() + (d_in_db <= 0.05).mean()))}**).",
        ])

    # 9. Muestra de entrenamiento Master+ (si se corrió)
    if out_tr is not None:
        df_t = out_tr["df_res"]
        pt_r = df_t["p_real"].to_numpy()
        Dt_r = df_t["D_real"].to_numpy()
        nft_r = df_t["n_fuera_real"].to_numpy()
        lines.extend([
            "",
            "---",
            "",
            f"## 9. Verificación sobre muestra de entrenamiento Master+ (n = {cuenta(len(df_t))})",
            "",
            "Muestra aleatoria estratificada del entrenamiento Master+ (`D['fi_v'] & D['master']`, anterior al 16/09). "
            "Sirve para comprobar que la estructura de roles y la ganancia de reemparejamiento no son propias solo de la ventana V.",
            "",
            "| Escenario (muestra train Master+) | |p−0,5| media (pp) | |p−0,5| med (pp) | p∈[0,45;0,55] | p∈[0,48;0,52] | D media (ΔD) |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        mt0 = resumen_escenario(pt_r, pt_r, Dt_r, Dt_r, nft_r)
        lines.append(
            f"| **{ETIQUETAS_ESC['real']}** | {pp(mt0['dist_mean'])} | {pp(mt0['dist_p50'])} | "
            f"{pct(mt0['pct_45_55'])} | {pct(mt0['pct_48_52'])} | {num(mt0['D_mean'], 3)} (0,000) |"
        )
        for esc in ("a_misma_linea", "b_rol_t20_d_le", "b_rol_t20_sin_tope", "b_rol_t35_d_le",
                    "ab_exh_t20_d_le", "ab_exh_t20_sin_tope", "ab_exh_t35_d_le", "ab_exh_t35_sin_tope"):
            mt = resumen_escenario(pt_r, df_t[f"p_{esc}"].to_numpy(), df_t[f"D_{esc}"].to_numpy(), Dt_r, df_t[f"n_fuera_{esc}"].to_numpy())
            dD_s = f"{mt['dD_mean']:+.3f}".replace(".", ",")
            lines.append(
                f"| {ETIQUETAS_ESC[esc]} | {pp(mt['dist_mean'])} | {pp(mt['dist_p50'])} | "
                f"{pct(mt['pct_45_55'])} | {pct(mt['pct_48_52'])} | {num(mt['D_mean'], 3)} ({dD_s}) |"
            )

    # 10. Tres ejemplos concretos
    lines.extend([
        "",
        "---",
        "",
        "## 10. Tres ejemplos concretos en V Master+",
        "",
    ])
    for ej in ejemplos:
        ganador = "Azul" if ej["y"] == 1 else "Rojo"
        dD_ej = f"{ej['D_nuevo'] - ej['D_real']:+.3f}".replace(".", ",")
        lines.extend([
            f"### {ej['titulo']}",
            "",
            f"- **Partida:** `{ej['match_id']}` (ganó **{ganador}**).",
            f"- **Probabilidad azul (sin draft):** real **{pct(ej['p_real'])}** → solo `(a)` **{pct(ej['p_a'])}** → alternativa `{ej['escenario']}` **{pct(ej['p_nuevo'])}**.",
            f"- **Costo total de rol $D$:** real **{num(ej['D_real'], 3)}** → nuevo **{num(ej['D_nuevo'], 3)}** ($\\Delta D = {dD_ej}$).",
            "",
            "| Jugador | Rol principal | n_prev | Real (Eq · Línea) | d_real | Nuevo (Eq · Línea) | π(nuevo) | d_nuevo | Δd |",
            "|---|---|---:|---|---:|---|---:|---:|---:|",
        ])
        for f in ej["filas"]:
            marca = " *" if (f["eq_real"] != f["eq_nuevo"] or f["lin_real"] != f["lin_nuevo"]) else ""
            dd_s = f"{f['delta_d']:+.3f}".replace(".", ",")
            lines.append(
                f"| `{f['player_id']}`{marca} | {f['rol_principal']} | {cuenta(f['n_prev'])} | "
                f"{f['eq_real']} · {f['lin_real']} | {num(f['d_real'], 3)} | "
                f"{f['eq_nuevo']} · {f['lin_nuevo']} | {num(f['pi_nuevo'], 3)} | "
                f"{num(f['d_nuevo'], 3)} | {dd_s} |"
            )
        lines.append("")

    (SALIDA / "REPORTE_swaps.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    ap = argparse.ArgumentParser(description="Análisis contrafáctico de swaps en Master+ (LAS).")
    ap.add_argument("--muestra", type=int, default=0,
                    help="Si >0, corre prueba rápida sobre las primeras N partidas de V Master+ sin sobrescribir archivos finales.")
    ap.add_argument("--n-train-sample", type=int, default=5000,
                    help="Cantidad de partidas Master+ de entrenamiento para verificación (0 para omitir).")
    ap.add_argument("--sin-lock", action="store_true",
                    help="Solo para pruebas rápidas chicas (<30 s): omite el archivo RECURSO_EN_USO.txt.")
    args = ap.parse_args()

    es_completo = args.muestra <= 0
    ctx = reservar_recurso("neon-swaps") if (es_completo and not args.sin_lock) else contextmanager(lambda: (yield))()

    with ctx, threadpool_limits(limits=2):
        t_ini = time.perf_counter()
        print("Cargando cachés del escalón 9...", flush=True)
        D = cargar()
        t_arr = D["t"]
        y = D["y"]
        tr = D["fi_v"][D["master"][D["fi_v"]]]
        v_all = np.flatnonzero(D["val"])
        assert D["master"][v_all].all(), "V debe ser estrictamente Master+!"
        assert not D["val0"][tr].any() and t_arr[tr].max() < t_arr[v_all].min(), "Fuga temporal en tr!"

        # Partición cronológica intercalada del entrenamiento Master+ en dos mitades disjuntas
        tr_ord = tr[np.lexsort((D["Z"]["mids"][tr], t_arr[tr]))]
        tr_a = tr_ord[0::2]
        tr_b = tr_ord[1::2]

        corte = float(np.quantile(t_arr[~D["val0"]], 0.90))
        extras_all = extras_referencia(D, escalon=9)
        columnas, base, keep, grupos, orden_cols = cargar_esquema_sin_draft(D, extras_all)

        # Seleccionar índices a evaluar
        if es_completo:
            v_eval = v_all
            rng = np.random.default_rng(20261119)
            if args.n_train_sample > 0:
                tr_sample = np.sort(rng.choice(tr, size=min(args.n_train_sample, len(tr)), replace=False))
            else:
                tr_sample = np.array([], dtype=np.int64)
        else:
            v_eval = v_all[:args.muestra]
            tr_sample = np.array([], dtype=np.int64)

        eval_idx = np.concatenate([v_eval, tr_sample]).astype(np.int64)
        v_idx_in_eval = np.arange(len(v_eval), dtype=np.int64)
        tr_idx_in_eval = np.arange(len(v_eval), len(eval_idx), dtype=np.int64)

        # Construir historial sin fuga + tensores contrafácticos exactos
        T = construir_historial_y_tensores(
            D, corte, columnas, base, eval_idx, v_idx_in_eval, extraer_draft=True
        )

        # Ensamblar matriz X_sin_draft (NM, 320) exactamente idéntica a Sol
        ml_orig = grupos["ML"]
        mmr_orig = grupos["MMR_HIST"]
        extras_sin = extras_all[:, keep].copy()
        extras_sin[:, [keep.index(i) for i in range(ml_orig.start, ml_orig.stop)]] = T["ml_diff_all"]
        extras_sin[:, [keep.index(i) for i in range(mmr_orig.start, mmr_orig.stop)]] = T["mmr_diff_all"]
        X_sin_all = np.hstack([T["xd_sin"], extras_sin]).astype(np.float32)
        assert X_sin_all.shape[1] == 320
        del extras_sin

        print("Ajustando logística sin draft (completa y mitades disjuntas A / B)...", flush=True)
        mod_full = ajustar_modelo_sin_draft(X_sin_all, y, tr, C=0.05)
        mod_a = ajustar_modelo_sin_draft(X_sin_all, y, tr_a, C=0.05)
        mod_b = ajustar_modelo_sin_draft(X_sin_all, y, tr_b, C=0.05)

        # Validación directa con predict_proba sobre V
        def _pred_dir(mod: dict, idx: np.ndarray) -> np.ndarray:
            Xs = sparse.csr_matrix(((X_sin_all[idx] - mod["mu"].astype(np.float32)) / mod["sd"].astype(np.float32)).astype(np.float32))
            return mod["lr"].predict_proba(Xs)[:, 1]

        p_dir_v = _pred_dir(mod_full, v_eval)
        p_a_v = _pred_dir(mod_a, v_eval)
        p_b_v = _pred_dir(mod_b, v_eval)
        y_v = y[v_eval]

        info_modelos = {
            "n_tr": len(tr), "n_tr_a": len(tr_a), "n_tr_b": len(tr_b),
            "auc_full": float(roc_auc_score(y_v, p_dir_v)),
            "acc_full": float(((p_dir_v > 0.5) == y_v).mean()),
            "ll_full": float(log_loss(y_v, p_dir_v)),
            "auc_a": float(roc_auc_score(y_v, p_a_v)),
            "acc_a": float(((p_a_v > 0.5) == y_v).mean()),
            "auc_b": float(roc_auc_score(y_v, p_b_v)),
            "acc_b": float(((p_b_v > 0.5) == y_v).mean()),
        }
        print(
            f"Sin draft en V ({len(v_eval):,}): Full AUC {info_modelos['auc_full']:.4f} / acc {info_modelos['acc_full']:.2%} | "
            f"A {info_modelos['auc_a']:.4f} / {info_modelos['acc_a']:.2%} | "
            f"B {info_modelos['auc_b']:.4f} / {info_modelos['acc_b']:.2%}",
            flush=True,
        )

        # Factorizar pesos por modelo en C_lin[N_eval, 10, 5] + w_nl[10]
        B0_f, Clin_f, wnl_f = precomputar_modelo_eval(mod_full, T, base)
        B0_a, Clin_a, wnl_a = precomputar_modelo_eval(mod_a, T, base)
        B0_b, Clin_b, wnl_b = precomputar_modelo_eval(mod_b, T, base)

        # Pesos |coef|·sd por bloque para el reporte
        c_abs = np.abs(mod_full["coef_std"])
        idx_sl = base.index("share_lane")
        idx_off = base.index("offrole")
        xj_jl_idx = [l * 40 + idx_sl for l in range(5)] + [l * 40 + idx_off for l in range(5)]
        xj_j_idx = [i for i in range(200) if i not in set(xj_jl_idx)]
        pesos_bloques = [
            {"bloque": "Xj (solo jugador, 38×5)", "n_cols": len(xj_j_idx),
             "tipo": "Jugador → línea", "recalculo": "Exacto (se mueve con el jugador; cambia el peso según la línea asignada)",
             "sum_abs": float(c_abs[xj_j_idx].sum()), "max_abs": float(c_abs[xj_j_idx].max())},
            {"bloque": "Xj (share_lane, offrole ×5)", "n_cols": len(xj_jl_idx),
             "tipo": "Jugador × línea", "recalculo": "Exacto (recalculado causalmente con n_pl en la línea nueva)",
             "sum_abs": float(c_abs[xj_jl_idx].sum()), "max_abs": float(c_abs[xj_jl_idx].max())},
            {"bloque": "ML (margen por línea, 15×5)", "n_cols": 75,
             "tipo": "Jugador × línea", "recalculo": "Exacto (recalculado causalmente con S_pl y N_pl en la línea nueva)",
             "sum_abs": float(c_abs[200:275].sum()), "max_abs": float(c_abs[200:275].max())},
            {"bloque": "MMR_HIST (nivel histórico)", "n_cols": 3,
             "tipo": "Solo jugador (media eq.)", "recalculo": "Exacto (re-promediado en equipo azul − rojo)",
             "sum_abs": float(c_abs[275:278].sum()), "max_abs": float(c_abs[275:278].max())},
            {"bloque": "FOTO1_DIN (W/L dinámico)", "n_cols": 4,
             "tipo": "Solo jugador (media eq.)", "recalculo": "Exacto (re-promediado en equipo azul − rojo)",
             "sum_abs": float(c_abs[278:282].sum()), "max_abs": float(c_abs[278:282].max())},
            {"bloque": "ROL_WR (medias + mínimos)", "n_cols": 8,
             "tipo": "Jugador × línea + no lineal", "recalculo": "Exacto (wr_lane, exceso, logn, estreno, autofill + min por equipo)",
             "sum_abs": float(c_abs[282:290].sum()), "max_abs": float(c_abs[282:290].max())},
            {"bloque": "KALMAN_OUT (Elo + 5 líneas)", "n_cols": 12,
             "tipo": "Jugador + línea + no lineal", "recalculo": "Exacto (media, min, max, std, top2, forma, rivales y 5 slots de línea)",
             "sum_abs": float(c_abs[290:302].sum()), "max_abs": float(c_abs[290:302].max())},
            {"bloque": "KALMAN_MRG (margen + brecha)", "n_cols": 7,
             "tipo": "Jugador + no lineal", "recalculo": "Exacto (media, min, max, forma, incert., exceso y |mb − mr|)",
             "sum_abs": float(c_abs[302:309].sum()), "max_abs": float(c_abs[302:309].max())},
            {"bloque": "TEMPO_OBJ (FB, FT, trinket, dur.)", "n_cols": 7,
             "tipo": "Jugador + slots rol", "recalculo": "Exacto (totales/medias + slots específicos JG+SUP y BOT+TOP)",
             "sum_abs": float(c_abs[309:316].sum()), "max_abs": float(c_abs[309:316].max())},
            {"bloque": "ECON_MID (oro 15→20, efic.)", "n_cols": 4,
             "tipo": "Jugador + slots + no lineal", "recalculo": "Exacto (media, max por equipo y slot BOT+MID)",
             "sum_abs": float(c_abs[316:320].sum()), "max_abs": float(c_abs[316:320].max())},
        ]

        # Liberar X_sin_all para mantener RAM baja
        del X_sin_all
        gc.collect()

        # Cargar perfil_roles alineado a eval_idx
        print("Cargando perfil_roles.parquet alineado a las partidas Master+...", flush=True)
        mids_eval = D["Z"]["mids"][eval_idx].astype(str)
        players_eval = D["B"]["player"][eval_idx].reshape(len(eval_idx), 10).astype(str)
        roles_all = cargar_perfil_roles_alineado(mids_eval, players_eval)

        def _slice_roles(sl: np.ndarray) -> dict:
            return {k: v[sl] for k, v in roles_all.items()}

        def _slice_T(sl: np.ndarray) -> dict:
            return {
                "wr_lane_5": T["wr_lane_5"][sl],
                "wr_lane_diff_5": T["wr_lane_diff_5"][sl],
                "g11_eval": {k: v[sl] for k, v in T["g11_eval"].items()},
            }

        # Correr búsqueda exhaustiva en V Master+
        print(f"Ejecutando búsqueda exhaustiva en V Master+ ({len(v_eval):,} partidas)...", flush=True)
        out_v = correr_experimento_conjunto(
            "V Master+",
            mids_eval[v_idx_in_eval],
            y_v,
            players_eval[v_idx_in_eval],
            _slice_roles(v_idx_in_eval),
            _slice_T(v_idx_in_eval),
            B0_f, Clin_f[v_idx_in_eval], wnl_f,
            B0_a, Clin_a[v_idx_in_eval], wnl_a,
            B0_b, Clin_b[v_idx_in_eval], wnl_b,
            guardar_jugadores=True,
        )

        # Validar max |Δp| en asignación real
        p_cf_v = out_v["df_res"]["p_real"].to_numpy()
        err_own = float(np.max(np.abs(p_cf_v - p_dir_v)))
        err_sol = np.nan
        ruta_sol_v = SALIDA / "calib_v_sin_draft.parquet"
        if ruta_sol_v.is_file():
            df_sol_v = pd.read_parquet(ruta_sol_v).set_index("match_id").reindex(mids_eval[v_idx_in_eval])
            err_sol = float(np.max(np.abs(p_cf_v - df_sol_v["p_azul"].to_numpy())))
        validacion = {"err_own": err_own, "err_sol": err_sol}
        print(f"VALIDACIÓN sin draft en asignación real: max |Δp| vs propio = {err_own:.3e} | vs Sol = {err_sol:.3e}", flush=True)
        assert err_own < 1e-5, f"Error demasiado alto en contrafáctico sin draft: {err_own}"

        # Correr muestra de entrenamiento Master+ (solo si hay)
        out_tr = None
        if len(tr_idx_in_eval) > 0:
            print(f"Ejecutando búsqueda exhaustiva en muestra train Master+ ({len(tr_idx_in_eval):,} partidas)...", flush=True)
            out_tr = correr_experimento_conjunto(
                "Train Master+",
                mids_eval[tr_idx_in_eval],
                y[tr_sample],
                players_eval[tr_idx_in_eval],
                _slice_roles(tr_idx_in_eval),
                _slice_T(tr_idx_in_eval),
                B0_f, Clin_f[tr_idx_in_eval], wnl_f,
                B0_a, Clin_a[tr_idx_in_eval], wnl_a,
                B0_b, Clin_b[tr_idx_in_eval], wnl_b,
                guardar_jugadores=False,
            )

        # Chequeo secundario CON DRAFT (solo V Master+, 2^5 = 32 swaps misma línea)
        draft_res = evaluar_chequeo_con_draft(
            D, extras_all, tr, tr_a, tr_b, v_eval, T, v_idx_in_eval
        )
        print(
            f"VALIDACIÓN con draft (escalón 9): AUC {draft_res['auc_full']:.4f} / acc {draft_res['acc_full']:.2%} | "
            f"max |Δp| vs propio = {draft_res['err_direct']:.3e} | vs Sol = {draft_res['err_sol']:.3e}",
            flush=True,
        )
        assert draft_res["err_direct"] < 1e-5, f"Error en contrafáctico con draft: {draft_res['err_direct']}"

        # Mapa de calibración de Sol
        mapa_platt = None
        ruta_mapa = SALIDA / "calib_mapa.json"
        if ruta_mapa.is_file():
            mapa_platt = json.loads(ruta_mapa.read_text(encoding="utf-8"))

        # Agregar columnas de draft y Platt a df_res de V Master+
        df_res_v = out_v["df_res"]
        df_res_v["p_real_draft"] = draft_res["p_real"]
        df_res_v["p_a_misma_linea_draft"] = draft_res["p_best"]
        df_res_v["p_a_eleg_draft"] = draft_res["p_a_eleg"]
        df_res_v["p_b_indep_draft"] = draft_res["p_b_indep"]
        df_res_v["p_b_eleg_draft"] = draft_res["p_b_eleg"]
        df_res_v["p_a_indep_draft"] = draft_res["p_a_indep"]
        if mapa_platt is not None:
            if "sin_draft" in mapa_platt:
                a_s, b_s = mapa_platt["sin_draft"]["intercepto"], mapa_platt["sin_draft"]["pendiente"]
                df_res_v["p_real_platt"] = aplicar_platt(df_res_v["p_real"].to_numpy(), a_s, b_s)
                for esc in ESCENARIOS_PRINCIPALES:
                    df_res_v[f"p_{esc}_platt"] = aplicar_platt(df_res_v[f"p_{esc}"].to_numpy(), a_s, b_s)
            if "escalon9" in mapa_platt:
                a_9, b_9 = mapa_platt["escalon9"]["intercepto"], mapa_platt["escalon9"]["pendiente"]
                df_res_v["p_real_draft_platt"] = aplicar_platt(df_res_v["p_real_draft"].to_numpy(), a_9, b_9)
                df_res_v["p_a_misma_linea_draft_platt"] = aplicar_platt(df_res_v["p_a_misma_linea_draft"].to_numpy(), a_9, b_9)

        dur_s = time.perf_counter() - t_ini

        # Imprimir resumen rápido en consola
        p_r = df_res_v["p_real"].to_numpy()
        D_r = df_res_v["D_real"].to_numpy()
        print("\n=== RESUMEN V MASTER+ (p cruda, sin draft) ===", flush=True)
        for esc in ("real", "a_misma_linea", "b_rol_t20_d_le", "b_rol_t20_sin_tope",
                    "b_rol_t35_d_le", "ab_exh_t20_d_le", "ab_exh_t20_sin_tope",
                    "ab_exh_t35_d_le", "ab_exh_t35_sin_tope"):
            pe = p_r if esc == "real" else df_res_v[f"p_{esc}"].to_numpy()
            De = D_r if esc == "real" else df_res_v[f"D_{esc}"].to_numpy()
            nfe = df_res_v["n_fuera_real"].to_numpy() if esc == "real" else df_res_v[f"n_fuera_{esc}"].to_numpy()
            m_e = resumen_escenario(p_r, pe, De, D_r, nfe)
            print(
                f"  {esc:22s} | |p-0.5| med={100*m_e['dist_p50']:5.2f}pp mean={100*m_e['dist_mean']:5.2f}pp | "
                f"[45,55]={100*m_e['pct_45_55']:5.1f}% [48,52]={100*m_e['pct_48_52']:5.1f}% | "
                f"D={m_e['D_mean']:.3f} (dD={m_e['dD_mean']:+.3f})",
                flush=True,
            )

        if es_completo:
            print("\nGuardando parquets y reporte final...", flush=True)
            df_res_v.to_parquet(SALIDA / "swaps_resultados.parquet", engine="pyarrow", index=False)
            out_v["df_jug"].to_parquet(SALIDA / "swaps_jugadores.parquet", engine="pyarrow", index=False)
            n_v_f = len(df_res_v)
            filas_fr = []
            for tau_tag in ("t20", "t35"):
                tau_val = 0.20 if tau_tag == "t20" else 0.35
                for dmax in FRONTERA_DELTAS:
                    k_f = f"{tau_tag}_dmax_{'inf' if math.isinf(dmax) else f'{dmax:.2f}'}"
                    fr = out_v["frontera"][k_f]
                    d_in = 0.5 * (np.abs(fr["p_b_indep"] - 0.5) + np.abs(fr["p_a_indep"] - 0.5))
                    p45_in = 0.5 * ((np.abs(fr["p_b_indep"] - 0.5) <= 0.05).mean()
                                    + (np.abs(fr["p_a_indep"] - 0.5) <= 0.05).mean())
                    filas_fr.append({
                        "clave": k_f,
                        "tau": tau_val,
                        "dmax": float(dmax),
                        "dist_full_pp": float(100.0 * np.abs(fr["p"] - 0.5).mean()),
                        "dist_indep_pp": float(100.0 * d_in.mean()),
                        "pct_45_55_full": float((np.abs(fr["p"] - 0.5) <= 0.05).mean()),
                        "pct_45_55_indep": float(p45_in),
                        "D_mean": float(fr["D"].mean()),
                        "dD_mean": float((fr["D"] - D_r).mean()),
                        "pct_fuera": float(fr["n_fuera"].sum() / (n_v_f * 10.0)),
                        "pct_d_gt03": float(fr["d_jug_gt03"].sum() / (n_v_f * 10.0)),
                        "pct_mejora_d": float(fr["n_mejora"].sum() / (n_v_f * 10.0)),
                        "pct_empeora_d": float(fr["n_empeora"].sum() / (n_v_f * 10.0)),
                    })
            pd.DataFrame(filas_fr).to_parquet(SALIDA / "swaps_frontera.parquet", engine="pyarrow", index=False)
            ejemplos = seleccionar_tres_ejemplos(
                df_res_v, out_v["best_cfgs"], players_eval[v_idx_in_eval], _slice_roles(v_idx_in_eval)
            )
            escribir_reporte(
                info_modelos, validacion, pesos_bloques, out_v, out_tr, draft_res, mapa_platt, ejemplos, dur_s
            )
            print(f"[+] Guardados: swaps_resultados.parquet, swaps_jugadores.parquet, swaps_frontera.parquet y REPORTE_swaps.md ({dur_s:.1f} s)", flush=True)
        else:
            print(f"\n[PRUEBA --muestra {args.muestra}] Terminada con éxito en {dur_s:.1f} s (archivos finales no modificados).", flush=True)


if __name__ == "__main__":
    main()
