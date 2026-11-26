"""Ingeniería de variables y modelos in-game para predicción al minuto 10, 15 y 20.

Diseñado sin fuga temporal y respetando la deuda metodológica de `AGENTS.md` §3.F:
- §3.F.1 Sesgo de lado: codificación estrictamente antisimétrica (Azul − Rojo) más
  contexto simétrico del ritmo de partida y un único intercepto de lado / espejado.
- §3.F.4 Truncamiento temprano: enmascaramiento estricto de partidas con
  `game_duration <= 60 * minute` (evita leer el frame final del Nexo en Riot
  Timeline-V5 cuando la partida terminó entre `m-1` y `m`).
- §3.F.5 Limpieza de posiciones y ganador único: delegada en los cargadores.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

LANES: Tuple[str, ...] = ("BOTTOM", "JUNGLE", "MIDDLE", "TOP", "UTILITY")
TIMELINE_STATS: Tuple[str, ...] = (
    "gold",
    "xp",
    "level",
    "cs",
    "damage",
    "kills",
    "deaths",
    "assists",
)
MINUTES: Tuple[int, ...] = (10, 15, 20)

ATRIB_COLS: Tuple[str, ...] = (
    "ap",
    "ps_dano",
    "ps_dur",
    "ps_cc",
    "ps_mov",
    "ps_util",
    "rango",
    "alcance",
    "front",
    "engage",
    "poke",
    "enchanter",
    "asesino",
    "tirador",
)
GROWTH_COLS: Tuple[str, ...] = (
    "hpperlevel",
    "attackdamageperlevel",
    "armorperlevel",
    "attackspeedperlevel",
)


def mask_finished_matches(
    values: np.ndarray,
    game_duration_sec: np.ndarray,
    minute: int,
) -> np.ndarray:
    """Enmascara con NaN las filas cuya partida no superó `60 * minute` segundos.

    Riot Timeline-V5 guarda `frames[m]` también cuando la partida terminó entre
    `m - 1` y `m` (frame final al caer el Nexo), lo que filtraría el resultado si
    se usara sin verificar `game_duration > 60 * minute`.
    """
    out = np.array(values, dtype=np.float32, copy=True)
    dur = np.asarray(game_duration_sec, dtype=np.float64)
    if out.shape[0] != len(dur):
        raise ValueError("La primera dimensión de values debe coincidir con game_duration_sec.")
    ended = dur <= 60.0 * float(minute)
    out[ended] = np.nan
    return out


def fit_residual_gold_coeffs(
    gold: np.ndarray,
    cs: np.ndarray,
    kills: np.ndarray,
    assists: np.ndarray,
    fit_mask: np.ndarray,
) -> np.ndarray:
    """Ajusta OLS por línea `gold ~ 1 + cs + kills + assists` solo sobre `fit_mask`.

    Devuelve array `(5, 4)` con coeficientes `[intercept, cs, kills, assists]`
    por carril. No usa el target `win`, pero restringe a `fit_mask` para mantener
    causalidad estricta.
    """
    coeffs = np.zeros((5, 4), dtype=np.float64)
    mask_m = np.asarray(fit_mask, dtype=bool)
    for l in range(5):
        g_l = gold[mask_m, :, l].reshape(-1)
        c_l = cs[mask_m, :, l].reshape(-1)
        k_l = kills[mask_m, :, l].reshape(-1)
        a_l = assists[mask_m, :, l].reshape(-1)
        ok = np.isfinite(g_l) & np.isfinite(c_l) & np.isfinite(k_l) & np.isfinite(a_l)
        if ok.sum() < 20:
            continue
        X = np.column_stack([np.ones(ok.sum()), c_l[ok], k_l[ok], a_l[ok]])
        beta, _, _, _ = np.linalg.lstsq(X, g_l[ok].astype(np.float64), rcond=None)
        coeffs[l] = beta
    return coeffs.astype(np.float32)


def apply_residual_gold(
    gold: np.ndarray,
    cs: np.ndarray,
    kills: np.ndarray,
    assists: np.ndarray,
    coeffs: np.ndarray,
) -> np.ndarray:
    """Calcula el oro residual `gold - (b0 + b1*cs + b2*kills + b3*assists)` por slot.

    Mide economía no explicada por CS ni kills/asistencias (placas, torres,
    bounties y objetivos globales), preservando NaNs donde falte algún snapshot.
    """
    out = np.full_like(gold, np.nan, dtype=np.float32)
    for l in range(5):
        b0, b1, b2, b3 = coeffs[l]
        pred = b0 + b1 * cs[:, :, l] + b2 * kills[:, :, l] + b3 * assists[:, :, l]
        out[:, :, l] = gold[:, :, l] - pred
    return out


def causal_champ_player_tempo(
    game_creation: np.ndarray,
    match_ids: Sequence[str],
    champ_ids: np.ndarray,
    player_ids: np.ndarray,
    dur_min: np.ndarray,
    y_blue: np.ndarray,
    d_gold_10_lane: np.ndarray,
    d_gold_15_lane: np.ndarray,
    d_gold_20_lane: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Calcula rasgos causales (< partida) de escalado de campeón y ritmo de jugador.

    Todas las métricas de una partida `m` se leen estrictamente ANTES de
    actualizar acumuladores con las 10 filas de `m`, evitando cualquier fuga
    intra-partida o hacia atrás en el tiempo.
    """
    NM = len(y_blue)
    champ_ids = np.asarray(champ_ids, dtype=np.int64).reshape(NM, 2, 5)
    player_ids = np.asarray(player_ids, dtype=str).reshape(NM, 2, 5)
    dur_min = np.asarray(dur_min, dtype=np.float64)
    y_blue = np.asarray(y_blue, dtype=np.float64)

    _, p_inv = np.unique(player_ids.reshape(-1), return_inverse=True)
    pid_mat = p_inv.reshape(NM, 2, 5).tolist()
    n_p = int(p_inv.max()) + 1 if NM > 0 else 0

    cl_pairs = champ_ids * 5 + np.arange(5)[None, None, :]
    _, cl_inv = np.unique(cl_pairs.reshape(-1), return_inverse=True)
    cl_mat = cl_inv.reshape(NM, 2, 5).tolist()
    n_cl = int(cl_inv.max()) + 1 if NM > 0 else 0

    # Acumuladores por (campeón, línea)
    cl_e_w = [0.0] * n_cl
    cl_e_n = [0.0] * n_cl
    cl_l_w = [0.0] * n_cl
    cl_l_n = [0.0] * n_cl
    cl_sl_sum = [0.0] * n_cl
    cl_sl_n = [0.0] * n_cl

    # Acumuladores por jugador
    p_e_w = [0.0] * n_p
    p_e_n = [0.0] * n_p
    p_l_w = [0.0] * n_p
    p_l_n = [0.0] * n_p
    p_g15_s1 = [0.0] * n_p
    p_g15_s2 = [0.0] * n_p
    p_g15_n = [0.0] * n_p

    cbias = np.zeros((NM, 2, 5), dtype=np.float32)
    cslope = np.zeros((NM, 2, 5), dtype=np.float32)
    pbias = np.zeros((NM, 2, 5), dtype=np.float32)
    pvol = np.zeros((NM, 2, 5), dtype=np.float32)

    prior_var_g15 = 1.6  # orden de magnitud de var(Δoro@15 en miles) por carril
    k_cl, k_p, k_v = 30.0, 15.0, 10.0

    # Orden estricto cronológico (desempate determinista por match_id)
    order = np.lexsort((np.asarray(match_ids, dtype=str), np.asarray(game_creation, dtype=np.int64)))

    g10_l = d_gold_10_lane.tolist()
    g15_l = d_gold_15_lane.tolist()
    g20_l = d_gold_20_lane.tolist()

    for m in order:
        dm = float(dur_min[m])
        yb = float(y_blue[m])
        is_early = dm < 25.0
        is_late = dm > 30.0

        # 1) Lectura estrictamente previa (< m)
        for s in (0, 1):
            for l in range(5):
                cl = cl_mat[m][s][l]
                p = pid_mat[m][s][l]
                ew_c = (cl_l_w[cl] + 0.5 * k_cl) / (cl_l_n[cl] + k_cl) - (cl_e_w[cl] + 0.5 * k_cl) / (cl_e_n[cl] + k_cl)
                cbias[m, s, l] = ew_c
                cslope[m, s, l] = cl_sl_sum[cl] / (cl_sl_n[cl] + k_cl)

                ew_p = (p_l_w[p] + 0.5 * k_p) / (p_l_n[p] + k_p) - (p_e_w[p] + 0.5 * k_p) / (p_e_n[p] + k_p)
                pbias[m, s, l] = ew_p

                nv = p_g15_n[p]
                mean_v = p_g15_s1[p] / (nv + k_v)
                var_v = max(0.0, (p_g15_s2[p] + k_v * prior_var_g15) / (nv + k_v) - mean_v * mean_v)
                pvol[m, s, l] = math.sqrt(var_v)

        # 2) Actualización posterior a la partida
        for s in (0, 1):
            w_s = yb if s == 0 else (1.0 - yb)
            sgn = 1.0 if s == 0 else -1.0
            for l in range(5):
                cl = cl_mat[m][s][l]
                p = pid_mat[m][s][l]
                if is_early:
                    cl_e_w[cl] += w_s
                    cl_e_n[cl] += 1.0
                    p_e_w[p] += w_s
                    p_e_n[p] += 1.0
                elif is_late:
                    cl_l_w[cl] += w_s
                    cl_l_n[cl] += 1.0
                    p_l_w[p] += w_s
                    p_l_n[p] += 1.0

                g10_v = g10_l[m][l]
                g20_v = g20_l[m][l]
                if math.isfinite(g10_v) and math.isfinite(g20_v):
                    cl_sl_sum[cl] += sgn * (g20_v - g10_v)
                    cl_sl_n[cl] += 1.0

                g15_v = g15_l[m][l]
                if math.isfinite(g15_v):
                    val = sgn * g15_v
                    p_g15_s1[p] += val
                    p_g15_s2[p] += val * val
                    p_g15_n[p] += 1.0

    return {
        "cbias": cbias,
        "cslope": cslope,
        "pbias": pbias,
        "pvol": pvol,
    }


def build_composition_features(
    champ_ids: np.ndarray,
    atrib_df: Optional[pd.DataFrame],
    ddragon_df: Optional[pd.DataFrame],
) -> Dict[str, np.ndarray]:
    """Construye rasgos de composición y crecimiento por equipo y por slot.

    Todas las entradas son atributos estáticos de campeón (zero leakage).
    """
    NM = champ_ids.shape[0]
    ids = np.asarray(champ_ids, dtype=np.int64).reshape(NM, 10)
    uniq = np.unique(ids)
    cmap = {int(c): i for i, c in enumerate(uniq)}
    cidx = np.vectorize(cmap.__getitem__, otypes=[np.int64])(ids)

    # 1. Atributos tácticos (14 cols)
    if atrib_df is not None:
        A = atrib_df.set_index("champion_id")[list(ATRIB_COLS)].astype(float)
        A = A.fillna(A.mean()).reindex(uniq).fillna(A.mean())
    else:
        A = pd.DataFrame(0.0, index=uniq, columns=list(ATRIB_COLS))
    a_arr = A.to_numpy(np.float32)[cidx].reshape(NM, 2, 5, len(ATRIB_COLS))

    # 2. Crecimiento por nivel de Data Dragon (4 cols)
    if ddragon_df is not None:
        G = ddragon_df.set_index("champion_id")[list(GROWTH_COLS)].astype(float)
        G = G.fillna(G.mean()).reindex(uniq).fillna(G.mean())
    else:
        G = pd.DataFrame(0.0, index=uniq, columns=list(GROWTH_COLS))
    g_raw = G.to_numpy(np.float32)
    g_norm = (g_raw - g_raw.mean(axis=0)) / (g_raw.std(axis=0) + 1e-6)
    g_arr = g_norm[cidx].reshape(NM, 2, 5, len(GROWTH_COLS))

    # Tensor estático estandarizado por slot (NM, 2, 5, 18)
    a_raw = A.to_numpy(np.float32)
    a_norm = (a_raw - a_raw.mean(axis=0)) / (a_raw.std(axis=0) + 1e-6)
    slot_static = np.concatenate(
        [a_norm[cidx].reshape(NM, 2, 5, len(ATRIB_COLS)), g_arr],
        axis=-1,
    ).astype(np.float32)

    col_i = {c: i for i, c in enumerate(ATRIB_COLS)}

    def team_stats(E: np.ndarray, Gr: np.ndarray) -> Dict[str, np.ndarray]:
        ap = E[:, :, col_i["ap"]].sum(1)
        dur = E[:, :, col_i["ps_dur"]].sum(1)
        cc = E[:, :, col_i["ps_cc"]].sum(1)
        mov = E[:, :, col_i["ps_mov"]].sum(1)
        util = E[:, :, col_i["ps_util"]].sum(1)
        dano = E[:, :, col_i["ps_dano"]].sum(1)
        front = E[:, :, col_i["front"]].sum(1)
        engage = E[:, :, col_i["engage"]].sum(1)
        rango = E[:, :, col_i["rango"]].sum(1)
        alcance = E[:, :, col_i["alcance"]].mean(1) / 100.0
        poke = E[:, :, col_i["poke"]].sum(1)
        ench = E[:, :, col_i["enchanter"]].sum(1)
        ases = E[:, :, col_i["asesino"]].sum(1)
        tir = E[:, :, col_i["tirador"]].sum(1)
        tank_score = dur + 2.0 * front
        return {
            "n_ap": ap,
            "todo_ad": (ap <= 0.5).astype(np.float32),
            "todo_ap": (ap >= 4.0).astype(np.float32),
            "desb_dano": np.abs(ap - 2.5),
            "dur": dur,
            "cc": cc,
            "mov": mov,
            "util": util,
            "dano": dano,
            "n_front": front,
            "sin_front": (front == 0).astype(np.float32),
            "n_engage": engage,
            "sin_engage": (engage == 0).astype(np.float32),
            "n_rango": rango,
            "alcance": alcance,
            "n_poke": poke,
            "n_enchanter": ench,
            "n_asesino": ases,
            "n_tirador": tir,
            "sin_cc": (cc <= 3.0).astype(np.float32),
            "poco_dur": (dur <= 5.0).astype(np.float32),
            "tank_score": tank_score,
            "hp_grow": Gr[:, :, 0].mean(1),
            "ad_grow": Gr[:, :, 1].mean(1),
            "arm_grow": Gr[:, :, 2].mean(1),
            "as_grow": Gr[:, :, 3].mean(1),
        }

    az = team_stats(a_arr[:, 0], g_arr[:, 0])
    ro = team_stats(a_arr[:, 1], g_arr[:, 1])
    team_keys = list(az.keys())

    comp_anti_base = np.column_stack([az[k] - ro[k] for k in team_keys]).astype(np.float32)

    # Vulnerabilidades direccionales por equipo (para cruzar luego con Δoro_m)
    vuln = {
        "full_ad_vs_tank": (
            np.maximum(0.0, 1.3 - az["n_ap"]) * ro["tank_score"],
            np.maximum(0.0, 1.3 - ro["n_ap"]) * az["tank_score"],
        ),
        "poke_vs_sin_engage": (
            az["n_poke"] * ro["sin_engage"],
            ro["n_poke"] * az["sin_engage"],
        ),
        "asesino_vs_poco_dur": (
            az["n_asesino"] * ro["poco_dur"],
            ro["n_asesino"] * az["poco_dur"],
        ),
        "engage_vs_rango": (
            az["n_engage"] * ro["n_rango"],
            ro["n_engage"] * az["n_rango"],
        ),
        "hypercarry_comp": (
            az["n_tirador"] + az["n_enchanter"] + 0.5 * az["n_front"],
            ro["n_tirador"] + ro["n_enchanter"] + 0.5 * ro["n_front"],
        ),
    }
    cruces_base = np.column_stack([b - r for b, r in vuln.values()]).astype(np.float32)
    comp_anti = np.concatenate([comp_anti_base, cruces_base], axis=1)
    names_anti = [f"comp_{k}" for k in team_keys] + [f"comp_{k}" for k in vuln]

    sym_keys = ("n_ap", "dur", "cc", "n_front", "n_engage", "n_poke", "n_asesino", "n_tirador", "hp_grow", "as_grow")
    comp_sym = np.column_stack([az[k] + ro[k] for k in sym_keys]).astype(np.float32)
    names_sym = [f"sym_comp_{k}" for k in sym_keys]

    return {
        "comp_anti": comp_anti,
        "names_anti": names_anti,
        "comp_sym": comp_sym,
        "names_sym": names_sym,
        "vuln_blue": np.column_stack([b for b, _ in vuln.values()]).astype(np.float32),
        "vuln_red": np.column_stack([r for _, r in vuln.values()]).astype(np.float32),
        "vuln_names": list(vuln.keys()),
        "slot_static": slot_static,
    }


def _sigm(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))


def _antisym_2v2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Interacción 2v2 estrictamente antisimétrica: f(-a, -b) = -f(a, b).

    Mide sinergia superlineal cuando ambos carriles van en la misma dirección
    (`sign(a+b)*(a+b)^2 - sign(a)*a^2 - sign(b)*b^2`), más producto modulado.
    """
    s = a + b
    sup = np.sign(s) * (s * s) - np.sign(a) * (a * a) - np.sign(b) * (b * b)
    return sup.astype(np.float32)


def extract_minute_features(
    minute: int,
    L_sym: np.ndarray,
    tl_raw: np.ndarray,
    gold_resid: np.ndarray,
    exp_lane: Optional[Dict[str, np.ndarray]] = None,
    exp_trans: Optional[Dict[str, np.ndarray]] = None,
    comp: Optional[Dict[str, np.ndarray]] = None,
    tempo: Optional[Dict[str, np.ndarray]] = None,
    player_pre: Optional[Dict[str, np.ndarray]] = None,
    include_all_stats: bool = True,
    include_traj: bool = True,
    include_expected: bool = True,
    include_comp_cross: bool = True,
    include_player: bool = True,
) -> Dict[str, np.ndarray]:
    """Arma las matrices `(X_anti, X_sym, lane_b, lane_r)` para el minuto `minute`.

    `tl_raw` tiene forma `(NM, 2, 5, 3, 8)` con orden de minutos `(10, 15, 20)`
    y estadísticas `TIMELINE_STATS`, ya escalado (oro, XP y daño en miles) y con
    partidas terminadas (`dur <= 60 * m`) enmascaradas a NaN.

    Garantía: bajo intercambio Azul <-> Rojo y `L_sym -> -L_sym`, `X_anti` cambia
    de signo exactamente, `X_sym` permanece invariante y `lane_b <-> lane_r` se
    intercambian.
    """
    if minute not in MINUTES:
        raise ValueError(f"minute debe estar en {MINUTES}, recibido {minute}.")
    mi = MINUTES.index(minute)
    NM = tl_raw.shape[0]
    stat_idx = {s: i for i, s in enumerate(TIMELINE_STATS)}

    # Diferencias por línea (NM, 5, 8) y totales por equipo (NM, 2, 8)
    cur = tl_raw[:, :, :, mi, :]  # (NM, 2, 5, 8)
    d_lane = cur[:, 0] - cur[:, 1]  # (NM, 5, 8)
    tot_team = cur.sum(axis=2)      # (NM, 2, 8)
    d_team = tot_team[:, 0] - tot_team[:, 1]  # (NM, 8)
    s_team = tot_team[:, 0] + tot_team[:, 1]  # (NM, 8)

    g_l = d_lane[:, :, stat_idx["gold"]]
    x_l = d_lane[:, :, stat_idx["xp"]]
    g_tot = d_team[:, stat_idx["gold"]]
    k_tot = d_team[:, stat_idx["kills"]]

    anti_parts: List[np.ndarray] = [L_sym[:, None], g_l, x_l, k_tot[:, None]]
    anti_names: List[str] = (
        ["L_sym"]
        + [f"dg{minute}_{l}" for l in LANES]
        + [f"dxp{minute}_{l}" for l in LANES]
        + [f"dkills{minute}_tot"]
    )

    sym_parts: List[np.ndarray] = [
        np.abs(L_sym)[:, None],
        np.abs(g_tot)[:, None],
        np.abs(d_team[:, stat_idx["xp"]])[:, None],
        s_team[:, stat_idx["kills"]][:, None],
        s_team[:, stat_idx["damage"]][:, None],
        s_team[:, stat_idx["gold"]][:, None],
        s_team[:, stat_idx["deaths"]][:, None],
        d_lane[:, :, stat_idx["gold"]].std(axis=1)[:, None],
    ]
    sym_names: List[str] = [
        "abs_L_sym",
        f"abs_dg{minute}_tot",
        f"abs_dxp{minute}_tot",
        f"sum_kills{minute}",
        f"sum_dmg{minute}",
        f"sum_gold{minute}",
        f"sum_deaths{minute}",
        f"std_dg{minute}_lanes",
    ]

    # Canales por línea y lado para InGameNet: (NM, 5, d_lane)
    lb_parts: List[np.ndarray] = [cur[:, 0]]
    lr_parts: List[np.ndarray] = [cur[:, 1]]

    if include_all_stats:
        for sname in ("level", "cs", "damage", "kills", "deaths", "assists"):
            si = stat_idx[sname]
            anti_parts.append(d_lane[:, :, si])
            anti_names += [f"d{sname}{minute}_{l}" for l in LANES]

        # Oro residual (proxy placas/torres/bounties)
        gr_cur = gold_resid[:, :, :, mi]  # (NM, 2, 5)
        d_gr = gr_cur[:, 0] - gr_cur[:, 1]
        anti_parts.append(d_gr)
        anti_names += [f"dgresid{minute}_{l}" for l in LANES]
        lb_parts.append(gr_cur[:, 0, :, None])
        lr_parts.append(gr_cur[:, 1, :, None])

        # Umbrales de nivel (power spikes)
        lvl_b = cur[:, 0, :, stat_idx["level"]]
        lvl_r = cur[:, 1, :, stat_idx["level"]]
        thrs = (6, 8) if minute == 10 else ((11, 13) if minute == 15 else (11, 13, 16))
        for thr in thrs:
            cnt_d = (lvl_b >= thr).sum(axis=1) - (lvl_r >= thr).sum(axis=1)
            anti_parts.append(cnt_d[:, None].astype(np.float32))
            anti_names.append(f"d_lvl_ge{thr}_{minute}")

        # Agregados de equipo y concentración (max+min preserva antisimetría exacta)
        for sname in ("gold", "xp", "damage", "kills", "deaths", "assists"):
            si = stat_idx[sname]
            arr = d_lane[:, :, si]
            anti_parts.append(
                np.column_stack([
                    arr.sum(axis=1),
                    arr.max(axis=1) + arr.min(axis=1),
                    (arr[:, 0] + arr[:, 1] + arr[:, 2]) / 3.0,  # carries: BOT+JG+MID
                ]).astype(np.float32)
            )
            anti_names += [f"d{sname}{minute}_sum", f"d{sname}{minute}_extr", f"d{sname}{minute}_carries"]

        dgr_sum = d_gr.sum(axis=1)
        dgr_ext = d_gr.max(axis=1) + d_gr.min(axis=1)
        # Coordinación asistencias por kill
        apk_b = tot_team[:, 0, stat_idx["assists"]] / np.maximum(1.0, tot_team[:, 0, stat_idx["kills"]])
        apk_r = tot_team[:, 1, stat_idx["assists"]] / np.maximum(1.0, tot_team[:, 1, stat_idx["kills"]])
        anti_parts.append(np.column_stack([dgr_sum, dgr_ext, apk_b - apk_r]).astype(np.float32))
        anti_names += [f"dgresid{minute}_sum", f"dgresid{minute}_extr", f"d_apk{minute}"]

        sym_parts.append(
            np.column_stack([
                np.abs(d_lane[:, :, stat_idx["damage"]]).mean(axis=1),
                np.abs(d_lane[:, :, stat_idx["kills"]]).max(axis=1),
                np.abs(d_gr.sum(axis=1)),
            ]).astype(np.float32)
        )
        sym_names += [f"abs_ddmg{minute}_mean", f"max_abs_dkills{minute}", f"abs_dgresid{minute}"]

    # Trayectoria (solo minutos anteriores <= m!)
    if include_traj and mi >= 1:
        prev10 = tl_raw[:, :, :, 0, :]
        d10_lane = prev10[:, 0] - prev10[:, 1]
        step_15_10 = (tl_raw[:, :, :, 1, :] - tl_raw[:, :, :, 0, :])
        d15_10 = step_15_10[:, 0] - step_15_10[:, 1]  # (NM, 5, 8)
        for sname in ("gold", "xp", "cs", "damage", "kills"):
            si = stat_idx[sname]
            anti_parts.append(d15_10[:, :, si])
            anti_names += [f"vel15_10_{sname}_{l}" for l in LANES]
            anti_parts.append(d15_10[:, :, si].sum(axis=1)[:, None])
            anti_names.append(f"vel15_10_{sname}_tot")
        g10_tot = d10_lane[:, :, stat_idx["gold"]].sum(axis=1)
        g15_tot = (tl_raw[:, 0, :, 1, stat_idx["gold"]] - tl_raw[:, 1, :, 1, stat_idx["gold"]]).sum(axis=1)
        flip_10_15 = np.where(np.sign(g15_tot) != np.sign(g10_tot), np.sign(g15_tot), 0.0).astype(np.float32)
        mom_ratio_15 = ((g15_tot - g10_tot) / (np.abs(g10_tot) + 0.5)).astype(np.float32)
        anti_parts.append(np.column_stack([flip_10_15, mom_ratio_15]))
        anti_names += ["flip_gold_10_15", "mom_ratio_10_15"]
        lb_parts.append(step_15_10[:, 0, :, :5])  # gold, xp, level, cs, damage deltas
        lr_parts.append(step_15_10[:, 1, :, :5])

        if mi == 2:
            step_20_15 = tl_raw[:, :, :, 2, :] - tl_raw[:, :, :, 1, :]
            d20_15 = step_20_15[:, 0] - step_20_15[:, 1]
            d20_10 = (tl_raw[:, 0, :, 2, :] - tl_raw[:, 1, :, 2, :]) - d10_lane
            for sname in ("gold", "xp", "cs", "damage", "kills"):
                si = stat_idx[sname]
                anti_parts.append(d20_15[:, :, si])
                anti_names += [f"vel20_15_{sname}_{l}" for l in LANES]
                anti_parts.append(d20_15[:, :, si].sum(axis=1)[:, None])
                anti_names.append(f"vel20_15_{sname}_tot")
            for sname in ("gold", "xp", "damage"):
                si = stat_idx[sname]
                anti_parts.append(d20_10[:, :, si].sum(axis=1)[:, None])
                anti_names.append(f"trend20_10_{sname}_tot")
            acc_g = (d20_15[:, :, stat_idx["gold"]].sum(axis=1) - d15_10[:, :, stat_idx["gold"]].sum(axis=1))
            flip_15_20 = np.where(np.sign(g_tot) != np.sign(g15_tot), np.sign(g_tot), 0.0).astype(np.float32)
            # Drawdown respecto al pico previo: si iba +4k y cayó a +1.5k, -2.5k
            peak_prev = np.where(np.abs(g15_tot) >= np.abs(g10_tot), g15_tot, g10_tot)
            drawdown = (g_tot - peak_prev).astype(np.float32)
            anti_parts.append(np.column_stack([acc_g, flip_15_20, drawdown]))
            anti_names += ["acc_gold_20", "flip_gold_15_20", "drawdown_gold_20"]
            lb_parts.append(step_20_15[:, 0, :, :5])
            lr_parts.append(step_20_15[:, 1, :, :5])

    # Valores esperados por línea ("sorpresa" = real - esperado causal)
    if include_expected and exp_lane is not None:
        for sname in ("gold", "xp", "cs", "damage"):
            key = f"{sname}_{minute}"
            if key not in exp_lane:
                continue
            obs_l = d_lane[:, :, stat_idx[sname]]
            exp_l = exp_lane[key]
            surp_l = obs_l - exp_l
            anti_parts.extend([surp_l, surp_l.sum(axis=1)[:, None]])
            anti_names += [f"surp_{key}_{l}" for l in LANES] + [f"surp_{key}_tot"]
            lb_parts.append(np.stack([0.5 * exp_l, 0.5 * surp_l], axis=-1))
            lr_parts.append(np.stack([-0.5 * exp_l, -0.5 * surp_l], axis=-1))

        # Además oro @15 esperado sirve al minuto 10 (como en brainstorm_40)
        if minute == 10 and "gold_15" in exp_lane:
            exp15 = exp_lane["gold_15"]
            anti_parts.extend([g_l - exp15, exp15.sum(axis=1)[:, None]])
            anti_names += [f"surp10vs15_gold_{l}" for l in LANES] + ["exp15_gold_tot"]

        if exp_trans is not None and f"trans_gold_{minute}" in exp_trans:
            tr_exp = exp_trans[f"trans_gold_{minute}"]  # (NM, 5)
            tr_surp = g_l - tr_exp
            anti_parts.extend([tr_surp, tr_surp.sum(axis=1)[:, None]])
            anti_names += [f"tr_surp_gold{minute}_{l}" for l in LANES] + [f"tr_surp_gold{minute}_tot"]

    # Composición, escalado y cruces
    if include_comp_cross and comp is not None:
        anti_parts.append(comp["comp_anti"])
        anti_names += comp["names_anti"]
        sym_parts.append(comp["comp_sym"])
        sym_names += comp["names_sym"]

        if tempo is not None:
            cb_d = tempo["cbias"][:, 0] - tempo["cbias"][:, 1]  # (NM, 5)
            cs_d = tempo["cslope"][:, 0] - tempo["cslope"][:, 1]
            cb_tot = cb_d.sum(axis=1)
            cs_tot = cs_d.sum(axis=1)
            cb_sym = (tempo["cbias"][:, 0] + tempo["cbias"][:, 1]).sum(axis=1)
            anti_parts.append(np.column_stack([cb_d, cb_tot, cs_tot]).astype(np.float32))
            anti_names += [f"cbias_{l}" for l in LANES] + ["cbias_tot", "cslope_tot"]
            sym_parts.append(cb_sym[:, None].astype(np.float32))
            sym_names.append("sym_cbias_tot")
            lb_parts.append(np.stack([tempo["cbias"][:, 0], tempo["cslope"][:, 0]], axis=-1))
            lr_parts.append(np.stack([tempo["cbias"][:, 1], tempo["cslope"][:, 1]], axis=-1))

            # Cruces escalado × ventaja de oro (estrictamente antisimétricos)
            close_gate = np.exp(-np.abs(g_tot) / 3.0).astype(np.float32)
            abs_tanh_g = np.abs(np.tanh(g_tot / 3.0)).astype(np.float32)
            cb_b_tot = tempo["cbias"][:, 0].sum(axis=1)
            cb_r_tot = tempo["cbias"][:, 1].sum(axis=1)
            cb_behind = (cb_b_tot * _sigm(-g_tot / 3.0) - cb_r_tot * _sigm(+g_tot / 3.0)).astype(np.float32)
            anti_parts.append(
                np.column_stack([
                    cb_tot * close_gate,
                    cb_tot * abs_tanh_g,
                    cb_behind,
                    cs_tot * close_gate,
                ]).astype(np.float32)
            )
            anti_names += ["cruce_cbias_close", "cruce_cbias_absg", "cruce_cbias_behind", "cruce_cslope_close"]

        if "slot_static" in comp:
            lb_parts.append(comp["slot_static"][:, 0])
            lr_parts.append(comp["slot_static"][:, 1])

        # Cruces ritmo de partida × ventaja (siempre antisimétricos!)
        tot_k_norm = ((s_team[:, stat_idx["kills"]] - 6.0 * (minute / 10.0)) / 8.0).astype(np.float32)
        tot_d_norm = ((s_team[:, stat_idx["damage"]] - 30.0 * (minute / 10.0)) / 25.0).astype(np.float32)
        anti_parts.append(
            np.column_stack([
                g_tot * tot_k_norm,
                g_tot * tot_d_norm,
                L_sym * tot_k_norm,
                g_tot / (1.0 + np.log1p(np.exp(tot_k_norm))),
            ]).astype(np.float32)
        )
        anti_names += ["cruce_g_pace_k", "cruce_g_pace_dmg", "cruce_L_pace_k", "cruce_g_mod_k"]

        # Cruces vulnerabilidad de composición × estado de oro (exactamente antisimétricos)
        vb, vr = comp["vuln_blue"], comp["vuln_red"]
        vnames = comp["vuln_names"]
        sig_neg = _sigm(-g_tot / 2.5)[:, None]
        sig_pos = _sigm(+g_tot / 2.5)[:, None]
        # full_ad castiga al que va atrás; poke/asesino aprovechan al ir adelante
        for vi, vname in enumerate(vnames):
            if vname == "full_ad_vs_tank":
                cr_v = vb[:, vi : vi + 1] * sig_neg - vr[:, vi : vi + 1] * sig_pos
            elif vname == "hypercarry_comp":
                s_bn = _sigm(-g_l[:, 0:1] / 1.5)
                s_bp = _sigm(+g_l[:, 0:1] / 1.5)
                cr_v = vb[:, vi : vi + 1] * s_bp - vr[:, vi : vi + 1] * s_bn
            else:
                cr_v = vb[:, vi : vi + 1] * sig_pos - vr[:, vi : vi + 1] * sig_neg
            anti_parts.append(cr_v.astype(np.float32))
            anti_names.append(f"cruce_{vname}_estado")

        # Cruces entre carriles (2v2 bot+sup, mid+jg, top+jg) estrictamente antisimétricos
        anti_parts.append(
            np.column_stack([
                _antisym_2v2(g_l[:, 0], g_l[:, 4]),  # BOT + SUP
                _antisym_2v2(g_l[:, 2], g_l[:, 1]),  # MID + JG
                _antisym_2v2(g_l[:, 3], g_l[:, 1]),  # TOP + JG
            ]).astype(np.float32)
        )
        anti_names += ["cruce_2v2_botsup", "cruce_2v2_midjg", "cruce_2v2_topjg"]

    # Rasgos de jugadores in-game y cruces jugador × estado
    if include_player and player_pre is not None:
        # `p_mat`: (NM, 2, 5, d_p)
        p_mat = player_pre["p_mat"]
        p_cols = player_pre["p_cols"]
        pd_lane = p_mat[:, 0] - p_mat[:, 1]  # (NM, 5, d_p)
        p_idx = {c: i for i, c in enumerate(p_cols)}

        # Resumen por equipo y carril de variables clave de jugador
        anti_parts.append(pd_lane.mean(axis=1))
        anti_names += [f"jug_eq_{c}" for c in p_cols]
        for key_col in ("acum_d_gold_15", "v20_d_gold_15", "wr_acum", "estreno", "offrole"):
            if key_col in p_idx:
                ci = p_idx[key_col]
                anti_parts.append(pd_lane[:, :, ci])
                anti_names += [f"jug_{key_col}_{l}" for l in LANES]

        lb_parts.append(p_mat[:, 0])
        lr_parts.append(p_mat[:, 1])

        # Cruces jugador × estado de partida (todos estrictamente antisimétricos!)
        if "estreno" in p_idx:
            ci = p_idx["estreno"]
            eb, er = p_mat[:, 0, :, ci], p_mat[:, 1, :, ci]
            c_est = (eb * _sigm(g_l / 1.0) - er * _sigm(-g_l / 1.0)).sum(axis=1)
            anti_parts.append(c_est[:, None].astype(np.float32))
            anti_names.append("cruce_jug_estreno_lead")

        if "offrole" in p_idx:
            ci = p_idx["offrole"]
            ob, or_ = p_mat[:, 0, :, ci], p_mat[:, 1, :, ci]
            c_off = (ob * _sigm(g_l / 1.0) - or_ * _sigm(-g_l / 1.0)).sum(axis=1)
            anti_parts.append(c_off[:, None].astype(np.float32))
            anti_names.append("cruce_jug_offrole_lead")

        if "acum_d_gold_15" in p_idx:
            ci = p_idx["acum_d_gold_15"]
            sk = pd_lane[:, :, ci]
            c_sk_close = (sk * np.exp(-np.abs(g_l) / 1.5)).sum(axis=1)
            c_sk_abs = (sk * np.abs(g_l)).sum(axis=1)
            anti_parts.append(np.column_stack([c_sk_close, c_sk_abs]).astype(np.float32))
            anti_names += ["cruce_jug_skill_close", "cruce_jug_skill_absg"]

        if tempo is not None:
            pb_d = tempo["pbias"][:, 0] - tempo["pbias"][:, 1]
            pv_d = tempo["pvol"][:, 0] - tempo["pvol"][:, 1]
            pb_tot = pb_d.sum(axis=1)
            pv_tot = pv_d.sum(axis=1)
            tot_k_norm = ((s_team[:, stat_idx["kills"]] - 6.0 * (minute / 10.0)) / 8.0).astype(np.float32)
            anti_parts.append(
                np.column_stack([
                    pb_tot,
                    pv_tot,
                    pb_tot * np.exp(-np.abs(g_tot) / 3.0),
                    pv_tot * tot_k_norm,
                ]).astype(np.float32)
            )
            anti_names += ["jug_pbias_tot", "jug_pvol_tot", "cruce_jug_pbias_close", "cruce_jug_pvol_pace"]
            lb_parts.append(np.stack([tempo["pbias"][:, 0], tempo["pvol"][:, 0]], axis=-1))
            lr_parts.append(np.stack([tempo["pbias"][:, 1], tempo["pvol"][:, 1]], axis=-1))

        if "acum_deaths_pm" in p_idx:
            ci = p_idx["acum_deaths_pm"]
            dp_tot = pd_lane[:, :, ci].sum(axis=1)
            tot_k_norm = ((s_team[:, stat_idx["kills"]] - 6.0 * (minute / 10.0)) / 8.0).astype(np.float32)
            anti_parts.append((dp_tot * tot_k_norm)[:, None].astype(np.float32))
            anti_names.append("cruce_jug_deaths_pace")

    X_anti = np.concatenate(anti_parts, axis=1).astype(np.float32)
    X_sym = np.concatenate(sym_parts, axis=1).astype(np.float32)
    lane_b = np.concatenate(lb_parts, axis=-1).astype(np.float32)
    lane_r = np.concatenate(lr_parts, axis=-1).astype(np.float32)
    return {
        "X_anti": X_anti,
        "X_sym": X_sym,
        "lane_b": lane_b,
        "lane_r": lane_r,
        "names_anti": anti_names,
        "names_sym": sym_names,
    }


class InGameNet(nn.Module):
    """Red neuronal antisimétrica por carril + modulación FiLM + cabezas auxiliares.

    Propiedad de antisimetría exacta:
      logit(B, R) - b_side == -(logit(R, B) - b_side)
    donde `b_side` es un único sesgo escalar para la ventaja estructural Rojo/Azul
    (§3.F.1). El contexto simétrico (`x_sym`: ritmo total, magnitud de ventaja,
    composición agregada) modula multiplicativamente (FiLM) los canales
    antisimétricos sin romper la invariancia de intercambio.
    """

    def __init__(
        self,
        d_lane: int,
        d_anti: int,
        d_sym: int,
        d_aux_anti: int = 0,
        d_aux_sym: int = 0,
        lane_hidden: int = 24,
        trunk_hidden: int = 96,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.lane_emb = nn.Embedding(5, 8)
        self.lane_enc = nn.Sequential(
            nn.Linear(2 * d_lane + 8, 48),
            nn.LayerNorm(48),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(48, lane_hidden),
            nn.SiLU(),
        )
        # 5 carriles individuales + sum + (max+min) = 7 * lane_hidden
        d_pool = 7 * lane_hidden
        self.anti_proj = nn.Sequential(
            nn.Linear(d_pool + d_anti, trunk_hidden),
            nn.LayerNorm(trunk_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.sym_net = nn.Sequential(
            nn.Linear(d_sym, 48),
            nn.LayerNorm(48),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(48, trunk_hidden),
        )
        self.head_mlp = nn.Sequential(
            nn.Linear(trunk_hidden, 48),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(48, 1, bias=False),
        )
        self.skip_anti = nn.Linear(d_anti, 1, bias=False)
        self.b_side = nn.Parameter(torch.tensor(0.0))

        self.aux_anti_head = nn.Linear(trunk_hidden, d_aux_anti, bias=False) if d_aux_anti > 0 else None
        self.aux_sym_head = nn.Linear(trunk_hidden, d_aux_sym) if d_aux_sym > 0 else None

    def _encode_anti(self, x_lb: torch.Tensor, x_lr: torch.Tensor, x_anti: torch.Tensor, x_sym: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x_lb.shape[0]
        idx = torch.arange(5, device=x_lb.device).unsqueeze(0).expand(B, 5)
        e = self.lane_emb(idx)
        h_br = self.lane_enc(torch.cat([x_lb, x_lr, e], dim=-1))
        h_rb = self.lane_enc(torch.cat([x_lr, x_lb, e], dim=-1))
        h_lane = h_br - h_rb  # (B, 5, lane_hidden) exactamente antisimétrico

        h_sum = h_lane.sum(dim=1)
        h_ext = h_lane.max(dim=1).values + h_lane.min(dim=1).values  # antisimétrico!
        h_flat = h_lane.reshape(B, -1)
        h_comb = torch.cat([h_flat, h_sum, h_ext, x_anti], dim=-1)

        # Para garantizar f(-h) = -f(h) en todo el tronco, aplicamos antisimetrización explícita
        z_in = self.anti_proj(h_comb) - self.anti_proj(-h_comb)
        film_scale = 1.0 + 0.5 * torch.tanh(self.sym_net(x_sym))
        h_mod = z_in * film_scale
        return h_mod, film_scale

    def forward(
        self,
        x_lb: torch.Tensor,
        x_lr: torch.Tensor,
        x_anti: torch.Tensor,
        x_sym: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        h_mod, film = self._encode_anti(x_lb, x_lr, x_anti, x_sym)
        # Salida antisimétrica explícita + sesgo único de lado
        delta_logit = (self.head_mlp(h_mod) - self.head_mlp(-h_mod)).squeeze(-1) * 0.5
        logit = delta_logit + self.skip_anti(x_anti).squeeze(-1) + self.b_side
        aux_anti = self.aux_anti_head(h_mod) if self.aux_anti_head is not None else None
        aux_sym = self.aux_sym_head(film) if self.aux_sym_head is not None else None
        return logit, aux_anti, aux_sym
