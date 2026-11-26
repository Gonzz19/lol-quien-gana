"""Ideas 6, 7, 21, 22, 23 del brainstorming (docs/BRAINSTORM_NOCHE.md):
Nuevas señales causales para Master+ sobre Escalón 3 (`extras_referencia(D)`, V 0,6663 / 62,0 %):

1. `ROL_WR` (Idea 21 — ausente de las 85 cols de variables.npz y de MC/ML):
   - Winrate causal previo por `(player, lane)` (`wr_lane`),
   - Caída de autofill real (`wr_lane - wr_acum`),
   - Exceso de winrate en `(player, champ, lane)` encogido jerárquicamente hacia `wr_lane` (no hacia 0,5),
   - `logn_lane`, `estreno_lane`, peor jugador del equipo (`min` wr_lane y `min` wr_lane_diff) y `(1 - share_lane) * wr_lane_diff`.
2. `MMR_V2` (Idea 22 — mejora causal de `mmr_historico`):
   - `niv_ewma18` y `niv_fast6` encogido hacia `niv_ewma18` (corrige el sesgo de `nivel_v10` que encoge 33 % hacia la media global `mu0`),
   - Imputación causal de jugadores SIN foto 1 en `L2(p, m)` usando su propio `niv_ewma12(< t)`,
   - Velocidad de escalada (`niv_fast6 - niv_ewma18`), volatilidad del nivel de lobby (`niv_std`),
   - Desajuste post-foto 1 (`CLIMB_POST`, `extra_post=True`): `(niv_ewma10 - lp_foto1)` para detectar cuentas decaídas (ej. Master 0 LP jugando lobbies GM/Chall) y smurfs post-04/09,
   - Winrate reciente (`wr_ewma18`) y ponderado por dificultad del lobby (`wr_x_niv`).
3. `SMURF_TILT` (Ideas 6 y 7 dedicadas sobre Escalón 3):
   - Smurfs / cuentas nuevas sin foto 1 (`sin_foto`, `sin_foto * wr_acum`, `lowN_highWR`, racha previa `[-5, +5]`),
   - Historial causal de early-FF (`dur<=960s & pierde`), heavy-int (`deaths/min >= 0.42`), bajo daño no-support y `deaths_10 >= 2` (media y peor compañero `max`).
4. `SNOW_JUG` (Timeline/perfil faltante en `Xj`):
   - `early_aggression_10`, `deaths_10` residualizadas por `(champ, lane)`, proporción de jungla (`neutral_cs / total_cs`), volatilidad de `d_gold_15`.
5. `DRAFT_REC` (Idea 23 — meta reciente de parches 16.17–16.19 sin acortar historial de jugador):
   - Puntaje OOF causal (`bloque < b` dentro de `tr`) de `Logistic(DRAFT)` con decaimiento exponencial temporal (`tau = 21 días`), preservando todas las ~150k partidas para estimar los pesos de jugador.

Solo data/splits/train. Todo 100 % causal (< partida t). n_jobs=1 (< 16 GB RAM).
"""
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.brainstorm_base import (  # noqa: E402
    ajustar_predecir,
    cargar,
    comparar,
    evaluar,
    extras_referencia,
    mostrar,
)
from src.features.padron import verificar_foto1  # noqa: E402
from src.features.margin import ORDEN, _previo, lane_diffs, lane_scale, prior_winrate  # noqa: E402
from src.features.rating import lp_absoluto  # noqa: E402

CACHE = REPO / "data" / "analysis" / "brainstorm" / "gemini_nuevas_v2.npz"
LANES = ["BOTTOM", "JUNGLE", "MIDDLE", "TOP", "UTILITY"]


def _cargar_df_ordenado(D: dict) -> pd.DataFrame:
    cols = [
        "match_id", "game_creation", "game_duration", "team_id", "lane", "player_id",
        "champion_id", "win", "kills", "deaths", "total_damage_dealt_to_champions",
        "total_minions_killed", "neutral_minions_killed", "early_aggression_10",
        "deaths_10", "gold_15",
    ]
    df = pd.read_csv(REPO / "data" / "splits" / "train" / "full_dataset.csv", usecols=cols)
    df = df[df.lane.isin(LANES)]
    df = df[df.groupby(["match_id", "team_id"]).lane.transform("nunique") == 5]
    df = df[df.groupby("match_id").lane.transform("size") == 10]
    df = df[df.groupby("match_id").win.transform("sum") == 5].copy()
    df = df.sort_values(ORDEN).reset_index(drop=True)
    assert (df["match_id"].to_numpy()[::10] == D["Z"]["mids"]).all()
    return df


def _ewma_causal_por_jugador(pid: np.ndarray, val: np.ndarray, hl_slow: float = 18.0, hl_fast: float = 6.0, mu0: float = 0.0):
    """EWMA estrictamente causal (< fila actual) por jugador, encogiendo rápido -> lento -> mu0."""
    n = len(pid)
    ew_s = np.full(n, mu0, dtype=np.float64)
    ew_f = np.full(n, mu0, dtype=np.float64)
    std_s = np.zeros(n, dtype=np.float64)
    n_obs = np.zeros(n, dtype=np.float64)

    order = np.argsort(pid, kind="mergesort")
    ps = pid[order]
    vs = val[order]
    breaks = np.flatnonzero(ps[1:] != ps[:-1]) + 1
    starts = np.r_[0, breaks]
    ends = np.r_[breaks, n]

    a_s = 1.0 - np.exp(-np.log(2.0) / hl_slow)
    a_f = 1.0 - np.exp(-np.log(2.0) / hl_fast)
    k_s, k_f = 5.0, 4.0

    for st, en in zip(starts, ends):
        num_s, den_s = k_s * mu0, k_s
        num_f, den_f = 0.0, 0.0
        mean_run, m2_run, cnt = 0.0, 0.0, 0
        for idx in range(st, en):
            cur_s = num_s / den_s
            ew_s[order[idx]] = cur_s
            ew_f[order[idx]] = (num_f + k_f * cur_s) / (den_f + k_f)
            std_s[order[idx]] = np.sqrt(m2_run / cnt) if cnt >= 2 else 0.0
            n_obs[order[idx]] = float(cnt)
            x = vs[idx]
            if np.isfinite(x):
                num_s = (1.0 - a_s) * num_s + a_s * x
                den_s = (1.0 - a_s) * den_s + a_s
                num_f = (1.0 - a_f) * num_f + a_f * x
                den_f = (1.0 - a_f) * den_f + a_f
                cnt += 1
                delta = x - mean_run
                mean_run += delta / cnt
                std_delta = x - mean_run
                m2_run += delta * std_delta
    return ew_s, ew_f, std_s, n_obs


def _streak_causal_por_jugador(pid: np.ndarray, win: np.ndarray) -> np.ndarray:
    """Racha consecutiva previa [-5, +5]: +k si ganó las últimas k, -k si perdió."""
    n = len(pid)
    streak = np.zeros(n, dtype=np.float32)
    order = np.argsort(pid, kind="mergesort")
    ps = pid[order]
    ws = win[order]
    breaks = np.flatnonzero(ps[1:] != ps[:-1]) + 1
    starts = np.r_[0, breaks]
    ends = np.r_[breaks, n]
    for st, en in zip(starts, ends):
        cur = 0
        for idx in range(st, en):
            streak[order[idx]] = float(max(-5, min(5, cur)))
            if ws[idx] > 0.5:
                cur = cur + 1 if cur > 0 else 1
            else:
                cur = cur - 1 if cur < 0 else -1
    return streak


def construir_cache(D: dict) -> dict:
    t0 = time.time()
    NM = D["NM"]
    fit_rows = np.repeat(~D["val0"], 10)  # estrictamente < 16/09 (sin diam_v ni V)
    df = _cargar_df_ordenado(D)
    pid = df["player_id"].to_numpy()
    lane = df["lane"].to_numpy()
    champ = df["champion_id"].to_numpy()
    win = df["win"].to_numpy(np.float64)

    # 1. ROL_WR (estrictamente causal por jugador, (player, lane), (player, champ, lane))
    ones = pd.Series(1.0, index=df.index)
    w_ser = pd.Series(win, index=df.index)
    wr_all = prior_winrate(df, window=20, k=10)
    n_tot = _previo(ones, df["player_id"]).to_numpy(np.float64)
    pl_key = [df["player_id"], df["lane"]]
    n_pl = _previo(ones, pl_key).to_numpy(np.float64)
    w_pl = _previo(w_ser, pl_key).to_numpy(np.float64)
    wr_acum_v = wr_all["wr_acum"].to_numpy(np.float64)
    # Encoger wr_lane hacia el winrate propio del jugador (0.5 + wr_acum), k_l=8
    wr_lane = (w_pl + 8.0 * (0.5 + wr_acum_v)) / (n_pl + 8.0) - 0.5
    wr_lane_diff = wr_lane - wr_acum_v

    pcl_key = [df["player_id"], df["champion_id"], df["lane"]]
    n_pcl = _previo(ones, pcl_key).to_numpy(np.float64)
    w_pcl = _previo(w_ser, pcl_key).to_numpy(np.float64)
    wr_champ_lane_exc = (w_pcl + 6.0 * (0.5 + wr_lane)) / (n_pcl + 6.0) - 0.5 - wr_lane

    share_lane = n_pl / np.maximum(1.0, n_tot)
    estreno_lane = (n_pl == 0).astype(np.float64)
    autofill_drop = (1.0 - share_lane) * wr_lane_diff
    autofill_flag = ((n_tot >= 15) & (share_lane < 0.20)).astype(np.float64) * np.minimum(0.0, wr_lane_diff)

    def _agg_eq(mat: np.ndarray, with_min: bool = False, with_max: bool = False) -> np.ndarray:
        a = mat.reshape(NM, 2, 5, -1)
        parts = [(a[:, 0] - a[:, 1]).mean(axis=1)]
        if with_min:
            parts.append(a[:, 0].min(axis=1) - a[:, 1].min(axis=1))
        if with_max:
            parts.append(a[:, 0].max(axis=1) - a[:, 1].max(axis=1))
        return np.hstack(parts).astype(np.float32)

    ROL_WR = np.c_[
        _agg_eq(np.c_[wr_lane, wr_lane_diff, wr_champ_lane_exc, np.log1p(n_pl), estreno_lane, autofill_drop, autofill_flag]),
        _agg_eq(np.c_[wr_lane, wr_lane_diff, wr_champ_lane_exc], with_min=True)[:, -3:],
    ].astype(np.float32)
    print(f"  ROL_WR listo ({time.time()-t0:.1f}s)", flush=True)

    # 2. MMR_V2 (Foto 1 LP + EWMA causal + 2da pasada sin foto + desajuste post-t1)
    with sqlite3.connect(f"file:{REPO / 'data' / 'lol_las.db'}?mode=ro&immutable=1", uri=True) as con:
        verificar_foto1(con)            # foto 1 congelada (§9.S.67)
        foto = pd.read_sql_query(
            "SELECT m.player_id, p.tier, p.rank, p.league_points, p.wins + p.losses AS games "
            "FROM players p JOIN player_id_map m ON p.puuid = m.puuid",
            con,
        )
    foto["z"] = [lp_absoluto(tr, rk, lp) for tr, rk, lp in zip(foto["tier"], foto["rank"], foto["league_points"])]
    foto = foto.dropna(subset=["z"]).sort_values("games").drop_duplicates("player_id", keep="last").set_index("player_id")
    x1 = df["player_id"].map(foto["z"]).to_numpy(np.float64)
    has_f1 = np.isfinite(x1)

    # Nivel promedio de los otros 9 con x1 en esta partida (para historial previo)
    x1_m = np.where(has_f1, x1, 0.0).reshape(NM, 10)
    v1_m = has_f1.astype(np.float64).reshape(NM, 10)
    otros_v1 = v1_m.sum(axis=1, keepdims=True) - v1_m
    L1 = np.where(otros_v1 >= 3.0, (x1_m.sum(axis=1, keepdims=True) - x1_m) / np.maximum(1.0, otros_v1), np.nan).reshape(-1)
    mu0 = float(np.nanmean(L1[fit_rows]))

    ew18, ew6, std_l, n_l1 = _ewma_causal_por_jugador(pid, L1, hl_slow=18.0, hl_fast=6.0, mu0=mu0)
    # 2da pasada causal: imputar a los que no tienen foto 1 con SU propio ew18 previo (< t)
    x2 = np.where(has_f1, x1, np.where(n_l1 >= 2.0, ew18, np.nan))
    x2_m = np.where(np.isfinite(x2), x2, 0.0).reshape(NM, 10)
    v2_m = np.isfinite(x2).astype(np.float64).reshape(NM, 10)
    otros_v2 = v2_m.sum(axis=1, keepdims=True) - v2_m
    L2 = np.where(otros_v2 >= 3.0, (x2_m.sum(axis=1, keepdims=True) - x2_m) / np.maximum(1.0, otros_v2), np.nan).reshape(-1)
    ew2_18, ew2_6, _, _ = _ewma_causal_por_jugador(pid, L2, hl_slow=18.0, hl_fast=6.0, mu0=mu0)

    wr_ew18, wr_ew6, _, _ = _ewma_causal_por_jugador(pid, win, hl_slow=18.0, hl_fast=6.0, mu0=0.5)
    climb_niv = (ew6 - ew18) / 100.0
    climb_niv2 = (ew2_6 - ew2_18) / 100.0
    gap_f1_post = np.where(np.repeat(D["post"], 10) & has_f1, (ew6 - x1) / 100.0, 0.0)
    # Winrate ponderado por dificultad de lobby previo (en Z-score del nivel)
    zL = np.clip((L1 - mu0) / 300.0, -1.5, 2.5)
    perf_niv = np.where(np.isfinite(L1), (win - 0.5) * (1.0 + 0.35 * zL), np.nan)
    wr_w_ew, _, _, _ = _ewma_causal_por_jugador(pid, perf_niv, hl_slow=24.0, hl_fast=8.0, mu0=0.0)
    niv_x_wr = ((ew2_18 - mu0) / 300.0) * (wr_ew18 - 0.5)

    MMR_V2 = np.c_[
        _agg_eq(
            np.c_[
                (ew18 - mu0) / 100.0,
                (ew6 - mu0) / 100.0,
                climb_niv,
                (ew2_18 - mu0) / 100.0,
                climb_niv2,
                std_l / 100.0,
                wr_ew18 - 0.5,
                wr_w_ew,
                niv_x_wr,
            ]
        ),
        _agg_eq(np.c_[(ew2_18 - mu0) / 100.0, climb_niv2], with_min=True, with_max=True)[:, 2:],
    ].astype(np.float32)
    CLIMB_POST = _agg_eq(gap_f1_post[:, None], with_max=True).astype(np.float32)
    print(f"  MMR_V2 listo ({time.time()-t0:.1f}s)", flush=True)

    # 3. SMURF_TILT (Ideas 6 y 7 dedicadas)
    sin_foto = (~has_f1).astype(np.float64)
    dur_min = np.maximum(1.0, df["game_duration"].to_numpy(np.float64) / 60.0)
    early_ff = ((df["game_duration"] <= 960) & (df["win"] == 0)).to_numpy(np.float64)
    ff_20 = ((df["game_duration"] <= 1200) & (df["win"] == 0)).to_numpy(np.float64)
    heavy_int = ((df["deaths"].to_numpy(np.float64) / dur_min) >= 0.42).astype(np.float64)
    dpm = df["total_damage_dealt_to_champions"].to_numpy(np.float64) / dur_min
    afk_low = ((dpm < 130.0) & (lane != "UTILITY")).astype(np.float64)
    die2_10 = np.where(df["deaths_10"].notna(), (df["deaths_10"].to_numpy(np.float64) >= 2.0).astype(np.float64), np.nan)

    def _prior_rate(arr: np.ndarray, k_r: float = 20.0) -> np.ndarray:
        ok = np.isfinite(arr)
        mu_f = float(np.nanmean(arr[fit_rows]))
        s = pd.Series(np.where(ok, arr, 0.0), index=df.index)
        c = pd.Series(ok.astype(np.float64), index=df.index)
        return ((_previo(s, df["player_id"]) + k_r * mu_f) / (_previo(c, df["player_id"]) + k_r) - mu_f).to_numpy(np.float64)

    r_eff = _prior_rate(early_ff, 25.0)
    r_ff20 = _prior_rate(ff_20, 25.0)
    r_int = _prior_rate(heavy_int, 20.0)
    r_afk = _prior_rate(afk_low, 25.0)
    r_d10 = _prior_rate(die2_10, 15.0)
    streak = _streak_causal_por_jugador(pid, win).astype(np.float64)

    sm_wr = sin_foto * wr_acum_v
    sm_low = ((n_tot < 25) * np.maximum(0.0, wr_acum_v)).astype(np.float64)
    sm_no_foto_post = np.where(np.repeat(D["post"], 10), sin_foto, 0.0)

    SMURF_TILT = np.c_[
        _agg_eq(np.c_[sin_foto, sm_wr, sm_low, streak, r_eff, r_ff20, r_int, r_afk, r_d10]),
        _agg_eq(np.c_[r_eff, r_int, sm_wr], with_max=True)[:, 3:],
    ].astype(np.float32)
    SMURF_POST = _agg_eq(np.c_[sm_no_foto_post, sm_no_foto_post * wr_acum_v]).astype(np.float32)

    # 4. SNOW_JUG (Timeline/rol faltante en Xj)
    df = lane_diffs(df, ["gold_15"])
    tot_cs = df["total_minions_killed"].fillna(0.0) + df["neutral_minions_killed"].fillna(0.0)
    df["jg_share"] = df["neutral_minions_killed"].fillna(0.0) / np.maximum(1.0, tot_cs)
    df["early_agg_pm"] = df["early_aggression_10"]
    df["deaths_10_c"] = df["deaths_10"]
    stats_n = ["jg_share", "early_agg_pm", "deaths_10_c", "d_gold_15"]
    esc = lane_scale(df[fit_rows], stats_n)
    cl_key = [df["champion_id"], df["lane"]]
    res_list = []
    for c_name in stats_n:
        mu_l, sd_l = df["lane"].map(esc[c_name][0]), df["lane"].map(esc[c_name][1]).clip(lower=1e-6)
        ok_c = df[c_name].notna().astype(np.float64)
        v_c = df[c_name].fillna(0.0)
        mu_cl = (_previo(v_c, cl_key) + 15.0 * mu_l) / (_previo(ok_c, cl_key) + 15.0)
        res_c = np.where(ok_c > 0, (v_c - mu_cl) / sd_l, np.nan)
        ew_c, _, sd_c, _ = _ewma_causal_por_jugador(pid, res_c, hl_slow=20.0, hl_fast=8.0, mu0=0.0)
        res_list.append(ew_c)
        if c_name == "d_gold_15":
            res_list.append(sd_c)
    snow_mat = np.column_stack(res_list)
    SNOW_JUG = np.c_[
        _agg_eq(snow_mat),
        _agg_eq(snow_mat[:, [2, 4]], with_max=True)[:, 2:],
    ].astype(np.float32)

    # 5. CAMP_CURVA (curva temprana vs tardía causal por campeón y cruce contra agresividad)
    is_e = (df["game_duration"] < 1500).to_numpy(np.float64)
    is_l = (df["game_duration"] > 1800).to_numpy(np.float64)
    we = _previo(pd.Series(win * is_e, index=df.index), df["champion_id"]).to_numpy(np.float64)
    ne = _previo(pd.Series(is_e, index=df.index), df["champion_id"]).to_numpy(np.float64)
    wl = _previo(pd.Series(win * is_l, index=df.index), df["champion_id"]).to_numpy(np.float64)
    nl = _previo(pd.Series(is_l, index=df.index), df["champion_id"]).to_numpy(np.float64)
    wr_ce = (we + 25.0 * 0.5) / (ne + 25.0) - 0.5
    wr_cl = (wl + 25.0 * 0.5) / (nl + 25.0) - 0.5
    sc_c = wr_cl - wr_ce
    c_mat = np.c_[wr_ce, wr_cl, sc_c].reshape(NM, 2, 5, 3)
    ea_b, ea_r = snow_mat[:, 1].reshape(NM, 2, 5)[:, 0].mean(1), snow_mat[:, 1].reshape(NM, 2, 5)[:, 1].mean(1)
    CAMP_CURVA = np.c_[
        (c_mat[:, 0] - c_mat[:, 1]).mean(1),
        c_mat[:, 0, :, 2].max(1) - c_mat[:, 1, :, 2].max(1),
        c_mat[:, 0, :, 0].mean(1) * c_mat[:, 1, :, 1].mean(1) - c_mat[:, 1, :, 0].mean(1) * c_mat[:, 0, :, 1].mean(1),
        c_mat[:, 0, :, 2].mean(1) * ea_r - c_mat[:, 1, :, 2].mean(1) * ea_b,
    ].astype(np.float32)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE,
        ROL_WR=ROL_WR,
        MMR_V2=MMR_V2,
        CLIMB_POST=CLIMB_POST,
        SMURF_TILT=SMURF_TILT,
        SMURF_POST=SMURF_POST,
        SNOW_JUG=SNOW_JUG,
        CAMP_CURVA=CAMP_CURVA,
    )
    print(f"Caché guardada en {CACHE} ({time.time()-t0:.1f}s)", flush=True)
    return dict(np.load(CACHE))


def _draft_reciente_oof(D: dict, tr_ms: np.ndarray, te: np.ndarray, tau_dias: float = 21.0, n_bloques: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Puntaje OOF estricto hacia adelante (`bloque < b` dentro de tr) de DRAFT con peso exponencial."""
    t, y, DRAFT = D["t"], D["y"], D["DRAFT"]
    tau_ms = tau_dias * 86_400_000.0
    qs = np.quantile(t[tr_ms], np.linspace(0, 1, n_bloques + 1)[1:-1])
    bl = np.searchsorted(qs, t[tr_ms], side="right")
    z_tr = np.zeros(len(tr_ms), dtype=np.float32)
    for b in range(1, n_bloques):
        atr = tr_ms[bl < b]
        ate = tr_ms[bl == b]
        w = np.exp(-(t[ate].min() - t[atr]) / tau_ms)
        lr = LogisticRegression(C=0.05, max_iter=2000).fit(DRAFT[atr], y[atr], sample_weight=w)
        z_tr[bl == b] = logit(np.clip(lr.predict_proba(DRAFT[ate])[:, 1], 1e-5, 1.0 - 1e-5))
    w_all = np.exp(-(t[tr_ms].max() - t[tr_ms]) / tau_ms)
    lr_full = LogisticRegression(C=0.05, max_iter=2000).fit(DRAFT[tr_ms], y[tr_ms], sample_weight=w_all)
    z_te = logit(np.clip(lr_full.predict_proba(DRAFT[te])[:, 1], 1e-5, 1.0 - 1e-5)).astype(np.float32)
    return z_tr[:, None], z_te[:, None]


def main():
    D = cargar()
    ms = D["master"]
    G = dict(np.load(CACHE)) if CACHE.exists() else construir_cache(D)
    REF = extras_referencia(D)

    mpr_path = REPO / "data" / "analysis" / "brainstorm" / "gemini_mpr.npz"
    if mpr_path.exists():
        REF_M = np.c_[REF, np.load(mpr_path)["MATCHUPS"]].astype(np.float32)
        ref_tag = "+ MATCHUPS"
    else:
        REF_M = REF
        ref_tag = ""

    def run_ext(extra=None, base_x=REF, extra_post_add=None):
        X_pre = base_x if extra is None else np.c_[base_x, extra].astype(np.float32)
        if extra_post_add is None:
            return lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=X_pre, extra_post=False)
        return lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=[(X_pre, False), (extra_post_add.astype(np.float32), True)])

    def run_draft_rec(base_x=REF_M, tau_dias: float = 21.0):
        def _fn(tr, te):
            tr_m = tr[ms[tr]]
            zr, ze = _draft_reciente_oof(D, tr_m, te, tau_dias=tau_dias)
            X_full = np.zeros((D["NM"], base_x.shape[1] + 1), dtype=np.float32)
            X_full[:, :-1] = base_x
            X_full[tr_m, -1] = zr[:, 0]
            X_full[te, -1] = ze[:, 0]
            return ajustar_predecir(D, tr_m, te, extra=X_full, extra_post=False)
        return _fn

    REF4 = extras_referencia(D, escalon=4)
    g08_path = REPO / "data" / "analysis" / "brainstorm" / "pred_g08.npz"
    P = {}
    if g08_path.exists() and "po_log4" in np.load(g08_path).files:
        G8 = np.load(g08_path)
        P["ref: Escalón 4 (0,6674)"] = (G8["po_log4"], G8["pv_log4"])
    else:
        P["ref: Escalón 4 (0,6674)"] = evaluar(D, run_ext(base_x=REF4))

    VAR = {
        "+ ROL_WR (wr_lane/champ_lane/autofill)": run_ext(G["ROL_WR"], base_x=REF4),
        "+ MMR_V2 (EWMA/imputación/climb)": run_ext(G["MMR_V2"], base_x=REF4, extra_post_add=G["CLIMB_POST"]),
        "+ SMURF_TILT (Ideas 6 y 7)": run_ext(G["SMURF_TILT"], base_x=REF4, extra_post_add=G["SMURF_POST"]),
        "+ SNOW_JUG (jg_share/agg10/std_g15)": run_ext(G["SNOW_JUG"], base_x=REF4),
        "+ CAMP_CURVA (early vs late scaling)": run_ext(G["CAMP_CURVA"], base_x=REF4),
        "+ DRAFT_REC (tau=21d)": run_draft_rec(REF4, tau_dias=21.0),
        "+ DRAFT_REC (tau=14d)": run_draft_rec(REF4, tau_dias=14.0),
        "+ ROL_WR + MMR_V2 (s/Esc4)": run_ext(np.c_[G["ROL_WR"], G["MMR_V2"]], base_x=REF4, extra_post_add=G["CLIMB_POST"]),
    }
    for nom, fn in VAR.items():
        print(f"Evaluando {nom}...", flush=True)
        P[nom] = evaluar(D, fn)
    mostrar(comparar(D, P, "ref: Escalón 4 (0,6674)"))
    np.savez_compressed(
        REPO / "data" / "analysis" / "brainstorm" / "pred_g10.npz",
        **{f"o_{i}": P[n][0] for i, n in enumerate(P)},
        **{f"v_{i}": P[n][1] for i, n in enumerate(P)},
        names=np.array(list(P)),
    )


if __name__ == "__main__":
    main()
