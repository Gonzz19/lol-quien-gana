"""Idea G35 del brainstorming (`docs/BRAINSTORM_NOCHE.md`):
1. `MASTERY_CURVE` (`8` cols causales < t): curva empírica causal de maestría por `(campeón, línea)`
   cuando el piloto tiene poca experiencia (`n_cl <= 2` o `3..7` vs `>= 12`), midiendo la caída
   real de winrate de cada campeón al primer uso (p. ej. Nidalee/Azir/Riven vs Malphite/Annie)
   y el ajuste esperado específico `(cl, n_cl_piloto)`, algo invisible para `D["DRAFT"]` constante
   y para `info_dif` entero estático de Data Dragon.
2. `PER_JUG_ESC6` (`(NM, 2, 5, 16)`): exportación causal por jugador de las señales ganadoras
   de Escalón 6 (`ROL_WR`: `wr_lane`, `wr_lane_diff`, `wr_cl_exc`, `log1p_pl`, `autofill_drop`,
   `autofill_flag`; y `KALMAN/TEMPO`: `r_elo`, `r_slow`, `r_opp`, `r_mrg`, `r_mrg_slow`, `unc`,
   `clutch`, `fb_r`, `pbias`, `dmg_gold_eff`) para inyectarlas directamente al encoder por
   jugador de `RedNeuralACWideDeep` y `RedAttnWideDeep` después de cerrar `g_22`.

Caché: `data/analysis/brainstorm/gemini_g24.npz`.
Evaluado s/Escalón 6 sin mapa (`V 0,6709`) + placebo + chequeo ciego `Diamond-V`.
"""
import math
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.brainstorm_base import ajustar_predecir, cargar, comparar, evaluar, extras_referencia, mostrar  # noqa: E402
from src.features.margin import ORDEN, _previo, prior_winrate  # noqa: E402

CACHE = REPO / "data" / "analysis" / "brainstorm" / "gemini_g24.npz"
LANES = ["BOTTOM", "JUNGLE", "MIDDLE", "TOP", "UTILITY"]


def construir_cache_g24(D: dict, rehacer: bool = False) -> dict[str, np.ndarray]:
    if CACHE.exists() and not rehacer:
        z = np.load(CACHE)
        if int(z["nm"][0]) == D["NM"]:
            return {k: z[k] for k in z.files if k != "nm"}

    t0 = time.time()
    print("Construyendo caché gemini_g24.npz (MASTERY_CURVE + PER_JUG_ESC6)...", flush=True)
    NM = D["NM"]
    Z = D["Z"]
    mids = Z["mids"].astype(str)
    t_arr = D["t"].astype(np.int64)
    y_arr = D["y"].astype(np.int8)
    val0 = D["val0"].astype(bool)
    post = D["post"].astype(bool)
    champ = Z["champ"].reshape(NM, 10)
    dur_min = Z["D_dur_min"].astype(np.float32)
    dg15 = Z["D_d_gold_15"].astype(np.float32)
    dxp15 = Z["D_d_xp_15"].astype(np.float32)
    dk15 = Z["D_d_kills_15"].astype(np.float32)

    df = pd.read_csv(
        REPO / "data" / "splits" / "train" / "full_dataset.csv",
        usecols=[
            "match_id", "game_creation", "team_id", "lane", "player_id", "champion_id", "win",
            "first_blood_kill", "gold_share_pct", "damage_share_pct",
        ],
    )
    df = df[df.lane.isin(LANES)]
    df = df[df.groupby(["match_id", "team_id"]).lane.transform("nunique") == 5]
    df = df[df.groupby("match_id").lane.transform("size") == 10]
    df = df[df.groupby("match_id").win.transform("sum") == 5].copy()
    df = df.sort_values(ORDEN).reset_index(drop=True)
    assert (df["match_id"].to_numpy()[::10] == mids).all()

    # 1. Per-player ROL_WR (estrictamente causal < t)
    ones = pd.Series(1.0, index=df.index)
    w_ser = pd.Series(df["win"].to_numpy(np.float64), index=df.index)
    wr_all = prior_winrate(df, window=20, k=10)
    n_tot = _previo(ones, df["player_id"]).to_numpy(np.float64)
    n_pl = _previo(ones, [df["player_id"], df["lane"]]).to_numpy(np.float64)
    w_pl = _previo(w_ser, [df["player_id"], df["lane"]]).to_numpy(np.float64)
    wr_acum_v = wr_all["wr_acum"].to_numpy(np.float64)
    wr_lane = (w_pl + 8.0 * (0.5 + wr_acum_v)) / (n_pl + 8.0) - 0.5
    wr_lane_diff = wr_lane - wr_acum_v
    n_pcl = _previo(ones, [df["player_id"], df["champion_id"], df["lane"]]).to_numpy(np.float64)
    w_pcl = _previo(w_ser, [df["player_id"], df["champion_id"], df["lane"]]).to_numpy(np.float64)
    wr_cl_exc = (w_pcl + 6.0 * (0.5 + wr_lane)) / (n_pcl + 6.0) - 0.5 - wr_lane
    share_lane = n_pl / np.maximum(1.0, n_tot)
    autofill_drop = (1.0 - share_lane) * wr_lane_diff
    autofill_flag = ((n_tot >= 15) & (share_lane < 0.20)).astype(np.float64) * np.minimum(0.0, wr_lane_diff)

    n_pcl_mat = n_pcl.reshape(NM, 10).tolist()
    fb_arr = df["first_blood_kill"].fillna(0).to_numpy(np.float32).reshape(NM, 10)
    gs_raw = df["gold_share_pct"].to_numpy(np.float32).reshape(NM, 10)
    ds_raw = df["damage_share_pct"].to_numpy(np.float32).reshape(NM, 10)
    if float(np.nanmedian(gs_raw)) > 1.5:
        gs_raw, ds_raw = gs_raw / 100.0, ds_raw / 100.0
    dge_l = np.nan_to_num(ds_raw - gs_raw, nan=0.0).tolist()
    fb_l = fb_arr.tolist()
    del df, wr_all, ones, w_ser

    players = D["B"]["player"].reshape(NM, 10).astype(str)
    uniq_p, p_inv = np.unique(players.reshape(-1), return_inverse=True)
    pid_mat = p_inv.reshape(NM, 10).tolist()
    n_p = len(uniq_p)

    cl_pairs = champ * 5 + np.tile(np.arange(5), 2)[None, :]
    uniq_cl, cl_inv = np.unique(cl_pairs.reshape(-1), return_inverse=True)
    cl_mat = cl_inv.reshape(NM, 10).tolist()
    n_cl = len(uniq_cl)

    # Pesos @15 fijos en ~val0 & ~post para Kalman-Mrg y Clutch
    m_fit15 = (~val0) & (~post) & np.isfinite(dg15) & np.isfinite(dxp15) & np.isfinite(dk15) & (dur_min >= 15.0)
    X15 = np.c_[np.nan_to_num(dg15, 0.0) / 1000.0, np.nan_to_num(dxp15, 0.0) / 1000.0, np.nan_to_num(dk15, 0.0)]
    lr15 = LogisticRegression(C=1.0, max_iter=500).fit(X15[m_fit15], y_arr[m_fit15])
    p_exp15 = lr15.predict_proba(X15)[:, 1].astype(np.float64)
    valid15 = np.isfinite(dg15) & (dur_min >= 15.0)

    # 2. Barrido cronológico único: MASTERY_CURVE + Kalman/Tempo por jugador
    cl_low_w, cl_low_n = [0.0] * n_cl, [0.0] * n_cl   # n_pcl <= 2
    cl_mid_w, cl_mid_n = [0.0] * n_cl, [0.0] * n_cl   # 3 <= n_pcl <= 9
    cl_exp_w, cl_exp_n = [0.0] * n_cl, [0.0] * n_cl   # n_pcl >= 10
    cl_tot_w, cl_tot_n = [0.0] * n_cl, [0.0] * n_cl

    # Estados Kalman/Tempo por jugador (fieles a g_11)
    r_elo = [0.0] * n_p
    r_slow = [0.0] * n_p
    r_opp = [0.0] * n_p
    r_mrg = [0.0] * n_p
    r_mrg_slow = [0.0] * n_p
    cnt_p = [0] * n_p
    cl_sum, cl_w = [0.0] * n_p, [0.0] * n_p
    fb_sum, ew_cnt, dg_sum = [0.0] * n_p, [0.0] * n_p, [0.0] * n_p
    wr_e_sum, wr_e_n, wr_l_sum, wr_l_n = [0.0] * n_p, [0.0] * n_p, [0.0] * n_p, [0.0] * n_p

    dec20 = math.exp(-math.log(2.0) / 20.0)
    dec25 = math.exp(-math.log(2.0) / 25.0)
    dec30 = math.exp(-math.log(2.0) / 30.0)

    MASTERY_CURVE = np.zeros((NM, 8), dtype=np.float32)
    P_KT = np.zeros((NM, 10, 10), dtype=np.float32)
    P_MAST = np.zeros((NM, 10, 3), dtype=np.float32)

    order = np.lexsort((mids, t_arr))
    K_M = 40.0
    for m in order:
        ps = pid_mat[m]
        cls = cl_mat[m]
        npcl = n_pcl_mat[m]
        yb = float(y_arr[m])
        dm = float(dur_min[m])
        v15 = bool(valid15[m])
        p15 = float(p_exp15[m])

        # ---- Leer MASTERY_CURVE causal (< m) ----
        m_adj = [0.0] * 10
        m_steep_low = [0.0] * 10
        m_def = [0.0] * 10
        for slot in range(10):
            cl = cls[slot]
            ncl = npcl[slot]
            wr_all_cl = (cl_tot_w[cl] + 15.0 * 0.50) / (cl_tot_n[cl] + 30.0)
            wr_low_cl = (cl_low_w[cl] + K_M * (wr_all_cl - 0.02)) / (cl_low_n[cl] + K_M)
            wr_mid_cl = (cl_mid_w[cl] + K_M * wr_all_cl) / (cl_mid_n[cl] + K_M)
            wr_exp_cl = (cl_exp_w[cl] + K_M * (wr_all_cl + 0.015)) / (cl_exp_n[cl] + K_M)

            if ncl <= 2.5:
                wr_bin = wr_low_cl
            elif ncl <= 9.5:
                wr_bin = wr_mid_cl
            else:
                wr_bin = wr_exp_cl

            adj = wr_bin - wr_all_cl
            steep = wr_exp_cl - wr_low_cl
            nov_w = 1.0 / (1.0 + ncl / 4.0)
            st_x_l = steep * nov_w
            deficit = wr_exp_cl - wr_bin

            m_adj[slot] = adj
            m_steep_low[slot] = st_x_l
            m_def[slot] = deficit
            P_MAST[m, slot, 0] = adj
            P_MAST[m, slot, 1] = st_x_l
            P_MAST[m, slot, 2] = deficit

        MASTERY_CURVE[m, 0] = (sum(m_adj[:5]) - sum(m_adj[5:])) * 0.2
        MASTERY_CURVE[m, 1] = min(m_adj[:5]) - min(m_adj[5:])
        MASTERY_CURVE[m, 2] = (sum(m_steep_low[:5]) - sum(m_steep_low[5:])) * 0.2
        MASTERY_CURVE[m, 3] = max(m_steep_low[:5]) - max(m_steep_low[5:])
        MASTERY_CURVE[m, 4] = (sum(m_def[:5]) - sum(m_def[5:])) * 0.2
        MASTERY_CURVE[m, 5] = m_steep_low[1] - m_steep_low[6]               # JG steep novelty
        MASTERY_CURVE[m, 6] = m_steep_low[2] - m_steep_low[7]               # MID steep novelty
        MASTERY_CURVE[m, 7] = (m_adj[0] + m_adj[3]) - (m_adj[5] + m_adj[8]) # ADC + TOP

        # ---- Leer Kalman / Tempo por jugador (< m) ----
        rb = [r_elo[ps[i]] for i in range(5)]
        rr = [r_elo[ps[5 + i]] for i in range(5)]
        rmb = [r_mrg[ps[i]] for i in range(5)]
        rmr = [r_mrg[ps[5 + i]] for i in range(5)]
        for slot in range(10):
            p = ps[slot]
            P_KT[m, slot, 0] = r_elo[p] / 100.0
            P_KT[m, slot, 1] = r_slow[p] / 100.0
            P_KT[m, slot, 2] = r_opp[p] / 100.0
            P_KT[m, slot, 3] = r_mrg[p] / 100.0
            P_KT[m, slot, 4] = r_mrg_slow[p] / 100.0
            P_KT[m, slot, 5] = 1.0 / math.sqrt(1.0 + cnt_p[p])
            P_KT[m, slot, 6] = cl_sum[p] / (cl_w[p] + 10.0)
            P_KT[m, slot, 7] = (fb_sum[p] + 12.0 * 0.10) / (ew_cnt[p] + 12.0)
            P_KT[m, slot, 8] = ((wr_l_sum[p] + 6.0 * 0.5) / (wr_l_n[p] + 12.0)) - ((wr_e_sum[p] + 6.0 * 0.5) / (wr_e_n[p] + 12.0))
            P_KT[m, slot, 9] = dg_sum[p] / (ew_cnt[p] + 10.0)

        # ---- Actualizar estados DESPUÉS ----
        mb, mr = sum(rb) * 0.2, sum(rr) * 0.2
        m_rmb, m_rmr = sum(rmb) * 0.2, sum(rmr) * 0.2
        err_out = yb - 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, (mb - mr) / 200.0 - 0.105))))
        mrg_target = 0.70 * yb + 0.30 * (p15 if v15 else yb)
        err_mrg = mrg_target - 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, (m_rmb - m_rmr) / 200.0 - 0.105))))
        lobby_elo = (mb + mr) * 0.5
        lobby_mrg = (m_rmb + m_rmr) * 0.5
        fb_row = fb_l[m]
        dge_row = dge_l[m]

        for slot in range(10):
            p = ps[slot]
            cl = cls[slot]
            ncl = npcl[slot]
            sgn = 1.0 if slot < 5 else -1.0
            win_p = yb if slot < 5 else (1.0 - yb)

            cl_tot_w[cl] += win_p
            cl_tot_n[cl] += 1.0
            if ncl <= 2.5:
                cl_low_w[cl] += win_p
                cl_low_n[cl] += 1.0
            elif ncl <= 9.5:
                cl_mid_w[cl] += win_p
                cl_mid_n[cl] += 1.0
            else:
                cl_exp_w[cl] += win_p
                cl_exp_n[cl] += 1.0

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

            if v15:
                cl_sum[p] = cl_sum[p] * dec20 + (yb - p15) * sgn
                cl_w[p] = cl_w[p] * dec20 + 1.0
            fb_sum[p] = fb_sum[p] * dec25 + fb_row[slot]
            dg_sum[p] = dg_sum[p] * dec25 + dge_row[slot]
            ew_cnt[p] = ew_cnt[p] * dec25 + 1.0
            if dm < 26.0:
                wr_e_sum[p] = wr_e_sum[p] * dec30 + win_p
                wr_e_n[p] = wr_e_n[p] * dec30 + 1.0
            else:
                wr_l_sum[p] = wr_l_sum[p] * dec30 + win_p
                wr_l_n[p] = wr_l_n[p] * dec30 + 1.0

    P_ROL = np.stack([
        wr_lane, wr_lane_diff, wr_cl_exc, np.log1p(n_pl), autofill_drop, autofill_flag
    ], axis=-1).reshape(NM, 2, 5, 6).astype(np.float32)
    PER_JUG_ESC6 = np.concatenate([P_ROL, P_KT.reshape(NM, 2, 5, 10), P_MAST.reshape(NM, 2, 5, 3)], axis=-1).astype(np.float32)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE,
        nm=np.array([NM]),
        MASTERY_CURVE=MASTERY_CURVE,
        PER_JUG_ESC6=PER_JUG_ESC6,
    )
    print(f"  gemini_g24.npz listo en {time.time() - t0:.1f}s", flush=True)
    return {"MASTERY_CURVE": MASTERY_CURVE, "PER_JUG_ESC6": PER_JUG_ESC6}


def main():
    t0 = time.time()
    D = cargar()
    G24 = construir_cache_g24(D)
    REF7 = extras_referencia(D, escalon=7)
    ms = D["master"]

    def solo_m(extra=None):
        tot = REF7 if extra is None else np.c_[REF7, extra].astype(np.float32)
        return lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=tot, extra_post=False, con_mapa=False)

    P = {"ref: Escalón 7 (sin mapa)": evaluar(D, solo_m())}
    P["+ MASTERY_CURVE (8)"] = evaluar(D, solo_m(G24["MASTERY_CURVE"]))
    rng = np.random.default_rng(99)
    P["Placebo g24 (8 perm)"] = evaluar(D, solo_m(G24["MASTERY_CURVE"][rng.permutation(D["NM"])]))
    mostrar(comparar(D, P, "ref: Escalón 7 (sin mapa)"))

    fi_v, dv = D["fi_v"], D["diam_v"]
    ydv = D["y"][dv]
    p_dv_ref = solo_m()(fi_v, dv)
    auc_ref = roc_auc_score(ydv, p_dv_ref)
    rng = np.random.default_rng(77)
    boots = rng.integers(0, len(ydv), size=(1500, len(ydv)))
    p_dv = solo_m(G24["MASTERY_CURVE"])(fi_v, dv)
    auc_p = roc_auc_score(ydv, p_dv)
    ds = np.array([roc_auc_score(ydv[b], p_dv[b]) - roc_auc_score(ydv[b], p_dv_ref[b]) for b in boots])
    print(f"\nChequeo ciego Diamond-V ({len(ydv):,} partidas): ref={auc_ref:.4f}")
    print(f"  + MASTERY_CURVE (8)      AUC {auc_p:.4f}  Δ {auc_p - auc_ref:+.4f} [{np.percentile(ds, 2.5):+.4f}; {np.percentile(ds, 97.5):+.4f}]")
    print(f"Listo en {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
