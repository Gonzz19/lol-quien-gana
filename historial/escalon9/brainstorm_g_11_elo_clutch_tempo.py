"""Ideas G24-G28 del brainstorming (docs/BRAINSTORM_NOCHE.md): señales causales no explotadas
sobre la referencia Escalón 3 + MATCHUPS (y contra Escalón 3 `0,6663`):

  1. ELO DECRECIENTE (claves internas `KALMAN_OUT`/`KALMAN_MRG` por compatibilidad con cachés):
     Elo causal en línea de doble señal (resultado + atracción al lobby, con sesgo azul/rojo y paso
     decreciente con las partidas vistas, sin propagación bayesiana de varianza), + flujo con margen
     @15 + velocidad de MMR (`elo - slow`). Cero Foto 1 (100 % causal desde t=0 en todo el grafo).
  2. CLUTCH_15: residual causal de cierre/throw pos-minuto 15 (`win - E[win | @15]`), + winrate
     estando adelante (`dg15 > +1000`) / atrás (`dg15 < -1000`) + eficiencia weakside. Los pesos
     @15 fijos se ajustan únicamente en pre-Foto 1 (`~val0 & ~post`).
  3. TEMPO_OBJ: tasas causales por jugador de First Blood, First Tower, swap de trinket (`item6 != 3340`),
     preferencia de duración (`wr_late - wr_early`), curva early/late causal del campeón y su cruce
     (`player_tempo * champ_tempo`).
  4. ECON_MID: transición 15->20 en oro contra rival de línea (`(g20-g15)_b - (g20-g15)_r`) + eficiencia
     `damage_share - gold_share`.

Caché guardada en `data/analysis/brainstorm/gemini_g11.npz`.
"""
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.brainstorm_base import ajustar_predecir, cargar, comparar, evaluar, extras_referencia, mostrar  # noqa: E402

CACHE = REPO / "data" / "analysis" / "brainstorm" / "gemini_g11.npz"


def _ema_decay(alpha: float):
    """Decay factor 1 - alpha."""
    return 1.0 - alpha


def construir_cache_g11(D: dict, rehacer: bool = False) -> dict[str, np.ndarray]:
    if CACHE.exists() and not rehacer:
        z = np.load(CACHE)
        if int(z["nm"][0]) == D["NM"]:
            return {k: z[k] for k in z.files if k != "nm"}

    t0 = time.time()
    print("Construyendo caché gemini_g11.npz...", flush=True)
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

    # Identificadores de jugador exactamente alineados con (NM, 2, 5)
    players = D["B"]["player"].reshape(NM, 10).astype(str)
    uniq_p, p_inv = np.unique(players.reshape(-1), return_inverse=True)
    pid_mat = p_inv.reshape(NM, 10).tolist()
    n_p = len(uniq_p)

    # Campeon x linea
    cl_pairs = champ * 5 + np.tile(np.arange(5), 2)[None, :]
    uniq_cl, cl_inv = np.unique(cl_pairs.reshape(-1), return_inverse=True)
    cl_mat = cl_inv.reshape(NM, 10).tolist()
    n_cl = len(uniq_cl)

    # Cargar columnas puntuales desde full_dataset.csv alineadas por (match_id, player_id)
    cols_csv = [
        "match_id", "player_id", "first_blood_kill", "first_tower_kill",
        "gold_share_pct", "damage_share_pct", "gold_15", "gold_20", "item6",
    ]
    df = pd.read_csv(REPO / "data" / "splits" / "train" / "full_dataset.csv", usecols=cols_csv)
    df = df.drop_duplicates(["match_id", "player_id"])
    keys = pd.DataFrame({
        "match_id": np.repeat(mids, 10),
        "player_id": players.reshape(-1),
        "_pos": np.arange(NM * 10, dtype=np.int64),
    })
    M = keys.merge(df, on=["match_id", "player_id"], how="left").sort_values("_pos", kind="mergesort")
    del df, keys

    fb_arr = M["first_blood_kill"].fillna(0).to_numpy(np.float32).reshape(NM, 10)
    ft_arr = M["first_tower_kill"].fillna(0).to_numpy(np.float32).reshape(NM, 10)
    gs_raw = M["gold_share_pct"].to_numpy(np.float32).reshape(NM, 10)
    ds_raw = M["damage_share_pct"].to_numpy(np.float32).reshape(NM, 10)
    if float(np.nanmedian(gs_raw)) > 1.5:
        gs_raw = gs_raw / 100.0
        ds_raw = ds_raw / 100.0
    gs_arr = np.nan_to_num(gs_raw, nan=0.20)
    ds_arr = np.nan_to_num(ds_raw, nan=0.20)
    g15_p = M["gold_15"].to_numpy(np.float32).reshape(NM, 10)
    g20_p = M["gold_20"].to_numpy(np.float32).reshape(NM, 10)
    it6_arr = M["item6"].fillna(3340).to_numpy(np.int32).reshape(NM, 10)
    del M

    swap_arr = (it6_arr != 3340).astype(np.float32)
    dmg_gold_eff = (ds_arr - gs_arr).astype(np.float32)
    surge_20_15 = (g20_p - g15_p).astype(np.float32)
    # Ventaja por linea 15->20: azul_l - rojo_l
    surge_valid = np.isfinite(surge_20_15[:, :5]) & np.isfinite(surge_20_15[:, 5:])
    surge_diff = np.where(surge_valid, (surge_20_15[:, :5] - surge_20_15[:, 5:]) / 1000.0, 0.0)

    # Pesos fijos para prob esperada @15 ajustados ESTRICTAMENTE en ~val0 & ~post (junio-agosto)
    m_fit15 = (~val0) & (~post) & np.isfinite(dg15) & np.isfinite(dxp15) & np.isfinite(dk15) & (dur_min >= 15.0)
    X15 = np.c_[
        np.nan_to_num(dg15, 0.0) / 1000.0,
        np.nan_to_num(dxp15, 0.0) / 1000.0,
        np.nan_to_num(dk15, 0.0),
    ]
    lr15 = LogisticRegression(C=1.0, max_iter=500).fit(X15[m_fit15], y_arr[m_fit15])
    p_exp15_all = lr15.predict_proba(X15)[:, 1].astype(np.float64)
    valid15 = np.isfinite(dg15) & (dur_min >= 15.0)
    print(f"  Modelo @15 pre-Foto1 (n={int(m_fit15.sum())}): coef={np.round(lr15.coef_[0], 3)}", flush=True)

    # Salidas
    KALMAN_OUT = np.zeros((NM, 12), dtype=np.float32)
    KALMAN_MRG = np.zeros((NM, 7), dtype=np.float32)
    CLUTCH_15 = np.zeros((NM, 8), dtype=np.float32)
    TEMPO_OBJ = np.zeros((NM, 10), dtype=np.float32)
    ECON_MID = np.zeros((NM, 4), dtype=np.float32)

    # Estados por jugador (listas Python para maxima velocidad)
    r_elo = [0.0] * n_p
    r_slow = [0.0] * n_p
    r_opp = [0.0] * n_p
    r_mrg = [0.0] * n_p
    r_mrg_slow = [0.0] * n_p
    cnt_p = [0] * n_p

    # Clutch @15 por jugador
    cl_sum = [0.0] * n_p
    cl_w = [0.0] * n_p
    ah_win = [0.0] * n_p
    ah_n = [0.0] * n_p
    bh_win = [0.0] * n_p
    bh_n = [0.0] * n_p
    wk_sum = [0.0] * n_p
    wk_w = [0.0] * n_p

    # Tempo, FB/FT, vision, mid econ por jugador
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

    # Curva early/late por (campeon, linea)
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
    gs_l = gs_arr.tolist()
    dge_l = dmg_gold_eff.tolist()
    sd_l = surge_diff.tolist()
    sv_l = surge_valid.tolist()

    for m in order:
        ps = pid_mat[m]
        cls = cl_mat[m]
        yb = float(y_arr[m])
        dm = float(dur_min[m])
        v15 = bool(valid15[m])
        p15 = float(p_exp15_all[m])
        dg15_m = float(dg15[m]) if np.isfinite(dg15[m]) else 0.0

        # ---- 1. Leer estados CAUSALES antes de actualizar ----
        rb = [r_elo[ps[i]] for i in range(5)]
        rr = [r_elo[ps[5 + i]] for i in range(5)]
        sb = [r_slow[ps[i]] for i in range(5)]
        sr_ = [r_slow[ps[5 + i]] for i in range(5)]
        ob = [r_opp[ps[i]] for i in range(5)]
        or_ = [r_opp[ps[5 + i]] for i in range(5)]

        mb = sum(rb) * 0.2
        mr = sum(rr) * 0.2
        rb_s = sorted(rb)
        rr_s = sorted(rr)
        stdb = math.sqrt(sum((x - mb) ** 2 for x in rb) * 0.2)
        stdr = math.sqrt(sum((x - mr) ** 2 for x in rr) * 0.2)

        KALMAN_OUT[m, 0] = (mb - mr) / 100.0
        KALMAN_OUT[m, 1] = (rb_s[0] - rr_s[0]) / 100.0
        KALMAN_OUT[m, 2] = (rb_s[4] - rr_s[4]) / 100.0
        KALMAN_OUT[m, 3] = (stdb - stdr) / 100.0
        KALMAN_OUT[m, 4] = ((rb_s[3] + rb_s[4]) - (rr_s[3] + rr_s[4])) * 0.5 / 100.0
        KALMAN_OUT[m, 5] = (sum(rb[i] - sb[i] for i in range(5)) - sum(rr[i] - sr_[i] for i in range(5))) * 0.2 / 100.0
        KALMAN_OUT[m, 6] = (sum(ob) - sum(or_)) * 0.2 / 100.0
        for l in range(5):
            KALMAN_OUT[m, 7 + l] = (rb[l] - rr[l]) / 100.0

        rmb = [r_mrg[ps[i]] for i in range(5)]
        rmr = [r_mrg[ps[5 + i]] for i in range(5)]
        rmb_s = sorted(rmb)
        rmr_s = sorted(rmr)
        unc_b = sum(1.0 / math.sqrt(1.0 + cnt_p[ps[i]]) for i in range(5)) * 0.2
        unc_r = sum(1.0 / math.sqrt(1.0 + cnt_p[ps[5 + i]]) for i in range(5)) * 0.2
        KALMAN_MRG[m, 0] = (sum(rmb) - sum(rmr)) * 0.2 / 100.0
        KALMAN_MRG[m, 1] = (rmb_s[0] - rmr_s[0]) / 100.0
        KALMAN_MRG[m, 2] = (rmb_s[4] - rmr_s[4]) / 100.0
        KALMAN_MRG[m, 3] = (sum(rmb[i] - r_mrg_slow[ps[i]] for i in range(5)) - sum(rmr[i] - r_mrg_slow[ps[5 + i]] for i in range(5))) * 0.2 / 100.0
        KALMAN_MRG[m, 4] = unc_b - unc_r
        # Diferencia entre elo por resultado y elo por grafo de oponentes (over/under-performer vs lobby!)
        KALMAN_MRG[m, 5] = ((sum(rb) - sum(ob)) - (sum(rr) - sum(or_))) * 0.2 / 100.0
        # Spread del lobby conjunto
        KALMAN_MRG[m, 6] = abs(mb - mr) / 100.0

        # Clutch & ahead/behind & weakside
        clv = [cl_sum[ps[i]] / (cl_w[ps[i]] + 10.0) for i in range(10)]
        ahv = [(ah_win[ps[i]] + 8.0 * 0.78) / (ah_n[ps[i]] + 8.0) - 0.78 for i in range(10)]
        bhv = [(bh_win[ps[i]] + 8.0 * 0.22) / (bh_n[ps[i]] + 8.0) - 0.22 for i in range(10)]
        wkv = [wk_sum[ps[i]] / (wk_w[ps[i]] + 10.0) for i in range(10)]
        CLUTCH_15[m, 0] = (sum(clv[:5]) - sum(clv[5:])) * 0.2
        CLUTCH_15[m, 1] = max(clv[:5]) - max(clv[5:])
        CLUTCH_15[m, 2] = min(clv[:5]) - min(clv[5:])
        CLUTCH_15[m, 3] = (sum(ahv[:5]) - sum(ahv[5:])) * 0.2
        CLUTCH_15[m, 4] = min(ahv[:5]) - min(ahv[5:])
        CLUTCH_15[m, 5] = (sum(bhv[:5]) - sum(bhv[5:])) * 0.2
        CLUTCH_15[m, 6] = max(bhv[:5]) - max(bhv[5:])
        CLUTCH_15[m, 7] = (sum(wkv[:5]) - sum(wkv[5:])) * 0.2

        # Tempo, FB/FT, vision swap, player*champ tempo
        fbv = [(fb_sum[ps[i]] + 12.0 * 0.10) / (ew_cnt[ps[i]] + 12.0) for i in range(10)]
        ftv = [(ft_sum[ps[i]] + 12.0 * 0.10) / (ew_cnt[ps[i]] + 12.0) for i in range(10)]
        swv = [(sw_sum[ps[i]] + 10.0 * 0.79) / (ew_cnt[ps[i]] + 10.0) for i in range(10)]
        durv = [(dur_sum[ps[i]] + 10.0 * 28.0) / (ew_cnt[ps[i]] + 10.0) - 28.0 for i in range(10)]
        pbias = [
            ((wr_l_sum[ps[i]] + 6.0 * 0.5) / (wr_l_n[ps[i]] + 12.0))
            - ((wr_e_sum[ps[i]] + 6.0 * 0.5) / (wr_e_n[ps[i]] + 12.0))
            for i in range(10)
        ]
        cbias = [
            ((cl_l_win[cls[i]] + 10.0 * 0.5) / (cl_l_n[cls[i]] + 20.0))
            - ((cl_e_win[cls[i]] + 10.0 * 0.5) / (cl_e_n[cls[i]] + 20.0))
            for i in range(10)
        ]
        TEMPO_OBJ[m, 0] = sum(fbv[:5]) - sum(fbv[5:])
        TEMPO_OBJ[m, 1] = (fbv[1] + fbv[4]) - (fbv[6] + fbv[9])  # JG + SUP FB (slots: BOT,JG,MID,TOP,SUP)
        TEMPO_OBJ[m, 2] = sum(ftv[:5]) - sum(ftv[5:])
        TEMPO_OBJ[m, 3] = (ftv[0] + ftv[3]) - (ftv[5] + ftv[8])  # BOT + TOP FT
        TEMPO_OBJ[m, 4] = sum(swv[:5]) - sum(swv[5:])
        TEMPO_OBJ[m, 5] = (sum(durv[:5]) - sum(durv[5:])) * 0.2
        TEMPO_OBJ[m, 6] = (sum(pbias[:5]) - sum(pbias[5:])) * 0.2
        TEMPO_OBJ[m, 7] = (sum(cbias[:5]) - sum(cbias[5:])) * 0.2
        TEMPO_OBJ[m, 8] = max(cbias[:5]) - max(cbias[5:])
        TEMPO_OBJ[m, 9] = (sum(pbias[i] * cbias[i] for i in range(5)) - sum(pbias[5 + i] * cbias[5 + i] for i in range(5))) * 0.2

        # Mid econ 15->20 + dmg-gold eff
        srv = [sr_sum[ps[i]] / (sr_w[ps[i]] + 10.0) for i in range(10)]
        dgev = [dg_sum[ps[i]] / (ew_cnt[ps[i]] + 10.0) for i in range(10)]
        ECON_MID[m, 0] = (sum(srv[:5]) - sum(srv[5:])) * 0.2
        ECON_MID[m, 1] = max(srv[:5]) - max(srv[5:])
        ECON_MID[m, 2] = (sum(dgev[:5]) - sum(dgev[5:])) * 0.2
        ECON_MID[m, 3] = (dgev[0] + dgev[2]) - (dgev[5] + dgev[7])  # ADC(0) + MID(2) dmg-gold eff

        # ---- 2. Actualizar estados estrictamente DESPUES ----
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
        fb_row = fb_l[m]
        ft_row = ft_l[m]
        sw_row = sw_l[m]
        gs_row = gs_l[m]
        dge_row = dge_l[m]
        sd_row = sd_l[m]
        sv_row = sv_l[m]

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

            # Clutch
            if v15:
                c_res = (yb - p15) * sgn
                cl_sum[p] = cl_sum[p] * dec20 + c_res
                cl_w[p] = cl_w[p] * dec20 + 1.0
                if gs_row[slot] < 0.20:
                    wk_sum[p] = wk_sum[p] * dec20 + c_res
                    wk_w[p] = wk_w[p] * dec20 + 1.0
                dg_team = dg15_m * sgn
                if dg_team > 1000.0:
                    ah_win[p] = ah_win[p] * dec25 + win_p
                    ah_n[p] = ah_n[p] * dec25 + 1.0
                elif dg_team < -1000.0:
                    bh_win[p] = bh_win[p] * dec25 + win_p
                    bh_n[p] = bh_n[p] * dec25 + 1.0

            # Tempo / FB / FT / Swap / Dmg-Gold eff
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

    out = {
        "KALMAN_OUT": KALMAN_OUT,
        "KALMAN_MRG": KALMAN_MRG,
        "CLUTCH_15": CLUTCH_15,
        "TEMPO_OBJ": TEMPO_OBJ,
        "ECON_MID": ECON_MID,
    }
    np.savez_compressed(CACHE, nm=np.array([NM]), **out)
    print(f"Caché gemini_g11.npz lista en {time.time() - t0:.1f}s", flush=True)
    return out


def main():
    t0 = time.time()
    D = cargar()
    ms = D["master"]
    G = construir_cache_g11(D)
    REF = extras_referencia(D)
    mpr_path = REPO / "data" / "analysis" / "brainstorm" / "gemini_mpr.npz"
    MPR = np.load(mpr_path)["MATCHUPS"].astype(np.float32) if mpr_path.exists() else np.zeros((D["NM"], 0), np.float32)
    BASE = np.c_[REF, MPR].astype(np.float32)

    pc_path = REPO / "data" / "analysis" / "brainstorm" / "pred_combinado.npz"
    if pc_path.exists():
        PC = np.load(pc_path)
        P = {"ref: Escalón 3 (0,6663)": (PC["o2"], PC["v2"])}
    else:
        P = {"ref: Escalón 3": evaluar(D, lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=REF, extra_post=False))}

    g08_path = REPO / "data" / "analysis" / "brainstorm" / "pred_g08.npz"
    if g08_path.exists() and "po_log4" in np.load(g08_path).files:
        G8 = np.load(g08_path)
        P["ref4: + MATCHUPS"] = (G8["po_log4"], G8["pv_log4"])
    else:
        print(f"Evaluando Escalón 3 + MATCHUPS ({time.time()-t0:.1f}s)...", flush=True)
        P["ref4: + MATCHUPS"] = evaluar(D, lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=BASE, extra_post=False))

    for k in ["KALMAN_OUT", "KALMAN_MRG", "CLUTCH_15", "TEMPO_OBJ", "ECON_MID"]:
        print(f"Evaluando +{k} s/Esc4 ({time.time()-t0:.1f}s)...", flush=True)
        X = np.c_[BASE, G[k]].astype(np.float32)
        P[f"+ {k} (s/Esc4)"] = evaluar(D, lambda tr, te, M=X: ajustar_predecir(D, tr[ms[tr]], te, extra=M, extra_post=False))

    print(f"Evaluando todo g11 junto s/Esc4 ({time.time()-t0:.1f}s)...", flush=True)
    X_all = np.c_[BASE, G["KALMAN_OUT"], G["KALMAN_MRG"], G["CLUTCH_15"], G["TEMPO_OBJ"], G["ECON_MID"]].astype(np.float32)
    P["+ TODO g11 (s/Esc4)"] = evaluar(D, lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=X_all, extra_post=False))

    print("\n--- Contra ref4: Escalón 3 + MATCHUPS ---")
    mostrar(comparar(D, P, "ref4: + MATCHUPS"))
    print("\n--- Contra ref: Escalón 3 (0,6663) ---")
    mostrar(comparar(D, P, "ref: Escalón 3 (0,6663)"))
    print(f"Listo en {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
