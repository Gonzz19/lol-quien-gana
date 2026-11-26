"""Señales causales nunca explotadas en `variables.npz`, `MC/ML` ni `Escalón 6` (`g_20`):

1) `TRINKET_MACRO` (5 cols): disciplina causal de cambio de trinket (`item6`: Oracle `3364` /
   Farsight `3363` vs Yellow `3340`) residualizada contra `(campeón, línea, partida<=20m)` +
   eficiencia de slots (`n_items / gold_k`).
2) `BUILD_ARCH_ADAPT` (8 cols): perfil causal de build (`item0..item5`) aprendido de forma
   estrictamente online (`< t`) por co-ocurrencia con clases Data Dragon (`Fighter/Tank/Mage/
   Assassin/Marksman/Support`) -> desviación de estilo AP/AD/Tank por jugador, riesgo real de
   equipo full-AD vs tanque enemigo, y adaptación defensiva de botas (`3047 Steelcaps` vs AD /
   `3111 Mercs` vs AP).
3) `JUNGLA_Y_AGG` (6 cols): `neutral_minions_killed/min` y ratio `neutral/(neutral+lane)` en JG
   y robo/tempo de campamentos en laners + `early_aggression_10` residual del JUGADOR (solo existía
   `perf_early_aggression_10` de campeón en `variables.npz`!).
4) `BAYES_TRUESKILL` (8 cols): sistema Bayesiano dual `(mu, sigma^2)` con ruido de proceso por
   días inactivos `q * dt`, ganancia proporcional a `sigma^2`, objetivo residualizado respecto al
   prior causal del draft, y Lobby-Pull puro separado para detectar overperformers/smurfs
   (`mu_S - mu_L`) y asimetría de incertidumbre.

Todo evaluado sobre Escalón 6 sin mapa (`con_mapa=False`), con L2 diferencial (`g_14`),
chequeo ciego `Diamond-V` y placebo permutado.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

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

CACHE = REPO / "data" / "analysis" / "brainstorm" / "gemini_g20.npz"
CSV_PATH = REPO / "data" / "splits" / "train" / "full_dataset.csv"
CHAMP_CSV = REPO / "data" / "ddragon" / "16.17.1" / "champions.csv"
TAGS = ["Fighter", "Tank", "Mage", "Assassin", "Marksman", "Support"]
LANES = ["BOTTOM", "JUNGLE", "MIDDLE", "TOP", "UTILITY"]


def _cargar_champ_tags(champ_ids_arr: np.ndarray):
    ch = pd.read_csv(CHAMP_CSV, usecols=["champion_id", "tags"])
    c2v = {}
    for cid, tg in zip(ch["champion_id"].astype(int), ch["tags"].astype(str)):
        v = np.zeros(6, dtype=np.float32)
        for i, tname in enumerate(TAGS):
            if tname in tg:
                v[i] = 1.0
        s = v.sum()
        if s > 0:
            v /= s
        c2v[cid] = v
    uniq = np.unique(champ_ids_arr)
    cid_to_idx = {int(c): i for i, c in enumerate(uniq)}
    T_uniq = np.stack([c2v.get(int(c), np.full(6, 1.0 / 6.0, dtype=np.float32)) for c in uniq])
    return cid_to_idx, T_uniq


def construir_cache(D: dict) -> dict:
    if CACHE.exists():
        print(f"Usando caché existente: {CACHE}")
        z = np.load(CACHE)
        return {k: z[k] for k in z.files}

    t0 = time.time()
    print("Construyendo caché gemini_g20.npz desde full_dataset.csv...")
    Z = D["Z"]
    mids = np.asarray(Z["mids"])
    NM = len(mids)
    mid_to_i = {m: i for i, m in enumerate(mids)}
    lane_to_i = {l: i for i, l in enumerate(LANES)}

    usecols = [
        "match_id", "team_id", "lane", "champion_id",
        "game_duration", "gold_earned", "total_minions_killed",
        "neutral_minions_killed", "early_aggression_10",
        "item0", "item1", "item2", "item3", "item4", "item5", "item6",
    ]
    df = pd.read_csv(CSV_PATH, usecols=usecols)
    df = df[df["match_id"].isin(mid_to_i)].copy()
    mi = df["match_id"].map(mid_to_i).to_numpy(np.int32)
    si = np.where(df["team_id"].to_numpy() == 100, 0, 1).astype(np.int32)
    li = df["lane"].map(lane_to_i).to_numpy(np.int32)

    # Alinear exactamente en (NM, 2, 5)
    g_champ = np.zeros((NM, 2, 5), dtype=np.int32)
    g_dur = np.zeros(NM, dtype=np.float32)
    g_gold = np.zeros((NM, 2, 5), dtype=np.float32)
    g_cs = np.zeros((NM, 2, 5), dtype=np.float32)
    g_neu = np.zeros((NM, 2, 5), dtype=np.float32)
    g_eagg = np.full((NM, 2, 5), np.nan, dtype=np.float32)
    g_items = np.zeros((NM, 2, 5, 6), dtype=np.int32)
    g_trink = np.zeros((NM, 2, 5), dtype=np.int32)

    g_champ[mi, si, li] = df["champion_id"].to_numpy(np.int32)
    assert (g_champ == Z["champ"]).all(), "Desalineación champ vs variables.npz!"
    g_dur[mi] = np.maximum(df["game_duration"].to_numpy(np.float32) / 60.0, 5.0)
    g_gold[mi, si, li] = df["gold_earned"].to_numpy(np.float32)
    g_cs[mi, si, li] = df["total_minions_killed"].to_numpy(np.float32)
    g_neu[mi, si, li] = df["neutral_minions_killed"].to_numpy(np.float32)
    g_eagg[mi, si, li] = df["early_aggression_10"].to_numpy(np.float32)
    for k in range(6):
        g_items[mi, si, li, k] = df[f"item{k}"].fillna(0).to_numpy(np.int32)
    g_trink[mi, si, li] = df["item6"].fillna(0).to_numpy(np.int32)
    del df, mi, si, li

    # Jugadores desde base.npz
    players = D["B"]["player"]  # (NM, 2, 5) str
    flat_p = players.ravel()
    _, p_codes = np.unique(flat_p, return_inverse=True)
    p_idx = p_codes.reshape(NM, 2, 5).astype(np.int32)
    NP = int(p_idx.max()) + 1

    # Campeones y tags
    cid_to_idx, T_uniq = _cargar_champ_tags(Z["champ"])
    NC = len(T_uniq)
    c_idx = np.vectorize(lambda c: cid_to_idx[int(c)], otypes=[np.int32])(Z["champ"])
    # Pesos compuestos por campeón: phys, mag, tank
    T_match = T_uniq[c_idx]  # (NM, 2, 5, 6)
    c_phys = T_uniq[:, 0] + T_uniq[:, 3] + T_uniq[:, 4]  # Fighter+Assassin+Marksman
    c_mag = T_uniq[:, 2] + 0.5 * T_uniq[:, 5]            # Mage + 0.5 Support
    c_tank = T_uniq[:, 1] + 0.35 * T_uniq[:, 0]          # Tank + 0.35 Fighter

    # Mapear item IDs a índices compactos
    u_items, inv_it = np.unique(g_items, return_inverse=True)
    it_idx = inv_it.reshape(NM, 2, 5, 6).astype(np.int32)
    NI = len(u_items)
    steel_id = int(np.searchsorted(u_items, 3047)) if 3047 in u_items else -1
    mercs_id = int(np.searchsorted(u_items, 3111)) if 3111 in u_items else -1
    mejais_ids = {int(np.searchsorted(u_items, x)) for x in (3041, 1082) if x in u_items}

    # Baselines causales @15 para TrueSkill (ajustado estrictamente en primeras 15k partidas)
    dg15 = Z["D_d_gold_15"].astype(np.float32) / 1000.0
    dx15 = Z["D_d_xp_15"].astype(np.float32) / 1000.0
    dk15 = Z["D_d_kills_15"].astype(np.float32)
    y = D["y"].astype(np.float32)
    m_pre = np.arange(min(15000, NM))
    ok15 = np.isfinite(dg15[m_pre]) & np.isfinite(dx15[m_pre]) & np.isfinite(dk15[m_pre])
    X15_fit = np.c_[dg15[m_pre][ok15], dx15[m_pre][ok15], dk15[m_pre][ok15]]
    lr15 = LogisticRegression(C=0.1).fit(X15_fit, y[m_pre][ok15])
    dg15_f = np.nan_to_num(dg15, nan=0.0)
    dx15_f = np.nan_to_num(dx15, nan=0.0)
    dk15_f = np.nan_to_num(dk15, nan=0.0)
    z15_all = lr15.intercept_[0] + lr15.coef_[0, 0] * dg15_f + lr15.coef_[0, 1] * dx15_f + lr15.coef_[0, 2] * dk15_f
    p15_all = 1.0 / (1.0 + np.exp(-np.clip(z15_all, -8.0, 8.0)))
    has15 = np.isfinite(dg15) & np.isfinite(dx15) & np.isfinite(dk15)

    t_sec = D["t"].astype(np.int64)
    has_f1 = np.isfinite(D["B"]["lp_abs"])
    lp0 = np.where(has_f1, D["B"]["lp_abs"], 0.0).astype(np.float32)
    # Normalizar LP previo usando solo mu/sd de pre-post? No, scale global aprox (LP/500) causal:
    lp_prior = np.where(has_f1, (lp0 - 400.0) / 400.0, 0.0).astype(np.float32)

    # ---- Acumuladores causales ----
    # 1) Trinket + item eff por (champ, lane, is_short) y por jugador
    # stats tracked (5): swap(3363|3364), oracle(3364), farsight(3363), yellow(3340), items_per_k_gold
    cl_tr_sum = np.zeros((NC, 5, 2, 5), dtype=np.float64)
    cl_tr_n = np.zeros((NC, 5, 2), dtype=np.float64)
    p_tr_res = np.zeros((NP, 5), dtype=np.float32)
    p_tr_n = np.zeros(NP, dtype=np.int32)

    # 2) Build profiles
    item_tag_cnt = np.zeros((NI, 6), dtype=np.float32)
    item_tot_cnt = np.zeros(NI, dtype=np.float32)
    cl_b_sum = np.zeros((NC, 5, 6), dtype=np.float64)
    cl_b_n = np.zeros((NC, 5), dtype=np.float64)
    p_b_res = np.zeros((NP, 6), dtype=np.float32)  # desviación vs (champ, lane)
    p_b_n = np.zeros(NP, dtype=np.int32)
    # Adaptación defensiva de botas & greedy seal
    cl_boot_sum = np.zeros((NC, 5, 3), dtype=np.float64)  # [ad_boot_vs_phys, ap_boot_vs_mag, mejai]
    cl_boot_n = np.zeros((NC, 5, 3), dtype=np.float64)
    p_boot_res = np.zeros((NP, 3), dtype=np.float32)
    p_boot_n = np.zeros((NP, 3), dtype=np.int32)

    # 3) Jungla neutral & early_aggression_10 (4): [neu_pm, neu_share, eagg10, eagg_jg_sup]
    cl_ja_sum = np.zeros((NC, 5, 3), dtype=np.float64)
    cl_ja_n = np.zeros((NC, 5, 3), dtype=np.float64)
    p_ja_res = np.zeros((NP, 3), dtype=np.float32)
    p_ja_n = np.zeros((NP, 3), dtype=np.int32)

    # 4) TrueSkill Bayesiano + Lobby-Pull
    mu_S = np.zeros(NP, dtype=np.float64)
    v_S = np.ones(NP, dtype=np.float64)
    mu_L = np.zeros(NP, dtype=np.float64)
    v_L = np.ones(NP, dtype=np.float64)
    last_t = np.full(NP, -1, dtype=np.int64)
    init_L = np.zeros(NP, dtype=bool)
    cl_wr_w = np.zeros((NC, 5), dtype=np.float64)
    cl_wr_n = np.zeros((NC, 5), dtype=np.float64)

    # Salidas
    F_TRINKET = np.zeros((NM, 5), dtype=np.float32)
    F_BUILD = np.zeros((NM, 8), dtype=np.float32)
    F_JUNGLA = np.zeros((NM, 6), dtype=np.float32)
    F_TRUE = np.zeros((NM, 8), dtype=np.float32)

    PRIOR_B = np.full(6, 1.0 / 6.0, dtype=np.float32)
    k_sh = 10.0
    alpha_ew = 0.12
    q_S, q_L = 0.012, 0.015
    beta2 = 1.0
    t_zero = int(t_sec[0])

    for m in range(NM):
        ps = p_idx[m]          # (2, 5)
        cs = c_idx[m]          # (2, 5)
        tm = int(t_sec[m])
        is_sh = 1 if g_dur[m] < 20.0 else 0

        # ============ 1) LECTURA CAUSAL (< m) ============
        # 1a. TRINKET_MACRO
        n_p = p_tr_n[ps].astype(np.float32)
        w_p = n_p / (n_p + k_sh)
        tr_eff = p_tr_res[ps] * w_p[:, :, None]  # (2, 5, 5)
        tr_eq_mean = tr_eff.mean(axis=1)         # (2, 5)
        sup_jg_or = 0.5 * (tr_eff[:, 1, 1] + tr_eff[:, 4, 1])
        adc_mid_far = 0.5 * (tr_eff[:, 0, 2] + tr_eff[:, 2, 2])
        worst_yel = np.maximum(tr_eff[:, 1, 3], tr_eff[:, 4, 3])
        F_TRINKET[m, 0] = tr_eq_mean[0, 0] - tr_eq_mean[1, 0]
        F_TRINKET[m, 1] = sup_jg_or[0] - sup_jg_or[1]
        F_TRINKET[m, 2] = adc_mid_far[0] - adc_mid_far[1]
        F_TRINKET[m, 3] = worst_yel[0] - worst_yel[1]
        F_TRINKET[m, 4] = tr_eq_mean[0, 4] - tr_eq_mean[1, 4]

        # 1b. BUILD_ARCH_ADAPT
        nb_p = p_b_n[ps].astype(np.float32)
        wb_p = nb_p / (nb_p + 12.0)
        b_dev = p_b_res[ps] * wb_p[:, :, None]  # (2, 5, 6)
        pred_vec = T_match[m] + 1.5 * b_dev     # perfil corregido por build del jugador
        pred_mag = (pred_vec[:, :, 2] + 0.5 * pred_vec[:, :, 5]).sum(axis=1)
        pred_phys = (pred_vec[:, :, 0] + pred_vec[:, :, 3] + pred_vec[:, :, 4]).sum(axis=1)
        pred_tank = (pred_vec[:, :, 1] + 0.35 * pred_vec[:, :, 0]).sum(axis=1)
        dev_mag_team = (b_dev[:, :, 2] + 0.5 * b_dev[:, :, 5]).sum(axis=1)
        dev_tank_team = (b_dev[:, :, 1] + 0.35 * b_dev[:, :, 0]).sum(axis=1)
        dev_leth_team = (b_dev[:, :, 3] + b_dev[:, :, 4]).sum(axis=1)
        # Riesgo real de full-AD contra tanque enemigo
        full_ad_risk = np.maximum(0.0, 1.3 - pred_mag) * pred_tank[::-1]
        full_ap_risk = np.maximum(0.0, 1.3 - pred_phys) * pred_tank[::-1]
        # Adaptación defensiva de botas
        nbt = p_boot_n[ps].astype(np.float32)
        wbt = nbt / (nbt + 8.0)
        bt_eff = p_boot_res[ps] * wbt  # (2, 5, 3)
        bt_mean = bt_eff.mean(axis=1)  # (2, 3)
        F_BUILD[m, 0] = dev_mag_team[0] - dev_mag_team[1]
        F_BUILD[m, 1] = dev_tank_team[0] - dev_tank_team[1]
        F_BUILD[m, 2] = dev_leth_team[0] - dev_leth_team[1]
        F_BUILD[m, 3] = full_ad_risk[0] - full_ad_risk[1]
        F_BUILD[m, 4] = full_ap_risk[0] - full_ap_risk[1]
        F_BUILD[m, 5] = bt_mean[0, 0] - bt_mean[1, 0]  # acero vs AD
        F_BUILD[m, 6] = bt_mean[0, 1] - bt_mean[1, 1]  # mercs vs AP
        F_BUILD[m, 7] = bt_mean[0, 2] - bt_mean[1, 2]  # mejai / dark seal greed

        # 1c. JUNGLA_Y_AGG
        nja = p_ja_n[ps].astype(np.float32)
        wja = nja / (nja + k_sh)
        ja_eff = p_ja_res[ps] * wja  # (2, 5, 3) -> [neu_pm, neu_share, eagg10]
        jg_npm = ja_eff[:, 1, 0]
        jg_nsh = ja_eff[:, 1, 1]
        lan_npm = (ja_eff[:, 0, 0] + ja_eff[:, 2, 0] + ja_eff[:, 3, 0]) / 3.0
        agg_all = ja_eff[:, :, 2].mean(axis=1)
        agg_roam = 0.5 * (ja_eff[:, 1, 2] + ja_eff[:, 4, 2])
        agg_spread = ja_eff[:, :, 2].max(axis=1) - ja_eff[:, :, 2].min(axis=1)
        F_JUNGLA[m, 0] = jg_npm[0] - jg_npm[1]
        F_JUNGLA[m, 1] = jg_nsh[0] - jg_nsh[1]
        F_JUNGLA[m, 2] = lan_npm[0] - lan_npm[1]
        F_JUNGLA[m, 3] = agg_all[0] - agg_all[1]
        F_JUNGLA[m, 4] = agg_roam[0] - agg_roam[1]
        F_JUNGLA[m, 5] = agg_spread[0] - agg_spread[1]

        # 1d. BAYES_TRUESKILL
        # Inicializar lobby MMR en primera observación si hay Foto 1
        for s in (0, 1):
            for l in range(5):
                pi = int(ps[s, l])
                if not init_L[pi]:
                    if has_f1[m, s, l]:
                        mu_L[pi] = float(lp_prior[m, s, l])
                        v_L[pi] = 0.35
                    init_L[pi] = True
                lt = last_t[pi]
                if lt >= 0:
                    dt = min(30.0, max(0.0, (tm - lt) / 86400000.0))
                    v_S[pi] = min(1.0, v_S[pi] + q_S * dt)
                    v_L[pi] = min(1.0, v_L[pi] + q_L * dt)

        mS = mu_S[ps]   # (2, 5)
        vvS = v_S[ps]   # (2, 5)
        mL = mu_L[ps]   # (2, 5)
        V_tot = beta2 + float(vvS.sum()) / 25.0
        dmS = float(mS[0].mean() - mS[1].mean())
        zS = dmS / np.sqrt(1.0 + 0.35 * V_tot)
        dmL = float(mL[0].mean() - mL[1].mean())
        over0 = float((mS[0] - mL[0]).mean())
        over1 = float((mS[1] - mL[1]).mean())
        sdS0 = float(np.sqrt(vvS[0]).mean())
        sdS1 = float(np.sqrt(vvS[1]).mean())

        # Prior causal de draft para residualizar la sorpresa del resultado
        wr_b = (cl_wr_w[cs[0], np.arange(5)] + 12.5) / (cl_wr_n[cs[0], np.arange(5)] + 25.0)
        wr_r = (cl_wr_w[cs[1], np.arange(5)] + 12.5) / (cl_wr_n[cs[1], np.arange(5)] + 25.0)
        delta_draft = float(np.clip(np.sum(wr_b - wr_r) * 2.0, -0.75, 0.75))

        F_TRUE[m, 0] = zS
        F_TRUE[m, 1] = dmS
        F_TRUE[m, 2] = float(mS[0].max() - mS[1].max())
        F_TRUE[m, 3] = float(mS[0].min() - mS[1].min())
        F_TRUE[m, 4] = dmL
        F_TRUE[m, 5] = over0 - over1
        F_TRUE[m, 6] = sdS0 - sdS1
        F_TRUE[m, 7] = zS * dmL

        # ============ 2) ACTUALIZACIÓN POST-PARTIDA ============
        dur = float(g_dur[m])
        tr_k = g_trink[m]
        it_k = it_idx[m]
        n_nz = (g_items[m] > 0).sum(axis=2).astype(np.float32)
        gold_k = np.maximum(g_gold[m] / 1000.0, 1.0)
        it_eff = n_nz / gold_k

        sw_v = np.isin(tr_k, (3363, 3364)).astype(np.float32)
        or_v = (tr_k == 3364).astype(np.float32)
        fa_v = (tr_k == 3363).astype(np.float32)
        yl_v = (tr_k == 3340).astype(np.float32)
        tr_obs = np.stack([sw_v, or_v, fa_v, yl_v, it_eff], axis=2)  # (2, 5, 5)

        # Amenaza física/mágica del enemigo en esta partida
        enem_phys = np.array([float(c_phys[cs[1]].sum()), float(c_phys[cs[0]].sum())], dtype=np.float32)
        enem_mag = np.array([float(c_mag[cs[1]].sum()), float(c_mag[cs[0]].sum())], dtype=np.float32)

        # TrueSkill update
        p_exp = 1.0 / (1.0 + np.exp(-np.clip(dmS + 0.5 * delta_draft, -6.0, 6.0)))
        targ_S = float(y[m]) if not has15[m] else float(0.70 * y[m] + 0.30 * p15_all[m])
        err_S = targ_S - p_exp
        inv_sqrtV = 1.0 / np.sqrt(V_tot)
        lobby_mean_L = float(mL.mean())

        for s in (0, 1):
            sg = 1.0 if s == 0 else -1.0
            opp = 1 - s
            for l in range(5):
                pi = int(ps[s, l])
                ci = int(cs[s, l])

                # 2a. Trinket update
                ncl = cl_tr_n[ci, l, is_sh]
                base_tr = (cl_tr_sum[ci, l, is_sh] + 0.5 * 5.0) / (ncl + 5.0)
                res_tr = tr_obs[s, l] - base_tr
                p_tr_res[pi] = (1.0 - alpha_ew) * p_tr_res[pi] + alpha_ew * res_tr
                p_tr_n[pi] += 1
                cl_tr_sum[ci, l, is_sh] += tr_obs[s, l]
                cl_tr_n[ci, l, is_sh] += 1.0

                # 2b. Build update
                its = it_k[s, l]
                nz_its = [int(its[j]) for j in range(6) if g_items[m, s, l, j] > 0]
                if len(nz_its) > 0:
                    # Perfil causal actual de cada item comprado antes de actualizar item_tag_cnt
                    b_sum = np.zeros(6, dtype=np.float32)
                    for ii in nz_its:
                        b_sum += (item_tag_cnt[ii] + 15.0 * PRIOR_B) / (item_tot_cnt[ii] + 15.0)
                    b_vec = b_sum / len(nz_its)
                    base_b = (cl_b_sum[ci, l] + 10.0 * T_uniq[ci]) / (cl_b_n[ci, l] + 10.0)
                    p_b_res[pi] = (1.0 - 0.12) * p_b_res[pi] + 0.12 * (b_vec - base_b)
                    p_b_n[pi] += 1
                    cl_b_sum[ci, l] += b_vec
                    cl_b_n[ci, l] += 1.0
                    # Actualizar tabla de items con T_uniq[ci]
                    tvec = T_uniq[ci]
                    for ii in nz_its:
                        item_tag_cnt[ii] += tvec
                        item_tot_cnt[ii] += 1.0

                # Boot adaptation
                if steel_id >= 0 and enem_phys[s] >= 2.8:
                    has_st = 1.0 if steel_id in its else 0.0
                    b0 = (cl_boot_sum[ci, l, 0] + 1.5) / (cl_boot_n[ci, l, 0] + 5.0)
                    p_boot_res[pi, 0] = (1.0 - alpha_ew) * p_boot_res[pi, 0] + alpha_ew * (has_st - b0)
                    p_boot_n[pi, 0] += 1
                    cl_boot_sum[ci, l, 0] += has_st
                    cl_boot_n[ci, l, 0] += 1.0
                if mercs_id >= 0 and enem_mag[s] >= 2.2:
                    has_mc = 1.0 if mercs_id in its else 0.0
                    b1 = (cl_boot_sum[ci, l, 1] + 1.5) / (cl_boot_n[ci, l, 1] + 5.0)
                    p_boot_res[pi, 1] = (1.0 - alpha_ew) * p_boot_res[pi, 1] + alpha_ew * (has_mc - b1)
                    p_boot_n[pi, 1] += 1
                    cl_boot_sum[ci, l, 1] += has_mc
                    cl_boot_n[ci, l, 1] += 1.0
                if mejais_ids:
                    has_mj = 1.0 if any(x in mejais_ids for x in its) else 0.0
                    b2 = (cl_boot_sum[ci, l, 2] + 0.25) / (cl_boot_n[ci, l, 2] + 5.0)
                    p_boot_res[pi, 2] = (1.0 - alpha_ew) * p_boot_res[pi, 2] + alpha_ew * (has_mj - b2)
                    p_boot_n[pi, 2] += 1
                    cl_boot_sum[ci, l, 2] += has_mj
                    cl_boot_n[ci, l, 2] += 1.0

                # 2c. Jungla neutral & early_aggression_10
                npm = float(g_neu[m, s, l]) / dur
                tot_cs = float(g_cs[m, s, l] + g_neu[m, s, l])
                nsh = float(g_neu[m, s, l]) / tot_cs if tot_cs > 0 else 0.0
                b_npm = (cl_ja_sum[ci, l, 0] + 5.0 * (5.0 if l == 1 else 0.15)) / (cl_ja_n[ci, l, 0] + 5.0)
                p_ja_res[pi, 0] = (1.0 - alpha_ew) * p_ja_res[pi, 0] + alpha_ew * (npm - b_npm)
                p_ja_n[pi, 0] += 1
                cl_ja_sum[ci, l, 0] += npm
                cl_ja_n[ci, l, 0] += 1.0

                b_nsh = (cl_ja_sum[ci, l, 1] + 5.0 * (0.8 if l == 1 else 0.02)) / (cl_ja_n[ci, l, 1] + 5.0)
                p_ja_res[pi, 1] = (1.0 - alpha_ew) * p_ja_res[pi, 1] + alpha_ew * (nsh - b_nsh)
                p_ja_n[pi, 1] += 1
                cl_ja_sum[ci, l, 1] += nsh
                cl_ja_n[ci, l, 1] += 1.0

                ea = g_eagg[m, s, l]
                if np.isfinite(ea):
                    b_ea = (cl_ja_sum[ci, l, 2] + 20.0) / (cl_ja_n[ci, l, 2] + 5.0)
                    p_ja_res[pi, 2] = (1.0 - alpha_ew) * p_ja_res[pi, 2] + alpha_ew * (float(ea) - b_ea)
                    p_ja_n[pi, 2] += 1
                    cl_ja_sum[ci, l, 2] += float(ea)
                    cl_ja_n[ci, l, 2] += 1.0

                # 2d. TrueSkill & Lobby-Pull update
                vs = v_S[pi]
                mu_S[pi] += sg * 0.28 * (vs * inv_sqrtV) * err_S
                v_S[pi] = max(0.08, vs * (1.0 - 0.18 * vs / V_tot))

                # Lobby pull usando solo los otros 9 jugadores
                other_mean = (10.0 * lobby_mean_L - mL[s, l]) / 9.0
                vl = v_L[pi]
                gain_l = vl / (vl + 0.35)
                mu_L[pi] += 0.35 * gain_l * (other_mean - mu_L[pi])
                v_L[pi] = max(0.06, vl * (1.0 - 0.20 * gain_l))
                last_t[pi] = tm

                # Draft WR causal
                cl_wr_w[ci, l] += (1.0 if y[m] == (1.0 if s == 0 else 0.0) else 0.0)
                cl_wr_n[ci, l] += 1.0

    # Normalizar F_* sin NaN
    for arr in (F_TRINKET, F_BUILD, F_JUNGLA, F_TRUE):
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    out = {
        "TRINKET_MACRO": F_TRINKET,
        "BUILD_ARCH_ADAPT": F_BUILD,
        "JUNGLA_Y_AGG": F_JUNGLA,
        "BAYES_TRUESKILL": F_TRUE,
    }
    np.savez_compressed(CACHE, **out)
    print(f"Caché gemini_g20.npz lista en {time.time() - t0:.1f}s")
    return out


def _eval_l2_dif(D, REF6, extra=None, C=0.02, g_xd=0.5, g_dr=1.5, g_hi=3.0):
    ms = D["master"]
    Xd, post = D["Xd"], D["post"]
    tot = REF6 if extra is None else np.c_[REF6, extra].astype(np.float32)

    def pred_fn(tr, te):
        trm = tr[ms[tr]]
        mu, sd = Xd[trm].mean(0), Xd[trm].std(0) + 1e-6
        ref = trm
        me, se = tot[ref].mean(0), tot[ref].std(0) + 1e-6
        Ptr = sparse.hstack([
            D["DRAFT"][trm] * g_dr,
            sparse.csr_matrix(((Xd[trm] - mu) / sd) * g_xd),
            sparse.csr_matrix(((tot[trm] - me) / se) * g_hi),
        ]).tocsr()
        Pte = sparse.hstack([
            D["DRAFT"][te] * g_dr,
            sparse.csr_matrix(((Xd[te] - mu) / sd) * g_xd),
            sparse.csr_matrix(((tot[te] - me) / se) * g_hi),
        ]).tocsr()
        lr = LogisticRegression(C=C, max_iter=4000).fit(Ptr, D["y"][trm])
        return lr.predict_proba(Pte)[:, 1]

    return pred_fn


def main():
    t0 = time.time()
    D = cargar()
    C20 = construir_cache(D)
    REF6 = extras_referencia(D, escalon=6)
    ms = D["master"]

    def solo_m(extra=None):
        tot = REF6 if extra is None else np.c_[REF6, extra].astype(np.float32)
        return lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=tot, extra_post=False, con_mapa=False)

    TODO = np.c_[C20["TRINKET_MACRO"], C20["BUILD_ARCH_ADAPT"], C20["JUNGLA_Y_AGG"], C20["BAYES_TRUESKILL"]].astype(np.float32)
    rng = np.random.default_rng(123)
    perm = rng.permutation(D["NM"])
    TODO_PL = TODO[perm]

    P = {"ref: Escalón 6 (sin mapa)": evaluar(D, solo_m())}
    P["+ TRINKET_MACRO (5)"] = evaluar(D, solo_m(C20["TRINKET_MACRO"]))
    P["+ BUILD_ARCH_ADAPT (8)"] = evaluar(D, solo_m(C20["BUILD_ARCH_ADAPT"]))
    P["+ JUNGLA_Y_AGG (6)"] = evaluar(D, solo_m(C20["JUNGLA_Y_AGG"]))
    P["+ BAYES_TRUESKILL (8)"] = evaluar(D, solo_m(C20["BAYES_TRUESKILL"]))
    P["+ TODO g20 (27)"] = evaluar(D, solo_m(TODO))
    P["+ PLACEBO TODO g20"] = evaluar(D, solo_m(TODO_PL))
    P["+ Escalón 6 L2 dif (g14)"] = evaluar(D, _eval_l2_dif(D, REF6))
    P["+ TODO g20 c/L2 dif (g14)"] = evaluar(D, _eval_l2_dif(D, REF6, TODO))
    mostrar(comparar(D, P, "ref: Escalón 6 (sin mapa)"))

    # Chequeo ciego Diamond-V
    fi_v, dv = D["fi_v"], D["diam_v"]
    ydv = D["y"][dv]
    p_dv_ref = solo_m()(fi_v, dv)
    auc_ref = roc_auc_score(ydv, p_dv_ref)
    boots = rng.integers(0, len(ydv), size=(1500, len(ydv)))
    print(f"\nChequeo Diamond-V ({len(ydv):,} partidas): ref={auc_ref:.4f}")
    tests_dv = [
        ("+ TRINKET_MACRO (5)", solo_m(C20["TRINKET_MACRO"])),
        ("+ BUILD_ARCH_ADAPT (8)", solo_m(C20["BUILD_ARCH_ADAPT"])),
        ("+ JUNGLA_Y_AGG (6)", solo_m(C20["JUNGLA_Y_AGG"])),
        ("+ BAYES_TRUESKILL (8)", solo_m(C20["BAYES_TRUESKILL"])),
        ("+ TODO g20 (27)", solo_m(TODO)),
        ("+ PLACEBO TODO g20", solo_m(TODO_PL)),
        ("+ TODO g20 c/L2 dif", _eval_l2_dif(D, REF6, TODO)),
    ]
    for nom, fn in tests_dv:
        p_dv = fn(fi_v, dv)
        auc_p = roc_auc_score(ydv, p_dv)
        ds = np.array([roc_auc_score(ydv[b], p_dv[b]) - roc_auc_score(ydv[b], p_dv_ref[b]) for b in boots])
        print(f"  {nom:24s} AUC {auc_p:.4f}  Δ {auc_p - auc_ref:+.4f} [{np.percentile(ds, 2.5):+.4f}; {np.percentile(ds, 97.5):+.4f}]")

    print(f"Listo en {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
