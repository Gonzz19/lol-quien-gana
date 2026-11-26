"""Idea G41 (`docs/BRAINSTORM_NOCHE.md`): Descomposición ortogonal Kalman post-15 y de combate 1v1:
Mientras `LANE_KALMAN_15` (`g_30`) captura la habilidad 1v1 de construir ventaja `@15` (`V +0,0010`),
quedan dos ejes causales completamente ortogonales sin filtro Bayesiano por oponente/compañeros:

  1) `MACRO_KALMAN_POST15` (6 cols):
     - En `g_11`, `CLUTCH` promediaba `(y - E[y | @15])` por jugador SIN descontar la fuerza
       macro de sus 4 aliados ni de los 5 rivales, ni ajustar por la magnitud exacta de ventaja.
     - Aquí ajustamos `p15_causal = logit^{-1}(w0 * dg15 + w1 * dx15 + w2 * dk15)` usando pesos
       fijos aprendidos SOLO en `(~val0) & (~post)` (cero fuga), y aplicamos un Filtro de Kalman
       bipartito de 10 jugadores sobre el residuo puro de conversión post-15 `r_conv = y - p15`:
         `innov = r_conv - alpha * (sum_B mu_macro - sum_R mu_macro) * 0.2`
       Midiendo quién realmente cierra o remonta partidas después del min 15 ajustado por lobby:
         - versión estándar (q lento), versión rápida (forma macro), min/max del equipo,
         - y separación asimétrica: habilidad jugando POR DELANTE (`dg15 > +1k`) vs DESDE ATRÁS (`dg15 < -1k`)!
  2) `COMBAT_KALMAN_15` (5 cols):
     - Kalman 1v1 directo por carril sobre la presión de bajas y daño `@15`
       (`kills_15 - deaths_15` + `damage_15 / sd`), descontando el esperado causal `(cA, cB, l)`
       y el rating de combate del rival directo `mu_c[opp]`:
       separa a los jugadores que ganan escaramuzas / kill pressure de los que solo farmean CS pasivo.

Todo 100 % causal (`< m` en orden `(t, m)`), evaluado sobre **Escalón 9 oficial (`0,6739`)** y **Escalón 9 + LK_AGGS (0,6749)**,
con placebo permutado y chequeo ciego `Diamond-V`.
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
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

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

CACHE = REPO / "data" / "analysis" / "brainstorm" / "gemini_g31_macro_comb.npz"
LANES = ["BOTTOM", "JUNGLE", "MIDDLE", "TOP", "UTILITY"]


def _extraer_combate_15(D: dict):
    """Extrae (kills_15 - deaths_15 + 0.5*assists_15) y damage_15 por `(match, team, lane)`."""
    mids = D["Z"]["mids"].astype(str)
    NM = D["NM"]
    pos = pd.Series(np.arange(NM, dtype=np.int32), index=mids)
    idx_l = {l: i for i, l in enumerate(LANES)}

    df = pd.read_csv(
        REPO / "data" / "splits" / "train" / "full_dataset.csv",
        usecols=["match_id", "team_id", "lane", "kills_15", "deaths_15", "assists_15", "damage_15"],
    )
    df = df[df.lane.isin(LANES)].copy()
    df["m"] = df["match_id"].astype(str).map(pos)
    df = df.dropna(subset=["m"]).copy()
    df["m"] = df["m"].astype(np.int32)
    df["l"] = df["lane"].map(idx_l).astype(np.int32)
    df["e"] = (df["team_id"] == 200).astype(np.int32)

    KDA = np.zeros((NM, 2, 5), dtype=np.float32)
    DMG = np.zeros((NM, 2, 5), dtype=np.float32)
    CNT = np.zeros((NM, 2, 5), dtype=np.int8)

    m = df["m"].to_numpy()
    e = df["e"].to_numpy()
    l = df["l"].to_numpy()
    kda = (
        df["kills_15"].fillna(0.0) - df["deaths_15"].fillna(0.0) + 0.5 * df["assists_15"].fillna(0.0)
    ).to_numpy(np.float32)
    dmg = df["damage_15"].fillna(0.0).to_numpy(np.float32)

    np.add.at(KDA, (m, e, l), kda)
    np.add.at(DMG, (m, e, l), dmg)
    np.add.at(CNT, (m, e, l), 1)

    dur = D["Z"]["D_dur_min"]
    ok = (CNT[:, 0, :] == 1) & (CNT[:, 1, :] == 1) & (DMG[:, 0, :] > 0) & (DMG[:, 1, :] > 0) & (dur >= 15.0)[:, None]
    dkda = np.where(ok, KDA[:, 0, :] - KDA[:, 1, :], np.nan).astype(np.float32)
    ddmg = np.where(ok, (DMG[:, 0, :] - DMG[:, 1, :]) / 1000.0, np.nan).astype(np.float32)
    return dkda, ddmg, ok


def construir_macro_y_combate(D: dict, rehacer: bool = False) -> dict[str, np.ndarray]:
    if CACHE.exists() and not rehacer:
        z = np.load(CACHE)
        if int(z["nm"][0]) == D["NM"]:
            return {k: z[k] for k in z.files if k != "nm"}

    t0 = time.time()
    print("[g_31] Construyendo MACRO_KALMAN_POST15 y COMBAT_KALMAN_15 causales (< m)...", flush=True)
    NM = D["NM"]
    Z = D["Z"]
    mids = Z["mids"].astype(str)
    t_arr = D["t"].astype(np.int64)
    y_arr = D["y"].astype(np.int8)
    val0 = D["val0"].astype(bool)
    post = D["post"].astype(bool)
    champ = Z["champ"].reshape(NM, 2, 5)

    dur = Z["D_dur_min"].astype(np.float32)
    dg15 = Z["D_d_gold_15"].astype(np.float32)
    dx15 = Z["D_d_xp_15"].astype(np.float32)
    dk15 = Z["D_d_kills_15"].astype(np.float32)

    # 1) Predictor causal fijo de estado @15 ajustado SOLO en ~val0 & ~post & master
    ok15_eq = np.isfinite(dg15) & np.isfinite(dx15) & np.isfinite(dk15) & (dur >= 15.0)
    X15 = np.c_[
        np.nan_to_num(dg15, 0.0) / 1000.0,
        np.nan_to_num(dx15, 0.0) / 1000.0,
        np.nan_to_num(dk15, 0.0),
    ].astype(np.float64)
    fit15 = (~val0) & (~post) & D["master"] & ok15_eq
    lr15 = LogisticRegression(C=1.0, fit_intercept=False, max_iter=500).fit(X15[fit15], y_arr[fit15])
    p_exp15 = lr15.predict_proba(X15)[:, 1].astype(np.float64)

    # 2) Métricas de combate 1v1 @15 estandarizadas estrictamente en (~val0 & ~post & master)
    dkda, ddmg, ok_c = _extraer_combate_15(D)
    pre_m = (~val0) & (~post) & D["master"]
    sd_k = np.ones(5, dtype=np.float64)
    sd_d = np.ones(5, dtype=np.float64)
    for l in range(5):
        ml_ = pre_m & ok_c[:, l]
        sd_k[l] = max(0.5, float(np.nanstd(dkda[ml_, l])))
        sd_d[l] = max(0.2, float(np.nanstd(ddmg[ml_, l])))
    zc15 = np.where(ok_c, 0.55 * (dkda / sd_k[None, :]) + 0.45 * (ddmg / sd_d[None, :]), 0.0).astype(np.float64)

    players = D["B"]["player"].reshape(NM, 10).astype(str)
    uniq_p, p_inv = np.unique(players.reshape(-1), return_inverse=True)
    pid_mat = p_inv.reshape(NM, 10).tolist()
    n_p = len(uniq_p)

    uc, c_inv = np.unique(champ.reshape(-1), return_inverse=True)
    c_mat = c_inv.reshape(NM, 2, 5).tolist()
    nc = len(uc)

    cl_s = np.zeros((nc, 5), dtype=np.float64)
    cl_n = np.zeros((nc, 5), dtype=np.float64)
    pr_s = np.zeros((5, nc, nc), dtype=np.float64)
    pr_n = np.zeros((5, nc, nc), dtype=np.float64)
    K_CL, K_PR = 20.0, 15.0

    # Estados Kalman Macro Bipartito (conversión post-15 y - p_exp15)
    mu_mac = [0.0] * n_p
    var_mac = [0.04] * n_p
    mu_mac_f = [0.0] * n_p
    var_mac_f = [0.05] * n_p
    mu_ahead = [0.0] * n_p
    var_ahead = [0.04] * n_p
    mu_behind = [0.0] * n_p
    var_behind = [0.04] * n_p

    # Varianza del residuo (y - p15) ~ p*(1-p) ~ 0.16; contribución de 10 jugadores -> obs_var ~ 0.25
    OBS_M = 0.25
    QM_S, MIN_MS = 0.0008, 0.004
    QM_F, MIN_MF = 0.0030, 0.008

    # Estados Kalman Combate 1v1 @15
    mu_cb = [0.0] * n_p
    var_cb = [0.36] * n_p
    OBS_C = 0.85
    QC_S, MIN_CS = 0.008, 0.04

    MACRO_POST15 = np.zeros((NM, 6), dtype=np.float32)
    COMBAT_15 = np.zeros((NM, 5), dtype=np.float32)

    order = np.lexsort((mids, t_arr))
    ok15_l = ok15_eq.tolist()
    p15_l = p_exp15.tolist()
    dg15_l = np.nan_to_num(dg15, 0.0).tolist()
    okc_l = ok_c.tolist()
    zc_l = zc15.tolist()

    for m in order:
        ps = pid_mat[m]
        cb, cr = c_mat[m][0], c_mat[m][1]
        yb = float(y_arr[m])

        # ---- 1. Leer estados previos (< m) ----
        mb = [mu_mac[ps[i]] for i in range(5)]
        mr = [mu_mac[ps[5 + i]] for i in range(5)]
        mfb = [mu_mac_f[ps[i]] for i in range(5)]
        mfr = [mu_mac_f[ps[5 + i]] for i in range(5)]
        ahb = [mu_ahead[ps[i]] for i in range(5)]
        ahr = [mu_ahead[ps[5 + i]] for i in range(5)]
        bhb = [mu_behind[ps[i]] for i in range(5)]
        bhr = [mu_behind[ps[5 + i]] for i in range(5)]

        MACRO_POST15[m, 0] = (sum(mb) - sum(mr)) * 0.2
        MACRO_POST15[m, 1] = min(mb) - min(mr)
        MACRO_POST15[m, 2] = max(mb) - max(mr)
        MACRO_POST15[m, 3] = (sum(mfb) - sum(mfr)) * 0.2
        MACRO_POST15[m, 4] = (sum(ahb) - sum(ahr)) * 0.2
        MACRO_POST15[m, 5] = (sum(bhb) - sum(bhr)) * 0.2

        cb_r = [mu_cb[ps[l]] for l in range(5)]
        cr_r = [mu_cb[ps[5 + l]] for l in range(5)]
        vcb = [var_cb[ps[l]] for l in range(5)]
        vcr = [var_cb[ps[5 + l]] for l in range(5)]
        COMBAT_15[m, 0] = (sum(cb_r) - sum(cr_r)) * 0.2
        COMBAT_15[m, 1] = min(cb_r) - min(cr_r)
        COMBAT_15[m, 2] = max(cb_r) - max(cr_r)
        COMBAT_15[m, 3] = sum((cb_r[l] - cr_r[l]) / math.sqrt(vcb[l] + vcr[l] + 0.1) for l in range(5)) * 0.2
        COMBAT_15[m, 4] = (cb_r[1] - cr_r[1])  # presión directa de Jungla @15

        # ---- 2. Actualizar Macro Bipartito post-15 DESPUÉS ----
        if ok15_l[m]:
            res15 = yb - p15_l[m]
            pred_mac = (sum(mb) - sum(mr)) * 0.2
            innov_m = res15 - pred_mac
            tot_vm = OBS_M + (sum(var_mac[ps[i]] for i in range(10))) * 0.04
            for slot in range(10):
                p = ps[slot]
                sgn = 0.2 if slot < 5 else -0.2
                km = (var_mac[p] * sgn) / tot_vm
                mu_mac[p] += km * innov_m
                gain_w = (var_mac[p] * 0.04) / tot_vm
                var_mac[p] = max(MIN_MS, (1.0 - gain_w) * var_mac[p]) + QM_S

            pred_mf = (sum(mfb) - sum(mfr)) * 0.2
            innov_mf = res15 - pred_mf
            tot_vmf = OBS_M + (sum(var_mac_f[ps[i]] for i in range(10))) * 0.04
            for slot in range(10):
                p = ps[slot]
                sgn = 0.2 if slot < 5 else -0.2
                kmf = (var_mac_f[p] * sgn) / tot_vmf
                mu_mac_f[p] += kmf * innov_mf
                gain_wf = (var_mac_f[p] * 0.04) / tot_vmf
                var_mac_f[p] = max(MIN_MF, (1.0 - gain_wf) * var_mac_f[p]) + QM_F

            # Jugando por delante (>+1000g Azul) vs jugando por detrás (<-1000g Azul)
            dg_val = dg15_l[m]
            if dg_val > 1000.0:
                ah_idx, bh_idx = ps[:5], ps[5:]
                sgn_ab = 1.0
            elif dg_val < -1000.0:
                ah_idx, bh_idx = ps[5:], ps[:5]
                sgn_ab = -1.0
            else:
                ah_idx = None

            if ah_idx is not None:
                pred_ab = (sum(mu_ahead[p] for p in ah_idx) - sum(mu_behind[p] for p in bh_idx)) * 0.2
                innov_ab = (sgn_ab * res15) - pred_ab
                tot_vab = OBS_M + 0.04 * (sum(var_ahead[p] for p in ah_idx) + sum(var_behind[p] for p in bh_idx))
                for p in ah_idx:
                    v = var_ahead[p]
                    mu_ahead[p] += ((v * 0.2) / tot_vab) * innov_ab
                    var_ahead[p] = max(MIN_MS, (1.0 - (v * 0.04) / tot_vab) * v) + QM_S
                for p in bh_idx:
                    v = var_behind[p]
                    mu_behind[p] -= ((v * 0.2) / tot_vab) * innov_ab
                    var_behind[p] = max(MIN_MS, (1.0 - (v * 0.04) / tot_vab) * v) + QM_S

        # ---- 3. Actualizar Combate 1v1 @15 DESPUÉS ----
        ocm = okc_l[m]
        zcm = zc_l[m]
        for l in range(5):
            if not ocm[l]:
                continue
            z_obs = zcm[l]
            c1, c2 = cb[l], cr[l]
            exp_ind = (cl_s[c1, l] / (cl_n[c1, l] + K_CL)) - (cl_s[c2, l] / (cl_n[c2, l] + K_CL))
            exp_pair = (pr_s[l, c1, c2] + K_PR * exp_ind) / (pr_n[l, c1, c2] + K_PR)
            y_comb = z_obs - exp_pair

            pb, pr_ = ps[l], ps[5 + l]
            innov_c = y_comb - (mu_cb[pb] - mu_cb[pr_])
            tvc = var_cb[pb] + var_cb[pr_] + OBS_C
            kc_b = var_cb[pb] / tvc
            kc_r = var_cb[pr_] / tvc
            mu_cb[pb] += kc_b * innov_c
            mu_cb[pr_] -= kc_r * innov_c
            var_cb[pb] = max(MIN_CS, (1.0 - kc_b) * var_cb[pb]) + QC_S
            var_cb[pr_] = max(MIN_CS, (1.0 - kc_r) * var_cb[pr_]) + QC_S

            cl_s[c1, l] += z_obs
            cl_n[c1, l] += 1.0
            cl_s[c2, l] -= z_obs
            cl_n[c2, l] += 1.0
            pr_s[l, c1, c2] += z_obs
            pr_n[l, c1, c2] += 1.0
            pr_s[l, c2, c1] -= z_obs
            pr_n[l, c2, c1] += 1.0

    np.savez_compressed(CACHE, nm=np.array([NM]), MACRO_POST15=MACRO_POST15, COMBAT_15=COMBAT_15)
    print(f"[g_31] Caché guardada en {time.time() - t0:.1f}s", flush=True)
    return {"MACRO_POST15": MACRO_POST15, "COMBAT_15": COMBAT_15}


if __name__ == "__main__":
    D = cargar()
    NM, ms = D["NM"], D["master"]
    X9 = extras_referencia(D, escalon=9)
    LK_AGGS = np.load(REPO / "data" / "analysis" / "brainstorm" / "gemini_g30_lk15.npz")["LK_AGGS"]
    X9_LK = np.c_[X9, LK_AGGS].astype(np.float32)

    res = construir_macro_y_combate(D)
    MAC, CMB = res["MACRO_POST15"], res["COMBAT_15"]
    PL = np.c_[MAC, CMB][np.random.default_rng(29).permutation(NM)]

    def ml(X):
        return lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=X, extra_post=False, con_mapa=False)

    VAR = {
        "Escalón 9 oficial": ml(X9),
        "Esc9 + LK_AGGS (4)": ml(X9_LK),
        "Esc9 + MACRO_POST15 (6)": ml(np.c_[X9, MAC]),
        "Esc9 + COMBAT_15 (5)": ml(np.c_[X9, CMB]),
        "Esc9 + LK_AGGS + MACRO": ml(np.c_[X9_LK, MAC]),
        "Esc9 + LK_AGGS + COMBAT": ml(np.c_[X9_LK, CMB]),
        "Esc9 + TODO (LK+MAC+CMB)": ml(np.c_[X9_LK, MAC, CMB]),
        "placebo MAC+CMB s/Esc9": ml(np.c_[X9, PL]),
    }
    P = dict(Parallel(n_jobs=2, backend="threading")(delayed(lambda n: (n, evaluar(D, VAR[n])))(n) for n in VAR))
    mostrar(comparar(D, P, "Escalón 9 oficial"))

    vi = np.flatnonzero(D["val"])
    yv = D["y"][vi]
    print("\n--- LogLoss en V Master+ ---", flush=True)
    for n in VAR:
        pv = P[n][1]
        print(f"  {n:26s}: V AUC {roc_auc_score(yv, pv):.4f} | LL {log_loss(yv, pv):.5f}", flush=True)

    di = np.flatnonzero(D["diam_v"])
    yd = D["y"][di]
    B = np.random.default_rng(1).integers(0, len(di), size=(300, len(di)))
    qd = P["Escalón 9 oficial"][0][di]
    print("\n--- Chequeo ciego Diamond-V (10.506 partidas) ---", flush=True)
    for n in VAR:
        if n == "Escalón 9 oficial":
            continue
        pd_ = P[n][0][di]
        d = np.array([roc_auc_score(yd[b], pd_[b]) - roc_auc_score(yd[b], qd[b]) for b in B])
        print(f"  Diamond-V {n:24s}: {roc_auc_score(yd, pd_):.4f} (Δ {roc_auc_score(yd, pd_) - roc_auc_score(yd, qd):+.4f} [{np.percentile(d, 2.5):+.4f}; {np.percentile(d, 97.5):+.4f}])")
