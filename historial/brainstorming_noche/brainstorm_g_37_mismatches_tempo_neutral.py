"""
Brainstorming Noche — Gemini G37 (`brainstorm_g_37_mismatches_tempo_neutral.py`):
Explora 3 ejes ortogonales sobre `Escalón 9 oficial` (`con_mapa=False`, `V 0,6739`, `LL 0,64431`):

1. `INLANE_MISMATCH_PREC` (7 cols, causal desde Xj / MC / mmr_historico):
   - Choque asimétrico en la MISMA línea l: `offrole_B * share_lane_R - offrole_R * share_lane_B`
     y `estreno_B * share_champ_R - estreno_R * share_champ_B` (dos novatos se anulan, pero un
     autofill/estreno frente a un main/OTP se castiga fuerte).
   - Ponderación por precisión (peso `n_tot / (n_tot + 15)` menos promedio sin ponderar) en `MC`
     y `mmr_historico` (separa ventaja de veteranos verificados vs ruido de muestra chica).
   - Eslabón débil (`min_5`), carry estrella (`max_5`) y dispersión (`std_5`) del nivel compuesto
     individual (`MC` + `mmr_historico` + `wr_acum`), no probados aún sobre `Escalón 9`.

2. `TEMPO_SYM_X_DRAFT` (4 cols, strict causal `< m`):
   - `early_skew(c, l) = sum_{< m} sg * (-tanh((dur - 28)/6)) / (n + 50)`: inclinación causal
     de cada campeón×línea a ganar partidas cortas vs largas.
   - `Pace_m`: ritmo/sangre simétrico de los 10 jugadores previo a `m` (`kills_pm + deaths_pm`,
     invariante al intercambio Azul↔Rojo).
   - Interacciones estrictamente antisimétricas: `Pace_m * (Early_B - Early_R)`,
     `(Early_B + Early_R) * HGB_15`, `Pace_m * HGB_15` y divergencia de ritmo `std_5(Early_B) - std_5(Early_R)`.

3. `NEUTRAL_LVL15_CSV` (4 cols, strict causal `< m` desde `full_dataset.csv` no usado en `Xj`):
   - `neutral_minions_killed / dur_min` residual respecto a `(campeón, línea)`: eficiencia de
     farm/invade para Jungla (`l==1`) y robo de campamentos para líneas (`l!=1`).
   - Duelo causal en `level_15` descontando par `(cA, cB, l)` + experiencia previa de nivel `@15`.

Uso:
  python historial/brainstorming_noche/brainstorm_g_37_mismatches_tempo_neutral.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
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

CACHE = REPO / "data" / "analysis" / "brainstorm" / "gemini_g37_misc.npz"
CSV_PATH = REPO / "data" / "splits" / "train" / "full_dataset.csv"


def _construir_inlane_prec(D) -> np.ndarray:
    """Choques asimétricos en la misma línea, precisión por muestra y min/max/std del equipo."""
    NM = D["NM"]
    Xj = D["Xj"]
    cols = D["cols"]
    fi = D["fi_v"]

    C = np.load(REPO / "data" / "analysis" / "brainstorm" / "margen_grupos.npz")
    MC_raw = np.nan_to_num(C["MC"].mean(axis=-1), nan=0.0).astype(np.float64)  # (NM, 2, 5)
    A = np.load(REPO / "data" / "analysis" / "brainstorm" / "mmr_historico.npz")["A"]
    MMR_raw = np.nan_to_num(A[..., 0], nan=0.0).astype(np.float64)  # (NM, 2, 5)

    offrole = np.nan_to_num(Xj[:, :, :, cols.index("offrole")], nan=0.0).astype(np.float64)
    sh_lane = np.nan_to_num(Xj[:, :, :, cols.index("share_lane")], nan=0.0).astype(np.float64)
    estreno = np.nan_to_num(Xj[:, :, :, cols.index("estreno")], nan=0.0).astype(np.float64)
    sh_champ = np.nan_to_num(Xj[:, :, :, cols.index("share_champ")], nan=0.0).astype(np.float64)
    wr_ac = np.nan_to_num(Xj[:, :, :, cols.index("wr_acum")], nan=0.5).astype(np.float64) - 0.5
    ng = np.expm1(np.maximum(np.nan_to_num(Xj[:, :, :, cols.index("logn_tot")], nan=0.0), 0.0))

    # Escala estrictamente aprendida en fi_v (sin fuga a V ni bloques posteriores si escalamos por fi_v)
    mu_mc, sd_mc = MC_raw[fi].mean(), MC_raw[fi].std() + 1e-6
    mu_mmr, sd_mmr = MMR_raw[fi].mean(), MMR_raw[fi].std() + 1e-6
    mu_wr, sd_wr = wr_ac[fi].mean(), wr_ac[fi].std() + 1e-6

    mc_z = (MC_raw - mu_mc) / sd_mc
    mmr_z = (MMR_raw - mu_mmr) / sd_mmr
    wr_z = (wr_ac - mu_wr) / sd_wr
    comp = (mc_z + mmr_z + 0.5 * wr_z) / 2.5  # (NM, 2, 5)

    # 1-2) Asimetría directa en la misma línea l: autofill vs main, estreno vs OTP
    autofill_vs_main = (offrole[:, 0] * sh_lane[:, 1] - offrole[:, 1] * sh_lane[:, 0]).sum(axis=1)
    estreno_vs_otp = (estreno[:, 0] * sh_champ[:, 1] - estreno[:, 1] * sh_champ[:, 0]).sum(axis=1)

    # 3-4) Diferencia entre promedio ponderado por precisión (n/(n+15)) y promedio sin ponderar
    w = ng / (ng + 15.0)
    wn = w / (w.sum(axis=2, keepdims=True) + 1e-6)
    prec_mc = (wn[:, 0] * mc_z[:, 0]).sum(axis=1) - (wn[:, 1] * mc_z[:, 1]).sum(axis=1) - (mc_z[:, 0].mean(1) - mc_z[:, 1].mean(1))
    prec_mmr = (wn[:, 0] * mmr_z[:, 0]).sum(axis=1) - (wn[:, 1] * mmr_z[:, 1]).sum(axis=1) - (mmr_z[:, 0].mean(1) - mmr_z[:, 1].mean(1))

    # 5-7) Eslabón débil (min), carry estrella (max) y dispersión interna (std)
    weak = comp[:, 0].min(axis=1) - comp[:, 1].min(axis=1)
    star = comp[:, 0].max(axis=1) - comp[:, 1].max(axis=1)
    disp = comp[:, 0].std(axis=1) - comp[:, 1].std(axis=1)

    return np.column_stack([autofill_vs_main, estreno_vs_otp, prec_mc, prec_mmr, weak, star, disp]).astype(np.float32)


def _construir_tempo_sym_x_draft(D) -> np.ndarray:
    """Inclinación causal temprana/tardía de campeones × ritmo sangriento simétrico del lobby."""
    NM = D["NM"]
    Z = D["Z"]
    mids = Z["mids"]
    t = D["t"]
    y = D["y"]
    Xj = D["Xj"]
    cols = D["cols"]
    dur = np.nan_to_num(Z["D_dur_min"].astype(np.float64), nan=28.0)
    champ = Z["champ"].reshape(NM, 2, 5)

    HGB = np.load(REPO / "data" / "analysis" / "brainstorm" / "gemini_hgb_lineas.npz")["HGB"]
    hgb_sum = HGB[:, -1].astype(np.float64) if HGB.shape[1] == 6 else HGB.mean(axis=1).astype(np.float64)

    kp = np.nan_to_num(Xj[:, :, :, cols.index("acum_kills_pm")], nan=0.0).reshape(NM, 10).astype(np.float64)
    dp = np.nan_to_num(Xj[:, :, :, cols.index("acum_deaths_pm")], nan=0.0).reshape(NM, 10).astype(np.float64)
    ng = np.expm1(np.maximum(np.nan_to_num(Xj[:, :, :, cols.index("logn_tot")], nan=0.0).reshape(NM, 10), 0.0))
    rel = ng / (ng + 8.0)

    c_uniq = np.unique(champ)
    cmap = {int(c): i for i, c in enumerate(c_uniq)}
    NC = len(c_uniq)
    c_idx = np.vectorize(cmap.__getitem__, otypes=[np.int32])(champ)  # (NM, 2, 5)

    ord_idx = np.lexsort((mids, t))
    lanes5 = np.arange(5, dtype=np.int32)

    cl_n = np.zeros((NC, 5), dtype=np.float64)
    cl_skew = np.zeros((NC, 5), dtype=np.float64)

    # Media/desvío causal de kills+deaths por partida para centrar Pace_m en ~0
    pace_n = 0.0
    pace_sum = 0.0
    pace_sq = 0.0

    out_ord = np.zeros((NM, 4), dtype=np.float32)

    for step, m in enumerate(ord_idx):
        # Pace individual pre-partida (solo jugadores con muestra)
        raw_p = (kp[m] + dp[m])
        w_p = rel[m]
        pace_raw_m = float((raw_p * w_p).sum() / (w_p.sum() + 1e-6))

        if pace_n >= 200:
            mu_p = pace_sum / pace_n
            sd_p = np.sqrt(max(pace_sq / pace_n - mu_p**2, 1e-4))
            pace_z = float(np.clip((pace_raw_m - mu_p) / sd_p, -3.5, 3.5))

            cb = c_idx[m, 0]
            cr = c_idx[m, 1]
            eb = cl_skew[cb, lanes5] / (cl_n[cb, lanes5] + 50.0)
            er = cl_skew[cr, lanes5] / (cl_n[cr, lanes5] + 50.0)

            eb_m, er_m = float(eb.mean()), float(er.mean())
            hm = float(hgb_sum[m])
            out_ord[step] = (
                pace_z * (eb_m - er_m),
                (eb_m + er_m) * hm,
                pace_z * hm,
                float(eb.std() - er.std()),
            )

        # Actualizar estadísticos causalmente DESPUÉS de leer m
        if w_p.sum() >= 3.0:
            pace_n += 1.0
            pace_sum += pace_raw_m
            pace_sq += pace_raw_m**2

        ym = float(y[m])
        fast_score = float(-np.tanh((dur[m] - 28.0) / 6.0))
        # Si s=0 gana juego corto (+1 * +fast), fast champion sube score
        for s, sg in ((0, 2.0 * ym - 1.0), (1, 1.0 - 2.0 * ym)):
            cs = c_idx[m, s]
            np.add.at(cl_n, (cs, lanes5), 1.0)
            np.add.at(cl_skew, (cs, lanes5), sg * fast_score)

    out = np.zeros((NM, 4), dtype=np.float32)
    out[ord_idx] = out_ord
    return out


def _construir_neutral_lvl15_csv(D) -> np.ndarray:
    """Features causales `< m` desde columnas del CSV ausentes en `variables.npz` (neutral_cs y level_15)."""
    NM = D["NM"]
    Z = D["Z"]
    mids = Z["mids"]
    t = D["t"]
    player = D["B"]["player"]  # (NM, 2, 5)
    dur = np.nan_to_num(Z["D_dur_min"].astype(np.float64), nan=28.0)
    champ = Z["champ"].reshape(NM, 2, 5)

    print("Leyendo neutral_minions_killed y level_15 desde train CSV...")
    df = pd.read_csv(
        CSV_PATH,
        usecols=["match_id", "player_id", "neutral_minions_killed", "level_15"],
    ).drop_duplicates(["match_id", "player_id"])

    keys = pd.DataFrame(
        {
            "match_id": np.repeat(mids, 10),
            "player_id": player.ravel(),
        }
    )
    merged = keys.merge(df, on=["match_id", "player_id"], how="left")
    neut = np.nan_to_num(merged["neutral_minions_killed"].to_numpy(np.float64), nan=0.0).reshape(NM, 2, 5)
    lvl15 = merged["level_15"].to_numpy(np.float64).reshape(NM, 2, 5)
    neut_pm = neut / np.maximum(dur[:, None, None], 5.0)

    c_uniq = np.unique(champ)
    cmap = {int(c): i for i, c in enumerate(c_uniq)}
    NC = len(c_uniq)
    c_idx = np.vectorize(cmap.__getitem__, otypes=[np.int32])(champ).reshape(NM, 10)

    pids_flat = player.ravel()
    _, pid_inv = np.unique(pids_flat, return_inverse=True)
    pid_inv = pid_inv.reshape(NM, 10)
    NP = int(pid_inv.max()) + 1

    ord_idx = np.lexsort((mids, t))
    lanes10 = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=np.int32)

    cl_n = np.zeros((NC, 5), dtype=np.float64)
    cl_neut_sum = np.zeros((NC, 5), dtype=np.float64)
    cl_lvl_n = np.zeros((NC, 5), dtype=np.float64)
    cl_lvl_sum = np.zeros((NC, 5), dtype=np.float64)

    # Historial por jugador
    p_jg_n = np.zeros(NP, dtype=np.float64)
    p_jg_res = np.zeros(NP, dtype=np.float64)
    p_lane_n = np.zeros(NP, dtype=np.float64)
    p_lane_neut = np.zeros(NP, dtype=np.float64)
    p_lvl_n = np.zeros(NP, dtype=np.float64)
    p_lvl_res = np.zeros(NP, dtype=np.float64)

    out_ord = np.zeros((NM, 4), dtype=np.float32)

    neut_flat = neut_pm.reshape(NM, 10)
    lvl_flat = lvl15.reshape(NM, 10)

    # Slots en variables.npz: 0=BOTTOM, 1=JUNGLE, 2=MIDDLE, 3=TOP, 4=UTILITY
    for step, m in enumerate(ord_idx):
        ps = pid_inv[m]
        cs = c_idx[m]

        # Leer valores estrictamente previos antes de m
        jg_b = p_jg_res[ps[1]] / (p_jg_n[ps[1]] + 6.0)
        jg_r = p_jg_res[ps[6]] / (p_jg_n[ps[6]] + 6.0)

        ln_res = p_lane_neut[ps] / (p_lane_n[ps] + 8.0)
        ln_mask = np.array([1.0, 0.0, 1.0, 1.0, 1.0])
        steal_b = float((ln_res[:5] * ln_mask).sum() / 4.0)
        steal_r = float((ln_res[5:] * ln_mask).sum() / 4.0)

        lv_res = p_lvl_res[ps] / (p_lvl_n[ps] + 8.0)
        lv_b = float(lv_res[:5].mean())
        lv_r = float(lv_res[5:].mean())
        # Solo laners: 2=MIDDLE, 3=TOP
        lv_solo_b = float((lv_res[2] + lv_res[3]) * 0.5)
        lv_solo_r = float((lv_res[7] + lv_res[8]) * 0.5)

        out_ord[step] = (jg_b - jg_r, steal_b - steal_r, lv_b - lv_r, lv_solo_b - lv_solo_r)

        # Actualizar promedios de campeón y jugador DESPUÉS de leer m
        mu_neut = cl_neut_sum[cs, lanes10] / np.maximum(cl_n[cs, lanes10], 1.0)
        ok_cl = cl_n[cs, lanes10] >= 20
        r_neut = np.where(ok_cl, np.clip(neut_flat[m] - mu_neut, -3.0, 3.0), 0.0)

        for jg_idx in (1, 6):
            if ok_cl[jg_idx]:
                p_jg_n[ps[jg_idx]] += 1.0
                p_jg_res[ps[jg_idx]] += r_neut[jg_idx]
        for li in (0, 2, 3, 4, 5, 7, 8, 9):
            if ok_cl[li]:
                p_lane_n[ps[li]] += 1.0
                p_lane_neut[ps[li]] += r_neut[li]

        np.add.at(cl_n, (cs, lanes10), 1.0)
        np.add.at(cl_neut_sum, (cs, lanes10), neut_flat[m])

        lvm = lvl_flat[m]
        if np.isfinite(lvm).all():
            mu_lv = cl_lvl_sum[cs, lanes10] / np.maximum(cl_lvl_n[cs, lanes10], 1.0)
            ok_lv = cl_lvl_n[cs, lanes10] >= 20
            dlv = np.zeros(10, dtype=np.float64)
            for l in range(5):
                exp_diff = (mu_lv[l] - mu_lv[l + 5]) if (ok_lv[l] and ok_lv[l + 5]) else 0.0
                edge = np.clip((lvm[l] - lvm[l + 5]) - exp_diff, -3.0, 3.0)
                dlv[l] = edge
                dlv[l + 5] = -edge
            p_lvl_n[ps] += 1.0
            p_lvl_res[ps] += dlv
            np.add.at(cl_lvl_n, (cs, lanes10), 1.0)
            np.add.at(cl_lvl_sum, (cs, lanes10), lvm)

    out = np.zeros((NM, 4), dtype=np.float32)
    out[ord_idx] = out_ord
    return out


def _eval_diam(D, extra_list):
    ms = D["master"]
    tr = D["fi"]
    tr_m = tr[ms[tr]]
    te = np.flatnonzero(D["diam_v"])
    p = ajustar_predecir(D, tr_m, te, extra=extra_list, con_mapa=False, peso=None)
    yv = D["y"][te]
    return float(roc_auc_score(yv, p)), float(log_loss(yv, p)), p, yv


def _boot_dif(p, pr, yv, B=200, seed=1):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(yv), size=(B, len(yv)))
    d = np.array([roc_auc_score(yv[s], p[s]) - roc_auc_score(yv[s], pr[s]) for s in idx])
    return float(roc_auc_score(yv, p) - roc_auc_score(yv, pr)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    t0 = time.time()
    D = cargar()
    ref9 = extras_referencia(D, escalon=9)
    print(f"Base + Escalón 9 cargados ({ref9.shape[1]} cols extra) en {time.time() - t0:.1f}s")

    if CACHE.exists():
        C = np.load(CACHE)
        inlane = C["INLANE"]
        tempo = C["TEMPO"]
        neut = C["NEUT"]
        print(f"Caché cargada: {CACHE.name}")
    else:
        ta = time.time()
        inlane = _construir_inlane_prec(D)
        print(f"INLANE calculado: {inlane.shape} en {time.time() - ta:.1f}s (std={np.round(inlane.std(0), 4)})")
        tb = time.time()
        tempo = _construir_tempo_sym_x_draft(D)
        print(f"TEMPO calculado: {tempo.shape} en {time.time() - tb:.1f}s (std={np.round(tempo.std(0), 4)})")
        tc = time.time()
        neut = _construir_neutral_lvl15_csv(D)
        print(f"NEUT calculado: {neut.shape} en {time.time() - tc:.1f}s (std={np.round(neut.std(0), 4)})")
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, INLANE=inlane, TEMPO=tempo, NEUT=neut)
        print(f"Caché guardada en {CACHE.name}")

    rng = np.random.default_rng(37)
    plac_in = inlane[rng.permutation(D["NM"])]
    plac_tp = tempo[rng.permutation(D["NM"])]

    variantes = {
        "Escalón 9 oficial": [(ref9, False)],
        "+ INLANE_PREC (7)": [(ref9, False), (inlane, False)],
        "+ TEMPO_SYM_DRAFT (4)": [(ref9, False), (tempo, False)],
        "+ NEUTRAL_LVL15 (4)": [(ref9, False), (neut, False)],
        "+ INLANE + TEMPO (11)": [(ref9, False), (inlane, False), (tempo, False)],
        "+ TODO G37 (15)": [(ref9, False), (inlane, False), (tempo, False), (neut, False)],
        "+ placebo (11)": [(ref9, False), (plac_in, False), (plac_tp, False)],
    }

    ms = D["master"]
    P = {}
    D_res = {}
    for nombre, ex_list in variantes.items():
        po, pv = evaluar(D, lambda tr, te, _e=ex_list: ajustar_predecir(D, tr[ms[tr]], te, extra=_e, con_mapa=False))
        P[nombre] = (po, pv)
        auc_d, ll_d, pd_v, y_d = _eval_diam(D, ex_list)
        D_res[nombre] = (auc_d, ll_d, pd_v, y_d)

    T = comparar(D, P, ref="Escalón 9 oficial")
    yv = D["y"][D["val"]]
    pd_ref = D_res["Escalón 9 oficial"][2]
    y_d = D_res["Escalón 9 oficial"][3]
    T["LL V"] = [round(log_loss(yv, P[n][1]), 5) for n in T["variante"]]
    T["AUC DiamV"] = [round(D_res[n][0], 4) for n in T["variante"]]
    T["LL DiamV"] = [round(D_res[n][1], 5) for n in T["variante"]]
    dl = []
    for n in T["variante"]:
        d, lo, hi = _boot_dif(D_res[n][2], pd_ref, y_d)
        dl.append(f"{d:+.4f} [{lo:+.4f}; {hi:+.4f}]" if n != "Escalón 9 oficial" else "+0.0000 [ref]")
    T["Δ DiamV (IC)"] = dl

    print("\n=== RESULTADOS G37 SOBRE ESCALÓN 9 OFICIAL (con_mapa=False) ===")
    mostrar(T)


if __name__ == "__main__":
    main()
