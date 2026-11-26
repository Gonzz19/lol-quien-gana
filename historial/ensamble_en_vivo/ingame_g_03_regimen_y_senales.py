"""Paso (iii.a) + (iii.c) en vivo (CPU liviano < 1 GB RAM, 0 GPU).

Evalúa sobre `92k Master+` (bloques forward A2-A4, V Master+, Diamond-V ciego):
  (iii.a) Corregir mezcla de regímenes pre/post-Foto-1 en el previo:
    - `0_Lorig_A2`: línea base `92k` con `L_oof`
    - `1_Lorig_Reg`: `A2` + `L_orig * is_post` (pendiente propia para post-Foto-1)
    - `2_Esc9_A2`: `92k` con `L_esc9_92k`
    - `3_Esc9_Reg` + placebo `3p`: `A2` + `L_esc9_92k * is_post`
    - `4_Mix_Reg` + placebo `4p`: `L_mix = where(is_post, L_esc9, L_orig)` + `L_mix * is_post`
    - `5a_2Stg_Lorig_dEsc9`, `5b_2Stg_Lorig_dEns10`, `5c_2Stg_dEns10_Decay` + placebo `5p`:
      modelo base in-game de `92k` (`z_Lorig_92k`) + ajuste de 2-3 pesos hacia adelante
      sobre `post` (`A3 <- A2`, `A4 <- A2-A3`, `V/DiamV <- A2-A4`) usando `L_esc9 - L_orig`
      o `L_ens10 - L_orig` (sin dilución con las 78k pre-Foto-1).
  (iii.c) Señales in-game nuevas sin redundancia + placebo estricto por permutación:
    - `C1_CarryConc`: concentración de oro en carries (`BOT+MID`), eficiencia daño/oro,
      `max/min` z-oro por rol intra-equipo (estrictamente en `tr`) y peso de muertes máximas
    - `C2a_LK_AGGS`: Kalman 1v1 intra-carril `LK_AGGS (4 cols)` de `g_30`
    - `C2b_Post15`: `HGB_POST15 (3 cols: pred_xp15, pred_ofin, conv_post15)` + `conv * exp(-|g|/3.5)`
    - `C3_Scale_Accel`: `cslope`/`cbias` × cercanía + aceleración/reversión `10->15->20`

Uso:
    python -u historial/ensamble_en_vivo/ingame_g_03_regimen_y_senales.py
"""
from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.analysis.ingame_evaluar import (  # noqa: E402
    _delta_ci,
    _ece,
    _fit_logreg_antisym,
    _logit,
    _sigm,
)
from historial.ensamble_en_vivo.ingame_g_01_prior_y_gpu import _cargar_priors  # noqa: E402
from src.models.ingame import LANES, MINUTES, TIMELINE_STATS, extract_minute_features  # noqa: E402

IN_DIR = REPO / "data" / "analysis" / "ingame"
BR_DIR = REPO / "data" / "analysis" / "brainstorm"
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def _cov_acc(y: np.ndarray, p: np.ndarray, frac: float) -> float:
    k = max(1, int(frac * len(y)))
    idx = np.argsort(-np.abs(p - 0.5))[:k]
    return float(((p[idx] > 0.5) == y[idx]).mean())


def _permutar_cols(X_tr_block: np.ndarray, seed: int, is_post_only: np.ndarray | None = None) -> np.ndarray:
    """Permuta por filas; si `is_post_only` está dado, permuta solo dentro de `post` manteniendo 0 en `~post`."""
    rng = np.random.default_rng(seed)
    out = X_tr_block.copy()
    if is_post_only is None:
        return out[rng.permutation(len(out))]
    pos = np.flatnonzero(is_post_only)
    out[pos] = out[rng.permutation(pos)]
    return out


def _carry_conc_feats(tl_raw: np.ndarray, minute: int, tr_idx: np.ndarray, all_idx: np.ndarray) -> np.ndarray:
    """5 rasgos antisimétricos nuevos de concentración de oro, daño/oro y weak link."""
    mi = MINUTES.index(minute)
    st = {s: i for i, s in enumerate(TIMELINE_STATS)}
    cur = tl_raw[:, :, :, mi, :]  # (N, 2, 5, 8)
    gb, gr = cur[:, 0, :, st["gold"]], cur[:, 1, :, st["gold"]]
    db, dr = cur[:, 0, :, st["damage"]], cur[:, 1, :, st["damage"]]
    kb, kr = cur[:, 0, :, st["deaths"]], cur[:, 1, :, st["deaths"]]

    # Estandarización por rol calculada exclusivamente con `tr_idx` (azul+rojo)
    g_role_tr = np.r_[gb[tr_idx], gr[tr_idx]]
    mu_r = np.nanmean(g_role_tr, axis=0)
    sd_r = np.nanstd(g_role_tr, axis=0) + 1e-6

    sub_gb, sub_gr = gb[all_idx], gr[all_idx]
    zb = (sub_gb - mu_r) / sd_r
    zr = (sub_gr - mu_r) / sd_r
    max_z = zb.max(axis=1) - zr.max(axis=1)
    min_z = zb.min(axis=1) - zr.min(axis=1)

    c_share_b = (sub_gb[:, 0] + sub_gb[:, 2]) / (sub_gb.sum(axis=1) + 1.0)
    c_share_r = (sub_gr[:, 0] + sub_gr[:, 2]) / (sub_gr.sum(axis=1) + 1.0)
    carry_share = c_share_b - c_share_r

    sub_db, sub_dr = db[all_idx], dr[all_idx]
    dmg_eff = (sub_db.sum(axis=1) / (sub_gb.sum(axis=1) + 5.0)) - (sub_dr.sum(axis=1) / (sub_gr.sum(axis=1) + 5.0))

    sub_kb, sub_kr = kb[all_idx], kr[all_idx]
    death_max = (sub_kb.max(axis=1) / (sub_kb.sum(axis=1) + 1.0)) - (sub_kr.max(axis=1) / (sub_kr.sum(axis=1) + 1.0))

    return np.column_stack([carry_share, dmg_eff, max_z, min_z, death_max]).astype(np.float32)


def _scale_accel_feats(tl_raw: np.ndarray, minute: int, cbias: np.ndarray, cslope: np.ndarray) -> np.ndarray:
    """Interacciones antisimétricas de escalado de composición y aceleración de trayectoria."""
    mi = MINUTES.index(minute)
    st = {s: i for i, s in enumerate(TIMELINE_STATS)}
    g_tot = (tl_raw[:, 0, :, mi, st["gold"]] - tl_raw[:, 1, :, mi, st["gold"]]).sum(axis=1)
    cb_tot = (cbias[:, 0] - cbias[:, 1]).sum(axis=1)
    cs_tot = (cslope[:, 0] - cslope[:, 1]).sum(axis=1)
    parts = [
        cs_tot,
        cs_tot * np.tanh((1.5 - g_tot) / 3.0),
        cb_tot * np.exp(-np.abs(g_tot) / 4.0),
    ]
    if minute >= 15:
        g10 = (tl_raw[:, 0, :, 0, st["gold"]] - tl_raw[:, 1, :, 0, st["gold"]]).sum(axis=1)
        g15 = (tl_raw[:, 0, :, 1, st["gold"]] - tl_raw[:, 1, :, 1, st["gold"]]).sum(axis=1)
        rev_10_15 = -np.tanh(g10 / 2.0) * (g15 - g10)
        parts.append(rev_10_15)
    if minute == 20:
        g20 = g_tot
        accel = (g20 - g15) - (g15 - g10)
        rev_15_20 = -np.tanh(g15 / 2.5) * (g20 - g15)
        parts.extend([accel, rev_15_20])
    return np.column_stack(parts).astype(np.float32)


def main() -> None:
    cache_path = IN_DIR / "ingame_cache.npz"
    l9_path = IN_DIR / "L_esc9_92k.npz"
    if not cache_path.exists() or not l9_path.exists():
        sys.exit("Falta ingame_cache.npz o L_esc9_92k.npz")

    log("Cargando arrays selectivos (< 600 MB RAM)...")
    C = np.load(cache_path, allow_pickle=True)
    mids = C["mids"].astype(str)
    y = C["y"].astype(np.int64)
    chunk = C["chunk"].astype(np.int32)
    bloque = C["bloque"].astype(np.int32)
    master = C["master"].astype(bool)
    val = C["val"].astype(bool)
    diam_v = C["diam_v"].astype(bool)
    dur_sec = C["dur_sec"].astype(np.float32)
    L_orig = C["L_oof"].astype(np.float32)
    tl_raw = C["tl_raw"].astype(np.float32)
    gold_resid = C["gold_resid"].astype(np.float32)
    cbias = C["cbias"].astype(np.float32)
    cslope = C["cslope"].astype(np.float32)
    exp_lane = {k[4:]: C[k].astype(np.float32) for k in C.files if k.startswith("exp_")}
    exp_trans = {k[3:]: C[k].astype(np.float32) for k in C.files if k.startswith("tr_")}
    del C
    gc.collect()

    L9_z = np.load(l9_path)
    assert (L9_z["mids"].astype(str) == mids).all()
    L_esc9 = L9_z["L_esc9_92k"].astype(np.float32)
    del L9_z

    priors = _cargar_priors(mids, bloque, val, diam_v, L_orig)
    L_ens10 = priors["L_ens10"].astype(np.float32)
    L_esc9_post = priors["L_esc9"].astype(np.float32)
    del priors

    is_post = (bloque >= 1).astype(np.float32)
    L_mix = np.where(bloque >= 1, L_esc9, L_orig).astype(np.float32)
    d_esc9_vs_orig = np.where(np.isfinite(L_esc9_post) & np.isfinite(L_orig), L_esc9_post - L_orig, 0.0).astype(np.float32)
    d_ens10_vs_orig = np.where(np.isfinite(L_ens10) & np.isfinite(L_orig), L_ens10 - L_orig, 0.0).astype(np.float32)

    lk_aggs = np.load(BR_DIR / "gemini_g30_lk15.npz")["LK_AGGS"].astype(np.float32)
    hgb_post15 = np.load(BR_DIR / "gemini_hgb_ctx_p15.npz")["HGB_POST15"].astype(np.float32)

    lap_z = np.load(IN_DIR / "pred_ingame.npz")
    lap_best_key = {10: "3_Abl_A2_Stats_Traj_Exp", 15: "10_InGameNet_MultiTask", 20: "10_InGameNet_MultiTask"}
    lap_ref = {mm: {"v_idx": lap_z[f"v_idx_{mm}"], "y_v": lap_z[f"y_v_{mm}"], "p_v": lap_z[f"p_{mm}_{bk}"].astype(np.float32)}
               for mm, bk in lap_best_key.items()}
    del lap_z
    gc.collect()

    stat_i = {s: i for i, s in enumerate(TIMELINE_STATS)}
    filas = []
    saved = {}

    for mi, m in enumerate(MINUTES):
        tm = time.time()
        log(f"\n==================== MINUTO {m} (Régimen + Señales CPU) ====================")
        ok_tl = np.isfinite(tl_raw[:, :, :, : mi + 1, :]).all(axis=(1, 2, 3, 4))
        ok_m = (dur_sec > 60.0 * m) & ok_tl & np.isfinite(L_orig) & np.isfinite(L_esc9)

        v_idx = np.flatnonzero(val & ok_m)
        dv_idx = np.flatnonzero(diam_v & ok_m)
        folds_a24 = [(b, np.flatnonzero((chunk == b) & master & ok_m)) for b in (4, 5, 6)]
        tr_full = np.flatnonzero(np.isin(chunk, [1, 2, 3, 4, 5, 6]) & master & ok_m)
        a24_idx = np.concatenate([te for _, te in folds_a24])
        a34_idx = np.concatenate([te for _, te in folds_a24[1:]])

        rng_bv = np.random.default_rng(300 + m)
        boot_v = rng_bv.integers(0, len(v_idx), size=(1000, len(v_idx)))
        rng_dv = np.random.default_rng(600 + m)
        boot_dv = rng_dv.integers(0, len(dv_idx), size=(1000, len(dv_idx)))
        yv, ydv = y[v_idx], y[dv_idx]
        g_tot_m = np.nan_to_num((tl_raw[:, 0, :, mi, stat_i["gold"]] - tl_raw[:, 1, :, mi, stat_i["gold"]]).sum(axis=1), nan=0.0).astype(np.float32)

        def fit_eval_xa(Xa_all: np.ndarray, dyn_fn=None, cs=(0.01, 0.03, 0.05)):
            """Evalúa `Xa_all` (+ opcional `dyn_fn(tr, idx)`) OOF en folds_a24 y en V/DiamV."""
            def get_pair(tr, te):
                if dyn_fn is None:
                    return Xa_all[tr], Xa_all[te]
                return np.c_[Xa_all[tr], dyn_fn(tr, tr)], np.c_[Xa_all[te], dyn_fn(tr, te)]

            best_C = cs[0]
            if len(cs) > 1:
                best_ll = float("inf")
                for c_try in cs:
                    pa_t = np.full(len(y), np.nan, np.float32)
                    for b, te in folds_a24:
                        tr_b = np.flatnonzero(np.isin(chunk, [1, 2, 3, 4, 5, 6]) & (chunk < b) & master & ok_m)
                        Xtr, Xte = get_pair(tr_b, te)
                        pa_t[te] = _sigm(_fit_logreg_antisym(Xtr, y[tr_b], Xte, C=c_try)[1])
                    ll = log_loss(y[a24_idx], pa_t[a24_idx])
                    if ll < best_ll:
                        best_ll, best_C = ll, c_try

            pa = np.full(len(y), np.nan, np.float32)
            for b, te in folds_a24:
                tr_b = np.flatnonzero(np.isin(chunk, [1, 2, 3, 4, 5, 6]) & (chunk < b) & master & ok_m)
                Xtr, Xte = get_pair(tr_b, te)
                pa[te] = _sigm(_fit_logreg_antisym(Xtr, y[tr_b], Xte, C=best_C)[1])
            te_vd = np.r_[v_idx, dv_idx]
            Xtr_f, Xte_f = get_pair(tr_full, te_vd)
            _, z_vd, _ = _fit_logreg_antisym(Xtr_f, y[tr_full], Xte_f, C=best_C)
            p_vd = _sigm(z_vd)
            del Xtr_f, Xte_f
            gc.collect()
            return pa, p_vd[: len(v_idx)], p_vd[len(v_idx) :], best_C

        pa_m, pv_m, pdv_m = {}, {}, {}

        # Base A2 con distintos previos y regímenes (iii.a)
        f0 = extract_minute_features(m, L_orig, tl_raw, gold_resid, exp_lane=exp_lane, exp_trans=exp_trans,
                                     include_all_stats=True, include_traj=True, include_expected=True,
                                     include_comp_cross=False, include_player=False)
        Xa0 = f0["X_anti"]
        del f0; gc.collect()

        pa_m["0_Lorig_A2"], pv_m["0_Lorig_A2"], pdv_m["0_Lorig_A2"], c_orig = fit_eval_xa(Xa0)

        col_post_orig = (L_orig * is_post)[:, None]
        pa_m["1_Lorig_Reg"], pv_m["1_Lorig_Reg"], pdv_m["1_Lorig_Reg"], c_1r = fit_eval_xa(np.c_[Xa0, col_post_orig])
        pa_m["1p_Lorig_Reg_pl"], pv_m["1p_Lorig_Reg_pl"], pdv_m["1p_Lorig_Reg_pl"], _ = fit_eval_xa(
            np.c_[Xa0, _permutar_cols(col_post_orig, 11 + m, is_post_only=(is_post > 0.5) & ok_m)], cs=(c_1r,)
        )

        Xa_esc9 = Xa0.copy()
        Xa_esc9[:, 0] = L_esc9
        pa_m["2_Esc9_A2"], pv_m["2_Esc9_A2"], pdv_m["2_Esc9_A2"], _ = fit_eval_xa(Xa_esc9)

        col_post_e9 = (L_esc9 * is_post)[:, None]
        pa_m["3_Esc9_Reg"], pv_m["3_Esc9_Reg"], pdv_m["3_Esc9_Reg"], c_3r = fit_eval_xa(np.c_[Xa_esc9, col_post_e9])
        pa_m["3p_Esc9_Reg_pl"], pv_m["3p_Esc9_Reg_pl"], pdv_m["3p_Esc9_Reg_pl"], _ = fit_eval_xa(
            np.c_[Xa_esc9, _permutar_cols(col_post_e9, 21 + m, is_post_only=(is_post > 0.5) & ok_m)], cs=(c_3r,)
        )

        Xa_mix = Xa0.copy()
        Xa_mix[:, 0] = L_mix
        col_post_mix = (L_mix * is_post)[:, None]
        pa_m["4_Mix_Reg"], pv_m["4_Mix_Reg"], pdv_m["4_Mix_Reg"], c_4r = fit_eval_xa(np.c_[Xa_mix, col_post_mix])
        pa_m["4p_Mix_Reg_pl"], pv_m["4p_Mix_Reg_pl"], pdv_m["4p_Mix_Reg_pl"], _ = fit_eval_xa(
            np.c_[Xa_mix, _permutar_cols(col_post_mix, 31 + m, is_post_only=(is_post > 0.5) & ok_m)], cs=(c_4r,)
        )
        del Xa_esc9, Xa_mix; gc.collect()

        # Two-stage: z_92k(Lorig) fijo + ajuste 2-3 params en post (A3<-A2, A4<-A2-A3, V/DiamV<-A2-A4)
        z_a0 = _logit(pa_m["0_Lorig_A2"])
        z_v0 = _logit(pv_m["0_Lorig_A2"])
        z_dv0 = _logit(pdv_m["0_Lorig_A2"])
        d_ens_dec = (d_ens10_vs_orig * np.exp(-np.abs(g_tot_m) / 4.0)).astype(np.float32)

        def eval_two_stage(d_mat: np.ndarray, nome: str):
            pa_e = np.full(len(y), np.nan, np.float32)
            for bi, (_, te) in enumerate(folds_a24):
                if bi == 0:
                    pa_e[te] = pa_m["0_Lorig_A2"][te]
                else:
                    tr_prev = np.concatenate([folds_a24[j][1] for j in range(bi)])
                    X_pr = np.c_[z_a0[tr_prev], d_mat[tr_prev]]
                    X_te = np.c_[z_a0[te], d_mat[te]]
                    me = LogisticRegression(C=0.5, max_iter=1000).fit(X_pr, y[tr_prev])
                    pa_e[te] = me.predict_proba(X_te)[:, 1].astype(np.float32)
            X_a = np.c_[z_a0[a24_idx], d_mat[a24_idx]]
            me_f = LogisticRegression(C=0.5, max_iter=1000).fit(X_a, y[a24_idx])
            pv_e = me_f.predict_proba(np.c_[z_v0, d_mat[v_idx]])[:, 1].astype(np.float32)
            pdv_e = me_f.predict_proba(np.c_[z_dv0, d_mat[dv_idx]])[:, 1].astype(np.float32)
            pa_m[nome], pv_m[nome], pdv_m[nome] = pa_e, pv_e, pdv_e
            if not nome.endswith("_pl"):
                log(f"  {nome} coef(A2-A4): {np.round(me_f.coef_[0], 3)}, b={me_f.intercept_[0]:+.3f}")

        eval_two_stage(d_esc9_vs_orig[:, None], "5a_2Stg_Lorig_dEsc9")
        eval_two_stage(d_ens10_vs_orig[:, None], "5b_2Stg_Lorig_dEns10")
        eval_two_stage(np.c_[d_ens10_vs_orig, d_ens_dec], "5c_2Stg_dEns10_Dec")
        rng_p5 = np.random.default_rng(41 + m)
        d_pl = d_ens10_vs_orig.copy()
        for sl in (folds_a24[0][1], folds_a24[1][1], folds_a24[2][1], v_idx, dv_idx):
            d_pl[sl] = d_ens10_vs_orig[rng_p5.permutation(sl)]
        eval_two_stage(d_pl[:, None], "5p_2Stg_dEns10_pl")

        # (iii.c) Cribado de señales nuevas sobre 0_Lorig_A2 (con C=(0.01,0.03,0.05) + placebos permutados dentro de ok_m)
        pa_m["C1_CarryConc"], pv_m["C1_CarryConc"], pdv_m["C1_CarryConc"], c_c1 = fit_eval_xa(
            Xa0, dyn_fn=lambda tr_i, al_i: _carry_conc_feats(tl_raw, m, tr_i, al_i)
        )
        pa_m["C1p_placebo"], pv_m["C1p_placebo"], pdv_m["C1p_placebo"], _ = fit_eval_xa(
            Xa0, dyn_fn=lambda tr_i, al_i, sd=51 + m: _permutar_cols(_carry_conc_feats(tl_raw, m, tr_i, al_i), sd),
            cs=(c_c1,),
        )

        pa_m["C2a_LK_AGGS"], pv_m["C2a_LK_AGGS"], pdv_m["C2a_LK_AGGS"], c_c2a = fit_eval_xa(np.c_[Xa0, lk_aggs])
        pa_m["C2ap_placebo"], pv_m["C2ap_placebo"], pdv_m["C2ap_placebo"], _ = fit_eval_xa(
            np.c_[Xa0, _permutar_cols(lk_aggs, 61 + m, is_post_only=ok_m)], cs=(c_c2a,)
        )

        post15_ext = np.nan_to_num(
            np.c_[hgb_post15, hgb_post15[:, 2:3] * np.exp(-np.abs(g_tot_m) / 3.5)[:, None]], nan=0.0
        ).astype(np.float32)
        pa_m["C2b_HGBPost15"], pv_m["C2b_HGBPost15"], pdv_m["C2b_HGBPost15"], c_c2b = fit_eval_xa(np.c_[Xa0, post15_ext])
        pa_m["C2bp_placebo"], pv_m["C2bp_placebo"], pdv_m["C2bp_placebo"], _ = fit_eval_xa(
            np.c_[Xa0, _permutar_cols(post15_ext, 71 + m, is_post_only=ok_m)], cs=(c_c2b,)
        )

        sc_ac = np.nan_to_num(_scale_accel_feats(tl_raw, m, cbias, cslope), nan=0.0).astype(np.float32)
        pa_m["C3_ScaleAccel"], pv_m["C3_ScaleAccel"], pdv_m["C3_ScaleAccel"], c_c3 = fit_eval_xa(np.c_[Xa0, sc_ac])
        pa_m["C3p_placebo"], pv_m["C3p_placebo"], pdv_m["C3p_placebo"], _ = fit_eval_xa(
            np.c_[Xa0, _permutar_cols(sc_ac, 81 + m, is_post_only=ok_m)], cs=(c_c3,)
        )

        del Xa0; gc.collect()

        assert np.array_equal(lap_ref[m]["v_idx"], v_idx)
        assert np.array_equal(lap_ref[m]["y_v"], yv)
        p_lap_v = lap_ref[m]["p_v"]
        ref_v0 = pv_m["0_Lorig_A2"]
        ref_dv0 = pdv_m["0_Lorig_A2"]

        for nom, pv in pv_m.items():
            auc_v = roc_auc_score(yv, pv)
            acc_v = float(((pv > 0.5) == yv).mean())
            ll_v = log_loss(yv, pv)
            ece_v = _ece(yv, pv)
            c5, c10, c20 = _cov_acc(yv, pv, 0.05), _cov_acc(yv, pv, 0.10), _cov_acc(yv, pv, 0.20)
            aucs_b = [roc_auc_score(y[te], pa_m[nom][te]) for _, te in folds_a24]
            aucs_0 = [roc_auc_score(y[te], pa_m["0_Lorig_A2"][te]) for _, te in folds_a24]
            d_bl = np.mean([a - b0 for a, b0 in zip(aucs_b, aucs_0)])
            d_a34 = np.mean([a - b0 for a, b0 in zip(aucs_b[1:], aucs_0[1:])])
            n_ok_b = sum(a > b0 for a, b0 in zip(aucs_b, aucs_0))
            pdv = pdv_m[nom]
            auc_dv = roc_auc_score(ydv, pdv)
            acc_dv = float(((pdv > 0.5) == ydv).mean())
            d0, lo0, hi0 = _delta_ci(yv, pv, ref_v0, boot_v)
            dl, lol, hil = _delta_ci(yv, pv, p_lap_v, boot_v)
            ddv, lodv, hidv = _delta_ci(ydv, pdv, ref_dv0, boot_dv)
            print(
                f"  {nom:20s} | bloq {n_ok_b}/3 Δ{d_bl:+.4f} (A34 {d_a34:+.4f}) | "
                f"V {auc_v:.4f}/{acc_v:.2%} LL {ll_v:.4f} ECE {ece_v:.4f} [cov5/10/20 {c5:.1%}/{c10:.1%}/{c20:.1%}] | "
                f"Δorig {d0:+.4f} [{lo0:+.4f};{hi0:+.4f}] | Δlap {dl:+.4f} [{lol:+.4f};{hil:+.4f}] | "
                f"DiamV {auc_dv:.4f}/{acc_dv:.2%} ({ddv:+.4f} [{lodv:+.4f};{hidv:+.4f}])",
                flush=True,
            )
            filas.append({
                "minuto": m, "modelo": nom, "n_V": len(v_idx),
                "AUC_A2": aucs_b[0], "AUC_A3": aucs_b[1], "AUC_A4": aucs_b[2],
                "dBloq_vs_orig": d_bl, "dA34_vs_orig": d_a34, "n_Bloq_pos": n_ok_b,
                "AUC_V": auc_v, "Acc_V": acc_v, "LogLoss_V": ll_v, "ECE_V": ece_v,
                "cov5": c5, "cov10": c10, "cov20": c20,
                "dV_vs_origA2": d0, "lo0": lo0, "hi0": hi0,
                "dV_vs_lapBest": dl, "lol": lol, "hil": hil,
                "AUC_DiamV": auc_dv, "Acc_DiamV": acc_dv,
                "dDV_vs_origA2": ddv, "lodv": lodv, "hidv": hidv,
            })
            saved[f"pv_{m}_{nom}"] = pv
            saved[f"pdv_{m}_{nom}"] = pdv
            saved[f"pa_{m}_{nom}"] = pa_m[nom][a24_idx]
        saved[f"v_idx_{m}"] = v_idx
        saved[f"dv_idx_{m}"] = dv_idx
        saved[f"a24_idx_{m}"] = a24_idx
        saved[f"y_v_{m}"] = yv
        saved[f"y_dv_{m}"] = ydv
        saved[f"y_a24_{m}"] = y[a24_idx]
        pd.DataFrame(filas).to_csv(IN_DIR / "regimen_y_senales_92k.csv", index=False)
        log(f"Minuto {m} terminado en {time.time() - tm:.1f}s")

    np.savez_compressed(IN_DIR / "pred_regimen_y_senales_92k.npz", **saved)
    log(f"Listo total en {time.time() - T0:.1f}s")


if __name__ == "__main__":
    main()
