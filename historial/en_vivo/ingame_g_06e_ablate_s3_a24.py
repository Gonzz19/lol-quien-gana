"""Ablación rápida y placebo de `S3_FedAffinity` en A2-A4 ONLY (CPU < 1.5 GB, ZERO V/DiamV).

Descompone `S3_FedAffinity` (`Bloq +0.00060/+0.00072` en `0c` y `+0.00018/+0.00020 IC>0` vs `6_Final` con redes cacheadas):
  - `S3a_cbias_z`   : mean_l(cbias * z_gold) azul - rojo
  - `S3b_pbias_z`   : mean_l(pbias * z_gold) azul - rojo
  - `S3c_dps_z`     : mean_l((tirador + 0.5*ps_dano) * z_gold) azul - rojo
  - `S3d_pskill_z`  : mean_l(acum_d_gold_15 * z_gold) azul - rojo
  - `S3p_Placebo`   : permutación por filas de S3(4 cols) dentro de `tr_b` y `te`
  - `S3_dev_only`   : versión centrada intra-equipo (z - mean(z)) para aislar efecto puro de "quién lleva el oro"
  - `S3+wr_v20`     : S3 + mean_l(wr_v20 * z_gold)
  - `S3+S1`         : S3 + S1_LeadSat (min 15 y 20)

Uso:
    python -u historial/en_vivo/ingame_g_06e_ablate_s3_a24.py
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.analysis.ingame_evaluar import _fit_logreg_antisym, _logit, _prep_arrays, _sigm  # noqa: E402
from historial.ensamble_en_vivo.ingame_g_01_prior_y_gpu import _cargar_priors  # noqa: E402
from historial.ensamble_en_vivo.ingame_g_03_regimen_y_senales import _carry_conc_feats  # noqa: E402
from historial.en_vivo.ingame_g_06_a24_explore import _blk_ci, _oof_logreg_logits  # noqa: E402
from historial.en_vivo.ingame_g_06b_netv2_a24 import _cb_close_mat  # noqa: E402
from historial.en_vivo.ingame_g_06d_live_feats_a24 import _extra_live_groups, _select_C_chron  # noqa: E402
from src.models.ingame import TIMELINE_STATS, extract_minute_features  # noqa: E402

IN_DIR = REPO / "data" / "analysis" / "ingame"
T0 = time.time()
log = lambda s: print(f"[{time.time() - T0:6.1f}s] {s}", flush=True)


def _s3_variants(
    tl_raw: np.ndarray,
    tempo: dict[str, np.ndarray],
    slot_static: np.ndarray,
    p_mat: np.ndarray,
    minute: int,
    tr_idx: np.ndarray,
    idx: np.ndarray,
) -> dict[str, np.ndarray]:
    mi = {10: 0, 15: 1, 20: 2}[minute]
    gi = TIMELINE_STATS.index("gold")
    gb = np.nan_to_num(tl_raw[idx, 0, :, mi, gi], nan=0.0).astype(np.float32)
    gr = np.nan_to_num(tl_raw[idx, 1, :, mi, gi], nan=0.0).astype(np.float32)
    g_pool = np.r_[
        np.nan_to_num(tl_raw[tr_idx, 0, :, mi, gi], nan=0.0),
        np.nan_to_num(tl_raw[tr_idx, 1, :, mi, gi], nan=0.0),
    ]
    mu_g = g_pool.mean(axis=0).astype(np.float32)
    sd_g = np.maximum(g_pool.std(axis=0), 1e-4).astype(np.float32)
    zb = (gb - mu_g) / sd_g
    zr = (gr - mu_g) / sd_g
    zdb = zb - zb.mean(axis=1, keepdims=True)
    zdr = zr - zr.mean(axis=1, keepdims=True)

    cb = np.nan_to_num(tempo["cbias"][idx], nan=0.0)
    pb = np.nan_to_num(tempo["pbias"][idx], nan=0.0)
    ss = np.nan_to_num(slot_static[idx], nan=0.0)
    w_dps = ss[:, :, :, 13] + 0.5 * ss[:, :, :, 1]
    p_sk = np.nan_to_num(p_mat[idx, :, :, 0], nan=0.0)
    p_wr = np.nan_to_num(p_mat[idx, :, :, 7], nan=0.0)  # wr_v20

    def d(mat_b, mat_r, u_b, u_r):
        return (np.mean(mat_b * u_b, axis=1) - np.mean(mat_r * u_r, axis=1))[:, None].astype(np.float32)

    s3_full = np.hstack([d(cb[:, 0], cb[:, 1], zb, zr), d(pb[:, 0], pb[:, 1], zb, zr),
                         d(w_dps[:, 0], w_dps[:, 1], zb, zr), d(p_sk[:, 0], p_sk[:, 1], zb, zr)])
    s3_dev = np.hstack([d(cb[:, 0], cb[:, 1], zdb, zdr), d(pb[:, 0], pb[:, 1], zdb, zdr),
                        d(w_dps[:, 0], w_dps[:, 1], zdb, zdr), d(p_sk[:, 0], p_sk[:, 1], zdb, zdr)])
    return {
        "S3a_cbias_z": d(cb[:, 0], cb[:, 1], zb, zr),
        "S3b_pbias_z": d(pb[:, 0], pb[:, 1], zb, zr),
        "S3c_dps_z": d(w_dps[:, 0], w_dps[:, 1], zb, zr),
        "S3d_pskill_z": d(p_sk[:, 0], p_sk[:, 1], zb, zr),
        "S3e_wr_z": d(p_wr[:, 0], p_wr[:, 1], zb, zr),
        "S3_4cols": s3_full,
        "S3_dev_only": s3_dev,
        "S3+wr_v20": np.c_[s3_full, d(p_wr[:, 0], p_wr[:, 1], zb, zr)],
    }


def main() -> None:
    log("Iniciando ablación S3 en A2-A4 (g_06e, Min 15 y 20, ZERO V)...")
    C = np.load(IN_DIR / "ingame_cache.npz", allow_pickle=True)
    mids = C["mids"].astype(str)
    t = C["t"].astype(np.int64)
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
    exp_lane = {k[4:]: C[k].astype(np.float32) for k in C.files if k.startswith("exp_")}
    exp_trans = {k[3:]: C[k].astype(np.float32) for k in C.files if k.startswith("tr_")}
    tempo = {k: C[k].astype(np.float32) for k in ("cbias", "cslope", "pbias", "pvol")}
    slot_static = C["slot_static"].astype(np.float32)
    p_mat = C["p_mat"].astype(np.float32)
    del C; gc.collect()

    priors = _cargar_priors(mids, bloque, val, diam_v, L_orig)
    d_ens10 = np.where(np.isfinite(priors["L_ens10"]) & np.isfinite(L_orig), priors["L_ens10"] - L_orig, 0.0).astype(np.float32)
    del priors

    z4 = np.load(IN_DIR / "pred_mt_a4_gpu_92k.npz")
    gi = TIMELINE_STATS.index("gold")
    rows = []

    for mi, m in [(1, 15), (2, 20)]:
        log(f"\n==================== MINUTO {m} (ABLACIÓN S3) ====================")
        ok_tl = np.isfinite(tl_raw[:, :, :, : mi + 1, :]).all(axis=(1, 2, 3, 4))
        ok_m = (dur_sec > 60.0 * m) & ok_tl & np.isfinite(L_orig)
        folds_a24 = [(b, np.flatnonzero((chunk == b) & master & ok_m)) for b in (4, 5, 6)]
        a24_idx = np.concatenate([te for _, te in folds_a24])
        ya = y[a24_idx]
        bl_mask = [chunk[a24_idx] == b for b in (4, 5, 6)]
        boots_b = [np.random.default_rng(1000 + m).integers(0, int(bm.sum()), (1000, int(bm.sum()))) for bm in bl_mask]
        g_tot = np.nan_to_num((tl_raw[:, 0, :, mi, gi] - tl_raw[:, 1, :, mi, gi]).sum(axis=1), nan=0.0).astype(np.float32)

        f_a2 = extract_minute_features(m, L_orig, tl_raw, gold_resid, exp_lane=exp_lane, exp_trans=exp_trans,
                                       include_all_stats=True, include_traj=True, include_expected=True,
                                       include_comp_cross=False, include_player=False)
        p0b_ref = z4[f"pa_{m}_0b_A2_C1"]
        p2mt, p3mt, p4mt = z4[f"pa_{m}_2_MT_A4_bs512"], z4[f"pa_{m}_3_MT_A4C1_bs512"], z4[f"pa_{m}_4_MT_Res_A4C1_bs512"]
        p5b, p6f = z4[f"pa_{m}_5_Blend_LR_MT"], z4[f"pa_{m}_6_Final_dEns10"]

        fold_data = []
        for b, te in folds_a24:
            tr_b = np.flatnonzero(np.isin(chunk, [1, 2, 3, 4, 5, 6]) & (chunk < b) & master & ok_m)
            tr_b = tr_b[np.argsort(t[tr_b], kind="mergesort")]
            ar0b = _prep_arrays(f_a2, tr_b, te, L_orig, y)
            c1_tr = _carry_conc_feats(tl_raw, m, tr_b, tr_b)
            c1_te = _carry_conc_feats(tl_raw, m, tr_b, te)
            sd_c1 = np.maximum(c1_tr.std(0), 1e-6).astype(np.float32)
            c1s_tr, c1s_te = np.clip(c1_tr / sd_c1, -8, 8), np.clip(c1_te / sd_c1, -8, 8)
            _, z_te_0b, _ = _fit_logreg_antisym(np.c_[ar0b["Xa_tr"], c1s_tr], y[tr_b], np.c_[ar0b["Xa_te"], c1s_te], C=0.03)
            z_oof_0b = _oof_logreg_logits(np.c_[ar0b["Xa_tr"], c1s_tr], y[tr_b], t[tr_b], C=0.03, n_splits=4)
            b0 = float(np.mean(z_oof_0b))
            cb_tr = _cb_close_mat(tempo["cbias"], g_tot, L_orig, z_oof_0b - b0, tr_b)
            cb_te = _cb_close_mat(tempo["cbias"], g_tot, L_orig, z_te_0b - b0, te)
            ext_tr, ext_te = np.c_[c1_tr, cb_tr].astype(np.float32), np.c_[c1_te, cb_te].astype(np.float32)
            sd = np.maximum(ext_tr.std(0), 1e-6).astype(np.float32)
            Xa0c_tr = np.c_[ar0b["Xa_tr"], np.clip(ext_tr / sd, -8, 8)].astype(np.float32)
            Xa0c_te = np.c_[ar0b["Xa_te"], np.clip(ext_te / sd, -8, 8)].astype(np.float32)

            v_tr = _s3_variants(tl_raw, tempo, slot_static, p_mat, m, tr_b, tr_b)
            v_te = _s3_variants(tl_raw, tempo, slot_static, p_mat, m, tr_b, te)
            g_tr = _extra_live_groups(tl_raw, tempo, slot_static, p_mat, m, tr_b, tr_b)
            g_te = _extra_live_groups(tl_raw, tempo, slot_static, p_mat, m, tr_b, te)
            v_tr["S3+S1"] = np.c_[v_tr["S3_4cols"], g_tr["S1_LeadSat"]]
            v_te["S3+S1"] = np.c_[v_te["S3_4cols"], g_te["S1_LeadSat"]]
            rng = np.random.default_rng(900 + m * 10 + b)
            v_tr["S3p_Placebo"] = v_tr["S3_4cols"][rng.permutation(len(tr_b))]
            v_te["S3p_Placebo"] = v_te["S3_4cols"][rng.permutation(len(te))]
            for k in list(v_tr.keys()):
                sg = np.maximum(v_tr[k].std(0), 1e-6).astype(np.float32)
                v_tr[k] = np.clip(v_tr[k] / sg, -8, 8).astype(np.float32)
                v_te[k] = np.clip(v_te[k] / sg, -8, 8).astype(np.float32)
            fold_data.append((b, tr_b, te, Xa0c_tr, Xa0c_te, v_tr, v_te))

        p0c_lr = None
        for nom in ["0c_Ref", "S3a_cbias_z", "S3b_pbias_z", "S3c_dps_z", "S3d_pskill_z", "S3e_wr_z",
                    "S3_4cols", "S3p_Placebo", "S3_dev_only", "S3+wr_v20", "S3+S1"]:
            p_lr = np.full(len(y), np.nan, np.float32)
            for b, tr_b, te, Xa0c_tr, Xa0c_te, v_tr, v_te in fold_data:
                Xtr = Xa0c_tr if nom == "0c_Ref" else np.c_[Xa0c_tr, v_tr[nom]]
                Xte = Xa0c_te if nom == "0c_Ref" else np.c_[Xa0c_te, v_te[nom]]
                _, z_te, _ = _fit_logreg_antisym(Xtr, y[tr_b], Xte, C=0.01 if m == 20 else 0.02)
                p_lr[te] = _sigm(z_te)
            pa = p_lr[a24_idx]
            if nom == "0c_Ref":
                p0c_lr = pa
            pt_lr, lo_lr, hi_lr, _ = _blk_ci(ya, bl_mask, boots_b, pa, p0b_ref)
            pt_0c, lo_0c, hi_0c, e_0c = _blk_ci(ya, bl_mask, boots_b, pa, p0c_lr)

            cols = [pa, p2mt, p3mt, p4mt]
            bo = np.full(len(ya), np.nan, np.float32)
            for bi in range(3):
                bm = bl_mask[bi]
                zte = np.column_stack([_logit(c[bm]) for c in cols])
                if bi == 0:
                    bo[bm] = _sigm(zte.mean(1))
                else:
                    trm = np.isin(chunk[a24_idx], [4 + j for j in range(bi)])
                    lr = LogisticRegression(C=0.5, max_iter=1000).fit(np.column_stack([_logit(c[trm]) for c in cols]), ya[trm])
                    bo[bm] = lr.predict_proba(zte)[:, 1].astype(np.float32)
            s2 = bo.copy()
            for bi in (1, 2):
                bm = bl_mask[bi]
                trm = np.isin(chunk[a24_idx], [4 + j for j in range(bi)])
                me = LogisticRegression(C=0.5, max_iter=1000).fit(np.c_[_logit(bo[trm]), d_ens10[a24_idx[trm]]], ya[trm])
                s2[bm] = me.predict_proba(np.c_[_logit(bo[bm]), d_ens10[a24_idx[bm]]])[:, 1].astype(np.float32)
            pt5, lo5, hi5, _ = _blk_ci(ya, bl_mask, boots_b, bo, p5b)
            pt6, lo6, hi6, _ = _blk_ci(ya, bl_mask, boots_b, s2, p6f)
            print(f"  {nom:14s} | vs0c {[round(x,5) for x in e_0c]} {pt_0c:+.5f}[{lo_0c:+.5f};{hi_0c:+.5f}] | +g04 vs5 {pt5:+.5f}[{lo5:+.5f};{hi5:+.5f}] | vs6 {pt6:+.5f}[{lo6:+.5f};{hi6:+.5f}]", flush=True)
            rows.append({"minuto": m, "modelo": nom, "vs0c": pt_0c, "vs0c_lo": lo_0c, "vs0c_hi": hi_0c,
                         "vs5": pt5, "vs5_lo": lo5, "vs5_hi": hi5, "vs6": pt6, "vs6_lo": lo6, "vs6_hi": hi6})
        del f_a2, fold_data; gc.collect()
        pd.DataFrame(rows).to_csv(IN_DIR / "a24_ablate_s3_g06e.csv", index=False)

    log(f"Listo total en {time.time() - T0:.1f}s")


if __name__ == "__main__":
    main()
