"""Combina S3_FedAffinity + CB_CLOSE + InGameNetV2 (NoSkip, 5x5 attn) en A2-A4 ONLY (Min 15 y 20, ZERO V).

Compara estrictamente en A2-A4 (bloques forward chunks 4,5,6) contra `5_Blend_LR_MT` y `6_Final_dEns10` de g_04:
  1. `A_0d_g04`          : `[0d (=0c+S3_4cols), 2_MT, 3_MTC1, 4_MTRes]`
  2. `B_4M_NV2_0d`       : `[0d, 2_MT, 3_MTC1, NV2NoSkip(0d + S3 en Xa)]`
  3. `C_5M_NV2_0d`       : `[0d, 2_MT, 3_MTC1, 4_MTRes, NV2NoSkip(0d + S3 en Xa)]`
  4. `D_4M_NV2_LaneProd` : `[0d, 2_MT, 3_MTC1, NV2NoSkip(0d + S3 en Xa + 4 canales slot en lane_b/r)]`
  5. `E_5M_NV2_LaneProd` : ídem con 5 miembros
  + Etapa 2 forward (`d_ens10`) para cada variante.

Uso:
    python -u historial/en_vivo/ingame_g_06f_combo_a24.py
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
import torch
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.analysis.ingame_evaluar import _construir_aux_targets, _fit_logreg_antisym, _logit, _prep_arrays, _sigm  # noqa: E402
from historial.ensamble_en_vivo.ingame_g_01_prior_y_gpu import _cargar_priors  # noqa: E402
from historial.ensamble_en_vivo.ingame_g_03_regimen_y_senales import _carry_conc_feats  # noqa: E402
from historial.en_vivo.ingame_g_06_a24_explore import _blk_ci, _oof_logreg_logits  # noqa: E402
from historial.en_vivo.ingame_g_06b_netv2_a24 import _cb_close_mat, _fit_netv2_gpu  # noqa: E402
from historial.en_vivo.ingame_g_06e_ablate_s3_a24 import _s3_variants  # noqa: E402
from src.models.ingame import TIMELINE_STATS, extract_minute_features  # noqa: E402

IN_DIR = REPO / "data" / "analysis" / "ingame"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
T0 = time.time()
log = lambda s: print(f"[{time.time() - T0:6.1f}s] {s}", flush=True)


def _slot_prod_channels(
    tl_raw: np.ndarray,
    tempo: dict[str, np.ndarray],
    slot_static: np.ndarray,
    p_mat: np.ndarray,
    minute: int,
    tr_idx: np.ndarray,
    idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Canales por slot `(N, 5, 4)` simétricos bajo B<->R: `[z_g, cb*z_g, w_dps*z_g, p_sk*z_g]`."""
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

    cb = np.nan_to_num(tempo["cbias"][idx], nan=0.0).astype(np.float32)
    ss = np.nan_to_num(slot_static[idx], nan=0.0).astype(np.float32)
    w_dps = ss[:, :, :, 13] + 0.5 * ss[:, :, :, 1]
    p_sk = np.nan_to_num(p_mat[idx, :, :, 0], nan=0.0).astype(np.float32)

    ch_b = np.stack([zb, cb[:, 0] * zb, w_dps[:, 0] * zb, p_sk[:, 0] * zb], axis=-1).astype(np.float32)
    ch_r = np.stack([zr, cb[:, 1] * zr, w_dps[:, 1] * zr, p_sk[:, 1] * zr], axis=-1).astype(np.float32)
    return ch_b, ch_r


def main() -> None:
    log(f"Iniciando combo A2-A4 g_06f (Min 15 y 20, ZERO V, {DEV})...")
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
    end_gold_diff = C["end_gold_diff"].astype(np.float32)
    end_kills_diff = C["end_kills_diff"].astype(np.float32)
    exp_lane = {k[4:]: C[k].astype(np.float32) for k in C.files if k.startswith("exp_")}
    exp_trans = {k[3:]: C[k].astype(np.float32) for k in C.files if k.startswith("tr_")}
    comp = {
        "comp_anti": C["comp_anti"].astype(np.float32), "names_anti": list(C["comp_names_anti"]),
        "comp_sym": C["comp_sym"].astype(np.float32), "names_sym": list(C["comp_names_sym"]),
        "vuln_blue": C["vuln_blue"].astype(np.float32), "vuln_red": C["vuln_red"].astype(np.float32),
        "vuln_names": list(C["vuln_names"]), "slot_static": C["slot_static"].astype(np.float32),
    }
    tempo = {k: C[k].astype(np.float32) for k in ("cbias", "cslope", "pbias", "pvol")}
    player_pre = {"p_mat": C["p_mat"].astype(np.float32), "p_cols": list(C["p_cols"])}
    slot_static = C["slot_static"].astype(np.float32)
    p_mat = C["p_mat"].astype(np.float32)
    del C
    gc.collect()

    priors = _cargar_priors(mids, bloque, val, diam_v, L_orig)
    d_ens10 = np.where(np.isfinite(priors["L_ens10"]) & np.isfinite(L_orig), priors["L_ens10"] - L_orig, 0.0).astype(np.float32)
    del priors

    z4 = np.load(IN_DIR / "pred_mt_a4_gpu_92k.npz")
    gi = TIMELINE_STATS.index("gold")
    rows = []

    for mi, m in [(1, 15), (2, 20)]:
        tm = time.time()
        log(f"\n==================== MINUTO {m} (A2-A4 COMBO g_06f) ====================")
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
        f_a4 = extract_minute_features(m, L_orig, tl_raw, gold_resid, exp_lane=exp_lane, exp_trans=exp_trans,
                                       comp=comp, tempo=tempo, player_pre=player_pre,
                                       include_all_stats=True, include_traj=True, include_expected=True,
                                       include_comp_cross=True, include_player=True)

        p2mt, p3mt, p4mt = z4[f"pa_{m}_2_MT_A4_bs512"], z4[f"pa_{m}_3_MT_A4C1_bs512"], z4[f"pa_{m}_4_MT_Res_A4C1_bs512"]
        p5b, p6f = z4[f"pa_{m}_5_Blend_LR_MT"], z4[f"pa_{m}_6_Final_dEns10"]

        c_0d = 0.02 if m == 15 else 0.01
        p_0d = np.full(len(y), np.nan, np.float32)
        p_nv2_0d = np.full(len(y), np.nan, np.float32)
        p_nv2_lp = np.full(len(y), np.nan, np.float32)

        for b, te in folds_a24:
            tr_b = np.flatnonzero(np.isin(chunk, [1, 2, 3, 4, 5, 6]) & (chunk < b) & master & ok_m)
            tr_b = tr_b[np.argsort(t[tr_b], kind="mergesort")]
            ya_b, ma_b, ys_b, ms_b, _ = _construir_aux_targets(m, tr_b, tl_raw, dur_sec, end_gold_diff, end_kills_diff)

            # 0b OOF causal -> cb_close
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
            s3_tr = _s3_variants(tl_raw, tempo, slot_static, p_mat, m, tr_b, tr_b)["S3_4cols"]
            s3_te = _s3_variants(tl_raw, tempo, slot_static, p_mat, m, tr_b, te)["S3_4cols"]

            ext_tr = np.c_[c1_tr, cb_tr, s3_tr].astype(np.float32)
            ext_te = np.c_[c1_te, cb_te, s3_te].astype(np.float32)
            sd_ext = np.maximum(ext_tr.std(0), 1e-6).astype(np.float32)
            ext_s_tr = np.clip(ext_tr / sd_ext, -8, 8).astype(np.float32)
            ext_s_te = np.clip(ext_te / sd_ext, -8, 8).astype(np.float32)

            # 0d = 0c + S3_4cols
            Xa0d_tr = np.c_[ar0b["Xa_tr"], ext_s_tr].astype(np.float32)
            Xa0d_te = np.c_[ar0b["Xa_te"], ext_s_te].astype(np.float32)
            _, z_te_0d, _ = _fit_logreg_antisym(Xa0d_tr, y[tr_b], Xa0d_te, C=c_0d)
            z_oof_0d = _oof_logreg_logits(Xa0d_tr, y[tr_b], t[tr_b], C=c_0d, n_splits=4)
            p_0d[te] = _sigm(z_te_0d)
            del ar0b, Xa0d_tr, Xa0d_te

            # Redes InGameNetV2 (NoSkip) sobre f_a4 + [C1, CB_CLOSE, S3_4cols]
            ar4 = _prep_arrays(f_a4, tr_b, te, L_orig, y)
            ar4["Xa_tr"] = np.c_[ar4["Xa_tr"], ext_s_tr].astype(np.float32)
            ar4["Xa_te"] = np.c_[ar4["Xa_te"], ext_s_te].astype(np.float32)
            p_nv2_0d[te] = _sigm(
                np.mean([_fit_netv2_gpu(m, ar4, y[tr_b], (ya_b, ma_b, ys_b, ms_b), 80 * b + s, 512, 16, z_oof_0d, z_te_0d, use_skip=False) for s in (0, 1)], axis=0)
            )

            # LaneProd: agregar [z_g, cb*z_g, w_dps*z_g, p_sk*z_g] simétrico por slot a lane_b / lane_r
            lb_tr_x, lr_tr_x = _slot_prod_channels(tl_raw, tempo, slot_static, p_mat, m, tr_b, tr_b)
            lb_te_x, lr_te_x = _slot_prod_channels(tl_raw, tempo, slot_static, p_mat, m, tr_b, te)
            pool_lp = np.r_[lb_tr_x.reshape(-1, lb_tr_x.shape[-1]), lr_tr_x.reshape(-1, lr_tr_x.shape[-1])]
            sd_lp = np.maximum(pool_lp.std(axis=0), 1e-6).astype(np.float32)
            ar4["Lb_tr"] = np.concatenate([ar4["Lb_tr"], np.clip(lb_tr_x / sd_lp, -8, 8)], axis=-1).astype(np.float32)
            ar4["Lr_tr"] = np.concatenate([ar4["Lr_tr"], np.clip(lr_tr_x / sd_lp, -8, 8)], axis=-1).astype(np.float32)
            ar4["Lb_te"] = np.concatenate([ar4["Lb_te"], np.clip(lb_te_x / sd_lp, -8, 8)], axis=-1).astype(np.float32)
            ar4["Lr_te"] = np.concatenate([ar4["Lr_te"], np.clip(lr_te_x / sd_lp, -8, 8)], axis=-1).astype(np.float32)
            assert np.isfinite(ar4["Lb_tr"]).all() and np.isfinite(ar4["Lr_tr"]).all() and np.isfinite(ar4["Xa_tr"]).all()
            p_nv2_lp[te] = _sigm(
                np.mean([_fit_netv2_gpu(m, ar4, y[tr_b], (ya_b, ma_b, ys_b, ms_b), 85 * b + s, 512, 16, z_oof_0d, z_te_0d, use_skip=False) for s in (0, 1)], axis=0)
            )
            del ar4; gc.collect()

        del f_a2, f_a4; gc.collect()
        pa_0d, pa_nv2, pa_lp = p_0d[a24_idx], p_nv2_0d[a24_idx], p_nv2_lp[a24_idx]

        def fwd_blend_s2(cols: list[np.ndarray]):
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
            return bo, s2

        for nom, cols in [
            ("A_0d_g04", [pa_0d, p2mt, p3mt, p4mt]),
            ("B_4M_NV2_0d", [pa_0d, p2mt, p3mt, pa_nv2]),
            ("C_5M_NV2_0d", [pa_0d, p2mt, p3mt, p4mt, pa_nv2]),
            ("D_4M_NV2_LaneProd", [pa_0d, p2mt, p3mt, pa_lp]),
            ("E_5M_NV2_LaneProd", [pa_0d, p2mt, p3mt, p4mt, pa_lp]),
        ]:
            bo, s2 = fwd_blend_s2(cols)
            pt5, lo5, hi5, e5 = _blk_ci(ya, bl_mask, boots_b, bo, p5b)
            pt6, lo6, hi6, e6 = _blk_ci(ya, bl_mask, boots_b, s2, p6f)
            print(f"  {nom:18s} vs5: {[round(x,5) for x in e5]} | {pt5:+.5f}[{lo5:+.5f};{hi5:+.5f}]", flush=True)
            print(f"    -> +dEns10 vs6: {[round(x,5) for x in e6]} | {pt6:+.5f}[{lo6:+.5f};{hi6:+.5f}]", flush=True)
            rows.append({"minuto": m, "modelo": nom, "vs5_Bloq": pt5, "vs5_lo": lo5, "vs5_hi": hi5,
                         "vs6_Bloq": pt6, "vs6_lo": lo6, "vs6_hi": hi6})
        pd.DataFrame(rows).to_csv(IN_DIR / "a24_combo_g06f.csv", index=False)
        log(f"Minuto {m} terminado en {time.time() - tm:.1f}s")

    log(f"Listo total en {time.time() - T0:.1f}s")


if __name__ == "__main__":
    main()
