"""Chequeo único en V + Diamond-V del candidato elegido estrictamente en A2-A4 (Regla 06:43).

Selección previa congelada desde A2-A4 (g_06 .. g_06f, cero lecturas previas de V/DiamV):
  - Minuto 10: sin cambios (`6_Final_dEns10` de g_04, ninguno mejoró en A2-A4).
  - Minuto 15: `D_4M_NV2_LaneProd` (`[0d, 2_MT, 3_MTC1, NV2NoSkip(0d+S3+LaneProd)] + Etapa 2 d_ens10`),
    ganador claro en A2-A4 (`BloqMean +0.00053 [+0.00018; +0.00085]` vs `6_Final_dEns10` con 2 semillas),
    evaluado ahora con 4 semillas `[0,1,2,3]` replicando el presupuesto de g_04 + placebo de filas permutadas.
  - Minuto 20: `A_0d_g04` (`[0d (=0b+CB_CLOSE+S3_4cols), 2_MT, 3_MTC1, 4_MTRes] + Etapa 2 d_ens10`),
    ganador más limpio y parsimonioso en A2-A4 (`BloqMean +0.00020 [+0.00001; +0.00039]`) + placebo permutado.

Uso:
    python -u historial/en_vivo/ingame_g_07_final_v.py
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
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.analysis.ingame_evaluar import _construir_aux_targets, _delta_ci, _ece, _fit_logreg_antisym, _logit, _prep_arrays, _sigm  # noqa: E402
from historial.ensamble_en_vivo.ingame_g_01_prior_y_gpu import _cargar_priors  # noqa: E402
from historial.ensamble_en_vivo.ingame_g_03_regimen_y_senales import _carry_conc_feats, _cov_acc  # noqa: E402
from historial.en_vivo.ingame_g_06_a24_explore import _blk_ci, _oof_logreg_logits  # noqa: E402
from historial.en_vivo.ingame_g_06b_netv2_a24 import InGameNetV2, _cb_close_mat, _fit_netv2_gpu  # noqa: E402
from historial.en_vivo.ingame_g_06e_ablate_s3_a24 import _s3_variants  # noqa: E402
from historial.en_vivo.ingame_g_06f_combo_a24 import _slot_prod_channels  # noqa: E402
from src.models.ingame import TIMELINE_STATS, extract_minute_features  # noqa: E402

IN_DIR = REPO / "data" / "analysis" / "ingame"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
T0 = time.time()
log = lambda s: print(f"[{time.time() - T0:6.1f}s] {s}", flush=True)


def _check_antisym_v2() -> float:
    net = InGameNetV2(d_lane=65, d_anti=250, d_sym=22, d_aux_anti=2, d_aux_sym=1, use_skip=False).eval()
    rng = np.random.default_rng(7)
    lb = torch.tensor(rng.normal(size=(8, 5, 65)).astype(np.float32))
    lr = torch.tensor(rng.normal(size=(8, 5, 65)).astype(np.float32))
    xa = torch.tensor(rng.normal(size=(8, 250)).astype(np.float32))
    xs = torch.tensor(rng.normal(size=(8, 22)).astype(np.float32))
    with torch.no_grad():
        l_br = net(lb, lr, xa, xs)[0]
        l_rb = net(lr, lb, -xa, xs)[0]
        err = float(torch.max(torch.abs(l_br + l_rb - 2.0 * net.b_side)).item())
    assert err < 1e-5, f"Error antisimetria InGameNetV2: {err}"
    return err


def _mcnemar(y: np.ndarray, p1: np.ndarray, p0: np.ndarray) -> tuple[int, int, float]:
    c1 = (p1 > 0.5) == y
    c0 = (p0 > 0.5) == y
    n10 = int(np.sum(c1 & ~c0))
    n01 = int(np.sum(~c1 & c0))
    p = float(binomtest(n10, max(1, n10 + n01), 0.5).pvalue) if (n10 + n01) > 0 else 1.0
    return n10, n01, p


def _eval_and_log(
    m: int,
    nom: str,
    y: np.ndarray,
    yv: np.ndarray,
    ydv: np.ndarray,
    folds_a24: list[tuple[int, np.ndarray]],
    a24_idx: np.ndarray,
    boots_b: list[np.ndarray],
    boot_v: np.ndarray,
    boot_dv: np.ndarray,
    pa_cand: np.ndarray,
    pv_cand: np.ndarray,
    pdv_cand: np.ndarray,
    pa_ref6: np.ndarray,
    pv_ref6: np.ndarray,
    pdv_ref6: np.ndarray,
) -> dict:
    ya = y[a24_idx]
    bl_mask = [np.isin(a24_idx, te) for _, te in folds_a24]
    d_bl, lo_bl, hi_bl, e_bl = _blk_ci(ya, bl_mask, boots_b, pa_cand, pa_ref6)
    n_pos_bl = sum(x > 0 for x in e_bl)
    auc_v = float(roc_auc_score(yv, pv_cand))
    acc_v = float(((pv_cand > 0.5) == yv).mean())
    ll_v = float(log_loss(yv, pv_cand))
    ece_v = float(_ece(yv, pv_cand))
    c5, c10, c20 = _cov_acc(yv, pv_cand, 0.05), _cov_acc(yv, pv_cand, 0.10), _cov_acc(yv, pv_cand, 0.20)
    dv, lov, hiv = _delta_ci(yv, pv_cand, pv_ref6, boot_v)
    n10, n01, pmc = _mcnemar(yv, pv_cand, pv_ref6)

    auc_dv = float(roc_auc_score(ydv, pdv_cand))
    acc_dv = float(((pdv_cand > 0.5) == ydv).mean())
    ll_dv = float(log_loss(ydv, pdv_cand))
    ddv, lodv, hidv = _delta_ci(ydv, pdv_cand, pdv_ref6, boot_dv)

    print(
        f"  {nom:22s} | bloq {n_pos_bl}/3 {[round(x,5) for x in e_bl]} mean {d_bl:+.5f}[{lo_bl:+.5f};{hi_bl:+.5f}] | "
        f"V {auc_v:.4f}/{acc_v:.2%} LL {ll_v:.4f} ECE {ece_v:.4f} [cov {c5:.1%}/{c10:.1%}/{c20:.1%}] | "
        f"dV_vs6 {dv:+.5f}[{lov:+.5f};{hiv:+.5f}] McN {n10}/{n01} p={pmc:.4f} | "
        f"DiamV {auc_dv:.4f}/{acc_dv:.2%} LL {ll_dv:.4f} dDV_vs6 {ddv:+.5f}[{lodv:+.5f};{hidv:+.5f}]",
        flush=True,
    )
    return {
        "minuto": m, "modelo": nom, "n_V": len(yv),
        "dA2": e_bl[0], "dA3": e_bl[1], "dA4": e_bl[2],
        "dBloq_vs6": d_bl, "lo_bl": lo_bl, "hi_bl": hi_bl, "n_Bloq_pos": n_pos_bl,
        "AUC_V": auc_v, "Acc_V": acc_v, "LogLoss_V": ll_v, "ECE_V": ece_v,
        "cov5": c5, "cov10": c10, "cov20": c20,
        "dV_vs6": dv, "lo_v": lov, "hi_v": hiv, "mcn_10": n10, "mcn_01": n01, "mcn_p": pmc,
        "AUC_DiamV": auc_dv, "Acc_DiamV": acc_dv, "LogLoss_DiamV": ll_dv,
        "dDV_vs6": ddv, "lo_dv": lodv, "hi_dv": hidv,
    }


def _perm_partitions(arr: np.ndarray, slices: list[np.ndarray], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = arr.copy()
    for sl in slices:
        if len(sl) > 1:
            out[sl] = arr[rng.permutation(sl)]
    return out


def _run_blend_and_stage2(
    cols_a: list[np.ndarray],
    cols_v: list[np.ndarray],
    cols_dv: list[np.ndarray],
    y: np.ndarray,
    folds_a24: list[tuple[int, np.ndarray]],
    a24_idx: np.ndarray,
    v_idx: np.ndarray,
    dv_idx: np.ndarray,
    d_ens10: np.ndarray,
    tag: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ya = y[a24_idx]
    bl_mask = [np.isin(a24_idx, te) for _, te in folds_a24]
    pa_b = np.full(len(ya), np.nan, np.float32)
    for bi in range(3):
        bm = bl_mask[bi]
        zte = np.column_stack([_logit(c[bm]) for c in cols_a])
        if bi == 0:
            pa_b[bm] = _sigm(zte.mean(axis=1))
        else:
            trm = np.zeros(len(ya), dtype=bool)
            for j in range(bi):
                trm |= bl_mask[j]
            lr = LogisticRegression(C=0.5, max_iter=1000).fit(np.column_stack([_logit(c[trm]) for c in cols_a]), ya[trm])
            pa_b[bm] = lr.predict_proba(zte)[:, 1].astype(np.float32)

    meta_b = LogisticRegression(C=0.5, max_iter=1000).fit(np.column_stack([_logit(c) for c in cols_a]), ya)
    pv_b = meta_b.predict_proba(np.column_stack([_logit(c) for c in cols_v]))[:, 1].astype(np.float32)
    pdv_b = meta_b.predict_proba(np.column_stack([_logit(c) for c in cols_dv]))[:, 1].astype(np.float32)
    log(f"  {tag} blend pesos A2-A4: coef={np.round(meta_b.coef_[0], 3)}, b={meta_b.intercept_[0]:+.3f}")

    pa_s2 = pa_b.copy()
    for bi in (1, 2):
        bm = bl_mask[bi]
        trm = np.zeros(len(ya), dtype=bool)
        for j in range(bi):
            trm |= bl_mask[j]
        me = LogisticRegression(C=0.5, max_iter=1000).fit(np.c_[_logit(pa_b[trm]), d_ens10[a24_idx[trm]]], ya[trm])
        pa_s2[bm] = me.predict_proba(np.c_[_logit(pa_b[bm]), d_ens10[a24_idx[bm]]])[:, 1].astype(np.float32)

    me_f = LogisticRegression(C=0.5, max_iter=1000).fit(np.c_[_logit(pa_b), d_ens10[a24_idx]], ya)
    pv_s2 = me_f.predict_proba(np.c_[_logit(pv_b), d_ens10[v_idx]])[:, 1].astype(np.float32)
    pdv_s2 = me_f.predict_proba(np.c_[_logit(pdv_b), d_ens10[dv_idx]])[:, 1].astype(np.float32)
    log(f"  {tag} Stage2 dEns10 pesos A2-A4: coef={np.round(me_f.coef_[0], 3)}, b={me_f.intercept_[0]:+.3f}")
    return pa_b, pv_b, pdv_b, pa_s2, pv_s2, pdv_s2


def main() -> None:
    err_sym = _check_antisym_v2()
    log(f"Iniciando g_07_final_v (chequeo unico V + DiamV; antisym_err_V2={err_sym:.2e}, {DEV})...")

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
    del C; gc.collect()

    priors = _cargar_priors(mids, bloque, val, diam_v, L_orig)
    d_ens10 = np.where(np.isfinite(priors["L_ens10"]) & np.isfinite(L_orig), priors["L_ens10"] - L_orig, 0.0).astype(np.float32)
    del priors

    z4 = np.load(IN_DIR / "pred_mt_a4_gpu_92k.npz")
    gi = TIMELINE_STATS.index("gold")
    rows = []
    saved: dict[str, np.ndarray] = {}

    # Minuto 10: sin cambios (referencia g_04)
    for m_ref in (10,):
        v_idx = np.flatnonzero(val & (dur_sec > 60.0 * m_ref) & np.isfinite(tl_raw[:, :, :, :1, :]).all(axis=(1, 2, 3, 4)) & np.isfinite(L_orig))
        dv_idx = np.flatnonzero(diam_v & (dur_sec > 60.0 * m_ref) & np.isfinite(tl_raw[:, :, :, :1, :]).all(axis=(1, 2, 3, 4)) & np.isfinite(L_orig))
        assert np.array_equal(v_idx, z4["v_idx_10"]) and np.array_equal(dv_idx, z4["dv_idx_10"])
        a24_idx = z4["a24_idx_10"]
        folds_a24 = [(b, np.flatnonzero((chunk == b) & master & np.isin(np.arange(len(y)), a24_idx))) for b in (4, 5, 6)]
        boot_v = np.random.default_rng(410).integers(0, len(v_idx), (1000, len(v_idx)))
        boot_dv = np.random.default_rng(710).integers(0, len(dv_idx), (1000, len(dv_idx)))
        boots_b = [np.random.default_rng(1010).integers(0, len(te), (1000, len(te))) for _, te in folds_a24]
        log("\n==================== MINUTO 10 (SIN CAMBIOS: ref 6_Final_dEns10) ====================")
        rows.append(_eval_and_log(10, "6_Final_Ref", y, y[v_idx], y[dv_idx], folds_a24, a24_idx, boots_b, boot_v, boot_dv,
                                  z4["pa_10_6_Final_dEns10"], z4["pv_10_6_Final_dEns10"], z4["pdv_10_6_Final_dEns10"],
                                  z4["pa_10_6_Final_dEns10"], z4["pv_10_6_Final_dEns10"], z4["pdv_10_6_Final_dEns10"]))

    for mi, m in [(1, 15), (2, 20)]:
        tm = time.time()
        log(f"\n==================== MINUTO {m} (CHEQUEO UNICO V + DIAMV) ====================")
        ok_tl = np.isfinite(tl_raw[:, :, :, : mi + 1, :]).all(axis=(1, 2, 3, 4))
        ok_m = (dur_sec > 60.0 * m) & ok_tl & np.isfinite(L_orig)

        v_idx = np.flatnonzero(val & ok_m)
        dv_idx = np.flatnonzero(diam_v & ok_m)
        folds_a24 = [(b, np.flatnonzero((chunk == b) & master & ok_m)) for b in (4, 5, 6)]
        tr_full = np.flatnonzero(np.isin(chunk, [1, 2, 3, 4, 5, 6]) & master & ok_m)
        tr_full = tr_full[np.argsort(t[tr_full], kind="mergesort")]
        a24_idx = np.concatenate([te for _, te in folds_a24])
        assert np.array_equal(v_idx, z4[f"v_idx_{m}"])
        assert np.array_equal(dv_idx, z4[f"dv_idx_{m}"])
        assert np.array_equal(a24_idx, z4[f"a24_idx_{m}"])

        yv, ydv = y[v_idx], y[dv_idx]
        boot_v = np.random.default_rng(400 + m).integers(0, len(v_idx), size=(1000, len(v_idx)))
        boot_dv = np.random.default_rng(700 + m).integers(0, len(dv_idx), size=(1000, len(dv_idx)))
        rng_bl = np.random.default_rng(1000 + m)
        boots_b = [rng_bl.integers(0, len(te), size=(1000, len(te))) for _, te in folds_a24]
        g_tot = np.nan_to_num((tl_raw[:, 0, :, mi, gi] - tl_raw[:, 1, :, mi, gi]).sum(axis=1), nan=0.0).astype(np.float32)

        f_a2 = extract_minute_features(m, L_orig, tl_raw, gold_resid, exp_lane=exp_lane, exp_trans=exp_trans,
                                       include_all_stats=True, include_traj=True, include_expected=True,
                                       include_comp_cross=False, include_player=False)
        f_a4 = extract_minute_features(m, L_orig, tl_raw, gold_resid, exp_lane=exp_lane, exp_trans=exp_trans,
                                       comp=comp, tempo=tempo, player_pre=player_pre,
                                       include_all_stats=True, include_traj=True, include_expected=True,
                                       include_comp_cross=True, include_player=True) if m == 15 else None

        # Miembros cacheados de g_04
        pa2, pv2, pdv2 = z4[f"pa_{m}_2_MT_A4_bs512"], z4[f"pv_{m}_2_MT_A4_bs512"], z4[f"pdv_{m}_2_MT_A4_bs512"]
        pa3, pv3, pdv3 = z4[f"pa_{m}_3_MT_A4C1_bs512"], z4[f"pv_{m}_3_MT_A4C1_bs512"], z4[f"pdv_{m}_3_MT_A4C1_bs512"]
        pa4, pv4, pdv4 = z4[f"pa_{m}_4_MT_Res_A4C1_bs512"], z4[f"pv_{m}_4_MT_Res_A4C1_bs512"], z4[f"pdv_{m}_4_MT_Res_A4C1_bs512"]
        pa5, pv5, pdv5 = z4[f"pa_{m}_5_Blend_LR_MT"], z4[f"pv_{m}_5_Blend_LR_MT"], z4[f"pdv_{m}_5_Blend_LR_MT"]
        pa6, pv6, pdv6 = z4[f"pa_{m}_6_Final_dEns10"], z4[f"pv_{m}_6_Final_dEns10"], z4[f"pdv_{m}_6_Final_dEns10"]

        rows.append(_eval_and_log(m, "6_Final_Ref", y, yv, ydv, folds_a24, a24_idx, boots_b, boot_v, boot_dv,
                                  pa6, pv6, pdv6, pa6, pv6, pdv6))

        c_0d = 0.02 if m == 15 else 0.01
        te_vd = np.r_[v_idx, dv_idx]

        def run_0d(placebo: bool = False):
            p_a = np.full(len(y), np.nan, np.float32)
            z_oof_dict, z_te_dict, ext_dict = {}, {}, {}
            all_tasks = [(b, np.flatnonzero(np.isin(chunk, [1, 2, 3, 4, 5, 6]) & (chunk < b) & master & ok_m), te, True)
                         for b, te in folds_a24] + [(99, tr_full, te_vd, False)]
            for key_f, tr_i, te_i, is_blk in all_tasks:
                tr_i = tr_i[np.argsort(t[tr_i], kind="mergesort")]
                ar0b = _prep_arrays(f_a2, tr_i, te_i, L_orig, y)
                c1_tr = _carry_conc_feats(tl_raw, m, tr_i, tr_i)
                c1_te = _carry_conc_feats(tl_raw, m, tr_i, te_i)
                sd_c1 = np.maximum(c1_tr.std(0), 1e-6).astype(np.float32)
                c1s_tr, c1s_te = np.clip(c1_tr / sd_c1, -8, 8), np.clip(c1_te / sd_c1, -8, 8)
                n_oof = 4 if is_blk else 5
                _, z_te_0b, _ = _fit_logreg_antisym(np.c_[ar0b["Xa_tr"], c1s_tr], y[tr_i], np.c_[ar0b["Xa_te"], c1s_te], C=0.03)
                z_oof_0b = _oof_logreg_logits(np.c_[ar0b["Xa_tr"], c1s_tr], y[tr_i], t[tr_i], C=0.03, n_splits=n_oof)
                b0 = float(np.mean(z_oof_0b))
                cb_tr = _cb_close_mat(tempo["cbias"], g_tot, L_orig, z_oof_0b - b0, tr_i)
                cb_te = _cb_close_mat(tempo["cbias"], g_tot, L_orig, z_te_0b - b0, te_i)
                s3_tr = _s3_variants(tl_raw, tempo, slot_static, p_mat, m, tr_i, tr_i)["S3_4cols"]
                s3_te = _s3_variants(tl_raw, tempo, slot_static, p_mat, m, tr_i, te_i)["S3_4cols"]
                new_tr = np.c_[cb_tr, s3_tr].astype(np.float32)
                new_te = np.c_[cb_te, s3_te].astype(np.float32)
                if placebo:
                    new_tr = _perm_partitions(new_tr, [np.arange(len(tr_i))], 1100 + m * 10 + key_f)
                    if is_blk:
                        new_te = _perm_partitions(new_te, [np.arange(len(te_i))], 1200 + m * 10 + key_f)
                    else:
                        new_te = _perm_partitions(new_te, [np.arange(len(v_idx)), np.arange(len(v_idx), len(te_vd))], 1300 + m)

                ext_tr = np.c_[c1_tr, new_tr].astype(np.float32)
                ext_te = np.c_[c1_te, new_te].astype(np.float32)
                sd_e = np.maximum(ext_tr.std(0), 1e-6).astype(np.float32)
                ext_s_tr, ext_s_te = np.clip(ext_tr / sd_e, -8, 8).astype(np.float32), np.clip(ext_te / sd_e, -8, 8).astype(np.float32)

                Xa0d_tr, Xa0d_te = np.c_[ar0b["Xa_tr"], ext_s_tr].astype(np.float32), np.c_[ar0b["Xa_te"], ext_s_te].astype(np.float32)
                _, z_te_0d, _ = _fit_logreg_antisym(Xa0d_tr, y[tr_i], Xa0d_te, C=c_0d)
                z_oof_0d = _oof_logreg_logits(Xa0d_tr, y[tr_i], t[tr_i], C=c_0d, n_splits=n_oof)
                if is_blk:
                    p_a[te_i] = _sigm(z_te_0d)
                else:
                    p_vd = _sigm(z_te_0d)
                z_oof_dict[key_f] = z_oof_0d
                z_te_dict[key_f] = z_te_0d
                ext_dict[key_f] = (ext_s_tr, ext_s_te)
                del ar0b, Xa0d_tr, Xa0d_te
            return p_a[a24_idx], p_vd[: len(v_idx)], p_vd[len(v_idx) :], z_oof_dict, z_te_dict, ext_dict

        pa0d, pv0d, pdv0d, z_oof_d, z_te_d, ext_d = run_0d(placebo=False)
        pa0dp, pv0dp, pdv0dp, z_oof_dp, z_te_dp, ext_dp = run_0d(placebo=True)

        if m == 15:
            def run_nv2_laneprod(z_oof_map, z_te_map, ext_map, placebo: bool = False):
                pn_a = np.full(len(y), np.nan, np.float32)
                pn_v = pn_dv = None
                all_tasks = [(b, np.flatnonzero(np.isin(chunk, [1, 2, 3, 4, 5, 6]) & (chunk < b) & master & ok_m), te, True)
                             for b, te in folds_a24] + [(99, tr_full, te_vd, False)]
                for key_f, tr_i, te_i, is_blk in all_tasks:
                    tr_i = tr_i[np.argsort(t[tr_i], kind="mergesort")]
                    ya_b, ma_b, ys_b, ms_b, _ = _construir_aux_targets(m, tr_i, tl_raw, dur_sec, end_gold_diff, end_kills_diff)
                    ar4 = _prep_arrays(f_a4, tr_i, te_i, L_orig, y)
                    ext_s_tr, ext_s_te = ext_map[key_f]
                    ar4["Xa_tr"] = np.c_[ar4["Xa_tr"], ext_s_tr].astype(np.float32)
                    ar4["Xa_te"] = np.c_[ar4["Xa_te"], ext_s_te].astype(np.float32)

                    lb_tr_x, lr_tr_x = _slot_prod_channels(tl_raw, tempo, slot_static, p_mat, m, tr_i, tr_i)
                    lb_te_x, lr_te_x = _slot_prod_channels(tl_raw, tempo, slot_static, p_mat, m, tr_i, te_i)
                    if placebo:
                        rng_p = np.random.default_rng(1400 + key_f)
                        p_tr = rng_p.permutation(len(tr_i))
                        lb_tr_x, lr_tr_x = lb_tr_x[p_tr], lr_tr_x[p_tr]
                        if is_blk:
                            p_te = rng_p.permutation(len(te_i))
                            lb_te_x, lr_te_x = lb_te_x[p_te], lr_te_x[p_te]
                        else:
                            pv_perm = rng_p.permutation(len(v_idx))
                            pdv_perm = len(v_idx) + rng_p.permutation(len(dv_idx))
                            p_vd = np.r_[pv_perm, pdv_perm]
                            lb_te_x, lr_te_x = lb_te_x[p_vd], lr_te_x[p_vd]
                    pool_lp = np.r_[lb_tr_x.reshape(-1, lb_tr_x.shape[-1]), lr_tr_x.reshape(-1, lr_tr_x.shape[-1])]
                    sd_lp = np.maximum(pool_lp.std(axis=0), 1e-6).astype(np.float32)
                    ar4["Lb_tr"] = np.concatenate([ar4["Lb_tr"], np.clip(lb_tr_x / sd_lp, -8, 8)], axis=-1).astype(np.float32)
                    ar4["Lr_tr"] = np.concatenate([ar4["Lr_tr"], np.clip(lr_tr_x / sd_lp, -8, 8)], axis=-1).astype(np.float32)
                    ar4["Lb_te"] = np.concatenate([ar4["Lb_te"], np.clip(lb_te_x / sd_lp, -8, 8)], axis=-1).astype(np.float32)
                    ar4["Lr_te"] = np.concatenate([ar4["Lr_te"], np.clip(lr_te_x / sd_lp, -8, 8)], axis=-1).astype(np.float32)

                    seeds = [400 + s for s in range(4)] if not is_blk else [80 * key_f + s for s in range(4)]
                    z_mean = np.mean(
                        [_fit_netv2_gpu(m, ar4, y[tr_i], (ya_b, ma_b, ys_b, ms_b), sd_v, 512, 16, z_oof_map[key_f], z_te_map[key_f], use_skip=False)
                         for sd_v in seeds],
                        axis=0,
                    )
                    if is_blk:
                        pn_a[te_i] = _sigm(z_mean)
                    else:
                        p_vd = _sigm(z_mean)
                        pn_v, pn_dv = p_vd[: len(v_idx)], p_vd[len(v_idx) :]
                    del ar4; gc.collect()
                return pn_a[a24_idx], pn_v, pn_dv

            panv, pvnv, pdvnv = run_nv2_laneprod(z_oof_d, z_te_d, ext_d, placebo=False)
            panvp, pvnvp, pdvnvp = run_nv2_laneprod(z_oof_dp, z_te_dp, ext_dp, placebo=True)

            # Candidato único Min 15: D_4M_NV2_LaneProd + Etapa 2 d_ens10 (4 semillas)
            pa_b, pv_b, pdv_b, pa_s2, pv_s2, pdv_s2 = _run_blend_and_stage2(
                [pa0d, pa2, pa3, panv], [pv0d, pv2, pv3, pvnv], [pdv0d, pdv2, pdv3, pdvnv],
                y, folds_a24, a24_idx, v_idx, dv_idx, d_ens10, "M15_Cand",
            )
            # Placebo permutado completo de las features nuevas
            pa_bp, pv_bp, pdv_bp, pa_s2p, pv_s2p, pdv_s2p = _run_blend_and_stage2(
                [pa0dp, pa2, pa3, panvp], [pv0dp, pv2, pv3, pvnvp], [pdv0dp, pdv2, pdv3, pdvnvp],
                y, folds_a24, a24_idx, v_idx, dv_idx, d_ens10, "M15_Placebo",
            )
            rows.append(_eval_and_log(m, "M15_Blend_PreS2", y, yv, ydv, folds_a24, a24_idx, boots_b, boot_v, boot_dv,
                                      pa_b, pv_b, pdv_b, pa5, pv5, pdv5))
            rows.append(_eval_and_log(m, "M15_Cand_dEns10", y, yv, ydv, folds_a24, a24_idx, boots_b, boot_v, boot_dv,
                                      pa_s2, pv_s2, pdv_s2, pa6, pv6, pdv6))
            rows.append(_eval_and_log(m, "M15_Placebo_Feats", y, yv, ydv, folds_a24, a24_idx, boots_b, boot_v, boot_dv,
                                      pa_s2p, pv_s2p, pdv_s2p, pa6, pv6, pdv6))
            rows.append(_eval_and_log(m, "M15_Cand_vs_Placebo", y, yv, ydv, folds_a24, a24_idx, boots_b, boot_v, boot_dv,
                                      pa_s2, pv_s2, pdv_s2, pa_s2p, pv_s2p, pdv_s2p))
            saved["pa_15_cand"], saved["pv_15_cand"], saved["pdv_15_cand"] = pa_s2, pv_s2, pdv_s2
            saved["pa_15_plac"], saved["pv_15_plac"], saved["pdv_15_plac"] = pa_s2p, pv_s2p, pdv_s2p
            saved["pv_15_nv2"], saved["pdv_15_nv2"] = pvnv, pdvnv
            saved["v_idx_15"], saved["dv_idx_15"], saved["a24_idx_15"] = v_idx, dv_idx, a24_idx
        else:
            # Candidato único Min 20: A_0d_g04 + Etapa 2 d_ens10
            pa_b, pv_b, pdv_b, pa_s2, pv_s2, pdv_s2 = _run_blend_and_stage2(
                [pa0d, pa2, pa3, pa4], [pv0d, pv2, pv3, pv4], [pdv0d, pdv2, pdv3, pdv4],
                y, folds_a24, a24_idx, v_idx, dv_idx, d_ens10, "M20_Cand",
            )
            pa_bp, pv_bp, pdv_bp, pa_s2p, pv_s2p, pdv_s2p = _run_blend_and_stage2(
                [pa0dp, pa2, pa3, pa4], [pv0dp, pv2, pv3, pv4], [pdv0dp, pdv2, pdv3, pdv4],
                y, folds_a24, a24_idx, v_idx, dv_idx, d_ens10, "M20_Placebo",
            )
            rows.append(_eval_and_log(m, "M20_Blend_PreS2", y, yv, ydv, folds_a24, a24_idx, boots_b, boot_v, boot_dv,
                                      pa_b, pv_b, pdv_b, pa5, pv5, pdv5))
            rows.append(_eval_and_log(m, "M20_Cand_dEns10", y, yv, ydv, folds_a24, a24_idx, boots_b, boot_v, boot_dv,
                                      pa_s2, pv_s2, pdv_s2, pa6, pv6, pdv6))
            rows.append(_eval_and_log(m, "M20_Placebo_Feats", y, yv, ydv, folds_a24, a24_idx, boots_b, boot_v, boot_dv,
                                      pa_s2p, pv_s2p, pdv_s2p, pa6, pv6, pdv6))
            rows.append(_eval_and_log(m, "M20_Cand_vs_Placebo", y, yv, ydv, folds_a24, a24_idx, boots_b, boot_v, boot_dv,
                                      pa_s2, pv_s2, pdv_s2, pa_s2p, pv_s2p, pdv_s2p))
            saved["pa_20_cand"], saved["pv_20_cand"], saved["pdv_20_cand"] = pa_s2, pv_s2, pdv_s2
            saved["pa_20_plac"], saved["pv_20_plac"], saved["pdv_20_plac"] = pa_s2p, pv_s2p, pdv_s2p
            saved["v_idx_20"], saved["dv_idx_20"], saved["a24_idx_20"] = v_idx, dv_idx, a24_idx

        del f_a2, f_a4; gc.collect()
        pd.DataFrame(rows).to_csv(IN_DIR / "g07_final_v.csv", index=False)
        log(f"Minuto {m} terminado en {time.time() - tm:.1f}s")

    np.savez_compressed(IN_DIR / "pred_g07_final_v.npz", **saved)
    log(f"Listo total en {time.time() - T0:.1f}s")


if __name__ == "__main__":
    main()
