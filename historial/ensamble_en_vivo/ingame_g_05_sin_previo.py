"""Versión sin previo (`L_orig = 0`, sin Etapa 2) del modelo en vivo vigente (`6_Final_dEns10`).

Genera los números comparables "solo el estado" para la sección 3 de
`notebooks/eda_modelos_ganador.ipynb`, usando exactamente el mismo pipeline,
particiones, semillas (`20*b+s`, `30*b+s`, `40*b+s`, `200..400`), épocas (`16`)
y `bs=512` que `historial/ensamble_en_vivo/ingame_g_04_mt_a4_gpu.py`, pero poniendo
`L_orig = 0` en todas las variables de los 4 miembros (`0b_A2_C1`, `2_MT_A4`,
`3_MT_A4C1`, `4_MT_Res_A4C1` incluyendo su `init` residual) y omitiendo la Etapa 2
(`d_ens10`).

Mantiene la misma máscara `ok_m` (`np.isfinite(L_orig_real)`) que `g_04` para que
las filas evaluadas en `A2-A4`, `V Master+` (`16.331 / 16.238 / 14.971`) y
`Diamond-V` coincidan 1 a 1 con `6_Final_dEns10`.

Uso:
    python -u historial/ensamble_en_vivo/ingame_g_05_sin_previo.py
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
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.analysis.ingame_evaluar import (  # noqa: E402
    _construir_aux_targets,
    _ece,
    _fit_logreg_antisym,
    _logit,
    _oof_logreg_logits,
    _sigm,
)
from historial.ensamble_en_vivo.ingame_g_03_regimen_y_senales import _cov_acc  # noqa: E402
from historial.ensamble_en_vivo.ingame_g_04_mt_a4_gpu import (  # noqa: E402
    DEV,
    IN_DIR,
    _fit_mt_gpu,
    _prep_arrays_with_c1,
)
from src.models.ingame import MINUTES, extract_minute_features  # noqa: E402

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def _fast_auc(y: np.ndarray, s: np.ndarray) -> float:
    r = rankdata(s)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _delta_ci_fast(
    y: np.ndarray,
    p1: np.ndarray,
    p0: np.ndarray,
    boot_idx: np.ndarray,
) -> tuple[float, float, float]:
    d0 = _fast_auc(y, p1) - _fast_auc(y, p0)
    ds = np.array([_fast_auc(y[b], p1[b]) - _fast_auc(y[b], p0[b]) for b in boot_idx])
    return float(d0), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


def _blk_mean_ci(
    ya: np.ndarray,
    bl_mask: list[np.ndarray],
    boots_b: list[np.ndarray],
    p1: np.ndarray,
    p0: np.ndarray,
) -> tuple[float, float, float]:
    ds = [np.array([_fast_auc(ya[bm][s], p1[bm][s]) - _fast_auc(ya[bm][s], p0[bm][s]) for s in bt]) for bm, bt in zip(bl_mask, boots_b)]
    dm = np.mean(ds, axis=0)
    pt = float(np.mean([_fast_auc(ya[bm], p1[bm]) - _fast_auc(ya[bm], p0[bm]) for bm in bl_mask]))
    return pt, float(np.percentile(dm, 2.5)), float(np.percentile(dm, 97.5))


def _prep_arrays_zero_prior(
    fdict: dict,
    tr_idx: np.ndarray,
    te_idx: np.ndarray,
    L_zero: np.ndarray,
    y: np.ndarray,
    tl_raw: np.ndarray,
    minute: int,
    use_c1: bool,
) -> dict:
    ar = _prep_arrays_with_c1(fdict, tr_idx, te_idx, L_zero, y, tl_raw, minute, use_c1)
    # Anular explícitamente columna 0 (L_sym - L0) para que no quede constante != 0
    ar["Xa_tr"][:, 0] = 0.0
    ar["Xa_te"][:, 0] = 0.0
    return ar


def main() -> None:
    cache_path = IN_DIR / "ingame_cache.npz"
    vig_path = IN_DIR / "pred_mt_a4_gpu_92k.npz"
    if not cache_path.exists():
        sys.exit(f"Falta {cache_path}")
    if not vig_path.exists():
        sys.exit(f"Falta {vig_path}")

    log(f"Iniciando versión SIN PREVIO (`g_05`) en {DEV}...")
    C = np.load(cache_path, allow_pickle=True)
    t = C["t"].astype(np.int64)
    y = C["y"].astype(np.int64)
    chunk = C["chunk"].astype(np.int32)
    master = C["master"].astype(bool)
    val = C["val"].astype(bool)
    diam_v = C["diam_v"].astype(bool)
    dur_sec = C["dur_sec"].astype(np.float32)
    L_orig_real = C["L_oof"].astype(np.float32)
    L_zero = np.zeros_like(L_orig_real, dtype=np.float32)
    tl_raw = C["tl_raw"].astype(np.float32)
    gold_resid = C["gold_resid"].astype(np.float32)
    end_gold_diff = C["end_gold_diff"].astype(np.float32)
    end_kills_diff = C["end_kills_diff"].astype(np.float32)
    exp_lane = {k[4:]: C[k].astype(np.float32) for k in C.files if k.startswith("exp_")}
    exp_trans = {k[3:]: C[k].astype(np.float32) for k in C.files if k.startswith("tr_")}
    comp = {
        "comp_anti": C["comp_anti"].astype(np.float32),
        "names_anti": list(C["comp_names_anti"]),
        "comp_sym": C["comp_sym"].astype(np.float32),
        "names_sym": list(C["comp_names_sym"]),
        "vuln_blue": C["vuln_blue"].astype(np.float32),
        "vuln_red": C["vuln_red"].astype(np.float32),
        "vuln_names": list(C["vuln_names"]),
        "slot_static": C["slot_static"].astype(np.float32),
    }
    tempo = {k: C[k].astype(np.float32) for k in ("cbias", "cslope", "pbias", "pvol")}
    player_pre = {"p_mat": C["p_mat"].astype(np.float32), "p_cols": list(C["p_cols"])}
    del C
    gc.collect()

    vig = np.load(vig_path)
    filas = []
    saved = {}

    for mi, m in enumerate(MINUTES):
        tm = time.time()
        log(f"\n==================== MINUTO {m} (SIN PREVIO: L_orig=0, sin Etapa 2) ====================")
        ok_tl = np.isfinite(tl_raw[:, :, :, : mi + 1, :]).all(axis=(1, 2, 3, 4))
        ok_m = (dur_sec > 60.0 * m) & ok_tl & np.isfinite(L_orig_real)

        v_idx = np.flatnonzero(val & ok_m)
        dv_idx = np.flatnonzero(diam_v & ok_m)
        folds_a24 = [(b, np.flatnonzero((chunk == b) & master & ok_m)) for b in (4, 5, 6)]
        tr_full = np.flatnonzero(np.isin(chunk, [1, 2, 3, 4, 5, 6]) & master & ok_m)
        tr_full = tr_full[np.argsort(t[tr_full], kind="mergesort")]
        a24_idx = np.concatenate([te for _, te in folds_a24])

        assert np.array_equal(v_idx, vig[f"v_idx_{m}"])
        assert np.array_equal(dv_idx, vig[f"dv_idx_{m}"])
        assert np.array_equal(a24_idx, vig[f"a24_idx_{m}"])

        rng_bv = np.random.default_rng(400 + m)
        boot_v = rng_bv.integers(0, len(v_idx), size=(1000, len(v_idx)))
        rng_dv = np.random.default_rng(700 + m)
        boot_dv = rng_dv.integers(0, len(dv_idx), size=(1000, len(dv_idx)))
        ya, yv, ydv = y[a24_idx], y[v_idx], y[dv_idx]
        bl_mask = [chunk[a24_idx] == b for b in (4, 5, 6)]
        boots_b = [np.random.default_rng(1000 + m).integers(0, int(bm.sum()), (1000, int(bm.sum()))) for bm in bl_mask]

        f_a2 = extract_minute_features(
            m, L_zero, tl_raw, gold_resid, exp_lane=exp_lane, exp_trans=exp_trans,
            include_all_stats=True, include_traj=True, include_expected=True,
            include_comp_cross=False, include_player=False,
        )
        f_a4 = extract_minute_features(
            m, L_zero, tl_raw, gold_resid, exp_lane=exp_lane, exp_trans=exp_trans,
            comp=comp, tempo=tempo, player_pre=player_pre,
            include_all_stats=True, include_traj=True, include_expected=True,
            include_comp_cross=True, include_player=True,
        )
        # Verificar que L_orig quedó en 0 en abs_L_sym y cruce_L_pace_k también
        assert np.all(f_a4["X_anti"][:, 0] == 0.0)
        assert np.all(f_a4["X_sym"][:, 0] == 0.0)

        pa_m, pv_m, pdv_m = {}, {}, {}

        # 0. Logísticas lineales sin previo: A2 y A2 + C1_CarryConc
        c_c1 = 0.03
        for key, use_c1 in [("0_SinPrev_A2", False), ("0b_SinPrev_A2_C1", True)]:
            best_C, best_ll = 0.03, float("inf")
            for c_try in (0.01, 0.03, 0.05):
                pa_t = np.full(len(y), np.nan, np.float32)
                for b, te in folds_a24:
                    tr_b = np.flatnonzero(np.isin(chunk, [1, 2, 3, 4, 5, 6]) & (chunk < b) & master & ok_m)
                    ar = _prep_arrays_zero_prior(f_a2, tr_b, te, L_zero, y, tl_raw, m, use_c1)
                    pa_t[te] = _sigm(_fit_logreg_antisym(ar["Xa_tr"], y[tr_b], ar["Xa_te"], C=c_try)[1])
                ll = log_loss(ya, pa_t[a24_idx])
                if ll < best_ll:
                    best_ll, best_C = ll, c_try
            pa = np.full(len(y), np.nan, np.float32)
            for b, te in folds_a24:
                tr_b = np.flatnonzero(np.isin(chunk, [1, 2, 3, 4, 5, 6]) & (chunk < b) & master & ok_m)
                ar = _prep_arrays_zero_prior(f_a2, tr_b, te, L_zero, y, tl_raw, m, use_c1)
                pa[te] = _sigm(_fit_logreg_antisym(ar["Xa_tr"], y[tr_b], ar["Xa_te"], C=best_C)[1])
            te_vd = np.r_[v_idx, dv_idx]
            ar_f = _prep_arrays_zero_prior(f_a2, tr_full, te_vd, L_zero, y, tl_raw, m, use_c1)
            _, z_vd, _ = _fit_logreg_antisym(ar_f["Xa_tr"], y[tr_full], ar_f["Xa_te"], C=best_C)
            p_vd = _sigm(z_vd)
            pa_m[key], pv_m[key], pdv_m[key] = pa, p_vd[: len(v_idx)], p_vd[len(v_idx) :]
            if use_c1:
                c_c1 = best_C
            log(f"  {key}: C={best_C} (LL A2-A4={best_ll:.5f})")

        # Redes GPU sobre f_a4 sin previo (mismos seeds/épocas/bs=512 que g_04)
        pa_m2 = np.full(len(y), np.nan, np.float32)
        pa_m3 = np.full(len(y), np.nan, np.float32)
        pa_m4 = np.full(len(y), np.nan, np.float32)

        for b, te in folds_a24:
            tr_b = np.flatnonzero(np.isin(chunk, [1, 2, 3, 4, 5, 6]) & (chunk < b) & master & ok_m)
            tr_b = tr_b[np.argsort(t[tr_b], kind="mergesort")]
            ya_b, ma_b, ys_b, ms_b, _ = _construir_aux_targets(m, tr_b, tl_raw, dur_sec, end_gold_diff, end_kills_diff)

            ar_a2c1 = _prep_arrays_zero_prior(f_a2, tr_b, te, L_zero, y, tl_raw, m, use_c1=True)
            _, z_te_lr, _ = _fit_logreg_antisym(ar_a2c1["Xa_tr"], y[tr_b], ar_a2c1["Xa_te"], C=c_c1)
            z_oof_lr = _oof_logreg_logits(ar_a2c1["Xa_tr"], y[tr_b], t[tr_b], C=c_c1, n_splits=4)
            del ar_a2c1

            ar_a4 = _prep_arrays_zero_prior(f_a4, tr_b, te, L_zero, y, tl_raw, m, use_c1=False)
            z2 = np.mean([_fit_mt_gpu(m, ar_a4, y[tr_b], (ya_b, ma_b, ys_b, ms_b), seed=20 * b + s, bs=512, epochs=16) for s in (0, 1)], axis=0)
            pa_m2[te] = _sigm(z2)
            del ar_a4; gc.collect()

            ar_a4c1 = _prep_arrays_zero_prior(f_a4, tr_b, te, L_zero, y, tl_raw, m, use_c1=True)
            z3 = np.mean([_fit_mt_gpu(m, ar_a4c1, y[tr_b], (ya_b, ma_b, ys_b, ms_b), seed=30 * b + s, bs=512, epochs=16) for s in (0, 1)], axis=0)
            z4 = np.mean([_fit_mt_gpu(m, ar_a4c1, y[tr_b], (ya_b, ma_b, ys_b, ms_b), seed=40 * b + s, bs=512, epochs=16,
                                      init_tr=z_oof_lr, init_te=z_te_lr) for s in (0, 1)], axis=0)
            pa_m3[te], pa_m4[te] = _sigm(z3), _sigm(z4)
            del ar_a4c1; gc.collect()

        te_vd = np.r_[v_idx, dv_idx]
        ya_f, ma_f, ys_f, ms_f, _ = _construir_aux_targets(m, tr_full, tl_raw, dur_sec, end_gold_diff, end_kills_diff)

        ar_fa2c1 = _prep_arrays_zero_prior(f_a2, tr_full, te_vd, L_zero, y, tl_raw, m, use_c1=True)
        _, z_vd_lr, _ = _fit_logreg_antisym(ar_fa2c1["Xa_tr"], y[tr_full], ar_fa2c1["Xa_te"], C=c_c1)
        z_oof_f = _oof_logreg_logits(ar_fa2c1["Xa_tr"], y[tr_full], t[tr_full], C=c_c1, n_splits=5)
        del ar_fa2c1, f_a2; gc.collect()

        ar_fa4 = _prep_arrays_zero_prior(f_a4, tr_full, te_vd, L_zero, y, tl_raw, m, use_c1=False)
        zf2 = np.mean([_fit_mt_gpu(m, ar_fa4, y[tr_full], (ya_f, ma_f, ys_f, ms_f), seed=200 + s, bs=512, epochs=16) for s in range(4)], axis=0)
        del ar_fa4; gc.collect()

        ar_fa4c1 = _prep_arrays_zero_prior(f_a4, tr_full, te_vd, L_zero, y, tl_raw, m, use_c1=True)
        zf3 = np.mean([_fit_mt_gpu(m, ar_fa4c1, y[tr_full], (ya_f, ma_f, ys_f, ms_f), seed=300 + s, bs=512, epochs=16) for s in range(4)], axis=0)
        zf4 = np.mean([_fit_mt_gpu(m, ar_fa4c1, y[tr_full], (ya_f, ma_f, ys_f, ms_f), seed=400 + s, bs=512, epochs=16,
                                   init_tr=z_oof_f, init_te=z_vd_lr) for s in range(4)], axis=0)
        del ar_fa4c1, f_a4; gc.collect()

        for k, p_a, z_vd in [
            ("2_SinPrev_MT_A4_bs512", pa_m2, zf2),
            ("3_SinPrev_MT_A4C1_bs512", pa_m3, zf3),
            ("4_SinPrev_MT_Res_A4C1_bs512", pa_m4, zf4),
        ]:
            p_f = _sigm(z_vd)
            pa_m[k], pv_m[k], pdv_m[k] = p_a, p_f[: len(v_idx)], p_f[len(v_idx) :]

        # Ensamble OOF hacia adelante exacto sin previo (A2=mean, A3<-A2, A4<-A2+A3, V/DV<-A2-A4)
        cols_b = ["0b_SinPrev_A2_C1", "2_SinPrev_MT_A4_bs512", "3_SinPrev_MT_A4C1_bs512", "4_SinPrev_MT_Res_A4C1_bs512"]
        pa_b = np.full(len(y), np.nan, np.float32)
        for bi, (_, te) in enumerate(folds_a24):
            z_te = np.column_stack([_logit(pa_m[c][te]) for c in cols_b])
            if bi == 0:
                pa_b[te] = _sigm(z_te.mean(axis=1))
            else:
                tr_prev = np.concatenate([folds_a24[j][1] for j in range(bi)])
                z_pr = np.column_stack([_logit(pa_m[c][tr_prev]) for c in cols_b])
                meta_b = LogisticRegression(C=0.5, max_iter=1000).fit(z_pr, y[tr_prev])
                pa_b[te] = meta_b.predict_proba(z_te)[:, 1].astype(np.float32)
        Z_a = np.column_stack([_logit(pa_m[c][a24_idx]) for c in cols_b])
        meta = LogisticRegression(C=0.5, max_iter=1000).fit(Z_a, y[a24_idx])
        pv_b = meta.predict_proba(np.column_stack([_logit(pv_m[c]) for c in cols_b]))[:, 1].astype(np.float32)
        pdv_b = meta.predict_proba(np.column_stack([_logit(pdv_m[c]) for c in cols_b]))[:, 1].astype(np.float32)
        pa_m["5_SinPrev_Blend_LR_MT"], pv_m["5_SinPrev_Blend_LR_MT"], pdv_m["5_SinPrev_Blend_LR_MT"] = pa_b, pv_b, pdv_b
        log(f"  5_SinPrev_Blend pesos A2-A4: coef={np.round(meta.coef_[0], 3)}, b={meta.intercept_[0]:+.3f}")

        # Incluir 6_Final_dEns10 vigente en la tabla comparativa directa ("+ lo previo")
        pa_m["6_Vig_Final_dEns10"] = np.full(len(y), np.nan, np.float32)
        pa_m["6_Vig_Final_dEns10"][a24_idx] = vig[f"pa_{m}_6_Final_dEns10"]
        pv_m["6_Vig_Final_dEns10"] = vig[f"pv_{m}_6_Final_dEns10"]
        pdv_m["6_Vig_Final_dEns10"] = vig[f"pdv_{m}_6_Final_dEns10"]

        ref_a0 = pa_m["0_SinPrev_A2"][a24_idx]
        ref_v0 = pv_m["0_SinPrev_A2"]
        ref_dv0 = pdv_m["0_SinPrev_A2"]
        ref_blend = pv_m["5_SinPrev_Blend_LR_MT"]

        for nom in pv_m:
            pa_a = pa_m[nom][a24_idx]
            pv = pv_m[nom]
            pdv = pdv_m[nom]
            aucs_b = [_fast_auc(ya[bm], pa_a[bm]) for bm in bl_mask]
            bm_d, bm_lo, bm_hi = _blk_mean_ci(ya, bl_mask, boots_b, pa_a, ref_a0)
            auc_v = _fast_auc(yv, pv)
            acc_v = float(((pv > 0.5) == yv).mean())
            ll_v = log_loss(yv, pv)
            ece_v = _ece(yv, pv)
            c5, c10, c20 = _cov_acc(yv, pv, 0.05), _cov_acc(yv, pv, 0.10), _cov_acc(yv, pv, 0.20)
            d0, lo0, hi0 = _delta_ci_fast(yv, pv, ref_v0, boot_v)
            db, lob, hib = _delta_ci_fast(yv, pv, ref_blend, boot_v)
            auc_dv = _fast_auc(ydv, pdv)
            acc_dv = float(((pdv > 0.5) == ydv).mean())
            ddv, lodv, hidv = _delta_ci_fast(ydv, pdv, ref_dv0, boot_dv)
            print(
                f"  {nom:27s} | A2/3/4 {aucs_b[0]:.4f}/{aucs_b[1]:.4f}/{aucs_b[2]:.4f} (Bloq {bm_d:+.4f} [{bm_lo:+.4f};{bm_hi:+.4f}]) | "
                f"V {auc_v:.4f}/{acc_v:.2%} LL {ll_v:.4f} ECE {ece_v:.4f} [{c5:.1%}/{c10:.1%}/{c20:.1%}] | "
                f"dA2sp {d0:+.4f} [{lo0:+.4f};{hi0:+.4f}] | dBlendsp {db:+.4f} [{lob:+.4f};{hib:+.4f}] | "
                f"DiamV {auc_dv:.4f}/{acc_dv:.2%} ({ddv:+.4f} [{lodv:+.4f};{hidv:+.4f}])",
                flush=True,
            )
            filas.append({
                "minuto": m, "modelo": nom, "n_V": len(v_idx), "n_DV": len(dv_idx),
                "AUC_A2": aucs_b[0], "AUC_A3": aucs_b[1], "AUC_A4": aucs_b[2],
                "dBloq_vs_A2sp": bm_d, "bloq_lo": bm_lo, "bloq_hi": bm_hi,
                "AUC_V": auc_v, "Acc_V": acc_v, "LogLoss_V": ll_v, "ECE_V": ece_v,
                "cov5": c5, "cov10": c10, "cov20": c20,
                "dV_vs_A2sp": d0, "lo_A2sp": lo0, "hi_A2sp": hi0,
                "dV_vs_Blendsp": db, "lo_Blendsp": lob, "hi_Blendsp": hib,
                "AUC_DiamV": auc_dv, "Acc_DiamV": acc_dv,
                "dDV_vs_A2sp": ddv, "lodv": lodv, "hidv": hidv,
            })
            if not nom.startswith("6_Vig"):
                saved[f"pv_{m}_{nom}"] = pv
                saved[f"pdv_{m}_{nom}"] = pdv
                saved[f"pa_{m}_{nom}"] = pa_a

        saved[f"v_idx_{m}"] = v_idx
        saved[f"dv_idx_{m}"] = dv_idx
        saved[f"a24_idx_{m}"] = a24_idx
        saved[f"y_v_{m}"] = yv
        saved[f"y_dv_{m}"] = ydv
        saved[f"y_a24_{m}"] = ya
        pd.DataFrame(filas).to_csv(IN_DIR / "sin_previo_92k.csv", index=False)
        log(f"Minuto {m} sin previo terminado en {time.time() - tm:.1f}s")

    np.savez_compressed(IN_DIR / "pred_sin_previo_92k.npz", **saved)
    log(f"Listo total en {time.time() - T0:.1f}s")


if __name__ == "__main__":
    main()
