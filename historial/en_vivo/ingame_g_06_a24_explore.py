"""Exploración estricta en A2-A4 (SIN tocar V ni Diamond-V) para InGame (@10, @15, @20).

Prueba los Candidatos 1 y 2 acordados en COORDINACION.md [06:43] usando únicamente
los sub-bloques Master+ `A2, A3, A4` (`chunks 4, 5, 6`) y caches de `g_04`:
  1. `cbias × cercanía` causal (`CB_CLOSE`: `cb_tot`, `cb_tot*exp(-|g|/3)`,
     `cb_tot*exp(-|L_orig|)`, `cb_tot*exp(-|z_0b_oof|)` sin circularidad) en:
     - `0c_A2_C1_cb`: logística lineal Stage-1 (`92k`, `C` elegido en `A2-A4`)
     - `5_Blend_w0c`: ensamble `g_04` con `0c` y ajuste directo `+cb_close` en `A3-A4`
  2. GBDTs con parada temprana cronológica interna (`tr_s=88% -> va_s=12%` espejada,
     nunca `A2-A4` ni `V`) sobre `[f_a4 + C1 + CB_CLOSE + z_oof_0b + |z_oof_0b|]`:
     - `G1_CB_Res_ES`: CatBoost GPU Residual (`baseline = z_anti_0b`)
     - `G2_LGBM_Res_ES`: LightGBM Residual (`init_score = z_anti_0b`)
     - `G3_CB_Stand_ES`: CatBoost GPU sin baseline congelado (`z_0b` como rasgo)
  3. Re-ensamble causal (`5_Blend + GBDT`, `A2` promedio / `A3<-A2` / `A4<-A2+A3`)
     evaluando `BloqMean 95% IC` y pesos aprendidos en `A2-A4` (V nunca se lee).

Uso:
    python -u historial/en_vivo/ingame_g_06_a24_explore.py
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

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.analysis.ingame_evaluar import (  # noqa: E402
    _calibrar_b_side,
    _fit_logreg_antisym,
    _logit,
    _oof_logreg_logits,
    _prep_arrays,
    _sigm,
)
from historial.ensamble_en_vivo.ingame_g_03_regimen_y_senales import _carry_conc_feats  # noqa: E402
from src.models.ingame import MINUTES, TIMELINE_STATS, extract_minute_features  # noqa: E402

IN_DIR = REPO / "data" / "analysis" / "ingame"
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def _fast_auc(y: np.ndarray, s: np.ndarray) -> float:
    r = rankdata(s)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _blk_ci(ya: np.ndarray, bl_mask: list[np.ndarray], boots_b: list[np.ndarray], p1: np.ndarray, p0: np.ndarray):
    ds = [np.array([_fast_auc(ya[bm][s], p1[bm][s]) - _fast_auc(ya[bm][s], p0[bm][s]) for s in bt]) for bm, bt in zip(bl_mask, boots_b)]
    dm = np.mean(ds, axis=0)
    pt = float(np.mean([_fast_auc(ya[bm], p1[bm]) - _fast_auc(ya[bm], p0[bm]) for bm in bl_mask]))
    d_each = [float(_fast_auc(ya[bm], p1[bm]) - _fast_auc(ya[bm], p0[bm])) for bm in bl_mask]
    return pt, float(np.percentile(dm, 2.5)), float(np.percentile(dm, 97.5)), d_each


def _cb_close_mat(cbias: np.ndarray, g_tot: np.ndarray, L_orig: np.ndarray, z0b_arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
    cb_d = (cbias[idx, 0] - cbias[idx, 1]).sum(axis=1)
    # Nota: para antisimetría exacta de |z0b|, callers ya centran z0b_arr alrededor de su media
    return np.column_stack([
        cb_d,
        cb_d * np.exp(-np.abs(g_tot[idx]) / 3.0),
        cb_d * np.exp(-np.abs(L_orig[idx])),
        cb_d * np.exp(-np.abs(z0b_arr)),
    ]).astype(np.float32)


def _prep_ext(fdict: dict, tr_idx: np.ndarray, te_idx: np.ndarray, L_oof: np.ndarray, y: np.ndarray,
              tl_raw: np.ndarray, minute: int, cb_tr: np.ndarray | None = None, cb_te: np.ndarray | None = None) -> dict:
    ar = _prep_arrays(fdict, tr_idx, te_idx, L_oof, y)
    c1_tr = _carry_conc_feats(tl_raw, minute, tr_idx, tr_idx)
    c1_te = _carry_conc_feats(tl_raw, minute, tr_idx, te_idx)
    ext_tr = c1_tr if cb_tr is None else np.c_[c1_tr, cb_tr].astype(np.float32)
    ext_te = c1_te if cb_te is None else np.c_[c1_te, cb_te].astype(np.float32)
    sd = np.maximum(np.std(ext_tr, axis=0), 1e-6).astype(np.float32)
    ar["Xa_tr"] = np.c_[ar["Xa_tr"], np.clip(ext_tr / sd, -8.0, 8.0)].astype(np.float32)
    ar["Xa_te"] = np.c_[ar["Xa_te"], np.clip(ext_te / sd, -8.0, 8.0)].astype(np.float32)
    return ar


def _fit_cb_gpu_es(Xa_tr, Xs_tr, y_tr, Xa_te, Xs_te, z_tr, z_te, use_baseline: bool = True) -> tuple[np.ndarray, int]:
    """CatBoost GPU espejado con holdout cronológico interno 12% para parada temprana."""
    n = len(y_tr)
    cut = int(n * 0.88)
    b0 = float(np.mean(z_tr[:cut]))
    z_a_tr = (z_tr - b0).astype(np.float32)
    z_a_te = (z_te - b0).astype(np.float32)

    X_tr_base = np.c_[Xa_tr, z_a_tr[:, None], np.abs(z_a_tr)[:, None], Xs_tr].astype(np.float32)
    X_tr_swap = np.c_[-Xa_tr, -z_a_tr[:, None], np.abs(z_a_tr)[:, None], Xs_tr].astype(np.float32)
    X_te_p = np.c_[Xa_te, z_a_te[:, None], np.abs(z_a_te)[:, None], Xs_te].astype(np.float32)
    X_te_s = np.c_[-Xa_te, -z_a_te[:, None], np.abs(z_a_te)[:, None], Xs_te].astype(np.float32)

    X_s = np.r_[X_tr_base[:cut], X_tr_swap[:cut]]
    y_s = np.r_[y_tr[:cut], 1 - y_tr[:cut]]
    X_v = np.r_[X_tr_base[cut:], X_tr_swap[cut:]]
    y_v = np.r_[y_tr[cut:], 1 - y_tr[cut:]]

    if use_baseline:
        bs_s = np.r_[z_a_tr[:cut], -z_a_tr[:cut]]
        bs_v = np.r_[z_a_tr[cut:], -z_a_tr[cut:]]
        p_tr = Pool(X_s, y_s, baseline=bs_s)
        p_va = Pool(X_v, y_v, baseline=bs_v)
    else:
        p_tr = Pool(X_s, y_s)
        p_va = Pool(X_v, y_v)

    cb = CatBoostClassifier(
        iterations=1600,
        depth=5,
        learning_rate=0.03,
        l2_leaf_reg=30.0,
        border_count=64,
        task_type="GPU",
        early_stopping_rounds=80,
        use_best_model=True,
        verbose=0,
        random_seed=42,
    )
    cb.fit(p_tr, eval_set=p_va)
    best_it = int(cb.get_best_iteration())

    def pred_raw(X, base):
        pl = Pool(X, baseline=base) if use_baseline else Pool(X)
        return np.asarray(cb.predict(pl, prediction_type="RawFormulaVal")).reshape(-1)

    z_te_anti = 0.5 * (pred_raw(X_te_p, z_a_te if use_baseline else None) - pred_raw(X_te_s, -z_a_te if use_baseline else None))
    z_tr_anti = 0.5 * (pred_raw(X_tr_base, z_a_tr if use_baseline else None) - pred_raw(X_tr_swap, -z_a_tr if use_baseline else None))
    b_side = _calibrar_b_side(z_tr_anti, y_tr)
    return _sigm(b_side + z_te_anti), best_it


def _fit_lgb_res_es(Xa_tr, Xs_tr, y_tr, Xa_te, Xs_te, z_tr, z_te) -> tuple[np.ndarray, int]:
    """LightGBM Residual espejado con parada temprana cronológica 12%."""
    n = len(y_tr)
    cut = int(n * 0.88)
    b0 = float(np.mean(z_tr[:cut]))
    z_a_tr = (z_tr - b0).astype(np.float32)
    z_a_te = (z_te - b0).astype(np.float32)

    X_tr_base = np.c_[Xa_tr, z_a_tr[:, None], np.abs(z_a_tr)[:, None], Xs_tr].astype(np.float32)
    X_tr_swap = np.c_[-Xa_tr, -z_a_tr[:, None], np.abs(z_a_tr)[:, None], Xs_tr].astype(np.float32)
    X_te_p = np.c_[Xa_te, z_a_te[:, None], np.abs(z_a_te)[:, None], Xs_te].astype(np.float32)
    X_te_s = np.c_[-Xa_te, -z_a_te[:, None], np.abs(z_a_te)[:, None], Xs_te].astype(np.float32)

    X_s = np.r_[X_tr_base[:cut], X_tr_swap[:cut]]
    y_s = np.r_[y_tr[:cut], 1 - y_tr[:cut]]
    bs_s = np.r_[z_a_tr[:cut], -z_a_tr[:cut]]
    X_v = np.r_[X_tr_base[cut:], X_tr_swap[cut:]]
    y_v = np.r_[y_tr[cut:], 1 - y_tr[cut:]]
    bs_v = np.r_[z_a_tr[cut:], -z_a_tr[cut:]]

    clf = lgb.LGBMClassifier(
        n_estimators=1600,
        num_leaves=23,
        max_depth=5,
        learning_rate=0.025,
        max_bin=63,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.5,
        min_child_samples=300,
        reg_lambda=25.0,
        n_jobs=6,
        verbose=-1,
        random_state=42,
    )
    clf.fit(
        X_s, y_s, init_score=bs_s,
        eval_set=[(X_v, y_v)], eval_init_score=[bs_v],
        callbacks=[lgb.early_stopping(80, verbose=False)],
    )
    best_it = int(clf.best_iteration_ or clf.n_estimators)

    raw_te_p = np.asarray(clf.predict_proba(X_te_p, raw_score=True)).reshape(-1)
    raw_te_s = np.asarray(clf.predict_proba(X_te_s, raw_score=True)).reshape(-1)
    z_te_anti = z_a_te + 0.5 * (raw_te_p - raw_te_s)

    raw_tr_p = np.asarray(clf.predict_proba(X_tr_base, raw_score=True)).reshape(-1)
    raw_tr_s = np.asarray(clf.predict_proba(X_tr_swap, raw_score=True)).reshape(-1)
    z_tr_anti = z_a_tr + 0.5 * (raw_tr_p - raw_tr_s)

    b_side = _calibrar_b_side(z_tr_anti, y_tr)
    return _sigm(b_side + z_te_anti), best_it


def main() -> None:
    log("Iniciando exploración ESTRICTA A2-A4 (SIN V ni Diamond-V)...")
    C = np.load(IN_DIR / "ingame_cache.npz", allow_pickle=True)
    t = C["t"].astype(np.int64)
    y = C["y"].astype(np.int64)
    chunk = C["chunk"].astype(np.int32)
    master = C["master"].astype(bool)
    dur_sec = C["dur_sec"].astype(np.float32)
    L_orig = C["L_oof"].astype(np.float32)
    tl_raw = C["tl_raw"].astype(np.float32)
    gold_resid = C["gold_resid"].astype(np.float32)
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

    z4 = np.load(IN_DIR / "pred_mt_a4_gpu_92k.npz")
    stat_i = {s: i for i, s in enumerate(TIMELINE_STATS)}
    rows = []

    for mi, m in enumerate(MINUTES):
        tm = time.time()
        log(f"\n==================== MINUTO {m} (A2-A4 EXPLORE ONLY) ====================")
        ok_tl = np.isfinite(tl_raw[:, :, :, : mi + 1, :]).all(axis=(1, 2, 3, 4))
        ok_m = (dur_sec > 60.0 * m) & ok_tl & np.isfinite(L_orig)
        folds_a24 = [(b, np.flatnonzero((chunk == b) & master & ok_m)) for b in (4, 5, 6)]
        a24_idx = np.concatenate([te for _, te in folds_a24])
        assert np.array_equal(a24_idx, z4[f"a24_idx_{m}"])
        ya = y[a24_idx]
        bl_mask = [chunk[a24_idx] == b for b in (4, 5, 6)]
        boots_b = [np.random.default_rng(1000 + m).integers(0, int(bm.sum()), (1000, int(bm.sum()))) for bm in bl_mask]
        g_tot = np.nan_to_num((tl_raw[:, 0, :, mi, stat_i["gold"]] - tl_raw[:, 1, :, mi, stat_i["gold"]]).sum(axis=1), nan=0.0).astype(np.float32)

        f_a2 = extract_minute_features(
            m, L_orig, tl_raw, gold_resid, exp_lane=exp_lane, exp_trans=exp_trans,
            include_all_stats=True, include_traj=True, include_expected=True,
            include_comp_cross=False, include_player=False,
        )
        f_a4 = extract_minute_features(
            m, L_orig, tl_raw, gold_resid, exp_lane=exp_lane, exp_trans=exp_trans,
            comp=comp, tempo=tempo, player_pre=player_pre,
            include_all_stats=True, include_traj=True, include_expected=True,
            include_comp_cross=True, include_player=True,
        )

        # Preds base desde g_04 (solo A2-A4)
        p0b = z4[f"pa_{m}_0b_A2_C1"]
        p2mt = z4[f"pa_{m}_2_MT_A4_bs512"]
        p3mt = z4[f"pa_{m}_3_MT_A4C1_bs512"]
        p4mt = z4[f"pa_{m}_4_MT_Res_A4C1_bs512"]
        p5b = z4[f"pa_{m}_5_Blend_LR_MT"]

        # Primero elegir C de 0b_A2_C1_cb en A2-A4 (y precomputar z_oof_0b / z_te_0b por fold)
        fold_cache = []
        for b, te in folds_a24:
            tr_b = np.flatnonzero(np.isin(chunk, [1, 2, 3, 4, 5, 6]) & (chunk < b) & master & ok_m)
            tr_b = tr_b[np.argsort(t[tr_b], kind="mergesort")]
            ar0b = _prep_ext(f_a2, tr_b, te, L_orig, y, tl_raw, m, cb_tr=None, cb_te=None)
            # C=0.03 fue la base estable; probamos C de 0c sobre A2-A4 también
            _, z_te_0b, _ = _fit_logreg_antisym(ar0b["Xa_tr"], y[tr_b], ar0b["Xa_te"], C=0.03)
            z_oof_0b = _oof_logreg_logits(ar0b["Xa_tr"], y[tr_b], t[tr_b], C=0.03, n_splits=4)
            b0_tr = float(np.mean(z_oof_0b))
            cb_tr = _cb_close_mat(tempo["cbias"], g_tot, L_orig, z_oof_0b - b0_tr, tr_b)
            cb_te = _cb_close_mat(tempo["cbias"], g_tot, L_orig, z_te_0b - b0_tr, te)
            fold_cache.append((b, tr_b, te, z_oof_0b, z_te_0b, cb_tr, cb_te))

        best_C, best_ll = 0.03, float("inf")
        for c_try in (0.01, 0.03, 0.05):
            pa_t = np.full(len(y), np.nan, np.float32)
            for b, tr_b, te, _, _, cb_tr, cb_te in fold_cache:
                ar0c = _prep_ext(f_a2, tr_b, te, L_orig, y, tl_raw, m, cb_tr=cb_tr, cb_te=cb_te)
                pa_t[te] = _sigm(_fit_logreg_antisym(ar0c["Xa_tr"], y[tr_b], ar0c["Xa_te"], C=c_try)[1])
            ll = log_loss(ya, pa_t[a24_idx])
            if ll < best_ll:
                best_ll, best_C = ll, c_try

        p_0c = np.full(len(y), np.nan, np.float32)
        p_cb_res = np.full(len(y), np.nan, np.float32)
        p_lgb_res = np.full(len(y), np.nan, np.float32)
        p_cb_stand = np.full(len(y), np.nan, np.float32)
        cb_its, lgb_its, cbs_its = [], [], []

        for b, tr_b, te, z_oof_0b, z_te_0b, cb_tr, cb_te in fold_cache:
            ar0c = _prep_ext(f_a2, tr_b, te, L_orig, y, tl_raw, m, cb_tr=cb_tr, cb_te=cb_te)
            p_0c[te] = _sigm(_fit_logreg_antisym(ar0c["Xa_tr"], y[tr_b], ar0c["Xa_te"], C=best_C)[1])

            ar_a4ext = _prep_ext(f_a4, tr_b, te, L_orig, y, tl_raw, m, cb_tr=cb_tr, cb_te=cb_te)
            p_cb_res[te], it1 = _fit_cb_gpu_es(ar_a4ext["Xa_tr"], ar_a4ext["Xs_tr"], y[tr_b], ar_a4ext["Xa_te"], ar_a4ext["Xs_te"], z_oof_0b, z_te_0b, use_baseline=True)
            p_lgb_res[te], it2 = _fit_lgb_res_es(ar_a4ext["Xa_tr"], ar_a4ext["Xs_tr"], y[tr_b], ar_a4ext["Xa_te"], ar_a4ext["Xs_te"], z_oof_0b, z_te_0b)
            p_cb_stand[te], it3 = _fit_cb_gpu_es(ar_a4ext["Xa_tr"], ar_a4ext["Xs_tr"], y[tr_b], ar_a4ext["Xa_te"], ar_a4ext["Xs_te"], z_oof_0b, z_te_0b, use_baseline=False)
            cb_its.append(it1); lgb_its.append(it2); cbs_its.append(it3)

        del f_a2, f_a4, fold_cache; gc.collect()
        log(f"  0c C={best_C} | iters ES [A2,A3,A4]: CB_Res={cb_its}, LGBM_Res={lgb_its}, CB_Stand={cbs_its}")

        # Blends OOF hacia adelante en A2-A4 (A2=media, A3<-A2, A4<-A2+A3)
        def fwd_blend(cols: list[np.ndarray], label: str) -> np.ndarray:
            out = np.full(len(ya), np.nan, np.float32)
            for bi, b in enumerate((4, 5, 6)):
                bm = bl_mask[bi]
                z_te = np.column_stack([_logit(c[bm]) for c in cols])
                if bi == 0:
                    out[bm] = _sigm(z_te.mean(axis=1))
                else:
                    tr_m = np.isin(chunk[a24_idx], [4 + j for j in range(bi)])
                    z_tr = np.column_stack([_logit(c[tr_m]) for c in cols])
                    lr = LogisticRegression(C=0.5, max_iter=1000).fit(z_tr, ya[tr_m])
                    out[bm] = lr.predict_proba(z_te)[:, 1].astype(np.float32)
            lr_f = LogisticRegression(C=0.5, max_iter=1000).fit(np.column_stack([_logit(c) for c in cols]), ya)
            log(f"  Pesos A2-A4 {label:20s}: {np.round(lr_f.coef_[0], 3)} (b={lr_f.intercept_[0]:+.3f})")
            return out

        pa_0c = p_0c[a24_idx]
        pa_cbr = p_cb_res[a24_idx]
        pa_lgr = p_lgb_res[a24_idx]
        pa_cbs = p_cb_stand[a24_idx]

        pa_b_0c = fwd_blend([pa_0c, p2mt, p3mt, p4mt], "Blend_w0c")
        pa_b_cbr = fwd_blend([p0b, p2mt, p3mt, p4mt, pa_cbr], "Blend+CB_Res")
        pa_b_lgr = fwd_blend([p0b, p2mt, p3mt, p4mt, pa_lgr], "Blend+LGBM_Res")
        pa_b_all = fwd_blend([pa_0c, p2mt, p3mt, p4mt, pa_cbr, pa_lgr], "Blend_0c+CB+LGB")

        # Evaluar 0c vs 0b primero y cada candidato vs 5_Blend en BloqMean A2-A4
        c_pt, c_lo, c_hi, c_each = _blk_ci(ya, bl_mask, boots_b, pa_0c, p0b)
        print(f"  {'0c_A2_C1_cb vs 0b':24s} | A2/3/4 {[round(x,5) for x in c_each]} | BloqMean {c_pt:+.5f} [{c_lo:+.5f};{c_hi:+.5f}]", flush=True)

        evals = [
            ("5_Blend_LR_MT (ref)", p5b),
            ("G1_CB_Res_ES", pa_cbr),
            ("G2_LGBM_Res_ES", pa_lgr),
            ("G3_CB_Stand_ES", pa_cbs),
            ("B1_Blend_w0c", pa_b_0c),
            ("B2_Blend+CB_Res", pa_b_cbr),
            ("B3_Blend+LGBM_Res", pa_b_lgr),
            ("B4_Blend_0c+CB+LGB", pa_b_all),
        ]
        for nom, pa_k in evals:
            pt, lo, hi, each = _blk_ci(ya, bl_mask, boots_b, pa_k, p5b)
            aucs = [float(_fast_auc(ya[bm], pa_k[bm])) for bm in bl_mask]
            ll_a = float(log_loss(ya, np.clip(pa_k, 1e-6, 1.0 - 1e-6)))
            print(f"  {nom:24s} | A2/3/4 {aucs[0]:.4f}/{aucs[1]:.4f}/{aucs[2]:.4f} LL={ll_a:.5f} | vs 5_Blend: {[round(x,5) for x in each]} | BloqMean {pt:+.5f} [{lo:+.5f};{hi:+.5f}]", flush=True)
            rows.append({
                "minuto": m, "modelo": nom,
                "AUC_A2": aucs[0], "AUC_A3": aucs[1], "AUC_A4": aucs[2], "LL_A24": ll_a,
                "dA2": each[0], "dA3": each[1], "dA4": each[2],
                "BloqMean": pt, "lo": lo, "hi": hi,
            })
        pd.DataFrame(rows).to_csv(IN_DIR / "a24_explore_g06.csv", index=False)
        log(f"Minuto {m} terminado en {time.time() - tm:.1f}s")

    log(f"Listo total en {time.time() - T0:.1f}s")


if __name__ == "__main__":
    main()
