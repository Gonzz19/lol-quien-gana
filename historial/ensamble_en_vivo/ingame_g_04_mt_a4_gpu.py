"""Paso (iii.b) definitivo en GPU RTX 3070: InGameNet_MultiTask sobre `f_a4` limpio + `C1_CarryConc` + `dEns10` + `HGB_POST15`.

Hallazgos combinados de (i)-(iii.a)-(iii.c):
  - `10_InGameNet_MultiTask` de la laptop (`0,8428` @15, `0,8739` @20) usaba `bs=2048, ep=16`
    sobre `f_a4` completo (`comp + tempo + player_pre`) y `L_oof` uniforme en `92k`.
  - `C1_CarryConc` (5 rasgos de concentración/eficiencia desde `tl_raw`) ganó `+0,0005` (`3/3` bloq,
    IC > 0) en min 15 y 20 sobre `A2` (`92k`).
  - Desacoplar `dEns10 = L_ens10 - L_orig` en Etapa 2 (`A3<-A2`, `A4<-A2-A3`, `V/DiamV<-A2-A4`)
    sumó `+0,0016` @10, `+0,0007` @15, `+0,0003` @20 (todo con IC > 0 y placebo = 0).

Compara en minutos 10, 15, 20 (`92k Master+`, `A2-A4`, `V Master+`, `Diamond-V` + `cov5/10/20` + `ECE`):
  0. `0_Lorig_A2`, `0b_A2_C1`
  1. `1_MT_A4_bs2048`: reproducción GPU de `10_InGameNet_MultiTask` (`f_a4`, `L_orig`, `bs=2048, ep=16`)
  2. `2_MT_A4_bs512`: efecto puro de `bs=512, ep=16` sobre `f_a4`
  3. `3_MT_A4_C1_bs512`: `InGameNet_MT` (`bs=512`) sobre `f_a4 + C1_CarryConc`
  4. `4_MT_Res_A4C1_bs512`: `InGameNet` residual sobre `z_oof(0b_A2_C1)` (`bs=512`)
  5. `5_Blend_LR_MT`: ensamble OOF hacia adelante en `A2-A4` y ajustado en `A2-A4` para `V/DiamV`
  6. `6_Final_dEns10` (+ `6p_dEns10_placebo`): suma `d_ens10 = L_ens10 - L_orig` en etapa 2 post-Foto-1
  7. `7_Final_dEns10_Post15` (+ `7p_Post15_placebo`): suma `HGB_POST15` (`C2b`) en etapa 2

Uso:
    python -u historial/ensamble_en_vivo/ingame_g_04_mt_a4_gpu.py
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
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.analysis.ingame_evaluar import (  # noqa: E402
    _construir_aux_targets,
    _delta_ci,
    _ece,
    _fit_logreg_antisym,
    _logit,
    _oof_logreg_logits,
    _prep_arrays,
    _sigm,
)
from historial.ensamble_en_vivo.ingame_g_01_prior_y_gpu import _cargar_priors  # noqa: E402
from historial.ensamble_en_vivo.ingame_g_03_regimen_y_senales import _carry_conc_feats, _cov_acc  # noqa: E402
from src.models.ingame import InGameNet, MINUTES, TIMELINE_STATS, extract_minute_features  # noqa: E402

IN_DIR = REPO / "data" / "analysis" / "ingame"
BR_DIR = REPO / "data" / "analysis" / "brainstorm"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def _prep_arrays_with_c1(fdict: dict, tr_idx: np.ndarray, te_idx: np.ndarray, L_oof: np.ndarray,
                         y: np.ndarray, tl_raw: np.ndarray, minute: int, use_c1: bool) -> dict:
    ar = _prep_arrays(fdict, tr_idx, te_idx, L_oof, y)
    if not use_c1:
        return ar
    c1_tr = _carry_conc_feats(tl_raw, minute, tr_idx, tr_idx)
    c1_te = _carry_conc_feats(tl_raw, minute, tr_idx, te_idx)
    mu = np.zeros(c1_tr.shape[1], dtype=np.float32)  # antisimétrico -> media 0
    sd = np.maximum(np.std(c1_tr, axis=0), 1e-6).astype(np.float32)
    c1_tr_z = np.clip((c1_tr - mu) / sd, -8.0, 8.0).astype(np.float32)
    c1_te_z = np.clip((c1_te - mu) / sd, -8.0, 8.0).astype(np.float32)
    ar["Xa_tr"] = np.c_[ar["Xa_tr"], c1_tr_z].astype(np.float32)
    ar["Xa_te"] = np.c_[ar["Xa_te"], c1_te_z].astype(np.float32)
    return ar


def _fit_mt_gpu(
    minute: int,
    arrs: dict,
    y_tr: np.ndarray,
    aux_tr: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    seed: int = 0,
    bs: int = 512,
    epochs: int = 16,
    init_tr: np.ndarray | None = None,
    init_te: np.ndarray | None = None,
) -> np.ndarray:
    """InGameNet_MultiTask en RTX 3070 (modo limpio o residual sobre `init_tr/init_te`)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    Lb_tr, Lr_tr, Xa_tr, Xs_tr = arrs["Lb_tr"], arrs["Lr_tr"], arrs["Xa_tr"], arrs["Xs_tr"]
    Lb_te, Lr_te, Xa_te, Xs_te = arrs["Lb_te"], arrs["Lr_te"], arrs["Xa_te"], arrs["Xs_te"]

    n = len(y_tr)
    cut = int(n * 0.88)
    tr_s, va_s = np.arange(cut), np.arange(cut, n)

    if init_tr is not None:
        b0 = float(np.mean(init_tr))
        ia_tr = (init_tr - b0).astype(np.float32)
        ia_te = (init_te - b0).astype(np.float32)
        lr_val, wd_val = 1.5e-3, 8e-3
    else:
        ia_tr = ia_te = None
        lr_val, wd_val = 2.0e-3, 5e-3

    net = InGameNet(
        d_lane=Lb_tr.shape[-1],
        d_anti=Xa_tr.shape[1],
        d_sym=Xs_tr.shape[1],
        d_aux_anti=aux_tr[0].shape[1],
        d_aux_sym=aux_tr[2].shape[1],
        lane_hidden=24,
        trunk_hidden=96,
        dropout=0.25,
    ).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr_val, weight_decay=wd_val)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    t_lb = torch.tensor(Lb_tr[tr_s], device=DEV)
    t_lr = torch.tensor(Lr_tr[tr_s], device=DEV)
    t_xa = torch.tensor(Xa_tr[tr_s], device=DEV)
    t_xs = torch.tensor(Xs_tr[tr_s], device=DEV)
    t_y = torch.tensor(y_tr[tr_s], dtype=torch.float32, device=DEV)
    t_ya = torch.tensor(aux_tr[0][tr_s], device=DEV)
    t_ma = torch.tensor(aux_tr[1][tr_s], dtype=torch.float32, device=DEV)
    t_ys = torch.tensor(aux_tr[2][tr_s], device=DEV)
    t_ms = torch.tensor(aux_tr[3][tr_s], dtype=torch.float32, device=DEV)
    t_z0 = torch.tensor(ia_tr[tr_s], device=DEV) if ia_tr is not None else None

    v_lb = torch.tensor(Lb_tr[va_s], device=DEV)
    v_lr = torch.tensor(Lr_tr[va_s], device=DEV)
    v_xa = torch.tensor(Xa_tr[va_s], device=DEV)
    v_xs = torch.tensor(Xs_tr[va_s], device=DEV)
    v_y = torch.tensor(y_tr[va_s], dtype=torch.float32, device=DEV)
    v_z0 = torch.tensor(ia_tr[va_s], device=DEV) if ia_tr is not None else None

    best_loss = float("inf")
    best_state = None
    rng = np.random.default_rng(seed)

    for _ in range(epochs):
        net.train()
        perm = rng.permutation(len(tr_s))
        for b in range(0, len(perm), bs):
            ii = perm[b : b + bs]
            d_log, out_a, out_s = net(t_lb[ii], t_lr[ii], t_xa[ii], t_xs[ii])
            logit = (t_z0[ii] + d_log) if t_z0 is not None else d_log
            loss = F.binary_cross_entropy_with_logits(logit, t_y[ii])
            loss_a = (F.smooth_l1_loss(out_a, t_ya[ii], reduction="none") * t_ma[ii]).sum() / torch.clamp(t_ma[ii].sum(), min=1.0)
            if minute == 20:
                loss_s = (F.smooth_l1_loss(out_s, t_ys[ii], reduction="none") * t_ms[ii]).sum() / torch.clamp(t_ms[ii].sum(), min=1.0)
            else:
                loss_s = F.binary_cross_entropy_with_logits(out_s, t_ys[ii])
            loss = loss + 0.18 * loss_a + 0.08 * loss_s
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt.step()
        sch.step()

        net.eval()
        with torch.no_grad():
            vl = net(v_lb, v_lr, v_xa, v_xs)[0]
            if v_z0 is not None:
                vl = vl + v_z0
            vloss = float(F.binary_cross_entropy_with_logits(vl, v_y).item())
        if vloss < best_loss:
            best_loss = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    if best_state is not None:
        net.load_state_dict({k: v.to(DEV) for k, v in best_state.items()})
    net.eval()
    preds = []
    with torch.no_grad():
        for b in range(0, len(Xa_te), 4096):
            l_b = net(
                torch.tensor(Lb_te[b : b + 4096], device=DEV),
                torch.tensor(Lr_te[b : b + 4096], device=DEV),
                torch.tensor(Xa_te[b : b + 4096], device=DEV),
                torch.tensor(Xs_te[b : b + 4096], device=DEV),
            )[0].detach().cpu().numpy()
            if ia_te is not None:
                l_b = l_b + ia_te[b : b + 4096]
            preds.append(l_b)

    del t_lb, t_lr, t_xa, t_xs, t_y, t_ya, t_ma, t_ys, t_ms, t_z0
    del v_lb, v_lr, v_xa, v_xs, v_y, v_z0, net, opt, sch, best_state
    torch.cuda.empty_cache()
    return np.concatenate(preds).astype(np.float32)


def main() -> None:
    cache_path = IN_DIR / "ingame_cache.npz"
    if not cache_path.exists():
        sys.exit(f"Falta {cache_path}")

    log(f"Iniciando Paso (iii.b) en {DEV}...")
    C = np.load(cache_path, allow_pickle=True)
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

    priors = _cargar_priors(mids, bloque, val, diam_v, L_orig)
    L_ens10 = priors["L_ens10"].astype(np.float32)
    d_ens10 = np.where(np.isfinite(L_ens10) & np.isfinite(L_orig), L_ens10 - L_orig, 0.0).astype(np.float32)
    del priors

    hgb_post15 = np.load(BR_DIR / "gemini_hgb_ctx_p15.npz")["HGB_POST15"].astype(np.float32)

    lap_z = np.load(IN_DIR / "pred_ingame.npz")
    lap_best_key = {10: "3_Abl_A2_Stats_Traj_Exp", 15: "10_InGameNet_MultiTask", 20: "10_InGameNet_MultiTask"}
    lap_ref = {mm: {"v_idx": lap_z[f"v_idx_{mm}"], "y_v": lap_z[f"y_v_{mm}"], "p_v": lap_z[f"p_{mm}_{bk}"].astype(np.float32), "name": bk}
               for mm, bk in lap_best_key.items()}
    del lap_z
    gc.collect()

    stat_i = {s: i for i, s in enumerate(TIMELINE_STATS)}
    filas = []
    saved = {}

    for mi, m in enumerate(MINUTES):
        tm = time.time()
        log(f"\n==================== MINUTO {m} (f_a4 GPU + C1 + 2Stg dEns10) ====================")
        ok_tl = np.isfinite(tl_raw[:, :, :, : mi + 1, :]).all(axis=(1, 2, 3, 4))
        ok_m = (dur_sec > 60.0 * m) & ok_tl & np.isfinite(L_orig)

        v_idx = np.flatnonzero(val & ok_m)
        dv_idx = np.flatnonzero(diam_v & ok_m)
        folds_a24 = [(b, np.flatnonzero((chunk == b) & master & ok_m)) for b in (4, 5, 6)]
        tr_full = np.flatnonzero(np.isin(chunk, [1, 2, 3, 4, 5, 6]) & master & ok_m)
        tr_full = tr_full[np.argsort(t[tr_full], kind="mergesort")]
        a24_idx = np.concatenate([te for _, te in folds_a24])

        rng_bv = np.random.default_rng(400 + m)
        boot_v = rng_bv.integers(0, len(v_idx), size=(1000, len(v_idx)))
        rng_dv = np.random.default_rng(700 + m)
        boot_dv = rng_dv.integers(0, len(dv_idx), size=(1000, len(dv_idx)))
        yv, ydv = y[v_idx], y[dv_idx]
        g_tot_m = np.nan_to_num((tl_raw[:, 0, :, mi, stat_i["gold"]] - tl_raw[:, 1, :, mi, stat_i["gold"]]).sum(axis=1), nan=0.0).astype(np.float32)

        f_a2 = extract_minute_features(m, L_orig, tl_raw, gold_resid, exp_lane=exp_lane, exp_trans=exp_trans,
                                       include_all_stats=True, include_traj=True, include_expected=True,
                                       include_comp_cross=False, include_player=False)
        f_a4 = extract_minute_features(m, L_orig, tl_raw, gold_resid, exp_lane=exp_lane, exp_trans=exp_trans,
                                       comp=comp, tempo=tempo, player_pre=player_pre,
                                       include_all_stats=True, include_traj=True, include_expected=True,
                                       include_comp_cross=True, include_player=True)

        pa_m, pv_m, pdv_m = {}, {}, {}

        # 0. Logísticas lineales 92k: A2 y A2 + C1_CarryConc
        for key, use_c1 in [("0_Lorig_A2", False), ("0b_A2_C1", True)]:
            best_C, best_ll = 0.03, float("inf")
            for c_try in (0.01, 0.03, 0.05):
                pa_t = np.full(len(y), np.nan, np.float32)
                for b, te in folds_a24:
                    tr_b = np.flatnonzero(np.isin(chunk, [1, 2, 3, 4, 5, 6]) & (chunk < b) & master & ok_m)
                    ar = _prep_arrays_with_c1(f_a2, tr_b, te, L_orig, y, tl_raw, m, use_c1)
                    pa_t[te] = _sigm(_fit_logreg_antisym(ar["Xa_tr"], y[tr_b], ar["Xa_te"], C=c_try)[1])
                ll = log_loss(y[a24_idx], pa_t[a24_idx])
                if ll < best_ll:
                    best_ll, best_C = ll, c_try
            pa = np.full(len(y), np.nan, np.float32)
            for b, te in folds_a24:
                tr_b = np.flatnonzero(np.isin(chunk, [1, 2, 3, 4, 5, 6]) & (chunk < b) & master & ok_m)
                ar = _prep_arrays_with_c1(f_a2, tr_b, te, L_orig, y, tl_raw, m, use_c1)
                pa[te] = _sigm(_fit_logreg_antisym(ar["Xa_tr"], y[tr_b], ar["Xa_te"], C=best_C)[1])
            te_vd = np.r_[v_idx, dv_idx]
            ar_f = _prep_arrays_with_c1(f_a2, tr_full, te_vd, L_orig, y, tl_raw, m, use_c1)
            _, z_vd, _ = _fit_logreg_antisym(ar_f["Xa_tr"], y[tr_full], ar_f["Xa_te"], C=best_C)
            p_vd = _sigm(z_vd)
            pa_m[key], pv_m[key], pdv_m[key] = pa, p_vd[: len(v_idx)], p_vd[len(v_idx) :]
            if use_c1:
                c_c1 = best_C

        # Redes GPU sobre f_a4 (1: bs=2048 reprod laptop; 2: bs=512; 3: +C1 bs=512; 4: Res s/A2+C1 bs=512)
        pa_m1 = np.full(len(y), np.nan, np.float32)
        pa_m2 = np.full(len(y), np.nan, np.float32)
        pa_m3 = np.full(len(y), np.nan, np.float32)
        pa_m4 = np.full(len(y), np.nan, np.float32)

        for b, te in folds_a24:
            tr_b = np.flatnonzero(np.isin(chunk, [1, 2, 3, 4, 5, 6]) & (chunk < b) & master & ok_m)
            tr_b = tr_b[np.argsort(t[tr_b], kind="mergesort")]
            ya, ma, ys, ms_, _ = _construir_aux_targets(m, tr_b, tl_raw, dur_sec, end_gold_diff, end_kills_diff)

            ar_a2c1 = _prep_arrays_with_c1(f_a2, tr_b, te, L_orig, y, tl_raw, m, use_c1=True)
            _, z_te_lr, _ = _fit_logreg_antisym(ar_a2c1["Xa_tr"], y[tr_b], ar_a2c1["Xa_te"], C=c_c1)
            z_oof_lr = _oof_logreg_logits(ar_a2c1["Xa_tr"], y[tr_b], t[tr_b], C=c_c1, n_splits=4)
            del ar_a2c1

            ar_a4 = _prep_arrays_with_c1(f_a4, tr_b, te, L_orig, y, tl_raw, m, use_c1=False)
            z1 = np.mean([_fit_mt_gpu(m, ar_a4, y[tr_b], (ya, ma, ys, ms_), seed=10 * b + s, bs=2048, epochs=16) for s in (0, 1)], axis=0)
            z2 = np.mean([_fit_mt_gpu(m, ar_a4, y[tr_b], (ya, ma, ys, ms_), seed=20 * b + s, bs=512, epochs=16) for s in (0, 1)], axis=0)
            pa_m1[te], pa_m2[te] = _sigm(z1), _sigm(z2)
            del ar_a4; gc.collect()

            ar_a4c1 = _prep_arrays_with_c1(f_a4, tr_b, te, L_orig, y, tl_raw, m, use_c1=True)
            z3 = np.mean([_fit_mt_gpu(m, ar_a4c1, y[tr_b], (ya, ma, ys, ms_), seed=30 * b + s, bs=512, epochs=16) for s in (0, 1)], axis=0)
            z4 = np.mean([_fit_mt_gpu(m, ar_a4c1, y[tr_b], (ya, ma, ys, ms_), seed=40 * b + s, bs=512, epochs=16,
                                      init_tr=z_oof_lr, init_te=z_te_lr) for s in (0, 1)], axis=0)
            pa_m3[te], pa_m4[te] = _sigm(z3), _sigm(z4)
            del ar_a4c1; gc.collect()

        te_vd = np.r_[v_idx, dv_idx]
        ya_f, ma_f, ys_f, ms_f, _ = _construir_aux_targets(m, tr_full, tl_raw, dur_sec, end_gold_diff, end_kills_diff)

        ar_fa2c1 = _prep_arrays_with_c1(f_a2, tr_full, te_vd, L_orig, y, tl_raw, m, use_c1=True)
        _, z_vd_lr, _ = _fit_logreg_antisym(ar_fa2c1["Xa_tr"], y[tr_full], ar_fa2c1["Xa_te"], C=c_c1)
        z_oof_f = _oof_logreg_logits(ar_fa2c1["Xa_tr"], y[tr_full], t[tr_full], C=c_c1, n_splits=5)
        del ar_fa2c1, f_a2; gc.collect()

        ar_fa4 = _prep_arrays_with_c1(f_a4, tr_full, te_vd, L_orig, y, tl_raw, m, use_c1=False)
        zf1 = np.mean([_fit_mt_gpu(m, ar_fa4, y[tr_full], (ya_f, ma_f, ys_f, ms_f), seed=100 + s, bs=2048, epochs=16) for s in range(4)], axis=0)
        zf2 = np.mean([_fit_mt_gpu(m, ar_fa4, y[tr_full], (ya_f, ma_f, ys_f, ms_f), seed=200 + s, bs=512, epochs=16) for s in range(4)], axis=0)
        del ar_fa4; gc.collect()

        ar_fa4c1 = _prep_arrays_with_c1(f_a4, tr_full, te_vd, L_orig, y, tl_raw, m, use_c1=True)
        zf3 = np.mean([_fit_mt_gpu(m, ar_fa4c1, y[tr_full], (ya_f, ma_f, ys_f, ms_f), seed=300 + s, bs=512, epochs=16) for s in range(4)], axis=0)
        zf4 = np.mean([_fit_mt_gpu(m, ar_fa4c1, y[tr_full], (ya_f, ma_f, ys_f, ms_f), seed=400 + s, bs=512, epochs=16,
                                   init_tr=z_oof_f, init_te=z_vd_lr) for s in range(4)], axis=0)
        del ar_fa4c1, f_a4; gc.collect()

        for k, p_a, z_vd in [("1_MT_A4_bs2048", pa_m1, zf1),
                             ("2_MT_A4_bs512", pa_m2, zf2),
                             ("3_MT_A4C1_bs512", pa_m3, zf3),
                             ("4_MT_Res_A4C1_bs512", pa_m4, zf4)]:
            p_f = _sigm(z_vd)
            pa_m[k], pv_m[k], pdv_m[k] = p_a, p_f[: len(v_idx)], p_f[len(v_idx) :]

        # Blends OOF hacia adelante en A2-A4 (A2: promedio logits; A3 <- A2; A4 <- A2+A3) y calibración en A2-A4 -> V/DiamV
        blend_specs = [
            ("5_Blend_LR_MT", ["0b_A2_C1", "2_MT_A4_bs512", "3_MT_A4C1_bs512", "4_MT_Res_A4C1_bs512"]),
        ]
        for b_nom, cols_b in blend_specs:
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
            pa_m[b_nom], pv_m[b_nom], pdv_m[b_nom] = pa_b, pv_b, pdv_b
            log(f"  {b_nom} pesos A2-A4: coef={np.round(meta.coef_[0], 3)}, b={meta.intercept_[0]:+.3f}")

        # Etapa 2 post-Foto-1: + d_ens10 y (+ HGB_POST15), cada uno con placebo permutado dentro de particiones
        def eval_stage2(base_key: str, extra_mat: np.ndarray, nom: str, pl_seed: int | None = None):
            arr_u = extra_mat.copy()
            if pl_seed is not None:
                rng_p = np.random.default_rng(pl_seed)
                for sl in (folds_a24[0][1], folds_a24[1][1], folds_a24[2][1], v_idx, dv_idx):
                    arr_u[sl] = extra_mat[rng_p.permutation(sl)]
            pa_e = np.full(len(y), np.nan, np.float32)
            for bi, (_, te) in enumerate(folds_a24):
                if bi == 0:
                    pa_e[te] = pa_m[base_key][te]
                else:
                    tr_prev = np.concatenate([folds_a24[j][1] for j in range(bi)])
                    me = LogisticRegression(C=0.5, max_iter=1000).fit(np.c_[_logit(pa_m[base_key][tr_prev]), arr_u[tr_prev]], y[tr_prev])
                    pa_e[te] = me.predict_proba(np.c_[_logit(pa_m[base_key][te]), arr_u[te]])[:, 1].astype(np.float32)
            me_f = LogisticRegression(C=0.5, max_iter=1000).fit(np.c_[_logit(pa_m[base_key][a24_idx]), arr_u[a24_idx]], y[a24_idx])
            pv_e = me_f.predict_proba(np.c_[_logit(pv_m[base_key]), arr_u[v_idx]])[:, 1].astype(np.float32)
            pdv_e = me_f.predict_proba(np.c_[_logit(pdv_m[base_key]), arr_u[dv_idx]])[:, 1].astype(np.float32)
            pa_m[nom], pv_m[nom], pdv_m[nom] = pa_e, pv_e, pdv_e
            if pl_seed is None:
                log(f"  {nom} pesos A2-A4: coef={np.round(me_f.coef_[0], 3)}, b={me_f.intercept_[0]:+.3f}")

        eval_stage2("3_MT_A4C1_bs512", d_ens10[:, None], "6a_SingleMTC1_dEns10", pl_seed=None)
        eval_stage2("5_Blend_LR_MT", d_ens10[:, None], "6_Final_dEns10", pl_seed=None)
        eval_stage2("5_Blend_LR_MT", d_ens10[:, None], "6p_dEns10_placebo", pl_seed=800 + m)

        post15_ext = np.nan_to_num(
            np.c_[hgb_post15, hgb_post15[:, 2:3] * np.exp(-np.abs(g_tot_m) / 3.5)[:, None]], nan=0.0
        ).astype(np.float32)
        eval_stage2("6_Final_dEns10", post15_ext, "7_Final_dEns10_Post15", pl_seed=None)
        eval_stage2("6_Final_dEns10", post15_ext, "7p_Post15_placebo", pl_seed=900 + m)

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
                f"  {nom:22s} | bloq {n_ok_b}/3 Δ{d_bl:+.4f} (A34 {d_a34:+.4f}) | "
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
        pd.DataFrame(filas).to_csv(IN_DIR / "mt_a4_gpu_92k.csv", index=False)
        log(f"Minuto {m} terminado en {time.time() - tm:.1f}s")

    np.savez_compressed(IN_DIR / "pred_mt_a4_gpu_92k.npz", **saved)
    log(f"Listo total en {time.time() - T0:.1f}s")


if __name__ == "__main__":
    main()
