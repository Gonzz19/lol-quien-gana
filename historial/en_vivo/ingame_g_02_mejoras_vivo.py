"""Experimento (ii) y Prioridad 3: Escalón 9 sobre ~92k + mejoras no lineales en vivo (GPU RTX 3070).

Evalúa sobre `A2-A4 Master+` (3 bloques hacia adelante, 3/3 + IC), `V Master+` (IC pareado vs
referencia vigente) y `Diamond-V`, con placebo de filas permutadas para toda variable nueva:
  1. `L_esc9_92k`: `po_ref9`/`pv_ref9`/`p_dv_ref9` (`pred_g28_esc9.npz`) empalmado con
     Stage-1 Escalón 9 forward-chaining estricto en `chunks 1..3` (sin Foto 1 en `~post`).
  2. `H1_NoLin`: interacciones antisimétricas en vivo sobre `A2` limpio: decaimiento de
     `L_sym` con `|g_tot|`, saturación `tanh(g_tot/s)`, cruces `2v2`, ritmo×oro y sorpresa×cercanía
     (+ `H1_placebo` permutado).
  3. `H2_LK15_HGB`: señales causales de duelo `@15` (`LK_LANES` + `HGB`/`hgb_lineas_extra`)
     (+ `H2_placebo` permutado).
  4. `LGBM_Clean_Res`: LightGBM residual sobre `A2 + H1` (sin ruido `A4` pre-partida).
  5. `InGameNet_Clean_GPU`: `InGameNet_MultiTask` en RTX 3070 (4 semillas) sobre `A2 + H1` limpio
     y versión residual sobre `z_oof(A2+H1)`, más calibración y blend `A2+H1 + InGameNet`.

Uso:
    python historial/en_vivo/ingame_g_02_mejoras_vivo.py
"""
from __future__ import annotations

import gc
import math
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
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.analysis.ingame_evaluar import (  # noqa: E402
    _construir_aux_targets,
    _delta_ci,
    _ece,
    _fit_logreg_antisym,
    _fit_predict_lgb,
    _logit,
    _oof_logreg_logits,
    _prep_arrays,
    _sigm,
)
from historial.ensamble_en_vivo.ingame_g_01_prior_y_gpu import _cargar_priors, _fit_predict_ingamenet_gpu  # noqa: E402
from src.models.ingame import (  # noqa: E402
    InGameNet,
    MINUTES,
    TIMELINE_STATS,
    _antisym_2v2,
    extract_minute_features,
)

IN_DIR = REPO / "data" / "analysis" / "ingame"
BR_DIR = REPO / "data" / "analysis" / "brainstorm"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def construir_esc9_92k(mids: np.ndarray, chunk: np.ndarray, t: np.ndarray, bloque: np.ndarray,
                       val: np.ndarray, diam_v: np.ndarray, master: np.ndarray, L_orig: np.ndarray) -> np.ndarray:
    """Empalma `pred_g28_esc9.npz` (A2-A4, V, DiamV) con Escalón 9 OOF causal en chunks 1..3.

    En `chunks 1..3` (`~post`), enmascara `mmr_historico` y `FOTO1_DIN` (cols 150:157 de
    `extras_referencia(..., escalon=9)`) para no leer Foto 1 (04/09) en partidas anteriores a ella.
    """
    cache_l = IN_DIR / "L_esc9_92k.npz"
    if cache_l.exists():
        z = np.load(cache_l)
        if len(z["L_esc9_92k"]) == len(mids) and (z["mids"].astype(str) == mids.astype(str)).all():
            return z["L_esc9_92k"].astype(np.float32)

    from scripts.analysis.brainstorm_base import cargar, extras_referencia

    log("Calculando Escalón 9 causal para chunks 1..3 (sin Foto 1 en ~post)...")
    D = cargar()
    assert (D["Z"]["mids"].astype(str) == mids.astype(str)).all()
    X9 = extras_referencia(D, escalon=9).copy()
    # Sin Foto 1 en filas pre-foto (~post): columnas 150:157 son niv(3) + FOTO1_DIN(4)
    X9[~D["post"], 150:157] = 0.0

    priors = _cargar_priors(mids, bloque, val, diam_v, L_orig)
    out = priors["L_esc9"].copy()  # A2-A4 (13.8k), V (16.4k), DiamV (10.5k) ya finitos

    Xd = D["Xd"]
    DRAFT = D["DRAFT"]
    y = D["y"]
    for b in (1, 2, 3):
        te = np.flatnonzero(chunk == b)
        t_lim = int(t[te].min())
        tr = np.flatnonzero((chunk >= 0) & (chunk < b) & (t < t_lim) & master)
        mu_x, sd_x = Xd[tr].mean(0), Xd[tr].std(0) + 1e-6
        mu_e, sd_e = X9[tr].mean(0), X9[tr].std(0) + 1e-6
        Ptr = sparse.hstack([DRAFT[tr], sparse.csr_matrix(((Xd[tr] - mu_x) / sd_x).astype(np.float32)),
                             sparse.csr_matrix(((X9[tr] - mu_e) / sd_e).astype(np.float32))]).tocsr()
        Pte = sparse.hstack([DRAFT[te], sparse.csr_matrix(((Xd[te] - mu_x) / sd_x).astype(np.float32)),
                             sparse.csr_matrix(((X9[te] - mu_e) / sd_e).astype(np.float32))]).tocsr()
        lr = LogisticRegression(C=0.05, max_iter=3000).fit(Ptr, y[tr])
        p_te = np.clip(lr.predict_proba(Pte)[:, 1], 1e-5, 1.0 - 1e-5)
        out[te] = np.log(p_te / (1.0 - p_te)).astype(np.float32)
        log(f"  chunk {b}: tr={len(tr):,}, te={len(te):,}, AUC(master)={roc_auc_score(y[te[master[te]]], out[te[master[te]]]):.4f}")

    del D, X9, Xd, DRAFT
    gc.collect()
    np.savez_compressed(cache_l, mids=mids, L_esc9_92k=out)
    return out


def _agregar_h1_nolin(fdict: dict, minute: int, L_arr: np.ndarray, tl_raw: np.ndarray, exp_lane: dict) -> dict:
    """Agrega interacciones antisimétricas en vivo sobre A2 (cero fuga, antisimetría exacta)."""
    mi = MINUTES.index(minute)
    stat_idx = {s: i for i, s in enumerate(TIMELINE_STATS)}
    cur = tl_raw[:, :, :, mi, :]
    d_lane = cur[:, 0] - cur[:, 1]
    s_team = cur.sum(axis=2)[:, 0] + cur.sum(axis=2)[:, 1]
    g_l = d_lane[:, :, stat_idx["gold"]]
    g_tot = g_l.sum(axis=1)
    x_tot = d_lane[:, :, stat_idx["xp"]].sum(axis=1)
    tot_k_norm = ((s_team[:, stat_idx["kills"]] - 6.0 * (minute / 10.0)) / 8.0).astype(np.float32)
    tot_d_norm = ((s_team[:, stat_idx["damage"]] - 30.0 * (minute / 10.0)) / 25.0).astype(np.float32)

    exp_g_key = f"gold_{minute}"
    surp_g_tot = (g_l - exp_lane[exp_g_key]).sum(axis=1) if exp_g_key in exp_lane else np.zeros_like(g_tot)

    h1_cols = np.column_stack([
        L_arr * np.exp(-np.abs(g_tot) / 2.5),
        L_arr * np.exp(-np.abs(g_tot) / 5.0),
        np.tanh(g_tot / 3.0),
        np.tanh(g_tot / 6.0),
        np.tanh(x_tot / 3.0),
        g_tot * tot_k_norm,
        g_tot * tot_d_norm,
        L_arr * tot_k_norm,
        surp_g_tot * np.exp(-np.abs(g_tot) / 3.0),
        _antisym_2v2(g_l[:, 0], g_l[:, 4]),  # BOT + SUP
        _antisym_2v2(g_l[:, 2], g_l[:, 1]),  # MID + JG
        _antisym_2v2(g_l[:, 3], g_l[:, 1]),  # TOP + JG
    ]).astype(np.float32)
    h1_names = [
        "h1_L_decay25", "h1_L_decay50", "h1_tanh_g3", "h1_tanh_g6", "h1_tanh_x3",
        "h1_g_pace_k", "h1_g_pace_dmg", "h1_L_pace_k", "h1_surp_close",
        "h1_2v2_botsup", "h1_2v2_midjg", "h1_2v2_topjg",
    ]
    out = dict(fdict)
    out["X_anti"] = np.c_[fdict["X_anti"], h1_cols].astype(np.float32)
    out["names_anti"] = list(fdict["names_anti"]) + h1_names
    return out


def _agregar_h2_duelo(fdict: dict, minute: int, tl_raw: np.ndarray, lk_lanes: np.ndarray,
                      hgb_g15: np.ndarray, hgb_ext: dict) -> dict:
    """Agrega expectativas HGB externas + Kalman 1v1 por carril `@15` (antisimétricas)."""
    mi = MINUTES.index(minute)
    stat_idx = {s: i for i, s in enumerate(TIMELINE_STATS)}
    g_l = tl_raw[:, :, :, mi, stat_idx["gold"]][:, 0] - tl_raw[:, :, :, mi, stat_idx["gold"]][:, 1]
    x_l = tl_raw[:, :, :, mi, stat_idx["xp"]][:, 0] - tl_raw[:, :, :, mi, stat_idx["xp"]][:, 1]

    if minute == 10 and "gold_10" in hgb_ext:
        parts = [
            g_l - hgb_ext["gold_10"],
            x_l - hgb_ext["xp_10"],
            hgb_g15,
            lk_lanes[:, :5] * np.exp(-np.abs(g_l) / 2.0),
            lk_lanes[:, 5:6],
        ]
        names = (
            [f"h2_surp_hgbg10_{i}" for i in range(5)]
            + [f"h2_surp_hgbx10_{i}" for i in range(5)]
            + [f"h2_hgb_g15_{i}" for i in range(5)]
            + [f"h2_lk15_close_{i}" for i in range(5)]
            + ["h2_lk15_prec"]
        )
    else:
        parts = [
            g_l - hgb_g15,
            x_l - hgb_ext["xp_15"],
            lk_lanes[:, :5] * np.exp(-np.abs(g_l) / 2.0),
            lk_lanes[:, 5:6],
        ]
        names = (
            [f"h2_surp_hgbg15_{i}" for i in range(5)]
            + [f"h2_surp_hgbx15_{i}" for i in range(5)]
            + [f"h2_lk15_close_{i}" for i in range(5)]
            + ["h2_lk15_prec"]
        )

    out = dict(fdict)
    out["X_anti"] = np.c_[fdict["X_anti"], np.column_stack(parts).astype(np.float32)].astype(np.float32)
    out["names_anti"] = list(fdict["names_anti"]) + names
    return out


def _permutar_nuevas(f_full: dict, f_base: dict, seed: int) -> dict:
    """Placebo estricto: permuta por filas SOLO las columnas nuevas agregadas sobre `f_base`."""
    d0 = f_base["X_anti"].shape[1]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(f_full["X_anti"]))
    out = dict(f_full)
    Xa = f_full["X_anti"].copy()
    Xa[:, d0:] = Xa[perm, d0:]
    out["X_anti"] = Xa
    return out


def _fit_predict_ingamenet_res_gpu(
    minute: int,
    arrs: dict,
    y_tr: np.ndarray,
    init_tr: np.ndarray,
    init_te: np.ndarray,
    aux_tr: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    seed: int = 0,
    bs: int = 1024,
    epochs: int = 18,
) -> np.ndarray:
    """InGameNet MultiTask entrenado como residual puro sobre logits `init_tr/init_te`."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    b0 = float(np.mean(init_tr))
    init_a_tr = (init_tr - b0).astype(np.float32)
    init_a_te = (init_te - b0).astype(np.float32)

    Lb_tr, Lr_tr, Xa_tr, Xs_tr = arrs["Lb_tr"], arrs["Lr_tr"], arrs["Xa_tr"], arrs["Xs_tr"]
    Lb_te, Lr_te, Xa_te, Xs_te = arrs["Lb_te"], arrs["Lr_te"], arrs["Xa_te"], arrs["Xs_te"]

    n = len(y_tr)
    cut = int(n * 0.88)
    tr_s, va_s = np.arange(cut), np.arange(cut, n)

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
    opt = torch.optim.AdamW(net.parameters(), lr=1.5e-3, weight_decay=8e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    t_lb = torch.tensor(Lb_tr[tr_s], device=DEV)
    t_lr = torch.tensor(Lr_tr[tr_s], device=DEV)
    t_xa = torch.tensor(Xa_tr[tr_s], device=DEV)
    t_xs = torch.tensor(Xs_tr[tr_s], device=DEV)
    t_z0 = torch.tensor(init_a_tr[tr_s], device=DEV)
    t_y = torch.tensor(y_tr[tr_s], dtype=torch.float32, device=DEV)
    t_ya = torch.tensor(aux_tr[0][tr_s], device=DEV)
    t_ma = torch.tensor(aux_tr[1][tr_s], dtype=torch.float32, device=DEV)
    t_ys = torch.tensor(aux_tr[2][tr_s], device=DEV)
    t_ms = torch.tensor(aux_tr[3][tr_s], dtype=torch.float32, device=DEV)

    v_lb = torch.tensor(Lb_tr[va_s], device=DEV)
    v_lr = torch.tensor(Lr_tr[va_s], device=DEV)
    v_xa = torch.tensor(Xa_tr[va_s], device=DEV)
    v_xs = torch.tensor(Xs_tr[va_s], device=DEV)
    v_z0 = init_a_tr[va_s]
    v_y = y_tr[va_s]

    best_loss = float("inf")
    best_state = None
    rng = np.random.default_rng(seed)

    for _ in range(epochs):
        net.train()
        perm = rng.permutation(len(tr_s))
        for b in range(0, len(perm), bs):
            ii = perm[b : b + bs]
            d_log, out_a, out_s = net(t_lb[ii], t_lr[ii], t_xa[ii], t_xs[ii])
            logit = t_z0[ii] + d_log
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
            vl = v_z0 + net(v_lb, v_lr, v_xa, v_xs)[0].detach().cpu().numpy()
            vloss = log_loss(v_y, _sigm(vl))
        if vloss < best_loss:
            best_loss = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    if best_state is not None:
        net.load_state_dict({k: v.to(DEV) for k, v in best_state.items()})
    net.eval()
    preds = []
    with torch.no_grad():
        for b in range(0, len(Xa_te), 4096):
            d_b = net(
                torch.tensor(Lb_te[b : b + 4096], device=DEV),
                torch.tensor(Lr_te[b : b + 4096], device=DEV),
                torch.tensor(Xa_te[b : b + 4096], device=DEV),
                torch.tensor(Xs_te[b : b + 4096], device=DEV),
            )[0].detach().cpu().numpy()
            preds.append(init_a_te[b : b + 4096] + d_b)
    return np.concatenate(preds).astype(np.float32)


def main() -> None:
    cache_path = IN_DIR / "ingame_cache.npz"
    if not cache_path.exists():
        sys.exit(f"Falta {cache_path}")

    log(f"Iniciando Experimento (ii) + Prioridad 3 en {DEV}...")
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
    del C
    gc.collect()

    L_esc9 = construir_esc9_92k(mids, chunk, t, bloque, val, diam_v, master, L_orig)
    priors = _cargar_priors(mids, bloque, val, diam_v, L_orig)
    L_ens10 = priors["L_ens10"]
    d_ens10 = np.where(np.isfinite(L_ens10) & np.isfinite(L_esc9), L_ens10 - L_esc9, 0.0).astype(np.float32)
    del priors
    gc.collect()

    C = np.load(cache_path, allow_pickle=True)
    tl_raw = C["tl_raw"].astype(np.float32)
    gold_resid = C["gold_resid"].astype(np.float32)
    end_gold_diff = C["end_gold_diff"].astype(np.float32)
    end_kills_diff = C["end_kills_diff"].astype(np.float32)
    exp_lane = {k[4:]: C[k].astype(np.float32) for k in C.files if k.startswith("exp_")}
    exp_trans = {k[3:]: C[k].astype(np.float32) for k in C.files if k.startswith("tr_")}
    del C
    gc.collect()

    lk_lanes = np.load(BR_DIR / "gemini_g30_lk15.npz")["LK_LANES"].astype(np.float32)
    hgb_g15 = np.load(BR_DIR / "gemini_hgb_lineas.npz")["HGB"][:, :5].astype(np.float32)
    hgb_ext = dict(np.load(BR_DIR / "hgb_lineas_extra.npz"))

    lap_z = np.load(IN_DIR / "pred_ingame.npz")
    lap_best_key = {10: "3_Abl_A2_Stats_Traj_Exp", 15: "10_InGameNet_MultiTask", 20: "10_InGameNet_MultiTask"}
    lap_ref = {}
    for mm, bk in lap_best_key.items():
        lap_ref[mm] = {
            "v_idx": lap_z[f"v_idx_{mm}"],
            "y_v": lap_z[f"y_v_{mm}"],
            "p_v": lap_z[f"p_{mm}_{bk}"].astype(np.float32),
            "name": bk,
        }
    del lap_z
    gc.collect()

    filas = []
    saved = {}

    for mi, m in enumerate(MINUTES):
        tm = time.time()
        log(f"\n==================== MINUTO {m} (92k + Mejoras) ====================")
        ok_tl = np.isfinite(tl_raw[:, :, :, : mi + 1, :]).all(axis=(1, 2, 3, 4))
        ok_m = (dur_sec > 60.0 * m) & ok_tl & np.isfinite(L_orig) & np.isfinite(L_esc9)

        v_idx = np.flatnonzero(val & ok_m)
        dv_idx = np.flatnonzero(diam_v & ok_m)
        folds_a24 = [(b, np.flatnonzero((chunk == b) & master & ok_m)) for b in (4, 5, 6)]
        tr_full = np.flatnonzero(np.isin(chunk, [1, 2, 3, 4, 5, 6]) & master & ok_m)
        a24_idx = np.concatenate([te for _, te in folds_a24])
        rng = np.random.default_rng(200 + m)
        boot_v = rng.integers(0, len(v_idx), size=(1000, len(v_idx)))
        yv, ydv = y[v_idx], y[dv_idx]

        def eval_lr(fdict, L_arr, tune_C=False):
            best_C = 0.05
            if tune_C:
                best_ll = float("inf")
                for c_try in (0.01, 0.03, 0.05, 0.1):
                    pa_t = np.full(len(y), np.nan, np.float32)
                    for b, te in folds_a24:
                        tr_b = np.flatnonzero(np.isin(chunk, [1, 2, 3, 4, 5, 6]) & (chunk < b) & master & ok_m)
                        ar = _prep_arrays(fdict, tr_b, te, L_arr, y)
                        pa_t[te] = _sigm(_fit_logreg_antisym(ar["Xa_tr"], y[tr_b], ar["Xa_te"], C=c_try)[1])
                    ll = log_loss(y[a24_idx], pa_t[a24_idx])
                    if ll < best_ll:
                        best_ll, best_C = ll, c_try
            pa = np.full(len(y), np.nan, np.float32)
            for b, te in folds_a24:
                tr_b = np.flatnonzero(np.isin(chunk, [1, 2, 3, 4, 5, 6]) & (chunk < b) & master & ok_m)
                ar = _prep_arrays(fdict, tr_b, te, L_arr, y)
                pa[te] = _sigm(_fit_logreg_antisym(ar["Xa_tr"], y[tr_b], ar["Xa_te"], C=best_C)[1])
            te_vd = np.r_[v_idx, dv_idx]
            ar_f = _prep_arrays(fdict, tr_full, te_vd, L_arr, y)
            _, z_vd, _ = _fit_logreg_antisym(ar_f["Xa_tr"], y[tr_full], ar_f["Xa_te"], C=best_C)
            p_vd = _sigm(z_vd)
            return pa, p_vd[: len(v_idx)], p_vd[len(v_idx) :], best_C

        pa_m, pv_m, pdv_m = {}, {}, {}

        # Evaluar una variante por vez para mantener RAM < 1.5 GB
        f_tmp = extract_minute_features(m, L_orig, tl_raw, gold_resid, exp_lane=exp_lane, exp_trans=exp_trans,
                                        include_all_stats=True, include_traj=True, include_expected=True,
                                        include_comp_cross=False, include_player=False)
        pa_m["0_Lorig_A2"], pv_m["0_Lorig_A2"], pdv_m["0_Lorig_A2"], _ = eval_lr(f_tmp, L_orig, tune_C=True)
        del f_tmp; gc.collect()

        f_a2_esc9 = extract_minute_features(m, L_esc9, tl_raw, gold_resid, exp_lane=exp_lane, exp_trans=exp_trans,
                                            include_all_stats=True, include_traj=True, include_expected=True,
                                            include_comp_cross=False, include_player=False)
        pa_m["1_Esc9_A2"], pv_m["1_Esc9_A2"], pdv_m["1_Esc9_A2"], c_9 = eval_lr(f_a2_esc9, L_esc9, tune_C=True)

        f_h2 = _agregar_h2_duelo(f_a2_esc9, m, tl_raw, lk_lanes, hgb_g15, hgb_ext)
        pa_m["3_Esc9_A2_H2"], pv_m["3_Esc9_A2_H2"], pdv_m["3_Esc9_A2_H2"], _ = eval_lr(f_h2, L_esc9, tune_C=True)
        f_h2_pl = _permutar_nuevas(f_h2, f_a2_esc9, seed=20 + m)
        pa_m["3p_Esc9_H2_placebo"], pv_m["3p_Esc9_H2_placebo"], pdv_m["3p_Esc9_H2_placebo"], _ = eval_lr(f_h2_pl, L_esc9, tune_C=True)
        del f_h2, f_h2_pl; gc.collect()

        f_h1 = _agregar_h1_nolin(f_a2_esc9, m, L_esc9, tl_raw, exp_lane)
        pa_m["2_Esc9_A2_H1"], pv_m["2_Esc9_A2_H1"], pdv_m["2_Esc9_A2_H1"], c_h1 = eval_lr(f_h1, L_esc9, tune_C=True)
        f_h1_pl = _permutar_nuevas(f_h1, f_a2_esc9, seed=10 + m)
        pa_m["2p_Esc9_H1_placebo"], pv_m["2p_Esc9_H1_placebo"], pdv_m["2p_Esc9_H1_placebo"], _ = eval_lr(f_h1_pl, L_esc9, tune_C=True)
        del f_h1_pl, f_a2_esc9; gc.collect()

        f_h12 = _agregar_h2_duelo(f_h1, m, tl_raw, lk_lanes, hgb_g15, hgb_ext)
        pa_m["4_Esc9_A2_H12"], pv_m["4_Esc9_A2_H12"], pdv_m["4_Esc9_A2_H12"], _ = eval_lr(f_h12, L_esc9, tune_C=True)
        del f_h12; gc.collect()

        c_use = c_h1
        # LGBM residual limpio sobre 2_Esc9_A2_H1 + InGameNet_Clean_GPU + Residual_GPU (4 semillas)
        pa_lgb = np.full(len(y), np.nan, np.float32)
        pa_mt_clean = np.full(len(y), np.nan, np.float32)
        pa_mt_res = np.full(len(y), np.nan, np.float32)
        for b, te in folds_a24:
            tr_b = np.flatnonzero(np.isin(chunk, [1, 2, 3, 4, 5, 6]) & (chunk < b) & master & ok_m)
            ar = _prep_arrays(f_h1, tr_b, te, L_esc9, y)
            _, z_te_lr, _ = _fit_logreg_antisym(ar["Xa_tr"], y[tr_b], ar["Xa_te"], C=c_use)
            z_oof_lr = _oof_logreg_logits(ar["Xa_tr"], y[tr_b], t[tr_b], C=c_use, n_splits=4)
            z_lgb, _ = _fit_predict_lgb(ar["Xa_tr"], ar["Xs_tr"], y[tr_b], ar["Xa_te"], ar["Xs_te"],
                                        init_tr=z_oof_lr, init_te=z_te_lr, n_estimators=240)
            pa_lgb[te] = _sigm(z_lgb)
            ya, ma, ys, ms_, _ = _construir_aux_targets(m, tr_b, tl_raw, dur_sec, end_gold_diff, end_kills_diff)
            zc = np.mean([_fit_predict_ingamenet_gpu(m, ar, y[tr_b], (ya, ma, ys, ms_), seed=30 * b + s, bs=2048) for s in (0, 1)], axis=0)
            zr = np.mean([_fit_predict_ingamenet_res_gpu(m, ar, y[tr_b], z_oof_lr, z_te_lr, (ya, ma, ys, ms_), seed=40 * b + s, bs=2048) for s in (0, 1)], axis=0)
            pa_mt_clean[te] = _sigm(zc)
            pa_mt_res[te] = _sigm(zr)

        te_vd = np.r_[v_idx, dv_idx]
        ar_f = _prep_arrays(f_h1, tr_full, te_vd, L_esc9, y)
        _, z_vd_lr, _ = _fit_logreg_antisym(ar_f["Xa_tr"], y[tr_full], ar_f["Xa_te"], C=c_use)
        z_oof_f = _oof_logreg_logits(ar_f["Xa_tr"], y[tr_full], t[tr_full], C=c_use, n_splits=5)
        z_vd_lgb, _ = _fit_predict_lgb(ar_f["Xa_tr"], ar_f["Xs_tr"], y[tr_full], ar_f["Xa_te"], ar_f["Xs_te"],
                                       init_tr=z_oof_f, init_te=z_vd_lr, n_estimators=280)
        ya_f, ma_f, ys_f, ms_f, _ = _construir_aux_targets(m, tr_full, tl_raw, dur_sec, end_gold_diff, end_kills_diff)
        zc_vd = np.mean([_fit_predict_ingamenet_gpu(m, ar_f, y[tr_full], (ya_f, ma_f, ys_f, ms_f), seed=300 + s, bs=2048) for s in (0, 1, 2, 3)], axis=0)
        zr_vd = np.mean([_fit_predict_ingamenet_res_gpu(m, ar_f, y[tr_full], z_oof_f, z_vd_lr, (ya_f, ma_f, ys_f, ms_f), seed=400 + s, bs=2048) for s in (0, 1, 2, 3)], axis=0)
        del f_h1, ar_f; gc.collect()

        for k, p_a, z_vd in [("5_LGBM_Clean_Res", pa_lgb, z_vd_lgb),
                             ("6_MT_Clean_GPU", pa_mt_clean, zc_vd),
                             ("7_MT_Res_GPU", pa_mt_res, zr_vd)]:
            p_f = _sigm(z_vd)
            pa_m[k], pv_m[k], pdv_m[k] = p_a, p_f[: len(v_idx)], p_f[len(v_idx) :]

        # Blends estrictamente OOF hacia adelante en A2-A4 (A2: media igual; A3 <- A2; A4 <- A2+A3)
        # y calibración sobre A2-A4 completa para predecir V Master+ y Diamond-V sin fuga.
        blend_specs = [
            ("8_Blend_H1_MTClean", ["2_Esc9_A2_H1", "6_MT_Clean_GPU"]),
            ("9_Blend_H1_MTRes", ["2_Esc9_A2_H1", "7_MT_Res_GPU"]),
            ("10_Blend_All4", ["2_Esc9_A2_H1", "5_LGBM_Clean_Res", "6_MT_Clean_GPU", "7_MT_Res_GPU"]),
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

        # Mejora incremental de d_ens10 = L_ens10 - L_esc9 sobre Blend_All4 (A3 <- A2, A4 <- A2-A3; V/DiamV <- A2-A4)
        # más placebo por permutación de d_ens10 dentro de cada partición.
        rng_pl = np.random.default_rng(700 + m)
        d_ens_pl = d_ens10.copy()
        for mask_pl in (folds_a24[0][1], folds_a24[1][1], folds_a24[2][1], v_idx, dv_idx):
            d_ens_pl[mask_pl] = d_ens10[rng_pl.permutation(mask_pl)]

        for e_nom, d_arr in [("11_Blend_All4_dEns10", d_ens10), ("11p_dEns10_placebo", d_ens_pl)]:
            pa_e = np.full(len(y), np.nan, np.float32)
            for bi, (_, te) in enumerate(folds_a24):
                if bi == 0:
                    pa_e[te] = pa_m["10_Blend_All4"][te]
                else:
                    tr_prev = np.concatenate([folds_a24[j][1] for j in range(bi)])
                    X_pr = np.c_[_logit(pa_m["10_Blend_All4"][tr_prev]), d_arr[tr_prev]]
                    X_te = np.c_[_logit(pa_m["10_Blend_All4"][te]), d_arr[te]]
                    me = LogisticRegression(C=0.5, max_iter=1000).fit(X_pr, y[tr_prev])
                    pa_e[te] = me.predict_proba(X_te)[:, 1].astype(np.float32)
            X_a = np.c_[_logit(pa_m["10_Blend_All4"][a24_idx]), d_arr[a24_idx]]
            me_f = LogisticRegression(C=0.5, max_iter=1000).fit(X_a, y[a24_idx])
            pv_e = me_f.predict_proba(np.c_[_logit(pv_m["10_Blend_All4"]), d_arr[v_idx]])[:, 1].astype(np.float32)
            pdv_e = me_f.predict_proba(np.c_[_logit(pdv_m["10_Blend_All4"]), d_arr[dv_idx]])[:, 1].astype(np.float32)
            pa_m[e_nom], pv_m[e_nom], pdv_m[e_nom] = pa_e, pv_e, pdv_e
            if e_nom == "11_Blend_All4_dEns10":
                log(f"  {e_nom} pesos A2-A4 (z_blend, d_ens10): coef={np.round(me_f.coef_[0], 3)}, b={me_f.intercept_[0]:+.3f}")

        assert np.array_equal(lap_ref[m]["v_idx"], v_idx), f"v_idx distinto de pred_ingame.npz en m={m}"
        assert np.array_equal(lap_ref[m]["y_v"], yv), f"y_v distinto de pred_ingame.npz en m={m}"
        p_lap_v = lap_ref[m]["p_v"]
        rng_dv = np.random.default_rng(500 + m)
        boot_dv = rng_dv.integers(0, len(dv_idx), size=(1000, len(dv_idx)))

        ref_0 = pv_m["0_Lorig_A2"]
        ref_1 = pv_m["1_Esc9_A2"]
        ref_dv0 = pdv_m["0_Lorig_A2"]
        for nom, pv in pv_m.items():
            auc_v = roc_auc_score(yv, pv)
            acc_v = float(((pv > 0.5) == yv).mean())
            ll_v = log_loss(yv, pv)
            ece_v = _ece(yv, pv)
            aucs_b = [roc_auc_score(y[te], pa_m[nom][te]) for _, te in folds_a24]
            aucs_0 = [roc_auc_score(y[te], pa_m["0_Lorig_A2"][te]) for _, te in folds_a24]
            d_bl = np.mean([a - b0 for a, b0 in zip(aucs_b, aucs_0)])
            n_ok_b = sum(a > b0 for a, b0 in zip(aucs_b, aucs_0))
            pdv = pdv_m[nom]
            auc_dv = roc_auc_score(ydv, pdv)
            acc_dv = float(((pdv > 0.5) == ydv).mean())
            d0, lo0, hi0 = _delta_ci(yv, pv, ref_0, boot_v)
            d1, lo1, hi1 = _delta_ci(yv, pv, ref_1, boot_v)
            dl, lol, hil = _delta_ci(yv, pv, p_lap_v, boot_v)
            ddv, lodv, hidv = _delta_ci(ydv, pdv, ref_dv0, boot_dv)
            print(
                f"  {nom:22s} | bloq Δ{d_bl:+.4f} ({n_ok_b}/3) | V {auc_v:.4f}/{acc_v:.2%} LL {ll_v:.4f} ECE {ece_v:.4f} | "
                f"ΔorigA2 {d0:+.4f} [{lo0:+.4f};{hi0:+.4f}] | Δesc9A2 {d1:+.4f} [{lo1:+.4f};{hi1:+.4f}] | "
                f"Δlap({lap_ref[m]['name'][:6]}) {dl:+.4f} [{lol:+.4f};{hil:+.4f}] | "
                f"DiamV {auc_dv:.4f}/{acc_dv:.2%} ({ddv:+.4f} [{lodv:+.4f};{hidv:+.4f}])",
                flush=True,
            )
            filas.append({
                "minuto": m, "modelo": nom, "n_V": len(v_idx),
                "AUC_A2": aucs_b[0], "AUC_A3": aucs_b[1], "AUC_A4": aucs_b[2],
                "dBloq_vs_orig": d_bl, "n_Bloq_pos": n_ok_b,
                "AUC_V": auc_v, "Acc_V": acc_v, "LogLoss_V": ll_v, "ECE_V": ece_v,
                "dV_vs_origA2": d0, "lo0": lo0, "hi0": hi0,
                "dV_vs_esc9A2": d1, "lo1": lo1, "hi1": hi1,
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
        log(f"Minuto {m} terminado en {time.time() - tm:.1f}s")
        pd.DataFrame(filas).to_csv(IN_DIR / "mejoras_vivo_92k.csv", index=False)

    np.savez_compressed(IN_DIR / "pred_mejoras_vivo_92k.npz", **saved)
    log(f"Listo total en {time.time() - T0:.1f}s")


if __name__ == "__main__":
    main()
