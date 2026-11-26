"""Experimento (i) Prioridad 1 in-game: comparar Stage-1 L_oof (0,6676) vs Escalón 9 (0,6739)
vs Ensamble Escalón 10 (0,6749) sobre A2-A4 (13,8k Master+), V Master+ y Diamond-V.

Uso:
    python historial/ensamble_en_vivo/ingame_g_01_prior_y_gpu.py
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
    _prep_arrays,
    _sigm,
)
from src.models.ingame import InGameNet, MINUTES, TIMELINE_STATS, extract_minute_features  # noqa: E402

IN_DIR = REPO / "data" / "analysis" / "ingame"
BR_DIR = REPO / "data" / "analysis" / "brainstorm"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def _cargar_priors(mids: np.ndarray, bloque: np.ndarray, val: np.ndarray, diam_v: np.ndarray, L_orig: np.ndarray) -> dict[str, np.ndarray]:
    """Construye los logits pre-partida alineados con `ingame_cache.npz` sin fuga temporal."""
    Z_var = np.load(REPO / "data" / "analysis" / "embeddings_partidas" / "variables.npz", allow_pickle=True)
    assert len(Z_var["mids"]) == len(mids) and (Z_var["mids"].astype(str) == mids.astype(str)).all(), "Desalineación con variables.npz"
    del Z_var

    P = np.load(BR_DIR / "pred_g28_esc9.npz")
    m_a24 = np.isin(bloque, [1, 2, 3]) & (~diam_v)

    def armar(mix_w: tuple[float, float, float, float]) -> np.ndarray:
        out = np.full(len(mids), np.nan, dtype=np.float32)
        keys = ("ref9", "nac", "att", "cb")
        po = np.zeros(len(mids), dtype=np.float64)
        pv = np.zeros(int(val.sum()), dtype=np.float64)
        pdv = np.zeros(int(diam_v.sum()), dtype=np.float64)
        for w, k in zip(mix_w, keys):
            if w == 0.0:
                continue
            po[m_a24] += w * _logit(P[f"po_{k}"][m_a24])
            pv += w * _logit(P[f"pv_{k}"])
            pdv += w * _logit(P[f"p_dv_{k}"])
        out[m_a24] = po[m_a24].astype(np.float32)
        out[val] = pv.astype(np.float32)
        out[diam_v] = pdv.astype(np.float32)
        return out

    return {
        "L_orig": L_orig.astype(np.float32),
        "L_esc9": armar((1.0, 0.0, 0.0, 0.0)),
        "L_ens10": armar((0.50, 0.25, 0.0, 0.25)),
    }


def _fit_predict_ingamenet_gpu(
    minute: int,
    arrs: dict,
    y_tr: np.ndarray,
    aux_tr: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    seed: int = 0,
    bs: int = 512,
    epochs: int = 18,
) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)

    Lb_tr, Lr_tr = arrs["Lb_tr"], arrs["Lr_tr"]
    Xa_tr, Xs_tr = arrs["Xa_tr"], arrs["Xs_tr"]
    Lb_te, Lr_te = arrs["Lb_te"], arrs["Lr_te"]
    Xa_te, Xs_te = arrs["Xa_te"], arrs["Xs_te"]

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
    opt = torch.optim.AdamW(net.parameters(), lr=2.0e-3, weight_decay=5e-3)
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

    v_lb = torch.tensor(Lb_tr[va_s], device=DEV)
    v_lr = torch.tensor(Lr_tr[va_s], device=DEV)
    v_xa = torch.tensor(Xa_tr[va_s], device=DEV)
    v_xs = torch.tensor(Xs_tr[va_s], device=DEV)
    v_y = y_tr[va_s]

    best_loss = float("inf")
    best_state = None
    rng = np.random.default_rng(seed)

    for _ in range(epochs):
        net.train()
        perm = rng.permutation(len(tr_s))
        for b in range(0, len(perm), bs):
            ii = perm[b : b + bs]
            logit, out_a, out_s = net(t_lb[ii], t_lr[ii], t_xa[ii], t_xs[ii])
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
            vl = net(v_lb, v_lr, v_xa, v_xs)[0].detach().cpu().numpy()
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
            l_b = net(
                torch.tensor(Lb_te[b : b + 4096], device=DEV),
                torch.tensor(Lr_te[b : b + 4096], device=DEV),
                torch.tensor(Xa_te[b : b + 4096], device=DEV),
                torch.tensor(Xs_te[b : b + 4096], device=DEV),
            )[0].detach().cpu().numpy()
            preds.append(l_b)
    return np.concatenate(preds).astype(np.float32)


def main() -> None:
    cache_path = IN_DIR / "ingame_cache.npz"
    if not cache_path.exists():
        sys.exit(f"Falta {cache_path}")

    log(f"Cargando caché en {DEV}...")
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
    end_gold_diff = C["end_gold_diff"].astype(np.float32)
    end_kills_diff = C["end_kills_diff"].astype(np.float32)
    exp_lane = {k[4:]: C[k].astype(np.float32) for k in C.files if k.startswith("exp_")}
    exp_trans = {k[3:]: C[k].astype(np.float32) for k in C.files if k.startswith("tr_")}
    del C
    gc.collect()

    priors = _cargar_priors(mids, bloque, val, diam_v, L_orig)
    for k, arr in priors.items():
        log(f"Prior {k}: V Master+ AUC={roc_auc_score(y[val], arr[val]):.4f} | DiamV={roc_auc_score(y[diam_v], arr[diam_v]):.4f}")

    pred_lap = np.load(IN_DIR / "pred_ingame.npz") if (IN_DIR / "pred_ingame.npz").exists() else None
    stat_i = {s: i for i, s in enumerate(TIMELINE_STATS)}
    filas = []
    saved = {}

    for mi, m in enumerate(MINUTES):
        tm = time.time()
        log(f"\n==================== MINUTO {m} (A2-A4 13,8k) ====================")
        ok_tl = np.isfinite(tl_raw[:, :, :, : mi + 1, :]).all(axis=(1, 2, 3, 4))
        ok_m = (dur_sec > 60.0 * m) & ok_tl & np.isfinite(priors["L_orig"]) & np.isfinite(priors["L_esc9"])

        v_idx = np.flatnonzero(val & ok_m)
        dv_idx = np.flatnonzero(diam_v & ok_m)
        tr_13k = np.flatnonzero(np.isin(chunk, [4, 5, 6]) & master & ok_m)
        folds_a34 = [(b, np.flatnonzero((chunk == b) & master & ok_m)) for b in (5, 6)]
        a34_idx = np.concatenate([te for _, te in folds_a34])
        log(f"Válidas m={m}: tr_13k={len(tr_13k):,}, A3-A4={len(a34_idx):,}, V={len(v_idx):,}, DiamV={len(dv_idx):,}")

        rng = np.random.default_rng(100 + m)
        boot_v = rng.integers(0, len(v_idx), size=(1000, len(v_idx)))
        boot_dv = rng.integers(0, len(dv_idx), size=(1000, len(dv_idx)))
        yv, ydv = y[v_idx], y[dv_idx]
        pv_map: dict[str, np.ndarray] = {}
        pa_map: dict[str, np.ndarray] = {}
        pdv_map: dict[str, np.ndarray] = {}

        if pred_lap is not None:
            assert (pred_lap[f"v_idx_{m}"] == v_idx).all(), f"v_idx_{m} difiere"
            pv_map["lap_A2_92k"] = pred_lap[f"p_{m}_3_Abl_A2_Stats_Traj_Exp"]
            pv_map["lap_MT_92k"] = pred_lap[f"p_{m}_10_InGameNet_MultiTask"]

        for p_name, L_arr in priors.items():
            f_eq = extract_minute_features(
                m, L_arr, tl_raw, gold_resid,
                include_all_stats=False, include_traj=False, include_expected=False,
                include_comp_cross=False, include_player=False,
            )
            g_l = tl_raw[:, 0, :, mi, stat_i["gold"]] - tl_raw[:, 1, :, mi, stat_i["gold"]]
            exp_ref = exp_lane["gold_15"] if m <= 15 else exp_lane["gold_20"]
            f_ls = dict(f_eq)
            f_ls["X_anti"] = np.c_[f_eq["X_anti"], g_l - exp_ref].astype(np.float32)

            f_a2 = extract_minute_features(
                m, L_arr, tl_raw, gold_resid, exp_lane=exp_lane, exp_trans=exp_trans,
                include_all_stats=True, include_traj=True, include_expected=True,
                include_comp_cross=False, include_player=False,
            )

            # Elegir C de A2 en A3-A4 OOF (sin tocar V!)
            best_C, best_ll = 0.05, float("inf")
            for cand_C in (0.01, 0.03, 0.05, 0.1):
                pa_tmp = np.full(len(y), np.nan, np.float32)
                for b, te in folds_a34:
                    tr_b = np.flatnonzero(np.isin(chunk, [4, 5, 6]) & (chunk < b) & master & ok_m)
                    ar = _prep_arrays(f_a2, tr_b, te, L_arr, y)
                    pa_tmp[te] = _sigm(_fit_logreg_antisym(ar["Xa_tr"], y[tr_b], ar["Xa_te"], C=cand_C)[1])
                ll = log_loss(y[a34_idx], pa_tmp[a34_idx])
                if ll < best_ll:
                    best_ll, best_C = ll, cand_C

            def run_lr(fdict, C_val):
                pa = np.full(len(y), np.nan, np.float32)
                for b, te in folds_a34:
                    tr_b = np.flatnonzero(np.isin(chunk, [4, 5, 6]) & (chunk < b) & master & ok_m)
                    ar = _prep_arrays(fdict, tr_b, te, L_arr, y)
                    pa[te] = _sigm(_fit_logreg_antisym(ar["Xa_tr"], y[tr_b], ar["Xa_te"], C=C_val)[1])
                te_vd = np.r_[v_idx, dv_idx]
                ar_v = _prep_arrays(fdict, tr_13k, te_vd, L_arr, y)
                p = _sigm(_fit_logreg_antisym(ar_v["Xa_tr"], y[tr_13k], ar_v["Xa_te"], C=C_val)[1])
                return pa, p[: len(v_idx)], p[len(v_idx) :]

            pa_ls, pv_ls, pdv_ls = run_lr(f_ls, 0.1)
            pa_a2, pv_a2, pdv_a2 = run_lr(f_a2, best_C)
            pv_map[f"{p_name}_LineasSorp"] = pv_ls; pa_map[f"{p_name}_LineasSorp"] = pa_ls; pdv_map[f"{p_name}_LineasSorp"] = pdv_ls
            pv_map[f"{p_name}_A2"] = pv_a2; pa_map[f"{p_name}_A2"] = pa_a2; pdv_map[f"{p_name}_A2"] = pdv_a2

            # InGameNet MultiTask GPU sobre f_a2 (3 semillas)
            pa_mt = np.full(len(y), np.nan, np.float32)
            for b, te in folds_a34:
                tr_b = np.flatnonzero(np.isin(chunk, [4, 5, 6]) & (chunk < b) & master & ok_m)
                ar = _prep_arrays(f_a2, tr_b, te, L_arr, y)
                ya, ma, ys, ms_, _ = _construir_aux_targets(m, tr_b, tl_raw, dur_sec, end_gold_diff, end_kills_diff)
                zm = np.mean([_fit_predict_ingamenet_gpu(m, ar, y[tr_b], (ya, ma, ys, ms_), seed=10 * b + s) for s in (0, 1)], axis=0)
                pa_mt[te] = _sigm(zm)

            te_vd = np.r_[v_idx, dv_idx]
            ar_f = _prep_arrays(f_a2, tr_13k, te_vd, L_arr, y)
            ya_f, ma_f, ys_f, ms_f, _ = _construir_aux_targets(m, tr_13k, tl_raw, dur_sec, end_gold_diff, end_kills_diff)
            zm_f = np.mean([_fit_predict_ingamenet_gpu(m, ar_f, y[tr_13k], (ya_f, ma_f, ys_f, ms_f), seed=200 + s) for s in (0, 1, 2)], axis=0)
            pm_f = _sigm(zm_f)
            pv_mt, pdv_mt = pm_f[: len(v_idx)], pm_f[len(v_idx) :]
            pv_map[f"{p_name}_MT_GPU"] = pv_mt; pa_map[f"{p_name}_MT_GPU"] = pa_mt; pdv_map[f"{p_name}_MT_GPU"] = pdv_mt

            # Blend 50/50 prefijado A2 + MT_GPU (cero tuning en V)
            lv_bl = 0.5 * _logit(pv_a2) + 0.5 * _logit(pv_mt)
            ldv_bl = 0.5 * _logit(pdv_a2) + 0.5 * _logit(pdv_mt)
            pa_bl = pa_a2.copy()
            pa_bl[a34_idx] = _sigm(0.5 * _logit(pa_a2[a34_idx]) + 0.5 * _logit(pa_mt[a34_idx]))
            pv_map[f"{p_name}_Blend"] = _sigm(lv_bl); pa_map[f"{p_name}_Blend"] = pa_bl; pdv_map[f"{p_name}_Blend"] = _sigm(ldv_bl)

        ref_orig = pv_map["L_orig_A2"]
        ref_orig_dv = pdv_map["L_orig_A2"]
        ref_lap_k = "lap_MT_92k" if m >= 15 and pred_lap is not None else ("lap_A2_92k" if pred_lap is not None else "L_orig_A2")
        ref_lap = pv_map[ref_lap_k]

        for nom, pv in pv_map.items():
            if nom.startswith("lap_"):
                auc_v = roc_auc_score(yv, pv)
                acc_v = float(((pv > 0.5) == yv).mean())
                ll_v = log_loss(yv, pv)
                ece_v = _ece(yv, pv)
                print(f"  {nom:24s} | A3-4   n/a   | V {auc_v:.4f} / {acc_v:.2%} LL {ll_v:.4f} ECE {ece_v:.4f}", flush=True)
                continue
            auc_v = roc_auc_score(yv, pv)
            acc_v = float(((pv > 0.5) == yv).mean())
            ll_v = log_loss(yv, pv)
            ece_v = _ece(yv, pv)
            auc_a3 = roc_auc_score(y[folds_a34[0][1]], pa_map[nom][folds_a34[0][1]])
            auc_a4 = roc_auc_score(y[folds_a34[1][1]], pa_map[nom][folds_a34[1][1]])
            auc_a34 = roc_auc_score(y[a34_idx], pa_map[nom][a34_idx])
            auc_dv = roc_auc_score(ydv, pdv_map[nom])
            acc_dv = float(((pdv_map[nom] > 0.5) == ydv).mean())
            d_o, lo_o, hi_o = _delta_ci(yv, pv, ref_orig, boot_v)
            d_dv, lo_dv, hi_dv = _delta_ci(ydv, pdv_map[nom], ref_orig_dv, boot_dv)
            d_l, lo_l, hi_l = _delta_ci(yv, pv, ref_lap, boot_v)
            print(
                f"  {nom:24s} | A3-4 {auc_a34:.4f} ({auc_a3:.4f}/{auc_a4:.4f}) | "
                f"V {auc_v:.4f} / {acc_v:.2%} LL {ll_v:.4f} ECE {ece_v:.4f} | "
                f"ΔorigA2 {d_o:+.4f} [{lo_o:+.4f};{hi_o:+.4f}] | "
                f"DiamV {auc_dv:.4f}/{acc_dv:.2%} ({d_dv:+.4f} [{lo_dv:+.4f};{hi_dv:+.4f}]) | "
                f"Δlap {d_l:+.4f} [{lo_l:+.4f};{hi_l:+.4f}]",
                flush=True,
            )
            filas.append({
                "minuto": m, "modelo": nom, "n_V": len(v_idx),
                "AUC_A3": auc_a3, "AUC_A4": auc_a4, "AUC_A34": auc_a34,
                "AUC_V": auc_v, "Acc_V": acc_v, "LogLoss_V": ll_v, "ECE_V": ece_v,
                "dAUC_vs_LorigA2": d_o, "lo_orig": lo_o, "hi_orig": hi_o,
                "AUC_DiamV": auc_dv, "Acc_DiamV": acc_dv,
                "dDV_vs_LorigA2": d_dv, "lo_dv": lo_dv, "hi_dv": hi_dv,
                "dAUC_vs_lapBest": d_l, "lo_lap": lo_l, "hi_lap": hi_l,
            })
            saved[f"pv_{m}_{nom}"] = pv
        saved[f"v_idx_{m}"] = v_idx
        saved[f"y_v_{m}"] = yv
        log(f"Minuto {m} terminado en {time.time() - tm:.1f}s")
        pd.DataFrame(filas).to_csv(IN_DIR / "prior_esc9_10_13k.csv", index=False)

    np.savez_compressed(IN_DIR / "pred_prior_esc9_10_13k.npz", **saved)
    log(f"Guardado {IN_DIR / 'prior_esc9_10_13k.npz'} (total {time.time() - T0:.1f}s)")


if __name__ == "__main__":
    main()
