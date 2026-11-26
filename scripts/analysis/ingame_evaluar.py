"""Entrena y evalúa modelos in-game mejorados a los minutos 10, 15 y 20.

Lee `data/analysis/ingame/ingame_cache.npz` (generado por `ingame_preparar.py`)
con bajo consumo de RAM (<1.5 GB) y compara estrictamente causal (A2-A4 hacia
adelante para diseño/pesos, V Master+ para confirmación final, más chequeo ciego
Diamond-V):

1. Baselines comparables (`brainstorm_39` por equipo y `brainstorm_40` por líneas
   + sorpresa, entrenados sobre A2-A4 con el mismo L_oof y máscara `dur > 60*m`).
2. Ablación progresiva de variables:
   - `A1`: solo ampliar datos de entrenamiento (~13,8k -> ~92k Master+)
   - `A2`: + 8 stats por carril, oro residual, picos de nivel, distribución,
           trayectorias 10→15→20 y expectativas @10/@15/@20 + transición
   - `A3`: + composición del draft, crecimiento, `cbias` y cruces (ritmo×oro,
           escalado×oro, vulnerabilidades, sinergias 2v2)
   - `A4`: + rasgos de jugadores in-game y cruces jugador×estado
3. Modelos no lineales (sin stacking de logísticas) + objetivo auxiliar:
   - `LogReg_Full`, `LightGBM_Standalone`, `LightGBM_Residual`,
     `CatBoost_Residual`, `InGameNet_NoAux`, `InGameNet_MultiTask`, y
     `Ensemble_NoLin`.

Uso:
    python scripts/analysis/ingame_evaluar.py
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

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

torch.set_num_threads(4)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.models.ingame import InGameNet, MINUTES, TIMELINE_STATS, extract_minute_features  # noqa: E402

IN_DIR = REPO / "data" / "analysis" / "ingame"
CACHE_PATH = IN_DIR / "ingame_cache.npz"
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p)).astype(np.float32)


def _sigm(z: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))).astype(np.float32)


def _ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    dec = pd.qcut(p, n_bins, labels=False, duplicates="drop")
    df = pd.DataFrame({"p": p, "y": y, "d": dec}).groupby("d").agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
    return float((np.abs(df["p"] - df["y"]) * df["n"]).sum() / df["n"].sum())


def _delta_ci(y: np.ndarray, p1: np.ndarray, p0: np.ndarray, boot_idx: np.ndarray) -> tuple[float, float, float]:
    d0 = roc_auc_score(y, p1) - roc_auc_score(y, p0)
    ds = np.array([roc_auc_score(y[b], p1[b]) - roc_auc_score(y[b], p0[b]) for b in boot_idx])
    return float(d0), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


def _prep_arrays(
    feats: dict,
    tr_idx: np.ndarray,
    te_idx: np.ndarray,
    L_oof: np.ndarray,
    y: np.ndarray,
) -> dict:
    """Estandariza manteniendo antisimetría exacta de `X_anti` y simetría de `X_sym`."""
    p_base = float(y[tr_idx].mean())
    L0 = math.log(p_base / (1.0 - p_base))
    L_sym = (L_oof - L0).astype(np.float32)

    xa = np.nan_to_num(feats["X_anti"], nan=0.0, posinf=0.0, neginf=0.0).copy()
    xa[:, 0] = L_sym  # reemplaza col 0 con L_sym centrado para el fold
    xs = np.nan_to_num(feats["X_sym"], nan=0.0, posinf=0.0, neginf=0.0)

    # Escala RMS (sin restar media!) preserva (-x)/rms == -(x/rms)
    rms_a = np.sqrt(np.mean(xa[tr_idx] ** 2, axis=0)) + 1e-6
    mu_s = xs[tr_idx].mean(axis=0)
    sd_s = xs[tr_idx].std(axis=0) + 1e-6

    Xa_tr = (xa[tr_idx] / rms_a).astype(np.float32)
    Xa_te = (xa[te_idx] / rms_a).astype(np.float32)
    Xs_tr = ((xs[tr_idx] - mu_s) / sd_s).astype(np.float32)
    Xs_te = ((xs[te_idx] - mu_s) / sd_s).astype(np.float32)

    # Normalización simétrica de carriles (promedio de ambos lados)
    lb = np.nan_to_num(feats["lane_b"], nan=0.0, posinf=0.0, neginf=0.0)
    lr = np.nan_to_num(feats["lane_r"], nan=0.0, posinf=0.0, neginf=0.0)
    mu_l = 0.5 * (lb[tr_idx].mean(axis=0) + lr[tr_idx].mean(axis=0))
    var_l = 0.5 * (lb[tr_idx].var(axis=0) + lr[tr_idx].var(axis=0))
    sd_l = np.sqrt(var_l) + 1e-6

    Lb_tr = ((lb[tr_idx] - mu_l) / sd_l).astype(np.float32)
    Lr_tr = ((lr[tr_idx] - mu_l) / sd_l).astype(np.float32)
    Lb_te = ((lb[te_idx] - mu_l) / sd_l).astype(np.float32)
    Lr_te = ((lr[te_idx] - mu_l) / sd_l).astype(np.float32)

    return {
        "Xa_tr": Xa_tr,
        "Xa_te": Xa_te,
        "Xs_tr": Xs_tr,
        "Xs_te": Xs_te,
        "Lb_tr": Lb_tr,
        "Lr_tr": Lr_tr,
        "Lb_te": Lb_te,
        "Lr_te": Lr_te,
    }


def _fit_logreg_antisym(Xa_tr, y_tr, Xa_te, C: float = 0.05) -> tuple[np.ndarray, np.ndarray, LogisticRegression]:
    """Logística con espejo en las covariables antisimétricas + intercepto de lado."""
    lr = LogisticRegression(C=C, max_iter=3000)
    lr.fit(Xa_tr, y_tr)
    z_tr = lr.decision_function(Xa_tr).astype(np.float32)
    z_te = lr.decision_function(Xa_te).astype(np.float32)
    return z_tr, z_te, lr


def _oof_logreg_logits(Xa_tr, y_tr, t_tr, C: float = 0.05, n_splits: int = 5) -> np.ndarray:
    """Logits fuera de muestra dentro de `tr` (OOF cronológico) para residual boosting."""
    qs = np.quantile(t_tr, np.linspace(0, 1, n_splits + 1)[1:-1])
    blk = np.searchsorted(qs, t_tr, side="right")
    z_oof = np.zeros(len(y_tr), dtype=np.float32)
    for b in range(n_splits):
        val_m = blk == b
        fit_m = ~val_m
        lr = LogisticRegression(C=C, max_iter=2000).fit(Xa_tr[fit_m], y_tr[fit_m])
        z_oof[val_m] = lr.decision_function(Xa_tr[val_m]).astype(np.float32)
    return z_oof


def _calibrar_b_side(z_anti_tr: np.ndarray, y_tr: np.ndarray) -> float:
    """Calibra únicamente el sesgo escalar de lado `b_side` manteniendo pendiente=1."""
    b = 0.0
    y_f = y_tr.astype(np.float64)
    z_f = z_anti_tr.astype(np.float64)
    for _ in range(12):
        p = 1.0 / (1.0 + np.exp(-np.clip(z_f + b, -20.0, 20.0)))
        grad = np.mean(p - y_f)
        hess = np.mean(p * (1.0 - p)) + 1e-9
        b -= grad / hess
    return float(b)


def _fit_predict_lgb(
    Xa_tr, Xs_tr, y_tr, Xa_te, Xs_te,
    init_tr: np.ndarray | None = None,
    init_te: np.ndarray | None = None,
    n_estimators: int = 350,
) -> tuple[np.ndarray, lgb.LGBMClassifier]:
    X_tr = np.c_[Xa_tr, Xs_tr]
    X_te = np.c_[Xa_te, Xs_te]
    X_tr_m = np.r_[X_tr, np.c_[-Xa_tr, Xs_tr]]
    y_tr_m = np.r_[y_tr, 1 - y_tr]
    X_tr_s = np.c_[-Xa_tr, Xs_tr]
    X_te_s = np.c_[-Xa_te, Xs_te]

    clf = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        num_leaves=31,
        learning_rate=0.03,
        max_bin=63,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.75,
        min_child_samples=150,
        reg_lambda=5.0,
        n_jobs=6,
        verbose=-1,
        random_state=42,
    )
    if init_tr is not None:
        b0 = float(np.mean(init_tr))
        init_anti_tr = init_tr - b0
        init_anti_te = init_te - b0
        init_m = np.r_[init_anti_tr, -init_anti_tr]
        clf.fit(X_tr_m, y_tr_m, init_score=init_m)
        raw_p = np.asarray(clf.predict_proba(X_te, raw_score=True)).reshape(-1)
        raw_s = np.asarray(clf.predict_proba(X_te_s, raw_score=True)).reshape(-1)
        z_anti_te = init_anti_te + 0.5 * (raw_p - raw_s)
        raw_tr_p = np.asarray(clf.predict_proba(X_tr, raw_score=True)).reshape(-1)
        raw_tr_s = np.asarray(clf.predict_proba(X_tr_s, raw_score=True)).reshape(-1)
        z_anti_tr = init_anti_tr + 0.5 * (raw_tr_p - raw_tr_s)
    else:
        clf.fit(X_tr_m, y_tr_m)
        z_p = _logit(clf.predict_proba(X_te)[:, 1])
        z_s = _logit(clf.predict_proba(X_te_s)[:, 1])
        z_anti_te = 0.5 * (z_p - z_s)
        z_tr_p = _logit(clf.predict_proba(X_tr)[:, 1])
        z_tr_s = _logit(clf.predict_proba(X_tr_s)[:, 1])
        z_anti_tr = 0.5 * (z_tr_p - z_tr_s)

    b_side = _calibrar_b_side(z_anti_tr, y_tr)
    return (b_side + z_anti_te).astype(np.float32), clf


def _fit_predict_catboost(
    Xa_tr, Xs_tr, y_tr, Xa_te, Xs_te,
    init_tr: np.ndarray,
    init_te: np.ndarray,
) -> np.ndarray:
    b0 = float(np.mean(init_tr))
    init_anti_tr = init_tr - b0
    init_anti_te = init_te - b0

    X_tr = np.c_[Xa_tr, Xs_tr]
    X_tr_m = np.r_[X_tr, np.c_[-Xa_tr, Xs_tr]]
    y_tr_m = np.r_[y_tr, 1 - y_tr]
    init_m = np.r_[init_anti_tr, -init_anti_tr]

    pool_tr = Pool(X_tr_m, y_tr_m, baseline=init_m)
    cb = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.045,
        l2_leaf_reg=12.0,
        rsm=0.75,
        border_count=64,
        thread_count=6,
        verbose=0,
        random_seed=42,
    )
    cb.fit(pool_tr)

    def pred_raw(X, base):
        return np.asarray(cb.predict(Pool(X, baseline=base), prediction_type="RawFormulaVal")).reshape(-1)

    z_te_p = pred_raw(np.c_[Xa_te, Xs_te], init_anti_te)
    z_te_s = pred_raw(np.c_[-Xa_te, Xs_te], -init_anti_te)
    z_anti_te = 0.5 * (z_te_p - z_te_s)

    z_tr_p = pred_raw(X_tr, init_anti_tr)
    z_tr_s = pred_raw(np.c_[-Xa_tr, Xs_tr], -init_anti_tr)
    z_anti_tr = 0.5 * (z_tr_p - z_tr_s)

    b_side = _calibrar_b_side(z_anti_tr, y_tr)
    return (b_side + z_anti_te).astype(np.float32)


def _construir_aux_targets(
    minute: int,
    idx: np.ndarray,
    tl_raw: np.ndarray,
    dur_sec: np.ndarray,
    end_gold_diff: np.ndarray,
    end_kills_diff: np.ndarray,
    stats_norm: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Construye objetivos auxiliares futuros y sus máscaras de validez (§3.F.4).

    Solo se usa durante entrenamiento (nunca en inferencia).
    """
    stat_i = {s: i for i, s in enumerate(TIMELINE_STATS)}
    d_g15 = (tl_raw[idx, 0, :, 1, stat_i["gold"]] - tl_raw[idx, 1, :, 1, stat_i["gold"]]).sum(axis=1)
    d_x15 = (tl_raw[idx, 0, :, 1, stat_i["xp"]] - tl_raw[idx, 1, :, 1, stat_i["xp"]]).sum(axis=1)
    d_g20 = (tl_raw[idx, 0, :, 2, stat_i["gold"]] - tl_raw[idx, 1, :, 2, stat_i["gold"]]).sum(axis=1)
    d_x20 = (tl_raw[idx, 0, :, 2, stat_i["xp"]] - tl_raw[idx, 1, :, 2, stat_i["xp"]]).sum(axis=1)
    d_d20 = (tl_raw[idx, 0, :, 2, stat_i["damage"]] - tl_raw[idx, 1, :, 2, stat_i["damage"]]).sum(axis=1)
    g_end = end_gold_diff[idx]
    k_end = end_kills_diff[idx]
    dur_s = dur_sec[idx]

    if minute == 10:
        raw_anti = np.column_stack([d_g15, d_x15, d_g20, d_x20, g_end])
        mask_anti = np.column_stack([
            (dur_s > 900.0) & np.isfinite(d_g15),
            (dur_s > 900.0) & np.isfinite(d_x15),
            (dur_s > 1200.0) & np.isfinite(d_g20),
            (dur_s > 1200.0) & np.isfinite(d_x20),
            np.isfinite(g_end),
        ])
        early20 = (dur_s <= 1200.0).astype(np.float32)
        raw_sym = early20[:, None]
        mask_sym = np.ones_like(raw_sym, dtype=bool)
    elif minute == 15:
        raw_anti = np.column_stack([d_g20, d_x20, d_d20, g_end])
        mask_anti = np.column_stack([
            (dur_s > 1200.0) & np.isfinite(d_g20),
            (dur_s > 1200.0) & np.isfinite(d_x20),
            (dur_s > 1200.0) & np.isfinite(d_d20),
            np.isfinite(g_end),
        ])
        early20 = (dur_s <= 1200.0).astype(np.float32)
        raw_sym = early20[:, None]
        mask_sym = np.ones_like(raw_sym, dtype=bool)
    else:
        raw_anti = np.column_stack([g_end, k_end])
        mask_anti = np.isfinite(raw_anti)
        rem_dur = np.log1p(np.maximum(0.0, dur_s / 60.0 - 20.0)).astype(np.float32)
        raw_sym = rem_dur[:, None]
        mask_sym = np.ones_like(raw_sym, dtype=bool)

    if stats_norm is None:
        rms_a = np.array([
            np.sqrt(np.mean(raw_anti[mask_anti[:, j], j] ** 2)) + 1e-6 for j in range(raw_anti.shape[1])
        ], dtype=np.float32)
        stats_norm = {"rms_a": rms_a}
        if minute == 20:
            stats_norm["mu_s"] = float(raw_sym.mean())
            stats_norm["sd_s"] = float(raw_sym.std() + 1e-6)
    y_anti = np.nan_to_num(raw_anti, nan=0.0) / stats_norm["rms_a"]
    if minute == 20:
        y_sym = ((raw_sym - stats_norm["mu_s"]) / stats_norm["sd_s"]).astype(np.float32)
    else:
        y_sym = raw_sym.astype(np.float32)
    return y_anti.astype(np.float32), mask_anti, y_sym, mask_sym, stats_norm


def _fit_predict_ingamenet(
    minute: int,
    arrs: dict,
    y_tr: np.ndarray,
    aux_tr: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
    use_aux: bool,
    seed: int = 0,
) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)

    Lb_tr, Lr_tr = arrs["Lb_tr"], arrs["Lr_tr"]
    Xa_tr, Xs_tr = arrs["Xa_tr"], arrs["Xs_tr"]
    Lb_te, Lr_te = arrs["Lb_te"], arrs["Lr_te"]
    Xa_te, Xs_te = arrs["Xa_te"], arrs["Xs_te"]

    n = len(y_tr)
    cut = int(n * 0.88)  # último 12 % cronológico como parada temprana
    tr_s, va_s = np.arange(cut), np.arange(cut, n)

    d_aux_a = aux_tr[0].shape[1] if (use_aux and aux_tr is not None) else 0
    d_aux_s = aux_tr[2].shape[1] if (use_aux and aux_tr is not None) else 0

    net = InGameNet(
        d_lane=Lb_tr.shape[-1],
        d_anti=Xa_tr.shape[1],
        d_sym=Xs_tr.shape[1],
        d_aux_anti=d_aux_a,
        d_aux_sym=d_aux_s,
        lane_hidden=24,
        trunk_hidden=96,
        dropout=0.25,
    )
    opt = torch.optim.AdamW(net.parameters(), lr=2.0e-3, weight_decay=5e-3)
    epochs = 16
    bs = 2048
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    t_lb = torch.tensor(Lb_tr[tr_s])
    t_lr = torch.tensor(Lr_tr[tr_s])
    t_xa = torch.tensor(Xa_tr[tr_s])
    t_xs = torch.tensor(Xs_tr[tr_s])
    t_y = torch.tensor(y_tr[tr_s], dtype=torch.float32)

    if use_aux and aux_tr is not None:
        t_ya = torch.tensor(aux_tr[0][tr_s])
        t_ma = torch.tensor(aux_tr[1][tr_s], dtype=torch.float32)
        t_ys = torch.tensor(aux_tr[2][tr_s])
        t_ms = torch.tensor(aux_tr[3][tr_s], dtype=torch.float32)

    v_lb = torch.tensor(Lb_tr[va_s])
    v_lr = torch.tensor(Lr_tr[va_s])
    v_xa = torch.tensor(Xa_tr[va_s])
    v_xs = torch.tensor(Xs_tr[va_s])
    v_y = y_tr[va_s]

    best_loss = float("inf")
    best_state = None
    rng = np.random.default_rng(seed)

    for ep in range(epochs):
        net.train()
        perm = rng.permutation(len(tr_s))
        for b in range(0, len(perm), bs):
            ii = perm[b : b + bs]
            # Data augmentation simétrica dentro del batch (50% espejado para regularizar)
            logit, out_a, out_s = net(t_lb[ii], t_lr[ii], t_xa[ii], t_xs[ii])
            loss = F.binary_cross_entropy_with_logits(logit, t_y[ii])
            if use_aux and aux_tr is not None:
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
            vl = net(v_lb, v_lr, v_xa, v_xs)[0].numpy()
            vp = _sigm(vl)
            vloss = log_loss(v_y, vp)
        if vloss < best_loss:
            best_loss = vloss
            best_state = {k: v.clone() for k, v in net.state_dict().items()}

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        preds = []
        for b in range(0, len(Xa_te), 4096):
            l_b = net(
                torch.tensor(Lb_te[b : b + 4096]),
                torch.tensor(Lr_te[b : b + 4096]),
                torch.tensor(Xa_te[b : b + 4096]),
                torch.tensor(Xs_te[b : b + 4096]),
            )[0].numpy()
            preds.append(l_b)
    return np.concatenate(preds).astype(np.float32)


def main() -> None:
    if not CACHE_PATH.exists():
        sys.exit(f"Falta {CACHE_PATH}. Correr primero: python historial/ensamble_en_vivo/ingame_preparar.py")

    log(f"Cargando {CACHE_PATH}...")
    C = np.load(CACHE_PATH, allow_pickle=True)
    NM = len(C["y"])
    y = C["y"].astype(np.int64)
    t = C["t"].astype(np.int64)
    chunk = C["chunk"].astype(np.int32)
    master = C["master"].astype(bool)
    val = C["val"].astype(bool)
    diam_v = C["diam_v"].astype(bool)
    dur_sec = C["dur_sec"].astype(np.float32)
    L_oof = C["L_oof"].astype(np.float32)
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

    stat_i = {s: i for i, s in enumerate(TIMELINE_STATS)}
    resultados = []
    segmentos = []
    importancias = []
    saved_preds = {}

    for mi, m in enumerate(MINUTES):
        log(f"\n==================== MINUTO {m} ====================")
        # Validez estricta: partida viva > 60*m y snapshots <= m finitos + L_oof finito
        ok_tl = np.isfinite(tl_raw[:, :, :, : mi + 1, :]).all(axis=(1, 2, 3, 4))
        ok_m = (dur_sec > 60.0 * m) & ok_tl & np.isfinite(L_oof)

        v_idx = np.flatnonzero(val & ok_m)
        dv_idx = np.flatnonzero(diam_v & ok_m)
        a24_folds = [(b, np.flatnonzero((chunk == b) & master & ok_m)) for b in (4, 5, 6)]
        tr_full = np.flatnonzero(np.isin(chunk, [1, 2, 3, 4, 5, 6]) & master & ok_m)
        tr_13k = np.flatnonzero(np.isin(chunk, [4, 5, 6]) & master & ok_m)
        tr_full_all = np.flatnonzero(np.isin(chunk, [1, 2, 3, 4, 5, 6]) & ok_m)  # Master + Diamond

        log(f"Filas válidas m={m}: tr_full={len(tr_full):,} (A2-A4={len(tr_13k):,}), V={len(v_idx):,}, DiamV={len(dv_idx):,}")

        # ---- 1. Baselines comparables (13.8k y ~92k) ----
        f_base_eq = extract_minute_features(
            m, L_oof, tl_raw, gold_resid,
            include_all_stats=False, include_traj=False, include_expected=False,
            include_comp_cross=False, include_player=False,
        )
        # Base_Lineas_Sorp replicando brainstorm_40 (+ gold_15 esperado, o gold_20 al min 20)
        g_l = (tl_raw[:, 0, :, mi, stat_i["gold"]] - tl_raw[:, 1, :, mi, stat_i["gold"]])
        exp_ref = exp_lane["gold_15"] if m <= 15 else exp_lane["gold_20"]
        f_base_ls = dict(f_base_eq)
        f_base_ls["X_anti"] = np.c_[f_base_eq["X_anti"], g_l - exp_ref].astype(np.float32)

        def eval_logreg_config(feats_dict: dict, use_13k: bool = False, C: float = 0.05, use_all_tiers: bool = False):
            # Forward-chaining en A2..A4
            a24_p = np.full(NM, np.nan, dtype=np.float32)
            for b, te in a24_folds:
                tr_b = np.flatnonzero(
                    (np.isin(chunk, [4, 5, 6]) if use_13k else np.isin(chunk, [1, 2, 3, 4, 5, 6]))
                    & (chunk < b)
                    & (True if use_all_tiers else master)
                    & ok_m
                )
                if len(tr_b) < 200:
                    continue
                ar = _prep_arrays(feats_dict, tr_b, te, L_oof, y)
                _, z_te, _ = _fit_logreg_antisym(ar["Xa_tr"], y[tr_b], ar["Xa_te"], C=C)
                a24_p[te] = _sigm(z_te)
            # Final en V Master+ y Diamond-V
            tr_f = tr_13k if use_13k else (tr_full_all if use_all_tiers else tr_full)
            te_vd = np.concatenate([v_idx, dv_idx])
            ar_v = _prep_arrays(feats_dict, tr_f, te_vd, L_oof, y)
            _, z_vd, _ = _fit_logreg_antisym(ar_v["Xa_tr"], y[tr_f], ar_v["Xa_te"], C=C)
            p_vd = _sigm(z_vd)
            return a24_p, p_vd[: len(v_idx)], p_vd[len(v_idx) :]

        preds_v: dict[str, np.ndarray] = {}
        preds_a24: dict[str, np.ndarray] = {}
        preds_dv: dict[str, np.ndarray] = {}

        for nom, fdict, u13, uall in [
            ("0_Base_Equipo_13k", f_base_eq, True, False),
            ("1_Base_Lineas_Sorp_13k", f_base_ls, True, False),
            ("2_Abl_A1_MasDatos_92k", f_base_ls, False, False),
        ]:
            pa, pv, pdv = eval_logreg_config(fdict, use_13k=u13, C=0.1, use_all_tiers=uall)
            preds_a24[nom], preds_v[nom], preds_dv[nom] = pa, pv, pdv

        # ---- 2. Ablación progresiva de features ----
        f_a2 = extract_minute_features(
            m, L_oof, tl_raw, gold_resid, exp_lane=exp_lane, exp_trans=exp_trans,
            include_all_stats=True, include_traj=True, include_expected=True,
            include_comp_cross=False, include_player=False,
        )
        f_a3 = extract_minute_features(
            m, L_oof, tl_raw, gold_resid, exp_lane=exp_lane, exp_trans=exp_trans,
            comp=comp, tempo=tempo,
            include_all_stats=True, include_traj=True, include_expected=True,
            include_comp_cross=True, include_player=False,
        )
        f_a4 = extract_minute_features(
            m, L_oof, tl_raw, gold_resid, exp_lane=exp_lane, exp_trans=exp_trans,
            comp=comp, tempo=tempo, player_pre=player_pre,
            include_all_stats=True, include_traj=True, include_expected=True,
            include_comp_cross=True, include_player=True,
        )

        # Elegir C de LogReg usando SOLO A2..A4 (nunca V!)
        a24_all_idx = np.concatenate([te for _, te in a24_folds])
        best_C, best_ll = 0.05, float("inf")
        for cand_C in (0.01, 0.03, 0.1, 0.3):
            pa_c, _, _ = eval_logreg_config(f_a4, use_13k=False, C=cand_C)
            ll_c = log_loss(y[a24_all_idx], pa_c[a24_all_idx])
            if ll_c < best_ll:
                best_ll, best_C = ll_c, cand_C
        log(f"  C elegido en A2-A4 para LogReg_Full: {best_C} (LogLoss A2-A4={best_ll:.5f})")

        for nom, fdict in [
            ("3_Abl_A2_Stats_Traj_Exp", f_a2),
            ("4_Abl_A3_Compo_Cruces", f_a3),
            ("5_Abl_A4_Jugadores_Full", f_a4),
        ]:
            pa, pv, pdv = eval_logreg_config(fdict, use_13k=False, C=best_C)
            preds_a24[nom], preds_v[nom], preds_dv[nom] = pa, pv, pdv

        # Chequeo: ¿ayuda entrenar Stage-2 con Master+Diamond vs solo Master+?
        pa_md, pv_md, pdv_md = eval_logreg_config(f_a4, use_13k=False, C=best_C, use_all_tiers=True)
        preds_a24["5b_Full_MasterDiam"], preds_v["5b_Full_MasterDiam"], preds_dv["5b_Full_MasterDiam"] = pa_md, pv_md, pdv_md

        # ---- 3. Modelos no lineales + Red con objetivo auxiliar ----
        # Calculamos en A3+A4 (b=5,6 para OOF de ensemble) y sobre V + Diamond-V
        # Nota: también evaluamos en A2..A4 completo para selección de pesos del ensamble sin tocar V!
        models_nl = ("6_LGBM_Standalone", "7_LGBM_Residual", "8_CatBoost_Residual", "9_InGameNet_NoAux", "10_InGameNet_MultiTask")
        for mn in models_nl:
            preds_a24[mn] = np.full(NM, np.nan, dtype=np.float32)

        for b, te in a24_folds:
            tr_b = np.flatnonzero(np.isin(chunk, [1, 2, 3, 4, 5, 6]) & (chunk < b) & master & ok_m)
            ar = _prep_arrays(f_a4, tr_b, te, L_oof, y)
            z_tr_lr, z_te_lr, _ = _fit_logreg_antisym(ar["Xa_tr"], y[tr_b], ar["Xa_te"], C=best_C)
            z_oof_lr = _oof_logreg_logits(ar["Xa_tr"], y[tr_b], t[tr_b], C=best_C, n_splits=4)

            z_lgb_st, _ = _fit_predict_lgb(ar["Xa_tr"], ar["Xs_tr"], y[tr_b], ar["Xa_te"], ar["Xs_te"], n_estimators=260)
            z_lgb_rs, _ = _fit_predict_lgb(ar["Xa_tr"], ar["Xs_tr"], y[tr_b], ar["Xa_te"], ar["Xs_te"],
                                           init_tr=z_oof_lr, init_te=z_te_lr, n_estimators=260)
            z_cb_rs = _fit_predict_catboost(ar["Xa_tr"], ar["Xs_tr"], y[tr_b], ar["Xa_te"], ar["Xs_te"],
                                            init_tr=z_oof_lr, init_te=z_te_lr)

            ya, ma, ys, ms_, norm_st = _construir_aux_targets(m, tr_b, tl_raw, dur_sec, end_gold_diff, end_kills_diff)
            z_nn_no = _fit_predict_ingamenet(m, ar, y[tr_b], None, use_aux=False, seed=10 + b)
            z_nn_mt = _fit_predict_ingamenet(m, ar, y[tr_b], (ya, ma, ys, ms_), use_aux=True, seed=20 + b)

            preds_a24["6_LGBM_Standalone"][te] = _sigm(z_lgb_st)
            preds_a24["7_LGBM_Residual"][te] = _sigm(z_lgb_rs)
            preds_a24["8_CatBoost_Residual"][te] = _sigm(z_cb_rs)
            preds_a24["9_InGameNet_NoAux"][te] = _sigm(z_nn_no)
            preds_a24["10_InGameNet_MultiTask"][te] = _sigm(z_nn_mt)

        # Entrenamiento final sobre todo tr_full (~92k Master+) -> V Master+ y Diamond-V
        te_vd = np.concatenate([v_idx, dv_idx])
        ar_f = _prep_arrays(f_a4, tr_full, te_vd, L_oof, y)
        _, z_vd_lr, _ = _fit_logreg_antisym(ar_f["Xa_tr"], y[tr_full], ar_f["Xa_te"], C=best_C)
        z_oof_f = _oof_logreg_logits(ar_f["Xa_tr"], y[tr_full], t[tr_full], C=best_C, n_splits=5)

        z_vd_lgb_st, _ = _fit_predict_lgb(ar_f["Xa_tr"], ar_f["Xs_tr"], y[tr_full], ar_f["Xa_te"], ar_f["Xs_te"], n_estimators=320)
        z_vd_lgb_rs, lgb_clf = _fit_predict_lgb(ar_f["Xa_tr"], ar_f["Xs_tr"], y[tr_full], ar_f["Xa_te"], ar_f["Xs_te"],
                                                init_tr=z_oof_f, init_te=z_vd_lr, n_estimators=320)
        z_vd_cb_rs = _fit_predict_catboost(ar_f["Xa_tr"], ar_f["Xs_tr"], y[tr_full], ar_f["Xa_te"], ar_f["Xs_te"],
                                           init_tr=z_oof_f, init_te=z_vd_lr)

        ya_f, ma_f, ys_f, ms_f, _ = _construir_aux_targets(m, tr_full, tl_raw, dur_sec, end_gold_diff, end_kills_diff)
        z_vd_nn_no = _fit_predict_ingamenet(m, ar_f, y[tr_full], None, use_aux=False, seed=100)
        # 2 semillas para MultiTask InGameNet
        z_vd_nn_mt = 0.5 * (
            _fit_predict_ingamenet(m, ar_f, y[tr_full], (ya_f, ma_f, ys_f, ms_f), use_aux=True, seed=200)
            + _fit_predict_ingamenet(m, ar_f, y[tr_full], (ya_f, ma_f, ys_f, ms_f), use_aux=True, seed=201)
        )

        nv = len(v_idx)
        for nom, z_arr in [
            ("6_LGBM_Standalone", z_vd_lgb_st),
            ("7_LGBM_Residual", z_vd_lgb_rs),
            ("8_CatBoost_Residual", z_vd_cb_rs),
            ("9_InGameNet_NoAux", z_vd_nn_no),
            ("10_InGameNet_MultiTask", z_vd_nn_mt),
        ]:
            p_arr = _sigm(z_arr)
            preds_v[nom] = p_arr[:nv]
            preds_dv[nom] = p_arr[nv:]

        # Top importancias de LightGBM Residual
        feat_names = f_a4["names_anti"] + f_a4["names_sym"]
        imp = pd.Series(lgb_clf.booster_.feature_importance(importance_type="gain"), index=feat_names)
        imp = imp.sort_values(ascending=False).head(20)
        for fn, val_i in imp.items():
            importancias.append({"minuto": m, "feature": fn, "gain": float(val_i)})

        # Ensamble diverso (pesos ajustados SOLO en A2-A4 OOF, nunca en V!)
        ens_keys = ["5_Abl_A4_Jugadores_Full", "7_LGBM_Residual", "8_CatBoost_Residual", "10_InGameNet_MultiTask"]
        Z_a24 = np.column_stack([_logit(preds_a24[k][a24_all_idx]) for k in ens_keys])
        meta_lr = LogisticRegression(C=1.0, max_iter=1000).fit(Z_a24, y[a24_all_idx])
        log(f"  Pesos Ensemble (A2-A4 OOF): " + ", ".join(f"{k.split('_', 1)[1]}={w:.3f}" for k, w in zip(ens_keys, meta_lr.coef_[0])))

        Z_v = np.column_stack([_logit(preds_v[k]) for k in ens_keys])
        Z_dv = np.column_stack([_logit(preds_dv[k]) for k in ens_keys])
        preds_a24["11_Ensemble_NoLin"] = np.full(NM, np.nan, dtype=np.float32)
        preds_a24["11_Ensemble_NoLin"][a24_all_idx] = meta_lr.predict_proba(Z_a24)[:, 1].astype(np.float32)
        preds_v["11_Ensemble_NoLin"] = meta_lr.predict_proba(Z_v)[:, 1].astype(np.float32)
        preds_dv["11_Ensemble_NoLin"] = meta_lr.predict_proba(Z_dv)[:, 1].astype(np.float32)

        # ---- 4. Tablas comparativas con IC por bootstrap pareado ----
        rng = np.random.default_rng(100 + m)
        boot_v = rng.integers(0, len(v_idx), size=(500, len(v_idx)))
        yv = y[v_idx]
        ydv = y[dv_idx]
        ref_eq = preds_v["0_Base_Equipo_13k"]
        ref_ls = preds_v["1_Base_Lineas_Sorp_13k"]

        a34_all_idx = np.concatenate([te for b, te in a24_folds if b in (5, 6)])
        log(f"  Resultados en V Master+ (n={len(v_idx):,}) y A3-A4 OOF (n={len(a34_all_idx):,}):")
        for nom in preds_v:
            pv = preds_v[nom]
            pa_arr = preds_a24[nom]
            ok_a24 = np.isfinite(pa_arr[a24_all_idx])
            auc_a24 = roc_auc_score(y[a24_all_idx[ok_a24]], pa_arr[a24_all_idx[ok_a24]]) if ok_a24.any() else float("nan")
            auc_a34 = roc_auc_score(y[a34_all_idx], pa_arr[a34_all_idx])
            pdv_ = preds_dv[nom]
            auc_v = roc_auc_score(yv, pv)
            acc_v = float(((pv > 0.5) == yv).mean())
            ll_v = log_loss(yv, pv)
            ece_v = _ece(yv, pv)
            auc_dv = roc_auc_score(ydv, pdv_)
            d_eq, lo_eq, hi_eq = _delta_ci(yv, pv, ref_eq, boot_v)
            d_ls, lo_ls, hi_ls = _delta_ci(yv, pv, ref_ls, boot_v)
            print(
                f"    {nom:26s} | A3-4 {auc_a34:.4f} (A2-4 {auc_a24:.4f}) | V {auc_v:.4f} / {acc_v:.2%} LL {ll_v:.4f} ECE {ece_v:.4f} "
                f"| Δeq {d_eq:+.4f} [{lo_eq:+.4f};{hi_eq:+.4f}] | Δls {d_ls:+.4f} [{lo_ls:+.4f};{hi_ls:+.4f}] | DiamV {auc_dv:.4f}",
                flush=True,
            )
            resultados.append({
                "minuto": m,
                "modelo": nom,
                "n_V": len(v_idx),
                "AUC_A24": auc_a24,
                "AUC_A34": auc_a34,
                "AUC_V": auc_v,
                "Acc_V": acc_v,
                "LogLoss_V": ll_v,
                "ECE_V": ece_v,
                "dAUC_vs_Equipo": d_eq,
                "dAUC_eq_lo": lo_eq,
                "dAUC_eq_hi": hi_eq,
                "dAUC_vs_LineasSorp": d_ls,
                "dAUC_ls_lo": lo_ls,
                "dAUC_ls_hi": hi_ls,
                "AUC_DiamV": auc_dv,
            })

        # Comparaciones específicas clave: +Jugadores y +MultiTask
        d_jug, lo_j, hi_j = _delta_ci(yv, preds_v["5_Abl_A4_Jugadores_Full"], preds_v["4_Abl_A3_Compo_Cruces"], boot_v)
        d_aux, lo_a, hi_a = _delta_ci(yv, preds_v["10_InGameNet_MultiTask"], preds_v["9_InGameNet_NoAux"], boot_v)
        log(f"  [Test directo] +Jugadores sobre Compo+Cruces: ΔAUC={d_jug:+.4f} [{lo_j:+.4f};{hi_j:+.4f}]")
        log(f"  [Test directo] InGameNet MultiTask vs NoAux:  ΔAUC={d_aux:+.4f} [{lo_a:+.4f};{hi_a:+.4f}]")

        # Segmentos y cobertura del mejor modelo (Ensemble_NoLin vs Base_Lineas_Sorp_13k)
        p_best = preds_v["11_Ensemble_NoLin"]
        g_tot_v = (tl_raw[v_idx, 0, :, mi, stat_i["gold"]] - tl_raw[v_idx, 1, :, mi, stat_i["gold"]]).sum(axis=1)
        abs_g = np.abs(g_tot_v)
        conf_pre = np.abs(L_oof[v_idx] - np.median(L_oof[v_idx]))
        q20, q80 = np.quantile(conf_pre, [0.2, 0.8])
        segs = {
            "Oro parejo (<1.5k)": abs_g < 1.5,
            "Oro medio (1.5k-4k)": (abs_g >= 1.5) & (abs_g < 4.0),
            "Paliza (>=4k)": abs_g >= 4.0,
            "Pre-juego difícil (20%)": conf_pre <= q20,
            "Pre-juego fácil (20%)": conf_pre >= q80,
        }
        for sname, smask in segs.items():
            segmentos.append({
                "minuto": m,
                "segmento": sname,
                "n": int(smask.sum()),
                "AUC_BaseLineas": roc_auc_score(yv[smask], ref_ls[smask]),
                "AUC_Ensemble": roc_auc_score(yv[smask], p_best[smask]),
                "Acc_BaseLineas": float(((ref_ls[smask] > 0.5) == yv[smask]).mean()),
                "Acc_Ensemble": float(((p_best[smask] > 0.5) == yv[smask]).mean()),
            })
        ord_c = np.argsort(-np.abs(p_best - 0.5))
        ok_b = (p_best > 0.5) == yv
        log("  Cobertura Ensemble_NoLin: " + "  ".join(f"{c:.0%}→{ok_b[ord_c[:int(c * len(yv))]].mean():.2%}" for c in (0.05, 0.10, 0.20, 0.50)))

        saved_preds[f"v_idx_{m}"] = v_idx
        saved_preds[f"y_v_{m}"] = yv
        for nom, pv in preds_v.items():
            saved_preds[f"p_{m}_{nom}"] = pv

        # Guardado incremental por minuto + liberación de RAM
        pd.DataFrame(resultados).to_csv(IN_DIR / "resultados_ingame.csv", index=False)
        pd.DataFrame(segmentos).to_csv(IN_DIR / "segmentos_ingame.csv", index=False)
        pd.DataFrame(importancias).to_csv(IN_DIR / "importancias_ingame.csv", index=False)
        del f_base_eq, f_base_ls, f_a2, f_a3, f_a4, ar_f
        gc.collect()

    np.savez_compressed(IN_DIR / "pred_ingame.npz", **saved_preds)
    log("Resultados guardados en data/analysis/ingame/.")


if __name__ == "__main__":
    main()
