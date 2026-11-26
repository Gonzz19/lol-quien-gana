"""Candidato a Escalón 7 (`g_22`): Redes GPU (`RedNeuralACWideDeep` 6 semillas + `RedAttnWideDeep` 5 semillas,
destiladas OOF `@10/@15` dentro de `tr`) + `CatBoost GPU Residual v2` (`baseline = logit Escalón 6 + L2 dif`
forward-chaining causal dentro de `tr`), alimentados con tensores por jugador completos (`189` canales pre-t1
+ `12` canales post-t1 Foto 1/mapa) y el vector completo Escalón 6 (`1.486` cols).

Mantiene íntegras las fórmulas ganadoras de `g_08` (Escalón 5, V +0,0022) y `g_13` (`RedAttnWideDeep` +
`CatBoost GPU Residual v2`), corrigiendo normalización separada pre/post, parada temprana sobre `post`,
regularización fuerte en rama lineal (`weight_decay=1e-1`) y compacidad de features en CatBoost.

Evalúa estrictamente causal (`solo Master+` en entrenamiento):
  1) 3 sub-bloques A2..A4 (`Δ bloq` + IC pareado)
  2) `V Master+` (`16.428` partidas): AUC, IC, acierto, LogLoss, ECE, cobertura 5/10/20/30/50 %
  3) Segmentos difíciles de `brainstorm_12` (`sin estrenos`, `veteranos >= 50`, `18-24h`, `madrugada`)
  4) Chequeo ciego `Diamond-V` (`10.506` partidas)
  5) Guarda predicciones en `data/analysis/brainstorm/pred_g22_esc7.npz`.
"""
import argparse
import gc
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from catboost import CatBoostClassifier, Pool
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.brainstorm_base import (  # noqa: E402
    ajustar_predecir,
    cargar,
    comparar,
    evaluar,
    extras_referencia,
    mostrar,
)
from historial.escalon9.brainstorm_g_01_foto1_wl import construir_extras  # noqa: E402
from historial.ensamble_en_vivo.brainstorm_g_05_gpu_master import cargar_foto1_jugador  # noqa: E402
from historial.escalon9.brainstorm_g_06_comp_poisson import construir_composicion  # noqa: E402
from historial.ensamble_en_vivo.brainstorm_g_08_destilacion_gpu import (  # noqa: E402
    RedNeuralACWideDeep,
    construir_priv,
    probs_maestro_oof,
)
from historial.ensamble_en_vivo.brainstorm_g_13_attn_gpu_catboost import RedAttnWideDeep  # noqa: E402

BR_DIR = REPO / "data" / "analysis" / "brainstorm"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _logit(p):
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _sigm(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))


def _ece(y, p, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y)
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i + 1] if i < n_bins - 1 else p <= bins[i + 1])
        if m.any():
            ece += (m.sum() / n) * abs(y[m].mean() - p[m].mean())
    return float(ece)


def preparar_entradas_redes(D: dict, REF_tot: np.ndarray, extra_opt: np.ndarray | None):
    """Construye tensores causales por jugador (189 pre + 12 post = 201 cols) y skip Wide pre/post."""
    NM = D["NM"]
    F1 = cargar_foto1_jugador(D)                 # (NM, 2, 5, 4) post-t1
    COMP, J_CAMP = construir_composicion(D)      # (NM, 25), (NM, 2, 5, 14)

    A_mmr = np.load(BR_DIR / "mmr_historico.npz")["A"].astype(np.float32)
    with np.load(BR_DIR / "margen_grupos_escala_sin_v.npz") as z:
        MC = z["MC"].astype(np.float32)
        ML = z["ML"].astype(np.float32)
    with np.load(BR_DIR / "mc_ajuste.npz") as z:
        mc5 = z["mc5"].astype(np.float32)
        mc20 = z["mc20"].astype(np.float32)
        mcl = z["mcl"].astype(np.float32)
    H = np.load(BR_DIR / "conoce_matchup.npz")["H"].astype(np.float32)

    post_p = np.broadcast_to(D["post"][:, None, None, None], (NM, 2, 5, 1)).astype(np.float32)
    # Últimas 12 cols = [D["J"](7), F1(4), post_flag(1)], idéntico a g_05/g_08/g_13
    P_jug = np.concatenate(
        [D["Xj"], A_mmr, MC, ML, mc5, mc20, mcl, H, J_CAMP, D["J"], F1, post_p],
        axis=-1,
    ).reshape(NM, 10, -1).astype(np.float32)
    np.nan_to_num(P_jug, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Descomponer REF_tot separando los 4 canales post-t1 de FOTO1_DIN (cols 153..157 en _ref)
    F1_DIN = construir_extras(D)["FOTO1_DIN"].astype(np.float32)  # (NM, 4)
    # _ref layout: mcml(150) + niv(3) + FOTO1_DIN(4) [+ escalon>=4...]
    REF_pre = np.c_[REF_tot[:, :153], REF_tot[:, 157:]].astype(np.float32)
    W_pre = np.c_[D["Xd"], REF_pre, COMP].astype(np.float32)
    if extra_opt is not None:
        W_pre = np.c_[W_pre, extra_opt].astype(np.float32)
    W_post = np.c_[F1_DIN, (D["J"][:, 0] - D["J"][:, 1]).mean(1), (F1[:, 0] - F1[:, 1]).mean(1)].astype(np.float32)
    np.nan_to_num(W_pre, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(W_post, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    ids = D["Z"]["champ"].reshape(NM, 10)
    unicos = np.unique(ids)
    cmap = {int(c): i for i, c in enumerate(unicos)}
    camp_idx = np.vectorize(cmap.__getitem__, otypes=[np.int64])(ids)

    del A_mmr, MC, ML, mc5, mc20, mcl, H, J_CAMP
    gc.collect()
    return camp_idx, P_jug, W_pre, W_post, F1, COMP, len(unicos)


def construir_X_cb(D: dict, F1: np.ndarray, COMP: np.ndarray, extra_opt: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Construye bloque tabular compacto (~310 cols) para CatBoost GPU Residual v2."""
    NM = D["NM"]
    Xj = D["Xj"]
    c = {k: i for i, k in enumerate(D["cols"])}
    n_prev = np.expm1(Xj[..., c["logn_tot"]]).reshape(NM, 10)
    est = Xj[..., c["estreno"]]
    off = Xj[..., c["offrole"]]
    wra = Xj[..., c["wr_acum"]]
    lnc = Xj[..., c["logn_champ"]]
    shl = Xj[..., c["share_lane"]]

    with np.load(BR_DIR / "margen_grupos_escala_sin_v.npz") as z:
        MC_raw, ML_raw = z["MC"].astype(np.float32), z["ML"].astype(np.float32)
    A_mmr = np.load(BR_DIR / "mmr_historico.npz")["A"].astype(np.float32)
    MATCHUPS = np.load(BR_DIR / "gemini_mpr.npz")["MATCHUPS"].astype(np.float32)
    ROL_WR = np.load(BR_DIR / "gemini_nuevas_v2.npz")["ROL_WR"].astype(np.float32)
    with np.load(BR_DIR / "gemini_g11.npz") as z:
        K_OUT = z["KALMAN_OUT"].astype(np.float32)
        K_MRG = z["KALMAN_MRG"].astype(np.float32)
        T_OBJ = z["TEMPO_OBJ"].astype(np.float32)
        E_MID = z["ECON_MID"].astype(np.float32)

    h_art = ((D["t"] // 3_600_000) - 3) % 24
    ang = 2.0 * np.pi * h_art / 24.0
    mc_d = MC_raw[:, 0] - MC_raw[:, 1]
    ml_d = ML_raw[:, 0] - ML_raw[:, 1]
    tier_m = np.nan_to_num(np.nanmean(D["Z"]["tnum"], axis=1), nan=1.0).astype(np.float32)
    gm = (np.nan_to_num(D["Z"]["tnum"], nan=0.0) >= 2).sum(axis=1).astype(np.float32)

    X_pre = np.c_[
        D["Xd"],
        (wra[:, 0] - wra[:, 1]),
        (off[:, 0] - off[:, 1]),
        (est[:, 0] - est[:, 1]),
        (lnc[:, 0] - lnc[:, 1]),
        (shl[:, 0] - shl[:, 1]),
        mc_d.mean(axis=1),
        ml_d.mean(axis=1),
        mc_d[:, :, 12],
        ml_d[:, :, 12],
        MC_raw[:, 0].min(axis=1) - MC_raw[:, 1].min(axis=1),
        (A_mmr[:, 0, :, :3] - A_mmr[:, 1, :, :3]).mean(axis=1),
        (A_mmr[:, 0, :, 1] - A_mmr[:, 1, :, 1]),
        MATCHUPS,
        COMP,
        ROL_WR,
        K_OUT,
        K_MRG,
        T_OBJ,
        E_MID,
        est.reshape(NM, 10).sum(1),
        (est.reshape(NM, 10).sum(1) == 0).astype(np.float32),
        n_prev.min(1),
        n_prev.mean(1),
        off.reshape(NM, 10).sum(1),
        np.isnan(D["Z"]["tnum"]).sum(1),
        (A_mmr[:, 0, :, 0].mean(1) + A_mmr[:, 1, :, 0].mean(1)) * 0.5,
        tier_m,
        gm,
        np.sin(ang),
        np.cos(ang),
    ].astype(np.float32)
    if extra_opt is not None:
        X_pre = np.c_[X_pre, extra_opt].astype(np.float32)
    X_post = np.c_[D["mapa"], (F1[:, 0] - F1[:, 1]).mean(axis=1)].astype(np.float32)
    return X_pre, X_post


def ajuste_l2_dif(D, tr_idx, te_idx, REF_tot, l2_dif=True):
    """Ajusta logística Escalón 6 (opcionalmente con L2 diferencial g14)."""
    if not l2_dif:
        return ajustar_predecir(D, tr_idx, te_idx, extra=REF_tot, extra_post=False, con_mapa=False)
    Xd = D["Xd"]
    mu, sd = Xd[tr_idx].mean(0), Xd[tr_idx].std(0) + 1e-6
    me, se = REF_tot[tr_idx].mean(0), REF_tot[tr_idx].std(0) + 1e-6
    gx, gd, gh, C = 0.5, 1.5, 3.0, 0.02
    Pf = sparse.hstack([
        D["DRAFT"][tr_idx] * gd,
        sparse.csr_matrix(((Xd[tr_idx] - mu) / sd) * gx),
        sparse.csr_matrix(((REF_tot[tr_idx] - me) / se) * gh),
    ]).tocsr()
    Pp = sparse.hstack([
        D["DRAFT"][te_idx] * gd,
        sparse.csr_matrix(((Xd[te_idx] - mu) / sd) * gx),
        sparse.csr_matrix(((REF_tot[te_idx] - me) / se) * gh),
    ]).tocsr()
    lr = LogisticRegression(C=C, max_iter=4000).fit(Pf, D["y"][tr_idx])
    return lr.predict_proba(Pp)[:, 1].astype(np.float32)


def entrenar_redes_fold(
    D, camp_idx, P_jug, W_pre, W_post, n_camp, p_maestro_tr, tr, te, seeds, cls_model,
    alfa=0.50, lr_wide=1e-3, wd_wide=1e-1, lr_deep=8e-4, wd_deep=2.5e-2, drop=0.30, dv_idx=None,
):
    """Entrena ensemble multi-semilla con split pre/post y parada temprana por LogLoss en post."""
    t, y, post = D["t"], D["y"], D["post"]

    # Separación estricta pre/post idéntica a g_08/g_13
    pos_mask, pre_mask = post[tr], ~post[tr]
    if pos_mask.sum() >= 1000:
        c_pre = np.quantile(t[tr][pre_mask], 0.88) if pre_mask.any() else 0
        c_pos = np.quantile(t[tr][pos_mask], 0.80)
        is_es = np.where(post[tr], t[tr] >= c_pos, t[tr] >= c_pre)
        es_eval_mask = post[tr[is_es]]
    else:
        is_es = t[tr] >= np.quantile(t[tr], 0.85)
        es_eval_mask = np.ones(int(is_es.sum()), dtype=bool)

    tr_sub, es_sub = tr[~is_es], tr[is_es]
    q_soft_sub = alfa * y[tr_sub].astype(np.float64) + (1.0 - alfa) * p_maestro_tr[~is_es]
    y_tr = torch.tensor(q_soft_sub, dtype=torch.float32, device=DEV)
    y_es_np = y[es_sub]

    d_pre_jug = P_jug.shape[2] - 12
    n_fit_pre = P_jug[tr_sub, :, :d_pre_jug].reshape(-1, d_pre_jug)
    mu_jp, sd_jp = n_fit_pre.mean(0), n_fit_pre.std(0) + 1e-6
    post_fit = tr_sub[post[tr_sub]]
    if len(post_fit) > 0:
        n_fit_po = P_jug[post_fit, :, d_pre_jug:-1].reshape(-1, 11)
        mu_jpo, sd_jpo = n_fit_po.mean(0), n_fit_po.std(0) + 1e-6
        mu_wpo, sd_wpo = W_post[post_fit].mean(0), W_post[post_fit].std(0) + 1e-6
    else:
        mu_jpo, sd_jpo = np.zeros(11, np.float32), np.ones(11, np.float32)
        mu_wpo, sd_wpo = np.zeros(W_post.shape[1], np.float32), np.ones(W_post.shape[1], np.float32)
    mu_wp, sd_wp = W_pre[tr_sub].mean(0), W_pre[tr_sub].std(0) + 1e-6

    def build_t(idx):
        p_norm = np.empty_like(P_jug[idx])
        p_norm[..., :d_pre_jug] = np.clip((P_jug[idx, :, :d_pre_jug] - mu_jp) / sd_jp, -8.0, 8.0)
        m_post = post[idx][:, None, None]
        p_norm[..., d_pre_jug:-1] = np.where(
            m_post, np.clip((P_jug[idx, :, d_pre_jug:-1] - mu_jpo) / sd_jpo, -8.0, 8.0), 0.0
        )
        p_norm[..., -1] = P_jug[idx, :, -1]
        w_pre_n = np.clip((W_pre[idx] - mu_wp) / sd_wp, -8.0, 8.0)
        w_post_n = np.where(post[idx][:, None], np.clip((W_post[idx] - mu_wpo) / sd_wpo, -8.0, 8.0), 0.0)
        return (
            torch.tensor(camp_idx[idx], dtype=torch.long, device=DEV),
            torch.tensor(p_norm, dtype=torch.float32, device=DEV),
            torch.tensor(np.hstack([w_pre_n, w_post_n]).astype(np.float32), dtype=torch.float32, device=DEV),
        )

    C_tr, N_tr, W_tr = build_t(tr_sub)
    C_es, N_es, W_es = build_t(es_sub)
    C_te, N_te, W_te = build_t(te)
    if dv_idx is not None:
        C_dv, N_dv, W_dv = build_t(dv_idx)

    logits_te = np.zeros(len(te), dtype=np.float64)
    logits_dv = np.zeros(len(dv_idx), dtype=np.float64) if dv_idx is not None else None
    n_rows = len(tr_sub)

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if cls_model is RedNeuralACWideDeep:
            net = RedNeuralACWideDeep(n_camp, P_jug.shape[2], W_tr.shape[1], h_dim=96, drop=drop, k_nac=12).to(DEV)
        else:
            net = RedAttnWideDeep(n_camp, P_jug.shape[2], W_tr.shape[1], d_mod=96, n_heads=4, drop=drop).to(DEV)

        opt = torch.optim.AdamW([
            {"params": list(net.camp_lin.parameters()) + list(net.wide.parameters()), "weight_decay": wd_wide, "lr": lr_wide},
            {"params": [p for n, p in net.named_parameters() if not n.startswith(("camp_lin.", "wide."))], "weight_decay": wd_deep, "lr": lr_deep},
        ])
        best_ll, best_sd, pat = 1e9, None, 0
        for _ in range(55):
            net.train()
            perm = torch.randperm(n_rows, device=DEV)
            for b in perm.split(512):
                opt.zero_grad(set_to_none=True)
                out = net(C_tr[b], N_tr[b], W_tr[b])
                loss = F.binary_cross_entropy_with_logits(out, y_tr[b])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
                opt.step()
            net.eval()
            with torch.no_grad():
                p_es = []
                for i in range(0, len(es_sub), 4096):
                    p_es.append(torch.sigmoid(net(C_es[i:i + 4096], N_es[i:i + 4096], W_es[i:i + 4096])).cpu().numpy())
                ll = log_loss(y_es_np[es_eval_mask], np.clip(np.concatenate(p_es)[es_eval_mask], 1e-6, 1.0 - 1e-6))
            if ll < best_ll - 1e-5:
                best_ll = ll
                best_sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                pat = 0
            else:
                pat += 1
                if pat >= 6:
                    break

        net.load_state_dict({k: v.to(DEV) for k, v in best_sd.items()})
        net.eval()
        with torch.no_grad():
            z_te = []
            for i in range(0, len(te), 4096):
                z_te.append(net(C_te[i:i + 4096], N_te[i:i + 4096], W_te[i:i + 4096]).cpu().numpy())
            logits_te += np.concatenate(z_te)
            if dv_idx is not None:
                z_dv = []
                for i in range(0, len(dv_idx), 4096):
                    z_dv.append(net(C_dv[i:i + 4096], N_dv[i:i + 4096], W_dv[i:i + 4096]).cpu().numpy())
                logits_dv += np.concatenate(z_dv)
        del net, opt, best_sd
        torch.cuda.empty_cache()

    del C_tr, N_tr, W_tr, C_es, N_es, W_es, C_te, N_te, W_te
    if dv_idx is not None:
        del C_dv, N_dv, W_dv
    torch.cuda.empty_cache()

    logits_te /= len(seeds)
    if dv_idx is not None:
        logits_dv /= len(seeds)
        return logits_te.astype(np.float32), logits_dv.astype(np.float32)
    return logits_te.astype(np.float32)


def catboost_residual_v2_fold(D, X_pre, X_post, REF_tot, tr, te, dv_idx=None, l2_dif=True):
    """CatBoost GPU Residual v2 con baseline forward-chaining causal Escalón 6+ dentro de `tr`."""
    t, y, post = D["t"], D["y"], D["post"]
    qs = np.quantile(t[tr], np.linspace(0, 1, 7)[1:-1])
    ch = np.searchsorted(qs, t[tr], side="right")
    ref_tr = np.zeros(len(tr), dtype=np.float32)
    draft_tr = np.zeros(len(tr), dtype=np.float32)

    for b in range(1, 6):
        itr, ite = tr[ch < b], tr[ch == b]
        p_ref = ajuste_l2_dif(D, itr, ite, REF_tot, l2_dif=l2_dif)
        ref_tr[ch == b] = _logit(p_ref)
        lr_d = LogisticRegression(C=0.05, max_iter=2000).fit(D["DRAFT"][itr], y[itr])
        draft_tr[ch == b] = _logit(lr_d.predict_proba(D["DRAFT"][ite])[:, 1])

    p_ref_te = ajuste_l2_dif(D, tr, te, REF_tot, l2_dif=l2_dif)
    ref_te = _logit(p_ref_te).astype(np.float32)
    lr_dfull = LogisticRegression(C=0.05, max_iter=2000).fit(D["DRAFT"][tr], y[tr])
    draft_te = _logit(lr_dfull.predict_proba(D["DRAFT"][te])[:, 1]).astype(np.float32)

    def mat(idx, r_l, d_l):
        po = np.where(post[idx][:, None], X_post[idx], np.nan)
        return np.c_[r_l, d_l, np.abs(r_l), X_pre[idx], po, post[idx].astype(np.float32)].astype(np.float32)

    m_ok = ch >= 1
    tr_cb = tr[m_ok]
    X_tr = mat(tr_cb, ref_tr[m_ok], draft_tr[m_ok])
    ref_cb = ref_tr[m_ok].astype(np.float64)
    y_cb = y[tr_cb]
    X_te = mat(te, ref_te, draft_te)

    pre_m, pos_m = ~post[tr_cb], post[tr_cb]
    if pos_m.sum() >= 1000:
        c_pre = np.quantile(t[tr_cb][pre_m], 0.88) if pre_m.any() else 0
        c_pos = np.quantile(t[tr_cb][pos_m], 0.80)
        is_es = np.where(post[tr_cb], t[tr_cb] >= c_pos, t[tr_cb] >= c_pre)
    else:
        is_es = t[tr_cb] >= np.quantile(t[tr_cb], 0.85)

    cb = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.02,
        depth=5,
        l2_leaf_reg=150.0,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=42,
        task_type="GPU" if DEV.type == "cuda" else "CPU",
        verbose=False,
        early_stopping_rounds=120,
    )
    cb.fit(
        Pool(X_tr[~is_es], y_cb[~is_es], baseline=ref_cb[~is_es]),
        eval_set=Pool(X_tr[is_es], y_cb[is_es], baseline=ref_cb[is_es]),
        use_best_model=True,
    )
    p_te = cb.predict_proba(Pool(X_te, baseline=ref_te.astype(np.float64)))[:, 1].astype(np.float32)
    if dv_idx is not None:
        p_ref_dv = ajuste_l2_dif(D, tr, dv_idx, REF_tot, l2_dif=l2_dif)
        ref_dv = _logit(p_ref_dv).astype(np.float32)
        draft_dv = _logit(lr_dfull.predict_proba(D["DRAFT"][dv_idx])[:, 1]).astype(np.float32)
        X_dv = mat(dv_idx, ref_dv, draft_dv)
        p_dv = cb.predict_proba(Pool(X_dv, baseline=ref_dv.astype(np.float64)))[:, 1].astype(np.float32)
        return p_ref_te, p_te, p_ref_dv, p_dv
    return p_ref_te, p_te


def reporte_segmentos_y_cobertura(D, p, nom):
    vi = np.flatnonzero(D["val"])
    y = D["y"][vi]
    pv = p if len(p) == len(vi) else p[vi]
    Xj, cols = D["Xj"][vi], D["cols"]
    c = {k: i for i, k in enumerate(cols)}
    n_prev = np.expm1(Xj[..., c["logn_tot"]]).reshape(len(vi), 10)
    estr = Xj[..., c["estreno"]].reshape(len(vi), 10).sum(1)
    hora = ((D["t"][vi] / 3.6e6) - 3.0) % 24.0
    conf = np.maximum(pv, 1.0 - pv)
    ord_c = np.argsort(-conf)
    covs = {q: ((pv[ord_c[:max(1, int(len(vi) * q))]] > 0.5) == y[ord_c[:max(1, int(len(vi) * q))]]).mean()
            for q in (0.05, 0.10, 0.20, 0.30, 0.50)}
    segs = {
        "sin_estrenos": estr == 0,
        "veteranos>=50": n_prev.min(1) >= 50,
        "18-24h": (hora >= 18) & (hora < 24),
        "0-6h(madrug)": (hora >= 0) & (hora < 6),
    }
    print(f"\n[{nom}] V Master+ AUC={roc_auc_score(y, pv):.4f} ac={((pv>0.5)==y).mean():.2%} LL={log_loss(y, pv):.4f} ECE={_ece(y, pv):.4f}")
    print("  Cobertura:", " | ".join(f"{int(q*100)}%->{v:.1%}" for q, v in covs.items()))
    print("  Segmentos:", " | ".join(f"{k}={roc_auc_score(y[m], pv[m]):.4f}" for k, m in segs.items() if m.sum() >= 200))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--escalon", type=int, default=7, choices=[6, 7], help="Escalón base de extras_referencia (default=7 oficial)")
    parser.add_argument("--con_g20c", action="store_true", help="Incluir full_ad_risk (B[:, 3:4]) de gemini_g20.npz")
    parser.add_argument("--sin_l2dif", action="store_true", help="Usar Escalón base estándar C=0.05 en baseline CatBoost")
    args = parser.parse_args()

    t0 = time.time()
    D = cargar()
    ms = D["master"]
    REF_base = extras_referencia(D, escalon=args.escalon)
    extra_opt = None
    if args.con_g20c:
        extra_opt = np.load(BR_DIR / "gemini_g20.npz")["BUILD_ARCH_ADAPT"][:, 3:4].astype(np.float32)
    REF_tot = REF_base if extra_opt is None else np.c_[REF_base, extra_opt].astype(np.float32)

    camp_idx, P_jug, W_pre, W_post, F1, COMP, n_camp = preparar_entradas_redes(D, REF_tot, extra_opt)
    X_cb_pre, X_cb_post = construir_X_cb(D, F1, COMP, extra_opt)
    PRIV = construir_priv(D)
    F1_eq = (F1[:, 0] - F1[:, 1]).mean(1).astype(np.float32)

    fi, bl, NM = D["fi"], D["bloque"], D["NM"]
    fi_v, dv = D["fi_v"], D["diam_v"]

    po_ref = np.full(NM, np.nan)
    po_log = np.full(NM, np.nan)
    po_nac = np.full(NM, np.nan)
    po_att = np.full(NM, np.nan)
    po_cb = np.full(NM, np.nan)

    seeds_nac = [0, 1, 2, 101, 202, 303]
    seeds_att = [11, 22, 33, 44, 55]
    use_l2dif = not args.sin_l2dif

    print(f"[g_22 Esc{args.escalon}] Entrenando A2..A4 (3 pliegues) + V ({len(fi_v[ms[fi_v]]):,} Master+ en train), DEV={DEV}...")
    for b in D["EV"]:
        tr_b = fi[(bl[fi] < b) & ms[fi]]
        te_b = np.flatnonzero((bl == b) & ms)
        print(f"  Bloque {b}: tr={len(tr_b):,}, te={len(te_b):,} ({time.time()-t0:.0f}s)", flush=True)
        po_ref[te_b] = ajustar_predecir(D, tr_b, te_b, extra=REF_base, extra_post=False, con_mapa=False)
        p_maestro_b = probs_maestro_oof(D, PRIV, tr_b, REF_tot, F1_eq, n_folds=4)
        po_nac[te_b] = _sigm(entrenar_redes_fold(
            D, camp_idx, P_jug, W_pre, W_post, n_camp, p_maestro_b, tr_b, te_b,
            seeds_nac, RedNeuralACWideDeep, alfa=0.50, lr_wide=1e-3, wd_wide=1e-1, lr_deep=8e-4, wd_deep=2.5e-2, drop=0.30,
        ))
        po_att[te_b] = _sigm(entrenar_redes_fold(
            D, camp_idx, P_jug, W_pre, W_post, n_camp, p_maestro_b, tr_b, te_b,
            seeds_att, RedAttnWideDeep, alfa=0.55, lr_wide=1e-3, wd_wide=1.2e-1, lr_deep=6e-4, wd_deep=3e-2, drop=0.28,
        ))
        po_log[te_b], po_cb[te_b] = catboost_residual_v2_fold(
            D, X_cb_pre, X_cb_post, REF_tot, tr_b, te_b, l2_dif=use_l2dif
        )

    tr_v = fi_v[ms[fi_v]]
    te_v = np.flatnonzero(D["val"])
    dv_idx = np.flatnonzero(dv)
    print(f"  Ventana V: tr={len(tr_v):,}, te_V={len(te_v):,}, Diamond-V={len(dv_idx):,} ({time.time()-t0:.0f}s)", flush=True)
    pv_ref = ajustar_predecir(D, tr_v, te_v, extra=REF_base, extra_post=False, con_mapa=False)
    p_dv_ref = ajustar_predecir(D, tr_v, dv_idx, extra=REF_base, extra_post=False, con_mapa=False)

    p_maestro_v = probs_maestro_oof(D, PRIV, tr_v, REF_tot, F1_eq, n_folds=4)
    zn_v, zn_dv = entrenar_redes_fold(
        D, camp_idx, P_jug, W_pre, W_post, n_camp, p_maestro_v, tr_v, te_v,
        seeds_nac, RedNeuralACWideDeep, alfa=0.50, lr_wide=1e-3, wd_wide=1e-1, lr_deep=8e-4, wd_deep=2.5e-2, drop=0.30,
        dv_idx=dv_idx,
    )
    za_v, za_dv = entrenar_redes_fold(
        D, camp_idx, P_jug, W_pre, W_post, n_camp, p_maestro_v, tr_v, te_v,
        seeds_att, RedAttnWideDeep, alfa=0.55, lr_wide=1e-3, wd_wide=1.2e-1, lr_deep=6e-4, wd_deep=3e-2, drop=0.28,
        dv_idx=dv_idx,
    )
    pv_nac, p_dv_nac = _sigm(zn_v), _sigm(zn_dv)
    pv_att, p_dv_att = _sigm(za_v), _sigm(za_dv)

    pv_log, pv_cb, p_dv_log, p_dv_cb = catboost_residual_v2_fold(
        D, X_cb_pre, X_cb_post, REF_tot, tr_v, te_v, dv_idx=dv_idx, l2_dif=use_l2dif
    )

    out_npz = BR_DIR / ("pred_g22_esc7.npz" if args.escalon == 7 else "pred_g22_esc6.npz")
    np.savez_compressed(
        out_npz,
        escalon=np.array([args.escalon]),
        po_ref=po_ref, pv_ref=pv_ref, p_dv_ref=p_dv_ref,
        po_log=po_log, pv_log=pv_log, p_dv_log=p_dv_log,
        po_nac=po_nac, pv_nac=pv_nac, p_dv_nac=p_dv_nac,
        po_att=po_att, pv_att=pv_att, p_dv_att=p_dv_att,
        po_cb=po_cb, pv_cb=pv_cb, p_dv_cb=p_dv_cb,
    )
    print(f"Predicciones guardadas en {out_npz.name} ({time.time() - t0:.1f}s)", flush=True)

    def mix_logit(items):
        z_o = np.zeros(NM, dtype=np.float64)
        z_v = np.zeros(len(te_v), dtype=np.float64)
        for w, (po_i, pv_i) in items:
            z_o += w * _logit(np.nan_to_num(po_i, nan=0.5))
            z_v += w * _logit(np.nan_to_num(pv_i, nan=0.5))
        return (_sigm(z_o), _sigm(z_v))

    ref_k = f"ref: Escalón {args.escalon} (sin mapa)"
    lbl_log = f"Log Esc{args.escalon}" + (" + L2 dif" if use_l2dif else "")
    lbl_nac = f"Red NeuralAC Esc{args.escalon}+ (6sem)"
    lbl_att = f"Red AttnWideDeep Esc{args.escalon}+ (5sem)"
    P = {
        ref_k: (po_ref, pv_ref),
        lbl_log: (po_log, pv_log),
        lbl_nac: (po_nac, pv_nac),
        lbl_att: (po_att, pv_att),
        "CatBoost GPU Residual v2": (po_cb, pv_cb),
        "Mix 65/35 Log+NAC": mix_logit([(0.65, (po_log, pv_log)), (0.35, (po_nac, pv_nac))]),
        "Mix 55/30/15 Log+NAC+Att": mix_logit([(0.55, (po_log, pv_log)), (0.30, (po_nac, pv_nac)), (0.15, (po_att, pv_att))]),
        "Mix 45/25/15/15 Log+NAC+Att+CB": mix_logit([(0.45, (po_log, pv_log)), (0.25, (po_nac, pv_nac)), (0.15, (po_att, pv_att)), (0.15, (po_cb, pv_cb))]),
        "Mix 40/30/15/15 Log+NAC+Att+CB": mix_logit([(0.40, (po_log, pv_log)), (0.30, (po_nac, pv_nac)), (0.15, (po_att, pv_att)), (0.15, (po_cb, pv_cb))]),
        "Mix 50/25/25 Log+NAC+CB": mix_logit([(0.50, (po_log, pv_log)), (0.25, (po_nac, pv_nac)), (0.25, (po_cb, pv_cb))]),
    }
    mostrar(comparar(D, P, ref_k))

    for k in (
        ref_k,
        lbl_log,
        lbl_nac,
        lbl_att,
        "CatBoost GPU Residual v2",
        "Mix 65/35 Log+NAC",
        "Mix 45/25/15/15 Log+NAC+Att+CB",
        "Mix 40/30/15/15 Log+NAC+Att+CB",
    ):
        reporte_segmentos_y_cobertura(D, P[k][1], k)

    ydv = D["y"][dv]
    auc_ref_dv = roc_auc_score(ydv, p_dv_ref)
    rng = np.random.default_rng(77)
    boots = rng.integers(0, len(ydv), size=(1500, len(ydv)))
    print(f"\nChequeo ciego Diamond-V ({len(ydv):,} partidas): ref={auc_ref_dv:.4f}")
    dv_preds = {
        lbl_log: p_dv_log,
        f"Red NeuralAC Esc{args.escalon}+": p_dv_nac,
        f"Red AttnWideDeep Esc{args.escalon}+": p_dv_att,
        "CatBoost GPU Residual v2": p_dv_cb,
        "Mix 65/35 Log+NAC": _sigm(0.65 * _logit(p_dv_log) + 0.35 * _logit(p_dv_nac)),
        "Mix 55/30/15 Log+NAC+Att": _sigm(0.55 * _logit(p_dv_log) + 0.30 * _logit(p_dv_nac) + 0.15 * _logit(p_dv_att)),
        "Mix 45/25/15/15": _sigm(0.45 * _logit(p_dv_log) + 0.25 * _logit(p_dv_nac) + 0.15 * _logit(p_dv_att) + 0.15 * _logit(p_dv_cb)),
        "Mix 40/30/15/15": _sigm(0.40 * _logit(p_dv_log) + 0.30 * _logit(p_dv_nac) + 0.15 * _logit(p_dv_att) + 0.15 * _logit(p_dv_cb)),
    }
    for nom, pdv in dv_preds.items():
        auc_p = roc_auc_score(ydv, pdv)
        ds = np.array([roc_auc_score(ydv[b], pdv[b]) - roc_auc_score(ydv[b], p_dv_ref[b]) for b in boots])
        print(f"  {nom:26s} AUC {auc_p:.4f}  Δ {auc_p - auc_ref_dv:+.4f} [{np.percentile(ds, 2.5):+.4f}; {np.percentile(ds, 97.5):+.4f}]")

    print(f"\nFin g_22. Listo en {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
