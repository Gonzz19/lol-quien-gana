"""Ideas 18 y 19 del brainstorming (docs/BRAINSTORM_NOCHE.md):
1. Destilación con Información Privilegiada (@10/@15) en la Red Wide & Deep GPU sobre `solo Master+`
   (§9.S.39 había dejado explícito: 'El candidato natural es la red de grupos... Pendiente correrla
   en la PC con GPU', y nunca se corrió con solo Master+ + MC/ML + mapa + Foto 1 W/L).
   El maestro usa los snapshots @10/@15 de `variables.npz` (`D_d_gold_10/15`, `D_d_xp_10/15`,
   `D_d_kills_10/15`, `D_fb_azul`, `D_ft_azul`) por 3 pliegues estratificados por fecha *dentro*
   de cada `tr` (nunca ve `te`) para producir la etiqueta suave `q = alpha * y + (1 - alpha) * p_maestro`.
2. Experto Árboles en GPU (`CatBoostClassifier` sobre RTX 3070) en `solo Master+`:
   En §9.S.25/39 los boostings fallaron porque se les pasaban 800+ one-hots dispersos de campeón y
   425+ columnas por línea con Diamond mezclado. Acá reciben:
   * `logit_draft_oof` (1 columna causal de `Logistic(DRAFT)` entrenado fuera de pliegue dentro de `tr`),
   * resúmenes compactos por equipo + agregados max/min/std + MC/ML + mapa + Foto 1 W/L dinámico +
     interacciones clave (autofill, baja cobertura).

Evalúa cada uno solo y combinado en ensamble (logits) con la Logística (+ MC/ML + Foto1 W/L).
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from catboost import CatBoostClassifier, Pool
from scipy import sparse
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.brainstorm_base import cargar, comparar, evaluar, mostrar  # noqa: E402
from historial.ensamble_en_vivo.brainstorm_g_05_gpu_master import (  # noqa: E402
    DEV,
    RedWideDeep,
    cargar_foto1_jugador,
    preparar_datos_red,
)
from historial.escalon9.brainstorm_g_03_agregados import construir_bloques  # noqa: E402


def construir_priv(D: dict) -> np.ndarray:
    """Features privilegiadas in-game @10/@15 desde variables.npz (SOLO para maestro dentro de tr)."""
    Z = D["Z"]
    cols = []
    for k in ("D_d_gold_10", "D_d_xp_10", "D_d_kills_10", "D_d_gold_15", "D_d_xp_15", "D_d_kills_15"):
        a = Z[k].astype(np.float32)
        cols.append(np.nan_to_num(a, nan=0.0))
        if "15" in k:
            cols.append(np.isfinite(a).astype(np.float32))
    cols.append(Z["D_fb_azul"].astype(np.float32) - 0.5)
    cols.append(Z["D_ft_azul"].astype(np.float32) - 0.5)
    return np.column_stack(cols).astype(np.float32)


def probs_maestro_oof(D: dict, PRIV: np.ndarray, tr: np.ndarray, MC_ML: np.ndarray, F1_eq: np.ndarray, n_folds: int = 4) -> np.ndarray:
    """Probabilidades del maestro (@10/@15 + pre-partida) por pliegues estratificados en tiempo dentro de `tr`."""
    t, y, post, Xd, mapa = D["t"], D["y"], D["post"], D["Xd"], D["mapa"]
    ord_tr = np.argsort(t[tr], kind="mergesort")
    folds = np.empty(len(tr), dtype=np.int32)
    folds[ord_tr] = np.arange(len(tr)) % n_folds
    p_m = np.full(len(tr), 0.5, dtype=np.float64)
    Xpre = np.c_[Xd, MC_ML, PRIV].astype(np.float32)
    Xpo = np.c_[mapa, F1_eq].astype(np.float32)
    for k in range(n_folds):
        mtr = folds != k
        mte = folds == k
        itr, ite = tr[mtr], tr[mte]
        mu, sd = Xpre[itr].mean(0), Xpre[itr].std(0) + 1e-6
        ptr = post[itr]
        if ptr.any():
            me, se = Xpo[itr][ptr].mean(0), Xpo[itr][ptr].std(0) + 1e-6
            po_tr = np.where(ptr[:, None], (Xpo[itr] - me) / se, 0.0).astype(np.float32)
            po_te = np.where(post[ite][:, None], (Xpo[ite] - me) / se, 0.0).astype(np.float32)
        else:
            po_tr = np.zeros((len(itr), Xpo.shape[1]), dtype=np.float32)
            po_te = np.zeros((len(ite), Xpo.shape[1]), dtype=np.float32)
        Mtr = sparse.hstack([
            D["DRAFT"][itr],
            sparse.csr_matrix(((Xpre[itr] - mu) / sd).astype(np.float32)),
            sparse.csr_matrix(po_tr),
        ]).tocsr()
        Mte = sparse.hstack([
            D["DRAFT"][ite],
            sparse.csr_matrix(((Xpre[ite] - mu) / sd).astype(np.float32)),
            sparse.csr_matrix(po_te),
        ]).tocsr()
        lr = LogisticRegression(C=0.05, max_iter=3000).fit(Mtr, y[itr])
        p_m[mte] = lr.predict_proba(Mte)[:, 1]
    return np.clip(p_m, 1e-4, 1.0 - 1e-4)


class RedNeuralACWideDeep(RedWideDeep):
    """Wide&Deep GPU + módulo antisimétrico NeuralAC (duelo directo por línea, cruce 5x5 y sinergia FM)."""

    def __init__(self, n_camp: int, d_num: int, d_wide: int, h_dim: int = 96, drop: float = 0.30, k_nac: int = 12):
        super().__init__(n_camp, d_num, d_wide, h_dim=h_dim, drop=drop)
        self.k_nac = k_nac
        self.u_proj = nn.Linear(h_dim, k_nac, bias=False)
        self.w_proj = nn.Linear(h_dim, k_nac, bias=False)
        self.g_syn = nn.Linear(h_dim, k_nac, bias=False)
        self.nac_scale = nn.Parameter(torch.tensor([0.2, 0.1, 0.1]))

    def forward(self, camp, num, wide_x):
        bsz = len(camp)
        lin = self.lineas.expand(bsz, -1)
        cl_idx = camp * 5 + lin
        cl_w = self.camp_lin(cl_idx).squeeze(-1)
        z_wide = (cl_w[:, :5].sum(1) - cl_w[:, 5:].sum(1)) + self.wide(wide_x).squeeze(-1)

        x = torch.cat([self.camp(camp), self.linea(lin), num], dim=2)
        h = self.phi(x)
        hb, hr = h[:, :5], h[:, 5:]
        eb = torch.cat([hb.mean(1), hb.max(1).values], dim=1)
        er = torch.cat([hr.mean(1), hr.max(1).values], dim=1)
        z_deep = (self.g(torch.cat([eb, er], dim=1)) - self.g(torch.cat([er, eb], dim=1))).squeeze(-1)
        z_lado = self.lado_base + self.lado_lob(eb + er).squeeze(-1)

        ub, ur = torch.tanh(self.u_proj(hb)), torch.tanh(self.u_proj(hr))
        wb, wr = torch.tanh(self.w_proj(hb)), torch.tanh(self.w_proj(hr))
        s = self.k_nac ** 0.5
        z_duel = ((ub * wr).sum(dim=(1, 2)) - (ur * wb).sum(dim=(1, 2))) / s
        z_cross = ((ub.sum(1) * wr.sum(1)).sum(1) - (ur.sum(1) * wb.sum(1)).sum(1)) / (5.0 * s)
        gb, gr = torch.tanh(self.g_syn(hb)), torch.tanh(self.g_syn(hr))
        syn_b = 0.5 * ((gb.sum(1) ** 2 - (gb ** 2).sum(1)).sum(1)) / (4.0 * s)
        syn_r = 0.5 * ((gr.sum(1) ** 2 - (gr ** 2).sum(1)).sum(1)) / (4.0 * s)
        z_nac = self.nac_scale[0] * z_duel + self.nac_scale[1] * z_cross + self.nac_scale[2] * (syn_b - syn_r)
        return z_wide + 0.5 * z_deep + z_nac + z_lado


def entrenar_red_destilada(
    D, camp_idx, P_jug, W_pre, d_post_eq, n_camp, PRIV, MC_ML, F1_eq, tr_raw, te,
    alfa: float = 0.5, seeds=(0, 1, 2), usar_nac: bool = True,
):
    ms, t, y, post = D["master"], D["t"], D["y"], D["post"]
    tr = tr_raw[ms[tr_raw]]

    if alfa < 1.0:
        p_maestro = probs_maestro_oof(D, PRIV, tr, MC_ML, F1_eq, n_folds=4)
        q_soft = alfa * y[tr].astype(np.float64) + (1.0 - alfa) * p_maestro
    else:
        q_soft = y[tr].astype(np.float64)

    d_pre_jug = P_jug.shape[2] - 12
    n_tr_flat = P_jug[tr, :, :d_pre_jug].reshape(-1, d_pre_jug)
    mu_jp, sd_jp = n_tr_flat.mean(0), n_tr_flat.std(0) + 1e-6
    post_tr = tr[post[tr]]
    if len(post_tr) > 0:
        n_po_flat = P_jug[post_tr, :, d_pre_jug:-1].reshape(-1, 11)
        mu_jpo, sd_jpo = n_po_flat.mean(0), n_po_flat.std(0) + 1e-6
    else:
        mu_jpo, sd_jpo = np.zeros(11, np.float32), np.ones(11, np.float32)

    mu_wp, sd_wp = W_pre[tr].mean(0), W_pre[tr].std(0) + 1e-6
    if len(post_tr) > 0:
        mu_wpo, sd_wpo = d_post_eq[post_tr].mean(0), d_post_eq[post_tr].std(0) + 1e-6
    else:
        mu_wpo, sd_wpo = np.zeros(d_post_eq.shape[1], np.float32), np.ones(d_post_eq.shape[1], np.float32)

    def build_tensors(idx):
        p_norm = np.empty_like(P_jug[idx])
        p_norm[..., :d_pre_jug] = np.clip((P_jug[idx, :, :d_pre_jug] - mu_jp) / sd_jp, -8.0, 8.0)
        m_post = post[idx][:, None, None]
        p_norm[..., d_pre_jug:-1] = np.where(
            m_post, np.clip((P_jug[idx, :, d_pre_jug:-1] - mu_jpo) / sd_jpo, -8.0, 8.0), 0.0
        )
        p_norm[..., -1] = P_jug[idx, :, -1]
        w_pre_n = np.clip((W_pre[idx] - mu_wp) / sd_wp, -8.0, 8.0)
        w_post_n = np.where(
            post[idx][:, None], np.clip((d_post_eq[idx] - mu_wpo) / sd_wpo, -8.0, 8.0), 0.0
        )
        wide_n = np.hstack([w_pre_n, w_post_n]).astype(np.float32)
        return (
            torch.tensor(camp_idx[idx], dtype=torch.long, device=DEV),
            torch.tensor(p_norm, dtype=torch.float32, device=DEV),
            torch.tensor(wide_n, dtype=torch.float32, device=DEV),
        )

    pre_mask = ~post[tr]
    pos_mask = post[tr]
    if pos_mask.sum() >= 1000:
        c_pre = np.quantile(t[tr][pre_mask], 0.88) if pre_mask.any() else 0
        c_pos = np.quantile(t[tr][pos_mask], 0.80)
        is_es = np.where(post[tr], t[tr] >= c_pos, t[tr] >= c_pre)
        es_eval_mask = post[tr[is_es]]
    else:
        corte = np.quantile(t[tr], 0.85)
        is_es = t[tr] >= corte
        es_eval_mask = np.ones(int(is_es.sum()), dtype=bool)

    tr_sub = tr[~is_es]
    es_sub = tr[is_es]
    q_tr = torch.tensor(q_soft[~is_es], dtype=torch.float32, device=DEV)
    y_es_np = y[es_sub]

    C_tr, N_tr, W_tr = build_tensors(tr_sub)
    C_es, N_es, W_es = build_tensors(es_sub)
    C_te, N_te, W_te = build_tensors(te)

    preds = []
    for seed in seeds:
        torch.manual_seed(seed)
        net_cls = RedNeuralACWideDeep if usar_nac else RedWideDeep
        net = net_cls(n_camp, P_jug.shape[2], W_tr.shape[1], h_dim=96, drop=0.30).to(DEV)
        opt = torch.optim.AdamW([
            {"params": list(net.camp_lin.parameters()) + list(net.wide.parameters()), "weight_decay": 1e-1, "lr": 1e-3},
            {"params": [p for n, p in net.named_parameters() if not n.startswith(("camp_lin.", "wide."))], "weight_decay": 2e-2, "lr": 8e-4},
        ])
        best_ll, best_sd, pat = 1e9, None, 0
        n_rows = len(tr_sub)
        for _ in range(60):
            net.train()
            perm = torch.randperm(n_rows, device=DEV)
            for b in perm.split(512):
                opt.zero_grad()
                out = net(C_tr[b], N_tr[b], W_tr[b])
                loss = nn.functional.binary_cross_entropy_with_logits(out, q_tr[b])
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                p_es = []
                for i in range(0, len(es_sub), 4096):
                    p_es.append(torch.sigmoid(net(C_es[i:i+4096], N_es[i:i+4096], W_es[i:i+4096])).cpu().numpy())
                # Parada temprana contra etiqueta real y_es_np (nunca contra el maestro)
                ll = log_loss(y_es_np[es_eval_mask], np.clip(np.concatenate(p_es)[es_eval_mask], 1e-6, 1 - 1e-6))
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
            p_te = []
            for i in range(0, len(te), 4096):
                p_te.append(torch.sigmoid(net(C_te[i:i+4096], N_te[i:i+4096], W_te[i:i+4096])).cpu().numpy())
            preds.append(np.concatenate(p_te))
    del C_tr, N_tr, W_tr, C_es, N_es, W_es, C_te, N_te, W_te
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
    return np.mean(preds, axis=0)


def catboost_residual_gpu(D, MC_raw, ML_raw, A_mmr, NIV_EQ, F1, COMP, MATCHUPS, REF_BASE, tr_raw, te):
    """CatBoost GPU usando `baseline=logit_ref` estrictamente causal por bloques hacia adelante en `tr`.
    Los árboles se concentran 100 % en interacciones no lineales (veteranos/sin-estreno/smurfs/autofill)
    sobre la frontera lineal ya aprendida por la Logística Escalón 3 + MATCHUPS."""
    ms, t, y, post, Xj, mapa = D["master"], D["t"], D["y"], D["post"], D["Xj"], D["mapa"]
    tr = tr_raw[ms[tr_raw]]

    # 1. Logit causal de la logística Escalón 3 + MATCHUPS (6 bloques cronológicos hacia adelante) y draft_oof
    qs = np.quantile(t[tr], np.linspace(0, 1, 7)[1:-1])
    ch = np.searchsorted(qs, t[tr], side="right")
    ref_tr = np.zeros(len(tr), dtype=np.float32)
    draft_tr = np.zeros(len(tr), dtype=np.float32)
    X_lin = np.c_[REF_BASE, MATCHUPS].astype(np.float32)
    from scripts.analysis.brainstorm_base import ajustar_predecir

    for b in range(1, 6):
        itr, ite = tr[ch < b], tr[ch == b]
        p_ref = ajustar_predecir(D, itr, ite, extra=X_lin, extra_post=False, con_mapa=bool(post[itr].any()))
        ref_tr[ch == b] = logit(np.clip(p_ref, 1e-5, 1.0 - 1e-5))
        lr_d = LogisticRegression(C=0.05, max_iter=2000).fit(D["DRAFT"][itr], y[itr])
        draft_tr[ch == b] = logit(np.clip(lr_d.predict_proba(D["DRAFT"][ite])[:, 1], 1e-5, 1.0 - 1e-5))
    p_ref_te = ajustar_predecir(D, tr, te, extra=X_lin, extra_post=False)
    ref_te = logit(np.clip(p_ref_te, 1e-5, 1.0 - 1e-5)).astype(np.float32)
    lr_dfull = LogisticRegression(C=0.05, max_iter=2000).fit(D["DRAFT"][tr], y[tr])
    draft_te = logit(np.clip(lr_dfull.predict_proba(D["DRAFT"][te])[:, 1], 1e-5, 1.0 - 1e-5)).astype(np.float32)

    # 2. Rasgos de equipo + contexto de segmento de lobby (sin fuga)
    cols = {k: i for i, k in enumerate(D["cols"])}
    n_prev = np.expm1(Xj[..., cols["logn_tot"]]).reshape(D["NM"], 10)
    estr_cnt = Xj[..., cols["estreno"]].reshape(D["NM"], 10).sum(1)
    off_cnt = Xj[..., cols["offrole"]].reshape(D["NM"], 10).sum(1)
    sin_foto_cnt = np.isnan(D["Z"]["tnum"]).sum(1)
    abs_elo = (A_mmr[:, 0, :, 0].mean(1) + A_mmr[:, 1, :, 0].mean(1)) * 0.5

    Xeq_pre = np.c_[
        (Xj[:, 0] - Xj[:, 1]).mean(axis=1),
        (MC_raw[:, 0] - MC_raw[:, 1]).mean(axis=1),
        MC_raw[:, 0].min(axis=1) - MC_raw[:, 1].min(axis=1),
        (ML_raw[:, 0] - ML_raw[:, 1]).mean(axis=1),
        (A_mmr[:, 0, :, :3] - A_mmr[:, 1, :, :3]).mean(axis=1),
        MATCHUPS,
        COMP,
        estr_cnt,
        n_prev.min(1),
        off_cnt,
        sin_foto_cnt,
        abs_elo,
        D["tier_lobby"],
    ].astype(np.float32)
    Xeq_post = np.c_[
        mapa,
        (F1[:, 0] - F1[:, 1]).mean(axis=1),
    ].astype(np.float32)

    def feat_mat(idx, ref_l, draft_l):
        po = np.where(post[idx][:, None], Xeq_post[idx], np.nan)
        return np.c_[ref_l, draft_l, Xeq_pre[idx], po, post[idx].astype(np.float32)].astype(np.float32)

    m_ok = ch >= 1
    tr_cb = tr[m_ok]
    X_all_tr = feat_mat(tr_cb, ref_tr[m_ok], draft_tr[m_ok])
    ref_cb = ref_tr[m_ok].astype(np.float64)
    y_cb = y[tr_cb]
    X_all_te = feat_mat(te, ref_te, draft_te)

    # Split interno causal que garantiza filas post en eval
    pre_m, pos_m = ~post[tr_cb], post[tr_cb]
    if pos_m.sum() >= 1000:
        c_pre = np.quantile(t[tr_cb][pre_m], 0.88) if pre_m.any() else 0
        c_pos = np.quantile(t[tr_cb][pos_m], 0.80)
        is_es = np.where(post[tr_cb], t[tr_cb] >= c_pos, t[tr_cb] >= c_pre)
    else:
        is_es = t[tr_cb] >= np.quantile(t[tr_cb], 0.85)

    cb = CatBoostClassifier(
        iterations=1200,
        learning_rate=0.025,
        depth=4,
        l2_leaf_reg=100.0,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=42,
        task_type="GPU" if DEV.type == "cuda" else "CPU",
        verbose=False,
        early_stopping_rounds=100,
    )
    pool_tr = Pool(X_all_tr[~is_es], y_cb[~is_es], baseline=ref_cb[~is_es])
    pool_es = Pool(X_all_tr[is_es], y_cb[is_es], baseline=ref_cb[is_es])
    pool_te = Pool(X_all_te, baseline=ref_te.astype(np.float64))
    cb.fit(pool_tr, eval_set=pool_es, use_best_model=True)
    return cb.predict_proba(pool_te)[:, 1]


def main():
    t0 = time.time()
    D = cargar()
    NM, ms = D["NM"], D["master"]
    C = np.load(REPO / "data" / "analysis" / "brainstorm" / "margen_grupos.npz")
    MC_raw, ML_raw = C["MC"], C["ML"]
    A_mmr = np.load(REPO / "data" / "analysis" / "brainstorm" / "mmr_historico.npz")["A"].astype(np.float32)
    NIV_EQ = (A_mmr[:, 0] - A_mmr[:, 1])[..., :3].mean(axis=1).astype(np.float32)
    MC_ML_NIV = np.c_[
        (MC_raw[:, 0] - MC_raw[:, 1]).reshape(NM, -1),
        (ML_raw[:, 0] - ML_raw[:, 1]).reshape(NM, -1),
        NIV_EQ,
    ].astype(np.float32)
    F1 = cargar_foto1_jugador(D)
    F1_eq = (F1[:, 0] - F1[:, 1]).mean(axis=1).astype(np.float32)
    PRIV = construir_priv(D)

    from scripts.analysis.brainstorm_base import ajustar_predecir, extras_referencia
    from historial.escalon9.brainstorm_g_06_comp_poisson import construir_composicion

    REF_BASE = extras_referencia(D)
    COMP, J_CAMP = construir_composicion(D)
    mpr_path = REPO / "data" / "analysis" / "brainstorm" / "gemini_mpr.npz"
    MATCHUPS = np.load(mpr_path)["MATCHUPS"].astype(np.float32) if mpr_path.exists() else np.zeros((NM, 0), np.float32)
    X_ESC4 = np.c_[REF_BASE, MATCHUPS].astype(np.float32)

    camp_idx, P_jug_base, W_pre_base, d_post_eq, n_camp = preparar_datos_red(D, F1, MC_raw, ML_raw)
    # Inyectar A_mmr[..., :3] + atributos Meraki/CDragon al principio de P_jug (conservando 12 cols finales post!)
    A_flat = A_mmr[:, :, :, :3].reshape(NM, 10, -1)
    C_flat = J_CAMP.reshape(NM, 10, -1)
    P_jug = np.concatenate([A_flat, C_flat, P_jug_base], axis=-1).astype(np.float32)
    W_pre = np.c_[W_pre_base, NIV_EQ, MATCHUPS, COMP].astype(np.float32)

    # Cargar referencia oficial Escalón 3 (0,6663 / 62,0 %) desde pred_combinado.npz
    pc_path = REPO / "data" / "analysis" / "brainstorm" / "pred_combinado.npz"
    if pc_path.exists():
        PC = np.load(pc_path)
        p_ref3 = (PC["o2"], PC["v2"])
    else:
        p_ref3 = evaluar(D, lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=REF_BASE, extra_post=False))

    print(f"Evaluando Logística Escalón 3 + MATCHUPS ({time.time()-t0:.1f}s)...", flush=True)
    p_log4 = evaluar(D, lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=X_ESC4, extra_post=False))

    print(f"Evaluando Red Wide&Deep+NeuralAC GPU alfa=1.0 ({time.time()-t0:.1f}s)...", flush=True)
    p_red_nodist = evaluar(D, lambda tr, te: entrenar_red_destilada(
        D, camp_idx, P_jug, W_pre, d_post_eq, n_camp, PRIV, np.c_[MC_ML_NIV, MATCHUPS].astype(np.float32), F1_eq, tr, te, alfa=1.0
    ))

    print(f"Evaluando Red Wide&Deep+NeuralAC GPU Destilada alfa=0.5 ({time.time()-t0:.1f}s)...", flush=True)
    p_dist = evaluar(D, lambda tr, te: entrenar_red_destilada(
        D, camp_idx, P_jug, W_pre, d_post_eq, n_camp, PRIV, np.c_[MC_ML_NIV, MATCHUPS].astype(np.float32), F1_eq, tr, te, alfa=0.5
    ))

    print(f"Evaluando CatBoost GPU Residual (baseline=Log Esc3+MATCHUPS, {time.time()-t0:.1f}s)...", flush=True)
    p_cb = evaluar(D, lambda tr, te: catboost_residual_gpu(D, MC_raw, ML_raw, A_mmr, NIV_EQ, F1, COMP, MATCHUPS, REF_BASE, tr, te))

    def blend3(wa, wb, wc):
        eps = 1e-6
        lo = (wa * logit(np.clip(p_log4[0], eps, 1 - eps))
              + wb * logit(np.clip(p_dist[0], eps, 1 - eps))
              + wc * logit(np.clip(p_cb[0], eps, 1 - eps)))
        lv = (wa * logit(np.clip(p_log4[1], eps, 1 - eps))
              + wb * logit(np.clip(p_dist[1], eps, 1 - eps))
              + wc * logit(np.clip(p_cb[1], eps, 1 - eps)))
        return (expit(lo), expit(lv))

    def blend2_nodist(wa, wb):
        eps = 1e-6
        lo = wa * logit(np.clip(p_log4[0], eps, 1 - eps)) + wb * logit(np.clip(p_red_nodist[0], eps, 1 - eps))
        lv = wa * logit(np.clip(p_log4[1], eps, 1 - eps)) + wb * logit(np.clip(p_red_nodist[1], eps, 1 - eps))
        return (expit(lo), expit(lv))

    P = {
        "ref: Escalón 3 (0,6663)": p_ref3,
        "Logística (Esc3 + MATCHUPS)": p_log4,
        "Red NeuralAC GPU (alfa=1)": p_red_nodist,
        "Red NeuralAC Destilada (alfa=0.5)": p_dist,
        "CatBoost GPU Residual (base=Esc4)": p_cb,
        "Mix 65/35 Log4 + Red(alfa=1)": blend2_nodist(0.65, 0.35),
        "Mix 65/35 Log4 + RedDist": blend3(0.65, 0.35, 0.0),
        "Mix 55/25/20 Log4 + RedDist + CB": blend3(0.55, 0.25, 0.20),
        "Mix 45/30/25 Log4 + RedDist + CB": blend3(0.45, 0.30, 0.25),
        "Mix 60/40 Log4 + CB": blend3(0.60, 0.0, 0.40),
    }
    print("\n--- Contra referencia oficial Escalón 3 (0,6663) ---")
    mostrar(comparar(D, P, "ref: Escalón 3 (0,6663)"))
    print("\n--- Contra Logística Escalón 3 + MATCHUPS ---")
    mostrar(comparar(D, P, "Logística (Esc3 + MATCHUPS)"))
    np.savez_compressed(
        REPO / "data" / "analysis" / "brainstorm" / "pred_g08.npz",
        po_log4=p_log4[0], pv_log4=p_log4[1],
        po_red=p_red_nodist[0], pv_red=p_red_nodist[1],
        po_dist=p_dist[0], pv_dist=p_dist[1],
        po_cb=p_cb[0], pv_cb=p_cb[1],
    )
    print(f"Listo en {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
