"""Ideas G31-G32 del brainstorming (docs/BRAINSTORM_NOCHE.md):
1. Red AttnWideDeep GPU (10-Player Swap-Antisymmetric Self-Attention + NeuralAC + Wide Skip, 5 semillas):
   - Cada jugador recibe embeddings de campeón y línea + features causales completas (MC, ML, NIV, Meraki/CDragon, F1).
   - Bloque Multi-Head Self-Attention sobre los 10 jugadores (distinguiendo aliado vs rival) evaluado en ambos
     órdenes `(B, R) - (R, B)` para garantizar antisimetría matemática exacta frente al cambio de bando,
     permitiendo que Jungla/Mid/Support atiendan dinámicamente a los 9 restantes antes del pooling `[mean, max]`.
   - Conserva el skip lineal completo (`Esc4`) y calibración de lado (`lado_base + lado_lob`) + NeuralAC duelo por línea.
2. CatBoost GPU Residual v2 (`baseline = logit Escalón 4+`, con contexto por línea, `hora_dia`, `min_n_prev` y `sin_estrenos`):
   - A diferencia de v1 en `g_08` (que solo pasaba promedios de equipo de `Xj`/`MC`/`ML`), expone métricas clave por línea,
     mínimos/máximos de equipo, hora del día (`brainstorm_12` mostró distinta dificultad 18-24h vs 0-6h), `veteranos`,
     `sin_estrenos` y `MATCHUPS` individuales para capturar interacciones no lineales puras sin perder Esc4.

También puede incorporar bloques ganadores externos (`--extras ...`) y cargar `pred_g08.npz` para comparar
ensembles diversos (`Log + NeuralAC + AttnTransformer + CatBoost v2`).
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from catboost import CatBoostClassifier, Pool
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

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
from historial.ensamble_en_vivo.brainstorm_g_05_gpu_master import cargar_foto1_jugador, preparar_datos_red  # noqa: E402
from historial.escalon9.brainstorm_g_06_comp_poisson import construir_composicion  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RedAttnWideDeep(nn.Module):
    """Wide&Deep + Self-Attention antisimétrica de 10 jugadores + NeuralAC duelo directo por línea."""

    def __init__(self, n_camp: int, d_num: int, d_wide: int, d_mod: int = 96, n_heads: int = 4, drop: float = 0.28):
        super().__init__()
        self.camp_lin = nn.Embedding(n_camp * 5, 1)
        nn.init.zeros_(self.camp_lin.weight)
        self.wide = nn.Linear(d_wide, 1, bias=False)
        nn.init.zeros_(self.wide.weight)

        self.camp = nn.Embedding(n_camp, 24)
        self.linea = nn.Embedding(5, 8)
        self.t_role = nn.Parameter(torch.randn(2, d_mod) * 0.02)  # [0]=aliado, [1]=rival
        self.in_proj = nn.Sequential(
            nn.Linear(24 + 8 + d_num, d_mod),
            nn.LayerNorm(d_mod),
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Linear(d_mod, d_mod),
            nn.LayerNorm(d_mod),
            nn.SiLU(),
        )
        self.attn_ln1 = nn.LayerNorm(d_mod)
        self.attn = nn.MultiheadAttention(d_mod, num_heads=n_heads, dropout=drop, batch_first=True)
        self.attn_ln2 = nn.LayerNorm(d_mod)
        self.ffn = nn.Sequential(
            nn.Linear(d_mod, d_mod * 2),
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Linear(d_mod * 2, d_mod),
        )
        self.u_proj = nn.Linear(d_mod, 12, bias=False)
        self.w_proj = nn.Linear(d_mod, 12, bias=False)
        self.head = nn.Sequential(
            nn.Linear(d_mod * 4, d_mod),
            nn.LayerNorm(d_mod),
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Linear(d_mod, 1),
        )
        self.lado_base = nn.Parameter(torch.tensor(-0.10))
        self.lado_lob = nn.Linear(d_mod * 4, 1)
        nn.init.zeros_(self.lado_lob.weight)
        nn.init.zeros_(self.lado_lob.bias)
        self.scale_duel = nn.Parameter(torch.tensor(0.2))
        self.register_buffer("lineas", torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=torch.long))

    def _context(self, h0, idx_self, idx_opp):
        hs = h0[:, idx_self] + self.t_role[0].view(1, 1, -1)
        ho = h0[:, idx_opp] + self.t_role[1].view(1, 1, -1)
        seq = torch.cat([hs, ho], dim=1)
        qn = self.attn_ln1(seq)
        att_out, _ = self.attn(qn, qn, qn, need_weights=False)
        seq = seq + att_out
        seq = seq + self.ffn(self.attn_ln2(seq))
        s_tok = seq[:, :5]
        o_tok = seq[:, 5:]
        es = torch.cat([s_tok.mean(1), s_tok.max(1).values], dim=1)
        eo = torch.cat([o_tok.mean(1), o_tok.max(1).values], dim=1)
        return es, eo, s_tok, o_tok

    def forward(self, camp, num, wide_x):
        bsz = len(camp)
        lin = self.lineas.expand(bsz, -1)
        cl_w = self.camp_lin(camp * 5 + lin).squeeze(-1)
        z_wide = (cl_w[:, :5].sum(1) - cl_w[:, 5:].sum(1)) + self.wide(wide_x).squeeze(-1)

        h0 = self.in_proj(torch.cat([self.camp(camp), self.linea(lin), num], dim=2))
        idx_b = torch.arange(5, device=camp.device)
        idx_r = torch.arange(5, 10, device=camp.device)

        eb_1, er_1, sb, or_ = self._context(h0, idx_b, idx_r)
        er_2, eb_2, sr, ob = self._context(h0, idx_r, idx_b)

        z_deep = 0.5 * (
            (self.head(torch.cat([eb_1, er_1], dim=1)) - self.head(torch.cat([er_2, eb_2], dim=1))).squeeze(-1)
        )
        ub, ur = torch.tanh(self.u_proj(sb)), torch.tanh(self.u_proj(sr))
        wb, wr = torch.tanh(self.w_proj(ob)), torch.tanh(self.w_proj(or_))
        z_duel = self.scale_duel * (((ub * wr).sum(dim=(1, 2)) - (ur * wb).sum(dim=(1, 2))) / (12.0 ** 0.5))

        e_sym = 0.5 * (torch.cat([eb_1, er_1], dim=1) + torch.cat([er_2, eb_2], dim=1))
        z_lado = self.lado_base + self.lado_lob(e_sym).squeeze(-1)
        return z_wide + 0.5 * z_deep + z_duel + z_lado


def entrenar_attn(D, camp_idx, P_jug, W_pre, d_post_eq, n_camp, tr_raw, te, seeds=(0, 1, 2, 3, 4)):
    ms, t, y, post = D["master"], D["t"], D["y"], D["post"]
    tr = tr_raw[ms[tr_raw]]

    d_pre_jug = P_jug.shape[2] - 12
    mu_jp = P_jug[tr, :, :d_pre_jug].reshape(-1, d_pre_jug).mean(0)
    sd_jp = P_jug[tr, :, :d_pre_jug].reshape(-1, d_pre_jug).std(0) + 1e-6
    post_tr = tr[post[tr]]
    if len(post_tr) > 0:
        mu_jpo = P_jug[post_tr, :, d_pre_jug:-1].reshape(-1, 11).mean(0)
        sd_jpo = P_jug[post_tr, :, d_pre_jug:-1].reshape(-1, 11).std(0) + 1e-6
        mu_wpo, sd_wpo = d_post_eq[post_tr].mean(0), d_post_eq[post_tr].std(0) + 1e-6
    else:
        mu_jpo, sd_jpo = np.zeros(11, np.float32), np.ones(11, np.float32)
        mu_wpo, sd_wpo = np.zeros(d_post_eq.shape[1], np.float32), np.ones(d_post_eq.shape[1], np.float32)
    mu_wp, sd_wp = W_pre[tr].mean(0), W_pre[tr].std(0) + 1e-6

    def build_t(idx):
        p_norm = np.empty_like(P_jug[idx])
        p_norm[..., :d_pre_jug] = np.clip((P_jug[idx, :, :d_pre_jug] - mu_jp) / sd_jp, -8.0, 8.0)
        m_post = post[idx][:, None, None]
        p_norm[..., d_pre_jug:-1] = np.where(m_post, np.clip((P_jug[idx, :, d_pre_jug:-1] - mu_jpo) / sd_jpo, -8.0, 8.0), 0.0)
        p_norm[..., -1] = P_jug[idx, :, -1]
        w_pre_n = np.clip((W_pre[idx] - mu_wp) / sd_wp, -8.0, 8.0)
        w_post_n = np.where(post[idx][:, None], np.clip((d_post_eq[idx] - mu_wpo) / sd_wpo, -8.0, 8.0), 0.0)
        return (
            torch.tensor(camp_idx[idx], dtype=torch.long, device=DEV),
            torch.tensor(p_norm, dtype=torch.float32, device=DEV),
            torch.tensor(np.hstack([w_pre_n, w_post_n]).astype(np.float32), dtype=torch.float32, device=DEV),
        )

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
    y_tr = torch.tensor(y[tr_sub], dtype=torch.float32, device=DEV)
    y_es_np = y[es_sub]
    C_tr, N_tr, W_tr = build_t(tr_sub)
    C_es, N_es, W_es = build_t(es_sub)
    C_te, N_te, W_te = build_t(te)

    preds = []
    for seed in seeds:
        torch.manual_seed(seed)
        net = RedAttnWideDeep(n_camp, P_jug.shape[2], W_tr.shape[1], d_mod=96, n_heads=4, drop=0.28).to(DEV)
        opt = torch.optim.AdamW([
            {"params": list(net.camp_lin.parameters()) + list(net.wide.parameters()), "weight_decay": 1.2e-1, "lr": 1e-3},
            {"params": [p for n, p in net.named_parameters() if not n.startswith(("camp_lin.", "wide."))], "weight_decay": 3e-2, "lr": 6e-4},
        ])
        best_ll, best_sd, pat = 1e9, None, 0
        n_rows = len(tr_sub)
        for _ in range(55):
            net.train()
            perm = torch.randperm(n_rows, device=DEV)
            for b in perm.split(512):
                opt.zero_grad()
                out = net(C_tr[b], N_tr[b], W_tr[b])
                loss = nn.functional.binary_cross_entropy_with_logits(out, y_tr[b])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
                opt.step()
            net.eval()
            with torch.no_grad():
                p_es = []
                for i in range(0, len(es_sub), 4096):
                    p_es.append(torch.sigmoid(net(C_es[i:i + 4096], N_es[i:i + 4096], W_es[i:i + 4096])).cpu().numpy())
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
                p_te.append(torch.sigmoid(net(C_te[i:i + 4096], N_te[i:i + 4096], W_te[i:i + 4096])).cpu().numpy())
            preds.append(np.concatenate(p_te))
    del C_tr, N_tr, W_tr, C_es, N_es, W_es, C_te, N_te, W_te
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
    return np.mean(preds, axis=0)


def catboost_residual_v2(D, MC_raw, ML_raw, A_mmr, F1, COMP, MATCHUPS, REF4, tr_raw, te):
    """CatBoost GPU Residual v2 con variables por línea + min/max de equipo + hora del día + veteranos/sin-estrenos."""
    ms, t, y, post, Xj, mapa = D["master"], D["t"], D["y"], D["post"], D["Xj"], D["mapa"]
    tr = tr_raw[ms[tr_raw]]

    qs = np.quantile(t[tr], np.linspace(0, 1, 7)[1:-1])
    ch = np.searchsorted(qs, t[tr], side="right")
    ref_tr = np.zeros(len(tr), dtype=np.float32)
    draft_tr = np.zeros(len(tr), dtype=np.float32)

    for b in range(1, 6):
        itr, ite = tr[ch < b], tr[ch == b]
        p_ref = ajustar_predecir(D, itr, ite, extra=REF4, extra_post=False, con_mapa=bool(post[itr].any()))
        ref_tr[ch == b] = logit(np.clip(p_ref, 1e-5, 1.0 - 1e-5))
        lr_d = LogisticRegression(C=0.05, max_iter=2000).fit(D["DRAFT"][itr], y[itr])
        draft_tr[ch == b] = logit(np.clip(lr_d.predict_proba(D["DRAFT"][ite])[:, 1], 1e-5, 1.0 - 1e-5))
    p_ref_te = ajustar_predecir(D, tr, te, extra=REF4, extra_post=False, con_mapa=bool(post[tr].any()))
    ref_te = logit(np.clip(p_ref_te, 1e-5, 1.0 - 1e-5)).astype(np.float32)
    lr_dfull = LogisticRegression(C=0.05, max_iter=2000).fit(D["DRAFT"][tr], y[tr])
    draft_te = logit(np.clip(lr_dfull.predict_proba(D["DRAFT"][te])[:, 1], 1e-5, 1.0 - 1e-5)).astype(np.float32)

    cols = {k: i for i, k in enumerate(D["cols"])}
    NM = D["NM"]
    n_prev = np.expm1(Xj[..., cols["logn_tot"]]).reshape(NM, 10)
    est = Xj[..., cols["estreno"]]
    off = Xj[..., cols["offrole"]]
    wra = Xj[..., cols["wr_acum"]]
    lnc = Xj[..., cols["logn_champ"]]
    shl = Xj[..., cols["share_lane"]]

    # Hora del dia en ART (UTC-3): t en ms
    h_art = ((t // 3_600_000) - 3) % 24
    ang = 2.0 * np.pi * h_art / 24.0

    # Resumen por linea clave (dgold15, wr, MC/ML) + min/max
    mc_d = MC_raw[:, 0] - MC_raw[:, 1]  # (NM, 5, 15)
    ml_d = ML_raw[:, 0] - ML_raw[:, 1]
    X_pre = np.c_[
        (Xj[:, 0] - Xj[:, 1]).mean(axis=1),
        (wra[:, 0] - wra[:, 1]),            # 5 wr_acum diffs por linea
        (off[:, 0] - off[:, 1]),            # 5 offrole diffs por linea
        (est[:, 0] - est[:, 1]),            # 5 estreno diffs por linea
        (lnc[:, 0] - lnc[:, 1]),            # 5 logn_champ diffs por linea
        (shl[:, 0] - shl[:, 1]),            # 5 share_lane diffs por linea
        mc_d.mean(axis=1),
        ml_d.mean(axis=1),
        mc_d[:, :, 12],                     # MC d_gold_15 por linea (5)
        ml_d[:, :, 12],                     # ML d_gold_15 por linea (5)
        MC_raw[:, 0].min(axis=1) - MC_raw[:, 1].min(axis=1),
        (A_mmr[:, 0, :, :3] - A_mmr[:, 1, :, :3]).mean(axis=1),
        (A_mmr[:, 0, :, 1] - A_mmr[:, 1, :, 1]),  # nivel_v10 por linea (5)
        MATCHUPS,
        COMP,
        est.reshape(NM, 10).sum(1),
        (est.reshape(NM, 10).sum(1) == 0).astype(np.float32),
        n_prev.min(1),
        off.reshape(NM, 10).sum(1),
        np.isnan(D["Z"]["tnum"]).sum(1),
        (A_mmr[:, 0, :, 0].mean(1) + A_mmr[:, 1, :, 0].mean(1)) * 0.5,
        D["tier_lobby"],
        np.sin(ang),
        np.cos(ang),
    ].astype(np.float32)
    X_post = np.c_[mapa, (F1[:, 0] - F1[:, 1]).mean(axis=1)].astype(np.float32)

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
    return cb.predict_proba(Pool(X_te, baseline=ref_te.astype(np.float64)))[:, 1]


def main():
    t0 = time.time()
    D = cargar()
    NM = D["NM"]
    C = np.load(REPO / "data" / "analysis" / "brainstorm" / "margen_grupos.npz")
    MC_raw, ML_raw = C["MC"], C["ML"]
    A_mmr = np.load(REPO / "data" / "analysis" / "brainstorm" / "mmr_historico.npz")["A"].astype(np.float32)
    NIV_EQ = (A_mmr[:, 0] - A_mmr[:, 1])[..., :3].mean(axis=1).astype(np.float32)
    F1 = cargar_foto1_jugador(D)
    REF4 = extras_referencia(D, escalon=4)
    COMP, J_CAMP = construir_composicion(D)
    mpr_path = REPO / "data" / "analysis" / "brainstorm" / "gemini_mpr.npz"
    MATCHUPS = np.load(mpr_path)["MATCHUPS"].astype(np.float32) if mpr_path.exists() else np.zeros((NM, 0), np.float32)

    camp_idx, P_jug_base, W_pre_base, d_post_eq, n_camp = preparar_datos_red(D, F1, MC_raw, ML_raw)
    P_jug = np.concatenate([A_mmr[..., :3].reshape(NM, 10, -1), J_CAMP.reshape(NM, 10, -1), P_jug_base], axis=-1).astype(np.float32)
    W_pre = np.c_[W_pre_base, NIV_EQ, MATCHUPS, COMP].astype(np.float32)

    g08 = np.load(REPO / "data" / "analysis" / "brainstorm" / "pred_g08.npz")
    p_log4 = (g08["po_log4"], g08["pv_log4"])
    p_red = (g08["po_red"], g08["pv_red"])

    print(f"Evaluando Red AttnWideDeep GPU (5 semillas, {time.time()-t0:.1f}s)...", flush=True)
    p_attn = evaluar(D, lambda tr, te: entrenar_attn(D, camp_idx, P_jug, W_pre, d_post_eq, n_camp, tr, te))

    print(f"Evaluando CatBoost GPU Residual v2 ({time.time()-t0:.1f}s)...", flush=True)
    p_cb2 = evaluar(D, lambda tr, te: catboost_residual_v2(D, MC_raw, ML_raw, A_mmr, F1, COMP, MATCHUPS, REF4, tr, te))

    eps = 1e-6
    def mix(items):
        lo = sum(w * logit(np.clip(p[0], eps, 1 - eps)) for w, p in items)
        lv = sum(w * logit(np.clip(p[1], eps, 1 - eps)) for w, p in items)
        return (expit(lo), expit(lv))

    P = {
        "ref: Escalón 4 (0,6674)": p_log4,
        "g08 Red NeuralAC": p_red,
        "Red AttnWideDeep GPU (5 sem)": p_attn,
        "CatBoost GPU Residual v2": p_cb2,
        "Mix 65/35 Log4 + Attn": mix([(0.65, p_log4), (0.35, p_attn)]),
        "Mix 50/25/25 Log4 + NeuralAC + Attn": mix([(0.50, p_log4), (0.25, p_red), (0.25, p_attn)]),
        "Mix 40/20/20/20 Log4+NAC+Attn+CB2": mix([(0.40, p_log4), (0.20, p_red), (0.20, p_attn), (0.20, p_cb2)]),
    }
    mostrar(comparar(D, P, "ref: Escalón 4 (0,6674)"))
    np.savez_compressed(
        REPO / "data" / "analysis" / "brainstorm" / "pred_g13.npz",
        po_attn=p_attn[0], pv_attn=p_attn[1],
        po_cb2=p_cb2[0], pv_cb2=p_cb2[1],
    )
    print(f"Listo en {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
