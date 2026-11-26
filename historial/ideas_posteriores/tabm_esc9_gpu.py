"""TabM liviano en PyTorch puro (BatchEnsemble k=32 + embeddings PLR de numéricas) sobre Escalón 9.

Arquitectura fiel a TabM (Gorishniy et al., ICLR 2025) sin dependencias externas:
  - Embeddings numéricos PLR (`Periodic -> NLinear por rasgo -> ReLU`, n_bins=16, d_emb=4)
    sobre las ~653 numéricas causales de Escalón 9 (`D["Xd"]` + `extras_referencia(D, 9)`).
  - Rama `DRAFT` dispersa proyectada a 32 dims + skip lineal fuerte.
  - Condicionamiento simétrico al régimen `post` (Foto 1 activa vs ~post pre-04/09).
  - Backbone MLP con capas BatchEnsemble de rango 1 (`k = 32` submodelos, factores r/s
    inicializados en Rademacher {-1, +1}) y cabezas independientes por miembro.
  - Antisimetría exacta respecto a intercambio Azul <-> Rojo:
      logit_k(x, post) = 0.5 * (f_k(x, post) - f_k(-x, post)) + b_side_k
    y pérdida media BCE por submodelo (previene el colapso de cabezas del ensamble).

Evalúa estrictamente causal (`solo Master+`, bloques `A2..A4`, `V Master+`, `Diamond-V`)
como experto tabular adicional sobre `Escalón 9` y `L_ens10` (`pred_g28_esc9.npz`).

Uso:
    python -u historial/ideas_posteriores/tabm_esc9_gpu.py
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
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.analysis.brainstorm_base import cargar, extras_referencia  # noqa: E402
from historial.ensamble_en_vivo.brainstorm_g_22_red_y_catboost_esc6 import BR_DIR, DEV, _ece, _logit, _sigm  # noqa: E402

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def _fast_auc(y: np.ndarray, s: np.ndarray) -> float:
    r = rankdata(s)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


class PeriodicPLREmbeddings(nn.Module):
    """PLR: Periodic (sin/cos 2*pi*c*x) -> NLinear por feature -> ReLU."""

    def __init__(self, n_features: int, n_freqs: int = 12, d_emb: int = 4, sigma: float = 0.1):
        super().__init__()
        self.freq = nn.Parameter(torch.randn(n_features, n_freqs) * sigma)
        # n_bins = 2 * n_freqs
        self.weight = nn.Parameter(torch.empty(n_features, 2 * n_freqs, d_emb))
        self.bias = nn.Parameter(torch.zeros(n_features, d_emb))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        with torch.no_grad():
            self.weight.mul_(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_features) -> (B, n_features * d_emb)
        ang = (2.0 * math.pi) * x.unsqueeze(-1) * self.freq.unsqueeze(0)
        p = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # (B, N, 2*F)
        h = torch.einsum("bni,nio->bno", p, self.weight) + self.bias.unsqueeze(0)
        return F.relu(h).reshape(x.shape[0], -1)


class BatchEnsembleLinear(nn.Module):
    """Capa lineal compartida W con adaptadores rank-1 (r_k, s_k) y bias por miembro k."""

    def __init__(self, in_features: int, out_features: int, k: int = 32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        r_init = torch.randint(0, 2, (k, in_features), dtype=torch.float32) * 2.0 - 1.0
        s_init = torch.randint(0, 2, (k, out_features), dtype=torch.float32) * 2.0 - 1.0
        self.r = nn.Parameter(r_init)
        self.s = nn.Parameter(s_init)
        self.bias = nn.Parameter(torch.zeros(k, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, K, in_features) -> (B, K, out_features)
        xr = x * self.r.unsqueeze(0)
        y = F.linear(xr, self.weight)
        return y * self.s.unsqueeze(0) + self.bias.unsqueeze(0)


class TabMEsc9(nn.Module):
    """TabM + PLR antisimétrico para predicción tabular pre-partida sobre Escalón 9."""

    def __init__(
        self,
        d_num: int,
        d_draft: int,
        k: int = 32,
        n_freqs: int = 12,
        d_emb: int = 4,
        d_draft_emb: int = 32,
        d_hid: int = 192,
        drop: float = 0.35,
    ):
        super().__init__()
        self.k = k
        self.plr = PeriodicPLREmbeddings(d_num, n_freqs=n_freqs, d_emb=d_emb, sigma=0.1)
        self.draft_proj = nn.Linear(d_draft, d_draft_emb, bias=False)
        d_in = d_num * d_emb + d_draft_emb

        # Capa de entrada proyecta a (B, K, d_hid) sin materializar (B, K, d_in) gigante
        self.in_weight = nn.Parameter(torch.empty(d_hid, d_in))
        nn.init.kaiming_uniform_(self.in_weight, a=math.sqrt(5))
        self.in_r = nn.Parameter(torch.randint(0, 2, (k, d_in), dtype=torch.float32) * 2.0 - 1.0)
        self.in_s = nn.Parameter(torch.randint(0, 2, (k, d_hid), dtype=torch.float32) * 2.0 - 1.0)
        self.in_bias = nn.Parameter(torch.zeros(k, d_hid))

        self.post_gamma = nn.Parameter(torch.zeros(2, k, d_hid))
        self.norm1 = nn.LayerNorm(d_hid)
        self.drop1 = nn.Dropout(drop)
        self.block2 = BatchEnsembleLinear(d_hid, d_hid, k=k)
        self.norm2 = nn.LayerNorm(d_hid)
        self.drop2 = nn.Dropout(drop)

        self.head_w = nn.Parameter(torch.randn(k, d_hid) * (1.0 / math.sqrt(d_hid)))
        self.skip_draft = nn.Linear(d_draft, 1, bias=False)
        self.skip_num = nn.Linear(d_num, 1, bias=False)
        self.b_side = nn.Parameter(torch.zeros(k))

    def _raw_branch(self, x_num: torch.Tensor, x_draft: torch.Tensor, post_idx: torch.Tensor) -> torch.Tensor:
        e_num = self.plr(x_num)
        e_dr = self.draft_proj(x_draft)
        z = torch.cat([e_num, e_dr], dim=-1)  # (B, d_in)

        # Equivale exactamente a BatchEnsembleLinear sin allocar tensor 3D sobre d_in:
        # out[b, k, :] = ((z[b] * in_r[k]) @ in_weight.T) * in_s[k] + in_bias[k]
        zr = torch.einsum("bd,kd->bkd", z, self.in_r)
        h = torch.einsum("bkd,hd->bkh", zr, self.in_weight)
        h = h * self.in_s.unsqueeze(0) + self.in_bias.unsqueeze(0)
        # Regulación simétrica por régimen post (0 = pre-Foto1, 1 = post-Foto1)
        h = h * (1.0 + 0.5 * torch.tanh(self.post_gamma[post_idx]))
        h = self.drop1(F.silu(self.norm1(h)))
        h = h + self.drop2(F.silu(self.norm2(self.block2(h))))
        out_k = (h * self.head_w.unsqueeze(0)).sum(dim=-1)
        skip = (self.skip_draft(x_draft) + self.skip_num(x_num))  # (B, 1)
        return out_k + skip

    def forward(self, x_num: torch.Tensor, x_draft: torch.Tensor, post_idx: torch.Tensor) -> torch.Tensor:
        """Devuelve logits `(B, K)` estrictamente antisimétricos bajo `x -> -x`."""
        x_n2 = torch.cat([x_num, -x_num], dim=0)
        x_d2 = torch.cat([x_draft, -x_draft], dim=0)
        p_2 = torch.cat([post_idx, post_idx], dim=0)
        raw = self._raw_branch(x_n2, x_d2, p_2)
        b = x_num.shape[0]
        return 0.5 * (raw[:b] - raw[b:]) + self.b_side.unsqueeze(0)


def _norm_fit_transform(
    X_pre: np.ndarray,
    X_post: np.ndarray,
    post: np.ndarray,
    tr_sub: np.ndarray,
    idx_list: list[np.ndarray],
) -> list[np.ndarray]:
    """Estandariza pre (sobre `tr_sub`) y post (solo `tr_sub & post`, 0 en `~post`), clip [-6, 6]."""
    mu_pre = X_pre[tr_sub].mean(0)
    sd_pre = X_pre[tr_sub].std(0) + 1e-6
    po_m = tr_sub[post[tr_sub]]
    if len(po_m) >= 50:
        mu_po = X_post[po_m].mean(0)
        sd_po = X_post[po_m].std(0) + 1e-6
    else:
        mu_po = np.zeros(X_post.shape[1], dtype=np.float32)
        sd_po = np.ones(X_post.shape[1], dtype=np.float32)

    out = []
    for idx in idx_list:
        pre_n = np.clip((X_pre[idx] - mu_pre) / sd_pre, -6.0, 6.0).astype(np.float32)
        po_n = np.where(
            post[idx][:, None],
            np.clip((X_post[idx] - mu_po) / sd_po, -6.0, 6.0),
            0.0,
        ).astype(np.float32)
        out.append(np.hstack([pre_n, po_n]).astype(np.float32))
    return out


def _fit_predict_tabm_fold(
    D: dict,
    X_pre: np.ndarray,
    X_post: np.ndarray,
    tr: np.ndarray,
    te: np.ndarray,
    dv_idx: np.ndarray | None = None,
    seeds: tuple[int, ...] = (11, 22),
    bs: int = 384,
    max_epochs: int = 30,
    patience: int = 6,
) -> tuple[np.ndarray, np.ndarray | None]:
    t, y, post, DRAFT = D["t"], D["y"], D["post"], D["DRAFT"]

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
    idx_eval = [tr_sub, es_sub, te] + ([dv_idx] if dv_idx is not None else [])
    mats = _norm_fit_transform(X_pre, X_post, post, tr_sub, idx_eval)

    def to_dev_tensors(idx, mat):
        # DRAFT entra densificado por bloques pequeños al GPU (tr_sub ~110k * 850 float32 = ~370 MB)
        dr = torch.tensor(DRAFT[idx].toarray(), dtype=torch.float32, device=DEV)
        xn = torch.tensor(mat, dtype=torch.float32, device=DEV)
        pi = torch.tensor(post[idx].astype(np.int64), dtype=torch.long, device=DEV)
        return xn, dr, pi

    Xn_tr, Dr_tr, Pi_tr = to_dev_tensors(tr_sub, mats[0])
    Xn_es, Dr_es, Pi_es = to_dev_tensors(es_sub, mats[1])
    Xn_te, Dr_te, Pi_te = to_dev_tensors(te, mats[2])
    y_tr_t = torch.tensor(y[tr_sub], dtype=torch.float32, device=DEV)
    y_es_np = y[es_sub]
    if dv_idx is not None:
        Xn_dv, Dr_dv, Pi_dv = to_dev_tensors(dv_idx, mats[3])

    d_num = Xn_tr.shape[1]
    d_draft = Dr_tr.shape[1]
    n_rows = len(tr_sub)

    probs_te = np.zeros(len(te), dtype=np.float64)
    probs_dv = np.zeros(len(dv_idx), dtype=np.float64) if dv_idx is not None else None

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        net = TabMEsc9(d_num=d_num, d_draft=d_draft, k=32, n_freqs=12, d_emb=4, d_draft_emb=32, d_hid=192, drop=0.35).to(DEV)
        skip_ids = {id(p) for p in list(net.skip_draft.parameters()) + list(net.skip_num.parameters()) + list(net.draft_proj.parameters())}
        no_decay_suffixes = (".freq", ".r", ".s", "in_r", "in_s", "bias", "norm1.weight", "norm2.weight", "post_gamma", "b_side")
        decay_p, nodecay_p, skip_p = [], [], []
        for pname, p in net.named_parameters():
            if not p.requires_grad:
                continue
            if id(p) in skip_ids:
                skip_p.append(p)
            elif any(pname.endswith(suf) for suf in no_decay_suffixes):
                nodecay_p.append(p)
            else:
                decay_p.append(p)
        opt = torch.optim.AdamW([
            {"params": skip_p, "lr": 1.2e-3, "weight_decay": 5e-2},
            {"params": decay_p, "lr": 1.5e-3, "weight_decay": 3e-2},
            {"params": nodecay_p, "lr": 1.5e-3, "weight_decay": 0.0},
        ])
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-4)
        best_ll, best_sd, best_ep, pat = 1e9, None, 0, 0
        for ep in range(max_epochs):
            net.train()
            perm = torch.randperm(n_rows, device=DEV)
            for b in perm.split(bs):
                opt.zero_grad(set_to_none=True)
                logits_k = net(Xn_tr[b], Dr_tr[b], Pi_tr[b])  # (B, K)
                targets = y_tr_t[b].unsqueeze(1).expand_as(logits_k)
                loss = F.binary_cross_entropy_with_logits(logits_k, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
                opt.step()
            sch.step()

            net.eval()
            with torch.no_grad():
                pe = []
                for i in range(0, len(es_sub), 1024):
                    lk = net(Xn_es[i : i + 1024], Dr_es[i : i + 1024], Pi_es[i : i + 1024])
                    pe.append(torch.sigmoid(lk).mean(dim=1).cpu().numpy())
                p_es = np.concatenate(pe)
                ll = log_loss(y_es_np[es_eval_mask], np.clip(p_es[es_eval_mask], 1e-6, 1.0 - 1e-6))
            if ll < best_ll - 1e-5:
                best_ll = ll
                best_sd = {k_: v.detach().cpu().clone() for k_, v in net.state_dict().items()}
                best_ep = ep
                pat = 0
            else:
                pat += 1
                if pat >= patience:
                    break
        log(f"    [seed={seed}] best_es_ll={best_ll:.5f} (ep {best_ep + 1})")

        net.load_state_dict({k_: v.to(DEV) for k_, v in best_sd.items()})
        net.eval()
        with torch.no_grad():
            pt = []
            for i in range(0, len(te), 1024):
                lk = net(Xn_te[i : i + 1024], Dr_te[i : i + 1024], Pi_te[i : i + 1024])
                pt.append(torch.sigmoid(lk).mean(dim=1).cpu().numpy())
            probs_te += np.concatenate(pt)
            if dv_idx is not None:
                pd_ = []
                for i in range(0, len(dv_idx), 1024):
                    lk = net(Xn_dv[i : i + 1024], Dr_dv[i : i + 1024], Pi_dv[i : i + 1024])
                    pd_.append(torch.sigmoid(lk).mean(dim=1).cpu().numpy())
                probs_dv += np.concatenate(pd_)
        del net, opt, best_sd
        torch.cuda.empty_cache()

    del Xn_tr, Dr_tr, Pi_tr, Xn_es, Dr_es, Pi_es, Xn_te, Dr_te, Pi_te, y_tr_t
    if dv_idx is not None:
        del Xn_dv, Dr_dv, Pi_dv
    torch.cuda.empty_cache()

    probs_te = (probs_te / len(seeds)).astype(np.float32)
    if probs_dv is not None:
        probs_dv = (probs_dv / len(seeds)).astype(np.float32)
    return probs_te, probs_dv


def main() -> None:
    log(f"Iniciando TabM PyTorch puro (k=32 + PLR) en {DEV}...")
    D = cargar()
    D.pop("Xj", None)
    gc.collect()

    REF9 = extras_referencia(D, escalon=9)
    # Separar pre (sin cols 153:157 FOTO1_DIN) y post (cols 153:157) igual que brainstorm_g_28
    X_pre = np.c_[D["Xd"], REF9[:, :153], REF9[:, 157:]].astype(np.float32)
    X_post = REF9[:, 153:157].astype(np.float32)
    D.pop("Xd", None)
    D.pop("_ref", None)
    del REF9
    gc.collect()
    np.nan_to_num(X_pre, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(X_post, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    log(f"TabM inputs: X_pre={X_pre.shape}, X_post={X_post.shape}, DRAFT={D['DRAFT'].shape}")

    NM = D["NM"]
    fi, bl, ms, y = D["fi"], D["bloque"], D["master"], D["y"]
    po_tabm = np.full(NM, np.nan, dtype=np.float32)

    for b in D["EV"]:
        tf = time.time()
        tr_b = fi[bl[fi] < b]
        tr_b = tr_b[ms[tr_b]]
        te_b = np.flatnonzero(bl == b)
        p_te, _ = _fit_predict_tabm_fold(D, X_pre, X_post, tr_b, te_b)
        po_tabm[te_b] = p_te
        m_te = te_b[ms[te_b]]
        log(f"  [fold A{b + 1}] AUC={_fast_auc(y[m_te], po_tabm[m_te]):.4f} ({time.time() - tf:.1f}s)")

    tf = time.time()
    tr_v = D["fi_v"][ms[D["fi_v"]]]
    vi = np.flatnonzero(D["val"])
    di = np.flatnonzero(D["diam_v"])
    pv_tabm, pdv_tabm = _fit_predict_tabm_fold(D, X_pre, X_post, tr_v, vi, dv_idx=di)
    log(f"  [fold V+Diam] V AUC={_fast_auc(y[vi], pv_tabm):.4f} | DiamV={_fast_auc(y[di], pdv_tabm):.4f} ({time.time() - tf:.1f}s)")

    out_npz = BR_DIR / "pred_tabm_esc9.npz"
    np.savez_compressed(out_npz, po_tabm=po_tabm, pv_tabm=pv_tabm, pdv_tabm=pdv_tabm)
    log(f"[+] Guardado {out_npz}")

    # Evaluación completa vs Escalón 9 oficial y L_ens10 (`pred_g28_esc9.npz`)
    g28 = np.load(BR_DIR / "pred_g28_esc9.npz")
    po_r9, pv_r9, pdv_r9 = g28["po_ref9"], g28["pv_ref9"], g28["p_dv_ref9"]
    # L_ens10 oficial = Mix 50/25/25 Ref9+NAC+CB (§9.S.52 / §9.S.55)
    lo_e10 = 0.50 * _logit(po_r9) + 0.25 * _logit(g28["po_nac"]) + 0.25 * _logit(g28["po_cb"])
    lv_e10 = 0.50 * _logit(pv_r9) + 0.25 * _logit(g28["pv_nac"]) + 0.25 * _logit(g28["pv_cb"])
    ld_e10 = 0.50 * _logit(pdv_r9) + 0.25 * _logit(g28["p_dv_nac"]) + 0.25 * _logit(g28["p_dv_cb"])
    po_e10, pv_e10, pdv_e10 = _sigm(lo_e10), _sigm(lv_e10), _sigm(ld_e10)

    # Stacks OOF hacia adelante (A2 = media 0.5/0.5 o 0.8/0.2; A3 <- A2; A4 <- A2-A3; V/DV <- A2-A4) sin mirar V
    def fwd_stack(cols_o: list[np.ndarray], cols_v: list[np.ndarray], cols_dv: list[np.ndarray], label: str):
        po_s = np.full(NM, np.nan, dtype=np.float32)
        for bi, b in enumerate(D["EV"]):
            te_m = np.flatnonzero((bl == b) & ms)
            z_te = np.column_stack([_logit(c[te_m]) for c in cols_o])
            if bi == 0:
                po_s[te_m] = _sigm(z_te.mean(axis=1))
            else:
                tr_prev = np.flatnonzero(np.isin(bl, D["EV"][:bi]) & ms)
                z_pr = np.column_stack([_logit(c[tr_prev]) for c in cols_o])
                lr = LogisticRegression(C=0.5, max_iter=1000).fit(z_pr, y[tr_prev])
                po_s[te_m] = lr.predict_proba(z_te)[:, 1].astype(np.float32)
        a24_m = np.flatnonzero(np.isin(bl, D["EV"]) & ms)
        lr_f = LogisticRegression(C=0.5, max_iter=1000).fit(np.column_stack([_logit(c[a24_m]) for c in cols_o]), y[a24_m])
        log(f"  Stack {label} pesos A2-A4: coef={np.round(lr_f.coef_[0], 4)}, b={lr_f.intercept_[0]:+.4f}")
        pv_s = lr_f.predict_proba(np.column_stack([_logit(c) for c in cols_v]))[:, 1].astype(np.float32)
        pdv_s = lr_f.predict_proba(np.column_stack([_logit(c) for c in cols_dv]))[:, 1].astype(np.float32)
        return po_s, pv_s, pdv_s

    po_s1, pv_s1, pdv_s1 = fwd_stack([po_r9, po_tabm], [pv_r9, pv_tabm], [pdv_r9, pdv_tabm], "Ref9+TabM")
    po_m75, pv_m75, pdv_m75 = (
        _sigm(0.75 * _logit(po_r9) + 0.25 * _logit(po_tabm)),
        _sigm(0.75 * _logit(pv_r9) + 0.25 * _logit(pv_tabm)),
        _sigm(0.75 * _logit(pdv_r9) + 0.25 * _logit(pdv_tabm)),
    )
    po_s2, pv_s2, pdv_s2 = fwd_stack([po_e10, po_tabm], [pv_e10, pv_tabm], [pdv_e10, pdv_tabm], "Ens10+TabM")
    po_s3, pv_s3, pdv_s3 = fwd_stack(
        [po_r9, g28["po_nac"], g28["po_cb"], po_tabm],
        [pv_r9, g28["pv_nac"], g28["pv_cb"], pv_tabm],
        [pdv_r9, g28["p_dv_nac"], g28["p_dv_cb"], pdv_tabm],
        "Ref9+NAC+CB+TabM",
    )
    po_s4, pv_s4, pdv_s4 = fwd_stack(
        [po_r9, g28["po_nac"], g28["po_att"], g28["po_cb"], po_tabm],
        [pv_r9, g28["pv_nac"], g28["pv_att"], g28["pv_cb"], pv_tabm],
        [pdv_r9, g28["p_dv_nac"], g28["p_dv_att"], g28["p_dv_cb"], pdv_tabm],
        "5Exp_ConAtt+TabM",
    )
    po_e45, pv_e45, pdv_e45 = (
        _sigm(0.45 * _logit(po_r9) + 0.25 * _logit(g28["po_nac"]) + 0.15 * _logit(g28["po_att"]) + 0.15 * _logit(g28["po_cb"])),
        _sigm(0.45 * _logit(pv_r9) + 0.25 * _logit(g28["pv_nac"]) + 0.15 * _logit(g28["pv_att"]) + 0.15 * _logit(g28["pv_cb"])),
        _sigm(0.45 * _logit(pdv_r9) + 0.25 * _logit(g28["p_dv_nac"]) + 0.15 * _logit(g28["p_dv_att"]) + 0.15 * _logit(g28["p_dv_cb"])),
    )

    yv, ydv = y[vi], y[di]
    a24_masks = [np.flatnonzero((bl == b) & ms) for b in D["EV"]]
    boots_b = [np.random.default_rng(100 + b).integers(0, len(m_), (1000, len(m_))) for b, m_ in zip(D["EV"], a24_masks)]
    boot_v = np.random.default_rng(1).integers(0, len(yv), (1000, len(yv)))
    boot_dv = np.random.default_rng(2).integers(0, len(ydv), (1000, len(ydv)))

    def cov_acc(y_arr, p_arr, frac):
        k_n = max(1, int(round(frac * len(y_arr))))
        ord_i = np.argsort(-np.abs(p_arr - 0.5))[:k_n]
        return float(((p_arr[ord_i] > 0.5) == y_arr[ord_i]).mean())

    def eval_pair(po1, pv1, pd1, po0, pv0, pd0):
        ds_b = [np.array([_fast_auc(y[m_[s]], po1[m_[s]]) - _fast_auc(y[m_[s]], po0[m_[s]]) for s in bt]) for m_, bt in zip(a24_masks, boots_b)]
        dm_b = np.mean(ds_b, axis=0)
        b_pt = float(np.mean([_fast_auc(y[m_], po1[m_]) - _fast_auc(y[m_], po0[m_]) for m_ in a24_masks]))
        dv_ = np.array([_fast_auc(yv[s], pv1[s]) - _fast_auc(yv[s], pv0[s]) for s in boot_v])
        dd_ = np.array([_fast_auc(ydv[s], pd1[s]) - _fast_auc(ydv[s], pd0[s]) for s in boot_dv])
        return (
            (b_pt, float(np.percentile(dm_b, 2.5)), float(np.percentile(dm_b, 97.5))),
            (float(_fast_auc(yv, pv1) - _fast_auc(yv, pv0)), float(np.percentile(dv_, 2.5)), float(np.percentile(dv_, 97.5))),
            (float(_fast_auc(ydv, pd1) - _fast_auc(ydv, pd0)), float(np.percentile(dd_, 2.5)), float(np.percentile(dd_, 97.5))),
        )

    cands = [
        ("0_Ref9_Oficial", po_r9, pv_r9, pdv_r9),
        ("1_L_Ens10_50_25_25", po_e10, pv_e10, pdv_e10),
        ("1b_L_Ens10_45_25_15_15", po_e45, pv_e45, pdv_e45),
        ("2_TabM_Solo", po_tabm, pv_tabm, pdv_tabm),
        ("3_Mix75_25_Ref9_TabM", po_m75, pv_m75, pdv_m75),
        ("4_Stack_Ref9_TabM", po_s1, pv_s1, pdv_s1),
        ("5_Stack_Ens10_TabM", po_s2, pv_s2, pdv_s2),
        ("6_Stack_4Exp_TabM", po_s3, pv_s3, pdv_s3),
        ("7_Stack_5Exp_TabM", po_s4, pv_s4, pdv_s4),
    ]
    rows = []
    log("\n--- Resumen TabM sobre Escalón 9 / 10 ---")
    for nom, po_k, pv_k, pd_k in cands:
        b_vs_r9, v_vs_r9, d_vs_r9 = eval_pair(po_k, pv_k, pd_k, po_r9, pv_r9, pdv_r9)
        b_vs_e10, v_vs_e10, d_vs_e10 = eval_pair(po_k, pv_k, pd_k, po_e10, pv_e10, pdv_e10)
        aucs_b = [_fast_auc(y[m_], po_k[m_]) for m_ in a24_masks]
        auc_v = _fast_auc(yv, pv_k)
        acc_v = float(((pv_k > 0.5) == yv).mean())
        ll_v = float(log_loss(yv, np.clip(pv_k, 1e-6, 1.0 - 1e-6)))
        ece_v = _ece(yv, pv_k)
        c5, c10, c20 = cov_acc(yv, pv_k, 0.05), cov_acc(yv, pv_k, 0.10), cov_acc(yv, pv_k, 0.20)
        auc_dv = _fast_auc(ydv, pd_k)
        acc_dv = float(((pd_k > 0.5) == ydv).mean())
        print(
            f"  {nom:23s} | A2/3/4 {aucs_b[0]:.4f}/{aucs_b[1]:.4f}/{aucs_b[2]:.4f} | "
            f"V {auc_v:.4f}/{acc_v:.2%} LL {ll_v:.4f} ECE {ece_v:.4f} [{c5:.1%}/{c10:.1%}/{c20:.1%}] | "
            f"vsRef9 Bloq {b_vs_r9[0]:+.4f} [{b_vs_r9[1]:+.4f};{b_vs_r9[2]:+.4f}] V {v_vs_r9[0]:+.4f} [{v_vs_r9[1]:+.4f};{v_vs_r9[2]:+.4f}] | "
            f"vsEns10 Bloq {b_vs_e10[0]:+.4f} [{b_vs_e10[1]:+.4f};{b_vs_e10[2]:+.4f}] V {v_vs_e10[0]:+.4f} [{v_vs_e10[1]:+.4f};{v_vs_e10[2]:+.4f}] | "
            f"DiamV {auc_dv:.4f}/{acc_dv:.2%} (vsR9 {d_vs_r9[0]:+.4f} [{d_vs_r9[1]:+.4f};{d_vs_r9[2]:+.4f}] | vsE10 {d_vs_e10[0]:+.4f} [{d_vs_e10[1]:+.4f};{d_vs_e10[2]:+.4f}])",
            flush=True,
        )
        rows.append({
            "modelo": nom, "AUC_A2": aucs_b[0], "AUC_A3": aucs_b[1], "AUC_A4": aucs_b[2],
            "AUC_V": auc_v, "Acc_V": acc_v, "LogLoss_V": ll_v, "ECE_V": ece_v,
            "cov5": c5, "cov10": c10, "cov20": c20,
            "dBloq_vs_Ref9": b_vs_r9[0], "lo_Bloq_Ref9": b_vs_r9[1], "hi_Bloq_Ref9": b_vs_r9[2],
            "dV_vs_Ref9": v_vs_r9[0], "lo_V_Ref9": v_vs_r9[1], "hi_V_Ref9": v_vs_r9[2],
            "dBloq_vs_Ens10": b_vs_e10[0], "lo_Bloq_Ens10": b_vs_e10[1], "hi_Bloq_Ens10": b_vs_e10[2],
            "dV_vs_Ens10": v_vs_e10[0], "lo_V_Ens10": v_vs_e10[1], "hi_V_Ens10": v_vs_e10[2],
            "AUC_DiamV": auc_dv, "Acc_DiamV": acc_dv,
            "dDV_vs_Ref9": d_vs_r9[0], "lo_DV_Ref9": d_vs_r9[1], "hi_DV_Ref9": d_vs_r9[2],
            "dDV_vs_Ens10": d_vs_e10[0], "lo_DV_Ens10": d_vs_e10[1], "hi_DV_Ens10": d_vs_e10[2],
        })
    pd.DataFrame(rows).to_csv(BR_DIR / "tabm_esc9.csv", index=False)
    log(f"Listo total en {time.time() - T0:.1f}s")


if __name__ == "__main__":
    main()
