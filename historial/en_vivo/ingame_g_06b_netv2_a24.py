"""Paso 3 en A2-A4 (SIN tocar V): InGameNet_v2 (Atención 5x5 simétrica entre carriles + residual limpio + cbias_close).

Propiedad matemática exacta de antisimetría bajo Azul <-> Rojo:
  - Codifica por carril u_br = enc([x_lb, x_lr, e]), u_rb = enc([x_lr, x_lb, e]).
  - Representación antisimétrica h_lane = u_br - u_rb (cambia de signo al permutar lados).
  - Representación simétrica g_lane = u_br + u_rb (invariante al permutar lados).
  - Atención 5x5 multi-cabeza calculada sobre g_lane -> matriz de pesos invariante de lado;
    aplicada a proyecciones lineales sin sesgo de h_lane -> salida exactamente antisimétrica!
  - En modo residual puro (`use_skip=False`), elimina `skip_anti` redundante para que la red
    aprenda exclusivamente corrección no lineal sobre `z_oof(0c)`.

Evalúa SOLO en `A2-A4` (`chunks 4, 5, 6`, sin leer `V` ni `Diamond-V`) contra `5_Blend_LR_MT`
(`pred_mt_a4_gpu_92k.npz`), usando parada temprana cronológica interna 12 % dentro de `tr_b`.

Uso:
    python -u historial/en_vivo/ingame_g_06b_netv2_a24.py
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
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.analysis.ingame_evaluar import (  # noqa: E402
    _construir_aux_targets,
    _fit_logreg_antisym,
    _logit,
    _oof_logreg_logits,
    _sigm,
)
from historial.ensamble_en_vivo.ingame_g_04_mt_a4_gpu import DEV, IN_DIR  # noqa: E402
from historial.en_vivo.ingame_g_06_a24_explore import _blk_ci, _cb_close_mat, _fast_auc, _prep_ext  # noqa: E402
from src.models.ingame import MINUTES, TIMELINE_STATS, extract_minute_features  # noqa: E402

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


class InGameNetV2(nn.Module):
    """InGameNet con atención simétrica 5x5 entre carriles + FiLM + multitarea auxiliar."""

    def __init__(
        self,
        d_lane: int,
        d_anti: int,
        d_sym: int,
        d_aux_anti: int = 0,
        d_aux_sym: int = 0,
        lane_hidden: int = 24,
        n_heads: int = 4,
        trunk_hidden: int = 96,
        dropout: float = 0.25,
        use_skip: bool = True,
    ):
        super().__init__()
        self.use_skip = use_skip
        self.n_heads = n_heads
        self.head_dim = lane_hidden // n_heads
        assert lane_hidden % n_heads == 0

        self.lane_emb = nn.Embedding(5, 8)
        self.lane_enc = nn.Sequential(
            nn.Linear(2 * d_lane + 8, 48),
            nn.LayerNorm(48),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(48, lane_hidden),
            nn.SiLU(),
        )
        # Atención 5x5 sobre el contexto simétrico g_lane (invariante Azul <-> Rojo)
        self.q_proj = nn.Linear(lane_hidden, lane_hidden)
        self.k_proj = nn.Linear(lane_hidden, lane_hidden)
        self.v_anti = nn.Linear(lane_hidden, lane_hidden, bias=False)
        self.attn_out = nn.Linear(lane_hidden, lane_hidden, bias=False)

        # 5 carriles h_lane + 5 context-attended + sum + (max+min) = 12 * lane_hidden
        d_pool = 12 * lane_hidden
        self.anti_proj = nn.Sequential(
            nn.Linear(d_pool + d_anti, trunk_hidden),
            nn.LayerNorm(trunk_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.sym_net = nn.Sequential(
            nn.Linear(d_sym, 48),
            nn.LayerNorm(48),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(48, trunk_hidden),
        )
        self.head_mlp = nn.Sequential(
            nn.Linear(trunk_hidden, 48),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(48, 1, bias=False),
        )
        self.skip_anti = nn.Linear(d_anti, 1, bias=False) if use_skip else None
        self.b_side = nn.Parameter(torch.tensor(0.0))

        self.aux_anti_head = nn.Linear(trunk_hidden, d_aux_anti, bias=False) if d_aux_anti > 0 else None
        self.aux_sym_head = nn.Linear(trunk_hidden, d_aux_sym) if d_aux_sym > 0 else None

    def forward(
        self,
        x_lb: torch.Tensor,
        x_lr: torch.Tensor,
        x_anti: torch.Tensor,
        x_sym: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        B = x_lb.shape[0]
        idx = torch.arange(5, device=x_lb.device).unsqueeze(0).expand(B, 5)
        e = self.lane_emb(idx)
        u_br = self.lane_enc(torch.cat([x_lb, x_lr, e], dim=-1))
        u_rb = self.lane_enc(torch.cat([x_lr, x_lb, e], dim=-1))
        h_lane = u_br - u_rb   # (B, 5, H) antisimétrico
        g_lane = u_br + u_rb   # (B, 5, H) simétrico!

        # Pesos de atención 5x5 invariantes al intercambio de equipos
        q = self.q_proj(g_lane).view(B, 5, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, 5, hd)
        k = self.k_proj(g_lane).view(B, 5, self.n_heads, self.head_dim).transpose(1, 2)
        attn = torch.softmax((q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5), dim=-1)  # (B, nh, 5, 5)

        v = self.v_anti(h_lane).view(B, 5, self.n_heads, self.head_dim).transpose(1, 2)  # exactamente impar en h_lane
        h_ctx = (attn @ v).transpose(1, 2).contiguous().view(B, 5, -1)
        h_ctx = self.attn_out(h_ctx)  # exactamente impar (lineal sin sesgo sobre h_lane)

        h_sum = h_lane.sum(dim=1)
        h_ext = h_lane.max(dim=1).values + h_lane.min(dim=1).values
        h_comb = torch.cat([h_lane.reshape(B, -1), h_ctx.reshape(B, -1), h_sum, h_ext, x_anti], dim=-1)

        z_in = self.anti_proj(h_comb) - self.anti_proj(-h_comb)
        film = 1.0 + 0.5 * torch.tanh(self.sym_net(x_sym))
        h_mod = z_in * film

        delta = (self.head_mlp(h_mod) - self.head_mlp(-h_mod)).squeeze(-1) * 0.5
        if self.skip_anti is not None:
            delta = delta + self.skip_anti(x_anti).squeeze(-1)
        logit = delta + self.b_side
        aux_a = self.aux_anti_head(h_mod) if self.aux_anti_head is not None else None
        aux_s = self.aux_sym_head(film) if self.aux_sym_head is not None else None
        return logit, aux_a, aux_s


def _fit_netv2_gpu(
    minute: int,
    arrs: dict,
    y_tr: np.ndarray,
    aux_tr: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    seed: int = 0,
    bs: int = 512,
    epochs: int = 16,
    init_tr: np.ndarray | None = None,
    init_te: np.ndarray | None = None,
    use_skip: bool = True,
) -> np.ndarray:
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
        lr_val, wd_val = 1.5e-3, 1.0e-2
    else:
        ia_tr = ia_te = None
        lr_val, wd_val = 2.0e-3, 5e-3

    net = InGameNetV2(
        d_lane=Lb_tr.shape[-1],
        d_anti=Xa_tr.shape[1],
        d_sym=Xs_tr.shape[1],
        d_aux_anti=aux_tr[0].shape[1],
        d_aux_sym=aux_tr[2].shape[1],
        lane_hidden=24,
        n_heads=4,
        trunk_hidden=96,
        dropout=0.25,
        use_skip=use_skip,
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

    best_loss, best_state = float("inf"), None
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
    # Verificación rápida de antisimetría exacta antes de correr en GPU
    torch.manual_seed(0)
    tst = InGameNetV2(d_lane=20, d_anti=30, d_sym=12, use_skip=False)
    tst.eval()
    xlb, xlr = torch.randn(16, 5, 20), torch.randn(16, 5, 20)
    xa, xs = torch.randn(16, 30), torch.randn(16, 12)
    with torch.no_grad():
        op = tst(xlb, xlr, xa, xs)[0] - tst.b_side
        on = tst(xlr, xlb, -xa, xs)[0] - tst.b_side
        err = float((op + on).abs().max().item())
    assert err < 1e-5, f"Error de antisimetría: {err}"
    log(f"InGameNetV2 antisym_err={err:.2e} | Iniciando en A2-A4 ONLY ({DEV})...")

    C = np.load(IN_DIR / "ingame_cache.npz", allow_pickle=True)
    t = C["t"].astype(np.int64)
    y = C["y"].astype(np.int64)
    chunk = C["chunk"].astype(np.int32)
    master = C["master"].astype(bool)
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

    z4 = np.load(IN_DIR / "pred_mt_a4_gpu_92k.npz")
    stat_i = {s: i for i, s in enumerate(TIMELINE_STATS)}
    rows = []

    for mi, m in enumerate(MINUTES):
        tm = time.time()
        log(f"\n==================== MINUTO {m} (A2-A4 NetV2 + cb_close) ====================")
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

        p0b = z4[f"pa_{m}_0b_A2_C1"]
        p2mt = z4[f"pa_{m}_2_MT_A4_bs512"]
        p3mt = z4[f"pa_{m}_3_MT_A4C1_bs512"]
        p4mt = z4[f"pa_{m}_4_MT_Res_A4C1_bs512"]
        p5b = z4[f"pa_{m}_5_Blend_LR_MT"]

        # C de 0c y precomputo de cb_close causal + redes NetV2 (solo A2-A4)
        fold_cache = []
        for b, te in folds_a24:
            tr_b = np.flatnonzero(np.isin(chunk, [1, 2, 3, 4, 5, 6]) & (chunk < b) & master & ok_m)
            tr_b = tr_b[np.argsort(t[tr_b], kind="mergesort")]
            ar0b = _prep_ext(f_a2, tr_b, te, L_orig, y, tl_raw, m, cb_tr=None, cb_te=None)
            _, z_te_0b, _ = _fit_logreg_antisym(ar0b["Xa_tr"], y[tr_b], ar0b["Xa_te"], C=0.03)
            z_oof_0b = _oof_logreg_logits(ar0b["Xa_tr"], y[tr_b], t[tr_b], C=0.03, n_splits=4)
            b0_tr = float(np.mean(z_oof_0b))
            cb_tr = _cb_close_mat(tempo["cbias"], g_tot, L_orig, z_oof_0b - b0_tr, tr_b)
            cb_te = _cb_close_mat(tempo["cbias"], g_tot, L_orig, z_te_0b - b0_tr, te)
            fold_cache.append((b, tr_b, te, cb_tr, cb_te))

        best_C, best_ll = 0.03, float("inf")
        for c_try in (0.01, 0.03, 0.05):
            pa_t = np.full(len(y), np.nan, np.float32)
            for b, tr_b, te, cb_tr, cb_te in fold_cache:
                ar0c = _prep_ext(f_a2, tr_b, te, L_orig, y, tl_raw, m, cb_tr=cb_tr, cb_te=cb_te)
                pa_t[te] = _sigm(_fit_logreg_antisym(ar0c["Xa_tr"], y[tr_b], ar0c["Xa_te"], C=c_try)[1])
            ll = log_loss(ya, pa_t[a24_idx])
            if ll < best_ll:
                best_ll, best_C = ll, c_try

        p_0c = np.full(len(y), np.nan, np.float32)
        p_nv2_skip = np.full(len(y), np.nan, np.float32)
        p_nv2_noskip = np.full(len(y), np.nan, np.float32)

        for b, tr_b, te, cb_tr, cb_te in fold_cache:
            ya_b, ma_b, ys_b, ms_b, _ = _construir_aux_targets(m, tr_b, tl_raw, dur_sec, end_gold_diff, end_kills_diff)
            ar0c = _prep_ext(f_a2, tr_b, te, L_orig, y, tl_raw, m, cb_tr=cb_tr, cb_te=cb_te)
            _, z_te_0c, _ = _fit_logreg_antisym(ar0c["Xa_tr"], y[tr_b], ar0c["Xa_te"], C=best_C)
            z_oof_0c = _oof_logreg_logits(ar0c["Xa_tr"], y[tr_b], t[tr_b], C=best_C, n_splits=4)
            p_0c[te] = _sigm(z_te_0c)
            del ar0c

            ar_a4ext = _prep_ext(f_a4, tr_b, te, L_orig, y, tl_raw, m, cb_tr=cb_tr, cb_te=cb_te)
            z_sk = np.mean([
                _fit_netv2_gpu(m, ar_a4ext, y[tr_b], (ya_b, ma_b, ys_b, ms_b), seed=50 * b + s, bs=512, epochs=16, use_skip=True)
                for s in (0, 1)
            ], axis=0)
            z_ns = np.mean([
                _fit_netv2_gpu(m, ar_a4ext, y[tr_b], (ya_b, ma_b, ys_b, ms_b), seed=60 * b + s, bs=512, epochs=16,
                               init_tr=z_oof_0c, init_te=z_te_0c, use_skip=False)
                for s in (0, 1)
            ], axis=0)
            p_nv2_skip[te], p_nv2_noskip[te] = _sigm(z_sk), _sigm(z_ns)
            del ar_a4ext; gc.collect()

        del f_a2, f_a4, fold_cache; gc.collect()

        pa_0c = p_0c[a24_idx]
        pa_nv2_sk = p_nv2_skip[a24_idx]
        pa_nv2_ns = p_nv2_noskip[a24_idx]

        def fwd_blend(cols: list[np.ndarray], label: str) -> np.ndarray:
            out = np.full(len(ya), np.nan, np.float32)
            for bi in range(3):
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
            log(f"  Pesos A2-A4 {label:22s}: {np.round(lr_f.coef_[0], 3)} (b={lr_f.intercept_[0]:+.3f})")
            return out

        b_nv2_sk = fwd_blend([pa_0c, p2mt, p3mt, p4mt, pa_nv2_sk], "Blend_0c+NetV2_Skip")
        b_nv2_ns = fwd_blend([pa_0c, p2mt, p3mt, p4mt, pa_nv2_ns], "Blend_0c+NetV2_ResNoSk")
        b_nv2_both = fwd_blend([pa_0c, p2mt, p3mt, p4mt, pa_nv2_sk, pa_nv2_ns], "Blend_0c+BothNetV2")
        b_pure_v2 = fwd_blend([pa_0c, pa_nv2_sk, pa_nv2_ns], "Blend_Pure_0c+NetV2")

        for nom, pa_k in [
            ("5_Blend_LR_MT (ref)", p5b),
            ("NV2_Skip_A4C1cb", pa_nv2_sk),
            ("NV2_ResNoSkip_0c", pa_nv2_ns),
            ("B1_0c+NetV2_Skip", b_nv2_sk),
            ("B2_0c+NetV2_ResNoSkip", b_nv2_ns),
            ("B3_0c+BothNetV2", b_nv2_both),
            ("B4_Pure_0c+BothNetV2", b_pure_v2),
        ]:
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
        pd.DataFrame(rows).to_csv(IN_DIR / "a24_explore_netv2_g06b.csv", index=False)
        log(f"Minuto {m} terminado en {time.time() - tm:.1f}s")

    log(f"Listo total en {time.time() - T0:.1f}s")


if __name__ == "__main__":
    main()
