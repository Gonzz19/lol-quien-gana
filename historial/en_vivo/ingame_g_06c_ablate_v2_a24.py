"""Ablación limpia en A2-A4 (Min 15 y 20, SIN tocar V):

Aísla exactamente qué aporta cada pieza de `g_06b`:
  - `A_M1_ResNoSkip_0c`: InGameNet v1 residual SIN `skip_anti` redundante + CosineAnnealing sobre `0c`
  - `B_NV2_ResNoSkip_0c`: InGameNetV2 (Atención 5x5 + SIN `skip_anti`) sobre `0c`
  - `C1_4M_NoAtt`: blend de 4 miembros `[0c, 2_MT, 3_MTC1, A_M1_ResNoSkip_0c]`
  - `C2_4M_NV2`:   blend de 4 miembros `[0c, 2_MT, 3_MTC1, B_NV2_ResNoSkip_0c]`
  - `C3_5M_NV2`:   blend de 5 miembros `[0c, 2_MT, 3_MTC1, 4_MTRes, B_NV2_ResNoSkip_0c]`
  + Etapa 2 forward (`+ d_ens10`) comparado contra `6_Final_dEns10` en `A2-A4`.

Uso:
    python -u historial/en_vivo/ingame_g_06c_ablate_v2_a24.py
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
from historial.ensamble_en_vivo.ingame_g_01_prior_y_gpu import _cargar_priors  # noqa: E402
from historial.ensamble_en_vivo.ingame_g_04_mt_a4_gpu import DEV, IN_DIR  # noqa: E402
from historial.en_vivo.ingame_g_06_a24_explore import _blk_ci, _cb_close_mat, _fast_auc, _prep_ext  # noqa: E402
from historial.en_vivo.ingame_g_06b_netv2_a24 import _fit_netv2_gpu  # noqa: E402
from src.models.ingame import TIMELINE_STATS, extract_minute_features  # noqa: E402

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


class InGameNetV1NoSkip(nn.Module):
    """InGameNet v1 exacto (sin atención 5x5) pero SIN skip_anti para uso residual puro."""

    def __init__(self, d_lane: int, d_anti: int, d_sym: int, d_aux_anti: int = 0, d_aux_sym: int = 0):
        super().__init__()
        self.lane_emb = nn.Embedding(5, 8)
        self.lane_enc = nn.Sequential(
            nn.Linear(2 * d_lane + 8, 48), nn.LayerNorm(48), nn.SiLU(), nn.Dropout(0.25),
            nn.Linear(48, 24), nn.SiLU(),
        )
        self.anti_proj = nn.Sequential(
            nn.Linear(7 * 24 + d_anti, 96), nn.LayerNorm(96), nn.SiLU(), nn.Dropout(0.25),
        )
        self.sym_net = nn.Sequential(
            nn.Linear(d_sym, 48), nn.LayerNorm(48), nn.SiLU(), nn.Dropout(0.25), nn.Linear(48, 96),
        )
        self.head_mlp = nn.Sequential(nn.Linear(96, 48), nn.SiLU(), nn.Dropout(0.25), nn.Linear(48, 1, bias=False))
        self.b_side = nn.Parameter(torch.tensor(0.0))
        self.aux_anti_head = nn.Linear(96, d_aux_anti, bias=False) if d_aux_anti > 0 else None
        self.aux_sym_head = nn.Linear(96, d_aux_sym) if d_aux_sym > 0 else None

    def forward(self, x_lb, x_lr, x_anti, x_sym):
        B = x_lb.shape[0]
        e = self.lane_emb(torch.arange(5, device=x_lb.device).unsqueeze(0).expand(B, 5))
        h_lane = self.lane_enc(torch.cat([x_lb, x_lr, e], dim=-1)) - self.lane_enc(torch.cat([x_lr, x_lb, e], dim=-1))
        h_comb = torch.cat([h_lane.reshape(B, -1), h_lane.sum(1), h_lane.max(1).values + h_lane.min(1).values, x_anti], dim=-1)
        z_in = self.anti_proj(h_comb) - self.anti_proj(-h_comb)
        film = 1.0 + 0.5 * torch.tanh(self.sym_net(x_sym))
        h_mod = z_in * film
        delta = (self.head_mlp(h_mod) - self.head_mlp(-h_mod)).squeeze(-1) * 0.5 + self.b_side
        return delta, (self.aux_anti_head(h_mod) if self.aux_anti_head else None), (self.aux_sym_head(film) if self.aux_sym_head else None)


def _fit_v1_noskip_gpu(minute: int, arrs: dict, y_tr: np.ndarray, aux_tr, seed: int, init_tr: np.ndarray, init_te: np.ndarray) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    Lb_tr, Lr_tr, Xa_tr, Xs_tr = arrs["Lb_tr"], arrs["Lr_tr"], arrs["Xa_tr"], arrs["Xs_tr"]
    Lb_te, Lr_te, Xa_te, Xs_te = arrs["Lb_te"], arrs["Lr_te"], arrs["Xa_te"], arrs["Xs_te"]
    n = len(y_tr)
    cut = int(n * 0.88)
    tr_s, va_s = np.arange(cut), np.arange(cut, n)
    b0 = float(np.mean(init_tr))
    ia_tr = (init_tr - b0).astype(np.float32)
    ia_te = (init_te - b0).astype(np.float32)

    net = InGameNetV1NoSkip(Lb_tr.shape[-1], Xa_tr.shape[1], Xs_tr.shape[1], aux_tr[0].shape[1], aux_tr[2].shape[1]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1.5e-3, weight_decay=1.0e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=16)

    t_lb, t_lr = torch.tensor(Lb_tr[tr_s], device=DEV), torch.tensor(Lr_tr[tr_s], device=DEV)
    t_xa, t_xs = torch.tensor(Xa_tr[tr_s], device=DEV), torch.tensor(Xs_tr[tr_s], device=DEV)
    t_y = torch.tensor(y_tr[tr_s], dtype=torch.float32, device=DEV)
    t_ya, t_ma = torch.tensor(aux_tr[0][tr_s], device=DEV), torch.tensor(aux_tr[1][tr_s], dtype=torch.float32, device=DEV)
    t_ys, t_ms = torch.tensor(aux_tr[2][tr_s], device=DEV), torch.tensor(aux_tr[3][tr_s], dtype=torch.float32, device=DEV)
    t_z0 = torch.tensor(ia_tr[tr_s], device=DEV)
    v_lb, v_lr = torch.tensor(Lb_tr[va_s], device=DEV), torch.tensor(Lr_tr[va_s], device=DEV)
    v_xa, v_xs = torch.tensor(Xa_tr[va_s], device=DEV), torch.tensor(Xs_tr[va_s], device=DEV)
    v_y, v_z0 = torch.tensor(y_tr[va_s], dtype=torch.float32, device=DEV), torch.tensor(ia_tr[va_s], device=DEV)

    best_loss, best_state = float("inf"), None
    rng = np.random.default_rng(seed)
    for _ in range(16):
        net.train()
        perm = rng.permutation(len(tr_s))
        for b in range(0, len(perm), 512):
            ii = perm[b : b + 512]
            d_log, oa, os_ = net(t_lb[ii], t_lr[ii], t_xa[ii], t_xs[ii])
            loss = F.binary_cross_entropy_with_logits(t_z0[ii] + d_log, t_y[ii])
            la = (F.smooth_l1_loss(oa, t_ya[ii], reduction="none") * t_ma[ii]).sum() / torch.clamp(t_ma[ii].sum(), min=1.0)
            ls = ((F.smooth_l1_loss(os_, t_ys[ii], reduction="none") * t_ms[ii]).sum() / torch.clamp(t_ms[ii].sum(), min=1.0)
                  if minute == 20 else F.binary_cross_entropy_with_logits(os_, t_ys[ii]))
            loss = loss + 0.18 * la + 0.08 * ls
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0); opt.step()
        sch.step()
        net.eval()
        with torch.no_grad():
            vloss = float(F.binary_cross_entropy_with_logits(net(v_lb, v_lr, v_xa, v_xs)[0] + v_z0, v_y).item())
        if vloss < best_loss:
            best_loss = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    net.load_state_dict({k: v.to(DEV) for k, v in best_state.items()})
    net.eval()
    preds = []
    with torch.no_grad():
        for b in range(0, len(Xa_te), 4096):
            lb_ = net(torch.tensor(Lb_te[b:b+4096], device=DEV), torch.tensor(Lr_te[b:b+4096], device=DEV),
                      torch.tensor(Xa_te[b:b+4096], device=DEV), torch.tensor(Xs_te[b:b+4096], device=DEV))[0].cpu().numpy()
            preds.append(lb_ + ia_te[b:b+4096])
    del t_lb, t_lr, t_xa, t_xs, t_y, t_ya, t_ma, t_ys, t_ms, t_z0, v_lb, v_lr, v_xa, v_xs, v_y, v_z0, net, opt, sch, best_state
    torch.cuda.empty_cache()
    return np.concatenate(preds).astype(np.float32)


def main() -> None:
    log(f"Iniciando ablación g_06c en A2-A4 ONLY (Min 15 y 20, {DEV})...")
    C = np.load(IN_DIR / "ingame_cache.npz", allow_pickle=True)
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
        "comp_anti": C["comp_anti"].astype(np.float32), "names_anti": list(C["comp_names_anti"]),
        "comp_sym": C["comp_sym"].astype(np.float32), "names_sym": list(C["comp_names_sym"]),
        "vuln_blue": C["vuln_blue"].astype(np.float32), "vuln_red": C["vuln_red"].astype(np.float32),
        "vuln_names": list(C["vuln_names"]), "slot_static": C["slot_static"].astype(np.float32),
    }
    tempo = {k: C[k].astype(np.float32) for k in ("cbias", "cslope", "pbias", "pvol")}
    player_pre = {"p_mat": C["p_mat"].astype(np.float32), "p_cols": list(C["p_cols"])}
    del C
    gc.collect()

    priors = _cargar_priors(mids, bloque, val, diam_v, L_orig)
    d_ens10 = np.where(np.isfinite(priors["L_ens10"]) & np.isfinite(L_orig), priors["L_ens10"] - L_orig, 0.0).astype(np.float32)
    del priors

    z4 = np.load(IN_DIR / "pred_mt_a4_gpu_92k.npz")
    stat_i = {s: i for i, s in enumerate(TIMELINE_STATS)}
    rows = []

    for mi, m in [(1, 15), (2, 20)]:
        tm = time.time()
        log(f"\n==================== MINUTO {m} (A2-A4 ABLATION) ====================")
        ok_tl = np.isfinite(tl_raw[:, :, :, : mi + 1, :]).all(axis=(1, 2, 3, 4))
        ok_m = (dur_sec > 60.0 * m) & ok_tl & np.isfinite(L_orig)
        folds_a24 = [(b, np.flatnonzero((chunk == b) & master & ok_m)) for b in (4, 5, 6)]
        a24_idx = np.concatenate([te for _, te in folds_a24])
        ya = y[a24_idx]
        bl_mask = [chunk[a24_idx] == b for b in (4, 5, 6)]
        boots_b = [np.random.default_rng(1000 + m).integers(0, int(bm.sum()), (1000, int(bm.sum()))) for bm in bl_mask]
        g_tot = np.nan_to_num((tl_raw[:, 0, :, mi, stat_i["gold"]] - tl_raw[:, 1, :, mi, stat_i["gold"]]).sum(axis=1), nan=0.0).astype(np.float32)

        f_a2 = extract_minute_features(m, L_orig, tl_raw, gold_resid, exp_lane=exp_lane, exp_trans=exp_trans,
                                       include_all_stats=True, include_traj=True, include_expected=True,
                                       include_comp_cross=False, include_player=False)
        f_a4 = extract_minute_features(m, L_orig, tl_raw, gold_resid, exp_lane=exp_lane, exp_trans=exp_trans,
                                       comp=comp, tempo=tempo, player_pre=player_pre,
                                       include_all_stats=True, include_traj=True, include_expected=True,
                                       include_comp_cross=True, include_player=True)

        p2mt, p3mt, p4mt = z4[f"pa_{m}_2_MT_A4_bs512"], z4[f"pa_{m}_3_MT_A4C1_bs512"], z4[f"pa_{m}_4_MT_Res_A4C1_bs512"]
        p5b, p6f = z4[f"pa_{m}_5_Blend_LR_MT"], z4[f"pa_{m}_6_Final_dEns10"]

        fold_cache = []
        for b, te in folds_a24:
            tr_b = np.flatnonzero(np.isin(chunk, [1, 2, 3, 4, 5, 6]) & (chunk < b) & master & ok_m)
            tr_b = tr_b[np.argsort(t[tr_b], kind="mergesort")]
            ar0b = _prep_ext(f_a2, tr_b, te, L_orig, y, tl_raw, m)
            _, z_te_0b, _ = _fit_logreg_antisym(ar0b["Xa_tr"], y[tr_b], ar0b["Xa_te"], C=0.03)
            z_oof_0b = _oof_logreg_logits(ar0b["Xa_tr"], y[tr_b], t[tr_b], C=0.03, n_splits=4)
            b0_tr = float(np.mean(z_oof_0b))
            cb_tr = _cb_close_mat(tempo["cbias"], g_tot, L_orig, z_oof_0b - b0_tr, tr_b)
            cb_te = _cb_close_mat(tempo["cbias"], g_tot, L_orig, z_te_0b - b0_tr, te)
            fold_cache.append((b, tr_b, te, cb_tr, cb_te))

        p_0c = np.full(len(y), np.nan, np.float32)
        p_v1ns = np.full(len(y), np.nan, np.float32)
        p_nv2ns = np.full(len(y), np.nan, np.float32)
        for b, tr_b, te, cb_tr, cb_te in fold_cache:
            ya_b, ma_b, ys_b, ms_b, _ = _construir_aux_targets(m, tr_b, tl_raw, dur_sec, end_gold_diff, end_kills_diff)
            ar0c = _prep_ext(f_a2, tr_b, te, L_orig, y, tl_raw, m, cb_tr=cb_tr, cb_te=cb_te)
            _, z_te_0c, _ = _fit_logreg_antisym(ar0c["Xa_tr"], y[tr_b], ar0c["Xa_te"], C=0.01)
            z_oof_0c = _oof_logreg_logits(ar0c["Xa_tr"], y[tr_b], t[tr_b], C=0.01, n_splits=4)
            p_0c[te] = _sigm(z_te_0c)
            del ar0c
            ar_a4ext = _prep_ext(f_a4, tr_b, te, L_orig, y, tl_raw, m, cb_tr=cb_tr, cb_te=cb_te)
            p_v1ns[te] = _sigm(np.mean([_fit_v1_noskip_gpu(m, ar_a4ext, y[tr_b], (ya_b, ma_b, ys_b, ms_b), 70*b+s, z_oof_0c, z_te_0c) for s in (0, 1)], axis=0))
            p_nv2ns[te] = _sigm(np.mean([_fit_netv2_gpu(m, ar_a4ext, y[tr_b], (ya_b, ma_b, ys_b, ms_b), 60*b+s, 512, 16, z_oof_0c, z_te_0c, use_skip=False) for s in (0, 1)], axis=0))
            del ar_a4ext; gc.collect()
        del f_a2, f_a4, fold_cache; gc.collect()

        pa_0c, pa_v1ns, pa_nv2 = p_0c[a24_idx], p_v1ns[a24_idx], p_nv2ns[a24_idx]

        def fwd_blend_s2(cols: list[np.ndarray], nom: str):
            b_out = np.full(len(ya), np.nan, np.float32)
            for bi in range(3):
                bm = bl_mask[bi]
                zte = np.column_stack([_logit(c[bm]) for c in cols])
                if bi == 0:
                    b_out[bm] = _sigm(zte.mean(1))
                else:
                    trm = np.isin(chunk[a24_idx], [4 + j for j in range(bi)])
                    lr = LogisticRegression(C=0.5, max_iter=1000).fit(np.column_stack([_logit(c[trm]) for c in cols]), ya[trm])
                    b_out[bm] = lr.predict_proba(zte)[:, 1].astype(np.float32)
            s2 = b_out.copy()
            for bi in (1, 2):
                bm = bl_mask[bi]
                trm = np.isin(chunk[a24_idx], [4 + j for j in range(bi)])
                me = LogisticRegression(C=0.5, max_iter=1000).fit(np.c_[_logit(b_out[trm]), d_ens10[a24_idx[trm]]], ya[trm])
                s2[bm] = me.predict_proba(np.c_[_logit(b_out[bm]), d_ens10[a24_idx[bm]]])[:, 1].astype(np.float32)
            return b_out, s2

        for nom, cols in [
            ("C1_4M_V1NoSkip", [pa_0c, p2mt, p3mt, pa_v1ns]),
            ("C2_4M_NV2NoSkip", [pa_0c, p2mt, p3mt, pa_nv2]),
            ("C3_5M_NV2NoSkip", [pa_0c, p2mt, p3mt, p4mt, pa_nv2]),
        ]:
            bo, s2 = fwd_blend_s2(cols, nom)
            pt1, lo1, hi1, e1 = _blk_ci(ya, bl_mask, boots_b, bo, p5b)
            pt2, lo2, hi2, e2 = _blk_ci(ya, bl_mask, boots_b, s2, p6f)
            print(f"  {nom:18s} vs 5_Blend: {[round(x,5) for x in e1]} | Bloq {pt1:+.5f} [{lo1:+.5f};{hi1:+.5f}]", flush=True)
            print(f"  {nom+'_dE10':18s} vs 6_Final: {[round(x,5) for x in e2]} | Bloq {pt2:+.5f} [{lo2:+.5f};{hi2:+.5f}]", flush=True)
            rows.append({"minuto": m, "modelo": nom, "vs5_Bloq": pt1, "vs5_lo": lo1, "vs5_hi": hi1, "vs6_Bloq": pt2, "vs6_lo": lo2, "vs6_hi": hi2})
        pd.DataFrame(rows).to_csv(IN_DIR / "a24_ablate_v2_g06c.csv", index=False)
        log(f"Minuto {m} terminado en {time.time() - tm:.1f}s")

    log(f"Listo total en {time.time() - T0:.1f}s")


if __name__ == "__main__":
    main()
