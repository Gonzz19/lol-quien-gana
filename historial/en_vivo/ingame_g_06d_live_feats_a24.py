"""Cribado rápido de señales nuevas in-game en A2-A4 ONLY (CPU < 1.5 GB, ZERO acceso a V/DiamV).

Prueba sobre `0c_A2_C1_cb` y su mezcla con las redes cacheadas `[2_MT, 3_MTC1, 4_MTRes]` de g_04 en A2-A4:
  - S1_LeadSat   : compresión no lineal antisimétrica (tanh / arcsinh de oro, xp, carries y carriles)
  - S2_KillBounty: kda máx., kills sin morir, daño/oro de carries, neto K-D en carries
  - S3_FedAffinity: dot(cbias, z_gold), dot(pbias, z_gold), dot(dps_role, z_gold), dot(p_acum_dg15, z_gold)
  - S4_XpGoldDiv : extremos z-XP, brecha gold_z - xp_z, min(z_bot, z_mid) y velocidad carries
  - Selección estricta de C in (0.01, 0.02, 0.03) dentro de `tr_b` (CV cronológica interna).

Uso:
    python -u historial/en_vivo/ingame_g_06d_live_feats_a24.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.analysis.ingame_evaluar import _fit_logreg_antisym, _logit, _prep_arrays, _sigm  # noqa: E402
from historial.ensamble_en_vivo.ingame_g_01_prior_y_gpu import _cargar_priors  # noqa: E402
from historial.ensamble_en_vivo.ingame_g_03_regimen_y_senales import _carry_conc_feats  # noqa: E402
from historial.en_vivo.ingame_g_06_a24_explore import _blk_ci, _oof_logreg_logits  # noqa: E402
from historial.en_vivo.ingame_g_06b_netv2_a24 import _cb_close_mat  # noqa: E402
from src.models.ingame import MINUTES, TIMELINE_STATS, extract_minute_features  # noqa: E402

IN_DIR = REPO / "data" / "analysis" / "ingame"
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def _extra_live_groups(
    tl_raw: np.ndarray,
    tempo: dict[str, np.ndarray],
    slot_static: np.ndarray,
    p_mat: np.ndarray,
    minute: int,
    tr_idx: np.ndarray,
    idx: np.ndarray,
) -> dict[str, np.ndarray]:
    """Construye grupos de features causales y antisimétricas al minuto `minute`."""
    mi = MINUTES.index(minute)
    st = {s: i for i, s in enumerate(TIMELINE_STATS)}
    cur = np.nan_to_num(tl_raw[idx, :, :, mi, :], nan=0.0).astype(np.float32)
    gb, gr = cur[:, 0, :, st["gold"]], cur[:, 1, :, st["gold"]]
    xb, xr = cur[:, 0, :, st["xp"]], cur[:, 1, :, st["xp"]]
    db, dr = cur[:, 0, :, st["damage"]], cur[:, 1, :, st["damage"]]
    kb, kr = cur[:, 0, :, st["kills"]], cur[:, 1, :, st["kills"]]
    deb, der = cur[:, 0, :, st["deaths"]], cur[:, 1, :, st["deaths"]]
    ab, ar_ = cur[:, 0, :, st["assists"]], cur[:, 1, :, st["assists"]]

    # Estadísticos por rol calculados SOLO con tr_idx (simétrico B+R)
    gb_tr = np.nan_to_num(tl_raw[tr_idx, 0, :, mi, st["gold"]], nan=0.0)
    gr_tr = np.nan_to_num(tl_raw[tr_idx, 1, :, mi, st["gold"]], nan=0.0)
    g_pool = np.r_[gb_tr, gr_tr]
    mu_g = g_pool.mean(axis=0).astype(np.float32)
    sd_g = np.maximum(g_pool.std(axis=0), 1e-4).astype(np.float32)
    zb = (gb - mu_g) / sd_g
    zr = (gr - mu_g) / sd_g

    xb_tr = np.nan_to_num(tl_raw[tr_idx, 0, :, mi, st["xp"]], nan=0.0)
    xr_tr = np.nan_to_num(tl_raw[tr_idx, 1, :, mi, st["xp"]], nan=0.0)
    x_pool = np.r_[xb_tr, xr_tr]
    mu_x = x_pool.mean(axis=0).astype(np.float32)
    sd_x = np.maximum(x_pool.std(axis=0), 1e-4).astype(np.float32)
    zxb = (xb - mu_x) / sd_x
    zxr = (xr - mu_x) / sd_x

    # S1: compresión no lineal de ventaja
    dg_l = gb - gr
    dg_t = dg_l.sum(axis=1)
    dxp_t = (xb - xr).sum(axis=1)
    dg_c = (gb[:, 0] + gb[:, 2]) - (gr[:, 0] + gr[:, 2])
    ddmg_t = (db - dr).sum(axis=1)
    s1 = np.column_stack([
        np.tanh(dg_t / 2.5),
        np.tanh(dg_t / 6.0),
        np.arcsinh(dg_t),
        np.tanh(dxp_t / 2.5),
        np.arcsinh(dxp_t),
        np.tanh(dg_c / 1.5),
        np.tanh(ddmg_t / 5.0),
        np.tanh(dg_l / 1.5).sum(axis=1),
    ]).astype(np.float32)

    # S2: kill concentration, kda máx, unbroken kills, daño/oro carries, neto KD carries
    kda_b = (kb + 0.5 * ab) / (deb + 1.0)
    kda_r = (kr + 0.5 * ar_) / (der + 1.0)
    unbr_b = (kb * (deb == 0)).sum(axis=1)
    unbr_r = (kr * (der == 0)).sum(axis=1)
    dmg_c_eff = ((db[:, 0] + db[:, 2]) / (gb[:, 0] + gb[:, 2] + 2.0)) - (
        (dr[:, 0] + dr[:, 2]) / (gr[:, 0] + gr[:, 2] + 2.0)
    )
    k_c_share = ((kb[:, 0] + kb[:, 2]) / (kb.sum(axis=1) + 1.0)) - (
        (kr[:, 0] + kr[:, 2]) / (kr.sum(axis=1) + 1.0)
    )
    kd_c = ((kb[:, 0] + kb[:, 2] - deb[:, 0] - deb[:, 2])) - (
        (kr[:, 0] + kr[:, 2] - der[:, 0] - der[:, 2])
    )
    s2 = np.column_stack([
        kda_b.max(axis=1) - kda_r.max(axis=1),
        unbr_b - unbr_r,
        dmg_c_eff,
        k_c_share,
        kd_c,
    ]).astype(np.float32)

    # S3: afinidad escalado/rol/jugador × oro actual por rol (estricto B - R)
    cb = np.nan_to_num(tempo["cbias"][idx], nan=0.0)
    pb = np.nan_to_num(tempo["pbias"][idx], nan=0.0)
    ss = np.nan_to_num(slot_static[idx], nan=0.0)
    w_dps = ss[:, :, :, 13] + 0.5 * ss[:, :, :, 1]  # tirador + 0.5 * ps_dano
    p_sk = np.nan_to_num(p_mat[idx, :, :, 0], nan=0.0)  # acum_d_gold_15
    s3 = np.column_stack([
        np.mean(cb[:, 0] * zb, axis=1) - np.mean(cb[:, 1] * zr, axis=1),
        np.mean(pb[:, 0] * zb, axis=1) - np.mean(pb[:, 1] * zr, axis=1),
        np.mean(w_dps[:, 0] * zb, axis=1) - np.mean(w_dps[:, 1] * zr, axis=1),
        np.mean(p_sk[:, 0] * zb, axis=1) - np.mean(p_sk[:, 1] * zr, axis=1),
    ]).astype(np.float32)

    # S4: z-XP extremos + divergencia oro-xp + carries min + velocidad carries
    s4_cols = [
        zxb.max(axis=1) - zxr.max(axis=1),
        zxb.min(axis=1) - zxr.min(axis=1),
        (zb.mean(axis=1) - zxb.mean(axis=1)) - (zr.mean(axis=1) - zxr.mean(axis=1)),
        np.minimum(zb[:, 0], zb[:, 2]) - np.minimum(zr[:, 0], zr[:, 2]),
    ]
    if mi >= 1:
        prv = np.nan_to_num(tl_raw[idx, :, :, mi - 1, st["gold"]], nan=0.0)
        vc = ((gb[:, 0] + gb[:, 2]) - (prv[:, 0, 0] + prv[:, 0, 2])) - (
            (gr[:, 0] + gr[:, 2]) - (prv[:, 1, 0] + prv[:, 1, 2])
        )
        s4_cols.append(vc)
    s4 = np.column_stack(s4_cols).astype(np.float32)

    return {"S1_LeadSat": s1, "S2_KillBounty": s2, "S3_FedAffinity": s3, "S4_XpGoldDiv": s4}


def _select_C_chron(Xa_tr: np.ndarray, y_tr: np.ndarray, t_tr: np.ndarray, cs=(0.01, 0.02)) -> float:
    qs = np.quantile(t_tr, np.linspace(0, 1, 5)[1:-1])
    blk = np.searchsorted(qs, t_tr, side="right")
    best_c, best_ll = float(cs[0]), float("inf")
    for c in cs:
        p = np.zeros(len(y_tr), dtype=np.float32)
        for b in range(4):
            vm = blk == b
            lr = LogisticRegression(C=c, max_iter=2000).fit(Xa_tr[~vm], y_tr[~vm])
            p[vm] = lr.predict_proba(Xa_tr[vm])[:, 1]
        ll = log_loss(y_tr, p)
        if ll < best_ll:
            best_ll, best_c = ll, float(c)
    return best_c


def main() -> None:
    log("Iniciando cribado A2-A4 (g_06d, ZERO acceso a V/DiamV)...")
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
    exp_lane = {k[4:]: C[k].astype(np.float32) for k in C.files if k.startswith("exp_")}
    exp_trans = {k[3:]: C[k].astype(np.float32) for k in C.files if k.startswith("tr_")}
    tempo = {k: C[k].astype(np.float32) for k in ("cbias", "cslope", "pbias", "pvol")}
    slot_static = C["slot_static"].astype(np.float32)
    p_mat = C["p_mat"].astype(np.float32)
    del C
    gc.collect()

    priors = _cargar_priors(mids, bloque, val, diam_v, L_orig)
    d_ens10 = np.where(np.isfinite(priors["L_ens10"]) & np.isfinite(L_orig), priors["L_ens10"] - L_orig, 0.0).astype(np.float32)
    del priors

    z4 = np.load(IN_DIR / "pred_mt_a4_gpu_92k.npz")
    stat_i = {s: i for i, s in enumerate(TIMELINE_STATS)}
    rows = []

    for mi, m in enumerate(MINUTES):
        tm = time.time()
        log(f"\n==================== MINUTO {m} (A2-A4 LIVE FEATS) ====================")
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

        p0b_ref = z4[f"pa_{m}_0b_A2_C1"]
        p2mt, p3mt, p4mt = z4[f"pa_{m}_2_MT_A4_bs512"], z4[f"pa_{m}_3_MT_A4C1_bs512"], z4[f"pa_{m}_4_MT_Res_A4C1_bs512"]
        p5b, p6f = z4[f"pa_{m}_5_Blend_LR_MT"], z4[f"pa_{m}_6_Final_dEns10"]

        # Precomputar arrays base y grupos por fold
        fold_data = []
        for b, te in folds_a24:
            tr_b = np.flatnonzero(np.isin(chunk, [1, 2, 3, 4, 5, 6]) & (chunk < b) & master & ok_m)
            tr_b = tr_b[np.argsort(t[tr_b], kind="mergesort")]
            ar0b = _prep_arrays(f_a2, tr_b, te, L_orig, y)
            c1_tr = _carry_conc_feats(tl_raw, m, tr_b, tr_b)
            c1_te = _carry_conc_feats(tl_raw, m, tr_b, te)
            sd_c1 = np.maximum(c1_tr.std(0), 1e-6).astype(np.float32)
            c1s_tr = np.clip(c1_tr / sd_c1, -8, 8)
            c1s_te = np.clip(c1_te / sd_c1, -8, 8)
            _, z_te_0b, _ = _fit_logreg_antisym(np.c_[ar0b["Xa_tr"], c1s_tr], y[tr_b],
                                                np.c_[ar0b["Xa_te"], c1s_te], C=0.03)
            z_oof_0b = _oof_logreg_logits(np.c_[ar0b["Xa_tr"], c1s_tr],
                                          y[tr_b], t[tr_b], C=0.03, n_splits=4)
            b0 = float(np.mean(z_oof_0b))
            cb_tr = _cb_close_mat(tempo["cbias"], g_tot, L_orig, z_oof_0b - b0, tr_b)
            cb_te = _cb_close_mat(tempo["cbias"], g_tot, L_orig, z_te_0b - b0, te)
            ext_tr = np.c_[c1_tr, cb_tr].astype(np.float32)
            ext_te = np.c_[c1_te, cb_te].astype(np.float32)
            sd = np.maximum(ext_tr.std(0), 1e-6).astype(np.float32)
            Xa0c_tr = np.c_[ar0b["Xa_tr"], np.clip(ext_tr / sd, -8, 8)].astype(np.float32)
            Xa0c_te = np.c_[ar0b["Xa_te"], np.clip(ext_te / sd, -8, 8)].astype(np.float32)

            grp_tr = _extra_live_groups(tl_raw, tempo, slot_static, p_mat, m, tr_b, tr_b)
            grp_te = _extra_live_groups(tl_raw, tempo, slot_static, p_mat, m, tr_b, te)
            for k in grp_tr:
                sg = np.maximum(grp_tr[k].std(0), 1e-6).astype(np.float32)
                grp_tr[k] = np.clip(grp_tr[k] / sg, -8, 8).astype(np.float32)
                grp_te[k] = np.clip(grp_te[k] / sg, -8, 8).astype(np.float32)
            fold_data.append((b, tr_b, te, Xa0c_tr, Xa0c_te, grp_tr, grp_te))

        def eval_variant(nom: str, grp_keys: list[str], use_0c: bool = True):
            p_lr = np.full(len(y), np.nan, np.float32)
            best_cs = []
            for b, tr_b, te, Xa0c_tr, Xa0c_te, grp_tr, grp_te in fold_data:
                Xtr = Xa0c_tr if not grp_keys else np.column_stack([Xa0c_tr] + [grp_tr[k] for k in grp_keys])
                Xte = Xa0c_te if not grp_keys else np.column_stack([Xa0c_te] + [grp_te[k] for k in grp_keys])
                c_opt = _select_C_chron(Xtr, y[tr_b], t[tr_b])
                best_cs.append(c_opt)
                _, z_te, _ = _fit_logreg_antisym(Xtr, y[tr_b], Xte, C=c_opt)
                p_lr[te] = _sigm(z_te)
            pa = p_lr[a24_idx]
            pt_lr, lo_lr, hi_lr, e_lr = _blk_ci(ya, bl_mask, boots_b, pa, p0b_ref)

            # Blend forward con [p2mt, p3mt, p4mt] de g_04
            cols = [pa, p2mt, p3mt, p4mt]
            bo = np.full(len(ya), np.nan, np.float32)
            for bi in range(3):
                bm = bl_mask[bi]
                zte = np.column_stack([_logit(c[bm]) for c in cols])
                if bi == 0:
                    bo[bm] = _sigm(zte.mean(1))
                else:
                    trm = np.isin(chunk[a24_idx], [4 + j for j in range(bi)])
                    lr = LogisticRegression(C=0.5, max_iter=1000).fit(np.column_stack([_logit(c[trm]) for c in cols]), ya[trm])
                    bo[bm] = lr.predict_proba(zte)[:, 1].astype(np.float32)
            s2 = bo.copy()
            for bi in (1, 2):
                bm = bl_mask[bi]
                trm = np.isin(chunk[a24_idx], [4 + j for j in range(bi)])
                me = LogisticRegression(C=0.5, max_iter=1000).fit(np.c_[_logit(bo[trm]), d_ens10[a24_idx[trm]]], ya[trm])
                s2[bm] = me.predict_proba(np.c_[_logit(bo[bm]), d_ens10[a24_idx[bm]]])[:, 1].astype(np.float32)

            pt5, lo5, hi5, e5 = _blk_ci(ya, bl_mask, boots_b, bo, p5b)
            pt6, lo6, hi6, e6 = _blk_ci(ya, bl_mask, boots_b, s2, p6f)
            print(f"  {nom:16s} Cs={best_cs} | LRvs0b {[round(x,5) for x in e_lr]} Bloq {pt_lr:+.5f} [{lo_lr:+.5f};{hi_lr:+.5f}]", flush=True)
            print(f"    -> +MT_g04 vs5 {[round(x,5) for x in e5]} Bloq {pt5:+.5f} [{lo5:+.5f};{hi5:+.5f}] | vs6 {pt6:+.5f} [{lo6:+.5f};{hi6:+.5f}]", flush=True)
            rows.append({"minuto": m, "modelo": nom, "lr_vs0b": pt_lr, "lr_lo": lo_lr, "lr_hi": hi_lr,
                         "b_vs5": pt5, "b_lo": lo5, "b_hi": hi5, "s2_vs6": pt6, "s2_lo": lo6, "s2_hi": hi6})
            return pa

        pa_0c = eval_variant("0c_Ref", [])
        pa_s1 = eval_variant("S1_LeadSat", ["S1_LeadSat"])
        pa_s2 = eval_variant("S2_KillBounty", ["S2_KillBounty"])
        pa_s3 = eval_variant("S3_FedAffinity", ["S3_FedAffinity"])
        pa_s4 = eval_variant("S4_XpGoldDiv", ["S4_XpGoldDiv"])
        pa_all = eval_variant("S_All_1234", ["S1_LeadSat", "S2_KillBounty", "S3_FedAffinity", "S4_XpGoldDiv"])
        del f_a2, fold_data; gc.collect()
        pd.DataFrame(rows).to_csv(IN_DIR / "a24_live_feats_g06d.csv", index=False)
        log(f"Minuto {m} en {time.time() - tm:.1f}s")

    log(f"Listo total en {time.time() - T0:.1f}s")


if __name__ == "__main__":
    main()
