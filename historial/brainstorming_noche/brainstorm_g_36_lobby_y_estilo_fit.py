"""
Brainstorming Noche — Gemini G36 (`brainstorm_g_36_lobby_y_estilo_fit.py`):
Explora 3 ejes pre-partida genuinamente no explotados sobre `Escalón 9 oficial` (0,6739):

1. `F1_WL_LOBBY` (`extra_post=True`, 6 cols):
   `prior_lobby_level` (promedio ventana 10 / acum de los OTROS 9 jugadores) aplicado a
   `wr`, `net_wins = asinh((W - L)/10)` y `log_partidas` de Foto 1 + deltas dinámicos
   estrictamente `< m` post `04/09`. En `lp_vivo.db`, `LP + W/L` saltó +0,0449 en Elo
   frente a LP solo (`mmr_historico` solo miró `LP_f1` de los otros 9, ignorando su W/L
   y volumen de smurf). Strict `0` pre-`04/09` vía `extra_post=True`.

2. `CAUSAL_2HOP_LOBBY` (`extra_post=False`, 5 cols):
   Elo causal logístico por partida (`< m`) + `prior_lobby_level` causal `< m` de los
   OTROS 9 jugadores (`1-hop` ventana 10 / acum + `2-hop` propagación del nivel de lobby
   + desempeño `@15` centrado causalmente de los otros 9 + brecha `lobby_elo - elo_propio`).
   Funciona desde t=0, no envece tras el `04/09`, y cubre `Diamond-V` sin depender de Foto 1.

3. `STYLE_CHAMP_FIT` (`extra_post=False`, 4 cols):
   Compatibilidad causal `< m` entre el **perfil de estilo** relativo del jugador (12 stats
   `acum_*` normalizadas causalmente por línea y centradas por jugador para aislar estilo de
   habilidad global) y el **perfil ganador** / **centroide típico** de `(campeón, línea)`
   aprendido en línea antes de `m` (incluyendo interacción en `estreno`).

Uso:
  python historial/brainstorming_noche/brainstorm_g_36_lobby_y_estilo_fit.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
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
import scripts.analysis.ganador_mapa_lp as gm  # noqa: E402
from src.features.lobby import prior_lobby_level  # noqa: E402

CACHE = REPO / "data" / "analysis" / "brainstorm" / "gemini_g36_lobby_estilo.npz"


def _construir_f1_wl_lobby(D) -> np.ndarray:
    """`prior_lobby_level` sobre W/L + partidas (Foto 1 + dinámico < m) en `post`."""
    NM = D["NM"]
    mids = D["Z"]["mids"]
    t = D["t"]
    y = D["y"]
    post = D["post"]
    val0 = D["val0"]
    player = D["B"]["player"]  # (NM, 2, 5) str
    foto1, _ = gm._cargar_foto1(gm.RUTA_DB)

    idx_post = np.flatnonzero(post)
    ord_rel = np.lexsort((mids[idx_post], t[idx_post]))
    post_ord = idx_post[ord_rel]

    f1_w = foto1["wins"].to_dict()
    f1_l = foto1["losses"].to_dict()

    pids_flat = player[post_ord].ravel()
    uniq_p, pid_inv = np.unique(pids_flat, return_inverse=True)
    NP = len(uniq_p)
    w0 = np.array([f1_w.get(p, np.nan) for p in uniq_p], dtype=np.float64)
    l0 = np.array([f1_l.get(p, np.nan) for p in uniq_p], dtype=np.float64)
    tiene = np.isfinite(w0) & np.isfinite(l0)

    w_din = np.zeros(NP, dtype=np.float64)
    l_din = np.zeros(NP, dtype=np.float64)

    n_rows = len(post_ord) * 10
    wr_arr = np.full(n_rows, np.nan, dtype=np.float64)
    netw_arr = np.full(n_rows, np.nan, dtype=np.float64)
    logg_arr = np.full(n_rows, np.nan, dtype=np.float64)

    for k, m in enumerate(post_ord):
        base = k * 10
        p_m = pid_inv[base : base + 10]
        ok = tiene[p_m]
        if ok.any():
            pok = p_m[ok]
            wt = w0[pok] + w_din[pok]
            lt = l0[pok] + l_din[pok]
            tot = wt + lt
            slots = base + np.flatnonzero(ok)
            wr_arr[slots] = (wt + 15.0 * 0.5) / (tot + 30.0)
            netw_arr[slots] = np.arcsinh((wt - lt) / 10.0)
            logg_arr[slots] = np.log1p(np.maximum(tot, 0.0))
        ym = int(y[m])
        w_din[p_m[:5]] += ym
        l_din[p_m[:5]] += 1 - ym
        w_din[p_m[5:]] += 1 - ym
        l_din[p_m[5:]] += ym

    df = pd.DataFrame(
        {
            "match_id": np.repeat(mids[post_ord], 10),
            "game_creation": np.repeat(t[post_ord], 10),
            "player_id": pid_inv,
            "wr_din": wr_arr,
            "netw_din": netw_arr,
            "logg_din": logg_arr,
        }
    )
    en_fp = np.repeat(~val0[post_ord], 10)

    def calc_pl(col: str) -> tuple[np.ndarray, np.ndarray]:
        tmp = prior_lobby_level(df[["match_id", "game_creation", "player_id", col]], col, k=5, mu0=0.0, window=10, min_otros=3)
        mu0 = float(tmp.loc[en_fp, "nivel_lobby"].mean())
        if not np.isfinite(mu0):
            mu0 = 0.0
        res = prior_lobby_level(df[["match_id", "game_creation", "player_id", col]], col, k=5, mu0=mu0, window=10, min_otros=3)
        return res["nivel_v10"].to_numpy(np.float32), res["nivel_acum"].to_numpy(np.float32)

    wr_v10, wr_ac = calc_pl("wr_din")
    nw_v10, nw_ac = calc_pl("netw_din")
    lg_v10, _ = calc_pl("logg_din")
    wr_gap = np.where(np.isfinite(wr_arr), wr_v10 - wr_arr, 0.0).astype(np.float32)

    feats_post = np.stack([wr_v10, wr_ac, nw_v10, nw_ac, lg_v10, wr_gap], axis=-1).reshape(len(post_ord), 2, 5, 6)
    diff_post = (feats_post[:, 0] - feats_post[:, 1]).mean(axis=1).astype(np.float32)
    out = np.zeros((NM, 6), dtype=np.float32)
    out[post_ord] = diff_post
    return out


def _construir_causal_2hop_lobby(D) -> np.ndarray:
    """Lobby causal 1-hop / 2-hop sobre Elo causal y desempeño @15 en TODAS las partidas."""
    NM = D["NM"]
    mids = D["Z"]["mids"]
    t = D["t"]
    y = D["y"]
    Xj = D["Xj"]
    cols = D["cols"]
    player = D["B"]["player"]

    ord_idx = np.lexsort((mids, t))
    pids_ord = player[ord_idx].ravel()
    _, pid_inv = np.unique(pids_ord, return_inverse=True)
    NP = int(pid_inv.max()) + 1

    idx_dg15 = cols.index("acum_d_gold_15")
    idx_dx15 = cols.index("acum_d_xp_15")
    idx_dp = cols.index("acum_deaths_pm")
    idx_logn = cols.index("logn_tot")

    dg15 = np.nan_to_num(Xj[:, :, :, idx_dg15], nan=0.0).astype(np.float64)
    dx15 = np.nan_to_num(Xj[:, :, :, idx_dx15], nan=0.0).astype(np.float64)
    dp = np.nan_to_num(Xj[:, :, :, idx_dp], nan=0.0).astype(np.float64)
    ng = np.expm1(np.maximum(np.nan_to_num(Xj[:, :, :, idx_logn], nan=0.0), 0.0))

    elo = np.zeros(NP, dtype=np.float64)
    n_elo = np.zeros(NP, dtype=np.float64)

    W = 10
    k_shrink = 5.0

    buf_e1 = np.zeros((NP, W), dtype=np.float64)
    ptr_e1 = np.zeros(NP, dtype=np.int32)
    cnt_e1 = np.zeros(NP, dtype=np.int32)
    sum_e1_ac = np.zeros(NP, dtype=np.float64)
    buf_e1_sum = np.zeros(NP, dtype=np.float64)

    buf_p1 = np.zeros((NP, W), dtype=np.float64)
    buf_p1_sum = np.zeros(NP, dtype=np.float64)

    buf_2h = np.zeros((NP, W), dtype=np.float64)
    buf_2h_sum = np.zeros(NP, dtype=np.float64)

    # Media causal en línea de muertes/min para que el shrinkage a 0 sea exacto
    dp_n = 0.0
    dp_sum = 0.0

    out_ord = np.zeros((NM, 5), dtype=np.float32)

    for step, m in enumerate(ord_idx):
        base = step * 10
        ps = pid_inv[base : base + 10]
        e_pre = elo[ps]

        mu_dp = dp_sum / max(dp_n, 1.0) if dp_n >= 200 else 0.20
        ng_m = ng[m].ravel()
        rel_pf = ng_m / (ng_m + 5.0)
        pf_pre = np.clip(
            rel_pf * ((dg15[m].ravel() / 600.0) + 0.7 * (dx15[m].ravel() / 500.0) - 3.0 * (dp[m].ravel() - mu_dp)),
            -3.0,
            3.0,
        )
        # Actualizar media causal solo con jugadores con >=4 partidas previas
        mask_dp = ng_m >= 4.0
        if mask_dp.any():
            dp_n += float(mask_dp.sum())
            dp_sum += float(dp[m].ravel()[mask_dp].sum())

        n_prev = cnt_e1[ps].astype(np.float64)
        nw = np.minimum(n_prev, W)
        l_v10 = buf_e1_sum[ps] / (nw + k_shrink)
        l_ac = sum_e1_ac[ps] / (n_prev + k_shrink)
        l_2h = buf_2h_sum[ps] / (nw + k_shrink)
        l_pf = buf_p1_sum[ps] / (nw + k_shrink)
        div = l_v10 - e_pre

        f_m = np.stack([l_v10, l_ac, l_2h, l_pf, div], axis=-1)  # (10, 5)
        out_ord[step] = f_m[:5].mean(axis=0) - f_m[5:].mean(axis=0)

        other_e = (e_pre.sum() - e_pre) / 9.0
        other_pf = (pf_pre.sum() - pf_pre) / 9.0
        other_1h = (l_v10.sum() - l_v10) / 9.0

        ptrs = ptr_e1[ps]
        buf_e1_sum[ps] += other_e - buf_e1[ps, ptrs]
        buf_e1[ps, ptrs] = other_e
        sum_e1_ac[ps] += other_e

        buf_p1_sum[ps] += other_pf - buf_p1[ps, ptrs]
        buf_p1[ps, ptrs] = other_pf

        buf_2h_sum[ps] += other_1h - buf_2h[ps, ptrs]
        buf_2h[ps, ptrs] = other_1h

        ptr_e1[ps] = (ptrs + 1) % W
        cnt_e1[ps] += 1

        eb = e_pre[:5].mean()
        er = e_pre[5:].mean()
        p_hat = 1.0 / (1.0 + np.exp(-np.clip(eb - er, -8.0, 8.0)))
        err = float(y[m]) - p_hat
        k_vec = 0.24 / np.sqrt(1.0 + 0.15 * n_elo[ps])
        elo[ps[:5]] += k_vec[:5] * err
        elo[ps[5:]] -= k_vec[5:] * err
        n_elo[ps] += 1.0

    out = np.zeros((NM, 5), dtype=np.float32)
    out[ord_idx] = out_ord
    return out


def _construir_style_champ_fit(D) -> np.ndarray:
    """Compatibilidad causal entre estilo relativo del jugador y perfil ganador de (campeón, línea)."""
    NM = D["NM"]
    mids = D["Z"]["mids"]
    t = D["t"]
    y = D["y"]
    Xj = D["Xj"]
    cols = D["cols"]
    champ = D["Z"]["champ"].reshape(NM, 2, 5)

    style_cols = [
        "acum_kills_pm",
        "acum_assists_pm",
        "acum_damage_share_pct",
        "acum_kill_participation_pct",
        "acum_deaths_pm",
        "acum_damage_taken_pm",
        "acum_gold_pm",
        "acum_cs_pm",
        "acum_vision_pm",
        "acum_d_gold_10",
        "acum_d_gold_15",
        "acum_d_xp_15",
    ]
    idx_s = [cols.index(c) for c in style_cols]
    idx_logn = cols.index("logn_tot")
    idx_est = cols.index("estreno")

    S_raw = np.nan_to_num(Xj[:, :, :, idx_s], nan=0.0).reshape(NM, 10, len(style_cols)).astype(np.float64)
    logn_tot = np.nan_to_num(Xj[:, :, :, idx_logn], nan=0.0).reshape(NM, 10).astype(np.float64)
    estreno = np.nan_to_num(Xj[:, :, :, idx_est], nan=0.0).reshape(NM, 10).astype(np.float64)
    n_games = np.expm1(np.maximum(logn_tot, 0.0))
    rel = n_games / (n_games + 12.0)

    c_uniq = np.unique(champ)
    cmap = {int(c): i for i, c in enumerate(c_uniq)}
    NC = len(c_uniq)
    champ_idx = np.vectorize(cmap.__getitem__, otypes=[np.int32])(champ).reshape(NM, 10)

    ord_idx = np.lexsort((mids, t))
    KD = len(style_cols)
    lanes10 = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=np.int32)
    lanes5 = np.arange(5, dtype=np.int32)

    lane_n = np.zeros(5, dtype=np.float64)
    lane_sum = np.zeros((5, KD), dtype=np.float64)
    lane_sq = np.zeros((5, KD), dtype=np.float64)

    cl_n = np.zeros((NC, 5), dtype=np.float64)
    cl_sum_s = np.zeros((NC, 5, KD), dtype=np.float64)
    cl_win_s = np.zeros((NC, 5, KD), dtype=np.float64)

    out_ord = np.zeros((NM, 4), dtype=np.float32)

    for step, m in enumerate(ord_idx):
        sm = S_raw[m]  # (10, 12)
        rm = rel[m]    # (10,)
        estm = estreno[m]
        cm = champ_idx[m]

        ready = lane_n.min() >= 200
        if ready:
            mu_l = lane_sum / lane_n[:, None]
            sd_l = np.sqrt(np.maximum(lane_sq / lane_n[:, None] - mu_l**2, 1e-4))
            z = np.clip((sm - mu_l[lanes10]) / sd_l[lanes10], -3.5, 3.5)
            # Centrar por jugador a lo largo de las 12 dimensiones -> forma pura de estilo
            z_shape = (z - z.mean(axis=1, keepdims=True)) * rm[:, None]

            nc = cl_n[cm, lanes10]  # (10,)
            mu_c = cl_sum_s[cm, lanes10] / (nc[:, None] + 40.0)
            beta_c = cl_win_s[cm, lanes10] / (nc[:, None] + 80.0)
            dif = z_shape - mu_c

            dot_w = np.einsum("ij,ij->i", dif, beta_c) * rm
            beta_norm = np.linalg.norm(beta_c, axis=1) + 0.08
            cos_w = np.einsum("ij,ij->i", z_shape, beta_c) / beta_norm
            arch = -np.mean(dif**2, axis=1) * rm
            est_arch = arch * estm

            f10 = np.stack([dot_w, cos_w, arch, est_arch], axis=-1)
            out_ord[step] = (f10[:5].mean(axis=0) - f10[5:].mean(axis=0)).astype(np.float32)

        # Actualizar estadísticas de línea solo con jugadores que tienen >=4 partidas previas
        mask_mat = n_games[m] >= 4.0
        for l in range(5):
            # filas l (azul) y l+5 (rojo)
            for j in (l, l + 5):
                if mask_mat[j]:
                    xrow = sm[j]
                    lane_n[l] += 1.0
                    lane_sum[l] += xrow
                    lane_sq[l] += xrow**2

        if ready:
            ym = float(y[m])
            sgn10 = np.array([2.0 * ym - 1.0] * 5 + [1.0 - 2.0 * ym] * 5, dtype=np.float64)
            upd = rm > 0.01
            if upd.any():
                c_u = cm[upd]
                l_u = lanes10[upd]
                r_u = rm[upd]
                zs_u = z_shape[upd]
                mu_u = mu_c[upd]
                np.add.at(cl_n, (c_u, l_u), r_u)
                np.add.at(cl_sum_s, (c_u, l_u), r_u[:, None] * zs_u)
                np.add.at(cl_win_s, (c_u, l_u), (sgn10[upd] * r_u)[:, None] * (zs_u - mu_u))

    out = np.zeros((NM, 4), dtype=np.float32)
    out[ord_idx] = out_ord
    return out


def _eval_diam(D, extra_list):
    ms = D["master"]
    tr = D["fi"]
    tr_m = tr[ms[tr]]
    te = np.flatnonzero(D["diam_v"])
    p = ajustar_predecir(D, tr_m, te, extra=extra_list, peso=None)
    yv = D["y"][te]
    return float(roc_auc_score(yv, p)), float(log_loss(yv, p)), p, yv


def _boot_dif(p, pr, yv, B=200, seed=1):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(yv), size=(B, len(yv)))
    d = np.array([roc_auc_score(yv[s], p[s]) - roc_auc_score(yv[s], pr[s]) for s in idx])
    return float(roc_auc_score(yv, p) - roc_auc_score(yv, pr)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    t0 = time.time()
    D = cargar()
    ref9 = extras_referencia(D, escalon=9)
    print(f"Base + Escalón 9 cargados ({ref9.shape[1]} cols extra) en {time.time() - t0:.1f}s")

    if CACHE.exists():
        C = np.load(CACHE)
        f1_wl = C["F1_WL_LOBBY"]
        c_2hop = C["CAUSAL_2HOP_LOBBY"]
        st_fit = C["STYLE_CHAMP_FIT"]
        print(f"Caché cargada: {CACHE.name}")
    else:
        ta = time.time()
        f1_wl = _construir_f1_wl_lobby(D)
        print(f"F1_WL_LOBBY calculado: {f1_wl.shape} en {time.time() - ta:.1f}s (std post={np.round(f1_wl[D['post']].std(0), 4)})")
        tb = time.time()
        c_2hop = _construir_causal_2hop_lobby(D)
        print(f"CAUSAL_2HOP_LOBBY calculado: {c_2hop.shape} en {time.time() - tb:.1f}s (std={np.round(c_2hop.std(0), 4)})")
        tc = time.time()
        st_fit = _construir_style_champ_fit(D)
        print(f"STYLE_CHAMP_FIT calculado: {st_fit.shape} en {time.time() - tc:.1f}s (std={np.round(st_fit.std(0), 4)})")
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, F1_WL_LOBBY=f1_wl, CAUSAL_2HOP_LOBBY=c_2hop, STYLE_CHAMP_FIT=st_fit)
        print(f"Caché guardada en {CACHE.name}")

    rng = np.random.default_rng(36)
    plac_f1 = f1_wl.copy()
    ip = np.flatnonzero(D["post"])
    plac_f1[ip] = plac_f1[rng.permutation(ip)]
    plac_2h = c_2hop[rng.permutation(D["NM"])]

    variantes = {
        "Escalón 9 oficial": [(ref9, False)],
        "+ F1_WL_LOBBY (6)": [(ref9, False), (f1_wl, True)],
        "+ CAUSAL_2HOP_LOBBY (5)": [(ref9, False), (c_2hop, False)],
        "+ STYLE_CHAMP_FIT (4)": [(ref9, False), (st_fit, False)],
        "+ F1_WL + 2HOP (11)": [(ref9, False), (f1_wl, True), (c_2hop, False)],
        "+ TODO G36 (15)": [(ref9, False), (f1_wl, True), (c_2hop, False), (st_fit, False)],
        "+ placebo (11)": [(ref9, False), (plac_f1, True), (plac_2h, False)],
    }

    ms = D["master"]
    P = {}
    D_res = {}
    for nombre, ex_list in variantes.items():
        po, pv = evaluar(D, lambda tr, te, _e=ex_list: ajustar_predecir(D, tr[ms[tr]], te, extra=_e))
        P[nombre] = (po, pv)
        auc_d, ll_d, pd_v, y_d = _eval_diam(D, ex_list)
        D_res[nombre] = (auc_d, ll_d, pd_v, y_d)

    T = comparar(D, P, ref="Escalón 9 oficial")
    yv = D["y"][D["val"]]
    pd_ref = D_res["Escalón 9 oficial"][2]
    y_d = D_res["Escalón 9 oficial"][3]
    T["LL V"] = [round(log_loss(yv, P[n][1]), 5) for n in T["variante"]]
    T["AUC DiamV"] = [D_res[n][0] for n in T["variante"]]
    T["LL DiamV"] = [round(D_res[n][1], 5) for n in T["variante"]]
    dl = []
    for n in T["variante"]:
        d, lo, hi = _boot_dif(D_res[n][2], pd_ref, y_d)
        dl.append(f"{d:+.4f} [{lo:+.4f}; {hi:+.4f}]" if n != "Escalón 9 oficial" else "+0.0000 [ref]")
    T["Δ DiamV (IC)"] = dl

    print("\n=== RESULTADOS G36 SOBRE ESCALÓN 9 OFICIAL ===")
    mostrar(T)


if __name__ == "__main__":
    main()
