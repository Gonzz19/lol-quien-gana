"""Idea G43 (`docs/BRAINSTORM_NOCHE.md`): Tres ejes causales ortogonales a `LK_AGGS (4)` (`@15 gold+xp`):

1. **`GRAPH_LSQR_15 (4)`** — Resolución global estrictamente causal (`< chunk`, `scipy.sparse.linalg.lsqr`)
   del grafo 1v1 `@15` (y grafo de victorias por equipo): a diferencia del filtro hacia adelante `g_30`
   (que nunca corrige el crédito cuando un rival temprano luego demuestra ser Challenger `< t_chunk`),
   resuelve el sistema conjunto ponderado exponencialmente en el tiempo sobre `32` tramos cronológicos
   estrictamente anteriores (y tramos `< 16/09` para `V` y `Diamond-V`).
2. **`MID_15_20_KALMAN (3)`** — Kalman 1v1 sobre la progresión pura `15 -> 20 min` ortogonalizada respecto a `@15`:
   `dz_15_20 = z20 - beta_l * z15` (pendiente `beta_l` fijada únicamente en `~val0 & ~post & master`),
   descontando escalado causal del cruce `(cA, cB, l)` `< m`. Mide sidelaning y juego medio post-placas.
3. **`EFF_VIS_KALMAN (4)`** — Kalman 1v1 sobre eficiencia daño/oro (`damage_share - gold_share`) y visión/min,
   ortogonalizadas respecto a `(y - 0.5)` en `~val0 & ~post & master` y al cruce causal `(cA, cB, l)` `< m`.

Evaluado directamente sobre **Escalón 9 oficial (`V 0,6739`)** con placebo y chequeo ciego `Diamond-V`
(sin arrastrar `LK_AGGS`, que quedó en el límite por bloques/Diamond-V).
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr
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

BR_DIR = REPO / "data" / "analysis" / "brainstorm"
CACHE = BR_DIR / "gemini_g34_mid_eff_grafo.npz"
LANES = ["BOTTOM", "JUNGLE", "MIDDLE", "TOP", "UTILITY"]


def _extraer_todo_por_carril(D: dict):
    mids = D["Z"]["mids"].astype(str)
    NM = D["NM"]
    pos = pd.Series(np.arange(NM, dtype=np.int32), index=mids)
    idx_l = {l: i for i, l in enumerate(LANES)}

    cols = [
        "match_id", "team_id", "lane",
        "gold_15", "xp_15", "gold_20", "xp_20",
        "damage_share_pct", "gold_share_pct", "vision_score",
    ]
    df = pd.read_csv(REPO / "data" / "splits" / "train" / "full_dataset.csv", usecols=cols)
    df = df[df.lane.isin(LANES)].copy()
    df["m"] = df["match_id"].astype(str).map(pos)
    df = df.dropna(subset=["m"]).copy()

    m = df["m"].to_numpy(np.int32)
    e = (df["team_id"].to_numpy() == 200).astype(np.int32)
    l = df["lane"].map(idx_l).to_numpy(np.int32)

    names = ["gold_15", "xp_15", "gold_20", "xp_20", "damage_share_pct", "gold_share_pct", "vision_score"]
    T = np.zeros((len(names), NM, 2, 5), dtype=np.float32)
    CNT = np.zeros((NM, 2, 5), dtype=np.int8)
    np.add.at(CNT, (m, e, l), 1)
    for idx, col in enumerate(names):
        vals = df[col].fillna(0.0).to_numpy(np.float32)
        np.add.at(T[idx], (m, e, l), vals)

    dur = np.maximum(D["Z"]["D_dur_min"].astype(np.float32), 1.0)
    ok_pair = (CNT[:, 0, :] == 1) & (CNT[:, 1, :] == 1)
    ok15 = ok_pair & (T[0, :, 0, :] > 0) & (T[0, :, 1, :] > 0) & (dur >= 15.0)[:, None]
    ok20 = ok15 & (T[2, :, 0, :] > 0) & (T[2, :, 1, :] > 0) & (dur >= 20.0)[:, None]

    dg15 = np.where(ok15, (T[0, :, 0, :] - T[0, :, 1, :]) / 1000.0, np.nan)
    dx15 = np.where(ok15, (T[1, :, 0, :] - T[1, :, 1, :]) / 1000.0, np.nan)
    dg20 = np.where(ok20, (T[2, :, 0, :] - T[2, :, 1, :]) / 1000.0, np.nan)
    dx20 = np.where(ok20, (T[3, :, 0, :] - T[3, :, 1, :]) / 1000.0, np.nan)
    eff = np.where(
        ok_pair,
        (T[4, :, 0, :] - T[5, :, 0, :]) - (T[4, :, 1, :] - T[5, :, 1, :]),
        np.nan,
    )
    vis = np.where(ok_pair, (T[6, :, 0, :] - T[6, :, 1, :]) / dur[:, None], np.nan)
    return dg15, dx15, ok15, dg20, dx20, ok20, eff, vis, ok_pair


def _run_1v1_kalman(order, ok_l, z_l, pid_mat, c_mat, nc, k_cl=20.0, k_pr=15.0):
    NM = len(pid_mat)
    n_p = max(max(row) for row in pid_mat) + 1
    cl_s = np.zeros((nc, 5), dtype=np.float64)
    cl_n = np.zeros((nc, 5), dtype=np.float64)
    pr_s = np.zeros((5, nc, nc), dtype=np.float64)
    pr_n = np.zeros((5, nc, nc), dtype=np.float64)

    mu = [0.0] * n_p
    var = [0.36] * n_p
    OBS_V, Q_S, MIN_VS = 0.85, 0.008, 0.04

    aggs = np.zeros((NM, 3), dtype=np.float32)  # mean, min, max
    by_lane = np.zeros((NM, 5), dtype=np.float32)

    for m in order:
        ps = pid_mat[m]
        cb, cr = c_mat[m][0], c_mat[m][1]
        mb = [mu[ps[l]] for l in range(5)]
        mr = [mu[ps[5 + l]] for l in range(5)]
        aggs[m, 0] = (sum(mb) - sum(mr)) * 0.2
        aggs[m, 1] = min(mb) - min(mr)
        aggs[m, 2] = max(mb) - max(mr)
        for l in range(5):
            by_lane[m, l] = mb[l] - mr[l]

        om = ok_l[m]
        zm = z_l[m]
        for l in range(5):
            if not om[l]:
                continue
            z_obs = zm[l]
            c1, c2 = cb[l], cr[l]
            exp_ind = (cl_s[c1, l] / (cl_n[c1, l] + k_cl)) - (cl_s[c2, l] / (cl_n[c2, l] + k_cl))
            exp_pair = (pr_s[l, c1, c2] + k_pr * exp_ind) / (pr_n[l, c1, c2] + k_pr)
            y_skill = z_obs - exp_pair

            pb, pr_ = ps[l], ps[5 + l]
            innov = y_skill - (mu[pb] - mu[pr_])
            tv = var[pb] + var[pr_] + OBS_V
            kb = var[pb] / tv
            kr = var[pr_] / tv
            mu[pb] += kb * innov
            mu[pr_] -= kr * innov
            var[pb] = max(MIN_VS, (1.0 - kb) * var[pb]) + Q_S
            var[pr_] = max(MIN_VS, (1.0 - kr) * var[pr_]) + Q_S

            cl_s[c1, l] += z_obs
            cl_n[c1, l] += 1.0
            cl_s[c2, l] -= z_obs
            cl_n[c2, l] += 1.0
            pr_s[l, c1, c2] += z_obs
            pr_n[l, c1, c2] += 1.0
            pr_s[l, c2, c1] -= z_obs
            pr_n[l, c2, c1] += 1.0

    return aggs, by_lane


def construir_bloques_g34(D: dict, rehacer: bool = False) -> dict[str, np.ndarray]:
    if CACHE.exists() and not rehacer:
        z = np.load(CACHE)
        if int(z["nm"][0]) == D["NM"]:
            return {k: z[k] for k in z.files if k != "nm"}

    t0 = time.time()
    print("[g_34] Extrayendo @15, @20, eficiencia y visión...", flush=True)
    NM = D["NM"]
    mids = D["Z"]["mids"].astype(str)
    t_arr = D["t"].astype(np.int64)
    y_arr = D["y"].astype(np.float64)
    champ = D["Z"]["champ"].reshape(NM, 2, 5)
    pre_m = (~D["val0"]) & (~D["post"]) & D["master"]

    dg15, dx15, ok15, dg20, dx20, ok20, eff, vis, ok_pair = _extraer_todo_por_carril(D)

    players = D["B"]["player"].reshape(NM, 10).astype(str)
    uniq_p, p_inv = np.unique(players.reshape(-1), return_inverse=True)
    pid_mat_arr = p_inv.reshape(NM, 10)
    pid_mat = pid_mat_arr.tolist()
    n_p = len(uniq_p)

    uc, c_inv = np.unique(champ.reshape(-1), return_inverse=True)
    c_mat = c_inv.reshape(NM, 2, 5).tolist()
    nc = len(uc)

    # Normalización pre_m para z15 y dz_15_20 (ortogonal a z15)
    sd_g15 = np.ones(5)
    sd_x15 = np.ones(5)
    sd_g20 = np.ones(5)
    sd_x20 = np.ones(5)
    for l in range(5):
        m15 = pre_m & ok15[:, l]
        sd_g15[l] = max(0.1, float(np.nanstd(dg15[m15, l])))
        sd_x15[l] = max(0.1, float(np.nanstd(dx15[m15, l])))
        m20 = pre_m & ok20[:, l]
        sd_g20[l] = max(0.1, float(np.nanstd(dg20[m20, l])))
        sd_x20[l] = max(0.1, float(np.nanstd(dx20[m20, l])))

    z15 = np.where(ok15, 0.65 * (dg15 / sd_g15[None, :]) + 0.35 * (dx15 / sd_x15[None, :]), 0.0)
    z20 = np.where(ok20, 0.65 * (dg20 / sd_g20[None, :]) + 0.35 * (dx20 / sd_x20[None, :]), 0.0)

    dz1520 = np.zeros((NM, 5), dtype=np.float64)
    for l in range(5):
        m20 = pre_m & ok20[:, l]
        beta = float(np.dot(z15[m20, l], z20[m20, l]) / max(1e-6, np.dot(z15[m20, l], z15[m20, l])))
        raw_d = np.where(ok20[:, l], z20[:, l] - beta * z15[:, l], 0.0)
        sd_d = max(0.1, float(np.std(raw_d[m20])))
        dz1520[:, l] = raw_d / sd_d

    # Normalización eff & vis ortogonales a outcome (y - 0.5) en pre_m
    yw = y_arr - 0.5
    z_eff = np.zeros((NM, 5), dtype=np.float64)
    z_vis = np.zeros((NM, 5), dtype=np.float64)
    for l in range(5):
        mp = pre_m & ok_pair[:, l]
        ae = float(np.dot(yw[mp], eff[mp, l]) / max(1e-6, np.dot(yw[mp], yw[mp])))
        re = np.where(ok_pair[:, l], eff[:, l] - ae * yw, 0.0)
        z_eff[:, l] = re / max(0.1, float(np.std(re[mp])))

        av = float(np.dot(yw[mp], vis[mp, l]) / max(1e-6, np.dot(yw[mp], yw[mp])))
        rv = np.where(ok_pair[:, l], vis[:, l] - av * yw, 0.0)
        z_vis[:, l] = rv / max(0.1, float(np.std(rv[mp])))

    order = np.lexsort((mids, t_arr))

    print("[g_34] Corriendo Kalmans MID_15_20, EFF, VIS (< m)...", flush=True)
    mid_aggs, _ = _run_1v1_kalman(order, ok20.tolist(), dz1520.tolist(), pid_mat, c_mat, nc)
    eff_aggs, eff_lanes = _run_1v1_kalman(order, ok_pair.tolist(), z_eff.tolist(), pid_mat, c_mat, nc)
    vis_aggs, vis_lanes = _run_1v1_kalman(order, ok_pair.tolist(), z_vis.tolist(), pid_mat, c_mat, nc)

    MID_15_20 = mid_aggs.astype(np.float32)  # (NM, 3): mean, min, max
    # carries_eff (BOT+MID+TOP mean), mean_eff, sup_jg_vis, mean_vis
    EFF_VIS = np.c_[
        eff_aggs[:, 0],
        (eff_lanes[:, 0] + eff_lanes[:, 2] + eff_lanes[:, 3]) / 3.0,
        vis_aggs[:, 0],
        (vis_lanes[:, 1] + vis_lanes[:, 4]) * 0.5,
    ].astype(np.float32)

    print("[g_34] Calculando y_skill causal @15 y resolviendo GRAPH_LSQR_15 (< chunk)...", flush=True)
    # y_skill causal igual que g_30, para alimentar aristas del grafo LSQR sin fuga
    cl_s = np.zeros((nc, 5), dtype=np.float64)
    cl_n = np.zeros((nc, 5), dtype=np.float64)
    pr_s = np.zeros((5, nc, nc), dtype=np.float64)
    pr_n = np.zeros((5, nc, nc), dtype=np.float64)
    y_skill_15 = np.zeros((NM, 5), dtype=np.float64)
    ok_l = ok15.tolist()
    z15_l = z15.tolist()
    for m in order:
        cb, cr = c_mat[m][0], c_mat[m][1]
        om, zm = ok_l[m], z15_l[m]
        for l in range(5):
            if not om[l]:
                continue
            z_obs = zm[l]
            c1, c2 = cb[l], cr[l]
            exp_ind = (cl_s[c1, l] / (cl_n[c1, l] + 20.0)) - (cl_s[c2, l] / (cl_n[c2, l] + 20.0))
            exp_pair = (pr_s[l, c1, c2] + 15.0 * exp_ind) / (pr_n[l, c1, c2] + 15.0)
            y_skill_15[m, l] = z_obs - exp_pair
            cl_s[c1, l] += z_obs; cl_n[c1, l] += 1.0
            cl_s[c2, l] -= z_obs; cl_n[c2, l] += 1.0
            pr_s[l, c1, c2] += z_obs; pr_n[l, c1, c2] += 1.0
            pr_s[l, c2, c1] -= z_obs; pr_n[l, c2, c1] += 1.0

    # Particionar pre_val0 en K=32 chunks cronológicos estrictos + 1 chunk final para val0 (todo >= 16/09 lee < 16/09)
    pre_v_ord = np.array([m for m in order if not D["val0"][m]], dtype=np.int32)
    post_v_ord = np.array([m for m in order if D["val0"][m]], dtype=np.int32)
    chunks = np.array_split(pre_v_ord, 32) + [post_v_ord]

    GRAPH_LSQR = np.zeros((NM, 4), dtype=np.float32)  # lane_mean, lane_min, lane_max, win_graph_mean
    TAU_MS = 35.0 * 86_400_000.0
    DAMP_LANE = float(np.sqrt(12.0))
    DAMP_WIN = float(np.sqrt(18.0))

    # Acumuladores de aristas históricas ya vistas en chunks < c
    l_r, l_c, l_v, l_y, l_t = [], [], [], [], []
    w_r, w_c, w_v, w_y, w_t = [], [], [], [], []
    row_l = 0
    row_w = 0

    for c_idx, ch in enumerate(chunks):
        if len(ch) == 0:
            continue
        if c_idx > 0 and row_l > 0:
            t_cut = float(t_arr[ch].min())
            # Resolver grafo 1v1 @15 con decaimiento temporal < t_cut
            wt_l = np.sqrt(np.exp(-np.maximum(0.0, t_cut - np.array(l_t, dtype=np.float64)) / TAU_MS))
            wt_l_e = np.repeat(wt_l, 2)
            A_l = coo_matrix((np.array(l_v) * wt_l_e, (np.array(l_r), np.array(l_c))), shape=(row_l, n_p)).tocsr()
            mu_l = lsqr(A_l, np.array(l_y) * wt_l, damp=DAMP_LANE, atol=1e-4, btol=1e-4, iter_lim=120)[0]

            # Resolver grafo de victoria por equipo (blue - red ~ y - 0.5) < t_cut
            wt_w = np.sqrt(np.exp(-np.maximum(0.0, t_cut - np.array(w_t, dtype=np.float64)) / TAU_MS))
            wt_w_e = np.repeat(wt_w, 10)
            A_w = coo_matrix((np.array(w_v) * wt_w_e, (np.array(w_r), np.array(w_c))), shape=(row_w, n_p)).tocsr()
            mu_w = lsqr(A_w, np.array(w_y) * wt_w, damp=DAMP_WIN, atol=1e-4, btol=1e-4, iter_lim=120)[0]

            mb = mu_l[pid_mat_arr[ch, :5]]
            mr = mu_l[pid_mat_arr[ch, 5:]]
            GRAPH_LSQR[ch, 0] = (mb.mean(1) - mr.mean(1))
            GRAPH_LSQR[ch, 1] = (mb.min(1) - mr.min(1))
            GRAPH_LSQR[ch, 2] = (mb.max(1) - mr.max(1))
            GRAPH_LSQR[ch, 3] = (mu_w[pid_mat_arr[ch, :5]].mean(1) - mu_w[pid_mat_arr[ch, 5:]].mean(1))

        # No agregar val0 al grafo (chunks[-1] es >= 16/09, nunca se agrega para Diamond-V / V)
        if c_idx == len(chunks) - 1:
            continue

        # Agregar partidas del chunk c_idx para los chunks posteriores
        for m in ch:
            tm = int(t_arr[m])
            ps = pid_mat[m]
            # Arista de victoria
            w_r.extend([row_w] * 10)
            w_c.extend(ps)
            w_v.extend([0.2] * 5 + [-0.2] * 5)
            w_y.append(y_arr[m] - 0.5)
            w_t.append(tm)
            row_w += 1
            # Aristas 1v1 @15 válidas
            for l in range(5):
                if ok15[m, l]:
                    l_r.extend((row_l, row_l))
                    l_c.extend((ps[l], ps[5 + l]))
                    l_v.extend((1.0, -1.0))
                    l_y.append(y_skill_15[m, l])
                    l_t.append(tm)
                    row_l += 1

    np.savez_compressed(CACHE, nm=np.array([NM]), GRAPH_LSQR=GRAPH_LSQR, MID_15_20=MID_15_20, EFF_VIS=EFF_VIS)
    print(f"[g_34] Caché guardada en {time.time() - t0:.1f}s", flush=True)
    return {"GRAPH_LSQR": GRAPH_LSQR, "MID_15_20": MID_15_20, "EFF_VIS": EFF_VIS}


if __name__ == "__main__":
    D = cargar()
    NM, ms = D["NM"], D["master"]
    X9 = extras_referencia(D, escalon=9)

    res = construir_bloques_g34(D)
    GR = res["GRAPH_LSQR"]
    MID = res["MID_15_20"]
    EV = res["EFF_VIS"]
    TODO_ORT = np.c_[GR, MID, EV].astype(np.float32)
    PL = TODO_ORT[np.random.default_rng(43).permutation(NM)]

    def ml(X):
        return lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=X, extra_post=False, con_mapa=False)

    VAR = {
        "Escalón 9 oficial": ml(X9),
        "+ GRAPH_LSQR (4)": ml(np.c_[X9, GR]),
        "+ MID_15_20 (3)": ml(np.c_[X9, MID]),
        "+ EFF_VIS (4)": ml(np.c_[X9, EV]),
        "+ TODO G34 (11)": ml(np.c_[X9, TODO_ORT]),
        "placebo G34 (11)": ml(np.c_[X9, PL]),
    }
    P = dict(Parallel(n_jobs=2, backend="threading")(delayed(lambda n: (n, evaluar(D, VAR[n])))(n) for n in VAR))
    print("\n=== VS ESCALÓN 9 OFICIAL (V 0.6739) ===", flush=True)
    mostrar(comparar(D, P, "Escalón 9 oficial"))

    vi = np.flatnonzero(D["val"])
    yv = D["y"][vi]
    print("\n--- LogLoss en V Master+ ---", flush=True)
    for n in VAR:
        print(f"  {n:24s}: V AUC {roc_auc_score(yv, P[n][1]):.4f} | LL {log_loss(yv, P[n][1]):.5f}", flush=True)

    di = np.flatnonzero(D["diam_v"])
    yd = D["y"][di]
    B = np.random.default_rng(1).integers(0, len(di), size=(300, len(di)))
    qd9 = P["Escalón 9 oficial"][0][di]
    print("\n--- Chequeo ciego Diamond-V (10.506 partidas) ---", flush=True)
    for n in VAR:
        if n == "Escalón 9 oficial":
            continue
        pd_ = P[n][0][di]
        d9 = np.array([roc_auc_score(yd[b], pd_[b]) - roc_auc_score(yd[b], qd9[b]) for b in B])
        print(f"  {n:24s}: {roc_auc_score(yd, pd_):.4f} | vs9 {roc_auc_score(yd, pd_) - roc_auc_score(yd, qd9):+.4f} [{np.percentile(d9, 2.5):+.4f}; {np.percentile(d9, 97.5):+.4f}]", flush=True)
