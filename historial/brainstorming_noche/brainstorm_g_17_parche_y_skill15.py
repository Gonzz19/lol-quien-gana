"""Ideas G33-G35 del brainstorming (docs/BRAINSTORM_NOCHE.md):
Evaluadas sobre la referencia nueva **Escalón 4b + g11_puro** (o Escalón 4b) con causalidad estricta `< t`:

1. **PATCH_SHIFT (6 cols):** Deriva de fuerza por parche exacto (`game_version` major.minor) y EWMA 7d real
   de `(campeón, línea)` DESCONTANDO la fuerza histórica acumulada `(campeón, línea)` del split (`K=25/35`).
   En `V`, `65%` es `16.18` (presente en `A2..A4`) y `35%` es `16.19`, mientras el one-hot `DRAFT` promedia
   desde principios de año (`16.10..16.17`).
2. **KALMAN_IND15 (8 cols):** TrueSkill causal individual a minuto 15 (`gold_15`, `xp_15` de cada línea exacta
   `azul_l - rojo_l`) DESCONTANDO la expectativa causal previa del enfrentamiento `(c_azul, c_rojo, línea)`
   y ajustando por matchmaking pull de los 9 jugadores del lobby (`other9 - r_ind`).
3. **ROLE_SWAP (4 cols):** `participant_id % 5` codifica el rol asignado en Champion Select (`0=TOP,1=JG,2=MID,3=BOT,4=SUP`),
   distinto de `lane` cuando hubo swap de líneas; mide `swap`, `offrole sin swap` e historial causal de swaps.

Caché guardada en `data/analysis/brainstorm/gemini_g17.npz`.
"""
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.brainstorm_base import ajustar_predecir, cargar, comparar, evaluar, extras_referencia, mostrar  # noqa: E402

CACHE = REPO / "data" / "analysis" / "brainstorm" / "gemini_g17.npz"
CACHE_G11 = REPO / "data" / "analysis" / "brainstorm" / "gemini_g11.npz"

# Mapeo alfabetico D["B"]/D["Z"]: 0=BOTTOM, 1=JUNGLE, 2=MIDDLE, 3=TOP, 4=UTILITY
# En Match-V5 participant_id 1..5 / 6..10 siguen TOP(0), JG(1), MID(2), BOT(3), SUP(4)
PID_TO_ALPHA = {0: 3, 1: 1, 2: 2, 3: 0, 4: 4}


def construir_cache_g17(D: dict, rehacer: bool = False) -> dict[str, np.ndarray]:
    if CACHE.exists() and not rehacer:
        z = np.load(CACHE)
        if int(z["nm"][0]) == D["NM"]:
            return {k: z[k] for k in z.files if k != "nm"}

    t0 = time.time()
    print("Construyendo caché gemini_g17.npz...", flush=True)
    NM = D["NM"]
    Z = D["Z"]
    mids = Z["mids"].astype(str)
    t_arr = D["t"].astype(np.int64)
    y_arr = D["y"].astype(np.int8)
    champ = Z["champ"].reshape(NM, 10).astype(np.int32)
    dur_min = Z["D_dur_min"].astype(np.float32)

    players = D["B"]["player"].reshape(NM, 10).astype(str)
    _, p_inv = np.unique(players.reshape(-1), return_inverse=True)
    pid_mat = p_inv.reshape(NM, 10).tolist()
    n_p = int(p_inv.max()) + 1

    cl_pairs = champ * 5 + np.tile(np.arange(5), 2)[None, :]
    _, cl_inv = np.unique(cl_pairs.reshape(-1), return_inverse=True)
    cl_mat = cl_inv.reshape(NM, 10).tolist()
    n_cl = int(cl_inv.max()) + 1

    _, c_inv = np.unique(champ.reshape(-1), return_inverse=True)
    c_mat = c_inv.reshape(NM, 10).tolist()
    n_c = int(c_inv.max()) + 1

    # Cargar game_version, participant_id, gold_15, xp_15 de full_dataset.csv
    cols_csv = ["match_id", "player_id", "game_version", "participant_id", "gold_15", "xp_15"]
    df = pd.read_csv(REPO / "data" / "splits" / "train" / "full_dataset.csv", usecols=cols_csv)
    df = df.drop_duplicates(["match_id", "player_id"])

    # Patch por partida (major.minor)
    patch_ser = df.groupby("match_id")["game_version"].first().astype(str).str.extract(r"(\d+\.\d+)")[0].fillna("0.0")
    patch_m_str = pd.Series(mids).map(patch_ser).fillna("0.0").to_numpy()
    _, patch_inv = np.unique(patch_m_str, return_inverse=True)
    patch_id = patch_inv.tolist()

    keys = pd.DataFrame({
        "match_id": np.repeat(mids, 10),
        "player_id": players.reshape(-1),
        "_pos": np.arange(NM * 10, dtype=np.int64),
    })
    M = keys.merge(df[["match_id", "player_id", "participant_id", "gold_15", "xp_15"]],
                   on=["match_id", "player_id"], how="left").sort_values("_pos", kind="mergesort")
    del df, keys

    part_id = M["participant_id"].fillna(1).to_numpy(np.int32).reshape(NM, 10)
    assigned_alpha = np.vectorize(PID_TO_ALPHA.__getitem__)(np.mod(part_id - 1, 5))
    actual_alpha = np.tile(np.arange(5), 2)[None, :]
    is_swap_arr = (assigned_alpha != actual_alpha).astype(np.float32)

    g15 = M["gold_15"].to_numpy(np.float32).reshape(NM, 10)
    xp15 = M["xp_15"].to_numpy(np.float32).reshape(NM, 10)
    del M

    # Diferencia exacta por linea @15 (azul_l - rojo_l), validada >= 15 min
    valid15 = (dur_min >= 15.0) & np.all(np.isfinite(g15), axis=1) & np.all(np.isfinite(xp15), axis=1)
    lane_score15 = np.where(
        valid15[:, None],
        0.65 * (g15[:, :5] - g15[:, 5:]) / 1200.0 + 0.35 * (xp15[:, :5] - xp15[:, 5:]) / 1000.0,
        0.0,
    ).astype(np.float32)

    cols_d = {k: i for i, k in enumerate(D["cols"])}
    off_arr = D["Xj"][..., cols_d["offrole"]].reshape(NM, 10).astype(np.float32)

    PATCH_SHIFT = np.zeros((NM, 6), dtype=np.float32)
    KALMAN_IND15 = np.zeros((NM, 8), dtype=np.float32)
    ROLE_SWAP = np.zeros((NM, 4), dtype=np.float32)

    # Estados de PATCH_SHIFT
    w_cl = [0.0] * n_cl
    n_cl_cnt = [0.0] * n_cl
    w_c = [0.0] * n_c
    n_c_cnt = [0.0] * n_c
    w_patch_cl = defaultdict(float)
    n_patch_cl = defaultdict(float)
    w_patch_c = defaultdict(float)
    n_patch_c = defaultdict(float)
    ew_cl_w = [0.0] * n_cl
    ew_cl_n = [0.0] * n_cl
    ew_cl_last_t = [0] * n_cl
    TAU7_MS = 7.0 * 86400.0 * 1000.0

    # Estados de KALMAN_IND15
    mu_cl15 = [0.0] * n_cl
    n_cl15 = [0.0] * n_cl
    mu_mu15 = defaultdict(float)
    n_mu15 = defaultdict(float)
    r_ind = [0.0] * n_p
    r_ind_slow = [0.0] * n_p
    r_lane_ew = [0.0] * n_p
    r_lane_w = [0.0] * n_p
    cnt_p = [0] * n_p

    # Estados de swap
    sw_sum = [0.0] * n_p
    sw_cnt = [0.0] * n_p
    dec20 = math.exp(-math.log(2.0) / 20.0)

    order = np.lexsort((mids, t_arr))
    sw_l = is_swap_arr.tolist()
    off_l = off_arr.tolist()
    ls15_l = lane_score15.tolist()
    c_raw = champ.tolist()

    for m in order:
        tm = int(t_arr[m])
        pm = int(patch_id[m])
        yb = float(y_arr[m])
        ps = pid_mat[m]
        cls = cl_mat[m]
        cs = c_mat[m]
        craw = c_raw[m]
        v15 = bool(valid15[m])

        # ---- 1. PATCH_SHIFT causal ----
        dp_cl_s, dew_cl_s, dp_c_s = [0.0] * 10, [0.0] * 10, [0.0] * 10
        for i in range(10):
            cl = cls[i]
            ci = cs[i]
            w_all = (w_cl[cl] + 30.0 * 0.5) / (n_cl_cnt[cl] + 30.0)
            kpc = (pm, cl)
            w_p = (w_patch_cl[kpc] + 25.0 * w_all) / (n_patch_cl[kpc] + 25.0)
            dp_cl_s[i] = w_p - w_all

            # Decaimiento continuo 7d para EWMA
            dt = max(0, tm - ew_cl_last_t[cl]) if ew_cl_last_t[cl] > 0 else 0
            dec = math.exp(-math.log(2.0) * (dt / TAU7_MS)) if dt > 0 else 1.0
            wew = ew_cl_w[cl] * dec
            new = ew_cl_n[cl] * dec
            w_ew = (wew + 35.0 * w_all) / (new + 35.0)
            dew_cl_s[i] = w_ew - w_all

            wc_all = (w_c[ci] + 40.0 * 0.5) / (n_c_cnt[ci] + 40.0)
            kpcc = (pm, ci)
            wc_p = (w_patch_c[kpcc] + 35.0 * wc_all) / (n_patch_c[kpcc] + 35.0)
            dp_c_s[i] = wc_p - wc_all

        PATCH_SHIFT[m, 0] = sum(dp_cl_s[:5]) - sum(dp_cl_s[5:])
        PATCH_SHIFT[m, 1] = sum(dew_cl_s[:5]) - sum(dew_cl_s[5:])
        PATCH_SHIFT[m, 2] = sum(dp_c_s[:5]) - sum(dp_c_s[5:])
        PATCH_SHIFT[m, 3] = (dp_cl_s[1] + dp_cl_s[2]) - (dp_cl_s[6] + dp_cl_s[7])  # JG+MID
        PATCH_SHIFT[m, 4] = (dp_cl_s[0] + dp_cl_s[4]) - (dp_cl_s[5] + dp_cl_s[9])  # BOT+SUP
        PATCH_SHIFT[m, 5] = max(dp_cl_s[:5]) - max(dp_cl_s[5:])

        # ---- 2. KALMAN_IND15 causal ----
        rib = [r_ind[ps[i]] for i in range(5)]
        rir = [r_ind[ps[5 + i]] for i in range(5)]
        rib_s, rir_s = sorted(rib), sorted(rir)
        lrb = [r_lane_ew[ps[i]] / (r_lane_w[ps[i]] + 8.0) for i in range(5)]
        lrr = [r_lane_ew[ps[5 + i]] / (r_lane_w[ps[5 + i]] + 8.0) for i in range(5)]
        m_ib = sum(rib) * 0.2
        m_ir = sum(rir) * 0.2
        KALMAN_IND15[m, 0] = (m_ib - m_ir) / 100.0
        KALMAN_IND15[m, 1] = (rib_s[0] - rir_s[0]) / 100.0
        KALMAN_IND15[m, 2] = (rib_s[4] - rir_s[4]) / 100.0
        KALMAN_IND15[m, 3] = (sum(rib[i] - r_ind_slow[ps[i]] for i in range(5))
                              - sum(rir[i] - r_ind_slow[ps[5 + i]] for i in range(5))) * 0.2 / 100.0
        KALMAN_IND15[m, 4] = (sum(lrb) - sum(lrr)) * 0.2
        KALMAN_IND15[m, 5] = max(lrb) - max(lrr)
        KALMAN_IND15[m, 6] = min(lrb) - min(lrr)
        KALMAN_IND15[m, 7] = (lrb[1] + lrb[2]) - (lrr[1] + lrr[2])

        # ---- 3. ROLE_SWAP causal ----
        sw_row = sw_l[m]
        off_row = off_l[m]
        sw_hist = [sw_sum[ps[i]] / (sw_cnt[ps[i]] + 8.0) for i in range(10)]
        ROLE_SWAP[m, 0] = sum(sw_row[:5]) - sum(sw_row[5:])
        ROLE_SWAP[m, 1] = (sum(off_row[i] * (1.0 - sw_row[i]) for i in range(5))
                           - sum(off_row[5 + i] * (1.0 - sw_row[5 + i]) for i in range(5)))
        ROLE_SWAP[m, 2] = (sum(sw_hist[:5]) - sum(sw_hist[5:])) * 0.2
        ROLE_SWAP[m, 3] = (sw_row[1] * off_row[1]) - (sw_row[6] * off_row[6])

        # ---- 4. Actualizar estados DESPUÉS ----
        for i in range(10):
            cl = cls[i]
            ci = cs[i]
            wi = yb if i < 5 else (1.0 - yb)
            w_cl[cl] += wi
            n_cl_cnt[cl] += 1.0
            w_c[ci] += wi
            n_c_cnt[ci] += 1.0
            kpc = (pm, cl)
            w_patch_cl[kpc] += wi
            n_patch_cl[kpc] += 1.0
            kpcc = (pm, ci)
            w_patch_c[kpcc] += wi
            n_patch_c[kpcc] += 1.0

            dt = max(0, tm - ew_cl_last_t[cl]) if ew_cl_last_t[cl] > 0 else 0
            dec = math.exp(-math.log(2.0) * (dt / TAU7_MS)) if dt > 0 else 1.0
            ew_cl_w[cl] = ew_cl_w[cl] * dec + wi
            ew_cl_n[cl] = ew_cl_n[cl] * dec + 1.0
            ew_cl_last_t[cl] = tm

            p = ps[i]
            sw_sum[p] = sw_sum[p] * dec20 + sw_row[i]
            sw_cnt[p] = sw_cnt[p] * dec20 + 1.0

        # Expectativa causal @15 por matchup exacto y residuo individual por linea
        res_lane = [0.0] * 5
        ls_row = ls15_l[m]
        if v15:
            for l in range(5):
                clb, clr = cls[l], cls[5 + l]
                cb, cr = craw[l], craw[5 + l]
                eb = mu_cl15[clb] / (n_cl15[clb] + 12.0)
                er = mu_cl15[clr] / (n_cl15[clr] + 12.0)
                e_prior = eb - er
                kmu = (l, cb, cr)
                e_mu = (mu_mu15[kmu] + 15.0 * e_prior) / (n_mu15[kmu] + 15.0)
                act = float(ls_row[l])
                res_lane[l] = max(-2.2, min(2.2, act - e_mu))

                mu_cl15[clb] += act
                n_cl15[clb] += 1.0
                mu_cl15[clr] -= act
                n_cl15[clr] += 1.0
                mu_mu15[kmu] += act
                n_mu15[kmu] += 1.0

        diff_ind = (m_ib - m_ir) / 200.0 - 0.105
        p_hat_i = 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, diff_ind))))
        err_out = yb - p_hat_i
        lobby_ind = (m_ib + m_ir) * 0.5
        for slot in range(10):
            p = ps[slot]
            sgn = 1.0 if slot < 5 else -1.0
            lane = slot % 5
            n = cnt_p[p]
            lr_w = 28.0 / math.sqrt(1.0 + n / 25.0)
            lr_pull = 0.22 if n < 10 else 0.10 / math.sqrt(1.0 + (n - 10) / 35.0)
            other9 = (10.0 * lobby_ind - r_ind[p]) / 9.0
            if v15:
                ind_r = res_lane[lane] * sgn
                target_err = 0.65 * sgn * err_out + 0.35 * (0.35 * ind_r)
                r_lane_ew[p] = r_lane_ew[p] * dec20 + ind_r
                r_lane_w[p] = r_lane_w[p] * dec20 + 1.0
            else:
                target_err = sgn * err_out
            new_ind = r_ind[p] + lr_w * target_err + lr_pull * (other9 - r_ind[p])
            r_ind[p] = new_ind
            r_ind_slow[p] = 0.94 * r_ind_slow[p] + 0.06 * new_ind
            cnt_p[p] = n + 1

    out = {"PATCH_SHIFT": PATCH_SHIFT, "KALMAN_IND15": KALMAN_IND15, "ROLE_SWAP": ROLE_SWAP}
    np.savez_compressed(CACHE, nm=np.array([NM]), **out)
    print(f"Caché gemini_g17.npz lista en {time.time() - t0:.1f}s", flush=True)
    return out


def main():
    t0 = time.time()
    D = cargar()
    ms = D["master"]
    R6 = extras_referencia(D, escalon=6)
    G17 = construir_cache_g17(D)

    VAR = {
        "ref: Escalón 6 (sin mapa)": R6,
        "+ PATCH_SHIFT (6)": np.c_[R6, G17["PATCH_SHIFT"]],
        "+ KALMAN_IND15 (8)": np.c_[R6, G17["KALMAN_IND15"]],
        "+ ROLE_SWAP (4)": np.c_[R6, G17["ROLE_SWAP"]],
        "+ TODO g17 (18)": np.c_[R6, G17["PATCH_SHIFT"], G17["KALMAN_IND15"], G17["ROLE_SWAP"]],
    }

    def run_one(nom, X):
        X32 = np.asarray(X, dtype=np.float32)
        return nom, evaluar(D, lambda tr, te, M=X32: ajustar_predecir(D, tr[ms[tr]], te, extra=M, extra_post=False, con_mapa=False))

    P = dict(Parallel(n_jobs=2, backend="threading")(delayed(run_one)(n, X) for n, X in VAR.items()))
    mostrar(comparar(D, P, "ref: Escalón 6 (sin mapa)"))

    from sklearn.metrics import roc_auc_score
    di = np.flatnonzero(D["diam_v"])
    yd = D["y"][di]
    B = np.random.default_rng(1).integers(0, len(di), size=(300, len(di)))
    b0 = P["ref: Escalón 6 (sin mapa)"][0][di]
    print(f"\nChequeo Diamond-V ({len(di):,} partidas):")
    for nom in ["+ PATCH_SHIFT (6)", "+ KALMAN_IND15 (8)", "+ ROLE_SWAP (4)", "+ TODO g17 (18)"]:
        p = P[nom][0][di]
        d = np.array([roc_auc_score(yd[b], p[b]) - roc_auc_score(yd[b], b0[b]) for b in B])
        print(f"  {nom:24s} AUC {roc_auc_score(yd, p):.4f}  Δ {roc_auc_score(yd, p) - roc_auc_score(yd, b0):+.4f} "
              f"[{np.percentile(d, 2.5):+.4f}; {np.percentile(d, 97.5):+.4f}]")
    print(f"Listo en {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
