"""Idea G34 del brainstorming (`docs/BRAINSTORM_NOCHE.md`):
Señales estrictamente causales de expectativa de duelo 1v1 `@10/@15` y residuo de habilidad del
jugador ajustado por el enfrentamiento exacto `(champ_A vs champ_B, línea)` (`RES_DUELO_15`):

1. `MATCHUP_EXP_MARGIN` (`12` cols): diferencia esperada de `@10/@15` (`d_gold_15`, `d_xp_15`,
   `d_cs_15`, `d_gold_10`, más top/mid/jg/bot `@15`) entre `champ_A` y `champ_B` en la misma
   línea (< t), con simetría exacta Azul/Rojo y encogimiento empírico hacia `mu_cl(A) - mu_cl(B)`.
   `MATCHUPS` solo usaba winrate final, nunca el stomp temprano esperado en línea.
2. `RES_DUELO_15` (`10` cols + tensor `(NM, 2, 5, 8)`): residuo histórico causal del jugador
   `obs - E[margen | champ_A vs champ_B, línea, < t]` (EWMA rápida/lenta + min de equipo),
   corrigiendo la penalización injusta a jugadores de campeones débiles en línea (`Kayle vs Darius`)
   o inflados por picks dominantes de línea (`Draven/Renekton`).

Evaluado estrictamente s/Escalón 6 sin mapa (`V 0,6709`) + chequeo Diamond-V.
"""
import math
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.brainstorm_base import ajustar_predecir, cargar, comparar, evaluar, extras_referencia, mostrar  # noqa: E402
from src.features.margin import ORDEN  # noqa: E402

CACHE = REPO / "data" / "analysis" / "brainstorm" / "gemini_g23.npz"
LANES = ["BOTTOM", "JUNGLE", "MIDDLE", "TOP", "UTILITY"]


def construir_cache_g23(D: dict, rehacer: bool = False) -> dict[str, np.ndarray]:
    if CACHE.exists() and not rehacer:
        z = np.load(CACHE)
        if int(z["nm"][0]) == D["NM"]:
            return {k: z[k] for k in z.files if k != "nm"}

    t0 = time.time()
    print("Construyendo caché gemini_g23.npz (RES_DUELO_15 + MATCHUP_EXP_MARGIN)...", flush=True)
    NM = D["NM"]
    mids = D["Z"]["mids"].astype(str)
    t_arr = D["t"].astype(np.int64)
    fit0 = ~D["val0"]

    df = pd.read_csv(
        REPO / "data" / "splits" / "train" / "full_dataset.csv",
        usecols=[
            "match_id", "game_creation", "game_duration", "team_id", "lane", "player_id",
            "champion_id", "win", "gold_10", "gold_15", "xp_15", "cs_15",
        ],
    )
    df = df[df.lane.isin(LANES)]
    df = df[df.groupby(["match_id", "team_id"]).lane.transform("nunique") == 5]
    df = df[df.groupby("match_id").lane.transform("size") == 10]
    df = df[df.groupby("match_id").win.transform("sum") == 5].copy()
    df = df.sort_values(ORDEN).reset_index(drop=True)
    assert (df["match_id"].to_numpy()[::10] == mids).all()

    dur = df["game_duration"].to_numpy(np.float64).reshape(NM, 2, 5)[:, 0, 0]
    champ = df["champion_id"].to_numpy(np.int64).reshape(NM, 2, 5)
    pid_raw = df["player_id"].to_numpy().astype(str).reshape(NM, 2, 5)
    uniq_p, p_inv = np.unique(pid_raw.reshape(-1), return_inverse=True)
    pid = p_inv.reshape(NM, 2, 5).tolist()
    n_p = len(uniq_p)

    # 4 métricas de duelo directo (azul - rojo por línea): g15, xp15, cs15, g10
    def diff_mat(col: str, min_dur: float) -> np.ndarray:
        a = df[col].to_numpy(np.float64).reshape(NM, 2, 5)
        d = a[:, 0, :] - a[:, 1, :]
        ok = (dur >= min_dur)[:, None] & np.isfinite(d)
        sc = float(np.nanstd(d[fit0[:, None] & ok])) + 1e-6
        return np.where(ok, d / sc, np.nan)

    D4 = [diff_mat("gold_15", 900.0), diff_mat("xp_15", 900.0), diff_mat("cs_15", 900.0), diff_mat("gold_10", 600.0)]
    del df

    # Acumuladores por (campeon, linea) y por par canonico (min_c, max_c, l)
    cl_mat = (champ * 5 + np.arange(5)[None, None, :]).tolist()
    cl_sum = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    cl_cnt = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    pair_sum = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    pair_cnt = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    # EWMA causal por jugador para los 4 residuales de duelo (halving 16 y 40 partidas)
    r_fast = [[0.0] * n_p for _ in range(4)]
    r_slow = [[0.0] * n_p for _ in range(4)]
    r_cnt = [[0.0] * n_p for _ in range(4)]
    a_f = 1.0 - math.exp(-math.log(2.0) / 14.0)
    a_s = 1.0 - math.exp(-math.log(2.0) / 40.0)
    k_f, k_s, k_m, k_cl = 6.0, 10.0, 12.0, 20.0

    MATCHUP_EXP_MARGIN = np.zeros((NM, 12), dtype=np.float32)
    RES_DUELO_15 = np.zeros((NM, 10), dtype=np.float32)
    P_RES = np.zeros((NM, 2, 5, 8), dtype=np.float32)

    order = np.lexsort((mids, t_arr))
    ch_l = champ.tolist()
    d_lists = [m.tolist() for m in D4]

    for m in order:
        cb_row, cr_row = ch_l[m][0], ch_l[m][1]
        pb_row, pr_row = pid[m][0], pid[m][1]
        clb_row, clr_row = cl_mat[m][0], cl_mat[m][1]

        exp_m = [[0.0] * 5 for _ in range(4)]
        for l in range(5):
            cb, cr = cb_row[l], cr_row[l]
            clb, clr = clb_row[l], clr_row[l]
            if cb == cr:
                continue
            sgn = 1.0 if cb < cr else -1.0
            pk = (cb, cr, l) if cb < cr else (cr, cb, l)
            ps, pc = pair_sum[pk], pair_cnt[pk]
            cbs, cbc = cl_sum[clb], cl_cnt[clb]
            crs, crc = cl_sum[clr], cl_cnt[clr]
            for s_i in range(4):
                mu_b = cbs[s_i] / (cbc[s_i] + k_cl)
                mu_r = crs[s_i] / (crc[s_i] + k_cl)
                prior_dif = mu_b - mu_r
                exp_m[s_i][l] = (sgn * ps[s_i] + k_m * prior_dif) / (pc[s_i] + k_m)

        for s_i in range(4):
            MATCHUP_EXP_MARGIN[m, s_i] = sum(exp_m[s_i]) * 0.2
        MATCHUP_EXP_MARGIN[m, 4:9] = exp_m[0]               # g15 esperado por linea (5)
        MATCHUP_EXP_MARGIN[m, 9:12] = [exp_m[1][2], exp_m[1][1], exp_m[1][3]]  # xp15 mid/jg/top

        # Leer estado causal del jugador (fast & slow shrunk hacia 0)
        rf_b = [[0.0] * 5 for _ in range(4)]
        rf_r = [[0.0] * 5 for _ in range(4)]
        rs_b = [[0.0] * 5 for _ in range(4)]
        rs_r = [[0.0] * 5 for _ in range(4)]
        for l in range(5):
            pb_i, pr_i = pb_row[l], pr_row[l]
            for s_i in range(4):
                w_fb = r_cnt[s_i][pb_i] / (r_cnt[s_i][pb_i] + k_f)
                w_fr = r_cnt[s_i][pr_i] / (r_cnt[s_i][pr_i] + k_f)
                w_sb = r_cnt[s_i][pb_i] / (r_cnt[s_i][pb_i] + k_s)
                w_sr = r_cnt[s_i][pr_i] / (r_cnt[s_i][pr_i] + k_s)
                vfb, vfr = r_fast[s_i][pb_i] * w_fb, r_fast[s_i][pr_i] * w_fr
                vsb, vsr = r_slow[s_i][pb_i] * w_sb, r_slow[s_i][pr_i] * w_sr
                rf_b[s_i][l], rf_r[s_i][l] = vfb, vfr
                rs_b[s_i][l], rs_r[s_i][l] = vsb, vsr
                P_RES[m, 0, l, s_i] = vfb
                P_RES[m, 1, l, s_i] = vfr
                P_RES[m, 0, l, 4 + s_i] = vsb
                P_RES[m, 1, l, 4 + s_i] = vsr

        for s_i in range(4):
            RES_DUELO_15[m, s_i] = (sum(rf_b[s_i]) - sum(rf_r[s_i])) * 0.2
            RES_DUELO_15[m, 4 + s_i] = (sum(rs_b[s_i]) - sum(rs_r[s_i])) * 0.2
        RES_DUELO_15[m, 8] = min(rf_b[0]) - min(rf_r[0])
        RES_DUELO_15[m, 9] = max(rf_b[0]) - max(rf_r[0])

        # ---- Actualizar DESPUES de leer ----
        for s_i in range(4):
            obs_row = d_lists[s_i][m]
            for l in range(5):
                obs = obs_row[l]
                if not math.isfinite(obs):
                    continue
                cb, cr = cb_row[l], cr_row[l]
                clb, clr = clb_row[l], clr_row[l]
                cl_sum[clb][s_i] += obs
                cl_cnt[clb][s_i] += 1.0
                cl_sum[clr][s_i] -= obs
                cl_cnt[clr][s_i] += 1.0
                if cb != cr:
                    sgn = 1.0 if cb < cr else -1.0
                    pk = (cb, cr, l) if cb < cr else (cr, cb, l)
                    pair_sum[pk][s_i] += sgn * obs
                    pair_cnt[pk][s_i] += 1.0

                surp = obs - exp_m[s_i][l]
                pb_i, pr_i = pb_row[l], pr_row[l]
                r_fast[s_i][pb_i] = (1.0 - a_f) * r_fast[s_i][pb_i] + a_f * surp
                r_slow[s_i][pb_i] = (1.0 - a_s) * r_slow[s_i][pb_i] + a_s * surp
                r_cnt[s_i][pb_i] += 1.0
                r_fast[s_i][pr_i] = (1.0 - a_f) * r_fast[s_i][pr_i] - a_f * surp
                r_slow[s_i][pr_i] = (1.0 - a_s) * r_slow[s_i][pr_i] - a_s * surp
                r_cnt[s_i][pr_i] += 1.0

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE,
        nm=np.array([NM]),
        MATCHUP_EXP_MARGIN=MATCHUP_EXP_MARGIN,
        RES_DUELO_15=RES_DUELO_15,
        P_RES=P_RES,
    )
    print(f"  gemini_g23.npz listo en {time.time() - t0:.1f}s", flush=True)
    return {"MATCHUP_EXP_MARGIN": MATCHUP_EXP_MARGIN, "RES_DUELO_15": RES_DUELO_15, "P_RES": P_RES}


def main():
    t0 = time.time()
    D = cargar()
    G23 = construir_cache_g23(D)
    REF7 = extras_referencia(D, escalon=7)
    ms = D["master"]

    def solo_m(extra=None):
        tot = REF7 if extra is None else np.c_[REF7, extra].astype(np.float32)
        return lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=tot, extra_post=False, con_mapa=False)

    P = {"ref: Escalón 7 (sin mapa)": evaluar(D, solo_m())}
    P["+ MATCHUP_EXP_MARGIN (12)"] = evaluar(D, solo_m(G23["MATCHUP_EXP_MARGIN"]))
    P["+ RES_DUELO_15 (10)"] = evaluar(D, solo_m(G23["RES_DUELO_15"]))
    P["+ TODO g23 (22)"] = evaluar(D, solo_m(np.c_[G23["MATCHUP_EXP_MARGIN"], G23["RES_DUELO_15"]]))

    rng = np.random.default_rng(123)
    plac = G23["RES_DUELO_15"][rng.permutation(D["NM"])]
    P["Placebo g23 (10 perm)"] = evaluar(D, solo_m(plac))
    mostrar(comparar(D, P, "ref: Escalón 7 (sin mapa)"))

    fi_v, dv = D["fi_v"], D["diam_v"]
    ydv = D["y"][dv]
    p_dv_ref = solo_m()(fi_v, dv)
    auc_ref = roc_auc_score(ydv, p_dv_ref)
    rng = np.random.default_rng(77)
    boots = rng.integers(0, len(ydv), size=(1500, len(ydv)))
    print(f"\nChequeo ciego Diamond-V ({len(ydv):,} partidas): ref={auc_ref:.4f}")
    for nom, ex in [
        ("+ MATCHUP_EXP_MARGIN", G23["MATCHUP_EXP_MARGIN"]),
        ("+ RES_DUELO_15", G23["RES_DUELO_15"]),
        ("+ TODO g23", np.c_[G23["MATCHUP_EXP_MARGIN"], G23["RES_DUELO_15"]]),
    ]:
        p_dv = solo_m(ex)(fi_v, dv)
        auc_p = roc_auc_score(ydv, p_dv)
        ds = np.array([roc_auc_score(ydv[b], p_dv[b]) - roc_auc_score(ydv[b], p_dv_ref[b]) for b in boots])
        print(f"  {nom:24s} AUC {auc_p:.4f}  Δ {auc_p - auc_ref:+.4f} [{np.percentile(ds, 2.5):+.4f}; {np.percentile(ds, 97.5):+.4f}]")
    print(f"Listo en {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
