"""Idea G39 (`docs/BRAINSTORM_NOCHE.md`): Grafo 2v2 y cruces entre líneas `@15` (`MATCHUP_2V2_15`),
estrictamente causal `< m` en orden cronológico `(t, match_id)` y ortogonalizado respecto al
1v1 de `MATCHUP_EXP_MARGIN` (Escalón 9):

  1) BOT_2V2 (4 cols):
     - Sinergia aliada Bot+Sup `(c_bot, c_sup)` en margen combinado `@15` (`g15`, `xp15`) y winrate
       descontando fuerza individual previa de `(c_bot, BOT)` y `(c_sup, SUP)`.
     - Contracruce cruzado `Bot vs Enemy_Sup` `(c_bot, c_esup)` en `@15` (complementa el 1v1
       `Bot vs Bot` y `Sup vs Sup` de `g_23` con los otros 2 bordes del carril 2v2).
  2) MID_JG (3 cols):
     - Sinergia aliada Mid+Jungla `(c_jg, c_mid)` en margen `@15` (`g15`, `xp15`) y winrate.
  3) JG_PRESSURE (3 cols):
     - Presión cruzada Jungla enemigo vs laners `(c_jg, c_elane)` en `@15` (`g15` promedio en
       TOP/MID/BOT, `g15` en MID y `g15` en BOT, capturando campeones vulnerables a ganks/dives).

Todo medido sobre **Escalón 9 oficial (`V 0,6739`)**, con placebo permutado y chequeo ciego Diamond-V.
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
from sklearn.metrics import roc_auc_score

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
from src.features.margin import ORDEN  # noqa: E402

CACHE = REPO / "data" / "analysis" / "brainstorm" / "gemini_g29_2v2.npz"
LANES = ["BOTTOM", "JUNGLE", "MIDDLE", "TOP", "UTILITY"]  # 0=BOT, 1=JG, 2=MID, 3=TOP, 4=SUP


def _extraer_15_por_carril(D: dict):
    """Extrae gold_15 y xp_15 por `(match, team 0/1, lane 0..4)` alineados con D['Z']['mids']."""
    mids = D["Z"]["mids"].astype(str)
    NM = D["NM"]
    pos = pd.Series(np.arange(NM, dtype=np.int32), index=mids)
    idx_l = {l: i for i, l in enumerate(LANES)}

    df = pd.read_csv(
        REPO / "data" / "splits" / "train" / "full_dataset.csv",
        usecols=["match_id", "game_creation", "team_id", "lane", "gold_15", "xp_15", "game_duration"],
    )
    df = df[df.lane.isin(LANES)].copy()
    df["m"] = df["match_id"].astype(str).map(pos)
    df = df.dropna(subset=["m"]).copy()
    df["m"] = df["m"].astype(np.int32)
    df["l"] = df["lane"].map(idx_l).astype(np.int32)
    df["e"] = (df["team_id"] == 200).astype(np.int32)

    G15 = np.zeros((NM, 2, 5), dtype=np.float32)
    X15 = np.zeros((NM, 2, 5), dtype=np.float32)
    CNT = np.zeros((NM, 2, 5), dtype=np.int8)

    m = df["m"].to_numpy()
    e = df["e"].to_numpy()
    l = df["l"].to_numpy()
    g = df["gold_15"].fillna(0.0).to_numpy(np.float32)
    x = df["xp_15"].fillna(0.0).to_numpy(np.float32)
    np.add.at(G15, (m, e, l), g)
    np.add.at(X15, (m, e, l), x)
    np.add.at(CNT, (m, e, l), 1)

    dur = D["Z"]["D_dur_min"]
    ok = (CNT[:, 0, :] == 1) & (CNT[:, 1, :] == 1) & (G15[:, 0, :] > 0) & (G15[:, 1, :] > 0) & (dur >= 15.0)[:, None]
    dg = np.where(ok, (G15[:, 0, :] - G15[:, 1, :]) / 1000.0, np.nan).astype(np.float32)
    dx = np.where(ok, (X15[:, 0, :] - X15[:, 1, :]) / 1000.0, np.nan).astype(np.float32)
    return dg, dx, ok


def construir_matchup_2v2(D: dict, rehacer: bool = False) -> dict[str, np.ndarray]:
    if CACHE.exists() and not rehacer:
        z = np.load(CACHE)
        if int(z["nm"][0]) == D["NM"]:
            return {k: z[k] for k in z.files if k != "nm"}

    t0 = time.time()
    print("[g_29] Construyendo MATCHUP_2V2_15 causal...", flush=True)
    NM = D["NM"]
    mids = D["Z"]["mids"].astype(str)
    t_arr = D["t"].astype(np.int64)
    y_arr = D["y"].astype(np.int8)
    champ = D["Z"]["champ"].reshape(NM, 2, 5)
    dg, dx, ok = _extraer_15_por_carril(D)

    # Remap campeones a 0..n_c-1
    uc, inv = np.unique(champ, return_inverse=True)
    c_idx = inv.reshape(NM, 2, 5)
    nc = len(uc)

    # Baselines individuales por (champ, lane) en cada métrica para ortogonalizar el residuo 2v2
    # Lanes: 0=BOT, 1=JG, 2=MID, 3=TOP, 4=SUP
    ind_w_s = np.zeros((nc, 5), dtype=np.float64)
    ind_w_n = np.zeros((nc, 5), dtype=np.float64)
    ind_g_s = np.zeros((nc, 5), dtype=np.float64)
    ind_g_n = np.zeros((nc, 5), dtype=np.float64)
    ind_x_s = np.zeros((nc, 5), dtype=np.float64)

    # Pares aliados Bot+Sup: (c_bot, c_sup)
    bs_g_s = np.zeros((nc, nc), dtype=np.float64)
    bs_g_n = np.zeros((nc, nc), dtype=np.float64)
    bs_x_s = np.zeros((nc, nc), dtype=np.float64)
    bs_w_s = np.zeros((nc, nc), dtype=np.float64)
    bs_w_n = np.zeros((nc, nc), dtype=np.float64)

    # Contracruce cruzado Bot vs Enemy_Sup: (c_bot, c_esup)
    bes_g_s = np.zeros((nc, nc), dtype=np.float64)
    bes_g_n = np.zeros((nc, nc), dtype=np.float64)
    bes_x_s = np.zeros((nc, nc), dtype=np.float64)

    # Sinergia aliada Mid+JG: (c_jg, c_mid)
    jm_g_s = np.zeros((nc, nc), dtype=np.float64)
    jm_g_n = np.zeros((nc, nc), dtype=np.float64)
    jm_x_s = np.zeros((nc, nc), dtype=np.float64)
    jm_w_s = np.zeros((nc, nc), dtype=np.float64)
    jm_w_n = np.zeros((nc, nc), dtype=np.float64)

    # Presión Jungla vs Enemy Laner (top/mid/bot separados por carril l_opp in {0,2,3})
    je_g_s = np.zeros((3, nc, nc), dtype=np.float64)  # axis 0: 0->BOT, 1->MID, 2->TOP
    je_g_n = np.zeros((3, nc, nc), dtype=np.float64)

    C_2V2 = 15.0
    C_WIN = 20.0
    BOT_2V2 = np.zeros((NM, 4), dtype=np.float32)
    MID_JG = np.zeros((NM, 3), dtype=np.float32)
    JG_PRES = np.zeros((NM, 3), dtype=np.float32)

    order = np.lexsort((mids, t_arr))
    c_l = c_idx.tolist()
    dg_l = dg.tolist()
    dx_l = dx.tolist()
    ok_l = ok.tolist()

    for m in order:
        yb = float(y_arr[m])
        cb, cr = c_l[m][0], c_l[m][1]
        ok_m = ok_l[m]
        dg_m, dx_m = dg_l[m], dx_l[m]

        # ---- 1. Leer valores causales (< m) ----
        # Fuerza individual previa (win residual centrado en 0)
        def wr_ind(ch, l):
            return ind_w_s[ch, l] / (ind_w_n[ch, l] + C_WIN)

        def g_ind(ch, l):
            return ind_g_s[ch, l] / (ind_g_n[ch, l] + C_2V2)

        def x_ind(ch, l):
            return ind_x_s[ch, l] / (ind_g_n[ch, l] + C_2V2)

        bb, jb, mb, tb, sb = cb[0], cb[1], cb[2], cb[3], cb[4]
        br, jr, mr, tr_, sr = cr[0], cr[1], cr[2], cr[3], cr[4]

        # Bot+Sup ally synergy (Azul - Rojo)
        bs_g = (bs_g_s[bb, sb] / (bs_g_n[bb, sb] + C_2V2)) - (bs_g_s[br, sr] / (bs_g_n[br, sr] + C_2V2))
        bs_x = (bs_x_s[bb, sb] / (bs_g_n[bb, sb] + C_2V2)) - (bs_x_s[br, sr] / (bs_g_n[br, sr] + C_2V2))
        bs_w = (bs_w_s[bb, sb] / (bs_w_n[bb, sb] + C_WIN)) - (bs_w_s[br, sr] / (bs_w_n[br, sr] + C_WIN))
        # Bot vs Enemy_Sup cross-counter (symetric: B.bot vs R.sup MINUS R.bot vs B.sup)
        bes_g = (bes_g_s[bb, sr] / (bes_g_n[bb, sr] + C_2V2)) - (bes_g_s[br, sb] / (bes_g_n[br, sb] + C_2V2))
        BOT_2V2[m, 0] = bs_g
        BOT_2V2[m, 1] = bs_x
        BOT_2V2[m, 2] = bs_w
        BOT_2V2[m, 3] = bes_g

        # Mid+JG ally synergy
        jm_g = (jm_g_s[jb, mb] / (jm_g_n[jb, mb] + C_2V2)) - (jm_g_s[jr, mr] / (jm_g_n[jr, mr] + C_2V2))
        jm_x = (jm_x_s[jb, mb] / (jm_g_n[jb, mb] + C_2V2)) - (jm_x_s[jr, mr] / (jm_g_n[jr, mr] + C_2V2))
        jm_w = (jm_w_s[jb, mb] / (jm_w_n[jb, mb] + C_WIN)) - (jm_w_s[jr, mr] / (jm_w_n[jr, mr] + C_WIN))
        MID_JG[m, 0] = jm_g
        MID_JG[m, 1] = jm_x
        MID_JG[m, 2] = jm_w

        # Jungla vs Enemy Laners (BOT=0->slot0, MID=2->slot1, TOP=3->slot2)
        jp_bot = (je_g_s[0, jb, br] / (je_g_n[0, jb, br] + C_2V2)) - (je_g_s[0, jr, bb] / (je_g_n[0, jr, bb] + C_2V2))
        jp_mid = (je_g_s[1, jb, mr] / (je_g_n[1, jb, mr] + C_2V2)) - (je_g_s[1, jr, mb] / (je_g_n[1, jr, mb] + C_2V2))
        jp_top = (je_g_s[2, jb, tr_] / (je_g_n[2, jb, tr_] + C_2V2)) - (je_g_s[2, jr, tb] / (je_g_n[2, jr, tb] + C_2V2))
        JG_PRES[m, 0] = (jp_bot + jp_mid + jp_top) / 3.0
        JG_PRES[m, 1] = jp_mid
        JG_PRES[m, 2] = jp_bot

        # ---- 2. Actualizar contadores DESPUÉS ----
        # Winrate centered in [-0.5, +0.5]
        wy_b = yb - 0.5
        wy_r = -wy_b

        # Residuo de winrate descontando individuales previos
        bs_w_res_b = wy_b - (wr_ind(bb, 0) + wr_ind(sb, 4) - wr_ind(br, 0) - wr_ind(sr, 4))
        bs_w_res_r = -bs_w_res_b
        bs_w_s[bb, sb] += bs_w_res_b
        bs_w_n[bb, sb] += 1.0
        bs_w_s[br, sr] += bs_w_res_r
        bs_w_n[br, sr] += 1.0

        jm_w_res_b = wy_b - (wr_ind(jb, 1) + wr_ind(mb, 2) - wr_ind(jr, 1) - wr_ind(mr, 2))
        jm_w_s[jb, mb] += jm_w_res_b
        jm_w_n[jb, mb] += 1.0
        jm_w_s[jr, mr] -= jm_w_res_b
        jm_w_n[jr, mr] += 1.0

        for l in range(5):
            ind_w_s[cb[l], l] += wy_b
            ind_w_n[cb[l], l] += 1.0
            ind_w_s[cr[l], l] += wy_r
            ind_w_n[cr[l], l] += 1.0

        # Margen @15 por 2v2 y cruces
        if ok_m[0] and ok_m[4]:  # BOT & SUP válidos
            g_2v2 = 0.5 * (dg_m[0] + dg_m[4])
            x_2v2 = 0.5 * (dx_m[0] + dx_m[4])
            exp_g = 0.5 * (g_ind(bb, 0) + g_ind(sb, 4) - g_ind(br, 0) - g_ind(sr, 4))
            exp_x = 0.5 * (x_ind(bb, 0) + x_ind(sb, 4) - x_ind(br, 0) - x_ind(sr, 4))
            res_g = g_2v2 - exp_g
            res_x = x_2v2 - exp_x

            bs_g_s[bb, sb] += res_g
            bs_g_n[bb, sb] += 1.0
            bs_g_s[br, sr] -= res_g
            bs_g_n[br, sr] += 1.0
            bs_x_s[bb, sb] += res_x
            bs_x_s[br, sr] -= res_x

            # Contracruce Bot vs Enemy_Sup sobre el oro combinado de Botlane
            bes_g_s[bb, sr] += res_g
            bes_g_n[bb, sr] += 1.0
            bes_g_s[br, sb] -= res_g
            bes_g_n[br, sb] += 1.0

        if ok_m[1] and ok_m[2]:  # JG & MID válidos
            g_jm = 0.5 * (dg_m[1] + dg_m[2])
            x_jm = 0.5 * (dx_m[1] + dx_m[2])
            exp_gj = 0.5 * (g_ind(jb, 1) + g_ind(mb, 2) - g_ind(jr, 1) - g_ind(mr, 2))
            exp_xj = 0.5 * (x_ind(jb, 1) + x_ind(mb, 2) - x_ind(jr, 1) - x_ind(mr, 2))
            res_gj = g_jm - exp_gj
            res_xj = x_jm - exp_xj
            jm_g_s[jb, mb] += res_gj
            jm_g_n[jb, mb] += 1.0
            jm_g_s[jr, mr] -= res_gj
            jm_g_n[jr, mr] += 1.0
            jm_x_s[jb, mb] += res_xj
            jm_x_s[jr, mr] -= res_xj

        for k_idx, l_op in ((0, 0), (1, 2), (2, 3)):
            if ok_m[l_op]:
                res_jl = dg_m[l_op] - (g_ind(cb[l_op], l_op) - g_ind(cr[l_op], l_op))
                je_g_s[k_idx, jb, cr[l_op]] += res_jl
                je_g_n[k_idx, jb, cr[l_op]] += 1.0
                je_g_s[k_idx, jr, cb[l_op]] -= res_jl
                je_g_n[k_idx, jr, cb[l_op]] += 1.0

        for l in range(5):
            if ok_m[l]:
                g_val = dg_m[l]
                x_val = dx_m[l]
                ind_g_s[cb[l], l] += g_val
                ind_g_n[cb[l], l] += 1.0
                ind_g_s[cr[l], l] -= g_val
                ind_g_n[cr[l], l] += 1.0
                ind_x_s[cb[l], l] += x_val
                ind_x_s[cr[l], l] -= x_val

    BOT_2V2 = BOT_2V2.astype(np.float32)
    MID_JG = MID_JG.astype(np.float32)
    JG_PRES = JG_PRES.astype(np.float32)
    np.savez_compressed(CACHE, nm=np.array([NM]), BOT_2V2=BOT_2V2, MID_JG=MID_JG, JG_PRES=JG_PRES)
    print(f"[g_29] Caché guardada en {time.time() - t0:.1f}s", flush=True)
    return {"BOT_2V2": BOT_2V2, "MID_JG": MID_JG, "JG_PRES": JG_PRES}


if __name__ == "__main__":
    D = cargar()
    NM, ms = D["NM"], D["master"]
    X9 = extras_referencia(D, escalon=9)
    res = construir_matchup_2v2(D)
    BOT_2V2, MID_JG, JG_PRES = res["BOT_2V2"], res["MID_JG"], res["JG_PRES"]
    TODO = np.c_[BOT_2V2, MID_JG, JG_PRES].astype(np.float32)
    PL = TODO[np.random.default_rng(19).permutation(NM)]

    def ml(X):
        return lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=X, extra_post=False, con_mapa=False)

    VAR = {
        "Escalón 9 oficial": ml(X9),
        "+ BOT_2V2 (4)": ml(np.c_[X9, BOT_2V2]),
        "+ MID_JG (3)": ml(np.c_[X9, MID_JG]),
        "+ JG_PRES (3)": ml(np.c_[X9, JG_PRES]),
        "+ TODO 2V2 (10)": ml(np.c_[X9, TODO]),
        "placebo 2V2": ml(np.c_[X9, PL]),
    }
    P = dict(Parallel(n_jobs=2, backend="threading")(delayed(lambda n: (n, evaluar(D, VAR[n])))(n) for n in VAR))
    mostrar(comparar(D, P, "Escalón 9 oficial"))

    di = np.flatnonzero(D["diam_v"])
    yd = D["y"][di]
    B = np.random.default_rng(1).integers(0, len(di), size=(300, len(di)))
    qd = P["Escalón 9 oficial"][0][di]
    for n in VAR:
        if n == "Escalón 9 oficial":
            continue
        pd_ = P[n][0][di]
        d = np.array([roc_auc_score(yd[b], pd_[b]) - roc_auc_score(yd[b], qd[b]) for b in B])
        print(f"  Diamond-V {n:18s}: {roc_auc_score(yd, pd_):.4f} (Δ {roc_auc_score(yd, pd_) - roc_auc_score(yd, qd):+.4f} [{np.percentile(d, 2.5):+.4f}; {np.percentile(d, 97.5):+.4f}])")
