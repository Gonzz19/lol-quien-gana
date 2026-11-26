"""Idea G40 (`docs/BRAINSTORM_NOCHE.md`): **Filtro de Kalman 1v1 por carril `@15` (`LANE_KALMAN_15`)**,
descontando calidad del rival directo y ventaja esperada del cruce de campeones `< m`:

  - Mientras `MC/ML` (`margin.py`) promedia `d_gold_15` crudo del jugador sin descontar si
    enfrentó a un Challenger o un autofill, y `KALMAN_MRG` (`g_11`) actualiza a los 10 jugadores
    con una sola señal de equipo (penalizando a quien ganó línea si otra línea alimentó),
  - `LANE_KALMAN_15` evalúa cada duelo 1v1 (`BOT, JG, MID, TOP, SUP`) estandarizado por carril
    sobre `~val0 & ~post & master` (cero fuga), resta el esperado causal del cruce `(cA, cB, l)`
    leído antes de la partida, y actualiza la estimación Bayesiana `(mu_p, var_p)` del jugador
    frente a la fuerza estimada `mu_rival` de su oponente de carril:
      1) `LK_AGGS (4)`: `[media, min (eslabón débil), max, forma rápida q_fast]` (`Azul - Rojo`)
      2) `LK_LANES (6)`: `[dif por línea BOT..SUP (5), suma ponderada por precisión 1/sqrt(vB+vR)]`

Evaluado sobre **Escalón 9 oficial (`V 0,6739`)** con placebo permutado y chequeo ciego `Diamond-V`.
"""
import math
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
from pathlib import Path

import numpy as np
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
from historial.brainstorming_noche.brainstorm_g_29_matchup_2v2_15 import _extraer_15_por_carril  # noqa: E402

CACHE = REPO / "data" / "analysis" / "brainstorm" / "gemini_g30_lk15.npz"


def construir_lane_kalman_15(D: dict, rehacer: bool = False) -> dict[str, np.ndarray]:
    if CACHE.exists() and not rehacer:
        z = np.load(CACHE)
        if int(z["nm"][0]) == D["NM"]:
            return {k: z[k] for k in z.files if k != "nm"}

    t0 = time.time()
    print("[g_30] Construyendo LANE_KALMAN_15 causal (< m)...", flush=True)
    NM = D["NM"]
    mids = D["Z"]["mids"].astype(str)
    t_arr = D["t"].astype(np.int64)
    champ = D["Z"]["champ"].reshape(NM, 2, 5)
    dg, dx, ok = _extraer_15_por_carril(D)

    # Desvíos por carril fijados únicamente en pre-foto1 Master+ (~val0 & ~post) para homogeneizar roles
    pre_m = (~D["val0"]) & (~D["post"]) & D["master"]
    sd_g = np.ones(5, dtype=np.float64)
    sd_x = np.ones(5, dtype=np.float64)
    for l in range(5):
        m_l = pre_m & ok[:, l]
        sd_g[l] = max(0.1, float(np.nanstd(dg[m_l, l])))
        sd_x[l] = max(0.1, float(np.nanstd(dx[m_l, l])))

    z15 = np.where(ok, 0.65 * (dg / sd_g[None, :]) + 0.35 * (dx / sd_x[None, :]), 0.0).astype(np.float64)

    players = D["B"]["player"].reshape(NM, 10).astype(str)
    uniq_p, p_inv = np.unique(players.reshape(-1), return_inverse=True)
    pid_mat = p_inv.reshape(NM, 10).tolist()
    n_p = len(uniq_p)

    uc, c_inv = np.unique(champ.reshape(-1), return_inverse=True)
    c_mat = c_inv.reshape(NM, 2, 5).tolist()
    nc = len(uc)

    # Esperados causales por (champ, lane) y (cA, cB, lane) en z15
    cl_s = np.zeros((nc, 5), dtype=np.float64)
    cl_n = np.zeros((nc, 5), dtype=np.float64)
    pr_s = np.zeros((5, nc, nc), dtype=np.float64)
    pr_n = np.zeros((5, nc, nc), dtype=np.float64)
    K_CL, K_PR = 20.0, 15.0

    # Kalman 1v1 estándar y rápido por jugador
    mu = [0.0] * n_p
    var = [0.36] * n_p
    mu_f = [0.0] * n_p
    var_f = [0.45] * n_p

    OBS_V = 0.85
    Q_S, MIN_VS = 0.008, 0.04
    Q_F, MIN_VF = 0.035, 0.06

    LK_AGGS = np.zeros((NM, 4), dtype=np.float32)
    LK_LANES = np.zeros((NM, 6), dtype=np.float32)

    order = np.lexsort((mids, t_arr))
    ok_l = ok.tolist()
    z_l = z15.tolist()

    for m in order:
        ps = pid_mat[m]
        cb, cr = c_mat[m][0], c_mat[m][1]

        # ---- 1. Leer estados previos (< m) ----
        mb = [mu[ps[l]] for l in range(5)]
        mr = [mu[ps[5 + l]] for l in range(5)]
        fb = [mu_f[ps[l]] for l in range(5)]
        fr = [mu_f[ps[5 + l]] for l in range(5)]
        vb = [var[ps[l]] for l in range(5)]
        vr = [var[ps[5 + l]] for l in range(5)]

        LK_AGGS[m, 0] = (sum(mb) - sum(mr)) * 0.2
        LK_AGGS[m, 1] = min(mb) - min(mr)
        LK_AGGS[m, 2] = max(mb) - max(mr)
        LK_AGGS[m, 3] = (sum(fb) - sum(fr)) * 0.2

        prec_sum = 0.0
        for l in range(5):
            d_l = mb[l] - mr[l]
            LK_LANES[m, l] = d_l
            prec_sum += d_l / math.sqrt(vb[l] + vr[l] + 0.1)
        LK_LANES[m, 5] = prec_sum * 0.2

        # ---- 2. Actualizar DESPUÉS para cada carril válido @15 ----
        om = ok_l[m]
        zm = z_l[m]
        for l in range(5):
            if not om[l]:
                continue
            z_obs = zm[l]
            c1, c2 = cb[l], cr[l]
            exp_ind = (cl_s[c1, l] / (cl_n[c1, l] + K_CL)) - (cl_s[c2, l] / (cl_n[c2, l] + K_CL))
            exp_pair = (pr_s[l, c1, c2] + K_PR * exp_ind) / (pr_n[l, c1, c2] + K_PR)
            y_skill = z_obs - exp_pair

            pb, pr_ = ps[l], ps[5 + l]
            # Lento
            innov = y_skill - (mu[pb] - mu[pr_])
            tv = var[pb] + var[pr_] + OBS_V
            kb = var[pb] / tv
            kr = var[pr_] / tv
            mu[pb] += kb * innov
            mu[pr_] -= kr * innov
            var[pb] = max(MIN_VS, (1.0 - kb) * var[pb]) + Q_S
            var[pr_] = max(MIN_VS, (1.0 - kr) * var[pr_]) + Q_S

            # Rápido
            innov_f = y_skill - (mu_f[pb] - mu_f[pr_])
            tv_f = var_f[pb] + var_f[pr_] + OBS_V
            kbf = var_f[pb] / tv_f
            krf = var_f[pr_] / tv_f
            mu_f[pb] += kbf * innov_f
            mu_f[pr_] -= krf * innov_f
            var_f[pb] = max(MIN_VF, (1.0 - kbf) * var_f[pb]) + Q_F
            var_f[pr_] = max(MIN_VF, (1.0 - krf) * var_f[pr_]) + Q_F

            # Actualizar esperados de campeón/cruce
            cl_s[c1, l] += z_obs
            cl_n[c1, l] += 1.0
            cl_s[c2, l] -= z_obs
            cl_n[c2, l] += 1.0
            pr_s[l, c1, c2] += z_obs
            pr_n[l, c1, c2] += 1.0
            pr_s[l, c2, c1] -= z_obs
            pr_n[l, c2, c1] += 1.0

    np.savez_compressed(CACHE, nm=np.array([NM]), LK_AGGS=LK_AGGS, LK_LANES=LK_LANES)
    print(f"[g_30] Caché guardada en {time.time() - t0:.1f}s", flush=True)
    return {"LK_AGGS": LK_AGGS, "LK_LANES": LK_LANES}


if __name__ == "__main__":
    D = cargar()
    NM, ms = D["NM"], D["master"]
    X9 = extras_referencia(D, escalon=9)
    res = construir_lane_kalman_15(D)
    LK_AGGS, LK_LANES = res["LK_AGGS"], res["LK_LANES"]
    TODO = np.c_[LK_AGGS, LK_LANES].astype(np.float32)
    PL = TODO[np.random.default_rng(23).permutation(NM)]

    def ml(X):
        return lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=X, extra_post=False, con_mapa=False)

    VAR = {
        "Escalón 9 oficial": ml(X9),
        "+ LK_AGGS (4)": ml(np.c_[X9, LK_AGGS]),
        "+ LK_LANES (6)": ml(np.c_[X9, LK_LANES]),
        "+ TODO LK_15 (10)": ml(np.c_[X9, TODO]),
        "placebo LK_15": ml(np.c_[X9, PL]),
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
