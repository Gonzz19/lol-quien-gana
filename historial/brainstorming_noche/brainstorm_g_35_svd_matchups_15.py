"""Idea G44 (`docs/BRAINSTORM_NOCHE.md`): **Factorización Bilineal Causal de Matchups `1v1` y Sinergias `2v2` sobre `@15` (`CHAMPS_BI_15`)**:

Mientras `MATCHUP_EXP_MARGIN` (`g_23`) encoge con `K=15` pares raros hacia cero (y `g_29` 2v2 falló por dispersión extrema
de `170x170` pares sin factorizar, igual que `g_12` FM falló al entrenar sobre `win` binario ruidoso), aquí factorizamos
el residuo **continuo de carril `@15` (`z15 = 0.65*dg15 + 0.35*dx15`)**, que tiene 10x-20x mayor SNR que `win`:

1. **`BI_1V1 (6)`** — Para cada carril `l` (`0..4`), ajustamos causalmente (`chunks < c`, y `< 16/09` para `V`/`Diamond-V`)
   una matriz **antisimétrica de rango `2*4`**:
     `f_l(cB, cR) = u_l(cB)·v_l(cR) - u_l(cR)·v_l(cB)`
   sobre el residuo `z15(l) - (E[cB, l] - E[cR, l])` ponderado por `N_pairs < chunk` con regularización `L2`.
   Permite generalizar interacciones (p. ej. rango vs melee / anti-tanque) incluso para cruces con `0..3` partidas previas.
2. **`BI_2V2 (3)`** — Factorización bilineal causal de rango `4` sobre la interacción **pura `2v2`** (descontando `exp_ind` y `1v1`):
   - `bot_sup_syn`: sinergia aliada `(c_bot, c_sup)` en `(z15_bot + z15_sup)` (`Az - Rj`)
   - `bot_esup_ctr`: cruce cruzado `(c_bot vs c_sup_rival)` en `z15_bot` (`Az - Rj`)
   - `mid_jg_syn`: sinergia `(c_mid, c_jg)` en `(z15_mid + z15_jg)` (`Az - Rj`)

Evaluado directamente sobre **Escalón 9 oficial (`V 0,6739`)** con placebo permutado y chequeo ciego `Diamond-V`.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
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
from historial.brainstorming_noche.brainstorm_g_29_matchup_2v2_15 import _extraer_15_por_carril  # noqa: E402

CACHE = REPO / "data" / "analysis" / "brainstorm" / "gemini_g35_bi_matchups.npz"


def _fit_skew_bilinear(R: np.ndarray, W: np.ndarray, rank: int = 4, lam: float = 80.0, steps: int = 160, seed: int = 42) -> np.ndarray:
    """Factorización antisimétrica ponderada M = U@V.T - V@U.T minimizando 0.5*sum(W*(M-R)^2) + 0.5*lam*(||U||^2+||V||^2)."""
    nc = R.shape[0]
    w_tot = float(W.sum())
    if w_tot < 10.0:
        return np.zeros((nc, nc), dtype=np.float32)
    scale = max(1.0, w_tot / float(nc))
    Wn = W / scale
    lam_n = lam / scale

    rng = np.random.default_rng(seed)
    U = rng.normal(0.0, 0.03, size=(nc, rank))
    V = rng.normal(0.0, 0.03, size=(nc, rank))
    mU = np.zeros_like(U)
    mV = np.zeros_like(V)
    vU = np.zeros_like(U)
    vV = np.zeros_like(V)
    lr = 0.04
    b1, b2, eps = 0.9, 0.999, 1e-8

    for t in range(1, steps + 1):
        M = U @ V.T - V @ U.T
        G = Wn * (M - R)
        gU = 2.0 * (G @ V) + lam_n * U
        gV = -2.0 * (G @ U) + lam_n * V

        mU = b1 * mU + (1.0 - b1) * gU
        mV = b1 * mV + (1.0 - b1) * gV
        vU = b2 * vU + (1.0 - b2) * (gU * gU)
        vV = b2 * vV + (1.0 - b2) * (gV * gV)
        mU_h = mU / (1.0 - b1 ** t)
        mV_h = mV / (1.0 - b1 ** t)
        vU_h = vU / (1.0 - b2 ** t)
        vV_h = vV / (1.0 - b2 ** t)
        U -= lr * mU_h / (np.sqrt(vU_h) + eps)
        V -= lr * mV_h / (np.sqrt(vV_h) + eps)

    M = U @ V.T - V @ U.T
    M = 0.5 * (M - M.T)
    return np.clip(M, -2.5, 2.5).astype(np.float32)


def _fit_rect_bilinear(R: np.ndarray, W: np.ndarray, rank: int = 4, lam: float = 120.0, steps: int = 160, seed: int = 42) -> np.ndarray:
    """Factorización bilineal rectangular M = U@V.T minimizando 0.5*sum(W*(M-R)^2) + 0.5*lam*(||U||^2+||V||^2)."""
    nc = R.shape[0]
    w_tot = float(W.sum())
    if w_tot < 10.0:
        return np.zeros((nc, nc), dtype=np.float32)
    scale = max(1.0, w_tot / float(nc))
    Wn = W / scale
    lam_n = lam / scale

    rng = np.random.default_rng(seed)
    U = rng.normal(0.0, 0.03, size=(nc, rank))
    V = rng.normal(0.0, 0.03, size=(nc, rank))
    mU = np.zeros_like(U)
    mV = np.zeros_like(V)
    vU = np.zeros_like(U)
    vV = np.zeros_like(V)
    lr = 0.04
    b1, b2, eps = 0.9, 0.999, 1e-8

    for t in range(1, steps + 1):
        M = U @ V.T
        G = Wn * (M - R)
        gU = G @ V + lam_n * U
        gV = G.T @ U + lam_n * V

        mU = b1 * mU + (1.0 - b1) * gU
        mV = b1 * mV + (1.0 - b1) * gV
        vU = b2 * vU + (1.0 - b2) * (gU * gU)
        vV = b2 * vV + (1.0 - b2) * (gV * gV)
        mU_h = mU / (1.0 - b1 ** t)
        mV_h = mV / (1.0 - b1 ** t)
        vU_h = vU / (1.0 - b2 ** t)
        vV_h = vV / (1.0 - b2 ** t)
        U -= lr * mU_h / (np.sqrt(vU_h) + eps)
        V -= lr * mV_h / (np.sqrt(vV_h) + eps)

    return np.clip(U @ V.T, -2.0, 2.0).astype(np.float32)


def construir_bi_matchups(D: dict, rehacer: bool = False) -> dict[str, np.ndarray]:
    if CACHE.exists() and not rehacer:
        z = np.load(CACHE)
        if int(z["nm"][0]) == D["NM"]:
            return {k: z[k] for k in z.files if k != "nm"}

    t0 = time.time()
    print("[g_35] Construyendo factorización bilineal causal BI_1V1 y BI_2V2 (< chunk)...", flush=True)
    NM = D["NM"]
    mids = D["Z"]["mids"].astype(str)
    t_arr = D["t"].astype(np.int64)
    champ = D["Z"]["champ"].reshape(NM, 2, 5)
    dg, dx, ok = _extraer_15_por_carril(D)

    pre_m = (~D["val0"]) & (~D["post"]) & D["master"]
    sd_g = np.ones(5)
    sd_x = np.ones(5)
    for l in range(5):
        ml = pre_m & ok[:, l]
        sd_g[l] = max(0.1, float(np.nanstd(dg[ml, l])))
        sd_x[l] = max(0.1, float(np.nanstd(dx[ml, l])))
    z15 = np.where(ok, 0.65 * (dg / sd_g[None, :]) + 0.35 * (dx / sd_x[None, :]), 0.0).astype(np.float64)

    uc, c_inv = np.unique(champ.reshape(-1), return_inverse=True)
    c_arr = c_inv.reshape(NM, 2, 5)
    nc = len(uc)

    # Acumuladores suficientes (< chunk)
    cl_s = np.zeros((nc, 5), dtype=np.float64)
    cl_n = np.zeros((nc, 5), dtype=np.float64)
    s_1v1 = np.zeros((5, nc, nc), dtype=np.float64)
    n_1v1 = np.zeros((5, nc, nc), dtype=np.float64)

    # 2v2: bot+sup ally (sum residual), bot vs enemy sup (z_bot residual), mid+jg ally (sum residual)
    s_bs = np.zeros((nc, nc), dtype=np.float64)
    n_bs = np.zeros((nc, nc), dtype=np.float64)
    s_be = np.zeros((nc, nc), dtype=np.float64)
    n_be = np.zeros((nc, nc), dtype=np.float64)
    s_mj = np.zeros((nc, nc), dtype=np.float64)
    n_mj = np.zeros((nc, nc), dtype=np.float64)

    order = np.lexsort((mids, t_arr))
    pre_v_ord = np.array([m for m in order if not D["val0"][m]], dtype=np.int32)
    post_v_ord = np.array([m for m in order if D["val0"][m]], dtype=np.int32)
    chunks = np.array_split(pre_v_ord, 20) + [post_v_ord]

    BI_1V1 = np.zeros((NM, 6), dtype=np.float32)  # 5 carriles + media
    BI_2V2 = np.zeros((NM, 3), dtype=np.float32)  # bot_sup_syn, bot_esup_ctr, mid_jg_syn

    for c_idx, ch in enumerate(chunks):
        if len(ch) == 0:
            continue
        if c_idx > 0:
            # 1. Ajustar factorizaciones estrictamente con chunks < c_idx
            exp_ind = cl_s / (cl_n + 20.0)  # (nc, 5)
            M1v1 = np.zeros((5, nc, nc), dtype=np.float32)
            for l in range(5):
                W = n_1v1[l] + n_1v1[l].T
                R_raw = np.divide(s_1v1[l] - s_1v1[l].T, W, out=np.zeros_like(W), where=W > 0)
                E = exp_ind[:, l][:, None] - exp_ind[:, l][None, :]
                R_res = np.where(W > 0, R_raw - E, 0.0)
                M1v1[l] = _fit_skew_bilinear(R_res, W, rank=4, lam=80.0, steps=160, seed=40 + l)

            R_bs = np.divide(s_bs, n_bs, out=np.zeros_like(s_bs), where=n_bs > 0)
            M_bs = _fit_rect_bilinear(R_bs, n_bs, rank=4, lam=120.0, steps=160, seed=51)

            R_be = np.divide(s_be, n_be, out=np.zeros_like(s_be), where=n_be > 0)
            M_be = _fit_rect_bilinear(R_be, n_be, rank=4, lam=120.0, steps=160, seed=52)

            R_mj = np.divide(s_mj, n_mj, out=np.zeros_like(s_mj), where=n_mj > 0)
            M_mj = _fit_rect_bilinear(R_mj, n_mj, rank=4, lam=120.0, steps=160, seed=53)

            # 2. Predecir el chunk ch
            cb = c_arr[ch, 0, :]
            cr = c_arr[ch, 1, :]
            for l in range(5):
                BI_1V1[ch, l] = M1v1[l, cb[:, l], cr[:, l]]
            BI_1V1[ch, 5] = BI_1V1[ch, :5].mean(1)

            BI_2V2[ch, 0] = M_bs[cb[:, 0], cb[:, 4]] - M_bs[cr[:, 0], cr[:, 4]]
            BI_2V2[ch, 1] = M_be[cb[:, 0], cr[:, 4]] - M_be[cr[:, 0], cb[:, 4]]
            BI_2V2[ch, 2] = M_mj[cb[:, 2], cb[:, 1]] - M_mj[cr[:, 2], cr[:, 1]]

        # Jamás acumular val0 (`>= 16/09`)
        if c_idx == len(chunks) - 1:
            continue

        # 3. Acumular estadísticas suficientes de ch para los chunks posteriores
        exp_ind_cur = cl_s / (cl_n + 20.0)
        cb = c_arr[ch, 0, :]
        cr = c_arr[ch, 1, :]
        for l in range(5):
            vc = ok[ch, l]
            if not vc.any():
                continue
            m_ok = ch[vc]
            c1, c2, zv = cb[vc, l], cr[vc, l], z15[m_ok, l]
            np.add.at(cl_s[:, l], c1, zv)
            np.add.at(cl_n[:, l], c1, 1.0)
            np.add.at(cl_s[:, l], c2, -zv)
            np.add.at(cl_n[:, l], c2, 1.0)
            np.add.at(s_1v1[l], (c1, c2), zv)
            np.add.at(n_1v1[l], (c1, c2), 1.0)

        # Residuos 2v2 ortogonalizados contra esperados individuales actuales
        ok_bs = ok[ch, 0] & ok[ch, 4]
        if ok_bs.any():
            m_bs = ch[ok_bs]
            b0, r0 = cb[ok_bs, 0], cr[ok_bs, 0]
            b4, r4 = cb[ok_bs, 4], cr[ok_bs, 4]
            # Residuo total bot+sup tras restar individuales
            res_tot = (
                (z15[m_bs, 0] - (exp_ind_cur[b0, 0] - exp_ind_cur[r0, 0]))
                + (z15[m_bs, 4] - (exp_ind_cur[b4, 4] - exp_ind_cur[r4, 4]))
            )
            np.add.at(s_bs, (b0, b4), res_tot)
            np.add.at(n_bs, (b0, b4), 1.0)
            np.add.at(s_bs, (r0, r4), -res_tot)
            np.add.at(n_bs, (r0, r4), 1.0)

            res_bot = z15[m_bs, 0] - (exp_ind_cur[b0, 0] - exp_ind_cur[r0, 0])
            np.add.at(s_be, (b0, r4), res_bot)
            np.add.at(n_be, (b0, r4), 1.0)
            np.add.at(s_be, (r0, b4), -res_bot)
            np.add.at(n_be, (r0, b4), 1.0)

        ok_mj = ok[ch, 2] & ok[ch, 1]
        if ok_mj.any():
            m_mj = ch[ok_mj]
            b2, r2 = cb[ok_mj, 2], cr[ok_mj, 2]
            b1, r1 = cb[ok_mj, 1], cr[ok_mj, 1]
            res_mj = (
                (z15[m_mj, 2] - (exp_ind_cur[b2, 2] - exp_ind_cur[r2, 2]))
                + (z15[m_mj, 1] - (exp_ind_cur[b1, 1] - exp_ind_cur[r1, 1]))
            )
            np.add.at(s_mj, (b2, b1), res_mj)
            np.add.at(n_mj, (b2, b1), 1.0)
            np.add.at(s_mj, (r2, r1), -res_mj)
            np.add.at(n_mj, (r2, r1), 1.0)

    np.savez_compressed(CACHE, nm=np.array([NM]), BI_1V1=BI_1V1, BI_2V2=BI_2V2)
    print(f"[g_35] Caché guardada en {time.time() - t0:.1f}s", flush=True)
    return {"BI_1V1": BI_1V1, "BI_2V2": BI_2V2}


if __name__ == "__main__":
    D = cargar()
    NM, ms = D["NM"], D["master"]
    X9 = extras_referencia(D, escalon=9)
    res = construir_bi_matchups(D)
    B1, B2 = res["BI_1V1"], res["BI_2V2"]
    TODO = np.c_[B1, B2].astype(np.float32)
    PL = TODO[np.random.default_rng(71).permutation(NM)]

    def ml(X):
        return lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=X, extra_post=False, con_mapa=False)

    VAR = {
        "Escalón 9 oficial": ml(X9),
        "+ BI_1V1 (6)": ml(np.c_[X9, B1]),
        "+ BI_2V2 (3)": ml(np.c_[X9, B2]),
        "+ TODO BI (9)": ml(np.c_[X9, TODO]),
        "placebo BI (9)": ml(np.c_[X9, PL]),
    }
    P = dict(Parallel(n_jobs=2, backend="threading")(delayed(lambda n: (n, evaluar(D, VAR[n])))(n) for n in VAR))
    mostrar(comparar(D, P, "Escalón 9 oficial"))

    vi = np.flatnonzero(D["val"])
    yv = D["y"][vi]
    print("\n--- LogLoss en V Master+ ---", flush=True)
    for n in VAR:
        print(f"  {n:22s}: V AUC {roc_auc_score(yv, P[n][1]):.4f} | LL {log_loss(yv, P[n][1]):.5f}", flush=True)

    di = np.flatnonzero(D["diam_v"])
    yd = D["y"][di]
    B = np.random.default_rng(1).integers(0, len(di), size=(300, len(di)))
    qd = P["Escalón 9 oficial"][0][di]
    print("\n--- Chequeo ciego Diamond-V (10.506 partidas) ---", flush=True)
    for n in VAR:
        if n == "Escalón 9 oficial":
            continue
        pd_ = P[n][0][di]
        d = np.array([roc_auc_score(yd[b], pd_[b]) - roc_auc_score(yd[b], qd[b]) for b in B])
        print(f"  Diamond-V {n:16s}: {roc_auc_score(yd, pd_):.4f} (Δ {roc_auc_score(yd, pd_) - roc_auc_score(yd, qd):+.4f} [{np.percentile(d, 2.5):+.4f}; {np.percentile(d, 97.5):+.4f}])", flush=True)
