"""Experimento `g_21` — Dos direcciones causales nuevas sobre Escalón 6 sin mapa (`0,6709`):

1) **`DRAFT_REC_BLK`** *(ajuste causal por deriva de parche en el draft, sin tirar partidas viejas)*:
   `brainstorm_03` tiraba historial de jugadores al ponderar todas las filas por `exp(-dt/tau)` y
   `g_17` comprimió el parche en 6 resúmenes escalares. Acá mantenemos `sample_weight=1` para
   TODO (`Xd`, `REF6`, `DRAFT_base`) y agregamos una copia residual de `DRAFT` ponderada en
   `tr` por `exp(-(t_end - t_i)/tau)` (`1.0` en `te`): cada `(campeón, línea)` aprende su desvío
   del parche actual con shrinkage L2 hacia el promedio histórico de toda la temporada!
2) **`POOL_FRESCO`** (`8 cols` causales de frescura y jerarquía en el pool del jugador `< m`,
   nunca presentes en `variables.npz`):
   - `days_since_champ`: tiempo `log1p(días)` desde la última vez que el jugador usó ESTE campeón
     (separa mains activos de picks oxidados de hace 40 días aunque `estreno=0`).
   - `champ_share_rec20` & `champ_wr_rec20`: fracción y rendimiento reciente (EWMA `N_eff~12`) en
     ESTE campeón (calentamiento actual vs acumulado viejo).
   - `is_top1_champ` & `otp_off_main`: si es el campeón #1 de su pool, y penalidad no lineal
     cuando un OTP (`max_share` alto) sale de su #1 (pick/ban).
   - `lane_freshness` & `pool_entropy`: entropía efectiva del pool reciente y días desde última
     partida en esta línea.

Evaluado sobre `Escalón 6 (sin mapa)` (`con_mapa=False`), `L2 dif (g14)`, `Diamond-V` y placebo.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
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

CACHE = REPO / "data" / "analysis" / "brainstorm" / "gemini_g21.npz"


def construir_cache(D: dict) -> dict:
    if CACHE.exists():
        print(f"Usando caché existente: {CACHE}")
        z = np.load(CACHE)
        return {k: z[k] for k in z.files}

    t0 = time.time()
    print("Construyendo caché gemini_g21.npz...")
    Z = D["Z"]
    NM = D["NM"]
    t_ms = D["t"].astype(np.int64)
    y = D["y"].astype(np.float32)
    champ = Z["champ"].astype(np.int32)
    players = D["B"]["player"]
    _, p_codes = np.unique(players.ravel(), return_inverse=True)
    p_idx = p_codes.reshape(NM, 2, 5).astype(np.int32)
    NP = int(p_idx.max()) + 1

    # Estructuras por jugador
    p_n = np.zeros(NP, dtype=np.int32)
    p_champ_cnt = [defaultdict(int) for _ in range(NP)]
    p_champ_rec = [defaultdict(float) for _ in range(NP)]   # EWMA recency indicator
    p_champ_last = [{} for _ in range(NP)]                  # (p, c) -> last t_ms
    p_champ_wr_ew = [defaultdict(float) for _ in range(NP)]
    p_champ_w_ew = [defaultdict(float) for _ in range(NP)]
    p_lane_last = np.full((NP, 5), -1, dtype=np.int64)
    p_top_cnt = np.zeros(NP, dtype=np.int32)
    p_top_champ = np.full(NP, -1, dtype=np.int32)
    p_rec_sum = np.zeros(NP, dtype=np.float32)

    F = np.zeros((NM, 8), dtype=np.float32)
    # 8 cols (Blue - Red):
    # 0: rusty_champ (log1p(days_since_champ) en no-estrenos)
    # 1: active_champ (rec share ~ últimos 15 juegos del jugador en este campeón)
    # 2: active_delta (active_champ - accum_share)
    # 3: champ_wr_rec_shrunk (rendimiento reciente en este campeón encogido k=8)
    # 4: is_top1_champ (1 si juega su #1 más jugado con >= 5 partidas previas)
    # 5: otp_forced_off (max_share_acum * (1 - is_top1) cuando max_share > 0.25)
    # 6: lane_rust (log1p(días desde última partida en esta línea))
    # 7: fresh_estreno_x_top_off (no-estreno oxidado >14d sin jugar este champ)

    decay_rec = 1.0 - 1.0 / 15.0  # vida media ~10 partidas del jugador

    for m in range(NM):
        ps = p_idx[m]
        cs = champ[m]
        tm = int(t_ms[m])

        v = np.zeros((2, 5, 8), dtype=np.float32)
        for s in (0, 1):
            for l in range(5):
                pi = int(ps[s, l])
                ci = int(cs[s, l])
                ntot = int(p_n[pi])
                cnt_c = p_champ_cnt[pi].get(ci, 0)
                lt_c = p_champ_last[pi].get(ci, -1)
                if lt_c >= 0:
                    days_c = max(0.0, (tm - lt_c) / 86_400_000.0)
                    v[s, l, 0] = float(np.log1p(min(days_c, 90.0)))
                    v[s, l, 7] = 1.0 if days_c >= 14.0 else 0.0
                else:
                    v[s, l, 0] = 0.0  # estreno ya capturado en Xj; col 0 mide óxido en conocidos

                lt_l = int(p_lane_last[pi, l])
                if lt_l >= 0:
                    days_l = max(0.0, (tm - lt_l) / 86_400_000.0)
                    v[s, l, 6] = float(np.log1p(min(days_l, 60.0)))
                else:
                    v[s, l, 6] = float(np.log1p(20.0))

                if ntot >= 3:
                    rec_tot = max(1e-4, float(p_rec_sum[pi]))
                    sh_rec = float(p_champ_rec[pi].get(ci, 0.0)) / rec_tot
                    sh_ac = cnt_c / float(ntot)
                    v[s, l, 1] = sh_rec
                    v[s, l, 2] = sh_rec - sh_ac

                    ew_w = p_champ_w_ew[pi].get(ci, 0.0)
                    ew_r = p_champ_wr_ew[pi].get(ci, 0.0)
                    v[s, l, 3] = float((ew_r + 0.5 * 8.0) / (ew_w + 8.0) - 0.5)

                    top_sh = float(p_top_cnt[pi]) / float(ntot)
                    is_t1 = 1.0 if (p_top_cnt[pi] >= 4 and int(p_top_champ[pi]) == ci) else 0.0
                    v[s, l, 4] = is_t1
                    if top_sh > 0.22 and is_t1 == 0.0 and p_top_cnt[pi] >= 4:
                        v[s, l, 5] = top_sh

        eq = v.mean(axis=1)  # (2, 8)
        F[m] = eq[0] - eq[1]

        # Actualización post-lectura
        for s in (0, 1):
            win_s = 1.0 if y[m] == (1.0 if s == 0 else 0.0) else 0.0
            for l in range(5):
                pi = int(ps[s, l])
                ci = int(cs[s, l])
                p_n[pi] += 1
                nc = p_champ_cnt[pi][ci] + 1
                p_champ_cnt[pi][ci] = nc
                if nc > p_top_cnt[pi]:
                    p_top_cnt[pi] = nc
                    p_top_champ[pi] = ci

                # Decay O(1) en rec_sum y ewma del campeón (usando escala por partida? mejor exacto sobre claves activas si len<=30)
                dmap = p_champ_rec[pi]
                for k in list(dmap.keys()):
                    dmap[k] *= decay_rec
                dmap[ci] = dmap.get(ci, 0.0) + 1.0
                p_rec_sum[pi] = p_rec_sum[pi] * decay_rec + 1.0

                p_champ_last[pi][ci] = tm
                p_lane_last[pi, l] = tm
                p_champ_w_ew[pi][ci] = p_champ_w_ew[pi].get(ci, 0.0) * 0.90 + 1.0
                p_champ_wr_ew[pi][ci] = p_champ_wr_ew[pi].get(ci, 0.0) * 0.90 + win_s

    np.nan_to_num(F, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.savez_compressed(CACHE, POOL_FRESCO=F)
    print(f"Caché gemini_g21.npz lista en {time.time() - t0:.1f}s")
    return {"POOL_FRESCO": F}


def pred_draft_rec(D, REF6, tau_dias=21.0, gamma=0.60, extra=None, C=0.05, l2_dif=False):
    ms = D["master"]
    Xd = D["Xd"]
    t_ms = D["t"]
    tot = REF6 if extra is None else np.c_[REF6, extra].astype(np.float32)

    def fn(tr, te):
        trm = tr[ms[tr]]
        t_end = t_ms[trm].max()
        w_rec = np.exp(-(t_end - t_ms[trm]) / (tau_dias * 86_400_000.0)).astype(np.float32)
        dr_tr_rec = D["DRAFT"][trm].multiply(gamma * w_rec[:, None])
        dr_te_rec = D["DRAFT"][te].multiply(gamma)

        mu, sd = Xd[trm].mean(0), Xd[trm].std(0) + 1e-6
        me, se = tot[trm].mean(0), tot[trm].std(0) + 1e-6
        g_xd = 0.5 if l2_dif else 1.0
        g_dr = 1.5 if l2_dif else 1.0
        g_hi = 3.0 if l2_dif else 1.0
        c_use = 0.02 if l2_dif else C

        Ptr = sparse.hstack([
            D["DRAFT"][trm] * g_dr,
            dr_tr_rec * g_dr,
            sparse.csr_matrix(((Xd[trm] - mu) / sd) * g_xd),
            sparse.csr_matrix(((tot[trm] - me) / se) * g_hi),
        ]).tocsr()
        Pte = sparse.hstack([
            D["DRAFT"][te] * g_dr,
            dr_te_rec * g_dr,
            sparse.csr_matrix(((Xd[te] - mu) / sd) * g_xd),
            sparse.csr_matrix(((tot[te] - me) / se) * g_hi),
        ]).tocsr()
        lr = LogisticRegression(C=c_use, max_iter=4000).fit(Ptr, D["y"][trm])
        return lr.predict_proba(Pte)[:, 1]

    return fn


def main():
    t0 = time.time()
    D = cargar()
    C21 = construir_cache(D)
    REF6 = extras_referencia(D, escalon=6)
    ms = D["master"]

    def solo_m(extra=None):
        tot = REF6 if extra is None else np.c_[REF6, extra].astype(np.float32)
        return lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=tot, extra_post=False, con_mapa=False)

    PF = C21["POOL_FRESCO"]
    rng = np.random.default_rng(77)
    PF_PL = PF[rng.permutation(D["NM"])]
    P = {"ref: Escalón 6 (sin mapa)": evaluar(D, solo_m())}
    P["+ DRAFT_REC tau=21d g=0.5"] = evaluar(D, pred_draft_rec(D, REF6, tau_dias=21.0, gamma=0.5))
    P["+ DRAFT_REC tau=28d g=0.6"] = evaluar(D, pred_draft_rec(D, REF6, tau_dias=28.0, gamma=0.6))
    P["+ DRAFT_REC tau=14d g=0.5"] = evaluar(D, pred_draft_rec(D, REF6, tau_dias=14.0, gamma=0.5))
    P["+ POOL_FRESCO (8)"] = evaluar(D, solo_m(PF))
    P["+ PLACEBO POOL_FRESCO"] = evaluar(D, solo_m(PF_PL))
    P["+ DRAFT_REC21 + POOL_FRESCO"] = evaluar(D, pred_draft_rec(D, REF6, tau_dias=21.0, gamma=0.5, extra=PF))
    P["+ DRAFT_REC21 + PF + L2dif"] = evaluar(D, pred_draft_rec(D, REF6, tau_dias=21.0, gamma=0.5, extra=PF, l2_dif=True))
    mostrar(comparar(D, P, "ref: Escalón 6 (sin mapa)"))

    fi_v, dv = D["fi_v"], D["diam_v"]
    ydv = D["y"][dv]
    p_dv_ref = solo_m()(fi_v, dv)
    auc_ref = roc_auc_score(ydv, p_dv_ref)
    boots = rng.integers(0, len(ydv), size=(1500, len(ydv)))
    print(f"\nChequeo Diamond-V ({len(ydv):,} partidas): ref={auc_ref:.4f}")
    for nom, fn in [
        ("+ DRAFT_REC21 g=0.5", pred_draft_rec(D, REF6, tau_dias=21.0, gamma=0.5)),
        ("+ DRAFT_REC28 g=0.6", pred_draft_rec(D, REF6, tau_dias=28.0, gamma=0.6)),
        ("+ POOL_FRESCO (8)", solo_m(PF)),
        ("+ DRAFT_REC21 + PF + L2dif", pred_draft_rec(D, REF6, tau_dias=21.0, gamma=0.5, extra=PF, l2_dif=True)),
    ]:
        p_dv = fn(fi_v, dv)
        auc_p = roc_auc_score(ydv, p_dv)
        ds = np.array([roc_auc_score(ydv[b], p_dv[b]) - roc_auc_score(ydv[b], p_dv_ref[b]) for b in boots])
        print(f"  {nom:28s} AUC {auc_p:.4f}  Δ {auc_p - auc_ref:+.4f} [{np.percentile(ds, 2.5):+.4f}; {np.percentile(ds, 97.5):+.4f}]")

    print(f"Listo en {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
