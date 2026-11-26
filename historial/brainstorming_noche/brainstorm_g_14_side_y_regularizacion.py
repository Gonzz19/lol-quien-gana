"""Ideas G26-G27 del brainstorming (docs/BRAINSTORM_NOCHE.md):
1. SIDE_COUNTER (Efecto asimetrico Azul vs Rojo / B1 blind vs R5 counterpick causal < t):
   - La matriz DRAFT actual fuerza simetria estricta w_azul = -w_rojo. En SoloQ Master+, el Lado Rojo
     siempre tiene ultimo pick (R5) y gana 52,6 %. Calculamos el residual causal del campeon jugando
     en Azul vs jugando en Rojo descontando su winrate global y el sesgo base de lado (K=60).
2. GROUP_L2 (Regularizacion L2 diferencial por bloque sobre Escalon 4):
   - `brainstorm_base.py` usa un unico C=0,05 para 851 cols dispersas de DRAFT (var~0,03), 425 cols
     colineales de Xd, 150 cols de MC/ML y solo 17 cols densas de alto SNR (NIV+W1+MATCHUPS).
   - Escalar el bloque k por gamma_k equivale exactamente a darle C_k = gamma_k^2 * C (y/o bajar
     C global). Probamos configuraciones fijas por principio (fuerte sobre Xd, moderado sobre MC/ML
     y draft, suelto sobre NIV+W1+MATCHUPS).
"""
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.brainstorm_base import ajustar_predecir, cargar, comparar, evaluar, extras_referencia, mostrar  # noqa: E402

CACHE = REPO / "data" / "analysis" / "brainstorm" / "gemini_side_counter.npz"


def construir_side_counter(D: dict) -> np.ndarray:
    if CACHE.exists():
        z = np.load(CACHE)
        if int(z["nm"][0]) == D["NM"]:
            return z["SIDE_COUNTER"]

    t0 = time.time()
    print("Construyendo caché gemini_side_counter.npz...", flush=True)
    NM = D["NM"]
    t = D["t"]
    y = D["y"].astype(float)
    champ = D["Z"]["champ"].reshape(NM, 2, 5).astype(int)
    ord_idx = np.lexsort((np.arange(NM), t))

    # Contadores estrictamente causales < t por (champ, lane) en Blue (s=0) y Red (s=1)
    from collections import defaultdict
    g_tot, n_tot = defaultdict(float), defaultdict(int)
    g_b, n_b = defaultdict(float), defaultdict(int)
    g_r, n_r = defaultdict(float), defaultdict(int)

    # 8 cols: blue_blind_sum, red_cnt_sum, blue_top_mid_sup, red_top_mid_sup, + diff top/mid/sup, blue_side_bias, red_side_bias
    SIDE = np.zeros((NM, 8), dtype=np.float32)
    K_tot, K_side = 30.0, 60.0
    BASE_DIFF = 0.026  # 52.6% Red vs 47.4% Blue -> +/- 0.026 respecto a 0.5

    for m in ord_idx:
        ym = y[m]
        res_b = [0.0] * 5
        res_r = [0.0] * 5
        for l in range(5):
            cb, cr = int(champ[m, 0, l]), int(champ[m, 1, l])
            kb, kr = (cb, l), (cr, l)

            wb_all = (g_tot[kb] + K_tot * 0.5) / (n_tot[kb] + K_tot)
            eb_prior = max(0.05, min(0.95, wb_all - BASE_DIFF))
            res_b[l] = (g_b[kb] + K_side * eb_prior) / (n_b[kb] + K_side) - eb_prior

            wr_all = (g_tot[kr] + K_tot * 0.5) / (n_tot[kr] + K_tot)
            er_prior = max(0.05, min(0.95, wr_all + BASE_DIFF))
            res_r[l] = (g_r[kr] + K_side * er_prior) / (n_r[kr] + K_side) - er_prior

        # Nota: un mayor res_b ayuda a Blue (+), un mayor res_r ayuda a Red (-)
        SIDE[m, 0] = sum(res_b)
        SIDE[m, 1] = -sum(res_r)
        SIDE[m, 2] = res_b[2] + res_b[3] + res_b[4]  # MID(2)+TOP(3)+SUP(4)
        SIDE[m, 3] = -(res_r[2] + res_r[3] + res_r[4])
        SIDE[m, 4] = res_b[3] - res_r[3]  # TOP side interaction
        SIDE[m, 5] = res_b[2] - res_r[2]  # MID side interaction
        SIDE[m, 6] = res_b[4] - res_r[4]  # SUP side interaction
        SIDE[m, 7] = sum(res_b) - sum(res_r)

        # Actualizar despues
        for l in range(5):
            kb = (int(champ[m, 0, l]), l)
            kr = (int(champ[m, 1, l]), l)
            g_tot[kb] += ym
            n_tot[kb] += 1
            g_b[kb] += ym
            n_b[kb] += 1
            g_tot[kr] += (1.0 - ym)
            n_tot[kr] += 1
            g_r[kr] += (1.0 - ym)
            n_r[kr] += 1

    np.savez_compressed(CACHE, nm=np.array([NM]), SIDE_COUNTER=SIDE)
    print(f"Caché gemini_side_counter.npz lista en {time.time() - t0:.1f}s", flush=True)
    return SIDE


def ajustar_escalado(D, tr_raw, te, MCML, HI, s_draft=1.0, s_xd=1.0, s_mcml=1.0, s_hi=1.0, C=0.05, sin_mapa=False, xd_mask=None):
    ms = D["master"]
    tr = tr_raw[ms[tr_raw]]
    Xd_raw = D["Xd"] if xd_mask is None else D["Xd"][:, xd_mask]
    post = D["post"]
    mu, sd = Xd_raw[tr].mean(0), Xd_raw[tr].std(0) + 1e-6

    mm1, sd1 = MCML[tr].mean(0), MCML[tr].std(0) + 1e-6
    mm2, sd2 = HI[tr].mean(0), HI[tr].std(0) + 1e-6

    def armar(idx, is_tr=False):
        parts = [
            D["DRAFT"][idx] * s_draft,
            sparse.csr_matrix((((Xd_raw[idx] - mu) / sd) * s_xd).astype(np.float32)),
            sparse.csr_matrix((((MCML[idx] - mm1) / sd1) * s_mcml).astype(np.float32)),
            sparse.csr_matrix((((HI[idx] - mm2) / sd2) * s_hi).astype(np.float32)),
        ]
        if not sin_mapa and post[tr].any():
            mapa = D["mapa"]
            ref = tr[post[tr]]
            me, se = mapa[ref].mean(0), mapa[ref].std(0) + 1e-6
            parts.append(sparse.csr_matrix((np.where(post[idx][:, None], (mapa[idx] - me) / se, 0.0) * s_mcml).astype(np.float32)))
        return sparse.hstack(parts).tocsr()

    lr = LogisticRegression(C=C, max_iter=4000).fit(armar(tr, True), D["y"][tr])
    return lr.predict_proba(armar(te, False))[:, 1]


def main():
    t0 = time.time()
    D = cargar()
    ms = D["master"]
    R6 = extras_referencia(D, escalon=6)
    SIDE = construir_side_counter(D)

    cols_tile = np.tile(np.array(list(D["cols"])), 5)
    mask_nopoda = ~np.isin(cols_tile, ["wr_acum", "wr_v20"]) & ~np.char.startswith(cols_tile.astype(str), "perf_")
    mask_podaf = mask_nopoda & ~np.char.startswith(cols_tile.astype(str), "v20_")

    mcml = R6[:, :150]
    hi6 = R6[:, 150:]  # NIV(3)+W1(4)+MAT(10)+ROL_WR(10)+g11(33) = 60 cols

    P = {}
    print("Evaluando ref: Escalón 6 (sin mapa, 0,6709)...", flush=True)
    P["ref: Escalón 6 (sin mapa)"] = evaluar(
        D, lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=R6, extra_post=False, con_mapa=False)
    )
    print("Evaluando Esc6 + SIDE_COUNTER...", flush=True)
    X_sc = np.c_[R6, SIDE].astype(np.float32)
    P["Esc6 + SIDE_COUNTER (8)"] = evaluar(
        D, lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=X_sc, extra_post=False, con_mapa=False)
    )

    configs = [
        ("Esc6 L2 dif A (Xd*0.5, hi*2.0)", dict(s_draft=1.0, s_xd=0.5, s_mcml=0.8, s_hi=2.0, C=0.05, sin_mapa=True)),
        ("Esc6 L2 dif B (Xd*0.4, draft*1.3, hi*2.5)", dict(s_draft=1.3, s_xd=0.4, s_mcml=0.7, s_hi=2.5, C=0.05, sin_mapa=True)),
        ("Esc6 C=0.02 + L2 dif (Xd*0.5, draft*1.5, hi*3.0)", dict(s_draft=1.5, s_xd=0.5, s_mcml=0.9, s_hi=3.0, C=0.02, sin_mapa=True)),
        ("Esc6 + Poda#21 + L2 dif (Xd*0.5, hi*2.0)", dict(s_draft=1.0, s_xd=0.5, s_mcml=0.8, s_hi=2.0, C=0.05, sin_mapa=True, xd_mask=mask_nopoda)),
        ("Esc6 + Poda#21f + L2 dif", dict(s_draft=1.0, s_xd=0.5, s_mcml=0.8, s_hi=2.0, C=0.05, sin_mapa=True, xd_mask=mask_podaf)),
        ("Esc6 + SIDE + Poda#21 + L2 dif", dict(s_draft=1.0, s_xd=0.5, s_mcml=0.8, s_hi=2.0, C=0.05, sin_mapa=True, xd_mask=mask_nopoda)),
    ]
    for nom, kw in configs:
        print(f"Evaluando {nom}...", flush=True)
        hi_cur = np.c_[hi6, SIDE].astype(np.float32) if "SIDE" in nom else hi6
        P[nom] = evaluar(D, lambda tr, te, h=hi_cur, k=kw: ajustar_escalado(D, tr, te, mcml, h, **k))

    mostrar(comparar(D, P, "ref: Escalón 6 (sin mapa)"))

    # Chequeo ciego en Diamond-V del mejor candidato de L2 dif
    di = np.flatnonzero(D["diam_v"])
    yd = D["y"][di]
    B = np.random.default_rng(1).integers(0, len(di), size=(300, len(di)))
    b0 = P["ref: Escalón 6 (sin mapa)"][0][di]
    from sklearn.metrics import roc_auc_score
    print(f"\nChequeo Diamond-V ({len(di):,} partidas):")
    for nom, (po, _) in P.items():
        if nom.startswith("ref:"):
            continue
        p = po[di]
        d = np.array([roc_auc_score(yd[b], p[b]) - roc_auc_score(yd[b], b0[b]) for b in B])
        print(f"  {nom:48s} AUC {roc_auc_score(yd, p):.4f}  Δ {roc_auc_score(yd, p) - roc_auc_score(yd, b0):+.4f} "
              f"[{np.percentile(d, 2.5):+.4f}; {np.percentile(d, 97.5):+.4f}]")
    print(f"Listo en {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
