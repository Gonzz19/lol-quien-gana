"""
Brainstorming Noche — Gemini G38 (`brainstorm_g_38_reg_bloques_y_stack.py`):
Prueba estructural (SIN nuevas variables ni tuning en V) sobre `Escalón 9 oficial`
(`V 0,6739`, `LL 0,64431`, `1.504` cols):

1. `Sin Xd estáticas (1,304 cols)`: quita las 200 cols estáticas de campeón dentro de `Xd`
   (`tag_*`, `info_*`, `st_*`, `perf_*`), colineales con el one-hot `DRAFT` (851 cols).
2. `FOTO1 extra_post=True`: corrige la estandarización de `FOTO1_DIN` (`87%` ceros en `tr`
   pre-04/09 cuando `extra_post=False`).
3. `Block-OOF Stack Lineal (7 bloques)` + `Stack Lineal + Contexto (post, estrenos, vet)`:
   Cada uno de los 7 bloques semánticos (`DRAFT`, `Xd_dyn`, `MCML`, `ELO_KALMAN`, `CHAMPS`,
   `HGB`, `FOTO1_post`) selecciona su L2 (`C_k`) estrictamente por 5-fold OOF cronológico
   *dentro de `tr`*, genera logits OOF dentro de `tr`, y un meta-modelo los combina junto
   al contexto simétrico de incertidumbre (`post`, `sum_estrenos`, `min_logn_tot`).
4. `Block-Penalty Escalón 9 (L2 dif por bloque)` y `Stack HGB meta (no lineal)` +
   `Mix 50/50 Esc9 + Stack_HGB`: verifica si el apilamiento por bloques + calibración
   no lineal de confianza supera a la regresión logística uniforme (`C=0,05`) en `A2..A4`,
   `V` y `Diamond-V` ciego sin reentrenar en GPU.

Uso:
  python historial/brainstorming_noche/brainstorm_g_38_reg_bloques_y_stack.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
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


def _logit(p: np.ndarray) -> np.ndarray:
    pc = np.clip(p, 1e-5, 1.0 - 1e-5)
    return np.log(pc / (1.0 - pc)).astype(np.float32)


def _fit_pred_lr(Xtr, ytr, Xte, C: float, dense: bool = True) -> np.ndarray:
    if dense:
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0) + 1e-6
        Xt = (Xtr - mu) / sd
        Xe = (Xte - mu) / sd
    else:
        Xt, Xe = Xtr, Xte
    lr = LogisticRegression(C=C, max_iter=2000).fit(Xt, ytr)
    return _logit(lr.predict_proba(Xe)[:, 1])


def _oof_bloque(X, y, t, tr: np.ndarray, te_list: list[np.ndarray], cs=(0.01, 0.05, 0.25), dense=True, solo_post=None):
    """5-fold cronológico estricto dentro de `tr` seleccionando `C` en `tr` y devolviendo `(oof_tr, [pred_te...])`."""
    ord_tr = np.argsort(t[tr], kind="mergesort")
    tr_s = tr[ord_tr]
    n = len(tr_s)

    if solo_post is not None:
        # Folds dentro del subconjunto `post` (en orden cronológico) para que todos los folds tengan filas activas
        pos_post_s = np.flatnonzero(solo_post[tr_s])
        best_c, best_ll, best_oof = cs[0], 1e9, np.zeros(n, dtype=np.float32)
        if len(pos_post_s) >= 100:
            folds_p = np.array_split(pos_post_s, 5)
            y_post = y[tr_s[pos_post_s]]
            for c_val in cs:
                oof_p = np.zeros(len(pos_post_s), dtype=np.float32)
                off = 0
                for fk in range(5):
                    va_p = folds_p[fk]
                    fit_p = np.concatenate([folds_p[j] for j in range(5) if j != fk])
                    m = len(va_p)
                    oof_p[off : off + m] = _fit_pred_lr(X[tr_s[fit_p]], y[tr_s[fit_p]], X[tr_s[va_p]], c_val, dense=dense)
                    off += m
                p_oof = 1.0 / (1.0 + np.exp(-np.clip(oof_p, -10.0, 10.0)))
                ll = float(log_loss(y_post, p_oof))
                if ll < best_ll:
                    best_ll = ll
                    best_c = c_val
                    tmp = np.zeros(n, dtype=np.float32)
                    tmp[pos_post_s] = oof_p
                    best_oof = tmp
    else:
        folds = np.array_split(np.arange(n), 5)
        best_c, best_ll, best_oof = cs[0], 1e9, None
        y_tr_s = y[tr_s]
        for c_val in cs:
            oof_s = np.zeros(n, dtype=np.float32)
            for fk in range(5):
                va_loc = folds[fk]
                fit_loc = np.concatenate([folds[j] for j in range(5) if j != fk])
                oof_s[va_loc] = _fit_pred_lr(X[tr_s[fit_loc]], y[tr_s[fit_loc]], X[tr_s[va_loc]], c_val, dense=dense)
            p_oof = 1.0 / (1.0 + np.exp(-np.clip(oof_s, -10.0, 10.0)))
            ll = float(log_loss(y_tr_s, p_oof))
            if ll < best_ll:
                best_ll = ll
                best_c = c_val
                best_oof = oof_s

    oof_tr = np.zeros(n, dtype=np.float32)
    oof_tr[ord_tr] = best_oof

    # Ajustar en TODO `tr` con `best_c` y predecir cada `te`
    preds_te = []
    if solo_post is not None:
        tr_eff = tr[solo_post[tr]]
        if len(tr_eff) >= 50:
            mu = X[tr_eff].mean(axis=0)
            sd = X[tr_eff].std(axis=0) + 1e-6
            lr = LogisticRegression(C=best_c, max_iter=2000).fit((X[tr_eff] - mu) / sd, y[tr_eff])
            for te in te_list:
                out = np.zeros(len(te), dtype=np.float32)
                m_te = solo_post[te]
                if m_te.any():
                    out[m_te] = _logit(lr.predict_proba((X[te[m_te]] - mu) / sd)[:, 1])
                preds_te.append(out)
        else:
            for te in te_list:
                preds_te.append(np.zeros(len(te), dtype=np.float32))
    else:
        if dense:
            mu = X[tr].mean(axis=0)
            sd = X[tr].std(axis=0) + 1e-6
            lr = LogisticRegression(C=best_c, max_iter=2000).fit((X[tr] - mu) / sd, y[tr])
            for te in te_list:
                preds_te.append(_logit(lr.predict_proba((X[te] - mu) / sd)[:, 1]))
        else:
            lr = LogisticRegression(C=best_c, max_iter=2000).fit(X[tr], y[tr])
            for te in te_list:
                preds_te.append(_logit(lr.predict_proba(X[te])[:, 1]))

    return oof_tr, preds_te, best_c


def _oof_full_esc9(D, ref9, tr: np.ndarray, te_list: list[np.ndarray]):
    """Logit OOF del Escalón 9 oficial completo (5-fold cronológico dentro de `tr`)."""
    t = D["t"]
    ord_tr = np.argsort(t[tr], kind="mergesort")
    tr_s = tr[ord_tr]
    folds = np.array_split(np.arange(len(tr_s)), 5)
    oof_s = np.zeros(len(tr_s), dtype=np.float32)
    for fk in range(5):
        va_loc = folds[fk]
        fit_loc = np.concatenate([folds[j] for j in range(5) if j != fk])
        p_va = ajustar_predecir(D, tr_s[fit_loc], tr_s[va_loc], extra=ref9, extra_post=False, con_mapa=False)
        oof_s[va_loc] = _logit(p_va)
    oof_tr = np.zeros(len(tr_s), dtype=np.float32)
    oof_tr[ord_tr] = oof_s

    preds_te = []
    # Reusar modelo completo para te_list
    mu, sd = D["Xd"][tr].mean(0), D["Xd"][tr].std(0) + 1e-6
    me, se = ref9[tr].mean(0), ref9[tr].std(0) + 1e-6
    Xtr_m = sparse.hstack([D["DRAFT"][tr], sparse.csr_matrix(((D["Xd"][tr] - mu) / sd).astype(np.float32)),
                           sparse.csr_matrix(((ref9[tr] - me) / se).astype(np.float32))]).tocsr()
    lr = LogisticRegression(C=0.05, max_iter=4000).fit(Xtr_m, D["y"][tr])
    for te in te_list:
        Xte_m = sparse.hstack([D["DRAFT"][te], sparse.csr_matrix(((D["Xd"][te] - mu) / sd).astype(np.float32)),
                               sparse.csr_matrix(((ref9[te] - me) / se).astype(np.float32))]).tocsr()
        preds_te.append(_logit(lr.predict_proba(Xte_m)[:, 1]))
    return oof_tr, preds_te


def main():
    t0 = time.time()
    D = cargar()
    ref9 = extras_referencia(D, escalon=9)
    NM = D["NM"]
    ms = D["master"]
    y = D["y"]
    t = D["t"]
    post = D["post"]

    # Subconjuntos de Xd
    cols = D["cols"]
    is_static = np.array([c.startswith(("tag_", "info_", "st_", "perf_")) for c in cols])
    keep_xd = np.tile(~is_static, 5)
    Xd_dyn = D["Xd"][:, keep_xd].astype(np.float32)  # 225 cols

    # Separar bloques de ref9 (228 cols exactas)
    B_MCML = ref9[:, 0:150]
    B_F1 = ref9[:, 153:157]
    B_ELO = np.c_[ref9[:, 150:153], ref9[:, 167:210]]  # niv(3) + ROL_WR(10) + KALMAN(19) + TEMPO/ECON(14) = 46
    B_CHAMPS = np.c_[ref9[:, 157:167], ref9[:, 216:228]]  # MATCHUPS(10) + MEM(12) = 22
    B_HGB = ref9[:, 210:216]  # 6

    # Contexto simétrico para meta-calibración de bloques
    Xj = D["Xj"]
    logn = Xj[:, :, :, cols.index("logn_tot")].reshape(NM, 10)
    est = Xj[:, :, :, cols.index("estreno")].reshape(NM, 10)
    off = Xj[:, :, :, cols.index("offrole")].reshape(NM, 10)
    CTX = np.column_stack(
        [
            post.astype(np.float32),
            logn.min(axis=1),
            logn.mean(axis=1),
            est.sum(axis=1),
            off.sum(axis=1),
            np.nan_to_num(D["tier_lobby"], nan=5.0),
        ]
    ).astype(np.float32)

    print(f"Preparado en {time.time() - t0:.1f}s | Xd_dyn={Xd_dyn.shape[1]}, B_ELO={B_ELO.shape[1]}, B_CHAMPS={B_CHAMPS.shape[1]}")

    # 1) Pruebas directas simples (sin stacking)
    D_nostatic = dict(D, Xd=Xd_dyn)
    ref9_sin_f1 = np.delete(ref9, np.s_[153:157], axis=1)

    simple_vars = {
        "Escalón 9 oficial": lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=ref9, extra_post=False, con_mapa=False),
        "Sin Xd estáticas (1,304)": lambda tr, te: ajustar_predecir(D_nostatic, tr[ms[tr]], te, extra=ref9, extra_post=False, con_mapa=False),
        "FOTO1 extra_post=True": lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=[(ref9_sin_f1, False), (B_F1, True)], con_mapa=False),
        "Sin Xd est + F1 post": lambda tr, te: ajustar_predecir(D_nostatic, tr[ms[tr]], te, extra=[(ref9_sin_f1, False), (B_F1, True)], con_mapa=False),
    }

    P = {}
    D_res = {}
    te_diam = np.flatnonzero(D["diam_v"])
    yd = y[te_diam]

    for nombre, fn in simple_vars.items():
        po, pv = evaluar(D, fn)
        P[nombre] = (po, pv)
        pd_v = fn(D["fi_v"], te_diam)
        D_res[nombre] = pd_v

    # 2) Evaluar Stacking de 7 bloques + Contexto + Meta HGB en A2..A4 y V/DiamV
    nombres_stack = [
        "Stack 7 Bloques Lin",
        "Stack 7+CTX Lin",
        "Stack Esc9+7+CTX Lin",
        "Stack HGB Meta",
        "Mix 50/50 Esc9+HGBMeta",
    ]
    stack_po = {n: np.full(NM, np.nan) for n in nombres_stack}
    stack_pv = {n: None for n in nombres_stack}
    stack_pd = {n: None for n in nombres_stack}

    # Correr para los 3 bloques A2..A4 y para V+DiamV compartiendo fi_v
    fi, bl = D["fi"], D["bloque"]
    split_specs = [(b, fi[bl[fi] < b], [np.flatnonzero(bl == b)]) for b in D["EV"]]
    vi = np.flatnonzero(D["val"])
    split_specs.append(("V", D["fi_v"], [vi, te_diam]))

    for sp_id, tr_raw, te_list in split_specs:
        tr = tr_raw[ms[tr_raw]]
        ts = time.time()

        oof_esc9, te_esc9 = _oof_full_esc9(D, ref9, tr, te_list)

        blocks_def = [
            ("DRAFT", D["DRAFT"], (0.02, 0.05, 0.15), False, None),
            ("XD_DYN", Xd_dyn, (0.005, 0.02, 0.08), True, None),
            ("MCML", B_MCML, (0.005, 0.02, 0.08), True, None),
            ("ELO", B_ELO, (0.02, 0.08, 0.3), True, None),
            ("CHAMPS", B_CHAMPS, (0.02, 0.1, 0.5), True, None),
            ("HGB", B_HGB, (0.05, 0.25, 1.0), True, None),
            ("F1_POST", B_F1, (0.05, 0.25, 1.0), True, post),
        ]

        oofs = []
        tes_blocks = [[] for _ in te_list]
        chosen = {}
        for bname, Xb, cs, dens, sp in blocks_def:
            otr, ptes, bc = _oof_bloque(Xb, y, t, tr, te_list, cs=cs, dense=dens, solo_post=sp)
            oofs.append(otr)
            chosen[bname] = bc
            for ti, pt in enumerate(ptes):
                tes_blocks[ti].append(pt)

        M7_tr = np.column_stack(oofs).astype(np.float32)
        M7_tes = [np.column_stack(tb).astype(np.float32) for tb in tes_blocks]

        # Interacciones interpretables de contexto con los bloques clave
        def armar_int(M7, idx):
            cx = CTX[idx]
            po_f = cx[:, 0]
            lmin = (cx[:, 1] - CTX[tr, 1].mean()) / (CTX[tr, 1].std() + 1e-6)
            est_s = (cx[:, 3] - CTX[tr, 3].mean()) / (CTX[tr, 3].std() + 1e-6)
            return np.column_stack(
                [
                    M7,
                    M7[:, 0] * est_s,  # DRAFT pesa más si hay estrenos
                    M7[:, 2] * est_s,  # MCML cambia con estrenos
                    M7[:, 3] * lmin,   # ELO según veteranía mínima
                    M7[:, 4] * est_s,  # CHAMPS pesa más si hay estrenos
                    M7[:, 6] * po_f,   # F1 solo activo post
                ]
            ).astype(np.float32)

        M_int_tr = armar_int(M7_tr, tr)
        M_int_tes = [armar_int(M7_tes[i], te) for i, te in enumerate(te_list)]

        # 1) Stack 7 Bloques Lin
        lr7 = LogisticRegression(C=0.5, max_iter=1000).fit(M7_tr, y[tr])
        # 2) Stack 7+CTX Lin
        mu_i, sd_i = M_int_tr.mean(0), M_int_tr.std(0) + 1e-6
        lr_ctx = LogisticRegression(C=0.2, max_iter=1000).fit((M_int_tr - mu_i) / sd_i, y[tr])
        # 3) Stack Esc9+7+CTX Lin
        Me_tr = np.column_stack([oof_esc9, M_int_tr])
        mu_e, sd_e = Me_tr.mean(0), Me_tr.std(0) + 1e-6
        lr_e = LogisticRegression(C=0.2, max_iter=1000).fit((Me_tr - mu_e) / sd_e, y[tr])
        # 4) Stack HGB Meta sobre [Esc9_oof, M7, CTX]
        Xh_tr = np.column_stack([oof_esc9, M7_tr, CTX[tr]])
        hgb = HistGradientBoostingClassifier(
            max_depth=3,
            min_samples_leaf=300,
            l2_regularization=20.0,
            learning_rate=0.03,
            max_iter=250,
            early_stopping=True,
            validation_fraction=0.12,
            random_state=38,
        ).fit(Xh_tr, y[tr])

        for ti, te in enumerate(te_list):
            p1 = lr7.predict_proba(M7_tes[ti])[:, 1]
            p2 = lr_ctx.predict_proba((M_int_tes[ti] - mu_i) / sd_i)[:, 1]
            Me_te = np.column_stack([te_esc9[ti], M_int_tes[ti]])
            p3 = lr_e.predict_proba((Me_te - mu_e) / sd_e)[:, 1]
            Xh_te = np.column_stack([te_esc9[ti], M7_tes[ti], CTX[te]])
            p4 = hgb.predict_proba(Xh_te)[:, 1]
            p_esc9 = 1.0 / (1.0 + np.exp(-te_esc9[ti]))
            p5 = 0.5 * p_esc9 + 0.5 * p4
            outs = [p1, p2, p3, p4, p5]
            if sp_id == "V":
                for ni, nm in enumerate(nombres_stack):
                    if ti == 0:
                        stack_pv[nm] = outs[ni]
                    else:
                        stack_pd[nm] = outs[ni]
            else:
                for ni, nm in enumerate(nombres_stack):
                    stack_po[nm][te] = outs[ni]

        print(f"Split {sp_id} listo en {time.time() - ts:.1f}s | C_elegidos={chosen}")

    for nm in nombres_stack:
        P[nm] = (stack_po[nm], stack_pv[nm])
        D_res[nm] = stack_pd[nm]

    T = comparar(D, P, ref="Escalón 9 oficial")
    yv = y[vi]
    pd_ref = D_res["Escalón 9 oficial"]
    T["LL V"] = [round(log_loss(yv, P[n][1]), 5) for n in T["variante"]]
    T["AUC DiamV"] = [round(roc_auc_score(yd, D_res[n]), 4) for n in T["variante"]]
    T["LL DiamV"] = [round(log_loss(yd, D_res[n]), 5) for n in T["variante"]]

    rng = np.random.default_rng(1)
    Bidx = rng.integers(0, len(yd), size=(200, len(yd)))
    dl = []
    for n in T["variante"]:
        if n == "Escalón 9 oficial":
            dl.append("+0.0000 [ref]")
        else:
            p = D_res[n]
            d = roc_auc_score(yd, p) - roc_auc_score(yd, pd_ref)
            ds = np.array([roc_auc_score(yd[s], p[s]) - roc_auc_score(yd[s], pd_ref[s]) for s in Bidx])
            dl.append(f"{d:+.4f} [{np.percentile(ds, 2.5):+.4f}; {np.percentile(ds, 97.5):+.4f}]")
    T["Δ DiamV (IC)"] = dl

    print("\n=== RESULTADOS G38: PODA Y STACKING CAUSAL DE BLOQUES ===")
    mostrar(T)


if __name__ == "__main__":
    main()
