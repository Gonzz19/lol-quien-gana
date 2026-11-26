"""
¿El jugador convierte la ventaja o la tira? ¿Remonta o se cae? (pedido del usuario, 2026-11-17; §9.S.57)

Variables del modelo en vivo que dependen del estado de la partida y de cómo rinde cada jugador según ese estado:

1. Modelo del estado: P(gana azul | Δoro del equipo al minuto m), logística ajustada con los tramos anteriores a
   A2 (chunks 1-3). Residuo de un equipo en una partida = ganó − P(su equipo gana | Δoro).
2. Por jugador, en línea y en orden cronológico (solo partidas anteriores, vivas al minuto m):
   * conv  = media encogida del residuo cuando su equipo iba arriba (Δoro > +1k): convierte o tira;
   * remo  = media encogida del residuo cuando iba abajo (Δoro < −1k): remonta o se cae;
   * lo mismo con el oro de su propia línea (±500): conv_l, remo_l.
   Encogimiento hacia 0 con k = 10 partidas.
3. En vivo, al minuto m, con w = σ(Δoro / 1k) (peso de "va arriba el azul"):
   F = w · (Σ conv azul − Σ remo rojo) + (1 − w) · (Σ remo azul − Σ conv rojo)   (antisimétrica)
   y la versión por línea con el oro de cada línea; más los totales sin estado (Σ conv, Σ remo azul − rojo).

Se agrega a la mejor logística en vivo (`3_Abl_A2_Stats_Traj_Exp` de `ingame_evaluar.py`) con el mismo
protocolo: A2-A4 hacia adelante (chunks 4-6) y V Master+ / Diamond-V una sola vez, C = 0,01.

    python historial/en_vivo/ingame_convertir.py      # ~5 min
"""
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.models.ingame import MINUTES, TIMELINE_STATS, extract_minute_features  # noqa: E402
from scripts.analysis.ingame_evaluar import _delta_ci, _fit_logreg_antisym, _prep_arrays, _sigm  # noqa: E402

IN_DIR = REPO / "data" / "analysis" / "ingame"
K, UMBRAL_EQ, UMBRAL_L, C_LR = 10.0, 1.0, 0.5, 0.01
T0 = time.time()
log = lambda s: print(f"[{time.time() - T0:6.1f}s] {s}", flush=True)  # noqa: E731


def rasgos(mi, y, dur, tl, pid, chunk, orden):
    """conv / remo por jugador (equipo y línea) al minuto MINUTES[mi], leídos antes de cada partida."""
    m = MINUTES[mi]
    gi = TIMELINE_STATS.index("gold")
    g_l = tl[:, 0, :, mi, gi] - tl[:, 1, :, mi, gi]                     # (NM, 5) azul − rojo; tl_raw ya está en miles
    g_t = g_l.sum(1)
    vivo = (dur > 60.0 * m) & np.isfinite(g_t)
    fit = vivo & np.isin(chunk, [1, 2, 3])
    est = LogisticRegression(C=1e4).fit(g_t[fit, None], y[fit])
    p_b = np.where(vivo, est.predict_proba(np.nan_to_num(g_t)[:, None])[:, 1], 0.5)
    S = np.zeros((pid.max() + 1, 4))                                     # sumas: conv, remo, conv_l, remo_l
    N = np.zeros((pid.max() + 1, 4))
    R = np.zeros((len(y), 2, 5, 4), np.float32)
    print(f"    m={m}: |Δoro equipo| > {UMBRAL_EQ}k en {np.mean(np.abs(g_t[vivo]) > UMBRAL_EQ):.0%} de las partidas vivas",
          flush=True)
    for i in orden:
        R[i] = S[pid[i]] / (N[pid[i]] + K)
        if not vivo[i]:
            continue
        for s in (0, 1):
            sg = 1.0 if s == 0 else -1.0
            r = (y[i] - p_b[i]) * sg                                    # residuo de su equipo
            ge = g_t[i] * sg
            for l in range(5):
                p = pid[i, s, l]
                gl = g_l[i, l] * sg
                if ge > UMBRAL_EQ:
                    S[p, 0] += r; N[p, 0] += 1
                elif ge < -UMBRAL_EQ:
                    S[p, 1] += r; N[p, 1] += 1
                if gl > UMBRAL_L:
                    S[p, 2] += r; N[p, 2] += 1
                elif gl < -UMBRAL_L:
                    S[p, 3] += r; N[p, 3] += 1
    return R, g_l, g_t


def variables(R, g_l, g_t):
    w = 1.0 / (1.0 + np.exp(-np.nan_to_num(g_t)))                       # va arriba el azul
    cb, rb = R[:, 0, :, 0].sum(1), R[:, 0, :, 1].sum(1)
    cr, rr = R[:, 1, :, 0].sum(1), R[:, 1, :, 1].sum(1)
    F_eq = w * (cb - rr) + (1 - w) * (rb - cr)
    wl = 1.0 / (1.0 + np.exp(-np.nan_to_num(g_l) / 0.5))                # por línea
    F_l = (wl * (R[:, 0, :, 2] - R[:, 1, :, 3]) + (1 - wl) * (R[:, 0, :, 3] - R[:, 1, :, 2])).sum(1)
    return np.c_[F_eq, F_l, cb - cr, rb - rr].astype(np.float32), ["conv_remo_equipo", "conv_remo_linea",
                                                                   "conv_tot", "remo_tot"]


def main():
    C = np.load(IN_DIR / "ingame_cache.npz", allow_pickle=True)
    y, t, chunk = C["y"].astype(np.int64), C["t"], C["chunk"].astype(np.int32)
    master, val, diam_v = C["master"].astype(bool), C["val"].astype(bool), C["diam_v"].astype(bool)
    dur, L_oof = C["dur_sec"].astype(np.float32), C["L_oof"].astype(np.float32)
    tl, gold_resid = C["tl_raw"].astype(np.float32), C["gold_resid"].astype(np.float32)
    exp_lane = {k[4:]: C[k].astype(np.float32) for k in C.files if k.startswith("exp_")}
    exp_trans = {k[3:]: C[k].astype(np.float32) for k in C.files if k.startswith("tr_")}
    B = np.load(REPO / "data" / "analysis" / "brainstorm" / "base.npz", allow_pickle=True)
    assert len(B["player"]) == len(y)
    pid = np.unique(B["player"].reshape(-1), return_inverse=True)[1].reshape(len(y), 2, 5)
    orden = np.lexsort((C["mids"].astype(str), t))
    NM = len(y)
    filas = []
    for mi, m in enumerate(MINUTES):
        R, g_l, g_t = rasgos(mi, y, dur, tl, pid, chunk, orden)
        X_new, nombres = variables(R, g_l, g_t)
        log(f"minuto {m}: rasgos listos")
        ok_tl = np.isfinite(tl[:, :, :, : mi + 1, :]).all(axis=(1, 2, 3, 4))
        ok_m = (dur > 60.0 * m) & ok_tl & np.isfinite(L_oof)
        v_idx, dv_idx = np.flatnonzero(val & ok_m), np.flatnonzero(diam_v & ok_m)
        folds = [(b, np.flatnonzero((chunk == b) & master & ok_m)) for b in (4, 5, 6)]
        tr_full = np.flatnonzero(np.isin(chunk, [1, 2, 3, 4, 5, 6]) & master & ok_m)
        base = extract_minute_features(m, L_oof, tl, gold_resid, exp_lane=exp_lane, exp_trans=exp_trans,
                                       include_all_stats=True, include_traj=True, include_expected=True,
                                       include_comp_cross=False, include_player=False)
        variantes = {"A2 (mejor logística en vivo)": base}
        for nom, cols in (("+ convierte / remonta (equipo y línea)", [0, 1]),
                          ("+ convierte / remonta + totales sin estado", [0, 1, 2, 3])):
            f = dict(base)
            f["X_anti"] = np.c_[base["X_anti"], X_new[:, cols]].astype(np.float32)
            variantes[nom] = f
        P_a, P_v, P_d = {}, {}, {}
        for nom, f in variantes.items():
            pa = np.full(NM, np.nan, np.float32)
            for b, te in folds:
                tr_b = np.flatnonzero(np.isin(chunk, [1, 2, 3, 4, 5, 6]) & (chunk < b) & master & ok_m)
                ar = _prep_arrays(f, tr_b, te, L_oof, y)
                pa[te] = _sigm(_fit_logreg_antisym(ar["Xa_tr"], y[tr_b], ar["Xa_te"], C=C_LR)[1])
            te_vd = np.r_[v_idx, dv_idx]
            ar = _prep_arrays(f, tr_full, te_vd, L_oof, y)
            p = _sigm(_fit_logreg_antisym(ar["Xa_tr"], y[tr_full], ar["Xa_te"], C=C_LR)[1])
            P_a[nom], P_v[nom], P_d[nom] = pa, p[: len(v_idx)], p[len(v_idx):]
        ref = "A2 (mejor logística en vivo)"
        rng = np.random.default_rng(7 + m)
        boot = rng.integers(0, len(v_idx), size=(500, len(v_idx)))
        yv, ydv = y[v_idx], y[dv_idx]
        for nom in variantes:
            blo = []
            for b, te in folds:
                blo.append(roc_auc_score(y[te], P_a[nom][te]) - roc_auc_score(y[te], P_a[ref][te]))
            d, lo, hi = _delta_ci(yv, P_v[nom], P_v[ref], boot)
            fila = {"minuto": m, "variante": nom, "AUC A2": roc_auc_score(y[folds[0][1]], P_a[nom][folds[0][1]]),
                    "AUC A3": roc_auc_score(y[folds[1][1]], P_a[nom][folds[1][1]]),
                    "AUC A4": roc_auc_score(y[folds[2][1]], P_a[nom][folds[2][1]]),
                    "Δ bloques": float(np.mean(blo)), "bloques que mejora": int(sum(x > 0 for x in blo)),
                    "AUC V": roc_auc_score(yv, P_v[nom]), "Accuracy V": float(((P_v[nom] > 0.5) == yv).mean()),
                    "Δ V": d, "Δ V lo": lo, "Δ V hi": hi, "AUC Diamond-V": roc_auc_score(ydv, P_d[nom])}
            filas.append(fila)
            print(f"  {nom:45s} bloques {fila['Δ bloques']:+.4f} ({fila['bloques que mejora']}/3) | "
                  f"V {fila['AUC V']:.4f} / {fila['Accuracy V']:.2%} | Δ V {d:+.4f} [{lo:+.4f}; {hi:+.4f}] | "
                  f"DiamV {fila['AUC Diamond-V']:.4f}", flush=True)
        pd.DataFrame(filas).to_csv(IN_DIR / "convertir_remontar.csv", index=False)
    log("listo")


if __name__ == "__main__":
    main()
