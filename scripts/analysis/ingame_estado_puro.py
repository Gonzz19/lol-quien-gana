"""Modelo en vivo con **solo el estado** de la partida, sin nada previo (PROJECT_SUMMARY.md §9.S.60).

Para la sección 3 de `eda_modelos_ganador` ("cuánto se decide durante la partida"): la barra
"solo el estado" tiene que salir de las mismas filas que el modelo vigente (`ingame_g_04_mt_a4_gpu.py`)
y no contener información anterior a la partida. La versión "sin previo" de `ingame_g_05_sin_previo.py`
apaga `L_orig` pero conserva las expectativas por línea (Ridge + LightGBM sobre draft, composición y
jugadores), la composición y los rasgos de jugador; acá se apaga todo eso:

  * `L_orig = 0`, sin expectativas, sin composición/cruces, sin jugador.
  * Quedan las 8 estadísticas del timeline por línea al minuto m, el oro residual (oro que no explican
    CS, kills y asistencias en ese minuto: placas, torres, recompensas; estado puro, con coeficientes
    ajustados en el pasado) y las trayectorias desde los minutos anteriores (A2 sin lo previo).
  * También la versión básica: Δoro, Δxp y Δkills del equipo (la de `brainstorm_39`).

Logística espejada de `ingame_evaluar.py`, C elegido en A2-A4 hacia adelante, V Master+ ← chunks 1-6.
CPU, < 2 GB, ~3 min.

Uso:
    python scripts/analysis/ingame_estado_puro.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.analysis.ingame_evaluar import _fit_logreg_antisym, _prep_arrays, _sigm  # noqa: E402
from src.models.ingame import MINUTES, TIMELINE_STATS, extract_minute_features  # noqa: E402

IN_DIR = REPO / "data" / "analysis" / "ingame"
T0 = time.time()


def _basico(tl_raw: np.ndarray, mi: int) -> np.ndarray:
    si = {s: i for i, s in enumerate(TIMELINE_STATS)}
    tot = np.nan_to_num(tl_raw[:, :, :, mi, :]).sum(axis=2)  # (NM, 2, 8)
    d = tot[:, 0] - tot[:, 1]
    return np.column_stack([d[:, si["gold"]], d[:, si["xp"]], d[:, si["kills"]]]).astype(np.float32)


def _fit_basico(X: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(X[tr] ** 2, axis=0)) + 1e-6
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(X[tr] / rms, y[tr])
    return lr.predict_proba(X[te] / rms)[:, 1].astype(np.float32)


def main() -> None:
    C = np.load(IN_DIR / "ingame_cache.npz", allow_pickle=True)
    t = C["t"].astype(np.int64)
    y = C["y"].astype(np.int64)
    chunk = C["chunk"].astype(np.int32)
    master = C["master"].astype(bool)
    val = C["val"].astype(bool)
    diam_v = C["diam_v"].astype(bool)
    dur_sec = C["dur_sec"].astype(np.float32)
    L_orig = C["L_oof"].astype(np.float32)
    tl_raw = C["tl_raw"].astype(np.float32)
    gold_resid = C["gold_resid"].astype(np.float32)
    del C
    L0 = np.zeros_like(L_orig)

    filas = []
    for mi, m in enumerate(MINUTES):
        # mismas filas que ingame_g_04 (incluido isfinite(L_orig), aunque acá L no se usa)
        ok_tl = np.isfinite(tl_raw[:, :, :, : mi + 1, :]).all(axis=(1, 2, 3, 4))
        ok_m = (dur_sec > 60.0 * m) & ok_tl & np.isfinite(L_orig)
        v_idx = np.flatnonzero(val & ok_m)
        dv_idx = np.flatnonzero(diam_v & ok_m)
        folds = [(b, np.flatnonzero((chunk == b) & master & ok_m)) for b in (4, 5, 6)]
        a24 = np.concatenate([te for _, te in folds])
        tr_full = np.flatnonzero(np.isin(chunk, [1, 2, 3, 4, 5, 6]) & master & ok_m)

        f = extract_minute_features(m, L0, tl_raw, gold_resid, include_all_stats=True, include_traj=True,
                                    include_expected=False, include_comp_cross=False, include_player=False)
        Xb = _basico(tl_raw, mi)

        preds = {}
        # estado completo (A2 sin lo previo): C elegido por log loss en A2-A4 hacia adelante
        best = (np.inf, 0.03)
        for c in (0.01, 0.03, 0.05, 0.1):
            pa = np.full(len(y), np.nan, np.float32)
            for b, te in folds:
                tr_b = np.flatnonzero(np.isin(chunk, [1, 2, 3, 4, 5, 6]) & (chunk < b) & master & ok_m)
                ar = _prep_arrays(f, tr_b, te, L0, y)
                pa[te] = _sigm(_fit_logreg_antisym(ar["Xa_tr"], y[tr_b], ar["Xa_te"], C=c)[1])
            ll = log_loss(y[a24], pa[a24])
            if ll < best[0]:
                best, pa_best = (ll, c), pa
        te_vd = np.r_[v_idx, dv_idx]
        ar = _prep_arrays(f, tr_full, te_vd, L0, y)
        p_vd = _sigm(_fit_logreg_antisym(ar["Xa_tr"], y[tr_full], ar["Xa_te"], C=best[1])[1])
        preds["estado_completo"] = (pa_best, p_vd, best[1])

        pa = np.full(len(y), np.nan, np.float32)
        for b, te in folds:
            tr_b = np.flatnonzero(np.isin(chunk, [1, 2, 3, 4, 5, 6]) & (chunk < b) & master & ok_m)
            pa[te] = _fit_basico(Xb, y, tr_b, te)
        preds["estado_basico"] = (pa, _fit_basico(Xb, y, tr_full, te_vd), 1.0)

        for nom, (pa, p_vd, c) in preds.items():
            pv, pdv = p_vd[: len(v_idx)], p_vd[len(v_idx):]
            fila = {
                "minuto": m, "modelo": nom, "C": c, "n_V": len(v_idx), "n_DV": len(dv_idx),
                "AUC_A2": roc_auc_score(y[folds[0][1]], pa[folds[0][1]]),
                "AUC_A3": roc_auc_score(y[folds[1][1]], pa[folds[1][1]]),
                "AUC_A4": roc_auc_score(y[folds[2][1]], pa[folds[2][1]]),
                "AUC_V": roc_auc_score(y[v_idx], pv), "Acc_V": float(((pv > 0.5) == y[v_idx]).mean()),
                "LogLoss_V": log_loss(y[v_idx], pv),
                "AUC_DiamV": roc_auc_score(y[dv_idx], pdv), "Acc_DiamV": float(((pdv > 0.5) == y[dv_idx]).mean()),
            }
            filas.append(fila)
            print(f"[{time.time() - T0:5.0f}s] min {m} {nom:16s} C={c:<5} V {fila['AUC_V']:.4f} / "
                  f"{fila['Acc_V']:.2%}  DiamV {fila['AUC_DiamV']:.4f} / {fila['Acc_DiamV']:.2%}  "
                  f"A2-A4 {fila['AUC_A2']:.4f} {fila['AUC_A3']:.4f} {fila['AUC_A4']:.4f}", flush=True)

    out = IN_DIR / "estado_puro.csv"
    pd.DataFrame(filas).to_csv(out, index=False)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
