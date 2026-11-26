"""
¿Cuánto aporta cada grupo de eventos del timeline al modelo en vivo? (estimación rápida, pedido del usuario,
2026-11-18; amplía `ingame_eventos.py` y la §3.3 de eda_modelos_ganador)

Misma base y mismo esquema que `ingame_eventos.py`: logística en vivo ajustada con el ajuste Master+ sin las
partidas con timeline completo → logit fuera de la muestra; segunda etapa [logit base, eventos] ajustada con la
muestra de entrenamiento y medida en V Master+. Para cada grupo de eventos, dos números de ΔAUC en V con IC 95 %
por bootstrap pareado (300 réplicas), y la diferencia de accuracy (umbral 0,5):

* *solo:* base + ese grupo contra base;
* *arriba de los demás:* todos los grupos contra todos menos ese (lo que el grupo aporta y los otros no).

    python scripts/analysis/eventos_ablacion.py      # ~2 min con la caché de variables_eventos.npz
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.models.ingame import MINUTES, extract_minute_features  # noqa: E402
from scripts.analysis.ingame_evaluar import _delta_ci, _fit_logreg_antisym, _prep_arrays  # noqa: E402
from scripts.analysis.ingame_eventos import IN_DIR, P1, P2, SALIDA, construir, log  # noqa: E402

GRUPOS = {
    "Dragones (y punto de alma)": ["dragones", "alma_punto"],
    "Vacuolarvas": ["larvas"],
    "Heraldo": ["heraldo"],
    "Torres, placas e inhibidores": ["torres_ext", "torres_int", "torres_base", "placas", "inhibidores"],
    "Oro sin gastar / invertido": ["oro_sin_gastar", "oro_invertido"],
    "Muertos en el minuto y shutdowns": ["muertos_ahora", "shutdowns"],
    "Primera sangre (lado y minuto)": ["fb_lado", "fb_minuto_lado"],
}


def main():
    mids_e, grupos, XE = construir()
    nombres = P1 + P2
    col = {n: i for i, n in enumerate(nombres)}
    C = np.load(IN_DIR / "ingame_cache.npz", allow_pickle=True)
    y, chunk = C["y"].astype(np.int64), C["chunk"].astype(np.int32)
    master = C["master"].astype(bool)
    dur, L_oof = C["dur_sec"].astype(np.float32), C["L_oof"].astype(np.float32)
    tl, gold_resid = C["tl_raw"].astype(np.float32), C["gold_resid"].astype(np.float32)
    exp_lane = {k[4:]: C[k].astype(np.float32) for k in C.files if k.startswith("exp_")}
    exp_trans = {k[3:]: C[k].astype(np.float32) for k in C.files if k.startswith("tr_")}
    pos = {m: i for i, m in enumerate(C["mids"].astype(str))}
    fila = np.array([pos[m] for m in mids_e])
    en_muestra = np.zeros(len(y), bool)
    en_muestra[fila[grupos == "train"]] = True
    todas = [c for g in GRUPOS.values() for c in g]
    out = []
    for mi, m in enumerate(MINUTES):
        ok_tl = np.isfinite(tl[:, :, :, : mi + 1, :]).all(axis=(1, 2, 3, 4))
        ok_m = (dur > 60.0 * m) & ok_tl & np.isfinite(L_oof)
        f = extract_minute_features(m, L_oof, tl, gold_resid, exp_lane=exp_lane, exp_trans=exp_trans,
                                    include_all_stats=True, include_traj=True, include_expected=True,
                                    include_comp_cross=False, include_player=False)
        tr_base = np.flatnonzero(np.isin(chunk, [1, 2, 3, 4, 5, 6]) & master & ok_m & ~en_muestra)
        ok_e = np.isfinite(XE[m]).all(1) & ok_m[fila]
        idx_tr, idx_v = fila[(grupos == "train") & ok_e], fila[(grupos == "V") & ok_e]
        E = np.zeros((len(y), len(nombres)), np.float32)
        E[fila[ok_e]] = XE[m][ok_e]
        te = np.r_[idx_tr, idx_v]
        ar = _prep_arrays(f, tr_base, te, L_oof, y)
        zb = np.zeros(len(y), np.float32)
        zb[te] = _fit_logreg_antisym(ar["Xa_tr"], y[tr_base], ar["Xa_te"], C=0.01)[1]
        rms = np.sqrt((E[idx_tr] ** 2).mean(0)) + 1e-6

        def pred(cols):
            ci = [col[c] for c in cols]
            X = (lambda i: np.c_[zb[i], E[i][:, ci] / rms[ci]]) if ci else (lambda i: zb[i, None])
            lr = LogisticRegression(C=1.0, max_iter=3000).fit(X(idx_tr), y[idx_tr])
            return lr.predict_proba(X(idx_v))[:, 1]

        yv = y[idx_v]
        boot = np.random.default_rng(100 + m).integers(0, len(idx_v), size=(300, len(idx_v)))
        p_base, p_todo = pred([]), pred(todas)
        acc = lambda p: float(((p > 0.5) == yv).mean())  # noqa: E731
        a_base = acc(p_base)
        d, lo, hi = _delta_ci(yv, p_todo, p_base, boot)
        log(f"minuto {m}: V {len(idx_v):,}; base {roc_auc_score(yv, p_base):.4f} / {a_base:.2%}; "
            f"todos los grupos {d:+.4f} [{lo:+.4f}; {hi:+.4f}], accuracy {acc(p_todo) - a_base:+.2%}")
        out.append({"minuto": m, "grupo": "Todos juntos", "solo": d, "solo lo": lo, "solo hi": hi,
                    "arriba": np.nan, "arriba lo": np.nan, "arriba hi": np.nan,
                    "acc base": a_base, "Δacc solo": acc(p_todo) - a_base, "Δacc arriba": np.nan})
        for g, cols in GRUPOS.items():
            p_solo, p_sin = pred(cols), pred([c for c in todas if c not in cols])
            ds, ls, hs = _delta_ci(yv, p_solo, p_base, boot)
            da, la, ha = _delta_ci(yv, p_todo, p_sin, boot)
            ac_s, ac_a = acc(p_solo) - a_base, acc(p_todo) - acc(p_sin)
            out.append({"minuto": m, "grupo": g, "solo": ds, "solo lo": ls, "solo hi": hs,
                        "arriba": da, "arriba lo": la, "arriba hi": ha,
                        "acc base": a_base, "Δacc solo": ac_s, "Δacc arriba": ac_a})
            print(f"  {g:34s} solo {ds:+.4f} [{ls:+.4f}; {hs:+.4f}] acc {ac_s:+.2%} | arriba de los demás "
                  f"{da:+.4f} [{la:+.4f}; {ha:+.4f}] acc {ac_a:+.2%}", flush=True)
    pd.DataFrame(out).to_csv(SALIDA / "eventos_ablacion.csv", index=False)
    log("listo")


if __name__ == "__main__":
    main()
