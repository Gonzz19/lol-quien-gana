"""Idea 20 del brainstorming (docs/BRAINSTORM_NOCHE.md): HGB por línea @15 estrictamente causal
(forward-chaining por bloques cronológicos, SIN cross-fitting futuro como §9.S.36) y entrenado
SOLO con Master+ + MC/ML + mmr_historico + atributos de campeón + counter causal.

Evaluado sobre la nueva referencia vigente (+ MC/ML + niv_hist = 0,6642 en V), con n_jobs=1.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.brainstorm_base import ajustar_predecir, cargar, comparar, evaluar, extras_referencia, mostrar  # noqa: E402
from historial.escalon9.brainstorm_g_06_comp_poisson import construir_composicion  # noqa: E402

CACHE = REPO / "data" / "analysis" / "brainstorm" / "gemini_hgb_lineas.npz"
LANES = ["BOTTOM", "JUNGLE", "MIDDLE", "TOP", "UTILITY"]


def cargar_target_oro15(D: dict) -> np.ndarray:
    mids = D["Z"]["mids"]
    df = pd.read_csv(
        REPO / "data" / "splits" / "train" / "full_dataset.csv",
        usecols=["match_id", "team_id", "lane", "gold_15", "game_duration"],
    )
    df = df[df.match_id.isin(set(mids)) & df.lane.isin(LANES)].copy()
    df["g15"] = np.where(df["game_duration"] > 900, df["gold_15"], np.nan)
    w = df.pivot_table(index="match_id", columns=["team_id", "lane"], values="g15", aggfunc="first").reindex(mids)
    Y = np.c_[[(w[(100, l)] - w[(200, l)]).to_numpy() / 1000.0 for l in LANES]].T
    return Y.astype(np.float32)


def construir_hgb_causal(D: dict) -> np.ndarray:
    t0 = time.time()
    NM, t, ms, fit0 = D["NM"], D["t"], D["master"], ~D["val0"]
    C = np.load(REPO / "data" / "analysis" / "brainstorm" / "margen_grupos.npz")
    A_mmr = np.load(REPO / "data" / "analysis" / "brainstorm" / "mmr_historico.npz")["A"].astype(np.float32)
    _, J_CAMP = construir_composicion(D)
    mpr_path = REPO / "data" / "analysis" / "brainstorm" / "gemini_mpr.npz"
    MT = np.load(mpr_path)["MATCHUPS"][:, :5] if mpr_path.exists() else np.zeros((NM, 5), np.float32)

    # Tensor (NM, 2, 5, d_pre) por jugador (solo A_mmr[..., :3] como Escalón 2/3)
    P_jug = np.concatenate([D["Xj"], C["MC"], C["ML"], A_mmr[..., :3], J_CAMP], axis=-1).astype(np.float32)
    Y_g15 = cargar_target_oro15(D)

    # Cortes cronológicos estrictos dentro de fit0 (< 16/09); chunks >= t_16_09 predicen con todo fit0
    fi0 = np.flatnonzero(fit0)
    qs = np.quantile(t[fi0], [0.10, 0.20, 0.35, 0.50, 0.65, 0.78, 0.85, 0.90, 0.95])
    chunk = np.searchsorted(qs, t, side="right")  # 0..9 en fi0; >=9 en fit0 termina antes de 16/09

    PRED = np.zeros((NM, 5), dtype=np.float32)
    targets = [(k, np.flatnonzero((chunk == k) & fit0)) for k in range(1, 10)]
    targets.append((10, np.flatnonzero(~fit0)))

    for l in range(5):
        Xa_all = np.c_[P_jug[:, 0, l], P_jug[:, 1, l], MT[:, l:l + 1]]
        Xr_all = np.c_[P_jug[:, 1, l], P_jug[:, 0, l], -MT[:, l:l + 1]]
        tl0 = time.time()
        for b, te in targets:
            if len(te) == 0:
                continue
            t_lim = int(t[te].min())
            if b <= 9:
                tr = np.flatnonzero((chunk < b) & (t < t_lim) & fit0 & ms & np.isfinite(Y_g15[:, l]))
            else:
                tr = np.flatnonzero(fit0 & ms & np.isfinite(Y_g15[:, l]))
            if len(tr) < 500:
                continue
            yt = Y_g15[tr, l]
            hgb = HistGradientBoostingRegressor(
                max_iter=250,
                learning_rate=0.05,
                max_leaf_nodes=31,
                min_samples_leaf=200,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42,
            )
            hgb.fit(np.r_[Xa_all[tr], Xr_all[tr]], np.r_[yt, -yt])
            PRED[te, l] = 0.5 * (hgb.predict(Xa_all[te]) - hgb.predict(Xr_all[te]))
        vi = D["val"] & np.isfinite(Y_g15[:, l])
        corr = np.corrcoef(Y_g15[vi, l], PRED[vi, l])[0, 1]
        auc_l = roc_auc_score(Y_g15[vi, l] > 0, PRED[vi, l])
        print(f"  HGB {LANES[l]:6s} V corr={corr:.3f} AUC_linea={auc_l:.4f} ({time.time() - tl0:.1f}s)", flush=True)

    return np.c_[PRED, PRED.sum(axis=1)].astype(np.float32)


def main():
    rehacer = "--rehacer" in sys.argv
    solo_cache = "--solo-cache" in sys.argv
    D = cargar()
    ms = D["master"]
    if rehacer or not CACHE.exists():
        print("Construyendo HGB por línea causal (solo Master+, forward-chaining)...")
        HGB = construir_hgb_causal(D)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, HGB=HGB)
    if solo_cache:
        print(f"[+] {CACHE}")
        return
    HGB = np.load(CACHE)["HGB"]
    REF6 = extras_referencia(D, escalon=6)

    def solo_m(extra=None):
        tot = REF6 if extra is None else np.c_[REF6, extra].astype(np.float32)
        return lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=tot, extra_post=False, con_mapa=False)

    P = {}
    ref_k = "ref: Escalón 6 (sin mapa)"
    P[ref_k] = evaluar(D, solo_m())
    P["+ HGB línea @15 causal (5+suma)"] = evaluar(D, solo_m(HGB))
    P["+ solo suma HGB línea @15"] = evaluar(D, solo_m(HGB[:, -1:]))
    mostrar(comparar(D, P, ref_k))

    # Chequeo ciego Diamond-V
    fi_v, dv = D["fi_v"], D["diam_v"]
    ydv = D["y"][dv]
    p_dv_ref = solo_m()(fi_v, dv)
    auc_ref = roc_auc_score(ydv, p_dv_ref)
    rng = np.random.default_rng(42)
    boots = rng.integers(0, len(ydv), size=(1500, len(ydv)))
    print(f"\nChequeo Diamond-V ({len(ydv):,} partidas): ref={auc_ref:.4f}")
    for nom, ex in [("+ HGB línea @15 causal", HGB), ("+ solo suma HGB", HGB[:, -1:])]:
        p_dv = solo_m(ex)(fi_v, dv)
        auc_p = roc_auc_score(ydv, p_dv)
        ds = np.array([roc_auc_score(ydv[b], p_dv[b]) - roc_auc_score(ydv[b], p_dv_ref[b]) for b in boots])
        print(f"  {nom:28s} AUC {auc_p:.4f}  Δ {auc_p - auc_ref:+.4f} [{np.percentile(ds, 2.5):+.4f}; {np.percentile(ds, 97.5):+.4f}]")

    np.savez_compressed(
        REPO / "data" / "analysis" / "brainstorm" / "pred_g09.npz",
        **{f"o_{i}": P[n][0] for i, n in enumerate(P)},
        **{f"v_{i}": P[n][1] for i, n in enumerate(P)},
        names=np.array(list(P)),
    )


if __name__ == "__main__":
    main()
