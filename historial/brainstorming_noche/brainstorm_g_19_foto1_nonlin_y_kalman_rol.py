"""Ideas G37-G38 del brainstorming (docs/BRAINSTORM_NOCHE.md):
Evaluadas sobre el **Escalón 6 (sin mapa, 0,6709)** + chequeo ciego en **Diamond-V**:

1. **F1_NONLIN (9 cols causales post-t1):** Interacciones lineales faltantes de `Foto 1 + W/L dinámico` que explicaron
   parte de la ventaja de `Red NeuralAC`:
   - `zero_m * wr_f1` (distingue Master 0 LP hardstuck 48% WR vs recién ascendido 60% WR)
   - `wr_f1 * (logn_f1 + 5.3)` (winrate pesado por madurez de muestra) y `wr_f1 / (1 + exp(logn_f1))` (pendiente smurf)
   - `asinh((W - L) / 20)` y `tanh((W - L) / 40)` (escala acotada vs lineal cruda)
   - `max/min(wr_f1)` y `max/min(asinh_nw)` (carry más fuerte y eslabón más débil en Foto 1).
2. **KALMAN_ROL (8 cols causales `< t`):** Residuo de habilidad causal específico de la **línea jugada** (`r_rol[p, lane]`
   relativo al rating global `r_mrg[p]` del jugador usando el target mixto resultado/@15 de `g_11`) + rating rápido
   (`tau=10` partidas) - lento (`tau=45`) y dispersión `min/max/std` de `r_rol` del equipo.
3. **NIV_NONLIN (4 cols causales):** Extremos `min/max` del nivel histórico (`mmr_historico`), spread `max - min` del equipo
   y brecha de incertidumbre de nivel (`1/sqrt(n)`).

Caché guardada en `data/analysis/brainstorm/gemini_g19.npz`.
"""
import math
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.brainstorm_base import ajustar_predecir, cargar, comparar, evaluar, extras_referencia, mostrar  # noqa: E402

CACHE = REPO / "data" / "analysis" / "brainstorm" / "gemini_g19.npz"


def construir_f1_nonlin(D: dict) -> np.ndarray:
    """Construye 9 columnas no lineales de Foto 1 + W/L dinámico causal post-t1 (0 antes de t1)."""
    NM = D["NM"]
    pl = D["B"]["player"].reshape(-1)
    pr = np.repeat(D["post"], 10)
    with sqlite3.connect(f"file:{REPO / 'data' / 'lol_las.db'}?mode=ro&immutable=1", uri=True) as con:
        foto = pd.read_sql_query(
            "SELECT m.player_id, p.tier, p.league_points, p.wins, p.losses "
            "FROM players p JOIN player_id_map m ON p.puuid = m.puuid",
            con,
        ).drop_duplicates("player_id").set_index("player_id")
    has_f1 = pd.Series(pl).isin(foto.index).to_numpy() & pr
    al = foto.reindex(pl)
    W0 = al["wins"].fillna(0.0).to_numpy(np.float64)
    L0 = al["losses"].fillna(0.0).to_numpy(np.float64)
    zm = (has_f1 & (al["tier"].fillna("").to_numpy() == "MASTER") & (al["league_points"].fillna(-1).to_numpy() == 0)).astype(np.float64)

    y_jug = np.repeat(D["y"], 10).astype(np.float64)
    es_azul = np.tile(np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=bool), NM)
    win_slot = np.where(es_azul, y_jug, 1.0 - y_jug)
    df_post = pd.DataFrame({
        "_pos": np.arange(NM * 10),
        "player_id": pl,
        "t": np.repeat(D["t"], 10),
        "m": np.repeat(np.arange(NM), 10),
        "post": pr,
        "win": win_slot,
    })
    sub_p = df_post[df_post["post"]].sort_values(["t", "m"], kind="mergesort").copy()
    sub_p["dw_prev"] = sub_p.groupby("player_id", sort=False)["win"].cumsum() - sub_p["win"]
    sub_p["dn_prev"] = sub_p.groupby("player_id", sort=False).cumcount().astype(np.float64)
    sub_p["dl_prev"] = sub_p["dn_prev"] - sub_p["dw_prev"]
    dw_all = np.zeros(NM * 10, dtype=np.float64)
    dl_all = np.zeros(NM * 10, dtype=np.float64)
    dw_all[sub_p["_pos"].to_numpy()] = sub_p["dw_prev"].to_numpy()
    dl_all[sub_p["_pos"].to_numpy()] = sub_p["dl_prev"].to_numpy()

    W = W0 + dw_all
    L = L0 + dl_all
    tot = W + L
    wr = np.where(has_f1, (W + 10.0 * 0.5) / (tot + 20.0) - 0.5, 0.0)
    ln_raw = np.where(has_f1, np.log1p(tot), 0.0)
    nw_raw = np.where(has_f1, W - L, 0.0)

    c_zm_wr = np.where(has_f1, zm * wr, 0.0).reshape(NM, 2, 5)
    c_wr_ln = np.where(has_f1, wr * (ln_raw - 5.3), 0.0).reshape(NM, 2, 5)
    c_smurf = np.where(has_f1, wr / (1.0 + np.maximum(0.0, ln_raw - 3.0)), 0.0).reshape(NM, 2, 5)
    c_asinh = np.where(has_f1, np.arcsinh(nw_raw / 20.0), 0.0).reshape(NM, 2, 5)
    c_tanh = np.where(has_f1, np.tanh(nw_raw / 40.0), 0.0).reshape(NM, 2, 5)
    c_wr = wr.reshape(NM, 2, 5)

    def eq(a):
        return (a[:, 0].mean(1) - a[:, 1].mean(1))

    return np.column_stack([
        eq(c_zm_wr),
        eq(c_wr_ln),
        eq(c_smurf),
        eq(c_asinh),
        eq(c_tanh),
        c_wr[:, 0].max(1) - c_wr[:, 1].max(1),
        c_wr[:, 0].min(1) - c_wr[:, 1].min(1),
        c_asinh[:, 0].max(1) - c_asinh[:, 1].max(1),
        c_asinh[:, 0].min(1) - c_asinh[:, 1].min(1),
    ]).astype(np.float32)


def construir_cache_g19(D: dict, rehacer: bool = False) -> dict[str, np.ndarray]:
    if CACHE.exists() and not rehacer:
        z = np.load(CACHE)
        if int(z["nm"][0]) == D["NM"]:
            return {k: z[k] for k in z.files if k != "nm"}

    t0 = time.time()
    print("Construyendo caché gemini_g19.npz...", flush=True)
    NM = D["NM"]
    Z = D["Z"]
    mids = Z["mids"].astype(str)
    t_arr = D["t"].astype(np.int64)
    y_arr = D["y"].astype(np.int8)
    val0 = D["val0"].astype(bool)
    post = D["post"].astype(bool)
    dur_min = Z["D_dur_min"].astype(np.float32)
    dg15 = Z["D_d_gold_15"].astype(np.float32)
    dxp15 = Z["D_d_xp_15"].astype(np.float32)
    dk15 = Z["D_d_kills_15"].astype(np.float32)

    m_fit15 = (~val0) & (~post) & np.isfinite(dg15) & np.isfinite(dxp15) & np.isfinite(dk15) & (dur_min >= 15.0)
    X15 = np.c_[np.nan_to_num(dg15, 0.0) / 1000.0, np.nan_to_num(dxp15, 0.0) / 1000.0, np.nan_to_num(dk15, 0.0)]
    lr15 = LogisticRegression(C=1.0, max_iter=500).fit(X15[m_fit15], y_arr[m_fit15])
    p_exp15 = lr15.predict_proba(X15)[:, 1].astype(np.float64)
    valid15 = np.isfinite(dg15) & (dur_min >= 15.0)

    players = D["B"]["player"].reshape(NM, 10).astype(str)
    _, p_inv = np.unique(players.reshape(-1), return_inverse=True)
    pid_mat = p_inv.reshape(NM, 10).tolist()
    n_p = int(p_inv.max()) + 1

    F1_NONLIN = construir_f1_nonlin(D)

    KALMAN_ROL = np.zeros((NM, 8), dtype=np.float32)
    r_glo = [0.0] * n_p
    r_fast = [0.0] * n_p
    r_rol = [[0.0] * 5 for _ in range(n_p)]
    n_tot = [0] * n_p
    n_rol = [[0] * 5 for _ in range(n_p)]

    order = np.lexsort((mids, t_arr))
    for m in order:
        ps = pid_mat[m]
        yb = float(y_arr[m])
        # Rating especifico en esa linea (encogido hacia r_glo cuando tiene pocas partidas en el rol)
        eff = [0.0] * 10
        rdiff = [0.0] * 10
        fdiff = [0.0] * 10
        for s in range(10):
            p = ps[s]
            l = s % 5
            nr = n_rol[p][l]
            w_rol = nr / (nr + 12.0)
            rr_l = w_rol * r_rol[p][l] + (1.0 - w_rol) * r_glo[p]
            eff[s] = rr_l
            rdiff[s] = rr_l - r_glo[p]
            fdiff[s] = r_fast[p] - r_glo[p]

        KALMAN_ROL[m, 0] = (sum(eff[:5]) - sum(eff[5:])) * 0.2 / 100.0
        KALMAN_ROL[m, 1] = (sum(rdiff[:5]) - sum(rdiff[5:])) * 0.2 / 100.0
        KALMAN_ROL[m, 2] = (min(eff[:5]) - min(eff[5:])) / 100.0
        KALMAN_ROL[m, 3] = (max(eff[:5]) - max(eff[5:])) / 100.0
        KALMAN_ROL[m, 4] = (sum(fdiff[:5]) - sum(fdiff[5:])) * 0.2 / 100.0
        KALMAN_ROL[m, 5] = (rdiff[1] - rdiff[6]) / 100.0  # residual especifico de rol en JG
        KALMAN_ROL[m, 6] = (rdiff[0] + rdiff[4] - rdiff[5] - rdiff[9]) * 0.5 / 100.0  # bot+sup role residual
        KALMAN_ROL[m, 7] = (rdiff[2] + rdiff[3] - rdiff[7] - rdiff[8]) * 0.5 / 100.0  # solo lanes role residual

        mb = sum(r_glo[ps[i]] for i in range(5)) * 0.2
        mr = sum(r_glo[ps[5 + i]] for i in range(5)) * 0.2
        diff = (mb - mr) / 200.0 - 0.105
        p_hat = 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, diff))))
        tgt = 0.70 * yb + 0.30 * (float(p_exp15[m]) if valid15[m] else yb)
        err = tgt - p_hat
        lobby_m = (mb + mr) * 0.5

        for s in range(10):
            p = ps[s]
            l = s % 5
            sgn = 1.0 if s < 5 else -1.0
            n = n_tot[p]
            nr = n_rol[p][l]
            lr_g = 32.0 / math.sqrt(1.0 + n / 25.0)
            lr_r = 36.0 / math.sqrt(1.0 + nr / 15.0)
            lr_pull = 0.22 if n < 10 else 0.10 / math.sqrt(1.0 + (n - 10) / 35.0)
            other9 = (10.0 * lobby_m - r_glo[p]) / 9.0

            ng = r_glo[p] + lr_g * sgn * err + lr_pull * (other9 - r_glo[p])
            r_glo[p] = ng
            r_fast[p] = r_fast[p] + 22.0 * sgn * err + 0.12 * (other9 - r_fast[p])
            # El rating del rol tambien converge suavemente al pull del lobby
            r_rol[p][l] = r_rol[p][l] + lr_r * sgn * err + lr_pull * (other9 - r_rol[p][l])
            n_tot[p] = n + 1
            n_rol[p][l] = nr + 1

    A_mmr = np.load(REPO / "data" / "analysis" / "brainstorm" / "mmr_historico.npz")["A"].astype(np.float32)
    a0 = A_mmr[:, :, :, 0]
    NIV_NONLIN = np.column_stack([
        a0[:, 0].max(1) - a0[:, 1].max(1),
        a0[:, 0].min(1) - a0[:, 1].min(1),
        (a0[:, 0].max(1) - a0[:, 0].min(1)) - (a0[:, 1].max(1) - a0[:, 1].min(1)),
        a0[:, 0].std(1) - a0[:, 1].std(1),
    ]).astype(np.float32)

    out = {"F1_NONLIN": F1_NONLIN, "KALMAN_ROL": KALMAN_ROL, "NIV_NONLIN": NIV_NONLIN}
    np.savez_compressed(CACHE, nm=np.array([NM]), **out)
    print(f"Caché gemini_g19.npz lista en {time.time() - t0:.1f}s", flush=True)
    return out


def main():
    t0 = time.time()
    D = cargar()
    ms = D["master"]
    R6 = extras_referencia(D, escalon=6)
    G = construir_cache_g19(D)

    VAR = {
        "ref: Escalón 6 (sin mapa)": R6,
        "+ F1_NONLIN (9)": np.c_[R6, G["F1_NONLIN"]],
        "+ KALMAN_ROL (8)": np.c_[R6, G["KALMAN_ROL"]],
        "+ NIV_NONLIN (4)": np.c_[R6, G["NIV_NONLIN"]],
        "+ TODO g19 (21)": np.c_[R6, G["F1_NONLIN"], G["KALMAN_ROL"], G["NIV_NONLIN"]],
    }

    def run_one(nom, X):
        X32 = np.asarray(X, dtype=np.float32)
        return nom, evaluar(D, lambda tr, te, M=X32: ajustar_predecir(D, tr[ms[tr]], te, extra=M, extra_post=False, con_mapa=False))

    P = dict(Parallel(n_jobs=2, backend="threading")(delayed(run_one)(n, X) for n, X in VAR.items()))
    mostrar(comparar(D, P, "ref: Escalón 6 (sin mapa)"))

    di = np.flatnonzero(D["diam_v"])
    yd = D["y"][di]
    B = np.random.default_rng(1).integers(0, len(di), size=(300, len(di)))
    b0 = P["ref: Escalón 6 (sin mapa)"][0][di]
    print(f"\nChequeo Diamond-V ({len(di):,} partidas):")
    for nom in [k for k in VAR if not k.startswith("ref:")]:
        p = P[nom][0][di]
        d = np.array([roc_auc_score(yd[b], p[b]) - roc_auc_score(yd[b], b0[b]) for b in B])
        print(f"  {nom:24s} AUC {roc_auc_score(yd, p):.4f}  Δ {roc_auc_score(yd, p) - roc_auc_score(yd, b0):+.4f} "
              f"[{np.percentile(d, 2.5):+.4f}; {np.percentile(d, 97.5):+.4f}]")
    print(f"Listo en {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
