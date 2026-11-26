"""Idea 10 del brainstorming (docs/BRAINSTORM_NOCHE.md): W/L y volumen de temporada de Foto 1 + lp_blend.

En el pipeline actual (`ganador_mapa_lp.py` / `brainstorm_base.py`), de la Foto 1 (`players`,
04/09 22:51 UTC) solo se usa `lp_foto`, descartando `wins` y `losses` de la temporada y si el
jugador está en el piso de Master `0 LP`. En `lp_vivo.db` (4 servidores reales), a igual LP el
winrate de temporada y `log1p(W + L)` predicen +0,045 de AUC sobre el LP real porque miden el
rezago entre el MMR oculto y el LP visible (escaladores/smurfs vs cuentas estancadas).

Evaluación estrictamente causal sobre la referencia vigente (`solo Master+`, V 0,6557 / 61,3 %):
* `+ Foto 1 W/L estático`: `wr_f1` (Bayes k=20), `log1p(W+L)`, `W - L`, indicador `Master 0 LP`.
* `+ Foto 1 W/L dinámico causal`: `W_f1 + dW_post`, `L_f1 + dL_post` (solo partidas >= t1 previas).
* `+ lp_blend`: promedio ponderado por precisión `1 / (1 + (lp_incert / 40)^2)` entre `lp_causal` y `niv_v10`.
* `+ Foto 1 W/L + lp_blend` (con estático y con dinámico).

Todo se centra y estandariza estrictamente dentro de `tr[post[tr]]` (`extra_post=True`).
"""
import pickle
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.brainstorm_base import ajustar_predecir, cargar, comparar, evaluar, mostrar  # noqa: E402
from src.features.padron import verificar_foto1  # noqa: E402
from src.features.lobby import prior_lobby_level  # noqa: E402


def construir_extras(D: dict) -> dict[str, np.ndarray]:
    NM = D["NM"]
    B, Z = D["B"], D["Z"]
    pl = B["player"]  # (NM, 2, 5) strings
    post = D["post"]  # (NM,)
    pr = np.repeat(post, 10)

    with sqlite3.connect(f"file:{REPO / 'data' / 'lol_las.db'}?mode=ro&immutable=1", uri=True) as con:
        verificar_foto1(con)            # foto 1 congelada (§9.S.67)
        foto = pd.read_sql_query(
            "SELECT m.player_id, p.tier, p.league_points, p.wins, p.losses "
            "FROM players p JOIN player_id_map m ON p.puuid = m.puuid",
            con,
        ).drop_duplicates("player_id")

    f_map = foto.set_index("player_id")
    pids_flat = pl.reshape(-1)
    idx_valid = f_map.index
    has_raw = pd.Series(pids_flat).isin(idx_valid).to_numpy()
    has_f1 = has_raw & pr

    # Lookup vectorizado alineado al orden exacto (NM, 2, 5)
    aligned = f_map.reindex(pids_flat)
    W0 = aligned["wins"].fillna(0.0).to_numpy(np.float64)
    L0 = aligned["losses"].fillna(0.0).to_numpy(np.float64)
    zero_m = (
        has_f1
        & (aligned["tier"].fillna("").to_numpy() == "MASTER")
        & (aligned["league_points"].fillna(-1).to_numpy() == 0)
    ).astype(np.float32)

    def eq_4(wr: np.ndarray, logn: np.ndarray, netw: np.ndarray, zm: np.ndarray) -> np.ndarray:
        a = np.stack([wr.reshape(NM, 2, 5), logn.reshape(NM, 2, 5), netw.reshape(NM, 2, 5), zm.reshape(NM, 2, 5)], axis=-1)
        return (a[:, 0] - a[:, 1]).mean(axis=1).astype(np.float32)

    # 1. Estático desde Foto 1 (0 cuando ~post o sin foto; constante fija para mantener centrado)
    tot0 = W0 + L0
    wr_f1 = np.where(has_f1, (W0 + 10.0 * 0.5) / (tot0 + 20.0) - 0.5, 0.0).astype(np.float32)
    logn_f1 = np.where(has_f1, np.log1p(tot0) - 5.3, 0.0).astype(np.float32)
    netw_f1 = np.where(has_f1, W0 - L0, 0.0).astype(np.float32)
    FOTO1_EST = eq_4(wr_f1, logn_f1, netw_f1, zero_m)

    # 2. Dinámico causal post-t1: suma dW_obs y dL_obs de partidas anteriores desde t1
    y_jug = np.repeat(D["y"], 10).astype(np.float64)
    # y == 1 significa azul (primeros 5 slots de cada partida); slots 5..9 son rojo
    es_azul = np.tile(np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=bool), NM)
    win_slot = np.where(es_azul, y_jug, 1.0 - y_jug)

    df_post = pd.DataFrame({
        "_pos": np.arange(NM * 10),
        "player_id": pids_flat,
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

    Wt = W0 + dw_all
    Lt = L0 + dl_all
    tot_t = Wt + Lt
    wr_t = np.where(has_f1, (Wt + 10.0 * 0.5) / (tot_t + 20.0) - 0.5, 0.0).astype(np.float32)
    logn_t = np.where(has_f1, np.log1p(tot_t) - 5.3, 0.0).astype(np.float32)
    netw_t = np.where(has_f1, Wt - Lt, 0.0).astype(np.float32)
    FOTO1_DIN = eq_4(wr_t, logn_t, netw_t, zero_m)

    # 3. lp_blend ponderado por precisión desde mapa_lp_causal_v.pkl
    with open(REPO / "data" / "analysis" / "mapa_lp_causal_v.pkl", "rb") as f:
        cache = pickle.load(f)
    dpc, keys = cache["df_post_cron"], cache["df_npz_keys"].copy()
    fit_post = post & ~D["val0"]
    en_fp = dpc.match_id.isin(set(Z["mids"][fit_post]))
    mu_m = float(dpc.loc[en_fp, "lp_causal"].mean())
    nm_ = prior_lobby_level(dpc, "lp_causal", k=5, mu0=mu_m, window=10, min_otros=3)
    F = dpc[["match_id", "player_id", "lp_foto", "lp_causal", "lp_incert", "metodo"]].copy()
    F["mapa_v10"] = nm_["nivel_v10"].to_numpy(float)
    keys["_pos"] = np.arange(len(keys))
    M = keys.merge(F, on=["match_id", "player_id"], how="left").sort_values("_pos", kind="mergesort").reset_index(drop=True)
    tiene_lp = (M.lp_foto.notna() & pr).to_numpy()
    inc_raw = M.lp_incert.fillna(150.0).to_numpy(np.float32)
    w_lp = np.where(tiene_lp, 1.0 / (1.0 + (inc_raw / 40.0) ** 2), 0.0).astype(np.float32)
    lp_val = np.where(M.metodo.to_numpy() == "causal_lobby", M.mapa_v10.fillna(mu_m), M.lp_causal.fillna(mu_m)).astype(np.float32)
    niv_val = M.mapa_v10.fillna(mu_m).to_numpy(np.float32)
    blend_val = np.where(pr, w_lp * lp_val + (1.0 - w_lp) * niv_val - mu_m, 0.0).astype(np.float32)
    ba = blend_val.reshape(NM, 2, 5)
    LP_BLEND = (ba[:, 0] - ba[:, 1]).mean(axis=1)[:, None].astype(np.float32)

    print(f"Cobertura Foto 1 en post: has_f1={has_f1[pr].mean():.2%}, zero_m={zero_m[pr].mean():.2%}")
    return {
        "FOTO1_EST": FOTO1_EST,
        "FOTO1_DIN": FOTO1_DIN,
        "LP_BLEND": LP_BLEND,
        "EST_BLEND": np.hstack([FOTO1_EST, LP_BLEND]),
        "DIN_BLEND": np.hstack([FOTO1_DIN, LP_BLEND]),
    }


def main():
    D = cargar()
    ms = D["master"]
    EX = construir_extras(D)

    def solo_m(extra=None):
        return lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=extra, extra_post=True)

    VAR = {
        "referencia: solo Master+": solo_m(),
        "+ Foto 1 W/L estático": solo_m(EX["FOTO1_EST"]),
        "+ Foto 1 W/L dinámico causal": solo_m(EX["FOTO1_DIN"]),
        "+ lp_blend": solo_m(EX["LP_BLEND"]),
        "+ Foto 1 W/L estático + lp_blend": solo_m(EX["EST_BLEND"]),
        "+ Foto 1 W/L dinámico + lp_blend": solo_m(EX["DIN_BLEND"]),
    }
    P = dict(Parallel(n_jobs=4, backend="threading")(delayed(lambda n: (n, evaluar(D, VAR[n])))(n) for n in VAR))
    mostrar(comparar(D, P, "referencia: solo Master+"))


if __name__ == "__main__":
    main()
