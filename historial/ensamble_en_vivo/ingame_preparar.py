"""Construye y cachea tensores causales para el modelo in-game (@10, @15 y @20).

Genera `data/analysis/ingame/ingame_cache.npz` con bajo consumo de RAM (<2 GB):
1. `tl_raw` `(NM, 2, 5, 3, 8)`: 8 stats del timeline escaladas y con fin de
   partida (`game_duration <= 60 * m`) enmascarado a NaN (§3.F.4).
2. `gold_resid` `(NM, 2, 5, 3)`: oro residual causal por chunk (proxy de
   placas, torres y bounties).
3. `L_oof` `(NM,)`: logit pre-partida fuera de muestra (forward-chaining sobre
   7 tramos cronológicos en `fit0` + `V Master+`, ampliando el entrenamiento
   de la Etapa 2 de 13.860 a ~92.000+ partidas Master+).
4. `exp_lane` y `exp_trans`: expectativas causales por carril (`gold, xp, cs,
   damage` a @10/@15/@20 y transiciones `@10→@15`, `@15→@20`).
5. `comp`, `tempo` (`cbias, cslope, pbias, pvol`) y `player_pre` (rasgos causales
   por jugador y carril).

Uso:
    python historial/ensamble_en_vivo/ingame_preparar.py
"""
from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.features.margin import ORDEN  # noqa: E402
from src.models.ingame import (  # noqa: E402
    LANES,
    MINUTES,
    TIMELINE_STATS,
    apply_residual_gold,
    build_composition_features,
    causal_champ_player_tempo,
    fit_residual_gold_coeffs,
    mask_finished_matches,
)
from src.models.validacion_temporal import bloques_por_fecha, particion_v_master  # noqa: E402
from src.ganador import rutas  # noqa: E402

OUT_DIR = REPO / "data" / "analysis" / "ingame"
CACHE_PATH = OUT_DIR / "ingame_cache.npz"
ATRIB_LOCAL = rutas.ATRIB_CSV
DDRAGON_CSV = REPO / "data" / "ddragon" / "16.17.1" / "champions.csv"
BR_DIR = REPO / "data" / "analysis" / "brainstorm"

PLAYER_COLS = [
    "acum_d_gold_15",
    "v20_d_gold_15",
    "acum_d_gold_10",
    "v20_d_gold_10",
    "acum_d_xp_15",
    "acum_d_cs_15",
    "wr_acum",
    "wr_v20",
    "estreno",
    "offrole",
    "logn_tot",
    "logn_champ",
    "share_lane",
    "acum_deaths_pm",
    "acum_vision_pm",
    "acum_damage_pm",
    "acum_gold_pm",
]

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def _cargar_atributos_campeones() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    atrib_path = ATRIB_LOCAL if ATRIB_LOCAL.exists() else None
    atrib_df = pd.read_csv(atrib_path) if atrib_path is not None else None
    ddragon_df = pd.read_csv(DDRAGON_CSV) if DDRAGON_CSV.exists() else None
    return atrib_df, ddragon_df


def _extraer_timeline_y_csv(mids: np.ndarray) -> dict:
    NM = len(mids)
    usecols = [
        "match_id",
        "game_creation",
        "game_duration",
        "player_id",
        "team_id",
        "lane",
        "win",
        "champion_id",
        "gold_earned",
        "kills",
    ] + [f"{s}_{m}" for m in MINUTES for s in TIMELINE_STATS]

    log("Leyendo full_dataset.csv (columnas de timeline y fin de partida)...")
    df = pd.read_csv(REPO / "data" / "splits" / "train" / "full_dataset.csv", usecols=usecols)
    df = df[df["lane"].isin(LANES)]
    df = df[df.groupby(["match_id", "team_id"])["lane"].transform("nunique") == 5]
    df = df[df.groupby("match_id")["lane"].transform("size") == 10]
    df = df[df.groupby("match_id")["win"].transform("sum") == 5].copy()
    df = df.sort_values(ORDEN).reset_index(drop=True)
    assert len(df) == 10 * NM and (df["match_id"].to_numpy()[::10] == mids).all(), "Desalineación con variables.npz"

    dur_sec = df["game_duration"].to_numpy(np.float32)[::10]
    player_ids = df["player_id"].to_numpy().astype(str).reshape(NM, 2, 5)
    g_end = (df["gold_earned"].to_numpy(np.float32).reshape(NM, 2, 5) / 1000.0)
    k_end = df["kills"].to_numpy(np.float32).reshape(NM, 2, 5)
    end_gold_diff = (g_end[:, 0] - g_end[:, 1]).sum(axis=1).astype(np.float32)
    end_kills_diff = (k_end[:, 0] - k_end[:, 1]).sum(axis=1).astype(np.float32)

    tl_raw = np.empty((NM, 2, 5, len(MINUTES), len(TIMELINE_STATS)), dtype=np.float32)
    scale_div = {"gold": 1000.0, "xp": 1000.0, "damage": 1000.0}
    for mi, m in enumerate(MINUTES):
        for si, s in enumerate(TIMELINE_STATS):
            arr = df[f"{s}_{m}"].to_numpy(np.float32).reshape(NM, 2, 5)
            if s in scale_div:
                arr = arr / scale_div[s]
            tl_raw[:, :, :, mi, si] = mask_finished_matches(arr, dur_sec, m)

    del df, g_end, k_end
    gc.collect()
    return {
        "dur_sec": dur_sec,
        "player_ids": player_ids,
        "end_gold_diff": end_gold_diff,
        "end_kills_diff": end_kills_diff,
        "tl_raw": tl_raw,
    }


def _construir_pre_extras(D: dict, comp: dict, tempo: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construye señales pre-partida causales livianas (sin HGB lento) para Stage-1 y Expected State."""
    from historial.escalon9.brainstorm_g_01_foto1_wl import construir_extras
    from historial.escalon9.brainstorm_g_07_matchups_parche_rematch import construir_mpr
    from historial.escalon9.brainstorm_g_11_elo_clutch_tempo import construir_cache_g11
    from historial.escalon9.brainstorm_g_23_duelo_res_y_per_jug import construir_cache_g23

    NM = D["NM"]
    # 1. FOTO1_DIN (~3s)
    f1_din = construir_extras(D)["FOTO1_DIN"].astype(np.float32)

    # 2. MATCHUPS (~15s o desde caché)
    mpr_path = BR_DIR / "gemini_mpr.npz"
    if mpr_path.exists():
        matchups = np.load(mpr_path)["MATCHUPS"].astype(np.float32)
    else:
        mpr = construir_mpr(D)
        matchups = mpr["MATCHUPS"].astype(np.float32)

    # 3. KALMAN / TEMPO / ECON (~20s o desde caché)
    g11_path = BR_DIR / "gemini_g11.npz"
    if g11_path.exists():
        g11 = dict(np.load(g11_path))
    else:
        g11 = construir_cache_g11(D)
    k_feats = np.c_[g11["KALMAN_OUT"], g11["KALMAN_MRG"], g11["TEMPO_OBJ"], g11["ECON_MID"]].astype(np.float32)

    # 4. MATCHUP_EXP_MARGIN (~20s o desde caché)
    g23_path = BR_DIR / "gemini_g23.npz"
    if g23_path.exists():
        mem = np.load(g23_path)["MATCHUP_EXP_MARGIN"].astype(np.float32)
    else:
        mem = construir_cache_g23(D)["MATCHUP_EXP_MARGIN"].astype(np.float32)

    # 5. Nivel histórico (Foto 1 sobre todo el historial, ~8s o desde caché)
    mmr_path = BR_DIR / "mmr_historico.npz"
    if mmr_path.exists():
        A_mmr = np.load(mmr_path)["A"].astype(np.float32)
    else:
        import scripts.analysis.ganador_mapa_lp as gm
        from src.features.lobby import prior_lobby_level

        foto1, _ = gm._cargar_foto1(gm.RUTA_DB)
        pl = D["B"]["player"].reshape(-1)
        df_mmr = pd.DataFrame({
            "match_id": np.repeat(D["Z"]["mids"], 10),
            "game_creation": np.repeat(D["t"], 10),
            "team_id": np.tile(np.repeat([100, 200], 5), NM),
            "lane": np.tile(np.arange(5), 2 * NM),
            "player_id": pl,
        })
        df_mmr["x"] = df_mmr["player_id"].map(foto1["z"].to_dict())
        mu0 = float(df_mmr.loc[np.repeat(~D["val0"], 10), "x"].mean())
        N = prior_lobby_level(df_mmr, "x", k=5, mu0=mu0, window=10, min_otros=3)
        propio = df_mmr["x"].fillna(mu0).to_numpy()
        A_mmr = np.c_[
            N["nivel_acum"] - mu0,
            N["nivel_v10"] - mu0,
            np.log1p(N["n_nivel"]),
            N["nivel_v10"].to_numpy() - propio,
            df_mmr["x"].notna().astype(float),
        ].astype(np.float32).reshape(NM, 2, 5, -1)
        del df_mmr
    niv_eq = (A_mmr[:, 0] - A_mmr[:, 1])[..., :3].mean(axis=1).astype(np.float32)

    cb_diff = (tempo["cbias"][:, 0] - tempo["cbias"][:, 1]).astype(np.float32)
    pre_no_post = np.c_[matchups, k_feats, mem, niv_eq, comp["comp_anti"], cb_diff].astype(np.float32)
    np.nan_to_num(pre_no_post, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return pre_no_post, f1_din, mem


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from scripts.analysis.brainstorm_base import cargar

    log("Cargando base + variables...")
    D = cargar()
    NM = D["NM"]
    Z = D["Z"]
    mids = Z["mids"].astype(str)
    t = D["t"].astype(np.int64)
    y = D["y"].astype(np.int8)
    val0 = D["val0"].astype(bool)
    val = D["val"].astype(bool)
    diam_v = D["diam_v"].astype(bool)
    master = D["master"].astype(bool)
    post = D["post"].astype(bool)
    champ = Z["champ"].astype(np.int32).reshape(NM, 2, 5)

    # Extraer bloque compacto de jugador desde Xj y liberar Xj enseguida (ahorra ~612 MB RAM)
    cols_all = list(D["cols"])
    p_idx = [cols_all.index(c) for c in PLAYER_COLS]
    p_mat = D["Xj"][:, :, :, p_idx].astype(np.float32)  # (NM, 2, 5, 17)
    D.pop("Xj", None)
    gc.collect()

    # 1. Leer timeline de full_dataset.csv
    tl_info = _extraer_timeline_y_csv(mids)
    tl_raw = tl_info["tl_raw"]
    dur_sec = tl_info["dur_sec"]
    dur_min = dur_sec / 60.0
    player_ids = tl_info["player_ids"]

    # 2. Rasgos causales de tempo (campeón y jugador)
    log("Calculando rasgos causales de tempo (cbias, cslope, pbias, pvol)...")
    stat_i = {s: i for i, s in enumerate(TIMELINE_STATS)}
    d_g10_l = (tl_raw[:, 0, :, 0, stat_i["gold"]] - tl_raw[:, 1, :, 0, stat_i["gold"]])
    d_g15_l = (tl_raw[:, 0, :, 1, stat_i["gold"]] - tl_raw[:, 1, :, 1, stat_i["gold"]])
    d_g20_l = (tl_raw[:, 0, :, 2, stat_i["gold"]] - tl_raw[:, 1, :, 2, stat_i["gold"]])
    tempo = causal_champ_player_tempo(t, mids, champ, player_ids, dur_min, y, d_g10_l, d_g15_l, d_g20_l)

    # 3. Composición y atributos estáticos
    log("Construyendo atributos de composición y crecimiento...")
    atrib_df, ddragon_df = _cargar_atributos_campeones()
    comp = build_composition_features(champ, atrib_df, ddragon_df)

    # 4. Señales pre-partida causales para Stage-1 L_oof
    log("Armando señales pre-partida causales para Stage-1...")
    pre_no_post, f1_din, mem = _construir_pre_extras(D, comp, tempo)

    # 5. Definir tramos cronológicos (sub-chunks en bloque 0 + A2, A3, A4 + V)
    bloque = D["bloque"].copy()  # -1=V, 0=pre-A2, 1=A2, 2=A3, 3=A4 (+diam_v)
    chunk = np.full(NM, -1, dtype=np.int32)
    b0_mask = bloque == 0
    chunk[b0_mask] = bloques_por_fecha(t[b0_mask], 4)  # 0, 1, 2, 3
    chunk[bloque == 1] = 4  # A2
    chunk[bloque == 2] = 5  # A3
    chunk[bloque == 3] = 6  # A4 (< 16/09)
    chunk[val] = 7          # V Master+
    chunk[diam_v] = 8       # Diamond de ventana V (separado de A4)

    for ch_id in range(8):
        m_ch = chunk == ch_id
        log(f"  chunk {ch_id}: total={m_ch.sum():,}, master={(m_ch & master).sum():,}")

    # 6. Forward-chaining Stage-1 L_oof (Master+ < chunk)
    log("Calculando Stage-1 L_oof causal sobre chunks 1..7...")
    L_oof = np.full(NM, np.nan, dtype=np.float32)
    Xd = D["Xd"]
    DRAFT = D["DRAFT"]
    mapa_post = np.c_[D["mapa"], f1_din].astype(np.float32)

    folds = [(b, np.flatnonzero(chunk == b)) for b in range(1, 7)]
    folds.append((7, np.flatnonzero(val | diam_v)))

    for b, te in folds:
        t_lim = int(t[te].min())
        if b <= 6:
            tr = np.flatnonzero((chunk >= 0) & (chunk < b) & (t < t_lim) & master)
        else:
            tr = np.flatnonzero((~val0) & master)
        mu_x, sd_x = Xd[tr].mean(0), Xd[tr].std(0) + 1e-6
        mu_e, sd_e = pre_no_post[tr].mean(0), pre_no_post[tr].std(0) + 1e-6

        Ptr = [
            DRAFT[tr],
            sparse.csr_matrix(((Xd[tr] - mu_x) / sd_x).astype(np.float32)),
            sparse.csr_matrix(((pre_no_post[tr] - mu_e) / sd_e).astype(np.float32)),
        ]
        Pte = [
            DRAFT[te],
            sparse.csr_matrix(((Xd[te] - mu_x) / sd_x).astype(np.float32)),
            sparse.csr_matrix(((pre_no_post[te] - mu_e) / sd_e).astype(np.float32)),
        ]
        tr_post = tr[post[tr]]
        if len(tr_post) >= 500:
            mp, sp = mapa_post[tr_post].mean(0), mapa_post[tr_post].std(0) + 1e-6
            Ptr.append(sparse.csr_matrix(np.where(post[tr][:, None], (mapa_post[tr] - mp) / sp, 0.0).astype(np.float32)))
            Pte.append(sparse.csr_matrix(np.where(post[te][:, None], (mapa_post[te] - mp) / sp, 0.0).astype(np.float32)))

        lr = LogisticRegression(C=0.05, max_iter=3000).fit(sparse.hstack(Ptr).tocsr(), y[tr])
        p_te = np.clip(lr.predict_proba(sparse.hstack(Pte).tocsr())[:, 1], 1e-5, 1.0 - 1e-5)
        L_oof[te] = np.log(p_te / (1.0 - p_te)).astype(np.float32)

    v_mask = val & np.isfinite(L_oof)
    a24_mask = np.isin(chunk, [4, 5, 6]) & master & np.isfinite(L_oof)
    log(f"Stage-1 L_oof AUC -> A2-A4 Master+: {roc_auc_score(y[a24_mask], L_oof[a24_mask]):.4f} | "
        f"V Master+: {roc_auc_score(y[v_mask], L_oof[v_mask]):.4f}")

    # Liberar matrices pre-partida pesadas antes de calcular Ridge/LGBM por carril
    del D, Xd, mapa_post, f1_din
    gc.collect()

    # 7. Oro residual causal por chunk (plates/towers/bounties proxy)
    log("Calculando oro residual causal por carril (gold ~ cs + kills + assists)...")
    gold_resid = np.full((NM, 2, 5, len(MINUTES)), np.nan, dtype=np.float32)
    for mi, m in enumerate(MINUTES):
        g_m = tl_raw[:, :, :, mi, stat_i["gold"]]
        c_m = tl_raw[:, :, :, mi, stat_i["cs"]]
        k_m = tl_raw[:, :, :, mi, stat_i["kills"]]
        a_m = tl_raw[:, :, :, mi, stat_i["assists"]]
        for b, te in folds:
            t_lim = int(t[te].min())
            fit_m = (chunk >= 0) & (chunk < b) & (t < t_lim) & master if b <= 6 else (~val0) & master
            coeffs = fit_residual_gold_coeffs(g_m, c_m, k_m, a_m, fit_m)
            res_te = apply_residual_gold(g_m[te], c_m[te], k_m[te], a_m[te], coeffs)
            gold_resid[te, :, :, mi] = res_te

    # 8. Expectativas causales por carril (Ridge DRAFT + comp -> residual LightGBM)
    log("Calculando expectativas causales por carril (@10, @15, @20) con Ridge + LightGBM...")
    X_ridge_base = sparse.hstack([DRAFT, sparse.csr_matrix(comp["comp_anti"])]).tocsr()
    del DRAFT
    gc.collect()

    exp_lane: dict[str, np.ndarray] = {}
    slot_static = comp["slot_static"]  # (NM, 2, 5, 18)

    # Primero Ridge lineal sobre DRAFT + comp para gold, xp, cs, damage en 10, 15, 20
    for m_idx, m in enumerate(MINUTES):
        for sname in ("gold", "xp", "cs", "damage"):
            key = f"{sname}_{m}"
            d_obs = (tl_raw[:, 0, :, m_idx, stat_i[sname]] - tl_raw[:, 1, :, m_idx, stat_i[sname]]).astype(np.float32)
            pred = np.zeros((NM, 5), dtype=np.float32)
            for l in range(5):
                ok_l = np.isfinite(d_obs[:, l])
                for b, te in folds:
                    t_lim = int(t[te].min())
                    tr_m = ((chunk >= 0) & (chunk < b) & (t < t_lim) & master) if b <= 6 else ((~val0) & master)
                    tr = np.flatnonzero(tr_m & ok_l)
                    if len(tr) < 200:
                        continue
                    rg = Ridge(alpha=300.0, fit_intercept=False, solver="lsqr")
                    rg.fit(X_ridge_base[tr], d_obs[tr, l])
                    pred[te, l] = rg.predict(X_ridge_base[te]).astype(np.float32)
            exp_lane[key] = pred
    del X_ridge_base
    gc.collect()

    # Refinamiento no lineal LightGBM sobre el residuo de Ridge para gold/xp @10/@15/@20
    lgb_params = dict(
        n_estimators=90,
        num_leaves=31,
        learning_rate=0.06,
        max_bin=63,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        min_child_samples=150,
        n_jobs=6,
        verbose=-1,
        random_state=42,
    )
    mem_by_lane = {0: mem[:, 4:5], 1: mem[:, 5:6], 2: mem[:, 6:7], 3: mem[:, 7:8], 4: mem[:, 8:9]}
    for m_idx, m in enumerate(MINUTES):
        for sname in ("gold", "xp"):
            key = f"{sname}_{m}"
            d_obs = (tl_raw[:, 0, :, m_idx, stat_i[sname]] - tl_raw[:, 1, :, m_idx, stat_i[sname]]).astype(np.float32)
            base_pred = exp_lane[key]
            resid_tot = (d_obs - base_pred).astype(np.float32)
            for l in range(5):
                Xa = np.c_[
                    p_mat[:, 0, l],
                    p_mat[:, 1, l],
                    slot_static[:, 0, l],
                    slot_static[:, 1, l],
                    tempo["cbias"][:, 0, l : l + 1],
                    tempo["cbias"][:, 1, l : l + 1],
                    mem_by_lane[l],
                    comp["comp_anti"],
                ].astype(np.float32)
                Xr = np.c_[
                    p_mat[:, 1, l],
                    p_mat[:, 0, l],
                    slot_static[:, 1, l],
                    slot_static[:, 0, l],
                    tempo["cbias"][:, 1, l : l + 1],
                    tempo["cbias"][:, 0, l : l + 1],
                    -mem_by_lane[l],
                    -comp["comp_anti"],
                ].astype(np.float32)
                ok_l = np.isfinite(resid_tot[:, l])
                add_pred = np.zeros(NM, dtype=np.float32)
                for b, te in folds:
                    t_lim = int(t[te].min())
                    tr_m = ((chunk >= 0) & (chunk < b) & (t < t_lim) & master) if b <= 6 else ((~val0) & master)
                    tr = np.flatnonzero(tr_m & ok_l)
                    if len(tr) < 500:
                        continue
                    yt = resid_tot[tr, l]
                    reg = lgb.LGBMRegressor(**lgb_params)
                    reg.fit(np.r_[Xa[tr], Xr[tr]], np.r_[yt, -yt])
                    add_pred[te] = (0.5 * (reg.predict(Xa[te]) - reg.predict(Xr[te]))).astype(np.float32)
                exp_lane[key][:, l] = base_pred[:, l] + add_pred
            vi_m = val & np.isfinite(d_obs).all(axis=1)
            corrs = [np.corrcoef(d_obs[vi_m, l], exp_lane[key][vi_m, l])[0, 1] for l in range(5)]
            log(f"  exp_lane[{key}] corr en V por carril: " + " ".join(f"{c:.3f}" for c in corrs))

    # 9. Expectativas causales de transición (@10 -> @15 y @10+@15 -> @20) para oro
    log("Calculando expectativas causales de transición (@10→@15 y @15→@20)...")
    exp_trans: dict[str, np.ndarray] = {}
    for m_idx, m in ((1, 15), (2, 20)):
        d_obs_cur = (tl_raw[:, 0, :, m_idx, stat_i["gold"]] - tl_raw[:, 1, :, m_idx, stat_i["gold"]]).astype(np.float32)
        base_g = exp_lane[f"gold_{m}"]
        prev_idx = m_idx - 1
        dg_prev_all = (tl_raw[:, 0, :, prev_idx, stat_i["gold"]] - tl_raw[:, 1, :, prev_idx, stat_i["gold"]]).astype(np.float32)
        dxp_prev_all = (tl_raw[:, 0, :, prev_idx, stat_i["xp"]] - tl_raw[:, 1, :, prev_idx, stat_i["xp"]]).astype(np.float32)
        dk_prev_all = (tl_raw[:, 0, :, prev_idx, stat_i["kills"]] - tl_raw[:, 1, :, prev_idx, stat_i["kills"]]).astype(np.float32)
        pred_tr = np.zeros((NM, 5), dtype=np.float32)
        for l in range(5):
            feats_base = [
                dg_prev_all,
                dxp_prev_all[:, l : l + 1],
                dk_prev_all[:, l : l + 1],
                base_g[:, l : l + 1],
                (tempo["cbias"][:, 0, l : l + 1] - tempo["cbias"][:, 1, l : l + 1]),
                comp["comp_anti"],
            ]
            if m == 20:
                dg10_all = (tl_raw[:, 0, :, 0, stat_i["gold"]] - tl_raw[:, 1, :, 0, stat_i["gold"]]).astype(np.float32)
                feats_base.append(dg_prev_all - dg10_all)
            Xa = np.column_stack(feats_base).astype(np.float32)
            ok = np.isfinite(d_obs_cur[:, l]) & np.isfinite(Xa).all(axis=1)
            for b, te in folds:
                t_lim = int(t[te].min())
                tr_m = ((chunk >= 0) & (chunk < b) & (t < t_lim) & master) if b <= 6 else ((~val0) & master)
                tr = np.flatnonzero(tr_m & ok)
                if len(tr) < 500:
                    continue
                te_ok = te[np.isfinite(Xa[te]).all(axis=1)]
                if len(te_ok) == 0:
                    continue
                yt = d_obs_cur[tr, l]
                reg = lgb.LGBMRegressor(**lgb_params)
                reg.fit(np.r_[Xa[tr], -Xa[tr]], np.r_[yt, -yt])
                pred_tr[te_ok, l] = (0.5 * (reg.predict(Xa[te_ok]) - reg.predict(-Xa[te_ok]))).astype(np.float32)
        exp_trans[f"trans_gold_{m}"] = pred_tr

    # Guardar caché unificada
    save_dict = dict(
        mids=mids,
        t=t,
        y=y,
        val0=val0,
        val=val,
        diam_v=diam_v,
        master=master,
        chunk=chunk,
        bloque=bloque,
        dur_sec=dur_sec,
        L_oof=L_oof,
        tl_raw=tl_raw,
        gold_resid=gold_resid,
        end_gold_diff=tl_info["end_gold_diff"],
        end_kills_diff=tl_info["end_kills_diff"],
        p_mat=p_mat,
        p_cols=np.array(PLAYER_COLS),
        comp_anti=comp["comp_anti"],
        comp_names_anti=np.array(comp["names_anti"]),
        comp_sym=comp["comp_sym"],
        comp_names_sym=np.array(comp["names_sym"]),
        vuln_blue=comp["vuln_blue"],
        vuln_red=comp["vuln_red"],
        vuln_names=np.array(comp["vuln_names"]),
        slot_static=comp["slot_static"],
        cbias=tempo["cbias"],
        cslope=tempo["cslope"],
        pbias=tempo["pbias"],
        pvol=tempo["pvol"],
        **{f"exp_{k}": v for k, v in exp_lane.items()},
        **{f"tr_{k}": v for k, v in exp_trans.items()},
    )
    np.savez_compressed(CACHE_PATH, **save_dict)
    log(f"Guardado {CACHE_PATH} ({CACHE_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
