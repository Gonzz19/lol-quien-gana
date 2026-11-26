"""Constructores de las variables previas del modelo final (escalón 9).

**Código copiado tal cual** de los scripts donde se probó cada variable (hoy en `historial/` o en
`scripts/analysis/`), con solo tres cambios: no leen ni escriben cachés (de eso se ocupa
`src/ganador/cache.py`), reciben como argumento lo que antes leían de disco, y devuelven solo las
columnas que usa el modelo final. La equivalencia con el código original está verificada
(`PROJECT_SUMMARY.md` §9.S.73). Por eso conservan los nombres internos de entonces (MATCHUPS, ROL_WR...).

No usar directamente: la entrada es `src.ganador.variables.calcular_todas`.
"""
# Generado por la reorganización del 2026-11-22. Origen de cada función:
#   variables_partida      <- scripts/analysis/embeddings_partidas.py::construir
#   margen_campeon_linea   <- brainstorm_05_jugador.py::construir_cache
#   nivel_historico        <- brainstorm_07_mmr_historico.py::construir_cache
#   wl_foto1               <- brainstorm_g_01_foto1_wl.py::construir_extras (solo FOTO1_DIN)
#   contracruces           <- brainstorm_g_07_matchups_parche_rematch.py::construir_mpr (solo MATCHUPS)
#   winrate_linea          <- brainstorm_g_10_nuevas_senales.py::construir_cache (solo ROL_WR)
#   kalman_elo             <- brainstorm_g_11_elo_clutch_tempo.py::construir_cache_g11 (sin CLUTCH_15)
#   composicion_campeones  <- brainstorm_g_06_comp_poisson.py::construir_composicion
#   oro_linea_hgb          <- brainstorm_g_09_hgb_lineas_causal.py::construir_hgb_causal
#   margen_esperado_par    <- brainstorm_g_23_duelo_res_y_per_jug.py::construir_cache_g23 (solo MATCHUP_EXP_MARGIN)
# ruff: noqa
import json
import math
import sqlite3
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.features.change import play_rate, pool_change
from src.features.elo_reverso import cargar_foto
from src.features.fingerprint import STYLE_FEATURES, add_rates
from src.features.lobby import prior_lobby_level
from src.features.margin import ORDEN, _previo, lane_diffs, lane_scale, prior_margin, prior_margin_by, prior_winrate
from src.features.padron import verificar_foto1
from src.features.rating import lp_absoluto
from src.ganador import rutas

T0 = time.time()
LANES = ["BOTTOM", "JUNGLE", "MIDDLE", "TOP", "UTILITY"]      # orden lexicográfico = orden del sort
STATS = ["kills_pm", "assists_pm", "damage_pm", "damage_share_pct", "kill_participation_pct",
         "deaths_pm", "damage_taken_pm", "gold_pm", "cs_pm", "gold_share_pct", "vision_pm",
         "d_gold_10", "d_gold_15", "d_xp_15", "d_cs_15"]
TAGS = ["Fighter", "Tank", "Mage", "Assassin", "Marksman", "Support"]
CSTATS = ["attackrange", "hp", "armor", "spellblock", "movespeed", "attackdamage", "attackspeed"]
USE = ["match_id", "game_creation", "game_duration", "game_version", "player_id", "team_id", "win",
       "champion_id", "lane", "kills", "deaths", "assists", "kill_participation_pct",
       "damage_share_pct", "gold_share_pct", "total_damage_dealt_to_champions", "total_damage_taken",
       "gold_earned", "total_minions_killed", "neutral_minions_killed", "vision_score",
       "first_blood_kill", "first_tower_kill", "player_tier", "early_aggression_10"] + \
      [f"{s}_{m}" for m in (10, 15, 20) for s in ("gold", "xp", "cs", "damage", "kills", "deaths", "assists")]

def log(*a):
    s = " ".join(str(x) for x in a)
    print(f"[{time.time() - T0:6.0f}s] {s}", flush=True)


def cargar_foto1():
    """Foto 1 del ranking (04/09) por jugador y su fecha, congelada (`verificar_foto1`)."""
    uri = f"file:{rutas.DB.resolve()}?mode=ro"
    foto1 = cargar_foto(
        uri,
        "SELECT m.player_id, p.tier, p.rank, p.league_points, p.wins, p.losses "
        "FROM players p JOIN player_id_map m ON m.puuid = p.puuid",
    )
    with sqlite3.connect(uri, uri=True) as con:
        t1 = verificar_foto1(con)
    return foto1, t1


def variables_partida() -> dict:
    with sqlite3.connect(f"file:{rutas.DB}?mode=ro", uri=True) as con:
        t_foto = verificar_foto1(con)       # foto 1 congelada (§9.S.67)
    df = rutas.leer_partidas(usecols=USE)
    df = df[df.lane.isin(LANES)]
    df = df[df.groupby(["match_id", "team_id"]).lane.transform("nunique") == 5]
    df = df[df.groupby("match_id").lane.transform("size") == 10]
    # partidas sin ganador (win = 0 en las 10 filas, TAREAS §3): fuera del análisis
    df = df[df.groupby("match_id").win.transform("sum") == 5].copy()
    df = add_rates(df).sort_values(["game_creation", "match_id", "team_id", "lane"]).reset_index(drop=True)
    df = lane_diffs(df, ["gold_10", "gold_15", "xp_15", "cs_15"])
    log(f"filas {len(df):,}, partidas {len(df) // 10:,} ({time.time() - T0:.0f}s)")

    mids = df.match_id.to_numpy()[::10]
    t = df.game_creation.to_numpy()[::10]
    val = t >= rutas.inicio_evaluacion(t, t_foto)   # 16/09 en V y el simulacro; corte del split en el test
    val_fila = np.repeat(val, 10)

    # ---- historial del jugador (solo partidas anteriores; escalas e imputación congeladas en ~val)
    escala = lane_scale(df[~val_fila], STATS)
    M = prior_margin(df, STATS, 20, 5, 20, escala=escala)
    W = prior_winrate(df, 20)
    gp = df.groupby("player_id")
    n_tot = gp.cumcount()
    gc = df.groupby(["player_id", "champion_id"])
    n_ch = gc.cumcount()
    w_ch = gc.win.cumsum() - df.win
    w_tot = gp.win.cumsum() - df.win
    n_ln = df.groupby(["player_id", "lane"]).cumcount()
    E = pd.DataFrame({
        "logn_tot": np.log1p(n_tot), "logn_champ": np.log1p(n_ch), "estreno": (n_ch == 0) * 1.0,
        "wr_champ": (w_ch + 5 * (w_tot + 5) / (n_tot + 10)) / (n_ch + 5) - (w_tot + 5) / (n_tot + 10),
        "share_champ": n_ch / (n_tot + 1), "share_lane": (n_ln + 1) / (n_tot + 2),
        "offrole": ((n_tot >= 10) & (n_ln / n_tot.clip(lower=1) < 0.2)) * 1.0,
    })
    horas = (df.game_creation - gp.game_creation.shift(1)) / 3.6e6
    log_horas = np.log1p(horas)
    E["log_horas_desde"] = log_horas.fillna(log_horas[~val_fila].median())
    P = pool_change(df, 10, 5)
    E["ritmo"] = play_rate(df).fillna(0)
    log(f"historial listo ({time.time() - T0:.0f}s)")

    # ---- campeón: atributos estáticos + perfil de estilo típico (ajustado sin V)
    dd = json.load(open(rutas.DDRAGON_JSON, encoding="utf-8"))["data"]
    A = {}
    for v in dd.values():
        A[int(v["key"])] = [1.0 * (g in v["tags"]) for g in TAGS] + \
                           [v["info"][k] for k in ("attack", "defense", "magic", "difficulty")] + \
                           [v["stats"][k] for k in CSTATS]
    Acols = [f"tag_{g}" for g in TAGS] + ["info_atk", "info_def", "info_mag", "info_dif"] + \
            [f"st_{k}" for k in CSTATS]
    CA = pd.DataFrame(df.champion_id.map(A).tolist(), columns=Acols, index=df.index)
    CA = CA.fillna(CA[~val_fila].mean())
    tr = df[~val_fila]
    zl = (tr[STYLE_FEATURES] - tr.groupby("lane")[STYLE_FEATURES].transform("mean")) / \
         tr.groupby("lane")[STYLE_FEATURES].transform("std")
    g = zl.groupby(tr.champion_id)
    perfil = (g.sum() / (g.count() + 30)).add_prefix("perf_")     # encogido hacia 0 con 30 partidas
    CP = perfil.reindex(df.champion_id).fillna(0).set_index(df.index)

    X = pd.concat([M, W, E, P, CA, CP], axis=1).astype(np.float32)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    cols = list(X.columns)
    d = len(cols)
    Xj = X.to_numpy().reshape(-1, 2, 5, d)          # partida, equipo (100, 200), línea, variable

    # ---- desarrollo de la partida (por equipo, descriptivo)
    def eq(c):
        return df[c].to_numpy(np.float64).reshape(-1, 2, 5).sum(axis=2)
    dur = df.game_duration.to_numpy()[::10] / 60
    D = {"dur_min": dur}
    for m in (10, 15, 20):
        for s in ("gold", "xp", "kills"):
            v = eq(f"{s}_{m}")
            D[f"d_{s}_{m}"] = v[:, 0] - v[:, 1]                  # azul - rojo
    D["kills_tot_pm"] = eq("kills").sum(1) / dur
    D["d_kills_fin"] = eq("kills")[:, 0] - eq("kills")[:, 1]
    D["d_oro_fin"] = eq("gold_earned")[:, 0] - eq("gold_earned")[:, 1]
    D["dano_tot_pm"] = eq("total_damage_dealt_to_champions").sum(1) / dur
    D["vision_tot_pm"] = eq("vision_score").sum(1) / dur
    # 1 = azul, 0 = rojo, 0,5 = ningún jugador tiene la marca (torre de súbditos: 3,7 % de las partidas;
    # primera sangre por torre/súbditos: 0,3 %). Antes esos casos se contaban como rojo.
    for k, c in (("fb_azul", "first_blood_kill"), ("ft_azul", "first_tower_kill")):
        v = eq(c) > 0
        D[k] = np.where(v[:, 0], 1.0, np.where(v[:, 1], 0.0, 0.5))
    D["sin_20"] = np.isnan(D["d_gold_20"])
    y = df.win.to_numpy()[::10].astype(int)                       # gana azul
    orden = {"DIAMOND": 0, "MASTER": 1, "GRANDMASTER": 2, "CHALLENGER": 3}
    tnum = df.player_tier.map(orden).to_numpy(float).reshape(-1, 10)
    parche = df.game_version.astype(str).str.split(".").str[:2].str.join(".").to_numpy()[::10]
    return dict(Xj=Xj, cols=np.array(cols), mids=mids, t=t, val=val, y=y, tnum=tnum, parche=parche,
                champ=df.champion_id.to_numpy().reshape(-1, 2, 5),
                player=df.player_id.to_numpy().astype(str).reshape(-1, 2, 5),
                **{f"D_{k}": np.asarray(v) for k, v in D.items()})



def margen_campeon_linea(D: dict) -> dict:
    NM = D["NM"]
    USE = ["match_id", "game_creation", "game_duration", "player_id", "team_id", "win", "champion_id", "lane",
           "kills", "deaths", "assists", "kill_participation_pct", "damage_share_pct", "gold_share_pct",
           "total_damage_dealt_to_champions", "total_damage_taken", "gold_earned", "total_minions_killed",
           "neutral_minions_killed", "vision_score", "gold_10", "gold_15", "xp_15", "cs_15"]
    df = rutas.leer_partidas(usecols=USE)
    df = df[df.lane.isin(["BOTTOM", "JUNGLE", "MIDDLE", "TOP", "UTILITY"])]
    df = df[df.groupby(["match_id", "team_id"]).lane.transform("nunique") == 5]
    df = df[df.groupby("match_id").lane.transform("size") == 10]
    df = df[df.groupby("match_id").win.transform("sum") == 5].copy()
    df = add_rates(df).sort_values(["game_creation", "match_id", "team_id", "lane"]).reset_index(drop=True)
    df = lane_diffs(df, ["gold_10", "gold_15", "xp_15", "cs_15"])
    assert (df.match_id.to_numpy()[::10] == D["Z"]["mids"]).all()
    # Escala por línea congelada estrictamente antes de V (~val0: sin V Master+ ni Diamond-V)
    esc = lane_scale(df[np.repeat(~D["val0"], 10)], STATS)
    MC = prior_margin_by(df, STATS, ["champion_id"], "mc", k_g=10, escala=esc)
    ML = prior_margin_by(df, STATS, ["lane"], "ml", k_g=10, escala=esc)
    return dict(MC=MC.to_numpy(np.float32).reshape(NM, 2, 5, -1),
                        ML=ML.to_numpy(np.float32).reshape(NM, 2, 5, -1))


def nivel_historico(D: dict) -> dict:
    NM = D["NM"]
    foto1, t1 = cargar_foto1()
    pl = D["player"].reshape(-1)
    df = pd.DataFrame({"match_id": np.repeat(D["Z"]["mids"], 10), "game_creation": np.repeat(D["t"], 10),
                       "team_id": np.tile(np.repeat([100, 200], 5), NM), "lane": np.tile(np.arange(5), 2 * NM),
                       "player_id": pl})
    df["x"] = df.player_id.map(foto1["z"].to_dict())
    # Prior mu0 congelado estrictamente antes de V (~val0: sin V Master+ ni Diamond-V)
    mu0 = float(df.loc[np.repeat(~D["val0"], 10), "x"].mean())
    N = prior_lobby_level(df, "x", k=5, mu0=mu0, window=10, min_otros=3)
    propio = df.x.fillna(mu0).to_numpy()
    A = np.c_[N.nivel_acum - mu0, N.nivel_v10 - mu0, np.log1p(N.n_nivel), N.nivel_v10.to_numpy() - propio,
              df.x.notna().astype(float)].astype(np.float32)
    # Máscara temporal (issue #4): antes de la foto 1 su LP es posterior a la partida. Prior = 0 en
    # todas las columnas (nivel = mu0, n = 0, sin foto); para t >= t1 el LP de los rivales de partidas
    # previas, aun anteriores a t1, ya se conocía: esas filas no cambian.
    A[np.repeat(D["t"] < t1, 10)] = 0.0
    return dict(A=A.reshape(NM, 2, 5, -1))


def wl_foto1(D: dict) -> dict:
    NM = D["NM"]
    pl = D["player"]  # (NM, 2, 5) strings
    post = D["post"]  # (NM,)
    pr = np.repeat(post, 10)

    with sqlite3.connect(f"file:{rutas.DB}?mode=ro&immutable=1", uri=True) as con:
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
    return dict(FOTO1_DIN=FOTO1_DIN)



def contracruces(D: dict) -> dict:
    t0 = time.time()
    NM = D["NM"]
    t = D["t"]
    y = D["y"].astype(float)
    champ = D["Z"]["champ"].reshape(NM, 2, 5).astype(int)
    parche_str = pd_codes = None
    _, p_codes = np.unique(D["Z"]["parche"].astype(str), return_inverse=True)

    # Mapeo entero de jugadores para loops O(N) en ~3-5s
    _, pl_codes = np.unique(D["player"].reshape(-1), return_inverse=True)
    pl_codes = pl_codes.reshape(NM, 2, 5)

    ord_idx = np.lexsort((np.arange(NM), t))

    # Contadores causales
    g_cl = defaultdict(float)
    n_cl = defaultdict(int)
    ew_cl = defaultdict(lambda: 0.5)
    decay = 2.0 ** (-1.0 / 200.0)

    g_p = defaultdict(float)
    n_p = defaultdict(int)
    n_p_tot = defaultdict(int)

    # Counters (min_c, max_c, lane) y sinergias (slot_pair, min_c, max_c)
    g_cnt = defaultdict(float)
    n_cnt = defaultdict(int)
    g_syn = defaultdict(float)
    n_syn = defaultdict(int)
    SYN_PAIRS = ((0, 4), (3, 1), (2, 1))

    MATCHUPS = np.zeros((NM, 10), dtype=np.float32)  # 5 counters + sum_cnt + 3 syn + sum_syn
    PARCHE = np.zeros((NM, 18), dtype=np.float32)    # 5 x_parche + 5 x_rec + 5 pop + 3 sumas
    REMATCH = np.zeros((NM, 6), dtype=np.float32)    # tm_loss, tm_win, opp_loss, opp_win, rev_ex_tm_loss, ally_ex_opp_loss

    last_game = {}  # pl_id -> (t_ms, win_int, team_fset, enemy_fset)
    WIN_90M = 5_400_000

    for step, m in enumerate(ord_idx):
        ym = y[m]
        tm = int(t[m])
        pm = int(p_codes[m])
        n_games_tot = max(1, step)
        n_games_pat = max(1, n_p_tot[pm])

        # ---- 1. Calcular features ANTES de actualizar estadísticas con la partida m
        wr_side = np.empty((2, 5), dtype=float)
        xp_side = np.empty((2, 5), dtype=float)
        xr_side = np.empty((2, 5), dtype=float)
        pp_side = np.empty((2, 5), dtype=float)
        for s in (0, 1):
            for l in range(5):
                c = int(champ[m, s, l])
                key_cl = (c, l)
                ng, gg = n_cl[key_cl], g_cl[key_cl]
                e_cl = (gg + 15.0) / (ng + 30.0)
                wr_side[s, l] = e_cl
                ew = ew_cl[key_cl] if ng > 0 else e_cl
                key_pcl = (pm, c, l)
                np_c, gp_c = n_p[key_pcl], g_p[key_pcl]
                xp_side[s, l] = (gp_c + 50.0 * e_cl) / (np_c + 50.0) - e_cl
                xr_side[s, l] = ew - e_cl
                pop_s = ng / (2.0 * n_games_tot)
                pp_side[s, l] = (np_c + 20.0 * pop_s) / (2.0 * n_games_pat + 20.0) - pop_s

        dxp = xp_side[0] - xp_side[1]
        dxr = xr_side[0] - xr_side[1]
        dpp = pp_side[0] - pp_side[1]
        PARCHE[m, :5] = dxp
        PARCHE[m, 5:10] = dxr
        PARCHE[m, 10:15] = dpp
        PARCHE[m, 15] = dxp.sum()
        PARCHE[m, 16] = dxr.sum()
        PARCHE[m, 17] = dpp.sum()

        # Counters por línea
        for l in range(5):
            cb, cr = int(champ[m, 0, l]), int(champ[m, 1, l])
            if cb == cr:
                continue
            p_exp = float(np.clip(0.5 + (wr_side[0, l] - wr_side[1, l]), 0.1, 0.9))
            if cb < cr:
                key_c = (cb, cr, l)
                res = (g_cnt[key_c] + 25.0 * p_exp) / (n_cnt[key_c] + 25.0) - p_exp
            else:
                key_c = (cr, cb, l)
                p_min = 1.0 - p_exp
                res = -((g_cnt[key_c] + 25.0 * p_min) / (n_cnt[key_c] + 25.0) - p_min)
            MATCHUPS[m, l] = res
        MATCHUPS[m, 5] = MATCHUPS[m, :5].sum()

        # Sinergias clave por equipo (Azul - Rojo)
        for idx_p, (l1, l2) in enumerate(SYN_PAIRS):
            res_s = [0.0, 0.0]
            for s in (0, 1):
                c1, c2 = int(champ[m, s, l1]), int(champ[m, s, l2])
                if c1 <= c2:
                    key_s = (idx_p, c1, c2)
                else:
                    key_s = (idx_p, c2, c1)
                p_e = float(np.clip(0.5 + (wr_side[s, l1] - 0.5) + (wr_side[s, l2] - 0.5), 0.1, 0.9))
                res_s[s] = (g_syn[key_s] + 30.0 * p_e) / (n_syn[key_s] + 30.0) - p_e
            MATCHUPS[m, 6 + idx_p] = res_s[0] - res_s[1]
        MATCHUPS[m, 9] = MATCHUPS[m, 6:9].sum()

        # Rematch <= 90 min por equipo (Azul - Rojo)
        teams = [frozenset(int(x) for x in pl_codes[m, 0]), frozenset(int(x) for x in pl_codes[m, 1])]
        rem_eq = np.zeros((2, 6), dtype=float)
        for s in (0, 1):
            t_me, t_op = teams[s], teams[1 - s]
            for p in t_me:
                prev = last_game.get(p)
                if prev is None or (tm - prev[0]) > WIN_90M:
                    continue
                _, w_prev, p_tm_prev, p_op_prev = prev
                n_tm = len(p_tm_prev & t_me) - 1  # -1 por si p está incluido
                n_op = len(p_op_prev & t_op)
                n_cross_rev = len(p_tm_prev & t_op) - 1 if p in p_tm_prev else len(p_tm_prev & t_op)
                n_cross_ally = len(p_op_prev & t_me)
                if n_tm > 0:
                    rem_eq[s, 0 if w_prev == 0 else 1] += n_tm * 0.5
                if n_op > 0:
                    rem_eq[s, 2 if w_prev == 0 else 3] += n_op * 0.5
                if w_prev == 0 and n_cross_rev > 0:
                    rem_eq[s, 4] += n_cross_rev
                if w_prev == 0 and n_cross_ally > 0:
                    rem_eq[s, 5] += n_cross_ally
        REMATCH[m] = (rem_eq[0] - rem_eq[1]).astype(np.float32)

        # ---- 2. Actualizar estado causal DESPUÉS de la partida m
        n_p_tot[pm] += 1
        for s in (0, 1):
            ws = ym if s == 0 else (1.0 - ym)
            for l in range(5):
                c = int(champ[m, s, l])
                key_cl = (c, l)
                g_cl[key_cl] += ws
                n_cl[key_cl] += 1
                ew_cl[key_cl] = decay * ew_cl[key_cl] + (1.0 - decay) * ws
                key_pcl = (pm, c, l)
                g_p[key_pcl] += ws
                n_p[key_pcl] += 1
                p_id = int(pl_codes[m, s, l])
                last_game[p_id] = (tm, int(round(ws)), teams[s], teams[1 - s])

            for idx_p, (l1, l2) in enumerate(SYN_PAIRS):
                c1, c2 = int(champ[m, s, l1]), int(champ[m, s, l2])
                key_s = (idx_p, c1, c2) if c1 <= c2 else (idx_p, c2, c1)
                g_syn[key_s] += ws
                n_syn[key_s] += 1

        for l in range(5):
            cb, cr = int(champ[m, 0, l]), int(champ[m, 1, l])
            if cb == cr:
                continue
            if cb < cr:
                key_c = (cb, cr, l)
                g_cnt[key_c] += ym
            else:
                key_c = (cr, cb, l)
                g_cnt[key_c] += (1.0 - ym)
            n_cnt[key_c] += 1

    print(f"MPR causal construido en {time.time() - t0:.1f}s | rematch!=0 frac={(np.abs(REMATCH).sum(1) > 0).mean():.2%}", flush=True)
    return {"MATCHUPS": MATCHUPS}


def _cargar_df_ordenado(D: dict) -> pd.DataFrame:
    cols = [
        "match_id", "game_creation", "game_duration", "team_id", "lane", "player_id",
        "champion_id", "win", "kills", "deaths", "total_damage_dealt_to_champions",
        "total_minions_killed", "neutral_minions_killed", "early_aggression_10",
        "deaths_10", "gold_15",
    ]
    df = rutas.leer_partidas(usecols=cols)
    df = df[df.lane.isin(LANES)]
    df = df[df.groupby(["match_id", "team_id"]).lane.transform("nunique") == 5]
    df = df[df.groupby("match_id").lane.transform("size") == 10]
    df = df[df.groupby("match_id").win.transform("sum") == 5].copy()
    df = df.sort_values(ORDEN).reset_index(drop=True)
    assert (df["match_id"].to_numpy()[::10] == D["Z"]["mids"]).all()
    return df


def _ewma_causal_por_jugador(pid: np.ndarray, val: np.ndarray, hl_slow: float = 18.0, hl_fast: float = 6.0, mu0: float = 0.0):
    """EWMA estrictamente causal (< fila actual) por jugador, encogiendo rápido -> lento -> mu0."""
    n = len(pid)
    ew_s = np.full(n, mu0, dtype=np.float64)
    ew_f = np.full(n, mu0, dtype=np.float64)
    std_s = np.zeros(n, dtype=np.float64)
    n_obs = np.zeros(n, dtype=np.float64)

    order = np.argsort(pid, kind="mergesort")
    ps = pid[order]
    vs = val[order]
    breaks = np.flatnonzero(ps[1:] != ps[:-1]) + 1
    starts = np.r_[0, breaks]
    ends = np.r_[breaks, n]

    a_s = 1.0 - np.exp(-np.log(2.0) / hl_slow)
    a_f = 1.0 - np.exp(-np.log(2.0) / hl_fast)
    k_s, k_f = 5.0, 4.0

    for st, en in zip(starts, ends):
        num_s, den_s = k_s * mu0, k_s
        num_f, den_f = 0.0, 0.0
        mean_run, m2_run, cnt = 0.0, 0.0, 0
        for idx in range(st, en):
            cur_s = num_s / den_s
            ew_s[order[idx]] = cur_s
            ew_f[order[idx]] = (num_f + k_f * cur_s) / (den_f + k_f)
            std_s[order[idx]] = np.sqrt(m2_run / cnt) if cnt >= 2 else 0.0
            n_obs[order[idx]] = float(cnt)
            x = vs[idx]
            if np.isfinite(x):
                num_s = (1.0 - a_s) * num_s + a_s * x
                den_s = (1.0 - a_s) * den_s + a_s
                num_f = (1.0 - a_f) * num_f + a_f * x
                den_f = (1.0 - a_f) * den_f + a_f
                cnt += 1
                delta = x - mean_run
                mean_run += delta / cnt
                std_delta = x - mean_run
                m2_run += delta * std_delta
    return ew_s, ew_f, std_s, n_obs


def _streak_causal_por_jugador(pid: np.ndarray, win: np.ndarray) -> np.ndarray:
    """Racha consecutiva previa [-5, +5]: +k si ganó las últimas k, -k si perdió."""
    n = len(pid)
    streak = np.zeros(n, dtype=np.float32)
    order = np.argsort(pid, kind="mergesort")
    ps = pid[order]
    ws = win[order]
    breaks = np.flatnonzero(ps[1:] != ps[:-1]) + 1
    starts = np.r_[0, breaks]
    ends = np.r_[breaks, n]
    for st, en in zip(starts, ends):
        cur = 0
        for idx in range(st, en):
            streak[order[idx]] = float(max(-5, min(5, cur)))
            if ws[idx] > 0.5:
                cur = cur + 1 if cur > 0 else 1
            else:
                cur = cur - 1 if cur < 0 else -1
    return streak


def winrate_linea(D: dict) -> dict:
    t0 = time.time()
    NM = D["NM"]
    fit_rows = np.repeat(~D["val0"], 10)  # estrictamente < 16/09 (sin diam_v ni V)
    df = _cargar_df_ordenado(D)
    pid = df["player_id"].to_numpy()
    lane = df["lane"].to_numpy()
    champ = df["champion_id"].to_numpy()
    win = df["win"].to_numpy(np.float64)

    # 1. ROL_WR (estrictamente causal por jugador, (player, lane), (player, champ, lane))
    ones = pd.Series(1.0, index=df.index)
    w_ser = pd.Series(win, index=df.index)
    wr_all = prior_winrate(df, window=20, k=10)
    n_tot = _previo(ones, df["player_id"]).to_numpy(np.float64)
    pl_key = [df["player_id"], df["lane"]]
    n_pl = _previo(ones, pl_key).to_numpy(np.float64)
    w_pl = _previo(w_ser, pl_key).to_numpy(np.float64)
    wr_acum_v = wr_all["wr_acum"].to_numpy(np.float64)
    # Encoger wr_lane hacia el winrate propio del jugador (0.5 + wr_acum), k_l=8
    wr_lane = (w_pl + 8.0 * (0.5 + wr_acum_v)) / (n_pl + 8.0) - 0.5
    wr_lane_diff = wr_lane - wr_acum_v

    pcl_key = [df["player_id"], df["champion_id"], df["lane"]]
    n_pcl = _previo(ones, pcl_key).to_numpy(np.float64)
    w_pcl = _previo(w_ser, pcl_key).to_numpy(np.float64)
    wr_champ_lane_exc = (w_pcl + 6.0 * (0.5 + wr_lane)) / (n_pcl + 6.0) - 0.5 - wr_lane

    share_lane = n_pl / np.maximum(1.0, n_tot)
    estreno_lane = (n_pl == 0).astype(np.float64)
    autofill_drop = (1.0 - share_lane) * wr_lane_diff
    autofill_flag = ((n_tot >= 15) & (share_lane < 0.20)).astype(np.float64) * np.minimum(0.0, wr_lane_diff)

    def _agg_eq(mat: np.ndarray, with_min: bool = False, with_max: bool = False) -> np.ndarray:
        a = mat.reshape(NM, 2, 5, -1)
        parts = [(a[:, 0] - a[:, 1]).mean(axis=1)]
        if with_min:
            parts.append(a[:, 0].min(axis=1) - a[:, 1].min(axis=1))
        if with_max:
            parts.append(a[:, 0].max(axis=1) - a[:, 1].max(axis=1))
        return np.hstack(parts).astype(np.float32)

    ROL_WR = np.c_[
        _agg_eq(np.c_[wr_lane, wr_lane_diff, wr_champ_lane_exc, np.log1p(n_pl), estreno_lane, autofill_drop, autofill_flag]),
        _agg_eq(np.c_[wr_lane, wr_lane_diff, wr_champ_lane_exc], with_min=True)[:, -3:],
    ].astype(np.float32)
    print(f"  ROL_WR listo ({time.time()-t0:.1f}s)", flush=True)

    # 2. MMR_V2 (Foto 1 LP + EWMA causal + 2da pasada sin foto + desajuste post-t1)
    with sqlite3.connect(f"file:{rutas.DB}?mode=ro&immutable=1", uri=True) as con:
        verificar_foto1(con)            # foto 1 congelada (§9.S.67)
        foto = pd.read_sql_query(
            "SELECT m.player_id, p.tier, p.rank, p.league_points, p.wins + p.losses AS games "
            "FROM players p JOIN player_id_map m ON p.puuid = m.puuid",
            con,
        )
    foto["z"] = [lp_absoluto(tr, rk, lp) for tr, rk, lp in zip(foto["tier"], foto["rank"], foto["league_points"])]
    foto = foto.dropna(subset=["z"]).sort_values("games").drop_duplicates("player_id", keep="last").set_index("player_id")
    x1 = df["player_id"].map(foto["z"]).to_numpy(np.float64)
    has_f1 = np.isfinite(x1)

    # Nivel promedio de los otros 9 con x1 en esta partida (para historial previo)
    x1_m = np.where(has_f1, x1, 0.0).reshape(NM, 10)
    v1_m = has_f1.astype(np.float64).reshape(NM, 10)
    otros_v1 = v1_m.sum(axis=1, keepdims=True) - v1_m
    L1 = np.where(otros_v1 >= 3.0, (x1_m.sum(axis=1, keepdims=True) - x1_m) / np.maximum(1.0, otros_v1), np.nan).reshape(-1)
    mu0 = float(np.nanmean(L1[fit_rows]))

    ew18, ew6, std_l, n_l1 = _ewma_causal_por_jugador(pid, L1, hl_slow=18.0, hl_fast=6.0, mu0=mu0)
    # 2da pasada causal: imputar a los que no tienen foto 1 con SU propio ew18 previo (< t)
    x2 = np.where(has_f1, x1, np.where(n_l1 >= 2.0, ew18, np.nan))
    x2_m = np.where(np.isfinite(x2), x2, 0.0).reshape(NM, 10)
    v2_m = np.isfinite(x2).astype(np.float64).reshape(NM, 10)
    otros_v2 = v2_m.sum(axis=1, keepdims=True) - v2_m
    L2 = np.where(otros_v2 >= 3.0, (x2_m.sum(axis=1, keepdims=True) - x2_m) / np.maximum(1.0, otros_v2), np.nan).reshape(-1)
    ew2_18, ew2_6, _, _ = _ewma_causal_por_jugador(pid, L2, hl_slow=18.0, hl_fast=6.0, mu0=mu0)

    wr_ew18, wr_ew6, _, _ = _ewma_causal_por_jugador(pid, win, hl_slow=18.0, hl_fast=6.0, mu0=0.5)
    climb_niv = (ew6 - ew18) / 100.0
    climb_niv2 = (ew2_6 - ew2_18) / 100.0
    gap_f1_post = np.where(np.repeat(D["post"], 10) & has_f1, (ew6 - x1) / 100.0, 0.0)
    # Winrate ponderado por dificultad de lobby previo (en Z-score del nivel)
    zL = np.clip((L1 - mu0) / 300.0, -1.5, 2.5)
    perf_niv = np.where(np.isfinite(L1), (win - 0.5) * (1.0 + 0.35 * zL), np.nan)
    wr_w_ew, _, _, _ = _ewma_causal_por_jugador(pid, perf_niv, hl_slow=24.0, hl_fast=8.0, mu0=0.0)
    niv_x_wr = ((ew2_18 - mu0) / 300.0) * (wr_ew18 - 0.5)

    MMR_V2 = np.c_[
        _agg_eq(
            np.c_[
                (ew18 - mu0) / 100.0,
                (ew6 - mu0) / 100.0,
                climb_niv,
                (ew2_18 - mu0) / 100.0,
                climb_niv2,
                std_l / 100.0,
                wr_ew18 - 0.5,
                wr_w_ew,
                niv_x_wr,
            ]
        ),
        _agg_eq(np.c_[(ew2_18 - mu0) / 100.0, climb_niv2], with_min=True, with_max=True)[:, 2:],
    ].astype(np.float32)
    CLIMB_POST = _agg_eq(gap_f1_post[:, None], with_max=True).astype(np.float32)
    print(f"  MMR_V2 listo ({time.time()-t0:.1f}s)", flush=True)

    # 3. SMURF_TILT (Ideas 6 y 7 dedicadas)
    sin_foto = (~has_f1).astype(np.float64)
    dur_min = np.maximum(1.0, df["game_duration"].to_numpy(np.float64) / 60.0)
    early_ff = ((df["game_duration"] <= 960) & (df["win"] == 0)).to_numpy(np.float64)
    ff_20 = ((df["game_duration"] <= 1200) & (df["win"] == 0)).to_numpy(np.float64)
    heavy_int = ((df["deaths"].to_numpy(np.float64) / dur_min) >= 0.42).astype(np.float64)
    dpm = df["total_damage_dealt_to_champions"].to_numpy(np.float64) / dur_min
    afk_low = ((dpm < 130.0) & (lane != "UTILITY")).astype(np.float64)
    die2_10 = np.where(df["deaths_10"].notna(), (df["deaths_10"].to_numpy(np.float64) >= 2.0).astype(np.float64), np.nan)

    def _prior_rate(arr: np.ndarray, k_r: float = 20.0) -> np.ndarray:
        ok = np.isfinite(arr)
        mu_f = float(np.nanmean(arr[fit_rows]))
        s = pd.Series(np.where(ok, arr, 0.0), index=df.index)
        c = pd.Series(ok.astype(np.float64), index=df.index)
        return ((_previo(s, df["player_id"]) + k_r * mu_f) / (_previo(c, df["player_id"]) + k_r) - mu_f).to_numpy(np.float64)

    r_eff = _prior_rate(early_ff, 25.0)
    r_ff20 = _prior_rate(ff_20, 25.0)
    r_int = _prior_rate(heavy_int, 20.0)
    r_afk = _prior_rate(afk_low, 25.0)
    r_d10 = _prior_rate(die2_10, 15.0)
    streak = _streak_causal_por_jugador(pid, win).astype(np.float64)

    sm_wr = sin_foto * wr_acum_v
    sm_low = ((n_tot < 25) * np.maximum(0.0, wr_acum_v)).astype(np.float64)
    sm_no_foto_post = np.where(np.repeat(D["post"], 10), sin_foto, 0.0)

    SMURF_TILT = np.c_[
        _agg_eq(np.c_[sin_foto, sm_wr, sm_low, streak, r_eff, r_ff20, r_int, r_afk, r_d10]),
        _agg_eq(np.c_[r_eff, r_int, sm_wr], with_max=True)[:, 3:],
    ].astype(np.float32)
    SMURF_POST = _agg_eq(np.c_[sm_no_foto_post, sm_no_foto_post * wr_acum_v]).astype(np.float32)

    # 4. SNOW_JUG (Timeline/rol faltante en Xj)
    df = lane_diffs(df, ["gold_15"])
    tot_cs = df["total_minions_killed"].fillna(0.0) + df["neutral_minions_killed"].fillna(0.0)
    df["jg_share"] = df["neutral_minions_killed"].fillna(0.0) / np.maximum(1.0, tot_cs)
    df["early_agg_pm"] = df["early_aggression_10"]
    df["deaths_10_c"] = df["deaths_10"]
    stats_n = ["jg_share", "early_agg_pm", "deaths_10_c", "d_gold_15"]
    esc = lane_scale(df[fit_rows], stats_n)
    cl_key = [df["champion_id"], df["lane"]]
    res_list = []
    for c_name in stats_n:
        mu_l, sd_l = df["lane"].map(esc[c_name][0]), df["lane"].map(esc[c_name][1]).clip(lower=1e-6)
        ok_c = df[c_name].notna().astype(np.float64)
        v_c = df[c_name].fillna(0.0)
        mu_cl = (_previo(v_c, cl_key) + 15.0 * mu_l) / (_previo(ok_c, cl_key) + 15.0)
        res_c = np.where(ok_c > 0, (v_c - mu_cl) / sd_l, np.nan)
        ew_c, _, sd_c, _ = _ewma_causal_por_jugador(pid, res_c, hl_slow=20.0, hl_fast=8.0, mu0=0.0)
        res_list.append(ew_c)
        if c_name == "d_gold_15":
            res_list.append(sd_c)
    snow_mat = np.column_stack(res_list)
    SNOW_JUG = np.c_[
        _agg_eq(snow_mat),
        _agg_eq(snow_mat[:, [2, 4]], with_max=True)[:, 2:],
    ].astype(np.float32)

    # 5. CAMP_CURVA (curva temprana vs tardía causal por campeón y cruce contra agresividad)
    is_e = (df["game_duration"] < 1500).to_numpy(np.float64)
    is_l = (df["game_duration"] > 1800).to_numpy(np.float64)
    we = _previo(pd.Series(win * is_e, index=df.index), df["champion_id"]).to_numpy(np.float64)
    ne = _previo(pd.Series(is_e, index=df.index), df["champion_id"]).to_numpy(np.float64)
    wl = _previo(pd.Series(win * is_l, index=df.index), df["champion_id"]).to_numpy(np.float64)
    nl = _previo(pd.Series(is_l, index=df.index), df["champion_id"]).to_numpy(np.float64)
    wr_ce = (we + 25.0 * 0.5) / (ne + 25.0) - 0.5
    wr_cl = (wl + 25.0 * 0.5) / (nl + 25.0) - 0.5
    sc_c = wr_cl - wr_ce
    c_mat = np.c_[wr_ce, wr_cl, sc_c].reshape(NM, 2, 5, 3)
    ea_b, ea_r = snow_mat[:, 1].reshape(NM, 2, 5)[:, 0].mean(1), snow_mat[:, 1].reshape(NM, 2, 5)[:, 1].mean(1)
    CAMP_CURVA = np.c_[
        (c_mat[:, 0] - c_mat[:, 1]).mean(1),
        c_mat[:, 0, :, 2].max(1) - c_mat[:, 1, :, 2].max(1),
        c_mat[:, 0, :, 0].mean(1) * c_mat[:, 1, :, 1].mean(1) - c_mat[:, 1, :, 0].mean(1) * c_mat[:, 0, :, 1].mean(1),
        c_mat[:, 0, :, 2].mean(1) * ea_r - c_mat[:, 1, :, 2].mean(1) * ea_b,
    ].astype(np.float32)

    return dict(ROL_WR=ROL_WR)



def kalman_elo(D: dict) -> dict:

    t0 = time.time()
    NM = D["NM"]
    Z = D["Z"]
    mids = Z["mids"].astype(str)
    t_arr = D["t"].astype(np.int64)
    y_arr = D["y"].astype(np.int8)
    val0 = D["val0"].astype(bool)
    post = D["post"].astype(bool)
    champ = Z["champ"].reshape(NM, 10)
    dur_min = Z["D_dur_min"].astype(np.float32)
    dg15 = Z["D_d_gold_15"].astype(np.float32)
    dxp15 = Z["D_d_xp_15"].astype(np.float32)
    dk15 = Z["D_d_kills_15"].astype(np.float32)

    # Identificadores de jugador exactamente alineados con (NM, 2, 5)
    players = D["player"].reshape(NM, 10).astype(str)
    uniq_p, p_inv = np.unique(players.reshape(-1), return_inverse=True)
    pid_mat = p_inv.reshape(NM, 10).tolist()
    n_p = len(uniq_p)

    # Campeon x linea
    cl_pairs = champ * 5 + np.tile(np.arange(5), 2)[None, :]
    uniq_cl, cl_inv = np.unique(cl_pairs.reshape(-1), return_inverse=True)
    cl_mat = cl_inv.reshape(NM, 10).tolist()
    n_cl = len(uniq_cl)

    # Cargar columnas puntuales desde full_dataset.csv alineadas por (match_id, player_id)
    cols_csv = [
        "match_id", "player_id", "first_blood_kill", "first_tower_kill",
        "gold_share_pct", "damage_share_pct", "gold_15", "gold_20", "item6",
    ]
    df = rutas.leer_partidas(usecols=cols_csv)
    df = df.drop_duplicates(["match_id", "player_id"])
    keys = pd.DataFrame({
        "match_id": np.repeat(mids, 10),
        "player_id": players.reshape(-1),
        "_pos": np.arange(NM * 10, dtype=np.int64),
    })
    M = keys.merge(df, on=["match_id", "player_id"], how="left").sort_values("_pos", kind="mergesort")
    del df, keys

    fb_arr = M["first_blood_kill"].fillna(0).to_numpy(np.float32).reshape(NM, 10)
    ft_arr = M["first_tower_kill"].fillna(0).to_numpy(np.float32).reshape(NM, 10)
    gs_raw = M["gold_share_pct"].to_numpy(np.float32).reshape(NM, 10)
    ds_raw = M["damage_share_pct"].to_numpy(np.float32).reshape(NM, 10)
    if float(np.nanmedian(gs_raw)) > 1.5:
        gs_raw = gs_raw / 100.0
        ds_raw = ds_raw / 100.0
    gs_arr = np.nan_to_num(gs_raw, nan=0.20)
    ds_arr = np.nan_to_num(ds_raw, nan=0.20)
    g15_p = M["gold_15"].to_numpy(np.float32).reshape(NM, 10)
    g20_p = M["gold_20"].to_numpy(np.float32).reshape(NM, 10)
    it6_arr = M["item6"].fillna(3340).to_numpy(np.int32).reshape(NM, 10)
    del M

    swap_arr = (it6_arr != 3340).astype(np.float32)
    dmg_gold_eff = (ds_arr - gs_arr).astype(np.float32)
    surge_20_15 = (g20_p - g15_p).astype(np.float32)
    # Ventaja por linea 15->20: azul_l - rojo_l
    surge_valid = np.isfinite(surge_20_15[:, :5]) & np.isfinite(surge_20_15[:, 5:])
    surge_diff = np.where(surge_valid, (surge_20_15[:, :5] - surge_20_15[:, 5:]) / 1000.0, 0.0)

    # Pesos fijos para prob esperada @15 ajustados ESTRICTAMENTE en ~val0 & ~post (junio-agosto)
    m_fit15 = (~val0) & (~post) & np.isfinite(dg15) & np.isfinite(dxp15) & np.isfinite(dk15) & (dur_min >= 15.0)
    X15 = np.c_[
        np.nan_to_num(dg15, 0.0) / 1000.0,
        np.nan_to_num(dxp15, 0.0) / 1000.0,
        np.nan_to_num(dk15, 0.0),
    ]
    lr15 = LogisticRegression(C=1.0, max_iter=500).fit(X15[m_fit15], y_arr[m_fit15])
    p_exp15_all = lr15.predict_proba(X15)[:, 1].astype(np.float64)
    valid15 = np.isfinite(dg15) & (dur_min >= 15.0)
    print(f"  Modelo @15 pre-Foto1 (n={int(m_fit15.sum())}): coef={np.round(lr15.coef_[0], 3)}", flush=True)

    # Salidas
    KALMAN_OUT = np.zeros((NM, 12), dtype=np.float32)
    KALMAN_MRG = np.zeros((NM, 7), dtype=np.float32)
    CLUTCH_15 = np.zeros((NM, 8), dtype=np.float32)
    TEMPO_OBJ = np.zeros((NM, 10), dtype=np.float32)
    ECON_MID = np.zeros((NM, 4), dtype=np.float32)

    # Estados por jugador (listas Python para maxima velocidad)
    r_elo = [0.0] * n_p
    r_slow = [0.0] * n_p
    r_opp = [0.0] * n_p
    r_mrg = [0.0] * n_p
    r_mrg_slow = [0.0] * n_p
    cnt_p = [0] * n_p

    # Clutch @15 por jugador
    cl_sum = [0.0] * n_p
    cl_w = [0.0] * n_p
    ah_win = [0.0] * n_p
    ah_n = [0.0] * n_p
    bh_win = [0.0] * n_p
    bh_n = [0.0] * n_p
    wk_sum = [0.0] * n_p
    wk_w = [0.0] * n_p

    # Tempo, FB/FT, vision, mid econ por jugador
    fb_sum = [0.0] * n_p
    ft_sum = [0.0] * n_p
    sw_sum = [0.0] * n_p
    dur_sum = [0.0] * n_p
    ew_cnt = [0.0] * n_p
    wr_e_sum = [0.0] * n_p
    wr_e_n = [0.0] * n_p
    wr_l_sum = [0.0] * n_p
    wr_l_n = [0.0] * n_p
    sr_sum = [0.0] * n_p
    sr_w = [0.0] * n_p
    dg_sum = [0.0] * n_p

    # Curva early/late por (campeon, linea)
    cl_e_win = [0.0] * n_cl
    cl_e_n = [0.0] * n_cl
    cl_l_win = [0.0] * n_cl
    cl_l_n = [0.0] * n_cl

    dec20 = math.exp(-math.log(2.0) / 20.0)
    dec25 = math.exp(-math.log(2.0) / 25.0)
    dec30 = math.exp(-math.log(2.0) / 30.0)

    order = np.lexsort((mids, t_arr))
    fb_l = fb_arr.tolist()
    ft_l = ft_arr.tolist()
    sw_l = swap_arr.tolist()
    gs_l = gs_arr.tolist()
    dge_l = dmg_gold_eff.tolist()
    sd_l = surge_diff.tolist()
    sv_l = surge_valid.tolist()

    for m in order:
        ps = pid_mat[m]
        cls = cl_mat[m]
        yb = float(y_arr[m])
        dm = float(dur_min[m])
        v15 = bool(valid15[m])
        p15 = float(p_exp15_all[m])
        dg15_m = float(dg15[m]) if np.isfinite(dg15[m]) else 0.0

        # ---- 1. Leer estados CAUSALES antes de actualizar ----
        rb = [r_elo[ps[i]] for i in range(5)]
        rr = [r_elo[ps[5 + i]] for i in range(5)]
        sb = [r_slow[ps[i]] for i in range(5)]
        sr_ = [r_slow[ps[5 + i]] for i in range(5)]
        ob = [r_opp[ps[i]] for i in range(5)]
        or_ = [r_opp[ps[5 + i]] for i in range(5)]

        mb = sum(rb) * 0.2
        mr = sum(rr) * 0.2
        rb_s = sorted(rb)
        rr_s = sorted(rr)
        stdb = math.sqrt(sum((x - mb) ** 2 for x in rb) * 0.2)
        stdr = math.sqrt(sum((x - mr) ** 2 for x in rr) * 0.2)

        KALMAN_OUT[m, 0] = (mb - mr) / 100.0
        KALMAN_OUT[m, 1] = (rb_s[0] - rr_s[0]) / 100.0
        KALMAN_OUT[m, 2] = (rb_s[4] - rr_s[4]) / 100.0
        KALMAN_OUT[m, 3] = (stdb - stdr) / 100.0
        KALMAN_OUT[m, 4] = ((rb_s[3] + rb_s[4]) - (rr_s[3] + rr_s[4])) * 0.5 / 100.0
        KALMAN_OUT[m, 5] = (sum(rb[i] - sb[i] for i in range(5)) - sum(rr[i] - sr_[i] for i in range(5))) * 0.2 / 100.0
        KALMAN_OUT[m, 6] = (sum(ob) - sum(or_)) * 0.2 / 100.0
        for l in range(5):
            KALMAN_OUT[m, 7 + l] = (rb[l] - rr[l]) / 100.0

        rmb = [r_mrg[ps[i]] for i in range(5)]
        rmr = [r_mrg[ps[5 + i]] for i in range(5)]
        rmb_s = sorted(rmb)
        rmr_s = sorted(rmr)
        unc_b = sum(1.0 / math.sqrt(1.0 + cnt_p[ps[i]]) for i in range(5)) * 0.2
        unc_r = sum(1.0 / math.sqrt(1.0 + cnt_p[ps[5 + i]]) for i in range(5)) * 0.2
        KALMAN_MRG[m, 0] = (sum(rmb) - sum(rmr)) * 0.2 / 100.0
        KALMAN_MRG[m, 1] = (rmb_s[0] - rmr_s[0]) / 100.0
        KALMAN_MRG[m, 2] = (rmb_s[4] - rmr_s[4]) / 100.0
        KALMAN_MRG[m, 3] = (sum(rmb[i] - r_mrg_slow[ps[i]] for i in range(5)) - sum(rmr[i] - r_mrg_slow[ps[5 + i]] for i in range(5))) * 0.2 / 100.0
        KALMAN_MRG[m, 4] = unc_b - unc_r
        # Diferencia entre elo por resultado y elo por grafo de oponentes (over/under-performer vs lobby!)
        KALMAN_MRG[m, 5] = ((sum(rb) - sum(ob)) - (sum(rr) - sum(or_))) * 0.2 / 100.0
        # Spread del lobby conjunto
        KALMAN_MRG[m, 6] = abs(mb - mr) / 100.0

        # Clutch & ahead/behind & weakside
        clv = [cl_sum[ps[i]] / (cl_w[ps[i]] + 10.0) for i in range(10)]
        ahv = [(ah_win[ps[i]] + 8.0 * 0.78) / (ah_n[ps[i]] + 8.0) - 0.78 for i in range(10)]
        bhv = [(bh_win[ps[i]] + 8.0 * 0.22) / (bh_n[ps[i]] + 8.0) - 0.22 for i in range(10)]
        wkv = [wk_sum[ps[i]] / (wk_w[ps[i]] + 10.0) for i in range(10)]
        CLUTCH_15[m, 0] = (sum(clv[:5]) - sum(clv[5:])) * 0.2
        CLUTCH_15[m, 1] = max(clv[:5]) - max(clv[5:])
        CLUTCH_15[m, 2] = min(clv[:5]) - min(clv[5:])
        CLUTCH_15[m, 3] = (sum(ahv[:5]) - sum(ahv[5:])) * 0.2
        CLUTCH_15[m, 4] = min(ahv[:5]) - min(ahv[5:])
        CLUTCH_15[m, 5] = (sum(bhv[:5]) - sum(bhv[5:])) * 0.2
        CLUTCH_15[m, 6] = max(bhv[:5]) - max(bhv[5:])
        CLUTCH_15[m, 7] = (sum(wkv[:5]) - sum(wkv[5:])) * 0.2

        # Tempo, FB/FT, vision swap, player*champ tempo
        fbv = [(fb_sum[ps[i]] + 12.0 * 0.10) / (ew_cnt[ps[i]] + 12.0) for i in range(10)]
        ftv = [(ft_sum[ps[i]] + 12.0 * 0.10) / (ew_cnt[ps[i]] + 12.0) for i in range(10)]
        swv = [(sw_sum[ps[i]] + 10.0 * 0.79) / (ew_cnt[ps[i]] + 10.0) for i in range(10)]
        durv = [(dur_sum[ps[i]] + 10.0 * 28.0) / (ew_cnt[ps[i]] + 10.0) - 28.0 for i in range(10)]
        pbias = [
            ((wr_l_sum[ps[i]] + 6.0 * 0.5) / (wr_l_n[ps[i]] + 12.0))
            - ((wr_e_sum[ps[i]] + 6.0 * 0.5) / (wr_e_n[ps[i]] + 12.0))
            for i in range(10)
        ]
        cbias = [
            ((cl_l_win[cls[i]] + 10.0 * 0.5) / (cl_l_n[cls[i]] + 20.0))
            - ((cl_e_win[cls[i]] + 10.0 * 0.5) / (cl_e_n[cls[i]] + 20.0))
            for i in range(10)
        ]
        TEMPO_OBJ[m, 0] = sum(fbv[:5]) - sum(fbv[5:])
        TEMPO_OBJ[m, 1] = (fbv[1] + fbv[4]) - (fbv[6] + fbv[9])  # JG + SUP FB (slots: BOT,JG,MID,TOP,SUP)
        TEMPO_OBJ[m, 2] = sum(ftv[:5]) - sum(ftv[5:])
        TEMPO_OBJ[m, 3] = (ftv[0] + ftv[3]) - (ftv[5] + ftv[8])  # BOT + TOP FT
        TEMPO_OBJ[m, 4] = sum(swv[:5]) - sum(swv[5:])
        TEMPO_OBJ[m, 5] = (sum(durv[:5]) - sum(durv[5:])) * 0.2
        TEMPO_OBJ[m, 6] = (sum(pbias[:5]) - sum(pbias[5:])) * 0.2
        TEMPO_OBJ[m, 7] = (sum(cbias[:5]) - sum(cbias[5:])) * 0.2
        TEMPO_OBJ[m, 8] = max(cbias[:5]) - max(cbias[5:])
        TEMPO_OBJ[m, 9] = (sum(pbias[i] * cbias[i] for i in range(5)) - sum(pbias[5 + i] * cbias[5 + i] for i in range(5))) * 0.2

        # Mid econ 15->20 + dmg-gold eff
        srv = [sr_sum[ps[i]] / (sr_w[ps[i]] + 10.0) for i in range(10)]
        dgev = [dg_sum[ps[i]] / (ew_cnt[ps[i]] + 10.0) for i in range(10)]
        ECON_MID[m, 0] = (sum(srv[:5]) - sum(srv[5:])) * 0.2
        ECON_MID[m, 1] = max(srv[:5]) - max(srv[5:])
        ECON_MID[m, 2] = (sum(dgev[:5]) - sum(dgev[5:])) * 0.2
        ECON_MID[m, 3] = (dgev[0] + dgev[2]) - (dgev[5] + dgev[7])  # ADC(0) + MID(2) dmg-gold eff

        # ---- 2. Actualizar estados estrictamente DESPUES ----
        diff_elo = (mb - mr) / 200.0 - 0.105
        p_hat = 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, diff_elo))))
        err_out = yb - p_hat

        m_rmb = sum(rmb) * 0.2
        m_rmr = sum(rmr) * 0.2
        diff_mrg = (m_rmb - m_rmr) / 200.0 - 0.105
        p_hat_m = 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, diff_mrg))))
        mrg_target = 0.70 * yb + 0.30 * (p15 if v15 else yb)
        err_mrg = mrg_target - p_hat_m

        lobby_elo = (mb + mr) * 0.5
        lobby_mrg = (m_rmb + m_rmr) * 0.5
        fb_row = fb_l[m]
        ft_row = ft_l[m]
        sw_row = sw_l[m]
        gs_row = gs_l[m]
        dge_row = dge_l[m]
        sd_row = sd_l[m]
        sv_row = sv_l[m]

        for slot in range(10):
            p = ps[slot]
            cl = cls[slot]
            sgn = 1.0 if slot < 5 else -1.0
            win_p = yb if slot < 5 else (1.0 - yb)
            n = cnt_p[p]
            lr_w = 32.0 / math.sqrt(1.0 + n / 25.0)
            lr_pull = 0.22 if n < 10 else 0.10 / math.sqrt(1.0 + (n - 10) / 35.0)
            other9_e = (10.0 * lobby_elo - r_elo[p]) / 9.0
            other9_m = (10.0 * lobby_mrg - r_mrg[p]) / 9.0

            new_e = r_elo[p] + lr_w * sgn * err_out + lr_pull * (other9_e - r_elo[p])
            new_m = r_mrg[p] + lr_w * sgn * err_mrg + lr_pull * (other9_m - r_mrg[p])
            new_o = r_opp[p] + (0.28 if n < 10 else 0.12 / math.sqrt(1.0 + (n - 10) / 30.0)) * (other9_e - r_opp[p])

            r_elo[p] = new_e
            r_slow[p] = 0.94 * r_slow[p] + 0.06 * new_e
            r_mrg[p] = new_m
            r_mrg_slow[p] = 0.94 * r_mrg_slow[p] + 0.06 * new_m
            r_opp[p] = new_o
            cnt_p[p] = n + 1

            # Clutch
            if v15:
                c_res = (yb - p15) * sgn
                cl_sum[p] = cl_sum[p] * dec20 + c_res
                cl_w[p] = cl_w[p] * dec20 + 1.0
                if gs_row[slot] < 0.20:
                    wk_sum[p] = wk_sum[p] * dec20 + c_res
                    wk_w[p] = wk_w[p] * dec20 + 1.0
                dg_team = dg15_m * sgn
                if dg_team > 1000.0:
                    ah_win[p] = ah_win[p] * dec25 + win_p
                    ah_n[p] = ah_n[p] * dec25 + 1.0
                elif dg_team < -1000.0:
                    bh_win[p] = bh_win[p] * dec25 + win_p
                    bh_n[p] = bh_n[p] * dec25 + 1.0

            # Tempo / FB / FT / Swap / Dmg-Gold eff
            fb_sum[p] = fb_sum[p] * dec25 + fb_row[slot]
            ft_sum[p] = ft_sum[p] * dec25 + ft_row[slot]
            sw_sum[p] = sw_sum[p] * dec25 + sw_row[slot]
            dur_sum[p] = dur_sum[p] * dec25 + dm
            dg_sum[p] = dg_sum[p] * dec25 + dge_row[slot]
            ew_cnt[p] = ew_cnt[p] * dec25 + 1.0

            if dm < 26.0:
                wr_e_sum[p] = wr_e_sum[p] * dec30 + win_p
                wr_e_n[p] = wr_e_n[p] * dec30 + 1.0
                cl_e_win[cl] += win_p
                cl_e_n[cl] += 1.0
            elif dm >= 29.0:
                wr_l_sum[p] = wr_l_sum[p] * dec30 + win_p
                wr_l_n[p] = wr_l_n[p] * dec30 + 1.0
                cl_l_win[cl] += win_p
                cl_l_n[cl] += 1.0

            lane = slot % 5
            if sv_row[lane]:
                sr_sum[p] = sr_sum[p] * dec20 + sd_row[lane] * sgn
                sr_w[p] = sr_w[p] * dec20 + 1.0

    out = {
        "KALMAN_OUT": KALMAN_OUT,
        "KALMAN_MRG": KALMAN_MRG,
        "TEMPO_OBJ": TEMPO_OBJ,
        "ECON_MID": ECON_MID,
    }
    return out


def composicion_campeones(D: dict) -> tuple[np.ndarray, np.ndarray]:
    NM = D["NM"]
    C = pd.read_csv(rutas.ATRIB_CSV).set_index("champion_id").astype(float)
    C = C.fillna(C.mean())
    cols = list(C.columns)
    champ = D["Z"]["champ"].reshape(NM, 10)  # IDs reales de campeón por slot
    unicos = np.unique(champ)
    A = C.reindex(unicos).fillna(C.mean())
    cmap = {int(cid): i for i, cid in enumerate(unicos)}
    cidx = np.vectorize(cmap.__getitem__, otypes=[np.int64])(champ)
    V = A.to_numpy(np.float32)[cidx]  # (NM, 10, 14)

    def equipo(E):
        g = {k: E[:, :, i] for i, k in enumerate(cols)}
        ap_ = g["ap"].sum(1)
        return {
            "n_ap": ap_, "todo_ad": (ap_ <= 0.5) * 1.0, "todo_ap": (ap_ >= 4.0) * 1.0,
            "desbalance_dano": np.abs(ap_ - 2.5),
            "dur": g["ps_dur"].sum(1), "cc": g["ps_cc"].sum(1), "mov": g["ps_mov"].sum(1),
            "util": g["ps_util"].sum(1), "dano": g["ps_dano"].sum(1),
            "n_front": g["front"].sum(1), "sin_front": (g["front"].sum(1) == 0) * 1.0,
            "n_engage": g["engage"].sum(1), "sin_engage": (g["engage"].sum(1) == 0) * 1.0,
            "n_rango": g["rango"].sum(1), "alcance": g["alcance"].mean(1) / 100.0,
            "n_poke": g["poke"].sum(1), "n_enchanter": g["enchanter"].sum(1),
            "n_asesino": g["asesino"].sum(1), "n_tirador": g["tirador"].sum(1),
            "sin_cc": (g["ps_cc"].sum(1) <= 3.0) * 1.0, "poco_dur": (g["ps_dur"].sum(1) <= 5.0) * 1.0,
        }

    az, ro = equipo(V[:, :5]), equipo(V[:, 5:])
    X = np.c_[tuple(az[k] - ro[k] for k in az)]
    cruces = [
        az["n_poke"] * ro["sin_engage"] - ro["n_poke"] * az["sin_engage"],
        az["n_asesino"] * ro["poco_dur"] - ro["n_asesino"] * az["poco_dur"],
        az["n_ap"] * ro["dur"] - ro["n_ap"] * az["dur"],
        az["n_engage"] * ro["n_rango"] - ro["n_engage"] * az["n_rango"],
    ]
    COMP = np.c_[X, np.c_[tuple(cruces)]].astype(np.float32)
    Vz = ((V - C.mean().to_numpy(np.float32)) / (C.std().to_numpy(np.float32) + 1e-6)).reshape(NM, 2, 5, -1).astype(np.float32)
    return COMP, Vz


def cargar_target_oro15(D: dict) -> np.ndarray:
    mids = D["Z"]["mids"]
    df = rutas.leer_partidas(
        usecols=["match_id", "team_id", "lane", "gold_15", "game_duration"],
    )
    df = df[df.match_id.isin(set(mids)) & df.lane.isin(LANES)].copy()
    df["g15"] = np.where(df["game_duration"] > 900, df["gold_15"], np.nan)
    w = df.pivot_table(index="match_id", columns=["team_id", "lane"], values="g15", aggfunc="first").reindex(mids)
    Y = np.c_[[(w[(100, l)] - w[(200, l)]).to_numpy() / 1000.0 for l in LANES]].T
    return Y.astype(np.float32)


def oro_linea_hgb(D: dict, MC, ML, A, MATCHUPS, semilla: int = 42) -> dict:
    t0 = time.time()
    NM, t, ms, fit0 = D["NM"], D["t"], D["master"], ~D["val0"]
    C = {"MC": MC, "ML": ML}
    A_mmr = A.astype(np.float32)
    _, J_CAMP = composicion_campeones(D)
    MT = MATCHUPS[:, :5]

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
                random_state=semilla,
            )
            hgb.fit(np.r_[Xa_all[tr], Xr_all[tr]], np.r_[yt, -yt])
            PRED[te, l] = 0.5 * (hgb.predict(Xa_all[te]) - hgb.predict(Xr_all[te]))
        vi = D["val"] & np.isfinite(Y_g15[:, l])
        corr = np.corrcoef(Y_g15[vi, l], PRED[vi, l])[0, 1]
        auc_l = roc_auc_score(Y_g15[vi, l] > 0, PRED[vi, l])
        print(f"  HGB {LANES[l]:6s} V corr={corr:.3f} AUC_linea={auc_l:.4f} ({time.time() - tl0:.1f}s)", flush=True)

    return dict(HGB=np.c_[PRED, PRED.sum(axis=1)].astype(np.float32))


def margen_esperado_par(D: dict) -> dict:

    t0 = time.time()
    NM = D["NM"]
    mids = D["Z"]["mids"].astype(str)
    t_arr = D["t"].astype(np.int64)
    fit0 = ~D["val0"]

    df = rutas.leer_partidas(
        usecols=[
            "match_id", "game_creation", "game_duration", "team_id", "lane", "player_id",
            "champion_id", "win", "gold_10", "gold_15", "xp_15", "cs_15",
        ],
    )
    df = df[df.lane.isin(LANES)]
    df = df[df.groupby(["match_id", "team_id"]).lane.transform("nunique") == 5]
    df = df[df.groupby("match_id").lane.transform("size") == 10]
    df = df[df.groupby("match_id").win.transform("sum") == 5].copy()
    df = df.sort_values(ORDEN).reset_index(drop=True)
    assert (df["match_id"].to_numpy()[::10] == mids).all()

    dur = df["game_duration"].to_numpy(np.float64).reshape(NM, 2, 5)[:, 0, 0]
    champ = df["champion_id"].to_numpy(np.int64).reshape(NM, 2, 5)
    pid_raw = df["player_id"].to_numpy().astype(str).reshape(NM, 2, 5)
    uniq_p, p_inv = np.unique(pid_raw.reshape(-1), return_inverse=True)
    pid = p_inv.reshape(NM, 2, 5).tolist()
    n_p = len(uniq_p)

    # 4 métricas de duelo directo (azul - rojo por línea): g15, xp15, cs15, g10
    def diff_mat(col: str, min_dur: float) -> np.ndarray:
        a = df[col].to_numpy(np.float64).reshape(NM, 2, 5)
        d = a[:, 0, :] - a[:, 1, :]
        ok = (dur >= min_dur)[:, None] & np.isfinite(d)
        sc = float(np.nanstd(d[fit0[:, None] & ok])) + 1e-6
        return np.where(ok, d / sc, np.nan)

    D4 = [diff_mat("gold_15", 900.0), diff_mat("xp_15", 900.0), diff_mat("cs_15", 900.0), diff_mat("gold_10", 600.0)]
    del df

    # Acumuladores por (campeon, linea) y por par canonico (min_c, max_c, l)
    cl_mat = (champ * 5 + np.arange(5)[None, None, :]).tolist()
    cl_sum = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    cl_cnt = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    pair_sum = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    pair_cnt = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    # EWMA causal por jugador para los 4 residuales de duelo (halving 16 y 40 partidas)
    r_fast = [[0.0] * n_p for _ in range(4)]
    r_slow = [[0.0] * n_p for _ in range(4)]
    r_cnt = [[0.0] * n_p for _ in range(4)]
    a_f = 1.0 - math.exp(-math.log(2.0) / 14.0)
    a_s = 1.0 - math.exp(-math.log(2.0) / 40.0)
    k_f, k_s, k_m, k_cl = 6.0, 10.0, 12.0, 20.0

    MATCHUP_EXP_MARGIN = np.zeros((NM, 12), dtype=np.float32)
    RES_DUELO_15 = np.zeros((NM, 10), dtype=np.float32)
    P_RES = np.zeros((NM, 2, 5, 8), dtype=np.float32)

    order = np.lexsort((mids, t_arr))
    ch_l = champ.tolist()
    d_lists = [m.tolist() for m in D4]

    for m in order:
        cb_row, cr_row = ch_l[m][0], ch_l[m][1]
        pb_row, pr_row = pid[m][0], pid[m][1]
        clb_row, clr_row = cl_mat[m][0], cl_mat[m][1]

        exp_m = [[0.0] * 5 for _ in range(4)]
        for l in range(5):
            cb, cr = cb_row[l], cr_row[l]
            clb, clr = clb_row[l], clr_row[l]
            if cb == cr:
                continue
            sgn = 1.0 if cb < cr else -1.0
            pk = (cb, cr, l) if cb < cr else (cr, cb, l)
            ps, pc = pair_sum[pk], pair_cnt[pk]
            cbs, cbc = cl_sum[clb], cl_cnt[clb]
            crs, crc = cl_sum[clr], cl_cnt[clr]
            for s_i in range(4):
                mu_b = cbs[s_i] / (cbc[s_i] + k_cl)
                mu_r = crs[s_i] / (crc[s_i] + k_cl)
                prior_dif = mu_b - mu_r
                exp_m[s_i][l] = (sgn * ps[s_i] + k_m * prior_dif) / (pc[s_i] + k_m)

        for s_i in range(4):
            MATCHUP_EXP_MARGIN[m, s_i] = sum(exp_m[s_i]) * 0.2
        MATCHUP_EXP_MARGIN[m, 4:9] = exp_m[0]               # g15 esperado por linea (5)
        MATCHUP_EXP_MARGIN[m, 9:12] = [exp_m[1][2], exp_m[1][1], exp_m[1][3]]  # xp15 mid/jg/top

        # Leer estado causal del jugador (fast & slow shrunk hacia 0)
        rf_b = [[0.0] * 5 for _ in range(4)]
        rf_r = [[0.0] * 5 for _ in range(4)]
        rs_b = [[0.0] * 5 for _ in range(4)]
        rs_r = [[0.0] * 5 for _ in range(4)]
        for l in range(5):
            pb_i, pr_i = pb_row[l], pr_row[l]
            for s_i in range(4):
                w_fb = r_cnt[s_i][pb_i] / (r_cnt[s_i][pb_i] + k_f)
                w_fr = r_cnt[s_i][pr_i] / (r_cnt[s_i][pr_i] + k_f)
                w_sb = r_cnt[s_i][pb_i] / (r_cnt[s_i][pb_i] + k_s)
                w_sr = r_cnt[s_i][pr_i] / (r_cnt[s_i][pr_i] + k_s)
                vfb, vfr = r_fast[s_i][pb_i] * w_fb, r_fast[s_i][pr_i] * w_fr
                vsb, vsr = r_slow[s_i][pb_i] * w_sb, r_slow[s_i][pr_i] * w_sr
                rf_b[s_i][l], rf_r[s_i][l] = vfb, vfr
                rs_b[s_i][l], rs_r[s_i][l] = vsb, vsr
                P_RES[m, 0, l, s_i] = vfb
                P_RES[m, 1, l, s_i] = vfr
                P_RES[m, 0, l, 4 + s_i] = vsb
                P_RES[m, 1, l, 4 + s_i] = vsr

        for s_i in range(4):
            RES_DUELO_15[m, s_i] = (sum(rf_b[s_i]) - sum(rf_r[s_i])) * 0.2
            RES_DUELO_15[m, 4 + s_i] = (sum(rs_b[s_i]) - sum(rs_r[s_i])) * 0.2
        RES_DUELO_15[m, 8] = min(rf_b[0]) - min(rf_r[0])
        RES_DUELO_15[m, 9] = max(rf_b[0]) - max(rf_r[0])

        # ---- Actualizar DESPUES de leer ----
        for s_i in range(4):
            obs_row = d_lists[s_i][m]
            for l in range(5):
                obs = obs_row[l]
                if not math.isfinite(obs):
                    continue
                cb, cr = cb_row[l], cr_row[l]
                clb, clr = clb_row[l], clr_row[l]
                cl_sum[clb][s_i] += obs
                cl_cnt[clb][s_i] += 1.0
                cl_sum[clr][s_i] -= obs
                cl_cnt[clr][s_i] += 1.0
                if cb != cr:
                    sgn = 1.0 if cb < cr else -1.0
                    pk = (cb, cr, l) if cb < cr else (cr, cb, l)
                    pair_sum[pk][s_i] += sgn * obs
                    pair_cnt[pk][s_i] += 1.0

                surp = obs - exp_m[s_i][l]
                pb_i, pr_i = pb_row[l], pr_row[l]
                r_fast[s_i][pb_i] = (1.0 - a_f) * r_fast[s_i][pb_i] + a_f * surp
                r_slow[s_i][pb_i] = (1.0 - a_s) * r_slow[s_i][pb_i] + a_s * surp
                r_cnt[s_i][pb_i] += 1.0
                r_fast[s_i][pr_i] = (1.0 - a_f) * r_fast[s_i][pr_i] - a_f * surp
                r_slow[s_i][pr_i] = (1.0 - a_s) * r_slow[s_i][pr_i] - a_s * surp
                r_cnt[s_i][pr_i] += 1.0

    return {"MATCHUP_EXP_MARGIN": MATCHUP_EXP_MARGIN}

