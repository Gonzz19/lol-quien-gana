"""Planteamiento básico del mapa de LP partida a partida (LAS) y benchmark inicial.

Uso:
    python scripts/analysis/mapa_lp_basico.py
    python scripts/analysis/mapa_lp_basico.py --lp-vivo data/lp_vivo/lp_vivo.db --muestra-mapa 1000

Evalúa únicamente sobre el conjunto de desarrollo (~en_caja, 85 % de `evaluables`) y sobre
tramos contiguos de `lp_vivo.db` (`la2`), sin tocar la caja cerrada ni el split de test (< 04/09).
No modifica `data/lol_las.db`, los CSV de `data/` ni genera `data/lp_por_partida.csv`.
"""

from __future__ import annotations

import argparse
import contextlib
import sqlite3
import sys
import time
import zlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.ganador import rutas
from src.features.elo_reverso import (
    DIA_MS,
    FOTO2,
    adelante as adelante_regla,
    ajustar_regla,
    camino as camino_regla,
    cargar_train,
    cortes,
    evaluables,
)
from src.features.lp_mecanica import (
    ParametrosPaso,
    ajustar_k_diamond,
    ajustar_parametros_las,
    cargar_eventos_vivo_las,
    estimar_mapa_tramo,
    recorrer_adelante,
    suavizar_en_grilla,
    suavizar_lobby_jugador,
)

RUTA_VIVO_DEFECTO = rutas.LP_VIVO_DB


def resolver_lp_vivo(ruta_arg: Optional[Path]) -> Optional[Path]:
    if ruta_arg is not None:
        return ruta_arg if ruta_arg.exists() else None
    return RUTA_VIVO_DEFECTO if RUTA_VIVO_DEFECTO.exists() else None


def _metricas(y_real: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = np.asarray(y_pred, dtype=float) - np.asarray(y_real, dtype=float)
    ok = np.isfinite(err)
    if not ok.any():
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"), "medae": float("nan"), "p90": float("nan")}
    ae = np.abs(err[ok])
    return {
        "n": int(ok.sum()),
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt(np.mean(err[ok] ** 2))),
        "medae": float(np.median(ae)),
        "p90": float(np.quantile(ae, 0.90)),
    }


def _imprimir_tabla(titulo: str, filas: List[Tuple[str, Dict[str, float]]], extra_cols: Optional[Dict[str, str]] = None) -> None:
    print(f"\n=== {titulo} ===")
    hdr = f"{'Método':<30} {'n':>8} {'MAE':>8} {'RMSE':>8} {'MedAE':>8} {'p90':>8}"
    if extra_cols:
        hdr += "  " + "  ".join(f"{k:>10}" for k in extra_cols)
    print(hdr)
    print("-" * len(hdr))
    for nombre, m in filas:
        linea = (
            f"{nombre:<30} {m['n']:>8d} {m['mae']:>8.2f} {m['rmse']:>8.2f} "
            f"{m['medae']:>8.2f} {m['p90']:>8.2f}"
        )
        if extra_cols:
            for col, val_key in extra_cols.items():
                val = m.get(val_key, float("nan"))
                linea += f"  {val:>10.3f}" if np.isfinite(val) else f"  {'-':>10}"
        print(linea)


def cargar_roles_para_puuids(
    lol_db: Path,
    puuids: Sequence[str],
    t_max_ms: Optional[int] = None,
) -> pd.DataFrame:
    """Frecuencia histórica de roles para `puuids` en `lol_las.db` (solo lectura)."""
    if not lol_db.exists() or len(puuids) == 0:
        return pd.DataFrame(columns=["puuid", "team_position", "frac_hist", "tot_hist"])
    uri = f"file:{lol_db.resolve().as_posix()}?mode=ro"
    lista = list(dict.fromkeys(str(p) for p in puuids))
    partes = []
    with contextlib.closing(sqlite3.connect(uri, uri=True, timeout=60)) as con:
        for ini in range(0, len(lista), 400):
            lote = lista[ini : ini + 400]
            ph = ",".join("?" for _ in lote)
            if t_max_ms is not None:
                sql = (
                    "SELECT mp.puuid, mp.individual_position AS team_position, COUNT(*) AS n "
                    "FROM match_participants mp "
                    "JOIN matches m ON m.match_id = mp.match_id "
                    f"WHERE mp.puuid IN ({ph}) AND m.game_creation < ? "
                    "GROUP BY mp.puuid, mp.individual_position"
                )
                df_i = pd.read_sql_query(sql, con, params=[*lote, int(t_max_ms)])
            else:
                sql = (
                    "SELECT puuid, individual_position AS team_position, COUNT(*) AS n "
                    f"FROM match_participants WHERE puuid IN ({ph}) "
                    "GROUP BY puuid, individual_position"
                )
                df_i = pd.read_sql_query(sql, con, params=lote)
            if not df_i.empty:
                partes.append(df_i)
    if not partes:
        return pd.DataFrame(columns=["puuid", "team_position", "frac_hist", "tot_hist"])
    roles = pd.concat(partes, ignore_index=True).groupby(["puuid", "team_position"], as_index=False)["n"].sum()
    tot = roles.groupby("puuid")["n"].transform("sum")
    roles["frac_hist"] = roles["n"] / tot.clip(lower=1)
    roles["tot_hist"] = tot
    return roles[["puuid", "team_position", "frac_hist", "tot_hist"]]


def cargar_posiciones_tramo(lol_db: Path, partes: pd.DataFrame, t1: int, t2: int) -> pd.Series:
    """Devuelve `pos` alineado a `partes` desde `data/lol_las.db` (solo lectura)."""
    if not lol_db.exists() or partes.empty:
        return pd.Series("UNKNOWN", index=partes.index, dtype=object)
    uri = f"file:{lol_db.resolve().as_posix()}?mode=ro"
    sql = (
        "SELECT mp.match_id, pid.player_id, mp.individual_position AS pos "
        "FROM matches m "
        "JOIN match_participants mp ON mp.match_id = m.match_id "
        "JOIN player_id_map pid ON pid.puuid = mp.puuid "
        "WHERE m.game_creation >= ? AND m.game_creation < ?"
    )
    with contextlib.closing(sqlite3.connect(uri, uri=True, timeout=60)) as con:
        df_pos = pd.read_sql_query(sql, con, params=[int(t1), int(t2)])
    df_pos = df_pos.drop_duplicates(["match_id", "player_id"])
    merged = partes[["match_id", "player_id"]].merge(df_pos, on=["match_id", "player_id"], how="left")
    return merged["pos"].fillna("UNKNOWN").set_axis(partes.index)


def _construir_tramos_vivos(ev: pd.DataFrame, min_len: int) -> List[pd.DataFrame]:
    df = ev.sort_values(["puuid", "t_despues", "id"], kind="mergesort").reset_index(drop=True)
    n_antes = df["wins_antes"] + df["losses_antes"]
    n_desp = df["wins_despues"] + df["losses_despues"]
    same_chain = (df["puuid"] == df["puuid"].shift(1)) & (n_antes == n_desp.shift(1))
    df["tramo_id"] = (~same_chain).cumsum()
    largos = df.groupby("tramo_id").size()
    validos = set(largos[largos >= min_len].index)
    return [g.reset_index(drop=True) for tid, g in df[df["tramo_id"].isin(validos)].groupby("tramo_id", sort=False)]


def benchmark_tramos_vivos(
    ev_vivo: pd.DataFrame,
    roles_hist: pd.DataFrame,
    params: ParametrosPaso,
    a_reg: Dict[int, float],
    b_reg: Dict[int, float],
    corte_f1: Tuple[float, float],
    min_len: int = 10,
) -> None:
    """Evalúa trayectoria partida a partida en tramos contiguos reales de `lp_vivo.db`."""
    tramos = _construir_tramos_vivos(ev_vivo, min_len=min_len)
    if not tramos:
        print(f"\nNo hay tramos contiguos de >= {min_len} partidas en lp_vivo.db.")
        return

    frac_lookup = (
        roles_hist.set_index(["puuid", "team_position"])["frac_hist"].to_dict()
        if not roles_hist.empty
        else {}
    )
    reales, p_recta, p_camino, p_adelante, p_grilla, dentro_80, anchos_80 = [], [], [], [], [], [], []
    err_1paso = []
    for tr in tramos:
        pu = str(tr["puuid"].iloc[0])
        pos_arr = tr["team_position"].fillna("UNKNOWN").to_numpy(object)
        fr_arr = np.array([frac_lookup.get((pu, str(pos)), 0.2) for pos in pos_arr], dtype=float)
        win_arr = tr["es_win"].to_numpy(bool)
        ge = tr["game_end"].to_numpy(float)
        ts_ms = np.where(ge < 1e12, ge * 1000.0, ge).astype(np.int64)
        # Fallback por si faltara game_end
        bad_t = ts_ms <= 0
        if bad_t.any():
            ts_ms[bad_t] = tr["t_despues"].to_numpy(np.int64)[bad_t]
        t_ini = int(min(tr["t_antes"].iloc[0], ts_ms[0] - 60_000))
        t_fin = int(max(tr["t_despues"].iloc[-1], ts_ms[-1] + 60_000))
        z1 = float(tr["z_antes"].iloc[0])
        z2 = float(tr["z_despues"].iloc[-1])
        z_real = tr["z_antes"].to_numpy(float)

        lob_raw = tr["lobby"].to_numpy(float)
        nl_raw = tr["n_lob"].to_numpy(float)
        lob_c, nl_c = suavizar_lobby_jugador(lob_raw, nl_raw, modo="causal")
        lob_d, nl_d = suavizar_lobby_jugador(lob_raw, nl_raw, modo="descriptivo")

        # Recta temporal entre extremos observados
        tf = np.clip((ts_ms - t_ini) / max(t_fin - t_ini, 1), 0.0, 1.0)
        recta = z1 + (z2 - z1) * tf

        # Regla (a,b) de elo_reverso entre extremos
        part_df = pd.DataFrame({"game_creation": ts_ms, "win": win_arr})
        cam = camino_regla(part_df, z2, z1, a_reg, b_reg, corte_f1)[:-1]

        # Paso mecánico adelante (causal desde z1, sin usar z2)
        _, zpre_ad = recorrer_adelante(
            win_arr, ts_ms, lob_c, nl_c, pos_arr, fr_arr, z1, t_ini, t_fin, params
        )

        # Error a 1 partida vista desde cada z_antes real
        for k in range(len(tr)):
            zk_fin, _ = recorrer_adelante(
                [win_arr[k]],
                [ts_ms[k]],
                [lob_c[k]],
                [nl_c[k]],
                [pos_arr[k]],
                [fr_arr[k]],
                float(z_real[k]),
                int(tr["t_antes"].iloc[k]),
                int(tr["t_despues"].iloc[k]),
                params,
            )
            err_1paso.append(abs(zk_fin - float(tr["z_despues"].iloc[k])))

        # Suavizado grilla 1D bidireccional (con extremos y saldo W/L exactos)
        gr = suavizar_en_grilla(
            win_arr,
            ts_ms,
            lob_d,
            nl_d,
            pos_arr,
            fr_arr,
            z1=z1,
            sigma1=0.0,
            z2=z2,
            sigma2=0.0,
            t_ini=t_ini,
            t_fin=t_fin,
            params=params,
            dW_total=int(win_arr.sum()),
            dL_total=int((~win_arr).sum()),
            tau_lobby=0.0,
            step=10.0,
        )

        # Evaluamos posiciones intermedias k >= 1 (k=0 es el ancla z1)
        reales.append(z_real[1:])
        p_recta.append(recta[1:])
        p_camino.append(cam[1:])
        p_adelante.append(zpre_ad[1:])
        p_grilla.append(gr["z_media"][1:])
        dentro_80.append((z_real[1:] >= gr["z_p10"][1:]) & (z_real[1:] <= gr["z_p90"][1:]))
        anchos_80.append(gr["z_p90"][1:] - gr["z_p10"][1:])

    yr = np.concatenate(reales)
    m_rec = _metricas(yr, np.concatenate(p_recta))
    m_cam = _metricas(yr, np.concatenate(p_camino))
    m_ad = _metricas(yr, np.concatenate(p_adelante))
    m_gr = _metricas(yr, np.concatenate(p_grilla))
    cov_80 = float(np.concatenate(dentro_80).mean())
    ancho_med = float(np.median(np.concatenate(anchos_80)))
    m_gr["cov80"] = cov_80

    _imprimir_tabla(
        f"Tramos contiguos reales lp_vivo.db (>= {min_len} part.; {len(tramos)} tramos, {len(yr)} ptos intermedios)",
        [
            ("Recta (t)", m_rec),
            ("Camino regla (a,b)", m_cam),
            ("Paso adelante (causal)", m_ad),
            ("Suavizado grilla (2 anclas)", m_gr),
        ],
        extra_cols={"Cob [p10,p90]": "cov80"},
    )
    print(
        f"Paso 1-a-1 sin acumular (n={len(err_1paso)}): MAE={np.mean(err_1paso):.2f} LP | "
        f"Ancho mediano [p10,p90] grilla = {ancho_med:.1f} LP"
    )


def _grupos_por_jugador(partes: pd.DataFrame, lob: pd.Series, nlob: pd.Series) -> Dict[str, dict]:
    df = partes.sort_values(["player_id", "game_creation", "match_id"], kind="mergesort")
    if "pos" not in df.columns:
        df = df.assign(pos="UNKNOWN")
    cnt = df.groupby("player_id")["win"].transform("size")
    fr = df.groupby(["player_id", "pos"])["win"].transform("size") / cnt.clip(lower=1)
    lob_s = lob.reindex(df.index).to_numpy(float)
    nl_s = nlob.reindex(df.index).to_numpy(float)
    win_a = df["win"].to_numpy(bool)
    ts_a = df["game_creation"].to_numpy(np.int64)
    pos_a = df["pos"].to_numpy(object)
    fr_a = fr.to_numpy(float)
    grupos: Dict[str, dict] = {}
    for pid, pos_idx in df.groupby("player_id", sort=False).indices.items():
        idx = np.asarray(pos_idx, dtype=int)
        grupos[str(pid)] = {
            "win": win_a[idx],
            "ts": ts_a[idx],
            "lob": lob_s[idx],
            "nl": nl_s[idx],
            "pos": pos_a[idx],
            "fr": fr_a[idx],
            "df_partes": pd.DataFrame({"game_creation": ts_a[idx], "win": win_a[idx]}),
        }
    return grupos


def benchmark_foto1_a_foto2(
    dev: pd.DataFrame,
    foto1_dev: pd.DataFrame,
    partes: pd.DataFrame,
    t1: int,
    t2: int,
    params: ParametrosPaso,
    a_reg: Dict[int, float],
    b_reg: Dict[int, float],
    corte_f1: Tuple[float, float],
    seed: int = 42,
) -> None:
    """Evalúa cierre Foto 1 -> Foto 2 sobre desarrollo (85 %, `~en_caja`)."""
    # Lobby causal usando únicamente foto1 de jugadores no en_caja (cero fuga de foto2 ni en_caja)
    z_f1_map = partes["player_id"].map(foto1_dev["z"])
    ok = z_f1_map.notna().astype(float)
    zf = z_f1_map.fillna(0.0).astype(float)
    sum_tot = zf.groupby(partes["match_id"], sort=False).transform("sum") - zf
    cnt_tot = ok.groupby(partes["match_id"], sort=False).transform("sum") - ok
    lob_causal = sum_tot / cnt_tot.where(cnt_tot > 0)

    grupos = _grupos_por_jugador(partes, lob_causal, cnt_tot)

    # Tasa promedio de ocultas por partida observada estimada en dev sin usar valores por jugador
    obs_dev = dev["obs"].to_numpy(int)
    tasa_oc_global = float(dev.loc[obs_dev > 0, "ocultas"].sum() / max(int(dev.loc[obs_dev > 0, "obs"].sum()), 1))
    sin_oc = dev["ocultas"] == 0

    pids = list(dev.index)
    z1_arr = dev["z1"].to_numpy(float)
    z2_arr = dev["z2"].to_numpy(float)
    dw_arr = dev["dW"].to_numpy(int)
    dl_arr = dev["dL"].to_numpy(int)
    tier_arr = dev["tier1"].to_numpy(object)

    pred_reg_obs = np.full(len(pids), np.nan, dtype=float)
    pred_mec_obs = np.full(len(pids), np.nan, dtype=float)
    pred_mec_saldo = np.full(len(pids), np.nan, dtype=float)
    empty_b = np.array([], dtype=bool)
    empty_i = np.array([], dtype=np.int64)
    empty_f = np.array([], dtype=float)
    empty_o = np.array([], dtype=object)
    for i, pid in enumerate(pids):
        g = grupos.get(str(pid))
        if g is None:
            pred_reg_obs[i] = z1_arr[i]
            z_obs, _ = recorrer_adelante(
                empty_b, empty_i, empty_f, empty_f, empty_o, empty_f, z1_arr[i], t1, t2, params,
            )
            z_sal, _ = recorrer_adelante(
                empty_b, empty_i, empty_f, empty_f, empty_o, empty_f, z1_arr[i], t1, t2, params,
                dW_total=int(dw_arr[i]), dL_total=int(dl_arr[i]),
            )
        else:
            m_lob, n_eff = suavizar_lobby_jugador(g["lob"], g["nl"], modo="causal")
            pred_reg_obs[i] = adelante_regla(g["df_partes"], z1_arr[i], a_reg, b_reg, corte_f1)[-1]
            z_obs, _ = recorrer_adelante(
                g["win"], g["ts"], m_lob, n_eff, g["pos"], g["fr"], z1_arr[i], t1, t2, params,
                tasa_oc=tasa_oc_global, p_oc=0.5,
            )
            z_sal, _ = recorrer_adelante(
                g["win"], g["ts"], m_lob, n_eff, g["pos"], g["fr"], z1_arr[i], t1, t2, params,
                dW_total=int(dw_arr[i]), dL_total=int(dl_arr[i]),
            )
        pred_mec_obs[i] = z_obs
        pred_mec_saldo[i] = z_sal

    _imprimir_tabla(
        f"Cierre Foto 1 -> Foto 2 en desarrollo (n={len(dev)}, ~en_caja)",
        [
            ("No cambió (z1)", _metricas(z2_arr, z1_arr)),
            ("Regla (a,b) solo obs", _metricas(z2_arr, pred_reg_obs)),
            ("Paso mecánico solo obs", _metricas(z2_arr, pred_mec_obs)),
            ("Paso mecánico + saldo dW/dL", _metricas(z2_arr, pred_mec_saldo)),
        ],
    )

    sin_oc_mask = sin_oc.to_numpy(bool)
    if sin_oc_mask.any():
        _imprimir_tabla(
            f"Subconjunto sin partidas ocultas (ocultas == 0, n={int(sin_oc_mask.sum())})",
            [
                ("No cambió (z1)", _metricas(z2_arr[sin_oc_mask], z1_arr[sin_oc_mask])),
                ("Regla (a,b)", _metricas(z2_arr[sin_oc_mask], pred_reg_obs[sin_oc_mask])),
                ("Paso mecánico LAS", _metricas(z2_arr[sin_oc_mask], pred_mec_saldo[sin_oc_mask])),
            ],
        )

    print("\n--- Desglose por Tier1 (Paso mecánico + saldo dW/dL vs Regla solo obs) ---")
    print(f"{'Tier1':<14} {'n':>7} {'MAE z1':>9} {'MAE Regla':>10} {'MAE Mec(obs)':>13} {'MAE Mec(saldo)':>15}")
    for grupo_nombre, mask in [
        ("CHALLENGER", tier_arr == "CHALLENGER"),
        ("GRANDMASTER", tier_arr == "GRANDMASTER"),
        ("MASTER", tier_arr == "MASTER"),
        ("DIAMOND", tier_arr == "DIAMOND"),
    ]:
        if not mask.any():
            continue
        m_z1 = _metricas(z2_arr[mask], z1_arr[mask])["mae"]
        m_rg = _metricas(z2_arr[mask], pred_reg_obs[mask])["mae"]
        m_mo = _metricas(z2_arr[mask], pred_mec_obs[mask])["mae"]
        m_ms = _metricas(z2_arr[mask], pred_mec_saldo[mask])["mae"]
        print(f"{grupo_nombre:<14} {int(mask.sum()):>7d} {m_z1:>9.2f} {m_rg:>10.2f} {m_mo:>13.2f} {m_ms:>15.2f}")

    # Test de robustez ante baja cobertura del pool de jugadores en foto1 (enmascarado realista)
    rng = np.random.default_rng(seed)
    activos = np.flatnonzero(sin_oc_mask & (obs_dev >= 5))
    muestra_idx = rng.choice(activos, size=min(1500, len(activos)), replace=False) if len(activos) else []
    if len(muestra_idx):
        sel_pids = [str(pids[i]) for i in muestra_idx]
        sel_set = set(sel_pids)
        mids_sub = set(partes.loc[partes["player_id"].isin(sel_set), "match_id"])
        partes_sub = partes[partes["match_id"].isin(mids_sub)].sort_values(
            ["player_id", "game_creation", "match_id"], kind="mergesort"
        )
        f1_ids = foto1_dev.index.to_numpy(dtype=str)
        params_sin_c = ParametrosPaso(**{**params.__dict__, "c_lobby": 0.0})
        print(f"\n--- Robustez ante baja cobertura del pool en foto1 (n={len(muestra_idx)} dev activos sin ocultas) ---")
        print(
            f"{'Cob pool':>9} {'v=1, c=0.0':>12} {f'v=31, c={params.c_lobby:.1f}':>13} "
            f"{'v=31, c=0.0':>12} {'Ganancia vs v=1':>16}"
        )
        for cob in (1.0, 0.50, 0.25, 0.10, 0.05):
            if cob < 1.0:
                keep_ids = set(rng.choice(f1_ids, size=max(1, int(len(f1_ids) * cob)), replace=False))
                z_m = partes_sub["player_id"].map(foto1_dev["z"].where(foto1_dev.index.isin(keep_ids)))
            else:
                z_m = partes_sub["player_id"].map(foto1_dev["z"])
            ok_m = z_m.notna().astype(float)
            zf_m = z_m.fillna(0.0).astype(float)
            st_m = zf_m.groupby(partes_sub["match_id"], sort=False).transform("sum") - zf_m
            ct_m = ok_m.groupby(partes_sub["match_id"], sort=False).transform("sum") - ok_m
            lob_m_ser = st_m / ct_m.where(ct_m > 0)
            solo_sel = partes_sub[partes_sub["player_id"].isin(sel_set)].assign(
                _lob=lob_m_ser, _nl=ct_m
            )
            grps_sub = {str(k): g for k, g in solo_sel.groupby("player_id", sort=False)}
            e_v1, e_v31_c, e_v31_0 = [], [], []
            for idx_i in muestra_idx:
                pid = str(pids[idx_i])
                gs = grps_sub[pid]
                gb = grupos[pid]
                lob_raw = gs["_lob"].to_numpy(float)
                nl_raw = gs["_nl"].to_numpy(float)
                m1, n1 = suavizar_lobby_jugador(lob_raw, nl_raw, modo="causal", ventana=1)
                m31, n31 = suavizar_lobby_jugador(lob_raw, nl_raw, modo="causal", ventana=31)
                z_v1, _ = recorrer_adelante(
                    gb["win"], gb["ts"], m1, n1, gb["pos"], gb["fr"], z1_arr[idx_i], t1, t2,
                    params_sin_c, dW_total=int(dw_arr[idx_i]), dL_total=int(dl_arr[idx_i]),
                )
                z_c, _ = recorrer_adelante(
                    gb["win"], gb["ts"], m31, n31, gb["pos"], gb["fr"], z1_arr[idx_i], t1, t2,
                    params, dW_total=int(dw_arr[idx_i]), dL_total=int(dl_arr[idx_i]),
                )
                z_0, _ = recorrer_adelante(
                    gb["win"], gb["ts"], m31, n31, gb["pos"], gb["fr"], z1_arr[idx_i], t1, t2,
                    params_sin_c, dW_total=int(dw_arr[idx_i]), dL_total=int(dl_arr[idx_i]),
                )
                e_v1.append(abs(z_v1 - z2_arr[idx_i]))
                e_v31_c.append(abs(z_c - z2_arr[idx_i]))
                e_v31_0.append(abs(z_0 - z2_arr[idx_i]))
            m1_mae, mc_mae, m0_mae = float(np.mean(e_v1)), float(np.mean(e_v31_c)), float(np.mean(e_v31_0))
            print(f"{cob * 100:>8.0f}% {m1_mae:>12.2f} {mc_mae:>13.2f} {m0_mae:>12.2f} {m1_mae - mc_mae:>+16.2f}")


def benchmark_mapa_tramo(
    data_dir: Path,
    dev: pd.DataFrame,
    foto1_dev: pd.DataFrame,
    foto2_dev: pd.DataFrame,
    partes: pd.DataFrame,
    t1: int,
    t2: int,
    params: ParametrosPaso,
    n_muestra: int = 1000,
    seed: int = 42,
) -> None:
    """Evalúa `estimar_mapa_tramo` (2 pasadas descriptivo vs causal + anclas `principal`)."""
    activos = dev[dev["obs"] >= 3].index.to_numpy(dtype=str)
    if len(activos) == 0 or n_muestra <= 0:
        return
    rng = np.random.default_rng(seed)
    sel = rng.choice(activos, size=min(n_muestra, len(activos)), replace=False)
    sel_mitad_real = set(sel[: len(sel) // 2])
    sel_mitad_ancla = set(sel[len(sel) // 2 :])

    # Cargar predicciones del modelo principal únicamente para ~en_caja cuando están disponibles
    ruta_ep = data_dir / "analysis" / "elo_principal_pred.csv"
    anclas = None
    f1_enmascarada = foto1_dev
    if ruta_ep.exists():
        ep = pd.read_csv(ruta_ep, usecols=["player_id", "en_caja", "principal"])
        ep = ep[(~ep["en_caja"]) & ep["player_id"].isin(sel_mitad_ancla)].set_index("player_id")
        if not ep.empty:
            anclas = ep["principal"].dropna()
            f1_enmascarada = foto1_dev.drop(index=list(sel_mitad_ancla), errors="ignore")

    t0 = time.perf_counter()
    res_p1 = estimar_mapa_tramo(
        partes, f1_enmascarada, foto2_dev, t1, t2, params, anclas_z1=anclas, sigma_ancla_z1=80.0,
        modo="descriptivo", n_pasadas=1, jugadores_objetivo=sel,
    )
    dt_p1 = time.perf_counter() - t0

    t0 = time.perf_counter()
    res_p2 = estimar_mapa_tramo(
        partes, f1_enmascarada, foto2_dev, t1, t2, params, anclas_z1=anclas, sigma_ancla_z1=80.0,
        modo="descriptivo", n_pasadas=2, jugadores_objetivo=sel,
    )
    dt_p2 = time.perf_counter() - t0

    t0 = time.perf_counter()
    res_cau = estimar_mapa_tramo(
        partes, foto1_dev, None, t1, t2, params, modo="causal", jugadores_objetivo=sel,
    )
    dt_cau = time.perf_counter() - t0

    mask_sel = partes["player_id"].isin(sel)
    r1 = res_p1.loc[mask_sel]
    r2 = res_p2.loc[mask_sel]
    rc = res_cau.loc[mask_sel]

    print(f"\n=== Pipeline completo `estimar_mapa_tramo` (muestra={len(sel)} jugadores dev, {int(mask_sel.sum())} filas) ===")
    print(f"Tiempos: pasada=1 -> {dt_p1:.2f}s | pasadas=2 -> {dt_p2:.2f}s | causal -> {dt_cau:.2f}s")
    diff_p2 = np.abs(r2["z_media"] - r1["z_media"]).dropna()
    diff_cau = np.abs(r2["z_media"] - rc["z_media"]).dropna()
    print(
        f"|Pass2 - Pass1|: mean={diff_p2.mean():.2f}, p90={diff_p2.quantile(0.9):.2f} LP | "
        f"|Descriptivo - Causal|: mean={diff_cau.mean():.2f}, p90={diff_cau.quantile(0.9):.2f} LP"
    )
    r2 = r2.assign(ancho=r2["z_p90"] - r2["z_p10"])
    print("Desglose descriptivo (pass=2):")
    for met, grp in r2.groupby("metodo", observed=False):
        print(f"  {met:<24} filas={len(grp):>6d}  ancho_med_[p10,p90]={grp['ancho'].median():>6.1f} LP")
    rc = rc.assign(ancho=rc["z_p90"] - rc["z_p10"])
    print("Desglose causal:")
    for met, grp in rc.groupby("metodo", observed=False):
        print(f"  {met:<24} filas={len(grp):>6d}  ancho_med_[p10,p90]={grp['ancho'].median():>6.1f} LP")
    if sel_mitad_real and anclas is not None and not anclas.empty:
        # Verificar consistencia cuando reemplazamos foto1 real por principal
        solo_p = r2[r2["metodo"] == "ancla_principal_grilla"]
        print(f"Filas reconstruidas con ancla `principal` (foto1 ocultada): {len(solo_p)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark básico del mapa de LP partida a partida (LAS)")
    parser.add_argument("--data-dir", type=Path, default=RAIZ / "data")
    parser.add_argument("--lp-vivo", type=Path, default=None)
    parser.add_argument("--muestra-mapa", type=int, default=1000)
    parser.add_argument("--min-tramo-vivo", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t_ini_script = time.perf_counter()
    lol_db = args.data_dir / "lol_las.db"
    ruta_vivo = resolver_lp_vivo(args.lp_vivo)

    print("Cargando train [foto1 -> foto2] desde", args.data_dir, "...")
    d = cargar_train(args.data_dir, foto2=FOTO2)
    foto1, foto2, partes, t1, t2 = d["foto1"], d["foto2"], d["partes"], d["t1"], d["t2"]
    if lol_db.exists():
        partes = partes.copy()
        partes["pos"] = cargar_posiciones_tramo(lol_db, partes, t1, t2)

    ev = evaluables(foto1, foto2, partes)
    en_caja = np.array([zlib.crc32(("caja-" + str(i)).encode()) % 100 < 15 for i in ev.index])
    en_caja_ids = set(ev.index[en_caja])
    dev = ev[~en_caja].copy()
    print(f"Evaluables: {len(ev)} total | caja cerrada={int(en_caja.sum())} | desarrollo={len(dev)}")

    k_dia = ajustar_k_diamond(dev)
    ev_vivo = pd.DataFrame()
    roles_hist = pd.DataFrame()
    if ruta_vivo is not None:
        print(f"Ajustando parámetros LAS en {ruta_vivo} ...")
        ev_vivo = cargar_eventos_vivo_las(ruta_vivo)
        t_min_vivo = int(ev_vivo["t_antes"].min()) if not ev_vivo.empty else None
        roles_hist = cargar_roles_para_puuids(lol_db, ev_vivo["puuid"].unique(), t_max_ms=t_min_vivo)
        params = ajustar_parametros_las(ev_vivo, roles_hist=roles_hist, k_diamond=k_dia)
    else:
        print("Aviso: no se encontró lp_vivo.db; usando ParametrosPaso por defecto con K_diamond ajustado.")
        params = ParametrosPaso(k_diamond=k_dia)

    print(
        f"ParametrosPaso: K_master={params.k_master:.2f}, K_diamond={params.k_diamond:.2f}, "
        f"perdon={params.perdon:.4f}, piso_quieto={params.piso_quieto:.4f}, piso_baja={params.piso_baja:.2f}, "
        f"doble_base={params.doble_base:.4f}, c_lobby={params.c_lobby:.1f}"
    )
    print("gap_nodos  :", [round(v, 1) for v in params.gap_nodos])
    print("delta_nodos:", [round(v, 2) for v in params.delta_nodos])

    a_reg, b_reg = ajustar_regla(dev)
    corte_f1 = cortes(foto1)

    if not ev_vivo.empty:
        benchmark_tramos_vivos(
            ev_vivo, roles_hist, params, a_reg, b_reg, corte_f1, min_len=args.min_tramo_vivo
        )

    foto1_dev = foto1.drop(index=list(en_caja_ids), errors="ignore")
    foto2_dev = foto2.drop(index=list(en_caja_ids), errors="ignore")
    benchmark_foto1_a_foto2(
        dev, foto1_dev, partes, t1, t2, params, a_reg, b_reg, corte_f1, seed=args.seed
    )

    benchmark_mapa_tramo(
        args.data_dir, dev, foto1_dev, foto2_dev, partes, t1, t2, params,
        n_muestra=args.muestra_mapa, seed=args.seed,
    )
    print(f"\nScript completado en {time.perf_counter() - t_ini_script:.1f}s.")


if __name__ == "__main__":
    main()
