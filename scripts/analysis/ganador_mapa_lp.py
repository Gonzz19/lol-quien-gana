"""
Evaluación causal del mapa de LP mecánico (`src/features/lp_mecanica.py`) dentro del
modelo de predicción de ganador (`quién gana`), comparando contra el híbrido base
(`Draft + Xd_85` de `variables.npz`), la `foto1` estática y el `prior_lobby_level`
del compañero (`src/features/lobby.py`, §9.S.31).

Reglas de aislamiento respetadas:
* No lee `< 04/09` (test final), ni `foto2` (`28/10`), ni `en_caja`, ni escribe en
  `data/lol_las.db` ni en `data/lp_por_partida.csv` (la caché vive en
  `data/analysis/mapa_lp_causal_v.pkl`).
* El mapa se construye estrictamente en `modo="causal"` (`t >= t1`, sin mirar
  partidas ni lobbies futuros ni `foto2`), y todas las medias/desvíos de imputación
  y estandarización se ajustan exclusivamente sobre `fit` (nunca `V`).
"""

from __future__ import annotations

import argparse
import pickle
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from scripts.analysis.mapa_lp_basico import cargar_roles_para_puuids, resolver_lp_vivo
from src.features.elo_reverso import cargar_foto
from src.features.lobby import prior_lobby_level
from src.features.padron import verificar_foto1
from src.features.lp_mecanica import (
    ParametrosPaso,
    ajustar_parametros_las,
    cargar_eventos_vivo_las,
    estimar_mapa_tramo,
)

RUTA_DB = RAIZ / "data" / "lol_las.db"
RUTA_CSV = RAIZ / "data" / "splits" / "train" / "full_dataset.csv"
RUTA_NPZ = RAIZ / "data" / "analysis" / "embeddings_partidas" / "variables.npz"
RUTA_CACHE = RAIZ / "data" / "analysis" / "mapa_lp_causal_v.pkl"

N_BOOT = 400
SEED_BOOT = 1
C_LOGIT = 0.05


def _calibrar_params(db_las: Path, t1: int, ruta_vivo: Optional[Path]) -> Tuple[ParametrosPaso, str]:
    vivo = resolver_lp_vivo(ruta_vivo)
    if vivo is None or not vivo.exists():
        return ParametrosPaso(), "defecto"
    ev = cargar_eventos_vivo_las(vivo)
    puuids = ev["puuid"].dropna().unique().tolist()
    roles = cargar_roles_para_puuids(db_las, puuids, t_max_ms=t1)
    params = ajustar_parametros_las(ev, roles_hist=roles, k_diamond=18.75)
    return params, f"lp_vivo ({vivo.name}, {len(ev)} ev)"


def _cargar_foto1(db_las: Path) -> Tuple[pd.DataFrame, int]:
    uri = f"file:{db_las.resolve()}?mode=ro"
    foto1 = cargar_foto(
        uri,
        "SELECT m.player_id, p.tier, p.rank, p.league_points, p.wins, p.losses "
        "FROM players p JOIN player_id_map m ON m.puuid = p.puuid",
    )
    with sqlite3.connect(uri, uri=True) as con:
        t1 = verificar_foto1(con)           # foto 1 congelada (§9.S.67)
    return foto1, t1


def _construir_draft(champ: np.ndarray, fit: np.ndarray) -> csr_matrix:
    """One-hot `(lane, champion_id)` con vocabulario aprendido solo en `fit`."""
    nm = champ.shape[0]
    vocab: Dict[Tuple[int, int], int] = {}
    for l_idx in range(5):
        c_fit = np.unique(champ[fit, :, l_idx])
        for c in c_fit:
            vocab[(l_idx, int(c))] = len(vocab)

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    for l_idx in range(5):
        for eq, signo in ((0, 1.0), (1, -1.0)):
            cs = champ[:, eq, l_idx]
            for i, c in enumerate(cs):
                j = vocab.get((l_idx, int(c)))
                if j is not None:
                    rows.append(i)
                    cols.append(j)
                    data.append(signo)
    return csr_matrix((data, (rows, cols)), shape=(nm, len(vocab)), dtype=np.float32)


def _cargar_o_calcular_mapa(
    mids_npz: np.ndarray,
    t_npz: np.ndarray,
    y_npz: np.ndarray,
    t1: int,
    foto1: pd.DataFrame,
    params: ParametrosPaso,
    rehacer: bool,
) -> dict:
    if RUTA_CACHE.exists() and not rehacer:
        with open(RUTA_CACHE, "rb") as f:
            cache = pickle.load(f)
        if cache.get("t1") == t1 and cache.get("n_mids") == len(mids_npz):
            print(f"[cache] Reutilizando mapa causal: {RUTA_CACHE}")
            return cache

    print("[datos] Leyendo full_dataset.csv...")
    mids_set = set(mids_npz.tolist())
    t_max_v = int(t_npz.max())

    chunks_npz = []
    chunks_post = []
    cols = ["match_id", "game_creation", "player_id", "team_id", "lane", "win"]
    for ch in pd.read_csv(RUTA_CSV, chunksize=500_000, usecols=cols):
        en_npz = ch["match_id"].isin(mids_set)
        if en_npz.any():
            chunks_npz.append(ch.loc[en_npz])
        en_post = (
            ch["match_id"].str.startswith("LA2_")
            & (ch["game_creation"] >= t1)
            & (ch["game_creation"] <= t_max_v)
        )
        if en_post.any():
            chunks_post.append(ch.loc[en_post])

    df_npz = pd.concat(chunks_npz, ignore_index=True)
    df_npz = df_npz.sort_values(["game_creation", "match_id", "team_id", "lane"], kind="mergesort").reset_index(drop=True)
    assert len(df_npz) == 10 * len(mids_npz), f"Filas esperadas {10*len(mids_npz)}, leídas {len(df_npz)}"
    assert (df_npz["match_id"].to_numpy()[::10] == mids_npz).all(), "Desalineado respecto a variables.npz"
    assert (df_npz["win"].to_numpy()[::10] == y_npz).all(), "Desalineado en etiquetas y de variables.npz"

    df_post = pd.concat(chunks_post, ignore_index=True)
    df_post = df_post.sort_values(["player_id", "game_creation", "match_id"], kind="mergesort").reset_index(drop=True)
    df_post["pos"] = df_post["lane"].fillna("UNKNOWN")

    print(f"[mapa] Estimando mapa causal en {len(df_post):,} filas (>= t1, {df_post['match_id'].nunique():,} partidas)...")
    t0 = time.perf_counter()
    res_mapa = estimar_mapa_tramo(
        df_post,
        foto1,
        None,
        t1,
        int(df_post["game_creation"].max()) + 1,
        params,
        modo="causal",
        step=10.0,
    )
    dt = time.perf_counter() - t0
    print(f"[mapa] Listo en {dt:.1f} s.")

    df_post["lp_causal"] = res_mapa["z_media"].to_numpy(dtype=np.float64)
    df_post["lp_mediana"] = res_mapa["z_mediana"].to_numpy(dtype=np.float64)
    df_post["lp_incert"] = (res_mapa["z_p90"] - res_mapa["z_p10"]).to_numpy(dtype=np.float64)
    df_post["metodo"] = res_mapa["metodo"].to_numpy(dtype=object)

    f1_z = foto1["z"].to_dict()
    df_post["lp_foto"] = df_post["player_id"].map(f1_z).to_numpy(dtype=np.float64)

    df_post_cron = df_post.sort_values(["game_creation", "match_id", "team_id", "lane"], kind="mergesort").reset_index(drop=True)
    keep_cols = [
        "match_id",
        "game_creation",
        "player_id",
        "team_id",
        "lane",
        "lp_foto",
        "lp_causal",
        "lp_incert",
        "metodo",
    ]
    RUTA_CACHE.parent.mkdir(parents=True, exist_ok=True)
    cache = {
        "t1": t1,
        "n_mids": len(mids_npz),
        "params": params,
        "df_post_cron": df_post_cron[keep_cols],
        "df_npz_keys": df_npz[["match_id", "game_creation", "player_id", "team_id", "lane"]],
    }
    with open(RUTA_CACHE, "wb") as f:
        pickle.dump(cache, f)
    print(f"[cache] Guardado en {RUTA_CACHE}")
    return cache


def _metricas(y: np.ndarray, p: np.ndarray) -> Tuple[float, float, float]:
    p_clip = np.clip(p, 1e-6, 1.0 - 1e-6)
    return (
        float(roc_auc_score(y, p_clip)),
        float(accuracy_score(y, p_clip >= 0.5)),
        float(log_loss(y, p_clip)),
    )


def _diff_boot(
    y: np.ndarray,
    p_mod: np.ndarray,
    p_ref: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
) -> Tuple[float, float, float]:
    diff = float(roc_auc_score(y, p_mod) - roc_auc_score(y, p_ref))
    n = len(y)
    diffs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        diffs[b] = diff if yb.min() == yb.max() else roc_auc_score(yb, p_mod[idx]) - roc_auc_score(yb, p_ref[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return diff, float(lo), float(hi)


def _construir_diffs(arr: np.ndarray, modo: str) -> np.ndarray:
    """Convierte `(N, 2, 5, F)` a diferencias azul - rojo (`"equipo"` o `"por_linea"`)."""
    diff_lineas = arr[:, 0, :, :] - arr[:, 1, :, :]
    diff_eq = diff_lineas.mean(axis=1)
    if modo == "equipo":
        return diff_eq.astype(np.float32)
    nm, _, f = diff_lineas.shape
    return np.concatenate([diff_lineas.reshape(nm, 5 * f), diff_eq], axis=1).astype(np.float32)


def _estandarizar_y_ajustar(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_val: np.ndarray,
    base_sparse_fit: Optional[csr_matrix],
    base_sparse_val: Optional[csr_matrix],
    extra_fit: Optional[np.ndarray],
    extra_val: Optional[np.ndarray],
    post_fit: Optional[np.ndarray] = None,
    post_val: Optional[np.ndarray] = None,
    escalar_post_en_activos: bool = False,
) -> np.ndarray:
    partes_fit = []
    partes_val = []
    if base_sparse_fit is not None:
        partes_fit.append(base_sparse_fit)
        partes_val.append(base_sparse_val)

    if x_fit.shape[1] > 0:
        mu = x_fit.mean(axis=0)
        sd = np.maximum(x_fit.std(axis=0), 1e-6)
        partes_fit.append(csr_matrix(((x_fit - mu) / sd).astype(np.float32)))
        partes_val.append(csr_matrix(((x_val - mu) / sd).astype(np.float32)))

    if extra_fit is not None and extra_fit.shape[1] > 0:
        if escalar_post_en_activos and post_fit is not None:
            mu_e = extra_fit[post_fit].mean(axis=0)
            sd_e = np.maximum(extra_fit[post_fit].std(axis=0), 1e-6)
            ef = np.where(post_fit[:, None], (extra_fit - mu_e) / sd_e, 0.0)
            ev = np.where(post_val[:, None], (extra_val - mu_e) / sd_e, 0.0)
        else:
            mu_e = extra_fit.mean(axis=0)
            sd_e = np.maximum(extra_fit.std(axis=0), 1e-6)
            ef = (extra_fit - mu_e) / sd_e
            ev = (extra_val - mu_e) / sd_e
        partes_fit.append(csr_matrix(ef.astype(np.float32)))
        partes_val.append(csr_matrix(ev.astype(np.float32)))

    X_f = hstack(partes_fit, format="csr") if len(partes_fit) > 1 else partes_fit[0]
    X_v = hstack(partes_val, format="csr") if len(partes_val) > 1 else partes_val[0]
    clf = LogisticRegression(C=C_LOGIT, max_iter=5000, solver="lbfgs")
    clf.fit(X_f, y_fit)
    return clf.predict_proba(X_v)[:, 1]


def _imprimir_tabla(
    titulo: str,
    grupos: List[Tuple[str, Optional[str]]],
    tensores: Dict[str, np.ndarray],
    repr_modo: str,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    sp_fit: csr_matrix,
    sp_val: csr_matrix,
    m_fit: np.ndarray,
    m_val: np.ndarray,
    m10_val: Optional[np.ndarray] = None,
    post_fit: Optional[np.ndarray] = None,
    post_val: Optional[np.ndarray] = None,
    escalar_post: bool = False,
) -> Dict[str, np.ndarray]:
    print(titulo)
    preds: Dict[str, np.ndarray] = {}
    for nombre, clave in grupos:
        ext = None if clave is None else _construir_diffs(tensores[clave], repr_modo)
        p = _estandarizar_y_ajustar(
            x_fit, y_fit, x_val, sp_fit, sp_val,
            None if ext is None else ext[m_fit],
            None if ext is None else ext[m_val],
            post_fit=post_fit, post_val=post_val,
            escalar_post_en_activos=escalar_post,
        )
        preds[nombre] = p
        auc, acc, ll = _metricas(y_val, p)
        s_10 = ""
        if m10_val is not None:
            a10 = roc_auc_score(y_val[m10_val], p[m10_val])
            d10 = a10 - roc_auc_score(y_val[m10_val], preds["base"][m10_val]) if nombre != "base" else 0.0
            s_10 = f" | 10f1={a10:.5f} ({d10:+.5f})" if nombre != "base" else f" | 10f1={a10:.5f}"
        if nombre == "base":
            print(f"  {nombre:32s} | AUC={auc:.5f} | Acc={acc*100:.2f}% | LL={ll:.5f}{s_10}")
        else:
            d, lo, hi = _diff_boot(y_val, p, preds["base"], np.random.default_rng(SEED_BOOT))
            extra_niv = ""
            if "+ nivel(foto) [lobby.py]" in preds and nombre != "+ nivel(foto) [lobby.py]":
                df_d, df_lo, df_hi = _diff_boot(y_val, p, preds["+ nivel(foto) [lobby.py]"], np.random.default_rng(SEED_BOOT))
                extra_niv = f" | vs nivF={df_d:+.5f} [{df_lo:+.5f},{df_hi:+.5f}]"
            print(f"  {nombre:32s} | AUC={auc:.5f} ({d:+.5f} [{lo:+.5f},{hi:+.5f}]) | Acc={acc*100:.2f}% | LL={ll:.5f}{s_10}{extra_niv}")
    return preds


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluación causal del mapa de LP en quién gana")
    ap.add_argument("--rehacer", action="store_true", help="Recalcula el mapa LP causal ignorando caché")
    ap.add_argument("--lp-vivo", type=Path, default=None, help="Ruta opcional a lp_vivo.db")
    args = ap.parse_args()

    foto1, t1 = _cargar_foto1(RUTA_DB)
    params, src_params = _calibrar_params(RUTA_DB, t1, args.lp_vivo)
    print(f"[setup] Foto 1 t1={t1} ({pd.to_datetime(t1, unit='ms', utc=True)}), jugadores={len(foto1):,}")
    print(f"[setup] ParametrosPaso ({src_params}): k_m={params.k_master:.2f}, k_d={params.k_diamond:.2f}, "
          f"perdon={params.perdon:.3f}, doble_base={params.doble_base:.3f}, sigma_paso={params.sigma_paso:.2f}")

    z = np.load(RUTA_NPZ, allow_pickle=True)
    mids = z["mids"]
    Xj = z["Xj"]
    champ = z["champ"]
    t = z["t"]
    val = z["val"].astype(bool)
    y = z["y"].astype(np.int64)
    nm = len(mids)

    fit_all = ~val
    fit_post = (t >= t1) & (~val)
    post_mask = t >= t1
    assert not ((t < t1) & val).any(), "Validación V contiene partidas previas a Foto 1"
    print(f"[splits] Total={nm:,} | fit_all={fit_all.sum():,} | fit_post(A)={fit_post.sum():,} | V={val.sum():,}")

    cache = _cargar_o_calcular_mapa(mids, t, y, t1, foto1, params, rehacer=args.rehacer)
    df_post_cron: pd.DataFrame = cache["df_post_cron"]
    df_npz_keys: pd.DataFrame = cache["df_npz_keys"]

    mids_fit_post = set(mids[fit_post].tolist())
    en_fit_post_df = df_post_cron["match_id"].isin(mids_fit_post)
    mu0_foto = float(df_post_cron.loc[en_fit_post_df, "lp_foto"].mean())
    mu0_mapa = float(df_post_cron.loc[en_fit_post_df, "lp_causal"].mean())

    print("[lobby] Calculando prior_lobby_level sobre >= t1 (foto vs mapa causal)...")
    niv_foto = prior_lobby_level(df_post_cron, "lp_foto", k=5, mu0=mu0_foto, window=10, min_otros=3)
    niv_mapa = prior_lobby_level(df_post_cron, "lp_causal", k=5, mu0=mu0_mapa, window=10, min_otros=3)

    df_post_feat = df_post_cron.copy()
    df_post_feat["foto_niv_acum"] = niv_foto["nivel_acum"].to_numpy(dtype=np.float64)
    df_post_feat["foto_niv_v10"] = niv_foto["nivel_v10"].to_numpy(dtype=np.float64)
    df_post_feat["foto_niv_n"] = np.log1p(niv_foto["n_nivel"].to_numpy(dtype=np.float64))
    df_post_feat["mapa_niv_acum"] = niv_mapa["nivel_acum"].to_numpy(dtype=np.float64)
    df_post_feat["mapa_niv_v10"] = niv_mapa["nivel_v10"].to_numpy(dtype=np.float64)
    df_post_feat["mapa_niv_n"] = np.log1p(niv_mapa["n_nivel"].to_numpy(dtype=np.float64))

    df_npz_keys = df_npz_keys.copy()
    df_npz_keys["_pos_orig"] = np.arange(len(df_npz_keys), dtype=np.int64)
    merged = df_npz_keys.merge(
        df_post_feat.drop(columns=["game_creation", "team_id", "lane"]),
        on=["match_id", "player_id"],
        how="left",
    ).sort_values("_pos_orig", kind="mergesort").reset_index(drop=True)
    assert len(merged) == 10 * nm, "Merge multiplicó filas"

    # Diagnóstico en V
    en_v_rows = np.repeat(val, 10)
    v_met = merged.loc[en_v_rows, "metodo"].value_counts(dropna=False)
    diff_f1_v = (merged.loc[en_v_rows, "lp_causal"] - merged.loc[en_v_rows, "lp_foto"]).abs().dropna()
    print("\n=== COBERTURA EN VALIDACIÓN V (n=26,934 partidas = 269,340 filas) ===")
    print(f"Métodos mapa causal: {v_met.to_dict()}")
    print(f"Cobertura foto1:     {merged.loc[en_v_rows, 'lp_foto'].notna().mean()*100:.2f}% | "
          f"Mapa (!=sin_dato): {merged.loc[en_v_rows, 'lp_causal'].notna().mean()*100:.2f}% | "
          f"nivel>=1: {merged.loc[en_v_rows, 'foto_niv_n'].gt(0).mean()*100:.2f}%")
    print(f"|lp_causal - lp_foto| en V (n={len(diff_f1_v):,}): media={diff_f1_v.mean():.1f}, p50={diff_f1_v.median():.1f}, p90={diff_f1_v.quantile(0.9):.1f}")

    # Construir matrices (NM, 2, 5)
    post_row = np.repeat(post_mask, 10)
    tiene_foto = (merged["lp_foto"].notna() & post_row).to_numpy(dtype=np.float32)

    lp_foto_crudo = np.where(post_row, merged["lp_foto"].fillna(mu0_foto).to_numpy(dtype=np.float32) - mu0_foto, 0.0)
    lp_foto_fill = np.where(
        post_row,
        np.where(merged["lp_foto"].isna(), merged["foto_niv_v10"].fillna(mu0_foto), merged["lp_foto"]).astype(np.float32) - mu0_foto,
        0.0,
    )
    lp_mapa_crudo = np.where(post_row, merged["lp_causal"].fillna(mu0_mapa).to_numpy(dtype=np.float32) - mu0_mapa, 0.0)
    lp_mapa_limpio = np.where(
        post_row,
        np.where(merged["metodo"].to_numpy() == "causal_lobby", merged["mapa_niv_v10"].fillna(mu0_mapa), merged["lp_causal"].fillna(mu0_mapa)).astype(np.float32) - mu0_mapa,
        0.0,
    )
    dl_foto = np.where(tiene_foto > 0.5, lp_mapa_limpio - lp_foto_crudo, 0.0).astype(np.float32)

    incert_raw = merged["lp_incert"].to_numpy(dtype=np.float32)
    inc_mu_fit = float(np.nanmean(incert_raw[np.repeat(fit_post, 10)]))
    lp_inc = np.where(post_row, np.nan_to_num(incert_raw, nan=inc_mu_fit) - inc_mu_fit, 0.0)

    niv_f_acum = np.where(post_row, merged["foto_niv_acum"].fillna(mu0_foto).to_numpy(dtype=np.float32) - mu0_foto, 0.0)
    niv_f_v10 = np.where(post_row, merged["foto_niv_v10"].fillna(mu0_foto).to_numpy(dtype=np.float32) - mu0_foto, 0.0)
    niv_f_n = np.where(post_row, merged["foto_niv_n"].fillna(0.0).to_numpy(dtype=np.float32), 0.0)

    niv_m_acum = np.where(post_row, merged["mapa_niv_acum"].fillna(mu0_mapa).to_numpy(dtype=np.float32) - mu0_mapa, 0.0)
    niv_m_v10 = np.where(post_row, merged["mapa_niv_v10"].fillna(mu0_mapa).to_numpy(dtype=np.float32) - mu0_mapa, 0.0)
    niv_m_n = np.where(post_row, merged["mapa_niv_n"].fillna(0.0).to_numpy(dtype=np.float32), 0.0)
    brecha_mmr = (niv_m_v10 - lp_mapa_limpio).astype(np.float32)

    def _t(v: np.ndarray) -> np.ndarray:
        return v.reshape(nm, 2, 5)

    m10_val = (_t(tiene_foto)[val].sum(axis=(1, 2)) == 10)

    tensores = {
        "foto_crudo": np.stack([_t(lp_foto_crudo), _t(tiene_foto)], axis=-1),
        "foto_fill": _t(lp_foto_fill)[..., None],
        "mapa_crudo": _t(lp_mapa_crudo)[..., None],
        "mapa_fill": _t(lp_mapa_limpio)[..., None],
        "act_y_tf1": np.stack([_t(tiene_foto), _t(niv_f_n)], axis=-1),
        "niv_foto": np.stack([_t(niv_f_acum), _t(niv_f_v10), _t(niv_f_n)], axis=-1),
        "niv_mapa": np.stack([_t(niv_m_acum), _t(niv_m_v10), _t(niv_m_n)], axis=-1),
        "mapa_limpio": np.stack([_t(lp_mapa_limpio), _t(dl_foto), _t(lp_inc), _t(tiene_foto)], axis=-1),
        "niv_foto_y_mapa": np.stack([_t(niv_f_acum), _t(niv_f_v10), _t(niv_f_n), _t(lp_mapa_limpio), _t(dl_foto), _t(lp_inc), _t(tiene_foto)], axis=-1),
        "niv_mapa_y_mapa": np.stack([_t(niv_m_acum), _t(niv_m_v10), _t(niv_m_n), _t(lp_mapa_limpio), _t(dl_foto), _t(lp_inc), _t(tiene_foto), _t(brecha_mmr)], axis=-1),
    }

    draft_a = _construir_draft(champ, fit_all)
    draft_post = _construir_draft(champ, fit_post)
    Xd = (Xj[:, 0] - Xj[:, 1]).reshape(nm, -1)

    grupos = [
        ("base", None),
        ("+ lp_foto (fill niv)", "foto_fill"),
        ("+ lp_causal (fill niv)", "mapa_fill"),
        ("+ lp_causal (crudo lobby)", "mapa_crudo"),
        ("+ actividad (tf1 + n_niv)", "act_y_tf1"),
        ("+ lp_foto + tf1", "foto_crudo"),
        ("+ nivel(foto) [lobby.py]", "niv_foto"),
        ("+ nivel(mapa_lp)", "niv_mapa"),
        ("+ mapa_lp (limpio)", "mapa_limpio"),
        ("+ niv(foto) + mapa_lp", "niv_foto_y_mapa"),
        ("+ niv(mapa) + mapa_lp", "niv_mapa_y_mapa"),
    ]

    # Régimen 1 (A -> V)
    for repr_modo in ("equipo", "por_linea"):
        _imprimir_tabla(
            f"\n=== RÉGIMEN 1A: Draft solo (fit = A, n={fit_post.sum():,}) -> V ({repr_modo}) ===",
            grupos, tensores, repr_modo,
            np.empty((fit_post.sum(), 0), dtype=np.float32), y[fit_post],
            np.empty((val.sum(), 0), dtype=np.float32), y[val],
            draft_post[fit_post], draft_post[val], fit_post, val, m10_val=m10_val,
        )
        _imprimir_tabla(
            f"\n=== RÉGIMEN 1B: Híbrido Draft + Xd_85 (fit = A, n={fit_post.sum():,}) -> V ({repr_modo}) ===",
            grupos, tensores, repr_modo,
            Xd[fit_post], y[fit_post], Xd[val], y[val],
            draft_post[fit_post], draft_post[val], fit_post, val, m10_val=m10_val,
        )

    # Régimen 2 (~val -> V, escalado en post_t1)
    preds_r2_eq = _imprimir_tabla(
        f"\n=== RÉGIMEN 2: Híbrido Draft + Xd_85 (fit = ~val, n={fit_all.sum():,}) -> V (equipo) ===",
        grupos, tensores, "equipo",
        Xd[fit_all], y[fit_all], Xd[val], y[val],
        draft_a[fit_all], draft_a[val], fit_all, val, m10_val=m10_val,
        post_fit=post_mask[fit_all], post_val=post_mask[val], escalar_post=True,
    )
    preds_r2_ln = _imprimir_tabla(
        f"\n=== RÉGIMEN 2: Híbrido Draft + Xd_85 (fit = ~val, n={fit_all.sum():,}) -> V (por_linea) ===",
        grupos, tensores, "por_linea",
        Xd[fit_all], y[fit_all], Xd[val], y[val],
        draft_a[fit_all], draft_a[val], fit_all, val, m10_val=m10_val,
        post_fit=post_mask[fit_all], post_val=post_mask[val], escalar_post=True,
    )

    # Desglose en V usando representación equipo en Régimen 2
    print("\n=== DESGLOSE EN V (Régimen 2, equipo) ===")
    yv = y[val]
    tv = t[val]
    t_med = np.median(tv)
    lp_prom = np.nanmean(merged.loc[en_v_rows, "lp_causal"].to_numpy(dtype=np.float64).reshape(int(val.sum()), 10), axis=1)
    subgrupos = [
        ("V temprana (primera mitad)", tv <= t_med),
        ("V tardía (segunda mitad)", tv > t_med),
        ("Diamond (LP medio < 400)", lp_prom < 400.0),
        ("Master+ (LP medio >= 400)", lp_prom >= 400.0),
        ("Los 10 con foto1", m10_val),
        ("Falta >= 1 foto1", ~m10_val),
    ]
    mods_sub = ["base", "+ lp_causal (fill niv)", "+ nivel(foto) [lobby.py]", "+ niv(foto) + mapa_lp", "+ niv(mapa) + mapa_lp"]
    for s_nom, m_s in subgrupos:
        print(f"  [{s_nom} | n={m_s.sum():,}]")
        for mn in mods_sub:
            pm = preds_r2_eq[mn][m_s]
            auc, _, ll = _metricas(yv[m_s], pm)
            if mn == "base":
                print(f"    {mn:30s}: AUC={auc:.5f} | LL={ll:.5f}")
            else:
                db, lo, hi = _diff_boot(yv[m_s], pm, preds_r2_eq["base"][m_s], np.random.default_rng(SEED_BOOT))
                dn, _, _ = _diff_boot(yv[m_s], pm, preds_r2_eq["+ nivel(foto) [lobby.py]"][m_s], np.random.default_rng(SEED_BOOT))
                suf_n = "" if mn == "+ nivel(foto) [lobby.py]" else f" | vs nivF={dn:+.5f}"
                print(f"    {mn:30s}: AUC={auc:.5f} ({db:+.5f} [{lo:+.5f},{hi:+.5f}]) | LL={ll:.5f}{suf_n}")


if __name__ == "__main__":
    main()
