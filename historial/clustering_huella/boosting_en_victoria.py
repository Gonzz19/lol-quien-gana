"""
Prueba causal: ¿mejora la predicción del modelo de victoria si, ante sospecha de boosting
(huella estilística distinta + winrate reciente alto), reemplazamos el historial completo de esa
cuenta por SOLO las partidas previas del tramo sospechoso? (pedido del usuario, 2026-11-19).

Diseño estrictamente causal (sin fuga del resultado de la partida evaluada `t`, y sin tocar el test):

1. **Modelo de victoria base (`brainstorm_base.py`):** logística híbrida (draft + 85 variables +
   mapa de LP) evaluada estrictamente fuera de muestra en los bloques temporales `A2, A3, A4` y en
   `V Master+` (30.288 partidas Master+ fuera de muestra).
2. **Disparador causal en línea (usando SOLO partidas estrictamente anteriores a `t`):**
   * Con los embeddings 2-fold fuera de muestra de la red contrastiva (41 variables: 23 estadísticas
     + 18 hábitos), para cada jugador con $\ge W + 15$ partidas previas con timeline se compara su
     **ventana reciente** de $W$ partidas previas ($W \in \{8, 10, 15\}$) contra su **historial
     anterior** (todas las previas anteriores a esa ventana):
     - `d_estilo_prev`: distancia entre `emb(recientes W)` y `emb(anteriores)`.
     - `wr_rec` y `delta_wr_prev = wr_rec - wr_ant`.
   * Se definen cuatro regímenes estrictamente antes de empezar la partida `t`:
     - **Sospecha de boosting:** `d_estilo_prev >= p90` (o `p95`) Y `wr_rec >= 0,70` Y `delta_wr_prev >= +0,20`.
     - **Control (racha buena, misma huella):** `d_estilo_prev < p50` Y `wr_rec >= 0,70` Y `delta_wr_prev >= +0,20`.
     - **Control (cambio de huella sin WR alto):** `d_estilo_prev >= p90` Y `|delta_wr_prev| < 0,12`.
     - **Control (normal, estable):** `d_estilo_prev < p50` Y `|delta_wr_prev| < 0,12`.
3. **Reemplazo de perfil en inferencia + ablación de variables:** para el jugador disparado en la
   partida `t`, se recalculan sus variables de historial usando **únicamente sus últimas `W`
   partidas previas** (descartando el historial viejo del dueño original) y se evalúa con el **mismo
   modelo ya entrenado** (sin reajustar pesos). En `W = 15` se descompone el efecto por grupo de
   variables (`Solo margen`, `Solo WR`, `Solo rol`, `Solo campeón`, `Reset limpio`, `Reset total`) y
   se mide el impacto global sobre las 16.428 partidas de `V Master+`.
4. **Estudio de caso sobre las 8 candidatas de §5:** además se inspeccionan aparte las partidas de
   las 8 cuentas detectadas en §5 dentro de `A2–A4 + V`, midiendo el cambio de probabilidad y
   log-loss al usar solo las partidas previas dentro del subgrupo boosteado.

Salidas en `data/analysis/huella/boosting_en_victoria*.{csv,json}`. Usa las redes 2-fold de
`huella_patrones_cuenta.redes_2fold` (residuo por pliegue desde §9.S.77 punto 5; `--residuo global` es la
versión anterior, con salidas `_residuo_global`) y lee `patrones_cuentas_todos_roles*.csv` de ese script, que
debe correr antes con el mismo `--residuo`.

    python historial/clustering_huella/boosting_en_victoria.py [--residuo global]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.features.fingerprint import STYLE_FEATURES, add_rates  # noqa: E402
from src.features.margin import _residuo, lane_diffs, lane_scale  # noqa: E402
from scripts.analysis import brainstorm_base as bb  # noqa: E402
from scripts.analysis import huella_mejoras as hm  # noqa: E402
from scripts.analysis.huella_patrones_cuenta import redes_2fold  # noqa: E402

SALIDA = hm.SALIDA
FEATS_TL = STYLE_FEATURES + hm.HABITOS
LANES = ["BOTTOM", "JUNGLE", "MIDDLE", "TOP", "UTILITY"]
STATS_MARGEN = [
    "kills_pm", "assists_pm", "damage_pm", "damage_share_pct", "kill_participation_pct",
    "deaths_pm", "damage_taken_pm", "gold_pm", "cs_pm", "gold_share_pct", "vision_pm",
    "d_gold_10", "d_gold_15", "d_xp_15", "d_cs_15",
]
USE_TRAIN = [
    "match_id", "game_creation", "game_duration", "player_id", "team_id", "win",
    "champion_id", "lane", "kills", "deaths", "assists", "kill_participation_pct",
    "damage_share_pct", "gold_share_pct", "total_damage_dealt_to_champions",
    "total_damage_taken", "gold_earned", "total_minions_killed", "neutral_minions_killed",
    "vision_score", "gold_10", "gold_15", "xp_15", "cs_15",
]
T0 = time.time()
log = lambda s: print(f"[{time.time() - T0:6.1f}s] {s}", flush=True)  # noqa: E731


def cargar_train_residuos():
    """Carga el mismo train ordenado de `embeddings_partidas.py` y precalcula los residuos causales `r`."""
    df = pd.read_csv(ROOT / "data" / "splits" / "train" / "full_dataset.csv", usecols=USE_TRAIN)
    df = df[df.lane.isin(LANES)]
    df = df[df.groupby(["match_id", "team_id"]).lane.transform("nunique") == 5]
    df = df[df.groupby("match_id").lane.transform("size") == 10]
    df = df[df.groupby("match_id").win.transform("sum") == 5].copy()
    df = add_rates(df).sort_values(["game_creation", "match_id", "team_id", "lane"]).reset_index(drop=True)
    df = lane_diffs(df, ["gold_10", "gold_15", "xp_15", "cs_15"])

    escala = lane_scale(df, STATS_MARGEN)
    R = np.zeros((len(df), len(STATS_MARGEN)), dtype=np.float32)
    V = np.zeros((len(df), len(STATS_MARGEN)), dtype=np.float32)
    for j, c in enumerate(STATS_MARGEN):
        r, v = _residuo(df, c, 20.0, escala)
        R[:, j] = r.to_numpy(np.float32)
        V[:, j] = v.to_numpy(np.float32)
    return df, R, V


def perfil_desde_indices(ix_prev: np.ndarray, row_cur: dict, R: np.ndarray, V: np.ndarray,
                         wins_arr: np.ndarray, champs_arr: np.ndarray, lanes_arr: np.ndarray,
                         x_orig: np.ndarray, col_idx: dict) -> np.ndarray:
    """Recalcula las variables de historial del jugador usando SOLO las filas `ix_prev` (estrictamente < t)."""
    x = x_orig.copy()
    n = len(ix_prev)
    if n == 0:
        return x
    S_r = R[ix_prev].sum(axis=0)
    N_v = V[ix_prev].sum(axis=0)
    m_rec = S_r / (N_v + 5.0)
    for j, st in enumerate(STATS_MARGEN):
        x[col_idx[f"acum_{st}"]] = m_rec[j]
        x[col_idx[f"v20_{st}"]] = m_rec[j]

    w_tot = float(wins_arr[ix_prev].sum())
    wr_shrunk = (w_tot + 5.0) / (n + 10.0)
    x[col_idx["wr_acum"]] = wr_shrunk - 0.5
    x[col_idx["wr_v20"]] = wr_shrunk - 0.5

    ch_cur = row_cur["champion_id"]
    ln_cur = row_cur["lane"]
    m_ch = champs_arr[ix_prev] == ch_cur
    n_ch = int(m_ch.sum())
    w_ch = float(wins_arr[ix_prev][m_ch].sum())
    n_ln = int((lanes_arr[ix_prev] == ln_cur).sum())

    x[col_idx["logn_tot"]] = np.log1p(n)
    x[col_idx["logn_champ"]] = np.log1p(n_ch)
    x[col_idx["estreno"]] = 1.0 if n_ch == 0 else 0.0
    x[col_idx["wr_champ"]] = (w_ch + 5.0 * wr_shrunk) / (n_ch + 5.0) - wr_shrunk
    x[col_idx["share_champ"]] = n_ch / (n + 1.0)
    x[col_idx["share_lane"]] = (n_ln + 1.0) / (n + 2.0)
    x[col_idx["offrole"]] = 1.0 if (n >= 10 and (n_ln / max(n, 1)) < 0.2) else 0.0
    return x


def ajustar_modelos_bloques(D, con_mapa=True):
    """Ajusta los 4 modelos fuera de muestra (A2, A3, A4 y V Master+) devolviendo escaladores y LR."""
    Xd, post = D["Xd"], D["post"]
    fi, bl = D["fi"], D["bloque"]
    nd = D["DRAFT"].shape[1]
    nx = Xd.shape[1]
    modelos = {}

    def fit_one(tr):
        mu, sd = Xd[tr].mean(0), Xd[tr].std(0) + 1e-6
        Ptr = [D["DRAFT"][tr], sparse.csr_matrix((Xd[tr] - mu) / sd)]
        mapa_stats = None
        if con_mapa:
            ref = tr[post[tr]]
            me, se = D["mapa"][ref].mean(0), D["mapa"][ref].std(0) + 1e-6
            mapa_stats = (me, se)
            mp_tr = np.where(post[tr][:, None], (D["mapa"][tr] - me) / se, 0.0).astype(np.float32)
            Ptr.append(sparse.csr_matrix(mp_tr))
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=0.05, max_iter=1500, tol=1e-3).fit(sparse.hstack(Ptr).tocsr(), D["y"][tr])
        w_xd = lr.coef_[0][nd: nd + nx].astype(np.float64)
        return {"lr": lr, "mu": mu, "sd": sd, "mapa_stats": mapa_stats, "w_xd": w_xd}

    for b in D["EV"]:
        modelos[f"A{b + 1}"] = (np.flatnonzero((bl == b) & D["master"]), fit_one(fi[bl[fi] < b]))
    modelos["V"] = (np.flatnonzero(D["val"]), fit_one(D["fi_v"]))
    return modelos


def predecir_lote(D, mod_info, idx, Xj_custom=None):
    """Predice probabilidad de victoria azul en las partidas `idx`, con `Xj` opcional modificado."""
    Xj = D["Xj"] if Xj_custom is None else Xj_custom
    Xd = (Xj[idx, 0] - Xj[idx, 1]).reshape(len(idx), -1).astype(np.float32)
    Pte = [D["DRAFT"][idx], sparse.csr_matrix((Xd - mod_info["mu"]) / mod_info["sd"])]
    if mod_info["mapa_stats"] is not None:
        me, se = mod_info["mapa_stats"]
        mp = np.where(D["post"][idx][:, None], (D["mapa"][idx] - me) / se, 0.0).astype(np.float32)
        Pte.append(sparse.csr_matrix(mp))
    return mod_info["lr"].predict_proba(sparse.hstack(Pte).tocsr())[:, 1]


def boot_delta(y, p_new, p_old, seed=0, B=1000):
    rng = np.random.default_rng(seed)
    n = len(y)
    eps = 1e-15
    pn, po = np.clip(p_new, eps, 1 - eps), np.clip(p_old, eps, 1 - eps)
    ll_n = -(y * np.log(pn) + (1 - y) * np.log(1 - pn))
    ll_o = -(y * np.log(po) + (1 - y) * np.log(1 - po))
    br_n = (pn - y) ** 2
    br_o = (po - y) ** 2
    idx = rng.integers(0, n, size=(B, n))
    dll = (ll_n - ll_o)[idx].mean(axis=1)
    dbr = (br_n - br_o)[idx].mean(axis=1)
    return {
        "ll_old": float(ll_o.mean()), "ll_new": float(ll_n.mean()),
        "delta_ll": float((ll_n - ll_o).mean()),
        "delta_ll_lo": float(np.quantile(dll, 0.025)), "delta_ll_hi": float(np.quantile(dll, 0.975)),
        "brier_old": float(br_o.mean()), "brier_new": float(br_n.mean()),
        "delta_brier": float((br_n - br_o).mean()),
        "acc_old": float(((po > 0.5) == y).mean()), "acc_new": float(((pn > 0.5) == y).mean()),
        "auc_old": float(roc_auc_score(y, po)) if len(np.unique(y)) == 2 else float("nan"),
        "auc_new": float(roc_auc_score(y, pn)) if len(np.unique(y)) == 2 else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser(description="Boosting en el modelo de victoria (§10.8)")
    ap.add_argument("--residuo", choices=["pliegue", "global"], default="pliegue",
                    help="residuo de la huella por pliegue (default, §9.S.77 punto 5) o con todos los jugadores, "
                         "como antes (las salidas llevan el sufijo _residuo_global y leen las de "
                         "huella_patrones_cuenta.py --residuo global)")
    residuo = ap.parse_args().residuo
    sufijo = "_residuo_global" if residuo == "global" else ""
    log("cargando brainstorm_base y residuos causales del train...")
    D = bb.cargar()
    df_tr, R, V = cargar_train_residuos()
    assert len(df_tr) == D["NM"] * 10
    assert np.array_equal(df_tr["match_id"].to_numpy()[::10], D["Z"]["mids"])
    col_idx = {c: i for i, c in enumerate(D["cols"])}

    # 1. Redes contrastivas 2-fold fuera de muestra (las mismas de huella_patrones_cuenta.py, con su caché). Con
    #    el residuo por pliegue (default, §9.S.77 punto 5) cada jugador se proyecta con filas normalizadas solo
    #    con la otra mitad; `--residuo global` es la versión anterior (resid_timeline.pkl)
    tl, emb_of = redes_2fold(residuo)
    tl = tl.sort_values(["player_id", "game_creation"]).reset_index(drop=True)

    # 2. Calcular señales causales en línea para cada (match_id, player_id) en A2-A4 y V Master+
    eval_mask_m = D["val"] | ((D["bloque"] >= 1) & (D["bloque"] <= 3) & D["master"])
    eval_mids = set(D["Z"]["mids"][eval_mask_m])
    log(f"partidas Master+ fuera de muestra (A2-A4 + V): {len(eval_mids):,}")

    X_tl = tl[FEATS_TL].to_numpy(np.float32)
    W_tl = tl["win"].to_numpy(int)
    L_tl = tl["lane"].to_numpy()
    M_tl = tl["match_id"].to_numpy()
    # Precomputar para ventanas W = 8, 10, 15 (strictly prior games < t)
    WINDOWS = (8, 10, 15)
    MIN_ANT = 15

    log("calculando distancias de huella y WR previos estrictamente causales (antes de t)...")
    # Para cada W: dict (match_id, player_id) -> (d_estilo, wr_rec, wr_ant, delta_wr, role_shift)
    senales = {w: {} for w in WINDOWS}
    for pid, ix in tl.groupby("player_id", sort=False).indices.items():
        if len(ix) < min(WINDOWS) + MIN_ANT + 1 or pid not in emb_of:
            continue
        mids_p = M_tl[ix]
        # Posiciones dentro de ix que pertenecen a eval_mids
        ev_pos = [j for j, m in enumerate(mids_p) if m in eval_mids]
        if not ev_pos:
            continue
        f = emb_of[pid]
        Xp = X_tl[ix]
        Wp = W_tl[ix]
        Lp = L_tl[ix]
        cs_X = np.vstack([np.zeros((1, Xp.shape[1]), dtype=np.float32), np.cumsum(Xp, axis=0)])
        cs_W = np.concatenate([[0], np.cumsum(Wp)])

        # Reunimos todos los vectores (rec, ant) de este jugador para llamar a la red en un solo batch
        vecs, meta_keys = [], []
        for w in WINDOWS:
            for j in ev_pos:
                if j < w + MIN_ANT:
                    continue
                n_ant = j - w
                m_rec = (cs_X[j] - cs_X[j - w]) / w
                m_ant = cs_X[j - w] / n_ant
                wr_rec = (cs_W[j] - cs_W[j - w]) / w
                wr_ant = cs_W[j - w] / n_ant
                # Rol principal en el historial anterior
                vals, cnts = np.unique(Lp[: j - w], return_counts=True)
                main_ant = vals[int(np.argmax(cnts))]
                role_shift = 1.0 - float((Lp[j - w: j] == main_ant).mean())
                vecs.append(m_rec)
                vecs.append(m_ant)
                meta_keys.append((w, mids_p[j], pid, float(wr_rec), float(wr_ant), role_shift))
        if not vecs:
            continue
        E = f(np.stack(vecs))
        dists = np.linalg.norm(E[0::2] - E[1::2], axis=1)
        for dist_val, (w, mid, p_, wr_r, wr_a, rsh) in zip(dists, meta_keys):
            senales[w][(mid, p_)] = (float(dist_val), wr_r, wr_a, wr_r - wr_a, rsh)

    # 3. Ajustar modelos de victoria fuera de muestra (con mapa y sin mapa)
    log("ajustando modelos de victoria fuera de muestra (A2, A3, A4, V)...")
    mods_con = ajustar_modelos_bloques(D, con_mapa=True)
    mods_sin = ajustar_modelos_bloques(D, con_mapa=False)

    # Mapear índices de `df_tr` por player_id para reconstruir el perfil solo con las últimas W previas
    tr_by_player = df_tr.groupby("player_id", sort=False).indices
    row_of_mp = {(m, p): i for i, (m, p) in enumerate(zip(df_tr["match_id"].to_numpy(), df_tr["player_id"].to_numpy()))}
    # Posición de cada fila dentro de la lista ordenada del jugador
    pos_in_player = np.zeros(len(df_tr), dtype=int)
    for ix in tr_by_player.values():
        pos_in_player[ix] = np.arange(len(ix))

    wins_arr = df_tr["win"].to_numpy(int)
    champs_arr = df_tr["champion_id"].to_numpy(int)
    lanes_arr = df_tr["lane"].to_numpy()
    mids_arr = D["Z"]["mids"]
    players_3d = D["B"]["player"]

    # Predicciones base en todas las partidas de evaluación
    pred_base_con = np.full(D["NM"], np.nan)
    pred_base_sin = np.full(D["NM"], np.nan)
    bloque_de = {}
    for nombre_b, (idxs, minfo) in mods_con.items():
        pred_base_con[idxs] = predecir_lote(D, minfo, idxs)
        pred_base_sin[idxs] = predecir_lote(D, mods_sin[nombre_b][1], idxs)
        for m_i in idxs:
            bloque_de[int(m_i)] = nombre_b

    # 4. Evaluar reemplazo causal para cada ventana W y régimen (boosting vs racha buena vs cambio sin WR)
    resumen = []
    for w in WINDOWS:
        sen = senales[w]
        all_d = np.array([v[0] for v in sen.values()])
        p50_d, p90_d, p95_d = np.quantile(all_d, [0.50, 0.90, 0.95])
        log(f"\n=== Ventana causal W = {w} (n = {len(sen):,} jugador-partidas) | d_estilo p50={p50_d:.3f} p90={p90_d:.3f} p95={p95_d:.3f} ===")

        regimenes = {
            "Sospecha boosting (p90 & WR≥70% & Δ≥20pp)": lambda d, wr, wa, dw, rs: (d >= p90_d) and (wr >= 0.70) and (dw >= 0.20),
            "Sospecha boosting estricto (p95 & WR≥70% & Δ≥20pp)": lambda d, wr, wa, dw, rs: (d >= p95_d) and (wr >= 0.70) and (dw >= 0.20),
            "Sospecha boosting + cambio rol (p90 & WR≥70% & rol≥50%)": lambda d, wr, wa, dw, rs: (d >= p90_d) and (wr >= 0.70) and (dw >= 0.20) and (rs >= 0.50),
            "Control: Racha buena (d<p50 & WR≥70% & Δ≥20pp)": lambda d, wr, wa, dw, rs: (d < p50_d) and (wr >= 0.70) and (dw >= 0.20),
            "Control: Cambio huella sin WR alto (p90 & |Δ|<12pp)": lambda d, wr, wa, dw, rs: (d >= p90_d) and (abs(dw) < 0.12),
            "Control: Normal (d<p50 & |Δ|<12pp)": lambda d, wr, wa, dw, rs: (d < p50_d) and (abs(dw) < 0.12),
        }

        for rname, cond in regimenes.items():
            # Recolectar (match_idx, team_idx, lane_idx, pid, meta)
            hits = []
            for m_i, bname in bloque_de.items():
                mid = mids_arr[m_i]
                for ti in (0, 1):
                    for li in range(5):
                        pid = players_3d[m_i, ti, li]
                        v = sen.get((mid, pid))
                        if v is None:
                            continue
                        if cond(*v):
                            hits.append((m_i, ti, li, pid, bname, v))
            if len(hits) < 10:
                log(f"  {rname}: n={len(hits)} (muy pocos)")
                continue

            # Para cada hit, construimos la fila modificada de `Xj` reemplazando SOLO a ese jugador
            # (si hay >1 hit en la misma partida, se agrupan por partida para evaluar a nivel partida)
            by_match = {}
            for m_i, ti, li, pid, bname, v in hits:
                by_match.setdefault(m_i, []).append((ti, li, pid, bname, v))

            m_idxs = sorted(by_match)
            # Evaluar en el modelo CON mapa y SIN mapa
            p_old_c, p_new_c, p_old_s, p_new_s = [], [], [], []
            y_match = []
            # Persistencia del WR del jugador disparado en la partida futura t
            wr_futuro_jug, p_jug_old_c, p_jug_new_c, p_jug_old_s, p_jug_new_s = [], [], [], [], []

            for m_i in m_idxs:
                bname = bloque_de[m_i]
                mid = mids_arr[m_i]
                xj_mod = D["Xj"][m_i].copy()
                for ti, li, pid, _, _ in by_match[m_i]:
                    row_i = row_of_mp[(mid, pid)]
                    p_pos = pos_in_player[row_i]
                    ix_all_p = tr_by_player[pid]
                    ix_prev = ix_all_p[max(0, p_pos - w): p_pos]
                    row_cur = {"champion_id": champs_arr[row_i], "lane": lanes_arr[row_i]}
                    xj_mod[ti, li] = perfil_desde_indices(
                        ix_prev, row_cur, R, V, wins_arr, champs_arr, lanes_arr, xj_mod[ti, li], col_idx
                    )
                # Actualización exacta del logit: Δlogit = ((Xd_mod - Xd_orig) / sd) @ w_xd
                d_xd = ((xj_mod[0] - xj_mod[1]).reshape(-1) - D["Xd"][m_i]).astype(np.float64)

                def pred_one(minfo, p_old):
                    logit_old = np.log(np.clip(p_old, 1e-12, 1 - 1e-12) / np.clip(1 - p_old, 1e-12, 1 - 1e-12))
                    logit_new = logit_old + float((d_xd / minfo["sd"]) @ minfo["w_xd"])
                    return float(1.0 / (1.0 + np.exp(-logit_new)))

                po_c = float(pred_base_con[m_i])
                po_s = float(pred_base_sin[m_i])
                pn_c = pred_one(mods_con[bname][1], po_c)
                pn_s = pred_one(mods_sin[bname][1], po_s)
                ym = int(D["y"][m_i])

                p_old_c.append(po_c); p_new_c.append(pn_c)
                p_old_s.append(po_s); p_new_s.append(pn_s)
                y_match.append(ym)

                for ti, li, pid, _, _ in by_match[m_i]:
                    # Desde la perspectiva del equipo del jugador disparado
                    won = ym if ti == 0 else (1 - ym)
                    wr_futuro_jug.append(won)
                    p_jug_old_c.append(po_c if ti == 0 else 1 - po_c)
                    p_jug_new_c.append(pn_c if ti == 0 else 1 - pn_c)
                    p_jug_old_s.append(po_s if ti == 0 else 1 - po_s)
                    p_jug_new_s.append(pn_s if ti == 0 else 1 - pn_s)

            y_match = np.array(y_match)
            p_old_ca, p_new_ca = np.array(p_old_c), np.array(p_new_c)
            p_old_sa, p_new_sa = np.array(p_old_s), np.array(p_new_s)
            res_c = boot_delta(y_match, p_new_ca, p_old_ca)
            res_s = boot_delta(y_match, p_new_sa, p_old_sa)
            is_v = np.array([bloque_de[m_i] == "V" for m_i in m_idxs])
            res_v = boot_delta(y_match[is_v], p_new_ca[is_v], p_old_ca[is_v]) if is_v.sum() >= 20 else {}
            res_a = boot_delta(y_match[~is_v], p_new_ca[~is_v], p_old_ca[~is_v]) if (~is_v).sum() >= 20 else {}
            wr_fut = float(np.mean(wr_futuro_jug))
            dp_c = float(np.mean(p_jug_new_c) - np.mean(p_jug_old_c))

            log(f"  {rname}")
            log(f"    n={len(hits):4d} ({len(m_idxs):4d} part.: A2-4={(~is_v).sum()}, V={is_v.sum()}) | WR futuro real = {wr_fut:.1%} "
                f"| P(equipo jug) base→reset: {np.mean(p_jug_old_c):.1%}→{np.mean(p_jug_new_c):.1%} ({dp_c:+.1%})")
            log(f"    [Con mapa] LL {res_c['ll_old']:.4f}→{res_c['ll_new']:.4f} (Δ {res_c['delta_ll']:+.4f} "
                f"[{res_c['delta_ll_lo']:+.4f}; {res_c['delta_ll_hi']:+.4f}]) | "
                f"Acc {res_c['acc_old']:.1%}→{res_c['acc_new']:.1%} | AUC {res_c['auc_old']:.4f}→{res_c['auc_new']:.4f}")
            log(f"    [Sin mapa] LL {res_s['ll_old']:.4f}→{res_s['ll_new']:.4f} (Δ {res_s['delta_ll']:+.4f} "
                f"[{res_s['delta_ll_lo']:+.4f}; {res_s['delta_ll_hi']:+.4f}]) | "
                f"Acc {res_s['acc_old']:.1%}→{res_s['acc_new']:.1%} | AUC {res_s['auc_old']:.4f}→{res_s['auc_new']:.4f}")

            resumen.append({
                "ventana_W": w, "regimen": rname, "n_jug_part": len(hits), "n_partidas": len(m_idxs),
                "n_a24": int((~is_v).sum()), "n_v": int(is_v.sum()),
                "wr_futuro_real": wr_fut,
                "p_jug_base_con": float(np.mean(p_jug_old_c)), "p_jug_reset_con": float(np.mean(p_jug_new_c)),
                **{f"con_{k}": v for k, v in res_c.items()},
                **{f"sin_{k}": v for k, v in res_s.items()},
                "a24_con_delta_ll": res_a.get("delta_ll", np.nan),
                "a24_con_delta_auc": (res_a["auc_new"] - res_a["auc_old"]) if res_a else np.nan,
                "v_con_delta_ll": res_v.get("delta_ll", np.nan),
                "v_con_delta_auc": (res_v["auc_new"] - res_v["auc_old"]) if res_v else np.nan,
            })

    # 4b. Ablación por grupo de variables en W = 15 e impacto global sobre todo V Master+
    log("\n=== Ablación por grupo de variables (W = 15, con mapa) ===")
    w_ab = 15
    sen15 = senales[w_ab]
    all_d15 = np.array([v[0] for v in sen15.values()])
    p50_15, p90_15 = np.quantile(all_d15, [0.50, 0.90])
    margen_cols = [f"{p}_{s}" for p in ("acum", "v20") for s in STATS_MARGEN]
    grupos_ab = {
        "Solo margen (acum + v20)": margen_cols,
        "Solo WR (wr_acum + wr_v20)": ["wr_acum", "wr_v20"],
        "Solo rol (share_lane + offrole)": ["share_lane", "offrole"],
        "Solo campeón (logn_champ, estreno, wr_champ, share_champ)": ["logn_champ", "estreno", "wr_champ", "share_champ"],
        "Reset limpio (sin castigar logn_champ/logn_tot)": margen_cols + ["share_lane", "offrole", "share_champ", "wr_champ", "wr_acum", "wr_v20"],
        "Reset total": margen_cols + ["wr_acum", "wr_v20", "logn_tot", "logn_champ", "estreno", "wr_champ", "share_champ", "share_lane", "offrole"],
    }
    regs_ab = [
        ("Sospecha boosting (p90 & WR≥70% & Δ≥20pp)", lambda d, wr, wa, dw, rs: d >= p90_15 and wr >= 0.70 and dw >= 0.20),
        ("Control: Racha buena (d<p50 & WR≥70% & Δ≥20pp)", lambda d, wr, wa, dw, rs: d < p50_15 and wr >= 0.70 and dw >= 0.20),
        ("Control: Cambio huella sin WR alto (p90 & |Δ|<12pp)", lambda d, wr, wa, dw, rs: d >= p90_15 and abs(dw) < 0.12),
        ("Control: Normal (d<p50 & |Δ|<12pp)", lambda d, wr, wa, dw, rs: d < p50_15 and abs(dw) < 0.12),
    ]
    abl_rows = []
    for rname, cond in regs_ab:
        by_m = {}
        for m_i in bloque_de:
            mid = mids_arr[m_i]
            for ti in (0, 1):
                for li in range(5):
                    pid = players_3d[m_i, ti, li]
                    v = sen15.get((mid, pid))
                    if v and cond(*v):
                        by_m.setdefault(m_i, []).append((ti, li, pid))
        m_idxs = sorted(by_m)
        y_m = D["y"][m_idxs]
        p_old = pred_base_con[m_idxs]
        # Precalcular el vector reseteado completo una sola vez por (partida, jugador)
        resets_cache = {}
        for m_i in m_idxs:
            mid = mids_arr[m_i]
            for ti, li, pid in by_m[m_i]:
                row_i = row_of_mp[(mid, pid)]
                p_pos = pos_in_player[row_i]
                ix_prev = tr_by_player[pid][max(0, p_pos - w_ab): p_pos]
                resets_cache[(m_i, ti, li)] = perfil_desde_indices(
                    ix_prev, {"champion_id": champs_arr[row_i], "lane": lanes_arr[row_i]},
                    R, V, wins_arr, champs_arr, lanes_arr, D["Xj"][m_i, ti, li], col_idx,
                )
        for gname, gcols in grupos_ab.items():
            g_idx = [col_idx[c] for c in gcols]
            p_new = np.zeros(len(m_idxs))
            dp_j = []
            for k, m_i in enumerate(m_idxs):
                bname = bloque_de[m_i]
                xj_mod = D["Xj"][m_i].copy()
                for ti, li, _ in by_m[m_i]:
                    xj_mod[ti, li, g_idx] = resets_cache[(m_i, ti, li)][g_idx]
                d_xd = ((xj_mod[0] - xj_mod[1]).reshape(-1) - D["Xd"][m_i]).astype(np.float64)
                minfo = mods_con[bname][1]
                po = float(p_old[k])
                lo = np.log(np.clip(po, 1e-12, 1 - 1e-12) / np.clip(1 - po, 1e-12, 1 - 1e-12))
                pn = float(1.0 / (1.0 + np.exp(-(lo + float((d_xd / minfo["sd"]) @ minfo["w_xd"])))))
                p_new[k] = pn
                for ti, _, _ in by_m[m_i]:
                    dp_j.append((pn - po) if ti == 0 else ((1 - pn) - (1 - po)))
            rt = boot_delta(y_m, p_new, p_old)
            abl_rows.append({
                "regimen": rname, "grupo": gname, "n_partidas": len(m_idxs),
                "delta_p_jug": float(np.mean(dp_j)),
                "delta_ll": rt["delta_ll"], "delta_ll_lo": rt["delta_ll_lo"], "delta_ll_hi": rt["delta_ll_hi"],
                "delta_auc": rt["auc_new"] - rt["auc_old"],
                "acc_old": rt["acc_old"], "acc_new": rt["acc_new"],
            })

    # Impacto global en todo V Master+ (n = 16.428) al aplicar la regla W = 15 de sospecha de boosting
    v_idxs = mods_con["V"][0]
    p_v_old = pred_base_con[v_idxs].copy()
    p_v_new = p_v_old.copy()
    minfo_v = mods_con["V"][1]
    n_mod_v = 0
    for pos_k, m_i in enumerate(v_idxs):
        mid = mids_arr[m_i]
        hits_m = []
        for ti in (0, 1):
            for li in range(5):
                pid = players_3d[m_i, ti, li]
                v = sen15.get((mid, pid))
                if v and v[0] >= p90_15 and v[1] >= 0.70 and v[3] >= 0.20:
                    hits_m.append((ti, li, pid))
        if not hits_m:
            continue
        n_mod_v += 1
        xj_mod = D["Xj"][m_i].copy()
        for ti, li, pid in hits_m:
            row_i = row_of_mp[(mid, pid)]
            p_pos = pos_in_player[row_i]
            ix_prev = tr_by_player[pid][max(0, p_pos - w_ab): p_pos]
            xj_mod[ti, li] = perfil_desde_indices(
                ix_prev, {"champion_id": champs_arr[row_i], "lane": lanes_arr[row_i]},
                R, V, wins_arr, champs_arr, lanes_arr, xj_mod[ti, li], col_idx,
            )
        d_xd = ((xj_mod[0] - xj_mod[1]).reshape(-1) - D["Xd"][m_i]).astype(np.float64)
        po = float(p_v_old[pos_k])
        lo = np.log(np.clip(po, 1e-12, 1 - 1e-12) / np.clip(1 - po, 1e-12, 1 - 1e-12))
        p_v_new[pos_k] = float(1.0 / (1.0 + np.exp(-(lo + float((d_xd / minfo_v["sd"]) @ minfo_v["w_xd"])))))
    rg_v = boot_delta(D["y"][v_idxs], p_v_new, p_v_old)
    global_v = {"n_total": int(len(v_idxs)), "n_tocadas": int(n_mod_v), "pct_tocadas": n_mod_v / len(v_idxs), **rg_v}
    log(f"  Global V Master+ (n={len(v_idxs):,}, tocadas={n_mod_v}): ΔLL={rg_v['delta_ll']:+.5f} "
        f"[{rg_v['delta_ll_lo']:+.5f};{rg_v['delta_ll_hi']:+.5f}] | AUC {rg_v['auc_old']:.5f}→{rg_v['auc_new']:.5f}")

    # 5. Estudio de caso: las 8 candidatas reales de la Sección 5 (walk-forward dentro de su tramo boosteado)
    log("\n=== Estudio de caso: 8 candidatas reales de la Sección 5 (partidas dentro de A2-A4 + V) ===")
    pc = pd.read_csv(SALIDA / f"patrones_cuentas_todos_roles{sufijo}.csv")
    p95_c, p90_c = pc["d_estilo"].quantile(0.95), pc["silhouette"].quantile(0.90)
    cands8 = set(pc[(pc["d_estilo"] >= p95_c) & (pc["silhouette"] >= p90_c)
                    & (pc["delta_wr"] >= 0.15) & (pc["p_wr_gt"] < 0.05)]["player_id"])
    # Para cada una de las 8 cuentas, reconstruimos qué partidas pertenecieron a su subgrupo boosteado (bloques de 5)
    from scripts.analysis.huella_patrones_cuenta import kmeans_2_min_size
    boost_mids_of = {}
    for pid in cands8:
        sub = tl[tl["player_id"] == pid].sort_values("game_creation").reset_index(drop=True)
        n_b = len(sub) // 5
        b_idx = [np.arange(k * 5, (k + 1) * 5) for k in range(n_b)]
        E = emb_of[pid](np.stack([sub.loc[bi, FEATS_TL].to_numpy(np.float32).mean(0) for bi in b_idx]))
        lbl = kmeans_2_min_size(E, min_blocks=2, seed=0)
        if (lbl == 1).sum() > (lbl == 0).sum():
            lbl = 1 - lbl
        alt_mids = set(np.concatenate([sub.loc[b_idx[k], "match_id"].to_numpy() for k in range(n_b) if lbl[k] == 1]))
        boost_mids_of[pid] = alt_mids

    # En cada partida de A2-A4 + V perteneciente al tramo boosteado de una de las 8 cuentas (con >= 5 previas en el tramo),
    # comparamos predicción base (historial entero de la cuenta) vs solo previas dentro del tramo boosteado (estrictamente < t)
    caso_rows = []
    for m_i, bname in sorted(bloque_de.items()):
        mid = mids_arr[m_i]
        for ti in (0, 1):
            for li in range(5):
                pid = players_3d[m_i, ti, li]
                if pid not in boost_mids_of or mid not in boost_mids_of[pid]:
                    continue
                row_i = row_of_mp[(mid, pid)]
                p_pos = pos_in_player[row_i]
                ix_all_p = tr_by_player[pid]
                ix_prev_all = ix_all_p[:p_pos]
                # Filtrar SOLO previas estrictamente < t que pertenecen al tramo boosteado
                mids_prev = df_tr["match_id"].to_numpy()[ix_prev_all]
                ix_prev_boost = ix_prev_all[np.isin(mids_prev, list(boost_mids_of[pid]))]
                if len(ix_prev_boost) < 5:
                    continue
                xj_mod = D["Xj"][m_i].copy()
                row_cur = {"champion_id": champs_arr[row_i], "lane": lanes_arr[row_i]}
                xj_mod[ti, li] = perfil_desde_indices(
                    ix_prev_boost, row_cur, R, V, wins_arr, champs_arr, lanes_arr, xj_mod[ti, li], col_idx
                )
                d_xd = ((xj_mod[0] - xj_mod[1]).reshape(-1) - D["Xd"][m_i]).astype(np.float64)

                def pred_one(minfo, p_old):
                    logit_old = np.log(np.clip(p_old, 1e-12, 1 - 1e-12) / np.clip(1 - p_old, 1e-12, 1 - 1e-12))
                    logit_new = logit_old + float((d_xd / minfo["sd"]) @ minfo["w_xd"])
                    return float(1.0 / (1.0 + np.exp(-logit_new)))

                po_c, po_s = float(pred_base_con[m_i]), float(pred_base_sin[m_i])
                pn_c, pn_s = pred_one(mods_con[bname][1], po_c), pred_one(mods_sin[bname][1], po_s)
                ym = int(D["y"][m_i])
                won = ym if ti == 0 else (1 - ym)
                caso_rows.append({
                    "pid": pid, "match_id": mid, "bloque": bname, "n_prev_boost": len(ix_prev_boost),
                    "n_prev_total": len(ix_prev_all), "y": ym, "won_jug": won,
                    "p_old_con": po_c, "p_new_con": pn_c, "p_old_sin": po_s, "p_new_sin": pn_s,
                    "p_jug_old_con": po_c if ti == 0 else 1 - po_c,
                    "p_jug_new_con": pn_c if ti == 0 else 1 - pn_c,
                    "p_jug_old_sin": po_s if ti == 0 else 1 - po_s,
                    "p_jug_new_sin": pn_s if ti == 0 else 1 - pn_s,
                })

    cdf = pd.DataFrame(caso_rows)
    if len(cdf):
        rc = boot_delta(cdf["y"].to_numpy(), cdf["p_new_con"].to_numpy(), cdf["p_old_con"].to_numpy())
        rs = boot_delta(cdf["y"].to_numpy(), cdf["p_new_sin"].to_numpy(), cdf["p_old_sin"].to_numpy())
        log(f"  n = {len(cdf)} partidas en tramos boosteados (de {cdf['pid'].nunique()} cuentas) | "
            f"WR real = {cdf['won_jug'].mean():.1%}")
        log(f"  P(equipo jug) [Con mapa]: {cdf['p_jug_old_con'].mean():.1%} → {cdf['p_jug_new_con'].mean():.1%} "
            f"| LL {rc['ll_old']:.4f} → {rc['ll_new']:.4f} (Δ {rc['delta_ll']:+.4f} "
            f"[{rc['delta_ll_lo']:+.4f}; {rc['delta_ll_hi']:+.4f}]) | Acc {rc['acc_old']:.1%} → {rc['acc_new']:.1%}")
        log(f"  P(equipo jug) [Sin mapa]: {cdf['p_jug_old_sin'].mean():.1%} → {cdf['p_jug_new_sin'].mean():.1%} "
            f"| LL {rs['ll_old']:.4f} → {rs['ll_new']:.4f} (Δ {rs['delta_ll']:+.4f} "
            f"[{rs['delta_ll_lo']:+.4f}; {rs['delta_ll_hi']:+.4f}]) | Acc {rs['acc_old']:.1%} → {rs['acc_new']:.1%}")

    pd.DataFrame(resumen).to_csv(SALIDA / f"boosting_en_victoria{sufijo}.csv", index=False)
    pd.DataFrame(abl_rows).to_csv(SALIDA / f"boosting_en_victoria_ablacion{sufijo}.csv", index=False)
    (SALIDA / f"boosting_en_victoria_global_v{sufijo}.json").write_text(json.dumps(global_v, indent=2), encoding="utf-8")
    if len(cdf):
        cdf.to_csv(SALIDA / f"boosting_en_victoria_casos8{sufijo}.csv", index=False)
    log(f"listo: boosting_en_victoria{sufijo}.csv (+ ablacion, global_v, casos8)")


if __name__ == "__main__":
    main()
