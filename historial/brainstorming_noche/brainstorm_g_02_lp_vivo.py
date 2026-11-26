"""Idea 5 del brainstorming (docs/BRAINSTORM_NOCHE.md): señal predictiva real en `lp_vivo.db`.

Evalúa sobre las partidas Master+ en vivo de 4 servidores (LA2, BR1, NA1, EUW1) cuánto predicen:
* LP real previo a la partida (`lp_antes`, escala absoluta de `rating.lp_absoluto`);
* volumen de temporada (`log1p(W + L)`), winrate previo (`wr_vivo`), `W - L` y `Master 0 LP`;
* asimetría de LP por equipo (`max`, `min`, `std`, por rol) y racha causal previa (`t_despues < game_start`).

Nota de esquema: `lp_vivo.db` (`estado` y `eventos`) NO guarda la columna `hotStreak`; se reconstruye
una racha causal equivalente desde los eventos previos ya cerrados.

Solo lectura de la base externa; corte temporal 70 % -> 30 % dentro de `lp_vivo.db` (no toca `V` ni `test`).
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.features.rating import lp_absoluto  # noqa: E402

RUTA_DEFECTO = Path(r"G:\Mi unidad\Lol Database\data_completa\lp_vivo\lp_vivo.db")
ROLES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")


def cargar_vivo(ruta: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with sqlite3.connect(f"file:{ruta}?mode=ro", uri=True) as con:
        partidas = pd.read_sql_query(
            "SELECT match_id, platform, game_creation, game_start, game_end, duracion_s FROM partidas",
            con,
        )
        part = pd.read_sql_query(
            "SELECT match_id, puuid, team_id, win, team_position FROM participantes",
            con,
        )
        evs = pd.read_sql_query(
            "SELECT platform, puuid, t_antes, t_despues, tier_antes, rank_antes, lp_antes, "
            "wins_antes, losses_antes, wins_despues, losses_despues, tipo, estado, match_id "
            "FROM eventos",
            con,
        )
    return partidas, part, evs


def preparar_dataset(partidas: pd.DataFrame, part: pd.DataFrame, evs: pd.DataFrame) -> pd.DataFrame:
    # 1. Estado previo exacto por (match_id, puuid) desde eventos enlazados
    enl = evs[evs["estado"] == "enlazada"].copy()
    enl = enl.drop_duplicates(subset=["match_id", "puuid"], keep="first")
    enl["lp_abs"] = [lp_absoluto(t, r, lp) for t, r, lp in zip(enl["tier_antes"], enl["rank_antes"], enl["lp_antes"])]
    enl["tot_antes"] = enl["wins_antes"].astype(float) + enl["losses_antes"].astype(float)
    enl["wr_antes"] = (enl["wins_antes"].astype(float) + 10.0 * 0.5) / (enl["tot_antes"] + 20.0)
    enl["logn_antes"] = np.log1p(enl["tot_antes"])
    enl["netw_antes"] = enl["wins_antes"].astype(float) - enl["losses_antes"].astype(float)
    enl["cero_m"] = ((enl["tier_antes"] == "MASTER") & (enl["lp_antes"] == 0)).astype(float)

    # 2. Racha causal estrictamente anterior al inicio de cada partida (desde eventos previos cerrados)
    hist = evs[(evs["tipo"] == "partida") & (evs["wins_despues"] - evs["wins_antes"] + evs["losses_despues"] - evs["losses_antes"] == 1)].copy()
    hist["g_ev"] = (hist["wins_despues"] - hist["wins_antes"]).astype(int)
    hist = hist.sort_values(["puuid", "t_despues"], kind="mergesort")

    # Calcular saldo últimas 3 y 5 partidas antes de cada match_id usando timestamps
    df = part.merge(partidas, on="match_id", how="inner")
    df = df.merge(
        enl[["match_id", "puuid", "lp_abs", "tot_antes", "wr_antes", "logn_antes", "netw_antes", "cero_m", "t_antes"]],
        on=["match_id", "puuid"],
        how="left",
    )

    # Racha causal: unir solo eventos cerrados del mismo jugador con t_despues < game_start
    ev_por_p = {
        p: (g["t_despues"].to_numpy(np.int64), g["g_ev"].to_numpy(np.int8))
        for p, g in hist.groupby("puuid", sort=False)
    }
    racha3 = np.zeros(len(df), np.float32)
    jug_hist_n = np.zeros(len(df), np.float32)
    puuids = df["puuid"].to_numpy()
    starts = df["game_start"].to_numpy(np.int64)
    for i, (p, gs) in enumerate(zip(puuids, starts)):
        item = ev_por_p.get(p)
        if item is None:
            continue
        ts, gs_arr = item
        k = int(np.searchsorted(ts, gs, side="left"))
        if k > 0:
            jug_hist_n[i] = float(k)
            ult = gs_arr[max(0, k - 3) : k]
            racha3[i] = float((ult * 2 - 1).sum())
    df["racha3"] = racha3
    df["n_hist_vivo"] = jug_hist_n
    return df


def matriz_partidas(df: pd.DataFrame, min_conocidos: int = 8) -> tuple[pd.DataFrame, dict]:
    # Filtrar partidas 5v5 limpias con al menos `min_conocidos` jugadores enlazados al snapshot previo
    cnt = df.groupby("match_id")["lp_abs"].agg(conocidos="count", n="size")
    ok_ids = set(cnt[(cnt["n"] == 10) & (cnt["conocidos"] >= min_conocidos)].index)
    sub = df[df["match_id"].isin(ok_ids)].copy()
    sub["tiene_lp"] = sub["lp_abs"].notna().astype(float)

    # Imputar faltantes de ese lobby con la media de los conocidos de SU equipo/partida (sin cruzar el corte futuro)
    lob_mu = sub.groupby("match_id")["lp_abs"].transform("mean")
    sub["lp_imp"] = sub["lp_abs"].fillna(lob_mu)

    rows = []
    feats_lp, feats_wl, feats_disp, feats_racha = [], [], [], []
    # Orden por game_creation -> corte temporal estricto
    sub = sub.sort_values(["game_creation", "match_id", "team_id"], kind="mergesort")
    for mid, g in sub.groupby("match_id", sort=False):
        if len(g) != 10:
            continue
        g1 = g[g["team_id"] == 100]
        g2 = g[g["team_id"] == 200]
        if len(g1) != 5 or len(g2) != 5:
            continue
        win1 = int(g1["win"].iloc[0])
        if win1 + int(g2["win"].iloc[0]) != 1:
            continue

        def diff_mean(col: str) -> float:
            v1 = g1[col].dropna()
            v2 = g2[col].dropna()
            if len(v1) == 0 or len(v2) == 0:
                return 0.0
            return float(v1.mean() - v2.mean())

        lp1, lp2 = g1["lp_imp"].to_numpy(), g2["lp_imp"].to_numpy()
        dlp = float(lp1.mean() - lp2.mean())
        feats_lp.append([dlp, float(g1["tiene_lp"].mean() - g2["tiene_lp"].mean())])

        feats_wl.append([
            diff_mean("wr_antes"),
            diff_mean("logn_antes"),
            diff_mean("netw_antes"),
            diff_mean("cero_m"),
        ])
        feats_disp.append([
            float(lp1.max() - lp2.max()),
            float(lp1.min() - lp2.min()),
            float(lp1.std() - lp2.std()),
        ])
        feats_racha.append([float(g1["racha3"].mean() - g2["racha3"].mean())])
        rows.append({
            "match_id": mid,
            "platform": g["platform"].iloc[0],
            "t": int(g["game_creation"].iloc[0]),
            "y": win1,
            "conocidos": int(g["tiene_lp"].sum()),
            "dlp": dlp,
            "d_logn": feats_wl[-1][1],
            "d_wr": feats_wl[-1][0],
        })

    meta = pd.DataFrame(rows)
    return meta, {
        "lp": np.array(feats_lp, dtype=np.float32),
        "wl": np.array(feats_wl, dtype=np.float32),
        "disp": np.array(feats_disp, dtype=np.float32),
        "racha": np.array(feats_racha, dtype=np.float32),
    }


def eval_split(meta: pd.DataFrame, bloques: dict[str, np.ndarray], nombre: str = "GLOBAL", frac_tr: float = 0.7):
    t = meta["t"].to_numpy()
    corte = np.quantile(t, frac_tr)
    tr = t < corte
    te = ~tr
    y = meta["y"].to_numpy()
    print(f"\n=== {nombre}: train={tr.sum():,}, test={te.sum():,}, known_mean={meta['conocidos'].mean():.2f}, blue_wr={y[te].mean():.4f} ===")

    # Univariados sobre todo el conjunto (o test)
    for col, nom in [("dlp", "LP real (Azul-Rojo)"), ("d_wr", "WR temporada vivo"), ("d_logn", "log(W+L) vivo")]:
        x = meta[col].to_numpy()[te]
        auc = roc_auc_score(y[te], x)
        print(f"  Univariado test | {nom:24s}: AUC={auc:.4f} (dir={1-auc if auc<0.5 else auc:.4f})")

    modelos = {
        "Solo LP real": bloques["lp"],
        "Solo W/L temporada (wr, logN, netW, 0LP)": bloques["wl"],
        "LP real + W/L temporada": np.hstack([bloques["lp"], bloques["wl"]]),
        "LP real + W/L + dispersión (max/min/std)": np.hstack([bloques["lp"], bloques["wl"], bloques["disp"]]),
        "LP real + W/L + dispersión + racha3": np.hstack([bloques["lp"], bloques["wl"], bloques["disp"], bloques["racha"]]),
    }
    p_base = None
    rng = np.random.default_rng(1)
    boot = rng.integers(0, int(te.sum()), size=(400, int(te.sum())))
    for nom, X in modelos.items():
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        lr = LogisticRegression(C=0.2, max_iter=2000).fit((X[tr] - mu) / sd, y[tr])
        p = lr.predict_proba((X[te] - mu) / sd)[:, 1]
        auc = roc_auc_score(y[te], p)
        acc = accuracy_score(y[te], p > 0.5)
        ll = log_loss(y[te], p)
        if p_base is None:
            p_base = p
            ic = "—"
        else:
            d = np.array([roc_auc_score(y[te][s], p[s]) - roc_auc_score(y[te][s], p_base[s]) for s in boot])
            ic = f"{auc - roc_auc_score(y[te], p_base):+.4f} [{np.percentile(d, 2.5):+.4f}; {np.percentile(d, 97.5):+.4f}]"
        print(f"  {nom:42s} | AUC={auc:.4f} | Acc={acc:.3%} | LL={ll:.4f} | Δ vs LP={ic}")


def main():
    ruta = Path(sys.argv[1]) if len(sys.argv) > 1 else RUTA_DEFECTO
    partidas, part, evs = cargar_vivo(ruta)
    df = preparar_dataset(partidas, part, evs)
    print(f"Partidas totales en lp_vivo.db: {len(partidas):,}; participantes enlazados: {df['lp_abs'].notna().sum():,}/{len(df):,} ({df['lp_abs'].notna().mean():.1%})")
    for min_k in (6, 8, 10):
        meta, bloq = matriz_partidas(df, min_conocidos=min_k)
        print(f"Partidas 5v5 con >={min_k}/10 jugadores con LP previo enlazado: {len(meta):,}")
        if min_k == 8:
            eval_split(meta, bloq, f"4 SERVIDORES (>={min_k}/10 enlazados)")
            for plat in ("euw1", "na1", "br1", "la2"):
                m = (meta["platform"] == plat).to_numpy()
                if m.sum() >= 500:
                    eval_split(meta.loc[m].reset_index(drop=True), {k: v[m] for k, v in bloq.items()}, plat.upper())
        elif min_k == 10 and len(meta) >= 500:
            eval_split(meta, bloq, "SOLO 10/10 ENLAZADOS")


if __name__ == "__main__":
    main()
