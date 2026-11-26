"""
Firmas de jugador por ponderación de embeddings de campeón-línea.

Variante A (macro): firma global por jugador = media ponderada por frecuencia
    de emb(champ, lane) sobre todo su historial + pool_dispersion
    (reemplaza a unique_champions_count: mide amplitud Y diversidad real).
Variante B (posicional): una firma por (jugador, línea) solo con partidas de
    esa línea + flag de elegibilidad (n >= min_lane_games, 10 por defecto:
    solo se promedian los campeones jugados en esa línea con >=10 partidas).

Embedding: v1 gameplay SIN first_blood_rate_pct (22D, ver src/embeddings.ipynb).
Identidad: player_id (componente conexa, ver src/features/identity.py).
Partidas fuera de las 5 líneas canónicas se excluyen.

Uso:
    python player_signatures.py
    python player_signatures.py --full data/full_dataset.csv --out-dir data
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

CANONICAL_LANES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
MIN_PARTIDAS_EMB = 100
MIN_LANE_GAMES = 10

# v1 gameplay menos first_blood_rate_pct (ruidosa por consenso ANOVA+ablación)
EMB_FEATURES = [
    "avg_kills", "avg_deaths", "avg_assists", "avg_kda",
    "avg_cs_at_10", "avg_gold_at_10", "avg_xp_at_10", "avg_damage_at_10",
    "avg_cs_at_15", "avg_gold_at_15", "avg_xp_at_15", "avg_damage_at_15",
    "avg_cs_at_20", "avg_gold_at_20", "avg_xp_at_20", "avg_damage_at_20",
    "avg_total_damage", "avg_damage_taken", "avg_gold_earned", "avg_total_cs",
    "avg_vision_score", "first_tower_rate_pct",
]


def build_champion_embeddings(
    champions_df: pd.DataFrame,
) -> Tuple[Dict[Tuple[str, str], np.ndarray], Dict[str, np.ndarray], StandardScaler]:
    """Vectores emb(champ, lane) + vectores de línea (media). Solo pares con >=100 part."""
    fil = champions_df[champions_df["games_played"] >= MIN_PARTIDAS_EMB].reset_index(drop=True)
    fil = fil[fil["lane"].isin(CANONICAL_LANES)].reset_index(drop=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(fil[EMB_FEATURES])
    vecs = {(r["champion_name"], r["lane"]): X[i] for i, r in fil.iterrows()}
    lanes = {lane: X[(fil["lane"] == lane).values].mean(axis=0)
             for lane in CANONICAL_LANES if (fil["lane"] == lane).any()}
    return vecs, lanes, scaler


def _signatures_for_groups(games: pd.DataFrame, vecs: Dict,
                           by: List[str]) -> pd.DataFrame:
    """Media ponderada por frecuencia + dispersión del pool por grupo."""
    games = games.copy()
    games["_vec"] = games.apply(
        lambda r: vecs.get((r["champion_name"], r["lane"])), axis=1)
    games = games[games["_vec"].notna()].reset_index(drop=True)
    if games.empty:
        return pd.DataFrame()
    V = np.stack(games["_vec"].values)
    out_rows = []
    for keys, idx in games.groupby(by).indices.items():
        M = V[idx]
        centroid = M.mean(axis=0)
        disp = float(np.linalg.norm(M - centroid, axis=1).mean())
        row = {"n_games": len(idx), "pool_dispersion": round(disp, 4)}
        for j in range(M.shape[1]):
            row[f"sig_{j}"] = round(float(centroid[j]), 6)
        if isinstance(keys, tuple):
            for k, v in zip(by, keys):
                row[k] = v
        else:
            row[by[0]] = keys
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def signature_a(games: pd.DataFrame, vecs: Dict) -> pd.DataFrame:
    """Variante A: una firma por jugador (todo el historial)."""
    g = games[games["lane"].isin(CANONICAL_LANES)].reset_index(drop=True)
    return _signatures_for_groups(g, vecs, by=["player_id"])


def signature_b(games: pd.DataFrame, vecs: Dict,
                min_lane_games: int = MIN_LANE_GAMES) -> pd.DataFrame:
    """Variante B: una firma por (jugador, línea) + elegibilidad."""
    g = games[games["lane"].isin(CANONICAL_LANES)].reset_index(drop=True)
    g["_lane_key"] = g["lane"]
    df = _signatures_for_groups(g, vecs, by=["player_id", "_lane_key"])
    if df.empty:
        return df
    df = df.rename(columns={"_lane_key": "lane"})
    df["eligible"] = df["n_games"] >= min_lane_games
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Firmas de jugador desde embeddings (A+B).")
    parser.add_argument("--full", type=str, default="data/full_dataset.csv")
    parser.add_argument("--champions", type=str, default="data/champions_by_lane.csv")
    parser.add_argument("--out-dir", type=str, default="data")
    parser.add_argument("--min-lane-games", type=int, default=MIN_LANE_GAMES)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    print("[*] Cargando datasets...")
    champs = pd.read_csv(args.champions)
    vecs, lanes, _ = build_champion_embeddings(champs)
    print(f"[*] Embeddings: {len(vecs)} pares (champ,línea), {len(EMB_FEATURES)}D "
          f"(sin first_blood_rate_pct).")
    games = pd.read_csv(args.full, usecols=["player_id", "champion_name", "lane"])
    print(f"[*] Partidas: {len(games):,}")

    sig_a = signature_a(games, vecs)
    sig_a.to_csv(out_dir / "player_signatures_A.csv", index=False)
    print(f"[+] Variante A: {len(sig_a):,} firmas -> player_signatures_A.csv")

    sig_b = signature_b(games, vecs, min_lane_games=args.min_lane_games)
    sig_b.to_csv(out_dir / "player_signatures_B.csv", index=False)
    elig = int(sig_b["eligible"].sum()) if not sig_b.empty else 0
    print(f"[+] Variante B: {len(sig_b):,} firmas ({elig} elegibles con "
          f"n>={args.min_lane_games}) -> player_signatures_B.csv")


if __name__ == "__main__":
    main()
