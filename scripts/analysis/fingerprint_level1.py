#!/usr/bin/env python3
"""
Nivel 1 del protocolo de validación de la huella de estilo (PROJECT_SUMMARY.md §10.2).

Corre sobre el **train** (`data/splits/train/full_dataset.csv`, solo LAS, las 5
líneas canónicas y, por defecto, jugadores Master+ del padrón: `--min-tier`) la prueba de reidentificación de `src/features/fingerprint.py`
para cada combinación de:

* muestra A/B: aleatoria, campeones distintos, temporal;
* k (partidas por muestra): 5, 10, 20;
* representación:
  - `huella`: estadísticas residualizadas por campeón, línea y resultado;
  - `crudo`: las mismas estadísticas sin descontar el campeón (z por línea);
  - `pool`: frecuencia de cada campeón (distancia coseno);
  - `elo_linea`: tier del padrón + reparto de líneas de la muestra;
  - `embedding`: media de emb(campeón, línea) de la muestra, con el embedding
    construido sobre el `champions_by_lane.csv` del train (solo campeones).

Los baselines viven en `src/models/baselines/identificacion.py`.

Con más de `--max-jugadores` jugadores elegibles se toma una muestra al azar (la
matriz de distancias es n × n). Escribe `data/analysis/huella_nivel1.json`.

Además calcula qué parte de la varianza de cada estadística residualizada es del
jugador (`player_variance_share`, ICC(1)) y la escribe en
`data/analysis/huella_reparto_varianza.json`. Con `--solo-reparto` hace solo eso.

Uso:
    python scripts/analysis/fingerprint_level1.py [--max-jugadores 3000] [--semillas 3]
    python scripts/analysis/fingerprint_level1.py --solo-reparto
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.fingerprint import (CANONICAL_LANES, STYLE_FEATURES, TIER_ORDER,  # noqa: E402
                                      add_rates, evaluate, hard_negative_mask, keep_min_tier, make_halves,
                                      player_variance_share, profiles, residualize, standardize)
from src.models.baselines.identificacion import (elo_lane_profiles, embedding_profiles,  # noqa: E402
                                                 load_embeddings, pool_profiles)

USECOLS = ["match_id", "game_creation", "game_duration", "player_id", "team_id", "win", "champion_name", "lane",
           "kills", "deaths", "assists", "kill_participation_pct", "damage_share_pct", "gold_share_pct",
           "total_damage_dealt_to_champions", "total_damage_taken", "gold_earned",
           "total_minions_killed", "neutral_minions_killed", "vision_score", "player_tier",
           "cs_10", "gold_10", "xp_10", "damage_10", "early_aggression_10", "kills_10", "deaths_10",
           "assists_10", "cs_15", "gold_15", "xp_15", "damage_15"]


def load_train(path: Path, min_tier: Optional[str] = "MASTER") -> pd.DataFrame:
    """Train de LAS en las 5 líneas; con `min_tier`, solo jugadores de ese tier del padrón o más."""
    df = pd.read_csv(path, usecols=USECOLS)
    df = df[df["match_id"].str.startswith("LA2_") & df["lane"].isin(CANONICAL_LANES)]
    if min_tier:
        df = keep_min_tier(df, min_tier)
    return add_rates(df).reset_index(drop=True)


def player_info(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("player_id")
    return pd.DataFrame({"player_tier": g["player_tier"].agg(lambda s: s.dropna().mode().iat[0]
                                                               if s.notna().any() else np.nan),
                         "main_lane": g["lane"].agg(lambda s: s.mode().iat[0])})


def subsample(halves: dict, n_max: int, seed: int) -> dict:
    if len(halves) <= n_max:
        return halves
    keep = np.random.default_rng(seed).choice(list(halves), n_max, replace=False)
    return {p: halves[p] for p in keep}


def main() -> None:
    ap = argparse.ArgumentParser(description="Nivel 1: ¿hay huella de estilo?")
    ap.add_argument("--train", default=str(ROOT / "data" / "splits" / "train" / "full_dataset.csv"))
    ap.add_argument("--champions", default=str(ROOT / "data" / "splits" / "train" / "champions_by_lane.csv"),
                    help="tabla de campeón del train para el embedding (sin leakage)")
    ap.add_argument("--out", default=str(ROOT / "data" / "analysis" / "huella_nivel1.json"))
    ap.add_argument("--max-jugadores", type=int, default=3000)
    ap.add_argument("--semillas", type=int, default=3)
    ap.add_argument("--ks", default="5,10,20")
    ap.add_argument("--reparto-out", default=str(ROOT / "data" / "analysis" / "huella_reparto_varianza.json"))
    ap.add_argument("--solo-reparto", action="store_true", help="solo el reparto de varianza (~1 min)")
    ap.add_argument("--min-tier", default="MASTER", choices=TIER_ORDER + ["todos"],
                    help="población: jugadores con este tier del padrón o más (default MASTER)")
    args = ap.parse_args()

    t0 = time.time()
    raw = load_train(Path(args.train), None if args.min_tier == "todos" else args.min_tier)
    print(f"[*] Train: {len(raw):,} filas jugador-partida, {raw['match_id'].nunique():,} partidas, "
          f"{raw['player_id'].nunique():,} jugadores.")
    resid = residualize(raw)
    crudo = standardize(raw).loc[resid.index]        # mismas filas en las dos representaciones
    print(f"[*] Residualizadas: {len(resid):,} filas ({len(resid) / len(raw) * 100:.1f} %; el resto "
          f"cae en grupos campeón-línea-resultado con < 30 partidas o tiene faltantes).")
    share = player_variance_share(resid, min_games=40)
    n_games = resid.groupby("player_id").size()
    rep_out = Path(args.reparto_out)
    rep_out.parent.mkdir(parents=True, exist_ok=True)
    rep_out.write_text(json.dumps({"min_partidas": 40, "jugadores": int((n_games >= 40).sum()),
                                   "icc": share.round(4).to_dict()}, indent=2, ensure_ascii=False),
                       encoding="utf-8")
    print("[*] Parte de la varianza que es del jugador (ICC):")
    for f, v in share.items():
        print(f"    {f:<24} {v:.3f}")
    print(f"[+] {rep_out}")
    if args.solo_reparto:
        return
    info = player_info(resid)
    vecs, lane_vecs = load_embeddings(Path(args.champions))
    print(f"[*] Embedding: {len(vecs)} pares (campeón, línea) desde {args.champions}")

    results = []
    for mode in ["aleatoria", "campeones_distintos", "temporal"]:
        for k in [int(x) for x in args.ks.split(",")]:
            for seed in range(args.semillas if mode != "temporal" else 1):
                halves = make_halves(resid, k=k, mode=mode, seed=seed)
                n_eleg = len(halves)
                halves = subsample(halves, args.max_jugadores, seed)
                pids = list(halves)
                mask = hard_negative_mask(pids, info)
                reps = {
                    "huella": profiles(resid, halves, STYLE_FEATURES),
                    "crudo": profiles(crudo, halves, STYLE_FEATURES),
                    "pool": pool_profiles(resid, halves),
                    "elo_linea": elo_lane_profiles(resid, halves, info),
                    "embedding": embedding_profiles(resid, halves, vecs, lane_vecs),
                }
                for name, (_, A, B) in reps.items():
                    metric = "cosine" if name == "pool" else "euclidean"
                    r = evaluate(A, B, metric=metric, hard_mask=mask, seed=seed)
                    row = {"muestra": mode, "k": k, "semilla": seed, "representacion": name,
                           "jugadores_elegibles": n_eleg, **r}
                    results.append(row)
                    print(f"  {mode:<20} k={k:<3} s={seed} {name:<10} n={r['n_jugadores']:>5} "
                          f"auc={r['auc']:.3f} [{r['auc_ic95'][0]:.3f}, {r['auc_ic95'][1]:.3f}] "
                          f"dificil={r.get('auc_dificil', float('nan')):.3f} top1={r['top1']:.3f} "
                          f"p={r['p_valor']:.4f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"train": args.train, "champions": args.champions, "min_tier": args.min_tier, "filas": len(raw), "filas_residualizadas": len(resid),
            "features": STYLE_FEATURES, "max_jugadores": args.max_jugadores,
            "segundos": round(time.time() - t0, 1)}
    out.write_text(json.dumps({"meta": meta, "resultados": results}, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"[+] {out} ({meta['segundos']} s)")


if __name__ == "__main__":
    main()
