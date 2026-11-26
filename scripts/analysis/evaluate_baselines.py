"""
Baselines heurísticos con estadísticas de campeón (`champions_by_lane.csv`).

Seis targets:
    Target 0      Ganador de la partida (1 = gana Blue).
    Targets 1..5  Ganador de cada línea al minuto 15 (1 = Blue tiene más oro @15
                  que su rival directo de la misma posición).

Las reglas y las métricas viven en `src/models/baselines/draft_rules.py`; este
script carga los datos, arma el split e imprime.

Modo por defecto `sin-leakage`: split vigente (`master_plus_temporal_split`, §9.S.12:
test = 20 % más reciente de Master+, Diamond anterior al corte en el train), tablas de campeón calculadas solo con el train y
evaluación solo sobre el test. El modo `descriptivo` (el original) usa
`champions_by_lane.csv`, que agrega todas las partidas, incluidas las evaluadas.

Uso:
    python scripts/analysis/evaluate_baselines.py
    python scripts/analysis/evaluate_baselines.py --out data/baseline_metrics.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import pandas as pd
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.baselines.draft_rules import (LANES, lane_baseline,  # noqa: E402
                                              load_champion_stats, load_participants,
                                              match_baseline, stats_from_participants)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def split_participants(full_path: Path, players_path: Path, test_frac: float = 0.2):
    """(train, test, resumen) con el split vigente (`master_plus_temporal_split`, §9.S.12).

    Solo partidas de LAS: las de otros servidores se filtran acá (preprocesamiento).
    """
    from src.features.split import master_plus_temporal_split, match_tiers

    parts = load_participants(full_path, extra=("player_id", "game_creation"))
    parts = parts[parts["match_id"].str.startswith("LA2_")]
    players = pd.read_csv(players_path, usecols=["player_id", "player_tier"])
    tiers = match_tiers(parts, players)
    m = (parts.drop_duplicates("match_id")[["match_id", "game_creation"]]
         .assign(tier=lambda d: d["match_id"].map(tiers)))
    labels, resumen = master_plus_temporal_split(m, test_frac)
    lab = parts["match_id"].map(labels)
    return parts[lab == "train"], parts[lab == "test"], resumen


# ----------------------------------------------------------------------------
# Salida
# ----------------------------------------------------------------------------
def _fmt_row(name: str, m: Dict) -> str:
    return (f"{name:<9} | {m['n']:>9,} | {m['accuracy']*100:>7.2f}% | {m['roc_auc']:>7.4f} | "
            f"{m['f1_blue']:>6.4f} | {m['precision_blue']:>6.4f} | {m['recall_blue']:>6.4f} | "
            f"{m['always_red_accuracy']*100:>7.2f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description="Baselines con stats de campeón (6 targets).")
    ap.add_argument("--full", default="data/full_dataset.csv")
    ap.add_argument("--champions", default="data/champions_by_lane.csv")
    ap.add_argument("--players", default="data/players.csv")
    ap.add_argument("--modo", choices=["sin-leakage", "descriptivo"], default="sin-leakage",
                    help="sin-leakage: tablas de campeón con el train, evaluación en el test "
                         "(split vigente: 20 %% más reciente de Master+). descriptivo: tablas y "
                         "evaluación con todas las partidas (tiene leakage)")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--out", default=None, help="JSON opcional con todas las métricas")
    args = ap.parse_args()

    print("[*] Cargando datos...", flush=True)
    split = None
    if args.modo == "sin-leakage":
        train, parts, split = split_participants(Path(args.full), Path(args.players), args.test_frac)
        stats = stats_from_participants(train)
        reparto = {k: f"{v * 100:.1f}%" for k, v in split["reparto_test"].items()}
        print(f"[*] Split: corte {split['corte']} | train {split['train']:,} partidas | "
              f"test {split['test']:,} | Diamond desde el corte (aparte) {split['diamante_despues']:,} "
              f"| sin rango {split['sin_rango']:,}")
        print(f"    reparto por rango del test: {reparto} (por rango: {split['test_por_rango']})")
        print("[*] Tablas de campeón calculadas SOLO con el train; se evalúa SOLO el test.")
    else:
        stats = load_champion_stats(Path(args.champions))
        parts = load_participants(Path(args.full))
    print(f"[*] Evaluando {len(parts):,} participaciones, {parts['match_id'].nunique():,} partidas.")

    m = match_baseline(parts, stats)
    lanes = lane_baseline(parts, stats)

    header = (f"{'Target':<9} | {'N':>9} | {'Accuracy':>8} | {'ROC-AUC':>7} | {'F1':>6} | "
              f"{'Prec':>6} | {'Recall':>6} | {'Siempre Red':>8}")
    print("\n" + "=" * len(header))
    print("  TARGET 0 — GANADOR DE LA PARTIDA (media de WR campeón-línea por equipo)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    print(_fmt_row("PARTIDA", m))
    print(f"\n    Brier = {m['brier']:.4f} | empates de score = {m['ties_pct']:.2f}% "
          f"| excluidas sin ganador único = {m['excluded_incoherent']} "
          f"| fallback a WR global = {m['fallback_lane_wr_pct']:.3f}% de picks")

    print("\n" + "=" * len(header))
    print("  TARGETS 1..5 — GANADOR DE LÍNEA @15 (oro medio @15 campeón-línea)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for lane in LANES:
        print(_fmt_row(lane, lanes[lane]))
    print("-" * len(header))
    print(_fmt_row("GLOBAL", lanes["GLOBAL"]))
    print(f"\n    Duelos descartados: {lanes['_dropped_position']:,} por posición duplicada/vacía, "
          f"{lanes['_dropped_no_label']:,} sin oro @15 o con empate exacto.")
    if args.modo == "descriptivo":
        print("\n[!] Modo descriptivo: las stats de campeón incluyen las partidas evaluadas "
              "(leakage). Las métricas describen la tabla, no generalizan.")
    else:
        print("\n[+] Sin leakage: ninguna partida del test interviene en las tablas de campeón.")

    if args.out:
        Path(args.out).write_text(json.dumps({"modo": args.modo, "split": split,
                                              "match": m, "lanes": lanes}, indent=2),
                                  encoding="utf-8")
        print(f"[+] Métricas guardadas en {args.out}")


if __name__ == "__main__":
    main()
