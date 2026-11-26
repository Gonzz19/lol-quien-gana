#!/usr/bin/env python3
"""
scripts/analysis/build_daily_form.py
====================================
Calcula la forma del día por jugador y partida (`src/features/daily_form.py`) y
escribe `data/daily_form.csv` (una fila por jugador-partida, se cruza con
`full_dataset.csv` por `match_id` + `player_id`).

* Día de juego de 08:00 a 08:00 (ART).
* El winrate habitual de cada jugador sale **solo del train** del split vigente
  (`master_plus_temporal_split` vía `compute_split_labels`, §9.S.12), para no
  filtrar el test.
* `dia_completo` usa los listados de la ventana (`crawl_window_listings`): True si
  el jugador tiene bajadas todas sus partidas listadas, False si le falta alguna,
  vacío fuera de los días cubiertos enteros por el listado.

Uso:
    python scripts/analysis/build_daily_form.py
"""

import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd  # noqa: E402

from export_to_csv import compute_split_labels  # noqa: E402
from src.config import DATA_DIR, DEFAULT_DB_PATH  # noqa: E402
from src.features.daily_form import baseline_winrate, daily_form  # noqa: E402
from src.features.identity import attach_for_participants  # noqa: E402

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

VENTANA = "2026-09-01"


def expected_from_db(db_path: Path):
    """(listados jugador-partida, último día de juego cubierto entero) de la ventana."""
    conn = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True, timeout=60.0)
    try:
        attach_for_participants(conn)
        exp = pd.read_sql_query("""
            SELECT DISTINCT m.player_id, l.match_id FROM crawl_window_listings l
            JOIN player_id_map m ON m.puuid = l.players_puuid
            WHERE substr(l.match_id, 1, 4) = 'LA2_'
        """, conn)
        listed_at = conn.execute("SELECT MIN(listed_at) FROM crawl_window_players").fetchone()[0]
    finally:
        conn.close()
    # El listado se hizo a una hora dada: ese día de juego quedó cortado. El último
    # completo es el anterior (listed_at está en UTC; el día de juego, en ART).
    listado = datetime.strptime(listed_at, "%Y-%m-%d %H:%M:%S") - timedelta(hours=3 + 8)
    ultimo = (listado - timedelta(days=1)).strftime("%Y-%m-%d")
    return exp, ultimo


def main() -> None:
    full = pd.read_csv(DATA_DIR / "full_dataset.csv",
                       usecols=["match_id", "player_id", "win", "game_creation", "game_duration"])
    full = full[full["match_id"].str.startswith("LA2_")]
    print(f"[*] {len(full):,} filas jugador-partida de LAS.")
    labels, _tiers, resumen = compute_split_labels(DEFAULT_DB_PATH)
    train = full[full["match_id"].map(labels) == "train"]
    print(f"[*] Winrate habitual con el train (corte {resumen['corte']}, {resumen['train']:,} partidas).")
    exp, ultimo = expected_from_db(DEFAULT_DB_PATH)
    print(f"[*] Días completos posibles: {VENTANA} -> {ultimo} ({exp['player_id'].nunique():,} jugadores con listado).")
    out = daily_form(full, baseline_winrate(train), expected=exp, complete_days=(VENTANA, ultimo))
    out["split"] = out["match_id"].map(labels).fillna("fuera")
    path = DATA_DIR / "daily_form.csv"
    out.to_csv(path, index=False, encoding="utf-8")
    comp = out["dia_completo"].value_counts(dropna=False).to_dict()
    print(f"[+] {path.name}: {len(out):,} filas | días completos: {comp}")
    print(f"    partidas con al menos una previa en el día: {(out['n_previas_dia'] > 0).mean() * 100:.1f}%")


if __name__ == "__main__":
    main()
