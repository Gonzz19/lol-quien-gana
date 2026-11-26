#!/usr/bin/env python3
"""
scripts/identity/backfill_puuid_projects.py
===========================================
Llena `puuid_projects` (proyecto de cifrado de cada PUUID) con lo que ya está en
la base, sin pedir nada a Riot:

1. **Exacto:** `players` (P0), `player_puuid_map`, y las etiquetas 1/2/3 de
   `puuid_riot_ids` (la 0 está contaminada, ver PROJECT_SUMMARY.md §9.S.4).
2. **Deducido:** para los PUUIDs que quedan sin etiqueta, el proyecto mayoritario
   de los demás PUUIDs de sus partidas, si supera la pureza mínima
   (`src/features/puuid_projects.py`).

Es idempotente: re-ejecutarlo no duplica, y un 'inferred' nunca pisa un 'exact'.
Desde la ingesta nueva, cada partida descargada ya anota sus 10 PUUIDs como
exactos; este script solo hace falta una vez, para el historial.

Uso:
    python scripts/identity/backfill_puuid_projects.py --dry-run
    python scripts/identity/backfill_puuid_projects.py [--min-purity 0.95]
"""

import argparse
import sqlite3
import sys
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import DEFAULT_DB_PATH  # noqa: E402
from src.database import DatabaseManager  # noqa: E402
from src.features.puuid_projects import (  # noqa: E402
    MIN_PURITY,
    exact_projects,
    infer_projects_from_matches,
)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def plan(db_path: Path, min_purity: float):
    """(exactos, deducidos) calculados en solo lectura."""
    conn = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True)
    try:
        exact = exact_projects(conn)
        rows = conn.execute("SELECT match_id, puuid FROM match_participants;").fetchall()
    finally:
        conn.close()
    inferred = infer_projects_from_matches(rows, exact, min_purity=min_purity)
    return exact, inferred


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill del proyecto de cada PUUID (sin API).")
    ap.add_argument("--db", default=str(DEFAULT_DB_PATH))
    ap.add_argument("--min-purity", type=float, default=MIN_PURITY)
    ap.add_argument("--dry-run", action="store_true", help="Solo informa, no escribe")
    args = ap.parse_args()

    exact, inferred = plan(Path(args.db), args.min_purity)
    print(f"[*] Exactos: {len(exact):,} PUUIDs {dict(sorted(Counter(exact.values()).items()))}")
    print(f"[*] Deducidos (pureza >= {args.min_purity}): {len(inferred):,} PUUIDs "
          f"{dict(sorted(Counter(inferred.values()).items()))}")
    if args.dry_run:
        print("[dry-run] No se escribió nada.")
        return

    db = DatabaseManager(Path(args.db))
    n_exact = db.record_puuid_projects(list(exact.items()), source="exact")
    n_inf = db.record_puuid_projects(list(inferred.items()), source="inferred")
    with db._get_connection() as conn:
        por_fuente = conn.execute(
            "SELECT source, COUNT(*) FROM puuid_projects GROUP BY source;").fetchall()
    print(f"[+] Escritos: {n_exact:,} exactos, {n_inf:,} deducidos. "
          f"Estado de la tabla: {dict((r[0], r[1]) for r in por_fuente)}")


if __name__ == "__main__":
    main()
