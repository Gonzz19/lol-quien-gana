#!/usr/bin/env python3
"""
scripts/ops/identity_progress.py
=================================
Telemetría **solo lectura** del avance de la resolución de identidades.

Se puede correr en cualquier momento, incluso con
`scripts/identity/resolve_inverse_by_match.py` escribiendo en paralelo: abre la
base en `mode=ro` y no toma locks de escritura.

Uso:
    python scripts/ops/identity_progress.py
    python scripts/ops/identity_progress.py --watch        # refresca cada 30 s
    python scripts/ops/identity_progress.py --watch --every 10
"""

import argparse
import sqlite3
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import DEFAULT_DB_PATH


def _connect(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{Path(db_path).resolve().as_posix()}?mode=ro",
                           uri=True, timeout=60.0)


def snapshot(db_path: Path) -> dict:
    """Conteos de cobertura de identidad. Sin escrituras, sin red."""
    conn = _connect(db_path)
    try:
        c = conn.cursor()
        total_puuids = c.execute(
            "SELECT COUNT(DISTINCT puuid) FROM match_participants;").fetchone()[0]
        sin_identidad = c.execute("""
            SELECT COUNT(*) FROM (SELECT DISTINCT puuid FROM match_participants) mp
            WHERE NOT EXISTS (SELECT 1 FROM v_puuid_canonical v
                              WHERE v.alias_puuid = mp.puuid)
              AND NOT EXISTS (SELECT 1 FROM player_puuid_map m
                              WHERE m.puuid = mp.puuid)
              AND NOT EXISTS (SELECT 1 FROM puuid_riot_ids r
                              WHERE r.puuid = mp.puuid);
        """).fetchone()[0]
        participaciones = c.execute(
            "SELECT COUNT(*) FROM match_participants;").fetchone()[0]
        part_sin_identidad = c.execute("""
            SELECT COUNT(*) FROM match_participants mp
            WHERE NOT EXISTS (SELECT 1 FROM v_puuid_canonical v
                              WHERE v.alias_puuid = mp.puuid)
              AND NOT EXISTS (SELECT 1 FROM player_puuid_map m
                              WHERE m.puuid = mp.puuid)
              AND NOT EXISTS (SELECT 1 FROM puuid_riot_ids r
                              WHERE r.puuid = mp.puuid);
        """).fetchone()[0]
        identidades = c.execute("SELECT COUNT(*) FROM player_identities;").fetchone()[0]
        aliases = c.execute("SELECT COUNT(*) FROM player_puuid_aliases;").fetchone()[0]
        try:
            obs = c.execute("SELECT COUNT(*) FROM puuid_riot_ids;").fetchone()[0]
            obs_puuids = c.execute(
                "SELECT COUNT(DISTINCT puuid) FROM puuid_riot_ids;").fetchone()[0]
            renombres = c.execute("""
                SELECT COUNT(*) FROM (SELECT puuid FROM puuid_riot_ids
                                      GROUP BY puuid HAVING COUNT(*) > 1);
            """).fetchone()[0]
        except sqlite3.OperationalError:
            obs = obs_puuids = renombres = 0
        proyectos = c.execute(
            "SELECT project_index, key_hint, last_seen FROM api_projects "
            "ORDER BY project_index;").fetchall()
        return {
            "total_puuids": total_puuids, "sin_identidad": sin_identidad,
            "participaciones": participaciones,
            "part_sin_identidad": part_sin_identidad,
            "identidades": identidades, "aliases": aliases,
            "obs": obs, "obs_puuids": obs_puuids, "renombres": renombres,
            "proyectos": proyectos,
        }
    finally:
        conn.close()


def render(s: dict, previo: dict = None) -> str:
    cubiertos = s["total_puuids"] - s["sin_identidad"]
    pct = 100.0 * cubiertos / max(1, s["total_puuids"])
    pct_part = 100.0 * s["part_sin_identidad"] / max(1, s["participaciones"])
    delta = ""
    if previo:
        d = previo["sin_identidad"] - s["sin_identidad"]
        if d:
            delta = f"   (-{d:,} desde el último refresco)"

    out = [
        "=" * 62,
        "  COBERTURA DE IDENTIDAD DE JUGADORES",
        "=" * 62,
        f"  PUUIDs con identidad   : {cubiertos:,} / {s['total_puuids']:,}  ({pct:.2f} %)",
        f"  PUUIDs sin identidad   : {s['sin_identidad']:,}{delta}",
        f"  Participaciones huérfanas: {s['part_sin_identidad']:,} "
        f"de {s['participaciones']:,}  ({pct_part:.2f} %)",
        "",
        f"  Identidades canónicas  : {s['identidades']:,}",
        f"  Alias PUUID            : {s['aliases']:,}",
        f"  Observaciones riotId   : {s['obs']:,} sobre {s['obs_puuids']:,} PUUIDs",
        f"  PUUIDs con >1 nombre   : {s['renombres']:,}  (renombres detectados)",
        "",
        "  Proyectos de API registrados:",
    ]
    for idx, hint, last_seen in s["proyectos"]:
        out.append(f"    P{idx}: {hint}   último uso {last_seen}")
    out.append("=" * 62)
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Avance de la resolución de identidades (solo lectura).")
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB_PATH))
    parser.add_argument("--watch", action="store_true",
                        help="Refresca en bucle hasta Ctrl+C")
    parser.add_argument("--every", type=int, default=30,
                        help="Segundos entre refrescos con --watch (default: 30)")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"[X] No se encontró la base de datos en {db_path}.")
        sys.exit(1)

    previo = None
    while True:
        actual = snapshot(db_path)
        print(render(actual, previo))
        if not args.watch:
            return
        previo = actual
        try:
            time.sleep(max(5, args.every))
        except KeyboardInterrupt:
            print("\n[*] Monitor detenido (no se tocó la base).")
            return


if __name__ == "__main__":
    main()
