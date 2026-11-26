#!/usr/bin/env python3
"""
scripts/collection/snapshot_ladder.py
=====================================
Foto fechada del rango de los jugadores, guardada aparte en `ladder_snapshots`
(no pisa `players`). Sirve para el rating por partida (PROJECT_SUMMARY.md §9.S.21):
el rating se evalúa con la recorrida de una foto a la siguiente.

Fases:
* `ladder`: Challenger, Grandmaster y Master completos (3 requests) y todas las
  páginas de Diamond I-IV, con la key del proyecto 0 (los PUUID del padrón son de P0).
* `sin_rango`: los jugadores de nuestras partidas de LAS que no están en el padrón
  (`players`) ni en el ladder de esta foto, consultados uno por uno (league-v4
  entries/by-puuid) con la key del proyecto de su PUUID. De más a menos partidas;
  se puede cortar y retomar (salta a los que ya están en la foto).

Toda petición pasa por `RiotRateLimiter` (AGENTS.md §3.B). Verificar antes las keys
con `python scripts/ops/check_api_keys.py`.

Uso:
    python scripts/collection/snapshot_ladder.py --fase ladder
    python scripts/collection/snapshot_ladder.py --fase sin_rango --min-partidas 5
    python scripts/collection/snapshot_ladder.py --fase sin_rango --snapshot "2026-10-27 18:00:00"
"""

import argparse
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.config import (DATA_DIR, DEFAULT_DB_PATH, RIOT_API_KEYS, RIOT_API_PROJECT_INDICES,  # noqa: E402
                        project_index_for_key)
from src.database import DatabaseManager  # noqa: E402
from src.riot_client import RiotAPIClient  # noqa: E402

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

LOTE = 50


def fase_ladder(db: DatabaseManager, snapshot: str) -> None:
    p0_key = next((k for k in RIOT_API_KEYS if project_index_for_key(k) == 0), None)
    if not p0_key:
        sys.exit("[X] Hace falta la key de P0 en el .env: los PUUID del padrón son de P0.")
    client = RiotAPIClient(api_keys=[p0_key], project_indices=[0])
    t0 = time.time()
    jugadores = client.fetch_all_diamond_plus_players(include_diamond=True)
    if not jugadores:
        sys.exit("[X] Riot no devolvió jugadores: revisar la key y la conexión.")
    n = db.save_ladder_snapshot(snapshot, jugadores, project_index=0, fuente="ladder")
    por_tier = pd.Series([j["tier"] for j in jugadores]).value_counts().to_dict()
    print(f"[+] Foto {snapshot}: {n:,} jugadores del ladder guardados {por_tier} "
          f"({time.time() - t0:.0f} s, {client.get_telemetry()})")


def objetivos_sin_rango(db_path: Path, snapshot: str, min_partidas: int) -> pd.DataFrame:
    """Jugadores de LAS fuera del padrón y del ladder de la foto, con sus PUUID por proyecto."""
    d = pd.read_csv(DATA_DIR / "full_dataset.csv", usecols=["match_id", "player_id"])
    n = d[d["match_id"].str.startswith("LA2_")].groupby("player_id").size().rename("partidas")
    conn = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True, timeout=60.0)
    try:
        mapa = pd.read_sql_query("SELECT puuid, player_id FROM player_id_map", conn)
        padron = set(pd.read_sql_query("SELECT puuid FROM players", conn)["puuid"])
        en_foto = set(pd.read_sql_query("SELECT puuid FROM ladder_snapshots WHERE snapshot_at = ?",
                                        conn, params=(snapshot,))["puuid"])
        proyectos = pd.read_sql_query(
            "SELECT puuid, project_index, source FROM puuid_projects WHERE project_index IN (%s)"
            % ",".join(str(int(i)) for i in RIOT_API_PROJECT_INDICES or [0]), conn)
    finally:
        conn.close()
    cubiertos = set(mapa.loc[mapa["puuid"].isin(padron | en_foto), "player_id"])
    obj = n[(n >= min_partidas) & ~n.index.isin(cubiertos)].sort_values(ascending=False)
    proyectos = proyectos.merge(mapa, on="puuid")
    proyectos = proyectos[proyectos["player_id"].isin(obj.index)]
    # Por persona y proyecto, un PUUID: el 'exact' antes que el 'inferred'.
    proyectos = proyectos.sort_values("source").drop_duplicates(["player_id", "project_index"])
    opciones = proyectos.groupby("player_id").apply(
        lambda g: dict(zip(g["project_index"], g["puuid"])), include_groups=False).rename("opciones")
    return obj.to_frame().join(opciones, how="inner")


def fase_sin_rango(db: DatabaseManager, snapshot: str, min_partidas: int, maximo: int, minutos: float) -> None:
    obj = objetivos_sin_rango(Path(db.db_path), snapshot, min_partidas)
    if maximo:
        obj = obj.head(maximo)
    print(f"[*] Foto {snapshot}: {len(obj):,} jugadores sin rango por consultar "
          f"(≥ {min_partidas} partidas, de más a menos).")
    client = RiotAPIClient(api_keys=RIOT_API_KEYS, project_indices=RIOT_API_PROJECT_INDICES or None)

    def feedback(puuid, project, ok):
        if ok:
            db.record_puuid_projects([(puuid, project)], source="exact")
        else:
            db.forget_puuid_project(puuid, project)
    client.project_feedback = feedback

    t0, lote, hechos, fallas = time.time(), {}, 0, 0
    tiers = {}
    for k, (pid, fila) in enumerate(obj.iterrows(), 1):
        if minutos and time.time() - t0 > minutos * 60:
            print("[i] Tiempo límite alcanzado: se corta (retomar con el mismo --snapshot).")
            break
        r = client.get_soloq_entry_by_puuid(fila["opciones"])
        if r is None:
            fallas += 1
            continue
        lote.setdefault(r["project_index"], []).append(r)
        tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1
        hechos += 1
        if hechos % LOTE == 0:
            for pj, filas in lote.items():
                db.save_ladder_snapshot(snapshot, filas, project_index=pj, fuente="by_puuid")
            lote = {}
        if k % 500 == 0:
            ritmo = k / max(time.time() - t0, 1)
            print(f"[{k:,}/{len(obj):,}] {ritmo:.2f} jug/s, ETA {(len(obj) - k) / ritmo / 60:.0f} min | "
                  f"{dict(sorted(tiers.items(), key=lambda x: -x[1]))} | fallas {fallas}", flush=True)
    for pj, filas in lote.items():
        db.save_ladder_snapshot(snapshot, filas, project_index=pj, fuente="by_puuid")
    print(f"[+] {hechos:,} guardados, {fallas} sin respuesta ({time.time() - t0:.0f} s). Rangos: {tiers}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Foto fechada del rango en ladder_snapshots")
    ap.add_argument("--fase", choices=["ladder", "sin_rango", "todo"], default="todo")
    ap.add_argument("--snapshot", default=None,
                    help="etiqueta UTC 'YYYY-MM-DD HH:MM:SS'; por defecto la última foto de ladder (o ahora)")
    ap.add_argument("--min-partidas", type=int, default=1)
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--minutos", type=float, default=0)
    args = ap.parse_args()

    db = DatabaseManager(DEFAULT_DB_PATH)
    snapshot = args.snapshot
    if args.fase in ("ladder", "todo"):
        snapshot = snapshot or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        fase_ladder(db, snapshot)
    if args.fase in ("sin_rango", "todo"):
        if snapshot is None:
            with db._get_connection() as conn:
                snapshot = conn.execute(
                    "SELECT MAX(snapshot_at) FROM ladder_snapshots WHERE fuente = 'ladder'").fetchone()[0]
            if snapshot is None:
                sys.exit("[X] No hay foto de ladder: correr antes --fase ladder.")
        fase_sin_rango(db, snapshot, args.min_partidas, args.max, args.minutos)


if __name__ == "__main__":
    main()
