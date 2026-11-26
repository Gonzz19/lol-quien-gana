#!/usr/bin/env python3
"""
scripts/migrate_dead_puuids.py
==============================
Une PUUIDs de proyectos MUERTOS (indescifrables por las keys actuales) con las
identidades live, sin necesidad de Account-V1 sobre los PUUIDs muertos.

Método (determinista, 0 falsos positivos):
  Por cada jugador F del roster fresco (PUUID del proyecto canónico 0):
    1. Account-V1(F) -> (gameName, tagLine) [funciona: F lo minteó la key live].
    2. Match-V5 ids(F) -> últimos 100 match_ids [funciona por la misma razón].
    3. Intersección con `matches` locales.
    4. Por cada match en común (hasta K): Match-V5 detail(M) [match_id público,
       balanceable en el pool] -> participant_id donde puuid == F.
    5. El PUUID muerto local en (M, participant_id) es F con certeza:
       se registra el link legacy en `player_puuid_aliases` (sin reescribir
       historial; la unificación ocurre en lectura vía v_puuid_canonical).
  Al final: dedup de partidas dobles + reporte de cupos y cobertura.

Costo: ~3 requests por jugador (account + ids + 1-2 details). Idempotente y
reanaduble (los ya vinculados se omiten sin requests).

Uso:
    python scripts/migrate_dead_puuids.py --tiers CHALLENGER,GRANDMASTER
    python scripts/migrate_dead_puuids.py --tiers CHALLENGER,GRANDMASTER,MASTER
    python scripts/migrate_dead_puuids.py --dry-run
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Agregar raíz del proyecto a sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tqdm import tqdm

from src.config import (
    DEFAULT_DB_PATH,
    DEFAULT_PLATFORM,
    DEFAULT_REGIONAL_ROUTING,
    RIOT_API_KEYS,
)
from src.database import DatabaseManager
from src.riot_client import RiotAPIClient


def attribute_dead_puuid(match_detail: Dict, fresh_puuid: str) -> Optional[int]:
    """Devuelve el participant_id cuyo puuid == fresh_puuid en el Match DTO (puro)."""
    if not match_detail or not fresh_puuid:
        return None
    for p in match_detail.get("info", {}).get("participants", []) or []:
        if p.get("puuid") == fresh_puuid:
            try:
                return int(p.get("participantId", 0)) or None
            except (TypeError, ValueError):
                return None
    return None


def overlapping_local_matches(db: DatabaseManager, match_ids: List[str]) -> List[str]:
    """Intersección de match_ids con la BD local (una sola query)."""
    if not match_ids:
        return []
    with db._get_connection() as conn:
        cursor = conn.cursor()
        placeholders = ",".join("?" for _ in match_ids)
        cursor.execute(f"SELECT match_id FROM matches WHERE match_id IN ({placeholders});",
                       match_ids)
        return [r[0] for r in cursor.fetchall()]


def local_dead_puuid(db: DatabaseManager, match_id: str, participant_id: int) -> Optional[str]:
    with db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT puuid FROM match_participants WHERE match_id = ? "
                       "AND participant_id = ?;", (match_id, participant_id))
        row = cursor.fetchone()
        return row[0] if row else None


def migrate_player(db: DatabaseManager, canon_client: RiotAPIClient,
                   pool_client: RiotAPIClient, entry: Dict,
                   max_overlap_details: int = 2, dry_run: bool = False) -> str:
    """Migra UN jugador. Retorna 'known' | 'linked' | 'no_overlap' | 'unresolved'."""
    fresh = entry.get("puuid")
    if not fresh:
        return "unresolved"
    # Ya vinculado (map live o alias legacy): refrescar stats sin requests.
    if db.resolve_canonical_id(fresh):
        return "known"
    ident = canon_client.get_account_by_puuid(fresh)
    if not ident:
        return "unresolved"
    game_name, tag_line = ident
    canonical_id = DatabaseManager.canonical_id_for(game_name, tag_line)
    if not dry_run:
        db.upsert_player_identity(
            game_name, tag_line, fresh, 0,
            tier=entry.get("tier", "UNKNOWN"), rank=entry.get("rank", "I"),
            league_points=int(entry.get("league_points", 0)),
            wins=int(entry.get("wins", 0)), losses=int(entry.get("losses", 0)),
            summoner_id=entry.get("summoner_id", ""))
    try:
        recent = canon_client.get_match_ids_by_puuid(puuid=fresh, count=100, start=0)
    except Exception:
        return "unresolved"
    overlap = overlapping_local_matches(db, recent or [])
    if not overlap:
        return "no_overlap"
    for mid in overlap[:max(1, max_overlap_details)]:
        detail = pool_client.get_match_detail(mid)
        pid = attribute_dead_puuid(detail or {}, fresh)
        if not pid:
            continue
        dead = local_dead_puuid(db, mid, pid)
        if dead and dead != fresh:
            if not dry_run:
                db.record_puuid_alias(fresh, dead, game_name, tag_line)
            return "linked"
    return "no_overlap"


def _run_only_doubles(db: DatabaseManager, canon_client: RiotAPIClient,
                      pool_client: RiotAPIClient, dry_run: bool) -> None:
    """Resuelve solo los grupos (match, pid) con 2 PUUIDs (barato: 1 detail c/u)."""
    with db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT match_id, participant_id
            FROM match_participants
            GROUP BY match_id, participant_id
            HAVING COUNT(*) > 1;
        """)
        groups = cursor.fetchall()
    print(f"[*] Grupos duplicados pendientes: {len(groups)}")
    fixed_links = 0
    for match_id, pid in groups:
        detail = pool_client.get_match_detail(match_id)
        parts = (detail or {}).get("info", {}).get("participants", []) or []
        fresh = next((p.get("puuid") for p in parts
                      if int(p.get("participantId", -1)) == int(pid)), None)
        if not fresh:
            continue
        known_cid = db.resolve_canonical_id(fresh)
        ident = None
        if not known_cid:
            ident = canon_client.get_account_by_puuid(fresh)
            if not ident:
                continue
            if not dry_run:
                db.upsert_player_identity(ident[0], ident[1], fresh, 0)
        elif not dry_run:
            # Reutilizar game/tag ya registrados para etiquetar el link legacy.
            with db._get_connection() as conn3:
                cur3 = conn3.cursor()
                cur3.execute("SELECT game_name, tag_line FROM player_identities "
                             "WHERE canonical_id = ?;", (known_cid,))
                r3 = cur3.fetchone()
                ident = (r3[0], r3[1]) if r3 and r3[0] else None
        if not dry_run:
            with db._get_connection() as conn2:
                cur2 = conn2.cursor()
                cur2.execute("SELECT DISTINCT puuid FROM match_participants "
                             "WHERE match_id = ? AND participant_id = ?;", (match_id, pid))
                for (dead,) in cur2.fetchall():
                    if dead != fresh:
                        db.record_puuid_alias(fresh, dead,
                                              ident[0] if ident else None,
                                              ident[1] if ident else None)
                        fixed_links += 1
    print(f"[+] Links legacy registrados para dobles: {fixed_links}")
    if not dry_run:
        db.reload_translation_cache()
        print(f"[+] Dedup: {db.dedupe_doubled_participants()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vincula PUUIDs muertos con identidades live por solapamiento de partidas.")
    parser.add_argument("--tiers", type=str, default="CHALLENGER,GRANDMASTER",
                        help="Tiers a migrar (default: CHALLENGER,GRANDMASTER)")
    parser.add_argument("--api-keys", type=str, default=None,
                        help="Pool de keys (default: RIOT_API_KEYS de .env)")
    parser.add_argument("--canon-index", type=int, default=0,
                        help="Índice del proyecto canónico en el pool (default: 0)")
    parser.add_argument("--max-overlap-details", type=int, default=2,
                        help="Details a re-fetch por jugador (default: 2)")
    parser.add_argument("--only-doubles", action="store_true",
                        help="Solo resuelve los grupos duplicados (match_id, participant_id) "
                             "con doble PUUID: ~27 details + accounts puntuales en vez de "
                             "migrar el tier entero")
    parser.add_argument("--dry-run", action="store_true",
                        help="Resuelve sin escribir links (sí hace requests de lectura)")
    args = parser.parse_args()

    keys = ([k.strip() for k in args.api_keys.split(",") if k.strip()]
            if args.api_keys else list(RIOT_API_KEYS))
    keys = [k for k in keys if k and "TU-CLAVE" not in k and "XXXXXXXX" not in k]
    if not keys or args.canon_index >= len(keys):
        print("[X] Pool inválido o canon-index fuera de rango.")
        sys.exit(1)

    tiers = [t.strip().upper() for t in args.tiers.split(",") if t.strip()]
    db = DatabaseManager(DEFAULT_DB_PATH)
    db.register_api_projects(keys)
    print(f"[+] Pool registrado: {len(keys)} proyectos (canónico #{args.canon_index}).")
    canon_client = RiotAPIClient(api_keys=[keys[args.canon_index]],
                                 platform=DEFAULT_PLATFORM,
                                 regional_routing=DEFAULT_REGIONAL_ROUTING)
    pool_client = RiotAPIClient(api_keys=keys, platform=DEFAULT_PLATFORM,
                                regional_routing=DEFAULT_REGIONAL_ROUTING)

    print("=" * 65)
    print("  MIGRACIÓN DE PUUIDs MUERTOS POR SOLAPAMIENTO DE PARTIDAS")
    print("=" * 65)

    if args.only_doubles:
        return _run_only_doubles(db, canon_client, pool_client, args.dry_run)

    entries: List[Dict] = []
    for tier in tiers:
        batch = canon_client.get_apex_league(tier)
        print(f"[*] {tier}: {len(batch)} entradas (proyecto canónico #{args.canon_index}).")
        entries.extend(batch)
    print(f"[*] Total a procesar: {len(entries)} | dry-run={args.dry_run}")

    stats = {"known": 0, "linked": 0, "no_overlap": 0, "unresolved": 0}
    pbar = tqdm(entries, desc="Migrando identidades", unit="jug")
    for e in pbar:
        outcome = migrate_player(db, canon_client, pool_client, e,
                                 max_overlap_details=args.max_overlap_details,
                                 dry_run=args.dry_run)
        stats[outcome] += 1
        pbar.set_postfix({"link": stats["linked"], "conoc": stats["known"]})

    print(f"\n[+] Migración: {stats}")
    if not args.dry_run:
        db.reload_translation_cache()
        dedup = db.dedupe_doubled_participants()
        print(f"[+] Dedup de partidas dobles: {dedup}")
    print("\n  --- CUPOS ---")
    for tier, info in db.check_roster_quotas().items():
        print(f"  {'[OK]' if info['ok'] else '[AVISO]'} {tier:<12}: "
              f"{info['actual']:,} vs {info['expected']:,} :: {info['note']}")
    cov = db.get_identity_coverage()
    print(f"\n  Mapa live: {cov['complete']}/{cov['total_identities']} completos "
          f"({cov['num_projects']} proyectos).")
    print("=" * 65)


if __name__ == "__main__":
    main()
