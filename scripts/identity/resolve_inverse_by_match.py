#!/usr/bin/env python3
"""
scripts/resolve_inverse_by_match.py
====================================
Resolución inversa bottom-up: en vez de partir del jugador fresco y buscar
solapamiento (ciego con inactivos), parte de las PARTIDAS LOCALES que contienen
PUUIDs muertos y pide el Match DTO con una key viva.

Por cada match: Riot devuelve los 10 PUUIDs frescos (+ riotIdGameName/Tagline),
y el join (match_id, participant_id) contra las filas locales establece links
deterministas alias -> live (hasta 10 muertos por request, 0 Account-V1 en el
caso común). Resuelve inactivos y Diamond por igual: todo muerto tiene ≥1 fila.

Atribución por proyecto: este pase escribe IDENTIDADES + ALIAS legacy (suficiente
para traducir, coverage y export vía v_puuid_canonical). Las columnas live por
proyecto las completa scripts/register_key_project.py por roster.

POR QUÉ ESTA ES LA ÚNICA VÍA CON UNA KEY DE OTRO PROYECTO
---------------------------------------------------------
El PUUID está cifrado por *producto* de API. Una key cuyo proyecto no es el que
minteó los PUUIDs almacenados NO puede descifrarlos: `account/v1/by-puuid` sobre
un PUUID ajeno devuelve 400/404 siempre, y por eso ningún barrido por roster ni
Account-V1 sobre lo almacenado sirve.

`match/v5/matches/{matchId}` es la excepción: el matchId es texto plano y
universal, cualquier key puede pedir cualquier partida, y la respuesta trae los
10 participantes con `riotIdGameName`/`riotIdTagline` **sin cifrar**. El join por
`participantId` contra las filas locales es exacto y no depende del nombre.

Rendimiento: 1 request = hasta 10 identidades, contra 1 identidad por request de
Account-V1. Además Match-V5 tiene el límite de method más holgado (2000/10 s vs
1000/min).

Uso:
    python scripts/identity/resolve_inverse_by_match.py --dry-run
    python scripts/identity/resolve_inverse_by_match.py [--max-matches N]
"""

import argparse
import heapq
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

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
    RIOT_API_PROJECT_INDICES,
)
from src.database import DatabaseManager
from src.riot_client import RiotAPIClient


def uncovered_dead(db: DatabaseManager) -> Set[str]:
    """PUUIDs distintos en partidas sin ningún link (ni map, ni alias, ni observación riotId)."""
    with db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT puuid FROM match_participants
            WHERE puuid NOT IN (SELECT alias_puuid FROM v_puuid_canonical)
              AND puuid NOT IN (SELECT canonical_puuid FROM v_puuid_canonical)
              AND puuid NOT IN (SELECT puuid FROM player_puuid_map)
              AND puuid NOT IN (SELECT puuid FROM puuid_riot_ids);
        """)
        return {r[0] for r in cursor.fetchall()}


def build_coverage_index(db: DatabaseManager,
                         uncovered: Set[str]) -> Dict[str, Set[str]]:
    """{match_id: {puuids muertos que aporta}} para las partidas que aportan ≥1.

    Se construye con UNA consulta indexada restringida a los muertos (join contra
    una tabla TEMP), en vez de agregar las ~982k filas de `match_participants`.
    La versión anterior hacía ese `GROUP BY` completo **una vez por lote**, o sea
    ~34 pasadas enteras para un run completo.
    """
    if not uncovered:
        return {}
    index: Dict[str, Set[str]] = {}
    with db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS temp.uncovered_puuid;")
        cursor.execute("CREATE TEMP TABLE uncovered_puuid (puuid TEXT PRIMARY KEY);")
        cursor.executemany("INSERT OR IGNORE INTO uncovered_puuid (puuid) VALUES (?);",
                           ((p,) for p in uncovered))
        cursor.execute("""
            SELECT mp.match_id, mp.puuid
            FROM uncovered_puuid u
            INNER JOIN match_participants mp ON mp.puuid = u.puuid;
        """)
        for match_id, puuid in cursor.fetchall():
            index.setdefault(match_id, set()).add(puuid)
        cursor.execute("DROP TABLE IF EXISTS temp.uncovered_puuid;")
    return index


def greedy_match_order(index: Dict[str, Set[str]]) -> Iterator[Tuple[str, int]]:
    """Orden greedy de máxima cobertura, con heap perezoso y re-scoring incremental.

    Emite `(match_id, ganancia_real)` de mayor a menor ganancia. Una entrada cuya
    ganancia quedó obsoleta (porque otra partida ya cubrió parte de sus muertos)
    se reinserta con su valor actualizado en vez de recalcular todo el ranking:
    el costo total es O(|E| log |M|) en memoria, sin volver a tocar la base.

    El marcado es optimista —se asume que pedir la partida cubre sus muertos—.
    Si una resolución falla, el PUUID sigue sin vincular en la base y lo levanta
    la próxima corrida: el script es reanudable por diseño.
    """
    heap = [(-len(dead), match_id) for match_id, dead in index.items() if dead]
    heapq.heapify(heap)
    covered: Set[str] = set()
    while heap:
        neg_gain, match_id = heapq.heappop(heap)
        remaining = index[match_id] - covered
        if not remaining:
            continue
        if len(remaining) != -neg_gain:
            heapq.heappush(heap, (-len(remaining), match_id))
            continue
        yield match_id, len(remaining)
        covered |= remaining


def extract_fresh_identities(match_detail: Dict) -> List[Tuple[int, str, Optional[str], Optional[str]]]:
    """[(participant_id, fresh_puuid, game_name, tag_line)] desde el Match DTO."""
    out = []
    for p in (match_detail or {}).get("info", {}).get("participants", []) or []:
        try:
            pid = int(p.get("participantId", 0))
        except (TypeError, ValueError):
            continue
        if pid and p.get("puuid"):
            out.append((pid, p["puuid"], p.get("riotIdGameName"), p.get("riotIdTagline")))
    return out


def resolve_match(db: DatabaseManager, pool_client: RiotAPIClient, match_id: str,
                  project_index: Optional[int] = None,
                  dry_run: bool = False) -> Dict[str, int]:
    """Resuelve UNA partida local. Retorna {linked, already, no_identity, observed}.

    Endurecimiento respecto de la versión previa: los `riotId` del payload se
    persisten para **los 10 participantes**, no solo para los que hacía falta
    vincular. Antes se descartaban los del caso `already`, tirando información
    que la request ya había pagado. Esas observaciones son el puente entre
    proyectos de API y el historial de renombres (ver `puuid_riot_ids`).
    """
    stats = {"linked": 0, "already": 0, "no_identity": 0, "observed": 0}
    detail = pool_client.get_match_detail(match_id)
    if not detail:
        return stats

    fresh_rows = extract_fresh_identities(detail)
    game_creation = ((detail or {}).get("info", {}) or {}).get("gameCreation")

    # 1. Persistir TODO nombre observado, con su ventana temporal. Cuesta 0 requests.
    if not dry_run:
        observations = [(fresh, gn, tag) for _pid, fresh, gn, tag in fresh_rows if gn]
        stats["observed"] = db.record_riot_id_observations(
            observations, project_index=project_index, game_creation=game_creation)
        # Los 10 PUUIDs frescos vienen cifrados con el proyecto de esta key.
        if project_index is not None:
            db.record_puuid_projects([(fresh, project_index) for _pid, fresh, _gn, _tag in fresh_rows],
                                     source="exact")

    with db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT participant_id, puuid FROM match_participants WHERE match_id = ?;",
                       (match_id,))
        local = {int(r[0]): r[1] for r in cursor.fetchall()}

    # 2. Vincular el PUUID almacenado con el fresco, por participantId.
    for pid, fresh, gn, tag in fresh_rows:
        dead = local.get(pid)
        if not dead:
            continue
        if db.resolve_canonical_id(dead):
            stats["already"] += 1
            continue
        if not gn:
            # Fallback solo sobre el PUUID FRESCO, que sí pertenece a nuestro
            # proyecto. Pedir Account-V1 sobre `dead` daría 400/404 garantizado.
            fb = pool_client.get_account_by_puuid(fresh)
            if not fb:
                stats["no_identity"] += 1
                continue
            gn, tag = fb
            if not dry_run:
                db.record_riot_id_observations([(fresh, gn, tag)],
                                               project_index=project_index,
                                               game_creation=game_creation)
        if not dry_run:
            if dead == fresh:
                db.upsert_player_identity(
                    gn, tag, fresh,
                    project_index=project_index if project_index is not None else 0,
                )
            else:
                db.upsert_player_identity(gn, tag, fresh, project_index=None)
                db.record_puuid_alias(fresh, dead, gn, tag)
                db.record_riot_id_observations([(dead, gn, tag)],
                                               project_index=None,
                                               game_creation=game_creation)
        stats["linked"] += 1
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolución inversa bottom-up de PUUIDs muertos por Match-ID.")
    parser.add_argument("--api-keys", type=str, default=None,
                        help="Pool de keys (default: RIOT_API_KEYS de .env)")
    parser.add_argument("--max-matches", type=int, default=None,
                        help="Tope de details a descargar (default: hasta cubrir todo)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Solo cuenta cobertura sin escribir ni pedir details")
    args = parser.parse_args()

    keys = ([k.strip() for k in args.api_keys.split(",") if k.strip()]
            if args.api_keys else list(RIOT_API_KEYS))
    keys = [k for k in keys if k and "TU-CLAVE" not in k and "XXXXXXXX" not in k]
    if not keys:
        print("[X] Sin API keys válidas.")
        sys.exit(1)

    # Ranura de proyecto: NUNCA por posicion cuando el pool es un subconjunto de
    # los proyectos historicos, o el registro pisa el key_hint de una ranura ajena.
    indices = RIOT_API_PROJECT_INDICES or None
    if indices is not None and len(indices) != len(keys):
        print(f"[X] RIOT_API_PROJECT_INDICES ({len(indices)}) no coincide con el "
              f"pool ({len(keys)} keys).")
        sys.exit(1)

    db = DatabaseManager(DEFAULT_DB_PATH)
    registered = db.register_api_projects(keys, project_indices=indices)
    slot_clients: List[Tuple[int, RiotAPIClient]] = [
        (proj_idx, RiotAPIClient(api_key=k, platform=DEFAULT_PLATFORM,
                                 regional_routing=DEFAULT_REGIONAL_ROUTING))
        for proj_idx, k in zip(registered, keys)
    ]

    print("=" * 65)
    print("  RESOLUCION INVERSA BOTTOM-UP POR MATCH-ID")
    print("=" * 65)
    print(f"[*] Ranuras de proyecto en uso: {registered} "
          f"({'explicitas por .env' if indices else 'por posicion en el pool'})")
    uncovered = uncovered_dead(db)
    print(f"[*] Muertos distintos sin vincular: {len(uncovered):,}")

    print("[*] Construyendo indice de cobertura (1 consulta indexada)...")
    index = build_coverage_index(db, uncovered)
    order = list(greedy_match_order(index))
    total_gain = sum(g for _, g in order)
    print(f"[*] Plan greedy: {len(order):,} partidas cubren {total_gain:,} muertos "
          f"({total_gain / max(1, len(order)):.2f} por request).")

    if args.dry_run:
        rate_per_min = 47.5 * max(1, len(slot_clients))
        for corte in (0.5, 0.8, 0.9, 1.0):
            objetivo, acumulado, reqs = corte * total_gain, 0, 0
            for _m, g in order:
                if acumulado >= objetivo:
                    break
                acumulado += g
                reqs += 1
            print(f"    [dry-run] {corte:.0%} de cobertura -> {reqs:,} requests "
                  f"(~{reqs / rate_per_min:.0f} min a {rate_per_min:.1f} req/min)")
        return

    totals = {"linked": 0, "already": 0, "no_identity": 0, "observed": 0, "matches": 0}
    limite = args.max_matches or len(order)
    pbar = tqdm(total=min(limite, len(order)), desc="Resolviendo partidas", unit="match")
    for match_id, _gain in order[:limite]:
        active_slots = [(p_idx, c) for p_idx, c in slot_clients if c.num_active_keys > 0]
        if not active_slots:
            print("\n[!] Todas las claves del pool se agotaron. Cortando ordenado...")
            break
        proj_idx, client = min(active_slots, key=lambda item: item[1].rate_limiter.get_wait_time())
        st = resolve_match(db, client, match_id, project_index=proj_idx)
        for k in ("linked", "already", "no_identity", "observed"):
            totals[k] += st[k]
        totals["matches"] += 1
        pbar.update(1)
        pbar.set_postfix({"link": totals["linked"], "obs": totals["observed"]})
    pbar.close()

    print(f"\n[+] Details descargados: {totals['matches']:,} | "
          f"links nuevos: {totals['linked']:,} | ya vinculados: {totals['already']:,} | "
          f"sin identidad: {totals['no_identity']:,} | "
          f"observaciones riotId: {totals['observed']:,}")
    db.reload_translation_cache()
    print(f"[+] Dedup dobles: {db.dedupe_doubled_participants()}")
    mapped = db.materialize_player_id_map()
    print(f"[+] player_id_map permanente actualizado: {mapped:,} PUUIDs mapeados.")
    print(f"[*] Muertos restantes sin vincular: {len(uncovered_dead(db)):,}")
    print("=" * 65)


if __name__ == "__main__":
    main()

