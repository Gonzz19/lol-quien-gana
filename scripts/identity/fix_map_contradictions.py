#!/usr/bin/env python3
"""
scripts/identity/fix_map_contradictions.py
==========================================
Corrige PUUIDs asignados a la persona equivocada en `player_puuid_map`
(PROJECT_SUMMARY.md §9.S.9).

El problema
-----------
Hay PUUIDs con dos afirmaciones incompatibles:
* `player_puuid_aliases` (resolución inversa: join exacto por `(match_id,
  participant_id)` contra el Match DTO) dice que el PUUID es de la persona A;
* `player_puuid_map` dice que es de la identidad B.

Mientras el PUUID tenga fila en el mapa, la vista `v_puuid_canonical` ignora su
alias, así que queda unido a B. Si A y B son personas distintas, A queda partido
y B mezclado con partidas de A.

El criterio (lo decide Riot, no una heurística)
-----------------------------------------------
Se toma una partida guardada donde juega el PUUID y se pide a Riot con la key de
P0. En el mismo `participantId`, Riot devuelve el PUUID de P0 de quien jugó ese
lugar: el **dueño real**. Si el dueño real ya es la persona a la que hoy está
unido el PUUID, es un cambio de nombre y no se toca. Si es otra persona, la fila
del mapa es errónea: se borra y el PUUID se vincula a su dueño real con un alias.

Uso:
    python scripts/identity/fix_map_contradictions.py --dry-run
    python scripts/identity/fix_map_contradictions.py
"""

import argparse
import collections
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import DEFAULT_DB_PATH, RIOT_API_KEYS, project_index_for_key  # noqa: E402
from src.database import DatabaseManager  # noqa: E402
from src.features.identity import attach_for_participants  # noqa: E402

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def find_contradictions(conn, lado: str = "ambos") -> List[Tuple[str, str, int]]:
    """`[(puuid, canonical_id_del_mapa, project_index)]` cuyo alias nombra a otra identidad.

    Un alias tiene dos PUUIDs, el viejo (`alias_puuid`) y el fresco
    (`primary_puuid`), y los dos fueron del mismo humano en esa partida: el
    nombre del alias contradice al mapa si el PUUID del mapa aparece de
    cualquiera de los dos lados. `lado` = "alias", "primario" o "ambos".
    """
    alias_names: Dict[str, set] = collections.defaultdict(set)
    cols = {"alias": ["alias_puuid"], "primario": ["primary_puuid"],
            "ambos": ["alias_puuid", "primary_puuid"]}[lado]
    for col in cols:
        for puuid, g, t in conn.execute(
                f"SELECT {col}, game_name, tag_line FROM player_puuid_aliases WHERE game_name IS NOT NULL"):
            alias_names[puuid].add(f"{(g or '').strip().lower()}#{(t or '').strip().lower()}")
    out = []
    for puuid, cid, pj in conn.execute("SELECT puuid, canonical_id, project_index FROM player_puuid_map"):
        if puuid in alias_names and cid not in alias_names[puuid]:
            out.append((puuid, cid, int(pj)))
    return out


def classify(person_of_puuid: Optional[str], person_of_owner: Optional[str]) -> str:
    """'renombre' si el dueño real ya es la persona del PUUID; 'mal_asignado' si es otra;
    'sin_dueno' si el PUUID de P0 que dio Riot no está en la base (no se puede ubicar)."""
    if person_of_owner is None:
        return "sin_dueno"
    return "renombre" if person_of_owner == person_of_puuid else "mal_asignado"


def resolve_owner(conn, client, puuid: str) -> Optional[Tuple[str, str, str]]:
    """`(match_id, puuid_P0_del_dueño, nombre)` según Riot, o None si no se pudo."""
    row = conn.execute("SELECT match_id, participant_id FROM match_participants WHERE puuid = ? "
                       "ORDER BY match_id DESC LIMIT 1", (puuid,)).fetchone()
    if not row:
        return None
    match_id, pid = row
    detail = client.get_match_detail(match_id)
    for p in ((detail or {}).get("info", {}) or {}).get("participants", []) or []:
        if int(p.get("participantId", 0)) == int(pid):
            return match_id, p.get("puuid"), f"{p.get('riotIdGameName')}#{p.get('riotIdTagline')}"
    return None


def apply_fix(db: DatabaseManager, puuid: str, map_cid: str, project: int,
              owner_p0: str, owner_name: str) -> None:
    """Borra la fila errónea del mapa y vincula el PUUID a su dueño real."""
    game, _, tag = owner_name.partition("#")
    with db._get_connection() as conn:
        conn.execute("DELETE FROM player_puuid_map WHERE puuid = ? AND canonical_id = ? AND project_index = ?;",
                     (puuid, map_cid, project))
        conn.commit()
    db.record_puuid_alias(owner_p0, puuid, game or None, tag or None)


def fix_rows_for_puuid(db: DatabaseManager, client, puuid: str) -> dict:
    """Corrección fila por fila de un PUUID que representa a dos personas según la partida.

    Pasa cuando una migración física vieja reescribió filas de una persona con el
    PUUID de otra. Para cada partida guardada con ese PUUID se pide a Riot quién
    jugó ese `participantId` (key de P0). Si es otra persona, la fila (y su
    timeline) pasa a tener el PUUID de P0 que dio Riot, que es exacto. Las filas
    del dueño legítimo quedan como están.
    """
    stats = {"filas": 0, "corregidas": 0, "sin_respuesta": 0}
    with db._get_connection() as conn:
        attach_for_participants(conn)
        person = dict(conn.execute("SELECT puuid, player_id FROM temp.player_id_map"))
        rows = conn.execute("SELECT match_id, participant_id FROM match_participants WHERE puuid = ?",
                            (puuid,)).fetchall()
    for match_id, pid in rows:
        stats["filas"] += 1
        detail = client.get_match_detail(match_id)
        slot = next((p for p in ((detail or {}).get("info", {}) or {}).get("participants", []) or []
                     if int(p.get("participantId", 0)) == int(pid)), None)
        if slot is None:
            stats["sin_respuesta"] += 1
            continue
        real = slot.get("puuid")
        if not real or person.get(real) == person.get(puuid):
            continue  # fila del dueño legítimo
        with db._get_connection() as conn:
            conn.execute("UPDATE match_participants SET puuid = ? WHERE match_id = ? AND participant_id = ?;",
                         (real, match_id, pid))
            conn.execute("UPDATE match_timeline SET puuid = ? WHERE match_id = ? AND participant_id = ?;",
                         (real, match_id, pid))
            conn.commit()
        db.record_puuid_projects([(real, 0)], source="exact")
        if slot.get("riotIdGameName"):
            db.record_riot_id_observations([(real, slot["riotIdGameName"], slot.get("riotIdTagline") or "")],
                                           project_index=0)
        stats["corregidas"] += 1
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Corrige PUUIDs asignados a la persona equivocada.")
    ap.add_argument("--dry-run", action="store_true", help="Solo informa (igual consulta a Riot)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--lado", choices=["alias", "primario", "ambos"], default="ambos",
                    help="Qué lado de player_puuid_aliases contrastar con el mapa (por defecto ambos)")
    ap.add_argument("--filas-de-puuid", type=str, default=None, metavar="PUUID",
                    help="Corrige fila por fila un PUUID que representa a dos personas")
    args = ap.parse_args()

    from src.riot_client import RiotAPIClient
    p0_key = next((k for k in RIOT_API_KEYS if project_index_for_key(k) == 0), None)
    if not p0_key:
        print("[X] Hace falta la key de P0 en el .env: es la que devuelve el PUUID de P0 del dueño.")
        sys.exit(1)
    client = RiotAPIClient(api_keys=[p0_key], project_indices=[0])

    if args.filas_de_puuid:
        db = DatabaseManager(DEFAULT_DB_PATH)
        st = fix_rows_for_puuid(db, client, args.filas_de_puuid)
        n = db.materialize_player_id_map()
        print(f"[+] Filas: {st['filas']} | corregidas: {st['corregidas']} | sin respuesta: "
              f"{st['sin_respuesta']} | player_id_map recalculado ({n:,} PUUIDs).")
        return

    conn = sqlite3.connect(f"file:{Path(DEFAULT_DB_PATH).resolve()}?mode=ro", uri=True, timeout=60.0)
    attach_for_participants(conn)
    person = dict(conn.execute("SELECT puuid, player_id FROM temp.player_id_map"))
    casos = find_contradictions(conn, args.lado)[: args.limit]
    print(f"[*] Contradicciones mapa vs alias: {len(casos)}")

    decisiones = collections.Counter()
    a_corregir = []
    for i, (puuid, cid, pj) in enumerate(casos, 1):
        res = resolve_owner(conn, client, puuid)
        if res is None:
            decisiones["sin_respuesta"] += 1
            continue
        match_id, owner_p0, owner_name = res
        kind = classify(person.get(puuid), person.get(owner_p0))
        decisiones[kind] += 1
        if kind == "mal_asignado":
            a_corregir.append((puuid, cid, pj, owner_p0, owner_name, person.get(owner_p0)))
        if i % 50 == 0:
            print(f"\r[*] Verificados {i}/{len(casos)}: {dict(decisiones)}", end="", flush=True)
    conn.close()
    print(f"\n[*] Resultado: {dict(decisiones)}")
    for puuid, cid, pj, _o, name, owner in a_corregir[:10]:
        print(f"    {puuid[:10]} (P{pj}): mapa decía '{cid}', Riot dice '{name}' -> persona '{owner}'")
    if args.dry_run:
        print("[dry-run] No se escribió nada.")
        return
    db = DatabaseManager(DEFAULT_DB_PATH)
    for puuid, cid, pj, owner_p0, name, _owner in a_corregir:
        apply_fix(db, puuid, cid, pj, owner_p0, name)
    n = db.materialize_player_id_map()
    print(f"[+] Corregidos {len(a_corregir)} PUUIDs; player_id_map recalculado ({n:,} PUUIDs).")


if __name__ == "__main__":
    main()
