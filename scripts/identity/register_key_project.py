"""
Registro dinámico de proyectos de API Key y resolución de PUUIDs por jugador.

Contexto: Riot encripta el PUUID como HMAC(puuid_global, project_secret). Cada
proyecto de Riot Developer (cada posición del pool RIOT_API_KEYS) devuelve un
PUUID distinto para el MISMO humano. Este script:

  1. Registra el pool actual de keys en `api_projects` (índice = posición en la
     lista, estable aunque la key rote cada 24h; solo se guarda hint enmascarado).
  2. Detecta por fingerprint si una key "nueva" es en realidad una rotación de un
     proyecto ya conocido (compara PUUIDs muestrales) y lo advierte.
  3. Recorre cada jugador del roster en CADA proyecto, resuelve su identidad
     verdadera vía Account-V1 (gameName#tagLine, estable entre proyectos) y la
     guarda en `player_identities` + `player_puuid_map` (un PUUID por proyecto).
  4. Reporta cupos del padrón (CH=200, GM=500 exactos) e invariante
     (n_puuids por identidad == n_proyectos). SOLO ADVIERTE, nunca bloquea.

Uso:
    python scripts/register_key_project.py --api-keys KEY1,KEY2
    python scripts/register_key_project.py   # usa RIOT_API_KEYS de .env
    python scripts/register_key_project.py --tiers CHALLENGER,GRANDMASTER
    python scripts/register_key_project.py --include-diamond --dry-run

El hot path del crawler NO hace requests extra: solo lee el mapa local O(1).
Toda la resolución costosa vive en este script de mantenimiento (batch, idempotente,
reanaduble: los PUUIDs ya conocidos se omiten sin requests).
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Agregar raíz del proyecto a sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tqdm import tqdm

from src.config import (
    APEX_TIERS,
    DEFAULT_DB_PATH,
    DEFAULT_PLATFORM,
    DEFAULT_REGIONAL_ROUTING,
    RIOT_API_KEYS,
)
from src.database import DatabaseManager
from src.riot_client import RiotAPIClient


# ---------------------------------------------------------------------------
# Fingerprint de proyecto (función pura, testeable sin red)
# ---------------------------------------------------------------------------

def fingerprint_match(sample_puuids: Set[str],
                      stored_by_project: Dict[int, Set[str]]) -> Tuple[Optional[int], float]:
    """Compara PUUIDs muestrales de una key contra los PUUIDs almacenados por proyecto.

    Retorna (mejor_project_index, jaccard). Jaccard ~1.0 => la key pertenece a ese
    proyecto (rotación de key, NO proyecto nuevo). Cercano a 0 en todos => proyecto nuevo.
    """
    best_idx: Optional[int] = None
    best_score = 0.0
    sample = set(sample_puuids or [])
    if not sample:
        return None, 0.0
    for idx, stored in stored_by_project.items():
        stored = set(stored or [])
        if not stored:
            continue
        union = sample | stored
        score = len(sample & stored) / len(union) if union else 0.0
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx, best_score


def build_stored_by_project(db: DatabaseManager, limit_per_project: int = 300) -> Dict[int, Set[str]]:
    """Muestra de PUUIDs almacenados por proyecto para el fingerprint (lectura local)."""
    stored: Dict[int, Set[str]] = {}
    with db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT project_index FROM api_projects ORDER BY project_index;")
        indices = [r[0] for r in cursor.fetchall()]
        for idx in indices:
            cursor.execute(
                "SELECT puuid FROM player_puuid_map WHERE project_index = ? LIMIT ?;",
                (idx, limit_per_project),
            )
            stored[idx] = {r[0] for r in cursor.fetchall()}
    return stored


# ---------------------------------------------------------------------------
# Resolución de identidades (batch, con red; idempotente y reanudable)
# ---------------------------------------------------------------------------

def probe_stored_decryptable(db, client: RiotAPIClient, n: int = 5) -> bool:
    """Verifica con una muestra que las keys actuales descifren los PUUIDs guardados.

    Si ningún PUUID almacenado es legible (proyectos muertos tras rotar keys),
    la resolución por Account-V1 quemaría miles de requests en vano: abortar y
    derivar a scripts/migrate_dead_puuids.py (solapamiento de partidas).
    """
    with db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT puuid FROM players LIMIT ?;", (max(n, 5),))
        sample = [r[0] for r in cursor.fetchall() if r[0]]
    if not sample:
        return True  # padrón vacío: nada que verificar
    ok = 0
    for puuid in sample[:n]:
        if client.get_account_by_puuid(puuid):
            ok += 1
    print(f"[*] Probe de descifrado: {ok}/{min(n, len(sample))} PUUIDs legibles "
          f"con el pool actual.")
    return ok > 0
def resolve_roster_for_project(db: DatabaseManager, api_key: str, project_index: int,
                               tiers: List[str], include_diamond: bool,
                               dry_run: bool = False) -> Dict[str, int]:
    """Recorre el roster de UN proyecto y registra cada (jugador, PUUID).

    Retorna contadores {fetched, already_known, resolved, unresolved}.
    Los PUUIDs ya mapeados se omiten sin ningún request (O(1) local).
    """
    client = RiotAPIClient(api_keys=[api_key], platform=DEFAULT_PLATFORM,
                           regional_routing=DEFAULT_REGIONAL_ROUTING)
    stats = {"fetched": 0, "already_known": 0, "resolved": 0, "unresolved": 0}

    entries: List[Dict] = []
    for tier in tiers:
        batch = client.get_apex_league(tier)
        print(f"    [P{project_index}] {tier}: {len(batch)} entradas.")
        entries.extend(batch)

    if include_diamond:
        from src.config import DIAMOND_DIVISIONS
        for division in DIAMOND_DIVISIONS:
            page = 1
            while True:
                batch = client.get_diamond_division_entries(division, page=page)
                if not batch:
                    break
                entries.extend(batch)
                page += 1
            print(f"    [P{project_index}] DIAMOND {division}: acumulado {len(entries)}.")

    stats["fetched"] = len(entries)
    pbar = tqdm(entries, desc=f"[P{project_index}] Resolviendo identidades", unit="jug")
    for e in pbar:
        puuid = e.get("puuid")
        if not puuid:
            stats["unresolved"] += 1
            continue
        known = db.resolve_canonical_id(puuid)
        # Solo se omite si ya tiene fila LIVE: un PUUID conocido únicamente por
        # alias legacy (ej. P1-live apuntando a primario muerto) DEBE procesarse
        # para darle su fila en player_puuid_map y completar el invariante.
        # Si se omitiera, la traducción mapearía escrituras live -> PUUIDs muertos.
        if known and db.has_live_puuid(puuid):
            # Ya mapeado: refrescar stats de liga sin requests.
            if not dry_run:
                with db._get_connection() as conn:
                    cur = conn.cursor()
                    cur.execute("""
                        UPDATE player_identities SET tier=?, rank=?, league_points=?,
                            wins=?, losses=?, updated_at=CURRENT_TIMESTAMP
                        WHERE canonical_id=?;
                    """, (e.get("tier", "UNKNOWN"), e.get("rank", "I"),
                          int(e.get("league_points", 0)), int(e.get("wins", 0)),
                          int(e.get("losses", 0)), known))
                    conn.commit()
            stats["already_known"] += 1
            continue
        ident = client.get_account_by_puuid(puuid)
        if not ident:
            stats["unresolved"] += 1
            continue
        game_name, tag_line = ident
        if not dry_run:
            db.upsert_player_identity(
                game_name, tag_line, puuid, project_index,
                tier=e.get("tier", "UNKNOWN"), rank=e.get("rank", "I"),
                league_points=int(e.get("league_points", 0)),
                wins=int(e.get("wins", 0)), losses=int(e.get("losses", 0)),
                summoner_id=e.get("summoner_id", ""))
        stats["resolved"] += 1
        pbar.set_postfix({"nuevos": stats["resolved"], "conocidos": stats["already_known"]})
    return stats


def check_and_report(db: DatabaseManager) -> None:
    """Reporte warn-only de cupos e invariante multi-proyecto."""
    print("\n  --- CUPOS DEL PADRÓN (referencia: CH=200, GM=500) ---")
    for tier, info in db.check_roster_quotas().items():
        flag = "[OK]" if info["ok"] else "[AVISO]"
        print(f"  {flag} {tier:<12}: actual={info['actual']:,} esperado={info['expected']:,} :: {info['note']}")
    cov = db.get_identity_coverage()
    print("\n  --- INVARIANTE MULTI-PROYECTO (n_puuids == n_proyectos) ---")
    print(f"  Proyectos registrados: {cov['num_projects']} | Identidades: {cov['total_identities']} | "
          f"Completas: {cov['complete']} | Incompletas: {cov['incomplete']}")
    for item in cov["incomplete_detail"][:10]:
        cid = str(item['canonical_id']).encode("ascii", "replace").decode("ascii")
        print(f"    [AVISO] {cid}: {item['n_puuids']} PUUIDs "
              f"(se esperaban {cov['num_projects']})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Registra proyectos de API Key y resuelve PUUIDs por jugador (Account-V1).")
    parser.add_argument("--api-keys", type=str, default=None,
                        help="Keys separadas por comas (default: RIOT_API_KEYS de .env)")
    parser.add_argument("--tiers", type=str, default=",".join(APEX_TIERS),
                        help="Tiers apex a resolver (default: CHALLENGER,GRANDMASTER,MASTER)")
    parser.add_argument("--include-diamond", action="store_true",
                        help="Incluir barrido paginado de Diamond I-IV (lento: ~150+ requests por proyecto)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Solo diagnostica (fingerprint + conteos) sin escribir identidades")
    parser.add_argument("--skip-fingerprint", action="store_true",
                        help="Omitir detección de rotación vs proyecto nuevo")
    args = parser.parse_args()

    if args.api_keys:
        keys = [k.strip() for k in args.api_keys.split(",") if k.strip()]
    else:
        keys = list(RIOT_API_KEYS)
    keys = [k for k in keys if k and "TU-CLAVE" not in k and "XXXXXXXX" not in k]
    if not keys:
        print("[X] Sin API keys válidas (.env RIOT_API_KEYS o --api-keys).")
        sys.exit(1)

    tiers = [t.strip().upper() for t in args.tiers.split(",") if t.strip()]
    db = DatabaseManager(DEFAULT_DB_PATH)

    print("=" * 65)
    print("  REGISTRO MULTI-PROYECTO Y RESOLUCIÓN DE IDENTIDADES")
    print("=" * 65)
    print(f"[*] Keys en pool: {len(keys)} | Tiers: {tiers} "
          f"| Diamond: {'sí' if args.include_diamond else 'no'} | Dry-run: {args.dry_run}")

    stored_before = build_stored_by_project(db)

    if not args.dry_run:
        indices = db.register_api_projects(keys)
        print(f"[+] Proyectos registrados (índices): {indices}")
    else:
        indices = list(range(len(keys)))

    # Fingerprint: ¿rotación de proyecto conocido o proyecto genuinamente nuevo?
    if not args.skip_fingerprint:
        print("\n[*] Fingerprint de proyectos (muestra Challenger, 1 request por key)...")
        for pos, key in enumerate(keys):
            probe = RiotAPIClient(api_keys=[key], platform=DEFAULT_PLATFORM,
                                  regional_routing=DEFAULT_REGIONAL_ROUTING)
            sample_entries = probe.get_apex_league("CHALLENGER")
            sample = {e.get("puuid") for e in sample_entries if e.get("puuid")}
            best, score = fingerprint_match(sample, stored_before)
            if best is not None and score > 0.5:
                print(f"    [P{pos}] key {DatabaseManager.mask_key_hint(key)}: "
                      f"Jaccard={score:.2f} con proyecto {best} -> probable ROTACIÓN "
                      f"(misma ranura, no proyecto nuevo).")
                if best != pos:
                    print(f"    [AVISO] Reordená el pool para que esta key ocupe la posición {best}.")
            else:
                print(f"    [P{pos}] key {DatabaseManager.mask_key_hint(key)}: "
                      f"sin coincidencia (mejor Jaccard={score:.2f}) -> proyecto NUEVO o mapa vacío.")

    # Resolución por proyecto y por jugador
    totals = {"fetched": 0, "already_known": 0, "resolved": 0, "unresolved": 0}
    probe_client = RiotAPIClient(api_keys=keys, platform=DEFAULT_PLATFORM,
                                 regional_routing=DEFAULT_REGIONAL_ROUTING)
    if not args.dry_run and not probe_stored_decryptable(db, probe_client):
        print("\n[X] Las keys actuales NO descifran los PUUIDs almacenados "
              "(proyectos muertos).")
        print("    No se continúa para no quemar cuota en vano. Ejecutá:")
        print("      python scripts/migrate_dead_puuids.py --tiers CHALLENGER,GRANDMASTER")
        sys.exit(2)
    for pos, key in enumerate(keys):
        print(f"\n[*] Proyecto {pos}: resolviendo roster...")
        st = resolve_roster_for_project(db, key, pos, tiers,
                                        include_diamond=args.include_diamond,
                                        dry_run=args.dry_run)
        for k in totals:
            totals[k] += st[k]
        print(f"    -> fetched={st['fetched']} conocidos={st['already_known']} "
              f"nuevos={st['resolved']} sin resolver={st['unresolved']}")

    print(f"\n[+] Totales: {totals}")
    if not args.dry_run:
        db.reload_translation_cache()
    check_and_report(db)
    print("=" * 65)


if __name__ == "__main__":
    main()
