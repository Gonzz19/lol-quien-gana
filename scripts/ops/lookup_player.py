#!/usr/bin/env python3
"""
scripts/lookup_player.py
========================
Herramienta de consulta rápida para buscar jugadores por PUUID o por Riot ID (game_name#tag).
Acceso estrictamente de solo lectura (mode=ro) sobre data/lol_las.db.

Uso:
    python scripts/lookup_player.py <PUUID_o_Nombre>
    python scripts/lookup_player.py "nombre#tag"
    python scripts/lookup_player.py PUOUMz-fZgnpRGPeXsys6RQyXQbFyEMnnLPJ6xqXufUsGD
"""
import sys
import sqlite3
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DB_URI = "file:" + str(ROOT_DIR / "data" / "lol_las.db") + "?mode=ro"

def lookup(query_str: str):
    query_str = query_str.strip()
    if not query_str:
        print("[-] Especifica un PUUID o Riot ID para buscar.")
        return

    conn = sqlite3.connect(DB_URI, uri=True)
    c = conn.cursor()

    # 1. Buscar si la query es un PUUID exacto en player_puuid_map o player_puuid_aliases
    sql_by_puuid = """
        SELECT 
            pi.game_name, pi.tag_line, pi.tier, pi.rank, pi.league_points,
            pi.wins, pi.losses, m.canonical_id, m.project_index, m.puuid
        FROM player_puuid_map m
        JOIN player_identities pi ON m.canonical_id = pi.canonical_id
        WHERE m.puuid = ?
        UNION
        SELECT 
            a.game_name, a.tag_line, pi.tier, pi.rank, pi.league_points,
            pi.wins, pi.losses, 
            COALESCE(pi.canonical_id, LOWER(a.game_name || '#' || a.tag_line)),
            -1, a.alias_puuid
        FROM player_puuid_aliases a
        LEFT JOIN player_identities pi ON LOWER(a.game_name || '#' || a.tag_line) = pi.canonical_id
        WHERE a.alias_puuid = ? OR a.primary_puuid = ?
    """
    c.execute(sql_by_puuid, (query_str, query_str, query_str))
    rows = c.fetchall()

    # 2. Si no encontró por PUUID exacto, buscar por nombre o tag
    if not rows:
        like_query = f"%{query_str.lower()}%"
        sql_by_name = """
            SELECT 
                pi.game_name, pi.tag_line, pi.tier, pi.rank, pi.league_points,
                pi.wins, pi.losses, pi.canonical_id, m.project_index, m.puuid
            FROM player_identities pi
            LEFT JOIN player_puuid_map m ON pi.canonical_id = m.canonical_id
            WHERE LOWER(pi.game_name) LIKE ? 
               OR LOWER(pi.canonical_id) LIKE ?
            LIMIT 10
        """
        c.execute(sql_by_name, (like_query, like_query))
        rows = c.fetchall()

    if not rows:
        print(f"[-] No se encontró ningún registro para: '{query_str}'")
        conn.close()
        return

    print("=" * 70)
    print(f"RESULTADOS DE BÚSQUEDA PARA: {query_str}")
    print("=" * 70)
    
    # Agrupar por identidad canónica
    by_identity = {}
    for r in rows:
        gname, tag, tier, rank, lp, wins, losses, cid, p_idx, puuid = r
        if cid not in by_identity:
            by_identity[cid] = {
                "name": f"{gname}#{tag}" if gname else cid,
                "tier": f"{tier} {rank}" if tier else "UNKNOWN",
                "lp": lp or 0,
                "wins": wins or 0,
                "losses": losses or 0,
                "puuids": []
            }
        if puuid:
            by_identity[cid]["puuids"].append((p_idx, puuid))

    for cid, data in by_identity.items():
        total_games = data["wins"] + data["losses"]
        wr = (data["wins"] / total_games * 100) if total_games > 0 else 0
        print(f"\n[+] Invocador: {data['name']}")
        print(f"    Rango:     {data['tier']} ({data['lp']} LP)")
        print(f"    Historial: {data['wins']}W / {data['losses']}L ({wr:.1f}% WR)")
        print(f"    Identidad: {cid}")
        print("    PUUIDs conocidos:")
        for p_idx, p in set(data["puuids"]):
            tag_proj = f"Proyecto {p_idx}" if p_idx >= 0 else "Alias Legacy"
            print(f"      - [{tag_proj}] {p}")
            
        # Consultar cuántas partidas tiene en match_participants
        c.execute("""
            SELECT COUNT(*) FROM match_participants 
            WHERE puuid IN (
                SELECT puuid FROM player_puuid_map WHERE canonical_id = ?
                UNION
                SELECT alias_puuid FROM v_puuid_canonical WHERE canonical_id = ?
            )
        """, (cid, cid))
        games_in_db = c.fetchone()[0]
        print(f"    Partidas en base de datos local: {games_in_db}")

    print("\n" + "=" * 70)
    conn.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python scripts/lookup_player.py <PUUID_o_Nombre>")
        sys.exit(1)
    lookup(" ".join(sys.argv[1:]))

