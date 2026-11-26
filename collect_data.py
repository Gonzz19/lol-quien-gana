"""
Script principal de ejecución para la recolección de datos de League of Legends (Diamante+ en LAS).
"""

import sys
import argparse
from src.config import (
    RIOT_API_KEYS,
    DEFAULT_PLATFORM, 
    DEFAULT_REGIONAL_ROUTING, 
    DEFAULT_DB_PATH
)
from src.riot_client import RiotAPIClient
from src.database import DatabaseManager
from src.collector import DataCollector


def parse_args():
    parser = argparse.ArgumentParser(
        description="Crawler de partidas y estadísticas de Lol Diamante+ en LAS (SQLite)"
    )
    parser.add_argument(
        "--api-keys",
        type=str,
        default=None,
        help="Claves de Riot API separadas por comas (omite las de .env si se especifica)"
    )
    parser.add_argument(
        "--sync-players",
        action="store_true",
        help="Sincroniza la lista de jugadores Diamante+ desde la API de Riot"
    )

    parser.add_argument(
        "--apex-only",
        action="store_true",
        help="Sincroniza solo ligas Apex (Challenger, GM, Master), omitiendo Diamante"
    )
    parser.add_argument(
        "--collect-matches",
        action="store_true",
        help="Extrae las partidas y estadísticas de los jugadores en la BD"
    )
    parser.add_argument(
        "--matches-per-player",
        type=int,
        default=None,
        help="Fija una cantidad manual de partidas por jugador (omite los umbrales adaptativos por Tier)"
    )

    parser.add_argument(
        "--max-players",
        type=int,
        default=None,
        help="Límite máximo de jugadores a procesar (ideal para pruebas piloto)"
    )
    parser.add_argument(
        "--tier",
        type=str,
        default=None,
        help="Filtrar jugadores por tier (ej: CHALLENGER, GRANDMASTER, MASTER, DIAMOND)"
    )
    parser.add_argument(
        "--duration-minutes",
        type=float,
        default=None,
        help="Límite de tiempo máximo de ejecución en minutos (ej: 10 para 10 minutos)"
    )
    parser.add_argument(
        "--breadth-first",
        action="store_true",
        help="Modo amplitud: recorre jugadores nuevos en lugar de profundizar en los mismos"
    )
    parser.add_argument(
        "--backfill-timelines",
        action="store_true",
        help="Descarga el timeline (@10, @15, @20) para las partidas que ya están en la BD"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Cantidad de hilos de red concurrentes para descarga de partidas (por defecto automático según número de claves)"
    )
    parser.add_argument(
        "--require-canonical",
        action="store_true",
        help="Modo estricto: no recolectar si el mapa multi-proyecto está vacío (evita fragmentar PUUIDs)"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Muestra un resumen del estado actual de la base de datos"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    db = DatabaseManager(DEFAULT_DB_PATH)

    if args.summary:
        summary = db.get_database_summary()
        print("\n" + "="*60)
        print("  ESTADO DE LA BASE DE DATOS (SQLite)")
        print("="*60)
        print(f"  - Archivo: {summary['db_path']}")
        print(f"  - Tamaño en disco: {summary['file_size_mb']} MB")
        print(f"  - Total Jugadores: {summary['total_players']}")
        print(f"    Desglose por Tier: {summary['players_by_tier']}")
        print(f"  - Total Partidas Únicas: {summary['total_matches']}")
        print(f"  - Total Partidas con Timeline: {summary.get('total_matches_with_timeline', 0)}")
        print(f"  - Total Registros de Participantes: {summary['total_participant_records']}")
        if 'coverage' in summary:
            cov = summary['coverage']
            print("\n  - Cobertura de Jugadores (Coverage):")
            for t in ["CHALLENGER", "GRANDMASTER", "MASTER", "DIAMOND", "TOTAL"]:
                st = cov[t]
                tot_p = st['total']
                pct50 = f"{st['c50']/tot_p*100:.1f}%" if tot_p > 0 else "0%"
                pct100 = f"{st['c100']/tot_p*100:.1f}%" if tot_p > 0 else "0%"
                pct200 = f"{st['c200']/tot_p*100:.1f}%" if tot_p > 0 else "0%"
                pct300 = f"{st['c300']/tot_p*100:.1f}%" if tot_p > 0 else "0%"
                print(f"    {t:<12}: Total={tot_p:,} | >=50: {st['c50']} ({pct50}) | >=100: {st['c100']} ({pct100}) | >=200: {st['c200']} ({pct200}) | >=300: {st['c300']} ({pct300})")
        print("="*60 + "\n")
        return

    # Validar API Keys (CLI o .env)
    if args.api_keys:
        active_keys = [k.strip() for k in args.api_keys.split(",") if k.strip()]
    else:
        active_keys = RIOT_API_KEYS

    valid_keys = [k for k in active_keys if k and "TU-CLAVE" not in k and "XXXXXXXX" not in k]
    if not valid_keys:
        print("\n[!] Error: No se encontraron API Keys válidas en el archivo .env ni por línea de comandos.")
        print("    Por favor, abre el archivo .env y configura tu API Key obtenida de:")
        print("    https://developer.riotgames.com/\n")
        sys.exit(1)

    client = RiotAPIClient(
        api_keys=valid_keys,
        platform=DEFAULT_PLATFORM,
        regional_routing=DEFAULT_REGIONAL_ROUTING
    )
    print(f"\n[+] Riot API Client inicializado con {client.num_active_keys} clave(s) activa(s).")
    collector = DataCollector(riot_client=client, db_manager=db)


    # Si no se pasó ningún flag específico, realizar el flujo completo por defecto
    do_all = not (args.sync_players or args.collect_matches or args.backfill_timelines)

    if args.sync_players or do_all:
        include_diamond = not args.apex_only
        collector.sync_players(include_diamond=include_diamond)

    if args.backfill_timelines:
        max_duration_sec = args.duration_minutes * 60.0 if args.duration_minutes else None
        collector.backfill_timelines(max_duration_seconds=max_duration_sec)

    if args.collect_matches or do_all:
        max_duration_sec = args.duration_minutes * 60.0 if args.duration_minutes else None
        collector.collect_matches(
            matches_per_player=args.matches_per_player,
            max_players=args.max_players,
            tier_filter=args.tier,
            max_duration_seconds=max_duration_sec,
            depth_first=not args.breadth_first,
            max_workers=args.workers,
            require_canonical=args.require_canonical
        )


if __name__ == "__main__":
    main()

