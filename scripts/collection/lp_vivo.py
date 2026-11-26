#!/usr/bin/env python3
"""
scripts/collection/lp_vivo.py
=============================
Mide en vivo cuánto LP mueve cada partida en Master+ de LAS, BR, NA y EUW, en una base
aparte: `data/lp_vivo/lp_vivo.db` (PROJECT_SUMMARY.md §9.S.22). No toca `lol_las.db`.

Usa una sola key: Riot cifra los PUUID por proyecto, así que el ladder y las partidas
tienen que venir del mismo. Por defecto P1, para dejarle P0 al crawler. Toda petición pasa
por `RiotRateLimiter` (AGENTS.md §3.B). Se puede cortar y retomar: las partidas jugadas
durante el corte quedan como eventos `varias`, sin partida enlazada.

Uso:
    python scripts/collection/lp_vivo.py                     # sin límite, Ctrl+C para cortar
    python scripts/collection/lp_vivo.py --minutos 10 --intervalo 120
    python scripts/collection/lp_vivo.py --resumen           # estado de la base, sin red
    python scripts/collection/lp_vivo.py --exportar          # data/lp_vivo/lp_por_partida.csv
"""

import argparse
import sqlite3
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import APEX_TIERS, DATA_DIR, RIOT_API_KEYS, project_index_for_key  # noqa: E402
from src.lp_vivo import PLATAFORMAS, AlmacenLP, agregar_lobby, ciclo_ladder, resolver_evento  # noqa: E402
from src.riot_client import RiotAPIClient  # noqa: E402

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

RUTA = DATA_DIR / "lp_vivo" / "lp_vivo.db"


def ahora_ms() -> int:
    return int(time.time() * 1000)


def trabajar(platform: str, key: str, proyecto: int, partidas, almacen: AlmacenLP, intervalo: float,
             fin: float, parar: threading.Event) -> None:
    """Lee el ladder cada `intervalo` segundos y dedica el tiempo que sobra a buscar partidas."""
    cliente = RiotAPIClient(api_keys=[key], project_indices=[proyecto], platform=platform,
                            regional_routing=PLATAFORMAS[platform])
    previo, t_previo = almacen.cargar_estado(platform)
    while not parar.is_set() and time.time() < fin:
        t0 = time.time()
        t_previo, r = ciclo_ladder(cliente, almacen, platform, previo, t_previo, APEX_TIERS, proyecto, ahora_ms)
        print(f"[{time.strftime('%H:%M:%S')}] {platform}: {r}", flush=True)
        proxima = t0 + intervalo
        while not parar.is_set() and time.time() < min(proxima, fin):
            evs = almacen.pendientes(platform, ahora_ms(), limite=1)
            if not evs:
                parar.wait(min(5.0, max(0.0, proxima - time.time())))
                continue
            resolver_evento(partidas, almacen, evs[0], ahora_ms())


def exportar(ruta: Path) -> None:
    """Una fila por jugador y partida enlazada, con el LP de antes y de después."""
    from src.lp_vivo import cargar_enlazadas
    d = cargar_enlazadas(ruta)
    destino = ruta.parent / "lp_por_partida.csv"
    d.to_csv(destino, index=False)
    print(f"[+] {destino}: {len(d):,} filas, {d['match_id'].nunique():,} partidas")


def main() -> None:
    ap = argparse.ArgumentParser(description="LP por partida en vivo, Master+ de LAS, BR, NA y EUW")
    ap.add_argument("--minutos", type=float, default=0, help="0 = sin límite")
    ap.add_argument("--intervalo", type=float, default=180, help="segundos entre lecturas del ladder")
    ap.add_argument("--proyecto", type=int, default=1, help="proyecto de Riot de la key a usar")
    ap.add_argument("--plataformas", default=",".join(PLATAFORMAS))
    ap.add_argument("--resumen", action="store_true")
    ap.add_argument("--exportar", action="store_true")
    args = ap.parse_args()

    almacen = AlmacenLP(RUTA)
    if args.resumen:
        for platform, datos in sorted(almacen.resumen().items()):
            print(platform, datos)
        return
    if args.exportar:
        exportar(RUTA)
        return

    key = next((k for k in RIOT_API_KEYS if project_index_for_key(k) == args.proyecto), None)
    if not key:
        sys.exit(f"[X] No hay key del proyecto {args.proyecto} en el .env.")
    plataformas = [p.strip().lower() for p in args.plataformas.split(",")]
    # Un cliente por ruta: LAS, BR y NA comparten la cuota de Match-V5 de `americas`.
    por_ruta = {r: RiotAPIClient(api_keys=[key], project_indices=[args.proyecto], regional_routing=r)
                for r in {PLATAFORMAS[p] for p in plataformas}}
    fin = time.time() + args.minutos * 60 if args.minutos else float("inf")
    parar = threading.Event()
    hilos = [threading.Thread(target=trabajar, daemon=True, name=p,
                              args=(p, key, args.proyecto, por_ruta[PLATAFORMAS[p]], almacen, args.intervalo, fin, parar))
             for p in plataformas]
    for h in hilos:
        h.start()
    try:
        while any(h.is_alive() for h in hilos):
            time.sleep(1)
    except KeyboardInterrupt:
        print("[i] Cortando...", flush=True)
        parar.set()
        for h in hilos:
            h.join(timeout=30)
    for platform, datos in sorted(almacen.resumen().items()):
        print(platform, datos)


if __name__ == "__main__":
    main()
