#!/usr/bin/env python3
"""
scripts/collection/secuencia_fin_de_semana.py
=============================================
Recolección desatendida hasta una hora de corte (PROJECT_SUMMARY.md §9.S.11).

Objetivo del usuario: **50 partidas por persona** en Master+ (train) y, aparte,
**partidas recientes repartidas en todos los rangos** (test, ~20 % extra).

Pasos, en orden (pedido del usuario: primero completar Master, después el test);
cada uno es reanudable y se reintenta si se corta:
1. Crawler normal (`run.py`): menos partidas primero, meta de 50 y paginación hacia
   atrás. Sesiones mientras siga bajando la cantidad de Master+ con < 50.
2. Terminar la ventana del 01/09 (`run.py --ventana-desde 2026-09-01 --fase descargar`):
   cobertura pareja de todo Master+ desde el 01/09, la base del test.
3. Ciclos de partidas recientes: cada ciclo lista y descarga desde el día del
   ciclo anterior (`--ventana-desde <día> --fase todo`), hasta la hora de corte.
   Entre ciclos, en vez de dormir con las keys ociosas, **relleno** (pedido del
   usuario, 2026-10-25: la mayor cantidad de partidas, con prioridad GM+): crawler
   normal con meta de `META_RELLENO` partidas por jugador, primero GM y Challenger
   y, si termina antes, Master; dentro de cada grupo, LP alto primero (suave).

Uso:
    python scripts/collection/secuencia_fin_de_semana.py --hasta "2026-10-26 07:00"
    python scripts/collection/secuencia_fin_de_semana.py --hasta "2026-10-26 07:00" --desde-paso 2
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime, timedelta
from contextlib import closing
from pathlib import Path
from typing import Callable, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[2]
LOG = ROOT_DIR / "logs" / "secuencia_fin_de_semana.log"
DB = ROOT_DIR / "data" / "lol_las.db"
PRIMERA_VENTANA = "2026-09-01"
SEGUNDA_VENTANA = "2026-10-22"
MAX_SIN_AVANCE = 3          # intentos seguidos sin progreso antes de pasar al siguiente paso
META_RELLENO = 300          # partidas por jugador en el relleno entre ciclos del paso 3
TIERS_RELLENO = ("CHALLENGER,GRANDMASTER", "MASTER")  # en orden de prioridad
# Orden suave dentro de cada grupo (pedido del usuario, 2026-10-25): LP alto primero
# con peso 0,7, menos partidas con 0,3. Los Master altos juegan menos con Diamond.
PESO_LP_RELLENO = 0.7


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_py(args: List[str], minutos: float) -> int:
    """Corre run.py con su salida al log. `minutos` = tiempo máximo de la corrida."""
    cmd = [sys.executable, "-u", str(ROOT_DIR / "run.py"), *args, "--minutos", f"{max(1.0, minutos):.0f}"]
    log("$ " + " ".join(cmd[2:]))
    with open(LOG, "a", encoding="utf-8") as f:
        return subprocess.call(cmd, cwd=ROOT_DIR, stdout=f, stderr=subprocess.STDOUT,
                               env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})


def pendientes_ventana(desde: str) -> int:
    from src.config import DEFAULT_PLATFORM
    from src.database import open_readonly_connection
    from src.window_crawl import count_pending_matches, window_key, parse_window_date
    key = window_key(parse_window_date(desde), None)
    with closing(open_readonly_connection(DB, query_only=True)) as conn:
        return count_pending_matches(conn, key, DEFAULT_PLATFORM.upper() + "_")


def master_bajo_50() -> int:
    sys.path.insert(0, str(ROOT_DIR))
    from src.database import DatabaseManager
    rows = DatabaseManager(DB).get_canonicals_for_deep_crawl(
        tier="CHALLENGER,GRANDMASTER,MASTER", min_matches_target=50, only_incomplete=True)
    return len(rows)


def repetir_mientras_avance(paso: Callable[[], None], medir: Callable[[], int],
                            listo: Callable[[int], bool], minutos_restantes: Callable[[], float],
                            nombre: str) -> None:
    """Corre `paso` mientras `medir()` baje y quede tiempo; corta si `listo` o sin avance."""
    sin_avance, antes = 0, medir()
    while not listo(antes) and minutos_restantes() > 5 and sin_avance < MAX_SIN_AVANCE:
        paso()
        ahora = medir()
        log(f"{nombre}: {antes:,} -> {ahora:,}")
        sin_avance = sin_avance + 1 if ahora >= antes else 0
        antes = ahora
        if sin_avance:
            time.sleep(60)  # p. ej. sin red: esperar antes de reintentar


def relleno(minutos: float, restantes: Callable[[], float]) -> None:
    """Usa `minutos` de cuota en el crawler normal, GM+ primero y después Master.

    Si la sesión de un grupo termina antes (todos llegaron a la meta o agotaron el
    historial), el tiempo que sobra pasa al grupo siguiente.
    """
    fin = time.monotonic() + minutos * 60
    for tiers in TIERS_RELLENO:
        quedan = min((fin - time.monotonic()) / 60, restantes() - 30)
        if quedan < 5:
            break
        log(f"Relleno ({tiers}, meta {META_RELLENO}) por {quedan:.0f} min")
        run_py(["--tier", tiers, "--partidas-por-jugador", str(META_RELLENO),
                "--prioridad-lp", str(PESO_LP_RELLENO)], quedan)
    resto = (fin - time.monotonic())
    if resto > 60:
        time.sleep(min(resto, max(0.0, (restantes() - 30) * 60)))


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Recolección desatendida hasta una hora de corte.")
    ap.add_argument("--hasta", required=True, help='Hora de corte local, "AAAA-MM-DD HH:MM"')
    ap.add_argument("--desde-paso", type=int, default=1, choices=[1, 2, 3])
    args = ap.parse_args(argv)
    corte = datetime.strptime(args.hasta, "%Y-%m-%d %H:%M")
    restantes = lambda: (corte - datetime.now()).total_seconds() / 60
    sys.path.insert(0, str(ROOT_DIR))
    log(f"=== Secuencia hasta {corte:%Y-%m-%d %H:%M} (desde el paso {args.desde_paso}) ===")

    if args.desde_paso <= 1:
        log("PASO 1: crawler normal hasta 50 partidas por persona")
        repetir_mientras_avance(lambda: run_py([], restantes()), master_bajo_50,
                                lambda n: n == 0, restantes, "Master+ con < 50")

    if args.desde_paso <= 2:
        log("PASO 2: terminar la ventana del 01/09")
        repetir_mientras_avance(
            lambda: run_py(["--ventana-desde", PRIMERA_VENTANA, "--fase", "descargar"], restantes()),
            lambda: pendientes_ventana(PRIMERA_VENTANA), lambda n: n == 0, restantes, "pendientes 01/09")

    log("PASO 3: ciclos de partidas recientes")
    inicio = SEGUNDA_VENTANA
    antes = None
    while restantes() > 30:
        hoy = datetime.now().strftime("%Y-%m-%d")
        run_py(["--ventana-desde", inicio, "--fase", "todo"], restantes())
        pend = pendientes_ventana(inicio)
        log(f"ciclo desde {inicio}: pendientes {pend:,}")
        if pend == 0:
            inicio, antes = hoy, None  # el próximo ciclo lista solo lo jugado desde hoy
            if restantes() > 30:
                # Dejar acumular partidas nuevas, aprovechando la cuota en el relleno.
                relleno(min(180.0, restantes() - 30), restantes)
        elif antes is not None and pend >= antes:
            time.sleep(60)  # sin avance (p. ej. sin red): esperar antes de reintentar
        else:
            antes = pend
    log("=== Corte: fin de la secuencia ===")


if __name__ == "__main__":
    main()
