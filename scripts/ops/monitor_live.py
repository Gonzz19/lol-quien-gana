"""
LoL Match Crawler - Monitor de Telemetría en Vivo (Rich Terminal UI)
Ejecución: python scripts/ops/monitor_live.py
Presiona Ctrl+C en cualquier momento para salir. El crawler seguirá corriendo en background.
"""

import sys
import time
from contextlib import closing
from pathlib import Path
from datetime import datetime, timezone

# Asegurar UTF-8 en terminales Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich import box

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.database import DatabaseManager, open_readonly_connection  # noqa: E402

DB_PATH = ROOT_DIR / "data" / "lol_las.db"
COVERAGE_REFRESH_SECONDS = 60.0
HEARTBEAT_MAX_AGE_SECONDS = 360.0


def get_stats(conn, initial_matches=None, start_time=None, *, previous=None, db_path=DB_PATH):
    now = time.monotonic()
    stats = dict(previous or {})
    if previous is None or now - previous["slow_refresh_at"] >= COVERAGE_REFRESH_SECONDS:
        with conn:
            conn.execute("BEGIN")
            stats.update(
                total_participants=conn.execute("SELECT COUNT(*) FROM match_participants;").fetchone()[0],
                total_timelines=conn.execute("SELECT COUNT(DISTINCT match_id) FROM match_timeline;").fetchone()[0],
                total_discarded=conn.execute("SELECT COUNT(*) FROM discarded_matches;").fetchone()[0],
                tier_stats=DatabaseManager.coverage_stats_from_connection(conn),
                slow_refresh_at=now,
            )

    total_matches = conn.execute("SELECT COUNT(*) FROM matches;").fetchone()[0]
    db_size_mb = db_path.stat().st_size / (1024 * 1024) if db_path.exists() else 0.0
    elapsed_sec = max(1.0, now - start_time) if start_time is not None else 1.0
    new_in_session = (total_matches - initial_matches) if initial_matches is not None else 0
    historical_avg_per_min = (new_in_session / elapsed_sec) * 60.0

    last_metric = conn.execute("""
        SELECT timestamp, net_matches_per_min
        FROM crawler_metrics_timeline 
        ORDER BY id DESC LIMIT 1;
    """).fetchone()
    is_crawler_active = False
    last_recorded_rate = 0.0
    last_active_time = "N/A"

    if last_metric:
        last_active_time = last_metric[0]
        last_recorded_rate = last_metric[1] or 0.0
        try:
            dt = datetime.strptime(last_active_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            diff_sec = (datetime.now(timezone.utc) - dt).total_seconds()
            is_crawler_active = 0 <= diff_sec <= HEARTBEAT_MAX_AGE_SECONDS
        except (TypeError, ValueError):
            pass

    stats.update(total_matches=total_matches, db_size_mb=db_size_mb, elapsed_sec=elapsed_sec,
                 new_in_session=new_in_session, historical_avg_per_min=historical_avg_per_min,
                 is_crawler_active=is_crawler_active, last_recorded_rate=last_recorded_rate,
                 last_active_time=last_active_time)
    return stats

def render_dashboard(stats):
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=3),
    )

    # Header
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header_text = Text(f"[*] LOL LAS MATCH CRAWLER - TELEMETRIA EN VIVO [*]\nActualizado: {now_str}", style="bold cyan", justify="center")
    layout["header"].update(Panel(header_text, style="cyan", box=box.ROUNDED))

    # Main split
    layout["main"].split_row(
        Layout(name="left", ratio=1),
        Layout(name="right", ratio=1),
    )

    # Tabla 1: Volumetría y Velocidad
    t_speed = Table(title="Metricas de Rendimiento y Base de Datos", style="bold white", expand=True, box=box.ROUNDED)
    t_speed.add_column("Metrica", style="bold yellow")
    t_speed.add_column("Valor Actual", style="bold green", justify="right")

    t_speed.add_row("Total Partidas en BD", f"{stats['total_matches']:,}")
    t_speed.add_row("Total Participantes", f"{stats['total_participants']:,}")
    timeline_pct = (stats['total_timelines'] / stats['total_matches'] * 100) if stats['total_matches'] > 0 else 0
    t_speed.add_row("Cobertura Timeline @10/15/20", f"{stats['total_timelines']:,} ({timeline_pct:.1f}%)")
    t_speed.add_row("Remakes Filtrados (<=300s)", f"{stats['total_discarded']:,}")
    t_speed.add_row("Tamano en Disco (SQLite)", f"{stats['db_size_mb']:.2f} MB")
    t_speed.add_section()
    if stats["is_crawler_active"]:
        t_speed.add_row("Estado del Crawler", "[bold green]● EN EJECUCIÓN (Streaming)[/bold green]")
        t_speed.add_row("Velocidad Neta Actual", f"[bold green]{stats['last_recorded_rate']:.1f} part/min[/bold green]")
        t_speed.add_row("Rendimiento Proyectado", f"{stats['last_recorded_rate'] * 60:,.0f} part/hora")
    else:
        t_speed.add_row("Estado del Crawler", "[bold yellow]○ DETENIDO / INACTIVO[/bold yellow]")
        t_speed.add_row("Última Velocidad en Corrida", f"{stats['last_recorded_rate']:.1f} part/min")
        t_speed.add_row("Velocidad Instantánea", "[dim]0.0 part/min (sin proceso)[/dim]")

    t_speed.add_row("Partidas Nuevas (en sesión)", f"+{stats['new_in_session']:,}")
    mins = stats['elapsed_sec'] / 60.0
    t_speed.add_row("Tiempo Terminal Abierta", f"{mins:.1f} min ({stats['elapsed_sec']:.0f}s)")
    if not stats["is_crawler_active"]:
        t_speed.add_row("Promedio Diluido (Terminal)", f"[dim]{stats['historical_avg_per_min']:.1f} part/min[/dim]")

    layout["left"].update(Panel(t_speed, border_style="green", box=box.ROUNDED))

    # Tabla 2: Cobertura por Tier
    t_cov = Table(title="Cobertura de Jugadores por Tier (Target 50)", style="bold white", expand=True, box=box.ROUNDED)
    t_cov.add_column("Tier", style="bold magenta")
    t_cov.add_column("Total", justify="right")
    t_cov.add_column(">= 50", justify="right", style="bold cyan")
    t_cov.add_column("% 50", justify="right", style="bold yellow")
    t_cov.add_column(">= 100", justify="right")
    t_cov.add_column(">= 200", justify="right")
    t_cov.add_column(">= 300", justify="right")

    tiers_order = ["CHALLENGER", "GRANDMASTER", "MASTER", "DIAMOND", "TOTAL"]
    for t in tiers_order:
        data = stats["tier_stats"].get(t, {"total": 0, "c50": 0, "c100": 0, "c200": 0, "c300": 0})
        pct50 = (data["c50"] / data["total"] * 100) if data["total"] > 0 else 0
        t_cov.add_row(
            t,
            f"{data['total']:,}",
            f"{data['c50']:,}",
            f"{pct50:.1f}%",
            f"{data['c100']:,}",
            f"{data['c200']:,}",
            f"{data['c300']:,}",
        )

    layout["right"].update(Panel(t_cov, border_style="magenta", box=box.ROUNDED))

    # Footer
    footer_text = Text(
        "Solo lectura | Cobertura y volumen detallado cada minuto | Ctrl+C para salir",
        style="italic dim white",
        justify="center"
    )
    layout["footer"].update(Panel(footer_text, style="blue", box=box.ROUNDED))

    return layout

def main():
    console = Console()
    if not DB_PATH.exists():
        console.print(f"[red]Error: No se encontro la base de datos en '{DB_PATH}'.[/red]")
        sys.exit(1)

    try:
        with closing(open_readonly_connection(DB_PATH, timeout=10.0, query_only=True)) as conn:
            start_time = time.monotonic()
            stats = get_stats(conn, start_time=start_time, db_path=DB_PATH)
            initial_matches = stats["total_matches"]
            with Live(render_dashboard(stats), refresh_per_second=1, screen=True) as live:
                while True:
                    time.sleep(2.0)
                    stats = get_stats(conn, initial_matches, start_time, previous=stats, db_path=DB_PATH)
                    live.update(render_dashboard(stats))
    except KeyboardInterrupt:
        console.print("\n[bold green][+] Monitor cerrado. El crawler sigue ejecutandose normalmente en background.[/bold green]")

if __name__ == "__main__":
    main()

