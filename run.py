"""
run.py — Punto de entrada único: recolectar y validar en la misma corrida.

El problema que resuelve
------------------------
El crawler no está limitado por CPU ni por disco, sino por el rate limit de Riot
(18 req/1 s y 95 req/120 s por key, `AGENTS.md` §3.B). Cada vez que el pool de
claves queda en cooldown, el proceso se queda esperando con la máquina ociosa y
con ~98.000 partidas en la base que nadie auditó nunca.

`run.py` mete la validación **dentro de ese hueco**. Un hilo de fondo mira el
tiempo de espera del pool y, cuando detecta que no hay ranura disponible, usa
esa ventana para auditar una tanda de partidas ya descargadas. Cuando la ranura
se libera, la validación se corta y el crawler sigue. El resultado es que
validar sale prácticamente gratis en throughput: consume tiempo que igual se
estaba perdiendo.

Tres garantías de diseño:

* **La validación nunca escribe en la BD.** Abre en `mode=ro` (`src/validation.py`),
  así que no puede tomar un lock de escritura ni provocar `[WinError 32]`,
  aunque corra en paralelo con el writer del crawler.
* **El trabajo está cuantizado.** Se audita de a rebanadas de `--slice` partidas
  (≈ decenas de ms), para que una rebanada entre completa en la espera más
  corta y no se le robe la ranura al crawler cuando se libera.
* **El progreso es incremental y reanudable.** El cursor vive en
  `data/validation_state.json`; cada corrida retoma donde quedó la anterior y
  las partidas nuevas quedan por delante del cursor.

Uso
---
    python run.py                              # 60 min de crawler + validación oportunista
    python run.py --minutos 240                # sesión larga
    python run.py --ciclos 0                   # ciclos indefinidos (Ctrl+C para cortar)
    python run.py --solo-validar               # sin red: audita todo el backlog y sale
    python run.py --sin-validar                # solo crawler (comportamiento clásico)
    python run.py --minutos 240 --respaldar     # + sincroniza BD y CSVs a Google Drive al cerrar
    python run.py --sync-jugadores --minutos 90
    python run.py --reiniciar-cursor --solo-validar --identidad   # auditoría completa desde cero
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Terminal de Windows: los reportes llevan acentos y símbolos de bloque.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from scripts.ops.check_api_keys import USABLE_STATUSES, check_keys, describe, mask
from src.collector import DataCollector
from src.config import (
    DEFAULT_DB_PATH,
    DEFAULT_PLATFORM,
    DEFAULT_REGIONAL_ROUTING,
    RIOT_API_KEYS,
)
from src.database import DatabaseManager
from src.riot_client import RiotAPIClient
from src.window_crawl import DEFAULT_TIERS, WindowCrawler, parse_window_date
from src.validation import (
    AVISO,
    CHECKS,
    ERROR,
    INFO,
    IncrementalValidator,
    global_checks,
    open_readonly,
)

PLACEHOLDER_MARKERS = ("TU-CLAVE", "XXXXXXXX")


# --------------------------------------------------------------------------
# Validador oportunista
# --------------------------------------------------------------------------
class OpportunisticValidator(threading.Thread):
    """
    Hilo que audita la BD en los huecos que deja el rate limit.

    La decisión de trabajar o no se toma con `_budget()`, que devuelve cuántos
    segundos de validación se pueden gastar *ahora mismo*:

    * si el pool de claves está en cooldown ≥ `stall_threshold`, el presupuesto
      es esa espera (menos un margen): es tiempo que el crawler iba a perder;
    * si no hubo ninguna pasada en `idle_every` segundos, se concede un
      presupuesto mínimo, para que la auditoría avance igual cuando el rate
      limit no aprieta (crawler lento, pocas keys, red con latencia);
    * en cualquier otro caso, cero: la prioridad es la red.

    Sin cliente Riot (`client=None`) el presupuesto es infinito: es el modo
    `--solo-validar`, donde no hay nadie con quien competir.
    """

    def __init__(self, validator: IncrementalValidator,
                 client: Optional[RiotAPIClient] = None,
                 stall_threshold: float = 0.35,
                 idle_every: float = 45.0,
                 idle_budget: float = 0.5,
                 max_budget: float = 5.0,
                 save_every: float = 30.0):
        super().__init__(name="validador-oportunista", daemon=True)
        self.validator = validator
        self.client = client
        self.stall_threshold = stall_threshold
        self.idle_every = idle_every
        self.idle_budget = idle_budget
        self.max_budget = max_budget
        self.save_every = save_every

        self._parar = threading.Event()
        self._last_pass = 0.0
        # Telemetría propia del hilo (se lee recién después del join()).
        self.matches_validated = 0
        self.slices = 0
        self.harvested_seconds = 0.0
        self.stall_windows = 0
        self.caught_up = False
        self.error: Optional[BaseException] = None

    # -- decisión ----------------------------------------------------------
    def _pool_wait(self) -> float:
        """
        Segundos hasta la próxima ranura libre en el pool (0.0 si hay una ya).

        Sin claves vivas la espera es infinita —el crawler no va a pedir nada—,
        así que la auditoría se queda con toda la máquina mientras el supervisor
        intenta renovar el pool.
        """
        if self.client is None:
            return float("inf")
        waits = [ctx.rate_limiter.get_wait_time()
                 for ctx in self.client.key_pool if ctx.is_active]
        return min(waits) if waits else float("inf")

    def _pool_ocioso(self) -> bool:
        """
        True si ninguna clave viva registró una petición en la ventana larga.

        Pasa en dos momentos largos y perfectamente aprovechables: el arranque,
        y sobre todo la exportación de los 4 CSV al cierre de cada ciclo, que
        son varios minutos sin tocar la red. Se consulta **después** de
        `_pool_wait()`, que es quien purga los historiales vencidos.
        """
        if self.client is None:
            return True
        return all(not ctx.rate_limiter.long_history
                   for ctx in self.client.key_pool if ctx.is_active)

    def _budget(self) -> Tuple[float, bool]:
        """(segundos de validación permitidos ahora, si vienen de un stall)."""
        wait = self._pool_wait()
        if wait == float("inf"):
            return self.max_budget, False
        if wait >= self.stall_threshold:
            return min(wait - 0.05, self.max_budget), True
        if self._pool_ocioso():
            return self.max_budget, False
        if (time.time() - self._last_pass) >= self.idle_every:
            return self.idle_budget, False
        return 0.0, False

    # -- ciclo de vida -----------------------------------------------------
    def stop(self) -> None:
        self._parar.set()

    def run(self) -> None:
        next_save = time.time() + self.save_every
        try:
            while not self._parar.is_set():
                budget, from_stall = self._budget()
                if budget <= 0.0:
                    self._parar.wait(0.1)
                    continue

                started = time.monotonic()
                deadline = started + budget
                worked = 0
                while time.monotonic() < deadline and not self._parar.is_set():
                    n = self.validator.run_slice()
                    if n == 0:
                        self.caught_up = True
                        break
                    self.caught_up = False
                    worked += n
                    self.slices += 1

                self._last_pass = time.time()
                self.matches_validated += worked
                if from_stall:
                    self.harvested_seconds += time.monotonic() - started
                    self.stall_windows += 1

                if worked == 0:
                    # Backlog agotado: no tiene sentido reconsultar en bucle
                    # cerrado, las partidas nuevas tardan segundos en llegar.
                    self._parar.wait(min(5.0, self.idle_every))

                if time.time() >= next_save:
                    self.validator.save_state()
                    next_save = time.time() + self.save_every
        except BaseException as err:  # nunca tumbar la recolección por la auditoría
            self.error = err
        finally:
            try:
                self.validator.save_state()
            except Exception:
                pass
            self.validator.close()


# --------------------------------------------------------------------------
# Supervisor del pool de claves
# --------------------------------------------------------------------------
class KeyPoolSupervisor(threading.Thread):
    """
    Vigila el pool de API keys durante toda la corrida y lo repara en caliente.

    Las dev keys de Riot vencen cada 24 h (`AGENTS.md` §3.B), así que en una
    sesión larga es normal que una clave se muera a mitad de camino. El cliente
    la desactiva sola al primer HTTP 401/403, pero eso deja dos agujeros que
    este hilo tapa:

    * **Renovación sin reiniciar.** Cada `interval` segundos relee `.env`; si el
      usuario pegó claves nuevas, se reemplazan **por ranura** (`refresh_pool`),
      que es lo que preserva el proyecto de Riot y con él los PUUIDs ya
      guardados. Una ranura nueva no se agrega sola: sería un proyecto sin
      registrar y mezclaría PUUIDs.
    * **Muerte total del pool.** Si no queda ninguna clave viva, levanta
      `stop_event` y el crawler corta el recorrido en vez de seguir marcando
      jugadores como visitados sin bajarles nada. Si más tarde aparecen claves
      válidas en `.env`, el pool revive y la señal se baja.
    """

    def __init__(self, client: RiotAPIClient, stop_event: threading.Event,
                 env_path: Optional[Path] = None, interval: float = 300.0,
                 reload_env: bool = True):
        super().__init__(name="supervisor-de-claves", daemon=True)
        self.client = client
        self.stop_event = stop_event
        self.env_path = env_path or (ROOT_DIR / ".env")
        self.interval = interval
        self.reload_env = reload_env
        self._parar = threading.Event()
        self.rotaciones = 0
        self.bajas: List[str] = []

    def stop(self) -> None:
        self._parar.set()

    def keys_del_env(self) -> List[str]:
        """Relee las claves del `.env` sin arrastrar el caché de `src.config`."""
        return self.pool_del_env()[0]

    def pool_del_env(self):
        """`(keys, proyectos)` del `.env`, filtrados juntos para no correr las ranuras.

        `proyectos` es `None` si `RIOT_API_PROJECT_INDICES` no está o no tiene una
        entrada por key; entonces el reemplazo cae al criterio por posición.
        """
        try:
            from dotenv import dotenv_values
        except ImportError:
            return [], None
        valores = dotenv_values(self.env_path)
        crudas = (valores.get("RIOT_API_KEYS") or valores.get("RIOT_API_KEY") or "")
        keys = [k.strip() for k in crudas.replace(";", ",").split(",") if k.strip()]
        idx_raw = (valores.get("RIOT_API_PROJECT_INDICES") or "").replace(";", ",")
        if idx_raw.strip():
            try:
                indices = [int(i) for i in idx_raw.split(",") if i.strip()]
            except ValueError:
                return [], None
        else:
            indices = list(range(len(keys)))
        if len(indices) != len(keys):
            return [], None
        pares = [(k, indices[i])
                 for i, k in enumerate(keys) if not any(m in k for m in PLACEHOLDER_MARKERS)]
        keys_ok = [k for k, _ in pares]
        proyectos = [p for _, p in pares] if pares else None
        return keys_ok, proyectos

    def revisar(self) -> None:
        """Una pasada de supervisión. Separada del bucle para poder testearla."""
        for ctx in self.client.key_pool:
            if not ctx.is_active:
                marca = f"ranura {ctx.key_index} ({ctx.masked}): {ctx.deactivated_reason or 'inactiva'}"
                if marca not in self.bajas:
                    self.bajas.append(marca)

        if self.reload_env:
            candidatas, proyectos = self.pool_del_env()
            if candidatas:
                cambios = self.client.refresh_pool(candidatas, projects=proyectos)
                rotadas = cambios["reemplazadas"] + cambios["reactivadas"]
                if rotadas:
                    self.rotaciones += len(rotadas)
                    print(f"\n[+] Pool renovado desde .env: ranuras {sorted(set(rotadas))} "
                          f"al aire otra vez ({self.client.num_active_keys} clave(s) activa(s)).")

        if self.client.num_active_keys == 0:
            if not self.stop_event.is_set():
                print("\n[X] El pool se quedó sin claves vivas. Cortando la recolección; "
                      "la validación del dataset ya descargado sigue corriendo.")
                print(f"    Renová las claves en {self.env_path} y se retoman solas "
                      f"(revisión cada {self.interval:.0f} s).")
            self.stop_event.set()
        elif self.stop_event.is_set() and self.client.num_active_keys > 0:
            print(f"\n[+] Pool operativo de nuevo ({self.client.num_active_keys} clave(s)).")
            self.stop_event.clear()

    def toca_revisar(self, ahora: float, proxima: float) -> bool:
        """
        True si corresponde una pasada de supervisión.

        Además del turno periódico hay un disparo **urgente**: si el cliente ya
        declaró el pool agotado, esperar los 300 s del intervalo dejaría al
        crawler recorriendo el padrón en falso todo ese rato. La condición se
        apaga sola en cuanto `revisar()` levanta la señal de corte.
        """
        if self.client.pool_exhausted.is_set() and not self.stop_event.is_set():
            return True
        return ahora >= proxima

    def run(self) -> None:
        proxima = time.time() + self.interval
        while not self._parar.wait(1.0):
            if not self.toca_revisar(time.time(), proxima):
                continue
            try:
                self.revisar()
            except Exception as err:  # la supervisión nunca tumba la corrida
                print(f"[!] Supervisor de claves: {err!r}")
            proxima = time.time() + self.interval


def print_pool_report(client: RiotAPIClient, supervisor: Optional[KeyPoolSupervisor]) -> None:
    """Bloque de cierre del pool: reparto de carga real y bajas de la sesión."""
    filas = client.pool_status()
    total = sum(f["requests"] for f in filas) or 1
    _rule("POOL DE API KEYS (REPARTO Y VIGENCIA)")
    print(f"  {'Ranura':<8}{'Clave':<22}{'Estado':<12}{'Requests':>10}{'Cuota':>8}"
          f"{'429':>7}{'Errores':>9}{'PUUID ajeno':>13}")
    for f in filas:
        estado = "activa" if f["activa"] else (f["motivo_baja"] or "inactiva")
        print(f"  {f['slot']:<8}{f['key']:<22}{estado:<12}{f['requests']:>10,}"
              f"{f['requests'] / total * 100:>7.1f}%{f['rate_limited']:>7,}{f['errors']:>9,}"
              f"{f.get('puuid_mismatch', 0):>13,}")

    # El 400 "Exception decrypting" no es error de red ni éxito: sin este aviso una
    # sesión entera puede terminar con cero partidas y la telemetría en blanco.
    mismatches = sum(f.get("puuid_mismatch", 0) for f in filas)
    ok = sum(f["ok"] for f in filas)
    if mismatches:
        print(f"\n  [!] {mismatches:,} peticiones rebotaron con 'Exception decrypting': el PUUID "
              f"pertenece a un proyecto de API que este pool no cubre.")
        if mismatches >= max(1, ok):
            print("      Es el modo de falla dominante de esta corrida. El pool no alcanza para "
                  "los PUUIDs guardados; ver la tarea de proyectos 0-2 en TAREAS.md.")
    if supervisor is not None:
        print(f"\n  - Rotaciones de clave en caliente: {supervisor.rotaciones}")
        if supervisor.bajas:
            print("  - Bajas registradas durante la sesión:")
            for baja in supervisor.bajas:
                print(f"      · {baja}")
    print("=" * 65)


# --------------------------------------------------------------------------
# Reportes
# --------------------------------------------------------------------------
def _rule(title: str, width: int = 65) -> None:
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_validation_report(validator: IncrementalValidator,
                            worker: Optional[OpportunisticValidator],
                            db_path: Path,
                            include_identity: bool = False) -> int:
    """
    Imprime el reporte de auditoría y devuelve la cantidad de hallazgos `error`.
    Abre su propia conexión: la del validador pertenece al hilo de fondo.
    """
    conn = open_readonly(db_path)
    try:
        total_matches = conn.execute("SELECT COUNT(*) FROM matches;").fetchone()[0]
        pending = conn.execute("SELECT COUNT(*) FROM matches WHERE rowid > ?;",
                               (validator.state.last_rowid,)).fetchone()[0]
        globals_ = global_checks(conn, include_identity=include_identity)
    finally:
        conn.close()

    st = validator.state
    _rule("REPORTE DE VALIDACIÓN DEL DATASET (SOLO LECTURA)")
    print(f"  - Cursor: rowid {st.last_rowid:,} | partidas auditadas (acumulado): "
          f"{st.matches_validated:,} en {st.slices:,} rebanadas")
    cobertura = (st.matches_validated / total_matches * 100) if total_matches else 0.0
    print(f"  - Cobertura de auditoría: {min(cobertura, 100.0):.1f}% "
          f"({total_matches:,} partidas en BD, {pending:,} sin auditar)")
    if worker is not None:
        print(f"  - Esta sesión: {worker.matches_validated:,} partidas en "
              f"{worker.slices:,} rebanadas")
        print(f"  - Tiempo de rate limit reutilizado: {worker.harvested_seconds:.1f} s "
              f"en {worker.stall_windows:,} ventanas de espera")
        if worker.error is not None:
            print(f"  [!] El hilo de validación abortó: {worker.error!r}")

    grouped = validator.findings_by_severity()
    etiquetas = {ERROR: "ERRORES (invariantes rotos)",
                 AVISO: "AVISOS (anomalías recuperables)",
                 INFO: "INFORMATIVOS (deuda metodológica conocida, AGENTS.md §3.F)"}
    for severity in (ERROR, AVISO, INFO):
        rows = grouped[severity]
        print(f"\n  {etiquetas[severity]}:")
        if not rows:
            print("    (ninguno)")
            continue
        for slug, count, samples in rows:
            desc = CHECKS.get(slug, (severity, ""))[1]
            pct = (count / st.matches_validated * 100) if st.matches_validated else 0.0
            print(f"    - {slug:<26} {count:>8,} ({pct:5.2f}%)  {desc}")
            if samples:
                print(f"      ej.: {', '.join(samples[:3])}")

    print("\n  CHEQUEOS GLOBALES (entre tablas):")
    for name, count in globals_.items():
        marker = "[+]" if count == 0 else "[!]"
        print(f"    {marker} {name:<30} {count:,}")

    print("=" * 65 + "\n")
    return validator.error_count()


# --------------------------------------------------------------------------
# Respaldo a Google Drive
# --------------------------------------------------------------------------
def respaldar_a_drive(cambios: int) -> bool:
    """
    Sincroniza la BD y los 4 CSV a Google Drive al cerrar la sesión.

    Se dispara acá y no en un hook `pre-push` por una razón de costo: el push
    lleva código (la BD está en `.gitignore`, nunca viaja por git), así que
    colgarle ~1,3 GB de copia a cada `git push` castiga los commits de una
    línea. El momento en que la BD realmente cambió es el final de una corrida
    de recolección, que es exactamente acá.

    Por eso mismo `cambios <= 0` omite la subida: una sesión que no guardó nada
    —o una corrida de `--solo-validar`, que es solo lectura— no tiene nada nuevo
    que respaldar, y reescribir 687 MB en Drive para nada no es gratis.
    """
    if cambios <= 0:
        print("\n[i] --respaldar: la corrida no guardó nada nuevo; se omite la subida a Drive.")
        print("    Para forzarla igual: python scripts/backup/backup_to_drive.py")
        return False

    _rule("RESPALDO A GOOGLE DRIVE (DB + 4 CSV)")
    inicio = time.time()
    try:
        from scripts.backup.backup_to_drive import run_backup
        reporte = run_backup()
    except Exception as err:
        print(f"[X] El respaldo falló: {err!r}")
        print("=" * 65)
        return False

    ok = bool(reporte.get("integrity_ok")) and bool(reporte.get("uploaded_latest"))
    if ok:
        print(f"[+] Respaldo completado en {time.time() - inicio:.1f} s "
              f"({len(reporte.get('csvs', []))} CSV sincronizados).")
    elif not reporte.get("integrity_ok"):
        # La política §4.W es explícita: si quick_check falla no se sube nada,
        # para no pisar en la nube una copia sana con una corrupta.
        print("[X] No se subió nada: la verificación de integridad de la BD falló.")
    else:
        print("[X] El respaldo no llegó a subir la BD (ver el detalle de arriba).")
    print("=" * 65)
    return ok


# --------------------------------------------------------------------------
# Preflight de claves
# --------------------------------------------------------------------------
def resolve_keys(cli_keys: Optional[str]) -> List[str]:
    raw = [k.strip() for k in cli_keys.split(",")] if cli_keys else list(RIOT_API_KEYS)
    return [k for k in raw
            if k and not any(marker in k for marker in PLACEHOLDER_MARKERS)]


def preflight_keys(keys: List[str]) -> List[str]:
    """
    Sondea el pool antes de arrancar (`AGENTS.md` §3.B) y devuelve solo las
    claves usables. Una key vencida en el pool no falla ruidosamente: reparte
    404/401 silenciosos durante horas, así que se descarta acá.
    """
    _rule("PREFLIGHT: VIGENCIA DEL POOL DE API KEYS")
    results = check_keys(keys)
    usable = []
    for idx, key in enumerate(keys, 1):
        status = results[key]
        ok = status in USABLE_STATUSES
        print(f"  {'[+]' if ok else '[-]'} Clave #{idx} ({mask(key)}): {describe(status)}")
        if ok:
            usable.append(key)
    if len(usable) < len(keys):
        print(f"\n  [!] {len(keys) - len(usable)} clave(s) descartada(s). "
              f"Renovalas en https://developer.riotgames.com/ y actualizá .env")
    print("=" * 65)
    return usable


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run.py",
        description="Recolección de partidas + validación oportunista del dataset ya descargado.",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    g = p.add_argument_group("Recolección")
    g.add_argument("--minutos", type=float, default=60.0,
                   help="Duración de cada ciclo de crawler (por defecto 60)")
    g.add_argument("--ciclos", type=int, default=1,
                   help="Cantidad de ciclos; 0 = indefinido hasta Ctrl+C (por defecto 1)")
    g.add_argument("--sync-jugadores", action="store_true",
                   help="Sincroniza el padrón Diamante+ antes del primer ciclo")
    g.add_argument("--apex-only", action="store_true",
                   help="Con --sync-jugadores: solo ligas Apex (Challenger/GM/Master)")
    g.add_argument("--backfill-timelines", action="store_true",
                   help="Descarga timelines faltantes en vez de partidas nuevas")
    g.add_argument("--partidas-por-jugador", type=int, default=None,
                   help="Cupo fijo por jugador (omite los umbrales adaptativos por tier)")
    g.add_argument("--prioridad-lp", type=float, default=None, metavar="PESO",
                   help="Ordena la cola por LP alto (PESO, 0-1) y menos partidas (1-PESO); "
                        "por defecto, solo menos partidas primero")
    g.add_argument("--max-jugadores", type=int, default=None,
                   help="Límite de jugadores a procesar (pruebas piloto)")
    g.add_argument("--tier", type=str, default=DEFAULT_TIERS,
                   help=f"Tiers a recolectar, separados por coma (por defecto {DEFAULT_TIERS}: "
                        "Diamond queda fuera como meta de recolección desde 2026-10-22; "
                        "pasá --tier DIAMOND para incluirlo a propósito)")
    g.add_argument("--amplitud", action="store_true",
                   help="Recorrido en amplitud (por defecto: profundidad)")
    g.add_argument("--workers", type=int, default=None,
                   help="Hilos HTTP concurrentes (por defecto, automático según claves)")
    g.add_argument("--require-canonical", action="store_true",
                   help="Modo estricto: aborta si el mapa multi-proyecto está vacío")
    g.add_argument("--api-keys", type=str, default=None,
                   help="Claves separadas por comas (omite las de .env)")
    g.add_argument("--sin-preflight", action="store_true",
                   help="Salta el sondeo de vigencia de claves")
    g.add_argument("--vigilar-keys", type=float, default=300.0,
                   help="Cada cuántos segundos se supervisa y recarga el pool (por defecto 300)")
    g.add_argument("--sin-recarga-de-keys", action="store_true",
                   help="El supervisor no relee .env: solo informa las bajas del pool")
    g.add_argument("--respaldar", action="store_true",
                   help="Al terminar, sincroniza BD y CSVs a Google Drive si la corrida "
                        "guardó algo nuevo (equivale a scripts/backup/backup_to_drive.py)")

    w = p.add_argument_group(
        "Ventana de tiempo",
        "Recolecta a TODOS los jugadores de los tiers elegidos dentro de un rango de "
        "fechas, sin importar cuántas partidas tengan ya (para el split train/test "
        "temporal). Reanudable. No exporta CSV al terminar: eso va en el cierre.")
    w.add_argument("--ventana-desde", type=str, default=None, metavar="AAAA-MM-DD",
                   help="Inicio de la ventana (00:00 hora de Argentina). Activa este modo")
    w.add_argument("--ventana-hasta", type=str, default=None, metavar="AAAA-MM-DD",
                   help="Fin de la ventana, exclusivo (por defecto: hasta ahora)")
    w.add_argument("--fase", choices=["listar", "descargar", "todo", "estado"], default="todo",
                   help="listar = solo IDs y estimación; descargar = solo partidas pendientes; "
                        "todo = ambas; estado = muestra el avance sin pedir nada (por defecto todo)")
    w.add_argument("--ventana-tiers", type=str, default=DEFAULT_TIERS,
                   help=f"Tiers a cubrir (por defecto {DEFAULT_TIERS})")

    p.add_argument("--db", type=str, default=None, metavar="RUTA",
                   help="Base SQLite sobre la que corre todo (por defecto data/lol_las.db). "
                        "Para probar sobre una copia; los CSV del cierre se escriben junto a ella.")

    v = p.add_argument_group("Validación")
    v.add_argument("--sin-validar", action="store_true",
                   help="Solo crawler, sin auditoría de fondo")
    v.add_argument("--solo-validar", action="store_true",
                   help="Solo auditoría (sin red): consume todo el backlog y sale")
    v.add_argument("--slice", type=int, default=300, dest="slice_size",
                   help="Partidas por rebanada de auditoría (por defecto 300)")
    v.add_argument("--umbral-stall", type=float, default=0.35,
                   help="Espera de pool (s) a partir de la cual se valida (por defecto 0.35)")
    v.add_argument("--cada", type=float, default=45.0, dest="idle_every",
                   help="Segundos sin pasada tras los que se valida igual (por defecto 45)")
    v.add_argument("--reiniciar-cursor", action="store_true",
                   help="Borra el cursor y los hallazgos: reaudita la BD desde cero")
    v.add_argument("--identidad", action="store_true",
                   help="Incluye el chequeo global de cobertura de identidad (costoso)")
    v.add_argument("--estricto", action="store_true",
                   help="Sale con código 1 si la auditoría encontró errores")

    return p.parse_args(argv)


def print_window_status(st: dict) -> None:
    _rule(f"VENTANA {st['ventana']}")
    print(f"  - Jugadores en la ventana: {st['jugadores']:,} (listados: {st['listados']:,})")
    print(f"  - IDs de partidas únicos: {st['ids_unicos']:,} "
          f"(ya en la base: {st['ya_en_base']:,}, pendientes: {st['pendientes']:,})")
    print(f"  - Descarga estimada con el pool actual: {st['horas_estimadas']:.1f} h")


def ejecutar_ventana(args: argparse.Namespace, client, db, stop_event: threading.Event,
                     deadline: Optional[float]) -> int:
    """Modo `--ventana-desde`: lista y/o descarga. Devuelve partidas nuevas guardadas."""
    crawler = WindowCrawler(
        client, db, parse_window_date(args.ventana_desde),
        parse_window_date(args.ventana_hasta) if args.ventana_hasta else None,
        tiers=args.ventana_tiers)
    if args.fase == "estado":
        print_window_status(crawler.status())
        return 0
    if args.fase in ("listar", "todo"):
        _rule(f"FASE 1: LISTAR IDs ({crawler.key})")
        st = crawler.list_phase(stop_event=stop_event, deadline=deadline, workers=args.workers)
        print(f"  - Jugadores listados: {st['listados']:,} | fallidos (se reintentan): "
              f"{st['fallidos']:,} | IDs: {st['ids']:,}{' | CORTADO' if st['cortado'] else ''}")
    nuevas = 0
    if args.fase in ("descargar", "todo") and not stop_event.is_set():
        _rule(f"FASE 2: DESCARGAR PARTIDAS ({crawler.key})")
        st = crawler.download_phase(stop_event=stop_event, deadline=deadline, workers=args.workers)
        nuevas = st["matches_new"]
        print(f"  - Pendientes al inicio: {st['pendientes_al_inicio']:,} | nuevas: {nuevas:,} | "
              f"remakes descartados: {st['descartadas']:,} | fallidas (quedan pendientes): "
              f"{st['fallidas']:,}{' | CORTADO' if st['cortado'] else ''}")
    print_window_status(crawler.status())
    return nuevas


def resolve_db_path(args: argparse.Namespace) -> Optional[Path]:
    """Base sobre la que trabaja la corrida: `--db` o `data/lol_las.db`.

    Con `--db` todo lo que la corrida escribe (la base, `validation_state.json` y
    los CSV del cierre) queda junto a esa base, así que una corrida sobre una
    copia no toca `data/` (§9.S.76). Devuelve None si `--db` no existe: crear
    una base vacía por un error de tipeo recolectaría desde cero sin avisar.
    """
    if not args.db:
        return DEFAULT_DB_PATH
    path = Path(args.db).expanduser().resolve()
    if not path.is_file():
        print(f"[X] --db: no existe {path}. Copiá antes la base (ver PROJECT_SUMMARY.md §9.S.76).")
        return None
    print(f"[i] Base de esta corrida: {path} (los CSV del cierre se escriben en {path.parent})")
    return path


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    db_path = resolve_db_path(args)
    if db_path is None:
        return 2
    if args.respaldar and db_path != DEFAULT_DB_PATH:
        print("[X] --respaldar sube data/lol_las.db a Drive: no se combina con --db.")
        return 2

    if args.ventana_desde:
        try:
            desde = parse_window_date(args.ventana_desde)
            if args.ventana_hasta and parse_window_date(args.ventana_hasta) <= desde:
                print("[X] --ventana-hasta tiene que ser posterior a --ventana-desde.")
                return 2
        except ValueError:
            print("[X] Las fechas de la ventana van como AAAA-MM-DD.")
            return 2
        if args.fase == "estado":
            # Sin red: solo lee la base. La estimación de horas usa las keys del .env.
            from types import SimpleNamespace
            sin_red = SimpleNamespace(num_active_keys=max(1, len(resolve_keys(args.api_keys))))
            return ejecutar_ventana(args, sin_red, DatabaseManager(db_path),
                                    threading.Event(), None) and 0

    if args.solo_validar and args.sin_validar:
        print("[X] --solo-validar y --sin-validar se excluyen.")
        return 2

    validator = IncrementalValidator(db_path, slice_size=args.slice_size)
    if args.reiniciar_cursor:
        validator.reset()
        print("[*] Cursor de validación reiniciado: se auditará la BD completa.")

    # ---- Modo auditoría pura: sin red, a toda velocidad ------------------
    if args.solo_validar:
        conn = open_readonly(db_path)
        try:
            pending = conn.execute("SELECT COUNT(*) FROM matches WHERE rowid > ?;",
                                   (validator.state.last_rowid,)).fetchone()[0]
        finally:
            conn.close()
        _rule("AUDITORÍA COMPLETA DEL DATASET DESCARGADO")
        print(f"  - Partidas pendientes de auditar: {pending:,}")
        t0 = time.monotonic()
        done = validator.run_until_done()
        validator.close()
        print(f"  - Auditadas {done:,} partidas en {time.monotonic() - t0:.1f} s")
        errores = print_validation_report(validator, None, db_path,
                                          include_identity=args.identidad)
        if args.respaldar:
            print("[i] --respaldar ignorado: la auditoría es solo lectura y no cambió la BD.")
        return 1 if (args.estricto and errores) else 0

    # ---- Modo normal: crawler + auditoría oportunista ---------------------
    keys = resolve_keys(args.api_keys)
    if not keys:
        print("\n[X] No hay API keys válidas en .env ni en --api-keys.")
        print("    Conseguí una en https://developer.riotgames.com/ y cargala en .env\n")
        return 1
    if not args.sin_preflight:
        keys = preflight_keys(keys)
        if not keys:
            print("\n[X] Ninguna clave del pool está vigente. Renovalas antes de crawlear.\n")
            return 1

    db = DatabaseManager(db_path)
    client = RiotAPIClient(api_keys=keys,
                           platform=DEFAULT_PLATFORM,
                           regional_routing=DEFAULT_REGIONAL_ROUTING)
    collector = DataCollector(riot_client=client, db_manager=db)
    print(f"\n[+] Riot API Client inicializado con {client.num_active_keys} clave(s) activa(s).")

    worker: Optional[OpportunisticValidator] = None
    if not args.sin_validar:
        conn = open_readonly(db_path)
        try:
            pending = conn.execute("SELECT COUNT(*) FROM matches WHERE rowid > ?;",
                                   (validator.state.last_rowid,)).fetchone()[0]
        finally:
            conn.close()
        worker = OpportunisticValidator(validator, client=client,
                                        stall_threshold=args.umbral_stall,
                                        idle_every=args.idle_every)
        worker.start()
        print(f"[+] Validación oportunista activa: {pending:,} partidas pendientes, "
              f"rebanadas de {args.slice_size}, umbral de espera {args.umbral_stall:.2f} s.")

    # Señal de corte compartida: la levanta el supervisor cuando el pool muere,
    # y la mira el collector en cada paso de su recorrido.
    stop_event = threading.Event()
    supervisor = KeyPoolSupervisor(client, stop_event,
                                   interval=args.vigilar_keys,
                                   reload_env=not args.sin_recarga_de_keys)
    supervisor.start()

    max_duration_sec = args.minutos * 60.0 if args.minutos else None
    ciclo = 0
    # Filas nuevas escritas por esta corrida. Decide si el respaldo tiene sentido.
    cambios = 0
    try:
        if args.sync_jugadores:
            cambios += collector.sync_players(include_diamond=not args.apex_only) or 0

        if args.ventana_desde:
            deadline = time.monotonic() + max_duration_sec if max_duration_sec else None
            cambios += ejecutar_ventana(args, client, db, stop_event, deadline)

        while not args.ventana_desde and (args.ciclos == 0 or ciclo < args.ciclos):
            ciclo += 1
            if args.ciclos != 1:
                _rule(f"CICLO #{ciclo} ({args.minutos:.0f} MIN)")

            if args.backfill_timelines:
                cambios += collector.backfill_timelines(
                    max_duration_seconds=max_duration_sec,
                    stop_event=stop_event) or 0
            else:
                reporte_sesion = collector.collect_matches(
                    matches_per_player=args.partidas_por_jugador,
                    max_players=args.max_jugadores,
                    tier_filter=args.tier,
                    max_duration_seconds=max_duration_sec,
                    depth_first=not args.amplitud,
                    max_workers=args.workers,
                    require_canonical=args.require_canonical,
                    stop_event=stop_event,
                    peso_lp=args.prioridad_lp,
                )
                # Los cortes tempranos devuelven el resumen de la BD, sin esta clave.
                cambios += int((reporte_sesion or {}).get("matches_new", 0))
                if (reporte_sesion or {}).get("writer_timed_out"):
                    # Otro ciclo sumaría un segundo escritor sobre una base trabada.
                    print("\n[X] El escritor de la base no terminó: no se encadena otro ciclo.")
                    break

            if stop_event.is_set() and client.num_active_keys == 0:
                print("\n[X] Sin claves vivas: no se encadena otro ciclo.")
                break
    except KeyboardInterrupt:
        print("\n[i] Interrupción manual: cerrando de forma segura...")
        stop_event.set()
    finally:
        supervisor.stop()
        if worker is not None:
            worker.stop()
            worker.join(timeout=30.0)

    print_pool_report(client, supervisor)

    errores = 0
    if worker is not None:
        errores = print_validation_report(validator, worker, db_path,
                                          include_identity=args.identidad)

    # El respaldo va último: sube la BD y los CSV ya exportados por el crawler.
    respaldo_pedido_y_fallido = False
    if args.respaldar:
        respaldo_pedido_y_fallido = not respaldar_a_drive(cambios) and cambios > 0

    if respaldo_pedido_y_fallido:
        return 1
    return 1 if (args.estricto and errores) else 0


if __name__ == "__main__":
    sys.exit(main())
