"""
Recolección por ventana de tiempo (PROJECT_SUMMARY.md §9.S.6).

Para qué
--------
La recolección por meta (`collect_matches`, "hasta N partidas por jugador")
saltea a quien ya llegó a la meta. Para un split temporal train/test eso deja
huecos justo en los jugadores más maduros: sus partidas recientes solo entran si
aparecen en partidas de otro. Acá se cubre **a todos los jugadores de los tiers
elegidos por igual dentro de una ventana de fechas**, sin importar cuántas
partidas tengan ya.

Dos fases, ambas reanudables
----------------------------
1. **Listar** (`list_phase`): por cada jugador, los IDs de sus rankeadas dentro
   de la ventana. Es barato (~1 request cada 100 IDs) y deja el **número exacto**
   de partidas que faltan bajar antes de gastar cuota en descargas. Un jugador
   queda marcado como listado solo si Riot respondió bien: si se cae la red, se
   lo vuelve a listar en la próxima corrida.
2. **Descargar** (`download_phase`): baja las partidas listadas que todavía no
   están en la base. Cada partida anota el proyecto exacto de sus 10 PUUIDs
   (ingesta de §9.S.4). Si una partida no se puede bajar, queda pendiente.
"""

import concurrent.futures
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

# Argentina no tiene horario de verano: un desfase fijo es exacto y evita
# depender de la base de zonas horarias (en Windows `zoneinfo` necesita tzdata).
ART = timezone(timedelta(hours=-3))
DEFAULT_TIERS = "CHALLENGER,GRANDMASTER,MASTER"
MAX_IDS_PER_PLAYER = 2000  # tope de seguridad por jugador y ventana
SAVE_BATCH = 15

_PENDING_MATCHES = """
    FROM crawl_window_matches w
    WHERE w.window_key = ?
      AND substr(w.match_id, 1, ?) = ?
      AND NOT EXISTS (SELECT 1 FROM matches m WHERE m.match_id = w.match_id)
      AND NOT EXISTS (SELECT 1 FROM discarded_matches d WHERE d.match_id = w.match_id)
"""


def count_pending_matches(conn, key: str, platform_prefix: str) -> int:
    return conn.execute("SELECT COUNT(*) " + _PENDING_MATCHES,
                        (key, len(platform_prefix), platform_prefix)).fetchone()[0]


def parse_window_date(value: str) -> int:
    """`"2026-10-01"` -> segundos epoch de ese día a las 00:00 hora de Argentina."""
    day = datetime.strptime(value.strip(), "%Y-%m-%d").replace(tzinfo=ART)
    return int(day.timestamp())


def window_key(start: int, end: Optional[int]) -> str:
    """Clave estable de la ventana, legible: `2026-10-01..2026-10-25` o `2026-10-01..`."""
    fmt = lambda ts: datetime.fromtimestamp(ts, ART).strftime("%Y-%m-%d")
    return f"{fmt(start)}..{fmt(end) if end else ''}"


class WindowCrawler:
    """Recolección de todos los jugadores de `tiers` dentro de `[start, end)`.

    `start`/`end` en segundos epoch (lo que espera Match-V5). `end=None` = hasta
    ahora. La clave de la ventana no incluye `end` cuando es abierta, así que
    una ventana "desde X" se puede extender corrida tras corrida.
    """

    def __init__(self, client, db, start: int, end: Optional[int] = None,
                 tiers: str = DEFAULT_TIERS, platform: Optional[str] = None):
        if end is not None and end <= start:
            raise ValueError("La ventana termina antes de empezar.")
        self.client = client
        self.db = db
        self.start = int(start)
        self.end = int(end) if end is not None else None
        self.tiers = tiers
        self.key = window_key(self.start, self.end)
        # Solo se descargan partidas del servidor del proyecto (LAS = LA2). Match-V5
        # devuelve las partidas de un jugador en todos los servidores de su región
        # (BR1, NA1, LA1 también): 441 de 139.669 en la ventana del 01/09.
        from src.config import DEFAULT_PLATFORM
        self.platform_prefix = (platform or DEFAULT_PLATFORM).upper() + "_"


    # ------------------------------------------------------------------ #
    # Estado                                                             #
    # ------------------------------------------------------------------ #
    def roster(self) -> List[Dict[str, Any]]:
        """Una fila por persona del padrón en `tiers`, con sus PUUIDs por proyecto."""
        rows = self.db.get_canonicals_for_deep_crawl(
            tier=self.tiers, min_matches_target=10**9, only_incomplete=False)
        seen, out = set(), []
        for r in rows:
            if r["canonical_id"] not in seen:
                seen.add(r["canonical_id"])
                out.append(r)
        return out

    def listed_players(self) -> set:
        with self.db._get_connection() as conn:
            return {r[0] for r in conn.execute(
                "SELECT player_id FROM crawl_window_players WHERE window_key = ?;", (self.key,))}

    def relist(self, player_ids) -> int:
        """Desmarca jugadores ya listados para que la próxima `list_phase` los repita.

        Sirve para registrar sus partidas en `crawl_window_listings` cuando se
        listaron antes de que esa tabla existiera. Los IDs ya guardados no se
        duplican (INSERT OR IGNORE).
        """
        ids = list(player_ids)
        with self.db._get_connection() as conn:
            conn.executemany("DELETE FROM crawl_window_players WHERE window_key = ? AND player_id = ?;",
                             [(self.key, p) for p in ids])
            conn.commit()
        return len(ids)

    def pending_match_ids(self) -> List[str]:
        """Partidas listadas en la ventana que todavía no están (ni descartadas) en la base."""
        with self.db._get_connection() as conn:
            return [r[0] for r in conn.execute("SELECT w.match_id " + _PENDING_MATCHES + " ORDER BY w.match_id;",
                                              (self.key, len(self.platform_prefix), self.platform_prefix))]

    def status(self) -> Dict[str, Any]:
        """Foto de la ventana: jugadores listados, IDs, pendientes y horas estimadas."""
        roster_n = len(self.roster())
        with self.db._get_connection() as conn:
            listed = conn.execute("SELECT COUNT(*) FROM crawl_window_players WHERE window_key = ?;",
                                  (self.key,)).fetchone()[0]
            ids = conn.execute("SELECT COUNT(*) FROM crawl_window_matches WHERE window_key = ?;",
                               (self.key,)).fetchone()[0]
            pending = count_pending_matches(conn, self.key, self.platform_prefix)
        keys = max(1, int(getattr(self.client, "num_active_keys", 1) or 1))
        # 2 requests por partida (detalle + timeline); ~49,5 req/min por key (99 / 120 s).
        hours = pending * 2 / (49.5 * keys) / 60.0
        return {"ventana": self.key, "jugadores": roster_n, "listados": listed,
                "ids_unicos": ids, "ya_en_base": ids - pending, "pendientes": pending,
                "horas_estimadas": round(hours, 1)}

    # ------------------------------------------------------------------ #
    # Fase 1: listar                                                     #
    # ------------------------------------------------------------------ #
    def list_phase(self, stop_event: Optional[threading.Event] = None,
                   deadline: Optional[float] = None,
                   max_ids_per_player: int = MAX_IDS_PER_PLAYER,
                   workers: Optional[int] = None) -> Dict[str, Any]:
        """Lista los IDs de la ventana para cada jugador todavía no listado.

        Concurrente: con un solo hilo, cada request espera su respuesta (~1,2 s
        de ida y vuelta) y el pool queda la mitad del tiempo ocioso (48 req/min
        medidos contra 99 de cuota con 2 keys). El limitador de cada key sigue
        mandando sobre la cuota; los hilos solo tapan la latencia. Las
        escrituras a la base van todas por el hilo principal.
        """
        done = self.listed_players()
        todo = [r for r in self.roster() if r["canonical_id"] not in done]
        keys = max(1, int(getattr(self.client, "num_active_keys", 1) or 1))
        workers = workers or min(16, max(2, keys * 4))
        feedback: List[Tuple[str, int, bool]] = []
        lock = threading.Lock()

        def collect(puuid, project, ok):
            with lock:
                feedback.append((puuid, project, ok))

        def flush():
            with lock:
                pending = list(feedback)
                feedback.clear()
            if pending:
                self.db.apply_project_feedback(pending)

        def fetch(row):
            # Corre en un hilo trabajador: `last_request_ok()` es por hilo, así
            # que se lee acá, inmediatamente después de su propia petición.
            ids = self.client.get_match_ids_by_puuid(
                puuid=row["puuid"], count=max_ids_per_player, start=0,
                start_time=self.start, end_time=self.end,
                candidates=dict(row.get("candidates") or {}) or None)
            return row, list(ids or []), _request_ok(self.client)

        def store(row, ids, ok):
            with self.db._get_connection() as conn:
                conn.executemany(
                    "INSERT OR IGNORE INTO crawl_window_matches (window_key, match_id) VALUES (?, ?);",
                    [(self.key, m) for m in ids])
                conn.executemany(
                    "INSERT OR IGNORE INTO crawl_window_listings (window_key, players_puuid, match_id) "
                    "VALUES (?, ?, ?);", [(self.key, row.get("players_puuid") or row["puuid"], m) for m in ids])
                if ok:
                    conn.execute(
                        "INSERT OR REPLACE INTO crawl_window_players (window_key, player_id, n_ids) "
                        "VALUES (?, ?, ?);", (self.key, row["canonical_id"], len(ids)))
                conn.commit()
            stats["ids"] += len(ids)
            stats["listados" if ok else "fallidos"] += 1

        self.client.project_feedback = collect
        stats = {"listados": 0, "fallidos": 0, "ids": 0, "cortado": False}
        processed = 0
        try:
            it = iter(todo)
            in_flight: Dict[concurrent.futures.Future, Any] = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                agotado = False
                while True:
                    while not agotado and len(in_flight) < workers * 2:
                        if _should_stop(stop_event, deadline):
                            stats["cortado"] = True
                            agotado = True
                            break
                        row = next(it, None)
                        if row is None:
                            agotado = True
                            break
                        in_flight[pool.submit(fetch, row)] = row
                    if not in_flight:
                        break
                    done_f, _ = concurrent.futures.wait(
                        list(in_flight), return_when=concurrent.futures.FIRST_COMPLETED)
                    for fut in done_f:
                        row = in_flight.pop(fut)
                        try:
                            _row, ids, ok = fut.result()
                        except Exception:
                            _row, ids, ok = row, [], False  # se reintenta en la próxima corrida
                        store(_row, ids, ok)
                        processed += 1
                        if processed % 100 == 0:
                            flush()
                            print(f"\r[*] Listados {processed:,}/{len(todo):,} jugadores, "
                                  f"{stats['ids']:,} IDs...", end="", flush=True)
        finally:
            flush()
            self.client.project_feedback = None
        print()
        return stats

    # ------------------------------------------------------------------ #
    # Fase 2: descargar                                                  #
    # ------------------------------------------------------------------ #
    def download_phase(self, stop_event: Optional[threading.Event] = None,
                       deadline: Optional[float] = None,
                       workers: Optional[int] = None) -> Dict[str, Any]:
        """Baja las partidas pendientes de la ventana. Devuelve `matches_new` para run.py."""
        from src.collector import _fetch_single_match_task  # evita import circular

        pending = self.pending_match_ids()
        keys = max(1, int(getattr(self.client, "num_active_keys", 1) or 1))
        workers = workers or min(32, max(4, keys * 4))
        stats = {"pendientes_al_inicio": len(pending), "matches_new": 0,
                 "descartadas": 0, "fallidas": 0, "cortado": False}
        batch: List[tuple] = []

        REFRESH_EVERY = 5000
        refreshed_at = [0]

        def save():
            if batch:
                saved = self.db.save_matches_and_timelines_batch(list(batch))
                stats["matches_new"] += saved
                stats["descartadas"] += len(batch) - saved
                batch.clear()
            # Refresco periódico del mapa de identidad: si la descarga se corta a
            # la fuerza, lo bajado hasta el último refresco ya cuenta por persona.
            if stats["matches_new"] - refreshed_at[0] >= REFRESH_EVERY:
                try:
                    self.db.materialize_player_id_map()
                    refreshed_at[0] = stats["matches_new"]
                except Exception as err:
                    print(f"\n[!] No se pudo refrescar player_id_map: {err}")

        it = iter(pending)
        in_flight: Dict[concurrent.futures.Future, str] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            agotado = False
            while True:
                while not agotado and len(in_flight) < workers * 2:
                    if _should_stop(stop_event, deadline):
                        stats["cortado"] = True
                        agotado = True
                        break
                    mid = next(it, None)
                    if mid is None:
                        agotado = True
                        break
                    in_flight[pool.submit(_fetch_single_match_task, self.client, mid)] = mid
                if not in_flight:
                    break
                done, _ = concurrent.futures.wait(
                    list(in_flight), return_when=concurrent.futures.FIRST_COMPLETED)
                for fut in done:
                    in_flight.pop(fut)
                    try:
                        res = fut.result()
                    except Exception:
                        res = None
                    if res is None:
                        stats["fallidas"] += 1  # queda pendiente para la próxima corrida
                        continue
                    batch.append(res)
                    if len(batch) >= SAVE_BATCH:
                        save()
                if stats["matches_new"] and stats["matches_new"] % 300 < SAVE_BATCH:
                    print(f"\r[*] Descargadas {stats['matches_new']:,}/{len(pending):,} partidas...",
                          end="", flush=True)
        save()
        print()
        if stats["matches_new"] > 0:
            try:
                n = self.db.materialize_player_id_map()
                print(f"[*] player_id_map actualizado: {n:,} PUUIDs.")
            except Exception as err:
                print(f"[!] No se pudo actualizar player_id_map (queda el anterior): {err}")
        return stats


def _should_stop(stop_event: Optional[threading.Event], deadline: Optional[float]) -> bool:
    if stop_event is not None and stop_event.is_set():
        return True
    return deadline is not None and time.monotonic() >= deadline


def _request_ok(client) -> bool:
    check = getattr(client, "last_request_ok", None)
    ok = check() if callable(check) else True
    return ok is True
