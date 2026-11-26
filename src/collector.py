"""
Orquestador de recolección de datos: Invocadores, Partidas y Estadísticas.
"""

import time
import queue
import threading
from datetime import datetime
from pathlib import Path
import concurrent.futures
from typing import Optional, Dict, Any, Tuple, List
from tqdm import tqdm
from src.config import DEFAULT_TIER_MATCH_TARGETS
from src.riot_client import RiotAPIClient
from src.database import DatabaseManager
from export_to_csv import export_all_datasets


def _fetch_single_match_task(client: RiotAPIClient, match_id: str) -> Optional[Tuple[
    str, Dict[str, Any], Optional[Dict[str, Any]], Optional[int]
]]:
    """
    Descarga los DTO de detalle y timeline de una partida en un hilo trabajador.
    Si la partida es un remake (gameDuration <= 300s), se cancela de inmediato la descarga de Timeline,
    ahorrando el 50% de las peticiones a Riot.
    """
    match_detail = client.get_match_detail(match_id)
    if not match_detail:
        return None
    # Proyecto de la key que descargó la partida: los 10 PUUIDs del DTO vienen
    # cifrados con él. Se lee antes de la siguiente petición de este hilo.
    served = getattr(client, "last_served_project", None)
    served_project = served() if callable(served) else None
    if not isinstance(served_project, int):
        served_project = None

    # Si es un remake (<= 5 minutos), no descargar Timeline (ahorro del 50% de requests HTTP)
    info = match_detail.get("info", {})
    game_duration = info.get("gameDuration", 0)
    if game_duration <= 300:
        return (match_id, match_detail, None, served_project)

    timeline_detail = client.get_match_timeline(match_id)
    return (match_id, match_detail, timeline_detail, served_project)


# Tope de IDs que se recorren hacia atrás por jugador y sesión (10 páginas de 100):
# alcanza para cualquier jugador de Master+ en una temporada y acota a los inactivos.
MAX_HISTORY_IDS = 1000
DOWNLOAD_GRACE_SECONDS = 5.0
# Tope para esperar al hilo escritor al cerrar. La cola guarda hasta 200 partidas y
# un lote de 5 se escribe en milisegundos; cada conexión espera hasta 60 s un lock
# ajeno (`busy_timeout`). Cinco minutos cubren varios de esos reintentos y acotan
# un cierre que antes podía quedar colgado sin fin (§9.S.76).
WRITER_JOIN_TIMEOUT_SECONDS = 300.0
METRICS_INTERVAL_SECONDS = 60.0
COVERAGE_INTERVAL_SECONDS = 300.0

# Orden de tier para el puntaje de prioridad: un GM con 0 LP está por encima de un
# Master con 900 (el LP se reinicia al subir de tier).
_TIER_BASE = {"DIAMOND": 0, "MASTER": 1, "GRANDMASTER": 2, "CHALLENGER": 3}


def priorizar_por_lp(players: List[Dict[str, Any]], peso_lp: float) -> List[Dict[str, Any]]:
    """Reordena la cola: `peso_lp` x percentil de LP (alto primero) + el resto x percentil
    de "menos partidas primero" (PROJECT_SUMMARY.md §9.S.11, pedido del usuario 2026-10-25).

    Es suave: nadie sale de la cola, solo cambia el orden. Motivo medido sobre la BD: un
    Master con < 49 LP juega el 26 % de sus partidas en lobbies de rango Diamond; uno con
    > 474 LP, el 3 %. Con el relleno limitado por tiempo, priorizar LP alto trae más
    partidas Master+ y menos Diamond.
    """
    if not 0 <= peso_lp <= 1:
        raise ValueError("peso_lp tiene que estar entre 0 y 1.")
    n = len(players)
    if n < 2:
        return list(players)

    def pct(valores, descendente):
        orden = sorted(range(n), key=lambda i: valores[i], reverse=descendente)
        out = [0.0] * n
        for pos, i in enumerate(orden):
            out[i] = pos / (n - 1)  # 0 = mejor
        return out

    elo = [_TIER_BASE.get(str(p.get("tier", "")).upper(), 0) * 10_000 + int(p.get("league_points") or 0)
           for p in players]
    juegos = [int(p.get("existing_recorded_games") or 0) for p in players]
    p_lp, p_juegos = pct(elo, True), pct(juegos, False)
    score = [peso_lp * a + (1 - peso_lp) * b for a, b in zip(p_lp, p_juegos)]
    return [players[i] for i in sorted(range(n), key=lambda i: (score[i], i))]


class DataCollector:
    """
    Coordina la sincronización de jugadores y la extracción masiva de partidas con deduplicación y timeline.
    """

    def __init__(self, riot_client: RiotAPIClient, db_manager: DatabaseManager,
                 project_indices: Optional[List[int]] = None):
        self.client = riot_client
        self.db = db_manager
        self.project_indices = list(project_indices) if project_indices is not None else None

    def _canonical_client(self, canonical_project_index: int = 0,
                          project_indices: Optional[List[int]] = None) -> Optional[RiotAPIClient]:
        """Cliente pineado exclusivamente a la key del proyecto canónico (`project_index == 0`).

        League-V4 devuelve PUUIDs cifrados con el proyecto de la key que
        atiende; usar otro proyecto o balancear el roster entre proyectos
        contaminaría `players` y haría que la poda borre el padrón de P0.
        Usa el proyecto identificado por el cliente a partir de la clave, que
        conserva su significado aunque el preflight filtre o reordene el pool.
        """
        pool = getattr(self.client, "key_pool", None) or []
        indices = project_indices if project_indices is not None else self.project_indices
        if canonical_project_index != 0 or (indices is not None and len(indices) != len(pool)):
            return None
        for pos, ctx in enumerate(pool):
            project = ctx.project_index if ctx.project_known else (indices[pos] if indices is not None else None)
            if project != 0 or not ctx.is_active:
                continue
            print(f"[*] Operación pineada al proyecto canónico #0 (posición #{pos} del pool, {ctx}).")
            return RiotAPIClient(api_keys=[ctx.api_key], project_indices=[0], rate_limiter=ctx.rate_limiter,
                                 platform=self.client.platform,
                                 regional_routing=self.client.regional_routing)
        return None

    def sync_players(self, include_diamond: bool = True, canonical_key_index: int = 0,
                     project_indices: Optional[List[int]] = None) -> int:
        """
        Descarga todos los jugadores de Diamante+ de LAS y los almacena en SQLite.
        """
        print("\n" + "="*60)
        print("  FASE 1: SINCRONIZACIÓN DE JUGADORES (DIAMANTE+ EN LAS)")
        print("="*60)

        sync_client = self._canonical_client(canonical_key_index, project_indices=project_indices)
        if sync_client is None:
            print("[!] Sincronización abortada: se requiere acceso activo al proyecto canónico #0.")
            return 0
        players = sync_client.fetch_all_diamond_plus_players(include_diamond=include_diamond)
        if not players:
            print("[!] No se obtuvieron jugadores. Verifica la conexión y tu API Key.")
            return 0

        saved_count = self.db.save_players(players)
        print(f"\n[+] {saved_count} jugadores sincronizados con exito en SQLite.")

        # Poda de snapshot: el padrón de cada tier se reemplaza por el roster fresco
        # SOLO si coincide con el cupo oficial (exacto CH/GM, ±5% Master/Diamond).
        # Así los cupos quedan exactos por construcción sin riesgo ante syncs parciales.
        try:
            fresh_by_tier = {}
            for p in players:
                if p.get("puuid"):
                    fresh_by_tier.setdefault(str(p.get("tier", "UNKNOWN")).upper(), set()).add(p["puuid"])
            prune = self.db.prune_stale_tier_players(fresh_by_tier)
            for tier, info in prune.items():
                if info.get("replaced"):
                    print(f"[*] Padrón {tier} reemplazado: {info['kept']} vigentes, "
                          f"{info['pruned']} obsoletos/duplicados eliminados.")
                else:
                    print(f"[!] Padrón {tier} sin poda: {info.get('note')}")
        except Exception as err:
            print(f"[!] No se pudo podar el padrón: {err}")

        # Chequeo de cupos del padrón (SOLO ADVIERTE, nunca bloquea la sincronización)
        try:
            quotas = self.db.check_roster_quotas()
            for tier, info in quotas.items():
                if not info["ok"]:
                    print(f"[!] Cupo {tier}: {info['note']}")
        except Exception as err:
            print(f"[!] No se pudo verificar cupos del padrón: {err}")
        return saved_count

    def backfill_timelines(self, max_duration_seconds: Optional[float] = None,
                           stop_event: Optional[threading.Event] = None) -> int:
        """
        Descarga el timeline (@10, @15, @20) para partidas existentes en la base de datos que no lo tengan.
        Optimizado: consulta el mapa de participantes (participant_id -> puuid) en O(1) directamente de SQLite.
        """
        print("\n" + "="*60)
        print("  BACKFILL: RECUPERACIÓN DE TIMELINES PENDIENTES")
        print("="*60)

        missing_match_ids = self.db.get_matches_without_timeline()
        if not missing_match_ids:
            print("[+] Todas las partidas en la base de datos ya cuentan con timeline registrado.")
            return 0

        print(f"[*] Se encontraron {len(missing_match_ids)} partidas sin timeline.")
        start_time = time.time()
        saved_count = 0

        pbar = tqdm(missing_match_ids, desc="Progreso Timelines", unit="partida")
        for match_id in pbar:
            if stop_event is not None and stop_event.is_set():
                print("\n[i] Corte solicitado (pool de claves agotado o interrupción). "
                      "Finalizando backfill...")
                break
            if max_duration_seconds and (time.time() - start_time) >= max_duration_seconds:
                print(f"\n[i] Tiempo límite alcanzado ({max_duration_seconds / 60:.1f} min). Finalizando backfill...")
                break

            timeline_detail = self.client.get_match_timeline(match_id)
            if not timeline_detail:
                continue

            puuid_map = self.db.get_participant_puuid_map(match_id)
            saved = self.db.save_match_timeline(match_id, timeline_detail, puuid_map)
            if saved:
                saved_count += 1

        print(f"[+] {saved_count} timelines registrados con exito.")
        return saved_count

    def collect_matches(self, matches_per_player: Optional[int] = None, 
                        tier_targets: Optional[Dict[str, int]] = None,
                        max_players: Optional[int] = None, 
                        tier_filter: Optional[str] = None,
                        max_duration_seconds: Optional[float] = None,
                        depth_first: bool = True,
                        max_workers: Optional[int] = None,
                        require_canonical: bool = False,
                        stop_event: Optional[threading.Event] = None,
                        peso_lp: Optional[float] = None) -> Dict[str, Any]:
        """
        Recorre los jugadores guardados en la base de datos, extrae sus partidas y timelines (@10, @15, @20)
        evitando cualquier descarga de partida duplicada y aplicando umbrales adaptativos por Tier.
        Utiliza un pool concurrente de hilos (ThreadPoolExecutor) y guardado por lotes (Batch Commits).
        Si require_canonical es True, se niega a correr en modo legacy (mapa vacío): retorna el
        resumen sin recolectar para no seguir fragmentando PUUIDs. Costo del chequeo: un COUNT local (~ms).
        """
        print("\n" + "="*60)
        print("  FASE 2: RECOLECCIÓN Y PROCESAMIENTO DE PARTIDAS Y TIMELINE")
        print("="*60)

        # Resolver umbrales (Adaptativo por Tier o Manual Fijo)
        active_tier_targets = tier_targets or DEFAULT_TIER_MATCH_TARGETS
        is_adaptive = matches_per_player is None
        min_target_depth = min(active_tier_targets.values()) if is_adaptive else (matches_per_player or 80)

        # Traducción multi-proyecto: se carga el mapa alias->canónico a memoria UNA vez
        # (lookup O(1) en hot path, cero requests extra a Riot durante el crawl).
        # Chequeo barato (un COUNT local): va primero para que el modo estricto
        # no necesite ni siquiera un client válido.
        try:
            n_trans = self.db.reload_translation_cache()
            if n_trans:
                print(f"[*] Traducción multi-proyecto activa: {n_trans} PUUIDs alias -> canónico.")
        except Exception as err:
            n_trans = 0
            print(f"[!] No se pudo cargar el mapa de traducción (se continúa sin traducir): {err}")
        if require_canonical and not n_trans:
            print("[X] Modo estricto (--require-canonical): mapa multi-proyecto vacío. "
                  "Corre 'python scripts/register_key_project.py' antes de recolectar. "
                  "Crawler detenido para no fragmentar PUUIDs.")
            return self.db.get_database_summary()

        # El conteo de partidas por persona sale de `player_id_map`. Si la corrida
        # anterior se cortó antes de refrescarlo (Ctrl+C, corte de luz), las
        # partidas que bajó no cuentan y la cola prioriza mal: se refresca acá
        # (~11 s sobre 120k PUUIDs).
        try:
            self.db.materialize_player_id_map()
        except Exception as err:
            print(f"[!] No se pudo refrescar player_id_map al arrancar: {err}")

        platform_prefix = (getattr(self.client, "platform", None) or "la2")
        platform_prefix = (platform_prefix if isinstance(platform_prefix, str) else "la2").upper() + "_"

        # Concurrencia de hilos: proporcional al pool de claves activas (4 hilos por clave)
        workers = max_workers or min(32, max(4, self.client.num_active_keys * 4))

        # 1. Obtener jugadores (priorizando profundidad o nuevos)
        if depth_first:
            players = self.db.get_canonicals_for_deep_crawl(tier=tier_filter, min_matches_target=min_target_depth, only_incomplete=True)
        else:
            players = self.db.get_all_players(tier=tier_filter, unvisited_first=True)

        if peso_lp is not None and players:
            players = priorizar_por_lp(players, peso_lp)
            print(f"[*] Cola ordenada por LP (peso {peso_lp:.2f}) y menos partidas (peso {1 - peso_lp:.2f}).")

        if not players:
            print("[+] Todos los jugadores seleccionados ya alcanzaron la meta requerida.")
            return self.db.get_database_summary()

        if max_players and max_players < len(players):
            players = players[:max_players]

        session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")

        print(f"[*] Iniciando recolección para {len(players)} jugadores...")
        print(f"    - Concurrencia HTTP: {workers} hilos trabajadores en paralelo")
        if is_adaptive:
            print("    - Modo de Asignación: Umbrales Adaptativos por Tier (Densidad de Grafo / Solapamiento)")
            print(f"      Challenger: {active_tier_targets.get('CHALLENGER', 160)} | GM: {active_tier_targets.get('GRANDMASTER', 175)} | Master: {active_tier_targets.get('MASTER', 215)} | Diamond: {active_tier_targets.get('DIAMOND', 260)}")
        else:
            print(f"    - Partidas solicitadas por jugador (Fijo): {matches_per_player}")

        print(f"    - Estrategia de Recorrido: {'Profundidad (Arquetipos ML)' if depth_first else 'Amplitud'}")
        if tier_filter:
            print(f"    - Filtro de Tier: {tier_filter.upper()}")
        if max_duration_seconds:
            print(f"    - Límite de tiempo: {max_duration_seconds / 60:.1f} minutos")

        def _debe_parar() -> bool:
            """Corte unificado del recorrido: presupuesto de tiempo o señal externa.

            La señal externa la levanta `run.py` cuando el pool de claves se
            queda sin ninguna viva: sin esto el crawler seguiría recorriendo el
            padrón marcando jugadores como visitados con cero partidas bajadas.
            """
            nonlocal stop_requested
            if not stop_requested and stop_event is not None:
                stop_requested = stop_event.is_set()
            return stop_requested or (deadline is not None and time.monotonic() >= deadline)

        total_discovered_matches = 0
        total_new_matches = 0
        total_skipped_duplicates = 0
        total_discarded_matches = 0
        # Contadores aparte (§9.S.76). `total_write_errors`: partidas de lotes cuyo
        # commit falló; no son descartes por criterio, se vuelven a pedir en otra
        # sesión. `total_rejected_participants`: partidas que llegaron con un número
        # de participantes distinto de 10 (también suman en descartadas).
        total_write_errors = 0
        total_rejected_participants = 0
        writer_timed_out = False
        total_timelines_saved = 0
        players_crawled = 0
        start_time = time.monotonic()
        deadline = start_time + max_duration_seconds if max_duration_seconds else None
        start_datetime = datetime.now()
        stop_requested = False
        stopped_due_to_time = False

        self.client.reset_telemetry()
        last_metrics_log_time = start_time
        cached_cov = {}
        last_cov_time = None

        # Inicialización de la cola concurrente y DB Writer Thread para Streaming Ingestion
        match_queue: queue.Queue = queue.Queue(maxsize=200)
        writer_lock = threading.Lock()
        reserved_ids = set()
        reservation_lock = threading.Lock()

        # Lo que Riot confirma sobre el proyecto de cada PUUID en las listas de
        # partidas (éxito = descifrable, 400 = etiqueta falsa). Se acumula en
        # memoria y lo escribe el hilo escritor, que es el único que escribe.
        project_feedback: List[Tuple[str, int, bool]] = []
        feedback_lock = threading.Lock()

        def _collect_feedback(puuid: str, project_index: int, ok: bool) -> None:
            with feedback_lock:
                project_feedback.append((puuid, project_index, ok))

        def _flush_feedback() -> None:
            with feedback_lock:
                pending = list(project_feedback)
                project_feedback.clear()
            if pending:
                try:
                    self.db.apply_project_feedback(pending)
                except Exception as err:
                    print(f"\n[!] No se pudo guardar el feedback de proyectos: {err}")

        def _db_writer_worker():
            nonlocal total_new_matches, total_discarded_matches, total_timelines_saved
            nonlocal total_write_errors, total_rejected_participants
            micro_batch = []
            last_flush = time.monotonic()
            closing = False

            while not closing:
                try:
                    item = match_queue.get(timeout=0.3)
                    closing = item is None
                    if not closing:
                        micro_batch.append(item)
                    match_queue.task_done()
                except queue.Empty:
                    pass

                # Micro-commits fluidos: si acumulamos 5 partidas, o pasó 1.5s, o cierre
                should_flush = (
                    len(micro_batch) >= 5 or
                    (micro_batch and (time.monotonic() - last_flush) >= 1.5) or
                    (closing and micro_batch)
                )

                if should_flush:
                    # Este hilo es el único que guarda partidas: la diferencia del
                    # contador de la base es lo que rechazó este lote.
                    rejected_before = getattr(self.db, "partidas_rechazadas_participantes", 0)
                    try:
                        saved, timelines = self.db.save_match_batch_with_counts(micro_batch)
                        rejected = getattr(self.db, "partidas_rechazadas_participantes", 0) - rejected_before
                        with writer_lock:
                            total_new_matches += saved
                            total_discarded_matches += (len(micro_batch) - saved)
                            total_rejected_participants += rejected
                            total_timelines_saved += timelines
                    except Exception as err:
                        # Un commit fallido no es un descarte por criterio: va a su
                        # propio contador para que el reporte no lo confunda con remakes.
                        with writer_lock:
                            total_write_errors += len(micro_batch)
                        print(f"\n[!] Error en DB Writer Thread ({len(micro_batch)} partidas "
                              f"sin guardar): {err}")
                    finally:
                        with reservation_lock:
                            reserved_ids.difference_update(item[0] for item in micro_batch)
                        micro_batch.clear()
                        last_flush = time.monotonic()
                    _flush_feedback()
            _flush_feedback()

        db_writer_thread = threading.Thread(target=_db_writer_worker, daemon=True)

        # Barra de progreso para los jugadores
        player_pbar = tqdm(players, desc="Progreso Jugadores", unit="jugador")

        # Pool de trabajadores HTTP persistente para toda la sesión
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        active_futures: Dict[concurrent.futures.Future, str] = {}

        def _harvest(done):
            nonlocal total_discarded_matches
            for fut in done:
                mid = active_futures.pop(fut)
                try:
                    result = fut.result()
                except Exception:
                    result = None
                if result is not None:
                    match_queue.put(result)
                    continue
                with reservation_lock:
                    reserved_ids.discard(mid)
                with writer_lock:
                    total_discarded_matches += 1

        def _log_metrics_if_due():
            nonlocal last_metrics_log_time, last_cov_time, cached_cov
            now = time.monotonic()
            if now - last_metrics_log_time < METRICS_INTERVAL_SECONDS:
                return
            last_metrics_log_time = now
            if last_cov_time is None or now - last_cov_time >= COVERAGE_INTERVAL_SECONDS:
                last_cov_time = now
                try:
                    cached_cov = self.db.get_coverage_stats()
                except Exception as err:
                    print(f"\n[!] No se pudo refrescar la cobertura: {err}")
            try:
                elapsed = (time.monotonic() - start_time) / 60.0
                with writer_lock:
                    curr_new = total_new_matches
                    curr_discarded = total_discarded_matches
                    curr_write_errors = total_write_errors
                    curr_rejected = total_rejected_participants
                db_size = self.db.db_path.stat().st_size / (1024 * 1024)
                self.db.log_crawler_metric({
                    "session_id": session_id,
                    "elapsed_minutes": round(elapsed, 2),
                    "active_keys": self.client.num_active_keys,
                    "workers": workers,
                    "players_crawled": players_crawled,
                    "total_matches_in_db": self.db.get_total_matches(),
                    "new_matches_saved": curr_new,
                    "duplicate_matches_skipped": total_skipped_duplicates,
                    "discarded_matches": curr_discarded,
                    "net_matches_per_min": round(curr_new / max(0.1, elapsed), 2),
                    "requests_per_min": round(self.client.telemetry.get("total_requests", 0) / max(0.1, elapsed), 2),
                    "c50_challenger": cached_cov.get("CHALLENGER", {}).get("c50", 0),
                    "c50_grandmaster": cached_cov.get("GRANDMASTER", {}).get("c50", 0),
                    "c50_master": cached_cov.get("MASTER", {}).get("c50", 0),
                    "c50_total": cached_cov.get("TOTAL", {}).get("c50", 0),
                    "db_size_mb": round(db_size, 2),
                    "write_errors": curr_write_errors,
                    "rejected_participants": curr_rejected,
                })
            except Exception as err:
                print(f"\n[!] No se pudo registrar el latido del crawler: {err}")

        previous_feedback = getattr(self.client, "project_feedback", None)
        self.client.project_feedback = _collect_feedback
        db_writer_thread.start()
        try:
            for player in player_pbar:
                if _debe_parar():
                    stopped_due_to_time = True
                    break

                puuid = player.get("puuid")
                if not puuid:
                    continue
                # Fila del padrón a marcar como visitada. La cola canónica la da
                # aparte porque `puuid` es el representante para Riot, y marcar
                # con un PUUID que no está en `players` no marcaba a nadie.
                players_puuid = player.get("players_puuid") or puuid
                # {project_index: puuid} de esta persona: las listas de partidas se
                # reparten entre las keys que puedan descifrar alguno.
                candidates = dict(player.get("candidates") or {})

                tier = str(player.get("tier", "DIAMOND")).upper()
                target_for_player = active_tier_targets.get(tier, 260) if is_adaptive else matches_per_player

                # Calcular partidas existentes en BD en tiempo real (anti-congelamiento)
                existing_count = self.db.get_player_match_count(puuid)
                matches_to_request = max(0, target_for_player - int(existing_count))

                if matches_to_request == 0:
                    self.db.mark_player_crawled(players_puuid)
                    players_crawled += 1
                    _log_metrics_if_due()
                    continue

                # Paginación hacia atrás hasta completar la meta o agotar el historial.
                # Antes se cortaba cuando `page_start` llegaba a la meta (50): con las
                # ~100 partidas más recientes ya descargadas (p. ej. por la ventana),
                # el crawler nunca llegaba a las anteriores y el jugador no avanzaba.
                # Páginas de 100 (el máximo de Riot) para gastar menos requests.
                all_unprocessed_ids = {}
                page_start = 0
                needed = matches_to_request

                while needed > 0 and page_start < MAX_HISTORY_IDS:
                    if _debe_parar():
                        stopped_due_to_time = True
                        break

                    batch_count = 100
                    page_match_ids = self.client.get_match_ids_by_puuid(
                        puuid=puuid, count=batch_count, start=page_start,
                        candidates=candidates or None)
                    if not page_match_ids:
                        break

                    total_discovered_matches += len(page_match_ids)
                    # Solo partidas del servidor del proyecto: Match-V5 devuelve las de
                    # todos los servidores de la región (BR1, NA1, LA1...).
                    page_local = [m for m in page_match_ids if m.upper().startswith(platform_prefix)]
                    # La reserva se libera después del commit. Mirar ambas fuentes
                    # bajo el candado evita un hueco entre el filtro y la liberación.
                    with reservation_lock:
                        candidates_local = list(dict.fromkeys(
                            m for m in page_local if m not in reserved_ids and m not in all_unprocessed_ids
                        ))
                        page_unprocessed = self.db.filter_unprocessed_match_ids(candidates_local)
                    skipped = len(page_local) - len(page_unprocessed)
                    total_skipped_duplicates += skipped

                    to_take = page_unprocessed[:needed]
                    all_unprocessed_ids.update(dict.fromkeys(to_take))
                    needed -= len(to_take)

                    if len(page_match_ids) < batch_count:
                        break
                    page_start += len(page_match_ids)

                if _debe_parar():
                    stopped_due_to_time = True
                else:
                    self.db.mark_player_crawled(players_puuid)
                    players_crawled += 1

                # Despacho en streaming: encolar tareas de descarga
                for mid in all_unprocessed_ids:
                    if _debe_parar():
                        stopped_due_to_time = True
                        break

                    # Control de contrapresión (backpressure): mantener saturados los workers
                    while len(active_futures) >= workers * 2 and not _debe_parar():
                        done, _ = concurrent.futures.wait(
                            active_futures, timeout=0.2, return_when=concurrent.futures.FIRST_COMPLETED
                        )
                        _harvest(done)
                    if _debe_parar():
                        stopped_due_to_time = True
                        break
                    with reservation_lock:
                        reserved_ids.add(mid)
                    fut = executor.submit(_fetch_single_match_task, self.client, mid)
                    active_futures[fut] = mid

                # Drenar tareas completadas en segundo plano
                _harvest([f for f in active_futures if f.done()])

                with writer_lock:
                    curr_new = total_new_matches
                player_pbar.set_postfix({
                    "Nuevas": curr_new,
                    "Duplicados": total_skipped_duplicates,
                    "Cola": match_queue.qsize()
                })

                if not stopped_due_to_time:
                    _log_metrics_if_due()

                if stopped_due_to_time:
                    break
            while active_futures and not stopped_due_to_time:
                if _debe_parar():
                    stopped_due_to_time = True
                    break
                done, _ = concurrent.futures.wait(
                    active_futures, timeout=0.2, return_when=concurrent.futures.FIRST_COMPLETED
                )
                _harvest(done)
        finally:
            try:
                # Cancelar la cola del executor. Los workers HTTP en curso solo
                # devuelven datos, así que después de la gracia ya no pueden escribir.
                executor.shutdown(wait=False, cancel_futures=True)
                _harvest([f for f in active_futures if f.done() and not f.cancelled()])
                remaining = [f for f in active_futures if not f.cancelled()]
                if remaining:
                    concurrent.futures.wait(remaining, timeout=DOWNLOAD_GRACE_SECONDS)
                    _harvest([f for f in remaining if f.done()])
            finally:
                match_queue.put(None)
                # El centinela sigue a todos los resultados. Esperarlo garantiza que
                # ni el mapa, ni el reporte, ni el export se adelanten al commit.
                # Con tope (§9.S.76): un escritor trabado (lock ajeno sobre la base,
                # disco lleno) colgaba el cierre para siempre.
                db_writer_thread.join(timeout=WRITER_JOIN_TIMEOUT_SECONDS)
                writer_timed_out = db_writer_thread.is_alive()
                if writer_timed_out:
                    print(f"\n[!] AVISO: el escritor de la base no terminó en "
                          f"{WRITER_JOIN_TIMEOUT_SECONDS:.0f} s (cola: {match_queue.qsize()} "
                          "partidas). Se cierra sin esperarlo: no se actualiza player_id_map "
                          "ni se exportan los CSV, y los contadores del reporte pueden quedar "
                          "cortos. Revisá si otro proceso tiene la base tomada y corré "
                          "`python run.py --solo-validar` antes de volver a recolectar.")
                self.client.project_feedback = previous_feedback
                with reservation_lock:
                    reserved_ids.clear()

        # Refrescar el `player_id_map` permanente con los PUUIDs que entraron en la
        # sesión. Lo leen `get_player_match_count()` y la cola de recolección
        # profunda: sin esto, las partidas nuevas no cuentan para su persona hasta
        # la próxima resolución inversa. Va después del join del escritor para no
        # competir por el lock, y solo si la sesión guardó algo (~11 s sobre 100k
        # partidas). Un fallo acá no invalida la sesión: los datos ya están
        # guardados y el export recalcula su propio mapa.
        if total_new_matches > 0 and not writer_timed_out:
            try:
                n_mapped = self.db.materialize_player_id_map()
                print(f"[*] player_id_map actualizado: {n_mapped:,} PUUIDs.")
            except Exception as err:
                print(f"[!] No se pudo actualizar player_id_map (queda el anterior): {err}")

        if stopped_due_to_time:
            if stop_requested:
                print("\n[i] Corte solicitado (pool de claves agotado o interrupción). "
                      "Finalizando de forma segura...")
            else:
                print(f"\n[i] Tiempo límite alcanzado ({max_duration_seconds / 60:.1f} min). Finalizando de forma segura...")

        end_datetime = datetime.now()
        elapsed_min = (time.monotonic() - start_time) / 60
        matches_per_min = round(total_new_matches / max(0.001, elapsed_min), 2)
        matches_per_hour = round(matches_per_min * 60, 1)

        telemetry = self.client.get_telemetry()
        total_reqs = telemetry.get("total_requests", 0)
        requests_per_min = round(total_reqs / max(0.001, elapsed_min), 2)
        requests_per_hour = round(requests_per_min * 60, 1)

        summary = self.db.get_database_summary()
        coverage = summary["coverage"]

        # Meta global estimada (~500,000 partidas para ~300 part/jugador en Diamante+)
        ESTIMATED_TARGET_MATCHES = 500000
        current_matches = summary['total_matches']
        pct_completed = round((current_matches / ESTIMATED_TARGET_MATCHES) * 100, 2)
        matches_remaining = max(0, ESTIMATED_TARGET_MATCHES - current_matches)

        if matches_per_hour > 0:
            eta_hours = round(matches_remaining / matches_per_hour, 1)
            eta_days = round(eta_hours / 24.0, 1)
            eta_str = f"{eta_hours:,.1f} horas (~{eta_days} días continuos)"
        else:
            eta_str = "Indeterminado (0 part/hora en esta sesión)"

        print("\n" + "="*65)
        print("  ESTADÍSTICAS DETALLADAS DE LA SESIÓN DE EXTRACCIÓN")
        print("="*65)
        print(f"  - Inicio: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')} | Fin: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  - Tiempo transcurrido: {elapsed_min:.2f} minutos")
        print(f"  - Partidas descubiertas: {total_discovered_matches:,}")
        print(f"  - Partidas nuevas guardadas: {total_new_matches:,}")
        print(f"  - Partidas duplicadas evitadas: {total_skipped_duplicates:,}")
        print(f"  - Partidas descartadas (remakes <= 300s / errores): {total_discarded_matches:,}")
        print(f"    - de ellas, rechazadas por no traer 10 participantes: {total_rejected_participants:,}")
        print(f"  - Partidas perdidas por errores de escritura en la BD (se reintentan): {total_write_errors:,}")
        if writer_timed_out:
            print("  - [!] El escritor no terminó a tiempo: estos contadores pueden quedar cortos.")
        timeline_pct = 100 * total_timelines_saved / total_new_matches if total_new_matches else 0.0
        print(f"  - Timelines guardadas: {total_timelines_saved:,} ({timeline_pct:.1f}%)")

        print("\n  VELOCIDAD Y RENDIMIENTO (THROUGHPUT):")
        print(f"  - Partidas nuevas / minuto: {matches_per_min:.2f} part/min")
        print(f"  - Partidas nuevas / hora:   {matches_per_hour:,.1f} part/hora")
        print(f"  - Peticiones HTTP / minuto: {requests_per_min:.2f} req/min")
        print(f"  - Peticiones HTTP / hora:   {requests_per_hour:,.1f} req/hora")

        print("\n  TELEMETRÍA DE RED Y API (RIOT GAMES):")
        print(f"  - Total peticiones HTTP (Requests): {total_reqs:,}")
        print(f"  - Peticiones exitosas (HTTP 200): {telemetry.get('status_200', 0):,}")
        print(f"  - Rate Limits alcanzados (HTTP 429): {telemetry.get('status_429', 0)}")
        print(f"  - Timeouts / Cortes de conexión: {telemetry.get('timeouts', 0)}")
        print(f"  - Errores HTTP de servidor (5xx/4xx): {telemetry.get('http_errors', 0)}")

        print("\n  PROGRESO GLOBAL Y ESTIMACIÓN DE TIEMPO (ETA):")
        print(f"  - Partidas acumuladas en BD: {current_matches:,}")
        print(f"  - Meta estimada Diamante+ (~300 part/jug): {ESTIMATED_TARGET_MATCHES:,}")
        print(f"  - Porcentaje de recolección completado: {pct_completed:.2f}%")
        print(f"  - Partidas restantes estimadas: {matches_remaining:,}")
        print(f"  - Tiempo estimado restante (ETA a este rate): {eta_str}")

        print("\n  ESTADO ACTUAL DE LA BASE DE DATOS:")
        print(f"  - Total partidas únicas en BD: {current_matches:,}")
        print(f"  - Total participantes en BD: {summary['total_participant_records']:,}")
        print(f"  - Tamaño de la base de datos: {summary['file_size_mb']} MB")

        print("\n  MÉTRICAS DE COBERTURA DE JUGADORES (COVERAGE):")
        for t in ["CHALLENGER", "GRANDMASTER", "MASTER", "DIAMOND", "TOTAL"]:
            st = coverage[t]
            tot_p = st['total']
            pct50 = f"{st['c50']/tot_p*100:.1f}%" if tot_p > 0 else "0%"
            pct100 = f"{st['c100']/tot_p*100:.1f}%" if tot_p > 0 else "0%"
            pct200 = f"{st['c200']/tot_p*100:.1f}%" if tot_p > 0 else "0%"
            pct300 = f"{st['c300']/tot_p*100:.1f}%" if tot_p > 0 else "0%"
            print(f"  - {t:<12}: Total={tot_p:,} | >=50: {st['c50']} ({pct50}) | >=100: {st['c100']} ({pct100}) | >=200: {st['c200']} ({pct200}) | >=300: {st['c300']} ({pct300})")
        print("="*65 + "\n")

        # Auto-exportar los CSVs al terminar la sesión.
        # El destino se deriva de la BD que realmente se recolectó, no del DATA_DIR
        # global: en producción `db_path` es `data/lol_las.db` y el destino es el
        # mismo `data/` de siempre, pero una sesión sobre una BD temporal (tests)
        # escribe junto a esa BD y no puede pisar los CSV reales del repo.
        # Con el escritor todavía vivo el export leería una base a medio escribir.
        if writer_timed_out:
            print("[!] Export de CSV omitido: el escritor de la base no terminó. "
                  "Exportá a mano cuando termine: python export_to_csv.py")
        else:
            export_all_datasets(self.db.db_path, Path(self.db.db_path).parent)

        session_report = {
            "start_time": start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            "end_time": end_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            "elapsed_minutes": round(elapsed_min, 2),
            "matches_discovered": total_discovered_matches,
            "matches_new": total_new_matches,
            "matches_duplicate": total_skipped_duplicates,
            "matches_discarded": total_discarded_matches,
            "matches_rejected_participants": total_rejected_participants,
            "write_errors": total_write_errors,
            "writer_timed_out": writer_timed_out,
            "timelines_saved": total_timelines_saved,
            "matches_per_minute": matches_per_min,
            "matches_per_hour": matches_per_hour,
            "requests_per_minute": requests_per_min,
            "requests_per_hour": requests_per_hour,
            "pct_completed": pct_completed,
            "matches_remaining": matches_remaining,
            "eta": eta_str,
            "telemetry": telemetry,
            "database_summary": summary,
            "coverage": coverage
        }

        return session_report

