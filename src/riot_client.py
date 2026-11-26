"""
Cliente para interactuar con la API de Riot Games (League-V4, Match-V5).
"""

import sys
import time
import threading
import requests
from typing import Callable, List, Dict, Any, Optional, Tuple
from src.config import (
    DEFAULT_PLATFORM,

    DEFAULT_REGIONAL_ROUTING,
    QUEUE_SOLOQ_NAME,
    QUEUE_SOLOQ_ID,
    APEX_TIERS,
    DIAMOND_DIVISIONS,
    RIOT_API_KEYS,
    RIOT_API_KEY,
    RIOT_API_PROJECT_INDICES,
    project_index_for_key,
)
from src.rate_limiter import RiotRateLimiter


class KeyContext:
    """
    Contexto individual para una API Key: encapsula su RateLimiter, su sesión HTTP
    persistente y su telemetría propia.

    El índice (`key_index`) es la **ranura**, no la clave: identifica al proyecto
    de Riot al que pertenece (ver `RIOT_API_PROJECT_INDICES` en `src/config.py`).
    Renovar una dev key vencida cambia la cadena pero no el proyecto, así que el
    reemplazo en caliente conserva la ranura y con ella la validez de los PUUIDs
    ya guardados.
    """
    def __init__(self, api_key: str, key_index: int, project_index: Optional[int] = None):
        self.api_key = api_key.strip()
        self.key_index = key_index
        # Proyecto de Riot de esta ranura: define con qué proyecto vienen cifrados
        # los PUUIDs que devuelve y cuáles puede descifrar. Por defecto, la posición.
        self.project_index = key_index if project_index is None else int(project_index)
        # False cuando el proyecto es solo la posición (key ajena al .env): la key
        # sirve igual, pero sus respuestas no se usan para etiquetar PUUIDs.
        self.project_known = project_index is not None
        self.rate_limiter = RiotRateLimiter(warmup=True)
        self.session = requests.Session()
        self.session.headers.update({
            "X-Riot-Token": self.api_key,
            "Accept": "application/json"
        })
        self.is_active = True
        self.deactivated_reason: Optional[str] = None
        # Telemetría por clave: sin esto, un pool desbalanceado (o una clave que
        # se muere a mitad de corrida) es invisible en el total agregado.
        self.stats: Dict[str, int] = {"requests": 0, "ok": 0, "rate_limited": 0,
                                      "errors": 0, "puuid_mismatch": 0}
        self._stats_lock = threading.Lock()

    def note(self, campo: str) -> None:
        with self._stats_lock:
            self.stats[campo] = self.stats.get(campo, 0) + 1

    def replace_key(self, api_key: str) -> None:
        """Cambia la clave conservando la ranura (proyecto), y reactiva el contexto."""
        self.api_key = api_key.strip()
        self.session.headers["X-Riot-Token"] = self.api_key
        self.is_active = True
        self.deactivated_reason = None

    @property
    def masked(self) -> str:
        return f"{self.api_key[:8]}...{self.api_key[-4:]}" if len(self.api_key) > 12 else "INVALID"

    def __repr__(self) -> str:
        return f"<KeyContext #{self.key_index + 1}: {self.masked}>"


class RiotAPIClient:
    """
    Cliente robusto para Riot Games API con soporte integrado de Rate Limiting, 
    reintentos y Multi-Key Pool (balanceo de carga entre múltiples claves).
    """

    def __init__(self, api_key: Optional[str] = None, 
                 api_keys: Optional[List[str]] = None,
                 platform: str = DEFAULT_PLATFORM, 
                 regional_routing: str = DEFAULT_REGIONAL_ROUTING, 
                 rate_limiter: Optional[RiotRateLimiter] = None,
                 project_indices: Optional[List[int]] = None):
        self.platform = platform.lower()
        self.regional_routing = regional_routing.lower()

        # Determinar lista de claves
        if api_keys:
            raw_keys = api_keys
        elif api_key:
            raw_keys = [api_key]
        elif RIOT_API_KEYS:
            raw_keys = RIOT_API_KEYS
        elif RIOT_API_KEY:
            raw_keys = [RIOT_API_KEY]
        else:
            raw_keys = []

        # Proyecto de cada key: explícito si se pasa `project_indices`; si no, se
        # busca la key en el .env (`project_index_for_key`), nunca por su posición
        # en esta lista, que puede venir filtrada. Una key de proyecto desconocido
        # se usa igual, pero sus respuestas no anotan etiquetas de proyecto.
        raw_keys = [k for k in raw_keys if k and k.strip()]
        if project_indices is not None and len(project_indices) != len(raw_keys):
            raise ValueError("project_indices debe tener una entrada por API key.")

        self.key_pool: List[KeyContext] = []
        for idx, k in enumerate(raw_keys):
            if project_indices is not None:
                ctx = KeyContext(k, idx, project_indices[idx])
            else:
                proj = project_index_for_key(k)
                ctx = KeyContext(k, idx, proj)
                ctx.project_known = proj is not None
            self.key_pool.append(ctx)

        if not self.key_pool:
            # Fallback a clave vacía si no se proveyó ninguna
            self.key_pool = [KeyContext("", 0)]

        # Si se pasó un rate_limiter personalizado explícito para la clave 0, asignarlo
        if rate_limiter and self.key_pool:
            self.key_pool[0].rate_limiter = rate_limiter

        self._pool_lock = threading.Lock()
        self._telemetry_lock = threading.Lock()
        # Estado por hilo de la última petición: qué proyecto la atendió y si Riot
        # la rechazó por no poder descifrar el PUUID. Por hilo porque los
        # trabajadores del crawler piden partidas en paralelo.
        self._tls = threading.local()
        # Callback opcional `(puuid, project_index, descifrable)` con lo que Riot
        # confirmó en cada lista de partidas (lo usa el crawler para corregir el
        # directorio `puuid_projects`).
        self.project_feedback: Optional[Callable[[str, int, bool], None]] = None
        # Se levanta cuando no queda ninguna clave viva. No mata el proceso: el
        # orquestador (run.py) decide si renovar el pool o cerrar la sesión con
        # su reporte. Antes esto era un sys.exit() que, disparado desde un hilo
        # trabajador, solo mataba ese hilo y dejaba al crawler girando en falso.
        self.pool_exhausted = threading.Event()
        self.telemetry = {
            "total_requests": 0,
            "status_200": 0,
            "status_429": 0,
            "timeouts": 0,
            "http_errors": 0,
            # PUUID cifrado por otro proyecto de API: no es error de red ni exito,
            # pero es el modo de falla dominante cuando el pool no cubre todos los
            # proyectos que produjeron los PUUID guardados.
            "puuid_mismatch": 0
        }

    @property
    def api_key(self) -> str:
        return self.key_pool[0].api_key if self.key_pool else ""

    @property
    def session(self) -> requests.Session:
        return self.key_pool[0].session if self.key_pool else requests.Session()

    @property
    def rate_limiter(self) -> RiotRateLimiter:
        return self.key_pool[0].rate_limiter if self.key_pool else RiotRateLimiter()

    @property
    def num_active_keys(self) -> int:
        return len([k for k in self.key_pool if k.is_active])

    def refresh_pool(self, keys: List[str], allow_new_slots: bool = False,
                     projects: Optional[List[int]] = None) -> Dict[str, List[int]]:
        """
        Reemplaza en caliente las claves del pool por las de `keys`, **por ranura**.

        Por qué por ranura y no por cadena: el PUUID está cifrado por proyecto de
        Riot, no por clave (ver `src/config.py`). Renovar una dev key vencida
        devuelve una cadena nueva del *mismo* proyecto, así que pisar la ranura
        i con `keys[i]` mantiene intactos los PUUIDs ya guardados. Una clave en
        una ranura nueva, en cambio, es un proyecto sin registrar: se agrega solo
        con `allow_new_slots=True` y hay que pasarle
        `scripts/identity/register_key_project.py` antes de que sirva.

        Con `projects` (el `RIOT_API_PROJECT_INDICES` del `.env`), cada key nueva
        reemplaza a la ranura **de su mismo proyecto**, no a la de su misma
        posición. Es lo correcto cuando el pool arrancó sin alguna key (vencida al
        inicio): ahí las posiciones del pool y las del `.env` no coinciden.

        Devuelve {"reemplazadas": [...], "reactivadas": [...], "agregadas": [...]}
        con los índices de ranura afectados.
        """
        cambios: Dict[str, List[int]] = {"reemplazadas": [], "reactivadas": [], "agregadas": []}
        limpias = [k.strip() for k in keys if k and k.strip()]
        if projects is not None and len(projects) == len(limpias):
            with self._pool_lock:
                by_project = {ctx.project_index: ctx for ctx in self.key_pool if ctx.project_known}
                for key, proj in zip(limpias, projects):
                    ctx = by_project.get(int(proj))
                    if ctx is None:
                        if allow_new_slots:
                            nuevo = KeyContext(key, len(self.key_pool), int(proj))
                            self.key_pool.append(nuevo)
                            cambios["agregadas"].append(nuevo.key_index)
                        continue
                    if ctx.api_key == key:
                        if not ctx.is_active:
                            ctx.is_active = True
                            ctx.deactivated_reason = None
                            cambios["reactivadas"].append(ctx.key_index)
                        continue
                    ctx.replace_key(key)
                    cambios["reemplazadas"].append(ctx.key_index)
                if any(ctx.is_active for ctx in self.key_pool):
                    self.pool_exhausted.clear()
            return cambios
        with self._pool_lock:
            for idx, key in enumerate(limpias):
                if idx < len(self.key_pool):
                    ctx = self.key_pool[idx]
                    if ctx.api_key == key:
                        if not ctx.is_active:
                            ctx.is_active = True
                            ctx.deactivated_reason = None
                            cambios["reactivadas"].append(idx)
                        continue
                    ctx.replace_key(key)
                    cambios["reemplazadas"].append(idx)
                elif allow_new_slots:
                    self.key_pool.append(KeyContext(key, idx, idx))
                    cambios["agregadas"].append(idx)
            if any(ctx.is_active for ctx in self.key_pool):
                self.pool_exhausted.clear()
        return cambios

    def pool_status(self) -> List[Dict[str, Any]]:
        """Foto del pool para el reporte de cierre: una fila por ranura."""
        return [{
            "slot": ctx.key_index,
            "proyecto": ctx.project_index,
            "key": ctx.masked,
            "activa": ctx.is_active,
            "motivo_baja": ctx.deactivated_reason,
            **dict(ctx.stats),
        } for ctx in self.key_pool]

    @property
    def active_projects(self) -> List[int]:
        """Proyectos con al menos una key viva en el pool."""
        return sorted({k.project_index for k in self.key_pool if k.is_active})

    def project_wait_time(self, project_index: int) -> float:
        """Menor espera entre las keys vivas de ese proyecto (inf si no hay)."""
        waits = [k.rate_limiter.get_wait_time() for k in self.key_pool
                 if k.is_active and k.project_index == project_index]
        return min(waits) if waits else float("inf")

    def last_request_ok(self) -> bool:
        """True si la última petición de este hilo terminó en HTTP 200.

        Distingue "Riot respondió que no hay datos" (200 con lista vacía) de "la
        petición falló" (red, 5xx, reintentos agotados), que el resto de la API
        del cliente devuelve igual, como lista vacía o `None`.
        """
        return bool(getattr(self._tls, "ok", False))

    def last_served_project(self) -> Optional[int]:
        """Proyecto de la key que respondió con éxito la última petición de este hilo."""
        return getattr(self._tls, "served_project", None)

    def _select_key_and_acquire(self, exclude_ctxs: Optional[set] = None,
                                only_projects: Optional[set] = None) -> Optional[KeyContext]:
        """
        Selecciona la clave con menor tiempo de espera disponible y reserva la ranura.
        Thread-safe: sincronizado con self._pool_lock.

        Devuelve `None` si el pool quedó sin claves vivas; quien llama debe
        rendirse en vez de bloquearse, y `pool_exhausted` queda levantado.
        """
        exclude_set = exclude_ctxs or set()

        while True:
            with self._pool_lock:
                active_keys = [k for k in self.key_pool if k.is_active and k not in exclude_set
                               and (only_projects is None or k.project_index in only_projects)]
                if not active_keys and only_projects is not None \
                        and any(k.is_active for k in self.key_pool):
                    # Ninguna key viva de los proyectos pedidos: decide quien llama.
                    return None
                if not active_keys:
                    active_keys = [k for k in self.key_pool if k.is_active]

                if not active_keys:
                    if not self.pool_exhausted.is_set():
                        print("\n[X] Error: No hay ninguna API Key activa en el pool.")
                    self.pool_exhausted.set()
                    return None

                self.pool_exhausted.clear()
                wait_times = [(ctx, ctx.rate_limiter.get_wait_time()) for ctx in active_keys]
                wait_times.sort(key=lambda item: item[1])

                best_ctx, min_wait = wait_times[0]

                if min_wait <= 0.001:
                    best_ctx.rate_limiter.acquire_slot()
                    return best_ctx

            # Si todas están en cooldown, dormir fuera del lock para no bloquear a otros hilos
            time.sleep(min_wait + 0.02)

    def get_telemetry(self) -> Dict[str, int]:
        """Retorna una copia de las métricas de telemetría de red acumuladas."""
        with self._telemetry_lock:
            return dict(self.telemetry)

    def reset_telemetry(self) -> None:
        """Reinicia los contadores de telemetría para una nueva sesión."""
        with self._telemetry_lock:
            self.telemetry = {
                "total_requests": 0,
                "status_200": 0,
                "status_429": 0,
                "timeouts": 0,
                "http_errors": 0,
                "puuid_mismatch": 0
            }

    @staticmethod
    def _get_response(ctx: KeyContext, url: str, params: Optional[Dict[str, Any]],
                      timeout: float) -> requests.Response:
        response = ctx.session.get(url, params=params, timeout=timeout)
        ctx.rate_limiter.update_from_headers(response.headers)
        return response

    @staticmethod
    def _pause_for_rate_limit(ctx: KeyContext, response: requests.Response) -> None:
        try:
            retry_after = int(response.headers.get("Retry-After", 5))
        except (TypeError, ValueError):
            retry_after = 5
        ctx.rate_limiter.handle_rate_limit_exceeded(retry_after)

    def probe_apex_status(self, timeout: float = 5.0) -> int:
        """Sondea League-V4 una vez y conserva el código HTTP del diagnóstico."""
        ctx = self._select_key_and_acquire()
        if ctx is None:
            return 0
        url = (f"https://{self.platform}.api.riotgames.com"
               f"/lol/league/v4/challengerleagues/by-queue/{QUEUE_SOLOQ_NAME}")
        response = self._get_response(ctx, url, None, timeout)
        if response.status_code == 429:
            self._pause_for_rate_limit(ctx, response)
        return response.status_code

    def _request(self, url: str, params: Optional[Dict[str, Any]] = None, max_retries: int = 6,
                 only_projects: Optional[set] = None) -> Optional[Any]:
        """Realiza una petición HTTP despachando a la clave óptima del pool.

        `only_projects` restringe las keys elegibles (p. ej. al proyecto con el que
        está cifrado el PUUID de la URL). Tras la llamada, `last_served_project()`
        y `self._tls.decrypt_failed` dicen qué proyecto respondió y si Riot
        rechazó el PUUID por no poder descifrarlo.
        """
        retries = 0
        tried_ctxs = set()
        self._tls.served_project = None
        self._tls.ok = False
        self._tls.decrypt_failed = False
        n_elegibles = (self.num_active_keys if only_projects is None else
                       sum(1 for k in self.key_pool if k.is_active and k.project_index in only_projects))

        while retries <= max_retries:
            ctx = self._select_key_and_acquire(
                exclude_ctxs=tried_ctxs if len(tried_ctxs) < max(1, n_elegibles) else None,
                only_projects=only_projects)
            if ctx is None:
                return None
            with self._telemetry_lock:
                self.telemetry["total_requests"] += 1
            ctx.note("requests")

            try:
                response = self._get_response(ctx, url, params, 15)

                if response.status_code == 200:
                    with self._telemetry_lock:
                        self.telemetry["status_200"] += 1
                    ctx.note("ok")
                    self._tls.served_project = ctx.project_index if ctx.project_known else None
                    self._tls.ok = True
                    return response.json()

                if response.status_code == 429:
                    with self._telemetry_lock:
                        self.telemetry["status_429"] += 1
                    ctx.note("rate_limited")
                    self._pause_for_rate_limit(ctx, response)
                    retries += 1
                    continue

                if response.status_code in (401, 403):
                    with self._telemetry_lock:
                        self.telemetry["http_errors"] += 1
                    ctx.note("errors")
                    print(f"\n[X] Error de autenticación en {ctx} (HTTP {response.status_code}):")
                    print("    La API Key no es válida o ha expirado (las Development Keys duran 24 horas).")
                    ctx.is_active = False
                    ctx.deactivated_reason = f"HTTP {response.status_code}"
                    if self.num_active_keys == 0:
                        print("    Todas las API Keys del pool expiraron. Actualiza .env "
                              "(run.py las recarga en caliente sin reiniciar la sesión).")
                        self.pool_exhausted.set()
                        return None
                    print(f"    -> Desactivando {ctx}. Claves restantes activas: {self.num_active_keys}")
                    retries += 1
                    continue

                if response.status_code == 404:
                    return None

                if response.status_code == 400 and "decrypting" in response.text:
                    # El PUUID fue encriptado por otra cuenta; probar con las demás claves del pool.
                    # Se contabiliza porque, sin pool completo, este caso es el modo de
                    # falla dominante y era invisible: la petición no cuenta como error
                    # ni como éxito, así que una sesión entera podía terminar con cero
                    # partidas y la telemetría en blanco (ver TAREAS.md, proyectos 0-2).
                    with self._telemetry_lock:
                        self.telemetry["puuid_mismatch"] = self.telemetry.get("puuid_mismatch", 0) + 1
                    ctx.note("puuid_mismatch")
                    self._tls.decrypt_failed = True
                    tried_ctxs.add(ctx)
                    retries += 1
                    if len(tried_ctxs) >= n_elegibles:
                        return None
                    continue

                if response.status_code in (500, 502, 503, 504):
                    with self._telemetry_lock:
                        self.telemetry["http_errors"] += 1
                    ctx.note("errors")
                    print(f"\n[!] Error del servidor Riot (HTTP {response.status_code}). Reintentando en 3s...")
                    time.sleep(3.0)
                    retries += 1
                    continue

                with self._telemetry_lock:
                    self.telemetry["http_errors"] += 1
                ctx.note("errors")
                print(f"[!] Respuesta inesperada ({response.status_code}) en {url}: {response.text[:200]}")
                return None

            except requests.exceptions.RequestException as e:
                with self._telemetry_lock:
                    self.telemetry["timeouts"] += 1
                ctx.note("errors")
                print(f"[!] Error de red con {ctx} ({e}). Reintentando en 3s...")
                time.sleep(3.0)
                retries += 1

        print(f"[X] Se superó el límite de reintentos para {url}")
        return None



    # -------------------------------------------------------------------------
    # Endpoints League-V4 (Jugadores Diamante y Superiores)
    # -------------------------------------------------------------------------

    def get_apex_league(self, tier: str, queue: str = QUEUE_SOLOQ_NAME) -> List[Dict[str, Any]]:
        """
        Obtiene los jugadores de una liga Apex (CHALLENGER, GRANDMASTER, MASTER).
        """
        tier_lower = tier.lower()
        url = f"https://{self.platform}.api.riotgames.com/lol/league/v4/{tier_lower}leagues/by-queue/{queue}"
        data = self._request(url)
        if not data or "entries" not in data:
            return []

        players = []
        for entry in data.get("entries", []):
            players.append({
                "puuid": entry.get("puuid"),
                "summoner_id": entry.get("summonerId"),
                "tier": tier.upper(),
                "rank": "I",
                "league_points": entry.get("leaguePoints", 0),
                "wins": entry.get("wins", 0),
                "losses": entry.get("losses", 0)
            })
        return players

    def get_diamond_division_entries(self, division: str, page: int = 1, queue: str = QUEUE_SOLOQ_NAME) -> List[Dict[str, Any]]:
        """
        Obtiene una página de jugadores en DIAMOND (I, II, III, IV).
        """
        url = f"https://{self.platform}.api.riotgames.com/lol/league/v4/entries/{queue}/DIAMOND/{division.upper()}"
        data = self._request(url, params={"page": page})
        if not data or not isinstance(data, list):
            return []

        players = []
        for entry in data:
            players.append({
                "puuid": entry.get("puuid"),
                "summoner_id": entry.get("summonerId"),
                "tier": "DIAMOND",
                "rank": division.upper(),
                "league_points": entry.get("leaguePoints", 0),
                "wins": entry.get("wins", 0),
                "losses": entry.get("losses", 0)
            })
        return players

    def get_soloq_entry_by_puuid(self, options: Dict[int, str],
                                 queue: str = QUEUE_SOLOQ_NAME) -> Optional[Dict[str, Any]]:
        """Rango actual de ranked solo de una persona, por PUUID (league-v4 entries/by-puuid).

        `options` = `{project_index: puuid}` de la misma persona: cada key solo descifra
        el PUUID de su proyecto. Devuelve un dict con `puuid` y `project_index` usados,
        y `tier` = 'UNRANKED' si no tiene ranked solo. `None` si ningún proyecto respondió.
        """
        options = dict(options)
        base = f"https://{self.platform}.api.riotgames.com/lol/league/v4/entries/by-puuid"
        while options:
            project = min(options, key=self.project_wait_time)
            puuid = options.pop(project)
            data = self._request(f"{base}/{puuid}", only_projects={project})
            if data is None:
                if getattr(self._tls, "decrypt_failed", False):
                    self._report_project(puuid, project, False)
                continue
            self._report_project(puuid, project, True)
            entry = next((e for e in data if e.get("queueType") == queue), None) if isinstance(data, list) else None
            if entry is None:
                return {"puuid": puuid, "project_index": project, "tier": "UNRANKED", "rank": None,
                        "league_points": 0, "wins": 0, "losses": 0}
            return {"puuid": puuid, "project_index": project, "tier": entry.get("tier", "UNRANKED"),
                    "rank": entry.get("rank"), "league_points": entry.get("leaguePoints", 0),
                    "wins": entry.get("wins", 0), "losses": entry.get("losses", 0)}
        return None

    def fetch_all_diamond_plus_players(self, include_diamond: bool = True) -> List[Dict[str, Any]]:
        """
        Recorre Challenger, Grandmaster, Master y (opcionalmente) todas las páginas de Diamond I-IV.
        """
        all_players = []

        # 1. Apex Tiers
        for tier in APEX_TIERS:
            print(f"[*] Obteniendo liga {tier} en {self.platform.upper()}...")
            tier_players = self.get_apex_league(tier)
            print(f"    -> Encontrados {len(tier_players)} jugadores en {tier}.")
            all_players.extend(tier_players)

        # 2. Diamond I-IV
        if include_diamond:
            for division in DIAMOND_DIVISIONS:
                page = 1
                div_count = 0
                print(f"[*] Obteniendo DIAMOND {division}...")
                while True:
                    entries = self.get_diamond_division_entries(division, page=page)
                    if not entries:
                        break
                    all_players.extend(entries)
                    div_count += len(entries)
                    page += 1
                print(f"    -> Encontrados {div_count} jugadores en DIAMOND {division} ({page-1} páginas).")

        return all_players

    # -------------------------------------------------------------------------
    # Endpoints Match-V5 (Partidas y Estadísticas)
    # -------------------------------------------------------------------------

    def _report_project(self, puuid: str, project_index: int, ok: bool) -> None:
        if self.project_feedback is not None:
            try:
                self.project_feedback(puuid, project_index, ok)
            except Exception as err:  # el feedback nunca puede tumbar la recolección
                print(f"[!] Feedback de proyecto descartado: {err}")

    def _match_ids_page(self, default_puuid: str, options: Dict[int, str],
                        params: Dict[str, Any]) -> Optional[Any]:
        """Una página de IDs, mandando a cada key un PUUID que pueda descifrar.

        `options` = `{project_index: puuid}` de la misma persona. Se elige el
        proyecto cuya key esté más libre; si Riot no puede descifrar ese PUUID
        (etiqueta falsa), se informa, se descarta ese proyecto para el resto de
        las páginas y se prueba el siguiente. Sin opciones utilizables se cae al
        comportamiento anterior: `default_puuid` con todo el pool.
        """
        base = f"https://{self.regional_routing}.api.riotgames.com/lol/match/v5/matches/by-puuid"
        while options:
            project = min(options, key=self.project_wait_time)
            puuid = options[project]
            data = self._request(f"{base}/{puuid}/ids", params=params, only_projects={project})
            if data is None and getattr(self._tls, "decrypt_failed", False):
                self._report_project(puuid, project, False)
                del options[project]
                continue
            if self.last_served_project() is not None:
                self._report_project(puuid, project, True)
            return data
        return self._request(f"{base}/{default_puuid}/ids", params=params)

    def get_match_ids_by_puuid(self, puuid: str, count: int = 100, queue: int = QUEUE_SOLOQ_ID,
                                start: int = 0, start_time: Optional[int] = None,
                                candidates: Optional[Dict[int, str]] = None,
                                end_time: Optional[int] = None) -> List[str]:
        """
        Obtiene los IDs de partidas de Ranked Solo para un PUUID dado.
        Soporta lotes de hasta 100 por petición y paginación automática para count > 100.

        `candidates` (`{project_index: puuid}` de la misma persona) permite repartir
        las listas entre todas las keys del pool sin mandarle a ninguna un PUUID
        que no pueda descifrar (ver `_match_ids_page`). Los proyectos que Riot
        rechace se quitan también de `candidates`, para que las páginas
        siguientes del mismo jugador no los vuelvan a probar.
        """
        all_ids = []
        remaining = count
        current_start = start
        vivos = set(self.active_projects)
        options = {int(pj): pu for pj, pu in (candidates or {}).items() if pu and int(pj) in vivos}
        try:
            return self._paginate_match_ids(puuid, options, queue, current_start, remaining,
                                            start_time, all_ids, end_time)
        finally:
            if candidates is not None:
                for pj in [p for p in candidates if int(p) in vivos and int(p) not in options]:
                    candidates.pop(pj, None)

    def _paginate_match_ids(self, puuid: str, options: Dict[int, str], queue: int,
                            current_start: int, remaining: int, start_time: Optional[int],
                            all_ids: List[str], end_time: Optional[int] = None) -> List[str]:
        """Paginación de `get_match_ids_by_puuid` (lotes de hasta 100)."""
        while remaining > 0:
            batch_size = min(remaining, 100)
            params = {
                "queue": queue,
                "type": "ranked",
                "start": current_start,
                "count": batch_size
            }
            # Riot espera ambos en segundos epoch (no milisegundos como gameCreation).
            if start_time:
                params["startTime"] = int(start_time)
            if end_time:
                params["endTime"] = int(end_time)

            data = self._match_ids_page(puuid, options, params)
            if not data or not isinstance(data, list):
                break

            all_ids.extend(data)
            if len(data) < batch_size:
                # Ya no hay más partidas disponibles para este jugador
                break

            remaining -= len(data)
            current_start += len(data)

        return all_ids

    def get_account_by_puuid(self, puuid: str) -> Optional[Tuple[str, str]]:
        """
        Resuelve la identidad (gameName, tagLine) de un PUUID vía Account-V1.

        Devuelve None si el PUUID no es descifrable por las claves actuales
        (proyecto muerto tras rotar keys) o si la cuenta no existe (404).
        El Riot ID es estable entre proyectos, por lo que actúa como clave
        canónica para unificar los PUUIDs que Riot emite por proyecto.
        """
        url = (f"https://{self.regional_routing}.api.riotgames.com"
               f"/riot/account/v1/accounts/by-puuid/{puuid}")
        data = self._request(url)
        if not data or not data.get("gameName"):
            return None
        return data.get("gameName"), data.get("tagLine", "")

    def get_match_detail(self, match_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene los detalles completos y estadísticas de una partida por su match_id.
        """
        url = f"https://{self.regional_routing}.api.riotgames.com/lol/match/v5/matches/{match_id}"
        return self._request(url)

    def get_match_timeline(self, match_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene el timeline detallado de eventos y frames de una partida por su match_id.
        """
        url = f"https://{self.regional_routing}.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"
        return self._request(url)

