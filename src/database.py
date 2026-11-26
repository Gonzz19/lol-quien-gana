"""
Módulo de base de datos SQLite para almacenamiento y deduplicación de jugadores, partidas y estadísticas.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import List, Dict, Any, Set, Optional, Tuple, Generator
from src.config import DEFAULT_DB_PATH, TIER_ORDER_LIGA

# Una partida de SoloQ tiene exactamente 10 participantes. Las que llegan con otro
# número no se guardan (§9.S.76): romperían la invariante de 10 filas por partida.
PARTICIPANTES_POR_PARTIDA = 10


def sql_rango_tier(col: str) -> str:
    """Expresión SQL con la posición de `col` en el orden de liga (1 = IRON ... 10 =
    CHALLENGER). 0 para NULL, 'UNKNOWN' o cualquier valor fuera de la lista."""
    casos = " ".join(f"WHEN '{t}' THEN {i}" for i, t in enumerate(TIER_ORDER_LIGA, start=1))
    return f"(CASE UPPER({col}) {casos} ELSE 0 END)"


def sql_tier_maximo(col: str) -> str:
    """Agregado SQL: el tier más alto de `col` en el grupo según el orden de liga.

    Reemplaza a `MAX(col)`, que compara texto (DIAMOND > CHALLENGER > MASTER). Si
    ningún valor del grupo es un tier conocido devuelve `MAX(col)` tal cual, así un
    grupo solo con 'UNKNOWN' o NULL se comporta como antes.
    """
    casos = " ".join(f"WHEN {i} THEN '{t}'" for i, t in enumerate(TIER_ORDER_LIGA, start=1))
    return f"(CASE MAX({sql_rango_tier(col)}) {casos} ELSE MAX({col}) END)"


def _configure_read_queries(conn: sqlite3.Connection) -> None:
    # Un límite de caché, reservado por SQLite a medida que lee páginas.
    conn.execute("PRAGMA cache_size = -262144;")
    conn.execute("PRAGMA temp_store = MEMORY;")


def open_readonly_connection(db_path: Path, *, timeout: float = 60.0,
                             query_only: bool = False) -> sqlite3.Connection:
    """Abre la base principal en solo lectura. El llamador cierra la conexión."""
    conn = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro",
                           uri=True, timeout=timeout)
    try:
        _configure_read_queries(conn)
        if query_only:
            conn.execute("PRAGMA query_only = ON;")
        return conn
    except BaseException:
        conn.close()
        raise


class DatabaseManager:
    """
    Gestiona la conexión y operaciones en SQLite.
    Garantiza deduplicación de partidas e invocadores y transacciones atómicas.
    """

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Caché en memoria alias_puuid -> puuid canónico (proyecto 0).
        # Se carga una sola vez por sesión (cero requests a Riot en el hot path).
        self._translation_cache: Dict[str, str] = {}
        # Partidas que llegaron con un número de participantes distinto de 10 y no
        # se guardaron (acumulado de la vida del objeto; el collector lee la
        # diferencia por lote). No van a `discarded_matches`: un payload truncado
        # puede ser transitorio y conviene volver a pedirlo en otra sesión.
        self.partidas_rechazadas_participantes = 0
        # Participantes que exige el guardado. `None` desactiva el control: solo lo
        # usa la fixture `temp_db` de los tests, cuyas partidas de juguete traen 1 o
        # 2 participantes para probar otra cosa (identidad, export).
        self.participantes_esperados: Optional[int] = PARTICIPANTES_POR_PARTIDA
        self._init_tables()

    def _participantes_completos(self, match_id: str, participants: List[Dict[str, Any]]) -> bool:
        """True si la partida trae exactamente 10 participantes con PUUIDs distintos.

        Si no, la registra en el log y en `partidas_rechazadas_participantes` y
        devuelve False para que el llamador no la guarde. No lanza excepción: una
        partida rara no corta la recolección (§9.S.76).
        """
        esperados = self.participantes_esperados
        if esperados is None:
            return True
        puuids = {p.get("puuid") for p in participants if p.get("puuid")}
        if len(participants) == esperados and len(puuids) == esperados:
            return True
        self.partidas_rechazadas_participantes += 1
        print(f"\n[!] Partida {match_id} rechazada: {len(participants)} participantes "
              f"({len(puuids)} PUUIDs distintos); se esperan {esperados}.")
        return False

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout = 60000;")
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            _configure_read_queries(conn)
            yield conn
        finally:
            conn.close()

    def _init_tables(self) -> None:
        """Crea las tablas e índices si no existen."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Tabla de Jugadores
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS players (
                    puuid TEXT PRIMARY KEY,
                    summoner_id TEXT,
                    tier TEXT NOT NULL,
                    rank TEXT NOT NULL,
                    league_points INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_crawled_at TIMESTAMP
                );
            """)

            # Migración: asegurar que la columna last_crawled_at exista si la base ya fue creada
            try:
                cursor.execute("ALTER TABLE players ADD COLUMN last_crawled_at TIMESTAMP;")
            except sqlite3.OperationalError:
                pass  # La columna ya existe

            # Tabla de Partidas
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS matches (
                    match_id TEXT PRIMARY KEY,
                    game_creation INTEGER,
                    game_duration INTEGER,
                    game_version TEXT,
                    queue_id INTEGER,
                    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Tabla de Estadísticas de Participantes (10 por partida)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS match_participants (
                    match_id TEXT NOT NULL,
                    puuid TEXT NOT NULL,
                    participant_id INTEGER,
                    team_id INTEGER,
                    win INTEGER,
                    champion_id INTEGER,
                    champion_name TEXT,
                    individual_position TEXT,
                    kills INTEGER DEFAULT 0,
                    deaths INTEGER DEFAULT 0,
                    assists INTEGER DEFAULT 0,
                    kda REAL DEFAULT 0.0,
                    total_damage_dealt_to_champions INTEGER DEFAULT 0,
                    total_damage_taken INTEGER DEFAULT 0,
                    gold_earned INTEGER DEFAULT 0,
                    total_minions_killed INTEGER DEFAULT 0,
                    neutral_minions_killed INTEGER DEFAULT 0,
                    vision_score INTEGER DEFAULT 0,
                    champ_level INTEGER DEFAULT 1,
                    item0 INTEGER DEFAULT 0,
                    item1 INTEGER DEFAULT 0,
                    item2 INTEGER DEFAULT 0,
                    item3 INTEGER DEFAULT 0,
                    item4 INTEGER DEFAULT 0,
                    item5 INTEGER DEFAULT 0,
                    item6 INTEGER DEFAULT 0,
                    first_blood_kill INTEGER DEFAULT 0,
                    first_tower_kill INTEGER DEFAULT 0,
                    PRIMARY KEY (match_id, puuid),
                    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
                );
            """)

            # Tabla de Snapshots del Timeline (Minutos 10, 15 y 20)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS match_timeline (
                    match_id TEXT NOT NULL,
                    participant_id INTEGER NOT NULL,
                    puuid TEXT,
                    gold_10 INTEGER,
                    xp_10 INTEGER,
                    level_10 INTEGER,
                    cs_10 INTEGER,
                    damage_10 INTEGER,
                    kills_10 INTEGER,
                    deaths_10 INTEGER,
                    assists_10 INTEGER,
                    gold_15 INTEGER,
                    xp_15 INTEGER,
                    level_15 INTEGER,
                    cs_15 INTEGER,
                    damage_15 INTEGER,
                    kills_15 INTEGER,
                    deaths_15 INTEGER,
                    assists_15 INTEGER,
                    gold_20 INTEGER,
                    xp_20 INTEGER,
                    level_20 INTEGER,
                    cs_20 INTEGER,
                    damage_20 INTEGER,
                    kills_20 INTEGER,
                    deaths_20 INTEGER,
                    assists_20 INTEGER,
                    PRIMARY KEY (match_id, participant_id),
                    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
                );
            """)

            # Tabla de Partidas Descartadas (Remakes <= 300s o no válidas)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS discarded_matches (
                    match_id TEXT PRIMARY KEY,
                    reason TEXT,
                    discarded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Tabla de Telemetría Histórica y Métricas en Tiempo Real (Timeline de Extracción)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS crawler_metrics_timeline (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    elapsed_minutes REAL,
                    active_keys INTEGER,
                    workers INTEGER,
                    players_crawled INTEGER,
                    total_matches_in_db INTEGER,
                    new_matches_saved INTEGER,
                    duplicate_matches_skipped INTEGER,
                    discarded_matches INTEGER,
                    net_matches_per_min REAL,
                    requests_per_min REAL,
                    c50_challenger INTEGER,
                    c50_grandmaster INTEGER,
                    c50_master INTEGER,
                    c50_total INTEGER,
                    db_size_mb REAL
                );
            """)
            # Migración (§9.S.76): contadores aparte para los lotes que fallaron al
            # escribirse y para las partidas rechazadas por no traer 10 participantes.
            for col in ("write_errors", "rejected_participants"):
                try:
                    cursor.execute(f"ALTER TABLE crawler_metrics_timeline ADD COLUMN {col} INTEGER DEFAULT 0;")
                except sqlite3.OperationalError:
                    pass  # La columna ya existe

            # Tabla de Resolución y Auditoría de Identidades (PUUIDs Duales / Alias de Cuentas)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS player_puuid_aliases (
                    primary_puuid TEXT NOT NULL,
                    alias_puuid TEXT NOT NULL,
                    game_name TEXT,
                    tag_line TEXT,
                    matches_migrated INTEGER DEFAULT 0,
                    resolved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (primary_puuid, alias_puuid)
                );
            """)

            # Registro de proyectos de API Key (identidad multi-proyecto de PUUIDs).
            # Cada posición del pool RIOT_API_KEYS corresponde a un proyecto Riot distinto
            # con su propio project_secret; el PUUID de un mismo humano difiere por proyecto.
            # project_index = posición en la lista de claves (estable aunque la key rote cada 24h).
            # Solo se guarda un hint enmascarado, NUNCA la clave completa.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS api_projects (
                    project_index INTEGER PRIMARY KEY,
                    key_hint TEXT NOT NULL,
                    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Identidad canónica del jugador verdadero (estable entre proyectos).
            # canonical_id = lower(gameName#tagLine) verificado vía Account-V1.
            # El proyecto 0 es el canónico: su PUUID es el que se persiste en match_participants.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS player_identities (
                    canonical_id TEXT PRIMARY KEY,
                    game_name TEXT NOT NULL,
                    tag_line TEXT NOT NULL,
                    tier TEXT NOT NULL DEFAULT 'UNKNOWN',
                    rank TEXT NOT NULL DEFAULT 'I',
                    league_points INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Observaciones crudas nombre <-> PUUID vistas en el payload de Match-V5.
            # `riotIdGameName`/`riotIdTagline` NO están cifrados: son idénticos sin
            # importar qué key pidió la partida. Por eso esta tabla es el único puente
            # que funciona entre proyectos de API cuyos PUUID son mutuamente
            # indescifrables, y a la vez el historial de renombres por PUUID
            # (`first_seen`/`last_seen` acotan cuándo ese humano usó ese nombre).
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS puuid_riot_ids (
                    puuid TEXT NOT NULL,
                    canonical_id TEXT NOT NULL,
                    game_name TEXT NOT NULL,
                    tag_line TEXT NOT NULL,
                    project_index INTEGER,
                    observations INTEGER NOT NULL DEFAULT 1,
                    first_seen_game_creation INTEGER,
                    last_seen_game_creation INTEGER,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (puuid, canonical_id)
                );
            """)

            # Un PUUID por (jugador verdadero, proyecto). Invariante de integridad:
            # todo canonical_id debe tener exactamente len(api_projects) filas (una por proyecto).
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS player_puuid_map (
                    canonical_id TEXT NOT NULL REFERENCES player_identities(canonical_id),
                    project_index INTEGER NOT NULL REFERENCES api_projects(project_index),
                    puuid TEXT NOT NULL UNIQUE,
                    summoner_id TEXT DEFAULT '',
                    last_verified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (canonical_id, project_index)
                );
            """)

            # Índices de optimización de consultas
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_players_tier ON players(tier, rank);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_creation ON matches(game_creation);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_participants_puuid ON match_participants(puuid);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_participants_champion ON match_participants(champion_name);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_participants_position ON match_participants(individual_position);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timeline_match ON match_timeline(match_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON crawler_metrics_timeline(timestamp);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alias_puuid ON player_puuid_aliases(alias_puuid);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_primary_puuid ON player_puuid_aliases(primary_puuid);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_riotids_canonical ON puuid_riot_ids(canonical_id);")

            # Directorio de proyecto por PUUID: con qué proyecto de API está cifrado
            # cada PUUID conocido. Es lo que permite mandarle a Riot, para cada
            # jugador, un PUUID que la key elegida pueda descifrar.
            #   source = 'exact'    -> lo dijo Riot: la key de ese proyecto devolvió
            #                          o descifró el PUUID (ingesta, sync, mapa).
            #   source = 'inferred' -> deducido por los demás PUUIDs de sus partidas
            #                          (ver src/features/puuid_projects.py).
            # Un 'inferred' nunca pisa un 'exact'; el crawler corrige los
            # 'inferred' con cada respuesta de Riot (éxito o 400 de descifrado).
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS puuid_projects (
                    puuid TEXT PRIMARY KEY,
                    project_index INTEGER NOT NULL,
                    source TEXT NOT NULL CHECK (source IN ('exact', 'inferred')),
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_puuid_projects_proj ON puuid_projects(project_index);")

            # Recolección por ventana de tiempo (src/window_crawl.py). `window_key`
            # identifica la ventana ("2026-10-01..2026-10-25"); el estado por
            # jugador y por partida permite cortar y retomar sin repetir requests.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS crawl_window_players (
                    window_key TEXT NOT NULL,
                    player_id TEXT NOT NULL,
                    n_ids INTEGER NOT NULL,
                    listed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (window_key, player_id)
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS crawl_window_matches (
                    window_key TEXT NOT NULL,
                    match_id TEXT NOT NULL,
                    PRIMARY KEY (window_key, match_id)
                );
            """)
            # Qué partidas devolvió Riot para cada jugador listado. Riot garantiza
            # que el jugador jugó cada una, así que es la prueba directa de que no
            # quedó partido en dos: al bajarla, tiene que aparecer en ella
            # (scripts/ops/audit_duplicates.py, "listado_ausente_en_su_partida").
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS crawl_window_listings (
                    window_key TEXT NOT NULL,
                    players_puuid TEXT NOT NULL,  -- PUUID del padrón: estable, a diferencia de la etiqueta
                    match_id TEXT NOT NULL,
                    PRIMARY KEY (window_key, players_puuid, match_id)
                );
            """)

            # Tabla persistente del mapeo de identidades por Union-Find
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS player_id_map (
                    puuid TEXT PRIMARY KEY,
                    player_id TEXT NOT NULL
                );
            """)
            # Fotos fechadas del rango (PROJECT_SUMMARY.md §9.S.21). A diferencia de
            # `players`, que se pisa en cada sync, acá cada foto queda separada por
            # `snapshot_at`: el rating necesita el LP de cada jugador en más de una
            # fecha. `fuente`: 'ladder' (league-v4 por tier/división) o 'by_puuid'
            # (consulta individual de quien no aparece en el ladder; 'UNRANKED' si no
            # tiene ranked solo).
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ladder_snapshots (
                    snapshot_at TEXT NOT NULL,
                    puuid TEXT NOT NULL,
                    project_index INTEGER NOT NULL,
                    tier TEXT NOT NULL,
                    rank TEXT,
                    league_points INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0,
                    fuente TEXT NOT NULL,
                    PRIMARY KEY (snapshot_at, puuid)
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_player_id_map_pid ON player_id_map(player_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_puuidmap_puuid ON player_puuid_map(puuid);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_puuidmap_canonical ON player_puuid_map(canonical_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_identities_tier ON player_identities(tier, rank);")

            # Vista unificada alias -> canónico (única fuente de verdad para traducir).
            # Une el mapa live multi-proyecto con los links legacy (PUUIDs de proyectos
            # muertos, indescifrables por las keys actuales). Precedencia: el mapa live
            # se consulta primero en Python; la vista sirve a SQL (export/coverage/crawl).
            view_sql = """
                CREATE VIEW v_puuid_canonical AS
                SELECT m.puuid AS alias_puuid, c.puuid AS canonical_puuid,
                       m.canonical_id AS canonical_id
                FROM player_puuid_map m
                INNER JOIN player_puuid_map c
                  ON c.canonical_id = m.canonical_id AND c.project_index = 0
                UNION
                SELECT a.alias_puuid AS alias_puuid, a.primary_puuid AS canonical_puuid,
                       COALESCE(
                           (SELECT m2.canonical_id FROM player_puuid_map m2
                            WHERE m2.puuid = a.primary_puuid AND m2.project_index = 0),
                           NULLIF(LOWER(COALESCE(a.game_name, '')) || '#'
                                  || LOWER(COALESCE(a.tag_line, '')), '#'),
                           a.primary_puuid
                       ) AS canonical_id
                FROM player_puuid_aliases a
                WHERE NOT EXISTS (SELECT 1 FROM player_puuid_map m
                                  WHERE m.puuid = a.alias_puuid)
                UNION
                SELECT a.primary_puuid AS alias_puuid, a.primary_puuid AS canonical_puuid,
                       COALESCE(
                           (SELECT m2.canonical_id FROM player_puuid_map m2
                            WHERE m2.puuid = a.primary_puuid AND m2.project_index = 0),
                           MAX(NULLIF(LOWER(COALESCE(a.game_name, '')) || '#'
                                  || LOWER(COALESCE(a.tag_line, '')), '#')),
                           a.primary_puuid
                       ) AS canonical_id
                FROM player_puuid_aliases a
                WHERE NOT EXISTS (SELECT 1 FROM player_puuid_map m
                                  WHERE m.puuid = a.primary_puuid)
                  AND NOT EXISTS (SELECT 1 FROM player_puuid_aliases a2
                                  WHERE a2.alias_puuid = a.primary_puuid)
                GROUP BY a.primary_puuid;
            """.strip().removesuffix(";")
            current_view = cursor.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'view' AND name = 'v_puuid_canonical';"
            ).fetchone()
            if current_view is None or current_view[0] != view_sql:
                cursor.execute("SAVEPOINT migrate_canonical_view;")
                cursor.execute("DROP VIEW IF EXISTS v_puuid_canonical;")
                cursor.execute(view_sql)
                cursor.execute("RELEASE migrate_canonical_view;")

            conn.commit()

    def _canonical_puuid(self, puuid: str) -> str:
        """Traduce un PUUID de cualquier proyecto a su PUUID canónico (proyecto 0).

        O(1) en memoria, sin I/O ni requests a Riot. Si el PUUID no está mapeado,
        se devuelve intacto (pass-through seguro antes del primer registro de keys).
        """
        if not puuid:
            return puuid
        return self._translation_cache.get(puuid, puuid)

    def reload_translation_cache(self) -> int:
        """Carga (o recarga) el mapa alias->canónico a memoria. Retorna entradas cargadas."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT alias_puuid, canonical_puuid FROM v_puuid_canonical;")
            rows = cursor.fetchall()
        # Precedencia al mapa live: se carga primero el legacy y luego el live lo pisa.
        live_first: Dict[str, str] = {}
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT m.puuid AS alias_puuid, c.puuid AS canonical_puuid
                FROM player_puuid_map m
                INNER JOIN player_puuid_map c
                  ON c.canonical_id = m.canonical_id AND c.project_index = 0
                WHERE m.project_index != 0;
            """)
            for r in cursor.fetchall():
                live_first[r[0]] = r[1]
        merged = {r[0]: r[1] for r in rows}
        merged.update(live_first)
        self._translation_cache = merged
        return len(self._translation_cache)

    _PUUID_PROJECT_UPSERT_SQL = """
        INSERT INTO puuid_projects (puuid, project_index, source, updated_at)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(puuid) DO UPDATE SET
            project_index = excluded.project_index,
            source = excluded.source,
            updated_at = CURRENT_TIMESTAMP
        WHERE excluded.source = 'exact' OR puuid_projects.source = 'inferred';
    """

    def record_puuid_projects(self, rows: List[Tuple[str, int]], source: str = "exact") -> int:
        """Anota `(puuid, project_index)` en el directorio. Un 'inferred' no pisa un 'exact'."""
        if source not in ("exact", "inferred"):
            raise ValueError(f"source inválido: {source}")
        data = [(p, int(pj), source) for p, pj in rows or () if p and pj is not None]
        if not data:
            return 0
        with self._get_connection() as conn:
            conn.executemany(self._PUUID_PROJECT_UPSERT_SQL, data)
            conn.commit()
        return len(data)

    def forget_puuid_project(self, puuid: str, project_index: int) -> bool:
        """Borra la etiqueta `puuid -> project_index` porque esa key no pudo descifrarlo.

        Riot respondió `400 Exception decrypting`: la etiqueta era falsa. Borrarla
        (en vez de adivinar otra) deja al PUUID sin proyecto hasta que una
        respuesta exitosa lo confirme.
        """
        with self._get_connection() as conn:
            cur = conn.execute(
                "DELETE FROM puuid_projects WHERE puuid = ? AND project_index = ?;",
                (puuid, int(project_index)))
            conn.commit()
            return (cur.rowcount or 0) > 0

    def apply_project_feedback(self, feedback: List[Tuple[str, int, bool]]) -> Dict[str, int]:
        """Aplica lo que Riot confirmó: `(puuid, project, True)` = descifrable, `False` = no."""
        ok = [(p, pj) for p, pj, good in feedback or () if good]
        bad = [(p, pj) for p, pj, good in feedback or () if not good]
        n_ok = self.record_puuid_projects(ok, source="exact") if ok else 0
        n_bad = sum(self.forget_puuid_project(p, pj) for p, pj in bad)
        return {"confirmados": n_ok, "corregidos": n_bad}

    def materialize_player_id_map(self) -> int:
        """Calcula el Union-Find sobre el estado actual y materializa la tabla permanente player_id_map."""
        from src.features.identity import materialize_for_database
        with self._get_connection() as conn:
            return materialize_for_database(conn)

    def save_ladder_snapshot(self, snapshot_at: str, entries: List[Dict[str, Any]],
                             project_index: int, fuente: str) -> int:
        """Guarda filas de una foto fechada del rango, sin tocar `players`.

        Idempotente por `(snapshot_at, puuid)`: re-guardar la misma foto actualiza.
        """
        if fuente not in ("ladder", "by_puuid"):
            raise ValueError(f"fuente inválida: {fuente}")
        rows = [(snapshot_at, e["puuid"], int(project_index), str(e.get("tier", "UNRANKED")).upper(),
                 e.get("rank"), int(e.get("league_points") or 0), int(e.get("wins") or 0),
                 int(e.get("losses") or 0), fuente)
                for e in entries or () if e.get("puuid")]
        if not rows:
            return 0
        with self._get_connection() as conn:
            conn.executemany("""
                INSERT INTO ladder_snapshots
                    (snapshot_at, puuid, project_index, tier, rank, league_points, wins, losses, fuente)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(snapshot_at, puuid) DO UPDATE SET
                    project_index=excluded.project_index, tier=excluded.tier, rank=excluded.rank,
                    league_points=excluded.league_points, wins=excluded.wins,
                    losses=excluded.losses, fuente=excluded.fuente;
            """, rows)
            conn.commit()
        return len(rows)

    def ladder_snapshot_puuids(self, snapshot_at: str) -> set:
        """PUUIDs ya guardados en una foto (para retomar una foto cortada)."""
        with self._get_connection() as conn:
            return {r[0] for r in conn.execute(
                "SELECT puuid FROM ladder_snapshots WHERE snapshot_at = ?;", (snapshot_at,))}

    def save_players(self, players_data: List[Dict[str, Any]]) -> int:
        """
        Inserta o actualiza una lista de jugadores.
        Retorna la cantidad de registros procesados.
        """
        if not players_data:
            return 0

        query = """
            INSERT INTO players (puuid, summoner_id, tier, rank, league_points, wins, losses, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(puuid) DO UPDATE SET
                summoner_id=excluded.summoner_id,
                tier=excluded.tier,
                rank=excluded.rank,
                league_points=excluded.league_points,
                wins=excluded.wins,
                losses=excluded.losses,
                updated_at=CURRENT_TIMESTAMP;
        """

        rows = []
        for p in players_data:
            puuid = p.get("puuid")
            if not puuid:
                continue
            rows.append((
                puuid,
                p.get("summoner_id", ""),
                p.get("tier", "UNKNOWN"),
                p.get("rank", "I"),
                int(p.get("league_points", 0)),
                int(p.get("wins", 0)),
                int(p.get("losses", 0))
            ))

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, rows)
            conn.commit()

        return len(rows)

    def get_all_players(self, tier: Optional[str] = None, unvisited_first: bool = True) -> List[Dict[str, Any]]:
        """
        Obtiene la lista de jugadores registrados, opcionalmente filtrados por tier.
        Si unvisited_first es True, prioriza aquellos jugadores que nunca han sido consultados (last_crawled_at NULL).
        """
        query = "SELECT * FROM players"
        params = []
        if tier:
            tier_upper = tier.upper()
            if tier_upper == "APEX":
                target_tiers = ["CHALLENGER", "GRANDMASTER"]
            elif "," in tier_upper:
                target_tiers = [t.strip() for t in tier_upper.split(",") if t.strip()]
            else:
                target_tiers = [tier_upper]
            placeholders = ",".join(["?"] * len(target_tiers))
            query += f" WHERE tier IN ({placeholders})"
            params.extend(target_tiers)

        if unvisited_first:
            query += " ORDER BY CASE WHEN last_crawled_at IS NULL THEN 0 ELSE 1 END, last_crawled_at ASC, tier, league_points DESC;"
        else:
            query += " ORDER BY tier, league_points DESC;"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_players_for_deep_crawl(self, tier: Optional[str] = None, min_matches_target: int = 50, only_incomplete: bool = False) -> List[Dict[str, Any]]:
        """
        Prioriza jugadores para recolección en profundidad (arquetipos):
        1. Jugadores que ya tienen partidas en la BD pero aún no alcanzan el objetivo (50 partidas).
        2. Jugadores con mayor LP / elo alto no consultados recientemente.
        Si only_incomplete es True, excluye directamente a quienes ya alcanzaron min_matches_target.
        """
        query = """
            SELECT 
                p.puuid,
                p.summoner_id,
                p.tier,
                p.rank,
                p.league_points,
                p.last_crawled_at,
                COUNT(mp.match_id) AS existing_recorded_games
            FROM players p
            LEFT JOIN match_participants mp ON p.puuid = mp.puuid
        """
        params = []
        if tier:
            tier_upper = tier.upper()
            if tier_upper == "APEX":
                target_tiers = ["CHALLENGER", "GRANDMASTER"]
            elif "," in tier_upper:
                target_tiers = [t.strip() for t in tier_upper.split(",") if t.strip()]
            else:
                target_tiers = [tier_upper]
            placeholders = ",".join(["?"] * len(target_tiers))
            query += f" WHERE p.tier IN ({placeholders})"
            params.extend(target_tiers)

        query += " GROUP BY p.puuid "

        if only_incomplete:
            query += " HAVING COUNT(mp.match_id) < ? "
            params.append(min_matches_target)

        query += """
            ORDER BY 
                CASE WHEN COUNT(mp.match_id) > 0 AND COUNT(mp.match_id) < ? THEN 0 ELSE 1 END,
                CASE WHEN p.last_crawled_at IS NULL THEN 0 ELSE 1 END,
                p.last_crawled_at ASC,
                COUNT(mp.match_id) DESC,
                p.league_points DESC;
        """
        params.append(min_matches_target)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def mark_player_crawled(self, puuid: str) -> None:
        """Marca un jugador como consultado actualizando last_crawled_at."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE players SET last_crawled_at = CURRENT_TIMESTAMP WHERE puuid = ?;",
                (puuid,)
            )
            conn.commit()

    def get_player_match_count(self, puuid: str) -> int:
        """Retorna la cantidad exacta de partidas registradas para un jugador (unificando sus PUUIDs)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            has_map = cursor.execute("SELECT EXISTS(SELECT 1 FROM player_id_map);").fetchone()[0]
            alias_sql = """
                    SELECT m2.puuid
                    FROM player_id_map m1
                    INNER JOIN player_id_map m2 ON m2.player_id = m1.player_id
                    WHERE m1.puuid = ?
            """ if has_map else """
                    SELECT v2.alias_puuid
                    FROM v_puuid_canonical v1
                    INNER JOIN v_puuid_canonical v2 ON v2.canonical_id = v1.canonical_id
                    WHERE v1.alias_puuid = ? OR v1.canonical_puuid = ?
            """
            params = (puuid, puuid) if has_map else (puuid, puuid, puuid)
            cursor.execute(f"""
                SELECT COUNT(DISTINCT mp.match_id)
                FROM match_participants mp
                WHERE mp.puuid IN (
                    SELECT ?
                    UNION
                    {alias_sql}
                );
            """, params)
            return cursor.fetchone()[0]

    def is_match_saved(self, match_id: str) -> bool:
        """Verifica de forma rápida si un match_id ya está en la base de datos."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM matches WHERE match_id = ? LIMIT 1;", (match_id,))
            return cursor.fetchone() is not None


    def filter_unprocessed_match_ids(self, match_ids: List[str]) -> List[str]:
        """
        Filtra una lista de IDs de partidas y retorna únicamente las que NO han sido guardadas.
        Evita consultar detalles de partidas ya existentes en la base de datos.
        """
        if not match_ids:
            return []

        placeholders = ",".join("?" for _ in match_ids)
        query = f"""
            SELECT match_id FROM matches WHERE match_id IN ({placeholders})
            UNION
            SELECT match_id FROM discarded_matches WHERE match_id IN ({placeholders});
        """

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, match_ids + match_ids)
            existing = {row[0] for row in cursor.fetchall()}

        return [m_id for m_id in match_ids if m_id not in existing]

    def _identity_rows(self, p: Dict[str, Any], game_creation: int,
                       served_project: Optional[int]) -> Tuple[Optional[str], Optional[tuple], Optional[tuple]]:
        """PUUID a guardar + filas de identidad de un participante del Match DTO.

        El PUUID se guarda **tal cual lo devolvió Riot**, sin traducirlo: está
        cifrado con el proyecto de la key que pidió la partida (`served_project`).
        La unificación por persona ocurre en lectura (`player_id_map`). Traducirlo
        al escribir lo reemplazaba por el PUUID que había dejado la resolución de
        alias, que podía ser de un proyecto sin key en el pool (ver
        `PROJECT_SUMMARY.md` §9.S.4).

        Devuelve `(puuid, fila_riot_id | None, fila_proyecto | None)`.
        """
        puuid = p.get("puuid")
        if not puuid:
            return None, None, None
        riot_row = None
        gn = (p.get("riotIdGameName") or "").strip()
        tag = (p.get("riotIdTagline") or "").strip()
        if gn:
            riot_row = (puuid, self.canonical_id_for(gn, tag), gn, tag, served_project,
                        game_creation, game_creation)
        project_row = (puuid, int(served_project), "exact") if served_project is not None else None
        return puuid, riot_row, project_row

    def save_match_and_participants(self, match_detail: Dict[str, Any],
                                    served_project: Optional[int] = None) -> bool:
        """
        Guarda los datos generales de una partida y las estadísticas de sus 10 participantes en una sola transacción atómica.
        `served_project` es el proyecto de la key que descargó la partida (ver `_identity_rows`).
        Retorna True si fue guardada con éxito, False si ya existía o hubo un error.
        """
        metadata = match_detail.get("metadata", {})
        info = match_detail.get("info", {})
        match_id = metadata.get("matchId")

        if not match_id or not info:
            return False

        game_creation = info.get("gameCreation", 0)
        game_duration = info.get("gameDuration", 0)
        # Ignorar y registrar remakes (partidas <= 300s / 5 minutos)
        if game_duration <= 300:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR IGNORE INTO discarded_matches (match_id, reason, discarded_at)
                    VALUES (?, 'remake_duration_le_300', CURRENT_TIMESTAMP);
                """, (match_id,))
                conn.commit()
            return False
        game_version = info.get("gameVersion", "")
        queue_id = info.get("queueId", 0)
        participants = info.get("participants", [])
        if not self._participantes_completos(match_id, participants):
            return False

        match_insert_query = """
            INSERT OR IGNORE INTO matches (match_id, game_creation, game_duration, game_version, queue_id, collected_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP);
        """

        participant_insert_query = """
            INSERT OR IGNORE INTO match_participants (
                match_id, puuid, participant_id, team_id, win,
                champion_id, champion_name, individual_position,
                kills, deaths, assists, kda,
                total_damage_dealt_to_champions, total_damage_taken,
                gold_earned, total_minions_killed, neutral_minions_killed,
                vision_score, champ_level,
                item0, item1, item2, item3, item4, item5, item6,
                first_blood_kill, first_tower_kill
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """

        part_rows = []
        riot_id_rows = []
        project_rows = []
        for p in participants:
            puuid, riot_row, project_row = self._identity_rows(p, game_creation, served_project)
            if not puuid:
                continue
            if riot_row:
                riot_id_rows.append(riot_row)
            if project_row:
                project_rows.append(project_row)

            kills = int(p.get("kills", 0))
            deaths = int(p.get("deaths", 0))
            assists = int(p.get("assists", 0))
            kda = round((kills + assists) / max(1, deaths), 2)

            part_rows.append((
                match_id,
                puuid,
                int(p.get("participantId", 0)),
                int(p.get("teamId", 0)),
                1 if p.get("win") else 0,
                int(p.get("championId", 0)),
                p.get("championName", "Unknown"),
                p.get("individualPosition", "UNKNOWN"),
                kills,
                deaths,
                assists,
                kda,
                int(p.get("totalDamageDealtToChampions", 0)),
                int(p.get("totalDamageTaken", 0)),
                int(p.get("goldEarned", 0)),
                int(p.get("totalMinionsKilled", 0)),
                int(p.get("neutralMinionsKilled", 0)),
                int(p.get("visionScore", 0)),
                int(p.get("champLevel", 1)),
                int(p.get("item0", 0)),
                int(p.get("item1", 0)),
                int(p.get("item2", 0)),
                int(p.get("item3", 0)),
                int(p.get("item4", 0)),
                int(p.get("item5", 0)),
                int(p.get("item6", 0)),
                1 if p.get("firstBloodKill") else 0,
                1 if p.get("firstTowerKill") else 0
            ))

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(match_insert_query, (match_id, game_creation, game_duration, game_version, queue_id))
            cursor.executemany(participant_insert_query, part_rows)
            if riot_id_rows:
                cursor.executemany(self._RIOT_ID_UPSERT_SQL, riot_id_rows)
            if project_rows:
                cursor.executemany(self._PUUID_PROJECT_UPSERT_SQL, project_rows)
            conn.commit()

        return True

    def save_match_timeline(self, match_id: str, timeline_data: Dict[str, Any], participant_puuids: Optional[Dict[int, str]] = None) -> bool:
        """
        Extrae y almacena las estadísticas del timeline a los minutos 10, 15 y 20 para los 10 participantes.
        """
        if not match_id or not timeline_data:
            return False

        info = timeline_data.get("info", {})
        frames = info.get("frames", [])
        if not frames:
            return False

        # Conteo acumulado de Kills, Deaths y Assists hasta min 10, 15 y 20
        kda_events = {
            10: {pid: {"k": 0, "d": 0, "a": 0} for pid in range(1, 11)},
            15: {pid: {"k": 0, "d": 0, "a": 0} for pid in range(1, 11)},
            20: {pid: {"k": 0, "d": 0, "a": 0} for pid in range(1, 11)}
        }

        for f_idx, frame in enumerate(frames):
            for event in frame.get("events", []):
                if event.get("type") == "CHAMPION_KILL":
                    killer = event.get("killerId", 0)
                    victim = event.get("victimId", 0)
                    assisters = event.get("assistingParticipantIds", []) or []

                    for target_min in (10, 15, 20):
                        if f_idx <= target_min:
                            if 1 <= killer <= 10:
                                kda_events[target_min][killer]["k"] += 1
                            if 1 <= victim <= 10:
                                kda_events[target_min][victim]["d"] += 1
                            for aid in assisters:
                                if 1 <= aid <= 10:
                                    kda_events[target_min][aid]["a"] += 1

        frame_10 = frames[10].get("participantFrames") if len(frames) > 10 else None
        frame_15 = frames[15].get("participantFrames") if len(frames) > 15 else None
        frame_20 = frames[20].get("participantFrames") if len(frames) > 20 else None

        puuid_map = participant_puuids or {}
        rows = []
        for pid in range(1, 11):
            pid_str = str(pid)
            # Mismo PUUID que su fila de `match_participants` (sin traducir).
            puuid = puuid_map.get(pid, "")

            # Snapshot Minuto 10
            if frame_10 and pid_str in frame_10:
                p10 = frame_10[pid_str]
                dmg10 = p10.get("damageStats", {}) or {}
                g10, xp10, lvl10 = int(p10.get("totalGold", 0)), int(p10.get("xp", 0)), int(p10.get("level", 1))
                cs10 = int(p10.get("minionsKilled", 0)) + int(p10.get("jungleMinionsKilled", 0))
                d10 = int(dmg10.get("totalDamageDoneToChampions", 0))
                k10, de10, a10 = kda_events[10][pid]["k"], kda_events[10][pid]["d"], kda_events[10][pid]["a"]
            else:
                g10 = xp10 = lvl10 = cs10 = d10 = k10 = de10 = a10 = None

            # Snapshot Minuto 15
            if frame_15 and pid_str in frame_15:
                p15 = frame_15[pid_str]
                dmg15 = p15.get("damageStats", {}) or {}
                g15, xp15, lvl15 = int(p15.get("totalGold", 0)), int(p15.get("xp", 0)), int(p15.get("level", 1))
                cs15 = int(p15.get("minionsKilled", 0)) + int(p15.get("jungleMinionsKilled", 0))
                d15 = int(dmg15.get("totalDamageDoneToChampions", 0))
                k15, de15, a15 = kda_events[15][pid]["k"], kda_events[15][pid]["d"], kda_events[15][pid]["a"]
            else:
                g15 = xp15 = lvl15 = cs15 = d15 = k15 = de15 = a15 = None

            # Snapshot Minuto 20
            if frame_20 and pid_str in frame_20:
                p20 = frame_20[pid_str]
                dmg20 = p20.get("damageStats", {}) or {}
                g20, xp20, lvl20 = int(p20.get("totalGold", 0)), int(p20.get("xp", 0)), int(p20.get("level", 1))
                cs20 = int(p20.get("minionsKilled", 0)) + int(p20.get("jungleMinionsKilled", 0))
                d20 = int(dmg20.get("totalDamageDoneToChampions", 0))
                k20, de20, a20 = kda_events[20][pid]["k"], kda_events[20][pid]["d"], kda_events[20][pid]["a"]
            else:
                g20 = xp20 = lvl20 = cs20 = d20 = k20 = de20 = a20 = None

            rows.append((
                match_id, pid, puuid,
                g10, xp10, lvl10, cs10, d10, k10, de10, a10,
                g15, xp15, lvl15, cs15, d15, k15, de15, a15,
                g20, xp20, lvl20, cs20, d20, k20, de20, a20
            ))

        insert_query = """
            INSERT OR REPLACE INTO match_timeline (
                match_id, participant_id, puuid,
                gold_10, xp_10, level_10, cs_10, damage_10, kills_10, deaths_10, assists_10,
                gold_15, xp_15, level_15, cs_15, damage_15, kills_15, deaths_15, assists_15,
                gold_20, xp_20, level_20, cs_20, damage_20, kills_20, deaths_20, assists_20
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(insert_query, rows)
            conn.commit()

        return True

    def save_matches_and_timelines_batch(self, batch_items: List[tuple]) -> int:
        return self.save_match_batch_with_counts(batch_items)[0]

    def save_match_batch_with_counts(self, batch_items: List[tuple]) -> Tuple[int, int]:
        """
        Guarda un lote de partidas completas (Match DTO + Timeline DTO) en una sola transacción atómica.
        Reduce drásticamente las operaciones de I/O en disco (fsync) en Windows.

        Cada ítem es `(match_id, match_detail, timeline_detail)` o, desde el
        crawler, `(match_id, match_detail, timeline_detail, served_project)`, con el
        proyecto de la key que descargó la partida (ver `_identity_rows`).
        Retorna (partidas nuevas, partidas nuevas con timeline guardado).
        """
        if not batch_items:
            return 0, 0
        with self._get_connection() as conn:
            # La comprobación de IDs y los INSERT pertenecen a la misma transacción.
            conn.execute("BEGIN IMMEDIATE;")
            return self._write_match_batch(conn, batch_items)

    def _write_match_batch(self, conn: sqlite3.Connection, batch_items: List[tuple]) -> Tuple[int, int]:
        # Filtrar de antemano cualquier match_id que ya exista en la base de datos
        all_ids = [item[0] for item in batch_items if item and item[0]]
        cursor = conn.cursor()
        already_saved = set()
        if all_ids:
            placeholders = ",".join("?" for _ in all_ids)
            cursor.execute(f"SELECT match_id FROM matches WHERE match_id IN ({placeholders});", all_ids)
            already_saved = {r[0] for r in cursor.fetchall()}

        match_rows = []
        participant_rows = []
        timeline_rows = []
        timelines_saved = 0
        discarded_rows = []
        riot_id_rows = []
        project_rows = []

        for item in batch_items:
            match_id, match_detail, timeline_detail = item[0], item[1], item[2]
            served_project = item[3] if len(item) > 3 else None
            if not match_id or not match_detail or match_id in already_saved:
                continue
            already_saved.add(match_id)

            metadata = match_detail.get("metadata", {})
            info = match_detail.get("info", {})
            game_creation = info.get("gameCreation", 0)
            game_duration = info.get("gameDuration", 0)
            # Ignorar remakes (partidas <= 300s / 5 minutos) y registrarlas para no volver a consultarlas
            if game_duration <= 300:
                discarded_rows.append((match_id, "remake_duration_le_300"))
                continue
            game_version = info.get("gameVersion", "")
            queue_id = info.get("queueId", 0)
            participants = info.get("participants", [])
            if not self._participantes_completos(match_id, participants):
                continue

            match_rows.append((match_id, game_creation, game_duration, game_version, queue_id))

            puuid_map = {}
            for p in participants:
                puuid, riot_row, project_row = self._identity_rows(p, game_creation, served_project)
                if not puuid:
                    continue
                pid = int(p.get("participantId", 0))
                puuid_map[pid] = puuid
                if riot_row:
                    riot_id_rows.append(riot_row)
                if project_row:
                    project_rows.append(project_row)

                kills = int(p.get("kills", 0))
                deaths = int(p.get("deaths", 0))
                assists = int(p.get("assists", 0))
                kda = round((kills + assists) / max(1, deaths), 2)

                participant_rows.append((
                    match_id,
                    puuid,
                    pid,
                    int(p.get("teamId", 0)),
                    1 if p.get("win") else 0,
                    int(p.get("championId", 0)),
                    p.get("championName", "Unknown"),
                    p.get("individualPosition", "UNKNOWN"),
                    kills,
                    deaths,
                    assists,
                    kda,
                    int(p.get("totalDamageDealtToChampions", 0)),
                    int(p.get("totalDamageTaken", 0)),
                    int(p.get("goldEarned", 0)),
                    int(p.get("totalMinionsKilled", 0)),
                    int(p.get("neutralMinionsKilled", 0)),
                    int(p.get("visionScore", 0)),
                    int(p.get("champLevel", 1)),
                    int(p.get("item0", 0)),
                    int(p.get("item1", 0)),
                    int(p.get("item2", 0)),
                    int(p.get("item3", 0)),
                    int(p.get("item4", 0)),
                    int(p.get("item5", 0)),
                    int(p.get("item6", 0)),
                    1 if p.get("firstBloodKill") else 0,
                    1 if p.get("firstTowerKill") else 0
                ))

            if timeline_detail:
                t_info = timeline_detail.get("info", {})
                frames = t_info.get("frames", [])
                if frames:
                    timelines_saved += 1
                    kda_events = {
                        10: {pid: {"k": 0, "d": 0, "a": 0} for pid in range(1, 11)},
                        15: {pid: {"k": 0, "d": 0, "a": 0} for pid in range(1, 11)},
                        20: {pid: {"k": 0, "d": 0, "a": 0} for pid in range(1, 11)}
                    }

                    for f_idx, frame in enumerate(frames):
                        for event in frame.get("events", []):
                            if event.get("type") == "CHAMPION_KILL":
                                killer = event.get("killerId", 0)
                                victim = event.get("victimId", 0)
                                assisters = event.get("assistingParticipantIds", []) or []

                                for target_min in (10, 15, 20):
                                    if f_idx <= target_min:
                                        if 1 <= killer <= 10:
                                            kda_events[target_min][killer]["k"] += 1
                                        if 1 <= victim <= 10:
                                            kda_events[target_min][victim]["d"] += 1
                                        for aid in assisters:
                                            if 1 <= aid <= 10:
                                                kda_events[target_min][aid]["a"] += 1

                    frame_10 = frames[10].get("participantFrames") if len(frames) > 10 else None
                    frame_15 = frames[15].get("participantFrames") if len(frames) > 15 else None
                    frame_20 = frames[20].get("participantFrames") if len(frames) > 20 else None

                    for pid in range(1, 11):
                        pid_str = str(pid)
                        puuid = puuid_map.get(pid, "")

                        if frame_10 and pid_str in frame_10:
                            p10 = frame_10[pid_str]
                            dmg10 = p10.get("damageStats", {}) or {}
                            g10, xp10, lvl10 = int(p10.get("totalGold", 0)), int(p10.get("xp", 0)), int(p10.get("level", 1))
                            cs10 = int(p10.get("minionsKilled", 0)) + int(p10.get("jungleMinionsKilled", 0))
                            d10 = int(dmg10.get("totalDamageDoneToChampions", 0))
                            k10, de10, a10 = kda_events[10][pid]["k"], kda_events[10][pid]["d"], kda_events[10][pid]["a"]
                        else:
                            g10 = xp10 = lvl10 = cs10 = d10 = k10 = de10 = a10 = None

                        if frame_15 and pid_str in frame_15:
                            p15 = frame_15[pid_str]
                            dmg15 = p15.get("damageStats", {}) or {}
                            g15, xp15, lvl15 = int(p15.get("totalGold", 0)), int(p15.get("xp", 0)), int(p15.get("level", 1))
                            cs15 = int(p15.get("minionsKilled", 0)) + int(p15.get("jungleMinionsKilled", 0))
                            d15 = int(dmg15.get("totalDamageDoneToChampions", 0))
                            k15, de15, a15 = kda_events[15][pid]["k"], kda_events[15][pid]["d"], kda_events[15][pid]["a"]
                        else:
                            g15 = xp15 = lvl15 = cs15 = d15 = k15 = de15 = a15 = None

                        if frame_20 and pid_str in frame_20:
                            p20 = frame_20[pid_str]
                            dmg20 = p20.get("damageStats", {}) or {}
                            g20, xp20, lvl20 = int(p20.get("totalGold", 0)), int(p20.get("xp", 0)), int(p20.get("level", 1))
                            cs20 = int(p20.get("minionsKilled", 0)) + int(p20.get("jungleMinionsKilled", 0))
                            d20 = int(dmg20.get("totalDamageDoneToChampions", 0))
                            k20, de20, a20 = kda_events[20][pid]["k"], kda_events[20][pid]["d"], kda_events[20][pid]["a"]
                        else:
                            g20 = xp20 = lvl20 = cs20 = d20 = k20 = de20 = a20 = None

                        timeline_rows.append((
                            match_id, pid, puuid,
                            g10, xp10, lvl10, cs10, d10, k10, de10, a10,
                            g15, xp15, lvl15, cs15, d15, k15, de15, a15,
                            g20, xp20, lvl20, cs20, d20, k20, de20, a20
                        ))

        match_insert_query = """
            INSERT OR IGNORE INTO matches (match_id, game_creation, game_duration, game_version, queue_id, collected_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP);
        """
        participant_insert_query = """
            INSERT OR IGNORE INTO match_participants (
                match_id, puuid, participant_id, team_id, win,
                champion_id, champion_name, individual_position,
                kills, deaths, assists, kda,
                total_damage_dealt_to_champions, total_damage_taken,
                gold_earned, total_minions_killed, neutral_minions_killed,
                vision_score, champ_level,
                item0, item1, item2, item3, item4, item5, item6,
                first_blood_kill, first_tower_kill
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        timeline_insert_query = """
            INSERT OR REPLACE INTO match_timeline (
                match_id, participant_id, puuid,
                gold_10, xp_10, level_10, cs_10, damage_10, kills_10, deaths_10, assists_10,
                gold_15, xp_15, level_15, cs_15, damage_15, kills_15, deaths_15, assists_15,
                gold_20, xp_20, level_20, cs_20, damage_20, kills_20, deaths_20, assists_20
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """

        if match_rows:
            cursor.executemany(match_insert_query, match_rows)
        if participant_rows:
            cursor.executemany(participant_insert_query, participant_rows)
        if timeline_rows:
            cursor.executemany(timeline_insert_query, timeline_rows)
        if riot_id_rows:
            cursor.executemany(self._RIOT_ID_UPSERT_SQL, riot_id_rows)
        if project_rows:
            cursor.executemany(self._PUUID_PROJECT_UPSERT_SQL, project_rows)
        if discarded_rows:
            cursor.executemany("""
                INSERT OR IGNORE INTO discarded_matches (match_id, reason, discarded_at)
                VALUES (?, ?, CURRENT_TIMESTAMP);
            """, discarded_rows)
        conn.commit()
        return len(match_rows), timelines_saved

    def is_timeline_saved(self, match_id: str) -> bool:
        """Verifica si ya tenemos guardado el timeline para un match_id."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM match_timeline WHERE match_id = ? LIMIT 1;", (match_id,))
            return cursor.fetchone() is not None

    def get_matches_without_timeline(self, limit: Optional[int] = None) -> List[str]:
        """Obtiene IDs de partidas que aún no tienen timeline registrado."""
        query = """
            SELECT m.match_id FROM matches m
            LEFT JOIN match_timeline t ON m.match_id = t.match_id
            WHERE t.match_id IS NULL
        """
        if limit:
            query += f" LIMIT {int(limit)}"
        query += ";"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            return [row[0] for row in cursor.fetchall()]

    def get_total_matches(self) -> int:
        with self._get_connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM matches;").fetchone()[0]

    def get_database_summary(self) -> Dict[str, Any]:
        """Obtiene métricas resumidas del estado de la base de datos y tamaño en disco."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM players;")
            total_players = cursor.fetchone()[0]

            cursor.execute("SELECT tier, COUNT(*) FROM players GROUP BY tier;")
            players_by_tier = dict(cursor.fetchall())

            cursor.execute("SELECT COUNT(*) FROM matches;")
            total_matches = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM match_participants;")
            total_participant_records = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT match_id) FROM match_timeline;")
            total_matches_with_timeline = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM discarded_matches;")
            total_discarded_matches = cursor.fetchone()[0]

        file_size_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0
        file_size_mb = round(file_size_bytes / (1024 * 1024), 2)

        return {
            "db_path": str(self.db_path),
            "file_size_mb": file_size_mb,
            "total_players": total_players,
            "players_by_tier": players_by_tier,
            "total_matches": total_matches,
            "total_participant_records": total_participant_records,
            "total_matches_with_timeline": total_matches_with_timeline,
            "total_discarded_matches": total_discarded_matches,
            "coverage": self.get_coverage_stats()
        }

    def get_coverage_stats(self) -> Dict[str, Any]:
        with self._get_connection() as conn:
            return self.coverage_stats_from_connection(conn)

    @staticmethod
    def coverage_stats_from_connection(conn: sqlite3.Connection) -> Dict[str, Any]:
        """
        Calcula las métricas de cobertura (coverage50, coverage100, coverage200, coverage300)
        indicando la cantidad de jugadores que tienen >= 50, 100, 200 y 300 partidas guardadas en la BD.
        Si existe mapa multi-proyecto, agrupa por jugador verdadero (identidad canónica)
        sumando las partidas de todos sus PUUIDs; si no, usa el conteo legacy por PUUID.
        """
        tiers = ["CHALLENGER", "GRANDMASTER", "MASTER", "DIAMOND", "TOTAL"]
        stats = {t: {"total": 0, "c50": 0, "c100": 0, "c200": 0, "c300": 0} for t in tiers}

        cursor = conn.cursor()
        has_pid_map = cursor.execute("SELECT EXISTS(SELECT 1 FROM player_id_map);").fetchone()[0]
        if has_pid_map:
            # Por persona (`player_id`): suma las partidas de todos sus PUUIDs,
            # de cualquier proyecto, tal como se guardan desde la ingesta.
            # El tier de la persona es el más alto en orden de liga, no en orden
            # alfabético (`sql_tier_maximo`, §9.S.76).
            cursor.execute(f"""
                WITH games_per_person AS (
                    SELECT m.player_id, COUNT(DISTINCT mp.match_id) AS games_count
                    FROM match_participants mp
                    INNER JOIN player_id_map m ON m.puuid = mp.puuid
                    GROUP BY m.player_id
                )
                SELECT
                    COALESCE(NULLIF({sql_tier_maximo('id.tier')}, 'UNKNOWN'),
                             {sql_tier_maximo('p.tier')}) AS tier,
                    COALESCE(MAX(g.games_count), 0) AS games_count
                FROM players p
                LEFT JOIN player_id_map m ON m.puuid = p.puuid
                LEFT JOIN player_identities id ON id.canonical_id = m.player_id
                LEFT JOIN games_per_person g ON g.player_id = m.player_id
                GROUP BY COALESCE(m.player_id, p.puuid);
            """)
        elif cursor.execute("SELECT EXISTS(SELECT 1 FROM v_puuid_canonical);").fetchone()[0]:
            cursor.execute("""
                WITH games_per_canonical AS (
                    SELECT
                        COALESCE(v.canonical_id, mp.puuid) AS canonical_id,
                        COUNT(DISTINCT mp.match_id) AS games_count
                    FROM match_participants mp
                    LEFT JOIN v_puuid_canonical v ON mp.puuid = v.alias_puuid
                    GROUP BY COALESCE(v.canonical_id, mp.puuid)
                )
                SELECT
                    COALESCE(id.tier, p.tier) AS tier,
                    COALESCE(g.games_count, 0) AS games_count
                FROM players p
                LEFT JOIN v_puuid_canonical v ON v.alias_puuid = p.puuid
                LEFT JOIN player_identities id ON id.canonical_id = v.canonical_id
                LEFT JOIN games_per_canonical g ON g.canonical_id = COALESCE(v.canonical_id, p.puuid)
                GROUP BY COALESCE(v.canonical_id, p.puuid);
            """)
        else:
            cursor.execute("""
                SELECT
                    p.tier,
                    COUNT(mp.match_id) as games_count
                FROM players p
                LEFT JOIN match_participants mp ON p.puuid = mp.puuid
                GROUP BY p.puuid;
            """)
        rows = cursor.fetchall()

        for tier, count in rows:
            t = str(tier).upper() if tier else "UNKNOWN"
            if t not in stats:
                stats[t] = {"total": 0, "c50": 0, "c100": 0, "c200": 0, "c300": 0}

            stats[t]["total"] += 1
            stats["TOTAL"]["total"] += 1
            if count >= 50:
                stats[t]["c50"] += 1
                stats["TOTAL"]["c50"] += 1
            if count >= 100:
                stats[t]["c100"] += 1
                stats["TOTAL"]["c100"] += 1
            if count >= 200:
                stats[t]["c200"] += 1
                stats["TOTAL"]["c200"] += 1
            if count >= 300:
                stats[t]["c300"] += 1
                stats["TOTAL"]["c300"] += 1

        return stats

    def log_crawler_metric(self, metric_dict: Dict[str, Any]) -> None:
        """Registra un punto de telemetría en el timeline histórico de la sesión de crawling."""
        query = """
            INSERT INTO crawler_metrics_timeline (
                session_id, elapsed_minutes, active_keys, workers,
                players_crawled, total_matches_in_db, new_matches_saved,
                duplicate_matches_skipped, discarded_matches,
                net_matches_per_min, requests_per_min,
                c50_challenger, c50_grandmaster, c50_master, c50_total,
                db_size_mb, write_errors, rejected_participants
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        params = (
            metric_dict.get("session_id", "default"),
            metric_dict.get("elapsed_minutes", 0.0),
            metric_dict.get("active_keys", 1),
            metric_dict.get("workers", 4),
            metric_dict.get("players_crawled", 0),
            metric_dict.get("total_matches_in_db", 0),
            metric_dict.get("new_matches_saved", 0),
            metric_dict.get("duplicate_matches_skipped", 0),
            metric_dict.get("discarded_matches", 0),
            metric_dict.get("net_matches_per_min", 0.0),
            metric_dict.get("requests_per_min", 0.0),
            metric_dict.get("c50_challenger", 0),
            metric_dict.get("c50_grandmaster", 0),
            metric_dict.get("c50_master", 0),
            metric_dict.get("c50_total", 0),
            metric_dict.get("db_size_mb", 0.0),
            metric_dict.get("write_errors", 0),
            metric_dict.get("rejected_participants", 0),
        )
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()

    def get_crawler_metrics_timeline(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Devuelve los últimos puntos del timeline de métricas históricas."""
        query = "SELECT * FROM crawler_metrics_timeline ORDER BY id DESC LIMIT ?;"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def resolve_and_merge_player_alias(self, primary_puuid: str, alias_puuid: str, game_name: Optional[str] = None, tag_line: Optional[str] = None) -> int:
        """
        Unifica un PUUID alias (secundario/regional) con el PUUID principal de la cuenta:
        1. Registra el mapeo en player_puuid_aliases para auditoría estricta.
        2. Actualiza match_participants y match_timeline para migrar las partidas hacia primary_puuid.
        Retorna la cantidad de registros de participantes migrados.
        """
        if not primary_puuid or not alias_puuid or primary_puuid == alias_puuid:
            return 0

        with self._get_connection() as conn:
            cursor = conn.cursor()
            # 1. Contar cuántas partidas tiene el alias en match_participants
            cursor.execute("SELECT COUNT(*) FROM match_participants WHERE puuid = ?", (alias_puuid,))
            migrated_count = cursor.fetchone()[0]

            # 2. Registrar en player_puuid_aliases
            cursor.execute("""
                INSERT OR REPLACE INTO player_puuid_aliases (primary_puuid, alias_puuid, game_name, tag_line, matches_migrated)
                VALUES (?, ?, ?, ?, ?);
            """, (primary_puuid, alias_puuid, game_name, tag_line, migrated_count))

            # 3. Migrar match_participants hacia primary_puuid
            cursor.execute("""
                UPDATE match_participants 
                SET puuid = ? 
                WHERE puuid = ? 
                  AND match_id NOT IN (SELECT match_id FROM match_participants WHERE puuid = ?);
            """, (primary_puuid, alias_puuid, primary_puuid))

            # 4. Migrar match_timeline hacia primary_puuid
            cursor.execute("""
                UPDATE match_timeline 
                SET puuid = ? 
                WHERE puuid = ?;
            """, (primary_puuid, alias_puuid))

            conn.commit()
            return migrated_count

    def get_player_aliases(self, primary_puuid: str) -> List[Dict[str, Any]]:
        """Devuelve los alias conocidos registrados para un PUUID primario."""
        query = "SELECT * FROM player_puuid_aliases WHERE primary_puuid = ?;"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (primary_puuid,))
            return [dict(row) for row in cursor.fetchall()]

    def get_participant_puuid_map(self, match_id: str) -> Dict[int, str]:
        """Devuelve el mapa participant_id -> puuid de una partida (O(1), SQLite local)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT participant_id, puuid FROM match_participants WHERE match_id = ?;",
                (match_id,)
            )
            return {int(r[0]): r[1] for r in cursor.fetchall()}

    # ------------------------------------------------------------------
    # Sistema de identidades multi-proyecto (PUUIDs por API key)
    # ------------------------------------------------------------------

    @staticmethod
    def canonical_id_for(game_name: str, tag_line: str) -> str:
        """Identidad verdadera estable entre proyectos: lower(gameName#tagLine)."""
        return f"{(game_name or '').strip().lower()}#{(tag_line or '').strip().lower()}"

    @staticmethod
    def mask_key_hint(api_key: str) -> str:
        """Hint enmascarado para registro (nunca persistir la clave completa)."""
        k = (api_key or "").strip()
        return f"{k[:9]}...{k[-4:]}" if len(k) > 13 else "******"

    def register_api_projects(self, api_keys: List[str],
                              project_indices: Optional[List[int]] = None) -> List[int]:
        """Registra el pool de proyectos.

        Por defecto el `project_index` es la posición en la lista, que es estable
        mientras el pool sea el mismo y la key solo rote dentro de su proyecto.

        `project_indices` permite fijar la ranura explícitamente, y es obligatorio
        cuando el pool NO contiene todos los proyectos históricos: con una sola key
        de un proyecto nuevo, el default la registraría en la ranura 0 y
        **sobrescribiría el `key_hint` del proyecto 0 ajeno**, borrando la única
        pista de a quién pertenecía esa ranura. Ver `RIOT_API_PROJECT_INDICES`
        en `.env`.

        Retorna los índices registrados. No elimina proyectos históricos.
        """
        if project_indices is not None and len(project_indices) != len(api_keys):
            raise ValueError("project_indices debe tener el mismo largo que api_keys.")
        indices = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for pos, key in enumerate(api_keys):
                if not (key or "").strip():
                    continue
                idx = project_indices[pos] if project_indices is not None else pos
                cursor.execute("""
                    INSERT INTO api_projects (project_index, key_hint, first_seen, last_seen)
                    VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    ON CONFLICT(project_index) DO UPDATE SET
                        key_hint=excluded.key_hint,
                        last_seen=CURRENT_TIMESTAMP;
                """, (idx, self.mask_key_hint(key)))
                indices.append(idx)
            conn.commit()
        # Refrescar caché de traducción por si el mapa ya tenía filas de estos proyectos
        self.reload_translation_cache()
        return indices

    def get_num_projects(self) -> int:
        """Cantidad de proyectos de API key registrados (denominador del invariante)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM api_projects;")
            row = cursor.fetchone()
            return int(row[0]) if row else 0

    def upsert_player_identity(self, game_name: str, tag_line: str, puuid: str,
                               project_index: Optional[int] = 0, tier: str = "UNKNOWN",
                               rank: str = "I", league_points: int = 0,
                               wins: int = 0, losses: int = 0,
                               summoner_id: str = "") -> str:
        """Crea o actualiza la identidad canónica de un jugador y su PUUID en un proyecto.

        Retorna el canonical_id. Idempotente: re-ejecutar con los mismos datos no duplica.
        Si project_index es None, solo asegura la identidad SIN fila de mapa
        (pase inverso: la atribución por proyecto la completa register_key_project).
        """
        canonical_id = self.canonical_id_for(game_name, tag_line)
        if canonical_id == "#":
            raise ValueError("game_name y tag_line no pueden estar ambos vacíos")
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Asegurar la fila del proyecto (FK): el migrate puede correr antes de
            # registrar hints, y los tests usan identidades sin registro previo.
            if project_index is not None:
                cursor.execute("""
                    INSERT OR IGNORE INTO api_projects (project_index, key_hint)
                    VALUES (?, '');
                """, (int(project_index),))
            cursor.execute("""
                INSERT INTO player_identities
                    (canonical_id, game_name, tag_line, tier, rank, league_points, wins, losses, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(canonical_id) DO UPDATE SET
                    game_name=excluded.game_name,
                    tag_line=excluded.tag_line,
                    tier=CASE WHEN excluded.tier = 'UNKNOWN' THEN player_identities.tier ELSE excluded.tier END,
                    rank=CASE WHEN excluded.tier = 'UNKNOWN' THEN player_identities.rank ELSE excluded.rank END,
                    league_points=CASE WHEN excluded.tier = 'UNKNOWN' THEN player_identities.league_points ELSE excluded.league_points END,
                    wins=CASE WHEN excluded.tier = 'UNKNOWN' THEN player_identities.wins ELSE excluded.wins END,
                    losses=CASE WHEN excluded.tier = 'UNKNOWN' THEN player_identities.losses ELSE excluded.losses END,
                    updated_at=CURRENT_TIMESTAMP;
            """, (canonical_id, game_name.strip(), tag_line.strip(), tier, rank,
                  int(league_points), int(wins), int(losses)))
            if project_index is not None:
                cursor.execute("""
                    INSERT INTO player_puuid_map
                        (canonical_id, project_index, puuid, summoner_id, last_verified)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(canonical_id, project_index) DO UPDATE SET
                        puuid=excluded.puuid,
                        summoner_id=excluded.summoner_id,
                        last_verified=CURRENT_TIMESTAMP;
                """, (canonical_id, int(project_index), puuid, summoner_id or ""))
            conn.commit()
        return canonical_id

    def has_live_puuid(self, puuid: str) -> bool:
        """True si el PUUID tiene fila en el mapa live (distinto de solo-alias-legacy)."""
        if not puuid:
            return False
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM player_puuid_map WHERE puuid = ?;", (puuid,))
            return cursor.fetchone() is not None

    def resolve_canonical_id(self, puuid: str) -> Optional[str]:
        """Devuelve el canonical_id de un PUUID (O(1) local, sin requests).

        Busca primero en el mapa live y luego en los alias legacy.
        """
        if not puuid:
            return None
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT c.canonical_id FROM player_puuid_map m
                INNER JOIN player_identities c ON c.canonical_id = m.canonical_id
                WHERE m.puuid = ?;
            """, (puuid,))
            row = cursor.fetchone()
            if row:
                return row[0]
            cursor.execute("SELECT primary_puuid FROM player_puuid_aliases WHERE alias_puuid = ?;",
                           (puuid,))
            row = cursor.fetchone()
            if not row:
                return None
            primary = row[0]
            # Encadenar legacy -> live para devolver siempre el canonical_id uniforme.
            cursor.execute("SELECT canonical_id FROM player_puuid_map WHERE puuid = ?;",
                           (primary,))
            row2 = cursor.fetchone()
            return row2[0] if row2 else primary

    def get_puuids_for_canonical(self, canonical_id: str) -> List[Dict[str, Any]]:
        """Todas las filas (una por proyecto) conocidas de un jugador verdadero."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM player_puuid_map WHERE canonical_id = ? ORDER BY project_index;",
                (canonical_id,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_identity_coverage(self) -> Dict[str, Any]:
        """Verifica el invariante: cada identidad debe tener 1 PUUID por proyecto registrado.

        Retorna {num_projects, total_identities, complete, incomplete, incomplete_detail}.
        """
        num_projects = self.get_num_projects()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT canonical_id, COUNT(*) AS n_puuids
                FROM player_puuid_map
                GROUP BY canonical_id;
            """)
            rows = cursor.fetchall()
        complete = sum(1 for r in rows if int(r[1]) == num_projects)
        incomplete = [ {"canonical_id": r[0], "n_puuids": int(r[1])} for r in rows if int(r[1]) != num_projects ]
        return {
            "num_projects": num_projects,
            "total_identities": len(rows),
            "complete": complete,
            "incomplete": len(incomplete),
            "incomplete_detail": incomplete[:50],
        }

    def get_canonicals_for_deep_crawl(self, tier: Optional[str] = None,
                                      min_matches_target: int = 50,
                                      only_incomplete: bool = False) -> List[Dict[str, Any]]:
        """Variante canónica del deep crawl: agrupa partidas por jugador verdadero.

        Suma las participaciones de TODOS sus PUUIDs (todos los proyectos) bajo un solo
        conteo, con el PUUID del proyecto 0 como representante para consultar a Riot.
        Si aún no hay mapa registrado, delega al método legacy por PUUID.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM v_puuid_canonical;")
            has_canonical = (cursor.fetchone()[0] or 0) > 0
            cursor.execute("SELECT COUNT(*) FROM player_id_map;")
            has_pid_map = (cursor.fetchone()[0] or 0) > 0
            if not has_canonical and not has_pid_map:
                pass
            else:
                if has_pid_map:
                    query = """
                        WITH games_per_canonical AS (
                            SELECT
                                COALESCE(m.player_id, mp.puuid) AS canonical_id,
                                COUNT(DISTINCT mp.match_id) AS existing_recorded_games
                            FROM match_participants mp
                            LEFT JOIN player_id_map m ON mp.puuid = m.puuid
                            GROUP BY COALESCE(m.player_id, mp.puuid)
                        )
                        SELECT
                            COALESCE(m.player_id, v.canonical_id, p.puuid) AS canonical_id,
                            MIN(p.puuid) AS players_puuid,
                            COALESCE(NULLIF(id.tier, 'UNKNOWN'), NULLIF(p.tier, 'UNKNOWN'), 'DIAMOND') AS tier,
                            COALESCE(NULLIF(id.rank, ''), NULLIF(p.rank, ''), 'I') AS rank,
                            p.league_points AS league_points,
                            p.last_crawled_at AS last_crawled_at,
                            COALESCE(g.existing_recorded_games, 0) AS existing_recorded_games
                        FROM players p
                        LEFT JOIN player_id_map m ON m.puuid = p.puuid
                        LEFT JOIN v_puuid_canonical v ON v.alias_puuid = p.puuid
                        LEFT JOIN player_identities id ON id.canonical_id = COALESCE(m.player_id, v.canonical_id)
                        LEFT JOIN games_per_canonical g ON g.canonical_id = COALESCE(m.player_id, v.canonical_id, p.puuid)
                    """
                    group_expr = "COALESCE(m.player_id, v.canonical_id, p.puuid)"
                else:
                    query = """
                        WITH games_per_canonical AS (
                            SELECT 
                                COALESCE(v.canonical_id, mp.puuid) AS canonical_id,
                                COUNT(DISTINCT mp.match_id) AS existing_recorded_games
                            FROM match_participants mp
                            LEFT JOIN v_puuid_canonical v ON mp.puuid = v.alias_puuid
                            GROUP BY COALESCE(v.canonical_id, mp.puuid)
                        )
                        SELECT
                            COALESCE(v.canonical_id, p.puuid) AS canonical_id,
                            MIN(p.puuid) AS players_puuid,
                            COALESCE(NULLIF(id.tier, 'UNKNOWN'), NULLIF(p.tier, 'UNKNOWN'), 'DIAMOND') AS tier,
                            COALESCE(NULLIF(id.rank, ''), NULLIF(p.rank, ''), 'I') AS rank,
                            p.league_points AS league_points,
                            p.last_crawled_at AS last_crawled_at,
                            COALESCE(g.existing_recorded_games, 0) AS existing_recorded_games
                        FROM players p
                        LEFT JOIN v_puuid_canonical v ON v.alias_puuid = p.puuid
                        LEFT JOIN player_identities id ON id.canonical_id = v.canonical_id
                        LEFT JOIN games_per_canonical g ON g.canonical_id = COALESCE(v.canonical_id, p.puuid)
                    """
                    group_expr = "COALESCE(v.canonical_id, p.puuid)"
                params: List[Any] = []
                if tier:
                    tier_upper = tier.upper()
                    if tier_upper == "APEX":
                        target_tiers = ["CHALLENGER", "GRANDMASTER"]
                    elif "," in tier_upper:
                        target_tiers = [t.strip() for t in tier_upper.split(",") if t.strip()]
                    else:
                        target_tiers = [tier_upper]
                    placeholders = ",".join(["?"] * len(target_tiers))
                    query += f" WHERE COALESCE(NULLIF(id.tier, 'UNKNOWN'), NULLIF(p.tier, 'UNKNOWN'), 'DIAMOND') IN ({placeholders})"
                    params.extend(target_tiers)
                query += f" GROUP BY {group_expr} "
                if only_incomplete:
                    # `COALESCE(g.…, 0)` explícito: si se escribe el alias, SQLite lo
                    # resuelve contra la columna homónima de la CTE, que es NULL para
                    # quien no tiene partidas, y `NULL < ?` excluía de la cola a todo
                    # jugador del padrón con 0 partidas (13.444 Diamond al 2026-10-21).
                    query += " HAVING COALESCE(g.existing_recorded_games, 0) < ? "
                    params.append(min_matches_target)
                # Prioridad (pedido del usuario, 2026-10-23): los que MENOS partidas
                # tienen primero. Sus partidas traen a otros jugadores del padrón, así
                # que los que están cerca de la meta se completan de rebote. Los nunca
                # visitados van antes que los ya visitados con la misma cantidad: un
                # jugador que agotó su historial no tapa al resto en cada sesión.
                query += """
                    ORDER BY
                        CASE WHEN existing_recorded_games > 0 AND existing_recorded_games < ? THEN 0 ELSE 1 END,
                        CASE WHEN p.last_crawled_at IS NULL THEN 0 ELSE 1 END,
                        existing_recorded_games ASC,
                        p.last_crawled_at ASC;
                """
                params.append(min_matches_target)
                cursor.execute(query, params)
                cols = [d[0] for d in cursor.description]
                rows = [dict(zip(cols, r)) for r in cursor.fetchall()]
                candidates = self._project_candidates(cursor, by_player_id=has_pid_map)
                for row in rows:
                    cand = dict(candidates.get(row["canonical_id"], {}))
                    # El PUUID del padrón siempre es de P0: lo sincronizó la key de P0.
                    cand.setdefault(0, row["players_puuid"])
                    row["candidates"] = cand
                    row["puuid"] = cand[0]
                return rows
        return self.get_players_for_deep_crawl(tier=tier, min_matches_target=min_matches_target,
                                               only_incomplete=only_incomplete)

    def _project_candidates(self, cursor, by_player_id: bool) -> Dict[str, Dict[int, str]]:
        """`{persona: {project_index: puuid}}` con los PUUIDs de proyecto conocido de cada persona.

        Es lo que deja al crawler elegir, para cada jugador, una key que pueda
        descifrar el PUUID que le manda. Fuentes, en orden de confianza: `players`
        (P0, lo sincronizó la key de P0), `player_puuid_map` y `puuid_projects`
        ('exact' antes que 'inferred'). La persona es el `player_id` (o el
        `canonical_id` de la vista en bases sin `player_id_map`).
        """
        sources = """
            SELECT puuid, 0 AS project_index, 0 AS rank FROM players
            UNION ALL SELECT puuid, project_index, 0 FROM player_puuid_map
            UNION ALL SELECT puuid, project_index,
                             CASE source WHEN 'exact' THEN 0 ELSE 1 END FROM puuid_projects
        """
        if by_player_id:
            members = "SELECT player_id AS person, puuid FROM player_id_map"
        else:
            members = """
                SELECT canonical_id AS person, alias_puuid AS puuid FROM v_puuid_canonical
                UNION SELECT canonical_id, canonical_puuid FROM v_puuid_canonical
                UNION SELECT p.puuid, p.puuid FROM players p
                      WHERE p.puuid NOT IN (SELECT alias_puuid FROM v_puuid_canonical)
            """
        cursor.execute(f"""
            SELECT m.person, s.project_index, s.puuid, MIN(s.rank) AS rank
            FROM ({members}) m JOIN ({sources}) s ON s.puuid = m.puuid
            GROUP BY m.person, s.project_index, s.puuid
            ORDER BY m.person, s.project_index, rank, s.puuid;
        """)
        out: Dict[str, Dict[int, str]] = {}
        for person, project, puuid, _rank in cursor.fetchall():
            out.setdefault(person, {}).setdefault(int(project), puuid)  # el primero es el más confiable
        return out

    # Cupos oficiales del padrón LAS (CH y GM exactos; Master/Diamond referenciales).
    EXPECTED_ROSTER_SIZES = {
        "CHALLENGER": 200,
        "GRANDMASTER": 500,
        "MASTER": 8259,
        "DIAMOND": 31653,
    }

    def check_roster_quotas(self) -> Dict[str, Any]:
        """Compara el padrón `players` contra los cupos oficiales. SOLO ADVIERTE, nunca bloquea.

        Retorna {tier: {actual, expected, ok, note}}. CH/MASTER/etc duplicados por
        doble proyecto aparecen como actual > expected y generan warning.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT tier, COUNT(*) FROM players GROUP BY tier;")
            actual = {str(r[0]).upper(): int(r[1]) for r in cursor.fetchall()}
        report: Dict[str, Any] = {}
        for tier, expected in self.EXPECTED_ROSTER_SIZES.items():
            n = actual.get(tier, 0)
            report[tier] = {
                "actual": n,
                "expected": expected,
                "ok": n == expected,
                "note": ("OK" if n == expected
                         else f"ADVERTENCIA: se esperaban {expected}, hay {n} (posible duplicación multi-proyecto)" if n > expected
                         else f"AVISO: faltan {expected - n} registros por sincronizar"),
            }
        return report

    _RIOT_ID_UPSERT_SQL = """
        INSERT INTO puuid_riot_ids
            (puuid, canonical_id, game_name, tag_line, project_index,
             observations, first_seen_game_creation, last_seen_game_creation)
        VALUES (?, ?, ?, ?, ?, 1, ?, ?)
        ON CONFLICT(puuid, canonical_id) DO UPDATE SET
            observations = puuid_riot_ids.observations + 1,
            game_name = excluded.game_name,
            tag_line = excluded.tag_line,
            project_index = COALESCE(excluded.project_index,
                                     puuid_riot_ids.project_index),
            first_seen_game_creation = MIN(
                COALESCE(puuid_riot_ids.first_seen_game_creation,
                         excluded.first_seen_game_creation),
                COALESCE(excluded.first_seen_game_creation,
                         puuid_riot_ids.first_seen_game_creation)),
            last_seen_game_creation = MAX(
                COALESCE(puuid_riot_ids.last_seen_game_creation,
                         excluded.last_seen_game_creation),
                COALESCE(excluded.last_seen_game_creation,
                         puuid_riot_ids.last_seen_game_creation)),
            updated_at = CURRENT_TIMESTAMP;
    """

    def record_riot_id_observations(
            self,
            observations: List[Tuple[str, str, str]],
            project_index: Optional[int] = None,
            game_creation: Optional[int] = None) -> int:
        """Anota observaciones `(puuid, game_name, tag_line)` del payload de Match-V5.

        Estos nombres vienen en texto plano, así que valen para cualquier proyecto
        de API y son el puente entre espacios de PUUID mutuamente indescifrables.
        Acumula `observations` y estira la ventana `first_seen`/`last_seen` con el
        `gameCreation` de la partida, que es lo que permite ordenar renombres.

        Idempotente por `(puuid, canonical_id)`. Retorna cuántas filas se tocaron.
        """
        rows = []
        for puuid, game_name, tag_line in observations or ():
            if not puuid or not game_name:
                continue
            tag_line = tag_line or ""
            rows.append((puuid, self.canonical_id_for(game_name, tag_line),
                         game_name, tag_line, project_index,
                         game_creation, game_creation))
        if not rows:
            return 0
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(self._RIOT_ID_UPSERT_SQL, rows)
            conn.commit()
        return len(rows)

    def record_puuid_alias(self, primary_puuid: str, alias_puuid: str,
                           game_name: Optional[str] = None,
                           tag_line: Optional[str] = None) -> bool:
        """Registra un link legacy alias -> PUUID live SIN reescribir historial.

        A diferencia de resolve_and_merge_player_alias (migración física de filas),
        esto solo anota que `alias_puuid` (proyecto muerto, indescifrable) pertenece
        al mismo humano que `primary_puuid` (proyecto live). La unificación ocurre
        en lectura vía v_puuid_canonical (export/coverage/crawl). Idempotente.
        """
        if not primary_puuid or not alias_puuid or primary_puuid == alias_puuid:
            return False
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM match_participants WHERE puuid = ?;",
                           (alias_puuid,))
            n = cursor.fetchone()[0] or 0
            cursor.execute("""
                INSERT INTO player_puuid_aliases
                    (primary_puuid, alias_puuid, game_name, tag_line, matches_migrated)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(primary_puuid, alias_puuid) DO UPDATE SET
                    game_name=COALESCE(excluded.game_name, player_puuid_aliases.game_name),
                    tag_line=COALESCE(excluded.tag_line, player_puuid_aliases.tag_line),
                    matches_migrated=excluded.matches_migrated;
            """, (primary_puuid, alias_puuid, game_name, tag_line, int(n)))
            conn.commit()
        # Refrescar caché en memoria para que el hot path vea el link al instante.
        self._translation_cache[alias_puuid] = self._canonical_puuid(primary_puuid)
        return True

    def dedupe_doubled_participants(self) -> Dict[str, int]:
        """Fusiona filas dobles (mismo match_id + participant_id con 2 PUUIDs).

        Causa: una partida ingerida bajo dos proyectos muertos. Conserva una sola
        fila con el PUUID canónico (vía traducción) y alinea el timeline.
        Solo toca grupos donde AMBAS filas traducen al mismo canónico; el resto se
        reporta sin tocar. Retorna {groups, fixed, skipped}.
        """
        stats = {"groups": 0, "fixed": 0, "skipped": 0}
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT match_id, participant_id
                FROM match_participants
                GROUP BY match_id, participant_id
                HAVING COUNT(*) > 1;
            """)
            groups = cursor.fetchall()
        stats["groups"] = len(groups)
        for match_id, pid in groups:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT rowid, puuid FROM match_participants
                    WHERE match_id = ? AND participant_id = ? ORDER BY rowid;
                """, (match_id, pid))
                rows = cursor.fetchall()
                canons = {self._canonical_puuid(r[1]) for r in rows}
                if len(canons) != 1:
                    stats["skipped"] += 1
                    continue
                canon = next(iter(canons))
                keep = next((r for r in rows if r[1] == canon), rows[0])
                drop_ids = [r[0] for r in rows if r[0] != keep[0]]
                if keep[1] != canon:
                    cursor.execute("UPDATE match_participants SET puuid = ? WHERE rowid = ?;",
                                   (canon, keep[0]))
                if drop_ids:
                    cursor.execute(
                        f"DELETE FROM match_participants WHERE rowid IN "
                        f"({','.join('?' for _ in drop_ids)});", drop_ids)
                cursor.execute("UPDATE match_timeline SET puuid = ? WHERE match_id = ? "
                               "AND participant_id = ?;", (canon, match_id, pid))
                conn.commit()
                stats["fixed"] += 1
        # La traducción pudo sumar entradas: recargar para consistencia.
        self.reload_translation_cache()
        return stats

    def prune_stale_tier_players(self, fresh_by_tier: Dict[str, Set[str]]) -> Dict[str, Any]:
        """Elimina filas del padrón que ya no están en el roster fresco, CON GUARDIA.

        Solo poda un tier si el roster fresco coincide EXACTO (CH/GM) o dentro del
        ±5% (Master/Diamond) del cupo oficial: así un sync parcial por error de API
        jamás borra jugadores legítimos. Retorna {tier: {kept, pruned, replaced}}.
        """
        report: Dict[str, Any] = {}
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for tier, fresh in fresh_by_tier.items():
                t = str(tier).upper()
                expected = self.EXPECTED_ROSTER_SIZES.get(t)
                n = len(fresh)
                safe = False
                if expected:
                    if t in ("CHALLENGER", "GRANDMASTER"):
                        safe = (n == expected)
                    else:
                        safe = abs(n - expected) / expected <= 0.05
                else:
                    safe = n > 0
                if not safe or n == 0:
                    report[t] = {"kept": None, "pruned": 0, "replaced": False,
                                 "note": f"sin poda (fresco={n}, esperado={expected})"}
                    continue
                cursor.execute("CREATE TEMP TABLE IF NOT EXISTS _keep_puuids(puuid TEXT PRIMARY KEY);")
                cursor.execute("DELETE FROM _keep_puuids;")
                cursor.executemany("INSERT OR IGNORE INTO _keep_puuids(puuid) VALUES (?);",
                                   [(p,) for p in fresh])
                cursor.execute("SELECT COUNT(*) FROM players WHERE tier = ?;", (t,))
                before = cursor.fetchone()[0] or 0
                cursor.execute("""
                    DELETE FROM players WHERE tier = ?
                      AND puuid NOT IN (SELECT puuid FROM _keep_puuids);
                """, (t,))
                pruned = cursor.rowcount or 0
                conn.commit()
                report[t] = {"kept": before - pruned, "pruned": pruned, "replaced": True,
                             "note": f"padrón {t} reemplazado por snapshot fresco ({n})"}
            try:
                cursor.execute("DROP TABLE IF EXISTS _keep_puuids;")
            except sqlite3.OperationalError:
                pass
        return report


