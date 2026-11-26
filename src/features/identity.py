"""
Identidad de jugador por componentes conexas (`player_id`).

Problema
--------
Ningún identificador disponible es, por sí solo, una clave de persona válida:

* El **PUUID** es estable ante cambios de nombre, pero está cifrado por *producto*
  de API: el mismo humano tiene un PUUID distinto por cada proyecto de Riot
  registrado en `api_projects`.
* El **`canonical_id`** (`lower(gameName#tagLine)`) es portable entre proyectos,
  pero se rompe cuando el jugador se renombra.

Medido sobre la BD viva el 2026-09-07: 3.102 `canonical_id` apuntaban a más de un
`canonical_puuid` (misma persona partida en varias filas) y 158 `canonical_puuid`
apuntaban a más de un `canonical_id` (renombres). Agrupar por cualquiera de las
dos columnas por separado fragmenta al mismo humano.

Solución
--------
Union-find sobre las dos clases de arista a la vez:

* **Capa 1 (dentro de proyecto):** `alias_puuid — canonical_puuid`. Une los PUUIDs
  que `v_puuid_canonical` ya sabe que son el mismo jugador.
* **Capa 2 (entre proyectos):** todos los PUUIDs que comparten un mismo
  `canonical_id` con forma `name#tag`.

Cada componente conexa resultante es una persona, y su etiqueta es el `player_id`.

Etiqueta
--------
Se prefiere el `name#tag` mínimo lexicográficamente del componente; si el
componente no tiene ningún nombre conocido, se usa el PUUID mínimo. Es
determinista para un estado dado de la base.

> **Advertencia:** `player_id` es una clave *derivada del estado actual de la
> base*, no un identificador durable. Si más adelante se descubre un alias nuevo
> o un renombre, la etiqueta de ese componente puede cambiar entre exportaciones.
> Sirve para agrupar dentro de un dataset, no para citar a un jugador entre
> versiones del dataset.
"""

from typing import Dict, Iterable, List, Optional, Set, Tuple

# Nombre de la tabla temporal/permanente que materializa el mapa para consultas SQL.
TEMP_TABLE = "player_id_map"


class UnionFind:
    """Union-find con compresión de caminos y unión por tamaño."""

    def __init__(self) -> None:
        self._parent: Dict[str, str] = {}
        self._size: Dict[str, int] = {}

    def add(self, item: str) -> None:
        if item not in self._parent:
            self._parent[item] = item
            self._size[item] = 1

    def find(self, item: str) -> str:
        self.add(item)
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        # Compresión de caminos (iterativa: los componentes pueden ser largos).
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._size[ra] < self._size[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        self._size[ra] += self._size[rb]

    def components(self) -> Dict[str, List[str]]:
        """Devuelve {raiz: [miembros...]}."""
        out: Dict[str, List[str]] = {}
        for item in self._parent:
            out.setdefault(self.find(item), []).append(item)
        return out


def _is_name(candidate: Optional[str]) -> bool:
    """Un `canonical_id` es un nombre real solo si tiene forma `gameName#tagLine`."""
    return isinstance(candidate, str) and "#" in candidate


def _load_known_projects(conn) -> Dict[str, int]:
    """Devuelve `{puuid: project_index}` para todo PUUID cuyo proyecto de cifrado es conocido.

    Fuentes, solo las que dijo Riot (nunca deducidas):
    - `players`: padrón sincronizado siempre bajo el proyecto canónico 0.
    - `player_puuid_map`: mapeo explícito `(canonical_id, project_index, puuid)`.
    - `puuid_projects` con `source = 'exact'`: proyecto de la key que trajo o
      descifró el PUUID.
    - `puuid_riot_ids` con `project_index` 1, 2 o 3. La etiqueta 0 de esa tabla
      **no se usa**: la ingesta anterior marcaba como 0 cualquier PUUID traducido,
      aunque fuera de P1 o P3 (4.088 casos, §9.S.4). Las etiquetas confiables ya
      están copiadas en `puuid_projects` por el backfill.
    Las etiquetas 'inferred' tampoco entran: este invariante decide si dos
    componentes son personas distintas, y una deducción errónea partiría a una.
    """
    puuid_proj: Dict[str, int] = {}
    try:
        for (puuid,) in conn.execute("SELECT puuid FROM players;"):
            if puuid:
                puuid_proj[puuid] = 0
    except Exception:
        pass
    try:
        for puuid, proj in conn.execute(
            "SELECT puuid, project_index FROM player_puuid_map WHERE project_index IS NOT NULL;"
        ):
            if puuid and proj is not None:
                puuid_proj[puuid] = int(proj)
    except Exception:
        pass
    try:
        for puuid, proj in conn.execute(
            "SELECT puuid, project_index FROM puuid_projects WHERE source = 'exact';"
        ):
            if puuid and proj is not None:
                puuid_proj.setdefault(puuid, int(proj))
    except Exception:
        pass
    try:
        for puuid, proj in conn.execute(
            "SELECT puuid, project_index FROM puuid_riot_ids "
            "WHERE project_index IS NOT NULL AND project_index <> 0;"
        ):
            if puuid and proj is not None:
                puuid_proj.setdefault(puuid, int(proj))
    except Exception:
        pass
    return puuid_proj


def build_player_id_map(conn,
                        extra_puuids: Optional[Iterable[str]] = None) -> Dict[str, str]:
    """
    Construye `{puuid_crudo: player_id}` uniendo las tres capas de identidad:

    * **Capa 1 (dentro/entre proyectos por match):** `alias_puuid — canonical_puuid`
      de `v_puuid_canonical` (`player_puuid_map` + `player_puuid_aliases`).
    * **Capa 2 (entre proyectos por nombre canónico):** PUUIDs que comparten un mismo
      `canonical_id` (`name#tag`) en `v_puuid_canonical`.
    * **Capa 3 (observaciones directas de Match-V5):** `(puuid, canonical_id)` en
      `puuid_riot_ids`, que aporta el puente entre proyectos y el historial de
      renombres con ventana temporal (`first_seen_game_creation` / `last_seen_game_creation`).

    **Invariante de proyecto:** dos PUUIDs distintos cifrados bajo el *mismo*
    `project_index` jamás pueden pertenecer a la misma cuenta de Riot. Si dos
    componentes ya contienen PUUIDs del mismo proyecto, una arista por nombre
    (`Capa 2/3`) entre ellos indica **reuso de nombre** por cuentas distintas en
    distintos momentos (ej. `jugador_a#tag`, `jugador_b#tag1`, `jugador_c#tag`) y se
    rechaza.

    **Biyección componente ↔ `player_id`:** si tras bloquear una unión por reuso de
    nombre dos componentes distintos comparten el mismo nombre candidato, se
    prioriza primero un nombre no disputado del componente (en caso de renombre) y,
    si persiste la colisión, se desambigua con `~<prefijo_puuid>` para no volver a
    fusionarlos al agrupar por `player_id`.
    """
    uf = UnionFind()
    names: Dict[str, Set[str]] = {}           # puuid -> {name#tag conocidos}
    by_name: Dict[str, List[str]] = {}        # name#tag -> puuids que lo comparten
    last_seen: Dict[Tuple[str, str], int] = {}  # (puuid, name#tag) -> timestamp máx

    puuid_proj = _load_known_projects(conn)

    try:
        rows: List[Tuple[str, str, Optional[str]]] = conn.execute(
            "SELECT alias_puuid, canonical_puuid, canonical_id FROM v_puuid_canonical;"
        ).fetchall()
    except Exception:
        rows = []

    for alias_puuid, canonical_puuid, canonical_id in rows:
        if not alias_puuid:
            continue
        uf.add(alias_puuid)
        # Capa 1: el alias y su canónico son la misma persona (join exacto por match/participant).
        if canonical_puuid:
            uf.union(alias_puuid, canonical_puuid)
        # Capa 2: se acumula por nombre y se une después respetando el invariante de proyecto.
        if _is_name(canonical_id):
            names.setdefault(alias_puuid, set()).add(canonical_id)
            by_name.setdefault(canonical_id, []).append(alias_puuid)
            if canonical_puuid:
                names.setdefault(canonical_puuid, set()).add(canonical_id)

    # Incluir filas de `player_puuid_map` de proyectos != 0 aunque aún no tengan par en proyecto 0.
    try:
        map_rows: List[Tuple[str, str]] = conn.execute(
            "SELECT puuid, canonical_id FROM player_puuid_map;"
        ).fetchall()
    except Exception:
        map_rows = []

    for puuid, canonical_id in map_rows:
        if not puuid:
            continue
        uf.add(puuid)
        if _is_name(canonical_id):
            names.setdefault(puuid, set()).add(canonical_id)
            by_name.setdefault(canonical_id, []).append(puuid)

    # Capa 3: observaciones de `puuid_riot_ids` (puente entre proyectos + historial de renombres).
    try:
        riot_rows: List[Tuple[str, str, Optional[int]]] = conn.execute(
            "SELECT puuid, canonical_id, last_seen_game_creation FROM puuid_riot_ids;"
        ).fetchall()
    except Exception:
        riot_rows = []

    for puuid, canonical_id, seen_ts in riot_rows:
        if not puuid:
            continue
        uf.add(puuid)
        if _is_name(canonical_id):
            names.setdefault(puuid, set()).add(canonical_id)
            by_name.setdefault(canonical_id, []).append(puuid)
            if seen_ts is not None:
                key = (puuid, canonical_id)
                last_seen[key] = max(last_seen.get(key, 0), int(seen_ts))

    # Mapa de proyectos conocidos por cada raíz actual de Capa 1.
    comp_projects: Dict[str, Set[int]] = {}
    for root, members in uf.components().items():
        comp_projects[root] = {puuid_proj[m] for m in members if m in puuid_proj}

    # Unir por nombre (Capas 2 y 3) respetando el invariante: un componente no puede
    # contener dos PUUIDs distintos del mismo proyecto de cifrado.
    for cname, shared in sorted(by_name.items()):
        # Ordenar candidatos: primero los que no tienen timestamp viejo (o tienen el
        # más reciente) para que el dueño vigente del nombre se una primero con P0/P1/P2.
        unique_shared = sorted(
            set(shared),
            key=lambda p: (-last_seen.get((p, cname), 10**18), p),
        )
        accepted_roots: List[str] = []
        for candidate in unique_shared:
            rc = uf.find(candidate)
            pc = comp_projects.setdefault(
                rc, {puuid_proj[rc]} if rc in puuid_proj else set()
            )
            merged = False
            for target_root in list(accepted_roots):
                rt = uf.find(target_root)
                if rc == rt:
                    merged = True
                    break
                pt = comp_projects.setdefault(
                    rt, {puuid_proj[rt]} if rt in puuid_proj else set()
                )
                # Si ambos componentes ya tienen un PUUID del mismo proyecto, no son
                # la misma cuenta: rechazar la unión por nombre.
                if pc & pt:
                    continue
                uf.union(rt, rc)
                new_root = uf.find(rt)
                comp_projects[new_root] = pt | pc
                rc = new_root
                pc = comp_projects[new_root]
                merged = True
                break
            if not merged:
                accepted_roots.append(rc)

    for puuid in extra_puuids or ():
        if puuid:
            uf.add(puuid)

    # Etiqueta determinista por componente, garantizando biyección componente <-> player_id.
    all_components = list(uf.components().values())
    comp_known: List[Tuple[List[str], List[str]]] = []
    name_component_count: Dict[str, int] = {}
    for members in all_components:
        known = sorted({n for m in members for n in names.get(m, ())})
        comp_known.append((members, known))
        for n in known:
            name_component_count[n] = name_component_count.get(n, 0) + 1

    # Elegir etiqueta candidata: si el componente tiene varios nombres (por renombre)
    # y alguno es exclusivo de este componente, preferirlo sobre un nombre disputado.
    by_label: Dict[str, List[List[str]]] = {}
    for members, known in comp_known:
        if not known:
            label = min(members)
        else:
            uncontested = [n for n in known if name_component_count.get(n, 0) == 1]
            label = uncontested[0] if uncontested else known[0]
        by_label.setdefault(label, []).append(members)

    mapping: Dict[str, str] = {}
    for base_label, comp_list in by_label.items():
        if len(comp_list) == 1:
            for member in comp_list[0]:
                mapping[member] = base_label
        else:
            # Desambiguar determinísticamente componentes distintos que colisionan en
            # el mismo nombre (reuso de nombre entre cuentas del mismo proyecto).
            comp_list.sort(key=lambda mems: (-len(mems), min(mems)))
            for idx, members in enumerate(comp_list):
                final_label = base_label if idx == 0 else f"{base_label}~{min(members)[:8]}"
                for member in members:
                    mapping[member] = final_label
    return mapping


def audit_duplicates(conn, table: str = TEMP_TABLE) -> Dict[str, int]:
    """Auditoría de jugadores duplicados sobre un `player_id_map` ya construido.

    Cada contador debería ser 0 (salvo `nombres_compartidos`, ver abajo). Mide las
    cuatro formas en que un jugador puede quedar duplicado o fusionado:

    * `padron_duplicado`: personas con más de un PUUID en `players` (la misma
      persona dos veces en el padrón).
    * `misma_persona_dos_veces_en_partida`: partidas donde dos PUUIDs del mismo
      `player_id` juegan juntos. Es imposible en la realidad, así que delata una
      **fusión falsa** de dos personas.
    * `partidas_sin_10_personas`: partidas con menos o más de 10 `player_id`
      distintos (participaciones repetidas o fusiones).
    * `personas_sin_nombre_con_partidas`: componentes sin ningún `name#tag`
      conocido que tienen partidas. Son PUUIDs que no se pudieron unir a nadie:
      posibles **mitades sueltas** de una persona.
    * `nombres_compartidos`: `name#tag` que aparece en más de una persona. No es
      error por sí solo (dos cuentas distintas pueden usar el mismo nombre en
      épocas distintas: el invariante de proyecto las separa con `~`), pero
      acota cuántas personas podrían estar partidas.
    """
    q = lambda sql: conn.execute(sql).fetchone()[0] or 0
    return {
        "padron_duplicado": q(f"""
            SELECT COUNT(*) FROM (SELECT m.player_id FROM players p
                JOIN {table} m ON m.puuid = p.puuid GROUP BY m.player_id HAVING COUNT(*) > 1)"""),
        "misma_persona_dos_veces_en_partida": q(f"""
            SELECT COUNT(*) FROM (SELECT mp.match_id, m.player_id FROM match_participants mp
                JOIN {table} m ON m.puuid = mp.puuid
                GROUP BY mp.match_id, m.player_id HAVING COUNT(*) > 1)"""),
        "partidas_sin_10_personas": q(f"""
            SELECT COUNT(*) FROM (SELECT mp.match_id FROM match_participants mp
                JOIN {table} m ON m.puuid = mp.puuid
                GROUP BY mp.match_id HAVING COUNT(DISTINCT m.player_id) <> 10)"""),
        "personas_sin_nombre_con_partidas": q(f"""
            SELECT COUNT(DISTINCT m.player_id) FROM match_participants mp
                JOIN {table} m ON m.puuid = mp.puuid WHERE m.player_id NOT LIKE '%#%'"""),
        "nombres_compartidos": q(f"""
            SELECT COUNT(*) FROM (SELECT r.canonical_id FROM puuid_riot_ids r
                JOIN {table} m ON m.puuid = r.puuid
                GROUP BY r.canonical_id HAVING COUNT(DISTINCT m.player_id) > 1)"""),
        "listado_ausente_en_su_partida": _listed_but_absent(conn, table),
    }


def _listed_but_absent(conn, table: str) -> int:
    """Partidas que Riot listó para un jugador, ya descargadas, donde ese jugador no aparece.

    Riot garantiza que el jugador jugó cada partida de su lista, así que si no
    figura en ella es que sus PUUIDs de esa partida quedaron en otra persona
    (jugador partido en dos, típicamente tras un cambio de nombre). 0 en bases
    sin recolección por ventana.

    Se guarda el PUUID del padrón del jugador (estable) y no su `player_id`, que
    es una etiqueta que puede cambiar al recalcular el mapa: con la etiqueta, un
    cambio haría que el chequeo no encuentre al jugador y pase en silencio.
    """
    existe = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' "
                          "AND name = 'crawl_window_listings'").fetchone()
    if not existe:
        return 0  # base sin recolección por ventana
    return conn.execute(f"""
        WITH listed AS (
            SELECT DISTINCT l.match_id, cur.player_id
            FROM crawl_window_listings l
            LEFT JOIN {table} cur ON cur.puuid = l.players_puuid
            WHERE EXISTS (SELECT 1 FROM matches mt WHERE mt.match_id = l.match_id)
        )
        SELECT COUNT(*) FROM listed
        WHERE listed.player_id IS NULL  -- jugador sin persona en el mapa: también es falla
           OR NOT EXISTS (
            SELECT 1 FROM match_participants mp JOIN {table} m ON m.puuid = mp.puuid
            WHERE mp.match_id = listed.match_id AND m.player_id = listed.player_id)
    """).fetchone()[0] or 0



def suspicious_shared_names(conn, table: str = TEMP_TABLE) -> List[Tuple[str, List[str]]]:
    """Nombres compartidos entre personas **sin** prueba de que sean cuentas distintas.

    Un nombre compartido es legítimo si alguna pareja de esas personas tiene cada
    una un PUUID exacto del mismo proyecto: Riot da un solo PUUID por cuenta y
    proyecto, así que son dos cuentas. Si ninguna pareja tiene esa prueba, la
    persona podría estar partida en dos y se devuelve para revisar.
    """
    shared = conn.execute(f"""
        SELECT r.canonical_id, GROUP_CONCAT(DISTINCT m.player_id) FROM puuid_riot_ids r
        JOIN {table} m ON m.puuid = r.puuid
        GROUP BY r.canonical_id HAVING COUNT(DISTINCT m.player_id) > 1""").fetchall()
    projects = _load_known_projects(conn)
    members: Dict[str, Set[str]] = {}
    for puuid, pid in conn.execute(f"SELECT puuid, player_id FROM {table}"):
        members.setdefault(pid, set()).add(puuid)
    out = []
    for name, pids_csv in shared:
        pids = pids_csv.split(",")
        proj = {pid: {projects[p] for p in members.get(pid, ()) if p in projects} for pid in pids}
        proven = any(proj[a] & proj[b] for i, a in enumerate(pids) for b in pids[i + 1:])
        if not proven:
            out.append((name, pids))
    return out


def _seed_puuids_from_db(conn) -> Set[str]:
    """Reúne todos los PUUIDs de `match_participants` y `players` para sembrar el mapa."""
    puuids: Set[str] = set()
    try:
        for (p,) in conn.execute("SELECT DISTINCT puuid FROM match_participants;"):
            if p:
                puuids.add(p)
    except Exception:
        pass
    try:
        for (p,) in conn.execute("SELECT DISTINCT puuid FROM players;"):
            if p:
                puuids.add(p)
    except Exception:
        pass
    return puuids


def attach_player_id_map(conn, extra_puuids: Optional[Iterable[str]] = None) -> int:
    """
    Materializa el mapa en una tabla TEMP para que las consultas puedan agrupar
    por `player_id` sin traer 78k filas a pandas.

    Las tablas TEMP viven en una base separada, así que funcionan incluso con la
    conexión abierta en `mode=ro`. Devuelve la cantidad de PUUIDs mapeados.
    """
    mapping = build_player_id_map(conn, extra_puuids=extra_puuids)
    conn.execute(f"DROP TABLE IF EXISTS temp.{TEMP_TABLE};")
    conn.execute(
        f"CREATE TEMP TABLE {TEMP_TABLE} ("
        "    puuid TEXT PRIMARY KEY,"
        "    player_id TEXT NOT NULL"
        ");"
    )
    conn.executemany(
        f"INSERT INTO {TEMP_TABLE} (puuid, player_id) VALUES (?, ?);",
        mapping.items(),
    )
    conn.execute(
        f"CREATE INDEX temp.idx_{TEMP_TABLE}_pid ON {TEMP_TABLE}(player_id);"
    )
    return len(mapping)


def attach_for_participants(conn) -> int:
    """`attach_player_id_map` sembrado con todos los PUUID de `match_participants` y `players`."""
    return attach_player_id_map(conn, extra_puuids=_seed_puuids_from_db(conn))


def materialize_player_id_map(conn, extra_puuids: Optional[Iterable[str]] = None) -> int:
    """
    Materializa el mapa en una tabla permanente `player_id_map` en la base de datos
    (con índice sobre `player_id`), permitiendo que cualquier consulta SQL directa
    agrupe o filtre por persona física sin tener que recalcular el Union-Find en RAM.

    Mantiene la idempotencia: si la tabla ya existe, la reemplaza atómicamente.
    Devuelve la cantidad de PUUIDs mapeados.
    """
    mapping = build_player_id_map(conn, extra_puuids=extra_puuids)
    with conn:
        conn.execute("CREATE TABLE IF NOT EXISTS player_id_map ("
                     "    puuid TEXT PRIMARY KEY,"
                     "    player_id TEXT NOT NULL"
                     ");")
        conn.execute("DELETE FROM player_id_map;")
        conn.executemany(
            "INSERT INTO player_id_map (puuid, player_id) VALUES (?, ?);",
            mapping.items(),
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_player_id_map_pid ON player_id_map(player_id);"
        )
    return len(mapping)


def materialize_for_database(conn) -> int:
    """Materializa `player_id_map` permanente usando la misma semilla que `attach_for_participants`."""
    return materialize_player_id_map(conn, extra_puuids=_seed_puuids_from_db(conn))


