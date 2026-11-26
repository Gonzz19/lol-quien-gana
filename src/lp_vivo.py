"""
LP por partida medido en vivo (PROJECT_SUMMARY.md §9.S.22), en una base aparte de `lol_las.db`.

Se lee el ladder de Master+ cada pocos minutos y se comparan dos lecturas seguidas de cada
jugador. Si W + L subió 1, entre las dos lecturas jugó exactamente una partida y la resta
de LP es lo que esa partida movió. Después se busca la partida en Match-V5.

Tablas: `lecturas` (cada lectura del ladder), `estado` (última lectura de cada jugador),
`eventos` (cada cambio, con su partida si se pudo enlazar), `partidas` y `participantes`.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from src.config import QUEUE_SOLOQ_ID

PLATAFORMAS = {"la2": "americas", "br1": "americas", "na1": "americas", "euw1": "europe"}
HOLGURA_ANTES_MS = 25 * 60_000   # el ladder se atrasa respecto del fin de la partida (medido: hasta ~15 min)
HOLGURA_DESPUES_MS = 60_000
ATRASO_MAX = 10
REMAKE_S = 300
REINTENTO_MS = 5 * 60_000
MAX_INTENTOS = 72                # ~6 h: Match-V5 tarda en publicar algunas partidas

ESQUEMA = """
CREATE TABLE IF NOT EXISTS lecturas (
    platform TEXT NOT NULL, t_inicio INTEGER NOT NULL, t_fin INTEGER NOT NULL, jugadores INTEGER NOT NULL,
    PRIMARY KEY (platform, t_inicio));
CREATE TABLE IF NOT EXISTS estado (
    platform TEXT NOT NULL, puuid TEXT NOT NULL, tier TEXT, rank TEXT, lp INTEGER, wins INTEGER, losses INTEGER,
    en_ladder INTEGER NOT NULL, t_lectura INTEGER NOT NULL, PRIMARY KEY (platform, puuid));
CREATE TABLE IF NOT EXISTS eventos (
    id INTEGER PRIMARY KEY, platform TEXT NOT NULL, puuid TEXT NOT NULL, t_antes INTEGER, t_despues INTEGER NOT NULL,
    tier_antes TEXT, rank_antes TEXT, lp_antes INTEGER, wins_antes INTEGER, losses_antes INTEGER,
    tier_despues TEXT, rank_despues TEXT, lp_despues INTEGER, wins_despues INTEGER, losses_despues INTEGER,
    tipo TEXT NOT NULL, estado TEXT NOT NULL, match_id TEXT, intentos INTEGER NOT NULL DEFAULT 0,
    proximo_intento INTEGER NOT NULL DEFAULT 0);
CREATE INDEX IF NOT EXISTS ix_eventos_pendientes ON eventos (estado, platform, proximo_intento);
CREATE INDEX IF NOT EXISTS ix_eventos_jugador ON eventos (platform, puuid, estado);
CREATE TABLE IF NOT EXISTS partidas (
    match_id TEXT PRIMARY KEY, platform TEXT NOT NULL, queue_id INTEGER, game_version TEXT, game_creation INTEGER,
    game_start INTEGER, game_end INTEGER, duracion_s INTEGER);
CREATE TABLE IF NOT EXISTS participantes (
    match_id TEXT NOT NULL, puuid TEXT NOT NULL, team_id INTEGER, win INTEGER, champion TEXT, team_position TEXT,
    kills INTEGER, deaths INTEGER, assists INTEGER, gold INTEGER, cs INTEGER, PRIMARY KEY (match_id, puuid));
CREATE INDEX IF NOT EXISTS ix_participantes_puuid ON participantes (puuid);
"""


def clasificar(antes: Optional[dict], despues: dict) -> str:
    """Qué pasó entre dos lecturas, según la resta de W + L."""
    if antes is None:
        return "entro"   # primera vez que se lo ve: no hay LP de antes
    d_n = despues["wins"] + despues["losses"] - antes["wins"] - antes["losses"]
    if d_n == 1:
        return "partida"
    if d_n > 1:
        return "varias"  # no se puede repartir el LP entre las partidas
    return "reinicio" if d_n < 0 else "sin_partida"   # sin_partida: decay u otro cambio de LP


def cambio(antes: dict, despues: dict) -> bool:
    return any(antes[c] != despues[c] for c in ("tier", "rank", "league_points", "wins", "losses"))


def atrasada(antes: dict, despues: dict) -> bool:
    """Lectura de una réplica vieja: W + L baja un poco y en la lectura siguiente vuelve a subir.

    Medido: en 8 minutos, 189 jugadores rebotaron así, cada uno con una "partida" falsa al
    volver. Una baja grande no es un atraso sino un reinicio de la temporada.
    """
    d_n = despues["wins"] + despues["losses"] - antes["wins"] - antes["losses"]
    return -ATRASO_MAX <= d_n < 0


def comparar(previo: Dict[str, dict], actual: Dict[str, dict]) -> Tuple[List[Tuple[Optional[dict], dict]], List[str]]:
    """Compara dos lecturas. Devuelve los cambios (antes, después) y los que salieron del ladder."""
    cambios = []
    for puuid, nuevo in actual.items():
        viejo = previo.get(puuid)
        if viejo is None or (cambio(viejo, nuevo) and not atrasada(viejo, nuevo)):
            cambios.append((viejo, nuevo))
    salieron = [p for p, v in previo.items() if v.get("en_ladder", True) and p not in actual]
    return cambios, salieron


def elegir_partida(partidas: Iterable[dict], t_antes: int, t_despues: int) -> Optional[dict]:
    """La partida que explica un evento: terminó en su ventana y no fue un remake.

    `None` si no hay ninguna o si hay más de una, porque entonces no se sabe cuál movió el LP.
    """
    validas = [p for p in partidas if p["duracion_s"] > REMAKE_S
               and t_antes - HOLGURA_ANTES_MS < p["game_end"] <= t_despues + HOLGURA_DESPUES_MS]
    return validas[0] if len(validas) == 1 else None


def resumir_partida(detalle: dict) -> Tuple[dict, List[dict]]:
    """Lo que se guarda de la respuesta de Match-V5."""
    info = detalle["info"]
    inicio = info.get("gameStartTimestamp") or info.get("gameCreation")
    fin = info.get("gameEndTimestamp") or (inicio + info.get("gameDuration", 0) * 1000)
    partida = {"match_id": detalle["metadata"]["matchId"], "platform": info.get("platformId", "").lower(),
               "queue_id": info.get("queueId"), "game_version": info.get("gameVersion"),
               "game_creation": info.get("gameCreation"), "game_start": inicio, "game_end": fin,
               "duracion_s": info.get("gameDuration", 0)}
    jugadores = [{"match_id": partida["match_id"], "puuid": p["puuid"], "team_id": p.get("teamId"),
                  "win": int(bool(p.get("win"))), "champion": p.get("championName"),
                  "team_position": p.get("teamPosition"), "kills": p.get("kills"), "deaths": p.get("deaths"),
                  "assists": p.get("assists"), "gold": p.get("goldEarned"),
                  "cs": p.get("totalMinionsKilled", 0) + p.get("neutralMinionsKilled", 0)}
                 for p in info.get("participants", [])]
    return partida, jugadores


def agregar_lobby(d):
    """Suma `lobby_lp` (`z_antes` medio de los otros jugadores medidos de la partida) y `lobby_conocidos`.

    Usa solo el LP de antes de la partida. NaN si no se midió a ningún otro.
    """
    g = d.groupby("match_id")["z_antes"]
    otros = g.transform("count") - 1
    return d.assign(lobby_lp=(g.transform("sum") - d["z_antes"]) / otros.where(otros > 0), lobby_conocidos=otros)


SQL_ENLAZADAS = (
    "SELECT e.match_id, e.platform, e.puuid, e.t_antes, e.t_despues, e.tier_antes, e.rank_antes, e.lp_antes, "
    "e.wins_antes, e.losses_antes, e.tier_despues, e.rank_despues, e.lp_despues, p.game_end, p.duracion_s, "
    "p.game_version, j.team_id, j.win, j.champion, j.team_position FROM eventos e "
    "JOIN partidas p ON p.match_id = e.match_id "
    "JOIN participantes j ON j.match_id = e.match_id AND j.puuid = e.puuid WHERE e.estado = 'enlazada'")


def cargar_enlazadas(ruta: Path, plataformas: Optional[Iterable[str]] = None):
    """Una fila por jugador y partida enlazada, con el LP de antes y de después en la escala
    continua (`z_antes`, `z_despues`), `delta_lp` y el lobby (`agregar_lobby`).

    Abre la base en solo lectura y cierra la conexión al salir. El lobby se calcula antes de
    filtrar por `plataformas` (una partida es de un solo servidor, así que no cambia).
    """
    import pandas as pd
    from src.features.rating import lp_absoluto
    conn = sqlite3.connect(f"file:{Path(ruta).resolve()}?mode=ro", uri=True, timeout=60.0)
    try:
        d = pd.read_sql_query(SQL_ENLAZADAS, conn)
    finally:
        conn.close()
    for lado in ("antes", "despues"):
        d[f"z_{lado}"] = [lp_absoluto(t, r, l) for t, r, l in zip(d[f"tier_{lado}"], d[f"rank_{lado}"], d[f"lp_{lado}"])]
    d["delta_lp"] = d["z_despues"] - d["z_antes"]
    d = agregar_lobby(d)
    if plataformas is not None:
        d = d[d["platform"].isin(list(plataformas))].reset_index(drop=True)
    return d


class AlmacenLP:
    """Base `lp_vivo.db`. Cada operación abre y cierra su conexión (AGENTS.md §3.A)."""

    def __init__(self, ruta: Path):
        self.ruta = Path(ruta)
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.executescript(ESQUEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.ruta, timeout=60.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def cargar_estado(self, platform: str) -> Tuple[Dict[str, dict], Optional[int]]:
        with self._conn() as c:
            filas = c.execute("SELECT * FROM estado WHERE platform = ?", (platform,)).fetchall()
            t = c.execute("SELECT MAX(t_inicio) FROM lecturas WHERE platform = ?", (platform,)).fetchone()[0]
        return {f["puuid"]: {"puuid": f["puuid"], "tier": f["tier"], "rank": f["rank"], "league_points": f["lp"],
                             "wins": f["wins"], "losses": f["losses"], "en_ladder": bool(f["en_ladder"])}
                for f in filas}, t

    def guardar_lectura(self, platform: str, t_inicio: int, t_fin: int, jugadores: int,
                        eventos: List[Tuple[Optional[dict], dict, str, Optional[int]]],
                        estados: List[dict]) -> None:
        """Lectura, eventos `(antes, después, tipo, t_antes)` y estados nuevos, en una transacción."""
        filas = []
        for antes, d, tipo, t_antes in eventos:
            a = antes or {}
            filas.append((platform, d["puuid"], t_antes, t_fin, a.get("tier"), a.get("rank"), a.get("league_points"),
                          a.get("wins"), a.get("losses"), d["tier"], d["rank"], d["league_points"], d["wins"],
                          d["losses"], tipo, "pendiente" if tipo == "partida" else "no_aplica"))
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO lecturas VALUES (?, ?, ?, ?)", (platform, t_inicio, t_fin, jugadores))
            c.executemany(
                "INSERT INTO eventos (platform, puuid, t_antes, t_despues, tier_antes, rank_antes, lp_antes, "
                "wins_antes, losses_antes, tier_despues, rank_despues, lp_despues, wins_despues, losses_despues, "
                "tipo, estado) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", filas)
            c.executemany(
                "INSERT OR REPLACE INTO estado VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(platform, e["puuid"], e["tier"], e["rank"], e["league_points"], e["wins"], e["losses"],
                  int(e.get("en_ladder", True)), t_fin) for e in estados])

    def pendientes(self, platform: str, ahora: int, limite: int = 50) -> List[sqlite3.Row]:
        with self._conn() as c:
            return c.execute("SELECT * FROM eventos WHERE estado = 'pendiente' AND platform = ? AND proximo_intento <= ? "
                             "ORDER BY t_despues LIMIT ?", (platform, ahora, limite)).fetchall()

    def partidas_conocidas(self, ids: Iterable[str]) -> Dict[str, dict]:
        ids = list(ids)
        if not ids:
            return {}
        with self._conn() as c:
            filas = c.execute(f"SELECT * FROM partidas WHERE match_id IN ({','.join('?' * len(ids))})", ids).fetchall()
        return {f["match_id"]: dict(f) for f in filas}

    def guardar_partida(self, partida: dict, jugadores: List[dict]) -> None:
        with self._conn() as c:
            c.execute("INSERT OR IGNORE INTO partidas VALUES (:match_id, :platform, :queue_id, :game_version, "
                      ":game_creation, :game_start, :game_end, :duracion_s)", partida)
            c.executemany("INSERT OR IGNORE INTO participantes VALUES (:match_id, :puuid, :team_id, :win, :champion, "
                          ":team_position, :kills, :deaths, :assists, :gold, :cs)", jugadores)

    def enlazar_participantes(self, partida: dict, *otras: dict) -> int:
        """Enlaza a los participantes con una única candidata conocida en su ventana."""
        ids = {p["match_id"] for p in (partida, *otras)}
        placeholders = ",".join("?" for _ in ids)
        with self._conn() as c:
            c.execute("BEGIN IMMEDIATE")
            filas = c.execute(f"""
                WITH afectados AS (
                    SELECT DISTINCT e.id
                    FROM partidas p
                    JOIN participantes j ON j.match_id = p.match_id
                    JOIN eventos e ON e.platform = p.platform AND e.puuid = j.puuid
                    WHERE p.match_id IN ({placeholders}) AND e.estado = 'pendiente'
                )
                SELECT e.id, e.platform, e.puuid, e.wins_despues - e.wins_antes AS delta_w,
                       p.match_id, j.win,
                       EXISTS (SELECT 1 FROM eventos usado
                               WHERE usado.platform = e.platform AND usado.puuid = e.puuid
                                 AND usado.match_id = p.match_id) AS usada
                FROM afectados a
                JOIN eventos e ON e.id = a.id
                JOIN participantes j ON j.puuid = e.puuid
                JOIN partidas p ON p.match_id = j.match_id AND p.platform = e.platform
                WHERE p.queue_id = ? AND p.duracion_s > ?
                  AND p.game_end > e.t_antes - ? AND p.game_end <= e.t_despues + ?
                ORDER BY e.t_despues, e.id
            """, (*ids, QUEUE_SOLOQ_ID, REMAKE_S, HOLGURA_ANTES_MS, HOLGURA_DESPUES_MS)).fetchall()

            candidatas = {}
            for fila in filas:
                candidatas.setdefault(fila["id"], []).append(fila)

            updates = []
            ocupadas = set()
            for evento_id, opciones in candidatas.items():
                if len(opciones) != 1:
                    continue
                candidata = opciones[0]
                match_id = candidata["match_id"]
                clave = (candidata["platform"], candidata["puuid"], match_id)
                if match_id not in ids or candidata["usada"] or clave in ocupadas:
                    continue
                estado = "enlazada" if (candidata["delta_w"] == 1) == bool(candidata["win"]) else "inconsistente"
                updates.append((estado, match_id, evento_id))
                ocupadas.add(clave)
            c.executemany("UPDATE eventos SET estado = ?, match_id = ? WHERE id = ?", updates)
            return len(updates)

    def reprogramar(self, evento_id: int, ahora: int) -> None:
        with self._conn() as c:
            c.execute("UPDATE eventos SET intentos = intentos + 1, proximo_intento = ?, "
                      "estado = CASE WHEN intentos + 1 >= ? THEN 'sin_enlace' ELSE estado END "
                      "WHERE id = ? AND estado = 'pendiente'",
                      (ahora + REINTENTO_MS, MAX_INTENTOS, evento_id))

    def resumen(self) -> Dict[str, dict]:
        with self._conn() as c:
            filas = c.execute("SELECT platform, tipo, estado, COUNT(*) n FROM eventos GROUP BY 1, 2, 3").fetchall()
            lect = c.execute("SELECT platform, COUNT(*) n, MAX(jugadores) j FROM lecturas GROUP BY 1").fetchall()
        out = {f["platform"]: {"lecturas": f["n"], "jugadores": f["j"]} for f in lect}
        for f in filas:
            out.setdefault(f["platform"], {})[f"{f['tipo']}/{f['estado']}"] = f["n"]
        return out


def leer_ladder(cliente, tiers: Iterable[str]) -> Optional[Dict[str, dict]]:
    """Las ligas completas de una plataforma. `None` si alguna falló: una lectura parcial haría
    parecer que sus jugadores salieron del ladder."""
    actual = {}
    for tier in tiers:
        jugadores = cliente.get_apex_league(tier)
        if not cliente.last_request_ok() or not jugadores:
            return None
        actual.update({j["puuid"]: j for j in jugadores if j.get("puuid")})
    return actual


def ciclo_ladder(cliente, almacen: AlmacenLP, platform: str, previo: Dict[str, dict], t_previo: Optional[int],
                 tiers: Iterable[str], proyecto: int, reloj) -> Tuple[Optional[int], dict]:
    """Una lectura: compara con la anterior, consulta a los que salieron y guarda. Actualiza `previo`.

    La ventana del evento va del inicio de la lectura anterior al fin de esta, porque las tres
    ligas no se leen en el mismo instante.
    """
    t_inicio = reloj()
    actual = leer_ladder(cliente, tiers)
    if actual is None:
        return t_previo, {"fallo": True}
    primera = not previo
    cambios, salieron = comparar(previo, actual)
    eventos, estados = [], []
    for antes, nuevo in cambios:
        if not primera:
            eventos.append((antes, nuevo, clasificar(antes, nuevo), t_previo if antes is not None else None))
        estados.append({**nuevo, "en_ladder": True})
    for puuid in salieron:   # bajó a Diamond: el ladder ya no lo trae, se lo pide por PUUID
        r = cliente.get_soloq_entry_by_puuid({proyecto: puuid})
        if r is None:
            continue
        nuevo = {**r, "puuid": puuid}
        if atrasada(previo[puuid], nuevo):
            continue
        if cambio(previo[puuid], nuevo):
            eventos.append((previo[puuid], nuevo, clasificar(previo[puuid], nuevo), t_previo))
        # También sin cambio: Master tiene un tope de 10.000 y el corrido por otros sale sin jugar.
        # Si vuelve a aparecer, el ladder lo sigue midiendo igual.
        estados.append({**nuevo, "en_ladder": False})
    t_fin = reloj()
    almacen.guardar_lectura(platform, t_inicio, t_fin, len(actual), eventos, estados)
    previo.update({e["puuid"]: e for e in estados})
    return t_inicio, {"fallo": False, "jugadores": len(actual), "eventos": len(eventos), "salieron": len(salieron)}


def resolver_evento(cliente_partidas, almacen: AlmacenLP, evento, ahora: int) -> None:
    """Busca la partida de un evento y enlaza a los 10 jugadores. Si no aparece, reintenta más tarde."""
    desde_s = (evento["t_antes"] - HOLGURA_ANTES_MS) // 1000
    ids = list(dict.fromkeys(cliente_partidas.get_match_ids_by_puuid(
        evento["puuid"], count=5, start_time=desde_s)))
    conocidas = almacen.partidas_conocidas(ids)
    for match_id in ids:
        if match_id in conocidas:
            continue
        detalle = cliente_partidas.get_match_detail(match_id)
        if not detalle or "info" not in detalle:
            continue
        partida, jugadores = resumir_partida(detalle)
        almacen.guardar_partida(partida, jugadores)
        conocidas[match_id] = partida
    # Un detalle faltante todavía puede ser una segunda candidata del evento.
    if conocidas and len(conocidas) == len(ids):
        almacen.enlazar_participantes(*conocidas.values())
    almacen.reprogramar(evento["id"], ahora)
