"""
Motor de validación incremental **solo lectura** sobre la BD viva.

Por qué existe: el crawler pasa una fracción grande de su tiempo bloqueado
esperando ranura de rate limit (18 req/1 s, 95 req/120 s por key; ver
`AGENTS.md` §3.B). Ese tiempo es CPU y disco ociosos con la base ya llena de
partidas sin auditar. Este módulo parte la auditoría en **rebanadas** lo bastante
baratas como para caber dentro de una espera de rate limit, de modo que validar
no le robe throughput al crawler.

Tres piezas, deliberadamente separadas:

1. `build_facts()` — **función pura**: filas crudas (partida, participantes,
   timeline) → `MatchFacts` por partida. Sin BD, sin red.
2. `evaluate()` — **función pura**: `MatchFacts` → lista de slugs de hallazgo.
   Toda la doctrina de qué es un error vive acá y se testea sin abrir SQLite.
3. `IncrementalValidator` — el envoltorio con estado: cursor persistido en disco,
   conexión `mode=ro` y acumulación de hallazgos entre corridas.

Invariante de concurrencia: **nunca escribe en la BD**. Abre en `mode=ro`
(URI) igual que `scripts/ops/identity_progress.py`, así corre en paralelo con el
crawler sin tomar locks de escritura ni arriesgar `[WinError 32]`.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.database import open_readonly_connection

# Severidades. Ordenadas de peor a mejor para el reporte.
ERROR = "error"   # viola un invariante de AGENTS.md §3: el dato está mal
AVISO = "aviso"   # anomalía recuperable, no invalida la partida
INFO = "info"     # sesgo conocido / deuda metodológica de AGENTS.md §3.F

#: slug -> (severidad, explicación en una línea)
CHECKS: Dict[str, Tuple[str, str]] = {
    "duracion_invalida":         (ERROR, "gameDuration nulo o negativo"),
    "participantes_incompletos": (ERROR, "la partida no tiene exactamente 10 participantes"),
    "equipos_desbalanceados":    (ERROR, "los equipos 100/200 no son 5 y 5"),
    "ganador_incoherente":       (ERROR, "no hay exactamente 5 participantes con win=1"),
    "participant_id_duplicado":  (ERROR, "dos participantes comparten participantId"),
    "campeon_invalido":          (ERROR, "championId/championName vacio o <= 0"),
    "stats_negativas":           (ERROR, "kills/deaths/assists/gold negativos"),
    "remake_no_descartado":      (AVISO, "remake (<= 300 s) guardado en 'matches'"),
    "cola_no_soloq":             (AVISO, "queueId distinto de 420 (RANKED_SOLO_5x5)"),
    "timeline_faltante":         (AVISO, "partida no-remake sin ninguna fila de timeline"),
    "timeline_incompleto":       (AVISO, "timeline presente pero sin 10 participantes"),
    "timeline_de_remake":        (AVISO, "remake con timeline descargada (request desperdiciada)"),
    "snapshot10_ausente":        (AVISO, "partida >= 11 min sin snapshot @10"),
    "posicion_vacia":            (INFO, "individualPosition vacio o 'Invalid'"),
    "posicion_duplicada":        (INFO, "posicion repetida en un equipo (deuda AGENTS.md 3.F.5)"),
    "snapshot20_ausente":        (INFO, "partida >= 21 min sin snapshot @20 (deuda AGENTS.md 3.F.4)"),
}

#: Umbral de duración (s) por debajo del cual la partida es remake y se descarta.
REMAKE_MAX_SECONDS = 300
#: Cola de SoloQ 5v5 clasificatoria.
SOLOQ_QUEUE_ID = 420
#: Márgenes para exigir cada snapshot: un minuto de gracia sobre el corte nominal.
MIN_DURATION_FOR_SNAPSHOT_10 = 660    # 11 min
MIN_DURATION_FOR_SNAPSHOT_20 = 1260   # 21 min
#: Posiciones que Riot devuelve cuando no pudo inferir el rol.
EMPTY_POSITIONS = {"", "INVALID", "NONE", "UNKNOWN"}
#: Cuántos match_id de ejemplo se guardan por hallazgo (para poder investigarlos).
MAX_SAMPLES = 5

STATE_VERSION = 1


@dataclass
class MatchFacts:
    """Todo lo que hace falta saber de una partida para juzgarla, ya agregado."""
    match_id: str
    game_duration: Optional[int]
    queue_id: Optional[int]
    n_participants: int = 0
    n_blue: int = 0
    n_red: int = 0
    n_wins: int = 0
    n_distinct_participant_ids: int = 0
    n_champion_invalid: int = 0
    n_stats_negative: int = 0
    n_position_empty: int = 0
    n_position_duplicated: int = 0
    n_timeline_rows: int = 0
    n_gold10_null: int = 0
    n_gold20_null: int = 0


def _is_empty_position(pos: Any) -> bool:
    return str(pos or "").strip().upper() in EMPTY_POSITIONS


def build_facts(match_rows: Iterable[Sequence[Any]],
                participant_rows: Iterable[Sequence[Any]],
                timeline_rows: Iterable[Sequence[Any]]) -> List[MatchFacts]:
    """
    Agrega filas crudas en un `MatchFacts` por partida. Función pura.

    - `match_rows`:       (match_id, game_duration, queue_id)
    - `participant_rows`: (match_id, participant_id, team_id, win, champion_id,
                           champion_name, individual_position, kills, deaths,
                           assists, gold_earned)
    - `timeline_rows`:    (match_id, gold_10, gold_20)

    Las participaciones y timelines de partidas fuera de `match_rows` se ignoran:
    los huérfanos son un chequeo global, no de rebanada (ver `global_checks`).
    """
    facts: Dict[str, MatchFacts] = {}
    for mid, duration, queue_id in match_rows:
        facts[mid] = MatchFacts(match_id=mid,
                                game_duration=duration,
                                queue_id=queue_id)

    seen_participant_ids: Dict[str, set] = {}
    position_slots: Dict[str, Dict[Tuple[Any, str], int]] = {}

    for row in participant_rows:
        (mid, participant_id, team_id, win, champion_id, champion_name,
         position, kills, deaths, assists, gold) = row
        f = facts.get(mid)
        if f is None:
            continue
        f.n_participants += 1
        if team_id == 100:
            f.n_blue += 1
        elif team_id == 200:
            f.n_red += 1
        if win:
            f.n_wins += 1
        seen_participant_ids.setdefault(mid, set()).add(participant_id)
        if not champion_id or int(champion_id) <= 0 or not str(champion_name or "").strip():
            f.n_champion_invalid += 1
        if min(kills or 0, deaths or 0, assists or 0, gold or 0) < 0:
            f.n_stats_negative += 1
        if _is_empty_position(position):
            f.n_position_empty += 1
        else:
            slots = position_slots.setdefault(mid, {})
            key = (team_id, str(position).strip().upper())
            slots[key] = slots.get(key, 0) + 1

    for mid, ids in seen_participant_ids.items():
        facts[mid].n_distinct_participant_ids = len(ids)

    for mid, slots in position_slots.items():
        # Cuenta participantes involucrados en un choque, no choques distintos:
        # un doble TOP aporta 2, un triple aporta 3.
        facts[mid].n_position_duplicated = sum(c for c in slots.values() if c > 1)

    for mid, gold_10, gold_20 in timeline_rows:
        f = facts.get(mid)
        if f is None:
            continue
        f.n_timeline_rows += 1
        if gold_10 is None:
            f.n_gold10_null += 1
        if gold_20 is None:
            f.n_gold20_null += 1

    return list(facts.values())


def evaluate(f: MatchFacts) -> List[str]:
    """
    Juzga una partida ya agregada y devuelve los slugs de `CHECKS` que dispara.
    Función pura: acá vive toda la doctrina de qué cuenta como error.
    """
    found: List[str] = []
    duration = f.game_duration

    if duration is None or duration < 0:
        found.append("duracion_invalida")
        duration = None

    if duration is not None and duration <= REMAKE_MAX_SECONDS:
        found.append("remake_no_descartado")

    if f.queue_id is not None and f.queue_id != SOLOQ_QUEUE_ID:
        found.append("cola_no_soloq")

    if f.n_participants != 10:
        # Con el roster incompleto, el reparto por equipo y el ganador son
        # consecuencia del faltante y no hallazgos independientes: no se reportan.
        found.append("participantes_incompletos")
    else:
        if f.n_blue != 5 or f.n_red != 5:
            found.append("equipos_desbalanceados")
        if f.n_wins != 5:
            found.append("ganador_incoherente")

    if f.n_participants and f.n_distinct_participant_ids != f.n_participants:
        found.append("participant_id_duplicado")
    if f.n_champion_invalid:
        found.append("campeon_invalido")
    if f.n_stats_negative:
        found.append("stats_negativas")
    if f.n_position_empty:
        found.append("posicion_vacia")
    if f.n_position_duplicated:
        found.append("posicion_duplicada")

    es_remake = duration is not None and duration <= REMAKE_MAX_SECONDS
    if es_remake:
        if f.n_timeline_rows:
            found.append("timeline_de_remake")
    elif duration is not None:
        if f.n_timeline_rows == 0:
            found.append("timeline_faltante")
        elif f.n_timeline_rows != 10:
            found.append("timeline_incompleto")

    if f.n_timeline_rows and duration is not None:
        if duration >= MIN_DURATION_FOR_SNAPSHOT_10 and f.n_gold10_null == f.n_timeline_rows:
            found.append("snapshot10_ausente")
        if duration >= MIN_DURATION_FOR_SNAPSHOT_20 and f.n_gold20_null == f.n_timeline_rows:
            found.append("snapshot20_ausente")

    return found


def open_readonly(db_path: Path) -> sqlite3.Connection:
    """
    Conexión de solo lectura a la BD viva. `mode=ro` es el punto entero:
    garantiza a nivel SQLite que esta ruta jamás toma un lock de escritura,
    aunque el crawler esté ingiriendo en paralelo.
    """
    return open_readonly_connection(db_path, query_only=True)


def _placeholders(n: int) -> str:
    return ",".join("?" * n)


def global_checks(conn: sqlite3.Connection, include_identity: bool = False) -> Dict[str, int]:
    """
    Chequeos que no se pueden partir por partida: huérfanos y contradicciones
    entre tablas. Se corren una vez por sesión, no por rebanada.
    """
    c = conn.cursor()
    out: Dict[str, int] = {}
    out["participaciones_huerfanas"] = c.execute(
        "SELECT COUNT(*) FROM match_participants p "
        "WHERE NOT EXISTS (SELECT 1 FROM matches m WHERE m.match_id = p.match_id);"
    ).fetchone()[0]
    out["timelines_huerfanas"] = c.execute(
        "SELECT COUNT(*) FROM match_timeline t "
        "WHERE NOT EXISTS (SELECT 1 FROM matches m WHERE m.match_id = t.match_id);"
    ).fetchone()[0]
    out["partidas_sin_participantes"] = c.execute(
        "SELECT COUNT(*) FROM matches m "
        "WHERE NOT EXISTS (SELECT 1 FROM match_participants p WHERE p.match_id = m.match_id);"
    ).fetchone()[0]
    out["descartadas_pero_guardadas"] = c.execute(
        "SELECT COUNT(*) FROM discarded_matches d "
        "WHERE EXISTS (SELECT 1 FROM matches m WHERE m.match_id = d.match_id);"
    ).fetchone()[0]

    if include_identity:
        # Caro (recorre las ~1M de participaciones contra la vista de identidad):
        # solo bajo pedido explícito, nunca dentro del bucle de rebanadas.
        out["participaciones_sin_identidad"] = c.execute(
            "SELECT COUNT(*) FROM match_participants mp "
            "WHERE NOT EXISTS (SELECT 1 FROM v_puuid_canonical v "
            "                  WHERE v.alias_puuid = mp.puuid);"
        ).fetchone()[0]
    return out


@dataclass
class ValidationState:
    """Cursor y hallazgos acumulados, persistidos entre corridas."""
    last_rowid: int = 0
    matches_validated: int = 0
    slices: int = 0
    findings: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    started_at: str = ""
    updated_at: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {"version": STATE_VERSION, "last_rowid": self.last_rowid,
                "matches_validated": self.matches_validated, "slices": self.slices,
                "findings": self.findings, "started_at": self.started_at,
                "updated_at": self.updated_at}

    @classmethod
    def from_json(cls, raw: Dict[str, Any]) -> "ValidationState":
        if int(raw.get("version", 0)) != STATE_VERSION:
            return cls()
        return cls(last_rowid=int(raw.get("last_rowid", 0)),
                   matches_validated=int(raw.get("matches_validated", 0)),
                   slices=int(raw.get("slices", 0)),
                   findings=dict(raw.get("findings", {})),
                   started_at=str(raw.get("started_at", "")),
                   updated_at=str(raw.get("updated_at", "")))

    def record(self, check: str, match_id: str) -> None:
        entry = self.findings.setdefault(check, {"count": 0, "samples": []})
        entry["count"] += 1
        if len(entry["samples"]) < MAX_SAMPLES:
            entry["samples"].append(match_id)


class IncrementalValidator:
    """
    Validador con estado: mantiene el cursor, ejecuta rebanadas y acumula
    hallazgos. Una instancia por proceso; su método `run_slice()` está pensado
    para llamarse muchas veces, en ráfagas cortas, desde el hueco que deja el
    rate limit.
    """

    def __init__(self, db_path: Path, state_path: Optional[Path] = None,
                 slice_size: int = 300):
        self.db_path = Path(db_path)
        self.state_path = Path(state_path) if state_path else \
            self.db_path.parent / "validation_state.json"
        self.slice_size = slice_size
        self.state = self._load_state()
        self._conn: Optional[sqlite3.Connection] = None

    # -- estado en disco ---------------------------------------------------
    def _load_state(self) -> ValidationState:
        try:
            return ValidationState.from_json(
                json.loads(self.state_path.read_text(encoding="utf-8")))
        except Exception:
            return ValidationState()

    def save_state(self) -> None:
        self.state.updated_at = time.strftime("%Y-%m-%d %H:%M:%S")
        if not self.state.started_at:
            self.state.started_at = self.state.updated_at
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state.to_json(), indent=2, ensure_ascii=False),
                       encoding="utf-8")
        tmp.replace(self.state_path)

    def reset(self) -> None:
        self.state = ValidationState()
        self.save_state()

    # -- conexión ----------------------------------------------------------
    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = open_readonly(self.db_path)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # -- trabajo -----------------------------------------------------------
    def pending(self) -> int:
        """Partidas en la BD todavía por delante del cursor."""
        return self.conn.execute(
            "SELECT COUNT(*) FROM matches WHERE rowid > ?;",
            (self.state.last_rowid,)).fetchone()[0]

    def run_slice(self) -> int:
        """
        Valida una rebanada. Devuelve cuántas partidas auditó (0 = al día).

        El cursor es el `rowid` de `matches`, que crece monótonamente con la
        ingesta: avanzarlo equivale a "auditadas hasta acá", y las partidas que
        el crawler guarda mientras tanto quedan delante del cursor para la
        pasada siguiente.

        No guarda el estado en disco: eso lo decide quien orquesta, para no
        reescribir el JSON cientos de veces por minuto.
        """
        rows = self.conn.execute(
            "SELECT rowid, match_id, game_duration, queue_id FROM matches "
            "WHERE rowid > ? ORDER BY rowid LIMIT ?;",
            (self.state.last_rowid, self.slice_size)).fetchall()
        if not rows:
            return 0

        ids = [r[1] for r in rows]
        ph = _placeholders(len(ids))
        participant_rows = self.conn.execute(
            f"SELECT match_id, participant_id, team_id, win, champion_id, champion_name, "
            f"individual_position, kills, deaths, assists, gold_earned "
            f"FROM match_participants WHERE match_id IN ({ph});", ids).fetchall()
        timeline_rows = self.conn.execute(
            f"SELECT match_id, gold_10, gold_20 FROM match_timeline "
            f"WHERE match_id IN ({ph});", ids).fetchall()

        facts = build_facts([(r[1], r[2], r[3]) for r in rows],
                            participant_rows, timeline_rows)
        for f in facts:
            for check in evaluate(f):
                self.state.record(check, f.match_id)

        self.state.last_rowid = rows[-1][0]
        self.state.matches_validated += len(rows)
        self.state.slices += 1
        return len(rows)

    def run_until_done(self, max_seconds: Optional[float] = None) -> int:
        """Consume el backlog entero (o hasta agotar el presupuesto de tiempo)."""
        deadline = time.monotonic() + max_seconds if max_seconds else None
        total = 0
        while True:
            n = self.run_slice()
            if n == 0:
                break
            total += n
            if deadline and time.monotonic() >= deadline:
                break
        self.save_state()
        return total

    # -- reporte -----------------------------------------------------------
    def findings_by_severity(self) -> Dict[str, List[Tuple[str, int, List[str]]]]:
        """Hallazgos agrupados por severidad, ordenados de mayor a menor conteo."""
        grouped: Dict[str, List[Tuple[str, int, List[str]]]] = {ERROR: [], AVISO: [], INFO: []}
        for slug, entry in self.state.findings.items():
            severity = CHECKS.get(slug, (AVISO, ""))[0]
            grouped[severity].append((slug, entry.get("count", 0), entry.get("samples", [])))
        for rows in grouped.values():
            rows.sort(key=lambda item: item[1], reverse=True)
        return grouped

    def error_count(self) -> int:
        return sum(e.get("count", 0) for slug, e in self.state.findings.items()
                   if CHECKS.get(slug, (AVISO, ""))[0] == ERROR)
