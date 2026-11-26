"""
Proyecto de cifrado de cada PUUID (`puuid_projects`).

Por qué hace falta
------------------
Riot cifra el PUUID con el proyecto de la key que hace la petición. Para pedir
la lista de partidas de un jugador, la key tiene que recibir un PUUID de *su*
proyecto, o responde `400 Exception decrypting`. Sin saber de qué proyecto es
cada PUUID guardado, el crawler mandaba PUUIDs indescifrables y repartía mal la
carga entre keys.

Fuentes exactas (las dijo Riot)
-------------------------------
* `players`: el padrón lo sincroniza la key de P0.
* `player_puuid_map`: una fila por (persona, proyecto), escrita por ranura.
* `puuid_riot_ids` con etiqueta 1, 2 o 3 (la 0 está contaminada, ver §9.S.4).
* La ingesta, desde ahora: los 10 PUUIDs de cada Match DTO vienen cifrados con
  el proyecto de la key que lo descargó.

Deducción por partida
---------------------
Una partida guardada se descargó con una sola key, así que los PUUIDs crudos de
sus participantes son todos del mismo proyecto. Para un PUUID sin etiqueta se
cuentan los proyectos conocidos de los demás PUUIDs de sus partidas y se toma el
mayoritario si su proporción supera `min_purity`.

Validación (2026-10-21, leave-one-out sobre 30.149 PUUIDs de proyecto exacto):
con pureza >= 0,95 se cubre el 86,9 % con 98,65 % de acierto. El error viene de
filas que la ingesta vieja había traducido a otro proyecto. Por eso lo deducido
se guarda como `'inferred'` y el crawler lo corrige con la primera respuesta de
Riot.
"""

from collections import defaultdict
from typing import Dict, Iterable, Mapping, Tuple

MIN_PURITY = 0.95


def infer_projects_from_matches(rows: Iterable[Tuple[str, str]],
                                exact: Mapping[str, int],
                                min_purity: float = MIN_PURITY) -> Dict[str, int]:
    """`{puuid: proyecto}` deducido para los PUUIDs que no están en `exact`.

    `rows` son pares `(match_id, puuid)` de `match_participants`. Solo votan los
    PUUIDs de proyecto exacto, y nunca sobre sí mismos.
    """
    by_match: Dict[str, list] = defaultdict(list)
    for match_id, puuid in rows:
        if match_id and puuid:
            by_match[match_id].append(puuid)

    votes: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for puuids in by_match.values():
        known = [exact[p] for p in puuids if p in exact]
        if not known:
            continue
        tally: Dict[int, int] = defaultdict(int)
        for pj in known:
            tally[pj] += 1
        for p in puuids:
            if p in exact:
                continue
            for pj, n in tally.items():
                votes[p][pj] += n

    out: Dict[str, int] = {}
    for puuid, tally in votes.items():
        total = sum(tally.values())
        project, best = max(tally.items(), key=lambda kv: (kv[1], -kv[0]))
        if total and best / total >= min_purity:
            out[puuid] = project
    return out


def exact_projects(conn) -> Dict[str, int]:
    """`{puuid: proyecto}` de las fuentes exactas de la base (ver docstring del módulo)."""
    exact: Dict[str, int] = {}
    queries = (
        "SELECT puuid, project_index FROM puuid_riot_ids "
        "WHERE project_index IS NOT NULL AND project_index <> 0",
        "SELECT puuid, 0 FROM players",
        "SELECT puuid, project_index FROM player_puuid_map",
        "SELECT puuid, project_index FROM puuid_projects WHERE source = 'exact'",
    )
    for sql in queries:  # las posteriores pisan a las anteriores
        try:
            for puuid, pj in conn.execute(sql):
                if puuid and pj is not None:
                    exact[puuid] = int(pj)
        except Exception:
            continue
    return exact
