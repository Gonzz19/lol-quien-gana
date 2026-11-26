#!/usr/bin/env python3
"""Respuesta completa de match-v5 de las partidas de Master+ de A2-A4 y V (PROJECT_SUMMARY.md §9.S.71).

`lol_las.db` descarta casi toda la respuesta de Riot: no guarda bans, rol pedido ni autofill
(`selectedRolePreferences`), runas, hechizos ni el bloque `challenges`. Este script vuelve a pedir
las partidas y guarda el JSON entero comprimido en una base aparte, `data/crudas/partidas.db`, para
no tener que pedirlas otra vez. No toca `lol_las.db`. Toda petición pasa por `RiotAPIClient` y su
`RiotRateLimiter` (AGENTS.md §3.B).

Orden de descarga: una permutación con semilla fija de las partidas de Master+ de los sub-bloques
A2-A4 y de V (`brainstorm_base.cargar`), mezcladas: cualquier corte deja partidas para ajustar y
para medir. Cortar y retomar sigue por donde iba; subir --n agranda la misma muestra.

Uso:
    python scripts/collection/partidas_crudas.py --n 30300    # ~8 h 45 con una key de desarrollo
    python scripts/collection/partidas_crudas.py --estado     # sin red
"""
import argparse
import gzip
import json
import sqlite3
import sys
import time
from contextlib import closing
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

RUTA = ROOT / "data" / "crudas" / "partidas.db"
ESQUEMA = """
CREATE TABLE IF NOT EXISTS partidas (match_id TEXT PRIMARY KEY, estado TEXT, game_creation INTEGER,
    game_version TEXT, bajada_en INTEGER, json_gz BLOB);
"""


def conectar(ruta: Path = RUTA):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(ruta, timeout=60.0)
    conn.executescript(ESQUEMA)
    return closing(conn)


def previo_de_partida(detalle: dict) -> dict:
    """Lo que se conoce antes de jugar: bans por equipo y rol pedido de cada jugador.

    Devuelve {"bans": {100: [championId, ...], 200: [...]}, "jugadores": [dict por participante]}.
    Un ban -1 (el jugador no baneó) se descarta. `selectedRolePreferences` viene como
    "ASIGNADO.TIPO.PEDIDO1.PEDIDO2[.FILL]", con TIPO en PRIMARY / SECONDARY / AUTOFILL.
    """
    info = detalle["info"]
    bans = {t["teamId"]: [b["championId"] for b in sorted(t.get("bans") or [], key=lambda b: b.get("pickTurn", 0))
                          if b.get("championId", -1) > 0] for t in info.get("teams") or []}
    jugadores = []
    for p in info["participants"]:
        partes = (p.get("selectedRolePreferences") or "").split(".")
        tipo = partes[1] if len(partes) > 1 else None
        jugadores.append({"participant_id": p.get("participantId"), "team_id": p.get("teamId"),
                          "champion_id": p.get("championId"), "posicion": p.get("teamPosition"),
                          "asignada": p.get("positionAssignedByMatchmaking"), "tipo_rol": tipo,
                          "autofill": None if tipo is None else tipo == "AUTOFILL",
                          "pedidos": [x for x in partes[2:] if x != "FILL"], "acepta_fill": "FILL" in partes[2:]})
    return {"bans": bans, "jugadores": jugadores}


def orden_de_descarga() -> list:
    from scripts.analysis.brainstorm_base import cargar
    D = cargar()
    idx = np.flatnonzero(((D["bloque"] >= 1) & D["master"]) | D["val"])
    return list(D["Z"]["mids"][np.random.default_rng(20261116).permutation(idx)])


def estado() -> None:
    with conectar() as conn:
        for est, n, kb in conn.execute("SELECT estado, COUNT(*), SUM(LENGTH(json_gz)) / 1024 FROM partidas GROUP BY estado"):
            print(f"partidas {est}: {n:,} ({(kb or 0) / 1024:.0f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=30300, help="partidas de la muestra (acumulado, no adicional)")
    ap.add_argument("--estado", action="store_true")
    args = ap.parse_args()
    if args.estado:
        return estado()

    from src.riot_client import RiotAPIClient
    orden = orden_de_descarga()[: args.n]
    with conectar() as conn:
        hechas = {r[0] for r in conn.execute("SELECT match_id FROM partidas")}
    pendientes = [m for m in orden if m not in hechas]
    print(f"muestra {len(orden):,} | ya bajadas {len(orden) - len(pendientes):,} | pendientes {len(pendientes):,}", flush=True)
    cliente = RiotAPIClient()
    t0, ok, fallos_seguidos = time.time(), 0, 0
    try:
        for i, mid in enumerate(pendientes, 1):
            det = cliente.get_match_detail(mid)
            if det and det.get("info", {}).get("participants"):
                est, fallos_seguidos = "ok", 0
                ok += 1
            elif cliente.last_request_ok():          # 200 sin participantes
                est = "sin_datos"
            else:                                    # 404, red o key: no se marca, se reintenta al retomar
                fallos_seguidos += 1
                if fallos_seguidos >= 20:
                    sys.exit("[X] 20 fallos seguidos: revisar `python scripts/ops/check_api_keys.py`.")
                continue
            info = (det or {}).get("info", {})
            with conectar() as conn:
                conn.execute("INSERT OR REPLACE INTO partidas VALUES (?, ?, ?, ?, ?, ?)",
                             (mid, est, info.get("gameCreation"), info.get("gameVersion"), int(time.time() * 1000),
                              gzip.compress(json.dumps(det, separators=(",", ":")).encode("utf-8")) if est == "ok" else None))
                conn.commit()
            if i % 500 == 0:
                mins = (time.time() - t0) / 60
                print(f"[{time.strftime('%H:%M:%S')}] {i:,}/{len(pendientes):,}  ok {ok:,}  {ok / mins:.0f} partidas/min (neto)  "
                      f"faltan ~{(len(pendientes) - i) / (i / mins):.0f} min", flush=True)
    except KeyboardInterrupt:
        print("[i] Cortado; al relanzar sigue por donde iba.")
    print(f"listo: {ok:,} partidas nuevas en {(time.time() - t0) / 60:.0f} min")
    estado()


if __name__ == "__main__":
    main()
