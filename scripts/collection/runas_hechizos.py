#!/usr/bin/env python3
"""Runas, hechizos de invocador y nivel de cuenta por jugador y partida (§9.S.37).

`lol_las.db` no guarda esos campos del payload de match-v5, así que se vuelven a pedir las
partidas y se guardan en una base aparte: `data/runas/runas.db`. No toca `lol_las.db`.
Toda petición pasa por `RiotAPIClient` y su `RiotRateLimiter` (AGENTS.md §3.B).

Orden de descarga: una permutación con semilla fija de las partidas del último bloque del
ajuste y de la validación de `ganador_feats_grande.pkl` (config C, 5 bloques). Cortar y
retomar sigue por donde iba, y subir --n agranda la misma muestra (la muestra chica queda
adentro de la grande). Con --todo sigue después con el resto del train.

Uso:
    python scripts/collection/runas_hechizos.py --n 10000     # ~1 h 45 con 2 keys
    python scripts/collection/runas_hechizos.py --estado      # sin red
"""
import argparse
import pickle
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

RUTA = ROOT / "data" / "runas" / "runas.db"
FEATS = ROOT / "data" / "analysis" / "ganador_feats_grande.pkl"
COLS = ["match_id", "puuid", "participant_id", "team_id", "champion_id", "team_position", "summoner_level",
        "hechizo1", "hechizo2", "estilo_primario", "estilo_secundario", "runa_clave",
        "runa2", "runa3", "runa4", "runa5", "runa6", "stat_ofensa", "stat_flex", "stat_defensa"]
ESQUEMA = f"""
CREATE TABLE IF NOT EXISTS runas ({', '.join(c + (' TEXT' if c in ('match_id', 'puuid', 'team_position') else ' INTEGER') for c in COLS)},
    PRIMARY KEY (match_id, participant_id));
CREATE TABLE IF NOT EXISTS partidas (match_id TEXT PRIMARY KEY, estado TEXT, game_creation INTEGER, game_version TEXT, bajada_en INTEGER);
"""


def conectar():
    RUTA.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(RUTA, timeout=60.0)
    conn.executescript(ESQUEMA)
    return closing(conn)


def filas_de_partida(detalle: dict) -> list:
    """Una tupla por participante, en el orden de COLS."""
    mid = detalle["metadata"]["matchId"]
    filas = []
    for p in detalle["info"]["participants"]:
        perks = p.get("perks") or {}
        estilos = {s.get("description"): s for s in perks.get("styles") or []}
        pri, sec = estilos.get("primaryStyle") or {}, estilos.get("subStyle") or {}
        sel = [s.get("perk") for s in (pri.get("selections") or [])] + [s.get("perk") for s in (sec.get("selections") or [])]
        sel += [None] * (6 - len(sel))
        st = perks.get("statPerks") or {}
        filas.append((mid, p.get("puuid"), p.get("participantId"), p.get("teamId"), p.get("championId"),
                      p.get("teamPosition"), p.get("summonerLevel"), p.get("summoner1Id"), p.get("summoner2Id"),
                      pri.get("style"), sec.get("style"), *sel[:6], st.get("offense"), st.get("flex"), st.get("defense")))
    return filas


def orden_de_descarga(todo: bool) -> list:
    d = pickle.load(open(FEATS, "rb"))
    mids, t, val, fit = d["mids"], d["t"], d["val"], d["masks"]["C"]
    del d
    idx = np.where(fit)[0]
    bloque = np.searchsorted(np.quantile(t[idx], np.linspace(0, 1, 6)[1:-1]), t[idx], side="right")
    primero = np.r_[idx[bloque == 4], np.where(val)[0]]
    rng = np.random.default_rng(20261114)
    orden = list(mids[rng.permutation(primero)])
    if todo:
        resto = np.setdiff1d(np.arange(len(mids)), primero)
        orden += list(mids[rng.permutation(resto)])
    return orden


def estado() -> None:
    with conectar() as conn:
        for est, n in conn.execute("SELECT estado, COUNT(*) FROM partidas GROUP BY estado"):
            print(f"partidas {est}: {n:,}")
        print(f"filas de jugadores: {conn.execute('SELECT COUNT(*) FROM runas').fetchone()[0]:,}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=10000, help="partidas de la muestra (acumulado, no adicional)")
    ap.add_argument("--todo", action="store_true", help="después de la muestra, seguir con el resto del train")
    ap.add_argument("--estado", action="store_true")
    args = ap.parse_args()
    if args.estado:
        return estado()

    from src.riot_client import RiotAPIClient
    orden = orden_de_descarga(args.todo)[: args.n]
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
                filas, est, fallos_seguidos = filas_de_partida(det), "ok", 0
                ok += 1
            elif cliente.last_request_ok():          # 200 sin participantes
                filas, est = [], "sin_datos"
            else:                                    # 404, red o key: no se marca, se reintenta al retomar
                fallos_seguidos += 1
                if fallos_seguidos >= 20:
                    sys.exit("[X] 20 fallos seguidos: revisar `python scripts/ops/check_api_keys.py`.")
                continue
            with conectar() as conn:
                conn.executemany(f"INSERT OR REPLACE INTO runas VALUES ({', '.join('?' * len(COLS))})", filas)
                info = (det or {}).get("info", {})
                conn.execute("INSERT OR REPLACE INTO partidas VALUES (?, ?, ?, ?, ?)",
                             (mid, est, info.get("gameCreation"), info.get("gameVersion"), int(time.time() * 1000)))
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
