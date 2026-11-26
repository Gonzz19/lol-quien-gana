"""
Piloto: timelines completos de Match-V5 (eventos + frames por minuto) para una muestra (pedido del usuario,
2026-11-17; TAREAS "Volver a bajar los timelines con eventos y oro sin gastar").

El crawler guardó del timeline solo `totalGold`, XP, nivel, CS, daño y K/D/A a los 10/15/20. Acá se guarda el
**JSON crudo completo** (comprimido) para extraer después objetivos (dragones, larvas, Heraldo, torres, placas,
inhibidores), oro sin gastar → oro invertido, quién está muerto en el minuto, recompensas, primera sangre y
primera torre con su minuto, ítems, wards y posiciones, sin volver a pedir nada a la API.

* Muestra: `--n-train` partidas Master+ del ajuste (chunks 1-6 de `ingame_cache.npz`, anteriores al 16/09) y
  `--n-v` de V Master+, al azar con semilla fija, intercaladas (una descarga parcial ya sirve para las dos).
* Base aparte `data/timelines_eventos/timelines.db`: no toca `lol_las.db` ni los CSV (AGENTS.md §3.G). Toda
  apertura pasa por un context manager que cierra la conexión (§3.A).
* Toda petición pasa por `RiotAPIClient` → `RiotRateLimiter` del pool (§3.B); 429 con `Retry-After`.
* Retoma: las partidas ya guardadas se saltean. Ctrl+C corta ordenado.

    python scripts/collection/timelines_eventos.py --n-train 8000 --n-v 8000     # ~3 h con 2 keys
    python scripts/collection/timelines_eventos.py --resumen                      # estado, sin red
"""
import argparse
import json
import os
import sqlite3
import sys
import threading
import time
import zlib
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import closing, contextmanager
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.config import DATA_DIR  # noqa: E402

RUTA_DB = DATA_DIR / "timelines_eventos" / "timelines.db"
# Latido mientras se descarga: backup_to_drive.py no sube la base si el archivo se tocó hace < 5 min.
LATIDO = RUTA_DB.parent / "descargando.lock"
CACHE_INGAME = DATA_DIR / "analysis" / "ingame" / "ingame_cache.npz"


class Almacen:
    def __init__(self, ruta: Path = RUTA_DB):
        self.ruta = ruta
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS seleccion (
                             match_id TEXT PRIMARY KEY, grupo TEXT NOT NULL, orden INTEGER NOT NULL)""")
            c.execute("""CREATE TABLE IF NOT EXISTS timeline_crudo (
                             match_id TEXT PRIMARY KEY, json_zlib BLOB NOT NULL, bytes_json INTEGER NOT NULL,
                             bajado_en REAL NOT NULL)""")
            c.execute("CREATE TABLE IF NOT EXISTS sin_timeline (match_id TEXT PRIMARY KEY, motivo TEXT)")
            # Lotes pedidos con --ids: una partida puede estar en varios; no pisa el grupo de `seleccion`.
            c.execute("""CREATE TABLE IF NOT EXISTS lote (
                             grupo TEXT NOT NULL, match_id TEXT NOT NULL, PRIMARY KEY (grupo, match_id))""")
            c.commit()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.ruta, timeout=60)
        try:
            yield conn
        finally:
            conn.close()

    def guardar_seleccion(self, filas):
        with self._conn() as c:
            c.executemany("INSERT OR IGNORE INTO seleccion VALUES (?, ?, ?)", filas)
            c.commit()

    def pendientes(self):
        with self._conn() as c:
            return [r[0] for r in c.execute(
                """SELECT s.match_id FROM seleccion s
                   WHERE s.match_id NOT IN (SELECT match_id FROM timeline_crudo)
                     AND s.match_id NOT IN (SELECT match_id FROM sin_timeline)
                   ORDER BY s.orden""")]

    def registrar_lote(self, grupo: str, ids: list):
        """Anota el lote y suma a `seleccion` las que no estaban (con `orden` después de lo existente).
        Devuelve las pendientes del lote en el orden del archivo: solo esas se bajan."""
        with self._conn() as c:
            c.executemany("INSERT OR IGNORE INTO lote VALUES (?, ?)", [(grupo, m) for m in ids])
            desde = c.execute("SELECT COALESCE(MAX(orden), -1) + 1 FROM seleccion").fetchone()[0]
            c.executemany("INSERT OR IGNORE INTO seleccion VALUES (?, ?, ?)",
                          [(m, grupo, desde + k) for k, m in enumerate(ids)])
            c.commit()
            hechas = {r[0] for r in c.execute("SELECT match_id FROM timeline_crudo")}
            hechas |= {r[0] for r in c.execute("SELECT match_id FROM sin_timeline")}
        return [m for m in ids if m not in hechas]

    def resumen_lotes(self):
        with self._conn() as c:
            return c.execute(
                """SELECT l.grupo, COUNT(*), SUM(t.match_id IS NOT NULL), SUM(s.match_id IS NOT NULL)
                   FROM lote l LEFT JOIN (SELECT match_id FROM timeline_crudo) t USING (match_id)
                               LEFT JOIN sin_timeline s USING (match_id)
                   GROUP BY l.grupo ORDER BY MIN(l.rowid)""").fetchall()

    def guardar(self, filas_ok, filas_sin):
        with self._conn() as c:
            c.executemany("INSERT OR IGNORE INTO timeline_crudo VALUES (?, ?, ?, ?)", filas_ok)
            c.executemany("INSERT OR IGNORE INTO sin_timeline VALUES (?, ?)", filas_sin)
            c.commit()

    def resumen(self):
        with self._conn() as c:
            sel = dict(c.execute("SELECT grupo, COUNT(*) FROM seleccion GROUP BY grupo").fetchall())
            ok = dict(c.execute("""SELECT s.grupo, COUNT(*) FROM timeline_crudo t JOIN seleccion s USING (match_id)
                                   GROUP BY s.grupo""").fetchall())
            sin = c.execute("SELECT COUNT(*) FROM sin_timeline").fetchone()[0]
            mb = c.execute("SELECT COALESCE(SUM(LENGTH(json_zlib)), 0) / 1e6 FROM timeline_crudo").fetchone()[0]
        return sel, ok, sin, mb


def seleccionar(n_train: int, n_v: int, semilla: int = 0):
    C = np.load(CACHE_INGAME, allow_pickle=True)
    mids = C["mids"].astype(str)
    chunk, master, val = C["chunk"], C["master"].astype(bool), C["val"].astype(bool)
    rng = np.random.default_rng(semilla)
    tr = rng.permutation(np.flatnonzero(np.isin(chunk, [1, 2, 3, 4, 5, 6]) & master))[:n_train]
    v = rng.permutation(np.flatnonzero(val))[:n_v]
    filas = [(mids[i], "train") for i in tr] + [(mids[i], "V") for i in v]
    orden = rng.permutation(len(filas))                    # intercaladas
    return [(filas[j][0], filas[j][1], int(k)) for k, j in enumerate(orden)]


def seleccionar_todas(ya: set, desde: int, semilla: int = 1):
    """Todas las partidas de `lol_las.db` que falten, por prioridad: V y Master+ del ajuste (lo que usa el
    modelo en vivo), después el resto de la caché en vivo y al final el resto de la base. Al azar dentro de
    cada grupo; `orden` sigue después de la selección que ya había."""
    C = np.load(CACHE_INGAME, allow_pickle=True)
    mids = C["mids"].astype(str)
    chunk, master, val = C["chunk"], C["master"].astype(bool), C["val"].astype(bool)
    prio = np.isin(chunk, [1, 2, 3, 4, 5, 6]) & master
    # solo lectura y cerrada al salir (AGENTS.md §3.A: el context manager de sqlite3 no cierra la conexión)
    with closing(sqlite3.connect(f"file:{(DATA_DIR / 'lol_las.db').as_posix()}?mode=ro", uri=True)) as c_ro:
        todas = [r[0] for r in c_ro.execute("SELECT match_id FROM matches")]
    en_cache = set(mids)
    rng = np.random.default_rng(semilla)
    grupos = [("V", list(mids[val])), ("train", list(mids[prio & ~val])),
              ("cache", list(mids[~prio & ~val])), ("resto", [m for m in todas if m not in en_cache])]
    filas, k = [], desde
    for g, lista in grupos:
        for m in rng.permutation(np.array([m for m in lista if m not in ya], dtype=object)):
            filas.append((str(m), g, k))
            k += 1
    return filas


def mostrar_resumen(alm: Almacen):
    sel, ok, sin, mb = alm.resumen()
    for g in sorted(sel):
        print(f"  {g:6s}: {ok.get(g, 0):,} / {sel[g]:,} bajadas")
    print(f"  sin timeline (404 u otro): {sin:,}; tamaño comprimido {mb:,.0f} MB")
    lotes = alm.resumen_lotes()
    if lotes:
        print("  lotes (--ids):")
        for g, n, ok_n, sin_n in lotes:
            print(f"    {g:12s}: {ok_n or 0:,} / {n:,} bajadas, sin timeline {sin_n or 0}, "
                  f"faltan {n - (ok_n or 0) - (sin_n or 0):,}")


def leer_ids(ruta: Path) -> list:
    """Un match_id por línea; ignora vacías y duplicadas, conserva el orden."""
    vistos, ids = set(), []
    for linea in Path(ruta).read_text(encoding="utf-8").splitlines():
        m = linea.strip()
        if m and m not in vistos:
            vistos.add(m)
            ids.append(m)
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-v", type=int, default=8000)
    ap.add_argument("--hilos", type=int, default=4)
    ap.add_argument("--resumen", action="store_true")
    ap.add_argument("--ids", type=Path, help="archivo con un match_id por línea: se agregan a la selección "
                    "(no necesita ingame_cache.npz); por ejemplo, Diamond-V para la réplica de los eventos")
    ap.add_argument("--grupo", default="ids", help="nombre del grupo para las partidas de --ids")
    ap.add_argument("--db", type=Path, default=RUTA_DB, help="timelines.db donde se guarda (por defecto data/)")
    ap.add_argument("--todas", action="store_true",
                    help="suma a la selección todas las partidas de lol_las.db (~273.000; ~18 GB, ~50 h con 2 keys)")
    args = ap.parse_args()
    global LATIDO
    alm = Almacen(args.db)
    LATIDO = args.db.parent / "descargando.lock"
    if args.resumen:
        mostrar_resumen(alm)
        return
    if args.ids:
        with alm._conn() as c:
            ya = {r[0] for r in c.execute("SELECT match_id FROM seleccion")}
            desde = c.execute("SELECT COALESCE(MAX(orden), -1) + 1 FROM seleccion").fetchone()[0]
        lista = [m.strip() for m in args.ids.read_text(encoding="utf-8").splitlines() if m.strip()]
        nuevas = [(m, args.grupo, desde + k) for k, m in enumerate(m for m in lista if m not in ya)]
        print(f"[*] --ids: {len(nuevas):,} partidas nuevas en la selección (grupo {args.grupo})", flush=True)
        alm.guardar_seleccion(nuevas)
    elif args.todas:
        with alm._conn() as c:
            ya = {r[0] for r in c.execute("SELECT match_id FROM seleccion")}
            desde = c.execute("SELECT COALESCE(MAX(orden), -1) + 1 FROM seleccion").fetchone()[0]
        nuevas = seleccionar_todas(ya, desde)
        print(f"[*] --todas: {len(nuevas):,} partidas nuevas en la selección", flush=True)
        alm.guardar_seleccion(nuevas)
    else:
        alm.guardar_seleccion(seleccionar(args.n_train, args.n_v))
    pend = alm.pendientes()
    if args.ids:                                            # solo las de --ids, no lo que quedaba de --todas
        pedidas = set(lista)
        pend = [m for m in pend if m in pedidas]
    print(f"[*] pendientes: {len(pend):,} (base {args.db})", flush=True)
    if not pend:
        mostrar_resumen(alm)
        return

    from src.riot_client import RiotAPIClient
    cliente = RiotAPIClient()                               # pool del .env, un RiotRateLimiter por key
    print(f"[*] keys activas: {cliente.num_active_keys}", flush=True)
    parar = threading.Event()

    def bajar(mid):
        if parar.is_set():
            return mid, None, "cortado"
        tl = cliente.get_match_timeline(mid)
        if tl is None:
            return mid, None, "sin respuesta (404 o error)"
        crudo = json.dumps(tl, separators=(",", ":")).encode()
        return mid, (zlib.compress(crudo, 6), len(crudo)), None

    def latir():
        try:
            LATIDO.write_text(f"{os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}", encoding="utf-8")
        except OSError:
            pass

    latir()
    t0, hechos, sin_n, prox_informe = time.time(), 0, 0, 50
    buf_ok, buf_sin = [], []
    it = iter(pend)
    try:
        with ThreadPoolExecutor(args.hilos) as ex:
            vivos = {ex.submit(bajar, m) for _, m in zip(range(args.hilos * 2), it)}
            while vivos:
                listos, vivos = wait(vivos, return_when=FIRST_COMPLETED)
                for f in listos:
                    mid, dato, motivo = f.result()
                    if dato is not None:
                        buf_ok.append((mid, dato[0], dato[1], time.time()))
                        hechos += 1
                    elif motivo != "cortado":
                        buf_sin.append((mid, motivo))
                        sin_n += 1
                    if not parar.is_set():
                        siguiente = next(it, None)
                        if siguiente is not None:
                            vivos.add(ex.submit(bajar, siguiente))
                if len(buf_ok) + len(buf_sin) >= 25:
                    alm.guardar(buf_ok, buf_sin)
                    buf_ok, buf_sin = [], []
                    latir()
                if hechos >= prox_informe:
                    prox_informe += 250
                    h = (time.time() - t0) / 3600
                    tel = cliente.telemetry
                    resto = len(pend) - hechos - sin_n
                    print(f"[{time.strftime('%H:%M:%S')}] {hechos:,} bajadas ({hechos / h:,.0f}/h neto), "
                          f"sin timeline {sin_n}, faltan {resto:,} (~{resto / max(hechos / h, 1):.1f} h) | "
                          f"req {tel['total_requests']:,}, 200 {tel['status_200']:,}, 429 {tel['status_429']}, "
                          f"errores {tel['http_errors']}, timeouts {tel['timeouts']}", flush=True)
                if cliente.pool_exhausted.is_set():
                    print("[X] no quedan keys activas: corto", flush=True)
                    parar.set()
    except KeyboardInterrupt:
        print("\n[!] Ctrl+C: guardo lo bajado y corto", flush=True)
        parar.set()
    finally:
        alm.guardar(buf_ok, buf_sin)
        try:
            LATIDO.unlink()
        except OSError:
            pass
    h = (time.time() - t0) / 3600
    print(f"[+] sesión: {hechos:,} timelines en {h * 60:.0f} min ({hechos / max(h, 1e-9):,.0f}/h neto)", flush=True)
    mostrar_resumen(alm)


if __name__ == "__main__":
    main()
