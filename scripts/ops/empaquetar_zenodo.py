"""Arma el paquete de datos para Zenodo: la base del crawler, los splits, el LP en vivo y los timelines.

Sin nombres de jugadores. El ID de jugador del proyecto (`player_id` en los CSV, `canonical_id` en la base) es el
Riot ID en minúsculas (`nombre#tag`), así que se reemplaza por un seudónimo `j` + HMAC-SHA256 con una clave secreta
(16 hex), el mismo en todos los archivos: los joins y los modelos siguen funcionando igual. La clave se guarda en
`data/zenodo_clave.txt`, que no se sube; sin ella el seudónimo no se puede revertir ni probar contra un nombre.
`game_name` y `tag_line` quedan vacíos. Los PUUID quedan: Riot los cifra por proyecto.

Cada base se copia con la API de respaldo de SQLite (consistente aunque haya un WAL abierto); los originales no se
tocan. No sube nada: deja en `--salida` los archivos listos para subir a mano y un `MANIFIESTO.json` con tamaño y
SHA-256. Resultados, cachés y pesos de modelos no van: se regeneran con los scripts del repo.

    python scripts/ops/empaquetar_zenodo.py --timelines RUTA/timelines.db
    python scripts/ops/empaquetar_zenodo.py --sin-timelines          # sin la base de timelines (~15 GB)
"""
import argparse
from contextlib import closing
import hashlib
import hmac
import json
import secrets
import shutil
import sqlite3
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
CLAVE = DATA / "zenodo_clave.txt"
COLUMNAS_NOMBRE = ("game_name", "tag_line")
COLUMNAS_ID_CSV = ("player_id", "blue_player_id", "red_player_id")


def clave_secreta(ruta: Path = CLAVE) -> bytes:
    """La clave del HMAC: se crea una vez y se reutiliza, así un paquete nuevo da los mismos seudónimos."""
    if not ruta.exists():
        ruta.write_text(secrets.token_hex(32), encoding="utf-8")
    return bytes.fromhex(ruta.read_text(encoding="utf-8").strip())


def seudonimo(valor, clave: bytes):
    if valor is None or (isinstance(valor, float) and valor != valor):
        return valor
    return "j" + hmac.new(clave, str(valor).encode("utf-8"), hashlib.sha256).hexdigest()[:16]


def copiar_base(origen: Path, destino: Path) -> None:
    """Copia consistente de una base SQLite (respeta el WAL del origen, que se abre solo para leer)."""
    destino.unlink(missing_ok=True)
    with closing(sqlite3.connect(f"file:{origen.as_posix()}?mode=ro", uri=True)) as src:
        with closing(sqlite3.connect(destino)) as dst:
            src.backup(dst)


def _columnas(c, t):
    return [(r[1], bool(r[3])) for r in c.execute(f'PRAGMA table_info("{t}")')]


def _tablas(c):
    return [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")]


def anonimizar_base(base: Path, clave: bytes) -> dict:
    """En la copia: vacía `game_name`/`tag_line` y reemplaza por su seudónimo toda columna de texto que tenga IDs de
    jugador (`nombre#tag`; se detectan por contenido, no por nombre). Devuelve {tabla.columna: filas tocadas}."""
    tocadas = {}
    with closing(sqlite3.connect(base)) as c:      # closing: en Windows `with connect()` no cierra (AGENTS §3)
        c.create_function("seudonimo", 1, lambda v: seudonimo(v, clave), deterministic=True)
        for t in _tablas(c):
            for col, no_nulo in _columnas(c, t):
                if col.lower() in COLUMNAS_NOMBRE:
                    n = c.execute(f'UPDATE "{t}" SET "{col}" = ? WHERE "{col}" IS NOT NULL',
                                  ("" if no_nulo else None,)).rowcount
                    tocadas[f"{t}.{col}"] = n
                elif c.execute(f'SELECT 1 FROM "{t}" WHERE typeof("{col}") = \'text\' AND instr("{col}", \'#\') > 0 '
                               f'AND "{col}" NOT LIKE \'%{{%\' LIMIT 1').fetchone():
                    n = c.execute(f'UPDATE "{t}" SET "{col}" = seudonimo("{col}") WHERE "{col}" IS NOT NULL').rowcount
                    tocadas[f"{t}.{col}"] = n
        c.commit()
        c.execute("VACUUM")
    return tocadas


def quedan_ids(base: Path) -> list:
    """Columnas que todavía tienen un Riot ID (`game_name`/`tag_line` no vacío o un texto con '#'). Debe dar []."""
    out = []
    with closing(sqlite3.connect(f"file:{base.as_posix()}?mode=ro", uri=True)) as c:
        for t in _tablas(c):
            for col, _ in _columnas(c, t):
                if col.lower() in COLUMNAS_NOMBRE:
                    q = f'SELECT 1 FROM "{t}" WHERE "{col}" IS NOT NULL AND "{col}" != \'\' LIMIT 1'
                else:
                    q = (f'SELECT 1 FROM "{t}" WHERE typeof("{col}") = \'text\' AND instr("{col}", \'#\') > 0 '
                         f'AND "{col}" NOT LIKE \'%{{%\' LIMIT 1')
                if c.execute(q).fetchone():
                    out.append(f"{t}.{col}")
    return out


def anonimizar_csv(origen: Path, destino: Path, clave: bytes, filas: int = 500_000) -> int:
    """Copia el CSV con las columnas de ID de jugador seudonimizadas (por bloques, para no cargarlo entero)."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    cab = pd.read_csv(origen, nrows=0).columns
    cols = [c for c in COLUMNAS_ID_CSV if c in cab]
    if not cols:
        shutil.copy2(origen, destino)
        return 0
    cache, n = {}, 0
    for i, bloque in enumerate(pd.read_csv(origen, chunksize=filas, dtype={c: "string" for c in cols},
                                           keep_default_na=False, na_values=[""], low_memory=False)):
        for c in cols:
            bloque[c] = bloque[c].map(lambda v: cache.setdefault(v, seudonimo(v, clave)) if pd.notna(v) else v)
        bloque.to_csv(destino, mode="w" if i == 0 else "a", header=i == 0, index=False)
        n += len(bloque)
    return n


def comprimir(archivos: list, destino: Path, raiz: Path) -> None:
    destino.unlink(missing_ok=True)
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for f in archivos:
            z.write(f, f.relative_to(raiz).as_posix())


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 22), b""):
            h.update(bloque)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--salida", type=Path, default=DATA / "zenodo")
    ap.add_argument("--timelines", type=Path, default=DATA / "timelines_eventos" / "timelines.db",
                    help="timelines.db completa (la del compañero o la del Drive)")
    ap.add_argument("--sin-timelines", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)  # noqa: E731
    if not a.sin_timelines and not a.timelines.exists():
        sys.exit(f"No está {a.timelines}: pasá --timelines RUTA o --sin-timelines.")
    clave = clave_secreta()
    sal, tmp = a.salida, a.salida / "_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    resumen = {}

    def base(origen: Path, nombre: str) -> Path:
        destino = tmp / nombre
        log(f"copiando y anonimizando {origen.name} ...")
        copiar_base(origen, destino)
        resumen[nombre] = anonimizar_base(destino, clave)
        assert not quedan_ids(destino), quedan_ids(destino)
        return destino

    # 1. Base del crawler
    comprimir([base(DATA / "lol_las.db", "lol_las.db")], sal / "lol_las_db.zip", tmp)
    log("lol_las_db.zip listo")

    # 2. Splits (train, test, Diamante, partición y manifiesto)
    sp, archivos = DATA / "splits", []
    for f in sorted(x for x in sp.rglob("*") if x.is_file() and x.name != "desktop.ini"):
        destino = tmp / "splits" / f.relative_to(sp)
        if f.suffix == ".csv":
            n = anonimizar_csv(f, destino, clave)
            log(f"  {f.relative_to(sp)}: {n:,} filas")
        else:
            destino.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, destino)
        archivos.append(destino)
    comprimir(archivos, sal / "splits.zip", tmp)
    log(f"splits.zip listo ({len(archivos)} archivos)")

    # 3. LP en vivo
    lpv = base(DATA / "lp_vivo" / "lp_vivo.db", "lp_vivo.db")
    anonimizar_csv(DATA / "lp_vivo" / "lp_por_partida.csv", tmp / "lp_por_partida.csv", clave)
    comprimir([lpv, tmp / "lp_por_partida.csv"], sal / "lp_vivo.zip", tmp)
    log("lp_vivo.zip listo")

    # 4. Timelines con eventos (el JSON ya está comprimido adentro: se copia sin zip)
    if not a.sin_timelines:
        tl = base(a.timelines, "timelines.db")
        shutil.move(tl, sal / "timelines.db")
        log("timelines.db lista")

    shutil.rmtree(tmp)
    archivos = sorted(f for f in sal.iterdir() if f.is_file() and f.name != "MANIFIESTO.json")
    manifiesto = {"creado": time.strftime("%Y-%m-%d %H:%M:%S"), "filas_anonimizadas": resumen,
                  "archivos": {f.name: {"bytes": f.stat().st_size, "sha256": sha256(f)} for f in archivos}}
    (sal / "MANIFIESTO.json").write_text(json.dumps(manifiesto, indent=1, ensure_ascii=False), encoding="utf-8")
    for f in archivos:
        log(f"  {f.name}: {f.stat().st_size / 1e9:.2f} GB")
    log(f"listo en {sal}")


if __name__ == "__main__":
    main()
