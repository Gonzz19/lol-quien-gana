#!/usr/bin/env python3
"""
scripts/backup_to_drive.py
==========================
Genera un snapshot atómico y seguro de la base de datos SQLite (data/lol_las.db)
y lo sincroniza a la carpeta de Google Drive para escritorio, junto a los 4 CSVs.

Política de nube (ver PROJECT_SUMMARY.md §4.W):
  1. Un solo escritor: solo la máquina recolectora sube; el resto descarga (RO).
  2. `lol_las.db` = latest (overwrite) + `snapshots/lol_las_YYYYMMDD_HHMM_m<M>.db`
     versionados inmutables (nuevo si pasaron >= 7 días o >= 10k partidas nuevas).
  3. `PRAGMA quick_check` antes de subir: si falla, NO se sube nada.
  4. Manifiesto JSON con conteos junto a cada snapshot (provenance).
  5. `data/timelines_eventos/timelines.db` (timelines completos, ~18 GB) va a `timelines_eventos/` solo si
     creció respecto de su manifiesto, sin snapshot local y nunca con la descarga corriendo.

Uso:
    python scripts/backup/backup_to_drive.py [--dry-run]

    También lo invoca `run.py --respaldar` al cerrar una sesión de recolección,
    y solo si esa corrida guardó algo nuevo. NO hay ningún hook de git: la BD
    está en `.gitignore`, así que un `git push` nunca toca la nube (versiones
    anteriores de este docstring afirmaban lo contrario; verificado el
    2026-09-07, `.git/hooks/` no tiene nada instalado).
"""

import argparse
import json
import os
import re
import shutil
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Cargar variables de entorno (.env)
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "lol_las.db"
LOCAL_BACKUPS_DIR = PROJECT_ROOT / "data" / "backups"
DRIVE_PATH_ENV = os.getenv("GOOGLE_DRIVE_BACKUP_PATH", "").strip().strip('"').strip("'")

VERSIONED_EVERY_DAYS = int(os.getenv("BACKUP_VERSIONED_DAYS", "7"))
VERSIONED_EVERY_MATCHES = int(os.getenv("BACKUP_VERSIONED_MATCHES", "10000"))
VERSIONED_KEEP = int(os.getenv("BACKUP_VERSIONED_KEEP", "1"))

SNAPSHOT_RE = re.compile(r"^lol_las_(\d{8}_\d{4})_m(\d+)\.db$")


# ---------------------------------------------------------------------------
# Helpers puros / testeables
# ---------------------------------------------------------------------------

def check_integrity(db_path: Path) -> Tuple[bool, str]:
    """PRAGMA quick_check en modo RO. Retorna (ok, mensaje)."""
    try:
        conn = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True, timeout=60.0)
        try:
            row = conn.execute("PRAGMA quick_check;").fetchone()
            msg = str(row[0]) if row else "empty"
            return (msg.lower() == "ok", msg)
        finally:
            conn.close()
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def table_count(conn: sqlite3.Connection, table: str) -> int:
    try:
        row = conn.execute(f"SELECT COUNT(*) FROM {table};").fetchone()
        return int(row[0]) if row else 0
    except sqlite3.OperationalError:
        return 0  # tabla aún no creada (BD vieja) -> se reporta 0


def collect_manifest(db_path: Path) -> Dict[str, Any]:
    """Conteos de provenance para el manifiesto (lectura RO).

    NOTA (2026-09-04, auditoría Muse): se probó derivar
    participants = matches * 10 para ahorrar el COUNT(*) full-scan (~15s), pero
    la BD viva demuestra que el invariante 10/10 NO es exacto: hay 27 partidas
    con 20 filas (doble ingesta multi-proyecto, mismo participant_id con 2 PUUIDs
    distintos: 716,290 exactos vs 716,020 derivados). El manifiesto debe reportar
    el conteo EXACTO para no ocultar esa anomalía. El costo (~15s) cae solo sobre
    backup/pre-push, nunca sobre el hot path del crawler.
    """
    conn = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True, timeout=60.0)
    try:
        matches_cnt = table_count(conn, "matches")
        manifest = {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "matches": matches_cnt,
            "participants": table_count(conn, "match_participants"),
            "players": table_count(conn, "players"),
            "discarded": table_count(conn, "discarded_matches"),
            "map_rows": table_count(conn, "player_puuid_map"),
            "identities": table_count(conn, "player_identities"),
            "aliases": table_count(conn, "player_puuid_aliases"),
            "riot_ids": table_count(conn, "puuid_riot_ids"),
            "projects": table_count(conn, "api_projects"),
            "db_size_mb": round(Path(db_path).stat().st_size / (1024 * 1024), 2),
        }
        try:
            row = conn.execute(
                "SELECT MIN(game_version), MAX(game_version) FROM matches;").fetchone()
            manifest["patch_min"] = row[0] if row else None
            manifest["patch_max"] = row[1] if row else None
        except sqlite3.OperationalError:
            pass
        return manifest
    finally:
        conn.close()


MONOTONIC_MANIFEST_KEYS: Tuple[str, ...] = (
    "matches",
    "participants",
    "identities",
    "map_rows",
    "aliases",
    "riot_ids",
)


def check_no_regression(
    local_manifest: Dict[str, Any],
    remote_manifest: Optional[Dict[str, Any]],
    keys: Tuple[str, ...] = MONOTONIC_MANIFEST_KEYS,
) -> Tuple[bool, List[str]]:
    """Verifica que la BD local no retroceda frente al manifiesto remoto en Drive.

    Compara las tablas monótonas crecientes (`matches`, `participants`,
    `identities`, `map_rows`, `aliases`, `riot_ids`). Si alguna tiene menos filas
    en local que en Drive, retorna `(False, detalles)` para abortar la subida y
    evitar pisar un snapshot más avanzado con una BD local desactualizada.
    """
    if not remote_manifest:
        return True, []
    regressions: List[str] = []
    for k in keys:
        if k not in remote_manifest or remote_manifest[k] is None:
            continue
        local_val = int(local_manifest.get(k, 0) or 0)
        remote_val = int(remote_manifest.get(k, 0) or 0)
        if local_val < remote_val:
            regressions.append(f"{k}: local={local_val:,} < drive={remote_val:,}")
    return (len(regressions) == 0, regressions)


def snapshot_name(manifest: Dict[str, Any],
                  when: Optional[datetime] = None) -> str:
    when = when or datetime.now()
    return f"lol_las_{when.strftime('%Y%m%d_%H%M')}_m{int(manifest.get('matches', 0))}.db"


def parse_snapshot_name(name: str) -> Optional[Tuple[datetime, int]]:
    m = SNAPSHOT_RE.match(name or "")
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d_%H%M"), int(m.group(2))
    except ValueError:
        return None


def list_versioned(snaps_dir: Path) -> List[Tuple[str, datetime, int]]:
    """Snapshots versionados ordenados (más viejo primero)."""
    out = []
    if snaps_dir.is_dir():
        for f in snaps_dir.iterdir():
            parsed = parse_snapshot_name(f.name)
            if parsed and f.is_file():
                out.append((f.name, parsed[0], parsed[1]))
    out.sort(key=lambda t: (t[1], t[0]))
    return out


def should_create_versioned(existing: List[Tuple[str, datetime, int]],
                             manifest: Dict[str, Any],
                             now: Optional[datetime] = None,
                             every_days: int = VERSIONED_EVERY_DAYS,
                             every_matches: int = VERSIONED_EVERY_MATCHES) -> bool:
    """Nuevo versionado si no hay ninguno, si el último es viejo, o si hubo salto de matches."""
    now = now or datetime.now()
    if not existing:
        return True
    _, last_dt, last_matches = existing[-1]
    if now - last_dt >= timedelta(days=every_days):
        return True
    if int(manifest.get("matches", 0)) - last_matches >= every_matches:
        return True
    return False


def prune_versioned(snaps_dir: Path, keep: int = VERSIONED_KEEP) -> List[str]:
    """Conserva los `keep` más recientes (DB + manifiesto). Retorna eliminados."""
    existing = list_versioned(snaps_dir)
    deleted = []
    for name, _, _ in existing[:-max(keep, 1)] if len(existing) > keep else []:
        for suffix in ("", ".manifest.json"):
            target = snaps_dir / f"{name}{suffix}" if suffix else snaps_dir / name
            try:
                if target.exists():
                    target.unlink()
            except OSError:
                pass
        deleted.append(name)
    # Manifiestos huérfanos (sin .db) también se limpian
    if snaps_dir.is_dir():
        db_names = {n for n, _, _ in list_versioned(snaps_dir)}
        for f in snaps_dir.iterdir():
            if f.name.endswith(".manifest.json") and f.name[:-len(".manifest.json")] not in db_names:
                try:
                    f.unlink()
                    deleted.append(f.name)
                except OSError:
                    pass
    return deleted


def safe_drive_copy(src: Path, dst: Path, max_retries: int = 5, delay: float = 1.0) -> None:
    """
    Copia un archivo hacia Google Drive de forma atómica y segura:
    1. Escribe primero a un archivo temporal intermedio (.uploading.tmp) en el mismo directorio.
    2. Hace flush y fsync para asegurar persistencia física.
    3. Verifica que el tamaño copiado coincida exactamente con el archivo fuente.
    4. Reemplaza atómicamente el archivo destino (os.replace), reintentando si
       el filesystem virtual de Google Drive tiene un lock temporal.
    5. Garantiza que el archivo destino NUNCA quede truncado en 0 bytes ni corrupto.
    6. Limpia cualquier residuo temporal (.tmp) si ocurre un error.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_dst = dst.with_name(f".{dst.stem}_upload_{os.getpid()}_{int(time.time())}.tmp")
    src_size = src.stat().st_size

    try:
        with open(src, "rb") as fsrc, open(tmp_dst, "wb") as fdst:
            shutil.copyfileobj(fsrc, fdst, length=8 * 1024 * 1024)
            fdst.flush()
            try:
                os.fsync(fdst.fileno())
            except OSError:
                pass  # Filesystems virtuales de Drive a veces no soportan fsync

        # Verificar tamaño completo antes de reemplazar
        tmp_size = tmp_dst.stat().st_size
        if tmp_size != src_size:
            raise IOError(
                f"Tamaño incompleto al copiar '{src.name}': esperado {src_size} bytes, obtenido {tmp_size} bytes"
            )

        # Reemplazar atómicamente el destino con reintentos para tolerar locks de Drive Desktop
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                os.replace(tmp_dst, dst)
                return
            except OSError as e:
                last_err = e
                # Si dst está bloqueado por el indexador de Drive Desktop, intentar unlink previo
                if dst.exists():
                    try:
                        dst.unlink()
                    except OSError:
                        pass
                if attempt < max_retries:
                    time.sleep(delay)

        raise OSError(
            f"No se pudo reemplazar '{dst.name}' en Google Drive tras {max_retries} intentos: {last_err}"
        )
    finally:
        if tmp_dst.exists():
            try:
                tmp_dst.unlink()
            except OSError:
                pass



# ---------------------------------------------------------------------------
# Snapshot local + subida
# ---------------------------------------------------------------------------

def create_sqlite_snapshot(source_db: Path, target_db: Path) -> bool:
    if not source_db.exists():
        print(f"[X] Error: La base de datos fuente no existe en '{source_db}'.")
        return False

    target_db.parent.mkdir(parents=True, exist_ok=True)

    src = None
    dst = None
    try:
        src = sqlite3.connect(source_db)
        dst = sqlite3.connect(target_db)
        src.backup(dst)
        return True
    except Exception as e:
        print(f"[X] Error durante el snapshot de SQLite: {e}")
        return False
    finally:
        if dst is not None:
            dst.close()
        if src is not None:
            src.close()


CSV_FILES = [
    "full_dataset.csv",
    "players.csv",
    "matchups_1v1.csv",
    "champions_by_lane.csv"
]

# Timelines completos con eventos (base aparte, `scripts/collection/timelines_eventos.py`; PROJECT_SUMMARY §9.S.64).
# Llega a ~18 GB: se sube solo si creció, sin snapshot local intermedio (no entra en disco) y nunca con la
# descarga corriendo (el descargador renueva un latido en `descargando.lock` cada vez que guarda).
TIMELINES_SUBDIR = "timelines_eventos"
TIMELINES_DB_NAME = "timelines.db"
TIMELINES_LOCK_NAME = "descargando.lock"
TIMELINES_LOCK_MAX_AGE_S = 300


def backup_timelines(drive_target: Path, src_db: Path, dry_run: bool = False,
                     allow_regression: bool = False, now: Optional[float] = None) -> Dict[str, Any]:
    """Sube `timelines.db` a `<drive>/timelines_eventos/` si hay timelines nuevos.

    Estados posibles en `estado`: "no_existe", "descarga_en_curso", "sin_cambios", "regresion", "integridad",
    "dry_run", "subida".
    """
    rep: Dict[str, Any] = {"estado": "no_existe", "timelines": None}
    if not src_db.exists():
        return rep
    lock = src_db.parent / TIMELINES_LOCK_NAME
    now = time.time() if now is None else now
    if lock.exists() and now - lock.stat().st_mtime < TIMELINES_LOCK_MAX_AGE_S:
        rep["estado"] = "descarga_en_curso"
        return rep

    conn = sqlite3.connect(f"file:{src_db.resolve()}?mode=ro", uri=True, timeout=60.0)
    try:
        local_n = table_count(conn, "timeline_crudo")
        sin_n = table_count(conn, "sin_timeline")
    finally:
        conn.close()
    rep["timelines"] = local_n

    dest_dir = drive_target / TIMELINES_SUBDIR
    dest_db = dest_dir / TIMELINES_DB_NAME
    man_path = dest_dir / "timelines.manifest.json"
    remote_n = None
    if man_path.exists():
        try:
            remote_n = int(json.loads(man_path.read_text(encoding="utf-8")).get("timelines"))
        except Exception:
            remote_n = None
    if remote_n is not None and local_n < remote_n and not allow_regression:
        rep["estado"] = "regresion"
        rep["remoto"] = remote_n
        return rep
    if remote_n is not None and local_n == remote_n and dest_db.exists() \
            and dest_db.stat().st_size == src_db.stat().st_size:
        rep["estado"] = "sin_cambios"
        return rep
    if dry_run:
        rep["estado"] = "dry_run"
        return rep

    ok, msg = check_integrity(src_db)
    if not ok:
        rep["estado"] = "integridad"
        rep["detalle"] = msg
        return rep

    # Lectura abierta durante toda la copia: con el lock SHARED nadie puede confirmar escrituras a mitad de la
    # copia (ya se verificó que el descargador no corre).
    conn = sqlite3.connect(f"file:{src_db.resolve()}?mode=ro", uri=True, timeout=60.0)
    try:
        conn.execute("BEGIN")
        conn.execute("SELECT COUNT(*) FROM sin_timeline").fetchone()
        safe_drive_copy(src_db, dest_db)
        conn.rollback()
    finally:
        conn.close()
    manifest = {"created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "timelines": local_n,
                "sin_timeline": sin_n, "db_size_mb": round(src_db.stat().st_size / (1024 * 1024), 2)}
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    rep["estado"] = "subida"
    return rep


def _load_remote_manifest(drive_target: Path) -> Optional[Dict[str, Any]]:
    """Carga `lol_las.manifest.json` de Drive y completa `aliases`/`riot_ids` si el manifiesto era previo."""
    manifest_path = drive_target / "lol_las.manifest.json"
    dest_db = drive_target / "lol_las.db"
    remote: Dict[str, Any] = {}
    if manifest_path.exists():
        try:
            remote = json.loads(manifest_path.read_text(encoding="utf-8")) or {}
        except Exception:
            remote = {}
    if dest_db.exists() and (not remote or "aliases" not in remote or "riot_ids" not in remote):
        try:
            conn = sqlite3.connect(f"file:{dest_db.resolve()}?mode=ro", uri=True, timeout=30.0)
            try:
                if not remote:
                    for k, tbl in (
                        ("matches", "matches"),
                        ("identities", "player_identities"),
                        ("map_rows", "player_puuid_map"),
                        ("aliases", "player_puuid_aliases"),
                        ("riot_ids", "puuid_riot_ids"),
                    ):
                        remote[k] = table_count(conn, tbl)
                else:
                    remote.setdefault("aliases", table_count(conn, "player_puuid_aliases"))
                    remote.setdefault("riot_ids", table_count(conn, "puuid_riot_ids"))
            finally:
                conn.close()
        except Exception:
            pass
    return remote or None


def run_backup(db_path: Path = DB_PATH,
               drive_path: Optional[Path] = None,
               local_backups_dir: Path = LOCAL_BACKUPS_DIR,
               dry_run: bool = False,
               allow_regression: bool = False) -> Dict[str, Any]:
    """Ejecuta el backup completo. Retorna reporte con lo realizado."""
    report: Dict[str, Any] = {"integrity_ok": False, "uploaded_latest": False,
                              "regression_blocked": False, "regression_reasons": [],
                              "versioned": None, "pruned": [], "csvs": []}
    if not db_path.exists():
        print(f"[!] No se encontró '{db_path}'. Nada para respaldar.")
        return report

    ok, msg = check_integrity(db_path)
    report["integrity_ok"] = ok
    report["integrity_msg"] = msg
    if not ok:
        print(f"[X] Integridad comprometida (quick_check: {msg}). NO se sube nada a la nube.")
        return report
    print(f"[+] Integridad verificada (quick_check=ok).")

    manifest = collect_manifest(db_path)
    report["manifest"] = manifest
    print(f"[*] Manifiesto: {manifest['matches']:,} partidas, "
          f"{manifest['players']:,} jugadores, identidades={manifest['identities']:,}, "
          f"mapa={manifest['map_rows']:,}, alias={manifest['aliases']:,}, riot_ids={manifest['riot_ids']:,}.")

    drive_target = Path(drive_path) if drive_path else (
        Path(DRIVE_PATH_ENV) if DRIVE_PATH_ENV else None)

    if drive_target and drive_target.exists() and drive_target.is_dir():
        remote_manifest = _load_remote_manifest(drive_target)
        reg_ok, reg_reasons = check_no_regression(manifest, remote_manifest)
        if not reg_ok:
            report["regression_reasons"] = reg_reasons
            if not allow_regression:
                report["regression_blocked"] = True
                print(f"[X] REGRESIÓN DETECTADA respecto a Google Drive ({'; '.join(reg_reasons)}).")
                print("    Se aborta el backup para no sobrescribir una BD remota más avanzada.")
                print("    Ejecuta 'python scripts/backup/restore_from_drive.py' primero, o usa --force si es intencional.")
                return report
            print(f"[!] Advertencia: regresión detectada ({'; '.join(reg_reasons)}), continuando por --force.")

    local_backups_dir.mkdir(parents=True, exist_ok=True)
    temp_snapshot_db = local_backups_dir / "temp_snapshot.db"
    print(f"[*] 1. Generando snapshot atómico de SQLite ({manifest['db_size_mb']:.2f} MB)...")
    if dry_run:
        print("[dry-run] Se omitiría snapshot y subida.")
        return report
    if not create_sqlite_snapshot(db_path, temp_snapshot_db):
        return report

    try:
        if drive_target and drive_target.exists() and drive_target.is_dir():
            print("[*] 2. Sincronizando base de datos y CSVs a Google Drive...")
            dest_db = drive_target / "lol_las.db"
            safe_drive_copy(temp_snapshot_db, dest_db)
            report["uploaded_latest"] = True
            print(f"[+] Base de datos sincronizada: '{dest_db.name}' ({manifest['db_size_mb']:.2f} MB)")
            with open(drive_target / "lol_las.manifest.json", "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)

            data_dir = Path(db_path).parent
            for csv_name in CSV_FILES:
                local_csv = data_dir / csv_name
                if local_csv.exists():
                    safe_drive_copy(local_csv, drive_target / csv_name)
                    report["csvs"].append(csv_name)
                    print(f"[+] Dataset CSV sincronizado: '{csv_name}'")
                else:
                    print(f"[-] Advertencia: '{csv_name}' no existe en data/.")

            tl = backup_timelines(drive_target, data_dir / TIMELINES_SUBDIR / TIMELINES_DB_NAME,
                                  allow_regression=allow_regression)
            report["timelines"] = tl
            n_tl = f"{tl['timelines']:,}" if tl.get("timelines") is not None else "-"
            print({"no_existe": "[-] Timelines con eventos: no hay base local.",
                   "descarga_en_curso": "[!] Timelines con eventos: la descarga está corriendo; no se sube "
                                        "(se respalda al cortarla).",
                   "sin_cambios": f"[*] Timelines con eventos: sin cambios ({n_tl}); no se vuelve a subir.",
                   "regresion": f"[X] Timelines con eventos: local {n_tl} < Drive {tl.get('remoto', 0):,}; "
                                "no se sube (usar --force si es intencional).",
                   "integridad": f"[X] Timelines con eventos: quick_check falló ({tl.get('detalle')}); no se sube.",
                   "subida": f"[+] Timelines con eventos sincronizados: {n_tl} en "
                             f"'{TIMELINES_SUBDIR}/{TIMELINES_DB_NAME}'."}.get(tl["estado"], str(tl)))

            old_drive_zip = drive_target / "lol_las.zip"
            if old_drive_zip.exists():
                old_drive_zip.unlink()
                print("[*] ZIP viejo eliminado de Google Drive.")

            # 2.4 Snapshots versionados inmutables (solo si hay espacio suficiente)
            snaps_dir = drive_target / "snapshots"
            snaps_dir.mkdir(parents=True, exist_ok=True)
            existing = list_versioned(snaps_dir)
            drive_free = shutil.disk_usage(str(drive_target)).free
            db_size_bytes = int(manifest.get("db_size_mb", 0) * 1024 * 1024)
            min_free_for_snap = 2500 * 1024 * 1024 if db_size_bytes > 100 * 1024 * 1024 else 50 * 1024 * 1024
            if drive_free >= min_free_for_snap and should_create_versioned(existing, manifest):
                vname = snapshot_name(manifest)
                safe_drive_copy(temp_snapshot_db, snaps_dir / vname)
                with open(snaps_dir / f"{vname}.manifest.json", "w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2, ensure_ascii=False)
                report["versioned"] = vname
                print(f"[+] Snapshot versionado inmutable: 'snapshots/{vname}'")
            else:
                if drive_free < min_free_for_snap:
                    print(f"[*] Almacenamiento local bajo ({round(drive_free / (1024*1024), 1)} MB libres). Omitiendo snapshot duplicado para proteger disco.")
                else:
                    print("[*] Snapshot versionado no necesario (último vigente).")
            pruned = prune_versioned(snaps_dir, VERSIONED_KEEP)
            report["pruned"] = pruned
            if pruned:
                print(f"[*] Versionados antiguos podados ({len(pruned)}), se conservan {VERSIONED_KEEP}.")
        else:
            print("\n[i] NOTA: No se ha detectado o configurado la ruta de Google Drive.")
            if not DRIVE_PATH_ENV and not drive_path:
                print("    Configura GOOGLE_DRIVE_BACKUP_PATH en el archivo .env")
            else:
                print(f"    La ruta configurada no existe actualmente: '{drive_target}'")
    finally:
        if temp_snapshot_db.exists():
            temp_snapshot_db.unlink()
        local_zip = local_backups_dir / "lol_las.zip"
        if local_zip.exists():
            local_zip.unlink()

    return report


def main():
    parser = argparse.ArgumentParser(description="Backup atómico de SQLite + CSVs a Google Drive")
    parser.add_argument("--dry-run", action="store_true",
                        help="Verifica integridad y manifiesto sin subir nada")
    parser.add_argument("--force", action="store_true",
                        help="Permite sobrescribir Drive aunque el manifiesto local tenga menos filas")
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("  SINCRONIZACIÓN DIRECTA (DB + CSVS) -> GOOGLE DRIVE [SIN ZIPS]")
    print("=" * 65)
    start_time = time.time()
    run_backup(dry_run=args.dry_run, allow_regression=args.force)
    print(f"[*] Operación completada en {time.time() - start_time:.2f} segundos.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
