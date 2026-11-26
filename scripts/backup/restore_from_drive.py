#!/usr/bin/env python3
"""
scripts/restore_from_drive.py
=============================
Restaura la base de datos local desde Google Drive (flujo inverso al backup).

Reglas de seguridad:
  1. NUNCA sobreescribe sin respaldo: la BD local actual se copia primero a
     data/backups/lol_las_pre_restore_YYYYMMDD_HHMM.db (vía API backup).
  2. Verifica PRAGMA quick_check tras copiar; si falla, revierte automáticamente
     al respaldo pre-restore y aborta con error.
  3. Solo la máquina recolectora escribe a la nube; este script es para recuperar
     una máquina (o verificar un snapshot) en modo lectura posterior.

Uso:
    python scripts/restore_from_drive.py --list
    python scripts/restore_from_drive.py                          # latest
    python scripts/restore_from_drive.py --snapshot lol_las_20260901_1200_m71602.db
    python scripts/restore_from_drive.py --dest data/lol_las_check.db --dry-run
"""

import argparse
import json
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Agregar raíz del proyecto a sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.backup.backup_to_drive import (  # noqa: E402
    DRIVE_PATH_ENV,
    LOCAL_BACKUPS_DIR,
    check_integrity,
    list_versioned,
)


def resolve_drive(drive_arg: Optional[str] = None) -> Path:
    raw = (drive_arg or DRIVE_PATH_ENV or "").strip().strip('"').strip("'")
    if not raw:
        print("[X] Sin ruta de Drive (arg --drive-path o GOOGLE_DRIVE_BACKUP_PATH en .env).")
        sys.exit(1)
    drive = Path(raw)
    if not drive.is_dir():
        print(f"[X] La ruta de Drive no existe o no es carpeta: '{drive}'.")
        sys.exit(1)
    return drive


def local_pre_restore_backup(dest: Path, local_backups_dir: Path = LOCAL_BACKUPS_DIR) -> Optional[Path]:
    """Respalda la BD local actual antes de tocarla. None si no existía."""
    if not dest.exists():
        return None
    local_backups_dir.mkdir(parents=True, exist_ok=True)
    backup = local_backups_dir / f"lol_las_pre_restore_{datetime.now().strftime('%Y%m%d_%H%M')}.db"
    src = dst = None
    try:
        src = sqlite3.connect(dest)
        dst = sqlite3.connect(backup)
        src.backup(dst)
        print(f"[+] Respaldo pre-restore local: '{backup.name}'")
        return backup
    finally:
        if dst is not None:
            dst.close()
        if src is not None:
            src.close()


def show_manifest(path: Path) -> None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"    manifiesto: {data.get('matches', '?'):,} partidas, "
              f"{data.get('players', '?'):,} jugadores, mapa={data.get('map_rows', '?')}, "
              f"creado={data.get('created_at', '?')}")
    except (OSError, ValueError):
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Restaura la BD local desde Google Drive")
    parser.add_argument("--list", action="store_true",
                        help="Lista latest + versionados disponibles y sale")
    parser.add_argument("--snapshot", type=str, default=None,
                        help="Nombre del versionado en snapshots/ (default: latest lol_las.db)")
    parser.add_argument("--dest", type=str, default=str(ROOT_DIR / "data" / "lol_las.db"),
                        help="Destino local (default: data/lol_las.db)")
    parser.add_argument("--drive-path", type=str, default=None,
                        help="Carpeta de Drive (default: GOOGLE_DRIVE_BACKUP_PATH)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Muestra qué haría sin copiar nada")
    parser.add_argument("--no-backup", action="store_true",
                        help="Omitir respaldo pre-restore (no recomendado)")
    args = parser.parse_args()

    drive = resolve_drive(args.drive_path)

    if args.list:
        latest = drive / "lol_las.db"
        print(f"latest: {'lol_las.db' if latest.exists() else '(ausente)'}")
        if latest.exists():
            show_manifest(drive / "lol_las.manifest.json")
        print("versionados:")
        for name, dt, matches in list_versioned(drive / "snapshots"):
            print(f"  {name}  ({dt:%Y-%m-%d %H:%M}, {matches:,} partidas)")
            show_manifest(drive / "snapshots" / f"{name}.manifest.json")
        return

    if args.snapshot:
        source = drive / "snapshots" / args.snapshot
    else:
        source = drive / "lol_las.db"
    if not source.exists():
        print(f"[X] Origen inexistente en Drive: '{source}'. Usá --list para ver disponibles.")
        sys.exit(1)

    dest = Path(args.dest)
    print("=" * 65)
    print(f"  RESTORE: '{source.name}' -> '{dest}'")
    print("=" * 65)
    if args.dry_run:
        print("[dry-run] No se copió nada.")
        return

    pre_backup = None if args.no_backup else local_pre_restore_backup(dest)
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        ok, msg = check_integrity(dest)
        if not ok:
            raise RuntimeError(f"quick_check falló tras el restore: {msg}")
        print(f"[+] Restore OK, integridad verificada (quick_check=ok).")
    except Exception as e:
        print(f"[X] Restore fallido: {e}")
        if pre_backup and pre_backup.exists():
            shutil.copy2(pre_backup, dest)
            print(f"[!] Se revirtió al respaldo pre-restore: '{pre_backup.name}'")
        sys.exit(1)
    print("=" * 65)


if __name__ == "__main__":
    main()
