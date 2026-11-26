#!/usr/bin/env python3
"""
scripts/auto_nightly_pipeline.py
================================
Orquesta una corrida continua del crawler por el tiempo especificado (ej. 4 horas),
y al terminar ejecuta automáticamente:
1. Exportación de CSVs actualizada.
2. Respaldo explícito a Google Drive (`backup_to_drive.py`).
3. Commit en Git bajo la cuenta del usuario.
4. Push a GitHub (solo código y documentación: la BD está en `.gitignore`).

Nota: no existe ningún hook `pre-push` hacia Drive, pese a lo que afirmaban
versiones anteriores de este docstring (verificado el 2026-09-07). La nube se
sincroniza en el paso 2, nunca por el push.

Para una sesión nueva conviene `run.py --minutos N --respaldar`, que además
audita el dataset dentro del cooldown del rate limit (`PROJECT_SUMMARY.md` §9.N).
"""

import sys
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def map_has_rows(db_path=None) -> bool:
    """True si el mapa multi-proyecto ya tiene filas (probe RO, ~ms, sin red)."""
    import sqlite3
    db = Path(db_path) if db_path else (PROJECT_ROOT / "data" / "lol_las.db")
    try:
        conn = sqlite3.connect(f"file:{db.resolve()}?mode=ro", uri=True, timeout=30.0)
        try:
            row = conn.execute("SELECT COUNT(*) FROM player_puuid_map;").fetchone()
            return int(row[0]) > 0
        finally:
            conn.close()
    except Exception:
        return False


def get_ready_api_keys(db_path=None) -> list:
    """Retorna las API keys seguras para el crawler.
    
    Si el Proyecto 2 aún está en proceso de mapeo inicial (< 8,000 PUUIDs en player_puuid_map),
    se aísla el crawling a las Keys 0 y 1 para no colisionar en rate limits (HTTP 429)
    ni recibir PUUIDs sin traducir. Una vez que P2 supera 8,000 entradas, se incluye
    automáticamente en el pool activo.
    """
    from src.config import RIOT_API_KEYS
    if len(RIOT_API_KEYS) <= 2:
        return RIOT_API_KEYS
    
    import sqlite3
    db = Path(db_path) if db_path else (PROJECT_ROOT / "data" / "lol_las.db")
    try:
        conn = sqlite3.connect(f"file:{db.resolve()}?mode=ro", uri=True, timeout=30.0)
        try:
            row = conn.execute("SELECT COUNT(*) FROM player_puuid_map WHERE project_index = 2;").fetchone()
            p2_count = int(row[0]) if row else 0
            if p2_count >= 8000:
                print(f"[*] Proyecto 2 mapeado completamente ({p2_count:,} PUUIDs). Usando pool completo de 3 keys.")
                return RIOT_API_KEYS
            else:
                print(f"[*] Proyecto 2 en mapeo activo ({p2_count:,}/8,354 PUUIDs). Usando Keys 0 y 1 para evitar contención de rate limit.")
                return RIOT_API_KEYS[:2]
        finally:
            conn.close()
    except Exception:
        return RIOT_API_KEYS[:2]


def build_crawler_cmd(chunk_min) -> list:
    """Comando del crawler; agrega --require-canonical solo si el mapa ya existe."""
    cmd = [sys.executable, "-u", "collect_data.py", "--collect-matches",
           "--duration-minutes", str(chunk_min)]
    keys = get_ready_api_keys()
    if keys:
        cmd.extend(["--api-keys", ",".join(keys)])
    if map_has_rows():
        cmd.append("--require-canonical")
        print("[*] Mapa multi-proyecto detectado: crawler en modo estricto (--require-canonical).")
    else:
        print("[!] Mapa multi-proyecto vacío: crawler en modo legacy (corre register_key_project.py).")
    return cmd

def run_cmd(cmd, check=True):
    print(f"\n[*] Ejecutando: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=False)
    if check and res.returncode != 0:
        print(f"[X] Comando falló con código {res.returncode}")
        return False
    return True

def main():
    first_arg = sys.argv[1].lower() if len(sys.argv) > 1 else "240"
    is_infinite = first_arg in ("infinite", "--infinite", "-i", "indefinido", "loop")

    if is_infinite:
        chunk_min = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
        print("=" * 65)
        print(f"  PIPELINE DE RECOLECCIÓN INDEFINIDA (CICLOS DE {chunk_min:.0f} MIN) + DRIVE + GIT")
        print("=" * 65)
        cycle = 1
        while True:
            print("\n" + "#" * 65)
            print(f"  >>> INICIANDO CICLO #{cycle} ({chunk_min:.0f} MINUTOS) <<<")
            print("#" * 65)

            # 1. Ejecutar crawler por chunk_min
            crawler_cmd = build_crawler_cmd(chunk_min)
            run_cmd(crawler_cmd, check=False)

            # 2. Exportar los 4 datasets CSV actualizados
            export_cmd = [sys.executable, "export_to_csv.py"]
            run_cmd(export_cmd, check=False)

            # 3. Respaldo explícito a Google Drive (DB + CSVs sin comprimir)
            backup_cmd = [sys.executable, "scripts/backup_to_drive.py"]
            run_cmd(backup_cmd, check=False)

            # 4. Git commit & push (código y documentación únicamente)
            run_cmd(["git", "add", "PROJECT_SUMMARY.md", "AGENTS.md", "src/", "scripts/", ".env.example", ".gitignore"], check=False)
            timestamp = time.strftime("%Y-%m-%d %H:%M")
            commit_msg = f"Auto-update: Recolección masiva ciclo #{cycle} ({timestamp}) y documentación"
            run_cmd(["git", "commit", "-m", commit_msg], check=False)
            run_cmd(["git", "push", "origin", "main"], check=False)

            print(f"\n[+] Ciclo #{cycle} completado con éxito. Iniciando Ciclo #{cycle + 1} inmediatamente...\n")
            cycle += 1
    else:
        duration_min = float(first_arg)
        print("=" * 60)
        print(f"  PIPELINE AUTOMÁTICO DE RECOLECCIÓN ({duration_min:.1f} MINUTOS) + PUSH")
        print("=" * 60)

        # 1. Ejecutar crawler
        crawler_cmd = build_crawler_cmd(duration_min)
        print(f"[*] 1. Iniciando recolección por {duration_min:.1f} minutos...")
        run_cmd(crawler_cmd, check=False)

        # 2. Exportar datasets CSV actualizados
        print("\n[*] 2. Exportando datasets CSV actualizados...")
        export_cmd = [sys.executable, "export_to_csv.py"]
        run_cmd(export_cmd, check=False)

        # 3. Respaldo explícito a Google Drive (DB + CSVs sin comprimir)
        print("\n[*] 3. Sincronizando base de datos y CSVs a Google Drive...")
        backup_cmd = [sys.executable, "scripts/backup_to_drive.py"]
        run_cmd(backup_cmd, check=False)

        # 4. Git Add (código y documentación únicamente)
        print("\n[*] 4. Preparando cambios para Git...")
        run_cmd(["git", "add", "PROJECT_SUMMARY.md", "AGENTS.md", "src/", "scripts/", ".env.example", ".gitignore"], check=False)

        # 5. Git Commit
        timestamp = time.strftime("%Y-%m-%d %H:%M")
        commit_msg = f"Auto-update: Recolección masiva de partidas ({timestamp}) y documentación"
        print(f"\n[*] 5. Creando commit: '{commit_msg}'...")
        run_cmd(["git", "commit", "-m", commit_msg], check=False)

        # 6. Git Push (disparará el hook pre-push)
        print("\n[*] 6. Pusheando a origin main...")
        run_cmd(["git", "push", "origin", "main"], check=False)

        print("\n" + "=" * 60)
        print("  ¡PIPELINE COMPLETADO EXITOSAMENTE!")
        print("=" * 60 + "\n")

if __name__ == "__main__":
    main()

