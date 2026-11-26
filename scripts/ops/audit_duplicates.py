#!/usr/bin/env python3
"""
scripts/ops/audit_duplicates.py
===============================
Auditoría de jugadores duplicados o fusionados (solo lectura, sin red).

Recalcula el mapa de identidad en memoria (no usa el `player_id_map` guardado,
que podría estar viejo) y verifica:

* nadie aparece dos veces en el padrón;
* ninguna persona juega dos veces en la misma partida (fusión falsa);
* toda partida tiene exactamente 10 personas distintas;
* no quedan PUUIDs con partidas sin unir a ninguna persona con nombre;
* cada partida que Riot listó para un jugador (ventana) lo contiene al bajarla;
* cada nombre compartido entre personas tiene prueba de que son cuentas
  distintas (un PUUID exacto del mismo proyecto en cada una).

Sale con código 1 si algo falla. Es parte del día de congelamiento.

Uso:
    python scripts/ops/audit_duplicates.py
"""

import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import DEFAULT_DB_PATH  # noqa: E402
from src.database import open_readonly_connection  # noqa: E402
from src.features.identity import (  # noqa: E402
    attach_for_participants,
    audit_duplicates,
    suspicious_shared_names,
)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

CHECKS = ("padron_duplicado", "misma_persona_dos_veces_en_partida",
          "partidas_sin_10_personas", "personas_sin_nombre_con_partidas",
          "listado_ausente_en_su_partida")


def run(db_path: Path) -> int:
    conn = open_readonly_connection(db_path)
    try:
        t0 = time.monotonic()
        attach_for_participants(conn)
        res = audit_duplicates(conn)
        sospechosos = suspicious_shared_names(conn)
    finally:
        conn.close()
    fallas = 0
    print("=" * 65)
    print("  AUDITORÍA DE JUGADORES DUPLICADOS")
    print("=" * 65)
    for k in CHECKS:
        estado = "OK" if res[k] == 0 else "FALLA"
        fallas += res[k] > 0
        print(f"  [{estado:<5}] {k}: {res[k]:,}")
    estado = "OK" if not sospechosos else "FALLA"
    fallas += bool(sospechosos)
    print(f"  [{estado:<5}] nombres compartidos sin prueba de cuentas distintas: "
          f"{len(sospechosos)} (de {res['nombres_compartidos']} compartidos)")
    for name, pids in sospechosos[:20]:
        print(f"           - {name}: {', '.join(pids)}")
    print(f"  ({time.monotonic() - t0:.0f} s)")
    print("=" * 65)
    return 1 if fallas else 0


if __name__ == "__main__":
    sys.exit(run(DEFAULT_DB_PATH))
