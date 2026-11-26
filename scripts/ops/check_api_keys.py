"""
Script de diagnóstico rápido para verificar el estado y vigencia de las API keys en .env.
Ejecuta una petición de prueba para cada clave e informa si está ACTIVA (200), VENCIDA (401),
FORBIDDEN (403) o con RATE LIMIT (429).

Además de la CLI, expone `probe_key()` y `check_keys()` para que otros
orquestadores (p. ej. `run.py`) hagan el chequeo preventivo que exige
`AGENTS.md` §3.B sin duplicar la lógica ni el criterio de "clave usable".
"""

import sys
from pathlib import Path

# Agregar raíz del proyecto a sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import Dict, List

from src.config import RIOT_API_KEYS
from src.riot_client import RiotAPIClient

#: Estados en los que la clave sirve para crawlear. El 429 entra: la clave es
#: válida y solo está saturada en el instante del sondeo.
USABLE_STATUSES = {200, 429}

#: Etiqueta legible por código HTTP devuelto por el sondeo.
STATUS_LABELS = {
    200: "ACTIVA (HTTP 200 OK)",
    401: "VENCIDA / NO AUTORIZADA (HTTP 401 Unauthorized)",
    403: "PROHIBIDA (HTTP 403 Forbidden)",
    429: "RATE LIMITED (HTTP 429 - Intenta de nuevo en unos segundos)",
}


def mask(key: str) -> str:
    return f"{key[:9]}...{key[-4:]}" if len(key) > 13 else "******"


def probe_key(key: str, timeout: float = 5.0) -> int:
    """
    Sondea una clave con una petición barata a League-V4.

    Devuelve el código HTTP, o `0` si la petición ni siquiera llegó (sin red,
    DNS caído, timeout). Nunca lanza.
    """
    if not key or not key.strip():
        return 0
    try:
        client = RiotAPIClient(api_key=key)
        with client.session:
            return client.probe_apex_status(timeout)
    except Exception:
        return 0


def check_keys(keys: List[str], timeout: float = 5.0) -> Dict[str, int]:
    """clave -> código HTTP del sondeo. Una petición por clave."""
    return {key: probe_key(key, timeout=timeout) for key in keys}


def describe(status: int) -> str:
    if status == 0:
        return "ERROR DE CONEXION (no hubo respuesta)"
    return STATUS_LABELS.get(status, f"RESPUESTA INESPERADA (HTTP {status})")


def test_keys():
    print("=" * 60)
    print("  VERIFICACIÓN DE VIGENCIA DE RIOT API KEYS")
    print("=" * 60)

    if not RIOT_API_KEYS:
        print("[!] ERROR: No se encontraron API keys configuradas en .env (RIOT_API_KEYS)")
        sys.exit(1)

    print(f"[*] Total de claves detectadas: {len(RIOT_API_KEYS)}\n")

    results = check_keys(RIOT_API_KEYS)
    all_ok = True
    for idx, key in enumerate(RIOT_API_KEYS, 1):
        status = results[key]
        marker = "[+]" if status == 200 else ("[!]" if status in USABLE_STATUSES or status == 0 else "[-]")
        print(f"  {marker} Clave #{idx} ({mask(key)}): {describe(status)}")
        if status not in USABLE_STATUSES:
            all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("[OK] Todas las claves configuradas están vigentes y operativas.")
    else:
        print("[AVISO] Una o más claves requieren renovación. Actualiza el archivo .env.")
    print("=" * 60)


if __name__ == "__main__":
    test_keys()
