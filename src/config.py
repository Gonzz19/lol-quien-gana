"""
Módulo de configuración del proyecto.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno desde .env en la raíz del proyecto
ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# Claves de la API de Riot Games (soporta una o varias separadas por comas en RIOT_API_KEYS o RIOT_API_KEY)
raw_keys = os.getenv("RIOT_API_KEYS", os.getenv("RIOT_API_KEY", ""))
RIOT_API_KEYS = [k.strip() for k in raw_keys.replace(";", ",").split(",") if k.strip()]
RIOT_API_KEY = RIOT_API_KEYS[0] if RIOT_API_KEYS else ""

# Ranura de proyecto (`project_index`) que le corresponde a cada key del pool.
#
# Por qué existe: el PUUID está cifrado por *producto* de API, no por key. Dos keys
# de cuentas de Riot distintas devuelven PUUIDs distintos para el mismo humano, y
# ninguna puede descifrar los de la otra. `api_projects` numera esos espacios.
#
# Por defecto la ranura es la posición en el pool, lo cual solo es correcto si el
# pool contiene TODOS los proyectos históricos y en el mismo orden. Cuando se
# trabaja con un subconjunto (p. ej. una sola key propia, mientras las otras 3 son
# de otra persona), hay que declararlo o el registro pisa la ranura 0 ajena:
#
#     RIOT_API_PROJECT_INDICES=3
#
raw_indices = os.getenv("RIOT_API_PROJECT_INDICES", "").replace(";", ",")
RIOT_API_PROJECT_INDICES = [int(i.strip()) for i in raw_indices.split(",") if i.strip()]
if RIOT_API_PROJECT_INDICES and len(RIOT_API_PROJECT_INDICES) != len(RIOT_API_KEYS):
    raise ValueError(
        f"RIOT_API_PROJECT_INDICES tiene {len(RIOT_API_PROJECT_INDICES)} entradas "
        f"pero el pool tiene {len(RIOT_API_KEYS)} keys: deben coincidir."
    )


def project_index_for_key(key: str, env_keys=None, env_indices=None):
    """Proyecto de Riot de una key, buscado **por la key**, no por su posición.

    Se busca la key en `RIOT_API_KEYS` del `.env` y se toma la entrada
    correspondiente de `RIOT_API_PROJECT_INDICES` (o su posición en el `.env`, si
    no está declarado). La posición en una lista ya filtrada (p. ej. tras
    descartar keys vencidas) no sirve: correría las ranuras y etiquetaría PUUIDs
    de un proyecto como de otro. Devuelve `None` si la key no está en el `.env`.
    """
    env_keys = list(RIOT_API_KEYS if env_keys is None else env_keys)
    env_indices = list(RIOT_API_PROJECT_INDICES if env_indices is None else env_indices)
    k = (key or "").strip()
    if k not in env_keys:
        return None
    pos = env_keys.index(k)
    return env_indices[pos] if len(env_indices) == len(env_keys) else pos


# Plataforma por defecto (LAS = la2) y Enrutamiento Regional (Americas)
DEFAULT_PLATFORM = os.getenv("DEFAULT_PLATFORM", "la2").lower()
DEFAULT_REGIONAL_ROUTING = os.getenv("DEFAULT_REGIONAL_ROUTING", "americas").lower()

# Colas y Tiers de League of Legends
QUEUE_SOLOQ_NAME = "RANKED_SOLO_5x5"
QUEUE_SOLOQ_ID = 420  # ID numérico de 5v5 Ranked Solo games

APEX_TIERS = ["CHALLENGER", "GRANDMASTER", "MASTER"]
DIAMOND_DIVISIONS = ["I", "II", "III", "IV"]

# Orden de liga de menor a mayor. Es el que usa el SQL para "el tier más alto de una
# persona" (`src.database.sql_tier_maximo`): MAX(tier) sobre texto ordena
# alfabéticamente y daría DIAMOND > CHALLENGER (§9.S.76).
TIER_ORDER_LIGA = ("IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD",
                   "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER")

# Rutas de almacenamiento
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_DB_PATH = DATA_DIR / "lol_las.db"

# Configuración de Rate Limiting para Development Keys
# Riot Dev Key: 20 req / 1s, 100 req / 120s
RATE_LIMIT_SAFETY_MARGIN = 0.95  # Margen de seguridad para no rozar el límite exacto

# Umbrales adaptativos calculados por densidad de grafo para converger a ~300 partidas por jugador (Fase 2 dinámica)
DYNAMIC_TIER_MATCH_TARGETS = {
    "CHALLENGER": 160,
    "GRANDMASTER": 175,
    "MASTER": 215,
    "DIAMOND": 260
}

# Threshold temporal solicitado por el usuario: Primero acumular 50 partidas por jugador en DFS
TEMPORARY_DFS_TARGET = 50

# Umbrales activos por defecto para el DFS (temporalmente 50 para todos los tiers)
DEFAULT_TIER_MATCH_TARGETS = {
    "CHALLENGER": 50,
    "GRANDMASTER": 50,
    "MASTER": 50,
    "DIAMOND": 50
}


