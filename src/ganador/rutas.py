"""Rutas de los datos que usa el modelo de quién gana, y el **modo de evaluación**. Todo relativo a la raíz del repo.

Modos (`configurar`, `PROJECT_SUMMARY.md` §9.S.80 y `docs/evaluacion_test/PROTOCOLO.md`):

| Modo | Partidas (`FUENTES`) | Evaluación (`val0`) | Master+ | Caché |
| :--- | :--- | :--- | :--- | :--- |
| `v` (por defecto) | train | desde la mediana de las fechas posteriores a la foto 1 (16/09) | mediana del lobby | `data/cache/ganador/` |
| `simulacro` | train | igual que `v` (V hace de test) | regla del split (≥ 5) | `data/cache/ganador_simulacro/` |
| `test` | train ∪ test ∪ Diamond posterior al corte | desde el corte del split (27/09 00:00 ART) | regla del split (≥ 5) | `data/cache/ganador_test/` |
| `test_sin_v` | igual que `test` | desde el 16/09, como en `v`: se ajusta sin V; se mide solo el test | regla del split (≥ 5) | `data/cache/ganador_test_sin_v/` |

En todos, lo que se ajusta usa solo partidas anteriores al inicio de la evaluación (`~val0`); el modo `v` es
exactamente lo de antes (mismas cuentas bit a bit). Los modos `test` y `test_sin_v` los usa solo
`scripts/analysis/evaluacion_test.py` (`test_sin_v` es el anexo de §9.S.80: el modelo del simulacro sobre el test).
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"

TRAIN_CSV = DATA / "splits" / "train" / "full_dataset.csv"   # partidas de entrenamiento (export_to_csv --target splits)
TEST_CSV = DATA / "splits" / "test" / "full_dataset.csv"     # el test: solo lo lee el modo "test"
DIAMANTE_DESPUES_CSV = DATA / "splits" / "diamante" / "despues" / "full_dataset.csv"  # Diamond desde el corte
MATCH_SPLIT = DATA / "splits" / "match_split.csv"            # partida → train / test / ... (congelado, hash en el protocolo)
SPLIT_MANIFEST = DATA / "splits" / "split_manifest.json"
DB = DATA / "lol_las.db"                                     # base SQLite (padrón de la foto 1 del ranking)
ATRIB_CSV = DATA / "externos" / "atributos_campeones.csv"    # atributos de campeones (Meraki + CommunityDragon)
DDRAGON_JSON = DATA / "ddragon" / "16.17.1" / "champion.json"

CACHE_V = DATA / "cache" / "ganador"                         # variables calculadas (se invalidan solas)
CACHE_SIMULACRO = DATA / "cache" / "ganador_simulacro"
CACHE_TEST = DATA / "cache" / "ganador_test"
CACHE_TEST_SIN_V = DATA / "cache" / "ganador_test_sin_v"

# Usadas por scripts de scripts/analysis/ (antes apuntaban a G:\Mi unidad\..., issue #8)
PLAYERS_CSV = DATA / "players.csv"                           # perfil por jugador (export_to_csv, base completa)
FIRMAS_A_CSV = DATA / "player_signatures_A.csv"              # firmas de pool por jugador
LP_VIVO_DB = DATA / "lp_vivo" / "lp_vivo.db"                 # LP partida a partida (scripts/collection/lp_vivo.py)
DDRAGON_CSV = DATA / "ddragon" / "16.17.1" / "champions.csv"  # crecimiento por nivel de cada campeón (modelo en vivo)
CACHE_EN_VIVO = DATA / "analysis" / "en_vivo"                # métricas del modelo en vivo (scripts/analysis/modelo_en_vivo.py)

# ---------------------------------------------------------------------------------------- modo de evaluación
MODOS = ("v", "simulacro", "test", "test_sin_v")
CORTE_TEST = "2026-09-27"                                    # día del corte del split (00:00 ART, split_manifest.json)
ART = timezone(timedelta(hours=-3))

MODO = "v"
FUENTES: tuple = (TRAIN_CSV,)
CACHE = CACHE_V


def corte_test_ms() -> int:
    """Instante (ms) del corte del split: 27/09 00:00 hora de Argentina (`src/features/split.py`)."""
    return int(datetime.strptime(CORTE_TEST, "%Y-%m-%d").replace(tzinfo=ART).timestamp() * 1000)


def configurar(modo: str = "v", cache: Path | None = None) -> None:
    """Fija el modo de evaluación para todo `src.ganador` (fuentes, inicio de la evaluación, regla de Master+, caché).

    `cache` cambia solo la carpeta (por ejemplo, para verificar el modo `v` sin tocar `data/cache/ganador/`).
    """
    global MODO, FUENTES, CACHE
    if modo not in MODOS:
        raise ValueError(f"modo desconocido: {modo!r} (válidos: {MODOS})")
    MODO = modo
    FUENTES = (TRAIN_CSV, TEST_CSV, DIAMANTE_DESPUES_CSV) if modo in ("test", "test_sin_v") else (TRAIN_CSV,)
    CACHE = Path(cache) if cache is not None else {"v": CACHE_V, "simulacro": CACHE_SIMULACRO, "test": CACHE_TEST,
                                                     "test_sin_v": CACHE_TEST_SIN_V}[modo]


def leer_partidas(usecols=None, **kw):
    """Las filas de participación de las partidas del modo actual (`FUENTES`, concatenadas en ese orden).

    En el modo `v` es exactamente `pd.read_csv(TRAIN_CSV, usecols=...)`. El orden de las filas no importa:
    todos los constructores ordenan por fecha y `match_id` después de leer.
    """
    import pandas as pd
    partes = [pd.read_csv(f, usecols=usecols, **kw) for f in FUENTES]
    return partes[0] if len(partes) == 1 else pd.concat(partes, ignore_index=True)


def inicio_evaluacion(t: np.ndarray, t_foto: int):
    """Desde cuándo una partida es de evaluación (`val0 = t >= inicio`).

    Modos `v` y `simulacro`: la mediana de las fechas posteriores a la foto 1 (el 16/09, la cuenta de siempre).
    Modo `test`: el corte del split (27/09 00:00 ART), así que `val0` son exactamente las partidas del test (y las
    Diamond posteriores al corte), y todo lo anterior al corte es ajuste.
    Modo `test_sin_v`: el mismo 16/09 que en `v` (la mediana se toma solo sobre las partidas anteriores al corte, que
    son las del train), así que V no se ajusta; `evaluacion_test.py` mide solo las partidas desde el corte.
    """
    if MODO == "test":
        return corte_test_ms()
    if MODO == "test_sin_v":
        return np.median(t[(t >= t_foto) & (t < corte_test_ms())])
    return np.median(t[t >= t_foto])


def regla_master() -> str:
    """Regla de Master+ del modo: `mediana` en `v` (la de siempre), `split` (≥ 5 jugadores) en los otros dos."""
    return "mediana" if MODO == "v" else "split"
