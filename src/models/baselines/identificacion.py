"""
Baselines de identificación (escalón 1) para la prueba de `src/features/fingerprint.py`.

Cada función recibe las muestras A/B de `make_halves` y devuelve `(pids, A, B)`:
matrices jugador × dimensiones con el perfil de cada muestra, que después compara
`evaluate`.

* `pool`: frecuencia de cada campeón (distancia coseno). Vive en
  `src/features/fingerprint.py` y se reexporta acá.
* `elo_linea`: tier del padrón + reparto de líneas de la muestra.
* `embedding`: promedio de emb(campeón, línea) sobre las partidas de la muestra
  (PROJECT_SUMMARY.md §9.S.20). Describe al jugador solo por el *tipo* de campeones
  que elige: a diferencia de `pool`, dos campeones que se juegan parecido quedan
  cerca, así que sigue funcionando cuando A y B no comparten campeones.

El embedding es el de `src/features/player_signatures.py` (22 estadísticas medias
estandarizadas por campeón y línea, solo pares con ≥ 100 partidas). Para no tener
leakage se construye con el `champions_by_lane.csv` del train.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.features.fingerprint import CANONICAL_LANES, TIER_ORDER, pool_profiles  # noqa: F401
from src.features.player_signatures import build_champion_embeddings


def elo_lane_profiles(df: pd.DataFrame, halves: dict, info: pd.DataFrame) -> tuple:
    """Tier del padrón (ordinal) + fracción de partidas en cada línea de la muestra."""
    rank = {t: i for i, t in enumerate(TIER_ORDER)}
    pids = list(halves)

    def vec(p, idx):
        lanes = df.loc[idx, "lane"].value_counts(normalize=True).reindex(CANONICAL_LANES, fill_value=0)
        return np.r_[rank.get(info.at[p, "player_tier"], -1), lanes.to_numpy()]
    A = np.stack([vec(p, halves[p]["A"]) for p in pids])
    B = np.stack([vec(p, halves[p]["B"]) for p in pids])
    return pids, A, B


def load_embeddings(champions_csv: Path) -> Tuple[Dict, Dict]:
    """(emb(campeón, línea), vector medio de cada línea) desde un `champions_by_lane.csv`."""
    vecs, lane_vecs, _ = build_champion_embeddings(pd.read_csv(champions_csv))
    return vecs, lane_vecs


def embedding_profiles(df: pd.DataFrame, halves: Dict, vecs: Dict, lane_vecs: Dict) -> tuple:
    """Perfil = media de emb(campeón, línea) de las partidas de la muestra.

    Un par sin embedding (menos de 100 partidas en la tabla) usa el vector medio de
    su línea, para que cada muestra conserve sus k partidas.
    """
    dim = len(next(iter(lane_vecs.values())))
    keys = list(zip(df["champion_name"].astype(str), df["lane"].astype(str)))
    E = np.stack([vecs.get(k, lane_vecs.get(k[1], np.zeros(dim))) for k in keys])
    pos = pd.Series(np.arange(len(df)), index=df.index)
    pids = list(halves)
    A = np.stack([E[pos[halves[p]["A"]]].mean(axis=0) for p in pids])
    B = np.stack([E[pos[halves[p]["B"]]].mean(axis=0) for p in pids])
    return pids, A, B
