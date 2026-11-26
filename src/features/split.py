"""
Split train/test temporal con el test proporcional por rango (PROJECT_SUMMARY.md §9.S.12).

Pedido del usuario: el test (~20 %) tiene que tener el mismo reparto por rango que
el resto del dataset (si Master es el 80 % de las partidas, también es el 80 % del
test), y tiene que ser posterior al train.

Rango de una partida (regla del usuario, 2026-10-25)
----------------------------------------------------
El tier **más alto con al menos `MIN_JUGADORES_RANGO` = 5 jugadores de ese tier o
superior** en el lobby (DIAMOND < MASTER < GRANDMASTER < CHALLENGER): 5+ Challenger
es Challenger; 5+ entre GM y Challenger es GM; 5+ Master o superior es Master; 5+
Diamond o superior es Diamond. Cuentan solo los jugadores del padrón. Si ningún
nivel junta 5 (lobby con muchos jugadores fuera del padrón) queda `SIN_RANGO`.

Antes era la mediana de los tiers del padrón: un estrato Challenger exigía que la
mitad de los jugadores *del padrón presentes* fueran Challenger, y los jugadores
fuera del padrón no contaban. Con el umbral de 5 sobre el lobby completo de 10 la
regla es explícita y no depende de cuántos jugadores del padrón hay en la partida.

Construcción
------------
1. Un **único corte de fecha** $T$ (límite de día, 00:00 hora de Argentina): todo lo
   anterior es train.
2. Para cada rango $r$, el test toma las $q_r$ partidas **más recientes** posteriores
   a $T$, con
   $$q_r = \\operatorname{round}\\Big(\\frac{f}{1-f}\\, n^{\\text{train}}_r\\Big),$$
   así que $q_r / (n^{\\text{train}}_r + q_r) = f$ en cada rango. El reparto por rango
   del test es entonces el mismo que el del train.
3. $T$ es el **día más tardío** para el que todos los rangos tienen al menos $q_r$
   partidas posteriores (mover $T$ hacia atrás achica el train y agranda los
   candidatos a test, así que la condición es monótona).
4. Las partidas posteriores a $T$ que sobran quedan como `fuera`: no entran al
   train, porque romperían la separación temporal.
5. **Rangos del test** (`test_tiers`, decisión del usuario del 2026-10-24: el test
   no lleva Diamond). Los pasos 2–3 se aplican solo a esos rangos, así que el
   reparto del test coincide con el del train *restringido a esos rangos*. Las
   partidas de otros rangos anteriores a $T$ quedan en train (qué hacer con
   Diamond en el train lo decide el usuario después; `match_split.csv` trae el
   rango para filtrarlo) y las posteriores a $T$ quedan `fuera`.
6. **Sobrante al test** (`sobrante_al_test`, decisión del usuario del 2026-10-25):
   $T$ se sigue eligiendo con las cuotas $q_r$ (así ningún rango queda con menos del
   20 %), pero después **todas** las partidas de los rangos del test posteriores a
   $T$ van al test, no solo las $q_r$ más recientes. Evita tirar ~26.000 partidas
   Master posteriores al corte; a cambio, los rangos con muchas partidas recientes
   (Master) pesan algo más en el test que en el train, y las métricas se reportan
   también por rango.
7. **Base de la cuota** (`base_desde`, decisión del usuario del 2026-10-25): $q_r$ se
   calcula con el train de $r$ **desde `BASE_CUOTA_DESDE`** (la ventana del 01/09,
   cobertura pareja para todo el padrón) y no con todo el train. Antes de esa fecha
   la cantidad de partidas de cada tier refleja cuánto historial se bajó (a GM+ se
   lo fue a buscar más hondo), no el juego: con la base completa, bajar historial
   viejo de Challenger subía su cuota y corría el corte hacia atrás, mandando
   partidas Diamond posteriores al corte a `fuera`. El train sigue usando todo.

Esto corre en el preprocesamiento, no en los CSV (AGENTS.md §3.G).
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

TIER_ORDER = ["DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER"]
ART = timezone(timedelta(hours=-3))
SIN_RANGO = "SIN_RANGO"
MIN_JUGADORES_RANGO = 5  # jugadores del tier (o superior) que definen el rango del lobby
TEST_TIERS = ("MASTER", "GRANDMASTER", "CHALLENGER")  # el test no lleva Diamond (2026-10-24)
SOBRANTE_AL_TEST = True  # lo posterior al corte de esos rangos va todo al test (2026-10-25)
# Base de la cuota: solo el train desde la ventana del 01/09, donde la cobertura es
# pareja para todo el padrón (2026-10-25). Antes de esa fecha la profundidad depende
# de cuánto historial se bajó de cada tier.
BASE_CUOTA_DESDE = "2026-09-01"


def match_tiers(full: pd.DataFrame, players: pd.DataFrame,
                min_jugadores: int = MIN_JUGADORES_RANGO) -> pd.Series:
    """`match_id -> rango`: el tier más alto con >= `min_jugadores` de ese tier o superior.

    `full`: filas de participación con `match_id`, `player_id`.
    `players`: `player_id`, `player_tier` (el padrón, p. ej. `players.csv`).
    """
    rank = {t: i for i, t in enumerate(TIER_ORDER)}
    tier_of = players.set_index("player_id")["player_tier"].map(rank)
    t = full[["match_id", "player_id"]].assign(r=full["player_id"].map(tier_of)).dropna(subset=["r"])
    cnt = (t.groupby(["match_id", "r"]).size().unstack(fill_value=0)
           .reindex(columns=range(len(TIER_ORDER)), fill_value=0))
    # Acumulado desde arriba: columna i = jugadores de tier i o superior.
    acum = cnt.iloc[:, ::-1].cumsum(axis=1).iloc[:, ::-1]
    ok = acum >= min_jugadores
    # El tier más alto que cumple: el último True de cada fila.
    nivel = ok.to_numpy()[:, ::-1].argmax(axis=1)
    best = len(TIER_ORDER) - 1 - nivel
    out = pd.Series(np.array(TIER_ORDER, dtype=object)[best], index=acum.index)
    out[~ok.any(axis=1)] = SIN_RANGO
    return out.reindex(full["match_id"].unique()).fillna(SIN_RANGO)


def _day_floor(ms: int) -> int:
    d = datetime.fromtimestamp(ms / 1000, ART).replace(hour=0, minute=0, second=0, microsecond=0)
    return int(d.timestamp() * 1000)


def proportional_temporal_split(matches: pd.DataFrame, test_frac: float = 0.2,
                                test_tiers: Optional[Iterable[str]] = None,
                                sobrante_al_test: bool = False,
                                base_desde: Optional[str] = None
                                ) -> Tuple[pd.Series, Dict]:
    """Etiqueta cada partida como 'train', 'test' o 'fuera'.

    `matches`: una fila por partida con `match_id`, `game_creation` (ms) y `tier`.
    `test_tiers`: rangos que pueden ir al test (None = todos).
    `sobrante_al_test`: True = toda partida de esos rangos posterior al corte va al
    test (el corte se sigue fijando con las cuotas; paso 6 del docstring del módulo).
    `base_desde` ("AAAA-MM-DD", hora ART): la cuota de cada rango se calcula solo con
    el train desde esa fecha (paso 7). None = todo el train.
    Devuelve `(etiquetas indexadas por match_id, resumen)`.
    """
    if not 0 < test_frac < 1:
        raise ValueError("test_frac tiene que estar entre 0 y 1.")
    m = matches.drop_duplicates("match_id").set_index("match_id")
    elegible = m["tier"].isin(set(test_tiers)) if test_tiers is not None else pd.Series(True, index=m.index)
    if not elegible.any():
        raise ValueError("Ninguna partida tiene un rango de test_tiers.")
    ratio = test_frac / (1 - test_frac)
    base_ms = (int(datetime.strptime(base_desde, "%Y-%m-%d").replace(tzinfo=ART).timestamp() * 1000)
               if base_desde else -(2 ** 62))
    days = sorted({_day_floor(int(x)) for x in m["game_creation"]}, reverse=True)

    cutoff: Optional[int] = None
    for day in days:  # del más reciente hacia atrás: el primero que alcanza es el más tardío
        pre = m[elegible & (m["game_creation"] < day) & (m["game_creation"] >= base_ms)]
        post = m[elegible & (m["game_creation"] >= day)]
        n_train = pre.groupby("tier").size()
        quota = (n_train * ratio).round().astype(int)
        disponibles = post.groupby("tier").size().reindex(quota.index, fill_value=0)
        if len(pre) and (disponibles >= quota).all():
            cutoff = day
            break
    if cutoff is None:
        raise ValueError("No hay un corte que deje suficientes partidas recientes en todos los rangos.")

    labels = pd.Series("fuera", index=m.index)
    labels[m["game_creation"] < cutoff] = "train"
    post = m[elegible & (m["game_creation"] >= cutoff)]
    n_train = m[elegible & (labels == "train") & (m["game_creation"] >= base_ms)].groupby("tier").size()
    quota = (n_train * ratio).round().astype(int)
    for tier, q in quota.items():
        cand = post[post["tier"] == tier].sort_values("game_creation", ascending=False)
        labels[cand.index if sobrante_al_test else cand.index[:q]] = "test"

    resumen = {
        "corte": datetime.fromtimestamp(cutoff / 1000, ART).strftime("%Y-%m-%d"),
        "train": int((labels == "train").sum()),
        "test": int((labels == "test").sum()),
        "fuera": int((labels == "fuera").sum()),
        "reparto_train": (m[labels == "train"].groupby("tier").size() / (labels == "train").sum()).round(4).to_dict(),
        "reparto_test": (m[labels == "test"].groupby("tier").size() / max(1, (labels == "test").sum())).round(4).to_dict(),
        # El reparto con el que se compara el test: el del train en los rangos del test.
        "reparto_train_rangos_test": (m[elegible & (labels == "train")].groupby("tier").size()
                                      / max(1, (elegible & (labels == "train")).sum())).round(4).to_dict(),
        "rangos_test": sorted(set(test_tiers)) if test_tiers is not None else "todos",
    }
    return labels, resumen


# ---------------------------------------------------------------------------
# Regla vigente desde el 2026-10-25 (decisión del usuario, PROJECT_SUMMARY.md §9.S.12)
# ---------------------------------------------------------------------------
DIAMANTE_ANTES, DIAMANTE_DESPUES, SIN_RANGO_LABEL = "diamante_antes", "diamante_despues", "sin_rango"
# Diamond anterior al corte va al train (decisión del usuario, 2026-10-26): da perfil a
# los jugadores Diamond que aparecen en el test (1 de cada 4 participaciones) sin
# poner Diamond en el test. Es pasado respecto de todo el test: no hay leakage.
DIAMANTE_ANTES_AL_TRAIN = True


def master_plus_temporal_split(matches: pd.DataFrame, test_frac: float = 0.2,
                               test_tiers: Iterable[str] = TEST_TIERS,
                               diamante_antes_al_train: bool = DIAMANTE_ANTES_AL_TRAIN
                               ) -> Tuple[pd.Series, Dict]:
    """Test = el `test_frac` más reciente de TODO lo Master+, con un corte de día único.

    * `train` / `test`: partidas de `test_tiers` antes / desde el corte $T$. $T$ es el
      día más tardío (00:00 ART) con al menos `test_frac` de las partidas Master+ en
      $[T, \infty)$, así que el test es el `test_frac` más reciente de Master+ en su
      conjunto (no por rango: cada rango queda con lo que jugó después de $T$).
    * Diamond **anterior** a $T$: va al `train` si `diamante_antes_al_train` (vigente
      desde el 2026-10-26), para que los jugadores Diamond del test tengan perfil; si
      no, queda aparte como `diamante_antes`.
    * `diamante_despues`: Diamond desde $T$, nunca en el test; sirve para evaluar el
      modelo de Master+ en Diamond.
    * `sin_rango`: lobbies donde ningún nivel junta 5 jugadores del padrón.
    """
    if not 0 < test_frac < 1:
        raise ValueError("test_frac tiene que estar entre 0 y 1.")
    m = matches.drop_duplicates("match_id").set_index("match_id")
    eleg = m["tier"].isin(set(test_tiers))
    if not eleg.any():
        raise ValueError("Ninguna partida tiene un rango de test_tiers.")
    meta = test_frac * int(eleg.sum())
    days = sorted({_day_floor(int(x)) for x in m.loc[eleg, "game_creation"]}, reverse=True)
    cutoff = next((d for d in days if (eleg & (m["game_creation"] >= d)).sum() >= meta), None)
    if cutoff is None or not (eleg & (m["game_creation"] < cutoff)).any():
        raise ValueError("No hay un corte que deje train y test.")

    post = m["game_creation"] >= cutoff
    labels = pd.Series(SIN_RANGO_LABEL, index=m.index)
    labels[eleg & ~post] = "train"
    labels[eleg & post] = "test"
    diamond = m["tier"] == "DIAMOND"
    labels[diamond & ~post] = "train" if diamante_antes_al_train else DIAMANTE_ANTES
    labels[diamond & post] = DIAMANTE_DESPUES

    def reparto(mask):
        return (m.loc[mask].groupby("tier").size() / max(1, int(mask.sum()))).round(4).to_dict()

    resumen = {
        "corte": datetime.fromtimestamp(cutoff / 1000, ART).strftime("%Y-%m-%d"),
        **{k: int((labels == k).sum()) for k in
           ("train", "test", DIAMANTE_ANTES, DIAMANTE_DESPUES, SIN_RANGO_LABEL)},
        "train_diamante": int((diamond & ~post & (labels == "train")).sum()),
        "test_sobre_master_plus": round(float((labels == "test").sum() / eleg.sum()), 4),
        "test_por_rango": {t: round(float((eleg & post & (m["tier"] == t)).sum()
                                          / max(1, int((m["tier"] == t).sum()))), 4) for t in test_tiers},
        "reparto_train": reparto(labels == "train"),
        "reparto_test": reparto(labels == "test"),
        "rangos_test": sorted(set(test_tiers)),
    }
    return labels, resumen
