"""Validación temporal por bloques (forward-chaining) para decidir si una idea entra al modelo.

Regla del proyecto desde 2026-11-14 (PROJECT_SUMMARY.md §9.S.47): una variante se decide con
los bloques del conjunto de ajuste, y V solo la confirma. Así cada decisión se apoya en varias
ventanas de tiempo y V, que se reusó más de 70 veces, se gasta menos.

Esquema (el mismo que `ganador_logistica_oos.py` y `ganador_red_jugador.py --oos`):

* El ajuste se parte en `k` bloques de igual cantidad de partidas, por fecha.
* El bloque b ≥ 1 se predice con un modelo entrenado **solo con los bloques anteriores**
  (ventana creciente). El bloque 0 solo entrena.
* Por bloque se mide el ΔAUC de la variante contra la referencia, con IC por bootstrap pareado.
  El resumen es la media de los Δ por bloque, con IC por bootstrap estratificado por bloque.

Decisión (`decidir`):

* **entra:** mejora en al menos `ceil(frac · n)` de los n bloques evaluados (3 de 4 con
  frac = 0,75) **y** el IC 95 % de la media de los Δ queda entero arriba de 0.
* **empeora:** el IC de la media queda entero debajo de 0.
* **no concluyente:** lo demás.
"""
from math import ceil
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def es_master_plus(tnum: np.ndarray) -> np.ndarray:
    """Partida de Master+: regla oficial de `split.py` (al menos 5 jugadores del padrón ≥ Master).

    `tnum`: (partidas, 10) con Diamond = 0, Master = 1, GM = 2, Challenger = 3 y NaN sin dato.
    Los jugadores fuera del padrón (`NaN`) no cuentan para alcanzar el umbral de 5.
    """
    arr = np.asarray(tnum, dtype=float)
    return (arr >= 1.0).sum(axis=1) >= 5


def es_master_plus_split(tnum: np.ndarray, min_jugadores: int = 5) -> np.ndarray:
    """Partida de Master+ con la **regla del split** (`src/features/split.py`, §9.S.12): al menos
    `min_jugadores` = 5 jugadores del padrón con tier ≥ Master (los que no tienen dato no cuentan).

    Es la regla que define el test. Sobre las 187.817 partidas del train da exactamente el rango de
    `match_split.csv` (§9.S.80). `tnum` como en `es_master_plus`.
    """
    t = np.nan_to_num(np.asarray(tnum, dtype=float), nan=-1.0)
    return (t >= 1.0).sum(axis=1) >= min_jugadores


def rango_lobby(tnum: np.ndarray, min_jugadores: int = 5) -> np.ndarray:
    """Rango del lobby con la regla del split: el tier más alto con ≥ 5 jugadores de ese tier o superior.

    Devuelve 0 = Diamond, 1 = Master, 2 = GM, 3 = Challenger y -1 = sin rango.
    """
    t = np.nan_to_num(np.asarray(tnum, dtype=float), nan=-1.0)
    out = np.full(len(t), -1)
    for k in range(4):                                  # de abajo hacia arriba: el último que cumple gana
        out[(t >= k).sum(axis=1) >= min_jugadores] = k
    return out


def particion_v_master(val: np.ndarray, tnum: np.ndarray, regla: str = "mediana"):
    """Regla desde 2026-11-15 (§9.S.51): V = solo partidas de Master+ de la ventana de validación.

    Devuelve `(val_master, diamond_v)`: la nueva V y las partidas de Diamond de la ventana, que
    pasan al último bloque del ajuste (A4). `regla`: `"mediana"` (`es_master_plus`) o `"split"`
    (`es_master_plus_split`). Desde el merge de `gonchi` (2026-11-25) `es_master_plus` también usa
    ≥ 5 del padrón, así que las dos dan lo mismo; el nombre "mediana" queda por compatibilidad. Los
    números de V ya reportados (16.428 partidas) se calcularon con la mediana y no se recalcularon.
    """
    val = np.asarray(val, dtype=bool)
    if regla == "mediana":
        m = es_master_plus(tnum)
    elif regla == "split":
        m = es_master_plus_split(tnum)
    else:
        raise ValueError(f"regla desconocida: {regla!r}")
    return val & m, val & ~m


def bloques_por_fecha(t: np.ndarray, k: int) -> np.ndarray:
    """Bloque 0..k-1 de cada partida según su fecha, con cortes en los cuantiles de `t`."""
    t = np.asarray(t)
    if k < 2:
        raise ValueError("hacen falta al menos 2 bloques")
    cortes = np.quantile(t, np.linspace(0, 1, k + 1)[1:-1])
    return np.searchsorted(cortes, t, side="right")


def predecir_por_bloques(ajustar: Callable, X, y: np.ndarray, bloque: np.ndarray,
                         k_min: int = 1) -> np.ndarray:
    """Predicción fuera de muestra y hacia adelante de cada partida con bloque ≥ `k_min`.

    `ajustar(X_entrena, y_entrena)` devuelve una función `predecir(X) -> prob`. El bloque b se
    predice con lo entrenado en los bloques < b. Las partidas de bloques < `k_min` quedan NaN.
    """
    p = np.full(len(y), np.nan)
    for b in range(k_min, int(bloque.max()) + 1):
        antes, este = bloque < b, bloque == b
        if not este.any():
            continue
        predecir = ajustar(X[antes], y[antes])
        p[este] = predecir(X[este])
    return p


def comparar_por_bloques(y: np.ndarray, p_nuevo: np.ndarray, p_ref: np.ndarray, bloque: np.ndarray,
                         bloques_eval: Optional[Iterable[int]] = None, B: int = 200,
                         seed: int = 0) -> pd.DataFrame:
    """ΔAUC (nuevo − referencia) por bloque y su media, con IC 95 % por bootstrap pareado.

    Devuelve una fila por bloque y una fila `media` (media de los Δ por bloque; el bootstrap
    remuestrea dentro de cada bloque, así cada réplica conserva la composición por bloque).
    """
    if bloques_eval is None:
        bloques_eval = sorted(int(b) for b in np.unique(bloque[~np.isnan(p_nuevo)]))
    rng = np.random.default_rng(seed)
    filas, reps = [], []
    for b in bloques_eval:
        i = np.flatnonzero(bloque == b)
        yb, pn, pr = y[i], p_nuevo[i], p_ref[i]
        a_n, a_r = roc_auc_score(yb, pn), roc_auc_score(yb, pr)
        rb = np.empty(B)
        for r in range(B):
            s = rng.integers(0, len(i), len(i))
            rb[r] = roc_auc_score(yb[s], pn[s]) - roc_auc_score(yb[s], pr[s])
        reps.append(rb)
        filas.append({"bloque": str(b), "partidas": len(i), "AUC ref": a_r, "AUC nuevo": a_n,
                      "delta": a_n - a_r, "ic_lo": np.percentile(rb, 2.5), "ic_hi": np.percentile(rb, 97.5)})
    tab = pd.DataFrame(filas)
    media = np.mean(reps, axis=0)
    tab.loc[len(tab)] = {"bloque": "media", "partidas": int(tab.partidas.sum()),
                         "AUC ref": tab["AUC ref"].mean(), "AUC nuevo": tab["AUC nuevo"].mean(),
                         "delta": tab.delta.mean(), "ic_lo": np.percentile(media, 2.5),
                         "ic_hi": np.percentile(media, 97.5)}
    return tab


def decidir(tabla: pd.DataFrame, frac: float = 0.75) -> str:
    """'entra', 'empeora' o 'no concluyente' según la regla del módulo."""
    por_bloque = tabla[tabla.bloque != "media"]
    media = tabla[tabla.bloque == "media"].iloc[0]
    n_mejora = int((por_bloque.delta > 0).sum())
    if n_mejora >= ceil(frac * len(por_bloque)) and media.ic_lo > 0:
        return "entra"
    if media.ic_hi < 0:
        return "empeora"
    return "no concluyente"
