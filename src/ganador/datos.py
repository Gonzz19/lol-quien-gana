"""Partidas, partición temporal y matrices base del modelo de quién gana.

Partición (`PROJECT_SUMMARY.md` §9.S.47 y §9.S.51):

* **V** = ventana de validación (16/09 → 27/09) con mediana de tier del lobby ≥ Master;
* las partidas Diamond de esa ventana (`diam_v`) se usan solo como chequeo;
* el modelo que predice V se ajusta solo con partidas **anteriores** al 16/09 (`fi_v`);
* el test (`data/splits/test`) no se carga nunca acá salvo en el modo `test` (`rutas.configurar`, §9.S.80),
  donde "V" son las partidas del test, Master+ es la regla del split (≥ 5 jugadores) y el ajuste es todo lo
  anterior al corte del 27/09. En el modo `simulacro` V hace de test con esa misma regla.

Antes las cargaba `scripts/analysis/brainstorm_base.cargar`, que además exigía calcular el mapa de LP
(y con él `lp_vivo.db`) aunque el modelo final no lo usa. Esta versión no lo necesita.
"""
from __future__ import annotations

import sqlite3

import numpy as np

from src.features.draft import draft_matrix_by_lane
from src.features.padron import verificar_foto1
from src.models.validacion_temporal import bloques_por_fecha, particion_v_master
from src.ganador import rutas


def fecha_foto1() -> int:
    """Instante (ms) de la foto 1 del ranking, verificando que el padrón siga congelado."""
    with sqlite3.connect(f"file:{rutas.DB.resolve()}?mode=ro", uri=True) as con:
        return verificar_foto1(con)


def armar(Z: dict) -> dict:
    """Diccionario `D` que reciben los constructores de variables y el modelo.

    `Z` es la salida de `variables_partida` (85 variables por jugador, campeones, fechas, resultado...).
    """
    Xj, champ, t, y = Z["Xj"], Z["champ"], Z["t"], Z["y"].astype(int)
    val0 = Z["val"].astype(bool)
    NM = len(y)
    regla = rutas.regla_master()                      # "mediana" en el modo v; "split" (≥ 5) en simulacro y test
    val, diam_v = particion_v_master(val0, Z["tnum"], regla)
    fit = ~val
    post = t >= fecha_foto1()
    fi = np.flatnonzero(fit)
    ai = np.flatnonzero(post & ~val0)
    bloque = np.full(NM, -1)
    bloque[fi] = 0
    bloque[ai] = bloques_por_fecha(t[ai], 4)
    bloque[diam_v] = 3
    return dict(
        Z=Z, NM=NM, Xj=Xj, cols=list(Z["cols"]), y=y, t=t, player=Z["player"],
        val=val, val0=val0, diam_v=diam_v, fit=fit, fi=fi, fi_v=fi[~diam_v[fi]], post=post,
        master=particion_v_master(np.ones(NM, bool), Z["tnum"], regla)[0], regla_master=regla,
        Xd=(Xj[:, 0] - Xj[:, 1]).reshape(NM, -1).astype(np.float32),     # azul − rojo por línea
        DRAFT=draft_matrix_by_lane(champ),                                 # campeón × línea, con signo
        bloque=bloque, EV=[1, 2, 3],
    )
