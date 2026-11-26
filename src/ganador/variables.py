"""Variables previas a la partida del modelo final de quién gana (escalón 9).

Todas son **causales**: para cada partida se calculan solo con partidas anteriores (orden por fecha y
`match_id`), leyendo el estado de cada jugador antes de actualizarlo con el resultado de la partida.

| Grupo (archivo en `data/cache/ganador/`) | Qué mide | Columnas | `PROJECT_SUMMARY.md` |
| :--- | :--- | ---: | :--- |
| `variables_partida` | 85 variables por jugador (rendimiento previo, experiencia, campeón) y el draft | 425 + 851 | §9.S.41 |
| `margen_campeon_linea` | Rendimiento previo del jugador con ese campeón y en esa línea | 150 | §9.S.29 |
| `nivel_historico` | Nivel por emparejamiento: LP de la foto 1 de los rivales y compañeros previos (prior 0 antes de la foto) | 3 | §9.S.31, §9.S.75 |
| `wl_foto1` | Victorias y derrotas de temporada de la foto 1, actualizadas con lo observado | 4 | §9.S.52 |
| `contracruces` | Campeón contra campeón por línea y sinergias (empírico-bayesiano) | 10 | §9.S.52 |
| `winrate_linea` | Winrate previo por línea y por campeón-línea; caída por autofill | 10 | §9.S.52 |
| `kalman_elo` | Elo con filtro de Kalman, tempo, objetivos y economía 15→20 | 33 | §9.S.52 |
| `oro_linea_hgb` | Oro @15 de cada línea predicho antes de jugar (HGB entrenado hacia adelante) | 6 | §9.S.36, §9.S.52 |
| `margen_esperado_par` | Margen esperado @10/@15 del cruce exacto de campeones en cada línea | 12 | §9.S.52 |

Uso:

    from src.ganador.variables import calcular_todas
    D, V = calcular_todas()          # usa la caché si está vigente; si no, recalcula (~22 min la primera vez)
"""
from __future__ import annotations

import numpy as np

from src.ganador import _constructores as K
from src.ganador import cache, datos, rutas


SEMILLA = 42      # semilla principal del HGB por línea (`oro_linea_hgb`); la de siempre (§9.S.80)


def nombre_oro(semilla: int = SEMILLA) -> str:
    """Archivo de caché del HGB por línea: `oro_linea_hgb` con la semilla principal, `oro_linea_hgb_s{n}` con otra."""
    return "oro_linea_hgb" if semilla == SEMILLA else f"oro_linea_hgb_s{semilla}"


def calcular_todas(rehacer: bool = False, semilla: int = SEMILLA) -> tuple[dict, dict]:
    """Devuelve `(D, V)`: datos y partición (`D`) y un diccionario con cada grupo de variables (`V`).

    Usa el modo de evaluación de `rutas.configurar` (por defecto `v`). `semilla` es la del HGB por línea, el
    único grupo que depende de una semilla; los demás son deterministas y se comparten entre semillas.
    """
    modo = f"modo={rutas.MODO}"
    h_part = cache.huella(K.variables_partida, [*rutas.FUENTES, rutas.DB, rutas.DDRAGON_JSON], [modo])
    Z = cache.obtener("variables_partida", K.variables_partida, h_part, rehacer)
    D = datos.armar(Z)

    def grupo(nombre, funcion, archivos=(), deps=(), args=(), kwargs=None):
        h = cache.huella(funcion, [*rutas.FUENTES, *archivos], [h_part, modo, *deps])
        return cache.obtener(nombre, lambda: funcion(D, *args, **(kwargs or {})), h, rehacer), h

    V, H = {}, {}
    for nombre, funcion, archivos in (
        ("margen_campeon_linea", K.margen_campeon_linea, ()),
        ("nivel_historico", K.nivel_historico, (rutas.DB,)),
        ("wl_foto1", K.wl_foto1, (rutas.DB,)),
        ("contracruces", K.contracruces, ()),
        ("winrate_linea", K.winrate_linea, (rutas.DB,)),
        ("kalman_elo", K.kalman_elo, ()),
        ("margen_esperado_par", K.margen_esperado_par, ()),
    ):
        K.log(f"variables: {nombre}")
        V[nombre], H[nombre] = grupo(nombre, funcion, archivos)
    K.log(f"variables: oro_linea_hgb (semilla {semilla})")
    V["oro_linea_hgb"], _ = grupo(
        nombre_oro(semilla), K.oro_linea_hgb, (rutas.ATRIB_CSV,),
        deps=(H["margen_campeon_linea"], H["nivel_historico"], H["contracruces"], f"semilla={semilla}"),
        args=(V["margen_campeon_linea"]["MC"], V["margen_campeon_linea"]["ML"],
              V["nivel_historico"]["A"], V["contracruces"]["MATCHUPS"]),
        kwargs={"semilla": semilla},
    )
    return D, V


def matriz_extras(D: dict, V: dict) -> np.ndarray:
    """Columnas que se suman al draft y a las 85 variables por jugador (mismo orden que el escalón 9)."""
    NM = D["NM"]
    MC, ML = V["margen_campeon_linea"]["MC"], V["margen_campeon_linea"]["ML"]
    A = V["nivel_historico"]["A"]
    k = V["kalman_elo"]
    base = np.c_[(MC[:, 0] - MC[:, 1]).reshape(NM, -1), (ML[:, 0] - ML[:, 1]).reshape(NM, -1),
                 (A[:, 0] - A[:, 1])[..., :3].mean(axis=1), V["wl_foto1"]["FOTO1_DIN"]].astype(np.float32)
    return np.c_[base, V["contracruces"]["MATCHUPS"], V["winrate_linea"]["ROL_WR"],
                 k["KALMAN_OUT"], k["KALMAN_MRG"], k["TEMPO_OBJ"], k["ECON_MID"],
                 V["oro_linea_hgb"]["HGB"], V["margen_esperado_par"]["MATCHUP_EXP_MARGIN"]].astype(np.float32)
