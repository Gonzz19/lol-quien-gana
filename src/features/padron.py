"""La foto 1 del padrón, congelada (PROJECT_SUMMARY.md §9.S.68).

La tabla `players` guarda tier, LP y W/L de cada jugador del padrón, y `save_players` los pisa
con un upsert cada vez que se actualiza el padrón. Hoy la tabla entera es una sola foto, tomada
el 04/09/2026 a las 22:51:22 UTC, y de esa fecha salen cosas que no pueden moverse:

* el corte de V (mitad de las partidas posteriores a la foto);
* el LP y el W/L de la foto que usan como variables el nivel histórico, el W/L dinámico, el mapa
  de LP y el rating.

Si alguien actualiza el padrón y vuelve a armar las cachés, `MAX(updated_at)` deja de ser la foto 1,
el corte de V se corre y el W/L de la foto pasa a incluir partidas de V y del test: una fuga directa
del resultado. Por eso la fecha es una constante y toda lectura de la foto pasa por `verificar_foto1`,
que corta con un error si `players` ya no es la foto 1.
"""
import sqlite3

import pandas as pd

T_FOTO1_TXT = "2026-09-04 22:51:22"          # UTC, `players.updated_at` de toda la tabla
T_FOTO1_MS = int(pd.Timestamp(T_FOTO1_TXT, tz="UTC").value // 10**6)


class PadronCambiado(RuntimeError):
    """`players` ya no es la foto 1: alguien actualizó el padrón."""


def verificar_foto1(con: sqlite3.Connection) -> int:
    """Comprueba que `players` siga siendo la foto 1 y devuelve su fecha en ms (UTC).

    Falla si hay filas con otro `updated_at`: en ese caso tier, LP y W/L ya no son los de la foto
    y usarlos como variables metería información posterior al 04/09.
    """
    fechas = [r[0] for r in con.execute("SELECT DISTINCT updated_at FROM players ORDER BY updated_at")]
    if fechas != [T_FOTO1_TXT]:
        raise PadronCambiado(
            f"La tabla players ya no es la foto 1 ({T_FOTO1_TXT} UTC): tiene updated_at {fechas[:5]}. "
            "Para reproducir el análisis, usar una copia de la base que conserve la foto original.")
    return T_FOTO1_MS


def leer_foto1(con: sqlite3.Connection, columnas: str) -> pd.DataFrame:
    """`SELECT m.player_id, <columnas> FROM players p JOIN player_id_map m`, después de verificar.

    `columnas` se escribe con el alias `p.` (p. ej. "p.tier, p.league_points, p.wins, p.losses").
    """
    verificar_foto1(con)
    return pd.read_sql_query(
        f"SELECT m.player_id, {columnas} FROM players p JOIN player_id_map m ON p.puuid = m.puuid", con)
