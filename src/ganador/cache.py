"""Caché de variables que se invalida sola.

Calcular las variables previas lleva ~22 min (son causales, partida por partida); ajustar el modelo, segundos.
Por eso cada grupo de variables se guarda en `data/cache/ganador/<nombre>.npz` junto con una **huella**:
un hash del código que lo calcula y de sus insumos (archivos de datos y huellas de los grupos de los que
depende). Al pedirlo:

* si el archivo existe y la huella coincide, se carga;
* si no existe, o cambió el código, un archivo de datos o un grupo del que depende, se recalcula.

Así nunca se entrena con variables viejas sin darse cuenta (el problema de las cachés anteriores,
que se reutilizaban solo porque el archivo existía).
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from src.ganador import rutas

CLAVE = "__huella__"


def firma_archivo(ruta: Path) -> str:
    """Nombre, tamaño y fecha de modificación: cambia si el archivo se regenera."""
    st = ruta.stat()
    return f"{ruta.name}:{st.st_size}:{st.st_mtime_ns}"


def firma_codigo(funcion: Callable) -> str:
    """Hash del código de la función y de las funciones de su mismo módulo que llama (recursivo)."""
    modulo = inspect.getmodule(funcion)
    vistas, pila, partes = set(), [funcion], []
    while pila:
        f = pila.pop()
        if f.__name__ in vistas:
            continue
        vistas.add(f.__name__)
        partes.append(inspect.getsource(f))
        for nombre in f.__code__.co_names:
            g = getattr(modulo, nombre, None)
            if inspect.isfunction(g) and inspect.getmodule(g) is modulo:
                pila.append(g)
    return hashlib.sha256("\n".join(sorted(partes)).encode()).hexdigest()


def huella(funcion: Callable, archivos: Iterable[Path] = (), dependencias: Iterable[str] = ()) -> str:
    texto = "|".join([firma_codigo(funcion), *map(firma_archivo, archivos), *dependencias])
    return hashlib.sha256(texto.encode()).hexdigest()[:20]


def obtener(nombre: str, construir: Callable[[], dict], h: str, rehacer: bool = False) -> dict:
    """Devuelve el grupo `nombre` desde la caché si la huella `h` coincide; si no, lo calcula y lo guarda."""
    ruta = rutas.CACHE / f"{nombre}.npz"
    if ruta.exists() and not rehacer:
        with np.load(ruta, allow_pickle=True) as z:
            if str(z[CLAVE]) == h:
                return {k: z[k] for k in z.files if k != CLAVE}
    datos = construir()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ruta, **{CLAVE: np.array(h)}, **datos)
    return datos
