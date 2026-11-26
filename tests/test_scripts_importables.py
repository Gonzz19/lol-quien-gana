"""Guarda estructural de los scripts de análisis y del historial de pruebas.

Los scripts de `scripts/analysis/` (vigentes) y de `historial/` (experimentos descartados) no se
ejecutan en los tests porque necesitan los datos y tardan minutos. Esta guarda comprueba, sin
ejecutarlos, lo que una reorganización de carpetas suele romper:

1. que cada script compile;
2. que cada import entre scripts (`scripts.analysis.X`, `historial.G.X`) apunte a un archivo existente;
3. que la raíz del repo calculada con `Path(__file__).resolve().parents[k]` sea de verdad la raíz;
4. que lo vigente no importe nada del historial.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CARPETAS = [REPO / "scripts" / "analysis", REPO / "historial"]
SCRIPTS = sorted(p for c in CARPETAS for p in c.rglob("*.py"))
RE_RAIZ = re.compile(r"Path\(__file__\)\.resolve\(\)\.parents\[(\d+)\]")


def _id(p: Path) -> str:
    return p.relative_to(REPO).as_posix()


def _imports_locales(arbol: ast.AST) -> set[str]:
    mods: set[str] = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.ImportFrom) and nodo.module:
            if nodo.module.startswith(("scripts.analysis", "historial")):
                partes = nodo.module.split(".")
                if len(partes) >= 3:  # paquete.subcarpeta.modulo
                    mods.add(".".join(partes[:3]))
                else:  # `from scripts.analysis import x`
                    mods |= {f"{nodo.module}.{a.name}" for a in nodo.names}
        elif isinstance(nodo, ast.Import):
            for a in nodo.names:
                if a.name.startswith(("scripts.analysis.", "historial.")):
                    mods.add(".".join(a.name.split(".")[:3]))
    return mods


def test_hay_scripts():
    assert len(SCRIPTS) > 100


@pytest.mark.parametrize("ruta", SCRIPTS, ids=_id)
def test_script_compila_e_imports_existen(ruta: Path):
    fuente = ruta.read_text(encoding="utf-8")
    arbol = ast.parse(fuente, filename=str(ruta))  # 1. compila
    for mod in _imports_locales(arbol):  # 2. imports entre scripts
        destino = REPO.joinpath(*mod.split(".")).with_suffix(".py")
        assert destino.exists(), f"{_id(ruta)} importa {mod}, que no existe"
    for k in RE_RAIZ.findall(fuente):  # 3. raíz del repo
        raiz = ruta.resolve().parents[int(k)]
        assert raiz == REPO, f"{_id(ruta)}: parents[{k}] da {raiz}, no la raíz"


def test_lo_vigente_no_importa_del_historial():
    malos = []
    for ruta in (REPO / "scripts").rglob("*.py"):
        mods = _imports_locales(ast.parse(ruta.read_text(encoding="utf-8")))
        malos += [f"{_id(ruta)} -> {m}" for m in mods if m.startswith("historial.")]
    assert not malos, malos
