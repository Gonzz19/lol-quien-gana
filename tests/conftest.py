"""
Configuración y fixtures compartidas de pytest.

1. Inserta la raíz del repositorio en `sys.path` para que los tests importen
   `src.*` y `scripts.*` sin instalar el paquete. Centraliza el bootstrap que
   antes estaba duplicado en cada módulo de test.
2. Expone la fixture `temp_db`, única para toda la suite.
"""

import sys
import tempfile
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.database import DatabaseManager  # noqa: E402  (requiere el sys.path de arriba)


@pytest.fixture
def temp_db():
    """
    DatabaseManager sobre una BD SQLite en un directorio temporal.

    Aísla cada test de `data/lol_las.db`: nunca toca la base viva ni deja
    lockfiles `-wal`/`-shm` en el repo.

    Las partidas de juguete de la suite traen 1 o 2 participantes, así que acá
    se apaga el control de 10 participantes del guardado (§9.S.76). Los tests de
    ese control (`test_robustez_crawler.py`) lo vuelven a encender, que es el
    valor por defecto de `DatabaseManager`.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db = DatabaseManager(Path(tmpdir) / "test_lol.db")
        db.participantes_esperados = None
        yield db
