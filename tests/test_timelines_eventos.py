"""`timelines_eventos.py --ids`: baja solo el lote pedido, sin arrastrar lo pendiente de otras selecciones."""
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("timelines_eventos",
                                               REPO / "scripts" / "collection" / "timelines_eventos.py")
te = importlib.util.module_from_spec(_spec)
sys.modules["timelines_eventos"] = te
_spec.loader.exec_module(te)


def test_leer_ids_sin_vacias_ni_duplicadas(tmp_path):
    f = tmp_path / "ids.txt"
    f.write_text("LA2_3\n\nLA2_1\n  LA2_3 \nLA2_2\n", encoding="utf-8")
    assert te.leer_ids(f) == ["LA2_3", "LA2_1", "LA2_2"]


def test_registrar_lote_solo_pendientes_del_lote(tmp_path):
    alm = te.Almacen(tmp_path / "t.db")
    # selección vieja con pendientes que el lote no debe arrastrar
    alm.guardar_seleccion([("LA2_1", "cache", 0), ("LA2_9", "resto", 1)])
    alm.guardar([("LA2_1", b"x", 1, 0.0)], [("LA2_2", "404")])

    pend = alm.registrar_lote("diam_v", ["LA2_3", "LA2_1", "LA2_2", "LA2_4"])
    assert pend == ["LA2_3", "LA2_4"]                     # orden del archivo, sin bajadas ni sin_timeline
    assert "LA2_9" not in pend

    # el grupo histórico de `seleccion` no se pisa; las nuevas entran después del orden existente
    with alm._conn() as c:
        sel = dict(c.execute("SELECT match_id, grupo FROM seleccion").fetchall())
        orden = dict(c.execute("SELECT match_id, orden FROM seleccion").fetchall())
    assert sel["LA2_1"] == "cache" and sel["LA2_3"] == "diam_v"
    assert min(orden["LA2_3"], orden["LA2_4"]) > 1

    alm.guardar([("LA2_3", b"x", 1, 0.0)], [])
    assert alm.resumen_lotes() == [("diam_v", 4, 2, 1)]  # total, bajadas, sin timeline
    assert alm.registrar_lote("diam_v", ["LA2_3", "LA2_1", "LA2_2", "LA2_4"]) == ["LA2_4"]  # retoma
