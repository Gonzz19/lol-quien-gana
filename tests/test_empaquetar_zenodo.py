import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "ops"))
import empaquetar_zenodo as ez  # noqa: E402

CLAVE = bytes(32)


def _base(tmp_path):
    origen = tmp_path / "origen.db"
    with sqlite3.connect(origen) as c:
        c.execute("CREATE TABLE player_identities (canonical_id TEXT PRIMARY KEY, game_name TEXT NOT NULL, "
                  "tag_line TEXT NOT NULL, tier TEXT)")
        c.execute("CREATE TABLE player_id_map (puuid TEXT, player_id TEXT)")
        c.execute("CREATE TABLE matches (match_id TEXT, champion_name TEXT, crudo TEXT)")
        c.executemany("INSERT INTO player_identities VALUES (?, ?, ?, 'MASTER')",
                      [("jugador#las", "Jugador", "LAS"), ("otro#123", "Otro", "123")])
        c.executemany("INSERT INTO player_id_map VALUES (?, ?)", [("p1", "jugador#las"), ("p2", "otro#123")])
        c.execute("""INSERT INTO matches VALUES ('LA2_1', 'Ahri', '{"a": "#x"}')""")
    return origen


def test_anonimizar_base_seudonimo_consistente_y_sin_nombres(tmp_path):
    origen = _base(tmp_path)
    copia = tmp_path / "copia.db"
    ez.copiar_base(origen, copia)
    assert set(ez.quedan_ids(copia)) == {"player_identities.canonical_id", "player_identities.game_name",
                                         "player_identities.tag_line", "player_id_map.player_id"}
    ez.anonimizar_base(copia, CLAVE)
    assert ez.quedan_ids(copia) == []
    with sqlite3.connect(copia) as c:
        ids = dict(c.execute("SELECT puuid, player_id FROM player_id_map"))
        assert ids["p1"] == ez.seudonimo("jugador#las", CLAVE) and ids["p1"].startswith("j")
        # el mismo jugador da el mismo seudónimo en las dos tablas: los joins siguen andando
        assert c.execute("SELECT COUNT(*) FROM player_identities pi JOIN player_id_map m "
                         "ON m.player_id = pi.canonical_id").fetchone()[0] == 2
        assert c.execute("SELECT game_name, tag_line FROM player_identities").fetchall() == [("", ""), ("", "")]
        assert c.execute("SELECT champion_name, crudo FROM matches").fetchone() == ("Ahri", '{"a": "#x"}')
    with sqlite3.connect(origen) as c:                                                  # el original no se toca
        assert c.execute("SELECT game_name FROM player_identities WHERE tag_line = 'LAS'").fetchone() == ("Jugador",)


def test_anonimizar_csv_usa_el_mismo_seudonimo_que_la_base(tmp_path):
    origen = tmp_path / "m.csv"
    pd.DataFrame({"match_id": ["LA2_1", "LA2_1"], "blue_player_id": ["jugador#las", "otro#123"],
                  "red_player_id": ["otro#123", None], "puuid_raw": ["p1", "p2"]}).to_csv(origen, index=False)
    destino = tmp_path / "sal" / "m.csv"
    assert ez.anonimizar_csv(origen, destino, CLAVE, filas=1) == 2
    d = pd.read_csv(destino)
    assert d.blue_player_id.tolist() == [ez.seudonimo("jugador#las", CLAVE), ez.seudonimo("otro#123", CLAVE)]
    assert d.red_player_id.iloc[0] == ez.seudonimo("otro#123", CLAVE) and pd.isna(d.red_player_id.iloc[1])
    assert d.puuid_raw.tolist() == ["p1", "p2"] and not d.astype(str).apply(lambda s: s.str.contains("#")).any().any()
