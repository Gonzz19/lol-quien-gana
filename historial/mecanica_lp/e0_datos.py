"""Arma y cachea el train del elo (solo foto 1 → foto 2) más el rol de cada partida."""
import contextlib
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
SP = Path(sys.argv[1]); ROOT = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(ROOT))
from src.features.elo_reverso import cargar_train, evaluables  # noqa: E402

d = cargar_train(ROOT / "data")
p = d["partes"]
print("partes", p.shape, "desde", pd.to_datetime(p.game_creation.min(), unit="ms"), "hasta", pd.to_datetime(p.game_creation.max(), unit="ms"))
uri = f"file:{(ROOT / 'data' / 'lol_las.db').as_posix()}?mode=ro"
with contextlib.closing(sqlite3.connect(uri, uri=True, timeout=60)) as c:
    rol = pd.read_sql_query(
        "select mp.match_id, mi.player_id, mp.individual_position pos from match_participants mp join matches x on x.match_id=mp.match_id "
        "join player_id_map mi on mi.puuid=mp.puuid where x.game_creation >= ? and mp.match_id like 'LA2_%'", c, params=(d["t1"],))
p = p.merge(rol.drop_duplicates(["match_id", "player_id"]), on=["match_id", "player_id"], how="left")
print("rol nulo:", p.pos.isna().mean())
e = evaluables(d["foto1"], d["foto2"], p)
print("evaluables", len(e), e.tier2.value_counts().to_dict())
p.to_pickle(SP / "vivo" / "partes.pkl"); e.to_pickle(SP / "vivo" / "e.pkl")
d["foto1"].to_pickle(SP / "vivo" / "foto1.pkl"); d["foto2"].to_pickle(SP / "vivo" / "foto2.pkl")
pd.Series({"t1": d["t1"], "t2": d["t2"]}).to_pickle(SP / "vivo" / "ts.pkl")
