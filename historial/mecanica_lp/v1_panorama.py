"""Panorama de lp_vivo.db (copia de Drive, solo lectura)."""
import contextlib
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
SP = Path(sys.argv[1])
uri = f"file:{(SP / 'vivo' / 'lp_vivo.db').as_posix()}?mode=ro&immutable=1"

with contextlib.closing(sqlite3.connect(uri, uri=True, timeout=60)) as c:
    print(c.execute("select count(*) from eventos").fetchone(), "eventos")
    print(pd.read_sql_query("select platform, count(*) lecturas, min(t_inicio) t0, max(t_inicio) t1, max(jugadores) jug from lecturas group by 1", c).assign(
        t0=lambda d: pd.to_datetime(d.t0, unit="ms"), t1=lambda d: pd.to_datetime(d.t1, unit="ms")).to_string())
    print(pd.read_sql_query("select platform, tipo, estado, count(*) n from eventos group by 1,2,3", c).pivot_table(
        index=["tipo", "estado"], columns="platform", values="n", fill_value=0).astype(int).to_string())
    print(pd.read_sql_query("select count(*) partidas, min(game_end) t0, max(game_end) t1 from partidas", c))
    print(pd.read_sql_query("select platform, queue_id, count(*) n from partidas group by 1,2", c).to_string())
    print(pd.read_sql_query("select count(*) n, count(distinct puuid) jug from participantes", c))
    # cadencia de lecturas en LAS
    l = pd.read_sql_query("select t_inicio, t_fin from lecturas where platform='la2' order by t_inicio", c)
print("LAS: mediana entre lecturas (s)", (l.t_inicio.diff() / 1000).median(), "p95", (l.t_inicio.diff() / 1000).quantile(.95),
      "max (h)", (l.t_inicio.diff() / 3.6e6).max())
print("LAS: duración de la lectura (s)", ((l.t_fin - l.t_inicio) / 1000).median())
