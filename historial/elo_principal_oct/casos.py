"""Línea de tiempo de los peores casos: fechas, resultado y lobby de cada partida."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import cargar  # noqa

c = cargar()
e, partes, lobp, lob2 = c["e"], c["partes"], c["lobp"], c["lob2"]
t1, t2 = pd.to_datetime(c["t1"], unit="ms"), pd.to_datetime(c["t2"], unit="ms")
# Los jugadores se pasan por línea de comandos (p. ej. `python casos.py "nombre#tag" ...`);
# la lista original tenía Riot IDs reales y se sacó del repo.
for pid in sys.argv[1:]:
    p = partes[partes["player_id"] == pid].sort_values("game_creation")
    print(f"\n{pid}: {e.at[pid, 'tier1']} {e.at[pid, 'z1']:.0f} → {e.at[pid, 'tier2']} {e.at[pid, 'z2']:.0f}, "
          f"dN {e.at[pid, 'dN']}, obs {e.at[pid, 'obs']:.0f}, W/L temporada f1 {e.at[pid, 'wins1']}/{e.at[pid, 'losses1']}"
          f" f2 {e.at[pid, 'wins2']}/{e.at[pid, 'losses2']}, regla {c['regla'][pid]:.0f}, decay {c['decay'][pid]:.0f}")
    fechas = pd.to_datetime(p["game_creation"], unit="ms")
    print(f"  foto1 {t1:%m-%d} | partidas {fechas.min():%m-%d} → {fechas.max():%m-%d} | foto2 {t2:%m-%d}")
    filas = pd.DataFrame({"fecha": fechas.dt.strftime("%m-%d %H:%M"), "win": p["win"].to_numpy(),
                          "lobby_f2": lob2.loc[p.index, "lobby"].round(0).to_numpy(),
                          "lobby_pred": lobp.loc[p.index, "lobby"].round(0).to_numpy(),
                          "lp_regla": c["lp_fila"][p.index].round(0).to_numpy()})
    print(filas.head(6).to_string(index=False))
    if len(filas) > 6:
        print("  ...")
        print(filas.tail(3).to_string(index=False, header=False))
