"""Suma al cache de secuencias las variables de decay finas (las mismas que usa gb_exp2)."""
import pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import AQUI, cargar  # noqa
c = cargar(); e, partes = c["e"], c["partes"]
DIA = 86_400_000
ps = partes[partes["player_id"].isin(e.index)].sort_values(["player_id", "game_creation"])
g = ps.groupby("player_id")["game_creation"]
fin = ((c["t2"] - g.max()) / DIA).reindex(e.index).fillna(55)
ini = ((g.min() - c["t1"]) / DIA).reindex(e.index).fillna(55)
huecos = g.diff() / DIA
medio = (huecos - 14).clip(lower=0).groupby(ps["player_id"]).sum().reindex(e.index).fillna(0)
n14 = (huecos > 14).groupby(ps["player_id"]).sum().reindex(e.index).fillna(0)
ult = g.max().reindex(e.index)
a = ps.assign(u=ps["player_id"].map(ult))
antes = a[a["game_creation"] >= a["u"] - 14 * DIA].groupby("player_id").size().reindex(e.index).fillna(0)
with open(AQUI / "seq.pkl", "rb") as fh:
    S = pickle.load(fh)
S["est"] = S["est"].assign(exc_fin=((fin - 14).clip(lower=0) / 30).astype(np.float32),
                           exc_ini=((ini - 14).clip(lower=0) / 30).astype(np.float32),
                           exc_medio=(medio / 30).astype(np.float32), n_huecos14=(n14 / 3).astype(np.float32),
                           antes_fin=(antes / 20).astype(np.float32),
                           piso375=(e["z2"] == 375).astype(np.float32), piso275=(e["z2"] == 275).astype(np.float32))
with open(AQUI / "seq.pkl", "wb") as fh:
    pickle.dump(S, fh)
print(list(S["est"].columns))
