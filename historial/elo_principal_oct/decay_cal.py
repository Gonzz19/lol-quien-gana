"""Decay calibrado con los datos (decay_curva.py): Master+ pierde 75/día después de 14 días hasta 375 (D1 75);
Diamond I (y D2 76-99) pierde 50/día después de 28 días hasta 275 (D2 75); de 275 para abajo no hay decay.

Hacia atrás: si el LP después del hueco no está en un piso, el LP perdido se devuelve exacto; si está en
un piso, queda censurado (se sabe una cota). Variables por hueco final (última partida → foto 2), inicial
(foto 1 → primera partida) y del medio. Guarda decay_cal.pkl.
"""
import pickle
import numpy as np, pandas as pd
from comun import AQUI, cargar
DIA = 86_400_000
c = cargar(); e, partes = c["e"], c["partes"]


def devolver(z, d):
    """LP antes de un hueco de d días, dado el LP z después. Devuelve (devuelto, censura375, censura275, techo)."""
    z, d = np.asarray(z, float), np.asarray(d, float)
    dev = np.zeros_like(z); c375 = np.zeros_like(z); c275 = np.zeros_like(z); techo = np.zeros_like(z)
    master = z >= 400
    dev = np.where(master, 75 * np.clip(np.floor(d) - 14, 0, None), dev)
    d1 = (z > 275) & (z < 375)
    dev = np.where(d1, 50 * np.clip(np.floor(d) - 28, 0, None), dev)
    en375 = (z == 375) & (d > 14)
    en275 = (z == 275) & (d > 28)
    c375, c275 = en375.astype(float), en275.astype(float)
    techo = np.where(en375, 25 + 75 * np.clip(np.floor(d) - 14, 0, None), techo)
    techo = np.where(en275, 100 + 50 * np.clip(np.floor(d) - 28, 0, None) + 75 * np.clip(np.floor(d) - 14, 0, None), techo)
    return dev, c375, c275, techo


ps = partes[partes["player_id"].isin(e.index)].sort_values(["player_id", "game_creation"])
g = ps.groupby("player_id")["game_creation"]
fin = ((c["t2"] - g.max()) / DIA).reindex(e.index).fillna((c["t2"] - c["t1"]) / DIA)
ini = ((g.min() - c["t1"]) / DIA).reindex(e.index).fillna(0)
sin = ~e.index.isin(ps["player_id"].unique())
F = pd.DataFrame(index=e.index)
F["dc_fin"], F["c375_fin"], F["c275_fin"], F["techo_fin"] = devolver(e["z2"], fin)
# Hueco inicial: el LP después es el de la regla al llegar a la partida más vieja (sin decay del final).
z_viejo = c["regla"] + F["dc_fin"]
F["dc_ini"], F["c375_ini"], F["c275_ini"], F["techo_ini"] = devolver(z_viejo.round(), np.where(sin, 0, ini))
# Huecos del medio: LP antes de la partida más nueva del hueco (camino de la regla).
lp = c["lp_fila"].reindex(ps.index)
hueco = (g.shift(-1) - ps["game_creation"]) / DIA          # de esta partida a la siguiente (más nueva)
lp_despues = lp.groupby(ps["player_id"]).shift(-1)          # LP antes de la siguiente partida
m = hueco > 14
dv, c3, c2, tc = devolver(lp_despues[m].round().fillna(0), hueco[m])
med = pd.DataFrame({"pid": ps.loc[m, "player_id"], "dv": dv, "c": c3 + c2, "tc": tc})
agg = med.groupby("pid").agg(dc_medio=("dv", "sum"), cens_medio=("c", "sum"), techo_medio=("tc", "sum"))
F = F.join(agg).fillna(0)
F["base_dc"] = c["regla"] + F["dc_fin"] + F["dc_ini"] + F["dc_medio"]
print(F.describe().T.round(2).to_string())
pickle.dump(F, open(AQUI / "decay_cal.pkl", "wb"))
