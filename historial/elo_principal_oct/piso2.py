"""Peso en el error de los que están en un piso del decay (z2 = 275/325/375) con inactividad al final."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import cargar, evaluar, linea  # noqa
ROOT = Path(r"c:/Users/DarkA/OneDrive/Escritorio/Facultad/3er año/ML/TPF/ProyectoFinalML")
c = cargar(); e, partes = c["e"], c["partes"]
ult = partes.groupby("player_id")["game_creation"].max().reindex(e.index)
dias_fin = ((c["t2"] - ult) / 86_400_000).fillna(99)
en_piso = e["z2"].isin([275.0, 325.0, 375.0])
for umbral in (7, 14, 21):
    g = en_piso & (dias_fin > umbral)
    print(f"en piso y > {umbral} días sin jugar al final: {g.sum()} ({100 * g.mean():.1f} %), "
          f"de ellos Master+ en foto 1: {(e.loc[g, 'z1'] >= 400).mean():.2f}")
oof = pd.read_csv(ROOT / "data" / "analysis" / "boost_completo_oof.csv", index_col="player_id").reindex(e.index)
pred = c["regla"] + oof["gb"]
sq = (e["z1"] - pred) ** 2
g = en_piso & (dias_fin > 14)
print(f"\n% del error² en el grupo piso (>14 d): {100 * sq[g].sum() / sq.sum():.1f}")
print(linea("GB OOF, todos", evaluar(e, pred)))
print(linea("GB OOF, sin el grupo piso", evaluar(e, pred, e.index[~g])))
print(linea("GB OOF, solo el grupo piso", evaluar(e, pred, e.index[g])))
# Inactividad en el medio también (huecos > 14 días entre partidas)
t = partes.sort_values("game_creation").groupby("player_id")["game_creation"].diff() / 86_400_000
hueco_max = t.groupby(partes.loc[t.index, "player_id"]).max().reindex(e.index).fillna(0)
h = ~g & (hueco_max > 14) & (e["z2"] >= 0)
print(linea("GB OOF, hueco medio > 14 d (no piso)", evaluar(e, pred, e.index[h])), h.sum())
