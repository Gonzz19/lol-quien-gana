"""¿El decay tiene piso? Dónde terminan los que estuvieron inactivos mucho tiempo antes de la foto 2."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import cargar  # noqa

c = cargar()
e, partes = c["e"], c["partes"]
ult = partes.groupby("player_id")["game_creation"].max().reindex(e.index)
dias_fin = (c["t2"] - ult) / 86_400_000
pri = partes.groupby("player_id")["game_creation"].min().reindex(e.index)
dias_ini = (pri - c["t1"]) / 86_400_000
# Inactivos al final (> 20 días sin jugar antes de la foto 2) y Master+ en la foto 1
m = (dias_fin > 20) & (e["z1"] >= 400) & (e["ocultas"] <= 1)
print(f"Master+ en foto 1 con > 20 días sin jugar antes de foto 2 (sin ocultas): {m.sum()}")
print("z2 más frecuentes:", e.loc[m, "z2"].value_counts().head(8).to_dict())
print("tier2:", e.loc[m, "tier2"].value_counts().to_dict())
# Diamond: ¿también decae? ¿a dónde?
md = (dias_fin > 35) & (e["z1"] < 400) & (e["ocultas"] <= 1)
print(f"\nDiamond en foto 1, > 35 días sin jugar al final: {md.sum()}; cambio z2 - z1 por regla:",
      (e.loc[md, "z2"] - c["regla"][md] - (e.loc[md, "z1"] - c["regla"][md])).describe().round(0).to_dict())
print("  z2 - (lo que dice la regla hacia adelante) ≈ residuo:", (c["regla"][md] - e.loc[md, "z1"]).describe().round(0).to_dict())
# Cuánto cae un Master+ que NO llega al piso, por día pasado de gracia
r = (e["z1"] - c["regla"]).rename("residuo")   # LP perdido que las partidas no explican (signo: z1 > regla → perdió)
for gracia in (7, 10, 14, 21, 28):
    exc = (dias_fin - gracia).clip(lower=0)
    sel = m & (e["z2"] > 380) & (exc > 0)
    if sel.sum() > 20:
        k = np.median(r[sel] / exc[sel])
        print(f"gracia {gracia}: n={sel.sum()}, LP por día (mediana) {k:.1f}")
print(f"\nAl piso (z2 en 370–380, Master+ en foto 1): {((e['z2'].between(370, 380)) & (e['z1'] >= 400)).sum()}")
