"""Oráculo: si supiéramos las victorias y derrotas ocultas (de la foto 1), ¿cuánto error queda?"""
import sys; from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import cargar, evaluar, linea
c = cargar(); e = c["e"]; dev = ~c["caja"]
print(e[["obs", "obs_w"]].head(), (e["obs_w"] <= e["obs"]).mean())
oW = e["obs_w"]; oL = e["obs"] - e["obs_w"]
hW = (e["dW"] - oW); hL = (e["dL"] - oL)
print("ocultas negativas", (hW < 0).mean(), (hL < 0).mean())
g = np.searchsorted([400, c["corte"][0], c["corte"][1]], e["z2"].to_numpy(), side="right")
a = pd.Series(g).map(c["a"]).to_numpy(); b = pd.Series(g).map(c["b"]).to_numpy()
orac = c["regla"] - a * hW.clip(lower=0) + b * hL.clip(lower=0)
idx = e.index[dev]
for n, p in [("regla", c["regla"]), ("regla + ocultas exactas", orac), ("regla + ocultas exactas + decay", orac + c["decay"])]:
    print(linea(n, evaluar(e, p, idx)))
    for t in ["DIAMOND", "MASTER"]:
        m = dev & (e["tier2"] == t).to_numpy()
        print("   ", t, round(np.abs(e["z1"][m] - p[m]).mean(), 1))
