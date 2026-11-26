"""Única evaluación de la caja cerrada para los candidatos declarados en bitacora.md (y referencias)."""
import sys
import numpy as np, pandas as pd
from comun import AQUI, cargar, evaluar, registrar, linea
c = cargar(); e = c["e"]; caja = c["caja"]; idx = e.index[caja]
gm, ch = c["corte"]


def cargar_p(t):
    return np.load(AQUI / "preds" / f"{t}_caja.npy")


cands = {}
cands["regla al revés"] = c["regla"].to_numpy()
cands["anterior: 0,65 GRU + 0,35 GB"] = 0.65 * cargar_p("est_plus_s5") + 0.35 * cargar_p("gbF_l1")
cuatro = sys.argv[1].split(",")
cands["PRINCIPAL: promedio de " + "+".join(cuatro)] = np.mean([cargar_p(t) for t in cuatro], 0)
cands["ALTERNATIVA: " + sys.argv[2]] = cargar_p(sys.argv[2])
for n, p in cands.items():
    m = evaluar(e, pd.Series(p, index=e.index), idx)
    registrar("CAJA2 " + n, m, "segunda apertura de la caja, candidatos fijados antes")
    err = np.abs(e["z1"].to_numpy() - p)
    print("    por tier:", {t: round(err[caja & (e["tier2"] == t).to_numpy()].mean(), 1) for t in ["DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER"]})
