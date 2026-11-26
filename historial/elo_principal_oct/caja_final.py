"""Evaluación única en la caja cerrada de las dos candidatas fijadas antes de mirarla."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import AQUI, cargar, evaluar, linea, registrar  # noqa
c = cargar(); e = c["e"]; caja = c["caja"]; idx = e.index[caja]
L = lambda t: pd.Series(np.load(AQUI / "preds" / f"{t}_caja.npy"), index=e.index)  # noqa: E731
principal = 0.65 * L("est_plus_s5") + 0.35 * L("gbF_l1")
alternativa = (L("est_plus_s5") + L("est2_s5") + L("est2_l2_s5") + L("gbF_l1") + L("gbF_l2")) / 5
for nom, p in (("regla al revés", c["regla"]), ("GRU est_plus 5 semillas", L("est_plus_s5")), ("GB L1", L("gbF_l1")),
               ("PRINCIPAL 0,65 GRU + 0,35 GB", principal), ("ALTERNATIVA promedio de 5", alternativa)):
    registrar("CAJA final: " + nom, evaluar(e, p, idx))
err = (e["z1"] - principal)[caja]
grupo = np.where(e.loc[caja, "tier2"].isin(["CHALLENGER", "GRANDMASTER"]), "GM/Challenger",
                 e.loc[caja, "tier2"].str.capitalize())
print("\nPrincipal por rango (caja):")
print(pd.DataFrame({"g": grupo, "a": err.abs()}).groupby("g")["a"].agg(["size", "mean"]).round(1).to_string())
