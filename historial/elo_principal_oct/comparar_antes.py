"""Compara los modelos de antes del objetivo con los mejores de ahora: desarrollo (5 folds) y caja."""
import numpy as np, pandas as pd
from comun import AQUI, cargar, evaluar
c = cargar(); e = c["e"]; caja = c["caja"]; dev = ~caja
def P(t, sufijo=""): return pd.Series(np.load(AQUI / "preds" / f"{t}{sufijo}.npy"), index=e.index)
def mod(s):
    return {"regla al revés": c["regla"], "GB del pipeline (antes)": P("gb_comp", s), "GRU del pipeline (antes)": P("gru_base", s),
            "1.ª noche: 0,65 GRU + 0,35 GB": 0.65 * P("est_plus_s5", s) + 0.35 * P("gbF_l1", s),
            "2.ª noche: promedio de 5": P("final5", s), "2.ª noche: compuerta": P("comp_t3", s)}
grupo = np.where(e["tier2"].isin(["CHALLENGER", "GRANDMASTER"]), "GM+", e["tier2"].str.capitalize())
mov = (e["z1"] - e["z2"]).abs() > 300
for nom, idx_m, s in (("DESARROLLO (5 folds)", dev, ""), ("CAJA", caja, "_caja")):
    filas = {}
    for k, p in mod(s).items():
        m = evaluar(e, p, e.index[idx_m]); a = (e["z1"] - p).abs()
        for g in ("Diamond", "Master", "GM+"): m[g] = a[idx_m & (grupo == g)].mean()
        m["mov>300"] = a[idx_m & mov].mean(); m["resto"] = a[idx_m & ~mov].mean()
        filas[k] = m
    print(f"\n{nom}  n={idx_m.sum()}"); print(pd.DataFrame(filas).T.round(3).to_string())
a0 = (e["z1"] - mod("_caja")["GRU del pipeline (antes)"]).abs()[caja]; a1 = (e["z1"] - P("final5", "_caja")).abs()[caja]
print(f"\ncaja: jugadores donde el promedio de 5 erra menos que la GRU de antes: {(a1 < a0).mean():.1%}; "
      f"error >150 LP: antes {(a0 > 150).mean():.1%}, ahora {(a1 > 150).mean():.1%}")
