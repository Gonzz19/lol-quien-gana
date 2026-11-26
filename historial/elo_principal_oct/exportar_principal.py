"""Exporta al repo (data/analysis/elo_principal_pred.csv) las predicciones fuera de la muestra de cada paso
del modelo principal: 5 folds en el desarrollo (85 %) y, en la caja (15 %), el modelo entrenado con todo el
desarrollo. LP absolutos."""
import pickle
from pathlib import Path
import numpy as np, pandas as pd
from comun import AQUI, cargar, evaluar
c = cargar(); e = c["e"]; caja = np.asarray(c["caja"]); fold = np.asarray(c["fold"])
DEST = Path(r"C:/Users/DarkA/OneDrive/Escritorio/Facultad/3er año/ML/TPF/ProyectoFinalML/data/analysis/elo_principal_pred.csv")


def P(t):
    return np.where(caja, np.load(AQUI / "preds" / f"{t}_caja.npy"), np.load(AQUI / "preds" / f"{t}.npy"))


def grilla(tag, col):
    s = pickle.load(open(AQUI / f"{tag}.pkl", "rb"))
    out = s["todo"][col].to_numpy().copy()
    for k in range(5):
        m = (~caja) & (fold == k)
        out[m] = s[k][col].to_numpy()[m]
    return out


COL = {"regla": c["regla"].to_numpy(),
       "grilla_sin_lobby": grilla("grid_l0", "mediana"), "grilla": grilla("grid_l2.0", "mediana"),
       "grilla_p10": grilla("grid_l2.0", "p10"), "grilla_p90": grilla("grid_l2.0", "p90"),
       "catboost": P("cat_huber"), "catboost_grilla": P("cat_huber+grid_l2.0"),
       "gru_5": P("est_plus_s5"), "gru_cat_grilla": P("g_cg2_s5"),
       "cat_gru_grilla": P("cat_bag5_rsm+an_gru+grid_l2.0"), "cat100_grilla": P("cat_bag5_h100+grid_l2.0"),
       "tabpfn": P("tabpfn_mediana"), "tabpfn_grilla": P("tabpfn_g2_mediana")}
cinco = ["gru_5", "gru_cat_grilla", "cat_gru_grilla", "cat100_grilla", "tabpfn_grilla"]
COL["principal"] = np.mean([COL[k] for k in cinco], axis=0)
assert np.allclose(COL["principal"], P("final5"), atol=1e-6), "el promedio no coincide con final5"
df = pd.DataFrame(COL, index=e.index); df.index.name = "player_id"; df.insert(0, "en_caja", caja)
assert df.notna().all().all()
df.to_csv(DEST, float_format="%.2f")
for k in df.columns[1:]:
    m = evaluar(e, df[k]); print(f"{k:<18} MAE {m['MAE']:.1f} RMSE {m['RMSE']:.1f}")
print(DEST, df.shape)
