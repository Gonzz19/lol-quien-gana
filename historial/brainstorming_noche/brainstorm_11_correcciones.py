"""Brainstorming (docs/BRAINSTORM_NOCHE.md): las dos correcciones que Gemini marcó en la revisión de
brainstorm_07/08, medidas sobre el escalón 3 (V 0,6663).

1. `mu0` del nivel histórico calculado con lo anterior a V (`~val0`) en vez de `fit` (que incluía
   las Diamond de la ventana de V).
2. W/L de la foto 1 estandarizado con las filas de entrenamiento posteriores a la foto (vale 0 antes),
   en vez de con todas.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.brainstorm_base import REPO, ajustar_predecir, cargar, comparar, evaluar, mostrar  # noqa: E402
from historial.escalon9.brainstorm_g_01_foto1_wl import construir_extras  # noqa: E402
import scripts.analysis.ganador_mapa_lp as gm  # noqa: E402
from src.features.lobby import prior_lobby_level  # noqa: E402

D = cargar()
NM, ms = D["NM"], D["master"]
C = np.load(REPO / "data" / "analysis" / "brainstorm" / "margen_grupos.npz")
MCML = np.c_[(C["MC"][:, 0] - C["MC"][:, 1]).reshape(NM, -1), (C["ML"][:, 0] - C["ML"][:, 1]).reshape(NM, -1)].astype(np.float32)
A = np.load(REPO / "data" / "analysis" / "brainstorm" / "mmr_historico.npz")["A"]
NIV = (A[:, 0] - A[:, 1])[..., :3].mean(axis=1).astype(np.float32)
W1 = construir_extras(D)["FOTO1_DIN"]

foto1, _ = gm._cargar_foto1(gm.RUTA_DB)
df = pd.DataFrame({"match_id": np.repeat(D["Z"]["mids"], 10), "game_creation": np.repeat(D["t"], 10),
                   "team_id": np.tile(np.repeat([100, 200], 5), NM), "lane": np.tile(np.arange(5), 2 * NM),
                   "player_id": D["B"]["player"].reshape(-1)})
df["x"] = df.player_id.map(foto1["z"].to_dict())
mu0 = float(df.loc[np.repeat(~D["val0"], 10), "x"].mean())
mu0_viejo = float(df.loc[np.repeat(D["fit"], 10), "x"].mean())
print(f"mu0 con ~val0: {mu0:.2f}  (antes, con fit: {mu0_viejo:.2f})")
N = prior_lobby_level(df, "x", k=5, mu0=mu0, window=10, min_otros=3)
A2 = np.c_[N.nivel_acum - mu0, N.nivel_v10 - mu0, np.log1p(N.n_nivel)].astype(np.float32).reshape(NM, 2, 5, -1)
NIV2 = (A2[:, 0] - A2[:, 1]).mean(axis=1)


def m(bloques):
    return lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=bloques)


VAR = {"escalón 3 (como estaba)": m([(np.c_[MCML, NIV, W1], False)]),
       "1. mu0 con ~val0": m([(np.c_[MCML, NIV2, W1], False)]),
       "2. W/L foto estandarizado en post": m([(np.c_[MCML, NIV], False), (W1, True)]),
       "1 + 2": m([(np.c_[MCML, NIV2], False), (W1, True)])}
P = dict(Parallel(n_jobs=2, backend="threading")(delayed(lambda n: (n, evaluar(D, VAR[n])))(n) for n in VAR))
mostrar(comparar(D, P, "escalón 3 (como estaba)"))
