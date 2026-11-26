"""Brainstorming (docs/BRAINSTORM_NOCHE.md): modelo combinado con lo aprobado en la noche.

Escalera sobre el híbrido + mapa entrenado solo con Master+:
  + margen por campeón y línea (Claude, idea 11)
  + nivel histórico (Claude, idea 16)
  + W/L de la foto 1 dinámico (Gemini, idea 10; `brainstorm_g_01_foto1_wl.construir_extras`)
  + agregados + autofill/smurf (Gemini, idea 13; `brainstorm_g_03_agregados.construir_bloques`)
Cada escalón se compara contra el anterior.
"""
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.brainstorm_base import REPO, ajustar_predecir, cargar, comparar, evaluar, mostrar  # noqa: E402
from historial.escalon9.brainstorm_g_01_foto1_wl import construir_extras  # noqa: E402
from historial.escalon9.brainstorm_g_03_agregados import construir_bloques  # noqa: E402

D = cargar()
NM, ms, post = D["NM"], D["master"], D["post"]
C = np.load(REPO / "data" / "analysis" / "brainstorm" / "margen_grupos.npz")
MCML = np.c_[(C["MC"][:, 0] - C["MC"][:, 1]).reshape(NM, -1), (C["ML"][:, 0] - C["ML"][:, 1]).reshape(NM, -1)]
A = np.load(REPO / "data" / "analysis" / "brainstorm" / "mmr_historico.npz")["A"]
NIV = (A[:, 0] - A[:, 1])[..., :3].mean(axis=1)
W1 = construir_extras(D)["FOTO1_DIN"]                       # solo post 04/09 (0 antes)
BL = construir_bloques(D, C["MC"], C["ML"])
AGR = np.c_[BL["AGREGADOS"], BL["AUTOFILL_SMURF"]]


def modelo(*bloques):
    X = np.hstack(bloques).astype(np.float32)
    return lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=X, extra_post=False)


ESC = {
    "1. + margen campeón/línea": modelo(MCML),
    "2. + nivel histórico": modelo(MCML, NIV),
    "3. + W/L foto 1 dinámico": modelo(MCML, NIV, W1),
    "4. + agregados y autofill": modelo(MCML, NIV, W1, AGR),
}
P = dict(Parallel(n_jobs=2, backend="threading")(delayed(lambda n: (n, evaluar(D, ESC[n])))(n) for n in ESC))
nombres = list(ESC)
for a, b in zip(nombres[:-1], nombres[1:]):
    print(f"\n--- {b}  contra  {a}")
    mostrar(comparar(D, {a: P[a], b: P[b]}, a).iloc[1:])
print("\n--- todos contra el escalón 1")
mostrar(comparar(D, P, nombres[0]))
np.savez_compressed(REPO / "data" / "analysis" / "brainstorm" / "pred_combinado.npz",
                    **{f"v{i}": P[n][1] for i, n in enumerate(nombres)}, **{f"o{i}": P[n][0] for i, n in enumerate(nombres)})
