"""Idea 25 del brainstorming (docs/BRAINSTORM_NOCHE.md): pesos que dependen del tipo de lobby, sin partir los datos.

Los modelos por segmento (brainstorm_13) empeoraron porque reducen N. Acá, sobre el escalón 4, se agregan
productos  (diferencia azul − rojo de un resumen) × (rasgo simétrico del lobby), que siguen siendo
antisimétricos (invertir lados invierte el signo) y dejan que la pendiente de cada resumen cambie con el
lobby, entrenando con todo:

* resúmenes (azul − rojo, media por equipo): margen acumulado compuesto, margen por campeón compuesto,
  nivel histórico, W/L de la foto (winrate), experiencia con el campeón (log partidas), estrenos, matchups (suma);
* rasgos del lobby (simétricos): veteranía (mín. de log partidas previas), estrenos totales, nivel
  histórico medio del lobby, madrugada (0-6 h ART).
Todo previo a la partida.
"""
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.brainstorm_base import REPO, ajustar_predecir, cargar, comparar, evaluar, extras_referencia, mostrar  # noqa: E402

D = cargar()
NM, ms = D["NM"], D["master"]
R4 = extras_referencia(D, escalon=4)
Xj, c = D["Xj"], {k: i for i, k in enumerate(D["cols"])}
iac = [i for i, k in enumerate(D["cols"]) if k.startswith("acum_")]
C = np.load(REPO / "data" / "analysis" / "brainstorm" / "margen_grupos.npz")
A = np.load(REPO / "data" / "analysis" / "brainstorm" / "mmr_historico.npz")["A"]
eqd = lambda a: a[:, 0].mean(1) - a[:, 1].mean(1)  # noqa: E731
RES = np.c_[eqd(Xj[..., iac].mean(-1)), eqd(C["MC"].mean(-1)), eqd(A[..., 0]), R4[:, 153], eqd(Xj[..., c["logn_champ"]]),
            eqd(Xj[..., c["estreno"]]), R4[:, 157:162].sum(1)].astype(np.float32)
hora = ((D["t"] / 3.6e6) - 3) % 24
LOB = np.c_[Xj[..., c["logn_tot"]].reshape(NM, 10).min(1), Xj[..., c["estreno"]].reshape(NM, 10).sum(1),
            A[..., 0].reshape(NM, 10).mean(1), (hora < 6).astype(float)].astype(np.float32)
RES_z = (RES - RES[D["fi_v"]].mean(0)) / (RES[D["fi_v"]].std(0) + 1e-6)
LOB_z = (LOB - LOB[D["fi_v"]].mean(0)) / (LOB[D["fi_v"]].std(0) + 1e-6)
INT = np.column_stack([RES_z[:, i] * LOB_z[:, j] for i in range(RES.shape[1]) for j in range(LOB.shape[1])]).astype(np.float32)
INT_vet = np.column_stack([RES_z[:, i] * LOB_z[:, 0] for i in range(RES.shape[1])]).astype(np.float32)


def m(extra=None):
    X = R4 if extra is None else np.c_[R4, extra]
    return lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=X, extra_post=False)


VAR = {"escalón 4": m(), "+ resúmenes × veteranía (7)": m(INT_vet), "+ resúmenes × 4 rasgos del lobby (28)": m(INT),
       "+ ídem + rasgos del lobby solos": m(np.c_[INT, LOB])}
P = dict(Parallel(n_jobs=2, backend="threading")(delayed(lambda n: (n, evaluar(D, VAR[n])))(n) for n in VAR))
mostrar(comparar(D, P, "escalón 4"))
