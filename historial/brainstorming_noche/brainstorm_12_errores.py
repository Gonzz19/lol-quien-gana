"""Brainstorming (docs/BRAINSTORM_NOCHE.md): ¿dónde falla el escalón 3 en V Master+? Guía para ideas nuevas.

AUC y acierto por subgrupos de partidas (todo medido antes de la partida): cuántos jugadores sin
foto, mínimo de partidas previas del lobby, estrenos, fuera de rol, presencia de GM/Challenger,
hora, diferencia de nivel histórico entre equipos, y si el favorito del draft coincide con el de
los jugadores.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.brainstorm_base import REPO, ajustar_predecir, cargar, extras_referencia  # noqa: E402

D = cargar()
NM, ms = D["NM"], D["master"]
REF = extras_referencia(D, escalon=int(sys.argv[1]) if len(sys.argv) > 1 else 3)
vi = np.flatnonzero(D["val"])
tr = D["fi_v"][ms[D["fi_v"]]]
p = ajustar_predecir(D, tr, vi, extra=REF, extra_post=False)
p_draft = ajustar_predecir(D, tr, vi, con_mapa=False, extra=None) if False else None
y = D["y"][vi]
Xj, cols = D["Xj"][vi], D["cols"]
c = {k: i for i, k in enumerate(cols)}
J = D["J"][vi]
tn = D["Z"]["tnum"][vi]
n_prev = np.expm1(Xj[..., c["logn_tot"]]).reshape(len(vi), 10)
sinfoto = (J[..., 6] < 0.5).reshape(len(vi), 10).sum(1)
estr = Xj[..., c["estreno"]].reshape(len(vi), 10).sum(1)
off = Xj[..., c["offrole"]].reshape(len(vi), 10).sum(1)
gm = (np.nan_to_num(tn, nan=0) >= 2).sum(1)
hora = ((D["t"][vi] / 3.6e6) - 3) % 24
A = np.load(REPO / "data" / "analysis" / "brainstorm" / "mmr_historico.npz")["A"][vi]
dniv = np.abs(A[:, 0, :, 0].mean(1) - A[:, 1, :, 0].mean(1))
G = pd.DataFrame({"p": p, "y": y,
                  "sin foto en el lobby": pd.cut(sinfoto, [-1, 0, 1, 2, 10], labels=["0", "1", "2", "3+"]),
                  "mín. partidas previas": pd.cut(n_prev.min(1), [-1, 4, 19, 49, 1e9], labels=["0-4", "5-19", "20-49", "50+"]),
                  "estrenos en el lobby": pd.cut(estr, [-1, 0, 2, 4, 10], labels=["0", "1-2", "3-4", "5+"]),
                  "fuera de rol": pd.cut(off, [-1, 0, 1, 10], labels=["0", "1", "2+"]),
                  "GM/Chall en el lobby": pd.cut(gm, [-1, 0, 2, 10], labels=["0", "1-2", "3+"]),
                  "hora": pd.cut(hora, [0, 6, 12, 18, 24], right=False, labels=["0-6", "6-12", "12-18", "18-24"]),
                  "|Δ nivel histórico|": pd.qcut(dniv, 4, labels=["Q1 (parejo)", "Q2", "Q3", "Q4 (disparejo)"])})


def met(g):
    if g.y.nunique() < 2 or len(g) < 200:
        return pd.Series({"partidas": len(g), "AUC": np.nan, "acierto": np.nan})
    return pd.Series({"partidas": len(g), "AUC": roc_auc_score(g.y, g.p), "acierto": ((g.p > 0.5) == g.y).mean()})


print(f"V Master+ global: AUC {roc_auc_score(y, p):.4f}, acierto {((p > 0.5) == y).mean():.1%}")
for col in G.columns[2:]:
    t = G.groupby(col, observed=True)[["p", "y"]].apply(met)
    t["partidas"] = t.partidas.astype(int)
    print(f"\n{col}\n" + t.to_string(formatters={"AUC": "{:.4f}".format, "acierto": "{:.1%}".format}))
