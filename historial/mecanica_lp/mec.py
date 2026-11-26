"""Mecánicas del LP en LAS, medidas en vivo: parámetros compartidos por las pruebas."""
from pathlib import Path

import numpy as np
import pandas as pd

K = 20.0
BANDAS = [-.01, .02, .05, .1, .2, .3, .5, 1.0]
DIA = 86_400_000


def parametros(V: Path) -> dict:
    la = pd.read_pickle(V / "la_rico.pkl")
    l = la[~la.es_win]
    w = la[la.es_win & (la.tot_hist >= 30)].copy()
    w["bf"] = pd.cut(w.frac_hist, BANDAS, labels=False)
    td = w.groupby(["team_position", "bf"], observed=True).doble.mean().unstack()
    x = la[(la.lob_n >= 6) & ((la.es_win & la.dz.between(10, 31)) | (~la.es_win & la.dz.between(-30, -5)))]
    pj = x.groupby("puuid").agg(d=("delta", "mean"), gap=("gap", "mean"), n=("delta", "size"))
    pj = pj[pj.n >= 8]
    med = pj.groupby(pd.cut(pj.gap, [-3000, -400, -200, -100, 0, 100, 200, 400, 650, 3000]), observed=False).agg(g=("gap", "mean"), d=("d", "mean")).dropna()
    return {"perdon": ((l.dz == 0) & (l.z_antes != 400)).sum() / (l.z_antes != 400).sum(),
            "piso_quieto": ((l.z_antes == 400) & (l.dz == 0)).sum() / (l.z_antes == 400).sum(),
            "piso_baja": -l[(l.z_antes == 400) & (l.dz < 0)].dz.mean(),
            "td": {r: td.loc[r].to_numpy() for r in td.index},
            "doble": la[la.es_win].doble.mean(),
            "gx": med.g.to_numpy(), "dy": np.minimum(np.maximum.accumulate(med.d.to_numpy()), 10.0)}


def decay(z, dias):
    """Master+: 75 LP/día pasados 14 días, hasta 375. Diamond I: 50 LP/día pasados 28 días, hasta 275."""
    if z >= 400 and dias > 14:
        return max(z - 75 * (dias - 14), 375.0)
    if 300 <= z < 400 and dias > 28:
        return max(z - 50 * (dias - 28), 275.0)
    return z


def p_doble(P, pos, frac):
    if pos not in P["td"]:
        return P["doble"]
    return float(np.nan_to_num(P["td"][pos][np.digitize(frac, BANDAS[1:-1], right=True)], nan=P["doble"]))


def metricas(y, zi, pred, corte=None):
    y, zi, pred = np.asarray(y, float), np.asarray(zi, float), np.asarray(pred, float)
    err = pred - y
    d = y - zi
    return {"MAE": np.abs(err).mean(), "mediana": np.median(np.abs(err)), "RMSE": np.sqrt((err ** 2).mean()),
            "R2_cambio": 1 - (err ** 2).sum() / ((d - d.mean()) ** 2).sum(), "n": len(y)}
