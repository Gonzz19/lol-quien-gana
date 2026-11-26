"""Adelante enmascarando, con las ocultas por su paso esperado: el escenario del pasado.

Se conoce el total de W y L de la foto, así que el saldo de las ocultas se sabe; su orden, no. Se mide
también qué pasa con un arranque malo: como la dinámica contrae, un error en el arranque se olvida.
"""
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 240); pd.set_option("display.max_columns", 40)
SP = Path(sys.argv[1]); ROOT = Path(sys.argv[2]).resolve(); sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(ROOT))
from mapa import KD, _decae, prob_doble  # noqa: E402
from mec import DIA, K, parametros  # noqa: E402


def recorrer(z0, win, tt, M, pdb, P, t1, t2, oc_por_vista, p_oc, usa_delta=True):
    """Una recorrida con ocultas repartidas entre las vistas. z0 es un vector de arranques."""
    z = np.atleast_1d(np.asarray(z0, float)).copy()
    b = np.where(z >= 400, 14.0, 28.0)
    t_prev = t1
    gx, dy, q, pq, pb, pd_m = P["gx"], P["dy"], P["perdon"], P["piso_quieto"], P["piso_baja"], P["doble"]
    for k in range(len(win) + 1):
        tk = tt[k] if k < len(win) else t2
        b = b - (tk - t_prev) / DIA
        t_prev = tk
        falta = b < 0
        if falta.any():
            for d in np.unique(np.floor(-b[falta] + 1e-9)):
                i = falta & (np.floor(-b + 1e-9) == d)
                z[i] = _decae(z[i], d)
            b[falta] = 0.0
        if k == len(win):
            break
        K_ = np.where(z >= 400, K, KD)
        d = np.interp(M[k] - z, gx, dy) if (usa_delta and M[k] == M[k]) else 0.0
        if win[k]:
            z = z + (K_ + d) * (1 + pdb[k])
        else:
            dest = z - (K_ - d) * (1 - q)
            dest = np.where((z > 400.5) & (dest < 400), 400.0, dest)
            z = np.where((z >= 400) & (z <= 400.5), z - (1 - pq) * pb, dest)
        if oc_por_vista > 0:
            K_ = np.where(z >= 400, K, KD)
            d = np.interp(M[k] - z, gx, dy) if (usa_delta and M[k] == M[k]) else 0.0
            z = z + oc_por_vista * (p_oc * (K_ + d) * (1 + pd_m) - (1 - p_oc) * (K_ - d) * (1 - q))
        b = np.where(z >= 400, np.minimum(b + 1.0 + oc_por_vista, 14.0), np.minimum(b + 7.0 * (1 + oc_por_vista), 28.0))
    return z


def trozo(lista, c, P, t1, t2, sesgo_ini):
    filas = []
    for j, zi, zf, dw, dl, win, tt, lob, pdb in lista:
        r = np.random.default_rng(zlib.crc32(f"{j}-1".encode()))
        keep = r.random(len(win)) < c
        if not keep.any():
            keep[r.integers(len(win))] = True
        w, t_, l, p_ = win[keep], tt[keep], lob[keep], pdb[keep]
        wb = w.astype(bool)
        M = pd.Series(l).rolling(31, center=True, min_periods=1).mean().to_numpy()
        nv = int(keep.sum()); n_oc = max(dw + dl - nv, 0); oc = n_oc / nv
        oc_w = np.clip(dw - wb.sum(), 0, n_oc)
        p_real = oc_w / n_oc if n_oc else .5
        a = recorrer(zi, w, t_, M, p_, P, t1, t2, oc, p_real)[0]
        b = recorrer(zi, w, t_, M, p_, P, t1, t2, oc, .5)[0]
        s = recorrer([zi - sesgo_ini, zi + sesgo_ini], w, t_, M, p_, P, t1, t2, oc, p_real)
        reg = zi + 20.6 * dw - 19.5 * dl
        sd = recorrer(zi, w, t_, M, p_, P, t1, t2, oc, p_real, usa_delta=False)[0]
        filas.append(dict(j=j, a=a, b=b, s_lo=s[0], s_hi=s[1], reg=reg, sin_delta=sd))
    return filas


if __name__ == "__main__":
    V = SP / "vivo"
    P = parametros(V)
    pa = pd.read_pickle(V / "partes_lobby.pkl"); pob = pd.read_pickle(V / "pob.pkl"); ts = pd.read_pickle(V / "ts.pkl")
    f2 = pd.read_pickle(V / "foto2.pkl"); rol = pd.read_pickle(V / "roles.pkl")
    t1, t2 = int(ts.t1), int(ts.t2)
    grp = pob.tier2.map({"CHALLENGER": 0, "GRANDMASTER": 1, "MASTER": 2}).fillna(3).astype(int)
    lim = pob[(pob.obs / pob.dN.clip(lower=1) >= .95) & (pob.dN >= 40) & (grp <= 2)]
    h = pa[pa.player_id.isin(lim.index)].merge(rol[["player_id", "pos", "frac"]], on=["player_id", "pos"], how="left").sort_values(["player_id", "game_creation"])
    h["pd"] = prob_doble(P, h.pos.to_numpy(), h.frac.fillna(0).to_numpy())
    gr = {j: (x.win.to_numpy(), x.game_creation.to_numpy(), x.lobby.to_numpy(), x.pd.to_numpy()) for j, x in h.groupby("player_id", sort=False)}
    ids = lim.index.to_numpy()
    tareas = np.array([(j, lim.z1[j], lim.z2[j], lim.dW[j], lim.dL[j], *gr[j]) for j in ids], dtype=object)
    y1, y2 = lim.z1.to_numpy(), lim.z2.to_numpy()
    print("jugadores:", len(ids), "partidas (mediana):", lim.dN.median())
    filas = []
    for c in (1.0, .75, .5, .25, .1, .05, .03):
        out = Parallel(n_jobs=10)(delayed(trozo)(list(t), c, P, t1, t2, 150.0) for t in np.array_split(tareas, 30))
        R = pd.DataFrame([x for o in out for x in o]).set_index("j").loc[ids]
        filas.append({"cobertura": c, "con saldo W/L conocido": np.abs(R.a - y2).mean(), "sin saldo (p=0,5)": np.abs(R.b - y2).mean(), "sin δ, con saldo": np.abs(R.sin_delta - y2).mean(),
                      "regla con W/L reales": np.abs(R.reg - y2).mean(), "arranque -150": np.abs(R.s_lo - y2).mean(), "arranque +150": np.abs(R.s_hi - y2).mean(),
                      "error de arranque que queda (de 300)": (R.s_hi - R.s_lo).abs().median()})
        print({k: round(float(v), 1) for k, v in filas[-1].items()}, flush=True)
    print(pd.DataFrame(filas).round(1).to_string())
