"""Ablación hacia adelante (solo W/L observadas), y el puente contra datos en vivo."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 240); pd.set_option("display.max_columns", 40)
SP = Path(sys.argv[1]); sys.path.insert(0, str(Path(__file__).resolve().parent))
from mec import DIA, K, decay, metricas, p_doble, parametros  # noqa: E402

V = SP / "vivo"
P = parametros(V)
p = pd.read_pickle(V / "partes_lobby.pkl"); pob = pd.read_pickle(V / "pob.pkl"); ts = pd.read_pickle(V / "ts.pkl")
roles = pd.read_pickle(V / "roles.pkl").set_index(["player_id", "pos"]).frac
t1, t2 = int(ts.t1), int(ts.t2)
KD = 18.75
ag = {j: (h.win.to_numpy(), h.game_creation.to_numpy(), h.M.to_numpy(), h.pos.to_numpy()) for j, h in p[p.player_id.isin(pob.index)].groupby("player_id", sort=False)}


def recorrer(j, z0, dob=1, per=1, dl=1, pis=1, dec=1, kd=KD, dl_dia=1):
    z = float(z0)
    if j not in ag:
        return z if not dec else decay(z, (t2 - t1) / DIA)
    win, tt, M, pos = ag[j]
    t_prev = t1
    for k in range(len(win)):
        if dec:
            z = decay(z, (tt[k] - t_prev) / DIA)
        t_prev = tt[k]
        K_ = K if z >= 400 else kd
        d = float(np.interp(M[k] - z, P["gx"], P["dy"])) if (dl and not np.isnan(M[k]) and (dl_dia or z >= 400)) else 0.0
        if win[k]:
            z += (K_ + d) * (1 + (p_doble(P, pos[k], roles.get((j, pos[k]), 0.0)) if dob else 0.0))
        else:
            q = P["perdon"] if per else 0.0
            if pis and z >= 400 and z - (K_ - d) * (1 - q) < 400:
                z = 400.0 if z > 400.5 else z - (1 - P["piso_quieto"]) * P["piso_baja"]
            else:
                z -= (K_ - d) * (1 - q)
    return decay(z, (t2 - t_prev) / DIA) if dec else z


y = pob.z2.to_numpy(); zi = pob.z1.to_numpy()
grp = pob.tier2.map({"CHALLENGER": 0, "GRANDMASTER": 1, "MASTER": 2}).fillna(3).astype(int).to_numpy()
conf = {"Completo": {}, "sin dobles": dict(dob=0), "sin perdón": dict(per=0), "sin pisos": dict(pis=0), "sin δ(gap)": dict(dl=0), "sin decay": dict(dec=0),
        "sin δ en Diamond": dict(dl_dia=0), "K Diamond = 20": dict(kd=20.0), "K Diamond = 16,5": dict(kd=16.5), "solo K + decay": dict(dob=0, per=0, pis=0, dl=0),
        "solo K + δ + decay": dict(dob=0, per=0, pis=0)}
preds = {}
for nom, c in conf.items():
    preds[nom] = np.array([recorrer(j, z, **c) for j, z in pob.z1.items()])
print(pd.DataFrame({n: metricas(y, zi, v) for n, v in preds.items()}).T.round(3).to_string())
print("\nMAE por rango de la foto 2:")
nombres = ["C+GM", "Master", "Diamond"]
tab = {n: {"C+GM": np.abs(v - y)[grp <= 1].mean(), "Master": np.abs(v - y)[grp == 2].mean(), "Diamond": np.abs(v - y)[grp == 3].mean()} for n, v in preds.items()}
print(pd.DataFrame(tab).T.round(1).to_string())

# ¿Cuánto del error restante se debe a partidas ocultas, y qué pasa con los "limpios"?
c = preds["Completo"]
cob = (pob.obs / pob.dN.clip(lower=1)).to_numpy()
print("\nError del completo por cobertura:")
for a, b in [(0, .25), (.25, .5), (.5, .75), (.75, .95), (.95, 1.01)]:
    i = (cob >= a) & (cob < b)
    print(f"  cobertura [{a:.2f},{b:.2f}): n={i.sum():6d}  MAE {np.abs(c - y)[i].mean():6.1f}  RMSE {np.sqrt(((c - y)[i] ** 2).mean()):6.1f}")
pob_ = pob.assign(err=(c - y))
print("\nError según partidas entre fotos (cobertura ≥95 %):")
i = cob >= .95
n = pob.dN.to_numpy()
for a, b in [(1, 10), (10, 30), (30, 60), (60, 100), (100, 200), (200, 2000)]:
    k = i & (n >= a) & (n < b)
    print(f"  dN [{a},{b}): n={k.sum():5d}  MAE {np.abs(c - y)[k].mean():6.1f}  RMSE {np.sqrt(((c - y)[k] ** 2).mean()):6.1f}  sesgo {(c - y)[k].mean():+6.1f}")
print("\nEl 5 % que más se mueve (|Δz| mayor):")
dz = np.abs(y - zi); um = np.quantile(dz, .95)
print(f"  umbral {um:.0f}: MAE regla (obs) -, completo {np.abs(c - y)[dz >= um].mean():.1f}, resto {np.abs(c - y)[dz < um].mean():.1f}; % del error²: {((c - y)[dz >= um] ** 2).sum() / ((c - y) ** 2).sum():.3f}")

# PUENTE contra datos en vivo: dos anclas, error en las partidas del medio
print("\n=== Puente en vivo (LAS): anclas en primera y última lectura, error en el medio ===")
la = pd.read_pickle(V / "la_rico.pkl").sort_values(["puuid", "t_despues"])
la["pos"] = la.team_position
from zlib import crc32
filas = []
for pu, h in la.groupby("puuid", sort=False):
    h = h.reset_index(drop=True)
    # tramos contiguos: z_despues(k) == z_antes(k+1) y mismo estado de W/L
    corte = (h.z_antes.values[1:] != h.z_despues.values[:-1])
    ini = 0
    idx = list(np.where(corte)[0] + 1) + [len(h)]
    for fin in idx:
        if fin - ini >= 12:
            s = h.iloc[ini:fin]
            z = np.r_[s.z_antes.values, s.z_despues.values[-1]]
            win = s.es_win.values; n = len(s)
            fr = s.frac_hist.fillna(0).values; pos = s.pos.values
            gap_m = np.nanmean(s.lobby.values) - z[0]
            # 1. recta por partidas
            recta = np.linspace(z[0], z[-1], n + 1)
            # 2. regla con deriva, ajustada al cierre (a,b de LAS en Master, camino)
            pasos = np.where(win, 20.6, -19.5)
            r = np.r_[z[0], z[0] + np.cumsum(pasos)]
            regla = r + (z[-1] - r[-1]) * np.arange(n + 1) / n
            # 3. K=20 sin más, ajuste al cierre
            pasos = np.where(win, 20.0, -20.0); r = np.r_[z[0], z[0] + np.cumsum(pasos)]
            kk = r + (z[-1] - r[-1]) * np.arange(n + 1) / n
            # 4. mecánico: dobles esperados por rol + δ del cierre
            pd_ = np.array([p_doble(P, a, b) for a, b in zip(pos, fr)])
            pasos = np.where(win, 20.0 * (1 + pd_), -20.0 * (1 - P["perdon"])); r = np.r_[z[0], z[0] + np.cumsum(pasos)]
            mec = r + (z[-1] - r[-1]) * np.arange(n + 1) / n
            # sin anclar al final (solo arranque): deriva libre
            libre_regla = np.r_[z[0], z[0] + np.cumsum(np.where(win, 20.6, -19.5))]
            d_gap = float(np.interp(gap_m, P["gx"], P["dy"]))
            libre_mec = np.r_[z[0], z[0] + np.cumsum(np.where(win, (20.0 + d_gap) * (1 + pd_), -(20.0 - d_gap) * (1 - P["perdon"])))]
            for k in range(1, n):
                filas.append(dict(n=n, k=k, recta=abs(recta[k] - z[k]), regla=abs(regla[k] - z[k]), kk=abs(kk[k] - z[k]), mec=abs(mec[k] - z[k]),
                                  libre_regla=abs(libre_regla[k] - z[k]), libre_mec=abs(libre_mec[k] - z[k]), oracle=abs(mec[k] - z[k]), pu=pu))
        ini = fin
b = pd.DataFrame(filas)
print("tramos:", b.groupby(["pu", "n"]).ngroups, "jugadores", b.pu.nunique(), "puntos", len(b), "n mediana", b.groupby(["pu", "n"]).n.first().median())
print(b[["recta", "regla", "kk", "mec", "libre_regla", "libre_mec"]].mean().round(2).to_string())
print("por largo del tramo:")
print(b.groupby(pd.cut(b.n, [11, 15, 20, 30, 60]), observed=True)[["recta", "regla", "kk", "mec", "libre_regla", "libre_mec"]].mean().round(2).to_string())

# Varianza del paso: lo que aporta cada capa
la2 = la.copy()
la2["e_regla"] = la2.dz - np.where(la2.es_win, 21.5, -18.6)
print("\nVarianza por paso del ΔLP en LAS (LP²), todas las partidas enlazadas:")
print("  regla media:", round(la2.e_regla.var(), 1))
pd2 = np.array([p_doble(P, a, b) if t >= 30 else P["doble"] for a, b, t in zip(la2.team_position, la2.frac_hist.fillna(0), la2.tot_hist.fillna(0))])
ex = np.where(la2.es_win, 20 * (1 + pd2), -20 * (1 - P["perdon"]))
print("  con roles:", round((la2.dz - ex).var(), 1))
g = la2.groupby("puuid").delta.transform("median")
g = g.where(g.abs() <= 12, 0)
ex2 = np.where(la2.es_win, (20 + g) * (1 + pd2), -(20 - g) * (1 - P["perdon"]))
print("  con roles y δ del jugador (mediana propia):", round((la2.dz - ex2).var(), 1))
sinx = la2[(la2.dz.abs() > 5) & (la2.dz.abs() < 33) & (la2.z_antes > 420)]
print("  solo partidas normales, con δ propio:", round((sinx.dz - np.where(sinx.es_win, 20 + g[sinx.index], -20 + g[sinx.index])).var(), 2), "n", len(sinx))
print("Tasa por partida: dobles", round((la2.doble).mean(), 4), "ceros fuera del piso", round(((la2.dz == 0) & (la2.z_antes != 400)).mean(), 4))
