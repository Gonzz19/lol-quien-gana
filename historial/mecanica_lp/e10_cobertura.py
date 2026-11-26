"""¿Cuánto cuesta no tener las partidas? Dos pruebas enmascarando partidas, con la verdad conocida:

1) Punta: jugadores casi completos entre fotos. Se ocultan partidas al azar y se predice la otra foto.
2) Camino: tramos en vivo de LAS con LP real en cada partida. Con anclas en las puntas, ¿cuánto se erra
   el LP de las partidas que sí se ven, según cuántas se ocultan? Mide la promesa del mapa."""
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor as H

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 240); pd.set_option("display.max_columns", 40)
SP = Path(sys.argv[1]); ROOT = Path(sys.argv[2]).resolve(); sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(ROOT))
from mapa import Z0, adelante_v, invertir, prob_doble  # noqa: E402
from mec import DIA, parametros  # noqa: E402

COBS = (1.0, .75, .5, .25, .1, .05, .03)


def enmascarar(lista, c, P, t1, t2, seed):
    filas = []
    for j, zi, zf, win, tt, lob, pdb in lista:
        r = np.random.default_rng(zlib.crc32(f"{j}-{seed}".encode()))
        keep = r.random(len(win)) < c
        if not keep.any():
            keep[r.integers(len(win))] = True
        w, t_, l, p_ = win[keep], tt[keep], lob[keep], pdb[keep]
        M = pd.Series(l).rolling(31, center=True, min_periods=1).mean().to_numpy()
        tasa = (1 - c) / c    # el banco de días ve las partidas ocultas por su tasa
        F = adelante_v(w, t_, M, p_, P, t1, t2, tasa_oc=tasa)
        z_inv = invertir(F, zf)[0]
        z_fwd = float(np.interp(zi, Z0, F))
        wb = w.astype(bool)
        tercio = max(1, len(l) // 3)
        filas.append(dict(j=j, n=int(keep.sum()), z_fwd=z_fwd, z_inv=z_inv, net=int(wb.sum()) - int((~wb).sum()), g=int(wb.sum()), p=int((~wb).sum()),
                          lob=np.nanmean(l) if np.isfinite(l).any() else np.nan, lobv=np.nanmean(l[:tercio]) if np.isfinite(l[:tercio]).any() else np.nan,
                          lobn=np.nanmean(l[-tercio:]) if np.isfinite(l[-tercio:]).any() else np.nan, ini=(t_[0] - t1) / DIA, fin=(t2 - t_[-1]) / DIA))
    return filas


if __name__ == "__main__":
    V = SP / "vivo"
    P = parametros(V)
    pa = pd.read_pickle(V / "partes_lobby.pkl"); pob = pd.read_pickle(V / "pob.pkl"); ts = pd.read_pickle(V / "ts.pkl")
    f2 = pd.read_pickle(V / "foto2.pkl"); rol = pd.read_pickle(V / "roles.pkl")
    t1, t2 = int(ts.t1), int(ts.t2)
    z2o = f2.loc[f2.tier != "UNRANKED", "z"]
    pa = pa.assign(zo=pa.player_id.map(z2o))
    g = pa.groupby("match_id").zo
    pa["lob_f2"] = (g.transform("sum") - pa.zo.fillna(0)) / (g.transform("count") - pa.zo.notna().astype(int)).replace(0, np.nan)
    grp = pob.tier2.map({"CHALLENGER": 0, "GRANDMASTER": 1, "MASTER": 2}).fillna(3).astype(int)
    lim = pob[(pob.obs / pob.dN.clip(lower=1) >= .95) & (pob.dN >= 40) & (grp <= 2)]
    print("=== 1. Punta: Master+ con cobertura ≥95 % y ≥40 partidas ===")
    print("jugadores:", len(lim), "partidas (mediana):", lim.dN.median())
    h = pa[pa.player_id.isin(lim.index)].merge(rol[["player_id", "pos", "frac"]], on=["player_id", "pos"], how="left").sort_values(["player_id", "game_creation"])
    h["pd"] = prob_doble(P, h.pos.to_numpy(), h.frac.fillna(0).to_numpy())
    gr = {j: (x.win.to_numpy(), x.game_creation.to_numpy(), x.lob_f2.to_numpy(), x.pd.to_numpy()) for j, x in h.groupby("player_id", sort=False)}
    ids = lim.index.to_numpy()
    tareas = np.array([(j, lim.z1[j], lim.z2[j], *gr[j]) for j in ids], dtype=object)
    y1, y2 = lim.z1.to_numpy(), lim.z2.to_numpy()
    filas = []
    for c in COBS:
        out = Parallel(n_jobs=10)(delayed(enmascarar)(list(t), c, P, t1, t2, 1) for t in np.array_split(tareas, 30))
        R = pd.DataFrame([x for o in out for x in o]).set_index("j").loc[ids]
        regla = y2 - (20.6 * R.g - 19.5 * R.p)
        regla_esc = y2 - (20.6 * R.g - 19.5 * R.p) / c
        Xg = pd.DataFrame({"z2": y2, "reg": regla_esc - y2, "inv": R.z_inv - y2, "n": R.n, "lob": R.lob - y2, "lobv": R.lobv - y2, "lobn": R.lobn - y2, "ini": R.ini, "fin": R.fin, "N2": lim.partidas2.to_numpy()}, index=ids)
        fold = np.array([zlib.crc32(str(i).encode()) % 5 for i in ids])
        oof = np.zeros(len(ids))
        for k in range(5):
            m = H(loss="absolute_error", max_iter=250, learning_rate=.05, max_leaf_nodes=15, min_samples_leaf=25, early_stopping=False, random_state=0)
            m.fit(Xg[fold != k], (y1 - y2)[fold != k])
            oof[fold == k] = m.predict(Xg[fold == k])
        filas.append({"cobertura": c, "partidas_vistas": R.n.median(), "ADELANTE: mecánica": np.abs(R.z_fwd - y2).mean(), "ATRÁS: regla escalada": np.abs(regla_esc - y1).mean(),
                      "ATRÁS: inversión": np.abs(R.z_inv - y1).mean(), "ATRÁS: GB (lobby + inversión)": np.abs(y2 + oof - y1).mean(), "No cambió": np.abs(y2 - y1).mean()})
        print({k: round(v, 1) for k, v in filas[-1].items()}, flush=True)
    print(pd.DataFrame(filas).round(1).to_string())

    print("\n=== 2. Camino: tramos en vivo de LAS, anclas en las puntas, error en las partidas vistas ===")
    la = pd.read_pickle(V / "la_rico.pkl").sort_values(["puuid", "t_despues"])
    tramos = []
    for pu, x in la.groupby("puuid", sort=False):
        x = x.reset_index(drop=True)
        cortes = list(np.where(x.z_antes.values[1:] != x.z_despues.values[:-1])[0] + 1) + [len(x)]
        ini = 0
        for fin in cortes:
            if fin - ini >= 12:
                s = x.iloc[ini:fin]
                tramos.append((pu, np.r_[s.z_antes.values, s.z_despues.values[-1]], s.es_win.values.astype(bool), prob_doble(P, s.team_position.values, s.frac_hist.fillna(0).values)))
            ini = fin
    print("tramos:", len(tramos), "partidas (mediana):", int(np.median([len(t[2]) for t in tramos])))
    r = np.random.default_rng(0)
    filas = []
    for c in COBS[:-1]:
        err_b, err_m, err_u, sig = [], [], [], []
        for rep in range(20 if c < 1 else 1):
            for pu, z, w, pdb in tramos:
                n = len(w)
                vis = r.random(n) < c
                if not vis.any():
                    continue
                # paso esperado: visto con su resultado, oculto con ~50 % (W/L del tramo conocidas por la foto)
                oc_w = w[~vis].sum(); oc_l = (~vis).sum() - oc_w
                # Los ocultos se reparten con el saldo real conocido, sin saber el orden
                p_oc = oc_w / max((~vis).sum(), 1)
                pasos_r = np.where(vis, np.where(w, 20.6, -19.5), p_oc * 20.6 - (1 - p_oc) * 19.5)
                pasos_m = np.where(vis, np.where(w, 20.0 * (1 + pdb), -20.0 * (1 - P["perdon"])), p_oc * 20.0 * (1 + P["doble"]) - (1 - p_oc) * 20.0 * (1 - P["perdon"]))
                rr = np.r_[z[0], z[0] + np.cumsum(pasos_r)]; mm = np.r_[z[0], z[0] + np.cumsum(pasos_m)]
                cam_r = rr + (z[-1] - rr[-1]) * np.arange(n + 1) / n
                cam_m = mm + (z[-1] - mm[-1]) * np.arange(n + 1) / n
                # solo donde el mapa tiene que dar un valor: antes de cada partida vista, sin las puntas
                idx = np.where(vis)[0]; idx = idx[(idx > 0)]
                if len(idx):
                    err_b.extend(np.abs(cam_r[idx] - z[idx])); err_m.extend(np.abs(cam_m[idx] - z[idx]))
        filas.append({"cobertura": c, "regla + cierre": np.mean(err_b), "mecánica + cierre": np.mean(err_m), "p90 mecánica": np.quantile(err_m, .9)})
        print({k: round(v, 2) for k, v in filas[-1].items()}, flush=True)
    print(pd.DataFrame(filas).round(2).to_string())
    # Var por paso medida, y su extrapolación a tramos largos con puente browniano
    qs = (20.0 ** 2 * 4 * .25)   # (2K)² p(1-p) con p = 0,5: var de un resultado desconocido
    print("\nExtrapolación del error en el medio del puente, σ = sqrt(n·q)/2 (q: por paso):")
    for nombre, q in (("todo visto, q=30", 30.0), ("todo identificado, q=1", 1.0), ("oculta, q=400", qs)):
        print(nombre, {n: round(np.sqrt(n * q) / 2, 1) for n in (25, 50, 100, 200, 300)})
