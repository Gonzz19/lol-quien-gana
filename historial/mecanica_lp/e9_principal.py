"""a) Adelante con banco, con ρ; b) inversión; c) ¿la mecánica le agrega al principal? d) ocultas.
Solo desarrollo; la caja cerrada queda afuera."""
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor as H

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 240); pd.set_option("display.max_columns", 40)
SP = Path(sys.argv[1]); ROOT = Path(sys.argv[2]).resolve(); sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(ROOT))
from mapa import KD, Z0, adelante_v, invertir, prob_doble  # noqa: E402
from mec import DIA, metricas, parametros  # noqa: E402


def uno(args, P, t1, t2):
    j, zi, zf, win, tt, lob_f2, M_f12, pos_d = args
    M = pd.Series(lob_f2).rolling(31, center=True, min_periods=1).mean().to_numpy() if len(win) else lob_f2
    F = adelante_v(win, tt, M, pos_d, P, t1, t2)
    z_inv, pend, ancho = invertir(F, zf)
    z_fwd = float(adelante_v(win, tt, M_f12, pos_d, P, t1, t2, z0=zi)[0])
    n = len(win)
    def media(x):
        return float(np.nanmean(x)) if len(x) and np.isfinite(x).any() else np.nan
    return dict(player_id=j, z_inv=z_inv, pend=pend, ancho=ancho, z_fwd=z_fwd, dob=float(pos_d[win.astype(bool)].sum()) if n else 0.0,
                M_ini=media(lob_f2[:10]), M_fin=media(lob_f2[-10:]), M_all=media(lob_f2), n_obs=n,
                hueco_max=float(np.max(np.diff(np.r_[t1, tt, t2])) / DIA), dias_fin=float((t2 - (tt[-1] if n else t1)) / DIA), dias_ini=float(((tt[0] if n else t2) - t1) / DIA))


def trozo(lista, P, t1, t2):
    return [uno(a, P, t1, t2) for a in lista]


if __name__ == "__main__":
    V = SP / "vivo"
    P = parametros(V)
    pa = pd.read_pickle(V / "partes_lobby.pkl"); pob = pd.read_pickle(V / "pob.pkl"); ts = pd.read_pickle(V / "ts.pkl")
    f2 = pd.read_pickle(V / "foto2.pkl"); rol = pd.read_pickle(V / "roles.pkl")
    PR = pd.read_csv(ROOT / "data" / "analysis" / "elo_principal_pred.csv", index_col="player_id")
    t1, t2 = int(ts.t1), int(ts.t2)
    # Lobby con la foto 2 de los otros (para ir hacia atrás) y M con las dos fotos (para ir hacia adelante)
    z2o = f2.loc[f2.tier != "UNRANKED", "z"]
    pa = pa.assign(zo=pa.player_id.map(z2o))
    g = pa.groupby("match_id").zo
    pa["lob_f2"] = (g.transform("sum") - pa.zo.fillna(0)) / (g.transform("count") - pa.zo.notna().astype(int)).replace(0, np.nan)
    pa = pa[pa.player_id.isin(pob.index)].merge(rol[["player_id", "pos", "frac"]], on=["player_id", "pos"], how="left").sort_values(["player_id", "game_creation"])
    pa["frac"] = pa.frac.fillna(0.0)
    pa["pd"] = prob_doble(P, pa.pos.to_numpy(), pa.frac.to_numpy())
    grupos = {j: (h.win.to_numpy(), h.game_creation.to_numpy(), h.lob_f2.to_numpy(), h.M.to_numpy(), h.pd.to_numpy()) for j, h in pa.groupby("player_id", sort=False)}
    vacio = (np.array([], bool), np.array([], np.int64), np.array([]), np.array([]), np.array([]))
    ids = pob.index.to_numpy()
    tareas = [(j, pob.z1[j], pob.z2[j], *grupos.get(j, vacio)) for j in ids]
    t0 = time.time()
    out = Parallel(n_jobs=10, batch_size=1)(delayed(trozo)(list(t), P, t1, t2) for t in np.array_split(np.array(tareas, dtype=object), 40))
    X = pd.DataFrame([x for o in out for x in o]).set_index("player_id").loc[ids]
    X.to_pickle(V / "X_mec.pkl")
    print(f"recorridas listas en {time.time() - t0:.0f} s", flush=True)

    y1 = pob.z1.to_numpy(); y2 = pob.z2.to_numpy(); pr = PR.loc[ids]
    grp = pob.tier2.map({"CHALLENGER": 0, "GRANDMASTER": 1, "MASTER": 2}).fillna(3).astype(int).to_numpy()

    def rho(v, y):
        return sum(spearmanr(v[grp == k], y[grp == k]).statistic * (grp == k).sum() for k in range(4)) / len(y)

    a_ = np.array([24.5, 23.3, 20.6, 16.5])[grp]; b_ = np.array([23.6, 22.4, 19.5, 16.3])[grp]
    regla_ad = y1 + a_ * pob.obs_w.to_numpy() - b_ * (pob.obs - pob.obs_w).to_numpy()
    print("\n=== Hacia ADELANTE: desde la foto 1 hasta la foto 2, solo W/L observadas ===")
    f = {"No cambió": y1, "Regla a,b (ajustada a estas fotos)": regla_ad, "Mecánica en vivo + banco": X.z_fwd.to_numpy()}
    print(pd.DataFrame({k: {**metricas(y2, y1, v), "rho": rho(v, y2)} for k, v in f.items()}).T.round(3).to_string())
    print("por rango (MAE):", {n: round(np.abs(X.z_fwd - y2)[grp == k].mean(), 1) for k, n in enumerate(["CH", "GM", "M", "D"])}, "sesgo", round((X.z_fwd - y2).mean(), 1))
    print("regla por rango (MAE):", {n: round(np.abs(regla_ad - y2)[grp == k].mean(), 1) for k, n in enumerate(["CH", "GM", "M", "D"])})

    print("\n=== Hacia ATRÁS: predecir la foto 1 desde la foto 2 ===")
    f = {"Regla": pr.regla.to_numpy(), "Inversión de la mecánica": X.z_inv.to_numpy(), "Grilla 29/10": pr.grilla.to_numpy(), "Principal": pr.principal.to_numpy()}
    print(pd.DataFrame({k: {**metricas(y1, y2, v), "rho": rho(v, y1)} for k, v in f.items()}).T.round(3).to_string())
    print("pendiente de F en la solución:", X.pend.describe(percentiles=[.1, .25, .5, .75, .9]).round(2).to_dict())
    print("error de la inversión según la pendiente:", {f"{a}-{b}": round(np.abs(X.z_inv - y1)[(X.pend >= a) & (X.pend < b)].mean(), 1) for a, b in [(0, .25), (.25, .5), (.5, .75), (.75, .95), (.95, 2)]})

    print("\n=== ¿La mecánica le agrega al principal? residuo, 5 folds por jugador, desarrollo ===")
    fold = np.array([zlib.crc32(str(i).encode()) % 5 for i in ids])
    F = pd.DataFrame({"princ": pr.principal, "z2": y2, "gr": pr.grilla - pr.principal, "reg": pr.regla - pr.principal, "n": X.n_obs,
                      "inv_dif": X.z_inv - pr.principal, "pend": X.pend, "ancho": X.ancho, "dob": X.dob, "gap_ini": X.M_ini - y2, "gap_fin": X.M_fin - y2, "gap_all": X.M_all - y2,
                      "hueco": X.hueco_max, "d_fin": X.dias_fin, "d_ini": X.dias_ini}, index=ids)
    base = ["princ", "z2", "gr", "reg", "n"]
    res = {}
    for nom, cols in (("control: lo que ya tiene", base), ("control + inactividad", base + ["hueco", "d_fin", "d_ini"]), ("con la mecánica", base + ["hueco", "d_fin", "d_ini", "inv_dif", "pend", "ancho", "dob", "gap_ini", "gap_fin", "gap_all"])):
        oof = np.zeros(len(ids))
        for k in range(5):
            m = H(loss="absolute_error", max_iter=250, learning_rate=.04, max_leaf_nodes=15, min_samples_leaf=60, early_stopping=False, random_state=0)
            tr = fold != k
            m.fit(F.loc[tr, cols], (y1 - pr.principal.to_numpy())[tr])
            oof[~tr] = m.predict(F.loc[~tr, cols])
        v = pr.principal.to_numpy() + oof
        res[nom] = v
        print(f"{nom:28s} MAE {np.abs(v - y1).mean():6.2f}  RMSE {np.sqrt(((v - y1) ** 2).mean()):6.2f}  R² {metricas(y1, y2, v)['R2_cambio']:.3f}  ρ {rho(v, y1):.3f}", flush=True)
    p_ = pr.principal.to_numpy()
    print(f"{'Principal':28s} MAE {np.abs(p_ - y1).mean():6.2f}  RMSE {np.sqrt(((p_ - y1) ** 2).mean()):6.2f}  R² {metricas(y1, y2, p_)['R2_cambio']:.3f}  ρ {rho(p_, y1):.3f}")
    v = res["con la mecánica"]
    print("MAE por rango, mecánica contra principal:", {n: (round(np.abs(v - y1)[grp == k].mean(), 1), round(np.abs(p_ - y1)[grp == k].mean(), 1)) for k, n in enumerate(["CH", "GM", "M", "D"])})
