"""Prototipo: suavizado en grilla HACIA ATRÁS con las mecánicas medidas en vivo.

Mismo problema que el principal: conociendo la foto 2, nuestras partidas y el lobby (con la foto 2 de
los otros, sin foto 1 ni W/L de la foto 1), predecir el LP de la foto 1. Solo jugadores de desarrollo.

No invierte la dinámica: propaga el mensaje hacia atrás β_t(z) = Σ π·β_{t+1}(z'(z)). Una contracción
hacia adelante no explota: varios z llevan al mismo z', y β queda ancho, que es la respuesta honesta.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 240); pd.set_option("display.max_columns", 40)
SP = Path(sys.argv[1]); ROOT = Path(sys.argv[2]).resolve(); sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(ROOT))
from mec import DIA, K, metricas, p_doble, parametros  # noqa: E402

V = SP / "vivo"
P = parametros(V)
pa = pd.read_pickle(V / "partes.pkl"); pob = pd.read_pickle(V / "pob.pkl"); ts = pd.read_pickle(V / "ts.pkl")
f2 = pd.read_pickle(V / "foto2.pkl")
roles = pd.read_pickle(V / "roles.pkl").set_index(["player_id", "pos"]).frac
PR = pd.read_csv(ROOT / "data" / "analysis" / "elo_principal_pred.csv", index_col="player_id")
t1, t2 = int(ts.t1), int(ts.t2)
KD = 18.75

# Lobby con la foto 2 de los otros (nada de la foto 1): lo mismo que existe en el pasado
z2 = f2.loc[f2.tier != "UNRANKED", "z"]
pa = pa.assign(zo=pa.player_id.map(z2))
g = pa.groupby("match_id").zo
pa["lobby"] = (g.transform("sum") - pa.zo.fillna(0)) / (g.transform("count") - pa.zo.notna().astype(int)).replace(0, np.nan)
pa = pa.sort_values(["player_id", "game_creation"])
pa["M"] = pa.groupby("player_id").lobby.transform(lambda s: s.rolling(31, center=True, min_periods=1).mean())

# Calibración de la medición: lobby de las 3 partidas más nuevas contra z2 (sin el target), por tramos
ult = pa[pa.player_id.isin(pob.index)].groupby("player_id").tail(3)
ult = ult.merge(pob[["z2"]], left_on="player_id", right_index=True).dropna(subset=["lobby"])
cortes = [-1000, 100, 200, 300, 400, 500, 700, 1000, 1500, 4000]
cal = ult.groupby(pd.cut(ult.z2, cortes), observed=True).agg(zm=("z2", "mean"), lm=("lobby", "mean"), s=("lobby", "std"))
print("lobby medio según z:\n", cal.round(0).to_string())
ZC, LC, SC = cal.zm.to_numpy(), cal.lm.to_numpy(), cal.s.to_numpy()

GRID = np.arange(-600.0, 3405.0, 5.0)
S = len(GRID)
idx0 = lambda v: (np.asarray(v) - GRID[0]) / 5.0


def interp_estado(beta, z_dest):
    """β evaluado en z_dest con interpolación lineal; fuera de la grilla vale 0."""
    x = idx0(z_dest)
    i = np.clip(np.floor(x).astype(int), 0, S - 2)
    w = np.clip(x - i, 0, 1)
    return beta[i] * (1 - w) + beta[i + 1] * w


L_MU = np.interp(GRID, ZC, LC)
L_S = np.interp(GRID, ZC, SC)
NU = 4.0


def lik_lobby(l):
    r = (l - L_MU) / L_S
    return (1 + r * r / NU) ** (-(NU + 1) / 2) / L_S


def decay_vec(z, dias):
    out = z.copy()
    if dias > 14:
        m = z >= 400
        out[m] = np.maximum(z[m] - 75 * (dias - 14), 375.0)
    if dias > 28:
        m = (z >= 300) & (z < 400)
        out[m] = np.maximum(z[m] - 50 * (dias - 28), 275.0)
    return out


RHO = {0: .03, 1: .03, 2: .13, 3: 1.09}   # ocultas por observada, según el rango (tasa del documento)
KG = np.where(GRID >= 400, K, KD)


def hacia_atras(j, zf, h, grupo, cfg, prior_hist):
    """Mensaje hacia atrás hasta la foto 1. Devuelve β en t1 como vector sobre GRID."""
    if not len(h[0]):
        beta = np.zeros(S); beta[int(np.clip(round(idx0(zf)), 0, S - 1))] = 1.0
        pi = cfg["pi_oculta"][grupo]
        z = decay_vec(GRID, (t2 - t1) / DIA)
        return (1 - pi) * interp_estado(gaussian_filter1d(beta, 2.0), z) + pi * gaussian_filter1d(beta, 12.0)
    win, tt, M, lob, pos = h
    beta = np.zeros(S); beta[int(np.clip(round(idx0(zf)), 0, S - 1))] = 1.0
    beta = gaussian_filter1d(beta, 1.0)
    rho = RHO[grupo] if cfg["ocultas"] else 0.0
    t_sig = t2
    pi = cfg["pi_oculta"][grupo]
    for k in range(len(win) - 1, -1, -1):
        # 1) Hueco hasta el evento siguiente: decay o partidas ocultas (mezcla)
        dias = (t_sig - tt[k]) / DIA
        if dias > 14:
            dz = decay_vec(GRID, dias)
            beta = (1 - pi) * interp_estado(beta, dz) + pi * beta
        t_sig = tt[k]
        # 2) La partida, con δ que depende del estado (brecha con el MMR local)
        d = np.interp(M[k] - GRID, P["gx"], P["dy"]) if (cfg["delta"] and not np.isnan(M[k])) else np.zeros(S)
        if cfg["delta_solo_master"]:
            d = np.where(GRID >= 400, d, 0.0)
        pd_ = p_doble(P, pos[k], roles.get((j, pos[k]), 0.0)) if cfg["dobles"] else 0.0
        q = P["perdon"] if cfg["perdon"] else 0.0
        if win[k]:
            normal = interp_estado(beta, GRID + KG + d)
            nuevo = (1 - pd_) * normal + pd_ * interp_estado(beta, GRID + 2 * (KG + d)) if pd_ > 0 else normal
        else:
            dest = GRID - (KG - d)
            if cfg["pisos"]:
                dest = np.where((GRID > 400) & (dest < 400), 400.0, dest)
                en0 = np.abs(GRID - 400) < 2.5
                dest = np.where(en0, 375.0, dest)
                base = interp_estado(beta, dest)
                base[en0] = P["piso_quieto"] * interp_estado(beta, GRID)[en0] + (1 - P["piso_quieto"]) * base[en0]
            else:
                base = interp_estado(beta, dest)
            nuevo = (1 - q) * base + q * beta
        # 3) Ruido del paso y ocultas
        var = cfg["q"] + rho * KG.mean() ** 2
        beta = gaussian_filter1d(nuevo, np.sqrt(var) / 5.0, truncate=4.0) if var > 0 else nuevo
        # 4) Medición
        if cfg["tau"] > 0 and not np.isnan(lob[k]):
            beta = beta * lik_lobby(lob[k]) ** cfg["tau"]
        s = beta.max()
        if s > 0:
            beta = beta / s
    dias = (tt[0] - t1) / DIA
    if dias > 14:
        dz = decay_vec(GRID, dias)
        beta = (1 - pi) * interp_estado(beta, dz) + pi * beta
    return beta


def trabajar(ids, cfg, datos, grupos, zfs, prior):
    filas = []
    for j in ids:
        b = hacia_atras(j, zfs[j], datos.get(j, ([], [], [], [], [])), grupos[j], cfg, prior)
        filas.append(b)
    return np.array(filas, dtype=np.float32)


def resumir(B, prior, y_idx=None):
    post = B * prior
    post = post / post.sum(1, keepdims=True).clip(min=1e-300)
    media = post @ GRID
    cdf = post.cumsum(1)
    med = GRID[(cdf >= .5).argmax(1)]
    sd = np.sqrt(np.maximum(post @ GRID ** 2 - media ** 2, 0))
    return media, med, sd, post


if __name__ == "__main__":
    n_muestra = int(sys.argv[3]) if len(sys.argv) > 3 else 6000
    rng = np.random.default_rng(7)
    ids = rng.choice(pob.index.to_numpy(), size=min(n_muestra, len(pob)), replace=False)
    sel = pa[pa.player_id.isin(ids)]
    datos = {j: (h.win.to_numpy(), h.game_creation.to_numpy(), h.M.to_numpy(), h.lobby.to_numpy(), h.pos.to_numpy()) for j, h in sel.groupby("player_id", sort=False)}
    grp = pob.tier2.map({"CHALLENGER": 0, "GRANDMASTER": 1, "MASTER": 2}).fillna(3).astype(int)
    grupos = grp.to_dict(); zfs = pob.z2.to_dict()
    prior = np.histogram(z2.clip(GRID[0], GRID[-1]), bins=np.r_[GRID - 2.5, GRID[-1] + 2.5])[0].astype(float) + .5
    prior = gaussian_filter1d(prior, 4.0); prior /= prior.sum()
    base = dict(delta=1, dobles=1, perdon=1, pisos=1, ocultas=1, q=4.0, tau=1.0, delta_solo_master=0, pi_oculta={0: .3, 1: .3, 2: .3, 3: .7})
    cfgs = {"Mecánico completo (q=4, τ=1)": {},
            "q=1": dict(q=1.0), "q=16": dict(q=16.0), "q=36": dict(q=36.0), "q=100": dict(q=100.0),
            "τ=0": dict(tau=0.0), "τ=0,5": dict(tau=.5), "τ=2": dict(tau=2.0),
            "sin δ": dict(delta=0), "δ solo en Master+": dict(delta_solo_master=1), "sin dobles ni perdón": dict(dobles=0, perdon=0), "sin ocultas": dict(ocultas=0)}
    y = pob.loc[ids, "z1"].to_numpy(); zf = pob.loc[ids, "z2"].to_numpy()
    PRp = PR.loc[ids]
    res = {"No cambió (z2)": zf, "Regla (documento)": PRp.regla.to_numpy(), "Grilla 29/10": PRp.grilla.to_numpy(),
           "CatBoost + grilla": PRp.catboost_grilla.to_numpy(), "GRU, 5 semillas": PRp.gru_5.to_numpy(), "Principal (5 modelos)": PRp.principal.to_numpy()}
    betas = {}
    trozos = np.array_split(ids, 16)
    for nom, c in cfgs.items():
        cfg = {**base, **c}
        out = Parallel(n_jobs=8, backend="loky")(delayed(trabajar)(t, cfg, {j: datos[j] for j in t if j in datos}, grupos, zfs, prior) for t in trozos)
        B = np.vstack(out)
        media, med, sd, _ = resumir(B, prior)
        res[nom + " · media"] = media; res[nom + " · mediana"] = med
        betas[nom] = B
        print(nom, "MAE media", round(np.abs(media - y).mean(), 1), "mediana", round(np.abs(med - y).mean(), 1), "RMSE", round(np.sqrt(((media - y) ** 2).mean()), 1), flush=True)
    cut = (732.0, 1260.0)
    t = pd.DataFrame({k: metricas(y, zf, v) for k, v in res.items()}).T
    gr_ids = grp.loc[ids].to_numpy()
    rho = {}
    for k, v in res.items():
        vals = []
        for g_ in range(4):
            m = gr_ids == g_
            if m.sum() > 5:
                vals.append((spearmanr(v[m], y[m]).statistic, m.sum()))
        rho[k] = sum(a * b for a, b in vals) / sum(b for _, b in vals)
    t["rho_dentro"] = pd.Series(rho)
    print(f"\n=== Hacia atrás, {len(ids)} jugadores de desarrollo: predecir z1 desde z2 ===\n", t.round(3).to_string())

    # Combinar: el principal como medición del LP en la foto 1 (fuera de fold), el mensaje del mecánico como verosimilitud
    print("\n=== Principal como prior y mensaje mecánico ===")
    pr = PRp.principal.to_numpy()
    mejor = "Mecánico completo (q=4, τ=1)"
    B = betas[mejor]
    for nom in [mejor, "q=16", "q=36", "τ=0", "τ=0,5"]:
        B = betas[nom]
        for s_p in (40, 80, 120, 200):
            pn = np.exp(-.5 * ((GRID[None, :] - pr[:, None]) / s_p) ** 2)
            post = B * pn
            post = post / post.sum(1, keepdims=True).clip(min=1e-300)
            m = post @ GRID
            print(f"{nom:32s} σ_p={s_p:3d}: MAE {np.abs(m - y).mean():6.1f}  RMSE {np.sqrt(((m - y) ** 2).mean()):6.1f}  R² {metricas(y, zf, m)['R2_cambio']:.3f}", flush=True)
    for w in (.25, .5, .75):
        m = w * res[mejor + " · media"] + (1 - w) * pr
        print(f"promedio {w:.2f} mecánico + principal: MAE {np.abs(m - y).mean():.1f}  RMSE {np.sqrt(((m - y) ** 2).mean()):.1f}  R² {metricas(y, zf, m)['R2_cambio']:.3f}")
    np.save(V / "ids_atras.npy", ids); pd.DataFrame(res, index=ids).to_pickle(V / "res_atras.pkl")
