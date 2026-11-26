"""¿Las mecánicas medidas en vivo (nov) explican el cambio entre fotos (sep-oct), que nunca vieron?

Se recorre HACIA ADELANTE desde la foto 1 y se compara con la foto 2. Es la dirección estable:
sirve para ver si la dinámica es correcta, antes de pensar en invertirla. Solo jugadores de
desarrollo: la caja cerrada queda afuera.
"""
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 240); pd.set_option("display.max_columns", 40)
SP = Path(sys.argv[1]); ROOT = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(ROOT))
from src.features.elo_reverso import ajustar_regla  # noqa: E402

V = SP / "vivo"
p = pd.read_pickle(V / "partes.pkl"); e = pd.read_pickle(V / "e.pkl")
f1 = pd.read_pickle(V / "foto1.pkl"); f2 = pd.read_pickle(V / "foto2.pkl"); ts = pd.read_pickle(V / "ts.pkl")
t1, t2 = int(ts.t1), int(ts.t2)
la = pd.read_pickle(V / "la_rico.pkl")

en_caja = np.array([zlib.crc32(("caja-" + str(i)).encode()) % 100 < 15 for i in e.index])
print("caja", en_caja.sum(), "desarrollo", (~en_caja).sum())
dev = e[~en_caja].copy()

# ---- Lobby de cada fila: LP de los otros 9 interpolado en el tiempo entre sus fotos
z1 = f1["z"]; z2 = f2.loc[f2.tier != "UNRANKED", "z"]
tf = (p.game_creation - t1) / (t2 - t1)
a = p.player_id.map(z1); b = p.player_id.map(z2)
zt = a + (b - a) * tf
zt = zt.fillna(b).fillna(a)
p = p.assign(zt=zt)
g = p.groupby("match_id").zt
p["lobby"] = (g.transform("sum") - p.zt.fillna(0)) / (g.transform("count") - p.zt.notna().astype(int)).replace(0, np.nan)
p = p.sort_values(["player_id", "game_creation"])
# MMR local: media móvil del lobby del jugador (±15 partidas observadas)
p["M"] = p.groupby("player_id").lobby.transform(lambda s: s.rolling(31, center=True, min_periods=1).mean())

# ---- Parámetros de las mecánicas, SOLO del en vivo (LAS)
K = 20.0
vivo_l = la[~la.es_win]
tasa_perdon = ((vivo_l.dz == 0) & (vivo_l.z_antes != 400)).sum() / (vivo_l.z_antes != 400).sum()
piso_quieto = ((vivo_l.z_antes == 400) & (vivo_l.dz == 0)).sum() / (vivo_l.z_antes == 400).sum()
piso_baja = -vivo_l[(vivo_l.z_antes == 400) & (vivo_l.dz < 0)].dz.mean()
print(f"derrota perdonada (0 LP, fuera del piso): {tasa_perdon:.4f} | en Master 0: se queda {piso_quieto:.3f}, si baja pierde {piso_baja:.1f}")

w = la[la.es_win & (la.tot_hist >= 30)].copy()
BANDAS = [-.01, .02, .05, .1, .2, .3, .5, 1.0]
w["bf"] = pd.cut(w.frac_hist, BANDAS, labels=False)
td = w.groupby(["team_position", "bf"], observed=True).doble.mean().unstack()
print("P(doble | victoria, rol, rareza) del en vivo:\n", td.round(3).to_string())
TD = {r: td.loc[r].to_numpy() for r in td.index}
p_base_doble = la[la.es_win].doble.mean()
print("doble medio:", round(p_base_doble, 4))

# δ(gap) por jugador, de los datos en vivo: gap = media del lobby - LP propio
x = la[(la.lob_n >= 6) & ((la.es_win & la.dz.between(10, 31)) | (~la.es_win & la.dz.between(-30, -5)))]
pj = x.groupby("puuid").agg(d=("delta", "mean"), gap=("gap", "mean"), n=("delta", "size"))
pj = pj[pj.n >= 8]
nud = np.array([-800, -300, -150, -50, 50, 150, 300, 500, 800], float)
cortes = [-3000, -400, -200, -100, 0, 100, 200, 400, 650, 3000]
med = pj.groupby(pd.cut(pj.gap, cortes), observed=False).agg(g=("gap", "mean"), d=("d", "mean"), n=("d", "size"))
print("δ(gap) por jugador, en vivo:\n", med.round(2).to_string())
gx = med.g.dropna().to_numpy(); dy = np.minimum.accumulate(med.d.dropna().to_numpy()[::-1])[::-1]  # monótona
dy = np.maximum.accumulate(med.d.dropna().to_numpy()); dy = np.minimum(dy, 10.0)
print("nodos δ(gap):", dict(zip(gx.round(0), dy.round(2))))
def delta_de(gap):
    return np.interp(gap, gx, dy)

# Diamond: K por fotos (único parámetro que no sale del en vivo; L1 con jugadores que siguen en Diamond)
dia = dev[(dev.z1 < 380) & (dev.z2 < 380) & (dev.z1 >= 0) & (dev.dW >= 0) & (dev.dL >= 0) & (dev.dN >= 10)]
def l1(K_): return np.abs((dia.z2 - dia.z1) - K_ * (dia.dW - dia.dL)).mean()
gr = np.arange(10, 30, .25); KD = gr[np.argmin([l1(k) for k in gr])]
print(f"K de Diamond por fotos (L1, {len(dia)} jugadores): {KD}  MAE {l1(KD):.1f}  | con 20: {l1(20):.1f}  | deriva por partida si se agrega: "
      f"{np.median((dia.z2 - dia.z1 - KD * (dia.dW - dia.dL)) / dia.dN):.2f}")

# ---- Recorrida hacia adelante
ag = dict(tuple(p[p.player_id.isin(dev.index)].groupby("player_id", sort=False)))
rol_tab = pd.read_pickle(V / "roles.pkl").set_index(["player_id", "pos"]).frac
rol_tot = pd.read_pickle(V / "roles.pkl").groupby("player_id").tot.first()

def decay(z, dias):
    """Master+: 75 LP/día pasados 14 días, hasta 375. Diamond I: 50 LP/día pasados ~28 días, hasta 275."""
    if z >= 400 and dias > 14:
        return max(z - 75 * (dias - 14), 375.0)
    if 300 <= z < 400 and dias > 28:
        return max(z - 50 * (dias - 28), 275.0)
    return z


def p_doble(pos, frac):
    if pos not in TD:
        return p_base_doble
    return np.nan_to_num(TD[pos][np.digitize(frac, BANDAS[1:-1], right=True)], nan=p_base_doble)

def recorrer(jug, f, usa_doble, usa_perdon, usa_delta, usa_piso, usa_decay, reales, Kd=KD, kmaster=K):
    h = ag.get(jug)
    z = float(f.z1)
    if h is None:
        n_oc_w = f.dW if reales else 0
        n_oc_l = f.dL if reales else 0
        return z + (kmaster if z >= 400 else Kd) * (n_oc_w - n_oc_l)
    win = h.win.to_numpy(); tiempo = h.game_creation.to_numpy(); M = h.M.to_numpy(); pos = h.pos.to_numpy()
    n = len(h)
    oc_w = max(f.dW - win.sum(), 0) if reales else 0
    oc_l = max(f.dL - (n - win.sum()), 0) if reales else 0
    oc_w_p, oc_l_p = oc_w / n, oc_l / n
    tot = rol_tot.get(jug, n)
    fr = {r: rol_tab.get((jug, r), 0.0) for r in set(pos)}
    t_prev = t1
    for k in range(n):
        if usa_decay:
            z = decay(z, (tiempo[k] - t_prev) / 86_400_000)
        t_prev = tiempo[k]
        K_ = kmaster if z >= 400 else Kd
        d = float(delta_de(M[k] - z)) if (usa_delta and not np.isnan(M[k])) else 0.0
        if usa_delta and z < 400:
            d = max(d, -1.0)
        ganar = K_ + d
        perder = K_ - d
        if win[k]:
            pd_ = p_doble(pos[k], fr[pos[k]]) if usa_doble else 0.0
            z += ganar * (1 + pd_)
        else:
            q = tasa_perdon if usa_perdon else 0.0
            if usa_piso and z >= 400 and z - perder * (1 - q) < 400:
                z = 400.0 if z > 400.5 else z - (1 - piso_quieto) * piso_baja
            else:
                z -= perder * (1 - q)
        if oc_w_p + oc_l_p > 0:
            K_ = kmaster if z >= 400 else Kd
            d = float(delta_de(M[k] - z)) if (usa_delta and not np.isnan(M[k])) else 0.0
            q = tasa_perdon if usa_perdon else 0.0
            z += oc_w_p * (K_ + d) * (1 + (p_base_doble if usa_doble else 0)) - oc_l_p * (K_ - d) * (1 - q)
    if usa_decay:
        z = decay(z, (t2 - t_prev) / 86_400_000)
    return z

def metricas(y, zi, pred, idx=None):
    if idx is not None:
        y, zi, pred = y[idx], zi[idx], pred[idx]
    err = pred - y
    d_real, d_pred = y - zi, pred - zi
    r2 = 1 - (err ** 2).sum() / ((d_real - d_real.mean()) ** 2).sum()
    return {"MAE": np.abs(err).mean(), "med": np.median(np.abs(err)), "RMSE": np.sqrt((err ** 2).mean()), "R2_cambio": r2, "n": len(y)}

a_, b_ = ajustar_regla(dev)
print("regla por fotos (a, b):", {k: (round(a_[k], 1), round(b_[k], 1)) for k in a_})

pob = dev[(dev.dW >= 0) & (dev.dL >= 0)].copy()
print("población:", len(pob))
y = pob.z2.to_numpy(); zi = pob.z1.to_numpy()
pred = {}
pred["No cambió"] = zi.copy()
grp = pob.tier2.map({"CHALLENGER": 0, "GRANDMASTER": 1, "MASTER": 2}).fillna(3).astype(int).to_numpy()
A = np.array([a_[g] for g in grp]); B = np.array([b_[g] for g in grp])
pred["Regla a,b ajustada a fotos, W/L reales (techo viejo)"] = zi + A * pob.dW.to_numpy() - B * pob.dL.to_numpy()
pred["Regla a,b ajustada a fotos, W/L observadas"] = zi + A * pob.obs_w.to_numpy() - B * (pob.obs - pob.obs_w).to_numpy()
conf = {
    "K=20 (en vivo) + K Diamond, W/L reales": dict(usa_doble=0, usa_perdon=0, usa_delta=0, usa_piso=0, usa_decay=0, reales=True),
    "+ dobles por rol": dict(usa_doble=1, usa_perdon=0, usa_delta=0, usa_piso=0, usa_decay=0, reales=True),
    "+ perdón": dict(usa_doble=1, usa_perdon=1, usa_delta=0, usa_piso=0, usa_decay=0, reales=True),
    "+ pisos": dict(usa_doble=1, usa_perdon=1, usa_delta=0, usa_piso=1, usa_decay=0, reales=True),
    "+ δ(gap)": dict(usa_doble=1, usa_perdon=1, usa_delta=1, usa_piso=1, usa_decay=0, reales=True),
    "+ decay": dict(usa_doble=1, usa_perdon=1, usa_delta=1, usa_piso=1, usa_decay=1, reales=True),
    "Completo, solo W/L observadas": dict(usa_doble=1, usa_perdon=1, usa_delta=1, usa_piso=1, usa_decay=1, reales=False),
}
for nom, c in conf.items():
    pred[nom] = np.array([recorrer(j, f, **c) for j, f in pob.iterrows()])
    print("listo:", nom, flush=True)

grupos = {"Todos": np.ones(len(pob), bool), "Challenger+GM": grp <= 1, "Master": grp == 2, "Diamond": grp == 3,
          "Cobertura ≥95 %": (pob.obs / pob.dN.clip(lower=1)).to_numpy() >= .95,
          "Master+ cobertura ≥95 % y ≥30 partidas": (grp <= 2) & ((pob.obs / pob.dN.clip(lower=1)).to_numpy() >= .95) & (pob.dN.to_numpy() >= 30)}
for g_, idx in grupos.items():
    t = pd.DataFrame({m: metricas(y, zi, v, idx) for m, v in pred.items()}).T
    print(f"\n=== {g_} (n={idx.sum()}) ===\n", t.round(3).to_string())

pd.DataFrame(pred, index=pob.index).to_pickle(V / "pred_adelante.pkl")
pob.to_pickle(V / "pob.pkl")
p.to_pickle(V / "partes_lobby.pkl")
