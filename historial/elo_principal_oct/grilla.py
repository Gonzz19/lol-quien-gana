"""Filtro de grilla (HMM sobre el LP discretizado), al revés desde la foto 2, en la GPU.

Estado: LP absoluto en una grilla de 5 LP. Se arranca con la verosimilitud concentrada en z2 y, de la
partida más nueva a la más vieja, en cada paso:
  1. se deshace el decay del hueco posterior a la partida (no lineal: Master+ 75/día después de 14 días
     hasta 375; Diamond I 50/día después de 28 días hasta 275; calibrado en decay_curva.py);
  2. se deshace la partida: Δ ~ N(±a o −b + media de las ocultas, s² + varianza de las ocultas), con las
     ocultas como Poisson compuesta de tasa ρ por partida observada (ρ por grupo, del fold de entrenamiento);
  3. (opcional) se multiplica por la verosimilitud del lobby: t de Student (ν = 4) templada por τ.
Al final: hueco foto 1 → partida más vieja, ocultas de los jugadores sin partidas, y prior = histograma
del LP de toda la foto 2 (sin etiquetas). Salidas por jugador: media, mediana, sd, p10, p90, P(Master+).

Uso: python grilla.py --tag grid [--tau 0.3] [--sin-prior]
"""
import argparse, pickle, time
import numpy as np, pandas as pd
import torch
from comun import AQUI, cargar, evaluar, registrar

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="grid")
ap.add_argument("--tau", type=float, default=0.0, help="templado del lobby (0 = sin lobby)")
ap.add_argument("--s", type=float, default=5.0, help="ruido por partida (LP)")
ap.add_argument("--sin-prior", action="store_true")
ap.add_argument("--pi-m", type=float, default=0.3, help="P(el hueco tuvo partidas ocultas) en Master+")
ap.add_argument("--pi-d", type=float, default=0.7, help="ídem en Diamond")
ap.add_argument("--wr-k", type=float, default=0.0,
                help="winrate de las ocultas por jugador: récord no observado de la temporada (foto 2 − observadas) "
                     "encogido hacia el del grupo con K partidas de prior (0 = solo el del grupo)")
args = ap.parse_args()
t0 = time.time()
DIA = 86_400_000
c = cargar(); e = c["e"]; partes = c["partes"]
dev, fold = ~c["caja"], c["fold"]
gpu = torch.device("cuda")
ZMIN, PASO, G, NFFT = -200.0, 5.0, 900, 2048
zg = torch.arange(G, device=gpu, dtype=torch.float32) * PASO + ZMIN

# Partidas por jugador, de la más nueva a la más vieja.
p = partes[partes["player_id"].isin(e.index)].copy()
p["lp"] = c["lp_fila"].reindex(p.index).to_numpy()
p["lobby"] = c["lobp"]["lobby"].reindex(p.index).to_numpy()
p = p.sort_values(["player_id", "game_creation"], ascending=[True, False])
p["hueco"] = ((p.groupby("player_id")["game_creation"].shift(1).fillna(c["t2"]) - p["game_creation"]) / DIA)
pos = {pid: i for i, pid in enumerate(e.index)}
fila = p["player_id"].map(pos).to_numpy()
orden = p.groupby("player_id").cumcount().to_numpy()
n, T = len(e), int(orden.max()) + 1
gm, ch = c["corte"]
grupo = np.where(p["lp"] >= ch, 0, np.where(p["lp"] >= gm, 1, np.where(p["lp"] >= 400, 2, 3)))
a_g = np.array([c["a"][k] for k in range(4)]); b_g = np.array([c["b"][k] for k in range(4)])


def matriz(valores, relleno):
    M = np.full((n, T), relleno, np.float32)
    M[fila, orden] = valores
    return torch.from_numpy(M).to(gpu)


WIN = matriz(p["win"].to_numpy(float), np.nan)
A_ = matriz(a_g[grupo], 0.0); B_ = matriz(b_g[grupo], 0.0)
HUECO = matriz(p["hueco"].to_numpy(float), 0.0)
LOB = matriz(p["lobby"].to_numpy(float), np.nan)
viejo = p.groupby("player_id")["game_creation"].min().reindex(e.index)
D_INI = torch.from_numpy(((viejo - c["t1"]) / DIA).fillna(0).to_numpy(np.float32)).to(gpu)
obs = e["obs"].to_numpy(); obs_w = e["obs_w"].to_numpy()
wr = torch.from_numpy(((obs_w + 5) / (obs + 10)).astype(np.float32)).to(gpu)
wr_t = torch.from_numpy((e["wins2"] / e["partidas2"]).to_numpy(np.float32)).to(gpu)
g2 = np.where(e["z2"] >= ch, 0, np.where(e["z2"] >= gm, 1, np.where(e["z2"] >= 400, 2, 3)))

# Lobby: lobby ≈ c0 + c1 z + ε por grupo, con la partida más nueva de cada jugador (su LP ≈ z2).
nueva = p[orden == 0]
cal = {}
for k in range(4):
    m = (g2[nueva["player_id"].map(pos).to_numpy()] == k) & nueva["lobby"].notna().to_numpy()
    zz = e["z2"].to_numpy()[nueva["player_id"].map(pos).to_numpy()[m]]
    c1, c0 = np.polyfit(zz, nueva["lobby"].to_numpy()[m], 1)
    cal[k] = (c0, c1, np.std(nueva["lobby"].to_numpy()[m] - (c0 + c1 * zz)))
print("lobby:", {k: tuple(round(v, 2) for v in t) for k, t in cal.items()}, flush=True)
C0 = torch.tensor([cal[k][0] for k in range(4)], device=gpu)
C1 = torch.tensor([cal[k][1] for k in range(4)], device=gpu)
RR = torch.tensor([cal[k][2] for k in range(4)], device=gpu)
GZ = torch.where(zg >= ch, 0, torch.where(zg >= gm, 1, torch.where(zg >= 400, 2, 3)))   # grupo de cada punto

# Prior: histograma del LP de la foto 2 completa, suavizado 10 LP.
zf = c["foto2"]["z"].dropna().to_numpy()
h = np.histogram(zf, bins=G, range=(ZMIN - PASO / 2, ZMIN + PASO * G - PASO / 2))[0].astype(float) + 0.5
h = np.convolve(h, np.exp(-0.5 * (np.arange(-6, 7) * PASO / 10) ** 2), "same")
PRIOR = torch.from_numpy(h / h.sum()).float().to(gpu)

w = 2 * np.pi * torch.fft.rfftfreq(NFFT, d=PASO).to(gpu)       # frecuencias angulares (por LP)


def decay(z, d):
    """LP después de d días sin jugar empezando en z (vectorizado)."""
    d = d[:, None]
    dm = (torch.floor(d) - 14).clamp(min=0)
    master = z >= 400
    zm = z - 75 * dm
    dias_a_d1 = torch.ceil((z - 399) / 75).clamp(min=0)          # días de decay hasta caer de Master
    zz = torch.where(master, torch.where(zm >= 400, zm, torch.full_like(z, 375.0)), z)
    inicio_d = torch.where(master, 14 + dias_a_d1 + 14, torch.full_like(z, 28.0))
    dd = (torch.floor(d) - inicio_d).clamp(min=0)
    zona = (zz > 275) & (zz < 400)
    zz = torch.where(zona, torch.maximum(zz - 50 * dd, torch.full_like(z, 275.0)), zz)
    return zz


PI = torch.where(zg >= 400, args.pi_m, args.pi_d)[None, :]


def deshacer_decay(beta, d):
    """Mezcla: con prob. PI el hueco tuvo partidas que no vemos (no hubo decay)."""
    activo = d > 14
    if not activo.any():
        return beta
    zt = decay(zg[None, :].expand(int(activo.sum()), -1), d[activo])
    x = ((zt - ZMIN) / PASO).clamp(0, G - 1.001)
    i0 = x.floor().long()
    f = x - i0
    b = beta[activo]
    beta = beta.clone()
    beta[activo] = (1 - PI) * (b.gather(1, i0) * (1 - f) + b.gather(1, i0 + 1) * f) + PI * b
    return beta


def convolucion(beta, mu, var):
    B = torch.fft.rfft(beta, n=NFFT)
    H = torch.exp(1j * w[None, :] * mu[:, None] - 0.5 * w[None, :] ** 2 * var[:, None])
    return torch.fft.irfft(B * H, n=NFFT)[:, :G].clamp(min=0)


def correr(rho, m0, v0, wh):
    beta = torch.zeros((n, G), device=gpu)
    x = ((torch.from_numpy(e["z2"].to_numpy(np.float32)).to(gpu) - ZMIN) / PASO).clamp(0, G - 1.001)
    i0 = x.floor().long()
    beta.scatter_(1, i0[:, None], (1 - (x - i0))[:, None])
    beta.scatter_add_(1, (i0 + 1)[:, None], (x - i0)[:, None])
    RHO = torch.tensor(rho, device=gpu, dtype=torch.float32)
    WH = torch.tensor(wh, device=gpu, dtype=torch.float32)[torch.from_numpy(g2).to(gpu)]   # winrate de las ocultas
    if args.wr_k > 0:   # posterior Beta: récord no observado de la temporada + K partidas al winrate del grupo
        w_no = torch.from_numpy((e["wins2"] - e["obs_w"]).clip(lower=0).to_numpy(np.float32)).to(gpu)
        n_no = torch.from_numpy((e["partidas2"] - e["obs"]).clip(lower=0).to_numpy(np.float32)).to(gpu)
        WH = (w_no.clamp(max=n_no) + args.wr_k * WH) / (n_no + args.wr_k)
    for t in range(T):
        hay = ~torch.isnan(WIN[:, t])
        if not hay.any():
            break
        beta = deshacer_decay(beta, torch.where(hay, HUECO[:, t], torch.zeros_like(HUECO[:, t])))
        a, b = A_[:, t], B_[:, t]
        r = RHO[torch.from_numpy(g2).to(gpu)]
        mu = torch.where(WIN[:, t] == 1, a, -b) + r * (WH * a - (1 - WH) * b)
        var = args.s ** 2 + r * (WH * a ** 2 + (1 - WH) * b ** 2)
        mu, var = torch.where(hay, mu, 0.0), torch.where(hay, var, 0.0)
        beta = convolucion(beta, mu, var)
        if args.tau > 0:
            l = LOB[:, t]
            ok = hay & ~torch.isnan(l)
            gz = GZ[None, :]
            pred = C0[gz] + C1[gz] * zg[None, :]
            u = (l[:, None] - pred) / RR[gz]
            lik = (1 + u ** 2 / 4) ** (-2.5 * args.tau)
            beta = torch.where(ok[:, None], beta * lik, beta)
        beta = beta / beta.sum(1, keepdim=True).clamp(min=1e-30)
    beta = deshacer_decay(beta, D_INI)
    # Jugadores sin partidas observadas: todas ocultas.
    sin = torch.from_numpy(obs == 0).to(gpu)
    ga = torch.tensor(a_g, device=gpu, dtype=torch.float32)[torch.from_numpy(g2).to(gpu)]
    gb = torch.tensor(b_g, device=gpu, dtype=torch.float32)[torch.from_numpy(g2).to(gpu)]
    M0 = torch.tensor(m0, device=gpu, dtype=torch.float32)[torch.from_numpy(g2).to(gpu)]
    V0 = torch.tensor(v0, device=gpu, dtype=torch.float32)[torch.from_numpy(g2).to(gpu)]
    paso = WH * ga - (1 - WH) * gb
    mu0 = torch.where(sin, M0 * paso, 0.0)
    var0 = torch.where(sin, M0 * (WH * ga ** 2 + (1 - WH) * gb ** 2) + V0 * paso ** 2 + args.s ** 2, 0.0)
    beta = convolucion(beta, mu0, var0)
    post = beta if args.sin_prior else beta * PRIOR[None, :]
    post = post / post.sum(1, keepdim=True).clamp(min=1e-30)
    media = (post * zg).sum(1)
    sd = ((post * (zg - media[:, None]) ** 2).sum(1)).sqrt()
    cdf = post.cumsum(1)
    q = lambda p_: zg[(cdf < p_).sum(1).clamp(max=G - 1)]   # noqa: E731
    pm = post[:, zg >= 400].sum(1)
    return pd.DataFrame({k: v.cpu().numpy() for k, v in
                         dict(media=media, mediana=q(0.5), sd=sd, p10=q(0.1), p90=q(0.9), p_master=pm).items()},
                        index=e.index)


# Tasas del fold de entrenamiento (las únicas cuentas que tocan etiquetas: 4 números por grupo).
salidas = {}
for k in range(5):
    tr = dev & (fold != k)
    rho, m0, v0, wh = [], [], [], []
    for gg in range(4):
        m = tr & (g2 == gg)
        con = m & (obs > 0)
        rho.append(float(e["ocultas"][con].sum() / max(e["obs"][con].sum(), 1)))
        hw = (e["dW"] - e["obs_w"]).clip(lower=0)[m].sum()
        wh.append(float(hw / max(e["ocultas"][m].sum(), 1)))
        s0 = e["dN"][m & (obs == 0)]
        m0.append(float(s0.mean()) if len(s0) > 5 else 10.0); v0.append(float(s0.var()) if len(s0) > 5 else 100.0)
    salidas[k] = correr(rho, m0, v0, wh)
    print(f"fold {k}: ρ {np.round(rho, 2)} wr ocultas {np.round(wh, 3)} ({time.time() - t0:.0f} s)", flush=True)

# Corrida con las tasas de todo el desarrollo (para entrenar con todo y medir la caja cerrada).
rho, m0, v0, wh = [], [], [], []
for gg in range(4):
    m = dev & (g2 == gg)
    con = m & (obs > 0)
    rho.append(float(e["ocultas"][con].sum() / max(e["obs"][con].sum(), 1)))
    wh.append(float((e["dW"] - e["obs_w"]).clip(lower=0)[m].sum() / max(e["ocultas"][m].sum(), 1)))
    s0 = e["dN"][m & (obs == 0)]
    m0.append(float(s0.mean()) if len(s0) > 5 else 10.0); v0.append(float(s0.var()) if len(s0) > 5 else 100.0)
salidas["todo"] = correr(rho, m0, v0, wh)

# Expertos fuera de fold y variables (para el fold externo k, las de su corrida).
oof = pd.DataFrame(index=e.index, columns=salidas[0].columns, dtype=float)
for k in range(5):
    te = dev & (fold == k)
    oof[te] = salidas[k][te]
pickle.dump(salidas, open(AQUI / f"{args.tag}.pkl", "wb"))
for col in ("media", "mediana"):
    pr = oof[col]
    np.save(AQUI / "preds" / f"{args.tag}_{col}.npy", pr.to_numpy())
    registrar(f"{args.tag}_{col}", evaluar(e, pr, e.index[dev]),
              f"filtro de grilla, {col} posterior; τ lobby {args.tau}, s {args.s}, π {args.pi_m}/{args.pi_d}{', sin prior' if args.sin_prior else ''} ({time.time() - t0:.0f} s)")
