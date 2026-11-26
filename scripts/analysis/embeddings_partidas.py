"""Embeddings y clustering de partidas (pedido del usuario, 2026-11-14; §9.S.41).

Dos vistas de cada partida, solo con data/splits/train (el test no se toca):

1. *Antes de la partida* (lo que se sabe en la pantalla de carga). Por jugador: margen previo
   (acumulado y últimas 20), winrate previo, experiencia con campeón y línea, cambio de pool,
   ritmo de juego, horas desde la partida anterior, atributos del campeón (ddragon) y el perfil
   de estilo típico del campeón. Todo con partidas ANTERIORES a cada fila. Un autoencoder
   comprime los 5 jugadores de cada equipo (por línea) en 16 dimensiones: embedding de equipo.
   Partida = (equipo azul, equipo rojo).
2. *Desarrollo* (descriptivo, después de la partida): diferencias de oro/xp/kills a 10/15/20,
   duración, ritmo de kills, remontada, rendición temprana.

Clustering: K-Means, fuzzy C-means y GMM sobre los embeddings, ajustados SOLO con las partidas
anteriores a la validación de siempre (V = mitad posterior a la foto del 04/09); V solo se
asigna y se usa para medir.

Uso:  python scripts/analysis/embeddings_partidas.py            (~10 min, GPU opcional)
      --rehacer   recalcula las variables por jugador (si no, usa la caché)
Salida: data/analysis/embeddings_partidas/ (caché, figuras, resumen.json, log)."""
import argparse, json, sqlite3, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.features.fingerprint import add_rates, STYLE_FEATURES
from src.features.margin import lane_diffs, lane_scale, prior_margin, prior_winrate
from src.features.padron import verificar_foto1
from src.features.change import pool_change, play_rate

ap = argparse.ArgumentParser()
ap.add_argument("--rehacer", action="store_true")
ap.add_argument("--solo-variables", action="store_true",
                help="solo (re)construye variables.npz y sale sin correr el autoencoder ni el clustering")
ap.add_argument("--dim", type=int, default=16, help="dimensión del embedding de equipo")
ap.add_argument("--semilla", type=int, default=0)
ap.add_argument("--grupo", default="todo", choices=["todo", "historial", "campeon"],
                help="qué variables entran al embedding: todas, solo el historial del jugador o solo el campeón")
args = ap.parse_args()

OUT = REPO / "data" / "analysis" / "embeddings_partidas"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "variables.npz"
SAL = OUT if args.grupo == "todo" else OUT / args.grupo
SAL.mkdir(exist_ok=True)
LOG = None if args.solo_variables else open(SAL / "log.txt", "w", encoding="utf-8")
T0 = time.time()


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    if LOG is not None:
        LOG.write(s + "\n"); LOG.flush()


LANES = ["BOTTOM", "JUNGLE", "MIDDLE", "TOP", "UTILITY"]      # orden lexicográfico = orden del sort
STATS = ["kills_pm", "assists_pm", "damage_pm", "damage_share_pct", "kill_participation_pct",
         "deaths_pm", "damage_taken_pm", "gold_pm", "cs_pm", "gold_share_pct", "vision_pm",
         "d_gold_10", "d_gold_15", "d_xp_15", "d_cs_15"]
TAGS = ["Fighter", "Tank", "Mage", "Assassin", "Marksman", "Support"]
CSTATS = ["attackrange", "hp", "armor", "spellblock", "movespeed", "attackdamage", "attackspeed"]
USE = ["match_id", "game_creation", "game_duration", "game_version", "player_id", "team_id", "win",
       "champion_id", "lane", "kills", "deaths", "assists", "kill_participation_pct",
       "damage_share_pct", "gold_share_pct", "total_damage_dealt_to_champions", "total_damage_taken",
       "gold_earned", "total_minions_killed", "neutral_minions_killed", "vision_score",
       "first_blood_kill", "first_tower_kill", "player_tier", "early_aggression_10"] + \
      [f"{s}_{m}" for m in (10, 15, 20) for s in ("gold", "xp", "cs", "damage", "kills", "deaths", "assists")]


# =====================================================================================
# 1. Variables por jugador (antes de la partida) y del desarrollo
# =====================================================================================
def construir():
    with sqlite3.connect(f"file:{REPO}/data/lol_las.db?mode=ro", uri=True) as con:
        t_foto = verificar_foto1(con)       # foto 1 congelada (§9.S.67)
    df = pd.read_csv(REPO / "data" / "splits" / "train" / "full_dataset.csv", usecols=USE)
    df = df[df.lane.isin(LANES)]
    df = df[df.groupby(["match_id", "team_id"]).lane.transform("nunique") == 5]
    df = df[df.groupby("match_id").lane.transform("size") == 10]
    # partidas sin ganador (win = 0 en las 10 filas, TAREAS §3): fuera del análisis
    df = df[df.groupby("match_id").win.transform("sum") == 5].copy()
    df = add_rates(df).sort_values(["game_creation", "match_id", "team_id", "lane"]).reset_index(drop=True)
    df = lane_diffs(df, ["gold_10", "gold_15", "xp_15", "cs_15"])
    log(f"filas {len(df):,}, partidas {len(df) // 10:,} ({time.time() - T0:.0f}s)")

    mids = df.match_id.to_numpy()[::10]
    t = df.game_creation.to_numpy()[::10]
    mitad = np.median(t[t >= t_foto])
    val = t >= mitad
    val_fila = np.repeat(val, 10)

    # ---- historial del jugador (solo partidas anteriores; escalas e imputación congeladas en ~val)
    escala = lane_scale(df[~val_fila], STATS)
    M = prior_margin(df, STATS, 20, 5, 20, escala=escala)
    W = prior_winrate(df, 20)
    gp = df.groupby("player_id")
    n_tot = gp.cumcount()
    gc = df.groupby(["player_id", "champion_id"])
    n_ch = gc.cumcount()
    w_ch = gc.win.cumsum() - df.win
    w_tot = gp.win.cumsum() - df.win
    n_ln = df.groupby(["player_id", "lane"]).cumcount()
    E = pd.DataFrame({
        "logn_tot": np.log1p(n_tot), "logn_champ": np.log1p(n_ch), "estreno": (n_ch == 0) * 1.0,
        "wr_champ": (w_ch + 5 * (w_tot + 5) / (n_tot + 10)) / (n_ch + 5) - (w_tot + 5) / (n_tot + 10),
        "share_champ": n_ch / (n_tot + 1), "share_lane": (n_ln + 1) / (n_tot + 2),
        "offrole": ((n_tot >= 10) & (n_ln / n_tot.clip(lower=1) < 0.2)) * 1.0,
    })
    horas = (df.game_creation - gp.game_creation.shift(1)) / 3.6e6
    log_horas = np.log1p(horas)
    E["log_horas_desde"] = log_horas.fillna(log_horas[~val_fila].median())
    P = pool_change(df, 10, 5)
    E["ritmo"] = play_rate(df).fillna(0)
    log(f"historial listo ({time.time() - T0:.0f}s)")

    # ---- campeón: atributos estáticos + perfil de estilo típico (ajustado sin V)
    dd = json.load(open(REPO / "data" / "ddragon" / "16.17.1" / "champion.json", encoding="utf-8"))["data"]
    A = {}
    for v in dd.values():
        A[int(v["key"])] = [1.0 * (g in v["tags"]) for g in TAGS] + \
                           [v["info"][k] for k in ("attack", "defense", "magic", "difficulty")] + \
                           [v["stats"][k] for k in CSTATS]
    Acols = [f"tag_{g}" for g in TAGS] + ["info_atk", "info_def", "info_mag", "info_dif"] + \
            [f"st_{k}" for k in CSTATS]
    CA = pd.DataFrame(df.champion_id.map(A).tolist(), columns=Acols, index=df.index)
    CA = CA.fillna(CA[~val_fila].mean())
    tr = df[~val_fila]
    zl = (tr[STYLE_FEATURES] - tr.groupby("lane")[STYLE_FEATURES].transform("mean")) / \
         tr.groupby("lane")[STYLE_FEATURES].transform("std")
    g = zl.groupby(tr.champion_id)
    perfil = (g.sum() / (g.count() + 30)).add_prefix("perf_")     # encogido hacia 0 con 30 partidas
    CP = perfil.reindex(df.champion_id).fillna(0).set_index(df.index)

    X = pd.concat([M, W, E, P, CA, CP], axis=1).astype(np.float32)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    cols = list(X.columns)
    d = len(cols)
    Xj = X.to_numpy().reshape(-1, 2, 5, d)          # partida, equipo (100, 200), línea, variable

    # ---- desarrollo de la partida (por equipo, descriptivo)
    def eq(c):
        return df[c].to_numpy(np.float64).reshape(-1, 2, 5).sum(axis=2)
    dur = df.game_duration.to_numpy()[::10] / 60
    D = {"dur_min": dur}
    for m in (10, 15, 20):
        for s in ("gold", "xp", "kills"):
            v = eq(f"{s}_{m}")
            D[f"d_{s}_{m}"] = v[:, 0] - v[:, 1]                  # azul - rojo
    D["kills_tot_pm"] = eq("kills").sum(1) / dur
    D["d_kills_fin"] = eq("kills")[:, 0] - eq("kills")[:, 1]
    D["d_oro_fin"] = eq("gold_earned")[:, 0] - eq("gold_earned")[:, 1]
    D["dano_tot_pm"] = eq("total_damage_dealt_to_champions").sum(1) / dur
    D["vision_tot_pm"] = eq("vision_score").sum(1) / dur
    # 1 = azul, 0 = rojo, 0,5 = ningún jugador tiene la marca (torre de súbditos: 3,7 % de las partidas;
    # primera sangre por torre/súbditos: 0,3 %). Antes esos casos se contaban como rojo.
    for k, c in (("fb_azul", "first_blood_kill"), ("ft_azul", "first_tower_kill")):
        v = eq(c) > 0
        D[k] = np.where(v[:, 0], 1.0, np.where(v[:, 1], 0.0, 0.5))
    D["sin_20"] = np.isnan(D["d_gold_20"])
    y = df.win.to_numpy()[::10].astype(int)                       # gana azul
    orden = {"DIAMOND": 0, "MASTER": 1, "GRANDMASTER": 2, "CHALLENGER": 3}
    tnum = df.player_tier.map(orden).to_numpy(float).reshape(-1, 10)
    parche = df.game_version.astype(str).str.split(".").str[:2].str.join(".").to_numpy()[::10]
    np.savez_compressed(CACHE, Xj=Xj, cols=np.array(cols), mids=mids, t=t, val=val, y=y,
                        tnum=tnum, parche=parche, champ=df.champion_id.to_numpy().reshape(-1, 2, 5),
                        **{f"D_{k}": np.asarray(v) for k, v in D.items()})
    log(f"caché escrita: {Xj.shape} ({time.time() - T0:.0f}s)")


if args.rehacer or not CACHE.exists():
    construir()
if args.solo_variables:
    sys.exit(0)
Z = np.load(CACHE, allow_pickle=True)
Xj, cols, y, val, t = Z["Xj"], list(Z["cols"]), Z["y"], Z["val"], Z["t"]
D = {k[2:]: Z[k] for k in Z.files if k.startswith("D_")}
tnum, parche, champ = Z["tnum"], Z["parche"], Z["champ"]
NM, d = len(y), Xj.shape[-1]
fit = ~val
CAMP_PREF = ("tag_", "info_", "st_", "perf_")
sel = [i for i, c in enumerate(cols) if args.grupo == "todo" or
       (args.grupo == "campeon") == c.startswith(CAMP_PREF)]
Xe, ce = Xj[..., sel], [cols[i] for i in sel]
de = len(sel)
log(f"partidas {NM:,} (ajuste {fit.sum():,}, V {val.sum():,}); {d} variables por jugador, "
    f"{de} al embedding ({args.grupo})")

# =====================================================================================
# 2. Embedding de equipo: autoencoder (5 jugadores por línea -> dim)
# =====================================================================================
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

rng = np.random.default_rng(args.semilla)
torch.manual_seed(args.semilla)
dev = "cuda" if torch.cuda.is_available() else "cpu"
T = Xe.reshape(NM * 2, 5 * de)                                     # fila = equipo
mu = T[np.repeat(fit, 2)].mean(0); sd = T[np.repeat(fit, 2)].std(0) + 1e-6
Ts = ((T - mu) / sd).astype(np.float32)
Ts = np.clip(Ts, -6, 6)


class AE(nn.Module):
    def __init__(self, n, k):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n, 512), nn.GELU(), nn.Linear(512, 128), nn.GELU(), nn.Linear(128, k))
        self.dec = nn.Sequential(nn.Linear(k, 128), nn.GELU(), nn.Linear(128, 512), nn.GELU(), nn.Linear(512, n))

    def forward(self, x):
        z = self.enc(x)
        return z, self.dec(z)


ae = AE(5 * de, args.dim).to(dev)
opt = torch.optim.AdamW(ae.parameters(), 1e-3, weight_decay=1e-5)
idx_fit = np.flatnonzero(np.repeat(fit, 2))
idx_val = np.flatnonzero(np.repeat(val, 2))
Xt = torch.tensor(Ts, device=dev)
for ep in range(30):
    ae.train()
    perm = rng.permutation(idx_fit)
    for b in range(0, len(perm), 1024):
        xb = Xt[perm[b:b + 1024]]
        _, rec = ae(nn.functional.dropout(xb, 0.1))               # denoising
        loss = ((rec - xb) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    if ep % 10 == 9 or ep == 0:
        ae.eval()
        with torch.no_grad():
            rv = np.mean([((ae(Xt[idx_val[b:b + 8192]])[1] - Xt[idx_val[b:b + 8192]]) ** 2).mean().item()
                          for b in range(0, len(idx_val), 8192)])
        log(f"  AE época {ep + 1}: MSE ajuste {loss.item():.3f}, V {rv:.3f}")
ae.eval()
with torch.no_grad():
    Eq = np.concatenate([ae(Xt[b:b + 8192])[0].cpu().numpy() for b in range(0, len(Ts), 8192)])
pca = PCA(args.dim, random_state=0).fit(Ts[idx_fit])
log(f"PCA {args.dim} comp.: varianza explicada {pca.explained_variance_ratio_.sum():.3f} "
    f"(1a {pca.explained_variance_ratio_[0]:.3f})")
Ep = pca.transform(Ts)
# varianza reconstruida por el AE (comparable con la de PCA)
r2_ae = 1 - rv / Ts[idx_val].var(0).mean()
log(f"AE: R² de reconstrucción en V {r2_ae:.3f}")

Eq = Eq.reshape(NM, 2, -1)
Ep = Ep.reshape(NM, 2, -1)
emb_mu, emb_sd = Eq[fit].reshape(-1, args.dim).mean(0), Eq[fit].reshape(-1, args.dim).std(0) + 1e-6
Eqs = (Eq - emb_mu) / emb_sd

# =====================================================================================
# 3. Clustering: K-Means, fuzzy C-means, GMM
# =====================================================================================
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, normalized_mutual_info_score


def cmeans(Xf, c, m=2.0, it=150, seed=0):
    """Fuzzy C-means (Bezdek). Devuelve centros y pertenencias U (n x c)."""
    r = np.random.default_rng(seed)
    C = Xf[r.choice(len(Xf), c, replace=False)]
    for _ in range(it):
        d2 = ((Xf[:, None, :] - C[None]) ** 2).sum(-1) + 1e-9
        U = 1 / (d2 ** (1 / (m - 1)) * (1 / d2 ** (1 / (m - 1))).sum(1, keepdims=True))
        Um = U ** m
        Cn = (Um.T @ Xf) / Um.sum(0)[:, None]
        if np.abs(Cn - C).max() < 1e-5:
            break
        C = Cn
    return C, U


def membresias(Xf, C, m=2.0):
    d2 = ((Xf[:, None, :] - C[None]) ** 2).sum(-1) + 1e-9
    return 1 / (d2 ** (1 / (m - 1)) * (1 / d2 ** (1 / (m - 1))).sum(1, keepdims=True))


def elegir_k(Xf, nombre, ks=range(2, 11)):
    """Silhouette (muestra de 20k) y BIC del GMM; k = máximo silhouette."""
    s = rng.choice(len(Xf), min(20000, len(Xf)), replace=False)
    res = {}
    for k in ks:
        km = MiniBatchKMeans(k, random_state=0, n_init=5, batch_size=4096).fit(Xf)
        sil = silhouette_score(Xf[s], km.predict(Xf[s]))
        gm = GaussianMixture(k, covariance_type="diag", random_state=0).fit(Xf[s])
        res[k] = {"silhouette": round(float(sil), 4), "bic_gmm": round(float(gm.bic(Xf[s])), 0)}
    log(f"  {nombre}: " + "  ".join(f"k={k} sil {v['silhouette']:.3f}" for k, v in res.items()))
    kb = max(res, key=lambda k: res[k]["silhouette"])
    kbic = min(res, key=lambda k: res[k]["bic_gmm"])
    log(f"  {nombre}: mejor k por silhouette {kb}, por BIC del GMM {kbic}")
    return res, kb, kbic


resumen = {"n_partidas": int(NM), "n_ajuste": int(fit.sum()), "n_val": int(val.sum()),
           "variables_por_jugador": cols, "r2_ae_val": round(float(r2_ae), 4),
           "pca_var": round(float(pca.explained_variance_ratio_.sum()), 4)}

# ---- 3a. tipos de EQUIPO (antes de la partida)
log("\n== Tipos de equipo (embedding del AE) ==")
TE = Eqs.reshape(NM * 2, -1)
tfit = np.repeat(fit, 2)
res_e, kb_e, kbic_e = elegir_k(TE[tfit], "equipo-AE")
res_p, kb_p, _ = elegir_k(((Ep - Ep[fit].reshape(-1, args.dim).mean(0)) /
                           Ep[fit].reshape(-1, args.dim).std(0)).reshape(NM * 2, -1)[tfit], "equipo-PCA")
resumen["k_equipo"] = {"ae": res_e, "pca": res_p}

# k para describir: el de silhouette suele ser 2 (poco informativo); usamos también uno fijo
K_EQ = int(max(kb_e, 6)) if kb_e < 4 else int(kb_e)
km_e = MiniBatchKMeans(K_EQ, random_state=0, n_init=10, batch_size=4096).fit(TE[tfit])
lab_e = km_e.predict(TE).reshape(NM, 2)
Cc, Uf = cmeans(TE[tfit][rng.choice(tfit.sum(), 60000, replace=False)], K_EQ)
U = membresias(TE, Cc).reshape(NM, 2, K_EQ)
pc = float((U.reshape(-1, K_EQ) ** 2).sum(1).mean())
amb = float((U.reshape(-1, K_EQ).max(1) < 0.5).mean())
log(f"  K-Means k={K_EQ}; fuzzy C-means: coef. de partición {pc:.3f} (1/k = {1 / K_EQ:.3f} = sin estructura), "
    f"{amb:.1%} de equipos con pertenencia máxima < 0,5")
nmi_fc = normalized_mutual_info_score(lab_e.ravel(), U.reshape(-1, K_EQ).argmax(1))
log(f"  acuerdo K-Means vs C-means (NMI) {nmi_fc:.3f}")
resumen["cmeans_equipo"] = {"k": K_EQ, "coef_particion": round(pc, 4), "ambiguos": round(amb, 4),
                            "nmi_vs_kmeans": round(float(nmi_fc), 4)}

# perfil de cada tipo: media por variable (sobre el equipo, en z), top diferenciales
Xeq = Xe.mean(axis=2)                                             # media de los 5 jugadores
zeq = (Xeq - Xeq[fit].reshape(-1, de).mean(0)) / (Xeq[fit].reshape(-1, de).std(0) + 1e-6)
win_eq = np.c_[y, 1 - y]
perfiles = {}
log(f"\n  {'tipo':>4} {'%':>6} {'gana':>6} {'gana V':>7}  rasgos (z medio del equipo)")
for k in range(K_EQ):
    m = lab_e == k
    mf, mv = m & fit[:, None], m & val[:, None]
    zz = zeq[m].mean(0)
    top = np.argsort(-np.abs(zz))[:6]
    perfiles[k] = {"frac": round(float(m.mean()), 4), "gana": round(float(win_eq[mf].mean()), 4),
                   "gana_V": round(float(win_eq[mv].mean()), 4),
                   "rasgos": {ce[i]: round(float(zz[i]), 2) for i in top},
                   "tier_medio": round(float(np.nanmean(np.where(m[:, :, None], tnum.reshape(NM, 2, 5), np.nan))), 3)}
    log(f"  {k:>4} {m.mean():6.1%} {win_eq[mf].mean():6.3f} {win_eq[mv].mean():7.3f}  " +
        ", ".join(f"{ce[i]} {zz[i]:+.2f}" for i in top))
resumen["tipos_equipo"] = perfiles

# matriz tipo azul x tipo rojo: winrate del azul (ajuste) y en V
log("\n  winrate del azul por (tipo azul, tipo rojo), en V [n]:")
WR = np.full((K_EQ, K_EQ), np.nan); NN = np.zeros((K_EQ, K_EQ), int)
for i in range(K_EQ):
    row = []
    for j in range(K_EQ):
        m = val & (lab_e[:, 0] == i) & (lab_e[:, 1] == j)
        NN[i, j] = m.sum()
        WR[i, j] = y[m].mean() if m.sum() else np.nan
        row.append(f"{WR[i, j]:.2f}[{NN[i, j]}]")
    log(f"  {i}: " + " ".join(row))
resumen["matriz_tipos_V"] = {"wr": np.round(WR, 3).tolist(), "n": NN.tolist()}

# ---- 3b. tipos de PARTIDA antes de jugarla (simétrico: no depende de quién es azul)
log("\n== Tipos de partida, antes de jugarla (simétrico) ==")
SM = np.c_[(Eqs[:, 0] + Eqs[:, 1]) / 2, np.abs(Eqs[:, 0] - Eqs[:, 1])]
SM = (SM - SM[fit].mean(0)) / SM[fit].std(0)
res_m, kb_m, kbic_m = elegir_k(SM[fit], "partida-previa")
K_PM = int(max(kb_m, 6)) if kb_m < 4 else int(kb_m)
lab_pm = MiniBatchKMeans(K_PM, random_state=0, n_init=10, batch_size=4096).fit(SM[fit]).predict(SM)
resumen["k_partida_previa"] = res_m

# ---- 3c. tipos de partida por su DESARROLLO (simétrico)
log("\n== Tipos de partida por su desarrollo (después) ==")
lider15 = np.sign(D["d_gold_15"])
remonta = (lider15 != 0) & (lider15 != np.where(y == 1, 1, -1))
DV = np.c_[D["dur_min"], D["kills_tot_pm"], D["dano_tot_pm"], D["vision_tot_pm"],
           np.abs(D["d_gold_10"]), np.abs(D["d_gold_15"]), np.abs(np.nan_to_num(D["d_gold_20"], nan=0)),
           np.abs(D["d_kills_15"]), np.abs(D["d_oro_fin"]), np.abs(D["d_kills_fin"]), remonta, D["sin_20"]]
dv_cols = ["dur_min", "kills_tot_pm", "dano_tot_pm", "vision_tot_pm", "|dgold10|", "|dgold15|", "|dgold20|",
           "|dkills15|", "|doro_fin|", "|dkills_fin|", "remonta15", "sin_20"]
DV = np.nan_to_num(DV.astype(float), nan=0.0)                    # sin timeline: diferencias en 0
DVs = (DV - DV[fit].mean(0)) / DV[fit].std(0)
res_d, kb_d, _ = elegir_k(DVs[fit], "desarrollo")
K_D = int(max(kb_d, 5)) if kb_d < 4 else int(kb_d)
km_d = MiniBatchKMeans(K_D, random_state=0, n_init=10, batch_size=4096).fit(DVs[fit])
lab_d = km_d.predict(DVs)
resumen["k_desarrollo"] = res_d
des = {}
log(f"  {'tipo':>4} {'%':>6}  " + " ".join(f"{c:>12}" for c in dv_cols))
for k in range(K_D):
    m = lab_d == k
    des[k] = {"frac": round(float(m.mean()), 4), **{c: round(float(DV[m, i].mean()), 2) for i, c in enumerate(dv_cols)}}
    log(f"  {k:>4} {m.mean():6.1%}  " + " ".join(f"{DV[m, i].mean():12.2f}" for i in range(len(dv_cols))))
resumen["tipos_desarrollo"] = des

# ---- 3d. ¿lo previo anticipa el desarrollo?
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

nmi_pd = normalized_mutual_info_score(lab_pm[val], lab_d[val])
log(f"\n  NMI(tipo previo, tipo de desarrollo) en V: {nmi_pd:.4f}  (0 = independientes)")
ct = pd.crosstab(lab_pm[val], lab_d[val], normalize="index").round(3)
log("  reparto del desarrollo dentro de cada tipo previo (filas, V):\n" + ct.to_string())
base_d = pd.Series(lab_d[val]).value_counts(normalize=True).sort_index().round(3)
log("  reparto global en V: " + " ".join(f"{k}:{v}" for k, v in base_d.items()))
resumen["nmi_previo_desarrollo_V"] = round(float(nmi_pd), 4)
resumen["crosstab_previo_desarrollo_V"] = ct.to_dict()

# =====================================================================================
# 4. ¿Suman a quién gana? Logística base (azul - rojo por línea) + embeddings / tipos
# =====================================================================================
log("\n== ¿Suma a quién gana? (ajuste sin V, AUC en V) ==")
Xd = (Xj[:, 0] - Xj[:, 1]).reshape(NM, -1)
xs_mu, xs_sd = Xd[fit].mean(0), Xd[fit].std(0) + 1e-6
Xd = (Xd - xs_mu) / xs_sd


def auc_de(X, C=0.05):
    lr = LogisticRegression(C=C, max_iter=2000).fit(X[fit], y[fit])
    return lr.predict_proba(X[val])[:, 1]


def boot(p1, p0, B=500):
    r = np.random.default_rng(1)
    yv = y[val]; n = len(yv); out = []
    for _ in range(B):
        s = r.integers(0, n, n)
        out.append(roc_auc_score(yv[s], p1[s]) - roc_auc_score(yv[s], p0[s]))
    return np.percentile(out, [2.5, 97.5])


p0 = auc_de(Xd)
a0 = roc_auc_score(y[val], p0)
log(f"  base (todas las variables por jugador, azul - rojo por línea): AUC {a0:.4f}")
pares = np.zeros((NM, K_EQ * K_EQ), np.float32)
for i in range(NM):
    a, b = lab_e[i]
    pares[i, a * K_EQ + b] += 1; pares[i, b * K_EQ + a] -= 1
Uf2 = (U[:, 0] - U[:, 1])
pruebas = {"+ embedding AE (azul - rojo)": np.c_[Xd, Eqs[:, 0] - Eqs[:, 1]],
           "+ cruce de tipos de equipo (antisimétrico)": np.c_[Xd, pares],
           "+ pertenencias C-means (azul - rojo)": np.c_[Xd, Uf2]}
resumen["auc"] = {"base": round(float(a0), 4)}
for nom, X in pruebas.items():
    p = auc_de(X)
    a = roc_auc_score(y[val], p)
    lo, hi = boot(p, p0)
    log(f"  {nom}: AUC {a:.4f} ({a - a0:+.4f} [{lo:+.4f}; {hi:+.4f}])")
    resumen["auc"][nom] = {"auc": round(float(a), 4), "delta": round(float(a - a0), 4),
                           "ic95": [round(float(lo), 4), round(float(hi), 4)]}
# solo el embedding (sin la base): cuánto de quién gana resume
p = auc_de(Eqs[:, 0] - Eqs[:, 1], C=1.0)
log(f"  solo embedding AE ({args.dim} dims, azul - rojo): AUC {roc_auc_score(y[val], p):.4f}")
resumen["auc"]["solo_embedding"] = round(float(roc_auc_score(y[val], p)), 4)

# AUC de la base dentro de cada tipo previo y de desarrollo: dónde acierta y dónde no
log("\n  AUC de la base por tipo de partida previo (V):")
por_tipo = {}
for k in range(K_PM):
    m = lab_pm[val] == k
    a = roc_auc_score(y[val][m], p0[m]) if m.sum() > 200 else np.nan
    tm = np.nanmean(tnum[val][m])
    por_tipo[k] = {"frac": round(float(m.mean()), 4), "auc": round(float(a), 4),
                   "gana_azul": round(float(y[val][m].mean()), 4), "tier_medio": round(float(tm), 3),
                   "dur_min": round(float(D["dur_min"][val][m].mean()), 1),
                   "remonta": round(float(remonta[val][m].mean()), 3)}
    log(f"  tipo {k}: {m.mean():5.1%}  AUC {a:.3f}  azul {y[val][m].mean():.3f}  tier {tm:.2f}  "
        f"dur {D['dur_min'][val][m].mean():.1f}  remonta15 {remonta[val][m].mean():.3f}")
resumen["por_tipo_previo_V"] = por_tipo

# =====================================================================================
# 5. Figuras
# =====================================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

s = rng.choice(np.flatnonzero(val), 6000, replace=False)
emb2 = TSNE(2, random_state=0, init="pca", perplexity=40).fit_transform(Eqs[s, 0])
fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
ax[0].scatter(emb2[:, 0], emb2[:, 1], c=lab_e[s, 0], cmap="tab10", s=3)
ax[0].set_title(f"Equipos (t-SNE del embedding AE), color = tipo K-Means")
ax[1].scatter(emb2[:, 0], emb2[:, 1], c=np.nanmean(tnum.reshape(NM, 2, 5)[s, 0], 1), cmap="viridis", s=3)
ax[1].set_title("color = tier medio del equipo")
li = ce.index("acum_gold_pm") if "acum_gold_pm" in ce else 0
ax[2].scatter(emb2[:, 0], emb2[:, 1], c=np.clip(zeq[s, 0, li], -2, 2), cmap="coolwarm", s=3)
ax[2].set_title(f"color = {ce[li]} (z, media del equipo)")
for a in ax:
    a.set_xticks([]); a.set_yticks([])
fig.tight_layout(); fig.savefig(SAL / "tsne_equipos.png", dpi=110); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(WR - y[val].mean(), cmap="RdBu_r", vmin=-0.08, vmax=0.08)
for i in range(K_EQ):
    for j in range(K_EQ):
        ax.text(j, i, f"{WR[i, j]:.2f}\n{NN[i, j]}", ha="center", va="center", fontsize=7)
ax.set_xlabel("tipo del equipo rojo"); ax.set_ylabel("tipo del equipo azul")
ax.set_title("Winrate del azul por cruce de tipos (V)")
fig.colorbar(im, label="winrate − media"); fig.tight_layout(); fig.savefig(SAL / "matriz_tipos.png", dpi=110); plt.close(fig)

json.dump(resumen, open(SAL / "resumen.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
np.savez_compressed(SAL / "embeddings.npz", mids=Z["mids"], emb_equipo=Eq, tipo_equipo=lab_e,
                    cmeans_equipo=U, tipo_previo=lab_pm, tipo_desarrollo=lab_d, val=val)
log(f"\nlisto ({time.time() - T0:.0f}s) -> {SAL}")
