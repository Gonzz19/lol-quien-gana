"""El escalón 9 llevado al dataset de DraftRec (Lee et al., WWW 2022; PROJECT_SUMMARY.md §9.S.59).

`draftrec_comparacion.py` reconstruyó el híbrido (draft + jugadores + margen de final de partida, todo como media
de los 5): 58,78 % / AUC 0,623 en su test contra 55,35 % de DraftRec. Acá, sobre los mismos splits y con C elegido
en val, se suman uno por uno los ingredientes del escalón 9 que se pueden reconstruir sin línea de tiempo, LP ni
W/L de temporada:

1. **Por línea** en vez de la media de los 5: azul − rojo de cada variable en cada rol (TOP, JUNGLE, MIDDLE,
   CARRY, SUPPORT), como `Xd` del modelo principal. Si un equipo no tiene los 5 roles, esa partida usa la media.
2. **Margen y winrate por (jugador, rol)** (escalones 2 y 4b): estadística − media del campeón, acumulada solo en
   las partidas de ese rol y encogida hacia el margen total; winrate en el rol encogido hacia el total.
3. **Elo decreciente causal** (análogo del Elo decreciente del escalón 6, solo con resultados): rating por jugador recorrido en
   orden de tiempo sobre las 279.893 partidas de los tres splits; cada partida se describe con los ratings de
   antes y después se actualiza. Diferencia por línea y partidas vistas.
4. **Contracruces por línea y sinergias** (escalón 4): winrate empírico-bayesiano causal del par (campeón azul,
   campeón rojo) en cada línea y de los pares de equipo (CARRY+SUPPORT, TOP+JUNGLE, MIDDLE+JUNGLE).
5. **Margen esperado por par de campeones en la línea** (escalón 9): media empírico-bayesiana causal de la
   diferencia de oro, CS y daño a campeones de final de partida entre los rivales de línea con esos campeones.

Los acumulados 3-5 recorren train → val → test en orden: el test se describe con todo lo anterior a cada partida
(igual que el modelo principal), pero los pesos de la logística se ajustan solo con train.

    python scripts/analysis/draftrec_mejor.py        # ~15-20 min (una pasada por los 16 GB)
"""
import argparse
import ast
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=str(REPO / "data" / "external" / "draftrec"))
ap.add_argument("--salida", default=str(REPO / "data" / "analysis" / "draftrec_mejor.csv"))
ap.add_argument("--k-elo", type=float, nargs="+", default=[0.03, 0.06, 0.1, 0.15])
ap.add_argument("--solo-pasadas", action="store_true", help="solo leer los 16 GB y guardar la caché (poca RAM)")
ap.add_argument("--rehacer", action="store_true", help="ignorar la caché de las pasadas")
args = ap.parse_args()
DIR = Path(args.dir)
T0 = time.time()
log = lambda s: print(f"[{time.time() - T0:6.0f}s] {s}", flush=True)  # noqa: E731

STATS = {"kda": 0, "dano_campeones": 15, "dano_objetivos": 22, "vision": 24, "dano_recibido": 26,
         "oro": 30, "cs": 34, "cs_jungla": 35}
SI = np.array(list(STATS.values())) + 14   # columnas de las 43 estadísticas dentro de las 57
PAR_STATS = [list(STATS).index(k) for k in ("oro", "cs", "dano_campeones")]
K_CH, K_WR, K_ROL = 5, 10, 5
ROLES = [4, 5, 6, 7, 8]                    # TOP, JUNGLE, MIDDLE, DUO_CARRY, DUO_SUPPORT
AZUL_T, ROJO_T, GANA = 4, 5, 5
SINERGIAS = [(3, 4), (0, 1), (2, 1)]        # (CARRY, SUPPORT), (TOP, JUNGLE), (MIDDLE, JUNGLE) en índices de línea

# ---------------------------------------------------------------- partidas de los tres splits, en orden de tiempo
U, H, T, S = [], [], [], []
for s in ("train", "val", "test"):
    d = pd.read_csv(DIR / f"{s}.csv")
    pares = np.array([[ast.literal_eval(v) for v in d[f"User{k}"]] for k in range(1, 11)])  # (10, n, 2)
    U.append(pares[:, :, 0].T); H.append(pares[:, :, 1].T); T.append(d.time.to_numpy()); S += [s] * len(d)
    log(f"{s}: {len(d):,} partidas")
U, H, T, S = np.vstack(U).astype(np.int64), np.vstack(H).astype(np.int64), np.concatenate(T), np.array(S)
N = len(T)
orden = np.argsort(T, kind="stable")
assert (S[orden][:(S == "train").sum()] == "train").all(), "los splits no son temporales"

A = np.load(DIR / "user_history.npy", mmap_mode="r")
NU, LH, F = A.shape
log(f"historial: {A.shape}")

NOM_V = (["log_n", "log_n_camp", "estreno", "share_camp", "wr_camp", "wr_tot", "wr_20", "share_rol", "fuera_rol"]
         + [f"m_acum_{k}" for k in STATS] + [f"m_20_{k}" for k in STATS] + [f"m_camp_{k}" for k in STATS])
NOM_R = ["wr_rol", "log_n_rol"] + [f"m_rol_{k}" for k in STATS]
NV, NR = len(NOM_V), len(NOM_R)
CACHE = Path(args.salida).with_name("draftrec_mejor_pasadas.npz")
GUARDAR = ("CAMP", "ROL", "EQ", "WIN", "ST", "BANS", "VAR", "VROL")
if CACHE.exists() and not args.rehacer:
    Cz = np.load(CACHE)
    CAMP, ROL, EQ, WIN, ST, BANS, VAR, VROL = (Cz[k] for k in GUARDAR)
    assert len(CAMP) == N, "la caché de las pasadas no corresponde a estos splits"
    log(f"pasadas leídas de {CACHE.name}")
else:
    # ---------------------------------------------------------------- pasada 1: media por campeón de las estadísticas (train)
    NC = 200
    tr_mask = S == "train"
    pares_tr = pd.DataFrame({"u": U[tr_mask].ravel(), "h": H[tr_mask].ravel()})
    pares_tr = pares_tr[pares_tr.u < NU].sort_values(["u", "h"])
    suma, cnt = np.zeros((NC, len(SI))), np.zeros(NC)
    for u, grp in pares_tr.groupby("u", sort=True):
        Hu = np.asarray(A[u])
        idx = grp.h.to_numpy()
        c = Hu[idx, 0].astype(int)
        np.add.at(suma, c, Hu[idx][:, SI])
        np.add.at(cnt, c, 1)
    media_camp = suma / np.maximum(cnt, 1)[:, None]
    del pares_tr
    log("pasada 1 lista (medias por campeón del train)")

    # ---------------------------------------------------------------- pasada 2: por (jugador, índice) lo previo a la partida
    VAR = np.zeros((N, 10, NV), np.float32)
    VROL = np.zeros((N, 10, NR), np.float32)
    CAMP = np.zeros((N, 10), np.int16)
    ROL = np.zeros((N, 10), np.int16)
    EQ = np.zeros((N, 10), np.int8)
    WIN = np.zeros((N, 10), np.int8)
    ST = np.zeros((N, 10, len(SI)), np.float32)
    BANS = np.zeros((N, 10), np.int16)

    slots = pd.DataFrame({"u": U.ravel(), "h": H.ravel(), "g": np.repeat(np.arange(N), 10), "k": np.tile(np.arange(10), N)})
    slots = slots[slots.u < NU].sort_values(["u", "h"])
    log(f"pasada 2: {slots.u.nunique():,} jugadores, {len(slots):,} lugares")
    for n_u, (u, grp) in enumerate(slots.groupby("u", sort=True)):
        Hu = np.asarray(A[u])
        n = int((Hu[:, 0] != 0).sum())
        camp, rol = Hu[:n, 0].astype(int), Hu[:n, 1].astype(int)
        gana = (Hu[:n, 13] == GANA).astype(float)
        marg = Hu[:n][:, SI] - media_camp[camp]
        cw = np.r_[0, np.cumsum(gana)]
        cm = np.r_[np.zeros((1, len(SI))), np.cumsum(marg, axis=0)]
        for h, g, k in zip(grp.h.to_numpy(), grp.g.to_numpy(), grp.k.to_numpy()):
            if h >= n:
                continue
            c, r = camp[h], rol[h]
            CAMP[g, k], ROL[g, k], EQ[g, k], WIN[g, k] = c, r, int(Hu[h, 2]), int(Hu[h, 13])
            ST[g, k] = Hu[h, SI]
            BANS[g] = Hu[h, 3:13]
            prev_c = camp[:h] == c
            nc = prev_c.sum()
            wr_tot = (cw[h] + 0.5 * K_WR) / (h + K_WR)
            wr_c = (gana[:h][prev_c].sum() + K_CH * wr_tot) / (nc + K_CH)
            a20 = max(0, h - 20)
            wr20 = (cw[h] - cw[a20] + 0.5 * K_WR) / (h - a20 + K_WR)
            prev_r = rol[:h] == r
            nr = prev_r.sum()
            m_acum = cm[h] / max(h, 1)
            m_20 = (cm[h] - cm[a20]) / max(h - a20, 1)
            m_c = (marg[:h][prev_c].sum(0) + K_CH * m_acum) / (nc + K_CH)
            VAR[g, k] = np.r_[np.log1p(h), np.log1p(nc), float(nc == 0), nc / (h + 1), wr_c - wr_tot, wr_tot, wr20,
                              (nr + 1) / (h + 2), float(h >= 10 and nr / h < 0.2), m_acum, m_20, m_c]
            wr_r = (gana[:h][prev_r].sum() + K_ROL * wr_tot) / (nr + K_ROL)
            m_r = (marg[:h][prev_r].sum(0) + K_ROL * m_acum) / (nr + K_ROL)
            VROL[g, k] = np.r_[wr_r - wr_tot, np.log1p(nr), m_r]
        if n_u % 10000 == 0:
            log(f"  pasada 2: {n_u:,} jugadores")
    del slots
    log("pasada 2 lista")
    np.savez(CACHE, **{k: globals()[k] for k in GUARDAR})
    log(f"pasadas guardadas en {CACHE.name}")
if args.solo_pasadas:
    raise SystemExit(0)

# ---------------------------------------------------------------- orientación: azul / rojo por el campo de equipo
ok_eq = ((EQ == AZUL_T).sum(1) == 5) & ((EQ == ROJO_T).sum(1) == 5)
y = np.zeros(N)
for g in range(N):
    k0 = int(np.flatnonzero(EQ[g] == AZUL_T)[0]) if ok_eq[g] else 0
    y[g] = float(WIN[g, k0] == GANA) if EQ[g, k0] == AZUL_T else float(WIN[g, k0] != GANA)
log(f"partidas con 5 + 5 jugadores identificados: {ok_eq.mean():.2%}; el primer pick es azul en "
    f"{(EQ[:, 0] == AZUL_T).mean():.1%}")

# posición de cada línea: idx[g, equipo, línea] = lugar k (o −1)
LIN = np.full((N, 2, 5), -1, np.int64)
for e, tcode in enumerate((AZUL_T, ROJO_T)):
    for li, r in enumerate(ROLES):
        m = (EQ == tcode) & (ROL == r)
        una = m.sum(1) == 1
        LIN[una, e, li] = np.argmax(m[una], axis=1)
ok_lin = ok_eq & (LIN >= 0).all((1, 2))
log(f"partidas con los 5 roles en los dos equipos: {ok_lin.mean():.2%}")


def por_linea(X):
    """(N, 10, f) → (N, 5·f) azul − rojo por línea; si no hay biyección de roles, la media del equipo."""
    out = np.zeros((N, 5, X.shape[2]), np.float32)
    az = (EQ == AZUL_T).astype(np.float32)
    ro = (EQ == ROJO_T).astype(np.float32)
    media = (np.einsum("gkf,gk->gf", X, az) / np.maximum(az.sum(1), 1)[:, None]
             - np.einsum("gkf,gk->gf", X, ro) / np.maximum(ro.sum(1), 1)[:, None])
    gi = np.flatnonzero(ok_lin)
    for li in range(5):
        out[gi, li] = X[gi, LIN[gi, 0, li]] - X[gi, LIN[gi, 1, li]]
    out[~ok_lin] = media[~ok_lin][:, None, :]
    return out.reshape(N, -1), media


XL_V, XM_V = por_linea(VAR)
XL_R, _ = por_linea(VROL)
log("variables por línea listas")

# ---------------------------------------------------------------- acumulados causales en orden de tiempo
def elo(K):
    R, n = {}, {}
    out = np.zeros((N, 5), np.float32)
    nvis = np.zeros((N, 5), np.float32)
    eq_d = np.zeros(N, np.float32)
    for g in orden:
        if not ok_eq[g]:
            continue
        us = U[g]
        rs = np.array([R.get(int(x), 0.0) for x in us])
        ns = np.array([n.get(int(x), 0) for x in us], float)
        az, ro = EQ[g] == AZUL_T, EQ[g] == ROJO_T
        d = rs[az].mean() - rs[ro].mean()
        eq_d[g] = d
        if ok_lin[g]:
            out[g] = rs[LIN[g, 0]] - rs[LIN[g, 1]]
            nvis[g] = np.log1p(ns[LIN[g, 0]]) - np.log1p(ns[LIN[g, 1]])
        else:
            out[g] = d
        e = 1.0 / (1.0 + np.exp(-d))
        delta = K * (y[g] - e)
        for kk, x in enumerate(us):
            x = int(x)
            R[x] = R.get(x, 0.0) + (delta if az[kk] else -delta)
            n[x] = n.get(x, 0) + 1
    return np.c_[eq_d, out, nvis]


val_m, te_m = S == "val", S == "test"
mejor_k, mejor_auc = None, -1
for K in args.k_elo:
    E = elo(K)
    a = roc_auc_score(y[val_m & ok_eq], E[val_m & ok_eq, 0])
    log(f"Elo K = {K}: AUC en val del Δ de equipo solo {a:.4f}")
    if a > mejor_auc:
        mejor_k, mejor_auc, ELO = K, a, E
log(f"Elo elegido en val: K = {mejor_k}")

K_EB, K_PAR = 20.0, 10.0
MU = np.zeros((N, 5), np.float32)            # contracruce por línea: WR EB del campeón azul contra el rojo − 0,5
SIN = np.zeros((N, 3), np.float32)           # sinergias: azul − rojo
PM = np.zeros((N, 5, len(PAR_STATS)), np.float32)   # margen esperado del par en la línea (azul − rojo)
cnt_mu, cnt_sin, cnt_pm = {}, {}, {}
for g in orden:
    if not ok_lin[g]:
        continue
    cb = CAMP[g, LIN[g, 0]].astype(int)
    cr = CAMP[g, LIN[g, 1]].astype(int)
    yb = y[g]
    for li in range(5):
        a_, b_ = cb[li], cr[li]
        key, sg = ((li, a_, b_), 1.0) if a_ <= b_ else ((li, b_, a_), -1.0)
        w, nn = cnt_mu.get(key, (0.0, 0))
        wr = (w + 0.5 * K_EB) / (nn + K_EB)                     # WR del primer campeón de la clave
        MU[g, li] = sg * (wr - 0.5)
        s_, n2 = cnt_pm.get(key, (np.zeros(len(PAR_STATS)), 0))
        PM[g, li] = sg * s_ / (n2 + K_PAR)
    for j, (l1, l2) in enumerate(SINERGIAS):
        v = []
        for e, cc in ((0, cb), (1, cr)):
            key = (j, cc[l1], cc[l2])
            w, nn = cnt_sin.get(key, (0.0, 0))
            v.append((w + 0.5 * K_EB) / (nn + K_EB) - 0.5)
        SIN[g, j] = v[0] - v[1]
    # actualizar después de describir la partida
    dif = ST[g, LIN[g, 0]][:, PAR_STATS] - ST[g, LIN[g, 1]][:, PAR_STATS]
    for li in range(5):
        a_, b_ = cb[li], cr[li]
        key, sg = ((li, a_, b_), 1.0) if a_ <= b_ else ((li, b_, a_), -1.0)
        w, nn = cnt_mu.get(key, (0.0, 0))
        cnt_mu[key] = (w + (yb if sg > 0 else 1 - yb), nn + 1)
        s_, n2 = cnt_pm.get(key, (np.zeros(len(PAR_STATS)), 0))
        cnt_pm[key] = (s_ + sg * dif[li], n2 + 1)
    for j, (l1, l2) in enumerate(SINERGIAS):
        for e, cc, ye in ((0, cb, yb), (1, cr, 1 - yb)):
            key = (j, cc[l1], cc[l2])
            w, nn = cnt_sin.get(key, (0.0, 0))
            cnt_sin[key] = (w + ye, nn + 1)
log("contracruces, sinergias y margen esperado por par listos")

# ---------------------------------------------------------------- draft (campeón × rol con signo; vocabulario del train)
CROL = {}
fil, col, val_ = [], [], []
for g in range(N):
    for k in range(10):
        if EQ[g, k] not in (AZUL_T, ROJO_T):
            continue
        key = (int(CAMP[g, k]), int(ROL[g, k]))
        if S[g] == "train":
            CROL.setdefault(key, len(CROL))
        j = CROL.get(key)
        if j is not None:
            fil.append(g); col.append(j); val_.append(1.0 if EQ[g, k] == AZUL_T else -1.0)
DR = sparse.csr_matrix((np.array(val_, np.float32), (fil, col)), shape=(N, len(CROL)), dtype=np.float32)
del fil, col, val_
log(f"draft: {len(CROL):,} pares campeón × rol")

# ---------------------------------------------------------------- modelos (C elegido por log loss en val)
tr_m = S == "train"
ev = {"train": tr_m & ok_eq, "val": val_m & ok_eq, "test": te_m & ok_eq}


def z(X):
    mu, sd = X[ev["train"]].mean(0), X[ev["train"]].std(0) + 1e-9
    return ((X - mu) / sd).astype(np.float32)


jug = [i for i, n_ in enumerate(NOM_V) if not n_.startswith("m_")]
BLOQ = {
    "medias": z(XM_V),
    "linea": z(XL_V),
    "rol": z(XL_R),
    "elo": z(ELO),
    "cruces": z(np.c_[MU, SIN]),
    "par": z(PM.reshape(N, -1)),
}
VARIANTES = [
    ("Híbrido: draft + jugadores + margen (media de los 5)", ["medias"]),
    ("1. + por línea en vez de la media", ["linea"]),
    ("2. + margen y winrate por (jugador, rol)", ["linea", "rol"]),
    ("3. + Elo causal", ["linea", "rol", "elo"]),
    ("4. + contracruces por línea y sinergias", ["linea", "rol", "elo", "cruces"]),
    ("5. + margen esperado por par de campeones (≈ escalón 9)", ["linea", "rol", "elo", "cruces", "par"]),
]


def armar(bloques, idx):
    return sparse.hstack([DR[idx]] + [sparse.csr_matrix(BLOQ[b][idx]) for b in bloques], format="csr", dtype=np.float32)


filas, preds = [], {}
for nombre, bloques in VARIANTES:
    Xtr, Xva = armar(bloques, ev["train"]), armar(bloques, ev["val"])
    mejor = None
    for C in (0.003, 0.01, 0.03, 0.1, 0.3):
        lr = LogisticRegression(C=C, max_iter=4000).fit(Xtr, y[ev["train"]])
        ll = log_loss(y[ev["val"]], lr.predict_proba(Xva)[:, 1])
        if mejor is None or ll < mejor[0]:
            mejor = (ll, C, lr)
    _, C, lr = mejor
    fila = {"modelo": nombre, "C": C, "columnas": Xtr.shape[1]}
    for s, m in ev.items():
        p = lr.predict_proba(armar(bloques, m))[:, 1]
        fila[f"AUC {s}"] = roc_auc_score(y[m], p)
        fila[f"acierto {s}"] = float(((p > 0.5) == y[m]).mean())
        if s == "test":
            preds[nombre] = p
    log(f"{nombre:58s} test: AUC {fila['AUC test']:.4f} | acierto {fila['acierto test']:.2%}  (C = {C})")
    filas.append(fila)

# IC pareado en test de cada escalón contra el anterior y del último contra el híbrido
yt = y[ev["test"]]
rng = np.random.default_rng(1)
BOOT = rng.integers(0, len(yt), size=(500, len(yt)))
nombres = [v[0] for v in VARIANTES]
for i, f in enumerate(filas):
    if i == 0:
        continue
    for ref_i, col_ in ((i - 1, "vs anterior"), (0, "vs híbrido")):
        p, q = preds[nombres[i]], preds[nombres[ref_i]]
        d = np.array([roc_auc_score(yt[s], p[s]) - roc_auc_score(yt[s], q[s]) for s in BOOT])
        da = np.array([((p[s] > .5) == yt[s]).mean() - ((q[s] > .5) == yt[s]).mean() for s in BOOT])
        f[f"ΔAUC {col_}"] = roc_auc_score(yt, p) - roc_auc_score(yt, q)
        f[f"ΔAUC {col_} IC"] = f"[{np.percentile(d, 2.5):+.4f}; {np.percentile(d, 97.5):+.4f}]"
        f[f"Δacierto {col_} IC"] = f"[{np.percentile(da, 2.5):+.2%}; {np.percentile(da, 97.5):+.2%}]"
T_ = pd.DataFrame(filas)
T_.to_csv(args.salida, index=False)
print("\nReferencia del paper (test): logística 52,55 % | NeuralAC 52,66 % | OptMatch 54,11 % | DraftRec 55,35 %")
pd.set_option("display.width", 250)
print(T_.drop(columns=[c for c in T_.columns if "train" in c]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

Path(args.salida).with_name("draftrec_mejor.json").write_text(json.dumps(
    {"elo_k": mejor_k, "ok_lin": float(ok_lin.mean()), "ok_eq": float(ok_eq.mean()),
     "primer_pick_azul": float((EQ[:, 0] == AZUL_T).mean())}, indent=1), encoding="utf-8")
log("[+] listo")
