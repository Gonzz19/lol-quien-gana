"""
Nuestra metodología sobre el dataset público de DraftRec (Lee et al., WWW 2022; PROJECT_SUMMARY.md §9.S.56).

Pedido del usuario (2026-11-16): comparar contra un trabajo publicado en SUS datos. DraftRec reporta en su test
(LoL, Master+ y Diamond 1-2, cola solo/dúo, junio-septiembre 2021, split temporal 85/5/10):
logística 52,55 %, NeuralAC 52,66 %, OptMatch 54,11 %, DraftRec 55,35 % de acierto (sin AUC).

Datos (https://github.com/dojeon-ai/DraftRec, en `data/external/draftrec/`):
* `train.csv`, `val.csv`, `test.csv`: una fila por partida; User1..User10 en orden de pick (azul = 1, 4, 5, 8, 9;
  rojo = 2, 3, 6, 7, 10); cada celda es (jugador, índice de esa partida en su historial).
* `user_history.npy`: float32 (62.466 jugadores, 1.145 partidas, 57). Por partida: campeón, rol, equipo, 10 bans,
  victoria del equipo (5 gana, 4 pierde) y 43 estadísticas de final de partida ya estandarizadas con el train.
  El historial está ordenado por fecha, así que lo previo a la partida es historial[:índice] (causal).

Lo que se puede reconstruir de nuestro híbrido (no hay línea de tiempo, ni rango, ni fechas por partida):
* Draft: campeón × rol con signo (+1 azul, −1 rojo), como §9.S.20.
* Por jugador, solo con partidas anteriores: partidas totales, partidas con el campeón, estreno, porción del
  campeón, winrate con el campeón (encogido), winrate total y de las últimas 20, porción del rol y fuera de rol
  (Champion Mastery, Pool, Rol Mastery, WR) y un **margen de final de partida**: estadística − media de ese
  campeón en el train, acumulado, últimas 20 y con ese campeón (nuestro margen usa @10/@15 contra el rival de
  línea; acá solo existen los totales del final).
* Todo entra como azul − rojo (media de los 5) en una logística; C se elige con val y se mide en test.

    python scripts/analysis/draftrec_comparacion.py            # ~15-25 min (lee los 16 GB dos veces)
"""
import argparse
import ast
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
ap.add_argument("--salida", default=str(REPO / "data" / "analysis" / "draftrec_comparacion.csv"))
args = ap.parse_args()
DIR = Path(args.dir)
T0 = time.time()
log = lambda s: print(f"[{time.time() - T0:6.0f}s] {s}", flush=True)  # noqa: E731

AZUL = [0, 3, 4, 7, 8]                     # User1, 4, 5, 8, 9 (índices 0-based)
ROJO = [1, 2, 5, 6, 9]                     # User2, 3, 6, 7, 10
# estadísticas para el margen (índices dentro de los 43 'stats' de categorical_ids.json)
STATS = {"kda": 0, "dano_campeones": 15, "dano_objetivos": 22, "vision": 24, "dano_recibido": 26,
         "oro": 30, "cs": 34, "cs_jungla": 35}
SI = np.array(list(STATS.values())) + 14   # columnas en el arreglo de 57
K_CH, K_WR = 5, 10                          # encogimiento con el campeón y del winrate

# ---------------------------------------------------------------- partidas
splits = {}
for s in ("train", "val", "test"):
    d = pd.read_csv(DIR / f"{s}.csv")
    pares = np.array([[ast.literal_eval(v) for v in d[f"User{k}"]] for k in range(1, 11)])  # (10, n, 2)
    splits[s] = {"u": pares[:, :, 0].T.astype(np.int64), "h": pares[:, :, 1].T.astype(np.int64), "t": d.time.to_numpy()}
    log(f"{s}: {len(d):,} partidas")

A = np.load(DIR / "user_history.npy", mmap_mode="r")
NU, L, F = A.shape
log(f"historial: {A.shape}")

# (jugador, índice) que hacen falta y a qué split pertenecen
need = {}
for s, d in splits.items():
    for u, h in zip(d["u"].ravel(), d["h"].ravel()):
        need.setdefault(int(u), set()).add(int(h))
en_train = {}
for u, h in zip(splits["train"]["u"].ravel(), splits["train"]["h"].ravel()):
    en_train.setdefault(int(u), set()).add(int(h))

# ---------------------------------------------------------------- pasada 1: media de cada estadística por campeón (train)
NC = 200
suma, cnt = np.zeros((NC, len(SI))), np.zeros(NC)
for u, hs in en_train.items():
    if u >= NU:
        continue
    H = np.asarray(A[u])
    idx = np.fromiter(hs, int)
    c = H[idx, 0].astype(int)
    np.add.at(suma, c, H[idx][:, SI])
    np.add.at(cnt, c, 1)
media_camp = suma / np.maximum(cnt, 1)[:, None]
log("pasada 1 lista (medias por campeón del train)")

# ---------------------------------------------------------------- pasada 2: variables previas de cada (jugador, índice)
NOMBRES = (["log_n", "log_n_camp", "estreno", "share_camp", "wr_camp", "wr_tot", "wr_20", "share_rol", "fuera_rol"]
           + [f"m_acum_{k}" for k in STATS] + [f"m_20_{k}" for k in STATS] + [f"m_camp_{k}" for k in STATS])
VAR, INFO = {}, {}                          # INFO: (campeón, rol, equipo, victoria) de la partida
for n_u, (u, hs) in enumerate(need.items()):
    if u >= NU:
        continue
    H = np.asarray(A[u])
    n = int((H[:, 0] != 0).sum())
    camp, rol = H[:n, 0].astype(int), H[:n, 1].astype(int)
    gana = (H[:n, 13] == 5).astype(float)
    marg = H[:n][:, SI] - media_camp[camp]                    # rendimiento contra lo esperado con ese campeón
    cw = np.r_[0, np.cumsum(gana)]                            # sumas acumuladas: índice h = partidas < h
    cm = np.r_[np.zeros((1, len(SI))), np.cumsum(marg, axis=0)]
    for h in hs:
        if h >= n:
            continue
        c, r = camp[h], rol[h]
        INFO[(u, h)] = (c, r, int(H[h, 2]), int(H[h, 13]))
        prev_c = camp[:h] == c
        nc = prev_c.sum()
        wr_tot = (cw[h] + 0.5 * K_WR) / (h + K_WR)
        wr_c = (gana[:h][prev_c].sum() + K_CH * wr_tot) / (nc + K_CH)
        a20 = max(0, h - 20)
        wr20 = (cw[h] - cw[a20] + 0.5 * K_WR) / (h - a20 + K_WR)
        nr = (rol[:h] == r).sum()
        m_acum = cm[h] / max(h, 1)
        m_20 = (cm[h] - cm[a20]) / max(h - a20, 1)
        m_c = (marg[:h][prev_c].sum(0) + K_CH * m_acum) / (nc + K_CH)
        VAR[(u, h)] = np.r_[np.log1p(h), np.log1p(nc), float(nc == 0), nc / (h + 1), wr_c - wr_tot, wr_tot, wr20,
                            (nr + 1) / (h + 2), float(h >= 10 and nr / h < 0.2), m_acum, m_20, m_c]
    if n_u % 10000 == 0:
        log(f"  pasada 2: {n_u:,}/{len(need):,} jugadores")
log("pasada 2 lista")

# ---------------------------------------------------------------- matrices por partida
CROL = {}


def armar(s):
    d = splits[s]
    n = len(d["t"])
    V = np.zeros((n, 10, len(NOMBRES)))
    camp = np.zeros((n, 10), int)
    rol = np.zeros((n, 10), int)
    y = np.zeros(n)
    for i in range(n):
        for k in range(10):
            u, h = int(d["u"][i, k]), int(d["h"][i, k])
            v = VAR.get((u, h))
            if v is not None:
                V[i, k] = v
            info = INFO.get((u, h))
            if info is not None:
                camp[i, k], rol[i, k] = info[0], info[1]
                if k == 0:                                     # User1 es azul: gana azul = gana su equipo
                    y[i] = float(info[3] == 5) if info[2] == 4 else float(info[3] != 5)
    Xp = V[:, AZUL].mean(1) - V[:, ROJO].mean(1)              # jugadores: azul − rojo
    filas, cols, vals = [], [], []
    for i in range(n):
        for ks, sg in ((AZUL, 1.0), (ROJO, -1.0)):
            for k in ks:
                key = (camp[i, k], rol[i, k])
                if s == "train":
                    CROL.setdefault(key, len(CROL))
                j = CROL.get(key)
                if j is not None:
                    filas.append(i); cols.append(j); vals.append(sg)
    return Xp, (filas, cols, vals, n), y


XP, DR, Y = {}, {}, {}
for s in ("train", "val", "test"):
    XP[s], DR[s], Y[s] = armar(s)
    log(f"{s}: matrices listas; gana azul {Y[s].mean():.1%}")
for s in DR:
    f, c, v, n = DR[s]
    DR[s] = sparse.csr_matrix((v, (f, c)), shape=(n, len(CROL)))
mu, sd = XP["train"].mean(0), XP["train"].std(0) + 1e-9
XP = {s: (x - mu) / sd for s, x in XP.items()}


def modelo(nombre, armar_X):
    mejor = None
    for C in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        lr = LogisticRegression(C=C, max_iter=3000).fit(armar_X("train"), Y["train"])
        ll = log_loss(Y["val"], lr.predict_proba(armar_X("val"))[:, 1])
        if mejor is None or ll < mejor[0]:
            mejor = (ll, C, lr)
    _, C, lr = mejor
    fila = {"modelo": nombre, "C": C}
    for s in ("train", "val", "test"):
        p = lr.predict_proba(armar_X(s))[:, 1]
        fila[f"AUC {s}"] = roc_auc_score(Y[s], p)
        fila[f"acierto {s}"] = float(((p > 0.5) == Y[s]).mean())
    log(f"{nombre:42s} test: AUC {fila['AUC test']:.4f} | acierto {fila['acierto test']:.2%}  (C = {C})")
    return fila


filas = [{"modelo": "Baseline (siempre la tasa del train)", "C": np.nan,
          **{f"AUC {s}": 0.5 for s in Y}, **{f"acierto {s}": float(((Y["train"].mean() > 0.5) == Y[s]).mean()) for s in Y}}]
filas.append(modelo("Logística Draft (campeón × rol)", lambda s: DR[s]))
jug = [i for i, n in enumerate(NOMBRES) if not n.startswith("m_")]
filas.append(modelo("Jugadores (sin margen)", lambda s: XP[s][:, jug]))
filas.append(modelo("Jugadores + margen de final de partida", lambda s: XP[s]))
filas.append(modelo("Híbrido: draft + jugadores + margen", lambda s: sparse.hstack([DR[s], sparse.csr_matrix(XP[s])]).tocsr()))
T = pd.DataFrame(filas)
T.to_csv(args.salida, index=False)
print("\nReferencia del paper (test): logística 52,55 % | NeuralAC 52,66 % | OptMatch 54,11 % | DraftRec 55,35 %")
print(T.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
log(f"[+] {args.salida}")
