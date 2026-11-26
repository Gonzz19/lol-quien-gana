"""
Mejoras de la huella de estilo (pedido del usuario, 2026-11-18; PROJECT_SUMMARY §10). Cuatro mejoras sobre el
nivel 1 (`fingerprint_level1.py`: distancia euclídea sin pesos sobre 23 estadísticas residualizadas por campeón,
línea y resultado), medidas todas igual: la parte que se aprende (pesos, red) se ajusta con la **mitad de los
jugadores** y el AUC de reidentificación se mide en la **otra mitad**, que el ajuste nunca vio.

* `--parte 1` — pesar las variables: Mahalanobis con la covarianza dentro del jugador ($S_w$) y LDA
  (autovectores de $S_w^{-1} S_b$, cada dirección pesada por $\\sqrt{\\lambda}$).
* `--parte 2` — hábitos de los timelines completos (orden de habilidades, primera vuelta a base, compras,
  wards, oro sin gastar, movimiento y posición, nivel 6, primera muerte, placas, control): en las partidas con
  timeline, solos y junto a las 23 estadísticas, con las mismas mitades.
* `--parte 3` — distancia aprendida no lineal (estimación del nivel 2): una red chica sobre el perfil medio,
  entrenada con pérdida contrastiva (InfoNCE) en los jugadores de ajuste.
* `--parte 4` — recencia: perfil con las partidas anteriores pesadas por antigüedad (vida media en partidas)
  para reconocer las k siguientes, contra el perfil con todas iguales.

Es descriptivo, como el nivel 1: la parte 2 usa también partidas de V (los timelines bajados). Salidas en
`data/analysis/huella/`.

    python scripts/analysis/huella_mejoras.py --parte 1      # ~4 min
    python scripts/analysis/huella_mejoras.py --parte 2      # ~10 min la primera vez (lee los timelines)
"""
import argparse
import json
import os
import sqlite3
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.features.fingerprint import (CANONICAL_LANES, STYLE_FEATURES, add_rates, hard_negative_mask,  # noqa: E402
                                      keep_min_tier, make_halves, residualize, row_auc)
from scripts.analysis.fingerprint_level1 import USECOLS, load_train, player_info  # noqa: E402
from src.ganador import rutas  # noqa: E402

SALIDA = ROOT / "data" / "analysis" / "huella"
RUTA_TL = Path(os.environ.get("HUELLA_TIMELINES", ROOT / "data" / "timelines_eventos" / "timelines.db"))
T0 = time.time()
log = lambda s: print(f"[{time.time() - T0:6.1f}s] {s}", flush=True)  # noqa: E731


# --------------------------------------------------------------------------------------------- métricas
def partir_jugadores(pids, semilla=0):
    """Mitad de ajuste y mitad de evaluación, por jugador."""
    rng = np.random.default_rng(semilla)
    p = np.array(sorted(pids))
    m = rng.random(len(p)) < 0.5
    return set(p[m]), set(p[~m])


def aprender(df, feats, pids_ajuste):
    """S_w (dentro del jugador, partidas centradas en su media) y S_b (entre medias) con los de ajuste."""
    d = df[df["player_id"].isin(pids_ajuste)]
    X = d[feats].to_numpy(float)
    mu = d.groupby("player_id")[feats].transform("mean").to_numpy(float)
    Sw = np.cov((X - mu).T) + 1e-6 * np.eye(len(feats))
    medias = d.groupby("player_id")[feats].mean()
    medias = medias[d.groupby("player_id").size() >= 10]
    Sb = np.cov(medias.to_numpy(float).T)
    ev, U = np.linalg.eigh(Sw)
    Wm = U @ np.diag(ev ** -0.5) @ U.T                       # S_w^{-1/2}
    lam, V = np.linalg.eigh(Wm @ Sb @ Wm)
    lam = np.clip(lam, 0, None)
    return {"euclidea": np.eye(len(feats)), "mahalanobis": Wm, "lda": Wm @ V @ np.diag(np.sqrt(lam))}


def perfiles(df, halves, feats, pesos=None):
    pids = list(halves)
    X = df[feats].to_numpy(float)
    pos = pd.Series(np.arange(len(df)), index=df.index)
    A = np.stack([X[pos[halves[p]["A"]]].mean(0) for p in pids])
    B = np.stack([X[pos[halves[p]["B"]]].mean(0) for p in pids])
    return pids, A, B


def auc_ic(D, mask=None, semilla=0, n_boot=500):
    r = row_auc(D, mask)
    r = r[np.isfinite(r)]
    rng = np.random.default_rng(semilla)
    b = [r[rng.integers(0, len(r), len(r))].mean() for _ in range(n_boot)]
    return float(r.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def medir(A, B, T, mask):
    D = cdist(A @ T, B @ T)
    auc, lo, hi = auc_ic(D)
    dif = auc_ic(D, mask)[0] if mask is not None else np.nan
    top1 = float((D.argmin(1) == np.arange(len(D))).mean())
    return {"auc": auc, "lo": lo, "hi": hi, "auc_dificil": dif, "top1": top1, "n": len(D)}


def imprimir(fila):
    print(f"  {fila['parte']:<10} {fila['muestra']:<20} k={fila['k']:<3} {fila['variante']:<30} "
          f"n={fila['n']:>5} AUC {fila['auc']:.3f} [{fila['lo']:.3f}; {fila['hi']:.3f}] "
          f"difícil {fila['auc_dificil']:.3f} top1 {fila['top1']:.3f}", flush=True)


def cargar_resid(residuo="ajuste"):
    """Train Master+ residualizado y la partición de jugadores (ajuste, evaluación).

    La partición se hace sobre los jugadores crudos, antes de residualizar, y con `residuo="ajuste"` la media y
    el desvío de cada grupo campeón-línea-resultado salen solo de los jugadores de ajuste (§9.S.77). Con
    `residuo="global"` se reproduce la versión anterior, que normalizaba también con los de evaluación.
    """
    raw = load_train(ROOT / "data" / "splits" / "train" / "full_dataset.csv", "MASTER")
    ajuste, evalu = partir_jugadores(raw["player_id"].unique())
    resid = residualize(raw, ajuste=ajuste if residuo == "ajuste" else None)
    log(f"train Master+: {len(resid):,} filas residualizadas ({residuo}), {resid['player_id'].nunique():,} "
        f"jugadores")
    return resid, ajuste, evalu


# --------------------------------------------------------------------------------------------- parte 1
def parte1(resid, info, ajuste, evalu):
    Ts = aprender(resid, STYLE_FEATURES, ajuste)
    ev = resid[resid["player_id"].isin(evalu)]
    filas = []
    for muestra, k in (("campeones_distintos", 10), ("campeones_distintos", 20), ("aleatoria", 10),
                       ("temporal", 20)):
        halves = make_halves(ev, k=k, mode=muestra, seed=0)
        pids, A, B = perfiles(ev, halves, STYLE_FEATURES)
        mask = hard_negative_mask(pids, info)
        for nom, T in Ts.items():
            f = {"parte": "1", "muestra": muestra, "k": k, "variante": nom, **medir(A, B, T, mask)}
            filas.append(f)
            imprimir(f)
    return filas, Ts


# --------------------------------------------------------------------------------------------- parte 3
def parte3(resid, info, Ts, ajuste, evalu, pasos=3000, semilla=0):
    import torch
    torch.manual_seed(semilla)
    X = torch.tensor(resid[STYLE_FEATURES].to_numpy(np.float32))
    grupos = {p: torch.tensor(ix) for p, ix in
              resid.reset_index(drop=True).groupby("player_id").indices.items() if p in ajuste and len(ix) >= 40}
    lista = list(grupos)
    d = len(STYLE_FEATURES)
    red = torch.nn.Sequential(torch.nn.Linear(d, 128), torch.nn.GELU(), torch.nn.Linear(128, 128), torch.nn.GELU(),
                              torch.nn.Linear(128, 32))
    opt = torch.optim.AdamW(red.parameters(), lr=2e-3, weight_decay=1e-4)
    rng = np.random.default_rng(semilla)
    for paso in range(pasos):
        k = int(rng.integers(5, 21))
        lote = rng.choice(len(lista), 256, replace=False)
        a, b = [], []
        for j in lote:
            ix = grupos[lista[j]]
            perm = ix[torch.randperm(len(ix))[: 2 * k]]
            a.append(X[perm[:k]].mean(0))
            b.append(X[perm[k:]].mean(0))
        za = torch.nn.functional.normalize(red(torch.stack(a)), dim=1)
        zb = torch.nn.functional.normalize(red(torch.stack(b)), dim=1)
        logits = za @ zb.T / 0.1
        y = torch.arange(len(lote))
        loss = (torch.nn.functional.cross_entropy(logits, y) + torch.nn.functional.cross_entropy(logits.T, y)) / 2
        opt.zero_grad()
        loss.backward()
        opt.step()
        if paso % 500 == 0:
            log(f"  red: paso {paso}, pérdida {loss.item():.3f}")
    red.eval()
    ev = resid[resid["player_id"].isin(evalu)]
    filas = []
    for muestra, k in (("campeones_distintos", 10), ("campeones_distintos", 20), ("aleatoria", 10)):
        halves = make_halves(ev, k=k, mode=muestra, seed=0)
        pids, A, B = perfiles(ev, halves, STYLE_FEATURES)
        mask = hard_negative_mask(pids, info)
        with torch.no_grad():
            Ea = torch.nn.functional.normalize(red(torch.tensor(A, dtype=torch.float32)), dim=1).numpy()
            Eb = torch.nn.functional.normalize(red(torch.tensor(B, dtype=torch.float32)), dim=1).numpy()
        for nom, (a, b, T) in {"mahalanobis (referencia)": (A, B, Ts["mahalanobis"]),
                               "red contrastiva": (Ea, Eb, np.eye(Ea.shape[1]))}.items():
            f = {"parte": "3", "muestra": muestra, "k": k, "variante": nom, **medir(a, b, T, mask)}
            filas.append(f)
            imprimir(f)
    return filas


# --------------------------------------------------------------------------------------------- parte 4
def parte4(resid, info, Ts, evalu, k=10, previas=60):
    ev = resid[resid["player_id"].isin(evalu)].sort_values(["player_id", "game_creation"])
    X = ev[STYLE_FEATURES].to_numpy(float)
    idx = ev.groupby("player_id").indices
    pids = [p for p, ix in idx.items() if len(ix) >= previas + k]
    mask = hard_negative_mask(pids, info)
    B = np.stack([X[idx[p][-k:]].mean(0) for p in pids])
    filas = []
    for vida in (None, 40, 20, 10):
        A = []
        for p in pids:
            prev = X[idx[p][-k - previas:-k]]
            edad = np.arange(len(prev))[::-1]                   # 0 = la más reciente
            w = np.ones(len(prev)) if vida is None else 0.5 ** (edad / vida)
            A.append((w[:, None] * prev).sum(0) / w.sum())
        nom = "todas iguales" if vida is None else f"vida media {vida} partidas"
        f = {"parte": "4", "muestra": f"siguientes (previas {previas})", "k": k, "variante": nom,
             **medir(np.stack(A), B, Ts["mahalanobis"], mask)}
        filas.append(f)
        imprimir(f)
    return filas


# --------------------------------------------------------------------------------------------- parte 2
HABITOS = ["max_q", "max_w", "max_e", "l1_q", "l1_w", "l1_e", "primera_vuelta_min", "visitas_tienda_10",
           "control_wards_15", "wards_puestos_10", "wards_rotos_15", "nivel6_min", "oro_sin_gastar_med",
           "movimiento_2_10", "avance_5_10", "primera_muerte_min", "placas_14", "control_15"]


def _habitos_partida(blob):
    t = json.loads(zlib.decompress(blob))
    fr = t["info"]["frames"]
    if len(fr) <= 15:
        return None
    ev = sorted((e for f in fr for e in f["events"]), key=lambda e: e["timestamp"])
    H = {p: {h: 0.0 for h in HABITOS} for p in range(1, 11)}
    puntos = {p: [0, 0, 0] for p in range(1, 11)}
    nivel_hab = {p: 0 for p in range(1, 11)}
    ult_compra = {p: -1e9 for p in range(1, 11)}
    for p in H:
        H[p]["primera_vuelta_min"] = H[p]["nivel6_min"] = H[p]["primera_muerte_min"] = 20.0
    for e in ev:
        ts, ty = e["timestamp"], e["type"]
        if ty == "SKILL_LEVEL_UP" and e.get("levelUpType") == "NORMAL":
            p, s = e.get("participantId", 0), e.get("skillSlot", 0)
            if 1 <= p <= 10 and 1 <= s <= 3:
                nivel_hab[p] += 1
                if nivel_hab[p] == 1:
                    H[p][("l1_q", "l1_w", "l1_e")[s - 1]] = 1.0
                if nivel_hab[p] <= 9:
                    puntos[p][s - 1] += 1
        elif ty == "ITEM_PURCHASED":
            p = e.get("participantId", 0)
            if 1 <= p <= 10:
                if ts > 90_000 and H[p]["primera_vuelta_min"] == 20.0 and ts < 1_200_000:
                    H[p]["primera_vuelta_min"] = ts / 60000
                if ts < 600_000 and ts - ult_compra[p] > 30_000:
                    H[p]["visitas_tienda_10"] += 1
                ult_compra[p] = ts
                if e.get("itemId") == 2055 and ts < 900_000:
                    H[p]["control_wards_15"] += 1
        elif ty == "WARD_PLACED" and ts < 600_000 and 1 <= e.get("creatorId", 0) <= 10:
            H[e["creatorId"]]["wards_puestos_10"] += 1
        elif ty == "WARD_KILL" and ts < 900_000 and 1 <= e.get("killerId", 0) <= 10:
            H[e["killerId"]]["wards_rotos_15"] += 1
        elif ty == "LEVEL_UP" and e.get("level") == 6 and 1 <= e.get("participantId", 0) <= 10:
            H[e["participantId"]]["nivel6_min"] = min(H[e["participantId"]]["nivel6_min"], ts / 60000)
        elif ty == "CHAMPION_KILL" and 1 <= e.get("victimId", 0) <= 10:
            v = e["victimId"]
            H[v]["primera_muerte_min"] = min(H[v]["primera_muerte_min"], ts / 60000)
        elif ty == "TURRET_PLATE_DESTROYED" and ts < 840_000 and 1 <= e.get("killerId", 0) <= 10:
            H[e["killerId"]]["placas_14"] += 1
    for p in range(1, 11):
        pts = puntos[p]
        if sum(pts):
            H[p][("max_q", "max_w", "max_e")[int(np.argmax(pts))]] = 1.0
        oro, mov, av = [], 0.0, []
        prev = None
        for m in range(2, 16):
            d = fr[m]["participantFrames"][str(p)]
            if m >= 3:
                oro.append(d["currentGold"])
            x, y = d["position"]["x"], d["position"]["y"]
            if m <= 10:
                if prev is not None:
                    mov += np.hypot(x - prev[0], y - prev[1])
                prev = (x, y)
                if m >= 5:
                    s = (x + y) / 2 if p <= 5 else 14800 - (x + y) / 2     # avance hacia la base rival
                    av.append(s)
        H[p]["oro_sin_gastar_med"] = float(np.mean(oro))
        H[p]["movimiento_2_10"] = mov / 1000
        H[p]["avance_5_10"] = float(np.mean(av)) / 1000
        H[p]["control_15"] = fr[15]["participantFrames"][str(p)].get("timeEnemySpentControlled", 0) / 1000
    return H


def _lote(filas):
    out = []
    for mid, blob in filas:
        try:
            H = _habitos_partida(blob)
        except Exception:
            H = None
        if H:
            for p, h in H.items():
                out.append((mid, p, *[h[k] for k in HABITOS]))
    return out


def habitos(hilos=6):
    cache = SALIDA / "habitos.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    filas = []
    with closing(sqlite3.connect(f"file:{RUTA_TL.as_posix()}?mode=ro", uri=True)) as con, \
            ProcessPoolExecutor(hilos) as ex:
        pend, n, ultimo = [], 0, 0
        while True:
            # tandas cortas por rowid: cada consulta termina y suelta el lock compartido, así el descargador
            # (que escribe en la misma base) puede guardar entre tanda y tanda
            lote = con.execute("SELECT rowid, match_id, json_zlib FROM timeline_crudo WHERE rowid > ? "
                               "ORDER BY rowid LIMIT 400", (ultimo,)).fetchall()
            if not lote:
                break
            ultimo = lote[-1][0]
            lote = [(m, b) for _, m, b in lote]
            pend.append(ex.submit(_lote, lote))
            if len(pend) >= 2 * hilos:
                filas += pend.pop(0).result()
                n += 1
                if n % 25 == 0:
                    log(f"  {len(filas) // 10:,} timelines leídos")
        for f in pend:
            filas += f.result()
    df = pd.DataFrame(filas, columns=["match_id", "participant_id", *HABITOS])
    df.to_pickle(cache)
    return df


def timeline_train():
    """Partidas Master+ de LAS con timeline completo: 23 estadísticas + 18 hábitos crudos, sin faltantes.

    Se cachea en `timeline_train.pkl` (lee el CSV de train entero, unos minutos).
    """
    cache = SALIDA / "timeline_train.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    H = habitos()
    log(f"hábitos: {H['match_id'].nunique():,} partidas")
    mids = set(H["match_id"])
    cols = list(dict.fromkeys(USECOLS + ["participant_id"]))
    df = pd.concat([ch[ch["match_id"].isin(mids)]
                    for ch in pd.read_csv(rutas.TRAIN_CSV, usecols=cols, chunksize=500_000)])
    df = df[df["match_id"].str.startswith("LA2_") & df["lane"].isin(CANONICAL_LANES)]
    df = add_rates(keep_min_tier(df, "MASTER")).merge(H, on=["match_id", "participant_id"], how="inner")
    df = df.dropna(subset=STYLE_FEATURES + HABITOS).reset_index(drop=True)
    SALIDA.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache)
    return df


def residuo_habitos(d, feats, ajuste=None, min_group=30):
    """z dentro de (campeón, línea, resultado), en grupos con >= `min_group` partidas.

    A diferencia de `residualize`, un hábito constante en el grupo (p. ej. el campeón siempre sube Q primero)
    vale 0 en vez de perder la fila. Con `ajuste` (conjunto de `player_id`), media, desvío y tamaño del grupo
    salen solo de esos jugadores y se aplican a todas las filas (§9.S.77); sin `ajuste`, de todas (la versión
    anterior, descriptiva).
    """
    by = ["champion_name", "lane", "win"]
    d = d.dropna(subset=feats)
    fit = d if ajuste is None else d[d["player_id"].isin(ajuste)]
    g = fit.groupby(by)[feats]
    grande = g.size() >= min_group
    mu, sd = g.mean()[grande], g.std()[grande].replace(0, np.nan)
    d = d.join(mu.add_suffix("__mu"), on=by, how="inner").join(sd.add_suffix("__sd"), on=by)
    for f in feats:
        d[f] = ((d[f] - d[f + "__mu"]) / d[f + "__sd"]).fillna(0.0)
    return d.drop(columns=[c for c in d.columns if c.endswith(("__mu", "__sd"))])


def cargar_timeline(residuo="ajuste", crudo=None):
    """Partidas con timeline residualizadas (estadísticas + hábitos) y la partición (ajuste, evaluación).

    Como `cargar_resid`: la partición se hace sobre los jugadores crudos, antes de residualizar, y con
    `residuo="ajuste"` la normalización sale solo de los jugadores de ajuste. `residuo="global"` reproduce la
    versión anterior con la misma partición. `crudo` reemplaza a `timeline_train()` (tests).
    """
    d = timeline_train() if crudo is None else crudo
    ajuste, evalu = partir_jugadores(d["player_id"].unique())
    resid = residuo_habitos(d, STYLE_FEATURES + HABITOS, ajuste=ajuste if residuo == "ajuste" else None)
    log(f"con timeline y residualizadas ({residuo}): {len(resid):,} filas, {resid['player_id'].nunique():,} "
        f"jugadores")
    return resid, ajuste, evalu


def parte2(residuo="ajuste"):
    resid, ajuste, evalu = cargar_timeline(residuo)
    feats = STYLE_FEATURES + HABITOS
    info = player_info(resid)
    ev = resid[resid["player_id"].isin(evalu)]
    filas = []
    for muestra, k in (("campeones_distintos", 10), ("campeones_distintos", 20), ("aleatoria", 10)):
        halves = make_halves(ev, k=k, mode=muestra, seed=0)
        if len(halves) < 200:
            continue
        for nom_f, fs in (("23 estadísticas", STYLE_FEATURES), ("solo hábitos", HABITOS), ("estadísticas + hábitos", feats)):
            Ts = aprender(resid, fs, ajuste)
            pids, A, B = perfiles(ev, halves, fs)
            mask = hard_negative_mask(pids, info)
            for nom_m in ("euclidea", "mahalanobis", "lda"):
                f = {"parte": "2", "muestra": muestra, "k": k, "variante": f"{nom_f} · {nom_m}",
                     **medir(A, B, Ts[nom_m], mask)}
                filas.append(f)
                imprimir(f)
    # cuánto de cada hábito es del jugador (ICC, jugadores con >= 30 partidas con timeline)
    n = resid.groupby("player_id").size()
    d = resid[resid["player_id"].isin(n[n >= 30].index)]
    g = d.groupby("player_id")[feats]
    w = g.var().mean()
    b = g.mean().var() - w / g.size().mean()
    icc = (b / (b + w)).sort_values(ascending=False)
    log("ICC (parte de la varianza que es del jugador):")
    for k_, v in icc.items():
        print(f"    {k_:<24} {v:.3f}")
    return filas, icc


def parte_red_habitos(residuo="ajuste", pasos=3000, semilla=0):
    """Red contrastiva sobre 23 estadísticas + 18 hábitos (misma arquitectura y split que huella_niveles.py)."""
    from scripts.analysis.huella_niveles import datos_timeline, entrenar_red
    resid, ajuste, evalu = datos_timeline(residuo)
    feats = STYLE_FEATURES + HABITOS
    log(f"timelines (red contrastiva): {len(resid):,} filas, {resid['player_id'].nunique():,} jugadores")
    info = player_info(resid)
    ev = resid[resid["player_id"].isin(evalu)]
    emb = entrenar_red(resid, feats, ajuste, pasos=pasos, semilla=semilla)
    filas = []
    for muestra, k in (("campeones_distintos", 10), ("campeones_distintos", 20), ("campeones_distintos", 30),
                       ("campeones_distintos", 40), ("campeones_distintos", 50), ("aleatoria", 10)):
        halves = make_halves(ev, k=k, mode=muestra, seed=0)
        if len(halves) < 200:
            continue
        pids, A, B = perfiles(ev, halves, feats)
        mask = hard_negative_mask(pids, info)
        Ea, Eb = emb(A), emb(B)
        f = {"parte": "red_habitos", "muestra": muestra, "k": k,
             "variante": "estadísticas + hábitos · red contrastiva",
             **medir(Ea, Eb, np.eye(Ea.shape[1]), mask)}
        filas.append(f)
        imprimir(f)
    return filas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parte", required=True, choices=["1", "2", "3", "4", "134", "red_habitos"])
    ap.add_argument("--residuo", choices=["ajuste", "global"], default="ajuste",
                    help="normalizar por grupo solo con los jugadores de ajuste (default) o con todos, como antes "
                         "del §9.S.77 (la salida lleva el sufijo _residuo_global)")
    ap.add_argument("--timelines", type=Path, default=None,
                    help="partes 2 y red_habitos: ruta de timelines.db (default data/timelines_eventos/; también "
                         "la variable HUELLA_TIMELINES). Se abre solo en lectura")
    args = ap.parse_args()
    global RUTA_TL
    if args.timelines is not None:
        RUTA_TL = args.timelines
    SALIDA.mkdir(parents=True, exist_ok=True)
    filas = []
    if args.parte in ("1", "3", "4", "134"):
        resid, ajuste, evalu = cargar_resid(args.residuo)
        info = player_info(resid)
        f1, Ts = parte1(resid, info, ajuste, evalu)
        if args.parte in ("1", "134"):
            filas += f1
        if args.parte in ("3", "134"):
            filas += parte3(resid, info, Ts, ajuste, evalu)
        if args.parte in ("4", "134"):
            filas += parte4(resid, info, Ts, evalu)
    if args.parte == "2":
        f2, icc = parte2(args.residuo)
        filas += f2
        sufijo = "_residuo_global" if args.residuo == "global" else ""
        icc.round(4).to_json(SALIDA / f"icc_habitos{sufijo}.json", force_ascii=False, indent=2)
    if args.parte == "red_habitos":
        filas += parte_red_habitos(args.residuo)
    out = SALIDA / (f"mejoras_{args.parte}.csv" if args.parte == "red_habitos" else f"mejoras_parte{args.parte}.csv")
    if args.residuo == "global":
        out = out.with_name(out.stem + "_residuo_global.csv")
    pd.DataFrame(filas).to_csv(out, index=False)
    log(f"listo: {out}")


if __name__ == "__main__":
    main()
