"""
¿Cuánto aportan los eventos del timeline al modelo en vivo? (piloto, pedido del usuario, 2026-11-17)

Lee los timelines crudos de `data/timelines_eventos/timelines.db` (`scripts/collection/timelines_eventos.py`) y
arma, para los minutos 10, 15 y 20, variables azul − rojo con **solo** lo ocurrido antes del minuto
(`timestamp < 60 000 · m`; frames hasta el índice m):

* Prioridad 1 — objetivos y estado instantáneo:
  dragones (total y punto de alma: 3+), larvas (`HORDE`), Heraldo, torres por nivel (exterior / interior /
  base), placas, inhibidores; oro sin gastar y oro invertido (total − sin gastar); muertos en el minuto m
  (momento de la muerte + temporizador por nivel); shutdowns cobrados.
* Prioridad 2 — tiempos y momentum:
  primera sangre (lado y minuto) y primera torre (lado y minuto); kills y Δ oro en los últimos 3 minutos;
  wards colocados y destruidos.

Medición (dos etapas, para no perder las ~94k partidas del modelo base):
1. Base = mejor logística en vivo (`3_Abl_A2_Stats_Traj_Exp`, C = 0,01) ajustada con el ajuste Master+
   (chunks 1-6) **sin** las partidas de la muestra → logit fuera de la muestra para la muestra y para V.
2. Segunda etapa: logística sobre [logit base, eventos] ajustada con la muestra de entrenamiento; se compara
   contra la segunda etapa con el logit solo (misma muestra) en la muestra de V, con bootstrap pareado.
   Bloques: la muestra de entrenamiento partida en 5 tramos por fecha; se predicen los tramos 3, 4 y 5 hacia
   adelante (cada uno con los anteriores).

    python scripts/analysis/ingame_eventos.py      # minutos, según cuántos timelines haya
"""
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.models.ingame import MINUTES, extract_minute_features  # noqa: E402

RUTA_DB = REPO / "data" / "timelines_eventos" / "timelines.db"
IN_DIR = REPO / "data" / "analysis" / "ingame"
SALIDA = IN_DIR / "eventos"
# temporizador de muerte (s) por nivel 1..18 (aprox. de la wiki; el escalado tardío casi no pesa antes del 20)
TIMER = np.array([10, 10, 12, 12, 14, 16, 20, 25, 28, 32.5, 35, 37.5, 40, 42.5, 45, 47.5, 50, 52.5])
T0 = time.time()
log = lambda s: print(f"[{time.time() - T0:6.1f}s] {s}", flush=True)  # noqa: E731

P1 = ["dragones", "alma_punto", "larvas", "heraldo", "torres_ext", "torres_int", "torres_base", "placas",
      "inhibidores", "oro_sin_gastar", "oro_invertido", "muertos_ahora", "shutdowns"]
P2 = ["fb_lado", "fb_minuto_lado", "ft_lado", "ft_minuto_lado", "kills_3min", "doro_3min", "wards_puestos",
      "wards_rotos"]


def lado(pid):
    return 0 if pid <= 5 else 1                              # participantes 1-5 = azul (100), 6-10 = rojo (200)


def variables_partida(tl):
    """dict minuto -> vector (P1 + P2) azul − rojo."""
    fr = tl["info"]["frames"]
    ev = sorted((e for f in fr for e in f["events"]), key=lambda e: e["timestamp"])
    out = {}
    for m in MINUTES:
        if len(fr) <= m:
            continue
        lim = 60000 * m
        v = {k: np.zeros(2) for k in P1 + P2}
        fb = ft = None
        nivel = {p: 1 for p in range(1, 11)}
        muerte = {}
        for e in ev:
            if e["timestamp"] >= lim:
                break
            t = e["type"]
            if t == "LEVEL_UP":
                nivel[e["participantId"]] = e["level"]
            elif t == "CHAMPION_KILL":
                vic = e["victimId"]
                muerte[vic] = e["timestamp"] + 1000 * TIMER[min(nivel.get(vic, 1), 18) - 1]
                k = e.get("killerId", 0)
                if 1 <= k <= 10:
                    if e.get("shutdownBounty", 0) > 0:
                        v["shutdowns"][lado(k)] += 1
                    if e["timestamp"] >= lim - 180000:
                        v["kills_3min"][lado(k)] += 1
            elif t == "CHAMPION_SPECIAL_KILL" and e.get("killType") == "KILL_FIRST_BLOOD" and fb is None:
                fb = (lado(e["killerId"]), e["timestamp"] / 60000)
            elif t == "ELITE_MONSTER_KILL":
                eq = 0 if e.get("killerTeamId") == 100 else 1
                mt = e.get("monsterType")
                if mt == "DRAGON":
                    v["dragones"][eq] += 1
                elif mt == "HORDE":
                    v["larvas"][eq] += 1
                elif mt == "RIFTHERALD":
                    v["heraldo"][eq] += 1
            elif t == "BUILDING_KILL":
                eq = 1 if e.get("teamId") == 100 else 0             # teamId = dueño del edificio destruido
                if e.get("buildingType") == "INHIBITOR_BUILDING":
                    v["inhibidores"][eq] += 1
                else:
                    tt = e.get("towerType", "")
                    clave = {"OUTER_TURRET": "torres_ext", "INNER_TURRET": "torres_int"}.get(tt, "torres_base")
                    v[clave][eq] += 1
                    if ft is None:
                        ft = (eq, e["timestamp"] / 60000)
            elif t == "TURRET_PLATE_DESTROYED":
                v["placas"][1 if e.get("teamId") == 100 else 0] += 1
            elif t == "WARD_PLACED" and 1 <= e.get("creatorId", 0) <= 10:
                v["wards_puestos"][lado(e["creatorId"])] += 1
            elif t == "WARD_KILL" and 1 <= e.get("killerId", 0) <= 10:
                v["wards_rotos"][lado(e["killerId"])] += 1
        for eq in (0, 1):
            v["alma_punto"][eq] = float(v["dragones"][eq] >= 3)
        for p, hasta in muerte.items():
            if hasta > lim:
                v["muertos_ahora"][lado(p)] += 1
        pf = fr[m]["participantFrames"]
        pf3 = fr[m - 3]["participantFrames"]
        for p in range(1, 11):
            d = pf[str(p)]
            v["oro_sin_gastar"][lado(p)] += d["currentGold"] / 1000
            v["oro_invertido"][lado(p)] += (d["totalGold"] - d["currentGold"]) / 1000
            v["doro_3min"][lado(p)] += (d["totalGold"] - pf3[str(p)]["totalGold"]) / 1000
        x = {k: v[k][0] - v[k][1] for k in P1 + P2}
        if fb is not None:
            s = 1.0 if fb[0] == 0 else -1.0
            x["fb_lado"], x["fb_minuto_lado"] = s, s * (m - fb[1]) / m   # antes = más peso
        if ft is not None:
            s = 1.0 if ft[0] == 0 else -1.0
            x["ft_lado"], x["ft_minuto_lado"] = s, s * (m - ft[1]) / m
        out[m] = np.array([x[k] for k in P1 + P2], np.float32)
    return out


CACHE_EV = SALIDA / "variables_eventos.npz"


def _procesar_lote(blobs):
    """Worker: lista de JSON comprimidos → lista de dict minuto → vector."""
    return [variables_partida(json.loads(zlib.decompress(b))) for b in blobs]


def construir(hilos: int = max(1, (os.cpu_count() or 2) - 2)):
    """Variables de eventos de todos los timelines bajados. Se cachean en `variables_eventos.npz` y se reusan
    mientras no cambie el conjunto de partidas (leer ~63.000 timelines con un núcleo tarda ~26 min)."""
    SALIDA.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(f"file:{RUTA_DB.as_posix()}?mode=ro", uri=True)) as con:
        ids = con.execute("""SELECT t.match_id, s.grupo FROM timeline_crudo t JOIN seleccion s USING (match_id)
                             ORDER BY t.match_id""").fetchall()
    mids = np.array([r[0] for r in ids])
    grupos = np.array([r[1] for r in ids])
    if CACHE_EV.exists():
        Z = np.load(CACHE_EV, allow_pickle=True)
        if len(Z["mids"]) == len(mids) and (Z["mids"].astype(str) == mids).all() \
                and list(Z["nombres"]) == P1 + P2:
            log(f"variables de eventos desde la caché ({len(mids):,} timelines)")
            return mids, grupos, {m: Z[f"X{m}"] for m in MINUTES}
    vacio = np.full(len(P1 + P2), np.nan, np.float32)
    X = {m: np.empty((len(mids), len(P1 + P2)), np.float32) for m in MINUTES}
    lote = 500
    with closing(sqlite3.connect(f"file:{RUTA_DB.as_posix()}?mode=ro", uri=True)) as con, \
            ProcessPoolExecutor(hilos) as ex:
        posicion = {m: i for i, m in enumerate(mids)}
        pendientes, hechos, ultimo, vistos = [], 0, 0, 0

        def recoger(par):
            nonlocal hechos
            ids_lote, fu = par
            for i, o in zip(ids_lote, fu.result()):
                for m in MINUTES:
                    X[m][i] = o.get(m, vacio)
            hechos += 1
            if hechos % 20 == 0:
                log(f"  {min(hechos * lote, len(mids)):,} / {len(mids):,} timelines leídos")

        while True:
            # tandas cortas por rowid: cada consulta termina y suelta el lock compartido, así el descargador
            # (que escribe en la misma base) puede guardar entre tanda y tanda
            filas = con.execute("SELECT rowid, match_id, json_zlib FROM timeline_crudo WHERE rowid > ? "
                                "ORDER BY rowid LIMIT ?", (ultimo, lote)).fetchall()
            if not filas:
                break
            ultimo = filas[-1][0]
            filas = [(posicion[m], b) for _, m, b in filas if m in posicion]   # las bajadas después, afuera
            vistos += len(filas)
            pendientes.append(([i for i, _ in filas], ex.submit(_procesar_lote, [b for _, b in filas])))
            if len(pendientes) >= 2 * hilos:            # pocos lotes en vuelo: la memoria no crece con la base
                recoger(pendientes.pop(0))
        for par in pendientes:
            recoger(par)
    assert vistos == len(mids), "timeline_crudo y seleccion desalineadas"
    np.savez_compressed(CACHE_EV, mids=mids, nombres=np.array(P1 + P2), **{f"X{m}": X[m] for m in MINUTES})
    return mids, grupos, X


def main():
    # import diferido: ingame_evaluar necesita lightgbm y catboost; así `variables_partida` y `construir` se
    # pueden importar sin ellos (los usa scripts/analysis/modelo_en_vivo_eventos.py, §9.S.78)
    from scripts.analysis.ingame_evaluar import _delta_ci, _fit_logreg_antisym, _prep_arrays, _sigm  # noqa: F401

    mids_e, grupos, XE = construir()
    log(f"timelines: {len(mids_e):,} (train {np.sum(grupos == 'train'):,}, V {np.sum(grupos == 'V'):,})")
    C = np.load(IN_DIR / "ingame_cache.npz", allow_pickle=True)
    y, chunk, t = C["y"].astype(np.int64), C["chunk"].astype(np.int32), C["t"].astype(np.int64)
    master, val = C["master"].astype(bool), C["val"].astype(bool)
    dur, L_oof = C["dur_sec"].astype(np.float32), C["L_oof"].astype(np.float32)
    tl, gold_resid = C["tl_raw"].astype(np.float32), C["gold_resid"].astype(np.float32)
    exp_lane = {k[4:]: C[k].astype(np.float32) for k in C.files if k.startswith("exp_")}
    exp_trans = {k[3:]: C[k].astype(np.float32) for k in C.files if k.startswith("tr_")}
    pos = {m: i for i, m in enumerate(C["mids"].astype(str))}
    fila = np.array([pos[m] for m in mids_e])
    en_muestra = np.zeros(len(y), bool)
    en_muestra[fila[grupos == "train"]] = True
    filas_out = []
    for mi, m in enumerate(MINUTES):
        ok_tl = np.isfinite(tl[:, :, :, : mi + 1, :]).all(axis=(1, 2, 3, 4))
        ok_m = (dur > 60.0 * m) & ok_tl & np.isfinite(L_oof)
        f = extract_minute_features(m, L_oof, tl, gold_resid, exp_lane=exp_lane, exp_trans=exp_trans,
                                    include_all_stats=True, include_traj=True, include_expected=True,
                                    include_comp_cross=False, include_player=False)
        tr_base = np.flatnonzero(np.isin(chunk, [1, 2, 3, 4, 5, 6]) & master & ok_m & ~en_muestra)
        ok_e = np.isfinite(XE[m]).all(1) & ok_m[fila]
        idx_tr = fila[(grupos == "train") & ok_e]
        idx_v = fila[(grupos == "V") & ok_e]
        E = np.full((len(y), XE[m].shape[1]), 0.0, np.float32)
        E[fila[ok_e]] = XE[m][ok_e]
        te = np.r_[idx_tr, idx_v]
        ar = _prep_arrays(f, tr_base, te, L_oof, y)
        z = _fit_logreg_antisym(ar["Xa_tr"], y[tr_base], ar["Xa_te"], C=0.01)[1]
        zb = np.zeros(len(y), np.float32)
        zb[te] = z
        log(f"minuto {m}: base ajustada con {len(tr_base):,}; muestra train {len(idx_tr):,}, V {len(idx_v):,}")
        rms = np.sqrt((E[idx_tr] ** 2).mean(0)) + 1e-6
        nombres = P1 + P2
        variantes = {"base (logística en vivo)": [],
                     "+ prioridad 1 (objetivos, oro invertido, muertos)": list(range(len(P1))),
                     "+ prioridad 1 + 2 (tiempos, momentum, wards)": list(range(len(nombres)))}

        def ajustar(cols, tr, tst):
            X = lambda i: np.c_[zb[i], E[i][:, cols] / rms[cols]] if cols else zb[i, None]  # noqa: E731
            lr = LogisticRegression(C=1.0, max_iter=3000).fit(X(tr), y[tr])
            return lr.predict_proba(X(tst))[:, 1], lr

        P = {}
        coefs = None
        for nom, cols in variantes.items():
            P[nom], lr = ajustar(cols, idx_tr, idx_v)
            if len(cols) == len(nombres):
                coefs = dict(zip(["logit_base"] + nombres, lr.coef_[0]))
        yv = y[idx_v]
        boot = np.random.default_rng(m).integers(0, len(idx_v), size=(500, len(idx_v)))
        ref = "base (logística en vivo)"
        for nom in variantes:
            blo = []
            q = np.quantile(t[idx_tr], [0.2, 0.4, 0.6, 0.8])      # 5 tramos por fecha dentro de la muestra
            tramo = np.searchsorted(q, t[idx_tr], side="right")
            for b in (2, 3, 4):
                trb, teb = idx_tr[tramo < b], idx_tr[tramo == b]
                if len(teb) > 200 and len(trb) > 500:
                    blo.append(roc_auc_score(y[teb], ajustar(variantes[nom], trb, teb)[0])
                               - roc_auc_score(y[teb], ajustar([], trb, teb)[0]))
            d, lo, hi = _delta_ci(yv, P[nom], P[ref], boot)
            r = {"minuto": m, "variante": nom, "n_train": len(idx_tr), "n_V": len(idx_v),
                 "Δ bloques": float(np.mean(blo)) if blo else np.nan, "bloques que mejora": int(sum(x > 0 for x in blo)),
                 "AUC V": roc_auc_score(yv, P[nom]), "Accuracy V": float(((P[nom] > 0.5) == yv).mean()),
                 "Δ V": d, "Δ V lo": lo, "Δ V hi": hi}
            filas_out.append(r)
            print(f"  {nom:52s} bloques {r['Δ bloques']:+.4f} ({r['bloques que mejora']}/{len(blo)}) | "
                  f"V {r['AUC V']:.4f} / {r['Accuracy V']:.2%} | Δ V {d:+.4f} [{lo:+.4f}; {hi:+.4f}]", flush=True)
        print("  coeficientes (escala RMS):", ", ".join(f"{k} {v:+.2f}" for k, v in
                                                     sorted(coefs.items(), key=lambda kv: -abs(kv[1]))[:12]))
        pd.DataFrame(filas_out).to_csv(SALIDA / "eventos_aporte.csv", index=False)
    log("listo")


if __name__ == "__main__":
    main()
