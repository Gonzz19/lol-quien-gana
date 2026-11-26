"""
Niveles 2 y 3 de la huella de estilo (pedido del usuario, 2026-11-18; protocolo en PROJECT_SUMMARY §10.1, mejoras en
§10.4). Todo con jugadores que el ajuste no vio: la mitad de los jugadores aprende los pesos o la red, la otra
mitad se evalúa.

* **Nivel 2: verificación con jugadores no vistos.** Pares (perfil A del jugador i, perfil B del jugador j): ¿es la
  misma persona? Con un **umbral único** para todos los pares (no por fila como el AUC del nivel 1): ROC-AUC, EER
  y TAR@FAR = 1 %, en función de k (partidas por perfil) y de la representación. Campeones distintos entre A y B.
* **Nivel 3: detección de cambio de persona** sobre secuencias armadas. Perfil de la cuenta con sus primeras
  `--historia` partidas, y después `--nuevas` partidas de uno de tres tipos:
  - *control*: las siguientes del mismo jugador (cambios naturales de campeón incluidos);
  - *boosting*: las de otro jugador no visto de la misma línea principal y tier igual o mayor;
  - *compartida*: alternadas entre los dos (empieza el dueño).
  El puntaje en cada paso es la distancia entre la ventana de las últimas w partidas nuevas y el perfil. El
  umbral de cada tamaño de ventana sale de los controles, para una tasa de falsas alarmas por partida de
  `--fa` (1 % = 1 cada 100 partidas). Se reporta la detección dentro de las 20 partidas, la demora mediana y la
  detección por tercil de distancia de estilo entre los dos jugadores.

Representaciones: euclídea sin pesos (nivel 1), LDA y red contrastiva, sobre las 23 estadísticas (train Master+) y
sobre estadísticas + hábitos (partidas con timeline completo). Es descriptivo, como el nivel 1.

    python scripts/analysis/huella_niveles.py      # ~15 min (más la primera vez, que arma las cachés)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.features.fingerprint import STYLE_FEATURES, TIER_ORDER, make_halves  # noqa: E402
from scripts.analysis.fingerprint_level1 import player_info  # noqa: E402
from scripts.analysis import huella_mejoras as hm  # noqa: E402

SALIDA = hm.SALIDA
T0 = time.time()
log = lambda s: print(f"[{time.time() - T0:6.1f}s] {s}", flush=True)  # noqa: E731


# --------------------------------------------------------------------------------------------- datos
def datos_train(residuo="ajuste"):
    """Train Master+ residualizado y la partición (ajuste, evaluación); ver `huella_mejoras.cargar_resid`."""
    return hm.cargar_resid(residuo)


def datos_timeline(residuo="ajuste", crudo=None):
    """Partidas con timeline (estadísticas + hábitos) residualizadas y la partición (ajuste, evaluación).

    La partición se hace sobre los jugadores crudos y, con `residuo="ajuste"`, la media, el desvío y el tamaño
    de cada grupo campeón-línea-resultado salen solo de los jugadores de ajuste (§9.S.77): la red y la LDA se
    evalúan con jugadores que no entraron en la normalización. `residuo="global"` es la versión anterior
    (descriptiva, todos los jugadores); se cachea en `resid_timeline.pkl`, que leen los análisis 2-fold con
    `--residuo global` (`huella_patrones_cuenta.datos_2fold`, que por defecto normaliza por pliegue).
    `crudo` reemplaza a `hm.timeline_train()` (tests).
    """
    if residuo == "global" and crudo is None:
        c = SALIDA / "resid_timeline.pkl"
        if c.exists():
            r = pd.read_pickle(c)
            return (r, *hm.partir_jugadores(hm.timeline_train()["player_id"].unique()))
        r, ajuste, evalu = hm.cargar_timeline("global")
        r.to_pickle(c)
        return r, ajuste, evalu
    return hm.cargar_timeline(residuo, crudo=crudo)


# --------------------------------------------------------------------------------------------- red
def entrenar_red(df, feats, ajuste, pasos=3000, semilla=0):
    import torch
    torch.manual_seed(semilla)
    X = torch.tensor(df[feats].to_numpy(np.float32))
    grupos = {p: torch.tensor(ix) for p, ix in df.reset_index(drop=True).groupby("player_id").indices.items()
              if p in ajuste and len(ix) >= 40}
    lista = list(grupos)
    red = torch.nn.Sequential(torch.nn.Linear(len(feats), 128), torch.nn.GELU(), torch.nn.Linear(128, 128),
                              torch.nn.GELU(), torch.nn.Linear(128, 32))
    opt = torch.optim.AdamW(red.parameters(), lr=2e-3, weight_decay=1e-4)
    rng = np.random.default_rng(semilla)
    for paso in range(pasos):
        k = int(rng.integers(3, 21))
        lote = rng.choice(len(lista), min(256, len(lista)), replace=False)
        a, b = [], []
        for j in lote:
            ix = grupos[lista[j]]
            perm = ix[torch.randperm(len(ix))[: 2 * k]]
            a.append(X[perm[:k]].mean(0))
            b.append(X[perm[k:]].mean(0))
        za = torch.nn.functional.normalize(red(torch.stack(a)), dim=1)
        zb = torch.nn.functional.normalize(red(torch.stack(b)), dim=1)
        y = torch.arange(len(lote))
        lg = za @ zb.T / 0.1
        loss = (torch.nn.functional.cross_entropy(lg, y) + torch.nn.functional.cross_entropy(lg.T, y)) / 2
        opt.zero_grad()
        loss.backward()
        opt.step()
    red.eval()

    def emb(M):
        with torch.no_grad():
            return torch.nn.functional.normalize(red(torch.tensor(M, dtype=torch.float32)), dim=1).numpy()
    return emb


def representaciones(df, feats, ajuste, con_red=True):
    """dict nombre -> función(perfiles medios) -> vectores donde se mide distancia euclídea."""
    Ts = hm.aprender(df, feats, ajuste)
    reps = {"euclídea": lambda M: M, "LDA": lambda M, T=Ts["lda"]: M @ T}
    if con_red:
        log("  entrenando la red contrastiva…")
        reps["red contrastiva"] = entrenar_red(df, feats, ajuste)
    return reps


# --------------------------------------------------------------------------------------------- nivel 2
def nivel2(df, feats, nombre_datos, reps, evalu, info, ks=(5, 10, 20, 40)):
    ev = df[df["player_id"].isin(evalu)]
    filas = []
    for k in ks:
        halves = make_halves(ev, k=k, mode="campeones_distintos", seed=0)
        if len(halves) < 200:
            continue
        pids, A, B = hm.perfiles(ev, halves, feats)
        t = info.reindex(pids)
        mismo = (t["player_tier"].to_numpy()[:, None] == t["player_tier"].to_numpy()[None, :]) & \
                (t["main_lane"].to_numpy()[:, None] == t["main_lane"].to_numpy()[None, :])
        for nom, f in reps.items():
            D = cdist(f(A), f(B))
            n = len(D)
            gen = np.diag(D)
            off = ~np.eye(n, dtype=bool)
            for tipo, imp in (("todos", D[off]), ("mismo tier y línea", D[off & mismo])):
                y = np.r_[np.ones(n), np.zeros(len(imp))]
                s = -np.r_[gen, imp]                                  # más cerca = más probable misma persona
                fpr, tpr, _ = roc_curve(y, s)
                eer = float(fpr[np.argmin(np.abs(fpr - (1 - tpr)))])
                tar1 = float(np.interp(0.01, fpr, tpr))
                fila = {"nivel": 2, "datos": nombre_datos, "k": k, "representacion": nom, "impostores": tipo,
                        "jugadores": n, "roc_auc": roc_auc_score(y, s), "eer": eer, "tar_far1": tar1}
                filas.append(fila)
                print(f"  N2 {nombre_datos:<24} k={k:<3} {nom:<16} {tipo:<19} n={n:>5} ROC-AUC {fila['roc_auc']:.3f} "
                      f"EER {eer:.1%} TAR@FAR1% {tar1:.1%}", flush=True)
    return filas


# --------------------------------------------------------------------------------------------- nivel 3
def armar_secuencias(df, feats, evalu, info, historia, nuevas, semilla=0):
    """Por cuenta: perfil (media de las primeras `historia`) y las `nuevas` siguientes de cada escenario."""
    rng = np.random.default_rng(semilla)
    ev = df[df["player_id"].isin(evalu)].sort_values(["player_id", "game_creation"])
    X = ev[feats].to_numpy(float)
    idx = {p: ix for p, ix in ev.groupby("player_id").indices.items() if len(ix) >= historia + nuevas}
    pids = list(idx)
    rank = {t: i for i, t in enumerate(TIER_ORDER)}
    t = info.reindex(pids)
    lane = t["main_lane"].to_numpy()
    tier = t["player_tier"].map(rank).fillna(0).to_numpy()
    cuentas = []
    for i, a in enumerate(pids):
        cand = np.flatnonzero((lane == lane[i]) & (tier >= tier[i]) & (np.arange(len(pids)) != i))
        if len(cand) == 0:
            continue
        b = pids[rng.choice(cand)]
        ia, ib = idx[a], idx[b]
        perfil = X[ia[:historia]].mean(0)
        propio = X[ia[historia:historia + nuevas]]
        inicio = rng.integers(0, len(ib) - nuevas + 1)
        otro = X[ib[inicio:inicio + nuevas]]
        alternado = np.where((np.arange(nuevas) % 2 == 0)[:, None], propio, otro)
        cuentas.append({"perfil": perfil, "control": propio, "boosting": otro, "compartida": alternado,
                        "perfil_otro": X[ib].mean(0)})
    return cuentas


def nivel3(cuentas, nombre_datos, reps, ventanas=(5, 10), fa=0.01):
    filas = []
    nuevas = len(cuentas[0]["control"])
    for nom, f in reps.items():
        P = f(np.stack([c["perfil"] for c in cuentas]))
        Po = f(np.stack([c["perfil_otro"] for c in cuentas]))
        dist_estilo = np.linalg.norm(P - Po, axis=1)
        terc = np.quantile(dist_estilo, [1 / 3, 2 / 3])
        for w in ventanas:
            # puntaje en cada paso t (1..nuevas): distancia de la media de las últimas min(t, w) partidas al perfil
            punt = {}
            for esc in ("control", "boosting", "compartida"):
                S = np.zeros((len(cuentas), nuevas))
                for t in range(1, nuevas + 1):
                    m = min(t, w)
                    V = f(np.stack([c[esc][t - m:t].mean(0) for c in cuentas]))
                    S[:, t - 1] = np.linalg.norm(V - P, axis=1)
                punt[esc] = S
            # umbral por tamaño de ventana: cuantil (1 − fa) del puntaje de los controles en ese paso
            tau = np.quantile(punt["control"], 1 - fa, axis=0)
            alarmas_ctrl = (punt["control"] > tau).mean()
            for esc in ("boosting", "compartida"):
                det = punt[esc] > tau
                hubo = det.any(1)
                demora = np.where(hubo, det.argmax(1) + 1, np.nan)
                fila = {"nivel": 3, "datos": nombre_datos, "representacion": nom, "ventana": w, "escenario": esc,
                        "cuentas": len(cuentas), "fa_por_100": 100 * alarmas_ctrl,
                        "detecta_en_20": float(hubo.mean()), "demora_mediana": float(np.nanmedian(demora)),
                        "detecta_en_5": float((demora <= 5).mean()), "detecta_en_10": float((demora <= 10).mean())}
                for j, (lo, hi) in enumerate(((-np.inf, terc[0]), (terc[0], terc[1]), (terc[1], np.inf))):
                    m = (dist_estilo > lo) & (dist_estilo <= hi)
                    fila[f"detecta_tercil{j + 1}"] = float(hubo[m].mean())
                filas.append(fila)
                print(f"  N3 {nombre_datos:<24} {nom:<16} w={w:<2} {esc:<10} n={len(cuentas):>4} "
                      f"falsas alarmas {fila['fa_por_100']:.1f}/100 | detecta: 5 part {fila['detecta_en_5']:.0%}, "
                      f"10 {fila['detecta_en_10']:.0%}, 20 {fila['detecta_en_20']:.0%} (demora mediana "
                      f"{fila['demora_mediana']:.0f}) | por distancia de estilo {fila['detecta_tercil1']:.0%} / "
                      f"{fila['detecta_tercil2']:.0%} / {fila['detecta_tercil3']:.0%}", flush=True)
    return filas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--historia", type=int, default=40)
    ap.add_argument("--nuevas", type=int, default=20)
    ap.add_argument("--fa", type=float, default=0.01)
    ap.add_argument("--residuo", choices=["ajuste", "global"], default="ajuste",
                    help="normalizar solo con los jugadores de ajuste (default, §9.S.77) o con todos (versión "
                         "anterior; las salidas llevan el sufijo _residuo_global)")
    args = ap.parse_args()
    SALIDA.mkdir(parents=True, exist_ok=True)
    filas2, filas3 = [], []
    for nombre, cargar, feats in (("23 estadísticas (train)", datos_train, STYLE_FEATURES),
                                  ("estadísticas + hábitos", datos_timeline, STYLE_FEATURES + hm.HABITOS)):
        df, ajuste, evalu = cargar(args.residuo)
        log(f"{nombre}: {len(df):,} filas, {df['player_id'].nunique():,} jugadores")
        info = player_info(df)
        reps = representaciones(df, feats, ajuste)
        filas2 += nivel2(df, feats, nombre, reps, evalu, info)
        cuentas = armar_secuencias(df, feats, evalu, info, args.historia, args.nuevas)
        log(f"  nivel 3: {len(cuentas):,} cuentas con {args.historia} + {args.nuevas} partidas")
        filas3 += nivel3(cuentas, nombre, reps, fa=args.fa)
    sufijo = "_residuo_global" if args.residuo == "global" else ""
    pd.DataFrame(filas2).to_csv(SALIDA / f"nivel2{sufijo}.csv", index=False)
    pd.DataFrame(filas3).to_csv(SALIDA / f"nivel3{sufijo}.csv", index=False)
    log("listo")


if __name__ == "__main__":
    main()
