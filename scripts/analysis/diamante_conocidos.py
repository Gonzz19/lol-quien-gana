"""
¿Cómo rinde el modelo previo en lobbies de Diamond cuyos jugadores conocemos bien? (pedido del usuario, 2026-11-17)

En Diamond-V (las Diamond de la ventana de V, a ciegas) el escalón 9 da 0,6437 / 60,3 % contra 0,6739 / 62,5 % en
V Master+. Hipótesis: rinde menos porque sabemos menos de esos jugadores (padrón casi todo Master+). Se separa
"cuánto sabemos" de "qué tier es" agrupando las partidas por el historial previo de los 10 jugadores
(`logn_tot` de `variables.npz` = log(1 + partidas previas), causal) y comparando Diamond-V con V Master+ en los
mismos grupos.

* Modelo: `--modelo hibrido` (híbrido + mapa, corre en cualquier máquina) o `--modelo escalon9` (necesita las
  cachés de `extras_referencia(D, 9)`; sin mapa, como `brainstorm_final.py`). Entrenado solo con Master+
  anterior al 16/09 (como el escalón 1) y, aparte, con Master+ y Diamond anteriores al 16/09.
* Grupos por partida: mínimo de partidas previas entre los 10 jugadores (≥ 20, ≥ 50, ≥ 100, ≥ 200) y cantidad de
  jugadores con ≥ 50 partidas previas.

    python scripts/analysis/diamante_conocidos.py --modelo hibrido     # ~3 min
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.brainstorm_base import ajustar_predecir, cargar, extras_referencia  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", choices=["hibrido", "escalon9"], default="hibrido")
    args = ap.parse_args()
    D = cargar()
    y, NM = D["y"], D["NM"]
    n_prev = np.expm1(D["Xj"][..., D["cols"].index("logn_tot")]).reshape(NM, 10)
    minimo = n_prev.min(1)
    con50 = (n_prev >= 50).sum(1)
    fi_v = D["fi_v"]
    te = np.flatnonzero(D["val"] | D["diam_v"])
    entrenos = {"solo Master+": fi_v[D["master"][fi_v]], "Master+ y Diamond": fi_v}
    P = {}
    for nom, tr in entrenos.items():
        if args.modelo == "escalon9":
            p = ajustar_predecir(D, tr, te, extra=extras_referencia(D, escalon=9), extra_post=False, con_mapa=False)
        else:
            p = ajustar_predecir(D, tr, te)
        P[nom] = np.full(NM, np.nan)
        P[nom][te] = p
        print(f"listo: entrenado con {nom} ({len(tr):,} partidas)", flush=True)
    filas = []
    # LP medio del lobby (mapa de LP causal, escala absoluta: Diamond ~0-400, Master+ 400+) y mediana de historial
    lp = np.nanmean(D["B"]["lp_abs"].reshape(NM, 10), 1)
    mediana = np.median(n_prev, 1)
    cortes_lp = [0, 200, 300, 400, 500, 600, 800, 1000, 99999]
    for a, b in zip(cortes_lp[:-1], cortes_lp[1:]):
        idx = np.flatnonzero((D["val"] | D["diam_v"]) & (lp >= a) & (lp < b))
        if len(idx) < 300:
            continue
        fila = {"conjunto": "LP medio del lobby", "grupo": f"{a}-{b if b < 99999 else '+'}", "partidas": len(idx),
                "% Master+": float(D["val"][idx].mean()), "previas (mediana del lobby)": np.median(mediana[idx])}
        for nom in entrenos:
            p = P[nom][idx]
            fila[f"AUC ({nom})"] = roc_auc_score(y[idx], p)
            fila[f"Accuracy ({nom})"] = float(((p > 0.5) == y[idx]).mean())
        filas.append(fila)
    for cn, cm in (("V Master+", D["val"]), ("Diamond-V", D["diam_v"])):
        for a, b in ((0, 10), (10, 30), (30, 60), (60, 120), (120, 99999)):
            idx = np.flatnonzero(cm & (mediana >= a) & (mediana < b))
            if len(idx) < 300:
                continue
            fila = {"conjunto": f"{cn}, mediana de previas del lobby", "grupo": f"{a}-{b if b < 99999 else '+'}",
                    "partidas": len(idx), "previas (mediana del lobby)": np.median(mediana[idx])}
            for nom in entrenos:
                p = P[nom][idx]
                fila[f"AUC ({nom})"] = roc_auc_score(y[idx], p)
                fila[f"Accuracy ({nom})"] = float(((p > 0.5) == y[idx]).mean())
            filas.append(fila)
    conjuntos = {"V Master+": D["val"], "Diamond-V": D["diam_v"]}
    grupos = {"todas": np.ones(NM, bool)}
    for k in (20, 50, 100, 200):
        grupos[f"los 10 con ≥ {k} previas"] = minimo >= k
    for k in (0, 5, 8):
        grupos[f"{k}+ jugadores con ≥ 50 previas" if k else "ninguno con ≥ 50 previas"] = \
            (con50 >= k) if k else (con50 == 0)
    for cn, cm in conjuntos.items():
        for gn, gm in grupos.items():
            idx = np.flatnonzero(cm & gm)
            if len(idx) < 300:
                continue
            fila = {"conjunto": cn, "grupo": gn, "partidas": len(idx), "previas (mediana del mínimo)": np.median(minimo[idx])}
            for nom in entrenos:
                p = P[nom][idx]
                fila[f"AUC ({nom})"] = roc_auc_score(y[idx], p)
                fila[f"Accuracy ({nom})"] = float(((p > 0.5) == y[idx]).mean())
            filas.append(fila)
    T = pd.DataFrame(filas)
    pd.set_option("display.width", 220)
    print(T.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    T.to_csv(REPO / "data" / "analysis" / f"diamante_conocidos_{args.modelo}.csv", index=False)


if __name__ == "__main__":
    main()
