"""Modelo en vivo de quién gana (10/15/20 min, logística con el previo del escalón 9) de punta a punta, en un solo comando.

Calcula (o carga de `data/cache/ganador/`, si siguen vigentes) las variables previas del escalón 9, el previo
hacia adelante, el estado del timeline y las expectativas por línea; ajusta el modelo y reporta en V Master+
(y Diamond-V como réplica) AUC, acierto contra "siempre rojo", log loss y ECE con IC 95 % por bootstrap, en cada
minuto y sobre **la misma cohorte** en los tres (partidas de V vivas a los 20). No toca el test.
Sin `torch` en la cuenta (se importa con `src.models.ingame`), sin GPU y sin `lightgbm`.

El modelo principal es **la etapa 1** (logística sobre el estado del minuto y el logit previo L). La etapa 2
(logit P = a + b z + c L) se calcula y se reporta como **variante descartada**: no pasa la regla de aceptación
de §9.S.47 (decisión en §9.S.78).

    python scripts/analysis/modelo_en_vivo.py               # usa la caché si está vigente
    python scripts/analysis/modelo_en_vivo.py --rehacer     # recalcula estado y previo del modelo en vivo

Escribe `data/cache/ganador/pred_en_vivo.npz` (`p_{m}` = modelo principal, `p_dos_etapas_{m}` = variante
descartada, con `mids`) y `data/analysis/en_vivo/metricas.csv` (columna `rol`). Detalle en PROJECT_SUMMARY.md §9.S.78.
"""
import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.ganador import modelo_en_vivo as MV, rutas, variables  # noqa: E402

B = 1000
ROL = {"etapa1": "principal", "dos_etapas": "descartada", "previo": "referencia"}
assert MV.MODELO_PRINCIPAL == "etapa1"


def _fmt(v):
    return f"{v[0]:.4f} [{v[1]:.4f}; {v[2]:.4f}]"


def _delta_bloques(y, pa1, pa0, bloques_rel, B=500):
    """Media de los ΔAUC de los bloques A3 y A4 (A2 es neutra por construcción), con IC por bootstrap dentro de cada uno."""
    from sklearn.metrics import roc_auc_score
    ds, reps = [], []
    for k, sl in enumerate(bloques_rel[1:]):
        yy, p1, p0 = y[sl], pa1[sl], pa0[sl]
        ds.append(roc_auc_score(yy, p1) - roc_auc_score(yy, p0))
        bb = np.random.default_rng(7 + k).integers(0, len(yy), size=(B, len(yy)))
        reps.append([roc_auc_score(yy[b], p1[b]) - roc_auc_score(yy[b], p0[b]) for b in bb])
    r = np.mean(reps, axis=0)
    return float(np.mean(ds)), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rehacer", action="store_true", help="recalcular el estado y el previo del modelo en vivo")
    a = ap.parse_args()
    t0 = time.time()

    D, V = variables.calcular_todas()
    I = MV.insumos(D, V, rehacer=a.rehacer)
    del D, V
    gc.collect()
    MV.log(f"insumos listos ({time.time() - t0:.0f} s)")

    y, chunk, mids = I["y"].astype(int), I["chunk"], I["mids"]
    vivas20 = MV.validas(I, 20)
    cohorte = np.flatnonzero((chunk == MV.TV) & vivas20)
    filas, salida, guardar = [], {}, {}
    for m in MV.MINUTOS:
        ok = MV.validas(I, m)
        v, dv = np.flatnonzero((chunk == MV.TV) & ok), np.flatnonzero((chunk == MV.TDV) & ok)
        R = MV.ajustar_predecir(I, m, np.r_[v, dv])
        p1, p2 = MV._sigm(R["z1_fin"]), MV._sigm(R["z2_fin"])
        # placebo: L permutada dentro de cada conjunto en la etapa 2
        rng = np.random.default_rng(41 + m)
        La = I["L"][R["a24"]].copy()
        corte = np.cumsum([0, *map(len, R["bloques"])])
        for k in range(3):
            La[corte[k]:corte[k + 1]] = rng.permutation(La[corte[k]:corte[k + 1]])
        Lf = I["L"][np.r_[v, dv]].copy()
        Lf[:len(v)], Lf[len(v):] = rng.permutation(Lf[:len(v)]), rng.permutation(Lf[len(v):])
        _, zpl, _ = MV.etapa2(R["z1_a24"], La, y[R["a24"]], corte, R["z1_fin"], Lf)
        ppl = MV._sigm(zpl)
        nv = len(v)
        ya = y[R["a24"]]
        rel = [slice(corte[k], corte[k + 1]) for k in range(3)]
        db = _delta_bloques(ya, MV._sigm(R["z2_a24"]), MV._sigm(R["z1_a24"]), rel)
        dV = MV.delta_auc_ic(y[v], p2[:nv], p1[:nv], B, 300 + m)
        dDV = MV.delta_auc_ic(y[dv], p2[nv:], p1[nv:], B, 600 + m)
        dPl = MV.delta_auc_ic(y[v], ppl[:nv], p1[:nv], B, 900 + m)
        print(f"\n=== Minuto {m}: C etapa 1 = {R['C1']}, etapa 2 (a, b, c) = "
              f"{tuple(round(x, 3) for x in R['etapa2'])}; entrenamiento {R['n_entrenamiento']:,}")
        print(f"  Δ dos etapas (descartada) − etapa 1 (principal): bloques A3-A4 {db[0]:+.4f} [{db[1]:+.4f}; {db[2]:+.4f}] | "
              f"V {dV[0]:+.4f} [{dV[1]:+.4f}; {dV[2]:+.4f}] | Diamond-V {dDV[0]:+.4f} [{dDV[1]:+.4f}; {dDV[2]:+.4f}] | "
              f"placebo V {dPl[0]:+.4f}")
        en_coh = np.isin(v, cohorte)
        for conj, idx, ps in (("V", v, {"etapa1": p1[:nv], "dos_etapas": p2[:nv]}),
                              ("V cohorte 20", v[en_coh], {"etapa1": p1[:nv][en_coh], "dos_etapas": p2[:nv][en_coh],
                                                          "previo": MV._sigm(I["L"][v[en_coh]])}),
                              ("Diamond-V", dv, {"etapa1": p1[nv:], "dos_etapas": p2[nv:]})):
            for modelo, p in ps.items():
                mt = MV.metricas_ic(y[idx], p, B, 10 + m)
                print(f"  {conj:13s} {modelo:10s} ({ROL[modelo]:10s}) n={mt['n']:,} | AUC {_fmt(mt['auc'])} | "
                      f"acierto {_fmt(mt['acierto'])} (siempre rojo {mt['siempre_rojo'][0]:.4f}) | "
                      f"log loss {_fmt(mt['log_loss'])} | ECE {_fmt(mt['ece'])}")
                filas.append({"minuto": m, "conjunto": conj, "modelo": modelo, "rol": ROL[modelo], "n": mt["n"],
                              **{f"{k}{s}": mt[k][i] for k in ("auc", "acierto", "log_loss", "ece", "siempre_rojo")
                                 for i, s in enumerate(("", "_lo", "_hi"))}})
        filas.append({"minuto": m, "conjunto": "Δ AUC dos etapas − etapa 1", "modelo": "dos_etapas", "rol": "descartada",
                      **{f"d_{n}{s}": d[i] for n, d in (("bloques", db), ("v", dV), ("diamond_v", dDV), ("placebo_v", dPl))
                         for i, s in enumerate(("", "_lo", "_hi"))},
                      "c_etapa1": R["C1"], "etapa2_a": R["etapa2"][0], "etapa2_b": R["etapa2"][1], "etapa2_c": R["etapa2"][2]})
        # p_{m}: modelo principal (etapa 1); p_dos_etapas_{m}: variante descartada (§9.S.78, "Decisión")
        guardar.update({f"mids_{m}": mids[np.r_[v, dv]], f"n_v_{m}": nv, f"p_{m}": MV._sigm(R["z_fin"]),
                        f"p_dos_etapas_{m}": p2, f"mids_a24_{m}": mids[R["a24"]], f"p_a24_{m}": MV._sigm(R["z_a24"]),
                        f"p_a24_dos_etapas_{m}": MV._sigm(R["z2_a24"])})
        gc.collect()
    guardar["modelo_principal"] = np.array(MV.MODELO_PRINCIPAL)
    rutas.CACHE_EN_VIVO.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).to_csv(rutas.CACHE_EN_VIVO / "metricas.csv", index=False)
    np.savez_compressed(rutas.CACHE / "pred_en_vivo.npz", **guardar)
    print(f"\n[+] {rutas.CACHE_EN_VIVO / 'metricas.csv'}\n[+] {rutas.CACHE / 'pred_en_vivo.npz'}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
