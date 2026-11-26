"""Candidato (no promovido): eventos del timeline encima del modelo en vivo en dos etapas (§9.S.64, §9.S.66, §9.S.78).

Los eventos de prioridad 1 de `ingame_eventos.py` (dragones y punto de alma, larvas, heraldo, torres por nivel,
placas, inhibidores, oro sin gastar e invertido, muertos en el minuto, shutdowns; azul − rojo, solo lo ocurrido
antes del minuto) entran en la **etapa 2**:

    logit P = a + b z + c L + γ·e        (contra la misma etapa 2 sin e, ajustada con las mismas partidas)

Solo hay timelines con eventos para V Master+ entera y ~la mitad de Master+ del ajuste, así que las dos variantes
se ajustan con las partidas de A2-A4 que tienen timeline (hacia adelante: A3 ← A2, A4 ← A2-A3, V ← A2-A4). Regla
de aceptación: media de los ΔAUC de los bloques A3-A4 con IC > 0, V una sola vez y réplica en Diamond-V (que hoy
no tiene timelines: se informa que falta). Los eventos se prueban sobre la estructura de dos etapas (la etapa 2 es
donde entran); "contra el presentado" compara con el modelo principal, que desde la decisión de §9.S.78 es la
etapa 1 sola.

    python scripts/analysis/modelo_en_vivo_eventos.py --db RUTA/timelines.db

Necesita la caché del modelo en vivo (`scripts/analysis/modelo_en_vivo.py`). Las variables de eventos se guardan
con `mids` en `data/analysis/en_vivo/eventos.npz` y se reusan mientras no cambie el conjunto de timelines.
"""
import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis import ingame_eventos as IE  # noqa: E402
from src.ganador import modelo_en_vivo as MV, rutas, variables  # noqa: E402

B = 1000


def _delta_bloques(y, p1, p0, rel, B=500):
    from sklearn.metrics import roc_auc_score
    ds, reps = [], []
    for k, sl in enumerate(rel):
        yy, a, b0 = y[sl], p1[sl], p0[sl]
        ds.append(roc_auc_score(yy, a) - roc_auc_score(yy, b0))
        bb = np.random.default_rng(70 + k).integers(0, len(yy), size=(B, len(yy)))
        reps.append([roc_auc_score(yy[s], a[s]) - roc_auc_score(yy[s], b0[s]) for s in bb])
    r = np.mean(reps, axis=0)
    return ds, float(np.mean(ds)), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", type=Path, default=IE.RUTA_DB, help="timelines.db (solo lectura)")
    ap.add_argument("--hilos", type=int, default=4)
    a = ap.parse_args()
    IE.RUTA_DB = a.db
    IE.SALIDA = rutas.CACHE_EN_VIVO
    IE.CACHE_EV = rutas.CACHE_EN_VIVO / "eventos.npz"
    mids_e, grupos, XE = IE.construir(a.hilos)
    MV.log(f"timelines con eventos: {len(mids_e):,}")

    D, V = variables.calcular_todas()
    I = MV.insumos(D, V)
    del D, V
    gc.collect()
    y, chunk, mids = I["y"].astype(int), I["chunk"], I["mids"]
    pos = {m: i for i, m in enumerate(mids)}
    dentro = np.array([m in pos for m in mids_e])
    fila = np.array([pos[m] for m in mids_e[dentro]])
    n1 = len(IE.P1)
    filas = []
    for m in MV.MINUTOS:
        E = np.full((len(y), n1), np.nan, np.float32)
        E[fila] = XE[m][dentro][:, :n1]
        tiene = np.isfinite(E).all(1)
        ok = MV.validas(I, m)
        v = np.flatnonzero((chunk == MV.TV) & ok)
        R = MV.ajustar_predecir(I, m, v)
        sel = tiene[R["a24"]]                       # partidas de A2-A4 con timeline, en orden
        a24 = R["a24"][sel]
        corte = np.cumsum([0, *[int(tiene[b].sum()) for b in R["bloques"]]])
        vs = tiene[v]
        rms = np.sqrt((E[a24] ** 2).mean(0)) + 1e-6
        Ea, Ev = E[a24] / rms, E[v[vs]] / rms
        z1a, Lf = R["z1_a24"][sel], I["L"][v[vs]]
        base = MV.etapa2(z1a, I["L"][a24], y[a24], corte, R["z1_fin"][vs], Lf)
        ev = MV.etapa2(z1a, I["L"][a24], y[a24], corte, R["z1_fin"][vs], Lf, Ea, Ev)
        rng = np.random.default_rng(5 + m)
        Epl, Evpl = Ea.copy(), Ev[rng.permutation(len(Ev))]
        for k in range(3):
            Epl[corte[k]:corte[k + 1]] = Epl[corte[k]:corte[k + 1]][rng.permutation(corte[k + 1] - corte[k])]
        pl = MV.etapa2(z1a, I["L"][a24], y[a24], corte, R["z1_fin"][vs], Lf, Epl, Evpl)
        rel = [slice(corte[1], corte[2]), slice(corte[2], corte[3])]
        db, dbm, dblo, dbhi = _delta_bloques(y[a24], MV._sigm(ev[0]), MV._sigm(base[0]), rel)
        yv = y[v[vs]]
        p0, p1, ppl = MV._sigm(base[1]), MV._sigm(ev[1]), MV._sigm(pl[1])
        dV = MV.delta_auc_ic(yv, p1, p0, B, 300 + m)
        dPl = MV.delta_auc_ic(yv, ppl, p0, B, 900 + m)
        # contra el modelo presentado (la etapa 1 sola desde la decisión de §9.S.78), en las mismas partidas de V
        p_pres = MV._sigm(R["z_fin"][vs])
        dP = MV.delta_auc_ic(yv, p1, p_pres, B, 500 + m)
        m0, m1 = MV.metricas_ic(yv, p0, B, 10 + m), MV.metricas_ic(yv, p1, B, 10 + m)
        print(f"\n=== Minuto {m}: A2/A3/A4 con timeline {np.diff(corte).tolist()}, V {vs.sum():,} de {len(v):,}")
        print(f"  bloques A3, A4: {db[0]:+.4f}, {db[1]:+.4f}; media {dbm:+.4f} [{dblo:+.4f}; {dbhi:+.4f}]")
        print(f"  V: Δ {dV[0]:+.4f} [{dV[1]:+.4f}; {dV[2]:+.4f}] | placebo {dPl[0]:+.4f} [{dPl[1]:+.4f}; {dPl[2]:+.4f}] | "
              f"contra el presentado {dP[0]:+.4f} [{dP[1]:+.4f}; {dP[2]:+.4f}] | Diamond-V: sin timelines")
        for nom, mt in (("sin eventos", m0), ("con eventos", m1)):
            print(f"  {nom}: AUC {mt['auc'][0]:.4f} | acierto {mt['acierto'][0]:.4f} [{mt['acierto'][1]:.4f}; "
                  f"{mt['acierto'][2]:.4f}] | log loss {mt['log_loss'][0]:.4f} | ECE {mt['ece'][0]:.4f}")
        print("  γ (escala RMS): " + ", ".join(f"{k} {g:+.3f}" for k, g in zip(IE.P1, ev[2][3:])))
        filas.append({"minuto": m, "n_a2": corte[1], "n_a3": corte[2] - corte[1], "n_a4": corte[3] - corte[2],
                      "n_v": int(vs.sum()), "d_A3": db[0], "d_A4": db[1], "d_bloques": dbm, "d_bloques_lo": dblo,
                      "d_bloques_hi": dbhi, "d_v": dV[0], "d_v_lo": dV[1], "d_v_hi": dV[2], "d_placebo_v": dPl[0],
                      "d_v_contra_presentado": dP[0], "auc_v_sin": m0["auc"][0], "auc_v_con": m1["auc"][0],
                      "acierto_v_sin": m0["acierto"][0], "acierto_v_con": m1["acierto"][0],
                      "log_loss_v_sin": m0["log_loss"][0], "log_loss_v_con": m1["log_loss"][0],
                      "ece_v_sin": m0["ece"][0], "ece_v_con": m1["ece"][0]})
    pd.DataFrame(filas).to_csv(rutas.CACHE_EN_VIVO / "eventos_dos_etapas.csv", index=False)
    print(f"\n[+] {rutas.CACHE_EN_VIVO / 'eventos_dos_etapas.csv'}")


if __name__ == "__main__":
    main()
