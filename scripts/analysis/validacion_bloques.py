"""
Validación temporal por bloques de las decisiones del modelo de quién gana (PROJECT_SUMMARY.md
§9.S.47). Usa `src/models/validacion_temporal.py`.

El conjunto de ajuste (las 160.883 partidas del train anteriores a V) se parte en 5 bloques por
fecha. Los bloques 1-4 se predicen hacia adelante, cada uno con lo entrenado en los anteriores.
Cada variante se decide con los 4 bloques, con la regla: mejora en ≥ 3 de 4 y el IC de la media
> 0. V se informa aparte, como confirmación. Modelo: el híbrido del EDA (§9.S.42: logística
C = 0,05 sobre el draft campeón × línea con signo y las 85 variables por jugador, azul − rojo por
línea), con la estandarización ajustada solo con las filas de entrenamiento de cada bloque.

Variantes que se pueden reproducir en esta máquina con la caché de `embeddings_partidas.py`:
sacar el draft, sacar cada familia, sin las últimas 20, la forma del día (§9.S.45; necesita
`ganador_forma_dia.py --solo-cache`).

    python scripts/analysis/validacion_bloques.py              # ~25 min, 6 hilos
    python scripts/analysis/validacion_bloques.py --hilos 3
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.features.draft import draft_matrix_by_lane  # noqa: E402
from src.models.validacion_temporal import (bloques_por_fecha, comparar_por_bloques, decidir,  # noqa: E402
                                            predecir_por_bloques)

ap = argparse.ArgumentParser()
ap.add_argument("--bloques", type=int, default=5)
ap.add_argument("--hilos", type=int, default=6)
ap.add_argument("--B", type=int, default=200, help="réplicas bootstrap")
args = ap.parse_args()
T0 = time.time()
SAL = REPO / "data" / "analysis" / "validacion_bloques"
SAL.mkdir(parents=True, exist_ok=True)

# ---- datos
Z = np.load(REPO / "data" / "analysis" / "embeddings_partidas" / "variables.npz", allow_pickle=True)
Xj, cols, y, val, champ, t = Z["Xj"], list(Z["cols"]), Z["y"], Z["val"], Z["champ"], Z["t"]
NM, fit = len(y), ~Z["val"]
Fh = np.load(REPO / "data" / "analysis" / "embeddings_partidas" / "forma_dia_corte6.npz", allow_pickle=True)
assert (Fh["mids"] == Z["mids"]).all()
Hj, hc = Fh["Hj"], list(Fh["hc"])

JC = {"logn_champ", "estreno", "wr_champ", "share_champ"}
FAMILIAS = {
    "Margen acumulado": lambda c: c.startswith("acum_"),
    "Margen últimas 20": lambda c: c.startswith("v20_"),
    "Winrate previo": lambda c: c in ("wr_acum", "wr_v20"),
    "Partidas, línea y rol": lambda c: c in ("logn_tot", "share_lane", "offrole"),
    "Pool": lambda c: c in ("conc_rec", "conc_ant", "d_pool", "otp_nuevo"),
    "Ritmo y descanso": lambda c: c in ("ritmo", "log_horas_desde"),
    "Experiencia con el campeón": lambda c: c in JC,
    "Campeón (Data Dragon)": lambda c: c.startswith(("tag_", "info_", "st_")),
    "Perfil típico del campeón": lambda c: c.startswith("perf_"),
}
fam = np.tile(np.array([next(f for f, g in FAMILIAS.items() if g(c)) for c in cols]), 5)
es_v20 = np.tile(np.array([c.startswith("v20_") or c == "wr_v20" for c in cols]), 5)

# todo azul − rojo por línea, sin estandarizar (se estandariza dentro de cada ajuste)
Xd = (Xj[:, 0] - Xj[:, 1]).reshape(NM, -1).astype(np.float32)


def hoy(nombres):
    j = [hc.index(c) for c in nombres]
    return (Hj[:, 0][:, :, j] - Hj[:, 1][:, :, j]).reshape(NM, -1).astype(np.float32)


HOY_M = hoy([c for c in hc if c.startswith("hoy_") and c != "hoy_victorias"] + ["n_dia"])
HOY_W = hoy(["hoy_victorias", "n_dia"])

DRAFT = draft_matrix_by_lane(champ)
print(f"datos listos ({time.time() - T0:.0f}s): {NM:,} partidas, draft {DRAFT.shape[1]:,} columnas")

# ---- variantes: (usar draft, máscara de columnas de Xd, extras densos)
TODO = np.ones(Xd.shape[1], bool)
VARIANTES = {"referencia (híbrido)": (True, TODO, [])}
VARIANTES["sin el draft (campeón × línea)"] = (False, TODO, [])
for f in FAMILIAS:
    VARIANTES[f"sin {f}"] = (True, fam != f, [])
VARIANTES["sin últimas 20 (margen y WR)"] = (True, ~es_v20, [])
VARIANTES["+ margen de hoy"] = (True, TODO, [HOY_M])
VARIANTES["sin últimas 20 + margen de hoy"] = (True, ~es_v20, [HOY_M])
VARIANTES["+ WR de hoy"] = (True, TODO, [HOY_W])


def armar(var, filas, media=None, desvio=None):
    con_draft, mk, extras = VARIANTES[var]
    D = np.hstack([Xd[filas][:, mk]] + [e[filas] for e in extras])
    if media is None:
        media, desvio = D.mean(0), D.std(0) + 1e-6
    D = (D - media) / desvio
    partes = ([DRAFT[filas]] if con_draft else []) + [sparse.csr_matrix(D)]
    return sparse.hstack(partes).tocsr(), media, desvio


def ajustador(var):
    def ajustar(idx_tr, y_tr):
        Xtr, mu, sd = armar(var, idx_tr[:, 0])
        lr = LogisticRegression(C=0.05, max_iter=3000).fit(Xtr, y_tr)
        return lambda idx_te: lr.predict_proba(armar(var, idx_te[:, 0], mu, sd)[0])[:, 1]
    return ajustar


fi = np.flatnonzero(fit)
bloque = bloques_por_fecha(t[fi], args.bloques)
IDX = np.arange(NM)[:, None]


def correr(var):
    t0 = time.time()
    aj = ajustador(var)
    p_oos = predecir_por_bloques(aj, IDX[fi], y[fi], bloque)
    p_val = aj(IDX[fi], y[fi])(IDX[val])
    print(f"  {var:40s} listo en {time.time() - t0:.0f}s", flush=True)
    return var, p_oos, p_val


res = Parallel(n_jobs=args.hilos, backend="threading")(delayed(correr)(v) for v in VARIANTES)
P = {v: (po, pv) for v, po, pv in res}
yb, yv = y[fi], y[val]

# ---- comparación
ref_o, ref_v = P["referencia (híbrido)"]
ev = sorted(set(bloque) - {0})
for b in ev:
    m = bloque == b
    fechas = pd.to_datetime(t[fi][m], unit="ms")
    print(f"bloque {b}: {m.sum():,} partidas, {fechas.min():%d/%m/%Y} → {fechas.max():%d/%m/%Y}, "
          f"AUC ref {roc_auc_score(yb[m], ref_o[m]):.4f}")
print(f"V: AUC ref {roc_auc_score(yv, ref_v):.4f}")

BOOT = np.random.default_rng(1).integers(0, len(yv), size=(args.B, len(yv)))
filas = []
for var, (po, pv) in P.items():
    if var.startswith("referencia"):
        continue
    tab = comparar_por_bloques(yb, po, ref_o, bloque, ev, B=args.B)
    bv = np.array([roc_auc_score(yv[s], pv[s]) - roc_auc_score(yv[s], ref_v[s]) for s in BOOT])
    fila = {"variante": var}
    for _, r in tab.iterrows():
        fila[f"Δ b{r.bloque}" if r.bloque != "media" else "Δ media"] = r.delta
    med = tab[tab.bloque == "media"].iloc[0]
    fila.update({"media lo": med.ic_lo, "media hi": med.ic_hi,
                 "bloques que mejora": int((tab[tab.bloque != "media"].delta > 0).sum()),
                 "decisión (bloques)": decidir(tab),
                 "Δ V": roc_auc_score(yv, pv) - roc_auc_score(yv, ref_v),
                 "V lo": np.percentile(bv, 2.5), "V hi": np.percentile(bv, 97.5)})
    filas.append(fila)
R = pd.DataFrame(filas)
R.to_csv(SAL / "resultado.tsv", sep="\t", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print("\n" + R.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
print(f"\n[+] {SAL / 'resultado.tsv'} ({time.time() - T0:.0f}s)")
