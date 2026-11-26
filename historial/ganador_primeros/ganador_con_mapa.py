"""
Validación por bloques y curva de aprendizaje del modelo de quién gana CON el mapa de LP
(PROJECT_SUMMARY.md §9.S.50). Repite §9.S.47 y §9.S.48 sumando al híbrido las variables del mapa
causal de §9.S.46 (`ganador_mapa_lp.py`, caché `data/analysis/mapa_lp_causal_v.pkl`).

Variables del mapa (representación "equipo": media azul − rojo de las 5 líneas, como el mejor
caso de §9.S.46). Existen solo desde la foto 1 (t1 = 04/09). Antes de t1 valen 0, y se
estandarizan con las filas de entrenamiento posteriores a t1 (`escalar_post` de §9.S.46):

* nivel(foto): nivel del lobby partida a partida con el LP de la foto (acumulado, últimas 10,
  log de partidas con nivel), de `src/features/lobby.py`;
* niv(mapa) + mapa: lo mismo con el LP del mapa, más LP del mapa, cambio desde la foto,
  incertidumbre, tiene foto y brecha nivel − LP (el mejor de §9.S.46);
* actividad: tiene foto y log de partidas con nivel.

Bloques: el mapa solo existe desde t1, así que los bloques de §9.S.47 no sirven (en los bloques
1-3 las variables valen 0). Se parte el tramo de ajuste posterior a t1 (A, 04/09 → 16/09,
26.934 partidas) en 4 sub-bloques por fecha. A2, A3 y A4 se predicen hacia adelante, cada uno con
todo el ajuste anterior (antes de t1 + los sub-bloques previos). V confirma.

    python historial/ganador_primeros/ganador_con_mapa.py      # ~25 min, 4 hilos
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
import scripts.analysis.ganador_mapa_lp as gm  # noqa: E402
from src.features.lobby import prior_lobby_level  # noqa: E402
from src.models.validacion_temporal import (bloques_por_fecha, comparar_por_bloques, decidir,  # noqa: E402
                                            particion_v_master)

ap = argparse.ArgumentParser()
ap.add_argument("--hilos", type=int, default=4)
ap.add_argument("--B", type=int, default=200)
args = ap.parse_args()
T0 = time.time()
SAL = REPO / "data" / "analysis" / "ganador_con_mapa"
SAL.mkdir(parents=True, exist_ok=True)

# ---- datos y variables del mapa (misma construcción que ganador_mapa_lp.main, §9.S.46)
Z = np.load(gm.RUTA_NPZ, allow_pickle=True)
mids, Xj, champ, t, val, y = Z["mids"], Z["Xj"], Z["champ"], Z["t"], Z["val"].astype(bool), Z["y"].astype(int)
NM = len(mids)
# Regla desde 2026-11-15 (§9.S.51): V = solo Master+; las de Diamond de la ventana pasan a A4
val0 = val
val, diam_v = particion_v_master(val0, Z["tnum"])
fit = ~val
print(f"V solo Master+: {val.sum():,} partidas; {diam_v.sum():,} de Diamond pasan a A4")
foto1, t1 = gm._cargar_foto1(gm.RUTA_DB)
cache = gm._cargar_o_calcular_mapa(mids, t, y, t1, foto1, None, rehacer=False)
post = t >= t1
fit_post = post & fit
dpc, keys = cache["df_post_cron"], cache["df_npz_keys"].copy()
en_fp = dpc.match_id.isin(set(mids[fit_post]))
mu_f, mu_m = float(dpc.loc[en_fp, "lp_foto"].mean()), float(dpc.loc[en_fp, "lp_causal"].mean())
nf = prior_lobby_level(dpc, "lp_foto", k=5, mu0=mu_f, window=10, min_otros=3)
nm_ = prior_lobby_level(dpc, "lp_causal", k=5, mu0=mu_m, window=10, min_otros=3)
F = dpc.copy()
for p, n in (("foto", nf), ("mapa", nm_)):
    F[f"{p}_acum"], F[f"{p}_v10"] = n["nivel_acum"].to_numpy(float), n["nivel_v10"].to_numpy(float)
    F[f"{p}_n"] = np.log1p(n["n_nivel"].to_numpy(float))
keys["_pos"] = np.arange(len(keys))
M = keys.merge(F.drop(columns=["game_creation", "team_id", "lane"]), on=["match_id", "player_id"], how="left") \
        .sort_values("_pos", kind="mergesort").reset_index(drop=True)
assert len(M) == 10 * NM
pr = np.repeat(post, 10)


def col(c, mu=0.0):
    return np.where(pr, M[c].fillna(mu).to_numpy(np.float32) - mu, 0.0)


tiene = (M.lp_foto.notna() & pr).to_numpy(np.float32)
lp_m = np.where(pr, np.where(M.metodo.to_numpy() == "causal_lobby", M.mapa_v10.fillna(mu_m), M.lp_causal.fillna(mu_m))
                .astype(np.float32) - mu_m, 0.0)
dl = np.where(tiene > 0.5, lp_m - col("lp_foto", mu_f), 0.0)
inc = M.lp_incert.to_numpy(np.float32)
inc_mu = float(np.nanmean(inc[np.repeat(fit_post, 10)]))
lp_inc = np.where(pr, np.nan_to_num(inc, nan=inc_mu) - inc_mu, 0.0)
nfa, nfv, nfn = col("foto_acum", mu_f), col("foto_v10", mu_f), col("foto_n")
nma, nmv, nmn = col("mapa_acum", mu_m), col("mapa_v10", mu_m), col("mapa_n")


def eq(*cs):
    a = np.stack([c.reshape(NM, 2, 5) for c in cs], axis=-1)
    return (a[:, 0] - a[:, 1]).mean(axis=1).astype(np.float32)          # "equipo"


EXTRA = {
    "+ actividad (tiene foto, partidas con nivel)": eq(tiene, nfn),
    "+ nivel(foto)": eq(nfa, nfv, nfn),
    "+ niv(mapa) + mapa": eq(nma, nmv, nmn, lp_m, dl, lp_inc, tiene, nmv - lp_m),
}
print(f"variables del mapa listas ({time.time() - T0:.0f}s); post t1: ajuste {fit_post.sum():,}, V {val.sum():,}")

# ---- híbrido (como validacion_bloques.py)
Xd = (Xj[:, 0] - Xj[:, 1]).reshape(NM, -1).astype(np.float32)
DRAFT = draft_matrix_by_lane(champ)


def ajustar_predecir(tr, te, extra=None):
    mu, sd = Xd[tr].mean(0), Xd[tr].std(0) + 1e-6
    partes = lambda f: [DRAFT[f], sparse.csr_matrix((Xd[f] - mu) / sd)]  # noqa: E731
    Ptr, Pte = partes(tr), partes(te)
    if extra is not None:
        ptr = post[tr]
        me, se = extra[tr][ptr].mean(0), extra[tr][ptr].std(0) + 1e-6
        Ptr.append(sparse.csr_matrix(np.where(post[tr][:, None], (extra[tr] - me) / se, 0).astype(np.float32)))
        Pte.append(sparse.csr_matrix(np.where(post[te][:, None], (extra[te] - me) / se, 0).astype(np.float32)))
    lr = LogisticRegression(C=0.05, max_iter=3000).fit(sparse.hstack(Ptr).tocsr(), y[tr])
    return lr.predict_proba(sparse.hstack(Pte).tocsr())[:, 1]


# ---- 1. validación por sub-bloques de A + V
fi = np.flatnonzero(fit)
ai = np.flatnonzero(fit_post & ~val0)                 # A original: 04/09 → 16/09
sub = bloques_por_fecha(t[ai], 4)                      # A1..A4 → 0..3
bloque_fit = np.zeros(NM, int) - 1
bloque_fit[fi] = 0                                     # antes de t1 y A1 solo entrenan
bloque_fit[ai] = sub
bloque_fit[diam_v] = 3                                 # las de Diamond de la ventana de V, a A4
EV = [1, 2, 3]
fi_v = fi[~diam_v[fi]]                                 # entrenamiento de V: todo lo anterior al 16/09
VARS = {"referencia (híbrido)": None, **EXTRA}


def correr(nombre):
    ex = VARS[nombre]
    po = np.full(NM, np.nan)
    for b in EV:
        tr = fi[bloque_fit[fi] < b]
        te = np.flatnonzero(bloque_fit == b)
        po[te] = ajustar_predecir(tr, te, ex)
    # el modelo de V se entrena solo con partidas anteriores a la ventana de V: las de Diamond de
    # la ventana (A4) se evalúan hacia adelante pero no entrenan a V (§9.S.51, sin mirar adelante)
    pv = ajustar_predecir(fi_v, np.flatnonzero(val), ex)
    print(f"  {nombre:46s} listo ({time.time() - T0:.0f}s)", flush=True)
    return nombre, po, pv


res = Parallel(n_jobs=args.hilos, backend="threading")(delayed(correr)(v) for v in VARS)
P = {n: (po, pv) for n, po, pv in res}
for b in EV:
    m = bloque_fit == b
    f = pd.to_datetime(t[m], unit="ms")
    print(f"sub-bloque A{b + 1}: {m.sum():,} partidas, {f.min():%d/%m %H:%M} → {f.max():%d/%m %H:%M}, "
          f"AUC ref {roc_auc_score(y[m], P['referencia (híbrido)'][0][m]):.4f}, "
          f"acierto ref {((P['referencia (híbrido)'][0][m] > 0.5) == y[m]).mean():.1%}")
yv = y[val]
BOOT = np.random.default_rng(1).integers(0, len(yv), size=(args.B, len(yv)))
ref_o, ref_v = P["referencia (híbrido)"]
mb = bloque_fit >= 1
filas = []
for n, (po, pv) in P.items():
    if n.startswith("referencia"):
        continue
    tab = comparar_por_bloques(y[mb], po[mb], ref_o[mb], bloque_fit[mb], EV, B=args.B)
    bv = np.array([roc_auc_score(yv[s], pv[s]) - roc_auc_score(yv[s], ref_v[s]) for s in BOOT])
    fila = {"variante": n}
    for _, r in tab.iterrows():
        fila["Δ media" if r.bloque == "media" else f"Δ A{int(r.bloque) + 1}"] = r.delta
    for b in EV:
        m = bloque_fit == b
        fila[f"acierto A{b + 1}"] = ((po[m] > 0.5) == y[m]).mean()
    med = tab[tab.bloque == "media"].iloc[0]
    fila.update({"media lo": med.ic_lo, "media hi": med.ic_hi, "decisión": decidir(tab),
                 "AUC V": roc_auc_score(yv, pv), "acierto V": ((pv > 0.5) == yv).mean(), "Δ V": roc_auc_score(yv, pv) - roc_auc_score(yv, ref_v),
                 "V lo": np.percentile(bv, 2.5), "V hi": np.percentile(bv, 97.5)})
    filas.append(fila)
R = pd.DataFrame(filas)
R.to_csv(SAL / "bloques.tsv", sep="\t", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 20)
print(f"\nV (Master+): AUC ref {roc_auc_score(yv, ref_v):.4f}, acierto ref {((ref_v > 0.5) == yv).mean():.1%}")
print(R.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

# ---- 2. curva de aprendizaje con el mapa (las N más recientes antes de V)
fo = fi_v[np.argsort(t[fi_v])]                         # sin las Diamond de la ventana de V
TAM = [5_000, 10_000, 20_000, 26_934, 40_000, 80_000, len(fo)]
best = EXTRA["+ niv(mapa) + mapa"]


def curva(n, con):
    tr = fo[-n:]
    return n, con, ajustar_predecir(tr, np.flatnonzero(val), best if con else None)


res = Parallel(n_jobs=args.hilos, backend="threading")(delayed(curva)(n, c) for n in TAM for c in (False, True))
C = {(n, c): p for n, c, p in res}
filas = []
for n in TAM:
    p0, p1 = C[(n, False)], C[(n, True)]
    b = np.array([roc_auc_score(yv[s], p1[s]) - roc_auc_score(yv[s], p0[s]) for s in BOOT])
    filas.append({"partidas": n, "desde": f"{pd.to_datetime(t[fo[-n:]].min(), unit='ms'):%d/%m/%Y}",
                  "post t1 en el ajuste": post[fo[-n:]].mean(), "AUC sin mapa": roc_auc_score(yv, p0),
                  "AUC con mapa": roc_auc_score(yv, p1), "acierto sin mapa": ((p0 > 0.5) == yv).mean(),
                  "acierto con mapa": ((p1 > 0.5) == yv).mean(), "Δ mapa": roc_auc_score(yv, p1) - roc_auc_score(yv, p0),
                  "lo": np.percentile(b, 2.5), "hi": np.percentile(b, 97.5)})
Cu = pd.DataFrame(filas)
Cu.to_csv(SAL / "curva.tsv", sep="\t", index=False)
print("\nCurva de aprendizaje (V), con y sin niv(mapa) + mapa:")
print(Cu.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"[+] {SAL} ({time.time() - T0:.0f}s)")
