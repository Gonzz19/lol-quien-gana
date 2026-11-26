"""
AUC y acierto en Train y en V de los modelos que marcan el camino de «quién gana» (PROJECT_SUMMARY.md §9.S.55).

Pedido del usuario (2026-11-16): todos los modelos medidos igual, en Train y en V, sin tocar el test. Hasta
acá la regla y la logística del draft solo tenían números en el test (§9.S.20, antes de la regla de no
tocarlo), y los modelos de quién gana solo tenían V.

* **V** = V Master+ de §9.S.51: partidas completas del train con fecha ≥ 16/09 (mediana de las fechas
  posteriores a la foto 1) y mediana del tier de los 10 ≥ Master. 16.428 partidas, las mismas para todos.
  Cada modelo predice V entrenado solo con partidas anteriores al 16/09.
* **Train** = dentro de la muestra: el mismo modelo que predice V, evaluado sobre las partidas con que se
  ajustó (cada uno con su conjunto de ajuste; columna `n train`). Para los stackings, los expertos ajustados
  con todo el ajuste predicen el ajuste y el meta (ajustado fuera de muestra, como siempre) los combina.

| Modelo | Qué es | De dónde sale |
| :--- | :--- | :--- |
| Baseline | siempre el rojo (la tasa del lado) | — |
| Regla | media del winrate campeón-línea, sin leakage (§9.S.13) | `src/models/baselines/draft_rules.py` |
| Logística Draft | campeón × línea con signo (§9.S.20) | `src/models/baselines/draft_logreg.py` |
| Campeones | one-hot de campeones con signo (§9.S.23) | `ganador_features.py` |
| Campeones + Jugador | + LP de la foto + huella (§9.S.23) | `ganador_features.py` |
| Stacking | margen por campeón + stacking de 6 expertos (§9.S.29) | `ganador_red_jugador.py` ×3 |
| MOE | 2 expertos + composición + modelo por línea (§9.S.37) | `ganador_red_jugador.py`, `ganador_logistica_oos.py` |
| Logística Master+ (Elo y duelos) | escalón 9 del brainstorming (§9.S.52) | `brainstorm_final.py` |

Un modelo cuyas cachés faltan se saltea con un aviso (los comandos para generarlas están en §9.S.55).

    python scripts/analysis/modelos_train_v.py            # ~3 min; escribe data/analysis/modelos_train_v.csv
"""
import pickle
import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.models.baselines.draft_logreg import complete_match_ids, draft_matrix, fit_temporal_C, match_target  # noqa: E402
from src.models.baselines.draft_rules import LANES, match_rule_scores, stats_from_participants  # noqa: E402
from src.features.padron import verificar_foto1  # noqa: E402

AN = REPO / "data" / "analysis"
SALIDA = AN / "modelos_train_v.csv"
ORDEN = {"DIAMOND": 0, "MASTER": 1, "GRANDMASTER": 2, "CHALLENGER": 3}
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))  # noqa: E731
filas = []


def registrar(nombre, descripcion, y_tr, p_tr, y_v, p_v):
    f = {"modelo": nombre, "descripcion": descripcion,
         "AUC train": roc_auc_score(y_tr, p_tr), "acierto train": float(((p_tr > 0.5) == y_tr).mean()),
         "AUC V": roc_auc_score(y_v, p_v), "acierto V": float(((p_v > 0.5) == y_v).mean()),
         "n train": int(len(y_tr)), "n V": int(len(y_v))}
    filas.append(f)
    print(f"  {nombre:34s} train {f['AUC train']:.4f} / {f['acierto train']:.1%} (n {f['n train']:,})   "
          f"V {f['AUC V']:.4f} / {f['acierto V']:.1%} (n {f['n V']:,})", flush=True)


# ------------------------------------------------------------------ partición común
COLS = ["match_id", "team_id", "win", "champion_name", "lane", "gold_15", "game_creation", "player_tier"]
parts = pd.read_csv(REPO / "data/splits/train/full_dataset.csv", usecols=COLS)
parts = parts[parts.lane.isin(LANES)]
ids = complete_match_ids(parts)
ids = ids[parts[parts.match_id.isin(ids)].groupby("match_id").win.sum().reindex(ids).to_numpy() == 5]  # un ganador
t = parts.groupby("match_id").game_creation.first().reindex(ids).to_numpy()
with sqlite3.connect(f"file:{REPO}/data/lol_las.db?mode=ro", uri=True) as con:
    t_foto = verificar_foto1(con)
MITAD = np.median(t[t >= t_foto])
# = es_master_plus: mediana del tier de los 10 (sin contar los que no tienen dato) ≥ Master
med = parts.assign(tn=parts.player_tier.map(ORDEN)).groupby("match_id").tn.median().reindex(ids)
MASTER = pd.Series(med.fillna(-1).to_numpy() >= 1.0, index=ids)
es_v = (t >= MITAD) & MASTER.to_numpy()
V_IDS = set(ids[es_v])
print(f"V Master+: {es_v.sum():,} partidas desde {pd.Timestamp(MITAD, unit='ms', tz='UTC'):%d/%m %H:%M} UTC", flush=True)


def en_v(mids):
    return np.array([m in V_IDS for m in mids])


# ------------------------------------------------------------------ 1-3. baseline, regla y logística del draft
tr_ids, v_ids = ids[t < MITAD], ids[es_v]
ids2 = tr_ids.append(v_ids)
tr, va = np.arange(len(tr_ids)), np.arange(len(tr_ids), len(ids2))
creation = parts.groupby("match_id").game_creation.first().reindex(ids2).to_numpy()
p_tr, p_v = parts[parts.match_id.isin(tr_ids)], parts[parts.match_id.isin(v_ids)]
y = match_target(parts, ids2)
c = y[tr].mean()
registrar("Baseline", "siempre el rojo", y[tr], np.full(len(tr), c), y[va], np.full(len(va), c))
stats = stats_from_participants(p_tr)
registrar("Regla", "media del winrate campeón-línea (sin leakage)", y[tr],
          match_rule_scores(p_tr, stats)[0].reindex(tr_ids)["prob"].to_numpy(), y[va],
          match_rule_scores(p_v, stats)[0].reindex(v_ids)["prob"].to_numpy())
X, _ = draft_matrix(parts, ids2, vocab=draft_matrix(parts, tr_ids)[1])
m, C, _ = fit_temporal_C(X, y, tr, creation)
registrar("Logística Draft", f"campeón × línea con signo (C = {C})", y[tr], m.predict_proba(X[tr])[:, 1],
          y[va], m.predict_proba(X[va])[:, 1])
del parts, p_tr, p_v, X

# ------------------------------------------------------------------ 4-5. campeones; campeones + LP + huella (§9.S.23)
ruta = AN / "ganador_feats.pkl"
if ruta.exists():
    d = pickle.load(open(ruta, "rb"))
    feat, yy, fit, val = d["feat"], d["y"], d["fit"], d["val"]
    mids = d["filas"].match_id.unique()
    assert len(mids) == len(yy)
    vm = val & en_v(mids)
    for nombre, desc, keys in [("Campeones", "one-hot de campeones con signo", ["onehot"]),
                               ("Campeones + Jugador", "campeones + LP de la foto + huella", ["onehot", "lp", "huella"])]:
        Z = np.hstack([feat[k].toarray() if hasattr(feat[k], "toarray") else feat[k] for k in keys])
        sc = StandardScaler(with_mean=False).fit(Z[fit])           # como ganador_modelos.py
        lr = LogisticRegression(C=0.05, max_iter=5000).fit(sc.transform(Z[fit]), yy[fit])
        registrar(nombre, desc, yy[fit], lr.predict_proba(sc.transform(Z[fit]))[:, 1],
                  yy[vm], lr.predict_proba(sc.transform(Z[vm]))[:, 1])
    del d, feat
else:
    print(f"  (falta {ruta.name}: correr ganador_features.py)")


# ------------------------------------------------------------------ 6-7. stacking de 6 y MOE
def stacking(nombre, desc, expertos, cfg="C"):
    """`expertos`: lista de (archivo _oos.npz, clave). Meta = logística sobre logits, ajustada fuera de muestra."""
    faltan = [f for f, _ in expertos if not (AN / f).exists()]
    if faltan:
        print(f"  (falta {', '.join(sorted(set(faltan)))}: se saltea {nombre})")
        return
    F = {f: np.load(AN / f) for f, _ in expertos}
    base = F[expertos[0][0]]
    for f in F.values():
        assert (f[f"{cfg}_oos_idx"] == base[f"{cfg}_oos_idx"]).all() and (f[f"{cfg}_val_idx"] == base[f"{cfg}_val_idx"]).all()
        if f"{cfg}_fit_idx" not in f.files:
            print(f"  (las cachés de {nombre} no traen la predicción en el ajuste: regenerarlas con el script actual)")
            return
    Zo = np.c_[[lg(F[f][f"{cfg}_oos_{k}"]) for f, k in expertos]].T
    Zv = np.c_[[lg(F[f][f"{cfg}_val_{k}"]) for f, k in expertos]].T
    Zf = np.c_[[lg(F[f][f"{cfg}_fit_{k}"]) for f, k in expertos]].T
    meta = LogisticRegression(C=1e4, max_iter=5000).fit(Zo, base[f"{cfg}_oos_y"])
    gm = pickle.load(open(AN / "ganador_feats_grande.pkl", "rb"))
    yy, mids = gm["y"], gm["mids"]
    vi, fi = base[f"{cfg}_val_idx"], base[f"{cfg}_fit_idx"]
    sel = en_v(mids[vi])
    registrar(nombre, desc, yy[fi], meta.predict_proba(Zf)[:, 1], yy[vi][sel], meta.predict_proba(Zv)[:, 1][sel])


SEIS = [(f"ganador_red_jugador_{n}_oos.npz", k) for n in ("margen", "poisson", "grupos") for k in ("logistica", "red_equipo")]
stacking("Stacking", "margen por campeón + stacking de 6 expertos", SEIS)
stacking("MOE", "2 expertos (red de grupos + logística) + composición + modelo por línea",
         [("ganador_red_jugador_grupos_oos.npz", "red_equipo"), ("ganador_logistica_composicion_oos.npz", "logistica")])

# ------------------------------------------------------------------ 8. escalón 9 del brainstorming
NECESARIAS = ["margen_grupos.npz", "mmr_historico.npz", "gemini_mpr.npz", "gemini_nuevas_v2.npz", "gemini_g11.npz",
              "gemini_hgb_lineas.npz", "gemini_g23.npz"]
faltan = [n for n in NECESARIAS if not (AN / "brainstorm" / n).exists()]
if faltan:
    print(f"  (faltan {', '.join(faltan)} en data/analysis/brainstorm: correr los scripts de brainstorm_final.py)")
else:
    from scripts.analysis.brainstorm_base import ajustar_predecir, cargar, extras_referencia
    D = cargar()
    Xe = extras_referencia(D, escalon=9)
    trb = D["fi_v"][D["master"][D["fi_v"]]]
    vib = np.flatnonzero(D["val"])
    pv, lr = ajustar_predecir(D, trb, vib, extra=Xe, extra_post=False, con_mapa=False, devolver_modelo=True)
    pt = ajustar_predecir(D, trb, trb, extra=Xe, extra_post=False, con_mapa=False)   # mismo ajuste, predice el ajuste
    registrar("Logística Master+ (Elo y duelos)",
              "escalón 9: Master+, margen jugador-campeón, nivel histórico, W/L foto, matchups, Elo decreciente, "
              "tempo, modelo por línea y margen esperado por duelo", D["y"][trb], pt, D["y"][vib], pv)

T = pd.DataFrame(filas)
T.to_csv(SALIDA, index=False)
print(f"[+] {SALIDA}")
