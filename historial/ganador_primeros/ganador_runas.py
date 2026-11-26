"""Quién gana: ¿suman las runas, los hechizos y el nivel de cuenta? (§9.S.37)

Entrada: data/runas/runas.db (scripts/collection/runas_hechizos.py), con una muestra de partidas
del bloque 4 del ajuste y de V. Todo lo que se usa se elige en la selección de campeones: es
información previa a la partida, sin fuga.

Variables por jugador:
  * nivel de cuenta: log, y si es < 100 (cuenta nueva);
  * hechizos (summoner1Id = tecla D, summoner2Id = F): sin Flash, Flash en la F, Smite fuera de la jungla, Teleport, Ignite, Exhaust, Ghost, Cleanse;
  * rareza: qué tan raro es su combo de hechizos y su runa clave para ese campeón en esa línea
    (frecuencia suavizada, contada solo con la parte de la muestra que se usa para ajustar).
Por partida entran como azul − rojo.

Evaluación: sobre la predicción del modelo de 2 expertos (red de grupos + logística con composición),
fuera de muestra en el bloque 4 y la de siempre en V. Una logística chica con
[logit(p_base), variables nuevas] se ajusta en la parte de la muestra del bloque 4 y se evalúa
en la parte de V, contra la misma logística solo con logit(p_base) (recalibración).
Diferencia pareada de AUC por bootstrap. Solo data/splits/train."""
import argparse, pickle, sqlite3, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss
warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
AN = REPO / "data" / "analysis"
ap = argparse.ArgumentParser()
ap.add_argument("--runas", default=str(REPO / "data" / "runas" / "runas.db"))
ap.add_argument("--feats", default=str(AN / "ganador_feats_grande.pkl"))
ap.add_argument("--red", default=str(AN / "ganador_red_jugador_grupos_oos.npz"))
ap.add_argument("--logistica", default=str(AN / "ganador_logistica_composicion_oos.npz"))
ap.add_argument("--boot", type=int, default=2000)
args = ap.parse_args()

lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

# ---------- predicción base del modelo de 2 expertos, por partida
d = pickle.load(open(args.feats, "rb"))
mids, y = d["mids"], d["y"].astype(int)
del d
R, L = np.load(args.red), np.load(args.logistica)
ir, il = R["C_oos_idx"], L["C_oos_idx"]
com, a, b = np.intersect1d(ir, il, return_indices=True)
meta = LogisticRegression().fit(np.c_[lg(R["C_oos_red_equipo"][a]), lg(L["C_oos_logistica"][b])], y[com])
z = np.full(len(mids), np.nan)
blq = np.full(len(mids), -1)
z[com] = meta.decision_function(np.c_[lg(R["C_oos_red_equipo"][a]), lg(L["C_oos_logistica"][b])])
blq[com] = L["C_oos_bloque"][b]
vi = L["C_val_idx"]
z[vi] = meta.decision_function(np.c_[lg(R["C_val_red_equipo"]), lg(L["C_val_logistica"])])
blq[vi] = 99
base = pd.DataFrame({"z": z, "y": y, "bloque": blq}, index=mids)
# (el meta se ajusta con los bloques 1-4; usarlo en el bloque 4 es leve optimismo para la base, no para lo nuevo)

# ---------- runas por jugador
with sqlite3.connect(f"file:{args.runas}?mode=ro", uri=True) as con:
    J = pd.read_sql_query("SELECT * FROM runas", con)
J = J[J.match_id.isin(base.index)]
J = J[J.match_id.map(base.bloque).isin([4, 99])]
cuenta = J.groupby("match_id").size()
J = J[J.match_id.map(cuenta) == 10].copy()
J["parte"] = np.where(J.match_id.map(base.bloque) == 4, "ajuste", "V")
print(f"partidas con runas: ajuste (bloque 4) {J[J.parte == 'ajuste'].match_id.nunique():,}, "
      f"V {J[J.parte == 'V'].match_id.nunique():,}")

FLASH, SMITE, TP, IGN, EXH, GHOST, CLEANSE = 4, 11, 12, 14, 3, 6, 1
sp = J[["hechizo1", "hechizo2"]].to_numpy()
tiene = lambda s: (sp == s).any(1).astype(float)
J["log_nivel"] = np.log(J.summoner_level.clip(lower=1))
J["cuenta_nueva"] = (J.summoner_level < 100).astype(float)
J["sin_flash"] = 1 - tiene(FLASH)
J["flash_en_f"] = (J.hechizo2 == FLASH).astype(float)        # summoner2Id = tecla F
J["smite_fuera"] = tiene(SMITE) * (J.team_position != "JUNGLE")
for nombre, s in (("tp", TP), ("ignite", IGN), ("exhaust", EXH), ("ghost", GHOST), ("cleanse", CLEANSE)):
    J[nombre] = tiene(s)
J["combo"] = np.sort(sp, 1)[:, 0] * 100 + np.sort(sp, 1)[:, 1]


def rareza(col):
    """1 - frecuencia suavizada de `col` para (campeón, línea), contada solo en la parte de ajuste"""
    aj = J[J.parte == "ajuste"]
    n_cl = aj.groupby(["champion_id", "team_position"]).size()
    n_x = aj.groupby(["champion_id", "team_position", col]).size()
    glob = aj[col].value_counts(normalize=True)
    k = list(zip(J.champion_id, J.team_position, J[col]))
    n = np.array([n_x.get(t, 0) for t in k], float)
    # en el ajuste, sin la propia fila (leave-one-out)
    n -= (J.parte == "ajuste").to_numpy()
    tot = np.array([n_cl.get(t[:2], 0) for t in k], float) - (J.parte == "ajuste").to_numpy()
    prior = J[col].map(glob).fillna(0).to_numpy()
    return 1 - (n + 5 * prior) / (tot + 5)


J["rareza_hechizos"] = rareza("combo")
J["rareza_runa"] = rareza("runa_clave")
J["runa_rara"] = (J.rareza_runa > 0.85).astype(float)
VARS = ["log_nivel", "cuenta_nueva", "sin_flash", "flash_en_f", "smite_fuera", "tp", "ignite", "exhaust", "ghost", "cleanse",
        "rareza_hechizos", "rareza_runa", "runa_rara"]
J["s"] = np.where(J.team_id == 100, 1, -1)
X = (J[VARS].mul(J.s, axis=0)).groupby(J.match_id).sum()
P = base.loc[X.index]
aj, va = (P.bloque == 4).to_numpy(), (P.bloque == 99).to_numpy()
print("\nmedias por jugador:", J[VARS].mean().round(3).to_dict())


def evaluar(cols):
    M = np.c_[P.z.to_numpy(), X[cols].to_numpy()] if cols else P.z.to_numpy()[:, None]
    sc = StandardScaler().fit(M[aj])
    m = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(M[aj]), P.y[aj])
    return m.predict_proba(sc.transform(M[va]))[:, 1], m


yv = P.y.to_numpy()[va]
p0, _ = evaluar([])
print(f"\nbase (modelo de 2, recalibrado) en la muestra de V: AUC {roc_auc_score(yv, p0):.4f}  log loss {log_loss(yv, p0):.4f}")
rng = np.random.default_rng(0)
B = [rng.integers(0, len(yv), len(yv)) for _ in range(args.boot)]
grupos = {"nivel de cuenta": ["log_nivel", "cuenta_nueva"],
          "hechizos": ["sin_flash", "flash_en_f", "smite_fuera", "tp", "ignite", "exhaust", "ghost", "cleanse"],
          "rareza (hechizos y runa)": ["rareza_hechizos", "rareza_runa", "runa_rara"],
          "todo": VARS}
for nombre, cols in grupos.items():
    p1, m = evaluar(cols)
    dif = [roc_auc_score(yv[i], p1[i]) - roc_auc_score(yv[i], p0[i]) for i in B]
    print(f"+ {nombre:26s} AUC {roc_auc_score(yv, p1):.4f}  dif {roc_auc_score(yv, p1) - roc_auc_score(yv, p0):+.4f} "
          f"IC95 [{np.percentile(dif, 2.5):+.4f}; {np.percentile(dif, 97.5):+.4f}]  log loss {log_loss(yv, p1):.4f}")
    if nombre == "todo":
        print("  coeficientes (estandarizados):", dict(zip(["z_base"] + cols, np.round(m.coef_[0], 3))))
