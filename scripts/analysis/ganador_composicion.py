"""Composición del draft con atributos externos de campeones (§9.S.37).

Fuentes (bajadas a data/externos/, ver bajar_campeones_externos.py):
  * CommunityDragon, `champions/{id}.json` del cliente: tipo de daño (físico / mágico / mixto)
    y playstyleInfo (daño, durabilidad, control, movilidad, utilidad; 0-3);
  * Meraki Analytics: roles finos (TANK, VANGUARD, ENCHANTER...), cuerpo a cuerpo / rango y
    rango de ataque.
Son atributos fijos del campeón, no salen de resultados: no hay fuga temporal.

Por equipo se arman variables de composición (cuánto daño mágico, si es todo AD, si no hay
frontline o engage, poke, rango...) y entran como azul − rojo. Lo nuevo frente al one-hot
es lo no aditivo: "todo AD" o "cero engage" no es suma de efectos por campeón.

Escribe data/analysis/ganador_composicion.pkl con {"mids", "nuevo": {"composicion": X}, "jug"}, para
`ganador_logistica_oos.py --extra ...:composicion` y `ganador_red_jugador.py --composicion`, y una
prueba rápida solo con el draft.
Solo data/splits/train."""
import argparse, json, pickle, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
AN, EXT = REPO / "data" / "analysis", REPO / "data" / "externos"
ap = argparse.ArgumentParser()
ap.add_argument("--feats", default=str(AN / "ganador_feats_grande.pkl"))
ap.add_argument("--salida", default=str(AN / "ganador_composicion.pkl"))
args = ap.parse_args()

# ---------- atributos por campeón
ruta_cd = EXT / "cdragon" / "champions_tactical.json"
cd = json.load(open(ruta_cd, encoding="utf-8")) if ruta_cd.exists() else {}
mk = {int(v["id"]): v for v in json.load(open(EXT / "meraki" / "champions.json", encoding="utf-8")).values()}
AP = {"kMagic": 1.0, "kPhysical": 0.0, "kMixed": 0.5}
FRONT, ENGAGE = {"TANK", "VANGUARD", "WARDEN", "JUGGERNAUT"}, {"VANGUARD", "CATCHER", "DIVER"}
filas = []
cd = {int(k): v for k, v in cd.items()}
MK_PS = {"damage": "damage", "durability": "toughness", "crowdControl": "control",
         "mobility": "mobility", "utility": "utility"}      # Meraki 1-3 si falta CommunityDragon
for cid in sorted(set(cd) | set(mk)):
    c, m = cd.get(cid, {}), mk.get(cid, {})
    ti, ps = c.get("tacticalInfo") or {}, c.get("playstyleInfo") or {}
    if not ps and m:
        ps = {k: m["attributeRatings"].get(v) for k, v in MK_PS.items()}
    roles = set(m.get("roles") or [])
    ap_ = AP.get(ti.get("damageType"))
    if ap_ is None and m:
        ap_ = 1.0 if m.get("adaptiveType") == "MAGIC_DAMAGE" else 0.0
    filas.append({"champion_id": cid, "ap": ap_,
                  "ps_dano": ps.get("damage"), "ps_dur": ps.get("durability"), "ps_cc": ps.get("crowdControl"),
                  "ps_mov": ps.get("mobility"), "ps_util": ps.get("utility"),
                  "rango": float(m.get("attackType") == "RANGED") if m else None,
                  "alcance": (m.get("stats", {}).get("attackRange", {}) or {}).get("flat") if m else None,
                  "front": float(bool(roles & FRONT)), "engage": float(bool(roles & ENGAGE)),
                  "poke": float("ARTILLERY" in roles), "enchanter": float("ENCHANTER" in roles),
                  "asesino": float("ASSASSIN" in roles), "tirador": float("MARKSMAN" in roles)})
C = pd.DataFrame(filas).set_index("champion_id").astype(float)
C = C.fillna(C.mean())
C.to_csv(EXT / "atributos_campeones.csv")
print(f"atributos de {len(C)} campeones (CDragon {len(cd)}, Meraki {len(mk)})")

# ---------- campeones por partida, en el orden de ganador_feats_grande.pkl
d = pickle.load(open(args.feats, "rb"))
mids, y, val, fit, feat = d["mids"], d["y"], d["val"], d["masks"]["C"], d["feat"]
camp = d["jug"]["campeon"]                     # (partida, puesto) con índices de campeón ordenados
df = pd.read_csv(REPO / "data" / "splits" / "train" / "full_dataset.csv", usecols=["champion_id"])
orden = np.array(sorted(df.champion_id.unique()))
del df, d
falta = set(orden) - set(C.index)
print(f"campeones del train sin atributos: {sorted(falta)}")
A = C.reindex(orden).fillna(C.mean())          # fila i = campeón de índice i
cols = list(A.columns)
V = A.to_numpy()[camp.astype(int)]             # (partida, 10, atributo)


def equipo(E):
    """E: (partida, 5, atributo) -> variables de composición del equipo"""
    g = {k: E[:, :, i] for i, k in enumerate(cols)}
    ap_ = g["ap"].sum(1)
    out = {"n_ap": ap_, "todo_ad": (ap_ <= 0.5) * 1., "todo_ap": (ap_ >= 4) * 1.,
           "desbalance_dano": np.abs(ap_ - 2.5),
           "dur": g["ps_dur"].sum(1), "cc": g["ps_cc"].sum(1), "mov": g["ps_mov"].sum(1),
           "util": g["ps_util"].sum(1), "dano": g["ps_dano"].sum(1),
           "n_front": g["front"].sum(1), "sin_front": (g["front"].sum(1) == 0) * 1.,
           "n_engage": g["engage"].sum(1), "sin_engage": (g["engage"].sum(1) == 0) * 1.,
           "n_rango": g["rango"].sum(1), "alcance": g["alcance"].mean(1) / 100,
           "n_poke": g["poke"].sum(1), "n_enchanter": g["enchanter"].sum(1),
           "n_asesino": g["asesino"].sum(1), "n_tirador": g["tirador"].sum(1),
           "sin_cc": (g["ps_cc"].sum(1) <= 3) * 1., "poco_dur": (g["ps_dur"].sum(1) <= 5) * 1.}
    return out


az, ro = equipo(V[:, :5]), equipo(V[:, 5:])
nombres = list(az)
X = np.c_[tuple(az[k] - ro[k] for k in nombres)]
# cruces con el rival: lo que una composición castiga en la otra
cruces = {"poke_vs_sin_engage": az["n_poke"] * ro["sin_engage"] - ro["n_poke"] * az["sin_engage"],
          "asesino_vs_poco_dur": az["n_asesino"] * ro["poco_dur"] - ro["n_asesino"] * az["poco_dur"],
          "ap_vs_dur": az["n_ap"] * ro["dur"] - ro["n_ap"] * az["dur"],
          "engage_vs_rango": az["n_engage"] * ro["n_rango"] - ro["n_engage"] * az["n_rango"]}
X = np.c_[X, np.c_[tuple(cruces.values())]]
nombres += list(cruces)
print(f"{X.shape[1]} columnas: {', '.join(nombres)}")
pct = lambda a: f"{a.mean():.1%}"
print(f"frecuencia (equipo azul): todo AD {pct(az['todo_ad'])}, todo AP {pct(az['todo_ap'])}, "
      f"sin frontline {pct(az['sin_front'])}, sin engage {pct(az['sin_engage'])}")
# "jug": atributos del campeón de cada jugador, (partida, puesto, atributo) con puesto = 5 * rojo + línea,
# estandarizados; para ganador_red_jugador.py --composicion
Vz = ((V - A.mean().to_numpy()) / (A.std().to_numpy() + 1e-9)).astype(np.float32)
pickle.dump({"mids": mids, "nuevo": {"composicion": X}, "nombres": nombres, "jug": Vz, "atributos": cols},
            open(args.salida, "wb"), protocol=4)
print(f"escrito {args.salida}")

# ---------- prueba rápida: solo el draft (sin jugadores), ajuste C y validación V
yb = y.astype(int)
fi, vi = np.where(fit)[0], np.where(val)[0]


def auc(M):
    sc = StandardScaler(with_mean=False).fit(M[fi])
    lr = LogisticRegression(C=0.05, max_iter=5000).fit(sc.transform(M[fi]), yb[fi])
    return roc_auc_score(yb[vi], lr.predict_proba(sc.transform(M[vi]))[:, 1]), lr


oh = feat["onehot"]
dr = np.c_[oh, feat["matchups"], feat["sinergias"]]
for nombre, M in [("one-hot", oh), ("one-hot + composición", np.c_[oh, X]), ("composición sola", X),
                  ("one-hot + matchups + sinergias", dr), ("  … + composición", np.c_[dr, X])]:
    a, lr = auc(M)
    print(f"solo draft | {nombre:32s} AUC V {a:.4f}")
a, lr = auc(X)
coef = pd.Series(lr.coef_[0], index=nombres).sort_values(key=np.abs, ascending=False)
print("coeficientes (estandarizados) de la composición sola:\n" + coef.round(3).to_string())
