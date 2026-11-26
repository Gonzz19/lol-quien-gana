"""Variables por partida para TODO el train (sin LP), para ajustar con más partidas.
Matchups y sinergias pasan a acumulados (solo partidas anteriores a cada una).
Dos huellas: con lo anterior a la foto 1 (config A) y con el 40 % más viejo (config B).
Solo data/splits/train; la validación es la misma de siempre (mitad 04/09 -> 27/09)."""
import argparse, sys, json, pickle, sqlite3
import numpy as np
import pandas as pd
from scipy import sparse

from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--root", default=str(REPO), help="raíz del repo (lee data/)")
ap.add_argument("--salida", default=str(REPO / "data" / "analysis" / "ganador_feats_grande.pkl"))
args = ap.parse_args()
ROOT, OUT = args.root, args.salida
sys.path.insert(0, ROOT)
from src.features.fingerprint import add_rates, residualize, STYLE_FEATURES
from src.features.padron import verificar_foto1  # noqa: E402

LANES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
with sqlite3.connect(f"file:{ROOT}/data/lol_las.db?mode=ro", uri=True) as con:
    t1 = verificar_foto1(con)

df = pd.read_csv(f"{ROOT}/data/splits/train/full_dataset.csv")
df = df[df.lane.isin(LANES)]
df = df[df.groupby(["match_id", "team_id"]).lane.transform("nunique") == 5]
df = df[df.groupby("match_id").lane.transform("size") == 10].copy()
df["s"] = np.where(df.team_id == 100, 1, -1)
df = df.sort_values(["game_creation", "match_id", "team_id", "lane"]).reset_index(drop=True)

# ---------- experiencia, cobertura, dificultad (igual que ganador_features.py)
gp = df.groupby("player_id")
df["n_tot"] = gp.cumcount()
gc = df.groupby(["player_id", "champion_id"])
df["n_champ"] = gc.cumcount()
df["w_champ"] = gc.win.cumsum() - df.win
df["w_tot"] = gp.win.cumsum() - df.win
df["n_lane"] = df.groupby(["player_id", "lane"]).cumcount()
df["x_logn_champ"] = np.log1p(df.n_champ)
df["x_estreno"] = (df.n_champ == 0).astype(float)
df["x_wr_champ"] = (df.w_champ + 5 * (df.w_tot + 5) / (df.n_tot + 10)) / (df.n_champ + 5) \
                   - (df.w_tot + 5) / (df.n_tot + 10)
df["x_share_champ"] = df.n_champ / (df.n_tot + 1)
df["x_share_lane"] = (df.n_lane + 1) / (df.n_tot + 2)
df["x_offrole"] = ((df.n_tot >= 10) & (df.n_lane / df.n_tot.clip(lower=1) < 0.2)).astype(float)
df["x_logn_tot"] = np.log1p(df.n_tot)
CAMP = ["x_logn_champ", "x_estreno", "x_wr_champ", "x_share_champ", "x_share_lane", "x_offrole"]
dd = json.load(open(f"{ROOT}/data/ddragon/16.17.1/champion.json", encoding="utf-8"))["data"]
df["x_dif"] = df.champion_id.map({int(v["key"]): v["info"]["difficulty"] for v in dd.values()})
df["x_dif"] = df.x_dif.fillna(df.x_dif.mean()) / 10
df["x_dif_estreno"] = df.x_dif * df.x_estreno
df["x_dif_logn"] = df.x_dif * df.x_logn_champ
DIF = ["x_dif", "x_dif_estreno", "x_dif_logn"]

# ---------- matchups y sinergias acumulados
K = 20
cl = df.groupby(["champion_id", "lane"])          # winrate previo del campeón en la línea
df["e_cl"] = ((cl.win.cumsum() - df.win) + 10) / (cl.cumcount() + 20)
mt = df.groupby("match_id").game_creation.first()
wide = df.pivot_table(index="match_id", columns=["team_id", "lane"], values=["champion_id", "e_cl"], aggfunc="first")
wide = wide.loc[mt.sort_values().index]
yb = df[df.s == 1].groupby("match_id").win.first().reindex(wide.index).to_numpy()

def previo(keys, w):
    """cuántas veces apareció cada clave antes y cuántas ganó (sin la fila actual)"""
    k = pd.DataFrame(keys).astype(int)
    g = k.assign(w=w).groupby(list(k.columns), sort=False).w
    return g.cumcount().to_numpy(), (g.cumsum() - w).to_numpy()

mt_cols = []
n = len(wide)
for l in LANES:
    cb, cr = wide[("champion_id", 100, l)].to_numpy(), wide[("champion_id", 200, l)].to_numpy()
    eb, er = wide[("e_cl", 100, l)].to_numpy(), wide[("e_cl", 200, l)].to_numpy()
    # dos perspectivas por partida, intercaladas en orden temporal: (c, o) y (o, c) son claves distintas
    c = np.r_[cb, cr]; o = np.r_[cr, cb]; w = np.r_[yb, 1 - yb]
    orden = np.argsort(np.r_[np.arange(n), np.arange(n)], kind="stable")
    cnt, wins = np.empty(2 * n), np.empty(2 * n)
    cnt[orden], wins[orden] = previo({"c": c[orden], "o": o[orden]}, w[orden])
    e = 0.5 + eb - er
    r = (wins[:n] + K * e) / (cnt[:n] + K) - e
    mt_cols.append(np.where(cb == cr, 0.0, r))
PARES = [("BOTTOM", "UTILITY"), ("JUNGLE", "MIDDLE"), ("JUNGLE", "TOP")]
sy_cols = []
for l1, l2 in PARES:
    vals = []
    for team, w in ((100, yb), (200, 1 - yb)):
        c1, c2 = wide[("champion_id", team, l1)].to_numpy(), wide[("champion_id", team, l2)].to_numpy()
        cnt, wins = previo({"a": c1, "b": c2}, w)
        e = wide[("e_cl", team, l1)].to_numpy() + wide[("e_cl", team, l2)].to_numpy() - 0.5
        vals.append((wins + K * e) / (cnt + K) - e)
    sy_cols.append(vals[0] - vals[1])
mids = wide.index.to_numpy()

# ---------- por partida
df = df.sort_values(["match_id", "lane"]).reset_index(drop=True)
row = pd.Series(np.arange(len(mids)), index=mids)[df.match_id].to_numpy()
lane_i = df.lane.map({l: i for i, l in enumerate(LANES)}).to_numpy()
s = df.s.to_numpy()

def por_linea(X, total=True):
    d = X.shape[1]
    out = np.zeros((len(mids), (6 if total else 5) * d))
    for li in range(5):
        sel = lane_i == li
        np.add.at(out[:, li * d:(li + 1) * d], row[sel], X[sel] * s[sel, None])
    if total:
        np.add.at(out[:, 5 * d:], row, X * s[:, None])
    return out

t = mt.reindex(mids).to_numpy()
q40 = np.quantile(t, 0.4)
feat = {"experiencia": por_linea(df[CAMP].to_numpy()), "cobertura": por_linea(df[["x_logn_tot"]].to_numpy()),
        "dificultad": por_linea(df[DIF].to_numpy()),
        "matchups": np.c_[tuple(mt_cols)], "sinergias": np.c_[tuple(sy_cols)]}
champs = {c: i for i, c in enumerate(sorted(df.champion_id.unique()))}
feat["onehot"] = sparse.csr_matrix((s, (row, df.champion_id.map(champs))), shape=(len(mids), len(champs))).toarray()
# por jugador, para la red con codificador compartido (ganador_red_jugador.py):
# arreglos (partida, puesto, ...) con puesto = 5 * (lado rojo) + línea
puesto = 5 * (s == -1) + lane_i

def por_jugador(X, dtype=np.float32):
    out = np.zeros((len(mids), 10) + X.shape[1:], dtype=dtype)
    out[row, puesto] = X
    return out

jug = {"num": por_jugador(df[CAMP + ["x_logn_tot"] + DIF].to_numpy()),
       "campeon": por_jugador(df.champion_id.map(champs).to_numpy(), np.int16),
       "n_campeones": len(champs)}
for nombre, hasta in (("huella_foto", t1), ("huella_40", q40)):
    Z = residualize(add_rates(df[df.game_creation < hasta]))
    HU = Z.groupby("player_id")[STYLE_FEATURES].mean()[Z.groupby("player_id").size() >= 10]
    hu = HU.reindex(df.player_id)
    feat[nombre] = por_linea(hu.fillna(0).to_numpy(), total=False)
    jug[nombre] = por_jugador(np.c_[hu.fillna(0).to_numpy(), hu.notna().all(axis=1).to_numpy()])

y = df[df.s == 1].groupby("match_id").win.first().reindex(mids).to_numpy()
mitad = np.median(t[t >= t1])
val = t >= mitad
masks = {"A": (t >= t1) & ~val, "B": (t >= q40) & ~val, "C": ~val}
pickle.dump({"feat": feat, "jug": jug, "mids": mids, "y": y, "t": t, "val": val, "masks": masks}, open(OUT, "wb"), protocol=4)
print("listo:", {k: v.shape for k, v in feat.items()})
print("validación", val.sum(), "| ajuste:", {k: int(m.sum()) for k, m in masks.items()},
      f"| corte 40 %: {pd.to_datetime(q40, unit='ms'):%d/%m}")
