"""Variables por partida para predecir quién gana (data/analysis/ganador_feats.pkl). Solo data/splits/train.
Esquema (PROJECT_SUMMARY.md §9.S.23):
  H: train antes de la foto 1 (04/09) -> huellas, matchups y sinergias de campeones
  F: foto 1 -> mitad del período       -> ajuste
  V: mitad -> 27/09                    -> validación
Todo lo de cada jugador sale de partidas anteriores a la que se predice."""
import argparse, sys, json, pickle, sqlite3
import numpy as np
import pandas as pd
from scipy import sparse

from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--root", default=str(REPO), help="raíz del repo (lee data/)")
ap.add_argument("--salida", default=str(REPO / "data" / "analysis" / "ganador_feats.pkl"))
args = ap.parse_args()
ROOT, OUT = args.root, args.salida
sys.path.insert(0, ROOT)
from src.features.elo_reverso import cargar_foto, cortes, grupo_de
from src.features.padron import verificar_foto1  # noqa: E402
from src.features.fingerprint import add_rates, residualize, STYLE_FEATURES

LANES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
db = f"file:{ROOT}/data/lol_las.db?mode=ro"
foto1 = cargar_foto(db, "SELECT m.player_id, p.tier, p.rank, p.league_points, p.wins, p.losses "
                        "FROM players p JOIN player_id_map m ON m.puuid = p.puuid")
with sqlite3.connect(db, uri=True) as con:
    t1 = verificar_foto1(con)

df = pd.read_csv(f"{ROOT}/data/splits/train/full_dataset.csv")
df = df[df.lane.isin(LANES)]
df = df[df.groupby(["match_id", "team_id"]).lane.transform("nunique") == 5]
df = df[df.groupby("match_id").lane.transform("size") == 10].copy()
df["s"] = np.where(df.team_id == 100, 1, -1)

# ---------- experiencia con el campeón y línea habitual (partidas anteriores del jugador)
df = df.sort_values(["game_creation", "match_id"])
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
df["x_logn_tot"] = np.log1p(df.n_tot)          # control de cobertura (cuántas partidas suyas tenemos)
CAMP = ["x_logn_champ", "x_estreno", "x_wr_champ", "x_share_champ", "x_share_lane", "x_offrole"]

# ---------- dificultad de Riot (Data Dragon, 1-10) y su cruce con la experiencia
dd = json.load(open(f"{ROOT}/data/ddragon/16.17.1/champion.json", encoding="utf-8"))["data"]
dif = {int(v["key"]): v["info"]["difficulty"] for v in dd.values()}
df["x_dif"] = df.champion_id.map(dif)
print(f"campeones sin dificultad en Data Dragon: {df.x_dif.isna().mean():.2%} de las filas")
df["x_dif"] = df.x_dif.fillna(df.x_dif.mean()) / 10
df["x_dif_estreno"] = df.x_dif * df.x_estreno
df["x_dif_logn"] = df.x_dif * df.x_logn_champ
DIF = ["x_dif", "x_dif_estreno", "x_dif_logn"]

# ---------- LP: foto 1 + paso fijo simétrico (±20 LP, sin usar foto 2 futura del 28/10)
post = df[df.game_creation >= t1]
a = {g: 20.0 for g in ("CHALLENGER", "GRANDMASTER", "MASTER", "DIAMOND")}
b = {g: 20.0 for g in ("CHALLENGER", "GRANDMASTER", "MASTER", "DIAMOND")}
corte = cortes(foto1)
z1 = foto1.z.reindex(post.player_id).to_numpy()
lp = np.full(len(post), np.nan)
estado = {}
for r, (pid, w, z) in enumerate(zip(post.player_id.to_numpy(), post.win.to_numpy(), z1)):
    cur = estado.get(pid, z)
    if cur != cur:
        continue
    lp[r] = cur
    g = grupo_de(cur, corte)
    estado[pid] = cur + a[g] if w == 1 else cur - b[g]
post = post.assign(lp_adelante=lp, lp_foto1=z1,
                   tier1=foto1.tier.reindex(post.player_id).to_numpy())

H = df[df.game_creation < t1]
# ---------- huella con H
Z = residualize(add_rates(H))
HU = Z.groupby("player_id")[STYLE_FEATURES].mean()[Z.groupby("player_id").size() >= 10]

# ---------- matchups y sinergias de campeones, estimados con H, suavizados hacia lo esperado
K = 20
wr_cl = H.groupby(["champion_id", "lane"]).win.mean()

def resid(n, w, e):
    """winrate suavizado hacia lo esperado `e`, menos `e` (0 sin datos)"""
    return (w + K * e) / (n + K) - e

wide = H.pivot_table(index=["match_id", "team_id"], columns="lane", values=["champion_id", "win"], aggfunc="first")
ch = wide["champion_id"].astype(int)
win_t = wide["win"]["TOP"].astype(int)
ch_blue = ch.xs(100, level="team_id")
ch_red = ch.xs(200, level="team_id").reindex(ch_blue.index)
win_blue = win_t.xs(100, level="team_id").reindex(ch_blue.index)
# matchups: (campeón, rival, línea) desde los dos lados
mu_tab = {}
for l in LANES:
    pares = pd.DataFrame({"c": np.r_[ch_blue[l], ch_red[l]], "o": np.r_[ch_red[l], ch_blue[l]],
                          "w": np.r_[win_blue, 1 - win_blue]})
    mu_tab[l] = pares.groupby(["c", "o"]).w.agg(["size", "sum"])
# sinergias: pares de compañeros en el mismo equipo
PARES = [("BOTTOM", "UTILITY"), ("JUNGLE", "MIDDLE"), ("JUNGLE", "TOP")]
win_team = win_t
sy_tab = {}
for l1, l2 in PARES:
    p = pd.DataFrame({"c1": ch[l1].to_numpy(), "c2": ch[l2].to_numpy(), "w": win_team.to_numpy()})
    sy_tab[(l1, l2)] = p.groupby(["c1", "c2"]).w.agg(["size", "sum"])

# ---------- features por partida, F y V
post = post.sort_values(["match_id", "lane"])
mids = post.match_id.unique()
row = pd.Series(np.arange(len(mids)), index=mids)[post.match_id].to_numpy()
lane_i = post.lane.map({l: i for i, l in enumerate(LANES)}).to_numpy()
s = post.s.to_numpy()

def por_linea(X, total=True):
    d = X.shape[1]
    out = np.zeros((len(mids), (6 if total else 5) * d))
    for li in range(5):
        sel = lane_i == li
        np.add.at(out[:, li * d:(li + 1) * d], row[sel], X[sel] * s[sel, None])
    if total:
        np.add.at(out[:, 5 * d:], row, X * s[:, None])
    return out

def por_equipo(col):
    v = post[col].to_numpy()
    k = np.isfinite(v)
    out = {}
    for lado, name in ((1, "az"), (-1, "ro")):
        sel = (s == lado) & k
        sm = np.bincount(row[sel], v[sel], len(mids))
        n = np.bincount(row[sel], minlength=len(mids))
        out[name] = np.where(n > 0, sm / np.maximum(n, 1), np.nan)
        out[name + "_n"] = n / 5
    d = out["az"] - out["ro"]
    return np.c_[np.nan_to_num(d / 100), np.isnan(d), out["az_n"] - out["ro_n"]]

feat = {}
feat["lp"] = np.c_[por_equipo("lp_adelante"), por_linea(np.nan_to_num(post[["lp_adelante"]].to_numpy() / 100), total=False)]
feat["huella"] = por_linea(HU.reindex(post.player_id).fillna(0).to_numpy(), total=False)
feat["experiencia"] = por_linea(post[CAMP].to_numpy())
feat["cobertura"] = por_linea(post[["x_logn_tot"]].to_numpy())
feat["dificultad"] = por_linea(post[DIF].to_numpy())
champs = {c: i for i, c in enumerate(sorted(df.champion_id.unique()))}
feat["onehot"] = sparse.csr_matrix((s, (row, post.champion_id.map(champs))), shape=(len(mids), len(champs)))

# matchups por línea (desde el azul) y sinergias (azul - rojo)
wide_p = post.pivot_table(index="match_id", columns=["team_id", "lane"], values="champion_id", aggfunc="first").reindex(mids)
cb = {l: wide_p[(100, l)].astype(int).to_numpy() for l in LANES}
cr = {l: wide_p[(200, l)].astype(int).to_numpy() for l in LANES}

def wr_exp(c, l):
    return wr_cl.reindex(pd.MultiIndex.from_arrays([c, np.repeat(l, len(c))])).fillna(0.5).to_numpy()

mt_cols, sy_cols = [], []
for l in LANES:
    tab = mu_tab[l].reindex(pd.MultiIndex.from_arrays([cb[l], cr[l]])).fillna(0)
    e = 0.5 + wr_exp(cb[l], l) - wr_exp(cr[l], l)
    mt_cols.append(resid(tab["size"].to_numpy(), tab["sum"].to_numpy(), e))
for l1, l2 in PARES:
    vals = []
    for c in (cb, cr):
        tab = sy_tab[(l1, l2)].reindex(pd.MultiIndex.from_arrays([c[l1], c[l2]])).fillna(0)
        e = wr_exp(c[l1], l1) + wr_exp(c[l2], l2) - 0.5
        vals.append(resid(tab["size"].to_numpy(), tab["sum"].to_numpy(), e))
    sy_cols.append(vals[0] - vals[1])
feat["matchups"] = np.c_[tuple(mt_cols)]
feat["sinergias"] = np.c_[tuple(sy_cols)]

t = post.groupby("match_id").game_creation.first().reindex(mids).to_numpy()
y = post[post.s == 1].groupby("match_id").win.first().reindex(mids).to_numpy()
mitad = np.median(t)
# rango del lobby (foto 1, conocida antes): Diamond si la mayoría de los que tienen foto es Diamond
es_dia = (post.tier1 == "DIAMOND").groupby(post.match_id).mean().reindex(mids)
con_foto = post.tier1.notna().groupby(post.match_id).mean().reindex(mids)
estrato = np.where(es_dia > 0.5, "DIAMOND", "MASTER+")
pickle.dump({"feat": feat, "y": y, "t": t, "fit": t < mitad, "val": t >= mitad, "estrato": estrato,
             "con_foto": con_foto.to_numpy(),
             "filas": post[["match_id", "player_id", "tier1", "win", "n_tot", "n_champ", "x_estreno",
                            "x_offrole", "x_dif", "game_creation"]].assign(val=post.game_creation >= mitad)},
            open(OUT, "wb"))
print("listo:", {k: (v.shape) for k, v in feat.items()}, "| ajuste", (t < mitad).sum(), "| validación", (t >= mitad).sum())
