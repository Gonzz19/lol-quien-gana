"""Dos pruebas de información nueva para quién gana (solo data/splits/train):

1. MARGEN: por cuánto le va bien a cada jugador, no solo si gana. Para cada estadística x
   (ataque, defensa, economía, visión, línea contra el rival directo) se descuenta el campeón
   y la línea con la media PREVIA del campeón en esa línea, pero NO el resultado:
       r = (x - media_previa(campeón, línea)) / desvío(línea)
   y el puntaje del jugador antes de cada partida es su media previa encogida hacia 0,
   S / (N + 5), sobre todas sus partidas anteriores ("acum") y sobre las últimas 20 ("v20").
   También el winrate previo (acumulado y de las últimas 20).
2. DUOS: pares del mismo equipo con >= 3 partidas previas juntos y ninguna enfrentados.
   Variable: (duos azul - duos rojo), cruzada con el tier del lobby (Diamond / Master+).

Se suman a la logística de ganador_red.py (configs A, B, C; misma validación) y se informa
la diferencia pareada de AUC contra la base, por bloque y por familia de estadísticas.
Todo sale de partidas anteriores a la que se predice. Las únicas constantes calculadas con
todo el train son la media y el desvío por línea de cada estadística (escala, sin resultado).
El tier del lobby sale de la foto del padrón: es simétrico entre lados y solo cambia el signo
del efecto del duo."""
import argparse, itertools, pickle, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.features.fingerprint import add_rates
from src.features.margin import lane_diffs, prior_margin, prior_winrate

ap = argparse.ArgumentParser()
ap.add_argument("--feats", default=str(REPO / "data" / "analysis" / "ganador_feats_grande.pkl"))
ap.add_argument("--salida", default=str(REPO / "data" / "analysis" / "ganador_margen_duos.pkl"))
ap.add_argument("--sin-evaluar", action="store_true", help="solo escribe el pickle")
args = ap.parse_args()

LANES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
FAMILIAS = {
    "ataque": ["kills_pm", "assists_pm", "damage_pm", "damage_share_pct", "kill_participation_pct"],
    "defensa": ["deaths_pm", "damage_taken_pm"],
    "economía": ["gold_pm", "cs_pm", "gold_share_pct"],
    "visión": ["vision_pm"],
    "línea": ["d_gold_10", "d_gold_15", "d_xp_15", "d_cs_15"],
}
STATS = [c for f in FAMILIAS.values() for c in f]
K_CL, K_J, VENTANA = 20, 5, 20

t0 = time.time()
df = pd.read_csv(REPO / "data" / "splits" / "train" / "full_dataset.csv")
todas = df[["match_id", "game_creation", "player_id", "team_id"]].copy()     # para los duos
df = df[df.lane.isin(LANES)]
df = df[df.groupby(["match_id", "team_id"]).lane.transform("nunique") == 5]
df = df[df.groupby("match_id").lane.transform("size") == 10].copy()
df = add_rates(df).sort_values(["game_creation", "match_id", "team_id", "lane"]).reset_index(drop=True)
df = lane_diffs(df, ["gold_10", "gold_15", "xp_15", "cs_15"])      # contra el rival directo de línea

# ---------- margen (src/features/margin.py, con tests de no-leakage)
J = pd.concat([prior_margin(df, STATS, K_CL, K_J, VENTANA), prior_winrate(df, VENTANA)], axis=1)
print(f"margen por jugador: {J.shape} ({time.time() - t0:.0f}s)", flush=True)

d = pickle.load(open(args.feats, "rb"))
feat, mids, y, t, val, masks = d["feat"], d["mids"], d["y"], d["t"], d["val"], d["masks"]
row = pd.Series(np.arange(len(mids)), index=mids).reindex(df.match_id).to_numpy()
assert not np.isnan(row).any() and len(set(row)) == len(mids), "las partidas no coinciden con el pickle"
row = row.astype(int)
lane_i = df.lane.map({l: i for i, l in enumerate(LANES)}).to_numpy()
s = np.where(df.team_id == 100, 1.0, -1.0)

def por_linea(X):
    """azul - rojo por línea y total del equipo"""
    k = X.shape[1]
    out = np.zeros((len(mids), 6 * k))
    for li in range(5):
        sel = lane_i == li
        np.add.at(out[:, li * k:(li + 1) * k], row[sel], X[sel] * s[sel, None])
    np.add.at(out[:, 5 * k:], row, X * s[:, None])
    return out

bloque = lambda nombres: por_linea(J[nombres].to_numpy())
NUEVO = {"winrate previo": bloque(["wr_acum", "wr_v20"]),
         "margen acum": bloque([f"acum_{c}" for c in STATS]),
         "margen v20": bloque([f"v20_{c}" for c in STATS])}
for fam, cs in FAMILIAS.items():
    NUEVO[f"fam {fam}"] = bloque([f"{p}_{c}" for p in ("acum", "v20") for c in cs])

# ---------- duos (con todas las partidas de 10 jugadores del train como historia)
t0 = time.time()
todas = todas[todas.groupby("match_id").player_id.transform("size") == 10]
todas = todas.sort_values(["game_creation", "match_id", "team_id", "player_id"])
pid = todas.player_id.astype("category").cat.codes.to_numpy().reshape(-1, 10)
tm = todas.team_id.to_numpy().reshape(-1, 10)
m_duo = todas.match_id.to_numpy()[::10]
assert (tm[:, :5] == 100).all() and (tm[:, 5:] == 200).all()
n = len(pid)
partes = []
for i, j in itertools.combinations(range(10), 2):
    partes.append(pd.DataFrame({"m": np.arange(n), "a": np.minimum(pid[:, i], pid[:, j]),
                                "b": np.maximum(pid[:, i], pid[:, j]),
                                "lado": 1 if j < 5 else (-1 if i >= 5 else 0)}))
P = pd.concat(partes, ignore_index=True).sort_values("m", kind="stable")
P["junto"] = (P.lado != 0).astype(int)
g = P.groupby(["a", "b"], sort=False)
P["prev_junto"] = g.junto.cumsum() - P.junto
P["prev_contra"] = g.cumcount() - P.prev_junto
P["duo"] = (P.lado != 0) & (P.prev_junto >= 3) & (P.prev_contra == 0)
duo = (P.duo * P.lado).groupby(P.m).sum()                       # duos azul - duos rojo
hay = P.duo.groupby(P.m).sum()
duo = pd.Series(duo.to_numpy(), index=m_duo).reindex(mids).fillna(0).to_numpy()
hay = pd.Series(hay.to_numpy(), index=m_duo).reindex(mids).fillna(0).to_numpy()
tier = pd.read_csv(REPO / "data" / "splits" / "match_split.csv", usecols=["match_id", "tier"]) \
         .set_index("match_id").tier.reindex(mids).to_numpy()
dia = tier == "DIAMOND"
NUEVO["duos"] = np.c_[duo * dia, duo * ~dia, np.sign(duo) * dia, np.sign(duo) * ~dia]
print(f"duos: {(hay > 0).mean():.1%} de las partidas con algún duo ({time.time() - t0:.0f}s)", flush=True)
# por jugador (partida, puesto = 5 * rojo + línea, variable), para la red con codificador compartido
jug_margen = np.zeros((len(mids), 10, J.shape[1]), dtype=np.float32)
jug_margen[row, 5 * (s == -1) + lane_i] = J.to_numpy()
pickle.dump({"mids": mids, "nuevo": NUEVO, "jug": jug_margen, "columnas": list(J.columns)},
            open(args.salida, "wb"), protocol=4)
if args.sin_evaluar:
    raise SystemExit(0)

# ---------- evaluación
yv = y[val]
B = np.random.default_rng(0).integers(0, val.sum(), (200, val.sum()))
COMUN = ["onehot", "experiencia", "cobertura", "dificultad", "matchups", "sinergias"]
HU = {"A": "huella_foto", "B": "huella_40", "C": None}

def ajustar(X, fit):
    sc = StandardScaler(with_mean=False).fit(X[fit])
    return LogisticRegression(C=0.05, max_iter=5000).fit(sc.transform(X[fit]), y[fit]).predict_proba(sc.transform(X[val]))[:, 1]

def informe(nombre, p, ref=None):
    extra = ""
    if ref is not None:
        dif = np.array([roc_auc_score(yv[b], p[b]) - roc_auc_score(yv[b], ref[b]) for b in B])
        extra = f"  vs base {roc_auc_score(yv, p) - roc_auc_score(yv, ref):+.4f} [{np.quantile(dif, .025):+.4f};{np.quantile(dif, .975):+.4f}]"
    print(f"  {nombre:38s} acc {accuracy_score(yv, p > .5):.3f}  AUC {roc_auc_score(yv, p):.4f}  logloss {log_loss(yv, p):.4f}{extra}", flush=True)

for cfg in "ABC":
    fit = masks[cfg]
    Xb = np.hstack([feat[k] for k in COMUN + ([HU[cfg]] if HU[cfg] else [])])
    print(f"\n=== Config {cfg} (ajuste {fit.sum()}, huella {HU[cfg] or 'no'}) ===")
    base = ajustar(Xb, fit)
    informe("base", base)
    solos = ["duos", "winrate previo", "margen acum", "margen v20"] + ([f"fam {f}" for f in FAMILIAS] if cfg == "A" else [])
    for k in solos:
        informe(f"base + {k}", ajustar(np.hstack([Xb, NUEVO[k]]), fit), base)
    margen = [NUEVO["winrate previo"], NUEVO["margen acum"], NUEVO["margen v20"]]
    informe("base + winrate + margen (todo)", ajustar(np.hstack([Xb] + margen), fit), base)
    informe("base + winrate + margen + duos", ajustar(np.hstack([Xb] + margen + [NUEVO["duos"]]), fit), base)
    informe("solo winrate + margen + duos", ajustar(np.hstack(margen + [NUEVO["duos"]]), fit), base)
    if HU[cfg]:
        sin = np.hstack([feat[k] for k in COMUN] + margen + [NUEVO["duos"]])
        informe("todo lo nuevo, SIN huella", ajustar(sin, fit), base)
