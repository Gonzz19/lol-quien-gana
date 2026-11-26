"""
Forma del día como margen: ¿cuánto mejor o peor que lo esperado viene jugando HOY el jugador,
antes de esta partida? ¿Predice quién gana más allá de su nivel general? (PROJECT_SUMMARY.md §9.S.45)

La racha de victorias del día ya se había descartado (§9.S.24, punto 3): los que vienen ganando
son los que ganan siempre. Acá se mide con el margen (§9.S.24): para cada estadística, el
residuo r = (x − media previa del campeón en la línea) / desvío de la línea, y

    hoy_<stat> = (S_día + k·m) / (N_día + k) − m

con S_día y N_día la suma y la cantidad de r de las partidas previas del jugador en el mismo
día de juego (de 06:00 a 06:00, hora de Argentina), m su margen acumulado previo y k = 3. Vale 0
en la primera partida del día: mide el desvío de hoy respecto de su nivel, no el nivel.

Sin leakage por construcción (`prior_margin_by`: toda suma es cumsum − valor propio). Solo
`data/splits/train`, la misma partición F/V y la caché de `embeddings_partidas.py`.

    python historial/ganador_primeros/ganador_forma_dia.py            # ~6 min
    python historial/ganador_primeros/ganador_forma_dia.py --corte 8  # día de 08:00 a 08:00
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.features.draft import draft_matrix_by_lane  # noqa: E402
from src.features.fingerprint import add_rates  # noqa: E402
from src.features.margin import lane_diffs, prior_margin_by, _residuo, lane_scale, _previo  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--corte", type=int, default=6, help="hora (ART) en que empieza el día de juego")
ap.add_argument("--k", type=float, default=3.0, help="encogimiento hacia el nivel del jugador")
ap.add_argument("--solo-cache", action="store_true",
                help="solo escribir la caché de la forma del día (la usa validacion_bloques.py)")
args = ap.parse_args()

LANES = ["BOTTOM", "JUNGLE", "MIDDLE", "TOP", "UTILITY"]
STATS = ["kills_pm", "assists_pm", "damage_pm", "damage_share_pct", "kill_participation_pct",
         "deaths_pm", "damage_taken_pm", "gold_pm", "cs_pm", "gold_share_pct", "vision_pm",
         "d_gold_10", "d_gold_15", "d_xp_15", "d_cs_15",
         # agregadas para la forma del día
         "d_xp_10", "d_cs_10", "d_damage_15", "early_aggression_10"]
USE = ["match_id", "game_creation", "game_duration", "game_version", "player_id", "team_id", "win",
       "champion_id", "lane", "kills", "deaths", "assists", "kill_participation_pct",
       "damage_share_pct", "gold_share_pct", "total_damage_dealt_to_champions", "total_damage_taken",
       "gold_earned", "total_minions_killed", "neutral_minions_killed", "vision_score",
       "early_aggression_10", "gold_10", "xp_10", "cs_10", "gold_15", "xp_15", "cs_15", "damage_15"]

# ---- filas (mismo filtro y orden que embeddings_partidas.py)
df = pd.read_csv(REPO / "data" / "splits" / "train" / "full_dataset.csv", usecols=USE)
df = df[df.lane.isin(LANES)]
df = df[df.groupby(["match_id", "team_id"]).lane.transform("nunique") == 5]
df = df[df.groupby("match_id").lane.transform("size") == 10]
df = df[df.groupby("match_id").win.transform("sum") == 5].copy()
df = add_rates(df).sort_values(["game_creation", "match_id", "team_id", "lane"]).reset_index(drop=True)
df = lane_diffs(df, ["gold_10", "gold_15", "xp_15", "cs_15", "xp_10", "cs_10", "damage_15"])

Z = np.load(REPO / "data" / "analysis" / "embeddings_partidas" / "variables.npz", allow_pickle=True)
assert (df.match_id.to_numpy()[::10] == Z["mids"]).all(), "las filas no coinciden con la caché"
Xj, cols, y, val, champ = Z["Xj"], list(Z["cols"]), Z["y"], Z["val"], Z["champ"]
NM, fit, yv = len(y), ~Z["val"], y[Z["val"]]

# ---- día de juego y forma del día
DIA_MS = 86_400_000
df["dia"] = (df.game_creation - (3 + args.corte) * 3_600_000) // DIA_MS     # ART = UTC − 3
escala = lane_scale(df, STATS)
H = prior_margin_by(df, STATS, ["dia"], "hoy", k_g=args.k, escala=escala)
pj = [df.player_id, df.dia]
n_dia = df.groupby(["player_id", "dia"]).cumcount()
# victorias de hoy, mismo esquema (comparación con la racha de §9.S.24)
wv = df.win.astype(float)
w_acum = (_previo(wv, df.player_id) + 5 * 0.5) / (df.groupby("player_id").cumcount() + 5)
H["hoy_victorias"] = (_previo(wv, pj) + args.k * w_acum) / (n_dia + args.k) - w_acum
# la partida anterior del mismo día (tilt): margen medio de esa sola partida, sin encoger
R = pd.DataFrame({c: _residuo(df, c, 20, escala)[0] for c in STATS})
prev = R.groupby([df.player_id, df.dia]).shift(1).fillna(0)
H = pd.concat([H, prev.add_prefix("ultima_")], axis=1)
H["n_dia"] = np.log1p(n_dia)
hc = list(H.columns)
print(f"filas {len(df):,}; con alguna partida previa en el día: {(n_dia > 0).mean():.1%}")

Hj = H.to_numpy(np.float32).reshape(NM, 2, 5, -1)
CACHE_H = REPO / "data" / "analysis" / "embeddings_partidas" / f"forma_dia_corte{args.corte}.npz"
np.savez_compressed(CACHE_H, Hj=Hj, hc=np.array(hc), mids=Z["mids"])
print(f"caché de la forma del día: {CACHE_H.relative_to(REPO)}")
if args.solo_cache:
    sys.exit(0)


def dif(cols_sel):
    j = [hc.index(c) for c in cols_sel]
    A = (Hj[:, 0][:, :, j] - Hj[:, 1][:, :, j]).reshape(NM, -1)
    return (A - A[fit].mean(0)) / (A[fit].std(0) + 1e-6)


# ---- base: el híbrido del EDA (draft + 85 variables por línea)
Xd = (Xj[:, 0] - Xj[:, 1]).reshape(NM, -1)
Xd = (Xd - Xd[fit].mean(0)) / (Xd[fit].std(0) + 1e-6)
DRAFT = draft_matrix_by_lane(champ, fit=fit)

BOOT = np.random.default_rng(1).integers(0, val.sum(), size=(400, val.sum()))


def ajustar(extra=None, base=None):
    base = Xd if base is None else base
    X = sparse.hstack([DRAFT, sparse.csr_matrix(base.astype(np.float32))] +
                      ([sparse.csr_matrix(extra.astype(np.float32))] if extra is not None else [])).tocsr()
    return LogisticRegression(C=0.05, max_iter=3000).fit(X[fit], y[fit]).predict_proba(X[val])[:, 1]


def comparar(p1, p0):
    b = np.array([roc_auc_score(yv[s], p1[s]) - roc_auc_score(yv[s], p0[s]) for s in BOOT])
    return roc_auc_score(yv, p1) - roc_auc_score(yv, p0), *np.percentile(b, [2.5, 97.5])


hoy_m = [f"hoy_{c}" for c in STATS]
ult = [f"ultima_{c}" for c in STATS]
p0 = ajustar()
print(f"\nbase (híbrido): AUC {roc_auc_score(yv, p0):.4f}  log loss {log_loss(yv, p0):.5f}")
VAR = {
    "victorias de hoy (racha, §9.S.24)": ["hoy_victorias", "n_dia"],
    "margen de hoy, 19 estadísticas": hoy_m + ["n_dia"],
    "margen de la partida anterior del día": ult + ["n_dia"],
    "margen de hoy + anterior + victorias": hoy_m + ult + ["hoy_victorias", "n_dia"],
}
filas = []
for nom, cs in VAR.items():
    p = ajustar(dif(cs))
    d, lo, hi = comparar(p, p0)
    filas.append((nom, len(cs) * 5, roc_auc_score(yv, p), d, lo, hi, log_loss(yv, p) - log_loss(yv, p0)))
    print(f"{nom:42s} cols {len(cs) * 5:4d}  AUC {filas[-1][2]:.4f}  Δ {d:+.4f} [{lo:+.4f}; {hi:+.4f}]  Δlogloss {filas[-1][6]:+.6f}")

# ---- sin las últimas 20: ¿la forma de hoy reemplaza a v20 cuando no está?
d_var = len(cols)
n_prev = np.expm1(Xj[:, :, :, cols.index("logn_tot")]).reshape(NM, 10)
print(f"\nCobertura de v20 (filas): sin partidas previas {(n_prev == 0).mean():.1%}, "
      f"1-19 previas {((n_prev > 0) & (n_prev < 20)).mean():.1%}, ≥ 20 previas {(n_prev >= 20).mean():.1%}")
es_v20 = np.array([c.startswith("v20_") or c == "wr_v20" for c in cols])
Xs = Xd[:, ~np.tile(es_v20, 5)]
wr_hoy = ["hoy_victorias", "n_dia"]
CONF = {
    "con v20 (base)": (None, None),
    "sin v20": (None, Xs),
    "sin v20 + margen de hoy": (dif(hoy_m + ["n_dia"]), Xs),
    "sin v20 + margen de hoy + WR de hoy": (dif(hoy_m + wr_hoy), Xs),
    "con v20 + margen de hoy + WR de hoy": (dif(hoy_m + wr_hoy), None),
}
PC = {nom: ajustar(e, b) for nom, (e, b) in CONF.items()}
# subgrupos por historial del lobby (mínimo de partidas previas entre los 10)
mn = n_prev[val].min(1)
GRUPOS = {"todas": np.ones(val.sum(), bool), "lobby con alguien < 20 previas": mn < 20,
          "los 10 con ≥ 20 previas": mn >= 20}
for g, mk in GRUPOS.items():
    print(f"\n[{g}] partidas {mk.sum():,}")
    bo = np.random.default_rng(2).integers(0, mk.sum(), size=(400, mk.sum()))
    ref = PC["sin v20"][mk]
    for nom, p in PC.items():
        pm = p[mk]
        b = np.array([roc_auc_score(yv[mk][s], pm[s]) - roc_auc_score(yv[mk][s], ref[s]) for s in bo])
        print(f"  {nom:40s} AUC {roc_auc_score(yv[mk], pm):.4f}   vs sin v20 {np.mean(b):+.4f} "
              f"[{np.percentile(b, 2.5):+.4f}; {np.percentile(b, 97.5):+.4f}]")

# ---- descriptivo a nivel jugador: ¿el margen de hoy predice su resultado más allá de su nivel?
# compuesto = media de los r de habilidad con signo (más es mejor); solo filas con ≥ 2 previas hoy
HAB = ["kills_pm", "assists_pm", "damage_pm", "kill_participation_pct", "gold_pm", "cs_pm",
       "d_gold_15", "d_xp_15", "d_cs_15"]
acum = _previo(R[HAB].mean(1) - R["deaths_pm"] / len(HAB), df.player_id) / \
       (df.groupby("player_id").cumcount() + 5)
hoy = H[[f"hoy_{c}" for c in HAB]].mean(1) - H["hoy_deaths_pm"] / len(HAB)
sel = (n_dia >= 2).to_numpy() & np.repeat(fit, 10)
q = pd.qcut(hoy[sel], 5, labels=False)
tab = pd.DataFrame({"q": q, "win": df.win[sel], "acum": acum[sel]}).groupby("q").agg(
    filas=("win", "size"), winrate=("win", "mean"), nivel_acum=("acum", "mean"))
print("\nJugadores con ≥ 2 partidas previas hoy (ajuste): quintil del margen de hoy compuesto")
print(tab.round(4).to_string())
comp = R[HAB].mean(1) - R["deaths_pm"] / len(HAB)
S = _previo(comp, df.player_id)
S20 = S - S.groupby(df.player_id).shift(20).fillna(0)
N = df.groupby("player_id").cumcount()
v20 = S20 / (np.minimum(N, 20) + 5)
for nom, ctrl in (("nivel acumulado", [acum]), ("nivel acumulado + últimas 20", [acum, v20])):
    Xc = np.column_stack([c[sel] for c in ctrl] + [hoy[sel]])
    m = LogisticRegression(C=1e6, max_iter=1000).fit(Xc / Xc.std(0), df.win[sel])
    print(f"logística por jugador, control {nom} (coef. por 1 desvío): " +
          "  ".join(f"{c:+.4f}" for c in m.coef_[0]) + "   (el último es hoy)")
