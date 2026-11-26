"""Techo del margen: ¿cuánto ganaría la logística si conociera MÁS historial de cada jugador? (§9.S.38)

Experimento de cota superior, tramposo a propósito: NO es un modelo usable. Para cada jugador se
calcula el mismo margen y winrate de §9.S.24 de tres formas, todas sin la partida que se predice:

    previo   sumas de las partidas ANTERIORES          S = cumsum - x            (lo que usa el modelo)
    futuro   sumas de las partidas POSTERIORES         S = total - cumsum
    oráculo  todas las demás partidas del train (LOO)  S = total - x

La media del campeón en la línea sigue la misma regla (previa en "previo", LOO en "futuro" y
"oráculo"). Se arma la misma diferencia azul - rojo por línea y total, se ajusta la misma logística
(C = 0,05, estandarizada) y se compara AUC en V. Lectura:
  * oráculo ~ previo  -> el historial ya alcanza: lo que falta es ruido propio de la partida.
  * oráculo >> previo -> falta historial; si la ganancia se concentra en lobbies con poca cobertura,
                         completar el historial con el crawler es el camino.
  * futuro ~ previo con N parecido -> lo que importa es la cantidad, no la dirección temporal.
Liviano a propósito (sin campeones, matchups ni red) para correr en ~3 GB de RAM: compara variantes
entre sí, no reemplaza al modelo de 2. Solo data/splits/train.

Uso (cada paso < 3 min; los puntajes quedan en --cache, ~0,8 GB):
    python historial/ganador_primeros/ganador_oraculo.py --modos previo futuro oráculo
    python historial/ganador_primeros/ganador_oraculo.py --modos previo_v20 futuro_v20 cercano_v20
    python historial/ganador_primeros/ganador_oraculo.py            # evalúa con lo que haya en caché"""
import argparse, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.features.fingerprint import add_rates
from src.features.margin import lane_diffs, lane_scale

ap = argparse.ArgumentParser()
ap.add_argument("--desde", default="2026-09-04", help="inicio del ajuste")
ap.add_argument("--corte", default="2026-09-16", help="inicio de V")
ap.add_argument("--cache", default=str(REPO / "data" / "analysis" / "ganador_oraculo"),
                help="carpeta donde guardar/leer los puntajes por modo (.npz)")
ap.add_argument("--modos", nargs="*", default=None, help="solo calcula estos modos y sale")
args = ap.parse_args()
CACHE = Path(args.cache); CACHE.mkdir(parents=True, exist_ok=True)

LANES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
STATS = ["kills_pm", "assists_pm", "damage_pm", "damage_share_pct", "kill_participation_pct",
         "deaths_pm", "damage_taken_pm", "gold_pm", "cs_pm", "gold_share_pct", "vision_pm",
         "d_gold_10", "d_gold_15", "d_xp_15", "d_cs_15"]
K_CL, K_J, K_WR = 20, 5, 10
COLS = ["match_id", "game_creation", "game_duration", "player_id", "team_id", "win", "champion_id", "lane",
        "kills", "deaths", "assists", "kill_participation_pct", "damage_share_pct", "gold_share_pct",
        "total_damage_dealt_to_champions", "total_damage_taken", "gold_earned", "total_minions_killed",
        "neutral_minions_killed", "vision_score", "gold_10", "gold_15", "xp_15", "cs_15"]

t0 = time.time()
df = pd.read_csv(REPO / "data" / "splits" / "train" / "full_dataset.csv", usecols=COLS)
df = df[df.lane.isin(LANES)]
df = df[df.groupby(["match_id", "team_id"]).lane.transform("nunique") == 5]
df = df[df.groupby("match_id").lane.transform("size") == 10]
df = add_rates(df).sort_values(["game_creation", "match_id", "team_id", "lane"]).reset_index(drop=True)
df = lane_diffs(df, ["gold_10", "gold_15", "xp_15", "cs_15"])
for c in ["kills", "deaths", "assists", "total_damage_dealt_to_champions", "total_damage_taken", "gold_earned",
          "total_minions_killed", "neutral_minions_killed", "vision_score", "gold_10", "gold_15", "xp_15", "cs_15"]:
    del df[c]
print(f"{len(df) // 10:,} partidas completas ({time.time() - t0:.0f}s)", flush=True)

pj, cl = df.player_id, [df.champion_id, df.lane]
uno = pd.Series(1.0, index=df.index)


V = 20          # ventana de los modos "cercanos"


def sumas(s, claves, modo):
    """Suma de las otras filas del grupo según el modo (nunca incluye la propia).

    previo / futuro / oráculo: todas las anteriores / posteriores / las demás.
    previo_v20 / futuro_v20: las 20 anteriores / las 20 posteriores más cercanas.
    cercano_v20: las 10 anteriores + las 10 posteriores más cercanas."""
    g = s.groupby(claves, sort=False)
    cum = g.cumsum()
    tot = g.transform("sum")
    if modo == "previo": return cum - s
    if modo == "futuro": return tot - cum
    if modo == "oráculo": return tot - s
    gc = cum.groupby(claves, sort=False)
    atras = lambda k: (cum - s) - (cum - s).groupby(claves, sort=False).shift(k).fillna(0)
    adelante = lambda k: gc.shift(-k).fillna(tot) - cum
    if modo == "previo_v20": return atras(V)
    if modo == "futuro_v20": return adelante(V)
    if modo == "cercano_v20": return atras(V // 2) + adelante(V // 2)
    raise ValueError(modo)


escala = lane_scale(df, STATS)
MODOS = ["previo", "futuro", "oráculo", "previo_v20", "futuro_v20", "cercano_v20"]
NOMBRES = STATS + ["wr", "logn"]
J = {}
for m in (args.modos or MODOS):
    f = CACHE / f"{m}.npz"
    if f.exists() and not args.modos:
        J[m] = dict(np.load(f)); continue
    J[m] = {}
    for c in STATS:
        x, v = df[c], df[c].notna().astype(float)
        mu, sd = df.lane.map(escala[c][0]), df.lane.map(escala[c][1])
        media_cl = (sumas(x.fillna(0), cl, m) + K_CL * mu) / (sumas(v, cl, m) + K_CL)
        r = ((x - media_cl) / sd).fillna(0)
        J[m][c] = (sumas(r, pj, m) / (sumas(v, pj, m) + K_J)).to_numpy()
    w = df.win.astype(float)
    J[m]["wr"] = ((sumas(w, pj, m) + K_WR / 2) / (sumas(uno, pj, m) + K_WR) - 0.5).to_numpy()
    J[m]["logn"] = np.log1p(sumas(uno, pj, m)).to_numpy()
    np.savez(f, **{k: v.astype(np.float32) for k, v in J[m].items()})
    print(f"  {m}: listo ({time.time() - t0:.0f}s)", flush=True)
if args.modos:
    raise SystemExit(0)

# ---------- matriz por partida: azul - rojo por línea (5 bloques) y total del equipo
n = len(df) // 10
mid = df.match_id.to_numpy()[::10]
assert (df.match_id.to_numpy().reshape(n, 10) == mid[:, None]).all()
s = np.where(df.team_id.to_numpy() == 100, 1.0, -1.0).reshape(n, 10)
lane_i = df.lane.map({l: i for i, l in enumerate(LANES)}).to_numpy().reshape(n, 10)
y = df.win.to_numpy().reshape(n, 10)[s == 1].reshape(n, 5)[:, 0].astype(int)
t = pd.to_datetime(df.game_creation.to_numpy()[::10], unit="ms")
fit = (t >= pd.Timestamp(args.desde)) & (t < pd.Timestamp(args.corte))
val = t >= pd.Timestamp(args.corte)
SETS = {"completo": NOMBRES, "sin conteo": STATS + ["wr"], "solo margen": STATS}


def matriz(m, cols=NOMBRES):
    A = np.stack([J[m][k] for k in cols], axis=1).reshape(n, 10, -1) * s[..., None]
    out = [np.where((lane_i == li)[..., None], A, 0).sum(1) for li in range(5)]
    return np.hstack(out + [A.sum(1)])


Np = J["previo"]["logn"].reshape(n, 10)                 # partidas previas (log1p) de cada jugador
pocas = (np.expm1(Np) < 10).sum(1)                       # jugadores del lobby con < 10 previas
Nf = np.expm1(J["futuro"]["logn"].reshape(n, 10)).mean(1)
print(f"ajuste {fit.sum():,} | V {val.sum():,} partidas | previas medias en V {np.expm1(Np[val]).mean():.0f}, "
      f"posteriores medias {Nf[val].mean():.0f}", flush=True)

yv = y[val]


def ajustar(X):
    sc = StandardScaler(with_mean=False).fit(X[fit])
    return LogisticRegression(C=0.05, max_iter=5000).fit(sc.transform(X[fit]), y[fit]).predict_proba(sc.transform(X[val]))[:, 1]


def ic(p, ref, sel=None):
    idx = np.arange(len(yv)) if sel is None else np.where(sel)[0]
    d = [roc_auc_score(yv[ii], p[ii]) - roc_auc_score(yv[ii], ref[ii])
         for ii in idx[np.random.default_rng(1).integers(0, len(idx), (200, len(idx)))]]
    return np.quantile(d, [.025, .975])


P = {}
for nombre_set, cols in SETS.items():
    print(f"\n=== V completo, variables: {nombre_set} ===")
    for m in MODOS:
        P[(m, nombre_set)] = p = ajustar(matriz(m, cols))
        ref = P[("previo", nombre_set)]
        extra = "" if m == "previo" else "  vs previo {:+.4f} [{:+.4f}; {:+.4f}]".format(
            roc_auc_score(yv, p) - roc_auc_score(yv, ref), *ic(p, ref))
        print(f"  {m:12s} AUC {roc_auc_score(yv, p):.4f}  acc {accuracy_score(yv, p > .5):.4f}  "
              f"logloss {log_loss(yv, p):.4f}{extra}", flush=True)

print("\n=== por cobertura del lobby (jugadores con < 10 partidas previas), solo margen ===")
pv = pocas[val]
for etiqueta, sel in [("0", pv == 0), ("1-2", (pv >= 1) & (pv <= 2)), ("3+", pv >= 3)]:
    print(f"  {etiqueta:4s} n={sel.sum():6,}", end="")
    if sel.sum() < 300:
        print("  (muy pocas)"); continue
    ref = P[("previo", "solo margen")]
    for m in ("oráculo", "cercano_v20"):
        p = P[(m, "solo margen")]
        a, b = ic(p, ref, sel)
        print(f"  {m} {roc_auc_score(yv[sel], p[sel]) - roc_auc_score(yv[sel], ref[sel]):+.4f} [{a:+.4f};{b:+.4f}]", end="")
    print(f"  (previo {roc_auc_score(yv[sel], ref[sel]):.4f})", flush=True)
print(f"\n({time.time() - t0:.0f}s)")
