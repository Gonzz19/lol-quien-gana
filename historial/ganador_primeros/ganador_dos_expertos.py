"""Diagnóstico del modelo elegido: stacking de 2 expertos (§9.S.34). Pedido del usuario, 2026-11-13.

Expertos: red por jugador de grupos (ganador_red_jugador_grupos_oos.npz, `red_equipo`) y
logística con Poisson, nivel del lobby, modelo por línea y composición del draft
(ganador_logistica_composicion_oos.npz, §9.S.36-37). Meta:
logística sobre los dos logits, ajustada con las predicciones fuera de muestra (bloques 1-4).

1. Qué hace: pesos y calibración en V (por decil de probabilidad).
2. Dónde acierta y dónde falla: AUC y acierto por grupo de partida (descriptores simétricos
   entre lados, calculados con partidas ANTERIORES), y por día de V.
3. Cuándo discrepan los dos expertos y quién tiene razón.
4. Mejoras del meta que no reentrenan expertos, todas elegidas con los bloques (sin V):
   compuerta por grupo (pesos que dependen del caso) y recalibración.
Solo data/splits/train."""
import argparse, pickle, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
AN = REPO / "data" / "analysis"
ap = argparse.ArgumentParser()
ap.add_argument("--red", default=str(AN / "ganador_red_jugador_grupos_oos.npz"))
ap.add_argument("--logistica", default=str(AN / "ganador_logistica_composicion_oos.npz"),
                help="desde §9.S.37 la logística incluye la composición del draft; antes: ganador_logistica_lineas_oos.npz (§9.S.36)")
ap.add_argument("--feats", default=str(AN / "ganador_feats_grande.pkl"))
args = ap.parse_args()

lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
sigm = lambda z: 1 / (1 + np.exp(-z))
R, L = np.load(args.red), np.load(args.logistica)
assert (R["C_oos_idx"] == L["C_oos_idx"]).all() and (R["C_val_idx"] == L["C_val_idx"]).all()
oi, vi, bl = R["C_oos_idx"], R["C_val_idx"], R["C_oos_bloque"]
yo, yv = R["C_oos_y"], R["C_val_y"]
Zo = np.c_[lg(R["C_oos_red_equipo"]), lg(L["C_oos_logistica"])]
Zv = np.c_[lg(R["C_val_red_equipo"]), lg(L["C_val_logistica"])]

# ---------- descriptores por partida (simétricos entre lados), solo con partidas anteriores
d = pickle.load(open(args.feats, "rb"))
mids, t = d["mids"], d["t"]
del d
LANES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
df = pd.read_csv(REPO / "data/splits/train/full_dataset.csv",
                 usecols=["match_id", "game_creation", "player_id", "team_id", "lane", "champion_id"])
df = df[df.lane.isin(LANES)]
df = df[df.groupby(["match_id", "team_id"]).lane.transform("nunique") == 5]
df = df[df.groupby("match_id").lane.transform("size") == 10]
df = df.sort_values(["game_creation", "match_id", "team_id", "lane"]).reset_index(drop=True)
df["n_prev"] = df.groupby("player_id").cumcount()
df["estreno"] = (df.groupby(["player_id", "champion_id"]).cumcount() == 0) & (df.n_prev >= 10)
df["n_lane"] = df.groupby(["player_id", "lane"]).cumcount()
df["offrole"] = (df.n_prev >= 10) & (df.n_lane / df.n_prev.clip(lower=1) < 0.2)
g = df.groupby("match_id")
desc = pd.DataFrame({"cob_min": g.n_prev.min(), "nuevos": g.n_prev.apply(lambda s: (s < 10).sum()),
                     "estrenos": g.estreno.sum(), "offrole": g.offrole.sum()}).reindex(mids)
tier = pd.read_csv(REPO / "data/splits/match_split.csv", index_col="match_id").tier.reindex(mids)
desc["lobby"] = np.where(tier.isin(["MASTER", "GRANDMASTER", "CHALLENGER"]), "Master+", "Diamond")
dia = pd.to_datetime(t, unit="ms") - pd.Timedelta(hours=3)
desc["dia"] = dia.strftime("%m-%d")
desc["hora"] = pd.cut(dia.hour, [-1, 5, 11, 17, 23], labels=["00-06", "06-12", "12-18", "18-24"]).astype(str)
GRUPOS = {
    "lobby": desc.lobby.to_numpy(),
    "cobertura mínima": pd.cut(desc.cob_min, [-1, 9, 29, 1e9], labels=["< 10", "10-29", ">= 30"]).astype(str).to_numpy(),
    "jugadores con < 10 previas": pd.cut(desc.nuevos, [-1, 0, 2, 10], labels=["0", "1-2", "3+"]).astype(str).to_numpy(),
    "estrenos de campeón": pd.cut(desc.estrenos, [-1, 0, 1, 10], labels=["0", "1", "2+"]).astype(str).to_numpy(),
    "fuera de rol": pd.cut(desc.offrole, [-1, 0, 1, 10], labels=["0", "1", "2+"]).astype(str).to_numpy(),
    "hora (ART)": desc.hora.to_numpy(),
}

# ---------- 1. meta y calibración
meta = LogisticRegression(C=1e4, max_iter=5000).fit(Zo, yo)
pv = meta.predict_proba(Zv)[:, 1]
po = None
print(f"pesos red / logística: {meta.coef_[0][0]:.3f} / {meta.coef_[0][1]:.3f}  constante {meta.intercept_[0]:+.3f}")
print(f"V: AUC {roc_auc_score(yv, pv):.4f}  acierto {accuracy_score(yv, pv > .5):.4f}  log loss {log_loss(yv, pv):.4f}"
      f"  (red sola {roc_auc_score(yv, Zv[:, 0]):.4f}, logística sola {roc_auc_score(yv, Zv[:, 1]):.4f})")
print("\n=== 1. Calibración en V (deciles de la probabilidad de que gane azul) ===")
q = pd.qcut(pv, 10, labels=False)
cal = pd.DataFrame({"p": pv, "y": yv, "q": q}).groupby("q").agg(n=("y", "size"), p_media=("p", "mean"), azul_gana=("y", "mean"))
print(cal.round(3).to_string())
ece = np.sum(cal.n * np.abs(cal.p_media - cal.azul_gana)) / len(pv)
print(f"error de calibración (ECE) {ece:.4f}; pendiente de la recalibración logit(p) -> y en V "
      f"{LogisticRegression(C=1e4).fit(lg(pv)[:, None], yv).coef_[0][0]:.3f} (1 = calibrado)")

# ---------- 2. por grupo
B = np.random.default_rng(0).integers(0, len(yv), (200, len(yv)))
gv = {k: v[vi] for k, v in GRUPOS.items()}
print("\n=== 2. Por grupo en V: n, AUC del stacking, de cada experto y acierto ===")
print(f"{'':34s}{'n':>7s}{'AUC':>8s}{'red':>8s}{'log':>8s}{'acierto':>9s}{'acierto 56 % seguro':>21s}")
u = np.quantile(np.abs(pv - .5), 1 - 0.56)
def fila(nom, s):
    a = lambda z: roc_auc_score(yv[s], z[s]) if len(set(yv[s])) > 1 else np.nan
    seg = s & (np.abs(pv - .5) >= u)
    print(f"{nom:34s}{s.sum():>7d}{a(pv):>8.4f}{a(Zv[:, 0]):>8.4f}{a(Zv[:, 1]):>8.4f}{accuracy_score(yv[s], pv[s] > .5):>9.3f}"
          f"{accuracy_score(yv[seg], pv[seg] > .5):>13.3f} ({seg.sum() / s.sum():.0%})")
fila("todo V", np.ones(len(yv), bool))
for k, v in gv.items():
    for c in sorted(set(v)):
        fila(f"{k}: {c}", v == c)
print("\npor día de V:")
dv = desc.dia.to_numpy()[vi]
for c in sorted(set(dv)):
    fila(f"  {c}", dv == c)

# ---------- 3. discrepancias
print("\n=== 3. Cuando los expertos discrepan ===")
dz = Zv[:, 0] - Zv[:, 1]
for nom, s in [("eligen el mismo ganador", np.sign(Zv[:, 0]) == np.sign(Zv[:, 1])),
               ("eligen ganadores distintos", np.sign(Zv[:, 0]) != np.sign(Zv[:, 1]))]:
    print(f"  {nom:28s} {s.mean():6.1%} de V | acierto red {accuracy_score(yv[s], Zv[s, 0] > 0):.3f}"
          f"  logística {accuracy_score(yv[s], Zv[s, 1] > 0):.3f}  stacking {accuracy_score(yv[s], pv[s] > .5):.3f}")
qd = pd.qcut(np.abs(dz), 5, labels=["muy poca", "poca", "media", "alta", "muy alta"])
for c in qd.categories:
    s = (qd == c)
    print(f"  discrepancia {c:9s}: acierto red {accuracy_score(yv[s], Zv[s, 0] > 0):.3f}  logística "
          f"{accuracy_score(yv[s], Zv[s, 1] > 0):.3f}  stacking {accuracy_score(yv[s], pv[s] > .5):.3f}  AUC stacking {roc_auc_score(yv[s], pv[s]):.4f}")

# ---------- 4. mejoras del meta (elegidas con los bloques, sin V)
go = {k: v[oi] for k, v in GRUPOS.items()}

def diseño(Z, grupos, cuales):
    cols = [Z]
    for k in cuales:
        cats = sorted(set(GRUPOS[k]))[1:]                    # una categoría de referencia
        for c in cats:
            D = (grupos[k] == c).astype(float)[:, None]
            cols += [D, Z * D]
    return np.hstack(cols)

def bloques(cuales, C=1e4):
    ll = []
    for k in (2, 3, 4):
        Xa, Xb = diseño(Zo[bl < k], {g: v[bl < k] for g, v in go.items()}, cuales), diseño(Zo[bl == k], {g: v[bl == k] for g, v in go.items()}, cuales)
        p = LogisticRegression(C=C, max_iter=5000).fit(Xa, yo[bl < k]).predict_proba(Xb)[:, 1]
        ll.append(log_loss(yo[bl == k], p))
    return np.mean(ll)

def en_v(cuales, C=1e4):
    m = LogisticRegression(C=C, max_iter=5000).fit(diseño(Zo, go, cuales), yo)
    return m.predict_proba(diseño(Zv, gv, cuales))[:, 1]

def dif(p):
    dd = np.array([roc_auc_score(yv[b], p[b]) - roc_auc_score(yv[b], pv[b]) for b in B])
    return f"{roc_auc_score(yv, p) - roc_auc_score(yv, pv):+.4f} [{np.quantile(dd, .025):+.4f};{np.quantile(dd, .975):+.4f}]"

base_ll = bloques([])
print(f"\n=== 4. Compuerta: pesos del meta según el grupo (criterio: log loss en los bloques 2-4, sin V) ===")
print(f"  sin compuerta                      bloques {base_ll:.5f}")
res = {}
for k in GRUPOS:
    if k == "hora (ART)" or True:
        ll = bloques([k]); p = en_v([k]); res[k] = ll
        print(f"  compuerta por {k:28s} bloques {ll:.5f} ({(ll - base_ll) * 1e4:+.1f}e-4) | V AUC {roc_auc_score(yv, p):.4f} vs sin {dif(p)}")
buenos = [k for k, ll in res.items() if ll < base_ll - 1e-5]
if len(buenos) > 1:
    ll = bloques(buenos); p = en_v(buenos)
    print(f"  compuerta por {buenos}: bloques {ll:.5f} ({(ll - base_ll) * 1e4:+.1f}e-4) | V AUC {roc_auc_score(yv, p):.4f} vs sin {dif(p)}")
