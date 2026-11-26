"""Huella separada en habilidad y estilo, y matchup de estilos por línea (pregunta del usuario).

* Habilidad: proyección de la huella sobre la dirección que más predice ganar (logística sobre la huella
  azul - rojo, ajustada solo con el ajuste). Un número por jugador.
* Estilo: la huella sin esa dirección, resumida con PCA (ejes ajustados solo con el ajuste).
* Matchup de estilos: en cada línea, s_azul_i * s_rojo_j - s_rojo_i * s_azul_j (i < j): antisimétrico, así que
  invertir los lados invierte el signo. Dice si "el tipo i le gana al tipo j" en esa línea.
* Red: el codificador compartido, más un término por línea psi(azul, rojo) - psi(rojo, azul).
Misma validación de siempre.

    python historial/ganador_primeros/ganador_features_grande.py
    python historial/ganador_primeros/ganador_huella_progresiva.py
    python historial/ganador_primeros/ganador_estilo_habilidad.py      # GPU, ~10 min
"""
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.features.fingerprint import STYLE_FEATURES  # noqa: E402

A = REPO / "data" / "analysis"
LANES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
COMUN = ["onehot", "experiencia", "cobertura", "dificultad", "matchups", "sinergias"]
N_EST = 4
t0 = time.time()
d = pickle.load(open(A / "ganador_feats_grande.pkl", "rb"))
h = pickle.load(open(A / "ganador_huella_progresiva.pkl", "rb"))
feat, y, t, val, masks = d["feat"], d["y"], d["t"], d["val"], d["masks"]

# ---------- huella por jugador en un tensor (partida, equipo, línea, 23)
mt = pd.read_csv(REPO / "data" / "splits" / "train" / "full_dataset.csv", usecols=["match_id", "game_creation"]) \
       .groupby("match_id").game_creation.first()
f = h["fila"]
pos = pd.Series(np.arange(len(t)), index=mt.loc[mt.index.isin(np.unique(f["match_id"]))].sort_values().index)
r = pos[f["match_id"]].to_numpy()
e = (f["team_id"] == 200).astype(int)
li = pd.Series(f["lane"]).map({l: i for i, l in enumerate(LANES)}).to_numpy().astype(int)
Hu = np.zeros((len(t), 2, 5, f["huella"].shape[1]), np.float32)
Hu[r, e, li] = f["huella"]
assert np.allclose(np.hstack([(Hu[:, 0, l] - Hu[:, 1, l]) for l in range(5)]), h["huella_prog"], atol=1e-4), "desalineado"

fit = masks["C"]
fit_idx = np.where(fit)[0]
yv = y[val]
B = np.random.default_rng(0).integers(0, val.sum(), (500, val.sum()))
res = {}

# ---------- habilidad: dirección que más predice ganar (solo ajuste)
dif_eq = (Hu[:, 0] - Hu[:, 1]).sum(1)                                     # suma de líneas, azul - rojo
sc = StandardScaler().fit(dif_eq[fit])
lr = LogisticRegression(C=0.05, max_iter=5000).fit(sc.transform(dif_eq[fit]), y[fit])
w = lr.coef_[0] / sc.scale_
w = w / np.linalg.norm(w)
print("Dirección de habilidad (pesos normalizados, mayores primero):")
for n, v in sorted(zip(STYLE_FEATURES, w), key=lambda x: -abs(x[1]))[:10]:
    print(f"   {n:24s} {v:+.3f}")
hab = Hu @ w                                                              # (n, 2, 5)
est = Hu - hab[..., None] * w                                             # sin la dirección de habilidad
pca = PCA(N_EST).fit(est[fit].reshape(-1, est.shape[-1]))
print("Varianza del estilo explicada por los ejes:", np.round(pca.explained_variance_ratio_, 3), flush=True)
S = pca.transform(est.reshape(-1, est.shape[-1])).reshape(len(t), 2, 5, N_EST).astype(np.float32)
for k in range(N_EST):
    top = sorted(zip(STYLE_FEATURES, pca.components_[k]), key=lambda x: -abs(x[1]))[:4]
    print(f"   eje {k}: " + ", ".join(f"{n} {v:+.2f}" for n, v in top))
print("corr(habilidad, ganar) por jugador en el ajuste:", round(np.corrcoef(hab[fit].reshape(-1), np.repeat(np.c_[y[fit], 1 - y[fit]], 5, 1).reshape(-1))[0, 1], 3))

# ---------- variables
habil = hab[:, 0] - hab[:, 1]                                             # (n, 5)
estilo = (S[:, 0] - S[:, 1]).reshape(len(t), -1)                          # (n, 5*N_EST)
cruces = []
for l in range(5):
    for i in range(N_EST):
        for j in range(i + 1, N_EST):
            cruces.append(S[:, 0, l, i] * S[:, 1, l, j] - S[:, 1, l, i] * S[:, 0, l, j])
cruces = np.c_[tuple(cruces)]
# estilo propio × habilidad del rival de línea (y viceversa), antisimétrico
est_x_hab = np.c_[tuple(S[:, 0, l, i] * hab[:, 1, l] - S[:, 1, l, i] * hab[:, 0, l] for l in range(5) for i in range(N_EST))]
base = [feat[k] for k in COMUN] + [h["huella_n"]]


def informe(nombre, p, ref=None):
    res[nombre] = p
    txt = f"{nombre:66s} acc {accuracy_score(yv, p > .5):.3f}  AUC {roc_auc_score(yv, p):.4f}  logloss {log_loss(yv, p):.4f}"
    if ref is not None:
        dd = [roc_auc_score(yv[b], p[b]) - roc_auc_score(yv[b], ref[b]) for b in B]
        txt += f"  vs ref {np.mean(dd):+.4f} [{np.quantile(dd, .025):+.4f}, {np.quantile(dd, .975):+.4f}]"
    print(txt, flush=True)


def logistica(X):
    s = StandardScaler(with_mean=False).fit(X[fit])
    m = LogisticRegression(C=0.05, max_iter=5000).fit(s.transform(X[fit]), y[fit])
    return m.predict_proba(s.transform(X[val]))[:, 1]


print()
ref = logistica(np.hstack(base + [h["huella_prog"]]))
informe("Referencia: huella completa (23 por línea)", ref)
informe(f"Habilidad (1) + estilo ({N_EST}) por línea, en vez de la huella", logistica(np.hstack(base + [habil, estilo])), ref)
informe("  + matchup de estilos por línea", logistica(np.hstack(base + [habil, estilo, cruces])), ref)
informe("  + matchup de estilos + estilo propio × habilidad del rival", logistica(np.hstack(base + [habil, estilo, cruces, est_x_hab])), ref)
informe("Huella completa + matchup de estilos + estilo × habilidad", logistica(np.hstack(base + [h["huella_prog"], cruces, est_x_hab])), ref)

# ---------- red: codificador compartido + término de línea contra línea
pr = pickle.load(open(A / "ganador_por_jugador.pkl", "rb")) if (A / "ganador_por_jugador.pkl").exists() else {}
dev = "cuda" if torch.cuda.is_available() else "cpu"
# Entrada por jugador: habilidad, estilo, y las variables de experiencia/cobertura/dificultad de la huella_n no
# están por jugador acá; se usan las de por línea de feat restadas como término lineal aparte.
J = np.concatenate([hab[..., None], S], -1)                               # (n, 2, 5, 1+N_EST)
Jm, Js = J[fit].reshape(-1, J.shape[-1]).mean(0), J[fit].reshape(-1, J.shape[-1]).std(0) + 1e-6
Jn = ((J - Jm) / Js).astype(np.float32)
L = np.hstack(base).astype(np.float32)
Ls = L[fit].std(0) + 1e-6
Ln = (L / Ls).astype(np.float32)


class Red(torch.nn.Module):
    def __init__(self, linea_vs_linea, oculta=64):
        super().__init__()
        ent = J.shape[-1] + 4
        self.lin = torch.nn.Embedding(5, 4)
        self.phi = torch.nn.Sequential(torch.nn.Linear(ent, oculta), torch.nn.ReLU(), torch.nn.Linear(oculta, 32), torch.nn.ReLU())
        self.rho = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
        self.lvl = linea_vs_linea
        if linea_vs_linea:
            self.psi = torch.nn.Sequential(torch.nn.Linear(2 * ent, oculta), torch.nn.ReLU(), torch.nn.Linear(oculta, 1))
        self.lineal = torch.nn.Linear(Ln.shape[1], 1)
        self.l5 = torch.arange(5)

    def forward(self, j, x):
        le = self.lin(self.l5.to(j.device)).expand(j.shape[0], 2, 5, -1)
        u = torch.cat([j, le], -1)                                        # (n, 2, 5, ent)
        g = self.rho(self.phi(u).sum(2)).squeeze(-1)
        out = g[:, 0] - g[:, 1] + self.lineal(x).squeeze(1)
        if self.lvl:
            ab = torch.cat([u[:, 0], u[:, 1]], -1); ba = torch.cat([u[:, 1], u[:, 0]], -1)
            out = out + (self.psi(ab) - self.psi(ba)).squeeze(-1).sum(1)
        return out


def entrenar(lvl, seed):
    torch.manual_seed(seed)
    idx = fit_idx[np.argsort(t[fit_idx])]
    corte = int(len(idx) * .85)
    tr, es = idx[:corte], idx[corte:]
    T = lambda a: torch.tensor(np.ascontiguousarray(a), device=dev)
    Jtr = T(np.r_[Jn[tr], Jn[tr][:, ::-1]]); Xtr = T(np.r_[Ln[tr], -Ln[tr]]); ytr = T(np.r_[y[tr], 1 - y[tr]].astype(np.float32))
    net = Red(lvl).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)

    def pred(i):
        net.eval()
        with torch.no_grad():
            return torch.sigmoid(net(T(Jn[i]), T(Ln[i]))).cpu().numpy()
    best, st, pac = 1e9, None, 0
    for ep in range(200):
        net.train()
        for b in torch.randperm(len(ytr), device=dev).split(1024):
            opt.zero_grad()
            torch.nn.functional.binary_cross_entropy_with_logits(net(Jtr[b], Xtr[b]), ytr[b]).backward()
            opt.step()
        ll = log_loss(y[es], pred(es))
        if ll < best - 1e-5:
            best, st, pac = ll, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            pac += 1
            if pac >= 10:
                break
    net.load_state_dict(st)
    return pred(np.where(val)[0])


print()
for nombre, lvl in (("Red: habilidad + estilo por jugador, sin línea contra línea", False),
                    ("Red: habilidad + estilo por jugador, con línea contra línea", True)):
    p = np.mean([entrenar(lvl, s) for s in (0, 1, 2)], axis=0)
    informe(nombre + ", 3 semillas", p, ref)
mejor_red = res["Red: habilidad + estilo por jugador, con línea contra línea, 3 semillas"]
informe("Promedio referencia + red con línea contra línea", (ref + mejor_red) / 2, ref)
pickle.dump(res, open(A / "ganador_estilo_habilidad.pkl", "wb"))
print(f"listo en {time.time() - t0:.0f} s")
