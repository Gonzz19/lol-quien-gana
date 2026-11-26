"""¿Cambia algo pasar las variables por jugador, sin restar azul - rojo?

Las mismas variables que la mejor logística, armadas de otra forma:
* L1: logística con los dos equipos por separado (el doble de columnas; puede aprender pesos distintos por lado).
* L2: logística azul - rojo, con el campeón por línea (campeón × línea) en lugar de por equipo.
* N1: MLP con los dos equipos concatenados (ve f(azul, rojo), no f(azul - rojo)).
* N2: codificador compartido por jugador: cada jugador (variables + línea + embedding de su campeón) pasa
  por la misma red, se suma por equipo y la salida es g(azul) - g(rojo). Antisimétrica por construcción.
Validación y ajuste iguales a los de antes. Las redes, espejadas, con parada temprana y 3 semillas.

    python historial/ganador_primeros/ganador_features_grande.py
    python historial/ganador_primeros/ganador_huella_progresiva.py
    python historial/ganador_primeros/ganador_por_jugador.py          # GPU, ~10 min
"""
import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
A = REPO / "data" / "analysis"
LANES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
COMUN = ["onehot", "experiencia", "cobertura", "dificultad", "matchups", "sinergias"]
t0 = time.time()

d = pickle.load(open(A / "ganador_feats_grande.pkl", "rb"))
h = pickle.load(open(A / "ganador_huella_progresiva.pkl", "rb"))
feat, y, t, val, masks = d["feat"], d["y"], d["t"], d["val"], d["masks"]

# ---------- variables por jugador (mismo código que ganador_features_grande.py)
df = pd.read_csv(REPO / "data" / "splits" / "train" / "full_dataset.csv")
df = df[df.lane.isin(LANES)]
df = df[df.groupby(["match_id", "team_id"]).lane.transform("nunique") == 5]
df = df[df.groupby("match_id").lane.transform("size") == 10].copy()
df = df.sort_values(["game_creation", "match_id", "team_id", "lane"]).reset_index(drop=True)
mids = df.groupby("match_id").game_creation.first().sort_values().index.to_numpy()
gp = df.groupby("player_id")
df["n_tot"] = gp.cumcount()
gc = df.groupby(["player_id", "champion_id"])
df["n_champ"] = gc.cumcount()
df["w_champ"] = gc.win.cumsum() - df.win
df["w_tot"] = gp.win.cumsum() - df.win
df["n_lane"] = df.groupby(["player_id", "lane"]).cumcount()
df["x_logn_champ"] = np.log1p(df.n_champ)
df["x_estreno"] = (df.n_champ == 0).astype(float)
df["x_wr_champ"] = (df.w_champ + 5 * (df.w_tot + 5) / (df.n_tot + 10)) / (df.n_champ + 5) - (df.w_tot + 5) / (df.n_tot + 10)
df["x_share_champ"] = df.n_champ / (df.n_tot + 1)
df["x_share_lane"] = (df.n_lane + 1) / (df.n_tot + 2)
df["x_offrole"] = ((df.n_tot >= 10) & (df.n_lane / df.n_tot.clip(lower=1) < 0.2)).astype(float)
df["x_logn_tot"] = np.log1p(df.n_tot)
dd = json.load(open(REPO / "data" / "ddragon" / "16.17.1" / "champion.json", encoding="utf-8"))["data"]
df["x_dif"] = df.champion_id.map({int(v["key"]): v["info"]["difficulty"] for v in dd.values()})
df["x_dif"] = df.x_dif.fillna(df.x_dif.mean()) / 10
df["x_dif_estreno"] = df.x_dif * df.x_estreno
df["x_dif_logn"] = df.x_dif * df.x_logn_champ
JUG = ["x_logn_champ", "x_estreno", "x_wr_champ", "x_share_champ", "x_share_lane", "x_offrole", "x_logn_tot", "x_dif", "x_dif_estreno", "x_dif_logn"]

f = h["fila"]
hu = pd.DataFrame(f["huella"], columns=[f"h{i}" for i in range(f["huella"].shape[1])])
hu["h_n"] = np.log1p(f["n"])
hu["match_id"], hu["team_id"], hu["lane"] = f["match_id"], f["team_id"], f["lane"]
df = df.merge(hu, on=["match_id", "team_id", "lane"], how="left", validate="one_to_one")
HCOL = [c for c in hu.columns if c.startswith("h")]
assert df[HCOL].notna().all().all()

pos = pd.Series(np.arange(len(mids)), index=mids)
df["r"] = pos[df.match_id].to_numpy()
df["eq"] = (df.team_id == 200).astype(int)
df["li"] = df.lane.map({l: i for i, l in enumerate(LANES)})
champs = {c: i for i, c in enumerate(sorted(df.champion_id.unique()))}
df["ci"] = df.champion_id.map(champs)
FJ = JUG + HCOL
P = np.zeros((len(mids), 2, 5, len(FJ)), np.float32)
ri, ei, li_ = df["r"].to_numpy(), df["eq"].to_numpy(), df["li"].to_numpy().astype(int)
P[ri, ei, li_] = df[FJ].to_numpy(np.float32)
C = np.zeros((len(mids), 2, 5), np.int64)
C[ri, ei, li_] = df.ci.to_numpy()
NC = len(champs)
EXTRA = np.hstack([feat["matchups"], feat["sinergias"]]).astype(np.float32)    # ya son azul - rojo
print(f"{len(mids):,} partidas, {len(FJ)} variables por jugador, {NC} campeones ({time.time() - t0:.0f} s)", flush=True)

fit_idx = np.where(masks["C"])[0]
fit_idx = fit_idx[np.argsort(t[fit_idx])]
vidx = np.where(val)[0]
yv = y[val]
B = np.random.default_rng(0).integers(0, val.sum(), (500, val.sum()))
dev = "cuda" if torch.cuda.is_available() else "cpu"
res = {}


def informe(nombre, p, ref=None):
    res[nombre] = p
    txt = f"{nombre:62s} acc {accuracy_score(yv, p > .5):.3f}  AUC {roc_auc_score(yv, p):.4f}  logloss {log_loss(yv, p):.4f}"
    if ref is not None:
        dif = [roc_auc_score(yv[b], p[b]) - roc_auc_score(yv[b], ref[b]) for b in B]
        txt += f"  vs ref {np.mean(dif):+.4f} [{np.quantile(dif, .025):+.4f}, {np.quantile(dif, .975):+.4f}]"
    print(txt, flush=True)


def logistica(X, Cr=0.05):
    sc = StandardScaler(with_mean=False).fit(X[fit_idx])
    m = LogisticRegression(C=Cr, max_iter=5000).fit(sc.transform(X[fit_idx]), y[fit_idx])
    return m.predict_proba(sc.transform(X[vidx]))[:, 1]


# ---------- logísticas
ref = logistica(np.hstack([feat[k] for k in COMUN] + [h["huella_prog"], h["huella_n"]]))
informe("Referencia: azul - rojo, campeón por equipo", ref)
oh_eq = np.zeros((len(mids), 2, NC), np.float32)
for e in (0, 1):
    for l in range(5):
        oh_eq[np.arange(len(mids)), e, C[:, e, l]] = 1
Xsep = np.hstack([P.reshape(len(mids), -1), oh_eq.reshape(len(mids), -1), EXTRA])
informe("L1: logística, los dos equipos por separado", logistica(Xsep), ref)
oh_cl = np.zeros((len(mids), 5 * NC), np.float32)
for l in range(5):
    oh_cl[np.arange(len(mids)), l * NC + C[:, 0, l]] += 1
    oh_cl[np.arange(len(mids)), l * NC + C[:, 1, l]] -= 1
Xcl = np.hstack([feat[k] for k in COMUN if k != "onehot"] + [oh_cl, h["huella_prog"], h["huella_n"]])
informe("L2: logística azul - rojo, campeón por línea", logistica(Xcl), ref)

# ---------- redes
mu = P[fit_idx].reshape(-1, len(FJ)).mean(0)
sd = P[fit_idx].reshape(-1, len(FJ)).std(0) + 1e-6
Pn = (P - mu) / sd
ex_sd = EXTRA[fit_idx].std(0) + 1e-6
EXn = EXTRA / ex_sd


class Concat(torch.nn.Module):
    """MLP sobre los 10 jugadores concatenados (azul y después rojo), con embedding de campeón."""
    def __init__(self, capas=(256, 128), emb=16, drop=.3):
        super().__init__()
        self.emb = torch.nn.Embedding(NC, emb)
        ent = 10 * (len(FJ) + emb) + EXn.shape[1]
        L = []
        for u in capas:
            L += [torch.nn.Linear(ent, u), torch.nn.ReLU(), torch.nn.Dropout(drop)]
            ent = u
        self.red = torch.nn.Sequential(*L, torch.nn.Linear(ent, 1))

    def forward(self, p, c, ex):
        z = torch.cat([p, self.emb(c)], -1).flatten(1)
        return self.red(torch.cat([z, ex], 1)).squeeze(1)


class Compartido(torch.nn.Module):
    """Mismo codificador para los 10; suma por equipo; g(azul) - g(rojo)."""
    def __init__(self, oculta=128, salida=32, emb=16, drop=.2):
        super().__init__()
        self.emb = torch.nn.Embedding(NC, emb)
        self.lin = torch.nn.Embedding(5, 4)
        ent = len(FJ) + emb + 4
        self.phi = torch.nn.Sequential(torch.nn.Linear(ent, oculta), torch.nn.ReLU(), torch.nn.Dropout(drop),
                                       torch.nn.Linear(oculta, salida), torch.nn.ReLU())
        self.rho = torch.nn.Sequential(torch.nn.Linear(salida, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
        self.w_ex = torch.nn.Linear(EXn.shape[1], 1)
        self.lineas = torch.arange(5)

    def forward(self, p, c, ex):
        li = self.lin(self.lineas.to(p.device)).expand(p.shape[0], 2, 5, -1)
        z = self.phi(torch.cat([p, self.emb(c), li], -1)).sum(2)        # (n, 2, salida)
        g = self.rho(z).squeeze(-1)                                     # (n, 2)
        return g[:, 0] - g[:, 1] + self.w_ex(ex).squeeze(1)


def entrenar(clase, seed, **kw):
    torch.manual_seed(seed)
    corte = int(len(fit_idx) * 0.85)
    tr, es = fit_idx[:corte], fit_idx[corte:]
    T = lambda a: torch.tensor(a, device=dev)
    # espejo: se intercambian los equipos, se invierten las diferencias y el resultado
    Ptr = T(np.r_[Pn[tr], Pn[tr][:, ::-1]]); Ctr = T(np.r_[C[tr], C[tr][:, ::-1]])
    Etr = T(np.r_[EXn[tr], -EXn[tr]]); ytr = T(np.r_[y[tr], 1 - y[tr]].astype(np.float32))
    net = clase(**kw).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)

    def pred(i):
        net.eval()
        with torch.no_grad():
            return torch.sigmoid(net(T(Pn[i]), T(C[i]), T(EXn[i]))).cpu().numpy()
    best, estado, pac = 1e9, None, 0
    for ep in range(200):
        net.train()
        for b in torch.randperm(len(ytr), device=dev).split(1024):
            opt.zero_grad()
            torch.nn.functional.binary_cross_entropy_with_logits(net(Ptr[b], Ctr[b], Etr[b]), ytr[b]).backward()
            opt.step()
        ll = log_loss(y[es], pred(es))
        if ll < best - 1e-5:
            best, estado, pac = ll, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            pac += 1
            if pac >= 10:
                break
    net.load_state_dict(estado)
    return pred(vidx)


for nombre, clase, kw in (("N1: MLP 256-128, los dos equipos concatenados", Concat, {}),
                          ("N1b: MLP 512-256, los dos equipos concatenados", Concat, {"capas": (512, 256), "drop": .4}),
                          ("N2: codificador compartido por jugador (128-32)", Compartido, {}),
                          ("N2b: codificador compartido más grande (256-64)", Compartido, {"oculta": 256, "salida": 64})):
    t1 = time.time()
    p = np.mean([entrenar(clase, s, **kw) for s in (0, 1, 2)], axis=0)
    informe(nombre + f", 3 semillas ({time.time() - t1:.0f} s)", p, ref)
informe("Promedio referencia + mejor red", (ref + max((v for k, v in res.items() if k.startswith("N")), key=lambda p: roc_auc_score(yv, p))) / 2, ref)
pickle.dump(res, open(A / "ganador_por_jugador.pkl", "wb"))
print(f"listo en {time.time() - t0:.0f} s")
