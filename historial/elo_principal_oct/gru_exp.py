"""Banco de pruebas de redes secuenciales para el objetivo. Desarrollo: 5 folds sobre el 85 %;
la caja cerrada (15 %) solo con --caja. Guarda las predicciones fuera de fold en preds/<tag>.npy."""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import AQUI, cargar, evaluar, registrar  # noqa

PASO_BASE = ["resultado", "aliados", "rivales", "lobby", "conocidos", "dias", "hueco", "azul"]
EST_BASE = ["z2", "pred_regla", "partidas2", "wr_temporada", "obs", "obs_w", "kal_media", "kal_sd", "decay"]

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--paso", default=",".join(PASO_BASE))
ap.add_argument("--est", default=",".join(EST_BASE))
ap.add_argument("--arq", default="gru", choices=["gru", "lstm", "bigru", "attn", "transformer", "deepsets"])
ap.add_argument("--oculta", type=int, default=48)
ap.add_argument("--capas", type=int, default=1)
ap.add_argument("--cabeza", type=int, default=64)
ap.add_argument("--dropout", type=float, default=0.0)
ap.add_argument("--wd", type=float, default=0.0)
ap.add_argument("--epocas", type=int, default=40)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--lote", type=int, default=256)
ap.add_argument("--semillas", type=int, default=1, help="promedio de N semillas")
ap.add_argument("--base", default="regla", choices=["regla", "kalman", "z2"])
ap.add_argument("--perdida", default="l1", choices=["l1", "huber", "l2"])
ap.add_argument("--max-largo", type=int, default=0, help="truncar a las N partidas más viejas (0 = todas)")
ap.add_argument("--caja", action="store_true", help="entrena con todo el desarrollo y mide la caja cerrada")
ap.add_argument("--nota", default="")
ap.add_argument("--delta", action="store_true", help="suma un delta por partida (cabeza de suma acumulada)")
ap.add_argument("--aux", type=float, default=0.0, help="peso de la tarea auxiliar: victorias y derrotas ocultas")
ap.add_argument("--mdn", type=int, default=0, help="mezcla de K Laplace; predice la mediana")
ap.add_argument("--grid", default="", help="variables del filtro de grilla (tag de grilla.py) como estáticas")
ap.add_argument("--anidado", action="store_true", help="genera preds/anid_<tag>.npy (5, n) para stacking sin fuga")
ap.add_argument("--semilla0", type=int, default=0, help="primera semilla")
ap.add_argument("--aug", type=float, default=0.0, help="prob. de borrar cada partida observada (aumento, solo entrenamiento)")
ap.add_argument("--extra-caja", default="", help="con --caja: modelo cuyas preds fuera de fold (dev) y _caja (caja) van como estática")
ap.add_argument("--extra", default="", help="predicciones fuera de fold anidadas como estáticas (archivo npy de anidado.py)")
args = ap.parse_args()

c = cargar()
e = c["e"]
with open(AQUI / "seq.pkl", "rb") as fh:
    S = pickle.load(fh)
paso = args.paso.split(",")
ip = [S["PASO"].index(x) for x in paso]
seqs = [s[:, ip] for s in S["seqs"]]
if args.max_largo:
    seqs = [s[-args.max_largo:] for s in seqs]   # el final del arreglo son las partidas más viejas
largos = np.array([len(s) for s in seqs])
# Para el aumento: cuánto LP mueve cada partida según la regla (a si ganó, -b si perdió, por grupo del paso).
_ir, _il = S["PASO"].index("resultado"), S["PASO"].index("lp_paso")
_z2 = e["z2"].to_numpy()
_gm, _ch = c["corte"]
MOV = []
for i, s0 in enumerate(S["seqs"]):
    lp_ = s0[:, _il] * 300 + _z2[i]
    gg = np.where(lp_ >= _ch, 0, np.where(lp_ >= _gm, 1, np.where(lp_ >= 400, 2, 3)))
    av = np.array([c["a"][k] for k in range(4)])[gg]; bv = np.array([c["b"][k] for k in range(4)])[gg]
    MOV.append(np.where(s0[:, _ir] > 0, av, -bv).astype(np.float32) / 100)
I_OBS = [i for i, x in enumerate(args.est.split(",")) if x == "obs"]
I_OBSW = [i for i, x in enumerate(args.est.split(",")) if x == "obs_w"]
I_REGLA = [i for i, x in enumerate(args.est.split(",")) if x == "pred_regla"]
est = S["est"][args.est.split(",")].to_numpy(np.float32)
base = {"regla": c["regla"], "kalman": c["km"], "z2": e["z2"].astype(float)}[args.base].to_numpy()
y = ((e["z1"].to_numpy() - base) / 100).astype(np.float32)
yaux = np.column_stack([(e["dW"] - e["obs_w"]).clip(lower=0) / 20,
                        (e["dL"] - (e["obs"] - e["obs_w"])).clip(lower=0) / 20]).astype(np.float32)
N_OUT = (3 * args.mdn if args.mdn else 1) + (2 if args.aux else 0)
GRID = None
if args.grid:
    _sal = pickle.load(open(AQUI / f"{args.grid}.pkl", "rb"))
    _r = c["regla"].to_numpy()
    GRID = {k: np.column_stack([(g["media"].to_numpy() - _r) / 100, (g["mediana"].to_numpy() - _r) / 100,
                                g["sd"].to_numpy() / 100, (g["p10"].to_numpy() - _r) / 100,
                                (g["p90"].to_numpy() - _r) / 100, g["p_master"].to_numpy()]).clip(-10, 10).astype(np.float32)
            for k, g in _sal.items()}
    est = np.hstack([est, np.zeros((len(est), 6), np.float32)])
def N_EXTRA():
    return int(EXTRA is not None or bool(args.extra_caja))


EXTRA = None
if args.extra:
    EXTRA = np.load(AQUI / "preds" / args.extra)        # (5, n): para el fold externo k, columna fuera de fold
    est = np.hstack([est, np.zeros((len(est), 1), np.float32)])
dev = ~c["caja"]
fold = c["fold"]
dispositivo = torch.device("cuda")


class Red(nn.Module):
    def __init__(self):
        super().__init__()
        h, d = args.oculta, args.dropout
        if args.arq in ("gru", "bigru", "attn"):
            self.rnn = nn.GRU(len(paso), h, num_layers=args.capas, batch_first=True, dropout=d if args.capas > 1 else 0,
                              bidirectional=args.arq == "bigru")
            sal = h * (2 if args.arq == "bigru" else 1)
        elif args.arq == "lstm":
            self.rnn = nn.LSTM(len(paso), h, num_layers=args.capas, batch_first=True, dropout=d if args.capas > 1 else 0)
            sal = h
        elif args.arq == "transformer":
            self.proy = nn.Linear(len(paso), h)
            capa = nn.TransformerEncoderLayer(h, 4, 2 * h, dropout=d, batch_first=True)
            self.enc = nn.TransformerEncoder(capa, args.capas)
            sal = 2 * h
        else:   # deepsets: suma y media de una red por partida
            self.phi = nn.Sequential(nn.Linear(len(paso), h), nn.ReLU(), nn.Linear(h, h))
            sal = 2 * h
        if args.arq == "attn":
            self.att = nn.Linear(h, 1)
            sal = 2 * h
        self.cab = nn.Sequential(nn.Dropout(d), nn.Linear(sal + est.shape[1], args.cabeza), nn.ReLU(),
                                 nn.Dropout(d), nn.Linear(args.cabeza, N_OUT))
        if args.delta:
            self.dl = nn.Sequential(nn.Linear(h * (2 if args.arq == "bigru" else 1) + est.shape[1], 32), nn.ReLU(),
                                    nn.Linear(32, 1))
            nn.init.zeros_(self.dl[2].weight); nn.init.zeros_(self.dl[2].bias)

    def forward(self, x, lg, s):
        mask = (torch.arange(x.shape[1], device=x.device)[None] < lg.to(x.device)[:, None]).float()
        suma = 0
        if args.arq in ("gru", "bigru", "lstm"):
            pk = nn.utils.rnn.pack_padded_sequence(x, lg, batch_first=True, enforce_sorted=False)
            o, out = self.rnn(pk)
            hT = out[0] if args.arq == "lstm" else out
            z = torch.cat([hT[-2], hT[-1]], 1) if args.arq == "bigru" else hT[-1]
            if args.delta:
                o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True, total_length=x.shape[1])
                dd = self.dl(torch.cat([o, s[:, None, :].expand(-1, o.shape[1], -1)], -1)).squeeze(-1)
                suma = (dd * mask).sum(1)
        elif args.arq == "attn":
            pk = nn.utils.rnn.pack_padded_sequence(x, lg, batch_first=True, enforce_sorted=False)
            o, hT = self.rnn(pk)
            o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True, total_length=x.shape[1])
            w = self.att(o).squeeze(-1).masked_fill(mask == 0, -1e9).softmax(1)
            z = torch.cat([hT[-1], (w[..., None] * o).sum(1)], 1)
        elif args.arq == "transformer":
            o = self.enc(self.proy(x), src_key_padding_mask=mask == 0)
            m = mask[..., None]
            z = torch.cat([(o * m).sum(1) / m.sum(1).clamp(min=1), o[:, 0]], 1)
        else:
            o = self.phi(x) * mask[..., None]
            z = torch.cat([o.sum(1) / 50, o.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)], 1)
        o = self.cab(torch.cat([z, s], 1))
        o = torch.cat([o[:, :1] + (suma[:, None] if torch.is_tensor(suma) else 0), o[:, 1:]], 1)
        return o


def lotes(idx, tam, rng=None):
    orden = idx[np.argsort(largos[idx], kind="stable")]
    bloques = [orden[i:i + tam] for i in range(0, len(orden), tam)]
    if rng is not None:
        rng.shuffle(bloques)
    aumentar = rng is not None and args.aug > 0
    for bl in bloques:
        lg = largos[bl].copy()
        ss, yy = est[bl].copy(), y[bl].copy()
        filas = []
        for j, k in enumerate(bl):
            sq = seqs[k]
            if aumentar and lg[j] > 1 and rng.random() < 0.5:
                quedan = rng.random(lg[j]) >= args.aug
                quedan[rng.integers(lg[j])] = True
                fuera = ~quedan
                d = MOV[k][fuera].sum()                  # la regla deja de deshacer esas partidas
                yy[j] -= d
                for i_ in I_REGLA: ss[j, i_] += d * 100 / 300
                for i_ in I_OBS: ss[j, i_] -= fuera.sum() / 100
                for i_ in I_OBSW: ss[j, i_] -= (fuera & (S["seqs"][k][:, _ir] > 0)).sum() / 100
                sq = sq[quedan]
                lg[j] = len(sq)
            filas.append(sq)
        x = np.zeros((len(bl), lg.max(), len(paso)), np.float32)
        for j, sq in enumerate(filas):
            x[j, :lg[j]] = sq
        yield (bl, torch.from_numpy(x).to(dispositivo), torch.from_numpy(lg),
               torch.from_numpy(ss).to(dispositivo), torch.from_numpy(yy).to(dispositivo),
               torch.from_numpy(yaux[bl]).to(dispositivo))


def mdn_partes(o):
    K = args.mdn
    return o[:, :K].log_softmax(1), o[:, K:2 * K], nn.functional.softplus(o[:, 2 * K:3 * K]) + 0.02


def punto(o):
    """Predicción puntual: la salida principal, o la mediana de la mezcla."""
    if not args.mdn:
        return o[:, 0]
    lp, mu, b = mdn_partes(o)
    g = torch.linspace(-15, 15, 601, device=o.device)
    cdf = (lp.exp()[:, None, :] * torch.where(g[None, :, None] < mu[:, None, :],
                                               0.5 * torch.exp((g[None, :, None] - mu[:, None, :]) / b[:, None, :]),
                                               1 - 0.5 * torch.exp(-(g[None, :, None] - mu[:, None, :]) / b[:, None, :]))).sum(2)
    return g[(cdf < 0.5).sum(1).clamp(max=600)]


def costo_total(o, t, ta):
    if args.mdn:
        lp, mu, b = mdn_partes(o)
        c = -torch.logsumexp(lp - (t[:, None] - mu).abs() / b - torch.log(2 * b), 1).mean()
    else:
        c = costo(o[:, 0], t)
    if args.aux:
        c = c + args.aux * (o[:, -2:] - ta).abs().mean()
    return c


def costo(p, t):
    if args.perdida == "huber":
        return nn.functional.huber_loss(p, t, delta=1.0)
    if args.perdida == "l2":
        return ((p - t) ** 2).mean()
    return (p - t).abs().mean()


def entrenar_predecir(entrena, predice, semilla):
    torch.manual_seed(semilla)
    rng = np.random.default_rng(semilla)
    red = Red().to(dispositivo)
    opt = torch.optim.AdamW(red.parameters(), lr=args.lr, weight_decay=args.wd)
    for _ in range(args.epocas):
        red.train()
        for bl, x, lg, s, yy, ya in lotes(entrena, args.lote, rng):
            opt.zero_grad()
            costo_total(red(x, lg, s), yy, ya).backward()
            opt.step()
    red.eval()
    out = np.zeros(len(e), np.float32)
    with torch.no_grad():
        for bl, x, lg, s, _, _ in lotes(predice, 1024):
            out[bl] = punto(red(x, lg, s)).cpu().numpy()
    return out[predice]


t0 = time.time()
SEM = range(args.semilla0, args.semilla0 + args.semillas)
if args.anidado:
    import itertools
    A = np.zeros((5, len(e)), np.float32)
    for k in range(5):
        entrena, predice = np.where(dev & (fold != k))[0], np.where(dev & (fold == k))[0]
        A[k, predice] = np.mean([entrenar_predecir(entrena, predice, s) for s in SEM], 0)
    for j, k in itertools.combinations(range(5), 2):
        entrena = np.where(dev & (fold != j) & (fold != k))[0]
        fj, fk = np.where(dev & (fold == j))[0], np.where(dev & (fold == k))[0]
        pr = np.mean([entrenar_predecir(entrena, np.concatenate([fj, fk]), s) for s in SEM], 0)
        A[k, fj], A[j, fk] = pr[:len(fj)], pr[len(fj):]
        print(f"par {j},{k} ({time.time() - t0:.0f} s)", flush=True)
    np.save(AQUI / "preds" / f"anid_{args.tag}.npy", A)
    sys.exit()
pred = np.full(len(e), np.nan)
if args.caja:
    if args.extra_caja:
        _oof = np.load(AQUI / "preds" / f"{args.extra_caja}.npy")
        _cj = np.load(AQUI / "preds" / f"{args.extra_caja}_caja.npy")
        est = np.hstack([est, np.zeros((len(est), 1), np.float32)]) if EXTRA is None else est
        est[:, -1] = ((np.where(dev, _oof, _cj) - c["regla"].to_numpy()) / 100).astype(np.float32)
    if GRID is not None:
        est[:, -6 - N_EXTRA():est.shape[1] - N_EXTRA()] = GRID["todo"]
    entrena, predice = np.where(dev)[0], np.where(~dev)[0]
    pred[predice] = np.mean([entrenar_predecir(entrena, predice, s) for s in SEM], 0)
    idx = e.index[~dev]
else:
    for k in range(5):
        entrena, predice = np.where(dev & (fold != k))[0], np.where(dev & (fold == k))[0]
        if EXTRA is not None:
            est[:, -1] = EXTRA[k]
        if GRID is not None:
            est[:, -6 - N_EXTRA():est.shape[1] - N_EXTRA()] = GRID[k]
        pred[predice] = np.mean([entrenar_predecir(entrena, predice, s) for s in SEM], 0)
    idx = e.index[dev]
final = pd.Series(base + 100 * pred, index=e.index)
(AQUI / "preds").mkdir(exist_ok=True)
np.save(AQUI / "preds" / f"{args.tag}{'_caja' if args.caja else ''}.npy", final.to_numpy())
if args.caja:   # la caja se mide una sola vez, con el candidato final (compuerta.py --caja)
    print(f"{args.tag}: caja lista ({time.time() - t0:.0f} s)")
    sys.exit()
m = evaluar(e, final, idx)
registrar(("CAJA " if args.caja else "") + args.tag, m, f"{args.nota} ({time.time() - t0:.0f} s)")
