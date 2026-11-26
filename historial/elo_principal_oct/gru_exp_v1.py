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
est = S["est"][args.est.split(",")].to_numpy(np.float32)
base = {"regla": c["regla"], "kalman": c["km"], "z2": e["z2"].astype(float)}[args.base].to_numpy()
y = ((e["z1"].to_numpy() - base) / 100).astype(np.float32)
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
                                 nn.Dropout(d), nn.Linear(args.cabeza, 1))

    def forward(self, x, lg, s):
        mask = (torch.arange(x.shape[1], device=x.device)[None] < lg.to(x.device)[:, None]).float()
        if args.arq in ("gru", "bigru", "lstm"):
            pk = nn.utils.rnn.pack_padded_sequence(x, lg, batch_first=True, enforce_sorted=False)
            out = self.rnn(pk)[1]
            hT = out[0] if args.arq == "lstm" else out
            z = torch.cat([hT[-2], hT[-1]], 1) if args.arq == "bigru" else hT[-1]
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
        return self.cab(torch.cat([z, s], 1)).squeeze(1)


def lotes(idx, tam, rng=None):
    orden = idx[np.argsort(largos[idx], kind="stable")]
    bloques = [orden[i:i + tam] for i in range(0, len(orden), tam)]
    if rng is not None:
        rng.shuffle(bloques)
    for bl in bloques:
        lg = largos[bl]
        x = np.zeros((len(bl), lg.max(), len(paso)), np.float32)
        for j, k in enumerate(bl):
            x[j, :lg[j]] = seqs[k]
        yield (bl, torch.from_numpy(x).to(dispositivo), torch.from_numpy(lg),
               torch.from_numpy(est[bl]).to(dispositivo), torch.from_numpy(y[bl]).to(dispositivo))


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
        for bl, x, lg, s, yy in lotes(entrena, args.lote, rng):
            opt.zero_grad()
            costo(red(x, lg, s), yy).backward()
            opt.step()
    red.eval()
    out = np.zeros(len(e), np.float32)
    with torch.no_grad():
        for bl, x, lg, s, _ in lotes(predice, 1024):
            out[bl] = red(x, lg, s).cpu().numpy()
    return out[predice]


t0 = time.time()
pred = np.full(len(e), np.nan)
if args.caja:
    entrena, predice = np.where(dev)[0], np.where(~dev)[0]
    pred[predice] = np.mean([entrenar_predecir(entrena, predice, s) for s in range(args.semillas)], 0)
    idx = e.index[~dev]
else:
    for k in range(5):
        entrena, predice = np.where(dev & (fold != k))[0], np.where(dev & (fold == k))[0]
        pred[predice] = np.mean([entrenar_predecir(entrena, predice, s) for s in range(args.semillas)], 0)
    idx = e.index[dev]
final = pd.Series(base + 100 * pred, index=e.index)
(AQUI / "preds").mkdir(exist_ok=True)
np.save(AQUI / "preds" / f"{args.tag}{'_caja' if args.caja else ''}.npy", final.to_numpy())
m = evaluar(e, final, idx)
registrar(("CAJA " if args.caja else "") + args.tag, m, f"{args.nota} ({time.time() - t0:.0f} s)")
