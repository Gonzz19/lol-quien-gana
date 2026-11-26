"""GRU del elo al revés con 20 % de jugadores fuera: MAE de train y de validación por época."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn

ROOT = Path(r"c:/Users/DarkA/OneDrive/Escritorio/Facultad/3er año/ML/TPF/ProyectoFinalML")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
from src.features.elo_reverso import ajustar_regla, al_reves, cargar_train, cortes, evaluables  # noqa
from elo_gru import PASO, ESTATICAS, secuencias  # noqa

EPOCAS = int(sys.argv[1]); OCULTAS = [int(h) for h in sys.argv[2].split(",")]; SEMILLA = 0
dev = torch.device("cuda")
d = cargar_train(ROOT / "data")
e = evaluables(d["foto1"], d["foto2"], d["partes"])
corte = cortes(d["foto2"])
a, b = ajustar_regla(e)
pred_regla = al_reves(d["partes"], d["foto2"]["z"], a, b, corte).reindex(e.index).fillna(e["z2"])
seqs = secuencias(e, d["partes"], d["foto2"]["z"], d["t2"])
largos = np.array([len(s) for s in seqs])
est = pd.DataFrame({"z2": (e["z2"] - 500) / 300, "pred_regla": (pred_regla - 500) / 300,
                    "partidas2": e["partidas2"] / 500, "wr_temporada": e["wins2"] / e["partidas2"],
                    "obs": e["obs"] / 100, "obs_w": e["obs_w"] / 100})[ESTATICAS].to_numpy(np.float32)
y = ((e["z1"] - pred_regla) / 100).to_numpy(np.float32)
rng0 = np.random.default_rng(123)
val = rng0.random(len(e)) < 0.2
idx_tr, idx_va = np.where(~val)[0], np.where(val)[0]
print(f"train {len(idx_tr)}, val {len(idx_va)}; regla en val {100*np.abs(y[idx_va]).mean():.1f}", flush=True)


class Red(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.gru = nn.GRU(len(PASO), h, batch_first=True)
        self.cabeza = nn.Sequential(nn.Linear(h + len(ESTATICAS), 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x, largo, s):
        pk = nn.utils.rnn.pack_padded_sequence(x, largo, batch_first=True, enforce_sorted=False)
        _, h = self.gru(pk)
        return self.cabeza(torch.cat([h[-1], s], 1)).squeeze(1)


def lotes(idx, tam, rng=None):
    orden = idx[np.argsort(largos[idx], kind="stable")]
    bloques = [orden[i:i + tam] for i in range(0, len(orden), tam)]
    if rng is not None:
        rng.shuffle(bloques)
    for bl in bloques:
        lg = largos[bl]
        x = np.zeros((len(bl), lg.max(), len(PASO)), np.float32)
        for j, k in enumerate(bl):
            x[j, :lg[j]] = seqs[k]
        yield (bl, torch.from_numpy(x).to(dev), torch.from_numpy(lg),
               torch.from_numpy(est[bl]).to(dev), torch.from_numpy(y[bl]).to(dev))


res = {}
for h in OCULTAS:
    torch.manual_seed(SEMILLA); rng = np.random.default_rng(SEMILLA)
    red = Red(h).to(dev); opt = torch.optim.Adam(red.parameters(), lr=2e-3)
    hist = []
    for ep in range(EPOCAS):
        red.train(); tot = 0.0
        for bl, x, lg, s, yy in lotes(idx_tr, 256, rng):
            opt.zero_grad(); p = (red(x, lg, s) - yy).abs().mean(); p.backward(); opt.step()
            tot += p.item() * len(bl)
        red.eval(); err = 0.0
        with torch.no_grad():
            for bl, x, lg, s, yy in lotes(idx_va, 1024):
                err += (red(x, lg, s) - yy).abs().sum().item()
        hist.append((ep + 1, round(100 * tot / len(idx_tr), 1), round(100 * err / len(idx_va), 1)))
        print(f"h={h} ep {ep+1}: train {hist[-1][1]} | val {hist[-1][2]}", flush=True)
    res[h] = hist
Path(__file__).with_name(f"gru_holdout_{EPOCAS}.json").write_text(json.dumps(res))
