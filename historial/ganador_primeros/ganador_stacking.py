"""Stacking final de quién gana: logística + red por jugador (§9.S.26).

Entrada: los <nombre>_oos.npz de `ganador_red_jugador.py --oos K --guardar ...`. Cada uno trae,
para sus expertos (logística y redes):
    * predicciones FUERA DE MUESTRA sobre el ajuste: el bloque temporal k predicho con modelos
      que solo vieron los bloques anteriores;
    * predicciones en V de los modelos ajustados con todo lo anterior a V.
El meta-modelo (logística sobre los logits de los expertos) se ajusta con las primeras y se
evalúa en V: nunca ve predicciones que un experto hizo sobre sus propias partidas de ajuste,
y V no interviene ni en los expertos ni en los pesos.
Se compara contra cada experto solo y contra el promedio simple de logits (sin pesos que
ajustar), con la diferencia pareada de AUC por bootstrap.
Solo data/splits/train (V es la validación de siempre, 16/09-27/09)."""
import argparse, itertools, warnings
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
AN = REPO / "data" / "analysis"
ap = argparse.ArgumentParser()
ap.add_argument("--expertos", nargs="+", default=[f"margen={AN / 'ganador_red_jugador_margen_oos.npz'}",
                                                  f"Poisson={AN / 'ganador_red_jugador_poisson_oos.npz'}"],
                help="etiqueta=archivo _oos.npz; los expertos de cada archivo se nombran '<experto> (<etiqueta>)'")
ap.add_argument("--config", default="C")
ap.add_argument("--ultimos", type=int, default=2, help="sensibilidad: meta ajustado solo con los últimos N bloques")
args = ap.parse_args()
cfg = args.config

logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
sigm = lambda z: 1 / (1 + np.exp(-z))
Zo, Zv, base = {}, {}, None
for item in args.expertos:
    etiqueta, ruta = item.split("=", 1)
    if not Path(ruta).exists():
        print(f"(sin {ruta}: se saltea '{etiqueta}')")
        continue
    f = np.load(ruta)
    if base is None:
        base = {k: f[f"{cfg}_{k}"] for k in ("oos_idx", "oos_bloque", "oos_y", "val_idx", "val_y")}
    assert (f[f"{cfg}_oos_idx"] == base["oos_idx"]).all() and (f[f"{cfg}_val_idx"] == base["val_idx"]).all(), \
        f"{ruta}: no son las mismas partidas ni los mismos bloques"
    for k in f.files:
        if k.startswith(f"{cfg}_oos_") and k[len(cfg) + 5:] not in ("idx", "bloque", "y"):
            nombre = f"{k[len(cfg) + 5:].replace('_', ' ').replace('logistica', 'logística')} ({etiqueta})"
            Zo[nombre], Zv[nombre] = logit(f[k]), logit(f[f"{cfg}_val_{k[len(cfg) + 5:]}"])
if base is None:
    raise SystemExit("no hay ningún archivo de expertos")
yo, yv, bl = base["oos_y"], base["val_y"], base["oos_bloque"]
nombres = list(Zo)
B = np.random.default_rng(0).integers(0, len(yv), (200, len(yv)))
print(f"fuera de muestra: {len(yo)} partidas en los bloques {bl.min()}-{bl.max()}; V: {len(yv)} partidas\n")

print("=== Cada experto solo: AUC fuera de muestra por bloque y en V ===")
print(f"{'':34s}" + "".join(f"{'bloque ' + str(b):>10s}" for b in np.unique(bl)) + f"{'V':>9s}{'acc V':>8s}{'logloss V':>11s}")
for n in nombres:
    print(f"{n:34s}" + "".join(f"{roc_auc_score(yo[bl == b], Zo[n][bl == b]):>10.4f}" for b in np.unique(bl)) +
          f"{roc_auc_score(yv, Zv[n]):>9.4f}{accuracy_score(yv, Zv[n] > 0):>8.3f}{log_loss(yv, sigm(Zv[n])):>11.4f}")


def informe(nombre, p, refs, pesos=""):
    aucs = np.array([roc_auc_score(yv[b], p[b]) for b in B])
    extra = ""
    for r in refs:
        dif = aucs - np.array([roc_auc_score(yv[b], Zv[r][b]) for b in B])
        extra += f"  vs {r.split(' (')[0][:9]} {roc_auc_score(yv, p) - roc_auc_score(yv, Zv[r]):+.4f} [{np.quantile(dif, .025):+.4f};{np.quantile(dif, .975):+.4f}]"
    print(f"  {nombre:30s} acc {accuracy_score(yv, p > .5):.3f}  AUC {roc_auc_score(yv, p):.4f}  logloss {log_loss(yv, p):.4f}{extra}{pesos}", flush=True)


def meta(grupo, sel):
    """logística sobre los logits de los expertos, ajustada con las partidas fuera de muestra `sel`"""
    m = LogisticRegression(C=1e4, max_iter=5000).fit(np.c_[[Zo[n][sel] for n in grupo]].T, yo[sel])
    return m.predict_proba(np.c_[[Zv[n] for n in grupo]].T)[:, 1], \
        "  pesos " + " / ".join(f"{w:.2f}" for w in m.coef_[0]) + f"  cte {m.intercept_[0]:+.3f}"


# combinaciones: logística + red de cada archivo, las mismas cruzadas, y todos juntos
etiquetas = sorted({n.split(" (")[1][:-1] for n in nombres}, key=[n.split(" (")[1][:-1] for n in nombres].index)
de = lambda et: [n for n in nombres if n.endswith(f"({et})")]
grupos = [de(et) for et in etiquetas]
if len(etiquetas) > 1:
    grupos += [list(g) for g in itertools.product(*[de(et) for et in etiquetas])] + [nombres]
recientes = bl > bl.max() - args.ultimos
for grupo in grupos:
    if len(grupo) < 2:
        continue
    refs = list(dict.fromkeys([grupo[0]] + [n for n in grupo if n.startswith("red")][:1]))
    print(f"\n=== {' + '.join(grupo)} ===")
    for n in grupo:
        informe(n, sigm(Zv[n]), [r for r in refs if r != n])
    informe("promedio simple de logits", sigm(np.mean([Zv[n] for n in grupo], axis=0)), refs)
    p, w = meta(grupo, np.ones(len(yo), bool))
    informe("stacking, todos los bloques", p, refs, w)
    p, w = meta(grupo, recientes)
    informe(f"stacking, últimos {args.ultimos} bloques", p, refs, w)
