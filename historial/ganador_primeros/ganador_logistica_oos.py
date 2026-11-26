"""Experto logístico con predicciones fuera de muestra para ganador_stacking.py, sin torch (§9.S.32).

Misma logística, mismos bloques y mismo formato de salida que `ganador_red_jugador.py --oos K`
(sin la red): el ajuste de la config se parte en K bloques por fecha y el bloque k >= 1 se
predice con la logística ajustada solo con los bloques anteriores; V se predice con todo el
ajuste. Variables: las de la config C de §9.S.29 (comunes + margen y duos + márgenes por grupo)
más los bloques extra que se pidan con --extra archivo.pkl:clave (los `nuevo` de
ganador_mmr.py, ganador_cambio.py...).

Escribe <guardar>_oos.npz con las claves que lee ganador_stacking.py. Solo data/splits/train."""
import argparse, pickle, time, warnings
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
AN = REPO / "data" / "analysis"
ap = argparse.ArgumentParser()
ap.add_argument("--feats", default=str(AN / "ganador_feats_grande.pkl"))
ap.add_argument("--margen", default=str(AN / "ganador_margen_duos.pkl"))
ap.add_argument("--grupos", default=str(AN / "ganador_margen_grupos.pkl"))
ap.add_argument("--poisson", help="salida de ganador_poisson.py: suma las diferencias de ataque y defensa por partida")
ap.add_argument("--extra", nargs="*", default=[], help="archivo.pkl:clave de su dict 'nuevo'")
ap.add_argument("--config", default="C")
ap.add_argument("--oos", type=int, default=5)
ap.add_argument("--guardar", required=True, help="ruta base; escribe <ruta>_oos.npz")
args = ap.parse_args()
cfg = args.config

d = pickle.load(open(args.feats, "rb"))
feat, mids, y, t, val, fit = d["feat"], d["mids"], d["y"], d["t"], d["val"], d["masks"][cfg]
X = [feat[k] for k in ["onehot", "experiencia", "cobertura", "dificultad", "matchups", "sinergias"]]
del d, feat
M = pickle.load(open(args.margen, "rb"))
assert (M["mids"] == mids).all()
X += [M["nuevo"][k] for k in ("winrate previo", "margen acum", "margen v20", "duos")]
del M
G = pickle.load(open(args.grupos, "rb"))
assert (G["mids"] == mids).all()
X += list(G["nuevo"].values())
del G
if args.poisson:                                   # como ganador_red_jugador.py --poisson
    P = pickle.load(open(args.poisson, "rb"))
    assert (P["mids"] == mids).all()
    X.append(np.hstack(list(P["poisson"].values())))
    print(f"+ Poisson: {X[-1].shape[1]} columnas")
    del P
for item in args.extra:
    ruta, clave = item.rsplit(":", 1)
    E = pickle.load(open(ruta, "rb"))
    assert (E["mids"] == mids).all(), f"{ruta} no corresponde a estas partidas"
    X.append(E["nuevo"][clave])
    print(f"+ {clave} ({Path(ruta).name}): {E['nuevo'][clave].shape[1]} columnas")
X = np.hstack(X)


def logistica(fit_idx, pred_idx, tambien_ajuste=False):
    """Con `tambien_ajuste`, devuelve además la predicción dentro de la muestra sobre `fit_idx` (Train)."""
    sc = StandardScaler(with_mean=False).fit(X[fit_idx])
    m = LogisticRegression(C=0.05, max_iter=5000).fit(sc.transform(X[fit_idx]), y[fit_idx])
    p = m.predict_proba(sc.transform(X[pred_idx]))[:, 1]
    return (p, m.predict_proba(sc.transform(X[fit_idx]))[:, 1]) if tambien_ajuste else p


t0 = time.time()
idx, vi = np.where(fit)[0], np.where(val)[0]
p_val, p_fit = logistica(idx, vi, tambien_ajuste=True)
print(f"V: AUC {roc_auc_score(y[vi], p_val):.4f}  acc {accuracy_score(y[vi], p_val > .5):.4f}  ({time.time() - t0:.0f}s)", flush=True)

# los mismos bloques que ganador_red_jugador.py --oos
bloque = np.searchsorted(np.quantile(t[idx], np.linspace(0, 1, args.oos + 1)[1:-1]), t[idx], side="right")
partes = {"idx": [], "bloque": [], "logistica": []}
for k in range(1, args.oos):
    antes, este = idx[bloque < k], idx[bloque == k]
    assert t[antes].max() <= t[este].min()
    partes["idx"].append(este); partes["bloque"].append(np.full(len(este), k))
    partes["logistica"].append(logistica(antes, este))
    print(f"  bloque {k}: ajuste {len(antes):>7d}, predice {len(este):>6d}  AUC {roc_auc_score(y[este], partes['logistica'][-1]):.4f}"
          f"  (desde {np.datetime64(int(t[este].min()), 'ms').astype('datetime64[D]')})", flush=True)
o = {k: np.concatenate(v) for k, v in partes.items()}
salida = Path(args.guardar)
np.savez(salida.with_name(salida.stem + "_oos.npz"),
         **{f"{cfg}_oos_{k}": v for k, v in o.items()},
         **{f"{cfg}_oos_y": y[o["idx"]], f"{cfg}_val_idx": vi, f"{cfg}_val_y": y[vi], f"{cfg}_val_t": t[vi],
            f"{cfg}_val_logistica": p_val, f"{cfg}_fit_idx": idx, f"{cfg}_fit_logistica": p_fit})
print(f"escrito {salida.with_name(salida.stem + '_oos.npz')}")
