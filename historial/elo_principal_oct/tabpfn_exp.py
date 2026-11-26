"""TabPFN v2 (pesos públicos, preentrenado; aprende en contexto) sobre las variables del GB final, 5 folds.

Uso: python tabpfn_exp.py [--max-train N] [--estimadores K]
"""
import argparse, pickle, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from tabpfn import TabPFNRegressor
from comun import AQUI, cargar, evaluar, registrar
from zoo import variables
P = (r"C:\Users\DarkA\.cache\huggingface\hub\models--Prior-Labs--TabPFN-v2-reg\snapshots"
     r"\02a44705ff7790b50cded404bc59adb94da456a1\tabpfn-v2-regressor.ckpt")
ap = argparse.ArgumentParser()
ap.add_argument("--max-train", type=int, default=10000)
ap.add_argument("--estimadores", type=int, default=4)
ap.add_argument("--tag", default="tabpfn")
ap.add_argument("--grid", default="", help="variables del filtro de grilla (tag de grilla.py)")
ap.add_argument("--caja", action="store_true", help="contexto del desarrollo → predice la caja (sin métricas)")
args = ap.parse_args()
c = cargar(); e = c["e"]; S = pickle.load(open(AQUI / "seq.pkl", "rb"))
X0 = variables(c, S); y = (e["z1"] - c["regla"]).to_numpy(np.float32)
SAL = pickle.load(open(AQUI / f"{args.grid}.pkl", "rb")) if args.grid else None


def matriz(k):
    if SAL is None:
        return X0.to_numpy(np.float32)
    g = SAL[k]
    G = pd.DataFrame({"g_media": g["media"] - c["regla"], "g_mediana": g["mediana"] - c["regla"], "g_sd": g["sd"],
                      "g_p10": g["p10"] - c["regla"], "g_p90": g["p90"] - c["regla"], "g_pm": g["p_master"]})
    return X0.join(G).to_numpy(np.float32)
dev, fold = ~c["caja"], c["fold"]
rng = np.random.default_rng(0)
t0 = time.time()
media, mediana = np.full(len(e), np.nan), np.full(len(e), np.nan)
for k in (["todo"] if args.caja else range(5)):
    X = matriz(k)
    tr = np.where(dev & (fold != k))[0] if not args.caja else np.where(dev)[0]
    te = np.where(dev & (fold == k))[0] if not args.caja else np.where(~dev)[0]
    if len(tr) > args.max_train:
        tr = rng.choice(tr, args.max_train, replace=False)
    m = TabPFNRegressor(model_path=P, device="cuda", n_estimators=args.estimadores, ignore_pretraining_limits=True,
                        random_state=0)
    m.fit(X[tr], y[tr])
    for i in range(0, len(te), 200):     # de a poco: la atención contra el contexto no entra en 8 GB de una
        tt = te[i:i + 200]
        out = m.predict(X[tt], output_type="full")
        media[tt], mediana[tt] = out["mean"], out["median"]
    print(f"fold {k} ({time.time() - t0:.0f} s)", flush=True)
for nombre, pr in (("media", media), ("mediana", mediana)):
    final = c["regla"] + pr
    tag = f"{args.tag}_{nombre}"
    if args.caja:
        np.save(AQUI / "preds" / f"{tag}_caja.npy", final.to_numpy()); print(f"{tag}: caja lista"); continue
    np.save(AQUI / "preds" / f"{tag}.npy", final.to_numpy())
    registrar(tag, evaluar(e, final, e.index[dev]),
              f"TabPFN v2, {args.max_train} filas de contexto, {args.estimadores} estimadores, {nombre} ({time.time() - t0:.0f} s)")
