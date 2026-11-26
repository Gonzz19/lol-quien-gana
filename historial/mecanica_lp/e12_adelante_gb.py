"""Mecánica + corrección supervisada, hacia adelante: ¿cuánto queda para el objetivo de métricas?
El residuo se aprende con 5 folds por jugador, sobre desarrollo. Sin la foto 2 del propio jugador."""
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor as H

sys.stdout.reconfigure(encoding="utf-8")
SP = Path(sys.argv[1]); sys.path.insert(0, str(Path(__file__).resolve().parent))
from mec import metricas  # noqa: E402

V = SP / "vivo"
X = pd.read_pickle(V / "X_mec.pkl"); pob = pd.read_pickle(V / "pob.pkl")
ids = pob.index.to_numpy(); y1 = pob.z1.to_numpy(); y2 = pob.z2.to_numpy()
grp = pob.tier2.map({"CHALLENGER": 0, "GRANDMASTER": 1, "MASTER": 2}).fillna(3).astype(int).to_numpy()


def rho(v):
    return sum(spearmanr(v[grp == k], y2[grp == k]).statistic * (grp == k).sum() for k in range(4)) / len(y2)


fold = np.array([zlib.crc32(str(i).encode()) % 5 for i in ids])
F = pd.DataFrame({"z1": y1, "fwd": X.z_fwd - y1, "n": X.n_obs, "dob": X.dob, "gap_ini": X.M_ini - y1, "gap_fin": X.M_fin - y1, "gap_all": X.M_all - y1,
                  "hueco": X.hueco_max, "d_fin": X.dias_fin, "d_ini": X.dias_ini, "N1": pob.partidas1.to_numpy(), "wr1": (pob.wins1 / pob.partidas1.clip(lower=1)).to_numpy(),
                  "obs_w": pob.obs_w.to_numpy()}, index=ids)
print(f"{'Mecánica sola':38s}", {k: round(v, 3) for k, v in metricas(y2, y1, X.z_fwd.to_numpy()).items() if k != 'n'}, "ρ", round(rho(X.z_fwd.to_numpy()), 3))
for nom, perd in (("Mecánica + GB (L1)", "absolute_error"), ("Mecánica + GB (L2)", "squared_error")):
    oof = np.zeros(len(ids))
    for k in range(5):
        m = H(loss=perd, max_iter=300, learning_rate=.04, max_leaf_nodes=15, min_samples_leaf=50, early_stopping=False, random_state=0)
        tr = fold != k
        m.fit(F[tr], (y2 - X.z_fwd.to_numpy())[tr])
        oof[~tr] = m.predict(F[~tr])
    v = X.z_fwd.to_numpy() + oof
    print(f"{nom:38s}", {k_: round(v_, 3) for k_, v_ in metricas(y2, y1, v).items() if k_ != 'n'}, "ρ", round(rho(v), 3),
          "por rango", {n: round(float(np.abs(v - y2)[grp == k].mean()), 1) for k, n in enumerate(["CH", "GM", "M", "D"])})
    np.save(V / f"fwd_gb_{perd}.npy", v)
a = (np.load(V / "fwd_gb_absolute_error.npy") + np.load(V / "fwd_gb_squared_error.npy")) / 2
print(f"{'Promedio L1 y L2':38s}", {k_: round(v_, 3) for k_, v_ in metricas(y2, y1, a).items() if k_ != 'n'}, "ρ", round(rho(a), 3))
print("Control sin la mecánica (mismas variables menos fwd, residuo de 'no cambió'):")
oof = np.zeros(len(ids))
c = [x for x in F.columns if x not in ("fwd", "dob")]
for k in range(5):
    m = H(loss="absolute_error", max_iter=300, learning_rate=.04, max_leaf_nodes=15, min_samples_leaf=50, early_stopping=False, random_state=0)
    tr = fold != k
    m.fit(F.loc[tr, c], (y2 - y1)[tr])
    oof[~tr] = m.predict(F.loc[~tr, c])
v = y1 + oof
print(f"{'GB sin mecánica':38s}", {k_: round(v_, 3) for k_, v_ in metricas(y2, y1, v).items() if k_ != 'n'}, "ρ", round(rho(v), 3))
