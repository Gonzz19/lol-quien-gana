"""
Modelo del LP partida a partida, entrenado con el LP medido en vivo (PROJECT_SUMMARY.md §9.S.54).

Datos: `data/lp_vivo/lp_vivo.db` (§9.S.22), una fila por jugador y partida con el LP de antes
y de después. Objetivo: $\\Delta LP = z_{\\text{después}} - z_{\\text{antes}}$, en la escala
continua de `rating.lp_absoluto`.

Variables, por niveles acumulados (`NIVELES`):

* `resultado`: victoria y rango (Challenger, GM, Master, Diamond).
* `partida`: + LP de antes, distancia al lobby, a los compañeros y a los rivales medidos,
  W/L de la temporada, duración y posición. Todo existe también en `lol_las.db`.
* `historia`: + racha, partidas previas observadas y horas desde la anterior.
* `lp_previo`: + lo último que ganó en una victoria y perdió en una derrota. Es un techo:
  usa LP real anterior, que fuera de `lp_vivo` no existe, y por eso no entra al recorrido.

Evaluación (`evaluar`), siempre con folds por jugador (`elo_reverso.folds_por_jugador`):

* *Por partida:* MAE, RMSE, $R^2$ y $R^2$ dentro del resultado (cuánto explica además de
  ganar o perder).
* *Recorrido:* en cada cadena de partidas seguidas de un jugador (W + L sube de a 1 y el LP
  empalma), se parte del LP real de la primera y se encadenan las predicciones. El error en
  el paso $s$ es el del mapa de LP a $s$ partidas de la última medición.
"""

from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from src.features.elo_reverso import GRUPOS, folds_por_jugador

POSICIONES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
DEPENDEN_DE_Z = ["z_antes", "gap_lobby", "gap_equipo", "gap_rival"]
NIVELES: Dict[str, List[str]] = {}
NIVELES["resultado"] = ["win", "grupo"]
NIVELES["partida"] = NIVELES["resultado"] + DEPENDEN_DE_Z + [
    "lobby_conocidos", "partidas_temp", "wr_temp", "dur_min"] + [f"pos_{p}" for p in POSICIONES]
NIVELES["historia"] = NIVELES["partida"] + ["racha", "previas_obs", "horas_desde_previa"]
NIVELES["lp_previo"] = NIVELES["historia"] + ["ganancia_previa", "perdida_previa"]
SIN_RECORRIDO = {"lp_previo"}
ATIPICO_LP = 40     # |ΔLP| mayor: la cola sin explicar de §9.S.22


def _racha(win: pd.Series, jugador: pd.Series) -> pd.Series:
    """Racha antes de cada partida: +n si ganó las n anteriores observadas, −n si las perdió."""
    prev = win.groupby(jugador).shift(1)
    signo = prev.map({1: 1, 0: -1})
    corte = (prev != prev.groupby(jugador).shift(1)) | prev.isna()
    bloque = corte.groupby(jugador).cumsum()
    largo = signo.groupby([jugador, bloque]).cumcount() + 1
    return (signo * largo).fillna(0)


def dependientes_de_z(d: pd.DataFrame, z: pd.Series) -> pd.DataFrame:
    """Las variables que cambian con el LP propio: el LP y la distancia a lobby, compañeros y rivales."""
    return pd.DataFrame({"z_antes": z, "gap_lobby": d["lobby_lp"] - z, "gap_equipo": d["equipo_lp"] - z,
                         "gap_rival": d["rival_lp"] - z}, index=d.index)


def preparar(d: pd.DataFrame) -> pd.DataFrame:
    """Ordena por jugador y tiempo y agrega las variables de `NIVELES`, las cadenas y el paso."""
    d = d.sort_values(["puuid", "game_end"]).reset_index(drop=True).copy()
    j = d["puuid"]
    d["grupo"] = d["tier_antes"].map({g: i for i, g in enumerate(GRUPOS)}).astype(float)

    # Compañeros y rivales medidos (sin el propio), con el LP de antes.
    eq = d.groupby(["match_id", "team_id"])["z_antes"].agg(["sum", "count"])
    tot = d.groupby("match_id")["z_antes"].agg(["sum", "count"])
    s_eq = eq["sum"].reindex(pd.MultiIndex.from_frame(d[["match_id", "team_id"]])).to_numpy()
    n_eq = eq["count"].reindex(pd.MultiIndex.from_frame(d[["match_id", "team_id"]])).to_numpy()
    s_tot, n_tot = tot["sum"].reindex(d["match_id"]).to_numpy(), tot["count"].reindex(d["match_id"]).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        d["equipo_lp"] = np.where(n_eq > 1, (s_eq - d["z_antes"]) / (n_eq - 1), np.nan)
        d["rival_lp"] = np.where(n_tot - n_eq > 0, (s_tot - s_eq) / (n_tot - n_eq), np.nan)
    d = pd.concat([d.drop(columns=[c for c in DEPENDEN_DE_Z if c in d]), dependientes_de_z(d, d["z_antes"])], axis=1)

    d["partidas_temp"] = d["wins_antes"] + d["losses_antes"]
    d["wr_temp"] = d["wins_antes"] / d["partidas_temp"].where(d["partidas_temp"] > 0)
    d["dur_min"] = d["duracion_s"] / 60
    for p in POSICIONES:
        d[f"pos_{p}"] = (d["team_position"] == p).astype(float)

    d["racha"] = _racha(d["win"], j)
    d["previas_obs"] = d.groupby(j).cumcount()
    d["horas_desde_previa"] = (d["game_end"] - d.groupby(j)["game_end"].shift(1)) / 3.6e6
    # Último ΔLP observado en una victoria (derrota) estrictamente antes de esta partida.
    for col, res in (("ganancia_previa", 1), ("perdida_previa", 0)):
        ult = d["delta_lp"].where(d["win"] == res).groupby(j).ffill()
        d[col] = ult.groupby(j).shift(1)

    # Cadenas: la partida sigue a la anterior del mismo jugador sin huecos.
    n = d["partidas_temp"]
    seguida = (j == j.shift(1)) & (n == n.shift(1) + 1) & (d["z_antes"] == d["z_despues"].shift(1))
    d["cadena"] = (~seguida).cumsum() - 1
    d["paso"] = d.groupby("cadena").cumcount()
    return d


def matriz(d: pd.DataFrame, cols: List[str]) -> np.ndarray:
    return d[cols].to_numpy(dtype=float)


# --- Modelos: fit(d, y) / predict(d), con d ya preparado ------------------------------------

class Regla:
    """ΔLP medio por (rango, resultado); si un rango no se vio, el medio del resultado."""

    def __init__(self, cols=None):
        self.cols = cols

    def fit(self, d, y):
        t = pd.DataFrame({"g": d["grupo"].to_numpy(), "w": d["win"].to_numpy(), "y": np.asarray(y)})
        self.por_grupo = t.groupby(["g", "w"])["y"].mean().to_dict()
        self.por_win = t.groupby("w")["y"].mean().to_dict()
        return self

    def predict(self, d):
        return np.array([self.por_grupo.get((g, w), self.por_win.get(w, 0.0))
                         for g, w in zip(d["grupo"], d["win"])])


class LinealPorResultado:
    """Ridge separado para victorias y derrotas (todas las variables interactúan con el resultado)."""

    def __init__(self, cols, alpha=10.0):
        self.cols, self.alpha = [c for c in cols if c != "win"], alpha

    def fit(self, d, y):
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        y = np.asarray(y)
        self.m = {}
        for w in (0, 1):
            k = (d["win"] == w).to_numpy()
            self.m[w] = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                                      Ridge(alpha=self.alpha)).fit(matriz(d[k], self.cols), y[k])
        return self

    def predict(self, d):
        out = np.zeros(len(d))
        for w, m in self.m.items():
            k = (d["win"] == w).to_numpy()
            if k.any():
                out[k] = m.predict(matriz(d[k], self.cols))
        return out


class GB:
    """LightGBM con pérdida de Huber ($\alpha = 8$ LP) por defecto.

    L1 predice la mediana (+20 al ganar) y la media es +21,6 por la cola de ganancias grandes:
    −0,86 LP por partida que al encadenar suma −10 LP a las 8–15 partidas. L2 no tiene sesgo
    pero erra más por partida. Huber queda en el medio y es el que menos erra en el recorrido.
    """

    def __init__(self, cols, objective="huber", **kw):
        self.cols = cols
        self.kw = dict(objective=objective, alpha=8.0, n_estimators=400, learning_rate=0.03, num_leaves=15,
                       min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                       random_state=0, verbose=-1)
        self.kw.update(kw)

    def fit(self, d, y):
        from lightgbm import LGBMRegressor
        self.m = LGBMRegressor(**self.kw).fit(matriz(d, self.cols), np.asarray(y))
        return self

    def predict(self, d):
        return self.m.predict(matriz(d, self.cols))


MODELOS: Dict[str, Callable] = {"regla": Regla, "lineal": LinealPorResultado, "gb": GB,
                                 "gb_l2": lambda cols: GB(cols, objective="l2")}


# --- Evaluación ------------------------------------------------------------------------------

def metricas(y, pred, win) -> Dict[str, float]:
    y, pred, win = np.asarray(y, float), np.asarray(pred, float), np.asarray(win)
    e = pred - y
    sst = ((y - y.mean()) ** 2).sum()
    sst_dentro = sum(((y[win == w] - y[win == w].mean()) ** 2).sum() for w in np.unique(win))
    sse = (e ** 2).sum()
    tip = np.abs(y) <= ATIPICO_LP
    return {"n": len(y), "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
            "r2": float(1 - sse / sst) if sst > 0 else np.nan,
            "r2_dentro": float(1 - sse / sst_dentro) if sst_dentro > 0 else np.nan,
            "mae_tipico": float(np.abs(e[tip]).mean()) if tip.any() else np.nan}


def recorrido(d: pd.DataFrame, modelo) -> pd.DataFrame:
    """Encadena las predicciones dentro de cada cadena desde el LP real de su primera partida.

    Devuelve `z_hat_antes` y `z_hat_despues` por fila; en el paso 0, `z_hat_antes` es el real.
    """
    d = d.sort_values(["cadena", "paso"])
    z_hat = pd.Series(np.nan, index=d.index)
    z_fin = pd.Series(np.nan, index=d.index)
    prev = None
    for s in range(int(d["paso"].max()) + 1):
        k = d.index[d["paso"] == s]
        if s == 0:
            z = d.loc[k, "z_antes"]
        else:
            z = prev.reindex(d.loc[k, "cadena"]).set_axis(k)
        x = d.loc[k].copy()
        x[DEPENDEN_DE_Z] = dependientes_de_z(x, z)
        fin = z + modelo.predict(x)
        z_hat[k], z_fin[k] = z, fin
        prev = pd.Series(fin.to_numpy(), index=d.loc[k, "cadena"].to_numpy())
    return pd.DataFrame({"z_hat_antes": z_hat, "z_hat_despues": z_fin})


def evaluar(d: pd.DataFrame, niveles=None, modelos=None, k: int = 5) -> Dict[str, pd.DataFrame]:
    """Predicciones fuera de fold (folds por jugador) por nivel y modelo, y el recorrido.

    Devuelve `oof` (una columna `pred_<modelo>_<nivel>` por combinación) y `recorrido`
    (`zhat_<modelo>_<nivel>` = LP encadenado después de la partida).
    """
    niveles = niveles or list(NIVELES)
    modelos = modelos or list(MODELOS)
    d = d.copy()
    fold = folds_por_jugador(d["puuid"], k)
    y = d["delta_lp"].to_numpy()
    for nivel in niveles:
        cols = NIVELES[nivel]
        for nombre in modelos:
            if nombre == "regla" and nivel != niveles[0]:
                continue   # la regla no usa variables: una sola vez
            col = f"pred_{nombre}_{nivel}" if nombre != "regla" else "pred_regla"
            zc = f"zhat_{nombre}_{nivel}" if nombre != "regla" else "zhat_regla"
            d[col] = np.nan
            d[zc] = np.nan
            for f in range(k):
                tr, te = fold != f, fold == f
                if not te.any():
                    continue
                m = MODELOS[nombre](cols).fit(d[tr], y[tr])
                d.loc[te, col] = m.predict(d[te])
                if nivel not in SIN_RECORRIDO:
                    r = recorrido(d[te], m)
                    d.loc[r.index, zc] = r["z_hat_despues"]
    return {"oof": d}


def tabla_partida(d: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for c in [c for c in d.columns if c.startswith("pred_")]:
        filas.append({"modelo": c[5:], **metricas(d["delta_lp"], d[c], d["win"])})
    return pd.DataFrame(filas).set_index("modelo")


TRAMOS = [(1, 1), (2, 3), (4, 7), (8, 15), (16, 10_000)]


def tabla_recorrido(d: pd.DataFrame) -> pd.DataFrame:
    """MAE del LP encadenado según cuántas partidas van desde el último LP real (paso + 1)."""
    filas = []
    s = d["paso"] + 1
    for c in [c for c in d.columns if c.startswith("zhat_")]:
        fila = {"modelo": c[5:]}
        e = (d[c] - d["z_despues"]).abs()
        for a, b in TRAMOS:
            k = (s >= a) & (s <= b) & e.notna()
            fila[f"{a}-{b}" if a != b else f"{a}"] = float(e[k].mean()) if k.any() else np.nan
        fila["todas"] = float(e.mean())
        filas.append(fila)
    t = pd.DataFrame(filas).set_index("modelo")
    t.columns = [c.replace("16-10000", "16+") for c in t.columns]
    return t
