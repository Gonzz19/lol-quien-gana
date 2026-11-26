"""Lectura de resultados ya guardados y bootstrap para los IC del informe.

Nada de acá ajusta modelos: se leen las predicciones y métricas que escriben los scripts vigentes.

| Qué | Archivo | Lo escribe |
| :--- | :--- | :--- |
| Resultado, fecha y partición de cada partida del train | `data/cache/ganador/variables_partida.npz` | `src/ganador` (§9.S.73) |
| Predicción del modelo previo en V Master+ | `data/cache/ganador/pred_v.npz` (`p_v`, `idx_v`) | `scripts/analysis/modelo_previo.py` |
| Métricas del modelo en vivo con IC | `data/analysis/en_vivo/metricas.csv` | `scripts/analysis/modelo_en_vivo.py` |
| Predicciones en vivo con `mids` | `data/cache/ganador/pred_en_vivo.npz` | `scripts/analysis/modelo_en_vivo.py` |
| Huella de estilo | `data/analysis/huella/mejoras_red_habitos.csv`, `mejoras_parte2.csv` | `huella_mejoras.py` (§9.S.77) |
| Matchmaking | `data/analysis/matchmaking/swaps_resultados.parquet` | `matchmaking_swaps.py` (§9.S.77) |
| Conteos del test (solo el manifiesto) | `data/splits/split_manifest.json` | `export_to_csv.py --target splits` |
| Evaluación única en el test (cuando exista) | `data/analysis/test_final/` | `scripts/analysis/evaluacion_test.py` |

**El test no se lee**: de `data/splits/test/` solo se usa el manifiesto con los conteos.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
CACHE = DATA / "cache" / "ganador"
VARIABLES = CACHE / "variables_partida.npz"
PRED_V = CACHE / "pred_v.npz"
PRED_EN_VIVO = CACHE / "pred_en_vivo.npz"
METRICAS_EN_VIVO = DATA / "analysis" / "en_vivo" / "metricas.csv"
HUELLA = DATA / "analysis" / "huella"
SWAPS = DATA / "analysis" / "matchmaking" / "swaps_resultados.parquet"
MANIFIESTO = DATA / "splits" / "split_manifest.json"
TEST_FINAL = DATA / "analysis" / "test_final"

FOTO1 = pd.Timestamp("2026-09-04 22:51:22")   # foto 1 del ranking (UTC), §9.S.68
CORTE_V = pd.Timestamp("2026-09-16")           # inicio de V
CORTE_TEST = pd.Timestamp("2026-09-27")        # inicio del test (manifiesto)
B = 1000                                        # réplicas de bootstrap


# --- partidas del train ------------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def partidas() -> pd.DataFrame:
    """Una fila por partida usable del train: `mid`, `t` (fecha UTC), `y` (1 = gana azul) y el tramo.

    Tramos (`src/ganador/datos.py`): `ajuste` = anterior al 16/09; `A1`-`A4` = posteriores a la foto 1 y anteriores
    a V, en 4 bloques de igual tamaño por fecha (subconjunto de `ajuste`); `V` = ventana 16/09-27/09 con lobby Master+;
    `Diamond-V` = el resto de esa ventana (control).
    """
    from src.models.validacion_temporal import bloques_por_fecha, particion_v_master

    z = np.load(VARIABLES, allow_pickle=True)
    t, val0, tnum = z["t"], z["val"].astype(bool), z["tnum"]
    val, diam_v = particion_v_master(val0, tnum)
    master = particion_v_master(np.ones(len(t), bool), tnum)[0]
    df = pd.DataFrame({"mid": z["mids"].astype(str), "t": pd.to_datetime(t, unit="ms"),
                       "y": z["y"].astype(int), "master": master})
    df["tramo"] = np.where(val, "V", np.where(diam_v, "Diamond-V", "ajuste"))
    post = (df["t"] >= FOTO1).to_numpy() & ~val0
    bloque = np.full(len(df), "", dtype=object)
    ai = np.flatnonzero(post)
    bloque[ai] = np.array(["A1", "A2", "A3", "A4"])[bloques_por_fecha(t[ai], 4)]
    df["bloque"] = bloque
    return df


def manifiesto() -> dict:
    return json.loads(MANIFIESTO.read_text(encoding="utf-8"))


def prediccion_previo_v() -> pd.DataFrame:
    """Probabilidad de que gane el azul según el modelo previo, en V Master+, con su resultado y fecha."""
    z = np.load(PRED_V)
    P = partidas()
    df = P.iloc[z["idx_v"]][["mid", "t", "y"]].copy()
    assert (P["tramo"].iloc[z["idx_v"]] == "V").all(), "pred_v no está alineado con V"
    df["p"] = z["p_v"]
    return df.reset_index(drop=True)


def prediccion_en_vivo(m: int) -> pd.DataFrame:
    """Predicción del modelo en vivo al minuto `m` en V Master+ (partidas vivas en `m`), unida por `mid`."""
    z = np.load(PRED_EN_VIVO)
    n = int(z[f"n_v_{m}"])
    df = pd.DataFrame({"mid": z[f"mids_{m}"][:n].astype(str), "p": z[f"p_{m}"][:n].astype(float)})
    P = partidas().set_index("mid")
    df = df.join(P[["t", "y", "tramo"]], on="mid")
    assert (df["tramo"] == "V").all(), "pred_en_vivo: las primeras n_v filas no son V"
    return df.drop(columns="tramo")


def metricas_en_vivo() -> pd.DataFrame:
    return pd.read_csv(METRICAS_EN_VIVO)


# --- bootstrap -------------------------------------------------------------------------------------------------

def pesos_bootstrap(n: int, b: int = B, semilla: int = 0) -> np.ndarray:
    """Matriz (b, n) con cuántas veces entra cada partida en cada réplica (remuestreo con reemplazo)."""
    rng = np.random.default_rng(semilla)
    return np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(b)]).astype(np.float32)


def auc_ponderado(y: np.ndarray, p: np.ndarray, W: np.ndarray) -> np.ndarray:
    """AUC de cada réplica (fila de W) en O(n): suma de rangos con pesos sobre `p` ordenado."""
    o = np.argsort(p, kind="stable")
    y = y[o]
    _, inicio, grupo = np.unique(p[o], return_index=True, return_inverse=True)   # empates: grupos de igual p
    out = []
    for i in range(0, len(W), 200):                      # por tandas, para no duplicar la matriz entera en memoria
        w = W[i:i + 200, o]
        wn, wp = w * (1 - y), w * y
        acum = np.cumsum(wn, axis=1) - wn                # negativos estrictamente antes (en el orden)
        neg_grupo = np.add.reduceat(wn, inicio, axis=1)  # negativos de cada grupo de empate
        neg_antes = acum[:, inicio][:, grupo] + 0.5 * neg_grupo[:, grupo]
        out.append((wp * neg_antes).sum(1) / (wp.sum(1) * wn.sum(1)))
    return np.concatenate(out)


def resumen(y: np.ndarray, p: np.ndarray, b: int = B, semilla: int = 0) -> dict:
    """AUC, acierto, log loss, ECE (10 deciles) y 'siempre rojo', cada uno con IC 95 % por bootstrap de partidas."""
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    W = pesos_bootstrap(len(y), b, semilla)
    ok = ((p > 0.5) == y).astype(float)
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    dec = pd.qcut(p, 10, labels=False)

    def ece(w):
        g = pd.DataFrame({"d": dec, "w": w, "wp": w * p, "wy": w * y}).groupby("d").sum()
        return float((np.abs(g.wp - g.wy)).sum() / g.w.sum())

    reps = {
        "auc": auc_ponderado(y, p, W),
        "acierto": W @ ok / W.sum(1),
        "log_loss": W @ ll / W.sum(1),
        "siempre_rojo": W @ (1 - y) / W.sum(1),
    }
    punto = {"auc": float(auc_ponderado(y, p, np.ones((1, len(y))))[0]), "acierto": ok.mean(),
             "log_loss": ll.mean(), "siempre_rojo": 1 - y.mean(), "ece": ece(np.ones(len(y)))}
    out = {k: (punto[k], *np.percentile(v, [2.5, 97.5])) for k, v in reps.items()}
    out["ece"] = (punto["ece"], np.nan, np.nan)   # su IC por bootstrap está sesgado hacia arriba (§9.S.78)
    out["n"] = len(y)
    return out


def proporcion_ic(k: float, n: float, z: float = 1.96) -> tuple[float, float, float]:
    """Proporción con IC de Wilson."""
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, c - h, c + h


# --- test (solo si existe la evaluación única) ------------------------------------------------------------------

def resultados_test() -> dict[str, pd.DataFrame]:
    """Los CSV de `data/analysis/test_final/` por nombre de archivo; vacío si la evaluación todavía no se corrió."""
    if not TEST_FINAL.is_dir():
        return {}
    return {f.stem: pd.read_csv(f) for f in sorted(TEST_FINAL.glob("*.csv"))}
