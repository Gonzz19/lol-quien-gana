r"""Métricas de la evaluación final con IC por bootstrap por bloques de días, ventanas y semillas (§9.S.80).

Lo usa `scripts/analysis/evaluacion_test.py` (y su `--simulacro` sobre V). Todo se fija antes de mirar el test
(`docs/evaluacion_test/PROTOCOLO.md`):

* **Métricas** (`metricas`): AUC, acierto (umbral 0,5), log loss, ECE (10 deciles de $p$) y la línea base
  **"siempre rojo"** $= \overline{1 - y}$. Se fija rojo de antemano y no "el lado más frecuente", porque elegir el
  lado mirando el resultado del test ya es usar el test. El acierto se reporta como ventaja sobre esa línea base.
* **IC por bootstrap por bloques de días** (`bootstrap_bloques`): se remuestrean **días** (00:00 ART) con
  reposición, $B \ge 1000$, y se toma el percentil 2,5-97,5. Las partidas de un mismo día comparten parche,
  jugadores en racha y sesiones (el mismo jugador aparece en decenas de partidas del día), así que no son
  independientes; remuestrear partidas sueltas da un IC demasiado angosto. Por jugador no se puede: cada partida
  tiene 10 jugadores y los grupos se solapan (no hay partición).
* **Ventanas cronológicas** (`ventanas`): bloques consecutivos de `TAM_VENTANA` = 2.500 partidas en orden de
  fecha; si el último queda con menos de la mitad, se une al anterior. Con 2.500 partidas el error estándar del
  AUC es ≈ 0,011. **Pendiente** (`pendiente_auc`): MCO del AUC de cada ventana contra su fecha media, en AUC por
  semana, con IC por bootstrap de partidas dentro de cada ventana.
"""
from __future__ import annotations

from datetime import timedelta, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

ART = timezone(timedelta(hours=-3))
TAM_VENTANA = 2500
B_MIN = 1000
NOMBRES = ("auc", "acierto", "log_loss", "ece", "siempre_rojo", "ventaja_rojo")


def ece(y: np.ndarray, p: np.ndarray, n: int = 10) -> float:
    """Error de calibración esperado con 10 deciles de $p$ (igual que `modelo_en_vivo.ece`)."""
    g = pd.DataFrame({"p": p, "y": y, "d": pd.qcut(p, n, labels=False, duplicates="drop")}).groupby("d") \
          .agg(p=("p", "mean"), r=("y", "mean"), n=("y", "size"))
    return float((np.abs(g.p - g.r) * g.n).sum() / g.n.sum())


def metricas(y: np.ndarray, p: np.ndarray) -> dict:
    """AUC, acierto, log loss, ECE, "siempre rojo" y la ventaja del acierto sobre "siempre rojo"."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    acierto = float(((p > 0.5) == y).mean())
    rojo = float((y == 0).mean())
    return {"auc": float(roc_auc_score(y, p)), "acierto": acierto,
            "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])), "ece": ece(y, p),
            "siempre_rojo": rojo, "ventaja_rojo": acierto - rojo}


def dia_art(t_ms: np.ndarray) -> np.ndarray:
    """Día calendario (hora de Argentina) de cada instante en ms, como entero (días desde 1970)."""
    return ((np.asarray(t_ms, dtype=np.int64) - 3 * 3_600_000) // 86_400_000).astype(np.int64)


def bootstrap_bloques(y: np.ndarray, p: np.ndarray, grupos: np.ndarray, B: int = B_MIN, semilla: int = 0,
                      funcion=metricas) -> dict:
    """Métricas con IC 95 % por bootstrap de **bloques** (grupos enteros remuestreados con reposición).

    Devuelve `{métrica: (valor, lo, hi)}` y `n`, `n_grupos`, `B`. Un remuestreo sin las dos clases (posible con
    muy pocos grupos) se descarta para el AUC.
    """
    if B < 1:
        raise ValueError("B tiene que ser positivo")
    y, p, grupos = np.asarray(y).astype(int), np.asarray(p, dtype=np.float64), np.asarray(grupos)
    punto = funcion(y, p)
    _, inv = np.unique(grupos, return_inverse=True)
    orden = np.argsort(inv, kind="mergesort")
    cortes = np.flatnonzero(np.diff(inv[orden])) + 1
    miembros = np.split(orden, cortes)
    G = len(miembros)
    rng = np.random.default_rng(semilla)
    reps = {k: [] for k in punto}
    for _ in range(B):
        idx = np.concatenate([miembros[g] for g in rng.integers(0, G, size=G)])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        for k, v in funcion(yy, p[idx]).items():
            reps[k].append(v)
    out = {"n": len(y), "n_grupos": G, "B": B}
    for k, v in punto.items():
        r = np.asarray(reps[k])
        out[k] = (v, float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5)))
    return out


def bootstrap_partidas(y, p, B: int = B_MIN, semilla: int = 0, funcion=metricas) -> dict:
    """El IC de antes (partidas sueltas como si fueran independientes), para comparar con el de bloques."""
    return bootstrap_bloques(y, p, np.arange(len(y)), B, semilla, funcion)


def ventanas(t: np.ndarray, tam: int = TAM_VENTANA) -> np.ndarray:
    """Ventana (0, 1, ...) de cada partida: bloques consecutivos de `tam` partidas en orden de fecha.

    El último bloque, si tiene menos de `tam / 2`, se une al anterior. Empates de fecha: orden estable de entrada.
    """
    t = np.asarray(t)
    n = len(t)
    orden = np.argsort(t, kind="mergesort")
    k = np.arange(n) // tam
    nv = int(k.max()) + 1 if n else 0
    if nv > 1 and (k == nv - 1).sum() < tam / 2:
        k[k == nv - 1] = nv - 2
    out = np.empty(n, dtype=np.int64)
    out[orden] = k
    return out


def metricas_por_ventana(y, p, t, tam: int = TAM_VENTANA, B: int = B_MIN, semilla: int = 0) -> pd.DataFrame:
    """Métricas por ventana cronológica con IC 95 % (bootstrap de partidas dentro de la ventana).

    Dentro de una ventana (uno o dos días) no hay días suficientes para remuestrear por bloques.
    """
    y, p, t = np.asarray(y).astype(int), np.asarray(p, dtype=np.float64), np.asarray(t, dtype=np.int64)
    w = ventanas(t, tam)
    filas = []
    for k in range(int(w.max()) + 1):
        s = np.flatnonzero(w == k)
        b = bootstrap_partidas(y[s], p[s], B, semilla + k)
        fila = {"ventana": k, "n": len(s), "desde": int(t[s].min()), "hasta": int(t[s].max()),
                "t_medio": float(np.mean(t[s]))}
        for m in NOMBRES:
            fila[m], fila[f"{m}_lo"], fila[f"{m}_hi"] = b[m]
        filas.append(fila)
    return pd.DataFrame(filas)


def pendiente_auc(y, p, t, tam: int = TAM_VENTANA, B: int = B_MIN, semilla: int = 0) -> dict:
    """Pendiente del AUC por ventana contra la fecha media de la ventana (AUC por semana), con IC 95 %.

    IC: en cada réplica se remuestrean las partidas dentro de cada ventana, se recalcula el AUC de cada una y la
    recta. Con una sola ventana la pendiente no está definida (NaN).
    """
    y, p, t = np.asarray(y).astype(int), np.asarray(p, dtype=np.float64), np.asarray(t, dtype=np.int64)
    w = ventanas(t, tam)
    nv = int(w.max()) + 1
    idx = [np.flatnonzero(w == k) for k in range(nv)]
    x = np.array([t[s].mean() for s in idx]) / (7 * 86_400_000)          # semanas
    if nv < 2:
        return {"pendiente_semana": np.nan, "lo": np.nan, "hi": np.nan, "n_ventanas": nv}

    def recta(aucs):
        return float(np.polyfit(x - x.mean(), aucs, 1)[0])

    punto = recta(np.array([roc_auc_score(y[s], p[s]) for s in idx]))
    rng = np.random.default_rng(semilla)
    reps = []
    for _ in range(B):
        a = []
        for s in idx:
            r = s[rng.integers(0, len(s), size=len(s))]
            a.append(roc_auc_score(y[r], p[r]) if y[r].min() != y[r].max() else np.nan)
        a = np.array(a)
        if np.isfinite(a).all():
            reps.append(recta(a))
    return {"pendiente_semana": punto, "lo": float(np.percentile(reps, 2.5)), "hi": float(np.percentile(reps, 97.5)),
            "n_ventanas": nv}


def resumen_semillas(por_semilla: dict[int, dict], principal: int) -> dict:
    """Media, desvío (ddof = 1), mínimo, máximo y rango de cada métrica entre semillas; la principal aparte."""
    ks = [k for k in NOMBRES if k in next(iter(por_semilla.values()))]
    out = {"principal": principal, "semillas": sorted(por_semilla)}
    for k in ks:
        v = np.array([por_semilla[s][k] for s in sorted(por_semilla)], dtype=float)
        out[k] = {"principal": float(por_semilla[principal][k]), "media": float(v.mean()),
                  "desvio": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "min": float(v.min()),
                  "max": float(v.max()), "rango": float(v.max() - v.min())}
    return out


def correlacion_predicciones(preds: dict[int, np.ndarray], principal: int) -> dict:
    """Correlación de Pearson de las predicciones de cada semilla con la principal y la mínima entre pares;
    fracción de partidas donde el ganador predicho coincide con el de la principal."""
    ss = sorted(preds)
    P = np.vstack([preds[s] for s in ss])
    R = np.corrcoef(P)
    i0 = ss.index(principal)
    return {"corr_con_principal": {int(s): float(R[i0, j]) for j, s in enumerate(ss) if s != principal},
            "corr_min_pares": float(R[np.triu_indices(len(ss), 1)].min()) if len(ss) > 1 else 1.0,
            "mismo_ganador_que_principal": {int(s): float(((preds[s] > 0.5) == (preds[principal] > 0.5)).mean())
                                           for s in ss if s != principal},
            "max_abs_dp_con_principal": {int(s): float(np.abs(preds[s] - preds[principal]).max())
                                         for s in ss if s != principal}}
