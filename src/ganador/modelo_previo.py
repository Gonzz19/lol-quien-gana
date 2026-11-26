"""Modelo previo de quién gana (escalón 9, `PROJECT_SUMMARY.md` §9.S.52 y §9.S.68).

Regresión logística con regularización L2 (C = 0,05) sobre tres bloques de columnas:

1. **draft**: campeón × línea con signo (+1 azul, −1 rojo), disperso (`D["DRAFT"]`);
2. **85 variables previas por jugador**, azul − rojo por línea (`D["Xd"]`, 425 columnas);
3. **extras** del escalón 9 (`variables.matriz_extras`: margen por campeón y línea, nivel histórico,
   W/L de la foto 1, contracruces, winrate por línea, Kalman-Elo, oro @15 predicho, margen esperado).

Los bloques 2 y 3 se estandarizan con media y desvío de las **filas de entrenamiento** (nunca con V).
El modelo de V se ajusta solo con partidas Master+ anteriores al 16/09 (`D["fi_v"]` ∩ `D["master"]`).

Es la misma cuenta que hacía `scripts/analysis/brainstorm_base.ajustar_predecir` con
`con_mapa=False, extra_post=False`, sin las opciones de exploración.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

C = 0.05
EPS = 1e-6
COBERTURAS = (0.05, 0.10, 0.20, 0.30, 0.50)


@dataclass
class ModeloPrevio:
    lr: LogisticRegression
    mu_x: np.ndarray
    sd_x: np.ndarray
    mu_e: np.ndarray
    sd_e: np.ndarray

    def matriz(self, D: dict, extras: np.ndarray, idx: np.ndarray) -> sparse.csr_matrix:
        return armar_matriz(D, extras, idx, self.mu_x, self.sd_x, self.mu_e, self.sd_e)

    def predecir(self, D: dict, extras: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Probabilidad de que gane el equipo azul en las partidas `idx`."""
        return self.lr.predict_proba(self.matriz(D, extras, idx))[:, 1]


def armar_matriz(D, extras, idx, mu_x, sd_x, mu_e, sd_e) -> sparse.csr_matrix:
    partes = [D["DRAFT"][idx],
              sparse.csr_matrix((D["Xd"][idx] - mu_x) / sd_x),
              sparse.csr_matrix(((extras[idx] - mu_e) / sd_e).astype(np.float32))]
    return sparse.hstack(partes).tocsr()


def ajustar(D: dict, extras: np.ndarray, tr: np.ndarray, c: float = C) -> ModeloPrevio:
    """Ajusta la logística con las partidas `tr`; la estandarización sale solo de esas filas."""
    Xd = D["Xd"]
    mu_x, sd_x = Xd[tr].mean(0), Xd[tr].std(0) + EPS
    mu_e, sd_e = extras[tr].mean(0), extras[tr].std(0) + EPS
    lr = LogisticRegression(C=c, max_iter=4000)
    lr.fit(armar_matriz(D, extras, tr, mu_x, sd_x, mu_e, sd_e), D["y"][tr])
    return ModeloPrevio(lr, mu_x, sd_x, mu_e, sd_e)


def filas_entrenamiento_v(D: dict) -> np.ndarray:
    """Partidas Master+ anteriores al 16/09 (las Diamond de la ventana de V nunca entran)."""
    return D["fi_v"][D["master"][D["fi_v"]]]


def metricas(y: np.ndarray, p: np.ndarray) -> dict:
    """AUC, acierto, log loss, ECE (10 deciles de p), cobertura y la línea base de 'siempre azul'."""
    y = np.asarray(y).astype(int)
    ok = (p > 0.5) == y
    g = pd.DataFrame({"p": p, "y": y, "d": pd.qcut(p, 10, labels=False)}).groupby("d") \
          .agg(p=("p", "mean"), r=("y", "mean"), n=("y", "size"))
    orden = np.argsort(-np.abs(p - 0.5))
    return {
        "n": len(y),
        "auc": roc_auc_score(y, p),
        "acierto": ok.mean(),
        "log_loss": log_loss(y, p),
        "ece": float((np.abs(g.p - g.r) * g.n).sum() / g.n.sum()),
        "linea_base": max(y.mean(), 1 - y.mean()),     # acierto de predecir siempre el lado más frecuente
        "cobertura": {c: ok[orden[:int(c * len(y))]].mean() for c in COBERTURAS},
    }


def evaluar_v(D: dict, extras: np.ndarray) -> tuple[ModeloPrevio, np.ndarray, dict]:
    """Ajusta con Master+ anterior al 16/09 y predice V Master+. Devuelve (modelo, p en V, métricas)."""
    tr = filas_entrenamiento_v(D)
    vi = np.flatnonzero(D["val"])
    modelo = ajustar(D, extras, tr)
    p = modelo.predecir(D, extras, vi)
    m = metricas(D["y"][vi], p)
    m["n_entrenamiento"], m["n_columnas"] = len(tr), modelo.lr.coef_.shape[1]
    return modelo, p, m
