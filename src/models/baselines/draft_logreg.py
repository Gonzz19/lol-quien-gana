r"""
Regresión logística sobre el draft (escalón 2; PROJECT_SUMMARY.md §9.S.20).

Codificación antisimétrica (tipo Bradley-Terry): una columna por (campeón, línea)
que vale +1 si está en Blue y −1 si está en Red,

$$P(\text{gana Blue}) = \sigma\Big(b + \sum_{i \in \text{Blue}} w_i - \sum_{i \in \text{Red}} w_i\Big),$$

así que $w$ es la fuerza del campeón en esa línea y $b$ la ventaja de lado. Con
`sides=True` cada (campeón, línea) tiene una columna por lado.

Para las líneas el target es quién tiene más oro @15 en el duelo; hay un modelo por
línea, con el duelo (`duel_matrix`: los dos campeones de la línea) o con el draft
completo (`draft_matrix`).

La regularización L2 (`C`) se elige con validación temporal dentro del train: se
ajusta con el primer 80 % (por fecha) y se elige el C de menor log loss en el
último 20 %; después se reentrena con todo el train. El vocabulario sale solo del
train: un (campeón, línea) que no aparece ahí queda en cero.
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from src.models.baselines.draft_rules import BLUE, LANES, coherent_match_ids

CS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
VAL_FRAC = 0.2


def complete_match_ids(parts: pd.DataFrame) -> pd.Index:
    """Partidas con un único ganador y exactamente un jugador por (equipo, línea)."""
    p = parts[parts["lane"].isin(LANES)]
    n = p.groupby(["match_id", "team_id", "lane"]).size()
    completas = n.groupby(level=0).agg(lambda s: len(s) == 2 * len(LANES) and (s == 1).all())
    return completas.index[completas].intersection(coherent_match_ids(parts))


def _sign(team_id: pd.Series) -> np.ndarray:
    return np.where(team_id.to_numpy() == BLUE, 1.0, -1.0)


def draft_matrix(parts: pd.DataFrame, match_ids: Sequence, vocab: Optional[pd.Index] = None,
                 sides: bool = False) -> Tuple[sparse.csr_matrix, pd.Index]:
    """Matriz partida × (campeón, línea) en el orden de `match_ids`.

    Sin `vocab`, se arma con las partidas de `match_ids` (pasar las del train).
    """
    ids = pd.Index(match_ids)
    p = parts[parts["match_id"].isin(ids) & parts["lane"].isin(LANES)]
    cl = p["champion_name"].astype(str) + "|" + p["lane"].astype(str)
    if vocab is None:
        vocab = pd.Index(sorted(cl.unique()))
    col = vocab.get_indexer(cl)
    row = ids.get_indexer(p["match_id"])
    keep = col >= 0
    s = _sign(p["team_id"])[keep]
    if sides:
        data, cols, width = np.ones(keep.sum()), col[keep] + np.where(s > 0, 0, len(vocab)), 2 * len(vocab)
    else:
        data, cols, width = s, col[keep], len(vocab)
    return sparse.csr_matrix((data, (row[keep], cols)), shape=(len(ids), width)), vocab


def duel_matrix(blue: Sequence[str], red: Sequence[str],
                vocab: Optional[pd.Index] = None) -> Tuple[sparse.csr_matrix, pd.Index]:
    """Una fila por duelo: +1 en el campeón de Blue y −1 en el de Red."""
    blue, red = np.asarray(blue, dtype=str), np.asarray(red, dtype=str)
    if vocab is None:
        vocab = pd.Index(sorted(set(blue) | set(red)))
    ib, ir = vocab.get_indexer(blue), vocab.get_indexer(red)
    rows = np.arange(len(blue))
    kb, kr = ib >= 0, ir >= 0
    X = sparse.csr_matrix((np.r_[np.ones(kb.sum()), -np.ones(kr.sum())],
                           (np.r_[rows[kb], rows[kr]], np.r_[ib[kb], ir[kr]])),
                          shape=(len(blue), len(vocab)))
    return X, vocab


def match_target(parts: pd.DataFrame, match_ids: Sequence) -> np.ndarray:
    """1 si gana Blue, en el orden de `match_ids`."""
    blue = parts[parts["team_id"] == BLUE].groupby("match_id")["win"].max()
    return blue.reindex(pd.Index(match_ids)).astype(int).to_numpy()


def lane_duels(parts: pd.DataFrame, match_ids: Sequence, lane: str) -> pd.DataFrame:
    """Por partida de `match_ids`: campeón y oro @15 de Blue y de Red en `lane`, y `y`
    (Blue con más oro). Sin oro en algún lado o con empate exacto, `y` es NaN."""
    p = parts[parts["match_id"].isin(pd.Index(match_ids)) & (parts["lane"] == lane)]
    w = p.pivot_table(index="match_id", columns="team_id", values=["champion_name", "gold_15"],
                      aggfunc="first").reindex(pd.Index(match_ids))
    d = pd.DataFrame({"champ_blue": w[("champion_name", BLUE)].astype(str),
                      "champ_red": w[("champion_name", 200)].astype(str),
                      "gold_blue": w[("gold_15", BLUE)].astype(float),
                      "gold_red": w[("gold_15", 200)].astype(float)})
    ok = d["gold_blue"].notna() & d["gold_red"].notna() & (d["gold_blue"] != d["gold_red"])
    d["y"] = np.where(ok, (d["gold_blue"] > d["gold_red"]).astype(float), np.nan)
    return d


def temporal_holdout(idx: np.ndarray, creation: np.ndarray,
                     val_frac: float = VAL_FRAC) -> Tuple[np.ndarray, np.ndarray]:
    """Ordena `idx` por fecha y separa el último `val_frac` como validación."""
    s = np.asarray(idx)[np.argsort(np.asarray(creation)[idx], kind="stable")]
    cut = int(round((1 - val_frac) * len(s)))
    return s[:cut], s[cut:]


def fit_temporal_C(X, y: np.ndarray, train_idx: np.ndarray, creation: np.ndarray,
                   cs: Sequence[float] = CS, val_frac: float = VAL_FRAC):
    """(modelo reentrenado con todo el train, C elegido, {C: log loss de validación})."""
    fit, val = temporal_holdout(train_idx, creation, val_frac)
    curva = {}
    for C in cs:
        m = LogisticRegression(C=C, max_iter=2000).fit(X[fit], y[fit])
        curva[C] = float(log_loss(y[val], m.predict_proba(X[val])[:, 1], labels=[0, 1]))
    C = min(curva, key=curva.get)
    return LogisticRegression(C=C, max_iter=2000).fit(X[train_idx], y[train_idx]), C, curva


def prob_metrics(y: np.ndarray, p: np.ndarray, n_boot: int = 1000, n_boot_auc: int = 200,
                 seed: int = 0) -> Dict[str, Dict[str, float]]:
    """Accuracy (umbral 0,5), AUC, log loss y Brier con IC 95 % por bootstrap de filas."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    pc = np.clip(p, 1e-15, 1 - 1e-15)
    por_fila = {"accuracy": ((p > 0.5) == y).astype(float),
                "log_loss": -(y * np.log(pc) + (1 - y) * np.log(1 - pc)),
                "brier": (p - y) ** 2}
    boots = [rng.integers(0, len(y), len(y)) for _ in range(max(n_boot, n_boot_auc))]
    out = {}
    for k, v in por_fila.items():
        vals = [v[b].mean() for b in boots[:n_boot]]
        out[k] = {"valor": float(v.mean()), "ic95": [float(np.percentile(vals, 2.5)),
                                                     float(np.percentile(vals, 97.5))]}
    auc = lambda yy, pp: roc_auc_score(yy, pp) if len(np.unique(yy)) == 2 else np.nan
    vals = [auc(y[b], p[b]) for b in boots[:n_boot_auc]]
    out["auc"] = {"valor": float(auc(y, p)), "ic95": [float(np.nanpercentile(vals, 2.5)),
                                                      float(np.nanpercentile(vals, 97.5))]}
    out["n"] = int(len(y))
    return out
