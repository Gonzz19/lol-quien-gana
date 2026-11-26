"""
Baselines heurísticos con estadísticas de campeón, sin entrenamiento (PROJECT_SUMMARY.md §9.S.1).

Seis targets:
    Target 0      Ganador de la partida (1 = gana Blue).
    Targets 1..5  Ganador de cada línea al minuto 15 (1 = Blue tiene más oro @15
                  que su rival directo de la misma posición).

Reglas:
    Partida  Score(equipo) = media de WR(campeón, línea) de sus 5 integrantes.
             Predice Blue si Score(Blue) > Score(Red).
    Línea    Predice Blue si avg_gold_at_15(campeón Blue, línea)
             > avg_gold_at_15(campeón Red, línea).

Las tablas de campeón se calculan con `stats_from_participants(train)` para que
ninguna partida evaluada intervenga en ellas (§9.S.13). `load_champion_stats` lee
`champions_by_lane.csv`, que agrega todas las partidas (modo descriptivo).

Filtros de evaluación (no tocan los CSV; ver AGENTS.md §3.F.5 y §3.G):
    * Partidas sin un único equipo ganador se excluyen del Target 0.
    * Un duelo de línea solo se evalúa si cada equipo tiene exactamente un
      jugador en esa posición.
    * Duelos sin oro @15 en alguno de los dos lados, o con empate exacto, no
      tienen ganador de línea y se excluyen.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)

LANES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
BLUE, RED = 100, 200
PARTICIPANT_COLS = ["match_id", "team_id", "win", "champion_name", "lane", "gold_15"]


# ----------------------------------------------------------------------------
# Tablas de campeón
# ----------------------------------------------------------------------------
def load_champion_stats(path: Path) -> Dict[str, pd.Series]:
    """Tablas de búsqueda desde champions_by_lane.csv."""
    c = pd.read_csv(path, usecols=["champion_name", "lane", "games_played",
                                   "wins", "winrate_pct", "avg_gold_at_15"])
    c = c[c["lane"].isin(LANES)]
    idx = pd.MultiIndex.from_arrays([c["champion_name"], c["lane"]])
    by_champ = c.groupby("champion_name")[["wins", "games_played"]].sum()
    gold_w = c.dropna(subset=["avg_gold_at_15"])
    return {
        "wr_lane": pd.Series((c["winrate_pct"] / 100.0).values, index=idx),
        "wr_champ": by_champ["wins"] / by_champ["games_played"],
        "gold_lane": pd.Series(c["avg_gold_at_15"].values, index=idx),
        # Fallback de oro: media de la línea ponderada por partidas jugadas.
        "gold_lane_mean": gold_w.groupby("lane").apply(
            lambda g: np.average(g["avg_gold_at_15"], weights=g["games_played"])),
    }


def load_participants(path: Path, extra=()) -> pd.DataFrame:
    return pd.read_csv(path, usecols=PARTICIPANT_COLS + list(extra))


def stats_from_participants(parts: pd.DataFrame) -> Dict[str, pd.Series]:
    """Las mismas tablas que `load_champion_stats`, calculadas solo con `parts`.

    Es lo que evita el leakage: con `parts` = partidas de train, ninguna partida de
    test interviene en el winrate ni en el oro medio @15 que se usan para
    predecirla. `champions_by_lane.csv` no sirve para esto porque agrega todas las
    partidas de la base.
    """
    p = parts[parts["lane"].isin(LANES)]
    g = p.groupby(["champion_name", "lane"]).agg(games=("win", "size"), wins=("win", "sum"),
                                                 gold=("gold_15", "mean"))
    by_champ = p.groupby("champion_name")["win"].agg(["sum", "size"])
    gold_lane_mean = p.dropna(subset=["gold_15"]).groupby("lane")["gold_15"].mean()
    return {
        "wr_lane": g["wins"] / g["games"],
        "wr_champ": by_champ["sum"] / by_champ["size"],
        "gold_lane": g["gold"],
        "gold_lane_mean": gold_lane_mean,
    }


# ----------------------------------------------------------------------------
# Preparación de etiquetas
# ----------------------------------------------------------------------------
def coherent_match_ids(parts: pd.DataFrame) -> pd.Index:
    """Partidas 5v5 con exactamente un equipo ganador (5 victorias en total)."""
    g = parts.groupby(["match_id", "team_id"])["win"].agg(["size", "sum"]).unstack("team_id")
    try:
        ok = ((g[("size", BLUE)] == 5) & (g[("size", RED)] == 5)
              & (g[("sum", BLUE)] + g[("sum", RED)] == 5)
              & g[("sum", BLUE)].isin([0, 5]))
    except KeyError:
        return pd.Index([])
    return g.index[ok.fillna(False)]


def build_lane_duels(parts: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Un duelo por (partida, línea) con exactamente un jugador por equipo.

    Devuelve (duelos, descartados por posición duplicada o vacía).
    """
    p = parts[parts["lane"].isin(LANES)]
    n_candidates = parts["match_id"].nunique() * len(LANES)
    sizes = p.groupby(["match_id", "team_id", "lane"]).size()
    single = sizes[sizes == 1].reset_index()[["match_id", "team_id", "lane"]]
    p1 = p.merge(single, on=["match_id", "team_id", "lane"])
    cols = ["champion_name", "gold_15"]
    blue = p1[p1["team_id"] == BLUE].set_index(["match_id", "lane"])[cols]
    red = p1[p1["team_id"] == RED].set_index(["match_id", "lane"])[cols]
    duels = blue.join(red, how="inner", lsuffix="_blue", rsuffix="_red").reset_index()
    return duels, n_candidates - len(duels)


# ----------------------------------------------------------------------------
# Métricas
# ----------------------------------------------------------------------------
def classification_metrics(y: np.ndarray, pred: np.ndarray, score: np.ndarray,
                           prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    out = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision_blue": float(precision_score(y, pred, zero_division=0)),
        "recall_blue": float(recall_score(y, pred, zero_division=0)),
        "f1_blue": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else float("nan"),
        "blue_rate": float(np.mean(y)),
        "always_red_accuracy": float(1.0 - np.mean(y)),
        "ties_pct": float(np.mean(score == 0) * 100.0),
    }
    if prob is not None:
        out["brier"] = float(brier_score_loss(y, prob))
    return out


# ----------------------------------------------------------------------------
# Reglas
# ----------------------------------------------------------------------------
def match_rule_scores(parts: pd.DataFrame, stats: Dict[str, pd.Series]) -> Tuple[pd.DataFrame, float]:
    """Por partida coherente: `y` (gana Blue), `diff` = Score(Blue) − Score(Red) y `prob`.

    Devuelve también el % de picks sin WR en su línea (se usa el WR global del
    campeón, y si tampoco existe, 0,5).
    """
    ids = coherent_match_ids(parts)
    p = parts[parts["match_id"].isin(ids)].copy()

    key = pd.MultiIndex.from_arrays([p["champion_name"], p["lane"]])
    p["wr"] = stats["wr_lane"].reindex(key).values
    miss = p["wr"].isna()
    p.loc[miss, "wr"] = p.loc[miss, "champion_name"].map(stats["wr_champ"])
    p["wr"] = p["wr"].fillna(0.5)

    team = p.groupby(["match_id", "team_id"]).agg(score=("wr", "mean"), win=("win", "max"))
    team = team.unstack("team_id")
    sb, sr = team[("score", BLUE)], team[("score", RED)]
    out = pd.DataFrame({"y": team[("win", BLUE)].astype(int), "diff": sb - sr,
                        "prob": sb / (sb + sr)})  # normalización tipo Bradley-Terry de las dos fuerzas
    return out, float(miss.mean() * 100.0)


def match_baseline(parts: pd.DataFrame, stats: Dict[str, pd.Series]) -> Dict:
    """Target 0: media de WR(campeón, línea) por equipo."""
    s, fallback = match_rule_scores(parts, stats)
    y, diff = s["y"].to_numpy(), s["diff"].to_numpy()
    res = classification_metrics(y, (diff > 0).astype(int), diff, s["prob"].to_numpy())
    res["excluded_incoherent"] = int(parts["match_id"].nunique() - len(s))
    res["fallback_lane_wr_pct"] = fallback
    return res


def lane_rule_scores(parts: pd.DataFrame, stats: Dict[str, pd.Series]) -> Tuple[pd.DataFrame, int, int]:
    """Duelos con etiqueta: `match_id`, `lane`, `y` (Blue con más oro @15) y `score`
    (oro medio @15 esperado de Blue − el de Red). Devuelve también los duelos
    descartados por posición y por falta de etiqueta."""
    duels, dropped_pos = build_lane_duels(parts)

    labeled = duels.dropna(subset=["gold_15_blue", "gold_15_red"])
    labeled = labeled[labeled["gold_15_blue"] != labeled["gold_15_red"]].copy()
    dropped_label = len(duels) - len(labeled)

    def expected_gold(champ_col: str) -> pd.Series:
        key = pd.MultiIndex.from_arrays([labeled[champ_col], labeled["lane"]])
        g = pd.Series(stats["gold_lane"].reindex(key).values, index=labeled.index)
        return g.fillna(labeled["lane"].map(stats["gold_lane_mean"]))

    labeled["y"] = (labeled["gold_15_blue"] > labeled["gold_15_red"]).astype(int)
    labeled["score"] = expected_gold("champion_name_blue") - expected_gold("champion_name_red")
    return labeled[["match_id", "lane", "y", "score"]], int(dropped_pos), int(dropped_label)


def lane_baseline(parts: pd.DataFrame, stats: Dict[str, pd.Series]) -> Dict:
    """Targets 1..5: oro medio @15 del campeón en su línea."""
    labeled, dropped_pos, dropped_label = lane_rule_scores(parts, stats)
    labeled["pred"] = (labeled["score"] > 0).astype(int)

    res = {}
    for lane in ["GLOBAL"] + LANES:
        sub = labeled if lane == "GLOBAL" else labeled[labeled["lane"] == lane]
        if len(sub) == 0:
            continue
        res[lane] = classification_metrics(sub["y"].values, sub["pred"].values,
                                           sub["score"].values)
    res["_dropped_position"] = dropped_pos
    res["_dropped_no_label"] = dropped_label
    return res
