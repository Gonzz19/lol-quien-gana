"""
Huella de estilo por jugador: nivel 1 del protocolo de validación (PROJECT_SUMMARY.md §10).

Pregunta: ¿las partidas de un mismo jugador se parecen entre sí más que a las de
otros jugadores, una vez descontado lo que explica el campeón?

Representación
--------------
Cada partida se describe con estadísticas de juego normalizadas por minuto o
tomadas a los minutos 10 y 15 (`STYLE_FEATURES`). Para separar al jugador del
campeón, cada valor se expresa como **desvío respecto de las partidas del mismo
campeón, en la misma línea y con el mismo resultado**:

$$z = \\frac{x - \\mu_{c,l,w}}{\\sigma_{c,l,w}}$$

Condicionar por el resultado ($w$) evita que la huella sea "gana mucho": un
ganador siempre tiene más oro. Los grupos con menos de `MIN_GROUP` partidas se
descartan, porque su media y su desvío son ruido.

El perfil de un conjunto de partidas es la media de sus $z$.

Prueba
------
Para cada jugador se arman dos muestras disjuntas de $k$ partidas (A y B) y se
calcula la distancia entre todos los perfiles A y todos los B. Si hay huella, la
B del mismo jugador queda más cerca de su A que las B de los demás:

* `auc`: media por jugador de la fracción de otros jugadores cuya B queda más
  lejos que la propia (0,5 = azar, 1 = reconocimiento perfecto).
* `top1`: fracción de jugadores cuya B más cercana es la propia (azar = $1/n$).
* `auc_dificil`: igual que `auc`, pero comparando solo contra jugadores del mismo
  tier y la misma línea principal.
* `p_valor`: prueba de permutación sobre `auc` (se reasigna al azar qué B
  corresponde a cada A).

Muestras A/B (`make_halves`):

* `aleatoria`: $2k$ partidas al azar, partidas entre A y B.
* `campeones_distintos`: A y B con conjuntos de campeones disjuntos. Es la
  prueba clave: el mismo jugador con otros campeones.
* `temporal`: las $k$ más viejas contra las $k$ más nuevas (estabilidad).
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

CANONICAL_LANES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
TIER_ORDER = ["DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER"]
MIN_GROUP = 30

STYLE_FEATURES = [
    "kills_pm", "deaths_pm", "assists_pm",
    "kill_participation_pct", "damage_share_pct", "gold_share_pct",
    "damage_pm", "damage_taken_pm", "gold_pm", "cs_pm", "vision_pm",
    "cs_10", "gold_10", "xp_10", "damage_10", "early_aggression_10",
    "kills_10", "deaths_10", "assists_10",
    "cs_15", "gold_15", "xp_15", "damage_15",
]


def player_tiers(df: pd.DataFrame) -> pd.Series:
    """Tier del padrón de cada jugador: la moda de `player_tier` en sus filas (NaN si no está)."""
    t = df.dropna(subset=["player_tier"]).groupby("player_id")["player_tier"].agg(lambda s: s.mode().iat[0])
    return t.reindex(df["player_id"].unique())


def keep_min_tier(df: pd.DataFrame, min_tier: str = "MASTER") -> pd.DataFrame:
    """Solo las filas de jugadores con tier del padrón >= `min_tier` (los que no están, afuera).

    Se aplica antes de `residualize`, para que las medias de grupo salgan de la
    misma población que se evalúa.
    """
    rank = {t: i for i, t in enumerate(TIER_ORDER)}
    tiers = player_tiers(df)
    keep = tiers[tiers.map(rank).ge(rank[min_tier])].index
    return df[df["player_id"].isin(keep)]


def add_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega las estadísticas por minuto que usa `STYLE_FEATURES`."""
    mins = df["game_duration"] / 60.0
    return df.assign(
        kills_pm=df["kills"] / mins,
        deaths_pm=df["deaths"] / mins,
        assists_pm=df["assists"] / mins,
        damage_pm=df["total_damage_dealt_to_champions"] / mins,
        damage_taken_pm=df["total_damage_taken"] / mins,
        gold_pm=df["gold_earned"] / mins,
        cs_pm=(df["total_minions_killed"] + df["neutral_minions_killed"]) / mins,
        vision_pm=df["vision_score"] / mins,
    )


def residualize(df: pd.DataFrame, features: List[str] = STYLE_FEATURES,
                by: Optional[List[str]] = None, min_group: int = MIN_GROUP,
                ajuste: Optional[set] = None) -> pd.DataFrame:
    """Reemplaza cada feature por su z dentro del grupo `by` (campeón, línea, resultado).

    Devuelve solo las filas de grupos con al menos `min_group` partidas y sin
    valores faltantes en `features`.

    Con `ajuste` (un conjunto de `player_id`), la media, el desvío y el tamaño de
    cada grupo salen solo de las filas de esos jugadores y se aplican a todas las
    filas: los jugadores de evaluación no entran en la normalización
    (PROJECT_SUMMARY §9.S.77). Sin `ajuste`, todas las filas aportan (descriptivo).
    """
    by = by or ["champion_name", "lane", "win"]
    d = df.dropna(subset=features).copy()
    if ajuste is None:
        size = d.groupby(by)[features[0]].transform("size")
        d = d[size >= min_group].copy()
        g = d.groupby(by)[features]
        sd = g.transform("std").replace(0, np.nan)
        d[features] = (d[features] - g.transform("mean")) / sd
        return d.dropna(subset=features)
    fit = d[d["player_id"].isin(ajuste)]
    g = fit.groupby(by)[features]
    stats = g.mean().add_suffix("__mu").join(g.std().replace(0, np.nan).add_suffix("__sd"))
    stats["__n"] = g.size()
    stats = stats[stats["__n"] >= min_group]
    d = d.join(stats, on=by, how="inner")
    for f in features:
        d[f] = (d[f] - d[f + "__mu"]) / d[f + "__sd"]
    d = d.drop(columns=list(stats.columns))
    return d.dropna(subset=features)


def standardize(df: pd.DataFrame, features: List[str] = STYLE_FEATURES) -> pd.DataFrame:
    """z global por línea, sin descontar el campeón: el baseline 'crudo'."""
    d = df.dropna(subset=features).copy()
    g = d.groupby("lane")[features]
    d[features] = (d[features] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    return d.dropna(subset=features)


def _split_champions(games: pd.DataFrame, k: int, rng: np.random.Generator) -> Optional[tuple]:
    """Reparte los campeones del jugador en dos grupos con al menos k partidas cada uno."""
    counts = games["champion_name"].value_counts()
    order = rng.permutation(counts.index.to_numpy())
    a, n_a = [], 0
    for champ in order:
        if n_a >= k:
            break
        a.append(champ)
        n_a += counts[champ]
    in_a = games["champion_name"].isin(a)
    if n_a < k or (~in_a).sum() < k:
        return None
    return games[in_a], games[~in_a]


def make_halves(df: pd.DataFrame, k: int, mode: str, seed: int = 0) -> Dict[str, Dict[str, pd.Index]]:
    """`{player_id: {'A': índices, 'B': índices}}` con k partidas en cada muestra."""
    rng = np.random.default_rng(seed)
    out = {}
    for pid, games in df.groupby("player_id"):
        if len(games) < 2 * k:
            continue
        if mode == "aleatoria":
            idx = rng.permutation(games.index.to_numpy())
            out[pid] = {"A": idx[:k], "B": idx[k:2 * k]}
        elif mode == "temporal":
            s = games.sort_values("game_creation").index.to_numpy()
            out[pid] = {"A": s[:k], "B": s[-k:]}
        elif mode == "campeones_distintos":
            pair = _split_champions(games, k, rng)
            if pair is None:
                continue
            a, b = pair
            out[pid] = {"A": rng.choice(a.index.to_numpy(), k, replace=False),
                        "B": rng.choice(b.index.to_numpy(), k, replace=False)}
        else:
            raise ValueError(f"Modo desconocido: {mode}")
    return out


def profiles(df: pd.DataFrame, halves: Dict, features: List[str]) -> tuple:
    """Matrices (jugadores × features) con la media de A y de B, en el orden de `halves`."""
    pids = list(halves)
    X = df[features].to_numpy()
    pos = pd.Series(np.arange(len(df)), index=df.index)
    A = np.stack([X[pos[halves[p]["A"]]].mean(axis=0) for p in pids])
    B = np.stack([X[pos[halves[p]["B"]]].mean(axis=0) for p in pids])
    return pids, A, B


def pool_profiles(df: pd.DataFrame, halves: Dict) -> tuple:
    """Baseline del pool: frecuencia de cada campeón en A y en B."""
    champs = {c: i for i, c in enumerate(sorted(df["champion_name"].unique()))}
    col = df["champion_name"].map(champs)
    pids = list(halves)

    def freq(idx):
        v = np.zeros(len(champs))
        np.add.at(v, col[idx].to_numpy(), 1.0)
        return v / v.sum()
    A = np.stack([freq(halves[p]["A"]) for p in pids])
    B = np.stack([freq(halves[p]["B"]) for p in pids])
    return pids, A, B


def row_auc(D: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Por fila i: fracción de j != i (dentro de `mask`) con D[i, j] > D[i, i]; empates valen 1/2."""
    n = D.shape[0]
    diag = np.diag(D)[:, None]
    valid = ~np.eye(n, dtype=bool) if mask is None else (mask & ~np.eye(n, dtype=bool))
    farther = ((D > diag) + 0.5 * (D == diag)) * valid
    cnt = valid.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, farther.sum(axis=1) / cnt, np.nan)


def evaluate(A: np.ndarray, B: np.ndarray, metric: str = "euclidean",
             hard_mask: Optional[np.ndarray] = None, n_perm: int = 1000,
             n_boot: int = 1000, seed: int = 0) -> Dict:
    """Métricas de reidentificación A -> B (ver docstring del módulo)."""
    rng = np.random.default_rng(seed)
    D = cdist(A, B, metric=metric)
    n = len(D)
    auc_rows = row_auc(D)
    auc = float(np.nanmean(auc_rows))
    top1 = float((D.argmin(axis=1) == np.arange(n)).mean())

    # Permutación: reasignar qué B es la "propia". Con los rangos por fila precalculados,
    # el AUC de la permutación pi es la media de R[i, pi(i)].
    R = (rankdata(D, axis=1) - 1) / (n - 1)          # 0 = la más cercana, 1 = la más lejana
    null = np.array([R[np.arange(n), rng.permutation(n)].mean() for _ in range(n_perm)])
    p = float((1 + (null >= auc).sum()) / (1 + n_perm))

    boot = np.array([np.nanmean(auc_rows[rng.integers(0, n, n)]) for _ in range(n_boot)])
    out = {"n_jugadores": n, "auc": round(auc, 4),
           "auc_ic95": [round(float(np.quantile(boot, 0.025)), 4), round(float(np.quantile(boot, 0.975)), 4)],
           "top1": round(top1, 4), "top1_azar": round(1 / n, 5),
           "p_valor": p, "auc_nulo_media": round(float(null.mean()), 4)}
    if hard_mask is not None:
        hard = row_auc(D, hard_mask)
        out["auc_dificil"] = round(float(np.nanmean(hard)), 4)
        out["n_con_rivales_dificiles"] = int(np.isfinite(hard).sum())
    return out


def hard_negative_mask(pids: List, info: pd.DataFrame) -> np.ndarray:
    """mask[i, j] = mismo tier y misma línea principal (`info` indexado por player_id)."""
    t = info.reindex(pids)
    tier = t["player_tier"].fillna("?").to_numpy()
    lane = t["main_lane"].fillna("?").to_numpy()
    return (tier[:, None] == tier[None, :]) & (lane[:, None] == lane[None, :]) & (tier[:, None] != "?")


def player_variance_share(df: pd.DataFrame, features: List[str] = STYLE_FEATURES,
                          min_games: int = 40) -> pd.Series:
    """Parte de la varianza de cada feature que se explica por el jugador (ICC(1)).

    Con $\sigma^2_b$ la varianza de las medias por jugador, $\sigma^2_w$ la media de
    las varianzas dentro de cada jugador y $\bar n$ las partidas medias por jugador:

    $$\text{ICC} = \frac{\sigma^2_b - \sigma^2_w/\bar n}{\sigma^2_b - \sigma^2_w/\bar n + \sigma^2_w}$$

    Restar $\sigma^2_w/\bar n$ descuenta el ruido que tienen las medias por estar
    hechas con pocas partidas. Solo jugadores con al menos `min_games` partidas.
    """
    n = df.groupby("player_id").size()
    d = df[df["player_id"].isin(n[n >= min_games].index)]
    g = d.groupby("player_id")[features]
    within = g.var().mean()
    between = g.mean().var() - within / g.size().mean()
    return (between / (between + within)).sort_values(ascending=False)
