"""
Rating de habilidad por jugador y partida: TrueSkill de equipos con la foto del
padrón como medición (PROJECT_SUMMARY.md §9.S.21).

Riot no da el rango histórico de un jugador, así que se estima su habilidad al
momento de cada partida con un sistema de rating que aprende de los resultados.

* **Escala en LP.** La habilidad de cada jugador es una normal $N(\\mu, \\sigma^2)$
  en "LP absolutos": Diamond IV 0 LP = 0, cada división suma 100 y Master 0 LP =
  400 (Master, Grandmaster y Challenger comparten una sola escala de LP). Solo
  importan las diferencias: $\\mu$ puede ser negativa (un Emerald en una partida
  Diamond).
* **Partida (TrueSkill de dos equipos, sin empates).** Con
  $c^2 = \\sum_i \\sigma_i^2 + n\\beta^2$ y $\\delta$ la ventaja del lado rojo:

  $$P(\\text{gana azul}) = \\Phi\\Big(\\frac{\\sum_{azul} \\mu - \\sum_{rojo} \\mu - \\delta}{c}\\Big)$$

  Con $t$ el mismo cociente visto desde el ganador:

  $$\\mu_i \\mathrel{{+}{=}} \\pm\\frac{\\sigma_i^2}{c} v(t), \\qquad
  \\sigma_i^2 \\mathrel{{\\times}{=}} 1 - \\frac{\\sigma_i^2}{c^2} w(t)$$

  con $v(t) = \\varphi(t)/\\Phi(t)$ y $w(t) = v(t)(v(t) + t)$.
* **Dinámica.** Antes de cada partida la incertidumbre crece por las partidas
  jugadas desde la anterior, incluidas las que no vimos:
  $\\sigma^2 \\mathrel{{+}{=}} \\tau^2 (1 + h)$, con $h$ las partidas no vistas por
  partida observada (de la cobertura del jugador frente a su W + L de la foto).
  Solo antes de la foto: después, la recolección por ventana es casi completa.
* **La foto como medición.** En su fecha, el LP de la foto $z$ corrige a cada
  jugador del padrón como un filtro de Kalman:
  $K = \\sigma^2/(\\sigma^2 + \\sigma_{foto}^2)$,
  $\\mu \\leftarrow \\mu + K(z - \\mu)$, $\\sigma^2 \\leftarrow (1 - K)\\sigma^2$.
  Ninguna partida usa la foto antes de su fecha.
* **Sin fuga del target.** Lo que se devuelve es el rating **previo** a cada
  partida: no depende de su resultado.

El bucle trabaja con floats de Python y no con arreglos de numpy: con 10 jugadores
por partida, el costo fijo de cada operación de numpy domina (≈ 25 veces más lento).
"""

from dataclasses import asdict, dataclass
from math import erf, exp, pi, sqrt
from typing import Optional, Sequence

import numpy as np
import pandas as pd

DIVISION_LP = {"IV": 0, "III": 100, "II": 200, "I": 300}
APEX = {"MASTER", "GRANDMASTER", "CHALLENGER"}
MASTER_BASE = 400
AZUL = 100

_SQRT2 = sqrt(2.0)
_SQRT2PI = sqrt(2.0 * pi)


# Debajo de Diamond la escala sigue hacia abajo: Emerald I 0 LP = −100, Emerald IV
# 0 LP = −400, Platinum I 0 LP = −500, etc. (cada división, 100).
DEBAJO_DE_DIAMOND = ["EMERALD", "PLATINUM", "GOLD", "SILVER", "BRONZE", "IRON"]


def lp_absoluto(tier: str, rank: str, lp: float) -> float:
    """LP en la escala continua: Diamond IV 0 LP = 0, Master 0 LP = 400, Emerald I 0 LP = −100.

    NaN para UNRANKED o un tier desconocido.
    """
    tier, rank = str(tier).upper(), str(rank).upper()
    if tier in APEX:
        return MASTER_BASE + float(lp)
    if rank not in DIVISION_LP:
        return float("nan")
    if tier == "DIAMOND":
        return DIVISION_LP[rank] + float(lp)
    if tier in DEBAJO_DE_DIAMOND:
        return -400 * (DEBAJO_DE_DIAMOND.index(tier) + 1) + DIVISION_LP[rank] + float(lp)
    return float("nan")


@dataclass(frozen=True)
class ParametrosRating:
    beta: float = 200.0        # ruido de rendimiento de una partida, por jugador
    tau: float = 15.0          # cambio de habilidad por partida jugada
    sigma0: float = 400.0      # incertidumbre inicial de un jugador nuevo
    sigma_foto: float = 100.0  # error de medición del LP de la foto como habilidad
    mu0: float = 0.0           # media inicial de un jugador nuevo
    ventaja_roja: float = 0.0  # δ: ventaja del lado rojo, en LP del equipo
    max_ocultas: float = 30.0  # tope de partidas no vistas por partida observada

    def to_dict(self) -> dict:
        return asdict(self)


def _phi(x: float) -> float:
    return exp(-0.5 * x * x) / _SQRT2PI


def _Phi(x: float) -> float:
    return 0.5 * (1.0 + erf(x / _SQRT2))


def _v_w(t: float) -> tuple:
    """v(t) = φ(t)/Φ(t) y w(t) = v(v + t), estables para t muy negativo."""
    cdf = _Phi(t)
    v = -t if cdf < 1e-12 else _phi(t) / cdf  # asintótico: φ(t)/Φ(t) ≈ −t
    return v, v * (v + t)


def prob_azul(mu: Sequence[float], s2: Sequence[float], azul: Sequence[int], rojo: Sequence[int],
              beta: float, ventaja_roja: float = 0.0) -> float:
    """P(gana el equipo azul) antes de la partida."""
    c = sqrt(sum(s2[i] for i in azul) + sum(s2[i] for i in rojo) + (len(azul) + len(rojo)) * beta * beta)
    return _Phi((sum(mu[i] for i in azul) - sum(mu[i] for i in rojo) - ventaja_roja) / c)


def actualizar(mu: list, s2: list, gan: Sequence[int], per: Sequence[int], beta: float,
               ventaja_gan: float = 0.0) -> None:
    """Actualiza in place a los ganadores `gan` y perdedores `per`.

    `ventaja_gan`: ventaja de lado del equipo ganador (+δ si ganó rojo, −δ si azul).
    """
    c2 = sum(s2[i] for i in gan) + sum(s2[i] for i in per) + (len(gan) + len(per)) * beta * beta
    c = sqrt(c2)
    v, w = _v_w((sum(mu[i] for i in gan) - sum(mu[i] for i in per) + ventaja_gan) / c)
    for signo, equipo in ((1.0, gan), (-1.0, per)):
        for i in equipo:
            mu[i] += signo * s2[i] / c * v
            s2[i] *= max(1.0 - s2[i] / c2 * w, 1e-6)


def medir(mu: list, s2: list, idx: Sequence[int], z: Sequence[float], sigma_foto: float) -> None:
    """Medición de Kalman: el LP `z` corrige a los jugadores `idx`."""
    r2 = sigma_foto * sigma_foto
    for i, zi in zip(idx, z):
        k = s2[i] / (s2[i] + r2)
        mu[i] += k * (zi - mu[i])
        s2[i] *= 1.0 - k


def ocultas_por_partida(partes: pd.DataFrame, foto: pd.DataFrame, desde_ms: int, hasta_ms: int,
                        tope: float) -> pd.Series:
    """Partidas no vistas por partida observada, por `player_id`, antes de la foto.

    `foto`: `player_id` y `partidas` (W + L del split según Riot). Las partidas
    observadas se cuentan entre `desde_ms` (inicio supuesto del split) y la foto.
    Sin partidas observadas no se define (el jugador no aparece en el resultado).
    """
    obs = (partes.loc[(partes["game_creation"] >= desde_ms) & (partes["game_creation"] < hasta_ms)]
           .groupby("player_id").size())
    total = foto.set_index("player_id")["partidas"]
    n = obs.reindex(total.index).fillna(0)
    h = ((total - n) / n.where(n > 0)).clip(lower=0, upper=tope)
    return h.dropna()


def correr(partes: pd.DataFrame, params: ParametrosRating, foto: Optional[pd.DataFrame] = None,
           t_foto_ms: Optional[int] = None, ocultas: Optional[pd.Series] = None) -> dict:
    """Corre el rating sobre todas las partidas, en orden cronológico.

    `partes`: una fila por jugador-partida con `match_id`, `game_creation`,
    `player_id`, `team_id` (100 = azul) y `win`. Las partidas sin ganador no
    actualizan. `foto`: `player_id` y `z` (LP absoluto), aplicada en `t_foto_ms`.
    `ocultas`: partidas no vistas por partida observada, por `player_id`.

    Devuelve `filas` (rating previo de cada jugador-partida, en orden cronológico),
    `partidas` (`p_azul_pre`, `gana_azul`: 1, 0 o −1 sin ganador) y `estado` (final).
    """
    df = partes.sort_values(["game_creation", "match_id", "team_id"], kind="mergesort").reset_index(drop=True)
    jugadores, idx_np = np.unique(df["player_id"].to_numpy(dtype=str), return_inverse=True)
    n_j = len(jugadores)
    pos = pd.Series(np.arange(n_j), index=jugadores)
    mu = [float(params.mu0)] * n_j
    s2 = [float(params.sigma0) ** 2] * n_j
    visto = [False] * n_j
    previas = [0] * n_j
    h = [0.0] * n_j
    if ocultas is not None and len(ocultas):
        comunes = ocultas.index.intersection(pos.index)
        for i, v in zip(pos[comunes].to_numpy(), ocultas[comunes].to_numpy()):
            h[int(i)] = float(v)

    foto_idx, foto_z = [], []
    if foto is not None and t_foto_ms is not None:
        f = foto[foto["player_id"].isin(pos.index)]
        foto_idx = pos[f["player_id"]].astype(int).tolist()
        foto_z = f["z"].astype(float).tolist()
    pendiente_foto = bool(foto_idx)

    t = df["game_creation"].to_numpy()
    idx = idx_np.tolist()
    es_azul = (df["team_id"].to_numpy() == AZUL).tolist()
    win = df["win"].to_numpy().tolist()
    _, inicio = np.unique(df["match_id"].to_numpy(dtype=str), return_index=True)
    inicio = np.sort(inicio)
    fin = np.r_[inicio[1:], len(df)]

    mu_pre = [0.0] * len(df)
    sig_pre = [0.0] * len(df)
    prev_pre = [0] * len(df)
    p_azul = [0.0] * len(inicio)
    gana_azul = [-1] * len(inicio)
    tau2 = params.tau ** 2
    beta, delta = params.beta, params.ventaja_roja

    for m, (a, b) in enumerate(zip(inicio.tolist(), fin.tolist())):
        if pendiente_foto and t[a] >= t_foto_ms:
            medir(mu, s2, foto_idx, foto_z, params.sigma_foto)
            for i in foto_idx:
                visto[i] = True
            pendiente_foto = False
        # Las partidas no vistas solo cuentan antes de la foto (después la cobertura es casi completa).
        antes_foto = t_foto_ms is None or t[a] < t_foto_ms
        azul, rojo, gan, per = [], [], [], []
        for r in range(a, b):
            i = idx[r]
            if visto[i]:
                s2[i] += tau2 * (1.0 + (h[i] if antes_foto else 0.0))
            visto[i] = True
            mu_pre[r] = mu[i]
            sig_pre[r] = sqrt(s2[i])
            prev_pre[r] = previas[i]
            (azul if es_azul[r] else rojo).append(i)
            (gan if win[r] == 1 else per).append(i)
        p_azul[m] = prob_azul(mu, s2, azul, rojo, beta, delta)
        if gan and per:
            gano_azul = es_azul[a] == (win[a] == 1)
            gana_azul[m] = int(gano_azul)
            actualizar(mu, s2, gan, per, beta, -delta if gano_azul else delta)
        for i in azul + rojo:
            previas[i] += 1

    if pendiente_foto:
        medir(mu, s2, foto_idx, foto_z, params.sigma_foto)

    filas = df[["match_id", "player_id", "team_id", "game_creation"]].copy()
    filas["mu_pre"] = mu_pre
    filas["sigma_pre"] = sig_pre
    filas["partidas_previas"] = prev_pre
    partidas = pd.DataFrame({"match_id": df["match_id"].to_numpy()[inicio], "game_creation": t[inicio],
                             "p_azul_pre": p_azul, "gana_azul": gana_azul})
    estado = pd.DataFrame({"player_id": jugadores, "mu": mu, "sigma": np.sqrt(s2), "partidas": previas})
    return {"filas": filas, "partidas": partidas, "estado": estado}


def log_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def metricas(partidas: pd.DataFrame) -> dict:
    """Log-loss, Brier y accuracy de `p_azul_pre` sobre las partidas con ganador.

    `log_loss_constante` es la referencia de predecir siempre la tasa de victoria
    azul de esas mismas partidas.
    """
    ok = partidas["gana_azul"] >= 0
    p = partidas.loc[ok, "p_azul_pre"].to_numpy(dtype=float)
    y = partidas.loc[ok, "gana_azul"].to_numpy(dtype=float)
    base = y.mean() if len(y) else 0.5
    return {"n": int(ok.sum()), "log_loss": log_loss(p, y), "brier": float(np.mean((p - y) ** 2)),
            "accuracy": float(np.mean((p > 0.5) == (y == 1))),
            "log_loss_constante": log_loss(np.full_like(y, base), y)}
