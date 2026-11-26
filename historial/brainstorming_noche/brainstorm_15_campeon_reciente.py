"""Idea 20 del brainstorming (docs/BRAINSTORM_NOCHE.md), sobre el escalón 4 (0,6674), apuntada a los lobbies
de veteranos sin estrenos, donde el modelo rinde menos.

* Forma reciente con el campeón: media de los residuos r (§9.S.24) del jugador en sus últimas 10 partidas
  previas con ese campeón, encogida (S10 / (N10 + 5)), menos su margen por campeón acumulado (el de MC).
  Para 6 estadísticas clave (kills, asistencias, daño, oro y CS por minuto, oro a los 15 contra el rival).
* Óxido con el campeón: log de horas desde la última partida del jugador con ese campeón (el máximo si
  nunca lo jugó), y log de horas desde la última partida en esa línea.
Todo causal: sumas acumuladas menos la fila propia, y diferencias con la fila anterior del grupo.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.brainstorm_base import ajustar_predecir, cargar, comparar, evaluar, extras_referencia, mostrar  # noqa: E402
from src.features.fingerprint import add_rates  # noqa: E402
from src.features.margin import _residuo, lane_diffs, lane_scale  # noqa: E402

STATS = ["kills_pm", "assists_pm", "damage_pm", "gold_pm", "cs_pm", "d_gold_15"]
CACHE = REPO / "data" / "analysis" / "brainstorm" / "campeon_reciente.npz"
D = cargar()
NM, ms = D["NM"], D["master"]
if not CACHE.exists():
    USE = ["match_id", "game_creation", "game_duration", "player_id", "team_id", "win", "champion_id", "lane",
           "kills", "deaths", "assists", "kill_participation_pct", "damage_share_pct", "gold_share_pct",
           "total_damage_dealt_to_champions", "total_damage_taken", "gold_earned", "total_minions_killed",
           "neutral_minions_killed", "vision_score", "gold_15"]
    df = pd.read_csv(REPO / "data" / "splits" / "train" / "full_dataset.csv", usecols=USE)
    df = df[df.lane.isin(["BOTTOM", "JUNGLE", "MIDDLE", "TOP", "UTILITY"])]
    df = df[df.groupby(["match_id", "team_id"]).lane.transform("nunique") == 5]
    df = df[df.groupby("match_id").lane.transform("size") == 10]
    df = df[df.groupby("match_id").win.transform("sum") == 5].copy()
    df = add_rates(df).sort_values(["game_creation", "match_id", "team_id", "lane"]).reset_index(drop=True)
    df = lane_diffs(df, ["gold_15"])
    assert (df.match_id.to_numpy()[::10] == D["Z"]["mids"]).all()
    esc = lane_scale(df, STATS)
    g = [df.player_id, df.champion_id]
    out = {}
    for s in STATS:
        r, v = _residuo(df, s, 20, esc)
        S = r.groupby(g, sort=False).cumsum() - r
        N = v.groupby(g, sort=False).cumsum() - v
        S10 = S - S.groupby(g, sort=False).shift(10).fillna(0)
        N10 = N - N.groupby(g, sort=False).shift(10).fillna(0)
        out[f"rec_{s}"] = S10 / (N10 + 5) - S / (N + 10)
    h = lambda keys: np.log1p(((df.game_creation - df.groupby(keys, sort=False).game_creation.shift(1)) / 3.6e6).fillna(24 * 365))  # noqa: E731
    out["oxido_champ"] = h(g)
    out["oxido_linea"] = h([df.player_id, df.lane])
    F = pd.DataFrame(out)
    np.savez_compressed(CACHE, F=F.to_numpy(np.float32).reshape(NM, 2, 5, -1), cols=np.array(F.columns))
F = np.load(CACHE)["F"]
dif = F[:, 0] - F[:, 1]
EQ = dif.mean(axis=1).astype(np.float32)
LIN = dif.reshape(NM, -1).astype(np.float32)
REF = extras_referencia(D, escalon=4)


def m(extra=None):
    X = REF if extra is None else np.c_[REF, extra]
    return lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=X, extra_post=False)


VAR = {"escalón 4": m(), "+ forma reciente con el campeón y óxido (equipo)": m(EQ),
       "+ ídem por línea": m(LIN), "+ solo óxido (por línea)": m(dif[..., -2:].reshape(NM, -1))}
P = dict(Parallel(n_jobs=2, backend="threading")(delayed(lambda n: (n, evaluar(D, VAR[n])))(n) for n in VAR))
mostrar(comparar(D, P, "escalón 4"))
