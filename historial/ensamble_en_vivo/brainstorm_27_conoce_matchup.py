"""Idea 28 del brainstorming (docs/BRAINSTORM_NOCHE.md): ¿el jugador conoce el matchup? Sobre el escalón 4.

Margen previo del jugador cuando enfrentó a ESE campeón rival en su línea (`prior_margin_by` con el campeón del rival
directo como grupo, k_g = 10, desvío respecto de su margen general), y log de las veces que lo enfrentó. Causal."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.brainstorm_base import ajustar_predecir, cargar, comparar, evaluar, extras_referencia, mostrar  # noqa: E402
from src.features.fingerprint import add_rates  # noqa: E402
from src.features.margin import lane_diffs, lane_scale, prior_margin_by  # noqa: E402

STATS = ["kills_pm", "assists_pm", "damage_pm", "deaths_pm", "gold_pm", "cs_pm", "d_gold_10", "d_gold_15", "d_cs_15"]
CACHE = REPO / "data" / "analysis" / "brainstorm" / "conoce_matchup.npz"
D = cargar()
NM, ms = D["NM"], D["master"]
if not CACHE.exists():
    USE = ["match_id", "game_creation", "game_duration", "player_id", "team_id", "win", "champion_id", "lane", "kills", "deaths",
           "assists", "kill_participation_pct", "damage_share_pct", "gold_share_pct", "total_damage_dealt_to_champions",
           "total_damage_taken", "gold_earned", "total_minions_killed", "neutral_minions_killed", "vision_score",
           "gold_10", "gold_15", "cs_15"]
    df = pd.read_csv(REPO / "data" / "splits" / "train" / "full_dataset.csv", usecols=USE)
    df = df[df.lane.isin(["BOTTOM", "JUNGLE", "MIDDLE", "TOP", "UTILITY"])]
    df = df[df.groupby(["match_id", "team_id"]).lane.transform("nunique") == 5]
    df = df[df.groupby("match_id").lane.transform("size") == 10]
    df = df[df.groupby("match_id").win.transform("sum") == 5].copy()
    df = add_rates(df).sort_values(["game_creation", "match_id", "team_id", "lane"]).reset_index(drop=True)
    df = lane_diffs(df, ["gold_10", "gold_15", "cs_15"])
    assert (df.match_id.to_numpy()[::10] == D["Z"]["mids"]).all()
    df["rival_champ"] = df.groupby(["match_id", "lane"]).champion_id.transform("sum") - df.champion_id
    H = prior_margin_by(df, STATS, ["rival_champ"], "cm", k_g=10, escala=lane_scale(df[np.repeat(~D["val0"], 10)], STATS))
    H["n_vs"] = np.log1p(df.groupby(["player_id", "rival_champ"]).cumcount())
    np.savez_compressed(CACHE, H=H.to_numpy(np.float32).reshape(NM, 2, 5, -1))
H = np.load(CACHE)["H"]
dif = H[:, 0] - H[:, 1]
R4 = extras_referencia(D, escalon=4)
m = lambda X: (lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=X, extra_post=False))  # noqa: E731
VAR = {"escalón 4": m(R4), "+ conoce el matchup (por línea)": m(np.c_[R4, dif.reshape(NM, -1)]),
       "+ conoce el matchup (equipo)": m(np.c_[R4, dif.mean(1)])}
P = dict(Parallel(n_jobs=2, backend="threading")(delayed(lambda n: (n, evaluar(D, VAR[n])))(n) for n in VAR))
mostrar(comparar(D, P, "escalón 4"))
