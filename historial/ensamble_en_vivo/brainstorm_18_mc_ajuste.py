"""Idea 22 del brainstorming (docs/BRAINSTORM_NOCHE.md): ajuste del margen por campeón (lo que más sumó en la noche).

Sobre el escalón 4, se reemplaza el margen por (jugador, campeón) con k_g = 10 por k_g = 5 y k_g = 20,
y se prueba el margen por (jugador, campeón, línea) además del por campeón.
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
from src.features.margin import lane_diffs, lane_scale, prior_margin_by  # noqa: E402

STATS = ["kills_pm", "assists_pm", "damage_pm", "damage_share_pct", "kill_participation_pct",
         "deaths_pm", "damage_taken_pm", "gold_pm", "cs_pm", "gold_share_pct", "vision_pm",
         "d_gold_10", "d_gold_15", "d_xp_15", "d_cs_15"]
CACHE = REPO / "data" / "analysis" / "brainstorm" / "mc_ajuste.npz"
D = cargar()
NM, ms = D["NM"], D["master"]
if not CACHE.exists():
    USE = ["match_id", "game_creation", "game_duration", "player_id", "team_id", "win", "champion_id", "lane",
           "kills", "deaths", "assists", "kill_participation_pct", "damage_share_pct", "gold_share_pct",
           "total_damage_dealt_to_champions", "total_damage_taken", "gold_earned", "total_minions_killed",
           "neutral_minions_killed", "vision_score", "gold_10", "gold_15", "xp_15", "cs_15"]
    df = pd.read_csv(REPO / "data" / "splits" / "train" / "full_dataset.csv", usecols=USE)
    df = df[df.lane.isin(["BOTTOM", "JUNGLE", "MIDDLE", "TOP", "UTILITY"])]
    df = df[df.groupby(["match_id", "team_id"]).lane.transform("nunique") == 5]
    df = df[df.groupby("match_id").lane.transform("size") == 10]
    df = df[df.groupby("match_id").win.transform("sum") == 5].copy()
    df = add_rates(df).sort_values(["game_creation", "match_id", "team_id", "lane"]).reset_index(drop=True)
    df = lane_diffs(df, ["gold_10", "gold_15", "xp_15", "cs_15"])
    assert (df.match_id.to_numpy()[::10] == D["Z"]["mids"]).all()
    esc = lane_scale(df, STATS)
    out = {f"mc{k}": prior_margin_by(df, STATS, ["champion_id"], "mc", k_g=k, escala=esc).to_numpy(np.float32)
           for k in (5, 20)}
    out["mcl"] = prior_margin_by(df, STATS, ["champion_id", "lane"], "mcl", k_g=10, escala=esc).to_numpy(np.float32)
    np.savez_compressed(CACHE, **{k: v.reshape(NM, 2, 5, -1) for k, v in out.items()})
C = np.load(CACHE)
dif = lambda a: (a[:, 0] - a[:, 1]).reshape(NM, -1).astype(np.float32)  # noqa: E731
R4 = extras_referencia(D, escalon=4)
SIN_MC = R4[:, 75:]                     # R4 = [MC (75) | ML (75) | NIV (3) | W1 (4) | MAT (10)]


def m(X):
    return lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=X, extra_post=False)


VAR = {"escalón 4 (MC k = 10)": m(R4), "MC k = 5": m(np.c_[dif(C["mc5"]), SIN_MC]),
       "MC k = 20": m(np.c_[dif(C["mc20"]), SIN_MC]), "+ margen por (campeón, línea)": m(np.c_[R4, dif(C["mcl"])])}
P = dict(Parallel(n_jobs=2, backend="threading")(delayed(lambda n: (n, evaluar(D, VAR[n])))(n) for n in VAR))
mostrar(comparar(D, P, "escalón 4 (MC k = 10)"))
