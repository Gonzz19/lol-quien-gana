"""Brainstorming (docs/BRAINSTORM_NOCHE.md): revisión de Gemini — la escala por línea (`lane_scale`) de los márgenes
por campeón y por línea se calculaba con todo el train (incluye estadísticas de partidas de la ventana de V).
Se recalcula con solo las partidas anteriores a V (`~val0`) y se compara el escalón 4."""
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

STATS = ["kills_pm", "assists_pm", "damage_pm", "damage_share_pct", "kill_participation_pct", "deaths_pm", "damage_taken_pm",
         "gold_pm", "cs_pm", "gold_share_pct", "vision_pm", "d_gold_10", "d_gold_15", "d_xp_15", "d_cs_15"]
CACHE = REPO / "data" / "analysis" / "brainstorm" / "margen_grupos_escala_sin_v.npz"
D = cargar()
NM, ms = D["NM"], D["master"]
if not CACHE.exists():
    USE = ["match_id", "game_creation", "game_duration", "player_id", "team_id", "win", "champion_id", "lane", "kills", "deaths",
           "assists", "kill_participation_pct", "damage_share_pct", "gold_share_pct", "total_damage_dealt_to_champions",
           "total_damage_taken", "gold_earned", "total_minions_killed", "neutral_minions_killed", "vision_score",
           "gold_10", "gold_15", "xp_15", "cs_15"]
    df = pd.read_csv(REPO / "data" / "splits" / "train" / "full_dataset.csv", usecols=USE)
    df = df[df.lane.isin(["BOTTOM", "JUNGLE", "MIDDLE", "TOP", "UTILITY"])]
    df = df[df.groupby(["match_id", "team_id"]).lane.transform("nunique") == 5]
    df = df[df.groupby("match_id").lane.transform("size") == 10]
    df = df[df.groupby("match_id").win.transform("sum") == 5].copy()
    df = add_rates(df).sort_values(["game_creation", "match_id", "team_id", "lane"]).reset_index(drop=True)
    df = lane_diffs(df, ["gold_10", "gold_15", "xp_15", "cs_15"])
    assert (df.match_id.to_numpy()[::10] == D["Z"]["mids"]).all()
    esc = lane_scale(df[np.repeat(~D["val0"], 10)], STATS)          # solo partidas anteriores a V
    MC = prior_margin_by(df, STATS, ["champion_id"], "mc", k_g=10, escala=esc)
    ML = prior_margin_by(df, STATS, ["lane"], "ml", k_g=10, escala=esc)
    np.savez_compressed(CACHE, MC=MC.to_numpy(np.float32).reshape(NM, 2, 5, -1), ML=ML.to_numpy(np.float32).reshape(NM, 2, 5, -1))
C = np.load(CACHE)
MCML2 = np.c_[(C["MC"][:, 0] - C["MC"][:, 1]).reshape(NM, -1), (C["ML"][:, 0] - C["ML"][:, 1]).reshape(NM, -1)].astype(np.float32)
R4 = extras_referencia(D, escalon=4)
R4b = np.c_[MCML2, R4[:, 150:]]
m = lambda X: (lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=X, extra_post=False))  # noqa: E731
P = dict(Parallel(n_jobs=2, backend="threading")(delayed(lambda n, X: (n, evaluar(D, m(X))))(n, X)
                                                   for n, X in (("escalón 4 (escala con todo el train)", R4), ("escala solo antes de V", R4b))))
mostrar(comparar(D, P, "escalón 4 (escala con todo el train)"))
