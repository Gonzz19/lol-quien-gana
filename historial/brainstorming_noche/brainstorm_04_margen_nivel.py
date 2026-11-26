"""Idea 9 del brainstorming (docs/BRAINSTORM_NOCHE.md): margen previo del jugador en lobbies del
mismo nivel que la partida.

El margen de §9.S.24 promedia todas las partidas del jugador. Rendir +1 desvío contra un lobby
de Diamond puede valer menos que hacerlo contra Master+. Acá, con `prior_margin_by` y el grupo
"lobby Master+ / Diamond" (mediana del tier ≥ Master, `es_master_plus`), cada fila recibe el
desvío del margen del jugador en lobbies como el actual, respecto de su margen general (vale 0
sin partidas previas en ese tipo de lobby). También el winrate previo en ese tipo de lobby.

Sobre la referencia: híbrido + mapa entrenado solo con Master+ (brainstorm_01).
Fuga: el tipo de lobby sale del padrón (foto 1). Para las partidas de V (posteriores a la foto),
su historial y su clasificación son anteriores: causal. Para las filas de entrenamiento
anteriores a la foto, la clasificación de su historial usa el tier del 04/09; puede afectar los
pesos, pero no infla V.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.brainstorm_base import ajustar_predecir, cargar, comparar, evaluar, mostrar  # noqa: E402
from src.features.fingerprint import add_rates  # noqa: E402
from src.features.margin import _previo, lane_diffs, lane_scale, prior_margin_by  # noqa: E402
from src.models.validacion_temporal import es_master_plus  # noqa: E402

CACHE = REPO / "data" / "analysis" / "brainstorm" / "margen_nivel.npz"
STATS = ["kills_pm", "assists_pm", "damage_pm", "damage_share_pct", "kill_participation_pct",
         "deaths_pm", "damage_taken_pm", "gold_pm", "cs_pm", "gold_share_pct", "vision_pm",
         "d_gold_10", "d_gold_15", "d_xp_15", "d_cs_15"]
D = cargar()
NM = D["NM"]
if not CACHE.exists():
    USE = ["match_id", "game_creation", "game_duration", "player_id", "team_id", "win", "champion_id", "lane",
           "kills", "deaths", "assists", "kill_participation_pct", "damage_share_pct", "gold_share_pct",
           "total_damage_dealt_to_champions", "total_damage_taken", "gold_earned", "total_minions_killed",
           "neutral_minions_killed", "vision_score", "gold_10", "gold_15", "xp_15", "cs_15"]
    LANES = ["BOTTOM", "JUNGLE", "MIDDLE", "TOP", "UTILITY"]
    df = pd.read_csv(REPO / "data" / "splits" / "train" / "full_dataset.csv", usecols=USE)
    df = df[df.lane.isin(LANES)]
    df = df[df.groupby(["match_id", "team_id"]).lane.transform("nunique") == 5]
    df = df[df.groupby("match_id").lane.transform("size") == 10]
    df = df[df.groupby("match_id").win.transform("sum") == 5].copy()
    df = add_rates(df).sort_values(["game_creation", "match_id", "team_id", "lane"]).reset_index(drop=True)
    df = lane_diffs(df, ["gold_10", "gold_15", "xp_15", "cs_15"])
    assert (df.match_id.to_numpy()[::10] == D["Z"]["mids"]).all()
    df["lob_m"] = np.repeat(es_master_plus(D["Z"]["tnum"]), 10).astype(int)
    H = prior_margin_by(df, STATS, ["lob_m"], "niv", k_g=10, escala=lane_scale(df, STATS))
    pj, g = df.player_id, [df.player_id, df.lob_m]
    w = df.win.astype(float)
    wr_tot = (_previo(w, pj) + 5) / (df.groupby("player_id").cumcount() + 10)
    H["niv_wr"] = (_previo(w, g) + 10 * wr_tot) / (df.groupby(["player_id", "lob_m"]).cumcount() + 10) - wr_tot
    H["niv_n"] = np.log1p(df.groupby(["player_id", "lob_m"]).cumcount())
    np.savez_compressed(CACHE, H=H.to_numpy(np.float32).reshape(NM, 2, 5, -1), cols=np.array(H.columns))
Hc = np.load(CACHE, allow_pickle=True)
H = Hc["H"]
DIF = H[:, 0] - H[:, 1]                               # (NM, 5, k)
EQ = DIF.mean(axis=1).astype(np.float32)
LIN = DIF.reshape(NM, -1).astype(np.float32)
ms = D["master"]


def solo_m(extra=None):
    return lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=extra, extra_post=False)


VAR = {"referencia: solo Master+": solo_m(), "+ margen en lobbies del mismo nivel (equipo)": solo_m(EQ),
       "+ margen en lobbies del mismo nivel (por línea)": solo_m(LIN)}
P = dict(Parallel(n_jobs=3, backend="threading")(delayed(lambda n: (n, evaluar(D, VAR[n])))(n) for n in VAR))
mostrar(comparar(D, P, "referencia: solo Master+"))
