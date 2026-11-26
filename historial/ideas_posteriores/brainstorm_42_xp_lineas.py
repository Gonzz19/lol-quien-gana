"""Brainstorming (docs/BRAINSTORM_NOCHE.md): modelo por línea para la XP y el oro @10 (mismo HGB causal hacia adelante de
`brainstorm_g_09`, cambiando el objetivo). Escribe `data/analysis/brainstorm/hgb_lineas_extra.npz` con las predicciones
previas a la partida de Δxp @15, Δoro @10 y Δxp @10 por línea (en miles), para la segunda etapa en vivo.

    python historial/ideas_posteriores/brainstorm_42_xp_lineas.py     # ~30 min
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import historial.escalon9.brainstorm_g_09_hgb_lineas_causal as g9  # noqa: E402
from scripts.analysis.brainstorm_base import cargar  # noqa: E402

D = cargar()
out = {}
for col, minuto in (("xp_15", 900), ("gold_10", 600), ("xp_10", 600)):
    def objetivo(D, col=col, minuto=minuto):
        mids = D["Z"]["mids"]
        df = pd.read_csv(REPO / "data" / "splits" / "train" / "full_dataset.csv",
                         usecols=["match_id", "team_id", "lane", col, "game_duration"])
        df = df[df.match_id.isin(set(mids)) & df.lane.isin(g9.LANES)].copy()
        df["v"] = np.where(df["game_duration"] > minuto, df[col], np.nan)
        w = df.pivot_table(index="match_id", columns=["team_id", "lane"], values="v", aggfunc="first").reindex(mids)
        return np.c_[[(w[(100, l)] - w[(200, l)]).to_numpy() / 1000.0 for l in g9.LANES]].T.astype(np.float32)
    g9.cargar_target_oro15 = objetivo
    print(f"=== {col} ===", flush=True)
    out[col] = g9.construir_hgb_causal(D)[:, :5]
np.savez_compressed(REPO / "data" / "analysis" / "brainstorm" / "hgb_lineas_extra.npz", **out)
print("listo")
