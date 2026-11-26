"""Superconjunto de variables por partida y estáticas para el banco de la GRU → seq.pkl."""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import AQUI, cargar  # noqa

ROOT = Path(r"c:/Users/DarkA/OneDrive/Escritorio/Facultad/3er año/ML/TPF/ProyectoFinalML")
sys.path.insert(0, str(ROOT))
from src.features.elo_reverso import grupo_de  # noqa

t0 = time.time()
c = cargar()
e, partes = c["e"], c["partes"]
# Rendimiento de cada jugador en cada partida (no mueve LP, pero dice si juega por encima de su lobby).
perf_cols = ["kda", "kill_participation_pct", "damage_share_pct", "gold_share_pct", "deaths", "vision_score",
             "game_duration"]
trozos = []
for ch in pd.read_csv(ROOT / "data" / "full_dataset.csv", chunksize=500_000,
                      usecols=["match_id", "player_id", "game_creation"] + perf_cols):
    trozos.append(ch[ch["match_id"].str.startswith("LA2_") & (ch["game_creation"] >= c["t1"])
                     & (ch["game_creation"] < c["t2"])])
perf = pd.concat(trozos).drop_duplicates(["match_id", "player_id"]).set_index(["match_id", "player_id"])
print(f"rendimiento cargado ({time.time() - t0:.0f} s)", flush=True)

p = partes[partes["player_id"].isin(e.index)].copy()
pf = perf.reindex(pd.MultiIndex.from_frame(p[["match_id", "player_id"]]))
lp = c["lp_fila"][p.index]
lobp, lob2 = c["lobp"].loc[p.index], c["lob2"].loc[p.index]
z2p = p["player_id"].map(e["z2"])
f = pd.DataFrame({
    "player_id": p["player_id"].to_numpy(), "t": p["game_creation"].to_numpy(),
    "resultado": np.where(p["win"] == 1, 1.0, -1.0),
    "aliados": ((lobp["aliados"] - lp) / 300).fillna(0).to_numpy(),
    "rivales": ((lobp["rivales"] - lp) / 300).fillna(0).to_numpy(),
    "lobby": ((lobp["lobby"] - 500) / 300).fillna(0).to_numpy(),
    "gap": ((lobp["lobby"] - lp) / 300).fillna(0).to_numpy(),
    "lobby2_rel": ((lob2["lobby"] - z2p) / 300).fillna(0).to_numpy(),
    "lp_paso": ((lp - z2p) / 300).fillna(0).to_numpy(),
    "conocidos": (lobp["conocidos"] / 9).to_numpy(),
    "dias": ((c["t2"] - p["game_creation"]) / 86_400_000 / 50).to_numpy(),
    "azul": (p["team_id"] == 100).astype(float).to_numpy(),
    "kda": np.log1p(pf["kda"].fillna(2).clip(0, 50)).to_numpy() - 1,
    "kp": (pf["kill_participation_pct"].fillna(50) / 100).to_numpy() - 0.5,
    "dmg": (pf["damage_share_pct"].fillna(20) / 100).to_numpy() - 0.2,
    "oro": (pf["gold_share_pct"].fillna(20) / 100).to_numpy() - 0.2,
    "muertes": (pf["deaths"].fillna(5) / 10).to_numpy() - 0.5,
    "duracion": (pf["game_duration"].fillna(1700) / 1800).to_numpy() - 1,
})
f = f.sort_values(["player_id", "t"], ascending=[True, False])
nuevo = f.groupby("player_id")["t"].shift(1).fillna(c["t2"])
f["hueco"] = ((nuevo - f["t"]) / 86_400_000 / 14).to_numpy()
f["racha"] = f.groupby("player_id")["resultado"].transform(lambda s: s.rolling(5, min_periods=1).mean())
PASO = [col for col in f.columns if col not in ("player_id", "t")]
grupos = {pid: g[PASO].to_numpy(np.float32) for pid, g in f.groupby("player_id", sort=False)}
vacia = np.zeros((1, len(PASO)), np.float32)
seqs = [grupos.get(pid, vacia) for pid in e.index]

ps = partes.sort_values("game_creation")
gc = ps.groupby("player_id")["game_creation"]
hueco_max = (gc.diff() / 86_400_000).groupby(ps["player_id"]).max().reindex(e.index).fillna(0)
dias_fin = ((c["t2"] - gc.max()) / 86_400_000).reindex(e.index).fillna(55)
dias_ini = ((gc.min() - c["t1"]) / 86_400_000).reindex(e.index).fillna(55)
g2 = e["z2"].map(lambda v: grupo_de(v, c["corte"]))
est = pd.DataFrame({
    "z2": (e["z2"] - 500) / 300, "pred_regla": (c["regla"] - 500) / 300, "partidas2": e["partidas2"] / 500,
    "wr_temporada": e["wins2"] / e["partidas2"], "obs": e["obs"] / 100, "obs_w": e["obs_w"] / 100,
    "kal_media": (c["km"] - e["z2"]) / 100, "kal_sd": np.sqrt(c["kv"]).fillna(0) / 100, "decay": c["decay"] / 500,
    "piso": e["z2"].isin([275.0, 325.0, 375.0]).astype(float), "dias_fin": dias_fin / 30, "dias_ini": dias_ini / 30,
    "hueco_max": hueco_max / 30, "g_ch": (g2 == 0).astype(float), "g_gm": (g2 == 1).astype(float),
    "g_ma": (g2 == 2).astype(float),
    "lobby_viejo": ((c["X_comp"]["lobby_viejo"] - e["z2"]) / 300).fillna(0),
    "lobby_reciente": ((c["X_comp"]["lobby_reciente"] - e["z2"]) / 300).fillna(0),
}, index=e.index).astype(np.float32)
with open(AQUI / "seq.pkl", "wb") as fh:
    pickle.dump({"PASO": PASO, "seqs": seqs, "est": est}, fh)
print(f"PASO = {PASO}\nEST = {list(est.columns)}\nlisto ({time.time() - t0:.0f} s)")
