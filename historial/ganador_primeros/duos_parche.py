"""Duos reales por parche y tier del lobby: pares de líneas distintas, siempre juntos / siempre enfrentados
(azar = 1), §9.S.27. Solo data/splits/train; no mira resultados. Uso: python historial/ganador_primeros/duos_parche.py ."""
import sys, itertools
import numpy as np, pandas as pd
REPO = sys.argv[1]
LANES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
df = pd.read_csv(f"{REPO}/data/splits/train/full_dataset.csv", usecols=["match_id", "game_creation", "game_version", "player_id", "team_id", "lane"])
df = df[df.lane.isin(LANES)]
df = df[df.groupby(["match_id", "team_id"]).lane.transform("nunique") == 5]
df = df[df.groupby("match_id").lane.transform("size") == 10]
df["li"] = df.lane.map({l: i for i, l in enumerate(LANES)})
df = df.sort_values(["game_creation", "match_id", "team_id", "li"]).reset_index(drop=True)
tier = pd.read_csv(f"{REPO}/data/splits/match_split.csv", usecols=["match_id", "tier"]).set_index("match_id").tier
pid = df.player_id.astype("category").cat.codes.to_numpy().reshape(-1, 10)
mid = df.match_id.to_numpy()[::10]
parche = df.game_version.str.split(".").str[:2].str.join(".").to_numpy()[::10]
lob = tier.reindex(mid).to_numpy()
n = len(pid)
partes = []
for i, j in itertools.combinations(range(10), 2):
    if i % 5 != j % 5:
        partes.append(pd.DataFrame({"m": np.arange(n), "a": np.minimum(pid[:, i], pid[:, j]),
                                    "b": np.maximum(pid[:, i], pid[:, j]), "junto": int((i < 5) == (j < 5))}))
P = pd.concat(partes, ignore_index=True)
P["parche"], P["lob"] = parche[P.m], lob[P.m]
# pares que se cruzan >= 4 veces DENTRO del mismo parche y tier de lobby
f = P.groupby(["parche", "lob", "a", "b"]).junto.agg(["size", "sum"]).reset_index()
f = f[f["size"] >= 4]
print("pares con >= 4 cruces en el parche: siempre juntos / siempre enfrentados (azar = 1); partidas del parche")
nm = pd.Series(parche).groupby([parche, lob]).size()
for p in sorted(set(parche), key=lambda v: [int(x) for x in v.split(".")]):
    fila = []
    for t in ("DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER"):
        s = f[(f.parche == p) & (f.lob == t)]
        J, C = (s["sum"] == s["size"]).sum(), (s["sum"] == 0).sum()
        fila.append(f"{t[:4]} {J:4d}/{C:4d}{'' if C < 10 else f' = {J / C:5.2f}'} ({nm.get((p, t), 0):6d})")
    print(f"  {p:6s}", " | ".join(fila))
