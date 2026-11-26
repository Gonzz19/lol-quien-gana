"""Duos con el nulo correcto (§9.S.27): con emparejamiento por roles, dos jugadores de líneas distintas caen juntos con P = 1/2, así que "siempre juntos" contra "siempre enfrentados" vale 1 por azar. Uso: python historial/ganador_primeros/duos_roles.py . (solo data/splits/train)."""
import sys, itertools
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(sys.argv[1])
LANES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
df = pd.read_csv(REPO / "data/splits/train/full_dataset.csv",
                 usecols=["match_id", "game_creation", "player_id", "team_id", "win", "lane"])
df = df[df.lane.isin(LANES)]
df = df[df.groupby(["match_id", "team_id"]).lane.transform("nunique") == 5]
df = df[df.groupby("match_id").lane.transform("size") == 10]
df["li"] = df.lane.map({l: i for i, l in enumerate(LANES)})
df = df.sort_values(["game_creation", "match_id", "team_id", "li"]).reset_index(drop=True)
tier = pd.read_csv(REPO / "data/splits/match_split.csv", usecols=["match_id", "tier"]).set_index("match_id").tier
pid = df.player_id.astype("category").cat.codes.to_numpy().reshape(-1, 10)
win = df.win.to_numpy().reshape(-1, 10)
mid = df.match_id.to_numpy()[::10]
lob = tier.reindex(mid).to_numpy()
n = len(pid)
# rol principal de cada jugador: el más jugado en el train (descriptivo)
rol = df.groupby("player_id").li.agg(lambda s: s.value_counts().idxmax())
rol_c = pd.Series(rol.to_numpy(), index=pd.Categorical(rol.index, categories=df.player_id.astype("category").cat.categories).codes)
partes = []
for i, j in itertools.combinations(range(10), 2):
    if i % 5 == j % 5:          # misma línea en esta partida: son rivales directos, no pueden ir juntos
        continue
    partes.append(pd.DataFrame({"m": np.arange(n), "i": i, "j": j, "a": np.minimum(pid[:, i], pid[:, j]),
                                "b": np.maximum(pid[:, i], pid[:, j]), "junto": int((i < 5) == (j < 5))}))
P = pd.concat(partes, ignore_index=True).sort_values(["m", "i", "j"], kind="stable").reset_index(drop=True)
P["lob"] = lob[P.m]
g = P.groupby(["a", "b"], sort=False)
P["pj"] = g.junto.cumsum() - P.junto
P["pc"] = g.cumcount() - P.pj
print("Pares de líneas distintas en cada cruce: bajo azar P(juntos) = 1/2 (4/8 de los otros puestos con otra línea)")
print("  P(juntos) en el primer cruce:", " ".join(f"{t}: {P[(P.pj + P.pc == 0) & (P.lob == t)].junto.mean():.3f}"
                                            for t in ("DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER")))
fin = P.groupby(["a", "b"]).agg(k=("junto", "size"), j=("junto", "sum"), m0=("m", "first"))
fin["lob"] = lob[fin.m0]
print("\n1) pares con k cruces (líneas distintas): siempre juntos / siempre enfrentados (azar: 1)")
for t_ in ("DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER"):
    f = fin[fin.lob == t_]
    fila = []
    for k in (3, 4, 5, 6):
        s = f[f.k == k]
        J, C = (s.j == k).sum(), (s.j == 0).sum()
        fila.append(f"k={k}: {J}/{C} = {J / max(C, 1):.2f} (reales≈{max(J - C, 0) / max(J, 1):.0%})")
    print(f"  {t_:12s}", " | ".join(fila))

print("\n2) WR del equipo del par HOY JUNTOS según su historia previa (pj juntos, pc enfrentados)")
H = P[P.junto == 1].copy()
H["gana"] = win[H.m, H.i]
for t_ in ("DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER"):
    s = H[H.lob == t_]
    celdas = []
    for pj, pc in ((0, 0), (1, 0), (2, 0), (3, 0), (5, 0), (0, 1), (0, 2), (0, 3), (1, 1), (2, 1), (1, 2)):
        c = s[(s.pj == pj) & (s.pc == pc)] if pj < 5 else s[(s.pj >= 5) & (s.pc == 0)]
        celdas.append(f"{pj}{'+' if pj == 5 else ''}/{pc}: {c.gana.mean():.3f} ({len(c)})")
    print(f"  {t_:12s}", "  ".join(celdas))

# ---- 3. resultado previo del par y si la partida anterior de los dos fue la misma
gc = df.game_creation.to_numpy()[::10]
P["gana_eq"] = win[P.m, P.i]                                   # gana el equipo del primero
g = P.groupby(["a", "b"], sort=False)
P["m_prev"] = g.m.shift()
last, prev_m = {}, np.full((n, 10), -1)
for m in range(n):
    for s_ in range(10):
        prev_m[m, s_] = last.get(pid[m, s_], -1); last[pid[m, s_]] = m
P["seguida"] = (prev_m[P.m, P.i] == prev_m[P.m, P.j]) & (prev_m[P.m, P.i] == P.m_prev)
P["prev_junto_gano"] = g.gana_eq.shift()                      # resultado del equipo de i en el cruce previo
P["prev_era_junto"] = g.junto.shift()
H = P[(P.junto == 1)].copy()
H["gana"] = win[H.m, H.i]
print("\n3) Par hoy juntos, cuya partida anterior de los dos fue la misma y también juntos:")
for t_ in ("DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER"):
    s = H[(H.lob == t_) & H.seguida & (H.prev_era_junto == 1)]
    gan, per = s[s.prev_junto_gano == 1], s[s.prev_junto_gano == 0]
    s2 = s[(s.pj >= 3) & (s.pc == 0)]
    print(f"  {t_:12s} seguidas {len(s):6d}  WR tras ganar juntos {gan.gana.mean():.3f} ({len(gan)})  "
          f"tras perder juntos {per.gana.mean():.3f} ({len(per)})  | pj>=3,pc=0: tras ganar "
          f"{s2[s2.prev_junto_gano == 1].gana.mean():.3f} ({(s2.prev_junto_gano == 1).sum()}) tras perder "
          f"{s2[s2.prev_junto_gano == 0].gana.mean():.3f} ({(s2.prev_junto_gano == 0).sum()})")
    ns = H[(H.lob == t_) & ~H.seguida & (H.pj >= 3) & (H.pc == 0)]
    print(f"  {'':12s} pj>=3,pc=0 NO seguidas: WR {ns.gana.mean():.3f} ({len(ns)})")
