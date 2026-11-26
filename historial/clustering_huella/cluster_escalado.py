"""Clustering por rol con (jugador, línea): ventaja a los 15 vs escalado vs solo estilo.

Solo lectura de los CSV de data/. Cada fila = (jugador, línea) con las partidas de ESA línea.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

ROOT, SEED = sys.argv[1], 42
MIN_PLAYER_GAMES, MIN_LANE_GAMES, MIN_LANE_PCT = 50, 20, 15.0
MIN_DUR_SCAL = 20 * 60  # el escalado se mide en partidas de >= 20 min
ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]

fd = pd.read_csv(f"{ROOT}/data/full_dataset.csv", usecols=[
    "match_id", "player_id", "lane", "win", "kill_participation_pct", "damage_share_pct",
    "gold_share_pct", "total_damage_taken", "total_minions_killed", "neutral_minions_killed",
    "vision_score", "cs_15", "early_aggression_10"])
fd = fd[fd["lane"].isin(ROLES)]
fd["total_cs"] = fd["total_minions_killed"] + fd["neutral_minions_killed"]
pl = pd.read_csv(f"{ROOT}/data/players.csv", usecols=["player_id", "games_recorded"])
mature = set(pl.loc[pl["games_recorded"] >= MIN_PLAYER_GAMES, "player_id"])
fd = fd[fd["player_id"].isin(mature)]

own = fd.groupby(["player_id", "lane"]).agg(
    games=("win", "size"), wr=("win", "mean"), kp=("kill_participation_pct", "mean"),
    dmg_share=("damage_share_pct", "mean"), gold_share=("gold_share_pct", "mean"),
    dmg_taken=("total_damage_taken", "mean"), total_cs=("total_cs", "mean"),
    vision=("vision_score", "mean"), cs15=("cs_15", "mean"), early_agg=("early_aggression_10", "mean"),
).reset_index()
own["role_focus"] = 100 * own["games"] / own.groupby("player_id")["games"].transform("sum")

mu = pd.read_csv(f"{ROOT}/data/matchups_1v1.csv", usecols=[
    "game_duration", "lane", "blue_player_id", "red_player_id", "diff_gold_15", "diff_cs_15",
    "diff_damage_15", "diff_cs_end", "diff_damage_end", "diff_gold_pct_snowball"])
dcols = ["diff_gold_15", "diff_cs_15", "diff_damage_15", "diff_cs_end", "diff_damage_end", "diff_gold_pct_snowball"]
b = mu.rename(columns={"blue_player_id": "player_id"}).drop(columns="red_player_id")
r = mu.rename(columns={"red_player_id": "player_id"}).drop(columns="blue_player_id")
r[dcols] = -r[dcols]
x = pd.concat([b, r], ignore_index=True)
x = x[x["player_id"].isin(mature)]
mins = x["game_duration"] / 60
# Escalado = ritmo con que crece la diferencia después del 15 menos el ritmo antes del 15
# (unidades por minuto). Positivo: amplía la ventaja (o recorta la desventaja) en el late.
late = x["game_duration"] >= MIN_DUR_SCAL
x["cs_scal"] = np.where(late, (x["diff_cs_end"] - x["diff_cs_15"]) / (mins - 15) - x["diff_cs_15"] / 15, np.nan)
x["dmg_scal"] = np.where(late, (x["diff_damage_end"] - x["diff_damage_15"]) / (mins - 15) - x["diff_damage_15"] / 15, np.nan)
x["gold_scal"] = np.where(late, x["diff_gold_pct_snowball"], np.nan)
riv = x.groupby(["player_id", "lane"])[["diff_gold_15", "diff_cs_15", "diff_cs_end", "diff_damage_end",
                                         "diff_gold_pct_snowball", "cs_scal", "dmg_scal", "gold_scal"]].mean().reset_index()

sig = pd.read_csv(f"{ROOT}/data/player_signatures_B.csv", usecols=["player_id", "lane", "n_games", "pool_dispersion"])
sig = sig[sig["n_games"] >= 10].drop(columns="n_games")

d = own.merge(riv, on=["player_id", "lane"], how="inner").merge(sig, on=["player_id", "lane"], how="left")
d = d[(d["games"] >= MIN_LANE_GAMES) & (d["role_focus"] >= MIN_LANE_PCT)]
d["pool_dispersion"] = d.groupby("lane")["pool_dispersion"].transform(lambda s: s.fillna(s.median()))

STYLE = ["role_focus", "pool_dispersion", "cs15", "total_cs", "gold_share", "dmg_share", "dmg_taken",
         "early_agg", "kp", "vision"]
LEVEL = ["diff_gold_15", "diff_cs_15", "diff_cs_end", "diff_damage_end", "diff_gold_pct_snowball"]
SCAL = ["gold_scal", "cs_scal", "dmg_scal"]
SETS = {"A actual (estilo + ventaja)": STYLE + LEVEL,
        "B estilo + escalado": STYLE + SCAL,
        "C solo estilo": STYLE}
d = d.dropna(subset=STYLE + LEVEL + SCAL)
print(f"(jugador, línea) con >= {MIN_LANE_GAMES} partidas en la línea y >= {MIN_LANE_PCT:.0f} %: {len(d)}")
print(d["lane"].value_counts().reindex(ROLES).to_string(), "\n")

print("¿El escalado sigue midiendo fuerza? corr(variable, WR en la línea), todos los roles:")
for f in LEVEL + SCAL + ["gold_share", "dmg_share", "kp"]:
    print(f"  {f:24s} {d[f].corr(d['wr']):+.2f}")
print("corr(escalado, ventaja al 15): "
      f"cs {d['cs_scal'].corr(d['diff_cs_15']):+.2f}, gold {d['gold_scal'].corr(d['diff_gold_15']):+.2f}, "
      f"dmg {d['dmg_scal'].corr(d['diff_damage_end']):+.2f}\n")


def eta2(y, g):
    m = y.mean()
    return float(sum(len(v) * (v.mean() - m) ** 2 for _, v in y.groupby(g)) / ((y - m) ** 2).sum())


rows = []
for role in ROLES:
    sub = d[d["lane"] == role]
    for name, feats in SETS.items():
        X = StandardScaler().fit_transform(sub[feats])
        sil = {}
        for k in range(2, 7):
            lb = KMeans(k, random_state=SEED, n_init=10).fit_predict(X)
            sil[k] = silhouette_score(X, lb)
        lb3 = KMeans(3, random_state=SEED, n_init=15).fit_predict(X)
        wr_c = sub["wr"].groupby(lb3).mean()
        z = pd.DataFrame(X, columns=feats).groupby(lb3).mean()
        desc = []
        for c in z.index:
            top = z.loc[c].abs().sort_values(ascending=False).index[:2]
            desc.append(f"{(lb3 == c).mean():.0%} WR{wr_c[c]:.1%} [" + ", ".join(f"{t} {z.loc[c, t]:+.1f}" for t in top) + "]")
        rows.append({"rol": role, "set": name, "N": len(sub), "k_best": max(sil, key=sil.get),
                     **{f"sil_k{k}": round(v, 3) for k, v in sil.items()},
                     "rango_WR_k3_pp": round(100 * (wr_c.max() - wr_c.min()), 1),
                     "eta2_WR_k3": round(eta2(sub["wr"].reset_index(drop=True), pd.Series(lb3)), 3),
                     "clusters_k3": " | ".join(desc)})

res = pd.DataFrame(rows)
pd.set_option("display.width", 250, "display.max_colwidth", 400)
print(res.drop(columns="clusters_k3").to_string(index=False))
print()
for _, r in res.iterrows():
    print(f"{r['rol']:8s} {r['set']:28s} {r['clusters_k3']}")
