"""Cluster de jugador global (sección 16) vs clustering dentro de cada rol principal.

Solo lectura de data/players.csv y data/player_signatures_A.csv.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

ROOT = sys.argv[1]
MIN_GAMES, K_GLOBAL, SEED = 50, 5, 42
FEATURES = {
    "role_focus_pct": "Role Focus %", "pool_dispersion": "Pool Dispersion",
    "avg_cs_at_15": "CS @15", "avg_total_cs": "Total CS End", "avg_gold_share_pct": "% Gold Share",
    "avg_damage_share_pct": "% Damage Share", "avg_damage_taken": "Damage Taken",
    "avg_early_aggression_10": "Early Agg @10",
    "avg_kill_participation_pct": "% Kill Part", "avg_vision_score": "Vision Score",
    "avg_gold_diff_at_15": "GD vs Rival @15", "avg_cs_diff_at_15": "ΔCS Rival @15",
    "avg_cs_diff_end": "ΔCS Rival End",
    "avg_damage_diff_end": "ΔDamage Rival End", "avg_gold_pct_diff_delta": "Δ%Gold Snowball",
}
F = list(FEATURES)
ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]

players = pd.read_csv(f"{ROOT}/data/players.csv")
sig = pd.read_csv(f"{ROOT}/data/player_signatures_A.csv", usecols=["player_id", "pool_dispersion"])
m = players[players["games_recorded"] >= MIN_GAMES].merge(sig, on="player_id", how="inner")
m = m.dropna(subset=F + ["main_lane"]).reset_index(drop=True)
m = m[m["main_lane"].isin(ROLES)].reset_index(drop=True)
print(f"N = {len(m)} jugadores (>= {MIN_GAMES} partidas, con firma)")
print(m["main_lane"].value_counts().reindex(ROLES).to_string(), "\n")

# 1. Cluster global, igual que la sección 16.
Xg = StandardScaler().fit_transform(m[F])
km = KMeans(K_GLOBAL, random_state=SEED, n_init=15).fit(Xg)
m["c_global"] = km.labels_
sil_global = silhouette_score(Xg, m["c_global"])
print(f"[Global] k = {K_GLOBAL}, silhouette = {sil_global:.3f}")
print(f"  NMI(cluster global, rol principal) = {normalized_mutual_info_score(m['main_lane'], m['c_global']):.3f}"
      f" | ARI = {adjusted_rand_score(m['main_lane'], m['c_global']):.3f}")
ct = pd.crosstab(m["c_global"], m["main_lane"]).reindex(columns=ROLES, fill_value=0)
print("  Cluster global x rol principal (% de cada cluster):")
print((ct.div(ct.sum(1), axis=0) * 100).round(0).astype(int).assign(N=ct.sum(1)).to_string(), "\n")
pureza = (ct.max(1).sum() / ct.values.sum())
print(f"  Pureza de rol de los clusters globales: {pureza:.1%}\n")

# Cuánto de cada variable explica el rol (eta^2 = var entre roles / var total).
eta = {}
for f in F:
    g = m.groupby("main_lane")[f]
    eta[f] = (g.size() * (g.mean() - m[f].mean()) ** 2).sum() / ((m[f] - m[f].mean()) ** 2).sum()
eta = pd.Series(eta).sort_values(ascending=False)
print("  eta^2 del rol por variable (fracción de la varianza que explica el rol):")
print("  " + ", ".join(f"{FEATURES[k]} {v:.2f}" for k, v in eta.items()))
print(f"  Promedio: {eta.mean():.2f}\n")

# 2. Cluster dentro de cada rol: estandarizado DENTRO del rol, barrido de k.
print("[Por rol] (estandarización dentro del rol)")
rows = []
for role in ROLES:
    sub = m[m["main_lane"] == role]
    Xr = StandardScaler().fit_transform(sub[F])
    sils = {}
    for k in range(2, 7):
        lb = KMeans(k, random_state=SEED, n_init=10).fit_predict(Xr)
        sils[k] = silhouette_score(Xr, lb)
    k_best = max(sils, key=sils.get)
    lb3 = KMeans(3, random_state=SEED, n_init=15).fit_predict(Xr)
    m.loc[sub.index, "c_rol"] = lb3
    # ¿El cluster global separa algo dentro del rol? (cuántos clusters globales usa el rol)
    glob_in_role = sub["c_global"].value_counts(normalize=True)
    ari = adjusted_rand_score(sub["c_global"], lb3)
    nmi = normalized_mutual_info_score(sub["c_global"], lb3)
    # Variables que definen los clusters del rol (mayor |z| de centroide).
    z = pd.DataFrame(Xr, index=sub.index).groupby(lb3).mean()
    z.columns = [FEATURES[f] for f in F]
    tops = []
    for c in z.index:
        s = z.loc[c].abs().sort_values(ascending=False).index[:2]
        tops.append(f"C{c}({(lb3 == c).mean():.0%}): " + ", ".join(f"{v} {z.loc[c, v]:+.1f}σ" for v in s))
    rows.append((role, len(sub), sils[3], k_best, sils[k_best], ari, nmi,
                 glob_in_role.iloc[0], glob_in_role.index[0]))
    print(f"  {role:8s} N={len(sub):5d} | sil(k=3)={sils[3]:.3f} | k_sil_max={k_best} ({sils[k_best]:.3f})"
          f" | ARI/NMI vs global={ari:.2f}/{nmi:.2f} | cluster global dominante C{glob_in_role.index[0]}"
          f" = {glob_in_role.iloc[0]:.0%}")
    for t in tops:
        print("      " + t)

# 3. Global sobre variables centradas por rol (quitar el efecto rol, un solo modelo).
Z = m[F].copy()
Z = Z.groupby(m["main_lane"]).transform(lambda s: (s - s.mean()) / s.std())
Xc = Z.values
km_c = KMeans(K_GLOBAL, random_state=SEED, n_init=15).fit(Xc)
print(f"\n[Global centrado por rol] k = {K_GLOBAL}, silhouette = {silhouette_score(Xc, km_c.labels_):.3f}"
      f" | NMI con rol = {normalized_mutual_info_score(m['main_lane'], km_c.labels_):.3f}"
      f" | ARI vs global original = {adjusted_rand_score(m['c_global'], km_c.labels_):.2f}")
sils_c = {k: silhouette_score(Xc, KMeans(k, random_state=SEED, n_init=10).fit_predict(Xc)) for k in range(2, 7)}
print("  silhouette por k: " + ", ".join(f"k={k}: {v:.3f}" for k, v in sils_c.items()))
