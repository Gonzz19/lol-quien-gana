"""K-Means contra e-means (FAISS 1.16) sobre el clustering de jugadores (PROJECT_SUMMARY.md §9.S.53).

Mismos datos que la §1.1 de `eda_reconocimiento_jugadores.ipynb` (ex sección 16 de
`clustering_jugadores.ipynb`) y mismas 15 variables que
`cluster_global_vs_rol.py`: jugadores con 50 partidas o más, 15 variables. Tres escenarios:

* `global`: estandarizado sobre todos (como la sección 16).
* `centrado_rol`: cada variable estandarizada dentro del rol principal (saca el efecto rol).
* `rol_<ROL>`: un clustering por rol, estandarizado dentro del rol.

Por cada k y método: SSE (mejor y mediana de 10 semillas), silhouette, estabilidad entre
semillas (ARI medio entre pares), tiempo y ARI contra K-Means. Exploratorio: `players.csv`
mezcla train y test, igual que el notebook.

    python scripts/analysis/cluster_emeans.py [--kmax 8] [--semillas 10]

Escribe `data/analysis/cluster_emeans.csv`. Solo lectura de `data/`.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.clustering import comparar  # noqa: E402

MIN_GAMES = 50
F = ["role_focus_pct", "pool_dispersion", "avg_cs_at_15", "avg_total_cs", "avg_gold_share_pct",
     "avg_damage_share_pct", "avg_damage_taken", "avg_early_aggression_10", "avg_kill_participation_pct",
     "avg_vision_score", "avg_gold_diff_at_15", "avg_cs_diff_at_15", "avg_cs_diff_end",
     "avg_damage_diff_end", "avg_gold_pct_diff_delta"]
ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]


def cargar() -> pd.DataFrame:
    players = pd.read_csv(ROOT / "data" / "players.csv")
    sig = pd.read_csv(ROOT / "data" / "player_signatures_A.csv", usecols=["player_id", "pool_dispersion"])
    m = players[players["games_recorded"] >= MIN_GAMES].merge(sig, on="player_id", how="inner")
    m = m.dropna(subset=F + ["main_lane"])
    return m[m["main_lane"].isin(ROLES)].reset_index(drop=True)


def escenarios(m: pd.DataFrame):
    yield "global", StandardScaler().fit_transform(m[F])
    yield "centrado_rol", m[F].groupby(m["main_lane"]).transform(lambda s: (s - s.mean()) / s.std()).values
    for rol in ROLES:
        yield f"rol_{rol}", StandardScaler().fit_transform(m.loc[m["main_lane"] == rol, F])


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--semillas", type=int, default=10)
    args = ap.parse_args()

    m = cargar()
    print(f"N = {len(m)} jugadores (>= {MIN_GAMES} partidas, con firma)")
    filas = []
    for nombre, X in escenarios(m):
        print(f"\n[{nombre}] n = {len(X)}")
        fs = comparar(X, range(2, args.kmax + 1), semillas=range(args.semillas))
        for f in fs:
            f["escenario"], f["n"] = nombre, len(X)
        filas += fs
        t = pd.DataFrame(fs).pivot(index="k", columns="metodo")
        print(t[["sse_mejor", "silhouette", "estabilidad", "ari_vs_kmeans", "segundos"]]
              .round({"sse_mejor": 0, "silhouette": 3, "estabilidad": 2, "ari_vs_kmeans": 2, "segundos": 2})
              .to_string())
    out = pd.DataFrame(filas)
    destino = ROOT / "data" / "analysis" / "cluster_emeans.csv"
    destino.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(destino, index=False)
    print(f"\n[+] {destino}")


if __name__ == "__main__":
    main()
