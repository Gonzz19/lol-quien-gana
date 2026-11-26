"""
Registro del LP de cada jugador al momento de cada partida, para el período entre
fotos (04/09 → 28/10; PROJECT_SUMMARY.md §9.S.21, "Registro de LP por partida").

Según qué fotos tiene el jugador:

* **puente** (las dos fotos): se deshace la regla desde la foto 2 y el camino se
  ancla en el LP real de la foto 1. Las partidas no observadas se descartan en la
  proporción real del jugador (si vimos 1 de cada 7, cada paso deshace 1 observada
  y 6 no observadas con su cambio esperado según el winrate total).
* **desde_foto2** (solo foto 2): NGBoost predice el LP de la foto 1 (residuo de la
  regla, con el winrate total entre las variables) y el camino se ancla ahí. Las no
  observadas, con la tasa de su rango.
* **desde_foto1** (solo foto 1): se avanza con la regla desde la foto 1, con la tasa
  de no observadas de su rango.
* **lobby** (ninguna foto, o UNRANKED en la foto 2): LP medio de los otros 9 de la
  partida en la foto 2.

Lo anterior al 04/09 (test) no se carga. Salida: `data/lp_por_partida.csv` con
`match_id, player_id, game_creation, lp` (LP absoluto antes de la partida), `rango` y
`metodo`. Es descriptivo: el puente y `desde_foto2` miran partidas posteriores, así que
no sirve como variable de un modelo que prediga el resultado de la partida.

    python scripts/analysis/build_lp_partida.py     # ~10 min (NGBoost ~7)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.elo_reverso import (adelante, ajustar_regla, al_reves, camino, cargar_train, cortes,  # noqa: E402
                                      evaluables, grupo_de, lobby_por_fila, rango_de, variables_estaticas)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    args = ap.parse_args()
    t0 = time.time()
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    from sklearn.tree import DecisionTreeRegressor

    d = cargar_train(Path(args.data_dir))
    foto1, foto2, partes = d["foto1"], d["foto2"], d["partes"]
    e = evaluables(foto1, foto2, partes)
    corte = cortes(foto2)
    a, b = ajustar_regla(e)
    g2 = e["z2"].map(lambda z: grupo_de(z, corte))
    tasa = (e.groupby(g2)["ocultas"].sum() / e.groupby(g2)["obs"].sum()).to_dict()
    proporcion = (e["ocultas"] / e["obs"].replace(0, np.nan)).fillna(0.0)
    wr2 = (foto2["wins"] / foto2["partidas"].replace(0, np.nan)).fillna(0.5)
    wr1 = (foto1["wins"] / foto1["partidas"].replace(0, np.nan)).fillna(0.5)

    jugadores = pd.Index(partes["player_id"].unique())
    puente = jugadores.intersection(e.index)
    solo2 = jugadores.intersection(foto2.index).difference(puente)
    solo1 = jugadores.intersection(foto1.index).difference(foto2.index)
    print(f"[i] {len(partes):,} filas. Jugadores: puente {len(puente):,}, desde foto 2 {len(solo2):,}, "
          f"desde foto 1 {len(solo1):,}, sin foto {len(jugadores) - len(puente) - len(solo2) - len(solo1):,}. "
          f"Carga {time.time() - t0:.0f} s", flush=True)

    # NGBoost entrenado con los que tienen las dos fotos; predice la foto 1 de los que solo tienen la 2.
    base = al_reves(partes, foto2["z"], a, b, corte)
    obs = partes.groupby("player_id")["win"].agg(obs="size", obs_w="sum")
    e2 = foto2.loc[solo2, ["z", "wins", "partidas"]].rename(columns={"z": "z2", "wins": "wins2", "partidas": "partidas2"})
    e2 = e2.join(obs).assign(ocultas=np.nan)
    X = variables_estaticas(e, partes, foto2["z"], base.reindex(e.index).fillna(e["z2"])).drop(columns="ocultas")
    X2 = variables_estaticas(e2, partes, foto2["z"], base.reindex(e2.index).fillna(e2["z2"])).drop(columns="ocultas")
    ngb = NGBRegressor(Dist=Normal, n_estimators=800, learning_rate=0.05, verbose=False, random_state=0,
                       Base=DecisionTreeRegressor(max_depth=5, min_samples_leaf=50))
    ngb.fit(X.fillna(-999), (e["z1"] - X["pred_regla"]) / 100)
    z1_pred = X2["pred_regla"] + 100 * ngb.predict(X2[X.columns].fillna(-999))
    print(f"[i] NGBoost listo ({time.time() - t0:.0f} s)", flush=True)

    lob = lobby_por_fila(partes, foto2["z"])["lobby"]
    filas = []
    for pid, p in partes.groupby("player_id", sort=False):
        p = p.sort_values("game_creation")
        if pid in puente:
            lp = camino(p, e.at[pid, "z2"], e.at[pid, "z1"], a, b, corte,
                        no_observadas=proporcion[pid], wr=wr2[pid])[:-1]
            metodo = "puente"
        elif pid in solo2 and foto2.at[pid, "z"] == foto2.at[pid, "z"]:   # UNRANKED en la foto 2 → lobby
            lp = camino(p, foto2.at[pid, "z"], z1_pred[pid], a, b, corte, no_observadas=tasa, wr=wr2[pid])[:-1]
            metodo = "desde_foto2"
        elif pid in solo1:
            lp = adelante(p, foto1.at[pid, "z"], a, b, corte, no_observadas=tasa, wr=wr1[pid])[:-1]
            metodo = "desde_foto1"
        else:
            lp = lob[p.index].to_numpy()
            metodo = "lobby"
        filas.append(pd.DataFrame({"match_id": p["match_id"].to_numpy(), "player_id": pid,
                                   "game_creation": p["game_creation"].to_numpy(), "lp": lp, "metodo": metodo}))
    out = pd.concat(filas, ignore_index=True)
    out["rango"] = rango_de(out["lp"].fillna(-1e9), corte)
    out.loc[out["lp"].isna(), "rango"] = np.nan
    out = out.sort_values(["game_creation", "match_id", "player_id"], kind="mergesort")
    out["lp"] = out["lp"].round(1)

    destino = Path(args.data_dir) / "lp_por_partida.csv"
    out[["match_id", "player_id", "game_creation", "lp", "rango", "metodo"]].to_csv(destino, index=False)
    resumen = out["metodo"].value_counts()
    print("[+] Filas por método:", {k: f"{v:,} ({100 * v / len(out):.1f} %)" for k, v in resumen.items()})
    print(f"[+] {destino} ({len(out):,} filas, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
