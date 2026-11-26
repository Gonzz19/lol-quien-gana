#!/usr/bin/env python3
"""
scripts/analysis/cobertura_perfiles_test.py
===========================================
¿Cuántos jugadores del test tienen perfil (partidas) en el train? Solo lectura de
data/lol_las.db.

Corte T = el día más tardío con al menos el 20 % de las partidas Master+ desde T
(`master_plus_temporal_split`). Compara dos trains:

* A: solo Master+ anterior a T (regla vigente: Diamond aparte).
* B: Master+ y Diamond anteriores a T (variante del usuario, 2026-10-26: el test no
  lleva Diamond, pero el Diamond viejo entra al train para tener perfiles).

Para cada participación del test (10 por partida) cuenta cuántas partidas tiene ese
jugador en el train, desglosado por el tier del jugador en el padrón.

Uso:
    python scripts/analysis/cobertura_perfiles_test.py
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.features.identity import attach_for_participants  # noqa: E402
from src.features.split import master_plus_temporal_split, match_tiers  # noqa: E402

UMBRALES = (1, 10, 20, 50)


def main() -> None:
    conn = sqlite3.connect(f"file:{ROOT / 'data' / 'lol_las.db'}?mode=ro", uri=True, timeout=120.0)
    try:
        attach_for_participants(conn)
        parts = pd.read_sql_query("""
            SELECT p.match_id, COALESCE(pim.player_id, p.puuid) AS player_id, m.game_creation
            FROM match_participants p
            LEFT JOIN player_id_map pim ON pim.puuid = p.puuid
            JOIN matches m ON m.match_id = p.match_id
            WHERE substr(p.match_id, 1, 4) = 'LA2_'""", conn)
        padron = pd.read_sql_query("""
            SELECT pim.player_id, MAX(pl.tier) AS player_tier
            FROM players pl JOIN player_id_map pim ON pim.puuid = pl.puuid
            GROUP BY pim.player_id""", conn)
    finally:
        conn.close()

    tiers = match_tiers(parts, padron)
    m = (parts.drop_duplicates("match_id")[["match_id", "game_creation"]]
         .assign(tier=lambda d: d["match_id"].map(tiers)))
    labels, res = master_plus_temporal_split(m, 0.2)
    print(f"Partidas LAS: {len(m):,} | corte T = {res['corte']}")
    print(f"  test (Master+ desde T): {res['test']:,} = {res['test_sobre_master_plus']:.1%} de Master+")
    print(f"  train Master+: {res['train']:,} | Diamond antes de T: {res['diamante_antes']:,} | "
          f"Diamond desde T: {res['diamante_despues']:,} | sin rango: {res['sin_rango']:,}")
    sin_rango_antes = int(((labels == "sin_rango") & (m.set_index("match_id")["game_creation"]
                                                      < m.loc[m["match_id"].isin(labels[labels == "test"].index),
                                                              "game_creation"].min())).sum())
    print(f"  (sin rango antes de T: {sin_rango_antes:,})")
    print(f"  test por rango: {res['test_por_rango']}\n")

    p = parts.assign(split=parts["match_id"].map(labels))
    tier_jug = padron.set_index("player_id")["player_tier"]
    test = p[p["split"] == "test"].copy()
    test["tier_jugador"] = test["player_id"].map(tier_jug).fillna("FUERA_DEL_PADRON")
    print("Participaciones del test por tier del jugador:")
    print((test["tier_jugador"].value_counts(normalize=True) * 100).round(1).to_string(), "\n")

    for nombre, etiquetas in (("A: train = solo Master+", {"train"}),
                              ("B: train = Master+ y Diamond antes de T", {"train", "diamante_antes"})):
        n_train = p[p["split"].isin(etiquetas)].groupby("player_id").size()
        test["n_train"] = test["player_id"].map(n_train).fillna(0).astype(int)
        filas = []
        for tj, g in [("TODOS", test)] + list(test.groupby("tier_jugador")):
            filas.append({"tier jugador": tj, "participaciones": len(g),
                          **{f">= {u}": f"{(g['n_train'] >= u).mean():.1%}" for u in UMBRALES},
                          "mediana": int(np.median(g["n_train"]))})
        print(f"[{nombre}] partidas del jugador en el train, por participación del test:")
        print(pd.DataFrame(filas).to_string(index=False), "\n")


if __name__ == "__main__":
    main()
