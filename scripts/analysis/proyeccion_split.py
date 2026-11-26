#!/usr/bin/env python3
"""
scripts/analysis/proyeccion_split.py
====================================
Proyección del split train/test (src/features/split.py) sobre el dataset FINAL,
mientras la recolección sigue corriendo. Solo lectura de data/lol_las.db.

Dataset proyectado = partidas ya bajadas + pendientes de la ventana del 01/09 +
estimación del paso 3 (lo jugado después del listado, hasta la hora de corte).

* Fecha de una pendiente: interpolada por el número de `match_id`, que en LA2 crece
  con el tiempo (se ajusta sobre las partidas ya bajadas).
* Rango de una pendiente: la regla de `match_tiers` aplicada a los jugadores Master+
  del padrón que la listaron (los Diamond no se listaron). Se corrige con la matriz P(rango real | rango del listado) medida en las
  partidas de la ventana que ya están bajadas, sorteando con semilla fija.
* Descartes (remakes, errores): se aplica la tasa observada en la ventana.
* Paso 3: se replica la última semana completa proyectada, escalada a los días que
  quedan entre el fin del listado y el corte.

Uso:
    python scripts/analysis/proyeccion_split.py --corte "2026-10-26 06:00"
"""
import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.features.identity import attach_for_participants  # noqa: E402
from src.features.split import (ART, BASE_CUOTA_DESDE, SIN_RANGO, SOBRANTE_AL_TEST, TEST_TIERS,  # noqa: E402
                                TIER_ORDER, master_plus_temporal_split, match_tiers,
                                proportional_temporal_split)

WINDOW = "2026-09-01.."
DAY_MS = 86_400_000


def _num(ids: pd.Series) -> np.ndarray:
    return ids.str.slice(4).astype(np.int64).to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corte", default="2026-10-26 06:00", help="Hora de corte de la recolección (ART)")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--base-completa", action="store_true",
                    help="Variante: cuota sobre todo el train (regla previa al 25/10)")
    ap.add_argument("--fusionar-apex", action="store_true",
                    help="Variante: GM y Challenger como un solo estrato (GRANDMASTER) para el split")
    args = ap.parse_args()
    corte_ms = int(datetime.strptime(args.corte, "%Y-%m-%d %H:%M").replace(tzinfo=ART).timestamp() * 1000)
    rng = np.random.default_rng(42)

    conn = sqlite3.connect(f"file:{ROOT / 'data' / 'lol_las.db'}?mode=ro", uri=True, timeout=120.0)
    try:
        attach_for_participants(conn)
        parts = pd.read_sql_query("""
            SELECT p.match_id, pim.player_id, m.game_creation
            FROM match_participants p
            JOIN player_id_map pim ON pim.puuid = p.puuid
            JOIN matches m ON m.match_id = p.match_id
            WHERE substr(p.match_id, 1, 4) = 'LA2_'""", conn)
        padron = pd.read_sql_query("""
            SELECT pim.player_id, MAX(pl.tier) AS player_tier
            FROM players pl JOIN player_id_map pim ON pim.puuid = pl.puuid
            GROUP BY pim.player_id""", conn)
        # Todas las partidas de la ventana (fuente completa). `crawl_window_listings`
        # (quién listó cada partida) se agregó después de listar a parte del padrón:
        # solo cubre a ~4.600 jugadores, así que se usa para el rango y no para contar.
        win = pd.read_sql_query("""
            SELECT w.match_id,
                   EXISTS (SELECT 1 FROM matches m WHERE m.match_id = w.match_id) AS bajada,
                   EXISTS (SELECT 1 FROM discarded_matches d WHERE d.match_id = w.match_id) AS descartada
            FROM crawl_window_matches w
            WHERE w.window_key = ? AND substr(w.match_id, 1, 4) = 'LA2_'""", conn, params=(WINDOW,))
        lst = pd.read_sql_query("""
            SELECT l.match_id, p.tier
            FROM crawl_window_listings l JOIN players p ON p.puuid = l.players_puuid
            WHERE l.window_key = ? AND substr(l.match_id, 1, 4) = 'LA2_'""", conn, params=(WINDOW,))
        fin_listado = conn.execute("SELECT MAX(listed_at) FROM crawl_window_players WHERE window_key = ?",
                                   (WINDOW,)).fetchone()[0]
    finally:
        conn.close()

    # 1. Partidas reales con su rango.
    tiers = match_tiers(parts, padron)
    real = (parts.drop_duplicates("match_id")[["match_id", "game_creation"]]
            .assign(tier=lambda d: d["match_id"].map(tiers)))

    # 2. Rango "del listado" (la misma regla de match_tiers sobre los Master+ listados,
    #    que no ve a los Diamond) y su corrección con lo ya bajado.
    por_match = win.set_index("match_id")
    lst_ids = lst.assign(player_id=lst["match_id"] + "_" + lst.index.astype(str))
    proxy = match_tiers(lst_ids[["match_id", "player_id"]],
                        lst_ids[["player_id", "tier"]].rename(columns={"tier": "player_tier"}))
    por_match["proxy"] = proxy.replace({SIN_RANGO: "DIAMOND"})  # < 5 Master+ listados
    # Sin detalle de listado: toman el reparto de las ya bajadas que tampoco lo tienen.
    por_match["proxy"] = por_match["proxy"].fillna("SIN_LISTADO")
    hechas = por_match[por_match["bajada"] == 1].join(real.set_index("match_id")["tier"])
    conf = pd.crosstab(hechas["proxy"], hechas["tier"], normalize="index")
    procesadas = por_match[(por_match["bajada"] == 1) | (por_match["descartada"] == 1)]
    tasa_desc = float(procesadas["descartada"].mean())

    pend = por_match[(por_match["bajada"] == 0) & (por_match["descartada"] == 0)].copy()
    pend = pend[rng.random(len(pend)) >= tasa_desc]  # los que se van a descartar
    pend["tier"] = [rng.choice(conf.columns, p=conf.loc[p].to_numpy()) if p in conf.index else p
                    for p in pend["proxy"]]

    # 3. Fecha por interpolación del número de match_id.
    ref = real[real["game_creation"] >= real["game_creation"].max() - 120 * DAY_MS]
    ref = ref.assign(n=_num(ref["match_id"])).sort_values("n")
    pend["game_creation"] = np.interp(_num(pend.index.to_series()), ref["n"], ref["game_creation"]).astype(np.int64)
    pend = pend.reset_index()[["match_id", "game_creation", "tier"]]

    # 4. Paso 3: lo jugado entre el fin del listado y el corte.
    # listed_at es CURRENT_TIMESTAMP de SQLite: UTC.
    fin_ms = int(datetime.strptime(fin_listado, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp() * 1000)
    base = pd.concat([real, pend], ignore_index=True)
    semana = base[(base["game_creation"] >= fin_ms - 7 * DAY_MS) & (base["game_creation"] < fin_ms)]
    dias = (corte_ms - fin_ms) / DAY_MS
    n3 = int(round(len(semana) * dias / 7)) if dias > 0 else 0
    p3 = semana.sample(n=n3, replace=n3 > len(semana), random_state=42).copy()
    p3["game_creation"] = fin_ms + rng.integers(0, max(1, corte_ms - fin_ms), size=n3)
    p3["match_id"] = [f"P3_{i}" for i in range(n3)]
    proy = pd.concat([base, p3], ignore_index=True)

    print(f"Reales: {len(real):,} | pendientes de la ventana (tras {tasa_desc:.1%} de descartes): "
          f"{len(pend):,} | paso 3 estimado ({dias:.1f} días): {n3:,}")
    print(f"TOTAL PROYECTADO (LAS): {len(proy):,}\n")
    print("Corrección del rango del listado, P(real | listado):")
    print((conf * 100).round(1).to_string(), "\n")

    if args.fusionar_apex:
        proy["tier"] = proy["tier"].replace({"CHALLENGER": "GRANDMASTER"})
        print("[Variante] GM + Challenger fusionados en un estrato (GRANDMASTER)\n")
    labels, res = proportional_temporal_split(proy, args.test_frac, test_tiers=TEST_TIERS,
                                              sobrante_al_test=SOBRANTE_AL_TEST,
                                              base_desde=None if args.base_completa else BASE_CUOTA_DESDE)
    proy = proy.set_index("match_id").assign(split=labels)
    proy["fecha"] = pd.to_datetime(proy["game_creation"], unit="ms", utc=True).dt.tz_convert(ART)
    print(f"Corte T = {res['corte']} | train {res['train']:,} | test {res['test']:,} | fuera {res['fuera']:,}\n")
    tab = proy.groupby(["tier", "split"]).size().unstack(fill_value=0)
    print(tab.reindex(TIER_ORDER + ["SIN_RANGO"]).dropna(how="all").to_string(), "\n")
    print("Fechas del test por rango:")
    for t in TEST_TIERS:
        f = proy.loc[(proy["split"] == "test") & (proy["tier"] == t), "fecha"]
        if len(f):
            q = f.quantile([0, 0.5, 1])
            print(f"  {t:12s} {len(f):6,} partidas | desde {q.iloc[0]:%d/%m %H:%M} | mediana {q.iloc[1]:%d/%m}"
                  f" | hasta {q.iloc[2]:%d/%m %H:%M}")
    print(f"\nReparto del test:  {res['reparto_test']}")
    print(f"Reparto del train (en los rangos del test): {res['reparto_train_rangos_test']}")
    # Variante: test = 20 % de TODO lo Master+ (un corte único); lo posterior al corte
    # de otros rangos se cuenta aparte (candidato a validación).
    eleg = proy["tier"].isin(TEST_TIERS)
    dias = sorted({int(x) for x in (proy["game_creation"] // DAY_MS) * DAY_MS}, reverse=True)
    for d in dias:
        if (eleg & (proy["game_creation"] >= d)).sum() >= args.test_frac * eleg.sum():
            post = proy["game_creation"] >= d
            print(f"\n[Test = {args.test_frac:.0%} de todo Master+] corte {pd.Timestamp(d, unit='ms', tz=ART):%d/%m}: "
                  f"test {(eleg & post).sum():,}, train {(~post).sum():,}, "
                  f"Diamond/sin rango posteriores {(~eleg & post).sum():,}")
            for t in TEST_TIERS:
                tt = proy["tier"] == t
                print(f"    {t:12s} test {(tt & post).sum():6,} = {(tt & post).sum() / max(1, tt.sum()):.1%} del rango")
            break
    _, vig = master_plus_temporal_split(proy.reset_index()[["match_id", "game_creation", "tier"]],
                                        args.test_frac)
    print(f"\n[REGLA VIGENTE: master_plus_temporal_split] {vig}")
    print("\nPartidas por semana (proyectadas):")
    sem = proy["fecha"].dt.tz_localize(None).dt.to_period("W-SUN").dt.start_time.dt.strftime("%d/%m")
    print(proy[proy["game_creation"] >= corte_ms - 70 * DAY_MS].groupby(sem).size().to_string())


if __name__ == "__main__":
    main()
