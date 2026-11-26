#!/usr/bin/env python3
"""
scripts/analysis/build_rating.py
================================
Rating de habilidad por jugador y partida (`src/features/rating.py`,
PROJECT_SUMMARY.md §9.S.21).

1. Ajusta los parámetros con una grilla: log-loss de la predicción previa a cada
   partida sobre la ventana de validación del **train** (`VAL_DESDE` al corte). El
   test no se evalúa.
2. Valida contra la foto del padrón: el rating de cada jugador justo antes de la
   foto, calculado **sin** usar la foto, contra su LP real (Spearman).
3. Corre el rating final sobre todas las partidas de LAS y escribe:
   * `data/rating_partida.csv`: rating previo de cada jugador en cada partida.
   * `data/analysis/rating_params.json`: grilla, parámetros elegidos y validaciones.

Uso:
    python scripts/analysis/build_rating.py              # grilla + salida (~15 min)
    python scripts/analysis/build_rating.py --sin-grilla # solo con los parámetros por defecto
"""

import argparse
import itertools
import json
import sqlite3
import sys
import time
from pathlib import Path
from statistics import NormalDist

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.config import DATA_DIR, DEFAULT_DB_PATH  # noqa: E402
from src.features.padron import T_FOTO1_TXT, verificar_foto1  # noqa: E402
from src.features.rating import (ParametrosRating, correr, lp_absoluto, metricas,  # noqa: E402
                                 ocultas_por_partida)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

TZ = "America/Argentina/Buenos_Aires"
# Supuesto: inicio del split cuyas W/L reporta la foto (para estimar la cobertura).
INICIO_SPLIT = "2026-01-08"
VAL_DESDE = "2026-08-15"
MIN_PREVIAS_VALIDACION = 20

GRILLA = {
    "beta": [800.0, 1600.0, 3200.0],
    "tau": [5.0, 20.0],
    "sigma0": [200.0, 400.0],
    "sigma_foto": [200.0, 800.0],
}


def ms(fecha: str, tz: str = TZ) -> int:
    return int(pd.Timestamp(fecha, tz=tz).value // 10**6)


def cargar_partes() -> pd.DataFrame:
    d = pd.read_csv(DATA_DIR / "full_dataset.csv",
                    usecols=["match_id", "player_id", "team_id", "win", "game_creation"])
    return d[d["match_id"].str.startswith("LA2_")].reset_index(drop=True)


def cargar_foto(db_path: Path) -> tuple:
    """Foto del padrón por `player_id`: LP absoluto `z`, `partidas` (W + L), tier. Y su fecha (ms)."""
    conn = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True, timeout=60.0)
    try:
        f = pd.read_sql_query("""
            SELECT m.player_id, p.tier, p.rank, p.league_points, p.wins + p.losses AS partidas
            FROM players p JOIN player_id_map m ON m.puuid = p.puuid
        """, conn)
        verificar_foto1(conn)
        t_foto = T_FOTO1_TXT
    finally:
        conn.close()
    f["z"] = [lp_absoluto(t, r, lp) for t, r, lp in zip(f["tier"], f["rank"], f["league_points"])]
    # Una persona con varias cuentas en el padrón: la cuenta con más partidas.
    f = f.dropna(subset=["z"]).sort_values("partidas").drop_duplicates("player_id", keep="last")
    return f.reset_index(drop=True), ms(t_foto, tz="UTC")


def delta_rojo(p_rojo: float, beta: float, n: int = 10) -> float:
    """Ventaja de lado que reproduce la tasa de victoria roja con habilidades iguales."""
    return NormalDist().inv_cdf(p_rojo) * (n ** 0.5) * beta


def validacion_foto(partes, foto, t_foto, params, ocultas) -> dict:
    """Spearman entre el rating justo antes de la foto (sin usarla) y el LP real."""
    antes = partes[partes["game_creation"] < t_foto]
    est = correr(antes, params, ocultas=ocultas)["estado"].rename(columns={"partidas": "observadas"})
    x = foto.merge(est, on="player_id")
    x = x[x["observadas"] >= MIN_PREVIAS_VALIDACION]
    out = {"jugadores": int(len(x)), "spearman": float(x["z"].corr(x["mu"], method="spearman"))}
    out["por_tier"] = {t: {"n": int(len(g)), "spearman": float(g["z"].corr(g["mu"], method="spearman"))}
                       for t, g in x.groupby("tier") if len(g) >= 30}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Rating TrueSkill por jugador y partida")
    ap.add_argument("--sin-grilla", action="store_true")
    ap.add_argument("--db", default=str(DEFAULT_DB_PATH))
    args = ap.parse_args()
    t0 = time.time()

    partes = cargar_partes()
    foto, t_foto = cargar_foto(Path(args.db))
    splits = pd.read_csv(DATA_DIR / "splits" / "match_split.csv", usecols=["match_id", "split"])
    corte = json.loads((DATA_DIR / "splits" / "split_manifest.json").read_text(encoding="utf-8"))["corte"]
    ocultas = ocultas_por_partida(partes, foto, ms(INICIO_SPLIT), t_foto, ParametrosRating().max_ocultas)
    print(f"[*] {partes['match_id'].nunique():,} partidas de LAS, {partes['player_id'].nunique():,} jugadores; "
          f"foto: {len(foto):,} jugadores ({pd.Timestamp(t_foto, unit='ms', tz='UTC'):%Y-%m-%d %H:%M} UTC); "
          f"cobertura estimada para {len(ocultas):,}.")

    train_ids = set(splits.loc[splits["split"] == "train", "match_id"])
    val = lambda p: p[p["match_id"].isin(train_ids) & (p["game_creation"] >= ms(VAL_DESDE))  # noqa: E731
                      & (p["game_creation"] < ms(corte))]
    gan = partes[partes["match_id"].isin(train_ids) & (partes["team_id"] == 100)].drop_duplicates("match_id")
    p_rojo = 1 - gan["win"].mean()

    resultados = []
    combos = [dict(zip(GRILLA, v)) for v in itertools.product(*GRILLA.values())] if not args.sin_grilla else [{}]
    for k, c in enumerate(combos, 1):
        base = ParametrosRating(**c)
        params = ParametrosRating(**{**base.to_dict(), "ventaja_roja": delta_rojo(p_rojo, base.beta)})
        r = correr(partes, params, foto=foto, t_foto_ms=t_foto, ocultas=ocultas)
        m = metricas(val(r["partidas"]))
        resultados.append({"params": params.to_dict(), "validacion": m})
        print(f"[{k}/{len(combos)}] {c} -> log-loss {m['log_loss']:.5f} "
              f"(constante {m['log_loss_constante']:.5f}), acc {m['accuracy']:.4f}  [{time.time() - t0:.0f} s]",
              flush=True)

    mejor = min(resultados, key=lambda x: x["validacion"]["log_loss"])
    params = ParametrosRating(**mejor["params"])
    print(f"[+] Elegidos: {params.to_dict()}")

    val_foto = validacion_foto(partes, foto, t_foto, params, ocultas)
    print(f"[+] Validación contra la foto (sin usarla): Spearman {val_foto['spearman']:.3f} "
          f"en {val_foto['jugadores']:,} jugadores con ≥ {MIN_PREVIAS_VALIDACION} partidas previas.")

    final = correr(partes, params, foto=foto, t_foto_ms=t_foto, ocultas=ocultas)
    filas = final["filas"].drop(columns=["team_id", "game_creation"])
    filas["en_padron"] = filas["player_id"].isin(foto["player_id"])
    filas.round({"mu_pre": 1, "sigma_pre": 1}).to_csv(DATA_DIR / "rating_partida.csv", index=False)

    out = DATA_DIR / "analysis" / "rating_params.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "meta": {"inicio_split_supuesto": INICIO_SPLIT, "val_desde": VAL_DESDE, "corte": corte,
                 "t_foto_ms": t_foto, "p_rojo_train": p_rojo, "segundos": round(time.time() - t0, 1)},
        "elegidos": params.to_dict(), "validacion": mejor["validacion"], "validacion_foto": val_foto,
        "grilla": resultados}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[+] {DATA_DIR / 'rating_partida.csv'} y {out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
