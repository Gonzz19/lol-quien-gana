"""Modelo del LP partida a partida con el LP medido en vivo (PROJECT_SUMMARY.md §9.S.54).

Lee `data/lp_vivo/lp_vivo.db` en solo lectura, arma las variables de
`src/features/lp_partida.py` y evalúa cada modelo y nivel con folds por jugador: error por
partida y error del recorrido (LP encadenado desde la última medición real).

    python scripts/analysis/lp_partida_modelo.py                       # LAS, todo
    python scripts/analysis/lp_partida_modelo.py --plataformas la2,br1 --niveles partida --modelos regla,gb
    python scripts/analysis/lp_partida_modelo.py --guardar gb:partida  # además, entrena con todo y guarda

Escribe en `data/analysis/`: `lp_partida_eval.csv` (por partida), `lp_partida_recorrido.csv`
(por tramo de pasos) y `lp_partida_oof.csv` (predicciones fuera de fold por fila). Con
`--guardar`, el modelo entrenado con todas las filas va a `data/models/lp_partida_<modelo>_<nivel>.joblib`.
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.features import lp_partida as L  # noqa: E402
from src.lp_vivo import cargar_enlazadas  # noqa: E402

DB = ROOT / "data" / "lp_vivo" / "lp_vivo.db"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", type=Path, default=DB)
    ap.add_argument("--plataformas", default="la2")
    ap.add_argument("--niveles", default=",".join(L.NIVELES))
    ap.add_argument("--modelos", default=",".join(L.MODELOS))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--guardar", default=None, help="modelo:nivel a entrenar con todo y guardar")
    args = ap.parse_args()

    plataformas = args.plataformas.split(",")
    d = L.preparar(cargar_enlazadas(args.db, plataformas))
    largo = d.groupby("cadena").size()
    print(f"{','.join(plataformas)}: {len(d):,} filas, {d['match_id'].nunique():,} partidas, "
          f"{d['puuid'].nunique():,} jugadores | cadenas de 2+ partidas: {(largo >= 2).sum():,} "
          f"({largo[largo >= 2].sum():,} filas, la más larga {largo.max()})")
    print(f"Desde {pd.to_datetime(d['game_end'].min(), unit='ms')} hasta {pd.to_datetime(d['game_end'].max(), unit='ms')}")

    t0 = time.time()
    oof = L.evaluar(d, args.niveles.split(","), args.modelos.split(","), args.folds)["oof"]
    tp, tr = L.tabla_partida(oof), L.tabla_recorrido(oof)
    print(f"\nPor partida (ΔLP, {args.folds} folds por jugador):")
    print(tp.round(3).to_string())
    print("\nRecorrido: MAE del LP encadenado según partidas desde el último LP real:")
    print(tr.round(2).to_string())
    print(f"\n({time.time() - t0:.0f} s)")

    out = ROOT / "data" / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    tp.to_csv(out / "lp_partida_eval.csv")
    tr.to_csv(out / "lp_partida_recorrido.csv")
    cols = ["match_id", "puuid", "platform", "game_end", "win", "z_antes", "z_despues", "delta_lp", "cadena", "paso"]
    oof[cols + [c for c in oof.columns if c.startswith(("pred_", "zhat_"))]].to_csv(out / "lp_partida_oof.csv", index=False)
    print(f"[+] {out / 'lp_partida_eval.csv'}, lp_partida_recorrido.csv, lp_partida_oof.csv")

    if args.guardar:
        import joblib
        nombre, nivel = args.guardar.split(":")
        m = L.MODELOS[nombre](L.NIVELES[nivel]).fit(d, d["delta_lp"])
        destino = ROOT / "data" / "models" / f"lp_partida_{nombre}_{nivel}.joblib"
        destino.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"modelo": m, "nivel": nivel, "cols": L.NIVELES[nivel], "plataformas": plataformas}, destino)
        print(f"[+] {destino}")


if __name__ == "__main__":
    main()
