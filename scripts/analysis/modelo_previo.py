"""Modelo previo de quién gana (escalón 9) de punta a punta, en un solo comando.

Calcula (o carga de `data/cache/ganador/`, si siguen vigentes) las variables previas, ajusta la
logística con las partidas Master+ anteriores al 16/09 y reporta AUC, acierto, log loss, calibración
y cobertura en V Master+. No toca el test. La primera corrida tarda ~22 min (8 de ellos el HGB por línea); después, ~45 s.

    python scripts/analysis/modelo_previo.py                # usa la caché si está vigente
    python scripts/analysis/modelo_previo.py --rehacer      # recalcula todas las variables
    python scripts/analysis/modelo_previo.py --comparar     # compara con data/analysis/brainstorm/pred_final.npz

Escribe `data/cache/ganador/pred_v.npz` (p en V e índices). Detalle en PROJECT_SUMMARY.md §9.S.73.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.ganador import modelo_previo, rutas, variables  # noqa: E402

REFERENCIA = REPO / "data" / "analysis" / "brainstorm" / "pred_final.npz"


def main():
    sys.stdout.reconfigure(encoding="utf-8")   # en Windows, con la salida redirigida, cp1252 no tiene "→"
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rehacer", action="store_true", help="recalcular todas las variables aunque la caché esté vigente")
    ap.add_argument("--comparar", action="store_true", help="comparar las predicciones con la referencia de brainstorm_final.py")
    a = ap.parse_args()

    D, V = variables.calcular_todas(rehacer=a.rehacer)
    X = variables.matriz_extras(D, V)
    _, p, m = modelo_previo.evaluar_v(D, X)
    print(f"Entrenamiento: {m['n_entrenamiento']:,} partidas de Master+ anteriores al 16/09; columnas: {m['n_columnas']:,}")
    print(f"V Master+ ({m['n']:,}): AUC {m['auc']:.4f} | acierto {m['acierto']:.1%} (siempre el lado más frecuente: "
          f"{m['linea_base']:.1%}) | log loss {m['log_loss']:.4f} | ECE {m['ece']:.4f}")
    print("Cobertura (acierto en la fracción más segura): " +
          "  ".join(f"{c:.0%} → {v:.1%}" for c, v in m["cobertura"].items()))
    vi = np.flatnonzero(D["val"])
    salida = rutas.CACHE / "pred_v.npz"
    salida.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(salida, p_v=p, idx_v=vi)
    print(f"[+] {salida}")

    if a.comparar:
        with np.load(REFERENCIA) as r:
            if not np.array_equal(r["idx_v"], vi):
                sys.exit("La referencia tiene otras partidas en V: no se puede comparar.")
            p_ref = r["p_v"]
        dif = np.abs(p_ref - p)
        print(f"Contra {REFERENCIA.name}: |Δp| máx {dif.max():.2e}, medio {dif.mean():.2e}; "
              f"misma predicción de ganador en {((p_ref > 0.5) == (p > 0.5)).mean():.2%}")


if __name__ == "__main__":
    main()
