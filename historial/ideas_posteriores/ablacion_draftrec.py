"""
¿Cuánto pierde el híbrido + mapa si se queda solo con lo que existe en el dataset de DraftRec? (pedido del usuario,
2026-11-16; PROJECT_SUMMARY.md §9.S.56)

DraftRec tiene, por partida del historial: campeón, rol, equipo, bans, victoria y estadísticas de final de partida.
No tiene LP ni rango, línea de tiempo (@10/@15) ni fecha por partida del historial. Se sacan esas capas de a una:

1. referencia: híbrido + mapa (§9.S.51);
2. sin el mapa de LP;
3. sin la línea de tiempo: variables `*_10`, `*_15`, `d_*` (diferencias contra el rival de línea);
4. sin las fechas: `log_horas_desde`, `ritmo`, `conc_rec`, `conc_ant` → "compatible con DraftRec".

Quedan: estadísticas de final de partida (acum / últimas 20), winrate, experiencia, pool, rol y los datos fijos de
cada campeón (Data Dragon, disponibles para cualquier dataset). Evaluación: bloques A2-A4 y V Master+.

    python historial/ideas_posteriores/ablacion_draftrec.py      # ~20 min
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.brainstorm_base import ajustar_predecir, cargar, comparar, evaluar, mostrar  # noqa: E402

LINEA_TIEMPO = lambda c: c.endswith("_10") or c.endswith("_15") or "_d_" in c  # noqa: E731  (d_pool no es línea de tiempo)
FECHAS = {"log_horas_desde", "ritmo", "conc_rec", "conc_ant"}


def main():
    D = cargar()
    cols = D["cols"]
    nf = len(cols)
    X3 = D["Xd"].reshape(D["NM"], -1, nf)                       # (partidas, líneas, variables)

    def con_columnas(keep):
        D2 = dict(D)
        D2["Xd"] = np.ascontiguousarray(X3[:, :, keep].reshape(D["NM"], -1))
        return D2

    todas = list(range(nf))
    sin_tl = [i for i, c in enumerate(cols) if not LINEA_TIEMPO(c)]
    compat = [i for i in sin_tl if cols[i] not in FECHAS]
    print("sacadas por línea de tiempo:", [c for c in cols if LINEA_TIEMPO(c)])
    print("sacadas por fechas:", sorted(FECHAS & set(cols)))
    print(f"quedan {len(compat)} de {nf} variables por jugador")
    variantes = [
        ("Híbrido + mapa (referencia)", D, True),
        ("sin mapa de LP", D, False),
        ("sin mapa ni línea de tiempo", con_columnas(sin_tl), False),
        ("compatible con DraftRec (sin mapa, línea de tiempo ni fechas)", con_columnas(compat), False),
    ]
    P = {}
    for nombre, Dx, mapa in variantes:
        P[nombre] = evaluar(Dx, lambda tr, te, Dx=Dx, mapa=mapa: ajustar_predecir(Dx, tr, te, con_mapa=mapa))
        print("listo:", nombre, flush=True)
    T = comparar(D, P, "Híbrido + mapa (referencia)")
    mostrar(T)
    T.to_csv(REPO / "data" / "analysis" / "ablacion_draftrec.csv", index=False)


if __name__ == "__main__":
    main()
