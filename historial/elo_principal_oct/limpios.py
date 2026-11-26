"""Error en los casos limpios: sin no observadas y sin huecos largos (solo queda la variación de LP por partida)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import cargar, evaluar, linea  # noqa
ROOT = Path(r"c:/Users/DarkA/OneDrive/Escritorio/Facultad/3er año/ML/TPF/ProyectoFinalML")
c = cargar(); e, partes = c["e"], c["partes"]
ps = partes.sort_values("game_creation")
g = ps.groupby("player_id")["game_creation"]
hueco = (g.diff() / 86_400_000).groupby(ps["player_id"]).max().reindex(e.index).fillna(0)
fin = ((c["t2"] - g.max()) / 86_400_000).reindex(e.index)
ini = ((g.min() - c["t1"]) / 86_400_000).reindex(e.index)
huecos = pd.concat([hueco, fin, ini], axis=1).max(axis=1)
oof = pd.read_csv(ROOT / "data" / "analysis" / "boost_completo_oof.csv", index_col="player_id").reindex(e.index)
gb = c["regla"] + oof["gb"]
for lim in (5, 10, 14):
    m = (e["ocultas"] == 0) & (huecos <= lim)
    print(f"limpios (0 ocultas, huecos <= {lim} d): n={m.sum()}")
    print("  " + linea("regla", evaluar(e, c["regla"], e.index[m])))
    print("  " + linea("GB OOF", evaluar(e, gb, e.index[m])))
    for t in ("MASTER", "DIAMOND"):
        mm = m & (e["tier2"] == t)
        r = (e["z1"] - c["regla"])[mm]
        print(f"    {t}: n={mm.sum()}, regla MAE {r.abs().mean():.1f}, sesgo {r.mean():.1f}; "
              f"MAE por partida {(r.abs() / e.loc[mm, 'obs'].clip(lower=1)).median():.2f}")
m = (e["ocultas"] == 0) & (huecos <= 10)
r = (e["z1"] - c["regla"])[m]
print("\nerror de la regla vs partidas (limpios):")
print(r.abs().groupby(pd.cut(e.loc[m, "obs"], [0, 5, 10, 25, 50, 100, 1000])).agg(["size", "mean", "median"]).round(1).to_string())
