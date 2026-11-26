"""¿Cómo baja el LP sin jugar? Jugadores con pocas partidas: LP al dejar de jugar ≈ z1 + lo que movieron sus partidas."""
import pickle, numpy as np, pandas as pd
from comun import cargar
c = cargar(); e = c["e"]; S = pickle.load(open("seq.pkl", "rb")); est = S["est"]
dev = ~c["caja"]
fin = est["dias_fin"] * 30
# LP al dejar de jugar: z1 + (lo que la regla mueve hacia adelante) = z1 + (z2_sin_decay - regla) ~ z1 + (z2 - regla) si no hubo decay en el medio
mov = e["z2"] - c["regla"]          # lo que movieron las partidas observadas (con la regla)
zl = e["z1"] + mov                   # LP al terminar la última partida (aprox.)
m = dev & (e["obs"] <= 10).to_numpy() & (e["ocultas"] <= 2).to_numpy() & (est["exc_ini"].to_numpy() == 0) & (est["exc_medio"].to_numpy() == 0)
d = pd.DataFrame({"zl": zl, "z2": e["z2"], "fin": fin, "z1": e["z1"]})[m]
d["dd"] = (np.floor(d["fin"]) - 14).clip(lower=0)
pd.set_option("display.width", 200)
for lo, hi in [(400, 500), (500, 700), (700, 1000), (275, 400), (100, 275), (0, 100)]:
    s = d[(d["zl"] >= lo) & (d["zl"] < hi)]
    print(f"\nLP al dejar {lo}-{hi}: {len(s)} jugadores")
    s = s.assign(b=pd.cut(s["dd"], [-1, 0, 2, 5, 10, 20, 50]))
    print(s.groupby("b", observed=True).agg(n=("z2", "size"), zl=("zl", "median"), z2_med=("z2", "median"),
          p375=("z2", lambda v: (v == 375).mean()), p275=("z2", lambda v: (v == 275).mean()),
          p0=("z2", lambda v: (v == 0).mean()), caida=("z2", lambda v: 0)).drop(columns="caida").round(2).to_string())

print("\n=== fino: días totales sin jugar al final (no excedente) ===")
d["fd"] = np.floor(d["fin"])
for lo, hi in [(400, 2000), (300, 400)]:
    s = d[(d["zl"] >= lo) & (d["zl"] < hi) & (d["fd"] >= 12)]
    s = s.assign(b=pd.cut(s["fd"], [11, 14, 15, 16, 18, 20, 24, 28, 29, 30, 32, 35, 40, 60]))
    print(f"\nLP al dejar {lo}-{hi}")
    print(s.groupby("b", observed=True).agg(n=("z2", "size"), zl=("zl", "median"), z2_med=("z2", "median"),
          p375=("z2", lambda v: (v == 375).mean()), p_325=("z2", lambda v: (v == 325).mean()),
          p275=("z2", lambda v: (v == 275).mean())).round(2).to_string())
