"""Brainstorming (PROJECT_SUMMARY.md §9.S.71): ítems. Build general de cada campeón, builds raras (outliers) y cómo
varía el winrate, con suavizado; y si el historial de builds del jugador suma al modelo previo (escalón 9).

Datos: `item0..item5` de `full_dataset.csv` (inventario FINAL, sin orden de compra) y `data/ddragon/16.17.1/item.json`
(Data Dragon: precio, etiquetas, de qué sale y en qué se convierte cada ítem).

Ítem completo = sin mejora posterior (`into` vacío), ≥ 1.600 de oro, ni consumible ni trinket ni botas. Las botas van aparte.

Cuidado con la lectura: el inventario final depende del resultado (el que gana tiene más oro y más ítems, y hay ítems
que se compran yendo adelante o atrás). Por eso el winrate se mide como RESIDUO: victoria − winrate esperado para esa
línea, esa cantidad de ítems completos y esa duración. Sigue siendo descriptivo, no causal.

  1. Build general: frecuencia de cada ítem completo por (campeón, línea).
  2. Rareza de una build = media de −log2 de la frecuencia suavizada de sus ítems en ese (campeón, línea), contada solo
     con partidas ANTERIORES. Winrate residual por quintil de rareza y en el 5 % más raro.
  3. Efecto de cada ítem dentro de su (campeón, línea): suma de residuos / (n + K), K = 100 (encogido hacia 0).
     Lista de los pares con efecto más alto y más bajo, con su error estándar.
  4. Modelo previo: por jugador (y por jugador-campeón), media de sus partidas anteriores de la rareza de la build y
     del efecto encogido de sus ítems (los dos calculados con partidas anteriores a cada una; el residuo va centrado
     en la media previa de su campeón-línea). Azul − rojo por línea
     sobre el escalón 9, en V Master+ y sub-bloques.

    python historial/ideas_posteriores/brainstorm_48_items.py     # ~10 min; tablas en data/analysis/brainstorm/items_*.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.brainstorm_base import ajustar_predecir, cargar, comparar, evaluar, extras_referencia, mostrar  # noqa: E402

BR = REPO / "data" / "analysis" / "brainstorm"
K_ITEM, K_JUG = 100.0, 5.0
LANES = ["BOTTOM", "JUNGLE", "MIDDLE", "TOP", "UTILITY"]

D = cargar()
NM, ms = D["NM"], D["master"]
meta = json.load(open(REPO / "data" / "ddragon" / "16.17.1" / "item.json", encoding="utf-8"))["data"]
nombre = {int(k): v["name"] for k, v in meta.items()}
completo = {int(k) for k, v in meta.items() if not v.get("into") and v["gold"]["total"] >= 1600
            and not {"Consumable", "Trinket", "Boots"} & set(v.get("tags", []))}
botas = {int(k) for k, v in meta.items() if "Boots" in v.get("tags", []) and v["gold"]["total"] >= 900}

IT = [f"item{i}" for i in range(6)]
df = pd.read_csv(REPO / "data" / "splits" / "train" / "full_dataset.csv",
                 usecols=["match_id", "game_creation", "game_duration", "player_id", "team_id", "lane", "win",
                          "champion_id", "champion_name"] + IT)
df = df[df.lane.isin(LANES)]
df = df[df.groupby(["match_id", "team_id"]).lane.transform("nunique") == 5]
df = df[df.groupby("match_id").lane.transform("size") == 10]
df = df[df.groupby("match_id").win.transform("sum") == 5]
df = df.sort_values(["game_creation", "match_id", "team_id", "lane"]).reset_index(drop=True)
assert (df.match_id.to_numpy()[::10] == D["Z"]["mids"]).all()
N = len(df)
previo_v = np.repeat(~D["val0"], 10)          # anteriores al 16/09: lo descriptivo no mira la ventana de V
items = df[IT].to_numpy(int)
es_comp = np.isin(items, list(completo))
df["n_comp"] = es_comp.sum(1)
df["dur5"] = np.clip(df.game_duration // 300, 3, 8)

# ---- winrate esperado por línea × cantidad de ítems completos × duración (ajustado antes del 16/09)
esp = df[previo_v].groupby(["lane", "n_comp", "dur5"]).win.agg(["sum", "count"])
esp["e"] = (esp["sum"] + 10) / (esp["count"] + 20)
df["res"] = df.win - df.merge(esp[["e"]], left_on=["lane", "n_comp", "dur5"], right_index=True, how="left")["e"].fillna(0.5).to_numpy()
print(f"partidas {NM:,}; ítems completos por jugador: media {df.n_comp.mean():.2f}; winrate según cantidad: " +
      "  ".join(f"{k}→{v:.1%}" for k, v in df[previo_v].groupby("n_comp").win.mean().items()))

# ---- formato largo: una fila por (jugador-partida, ítem completo), en orden temporal
fi, ci = np.nonzero(es_comp)
lg = pd.DataFrame({"fila": fi, "item": items[fi, ci]}).drop_duplicates()
lg["cl"] = (df.champion_id.to_numpy()[lg.fila] * 10 + df.lane.map({l: i for i, l in enumerate(LANES)}).to_numpy()[lg.fila])
cl_fila = df.champion_id.to_numpy() * 10 + df.lane.map({l: i for i, l in enumerate(LANES)}).to_numpy()
n_cl_prev = pd.Series(cl_fila).groupby(cl_fila).cumcount().to_numpy(float)      # partidas previas de ese campeón-línea
# residuo centrado en el campeón-línea (media de sus partidas ANTERIORES, encogida): así el efecto de un ítem se mide
# contra el promedio de su propio campeón y no arrastra que el campeón gane poco o mucho
s_cl_prev = df.res.groupby(cl_fila).cumsum().to_numpy() - df.res.to_numpy()
df["res_cl"] = df.res.to_numpy() - s_cl_prev / (n_cl_prev + 20.0)
lg["res"] = df.res_cl.to_numpy()[lg.fila]
g = lg.groupby(["cl", "item"], sort=False)
lg["n_prev"] = g.cumcount().to_numpy(float)                                       # veces previas de ese ítem ahí
lg["s_prev"] = g.res.cumsum().to_numpy() - lg.res.to_numpy()
# OJO: dentro de una partida los dos equipos comparten instante; el desfase máximo es de 9 filas (despreciable).
n_items_cl = lg.groupby("cl").item.transform("nunique").to_numpy(float)
lg["rareza"] = -np.log2((lg.n_prev + 1.0) / (n_cl_prev[lg.fila] + n_items_cl))
lg["efecto"] = lg.s_prev / (lg.n_prev + K_ITEM)
por_fila = lg.groupby("fila").agg(rareza=("rareza", "mean"), efecto=("efecto", "mean"))
df["rareza"] = por_fila.rareza.reindex(range(N)).to_numpy()
df["efecto"] = por_fila.efecto.reindex(range(N)).to_numpy()
madura = (n_cl_prev >= 300) & df.rareza.notna().to_numpy() & (df.n_comp.to_numpy() >= 2)

# ---- 1. build general
pre = lg[previo_v[lg.fila]]
tot_cl = pd.Series(cl_fila[previo_v]).value_counts()
gen = pre.groupby(["cl", "item"]).agg(n=("res", "size"), res=("res", "sum")).reset_index()
gen["n_cl"] = gen.cl.map(tot_cl)
gen["frecuencia"] = gen.n / gen.n_cl
gen["efecto"] = gen.res / (gen.n + K_ITEM)
gen["ee"] = 0.5 / np.sqrt(gen.n + K_ITEM)
camp = df.drop_duplicates("champion_id").set_index("champion_id").champion_name
gen["campeon"], gen["linea"], gen["item_nombre"] = (gen.cl // 10).map(camp), (gen.cl % 10).map(dict(enumerate(LANES))), gen.item.map(nombre)
gen = gen[gen.n_cl >= 300].sort_values(["n_cl", "cl", "frecuencia"], ascending=[False, True, False])
gen[["campeon", "linea", "n_cl", "item_nombre", "n", "frecuencia", "efecto", "ee"]].to_csv(BR / "items_build_general.csv", index=False)
top3 = gen.groupby("cl").head(3).groupby("cl").frecuencia.mean()
print(f"\n1. build general ({gen.cl.nunique()} campeón-línea con ≥ 300 partidas): los 3 ítems más comunes aparecen en promedio en el "
      f"{top3.mean():.0%} de las partidas; ítems completos distintos por campeón-línea: mediana {gen.groupby('cl').size().median():.0f}")
for cl in list(tot_cl.index[:4]):
    t = gen[gen.cl == cl].head(5)
    print(f"   {t.campeon.iloc[0]} {t.linea.iloc[0]} ({int(t.n_cl.iloc[0]):,}): " + ", ".join(f"{a} {b:.0%}" for a, b in zip(t.item_nombre, t.frecuencia)))

# ---- 2. rareza de la build y winrate residual
d2 = df[madura & previo_v].copy()
d2["z"] = d2.groupby(cl_fila[madura & previo_v]).rareza.transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
d2["q"] = pd.qcut(d2.z, [0, .2, .4, .6, .8, .95, 1.0], labels=["0-20", "20-40", "40-60", "60-80", "80-95", "95-100 (outliers)"])
t2 = d2.groupby("q", observed=True).agg(n=("win", "size"), winrate=("win", "mean"), residuo=("res", "mean"), ee=("res", lambda s: s.std() / np.sqrt(len(s))))
t2.to_csv(BR / "items_rareza_winrate.csv")
print("\n2. winrate según rareza de la build (percentil dentro del campeón-línea; residuo = victoria − esperado por ítems y duración)")
print(t2.assign(winrate=t2.winrate.map("{:.1%}".format), residuo=t2.residuo.map("{:+.2%}".format), ee=t2.ee.map("{:.2%}".format)).to_string())
exp = np.log1p(df.groupby(["player_id", "champion_id"]).cumcount())[madura & previo_v]
print(f"   correlación rareza ↔ log(partidas previas del jugador con el campeón): {np.corrcoef(d2.z, exp)[0, 1]:+.3f}")
for lo, hi, nm in ((0, 5, "< 5 partidas con el campeón"), (5, 30, "5-29"), (30, 1e9, "30+")):
    mk = (np.expm1(exp) >= lo) & (np.expm1(exp) < hi)
    o = d2[mk.to_numpy() & (d2.q == "95-100 (outliers)").to_numpy()]
    print(f"   outliers con {nm}: n {len(o):,}, residuo {o.res.mean():+.2%} ± {o.res.std() / np.sqrt(max(len(o), 1)):.2%}")

# ---- 3. efecto de cada ítem dentro de su campeón-línea
e3 = gen[gen.n >= 200].copy()
e3["z"] = e3.res / np.sqrt(e3.n) / 0.5
print(f"\n3. efecto encogido de cada ítem en su campeón-línea ({len(e3):,} pares con ≥ 200 usos; K = {K_ITEM:.0f}). "
      f"|z| > 3: {(e3.z.abs() > 3).sum()} (por azar se esperan {0.0027 * len(e3):.0f})")
cols3 = ["campeon", "linea", "item_nombre", "n", "frecuencia", "efecto", "z"]
for tit, t in (("más alto", e3.nlargest(10, "efecto")), ("más bajo", e3.nsmallest(10, "efecto"))):
    print(f"   {tit}:")
    print(t[cols3].assign(frecuencia=t.frecuencia.map("{:.0%}".format), efecto=t.efecto.map("{:+.1%}".format), z=t.z.map("{:+.1f}".format)).to_string(index=False))
por_item = pre.groupby("item").agg(n=("res", "size"), res=("res", "sum"))
por_item["efecto"] = por_item.res / (por_item.n + K_ITEM)
por_item["nombre"] = por_item.index.map(nombre)
por_item.sort_values("efecto").to_csv(BR / "items_efecto_global.csv")
pi = por_item[por_item.n >= 2000].sort_values("efecto")
print("   por ítem (todos los campeones), residuo más bajo: " + ", ".join(f"{a} {b:+.1%}" for a, b in zip(pi.nombre[:6], pi.efecto[:6])))
print("   por ítem, residuo más alto: " + ", ".join(f"{a} {b:+.1%}" for a, b in zip(pi.nombre[-6:][::-1], pi.efecto[-6:][::-1])))

# ---- 4. historial de builds del jugador en el modelo previo
mu_r = df.loc[previo_v, "rareza"].groupby(cl_fila[previo_v]).transform("mean").mean()
df["rz"] = df.rareza - pd.Series(df.rareza.to_numpy()).groupby(cl_fila).transform(lambda s: s.expanding().mean().shift()).fillna(mu_r).to_numpy()


def media_previa(col, claves):
    v = df[col].to_numpy(float)
    ok = ~np.isnan(v)
    tmp = pd.DataFrame({"v": np.where(ok, v, 0.0), "n": ok.astype(float)})
    ac = tmp.groupby([df[c] for c in claves], sort=False).cumsum().to_numpy()
    return (ac[:, 0] - tmp.v.to_numpy()) / (ac[:, 1] - tmp.n.to_numpy() + K_JUG)


F = np.stack([media_previa("rz", ["player_id"]), media_previa("efecto", ["player_id"]),
              media_previa("rz", ["player_id", "champion_id"]), media_previa("efecto", ["player_id", "champion_id"])],
             axis=1).astype(np.float32).reshape(NM, 2, 5, 4)
np.savez_compressed(BR / "items_jugador.npz", F=F, cols=np.array(["rareza_jug", "efecto_jug", "rareza_jug_camp", "efecto_jug_camp"]))
dif = F[:, 0] - F[:, 1]
X9 = extras_referencia(D, escalon=9)
m = lambda X: (lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=X, extra_post=False, con_mapa=False))  # noqa: E731
rng = np.random.default_rng(0)
VAR = {"escalón 9": m(X9),
       "+ builds del jugador por línea (20)": m(np.c_[X9, dif.reshape(NM, -1)]),
       "+ builds del jugador, media del equipo (4)": m(np.c_[X9, dif.mean(1)]),
       "placebo (filas permutadas, 20)": m(np.c_[X9, dif.reshape(NM, -1)[rng.permutation(NM)]])}
P = {n: evaluar(D, f) for n, f in VAR.items()}
print("\n4. historial de builds del jugador sobre el escalón 9")
mostrar(comparar(D, P, "escalón 9"))
