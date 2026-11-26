"""Genera notebooks/rating_jugadores.ipynb (rating por jugador y partida, PROJECT_SUMMARY.md §9.S.21).

Uso:
    python scripts/analysis/build_rating_nb.py notebooks/rating_jugadores.ipynb
    python -m jupyter execute --inplace notebooks/rating_jugadores.ipynb

Las secciones de evaluación (5 a 10) quedan afuera por defecto: se evalúa recién con
la recorrida de la foto del 04/09 a una segunda foto (decisión del 2026-10-27).
`--evaluacion` las incluye, para cuando esa foto exista.

Requiere haber corrido antes `scripts/analysis/build_rating.py`.
"""
import sys
from pathlib import Path

import nbformat as nbf

args = [a for a in sys.argv[1:] if not a.startswith("--")]
OUT = Path(args[0]) if args else Path("notebooks/rating_jugadores.ipynb")
EVALUACION = "--evaluacion" in sys.argv
cells, cells_eval = [], []
destino = cells


def md(s):
    destino.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    destino.append(nbf.v4.new_code_cell(s.strip()))


md(r"""
# Rating de los jugadores: el elo de cada uno en cada partida

### League of Legends, LAS. Sección aparte del EDA, con su propio análisis

Riot no da el rango de un jugador **al momento de cada partida**: las partidas no
traen el rango de los participantes y la API de ligas solo da el rango actual. Lo
único que tenemos es **una foto del padrón** (tabla `players`, 04/09/2026) con el
LP, las victorias y las derrotas de cada jugador Diamond+ ese día. Esta sección
estima la habilidad de cada jugador en cada partida con un **sistema de rating**
que aprende de los resultados (`src/features/rating.py`, `PROJECT_SUMMARY.md` §9.S.21):

* **TrueSkill de equipos.** La habilidad de cada jugador es una normal
  $N(\mu, \sigma^2)$; la del equipo es la suma. Después de cada partida, los
  ganadores suben y los perdedores bajan, más cuanto más sorprendente fue el
  resultado y cuanto más incierto era cada uno.
* **Escala en LP absolutos.** Diamond IV 0 LP = 0, cada división suma 100, Master 0
  LP = 400 (Master, Grandmaster y Challenger comparten una escala). $\mu$ puede ser
  negativa: un Emerald que cae en una partida Diamond.
* **Partidas que no vimos.** Antes de la foto tenemos solo una parte de las partidas
  de cada jugador. La incertidumbre crece entre dos partidas observadas según cuántas
  jugó en el medio sin que las viéramos (su cobertura frente a las W + L de la foto).
* **La foto como medición.** El 04/09, el LP de la foto corrige a cada jugador del
  padrón como un filtro de Kalman. Ninguna partida usa la foto antes de su fecha.
* **Sin fuga del target.** Cada fila tiene el rating **previo** a la partida: no
  depende de su resultado.

> **Estado: sin evaluar.** El rating se evalúa recién con la recorrida de la foto
> del 04/09 a una **segunda foto**: todos los jugadores del padrón arrancan con su LP
> real y se recorren las partidas posteriores, donde la cobertura es casi completa.
> La segunda foto da la pérdida. Hasta entonces esta sección describe los datos que
> sostienen el rating (secciones 1 a 4). Ver `PROJECT_SUMMARY.md` §9.S.21.

**Qué datos se miran.** Las descripciones de la foto y de la cobertura usan todo el
dataset, porque no involucran resultados. Lo que involucra resultados (sección 3)
usa solo partidas anteriores a la foto, que son del train: el test no se mira.

1. La foto del padrón.
2. Cobertura: cuántas de sus partidas tenemos.
3. ¿Vemos más victorias o más derrotas de las reales?
4. La foto envejece: filas sin tier en el tiempo.
5. Pendiente: evaluación con la segunda foto.
""")

code(r"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from IPython.display import display

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
for p in (ROOT, ROOT / "scripts" / "analysis"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.features.rating import ParametrosRating, correr, MASTER_BASE  # noqa: E402
from build_rating import (INICIO_SPLIT, MIN_PREVIAS_VALIDACION, VAL_DESDE, cargar_foto,  # noqa: E402
                          cargar_partes, ms)

DATA_DIR = ROOT / "data"
TZ = "America/Argentina/Buenos_Aires"
TIER_ORDER = ["CHALLENGER", "GRANDMASTER", "MASTER", "DIAMOND"]
TIER_COLORS = {"CHALLENGER": "#F0B232", "GRANDMASTER": "#C8282E", "MASTER": "#9B4FD1", "DIAMOND": "#86b6ef"}
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 12,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e1e0d9",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "legend.frameon": False, "figure.dpi": 110,
})

partes = cargar_partes()
foto, T_FOTO = cargar_foto(ROOT / "data" / "lol_las.db")
splits = pd.read_csv(DATA_DIR / "splits" / "match_split.csv", usecols=["match_id", "split", "tier"])
manifest = json.loads((DATA_DIR / "splits" / "split_manifest.json").read_text(encoding="utf-8"))
CORTE_MS = ms(manifest["corte"])
info = json.loads((DATA_DIR / "analysis" / "rating_params.json").read_text(encoding="utf-8"))
PARAMS = ParametrosRating(**info["elegidos"])
rating = pd.read_csv(DATA_DIR / "rating_partida.csv")

partes = partes.merge(splits[["match_id", "split"]], on="match_id", how="left")
rating = rating.merge(partes[["match_id", "player_id", "team_id", "win", "game_creation", "split"]],
                      on=["match_id", "player_id"])
train = rating[rating["split"] == "train"]
fecha = lambda t: pd.to_datetime(t, unit="ms", utc=True).dt.tz_convert(TZ)  # noqa: E731
print(f"Partidas de LAS: {partes['match_id'].nunique():,} | filas con rating: {len(rating):,} "
      f"| del train: {train['match_id'].nunique():,} partidas")
print(f"Foto del padrón: {len(foto):,} jugadores, {pd.Timestamp(T_FOTO, unit='ms', tz='UTC').tz_convert(TZ):%d/%m/%Y %H:%M} ART")
print("Parámetros elegidos:", {k: round(v, 1) for k, v in info["elegidos"].items()})
""")

md(r"""
## 1. La foto del padrón

LP absoluto de cada jugador del padrón el día de la foto. En Master+ el LP no tiene
techo: Grandmaster y Challenger no son un LP fijo sino los mejores puestos del
ladder (500 y 200 jugadores), así que sus cortes salen de la foto.
""")

code(r"""
fig, ax = plt.subplots(figsize=(13, 4.8))
bins = np.arange(0, foto["z"].max() + 50, 25)
for t in TIER_ORDER[::-1]:
    ax.hist(foto.loc[foto["tier"] == t, "z"], bins=bins, color=TIER_COLORS[t], label=t.capitalize(), alpha=0.95)
for t in ["GRANDMASTER", "CHALLENGER"]:
    corte = foto.loc[foto["tier"] == t, "z"].min()
    ax.axvline(corte, color=TIER_COLORS[t], linestyle="--", linewidth=1.2)
    ax.annotate(f"corte {t.capitalize()}: {corte:.0f}", xy=(corte, ax.get_ylim()[1] * 0.9), xytext=(4, 0),
                textcoords="offset points", fontsize=9, color=INK_2)
ax.axvline(MASTER_BASE, color=MUTED, linewidth=1)
ax.set_yscale("log")
ax.set_title("LP absoluto en la foto (Diamond IV 0 LP = 0, Master 0 LP = 400)")
ax.set_xlabel("LP absoluto")
ax.set_ylabel("Jugadores (escala log)")
ax.legend(loc="upper right")
plt.tight_layout()
plt.show()
display(foto.groupby("tier")["z"].describe()[["count", "min", "50%", "max"]].reindex(TIER_ORDER).round(0))
""")

md(r"""
## 2. Cobertura: cuántas de sus partidas tenemos

La foto dice cuántas partidas jugó cada uno en el split ($W + L$). Las comparamos con
las que tenemos entre el inicio del split y la foto. **Supuesto:** el split empezó el
`INICIO_SPLIT` (la fecha exacta no está confirmada). Una cobertura baja significa que
entre dos partidas observadas el jugador jugó muchas que no vimos, y el rating lo
refleja agrandando su incertidumbre.
""")

code(r"""
obs = (partes[(partes["game_creation"] >= ms(INICIO_SPLIT)) & (partes["game_creation"] < T_FOTO)]
       .groupby("player_id").size().rename("observadas"))
cob = foto.join(obs, on="player_id").fillna({"observadas": 0})
cob["cobertura"] = (cob["observadas"] / cob["partidas"].clip(lower=1)).clip(upper=1.5)
tabla = cob.groupby("tier").agg(jugadores=("cobertura", "size"),
                                 sin_partidas=("observadas", lambda s: 100 * (s == 0).mean()),
                                 p25=("cobertura", lambda s: 100 * s.quantile(0.25)),
                                 mediana=("cobertura", lambda s: 100 * s.median()),
                                 p75=("cobertura", lambda s: 100 * s.quantile(0.75))).reindex(TIER_ORDER)
print(f"Cobertura = partidas observadas desde el {INICIO_SPLIT} hasta la foto / (W + L) de la foto, en %:")
display(tabla.round(1))

fig, ax = plt.subplots(figsize=(10, 4.5))
datos = [cob.loc[(cob["tier"] == t) & (cob["observadas"] > 0), "cobertura"] * 100 for t in TIER_ORDER]
caja = ax.boxplot(datos, patch_artist=True, showfliers=False, widths=0.55, medianprops=dict(color=INK))
for patch, t in zip(caja["boxes"], TIER_ORDER):
    patch.set_facecolor(TIER_COLORS[t])
ax.set_xticks(range(1, 5), [t.capitalize() for t in TIER_ORDER])
ax.set_ylabel("% de sus partidas que tenemos")
ax.set_title("Cobertura por rango de la foto (jugadores con al menos una partida observada)")
plt.tight_layout()
plt.show()
""")

md(r"""
## 3. ¿Vemos más victorias o más derrotas de las reales?

Si nuestra recolección viera más victorias que derrotas (o al revés), el rating
aprendido de nuestras partidas estaría sesgado. Para cada jugador con al menos
`MIN_OBS` partidas observadas:

$$z = \frac{WR_{obs} - WR_{foto}}{\sqrt{WR_{foto}(1 - WR_{foto})/n}}$$

Sin sesgo, $z$ se reparte como una normal estándar. (Es aproximado: las partidas
observadas son parte de las que cuenta la foto, así que $z$ queda algo más angosta.)
""")

code(r"""
MIN_OBS = 20
pre = partes[(partes["game_creation"] >= ms(INICIO_SPLIT)) & (partes["game_creation"] < T_FOTO)]
wr_obs = pre.groupby("player_id")["win"].agg(["mean", "size"])
sql_wins = foto.set_index("player_id")
x = wr_obs.join(sql_wins[["tier", "partidas"]], how="inner")
import sqlite3
con = sqlite3.connect(f"file:{(DATA_DIR / 'lol_las.db').resolve()}?mode=ro", uri=True)
w = pd.read_sql_query("SELECT m.player_id, MAX(p.wins) AS wins FROM players p JOIN player_id_map m "
                      "ON m.puuid = p.puuid GROUP BY m.player_id", con).set_index("player_id")["wins"]
con.close()
x = x.join(w)
x = x[(x["size"] >= MIN_OBS) & (x["partidas"] > 0)]
x["wr_foto"] = x["wins"] / x["partidas"]
x["z"] = (x["mean"] - x["wr_foto"]) / np.sqrt(x["wr_foto"] * (1 - x["wr_foto"]) / x["size"])

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.hist(x["z"].clip(-5, 5), bins=np.arange(-5, 5.25, 0.25), density=True, color=CATEGORICAL[0], label="observado")
xs = np.linspace(-5, 5, 200)
ax.plot(xs, norm.pdf(xs), color=INK, linestyle="--", label="N(0, 1): sin sesgo")
ax.set_title(f"Winrate observado contra el de la foto ({len(x):,} jugadores con ≥ {MIN_OBS} partidas)")
ax.set_xlabel("z")
ax.legend()
plt.tight_layout()
plt.show()
display(x.groupby("tier")["z"].agg(["size", "mean", "std"]).reindex(TIER_ORDER).round(3))
""")

md(r"""
## 4. La foto envejece: filas sin tier en el tiempo

El tier de cada fila del dataset sale solo de la foto del 04/09. Un jugador que no
estaba en el padrón ese día (Emerald en partidas Diamond, quien subió después,
cuentas nuevas) queda sin tier, y cuanto más lejos de la foto, más pasa. Esta es la
motivación del rating: una foto sola no alcanza para el período del test. (Este
gráfico no usa resultados.)
""")

code(r"""
full_tier = pd.read_csv(DATA_DIR / "full_dataset.csv", usecols=["match_id", "player_tier", "game_creation"])
full_tier = full_tier[full_tier["match_id"].str.startswith("LA2_")]
full_tier["mes"] = fecha(full_tier["game_creation"]).dt.tz_localize(None).dt.to_period("M")
por_mes = full_tier.groupby("mes").agg(filas=("player_tier", "size"),
                                       sin_tier=("player_tier", lambda s: 100 * s.isna().mean()))
por_mes = por_mes[por_mes["filas"] >= 5000]
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.bar(por_mes.index.astype(str), por_mes["sin_tier"], color=CATEGORICAL[0])
for i, v in enumerate(por_mes["sin_tier"]):
    ax.text(i, v, f"{v:.1f} %", ha="center", va="bottom", fontsize=9, color=INK)
ax.set_title("Filas sin tier del padrón, por mes (meses con ≥ 5.000 filas)")
ax.set_ylabel("% de filas")
plt.tight_layout()
plt.show()
""")

destino = cells_eval

md(r"""
## 5. Ajuste de los parámetros

Se eligen con una grilla, minimizando la **log-loss** de la predicción previa a cada
partida en la ventana de validación del train (desde `VAL_DESDE` hasta el corte del
split). La ventaja del lado rojo $\delta$ no se ajusta: se fija para reproducir la
tasa de victoria roja del train con habilidades iguales,
$\delta = \Phi^{-1}(p_{rojo})\sqrt{10}\,\beta$.

* $\beta$: ruido de rendimiento de una partida (en LP, por jugador).
* $\tau$: cambio de habilidad por partida jugada.
* $\sigma_0$: incertidumbre de un jugador nuevo.
* $\sigma_{foto}$: error de la foto como medida de la habilidad.
""")

code(r"""
grilla = pd.DataFrame([{**g["params"], **{f"val_{k}": v for k, v in g["validacion"].items()}} for g in info["grilla"]])
cols = ["beta", "tau", "sigma0", "sigma_foto", "val_log_loss", "val_accuracy"]
display(grilla[cols].sort_values("val_log_loss").head(10).round(5))
v = info["validacion"]
print(f"Validación ({v['n']:,} partidas del train desde el {VAL_DESDE}): log-loss {v['log_loss']:.5f} contra "
      f"{v['log_loss_constante']:.5f} de predecir siempre la tasa azul; accuracy {100 * v['accuracy']:.2f} %.")

fig, ax = plt.subplots(figsize=(9, 4.5))
for k, ((tau, s0), g) in enumerate(grilla.groupby(["tau", "sigma0"])):
    g = g.groupby("beta")["val_log_loss"].min()
    ax.plot(g.index, g.to_numpy(), marker="o", color=CATEGORICAL[k], label=f"τ = {tau:.0f}, σ0 = {s0:.0f}")
ax.axhline(v["log_loss_constante"], color=MUTED, linestyle="--", label="constante")
ax.set_xscale("log")
ax.set_xlabel("β (LP)")
ax.set_ylabel("log-loss de validación")
ax.set_title("Grilla de parámetros (mejor σ_foto en cada punto)")
ax.legend(fontsize=8.5)
plt.tight_layout()
plt.show()
""")

md(r"""
## 6. Calibración de la predicción previa

Con el rating previo de los 10 jugadores, la probabilidad de que gane azul es
$\Phi\big((\sum_{azul}\mu - \sum_{rojo}\mu - \delta)/c\big)$. Si está calibrada, en
las partidas a las que el sistema les da 60 % gana azul el 60 % de las veces. Solo
partidas del train.
""")

code(r"""
def prob_partidas(df, params):
    g = df.assign(azul=df["team_id"] == 100, s2=df["sigma_pre"] ** 2)
    agg = g.groupby(["match_id", "azul"]).agg(mu=("mu_pre", "sum"), s2=("s2", "sum"), n=("mu_pre", "size"),
                                             win=("win", "max")).unstack("azul")
    c = np.sqrt(agg["s2"].sum(axis=1) + agg["n"].sum(axis=1) * params.beta ** 2)
    p = norm.cdf((agg[("mu", True)] - agg[("mu", False)] - params.ventaja_roja) / c)
    ok = agg[("win", True)] != agg[("win", False)]
    return pd.DataFrame({"p": p, "y": agg[("win", True)]})[ok]

pp = prob_partidas(train[train["game_creation"] >= ms(VAL_DESDE)], PARAMS)
pp["bin"] = pd.qcut(pp["p"], 10, duplicates="drop")
cal = pp.groupby("bin", observed=True).agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
fig, ax = plt.subplots(figsize=(6.5, 6))
ax.plot([0.3, 0.7], [0.3, 0.7], color=MUTED, linestyle="--", label="calibración perfecta")
z95 = 1.959963984540054
den = 1.0 + z95 * z95 / cal["n"]
ctr = (cal["y"] + z95 * z95 / (2.0 * cal["n"])) / den
sem = z95 * np.sqrt(np.maximum(cal["y"] * (1.0 - cal["y"]) / cal["n"] + z95 * z95 / (4.0 * cal["n"] ** 2), 0.0)) / den
lo_w, hi_w = np.clip(ctr - sem, 0.0, 1.0), np.clip(ctr + sem, 0.0, 1.0)
ax.plot(cal["p"], cal["y"], marker="o", color=CATEGORICAL[0], label="deciles de p (IC 95 % Wilson, partidas)")
ax.fill_between(cal["p"], lo_w, hi_w, color=CATEGORICAL[0], alpha=0.16, lw=0)
ax.set_xlabel("P(gana azul) prevista")
ax.set_ylabel("Tasa observada de victoria azul")
ax.set_title(f"Calibración en la validación ({len(pp):,} partidas)")
ax.legend()
plt.tight_layout()
plt.show()
print(f"Rango de las probabilidades: P5 = {pp['p'].quantile(0.05):.3f}, P95 = {pp['p'].quantile(0.95):.3f}")
""")

md(r"""
## 7. Validación contra la foto

La prueba independiente: se corre el rating **sin** la foto hasta justo antes de su
fecha, y se compara la habilidad estimada de cada jugador con su LP real. Si el
rating mide algo real, tienen que ordenar parecido (Spearman). Entran los jugadores
con al menos `MIN_PREVIAS_VALIDACION` partidas observadas.
""")

code(r"""
vf = info["validacion_foto"]
print(f"Spearman global: {vf['spearman']:.3f} ({vf['jugadores']:,} jugadores)")
display(pd.DataFrame(vf["por_tier"]).T.reindex([t for t in TIER_ORDER if t in vf["por_tier"]]).round(3))

antes = partes[partes["game_creation"] < T_FOTO]
est = correr(antes, PARAMS)["estado"].rename(columns={"partidas": "observadas"})
xv = foto.merge(est, on="player_id")
xv = xv[xv["observadas"] >= MIN_PREVIAS_VALIDACION]
fig, ax = plt.subplots(figsize=(8, 6))
for t in TIER_ORDER[::-1]:
    g = xv[xv["tier"] == t]
    ax.scatter(g["z"], g["mu"], s=6, alpha=0.4, color=TIER_COLORS[t], label=t.capitalize())
ax.set_xlabel("LP absoluto en la foto")
ax.set_ylabel("μ del rating justo antes de la foto (sin usarla)")
ax.set_title("Rating estimado contra LP real")
ax.legend(markerscale=3)
plt.tight_layout()
plt.show()
""")

md(r"""
## 8. Trayectorias de ejemplo

$\mu \pm 2\sigma$ previo a cada partida para algunos jugadores del padrón con muchas
partidas en el train, uno por rango. La línea vertical es la foto y el punto, su LP
ese día: ahí la incertidumbre se achica de golpe.
""")

code(r"""
fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), sharex=True)
cuenta = train.groupby("player_id").size()
for ax, t in zip(axes.flat, TIER_ORDER):
    cand = foto.loc[foto["tier"] == t, "player_id"]
    jug = cuenta.reindex(cand).dropna().sort_values().index[-1]
    g = train[train["player_id"] == jug].sort_values("game_creation")
    f = fecha(g["game_creation"]).dt.tz_localize(None)
    ax.fill_between(f, g["mu_pre"] - 2 * g["sigma_pre"], g["mu_pre"] + 2 * g["sigma_pre"],
                    color=TIER_COLORS[t], alpha=0.25, linewidth=0)
    ax.plot(f, g["mu_pre"], color=TIER_COLORS[t], linewidth=1.4)
    tf = pd.Timestamp(T_FOTO, unit="ms", tz="UTC").tz_convert(TZ).tz_localize(None)
    ax.axvline(tf, color=MUTED, linestyle="--", linewidth=1)
    ax.scatter([tf], [foto.loc[foto["player_id"] == jug, "z"].iloc[0]], color=INK, zorder=5, s=30)
    ax.set_title(f"{t.capitalize()}: {len(g)} partidas en el train")
    ax.set_ylabel("LP absoluto")
fig.autofmt_xdate()
plt.suptitle("μ ± 2σ previo a cada partida (train)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
""")

md(r"""
## 9. La incertidumbre en el tiempo

Mediana de $\sigma$ previo por semana, en el train, separando a los jugadores del
padrón del resto. Antes de la foto los del padrón tienen pocas partidas observadas y
mucha incertidumbre; la foto la baja de golpe.
""")

code(r"""
s = train.assign(semana=fecha(train["game_creation"]).dt.tz_localize(None).dt.to_period("W").dt.start_time)
s = s[s["semana"] >= s["semana"].quantile(0.02)]
fig, ax = plt.subplots(figsize=(12, 4.5))
for col, nombre, color in [(True, "en el padrón", CATEGORICAL[0]), (False, "fuera del padrón", CATEGORICAL[1])]:
    sub = s[s["en_padron"] == col]
    if not sub.empty:
        rows_w = []
        for sem, g_s in sub.groupby("semana"):
            a = g_s["sigma_pre"].dropna().to_numpy(float)
            h = min(0.499, 1.96 * 0.5 / np.sqrt(len(a)))
            lo_, med_, hi_ = np.quantile(a, [0.5 - h, 0.5, 0.5 + h])
            rows_w.append((sem, med_, lo_, hi_))
        df_w = pd.DataFrame(rows_w, columns=["semana", "med", "lo", "hi"]).sort_values("semana")
        ax.plot(df_w["semana"], df_w["med"], marker="o", markersize=3, color=color, label=f"{nombre} (IC 95 % mediana)")
        ax.fill_between(df_w["semana"], df_w["lo"], df_w["hi"], color=color, alpha=0.16)
ax.axvline(pd.Timestamp(T_FOTO, unit="ms", tz="UTC").tz_convert(TZ).tz_localize(None), color=MUTED,
           linestyle="--", label="foto")
ax.set_ylabel("σ previo (mediana e IC 95 % orden, LP)")
ax.set_title("Incertidumbre del rating por semana (train)")
ax.legend()
plt.tight_layout()
plt.show()
""")

md(r"""
## 10. El rating y el rango de la partida

El split le asigna a cada partida un rango (el más alto que juntan al menos 5
jugadores de ese rango o superior, según la foto). Si el rating es coherente, el
$\mu$ medio de los 10 jugadores tiene que ordenar los rangos. Solo train, desde la
ventana de validación (cuando el rating ya convergió).
""")

code(r"""
r10 = (train[train["game_creation"] >= ms(VAL_DESDE)].groupby("match_id")["mu_pre"].mean().rename("mu_medio")
       .to_frame().join(splits.set_index("match_id")["tier"]))
orden = [t for t in TIER_ORDER if t in set(r10["tier"])]
fig, ax = plt.subplots(figsize=(9, 4.5))
caja = ax.boxplot([r10.loc[r10["tier"] == t, "mu_medio"] for t in orden], patch_artist=True, showfliers=False,
                  widths=0.55, medianprops=dict(color=INK))
for patch, t in zip(caja["boxes"], orden):
    patch.set_facecolor(TIER_COLORS[t])
ax.set_xticks(range(1, len(orden) + 1), [t.capitalize() for t in orden])
ax.axhline(MASTER_BASE, color=MUTED, linestyle="--", linewidth=1)
ax.set_ylabel("μ medio de los 10 jugadores (LP)")
ax.set_title("Rating medio de la partida según su rango (train)")
plt.tight_layout()
plt.show()
display(r10.groupby("tier")["mu_medio"].describe()[["count", "25%", "50%", "75%"]].reindex(orden).round(0))
""")

destino = cells

md(r"""
## 5. Pendiente: evaluación con la segunda foto

**Por qué no se evalúa todavía.** Un sistema de rating solo mide habilidad
*relativa* a los rivales: si todos arrancan en el mismo valor, un grupo de
Challengers que juega entre sí no tiene forma de llegar a 2.400 LP. Evaluarlo antes
de la foto mide sobre todo esa falta de ancla. Además, elegir los parámetros por
cómo predicen partidas no sirve para el elo: la configuración que mejor predice
(log-loss 0,6919 contra 0,6926 de una constante) ordena a los jugadores peor contra
la foto que los parámetros por defecto (Spearman 0,45 contra 0,58).

**Cómo se va a evaluar.**

1. Se arranca en la foto del 04/09: cada jugador del padrón con su LP real.
2. Se recorren las partidas posteriores, actualizando el rating con cada una.
3. En la fecha de la segunda foto, se compara el rating de cada jugador con su LP
   real. Esa es la pérdida para ajustar $eta$, $	au$ y $\sigma_0$, y para medir
   cómo crece el error con la distancia en días.

La segunda foto es posterior al test, así que sus W/L incluyen resultados del test:
sirve para evaluar el rating, **no** para ajustar nada que alimente al modelo de
draft. Las secciones de evaluación ya escritas (ajuste, calibración, trayectorias)
se generan con `python scripts/analysis/build_rating_nb.py --evaluacion`.
""")

if EVALUACION:
    cells.extend(cells_eval)

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, OUT)
print(f"[+] {OUT} ({len(cells)} celdas)")
