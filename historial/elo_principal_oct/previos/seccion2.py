md(r"""
# 2. Modelos

Predicen el LP del 04/09 partiendo del LP del 28/10 y recorriendo las partidas
**hacia atrás**. Primero dos baselines; después tres modelos que aprenden (gradient
boosting, NGBoost y GRU), en tres niveles de mejoras que se van sumando; después el
stacking de la GRU con NGBoost, y al final la comparación de los modelos completos.

| Modelo | Qué hace |
| :--- | :--- |
| **No cambió** | $\hat z_1 = z_2$ |
| **Elo al revés** | Deshace cada partida: $-a$ por victoria y $+b$ por derrota, con $a, b$ del rango, multiplicado por la sorpresa del resultado |
| **Gradient boosting** | `HistGradientBoostingRegressor` (de la familia de XGBoost y LightGBM): árboles sobre histogramas que minimizan el error absoluto. Da un número |
| **NGBoost** | Natural Gradient Boosting: ajusta $\mu$ y $\sigma$ de una Normal por jugador con el gradiente natural. Da una distribución |
| **GRU** | Red recurrente que lee las partidas de una en una, de la más nueva a la más vieja, y junta su estado final con las variables del jugador |

**Dos mediciones.** *Dentro de la muestra:* se ajusta y se mide sobre los mismos
jugadores (el protocolo del train; optimista). *5 folds:* los jugadores se reparten en 5
grupos fijos (`folds_por_jugador`) y a cada uno lo predice un modelo entrenado con los
otros 4. Es la que dice cómo generaliza a jugadores que el modelo no vio. Los pocos
parámetros que no aprende el modelo ($a, b$ de la regla y la recta del lobby del Kalman)
se ajustan con todos.

Métricas: **MAE** y **RMSE** en LP, **R² del cambio** (qué parte del cambio real
$z_1 - z_2$ explica; "no cambió" da 0) y **ρ dentro del rango** (Spearman dentro del
rango del 28/10: si ordena bien a los que la foto 2 pone juntos).
""")

code(r"""
CORTE = cortes(foto2)   # rangos con los cortes de la foto 2, la que se conoce al arrancar
A, B = ajustar_regla(e)
cambio = e["z1"] - e["z2"]
ANALISIS = DATA_DIR / "analysis"

BOOST = pd.read_csv(ANALISIS / "elo_boost_pred.csv", index_col="player_id").reindex(e.index)


def leer_gru(nombre):
    ruta = ANALISIS / f"{nombre}_pred.csv"
    if not ruta.exists():
        print(f"Sin {ruta.name}: correr `scripts/analysis/elo_gru.py` (ver el docstring del generador).")
        return None
    return pd.read_csv(ruta).set_index("player_id")["pred_gru"].reindex(e.index)


# Por nivel y modelo: (dentro de la muestra, 5 folds).
PRED = {n: {"Gradient boosting": (BOOST[f"gb_{n}"], BOOST[f"gb_{n}_oof"]),
            "NGBoost": (BOOST[f"ngb_{n}"], BOOST[f"ngb_{n}_oof"]),
            "GRU": (leer_gru(f"elo_gru_{n}"), leer_gru(f"elo_gru_{n}_oof"))} for n in NIVELES}
BASELINES = {"No cambió": e["z2"].astype(float),
             "Elo al revés": al_reves(partes, foto2["z"], A, B, CORTE, modo="elo", beta=3200)
                             .reindex(e.index).fillna(e["z2"])}


def evaluar(pred, idx=None):
    idx = e.index if idx is None else idx
    err = e.loc[idx, "z1"] - pred[idx]
    ca = cambio[idx]
    rho = [(len(g), g["z1"].corr(pred[g.index], method="spearman"))
           for _, g in e.loc[idx].groupby("tier2") if len(g) >= 30 and pred[g.index].nunique() > 1]
    return {"MAE": err.abs().mean(), "RMSE": np.sqrt((err ** 2).mean()),
            "R² del cambio": 1 - (err ** 2).sum() / ((ca - ca.mean()) ** 2).sum(),
            "ρ dentro del rango": sum(n * r for n, r in rho) / sum(n for n, _ in rho) if rho else np.nan}


RESALTE = "background-color: #d8e8f8; color: #0b0b0b; font-weight: bold"   # legible también con tema oscuro
FORMATO = {"MAE": "{:.1f}", "RMSE": "{:.1f}", "R² del cambio": "{:.3f}", "ρ dentro del rango": "{:.3f}"}


def mostrar(tabla, titulo):
    display(tabla.style.format(FORMATO).set_caption(titulo)
            .highlight_min(subset=["MAE", "RMSE"], props=RESALTE)
            .highlight_max(subset=["R² del cambio", "ρ dentro del rango"], props=RESALTE))


def mostrar_nivel(modelos):
    # modelos: {nombre: (dentro de la muestra, 5 folds)}; los que no están (GRU sin correr) se saltean.
    modelos = {m: p for m, p in modelos.items() if p[0] is not None and p[1] is not None}
    mostrar(pd.DataFrame({m: evaluar(p[0]) for m, p in modelos.items()}).T, "Dentro de la muestra")
    mostrar(pd.DataFrame({m: evaluar(p[1]) for m, p in modelos.items()}).T, "5 folds por jugador")
""")

md(r"""
## 2.1 Baselines

* **No cambió:** el caso trivial, el LP del 04/09 es el del 28/10.
* **Elo al revés** (β = 3200, el mejor de una grilla): la referencia sin aprendizaje. Con
  β grande la sorpresa pesa poco y queda casi en la regla al revés, $-a$ por victoria y
  $+b$ por derrota.

No aprenden de los jugadores, así que no tienen versión fuera de fold.
""")

code(r"""
mostrar(pd.DataFrame({n: evaluar(p) for n, p in BASELINES.items()}).T, "Baselines")
""")

md(r"""
## 2.2 Sin mejoras

Los tres modelos predicen el cambio $z_1 - z_2$ directo, solo con lo mínimo del
jugador: su LP y su W + L en la foto 2 y cuántas partidas y victorias le observamos. La
GRU, además, lee por partida el resultado, cuántos días antes de la foto fue y el lado.
""")

code(r"""
mostrar_nivel(PRED["sin_mejoras"])
""")

md(r"""
## 2.3 Mejoras simples: residuo de la regla, deriva y variables

* **Residuo de la regla al revés.** El modelo ya no predice el cambio entero: se deshace
  la regla ($-a$ por victoria, $+b$ por derrota, con $a, b$ del rango en cada paso) y el
  modelo corrige lo que la regla erra, $z_1 - \hat z_1^{regla}$.
* **Deriva.** $a$ y $b$ se ajustan por separado: en Master+ $a > b$, así que un jugador
  de winrate 50 % sube $(a - b)/2$ LP por partida.
* **Variables.** La regla misma, el winrate observado (suavizado) y el de la temporada, y
  el lobby: el LP medio de los otros 9 de cada partida con su foto 2 (medio, de las 10
  más viejas, de las 10 más nuevas y el *gap* lobby − LP propio). Riot arma las partidas
  por MMR, así que el lobby mide el nivel del jugador. La GRU lee por partida el LP medio
  de aliados, de rivales y de los 9, y cuántos tienen foto.
""")

code(r"""
mostrar_nivel(PRED["simple"])
""")

md(r"""
## 2.4 Mejoras completas: decay, Kalman y lobby con el LP predicho

* **Lobby con el LP predicho.** El LP de cada uno de los otros 9 ya no es el de su foto 2
  sino el que tenía *en esa partida*, deshaciendo la regla desde su foto 2. No usa el
  target.
* **Kalman como variable.** Un suavizador de Kalman al revés: deshace las partidas
  observadas, suma incertidumbre por las no observadas (con la tasa de su rango) y usa el
  lobby de cada partida como medición ruidosa del LP. Su media y su desvío en la foto 1
  entran como variables.
* **Decay total.** Lo que perdería un Master+ por inactividad entre fotos (75 LP por día
  pasados 14 días sin jugar). La GRU lee además, por partida, los días hasta la siguiente.
""")

code(r"""
mostrar_nivel(PRED["completo"])

filas = [{"nivel": n.replace("_", " "), "modelo": m, "medición": med, **evaluar(p[i])}
         for n in NIVELES for m, p in PRED[n].items()
         for i, med in ((0, "dentro de la muestra"), (1, "5 folds")) if p[i] is not None]
prog = pd.DataFrame(filas)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
for ax, met in zip(axes, ["MAE", "RMSE"]):
    sns.lineplot(data=prog, x="nivel", y=met, hue="modelo", style="medición", palette=COLOR_MODELO,
                 markers=True, dashes={"dentro de la muestra": (2, 2), "5 folds": ""}, linewidth=2, ax=ax)
    ax.set_xlabel(""); ax.set_ylabel(f"{met} (LP)")
    ax.set_title(f"{met} según las mejoras")
axes[1].get_legend().remove()
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.show()
""")

md(r"""
## 2.5 Stacking: la GRU con la μ y σ de NGBoost

La GRU completa recibe además, como variables, la predicción de NGBoost ($\mu$, el
residuo de la regla que predice) y su incertidumbre ($\sigma$). La idea: la GRU es la que
mejor generaliza y NGBoost el que da incertidumbre por jugador.

**Sin fuga.** Si la GRU viera la $\mu$ de un NGBoost entrenado con el mismo jugador,
aprendería a confiar en una predicción que fuera de la muestra es peor. Por eso:

* *Dentro de la muestra:* la $\mu$ y $\sigma$ de cada jugador vienen de NGBoost fuera de
  fold.
* *5 folds:* folds anidados. La GRU del fold $k$ se entrena con jugadores cuya $\mu$ salió
  de NGBoost que no vieron ni a ellos ni al fold $k$, y predice el fold $k$ con la $\mu$
  de un NGBoost que no vio el fold $k$.

Como referencia, el **promedio** simple de la GRU y NGBoost, que no aprende nada.
""")

code(r"""
STACK = (leer_gru("elo_gru_completo_con_ngb"), leer_gru("elo_gru_completo_con_ngb_oof"))
comp = PRED["completo"]
st = {"NGBoost": comp["NGBoost"], "GRU": comp["GRU"]}
if comp["GRU"][0] is not None:
    st["Promedio GRU y NGBoost"] = tuple((comp["GRU"][i] + comp["NGBoost"][i]) / 2 for i in (0, 1))
st["GRU + NGBoost"] = STACK
mostrar_nivel(st)
""")

md(r"""
## 2.6 Error por rango

Desde acá, los modelos completos (más el stacking) contra el Elo al revés, **con las
predicciones de los 5 folds**. MAE según el rango del jugador el 28/10.
""")

code(r"""
FINALES = {"Elo al revés": BASELINES["Elo al revés"], **{m: p[1] for m, p in PRED["completo"].items()},
           "GRU + NGBoost": STACK[1]}
FINALES = {m: p for m, p in FINALES.items() if p is not None}

filas = [{"modelo": n, "rango": t.capitalize(), "MAE": (e["z1"] - p)[e["tier2"] == t].abs().mean()}
         for n, p in FINALES.items() for t in GRUPOS]
por_rango = pd.DataFrame(filas)
n_rango = e["tier2"].value_counts()
display(por_rango.pivot(index="modelo", columns="rango", values="MAE")[[t.capitalize() for t in GRUPOS]]
        .reindex(list(FINALES)).style.format("{:.1f}").set_caption("MAE (LP), 5 folds"))

fig, ax = plt.subplots(figsize=(11, 4.4))
sns.barplot(data=por_rango, x="rango", y="MAE", hue="modelo", palette=COLOR_MODELO, ax=ax)
for c in ax.containers:
    ax.bar_label(c, fmt="%.0f", padding=2, fontsize=8)
ax.set_xticks(range(len(GRUPOS)), [f"{t.capitalize()}\n(n = {n_rango.get(t, 0):,})" for t in GRUPOS])
ax.set_xlabel(""); ax.set_ylabel("MAE (LP)")
ax.set_title("Error por rango el 28/10 (5 folds)")
ax.grid(axis="x", visible=False)
ax.legend(title=None)
plt.tight_layout()
plt.show()
""")

md(r"""
## 2.7 Error según cuántas partidas se recorren hacia atrás

Partidas jugadas entre fotos según Riot. El test va más lejos que el train: esta curva
anticipa cuánto se degrada.
""")

code(r"""
TRAMOS = ["1–10", "11–25", "26–50", "51–100", "101–200", "> 200"]
tramos = pd.cut(e["dN"], [0, 10, 25, 50, 100, 200, 10_000], labels=TRAMOS)
curva = pd.concat([(e["z1"] - p).abs().groupby(tramos, observed=True).mean().rename("MAE").reset_index()
                   .assign(modelo=n) for n, p in FINALES.items()], ignore_index=True)
curva["dN"] = curva["dN"].astype(str)

fig, ax = plt.subplots(figsize=(10, 4.4))
sns.lineplot(data=curva, x="dN", y="MAE", hue="modelo", palette=COLOR_MODELO, marker="o", linewidth=2,
             sort=False, ax=ax)
cuenta = tramos.value_counts().reindex(TRAMOS)
ax.set_xticks(range(len(TRAMOS)), [f"{t}\n({n:,})" for t, n in cuenta.items()])
ax.set_xlabel("Partidas entre fotos (jugadores)"); ax.set_ylabel("MAE (LP)")
ax.set_title("El error crece con la distancia a la foto (5 folds)")
ax.legend(title=None)
plt.tight_layout()
plt.show()
""")

md(r"""
## 2.8 Los que más se movieron

Tres grupos del mismo tamaño $n$, donde $n$ es la cantidad de jugadores que cambiaron de
rango:

* **Cambiaron de rango.**
* **Más cambio de LP:** los $n$ con mayor $|z_2 - z_1|$.
* **Más cambio de LP sin cambiar de rango:** los $n$ con mayor $|z_2 - z_1|$ entre los que
  quedaron en el mismo rango.

**¿Subestima o sobreestima el cambio?** El sesgo es el error con signo en la dirección
del cambio real, $(\hat z_1 - z_1)\,\mathrm{sign}(z_1 - z_2)$. Negativo: el modelo **se
queda corto** (predice que el jugador se movió menos de lo que se movió). Positivo: **se
pasa**.
""")

code(r"""
dlp = (e["z2"] - e["z1"]).abs()
cambio_rango = e.index[e["tier1"] != e["tier2"]]
n = len(cambio_rango)
mas_lp = dlp.nlargest(n).index
mismo_rango = dlp[e["tier1"] == e["tier2"]].nlargest(n).index
GRUPOS_J = {"Todos": e.index, "Cambiaron de rango": cambio_rango, "Más cambio de LP": mas_lp,
            "Más cambio de LP, mismo rango": mismo_rango}

comp_n = len(cambio_rango.intersection(mas_lp))
print(f"n = {n:,} ({100 * n / len(e):.1f} % de los {len(e):,} jugadores).")
print(f"Cambiaron de rango ∩ más cambio de LP: {comp_n:,} ({100 * comp_n / n:.1f} %).")
print(f"|ΔLP| mínimo para entrar: más cambio de LP {dlp[mas_lp].min():.0f} LP; "
      f"mismo rango {dlp[mismo_rango].min():.0f} LP. Mediana de |ΔLP| de los que cambiaron de rango: "
      f"{dlp[cambio_rango].median():.0f} LP.")

signo = np.sign(e["z1"] - e["z2"])
filas = [{"grupo": gn, "modelo": mn, **evaluar(p, idx),
          "sesgo": ((p[idx] - e.loc[idx, "z1"]) * signo[idx]).mean(),
          "% se queda corto": 100 * ((p[idx] - e.loc[idx, "z1"]) * signo[idx] < 0).mean()}
         for gn, idx in GRUPOS_J.items() for mn, p in FINALES.items()]
por_grupo = pd.DataFrame(filas)
for val, fmt, cap in (("MAE", "{:.1f}", "MAE (LP)"), ("RMSE", "{:.1f}", "RMSE (LP)"),
                      ("% se queda corto", "{:.1f}", "% de jugadores en que se queda corto")):
    display(por_grupo.pivot(index="modelo", columns="grupo", values=val)[list(GRUPOS_J)].reindex(list(FINALES))
            .style.format(fmt).set_caption(cap + ", 5 folds"))

fig, axes = plt.subplots(2, 1, figsize=(11, 8.4), sharex=True)
for ax, col, titulo in ((axes[0], "MAE", "Error en los jugadores que más se movieron (5 folds)"),
                        (axes[1], "sesgo", "Sesgo: negativo = se queda corto, positivo = se pasa")):
    sns.barplot(data=por_grupo, x="grupo", y=col, hue="modelo", palette=COLOR_MODELO, order=list(GRUPOS_J), ax=ax)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.0f", padding=2, fontsize=8)
    ax.set_xlabel(""); ax.set_ylabel(col + " (LP)")
    ax.set_title(titulo)
    ax.grid(axis="x", visible=False)
    ax.legend(title=None)
axes[1].axhline(0, color=INK_2, linewidth=1)
axes[1].get_legend().remove()
axes[1].set_xticks(range(len(GRUPOS_J)), [f"{g}\n(n = {len(i):,})" for g, i in GRUPOS_J.items()])
plt.tight_layout()
plt.show()
""")

md(r"""
## 2.9 LP partida a partida de 3 jugadores

Los modelos predicen solo el LP del 04/09 (acá, con las predicciones de los 5 folds). Su
camino sale de deshacer la regla desde el 28/10 y repartir la corrección del modelo en
partes iguales entre las partidas: arranca en lo que predice el modelo y termina en la
foto 2. La barra verde es el intervalo del 95 % de NGBoost en la foto 1. La recta gris une
el LP real de las dos fotos. Jugadores al azar entre los que tienen al menos 20 partidas
observadas.
""")

code(r"""
fecha = lambda ms: pd.to_datetime(ms, unit="ms")  # noqa: E731
rng = np.random.default_rng(7)
elegidos = rng.choice(e.index[e["obs"] >= 20].to_numpy(), size=3, replace=False)
por_jugador = partes[partes["player_id"].isin(elegidos)].groupby("player_id")
curvas = [m for m in FINALES if m != "Elo al revés"]
SIGMA_NGB = BOOST["ngb_completo_oof_sigma"]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
for ax, pid in zip(axes, elegidos):
    p = por_jugador.get_group(pid).sort_values("game_creation")
    xs = fecha(np.r_[d["t1"], p["game_creation"].to_numpy(), d["t2"]])
    for m in curvas:
        lp = camino(p, e.at[pid, "z2"], FINALES[m][pid], A, B, CORTE)
        ax.step(xs, np.r_[lp, lp[-1]], where="post", color=COLOR_MODELO[m], linewidth=1.6, label=m)
    ax.errorbar(fecha(d["t1"]), FINALES["NGBoost"][pid], yerr=1.96 * SIGMA_NGB[pid], color=COLOR_MODELO["NGBoost"],
                capsize=4, linewidth=1.2, label="NGBoost IC 95 %")
    ax.plot(fecha([d["t1"], d["t2"]]), [e.at[pid, "z1"], e.at[pid, "z2"]], color=MUTED, linestyle="--",
            linewidth=1.4, label="Recta foto 1 → foto 2")
    ax.scatter(fecha([d["t1"], d["t2"]]), [e.at[pid, "z1"], e.at[pid, "z2"]], color=INK, zorder=5, s=25,
               label="LP real en las fotos")
    ax.set_title(f"{e.at[pid, 'tier1'].capitalize()} → {e.at[pid, 'tier2'].capitalize()} · "
                 f"{int(e.at[pid, 'obs'])} partidas", fontsize=10)
    ax.tick_params(axis="x", rotation=30)
axes[0].set_ylabel("LP absoluto")
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.show()
""")
