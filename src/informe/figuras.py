"""Una función por figura o tabla del notebook de la entrega.

Cada función lee resultados ya guardados (`src/informe/datos.py`), arma la figura con el estilo de
`src/informe/estilo.py`, la guarda en `notebooks/figuras/informe/<nombre>.png` y la devuelve. Las tablas se
devuelven como `Styler` de pandas, con los números en castellano.

Los números que no salen de un archivo sino del texto de `PROJECT_SUMMARY.md` están en constantes con la
sección citada al lado (`APORTES_PREVIO`, `SUMA_PREVIO_DOC`).
"""
from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown

from src.informe import datos as D
from src.informe import estilo as E
from src.informe.estilo import AZUL, GRIS, GRIS_CLARO, MODELO, RAMPA, ROJO, TINTA, TINTA_2

E.aplicar()


def _tabla(df: pd.DataFrame):
    """Styler con aspecto sobrio: sin índice, encabezado a la izquierda."""
    return (df.style.hide(axis="index")
            .set_table_styles([{"selector": "th", "props": "text-align:left; font-weight:600; padding:4px 12px;"},
                               {"selector": "td", "props": "text-align:left; padding:4px 12px;"}]))


# =================================================================================================================
# 1. La pregunta y los datos
# =================================================================================================================

def tabla_datos():
    """Cuántas partidas hay en cada tramo de la partición temporal."""
    P = D.partidas()
    m = D.manifiesto()
    ajuste_master = int(((P.tramo == "ajuste") & P.master).sum())
    filas = [
        ("Ajuste", "nov. 2024 → 15/09", E.entero(ajuste_master),
         "Master+ anteriores a V; con ellas se estiman los pesos"),
        ("Bloques hacia adelante", "07/09 → 15/09",
         " / ".join(E.entero(((P.bloque == b) & P.master).sum()) for b in ["A2", "A3", "A4"]),
         "cada bloque se predice con lo anterior: elige variantes sin mirar V"),
        ("Validación (V)", "16/09 → 26/09", E.entero((P.tramo == "V").sum()),
         "lobbies Master+; se mira una vez por variante"),
        ("Diamante de la ventana de V", "16/09 → 26/09", E.entero((P.tramo == "Diamond-V").sum()),
         "control: nunca se usó para decidir"),
        ("Test", "desde el 27/09", E.entero(m["test"]),
         f"{E.pct(m['reparto_test']['MASTER'])} Master; se evalúa una sola vez"),
    ]
    return _tabla(pd.DataFrame(filas, columns=["Tramo", "Fechas", "Partidas", "Para qué"]))


def fig_linea_tiempo():
    """Partidas por día y los tramos de la partición: ajuste, bloques hacia adelante, V y test."""
    P = D.partidas()
    desde, hasta = pd.Timestamp("2026-06-01"), pd.Timestamp("2026-10-27")
    por_dia = P[P.t >= desde].set_index("t").resample("D").size()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.2, 4.8), sharex=True, height_ratios=[1.2, 1])
    m = D.manifiesto()
    a2 = P.loc[P.bloque == "A2", "t"].min()
    carriles = [  # (fila, inicio, fin, color, nombre, texto al costado)
        (3, desde, D.CORTE_V, GRIS_CLARO, "Ajuste",
         f"{E.entero(((P.tramo == 'ajuste') & P.master).sum())} Master+ desde nov. 2024"),
        (2, a2, D.CORTE_V, GRIS, "Bloques hacia adelante", "A2 · A3 · A4"),
        (1, D.CORTE_V, D.CORTE_TEST, MODELO, "Validación (V)", f"{E.entero((P.tramo == 'V').sum())} Master+"),
        (0, D.CORTE_TEST, hasta, TINTA_2, "Test (no se carga)", f"{E.entero(m['test'])} partidas"),
    ]
    for fila, a, b, color, nombre, texto in carriles:
        ax1.barh(fila, (b - a) / pd.Timedelta(days=1), left=a, height=0.6, color=color,
                 edgecolor="white", linewidth=2, hatch="////" if fila == 0 else None)
        if fila == 3:
            ax1.text(desde + pd.Timedelta(days=3), fila, texto, va="center", fontsize=9, color=TINTA)
        elif fila == 0:
            ax1.text(a - pd.Timedelta(days=2), fila, texto, va="center", ha="right", fontsize=9, color=TINTA_2)
        else:
            ax1.text(b + pd.Timedelta(days=2), fila, texto, va="center", fontsize=9, color=TINTA_2)
    for b in ["A3", "A4"]:  # separadores de bloques
        ax1.plot([P.loc[P.bloque == b, "t"].min()] * 2, [1.68, 2.32], color="white", lw=2.5)
    for ax in (ax1, ax2):
        ax.axvline(D.FOTO1, color=TINTA_2, lw=0.9)
    ax1.text(D.FOTO1 - pd.Timedelta(days=1.5), 3.75, "foto 1 del ranking (04/09)", fontsize=8.8,
             color=TINTA_2, ha="right", va="center")
    ax1.set_yticks([3, 2, 1, 0], [c[4] for c in carriles])
    ax1.set_ylim(-0.5, 4.1)
    ax1.grid(False)
    ax1.spines["left"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    E.titulos(ax1, "Cómo se parte el tiempo", "Todo lo que se usa para decidir es anterior a lo que se evalúa")

    ax2.bar(por_dia.index, por_dia.values, width=0.8, color=GRIS_CLARO)
    v = por_dia[(por_dia.index >= D.CORTE_V) & (por_dia.index < D.CORTE_TEST)]
    ax2.bar(v.index, v.values, width=0.8, color=MODELO)
    ax2.set_ylabel("Partidas por día")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: E.entero(x)))
    ax2.set_xlim(desde, hasta)
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    meses = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: meses[mdates.num2date(x).month - 1]
                                                    + " " + str(mdates.num2date(x).year)))
    return E.guardar(fig, "01_linea_de_tiempo")


# =================================================================================================================
# 2. Modelo previo
# =================================================================================================================

# Escalera de grupos de variables del modelo previo, cada paso medido contra el anterior en V Master+ con bootstrap
# pareado. Fuente: PROJECT_SUMMARY.md §9.S.52 ("Escalera aceptada", escalones 1-4b, 6, 7 y 9; los ensambles 5, 8 y
# 10 no son parte del modelo final). Las dos correcciones de fuga: §9.S.68 (tabla "Validación pareada", V Master+)
# y §9.S.75 (tabla "Resultado").
APORTES_PREVIO = [
    # (texto, AUC acumulado, Δ, IC inferior, IC superior, ¿corrección?)
    ("Punto de partida: draft y perfil de cada jugador", 0.6531, None, None, None, False),
    ("Ajustar solo con partidas Master+", 0.6557, 0.0026, 0.0012, 0.0038, False),
    ("Rendimiento del jugador con ese campeón y en esa línea", 0.6614, 0.0058, 0.0028, 0.0085, False),
    ("Nivel oculto por emparejamiento (los rivales que le tocan)", 0.6642, 0.0028, 0.0017, 0.0039, False),
    ("Victorias y derrotas de la temporada", 0.6663, 0.0021, 0.0004, 0.0034, False),
    ("Contracruces y sinergias de campeones", 0.6674, 0.0011, 0.0006, 0.0017, False),
    ("Winrate por línea y costo de jugar fuera de rol", 0.6682, 0.0009, 0.0000, 0.0016, False),
    ("Elo dinámico, ritmo, objetivos y economía de cada jugador", 0.6707, 0.0024, 0.0010, 0.0040, False),
    ("Oro esperado de cada línea antes de jugar", 0.6722, 0.0012, 0.0006, 0.0019, False),
    ("Ventaja esperada a los 10 y 15 min por par de campeones", 0.6739, 0.0018, 0.0008, 0.0027, False),
    ("Estadísticas de normalización solo con datos anteriores a V", 0.6740, 0.00013, -0.00013, 0.00038, True),
    ("Sin el LP de la foto en partidas anteriores a la foto", 0.6727, -0.0013, -0.0025, -0.0002, True),
]


def fig_aportes_previo():
    """Cuánto suma cada grupo de variables al AUC del modelo previo en V (ΔAUC con IC 95 %)."""
    filas = APORTES_PREVIO[::-1]
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    y = np.arange(len(filas))
    for i, (txt, auc, d, lo, hi, corr) in enumerate(filas):
        if d is not None:
            ax.barh(i, d, height=0.5, color=GRIS if corr else MODELO)
            ax.plot([lo, hi], [i, i], color=TINTA, lw=1.1)
            ax.plot([lo, lo], [i - 0.13, i + 0.13], color=TINTA, lw=1.1)
            ax.plot([hi, hi], [i - 0.13, i + 0.13], color=TINTA, lw=1.1)
        ax.text(0.0105, i, E.num(auc), va="center", ha="right", fontsize=9.5,
                color=TINTA, fontweight="semibold" if i == 0 else "normal")
    ax.text(0.0105, len(filas) - 0.45, "AUC", ha="right", va="center", fontsize=9, color=TINTA_2)
    ax.axhline(1.5, color=GRIS_CLARO, lw=1)
    ax.set_yticks(y, [f[0] for f in filas], fontsize=9.5)
    ax.axvline(0, color=E.EJE, lw=1)
    ax.set_xlim(-0.003, 0.0108)
    ax.set_ylim(-0.6, len(filas) - 0.2)
    ax.tick_params(axis="y", colors=TINTA)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    E.eje_coma(ax, "x", 3)
    ax.set_xlabel("Cuánto sube el AUC en V al agregar el grupo (IC 95 %, bootstrap pareado)")
    E.titulos(ax, "Qué aporta cada grupo de variables al modelo previo",
              "En orden de incorporación; en gris, las dos correcciones de fuga temporal que dejan el modelo final")
    return E.guardar(fig, "02_aportes_modelo_previo")


def tabla_previo():
    """Métricas del modelo previo en V Master+ con IC 95 % (bootstrap de partidas, 1.000 réplicas)."""
    v = D.prediccion_previo_v()
    r = D.resumen(v.y.values, v.p.values)
    filas = [
        ("AUC", E.ic(*r["auc"])),
        ("Acierto", E.ic(*r["acierto"], d=1, porcentaje=True)),
        ("Acierto de «siempre rojo»", E.ic(*r["siempre_rojo"], d=1, porcentaje=True)),
        ("Log loss", E.ic(*r["log_loss"])),
        ("Error de calibración (ECE, 10 deciles)", E.num(r["ece"][0])),
    ]
    return _tabla(pd.DataFrame(filas, columns=[f"V Master+ ({E.entero(r['n'])} partidas)", "Valor [IC 95 %]"]))


def respuesta_previo():
    """La respuesta de la sección en una línea, con los números calculados (no escritos a mano)."""
    v = D.prediccion_previo_v()
    r = D.resumen(v.y.values, v.p.values)
    return Markdown(f"**Respuesta.** En V Master+, entre una partida que ganó el azul y otra que ganó el rojo, el "
                    f"modelo le da más chances al azul en la primera el {E.pct(r['auc'][0])} de las veces "
                    f"(AUC {E.ic(*r['auc'])}), y acierta el ganador en el "
                    f"{E.ic(*r['acierto'], d=1, porcentaje=True)}, contra {E.pct(r['siempre_rojo'][0])} de apostar "
                    f"siempre al rojo.")


def fig_separacion_previo():
    """Distribución de P(gana azul) según quién ganó de verdad."""
    v = D.prediccion_previo_v()
    bins = np.linspace(0, 1, 41)
    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    for gano, color, txt in [(1, AZUL, "Ganó el lado azul"), (0, ROJO, "Ganó el lado rojo")]:
        h, _ = np.histogram(v.p[v.y == gano], bins=bins, density=True)
        ax.fill_between(bins[:-1], h, step="post", color=color, alpha=0.12, lw=0)
        ax.step(bins[:-1], h, where="post", color=color, lw=2, label=txt)
    ax.axvline(0.5, color=GRIS, lw=1)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel("Densidad")
    E.eje_coma(ax, "x", 0, porcentaje=True)
    ax.set_xlabel("Probabilidad de que gane el azul, antes de la partida")
    ax.legend(loc="upper right")
    E.titulos(ax, "El modelo separa, pero con mucha superposición",
              "V Master+: cada curva es la distribución de la predicción en las partidas que ganó cada lado")
    return E.guardar(fig, "03_separacion_previo")


def fig_calibracion_cobertura():
    """(a) Calibración por deciles con IC de Wilson; (b) acierto en la fracción de partidas más seguras."""
    v = D.prediccion_previo_v()
    y, p = v.y.values, v.p.values
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 4.0))

    g = pd.DataFrame({"p": p, "y": y, "d": pd.qcut(p, 10, labels=False)}).groupby("d").agg(
        p=("p", "mean"), k=("y", "sum"), n=("y", "size"))
    ic = np.array([D.proporcion_ic(k, n) for k, n in zip(g.k, g.n)])
    a.plot([0.15, 0.85], [0.15, 0.85], color=GRIS, lw=1)
    a.errorbar(g.p, ic[:, 0], yerr=[ic[:, 0] - ic[:, 1], ic[:, 2] - ic[:, 0]], fmt="none", ecolor=TINTA, lw=1.1)
    a.plot(g.p, ic[:, 0], "o", color=MODELO, ms=7, mec="white", mew=1.5)
    a.text(0.63, 0.53, "calibración perfecta", color=TINTA_2, fontsize=8.8, ha="left", va="top")
    for ax in (a,):
        E.eje_coma(ax, "x", 0, porcentaje=True)
        E.eje_coma(ax, "y", 0, porcentaje=True)
    a.set_xlabel("Probabilidad predicha (media del decil)")
    a.set_ylabel("Frecuencia real de victoria azul")
    a.grid(axis="x")
    E.titulos(a, "Calibración en V", "Deciles de la predicción, IC 95 % de Wilson")

    orden = np.argsort(-np.abs(p - 0.5))
    ok = ((p > 0.5) == y)[orden].astype(float)
    fr = np.linspace(0.05, 1, 39)
    W = D.pesos_bootstrap(len(y), 500)[:, orden]
    curva, lo, hi = [], [], []
    for f in fr:
        k = int(f * len(y))
        curva.append(ok[:k].mean())
        reps = (W[:, :k] @ ok[:k]) / W[:, :k].sum(1)
        lo.append(np.percentile(reps, 2.5))
        hi.append(np.percentile(reps, 97.5))
    b.fill_between(fr, lo, hi, color=MODELO, alpha=0.12, lw=0)
    b.plot(fr, curva, color=MODELO)
    base = 1 - y.mean()
    b.axhline(base, color=GRIS, lw=1)
    b.text(0.98, base + 0.008, f"siempre rojo: {E.pct(base)}", ha="right", fontsize=8.8, color=TINTA_2)
    for f in (0.1, 0.2, 0.5, 1.0):
        k = int(f * len(y))
        b.plot(f, ok[:k].mean(), "o", color=MODELO, ms=7, mec="white", mew=1.5)
        b.annotate(E.pct(ok[:k].mean()), (f, ok[:k].mean()), xytext=(0, 9), textcoords="offset points",
                   ha="center", fontsize=8.8, color=TINTA)
    b.set_xlim(0, 1.03)
    b.set_ylim(0.48, 0.86)
    E.eje_coma(b, "x", 0, porcentaje=True)
    E.eje_coma(b, "y", 0, porcentaje=True)
    b.set_xlabel("Fracción de partidas, de la más segura a la menos")
    b.set_ylabel("Acierto")
    E.titulos(b, "Acierto según cuánta confianza tiene", "Banda: IC 95 % por bootstrap de partidas")
    return E.guardar(fig, "04_calibracion_y_cobertura")


# =================================================================================================================
# 3. Modelo en vivo
# =================================================================================================================

def _fila(M, conjunto, modelo, minuto):
    f = M[(M.conjunto == conjunto) & (M.modelo == modelo) & (M.minuto == minuto)]
    assert len(f) == 1, (conjunto, modelo, minuto)
    return f.iloc[0]


def tabla_en_vivo():
    """Modelo en vivo en V Master+: misma cohorte (vivas al 20) y cada minuto con todas sus partidas vivas."""
    M = D.metricas_en_vivo()
    filas = []
    for m in (10, 15, 20):
        c, t = _fila(M, "V cohorte 20", "etapa1", m), _fila(M, "V", "etapa1", m)
        filas.append((f"Minuto {m}", E.ic(c.auc, c.auc_lo, c.auc_hi),
                      E.ic(c.acierto, c.acierto_lo, c.acierto_hi, 1, True),
                      f"{E.num(t.auc)} · {E.pct(t.acierto)} ({E.entero(t.n)})"))
    p = _fila(M, "V cohorte 20", "previo", 10)
    filas.insert(0, ("Antes (solo el previo)", E.ic(p.auc, p.auc_lo, p.auc_hi),
                     E.ic(p.acierto, p.acierto_lo, p.acierto_hi, 1, True), "—"))
    n = int(_fila(M, "V cohorte 20", "etapa1", 20).n)
    cols = ["Momento", f"AUC, misma cohorte ({E.entero(n)})", "Acierto, misma cohorte",
            "AUC · acierto con todas las vivas (n)"]
    return _tabla(pd.DataFrame(filas, columns=cols))


def respuesta_en_vivo():
    """La respuesta de la sección 3 con los números de `metricas.csv`."""
    M = D.metricas_en_vivo()
    f = {m: _fila(M, "V cohorte 20", "etapa1", m) for m in (10, 15, 20)}
    p = _fila(M, "V cohorte 20", "previo", 10)
    return Markdown(f"**Respuesta.** Sobre las mismas {E.entero(f[20].n)} partidas de V, el AUC pasa de "
                    f"{E.num(p.auc)} antes de jugar a {E.num(f[10].auc)} / {E.num(f[15].auc)} / {E.num(f[20].auc)} "
                    f"a los 10 / 15 / 20 minutos, y el acierto de {E.pct(p.acierto)} a {E.pct(f[10].acierto)} / "
                    f"{E.pct(f[15].acierto)} / {E.pct(f[20].acierto)} (IC en la tabla). Antes de jugar estas "
                    f"partidas son más difíciles que el promedio de V: las que terminan antes del minuto 20 suelen "
                    f"ser las más desparejas.")


def fig_en_vivo_minutos():
    """AUC y acierto del modelo en vivo a los 10/15/20 con IC, contra el previo solo y 'siempre rojo'."""
    M = D.metricas_en_vivo()
    xs = [0, 1, 2, 3]
    etq = ["Antes", "Minuto 10", "Minuto 15", "Minuto 20"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    for ax, met, d, porc, nombre in [(axes[0], "auc", 2, False, "AUC"), (axes[1], "acierto", 0, True, "Acierto")]:
        prev = _fila(M, "V cohorte 20", "previo", 10)
        filas = [_fila(M, "V cohorte 20", "etapa1", m) for m in (10, 15, 20)]
        val = [f[met] for f in filas]
        lo = [f[f"{met}_lo"] for f in filas]
        hi = [f[f"{met}_hi"] for f in filas]
        ax.axhline(prev[met], color=GRIS, lw=1)
        ax.text(4.05, prev[met], "solo el previo", ha="right", va="bottom", fontsize=8.8, color=TINTA_2)
        ax.errorbar([0], [prev[met]], yerr=[[prev[met] - prev[f"{met}_lo"]], [prev[f"{met}_hi"] - prev[met]]],
                    fmt="o", color=GRIS, ms=7, mec="white", mew=1.5, ecolor=TINTA, elinewidth=1.1)
        ax.plot(xs[1:], val, color=MODELO)
        ax.errorbar(xs[1:], val, yerr=[np.subtract(val, lo), np.subtract(hi, val)], fmt="o", color=MODELO, ms=7,
                    mec="white", mew=1.5, ecolor=TINTA, elinewidth=1.1)
        for x, v_ in zip(xs[1:], val):
            ax.annotate(E.pct(v_) if porc else E.num(v_), (x, v_), xytext=(-8, 8), textcoords="offset points",
                        ha="right", fontsize=8.8)
        if porc:
            base = filas[0]["siempre_rojo"]
            ax.axhline(base, color=GRIS, lw=1)
            ax.text(4.05, base, "siempre rojo", ha="right", va="bottom", fontsize=8.8, color=TINTA_2)
            ax.set_ylim(0.5, 0.82)
        else:
            ax.set_ylim(0.62, 0.9)
        ax.set_xticks(xs, etq)
        ax.set_xlim(-0.4, 4.1)
        E.eje_coma(ax, "y", d, porcentaje=porc)
        E.titulos(ax, nombre, f"V Master+, las mismas {E.entero(filas[0]['n'])} partidas en los tres minutos")
    return E.guardar(fig, "05_en_vivo_por_minuto")


# Pasos intermedios de "cuánto suma lo previo en vivo", V Master+, cada minuto con sus partidas vivas.
# Fuente: PROJECT_SUMMARY.md §9.S.60 punto 6 (tabla "V Master+ (AUC)", filas "Solo el estado" y "+ expectativas por
# línea"; corrida del 17/11 sin predicciones guardadas, así que no tienen IC). El último paso (el modelo que se
# presenta) sale de data/analysis/en_vivo/metricas.csv, con IC.
SUMA_PREVIO_DOC = {
    "Solo el estado del minuto": {10: (0.7517, 0.687), 15: (0.8189, 0.737), 20: (0.8586, 0.774)},
    "+ lo que se esperaba de cada línea": {10: (0.7681, 0.695), 15: (0.8295, 0.750), 20: (0.8660, 0.782)},
}


def fig_suma_previo():
    """Cuánto suma lo que se sabía antes de la partida al modelo en vivo (AUC por minuto)."""
    M = D.metricas_en_vivo()
    pasos = list(SUMA_PREVIO_DOC) + ["+ predicción previa (modelo final)"]
    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    paso_x = 0.22
    for i, m in enumerate((10, 15, 20)):
        f = _fila(M, "V", "etapa1", m)
        vals = [SUMA_PREVIO_DOC[p][m][0] for p in pasos[:2]] + [f.auc]
        xs = [i + (j - 1) * paso_x for j in range(3)]
        ax.plot(xs, vals, color=E.GRIS_CLARO, lw=1.5, zorder=1)
        ax.plot([xs[2]] * 2, [f.auc_lo, f.auc_hi], color=TINTA, lw=1.1, zorder=2)
        for j, (x, v) in enumerate(zip(xs, vals)):
            ax.plot(x, v, "o", color=RAMPA[j], ms=9, mec="white", mew=1.5, zorder=3,
                    label=pasos[j] if i == 0 else None)
        ax.annotate(f"+{E.num(vals[2] - vals[0], 3)}", (xs[2], f.auc_hi), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=9, color=TINTA)
        ax.annotate(E.num(vals[0], 3), (xs[0], vals[0]), xytext=(0, -15), textcoords="offset points", ha="center",
                    fontsize=8.6, color=TINTA_2)
    ax.set_xticks([0, 1, 2], ["Minuto 10", "Minuto 15", "Minuto 20"])
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0.73, 0.90)
    E.eje_coma(ax, "y", 2)
    ax.set_ylabel("AUC en V")
    ax.legend(loc="upper left")
    E.titulos(ax, "Lo que se sabía antes de jugar sigue sumando en vivo",
              "V Master+, partidas vivas en cada minuto; arriba, lo que suma en total; IC 95 % solo en el modelo final")
    return E.guardar(fig, "06_cuanto_suma_lo_previo")


# =================================================================================================================
# 4 y 5. Test y robustez
# =================================================================================================================

AVISO_TEST = ("**Pendiente de la evaluación única en el test.** Esta celda se completa sola cuando exista "
              "`data/analysis/test_final/`.")


def _previo_test() -> dict | None:
    """Acierto y AUC del modelo previo en el test (semilla principal, IC por bloques de días), o None."""
    met = next((df for nombre, df in D.resultados_test().items() if nombre.startswith("metricas")), None)
    if met is None:
        return None
    f = met[(met.modelo == "previo") & (met.conjunto == "evaluacion") & (met.ic == "dias")]
    if f.empty:
        return None
    f = f.iloc[0]
    return {"acierto": (f.acierto, f.acierto_lo, f.acierto_hi), "auc": (f.auc, f.auc_lo, f.auc_hi), "n": int(f.n)}


N_V_MASTER = 16_428
RECORRIDO_PREVIO = [
    # (texto, acierto en V Master+, AUC en V Master+): §9.S.55 (modelos_train_v.csv) y §9.S.52/§9.S.75
    ("Siempre\nel rojo", 0.522, 0.500),
    ("Solo el\ndraft", 0.538, 0.5497),
    ("Draft + LP\nde la foto\ny huella", 0.556, 0.5817),
    ("Stacking de\n6 expertos", 0.614, 0.6574),
    ("Mezcla de\n2 expertos\n(MOE)", 0.617, 0.6604),
    ("Logística\nMaster+ con\ncontracruces", 0.619, 0.6674),
    ("+ Elo, oro\nesperado\ny duelos", 0.6264, 0.6740),
    ("Sin fugas\n(modelo\nfinal)", 0.6241, 0.6727),
]


def fig_recorrido_previo():
    """El recorrido del modelo previo: acierto (y AUC) de cada hito en V, y el final en el test si ya existe."""
    T = _previo_test()
    pasos = [(t, a, auc, *D.proporcion_ic(a * N_V_MASTER, N_V_MASTER)[1:], False) for t, a, auc in RECORRIDO_PREVIO]
    if T is not None:
        pasos.append(("Modelo final\nen el test", T["acierto"][0], T["auc"][0], T["acierto"][1], T["acierto"][2], True))
    fig, ax = plt.subplots(figsize=(10.2, 4.6))
    x = np.arange(len(pasos))
    colores = [GRIS] + [RAMPA[0]] * (len(RECORRIDO_PREVIO) - 2) + [MODELO] + ([RAMPA[2]] if T is not None else [])
    for i, (txt, a, auc, lo, hi, es_test) in enumerate(pasos):
        ax.bar(i, a - 0.5, bottom=0.5, width=0.62, color=colores[i])
        ax.errorbar(i, a, yerr=[[a - lo], [hi - a]], fmt="none", ecolor=TINTA, elinewidth=1, capsize=3)
        ax.text(i, hi + 0.003, E.pct(a), ha="center", va="bottom", fontsize=9.2, color=TINTA,
                fontweight="semibold" if i >= len(RECORRIDO_PREVIO) - 1 else "normal")
        if auc > 0.5:
            ax.text(i, 0.503, f"AUC\n{E.num(auc, 3)}", ha="center", va="bottom", fontsize=7.8,
                    color="white" if i else TINTA_2)
        if 0 < i and not es_test:
            d = a - pasos[i - 1][1]
            ax.text(i - 0.5, max(a, pasos[i - 1][1]) + 0.016, f"{'+' if d >= 0 else '−'}{E.num(abs(d) * 100, 1)}",
                    ha="center", fontsize=8, color=TINTA_2)
    if T is not None:
        ax.axvline(len(RECORRIDO_PREVIO) - 0.5, color=E.EJE, lw=1, ls=":")
    ax.set_xticks(x, [p[0] for p in pasos], fontsize=8.6)
    ax.set_ylim(0.5, 0.665)
    ax.set_xlim(-0.6, len(pasos) - 0.4)
    E.eje_coma(ax, "y", 0, porcentaje=True)
    ax.set_ylabel("Acierto (IC 95 %)")
    ax.grid(axis="x", visible=False)
    sub = ("V Master+ (16.428 partidas) hasta el modelo final; " +
           ("la última barra es el test, con IC por bloques de días" if T is not None else
            "el test se agrega cuando se corra la evaluación única") + "; entre barras, puntos de acierto")
    E.titulos(ax, "Del «siempre el rojo» al modelo final: lo que más suma es saber quién juega", sub)
    return E.guardar(fig, "04b_recorrido_previo")


DRAFTREC = {"paper": (0.5535, 28_690), "nuestro_en_su_dataset": (0.5927, 28_690)}


def fig_draftrec():
    """Contra DraftRec (Lee et al., WWW 2022) en su propio test: el paper y nuestra metodología (§9.S.59), las dos
    medidas sobre las mismas 28.690 partidas. No se agrega nada estimado ni medido en otro dataset."""
    pasos = [  # (texto, acierto, lo, hi, color)
        ("DraftRec\n(paper)", *_binom(*DRAFTREC["paper"]), GRIS),
        ("Nuestro modelo previo\n(con lo que el\ndataset permite)", *_binom(*DRAFTREC["nuestro_en_su_dataset"]),
         MODELO),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for i, (txt, a, lo, hi, color) in enumerate(pasos):
        ax.bar(i, a - 0.5, bottom=0.5, width=0.5, color=color)
        ax.errorbar(i, a, yerr=[[a - lo], [hi - a]], fmt="none", ecolor=TINTA, elinewidth=1, capsize=3)
        ax.text(i, hi + 0.003, E.pct(a), ha="center", va="bottom", fontsize=9.5, color=TINTA)
    ax.set_xticks(range(len(pasos)), [p[0] for p in pasos], fontsize=9)
    ax.set_xlim(-0.6, len(pasos) - 0.4)
    ax.set_ylim(0.5, 0.62)
    E.eje_coma(ax, "y", 0, porcentaje=True)
    ax.set_ylabel("Acierto (IC 95 %)")
    ax.grid(axis="x", visible=False)
    d = (DRAFTREC["nuestro_en_su_dataset"][0] - DRAFTREC["paper"][0]) * 100
    E.titulos(ax, f"Contra DraftRec, en sus propios datos: +{E.num(d, 1)} puntos",
              f"Test de DraftRec, las mismas {E.entero(DRAFTREC['paper'][1])} partidas; IC binomial")
    return E.guardar(fig, "04c_draftrec")


def _binom(p: float, n: int) -> tuple[float, float, float]:
    return D.proporcion_ic(p * n, n)


def seccion_test():
    """Resultado en el test si ya se corrió la evaluación única; si no, un aviso."""
    R = D.resultados_test()
    met = next((df for nombre, df in R.items() if nombre.startswith("metricas")), None)
    if met is None:
        return Markdown(AVISO_TEST)
    cols = [c for c in met.columns if c in {"modelo", "minuto", "conjunto", "n"} or
            c.split("_")[0] in {"auc", "acierto", "log", "ece", "siempre"}]
    return _tabla(met[cols].round(4))


def _filas_test_vs_v() -> list[tuple[str, pd.Series, pd.Series]] | None:
    """(etiqueta, fila de V, fila del test) para el previo y el en vivo a 10/15/20, todos en la cohorte de las partidas
    vivas al minuto 20 (la misma definición en los dos conjuntos); None si todavía no hay evaluación del test."""
    met = D.resultados_test().get("metricas")
    if met is None:
        return None
    M = D.metricas_en_vivo()
    t = met[(met.conjunto == "cohorte 20") & (met.ic == "dias")]

    def de_test(modelo, m):
        f = t[(t.modelo == modelo) & (pd.to_numeric(t.minuto, errors="coerce") == m)]
        assert len(f) == 1, (modelo, m)
        return f.iloc[0]
    filas = [("Antes de\nla partida", _fila(M, "V cohorte 20", "previo", 10), de_test("previo_hacia_adelante", 10))]
    filas += [(f"Minuto {m}", _fila(M, "V cohorte 20", "etapa1", m), de_test("en_vivo", m)) for m in (10, 15, 20)]
    return filas


def fig_test_vs_v():
    """AUC y acierto en V y en el test, lado a lado, con la misma cohorte en los dos: ¿se sostiene lo de V?"""
    filas = _filas_test_vs_v()
    if filas is None:
        return Markdown(AVISO_TEST)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    x = np.arange(len(filas))
    dx = 0.2
    for ax, met, d, porc, nombre in [(axes[0], "auc", 2, False, "AUC"), (axes[1], "acierto", 0, True, "Acierto")]:
        for k, (lado, color, etq) in enumerate([(-dx, RAMPA[0], "Validación (V)"), (dx, MODELO, "Test")]):
            v = np.array([f[k + 1][met] for f in filas])
            lo = np.array([f[k + 1][f"{met}_lo"] for f in filas])
            hi = np.array([f[k + 1][f"{met}_hi"] for f in filas])
            ax.errorbar(x + lado, v, yerr=[v - lo, hi - v], fmt="o", color=color, ms=8, mec="white", mew=1.5,
                        ecolor=TINTA, elinewidth=1, capsize=0, label=etq, zorder=3)
            for xi, vi, hii in zip(x + lado, v, hi):               # el número arriba de cada punto
                ax.annotate(E.pct(vi) if porc else E.num(vi, 3), (xi, hii), xytext=(0, 7), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8.4, color=TINTA if k else TINTA_2)
        if porc:
            base = filas[0][2]["siempre_rojo"]
            ax.axhline(base, color=GRIS, lw=1)
            ax.text(len(filas) - 0.45, base, "siempre rojo (test)", ha="right", va="bottom", fontsize=8.6,
                    color=TINTA_2)
            ax.set_ylim(0.5, 0.82)
        else:
            ax.set_ylim(0.62, 0.9)
        ax.set_xticks(x, [f[0] for f in filas])
        ax.set_xlim(-0.5, len(filas) - 0.4)
        ax.grid(axis="x", visible=False)
        E.eje_coma(ax, "y", d, porcentaje=porc)
        ax.set_title(nombre, loc="left", fontsize=10.5, pad=14)
    axes[0].set_ylabel("IC 95 %")
    axes[0].legend(loc="upper left", frameon=False, handletextpad=0.3)
    nv, nt = int(filas[0][1]["n"]), int(filas[0][2]["n"])
    fig.suptitle("En el test, los dos modelos rinden lo mismo que en V\n", x=0.01, ha="left", fontsize=11.5,
                 fontweight="semibold")
    fig.text(0.01, 0.93, f"Partidas vivas al minuto 20 de cada conjunto ({E.entero(nv)} en V, {E.entero(nt)} en el "
             "test). IC 95 %: bootstrap de días en el test y de partidas en V; casi todos son más chicos que el punto",
             ha="left", va="top", fontsize=9.2, color=TINTA_2, in_layout=False)
    return E.guardar(fig, "04a_test_contra_v")


def _auc_ventanas(df: pd.DataFrame, k: int, b: int = 500):
    """AUC por ventana cronológica de igual tamaño, con IC, y pendiente por día con IC (bootstrap por ventana)."""
    df = df.sort_values("t").reset_index(drop=True)
    corte = np.array_split(np.arange(len(df)), k)
    centro, auc, reps = [], [], []
    for i, idx in enumerate(corte):
        s = df.iloc[idx]
        y, p = s.y.values.astype(float), s.p.values
        centro.append((s.t.mean() - df.t.min()).total_seconds() / 86400)
        auc.append(D.auc_ponderado(y, p, np.ones((1, len(y))))[0])
        reps.append(D.auc_ponderado(y, p, D.pesos_bootstrap(len(y), b, semilla=i)))
    reps = np.array(reps)                                    # (k, b)
    x = np.array(centro) - np.mean(centro)
    pend = (reps * x[:, None]).sum(0) / (x ** 2).sum()
    p0 = float((np.array(auc) * x).sum() / (x ** 2).sum())
    return (np.array(centro), np.array(auc), np.percentile(reps, 2.5, 1), np.percentile(reps, 97.5, 1),
            (p0, *np.percentile(pend, [2.5, 97.5])))


def fig_degradacion_v(k: int = 4):
    """¿Empeora el modelo a medida que V se aleja del ajuste? AUC por ventana cronológica de V."""
    series = [("Antes de la partida", D.prediccion_previo_v())] + \
             [(f"En vivo, minuto {m}", D.prediccion_en_vivo(m)) for m in (10, 15, 20)]
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 3.4))
    for ax, (nombre, df) in zip(axes, series):
        c, a, lo, hi, (p0, plo, phi) = _auc_ventanas(df, k)
        ax.plot(c, a, color=MODELO)
        ax.errorbar(c, a, yerr=[a - lo, hi - a], fmt="o", color=MODELO, ms=6.5, mec="white", mew=1.4,
                    ecolor=TINTA, elinewidth=1)
        ax.set_ylim(np.mean(a) - 0.045, np.mean(a) + 0.045)
        E.eje_coma(ax, "y", 2)
        ax.set_xticks([0, 2, 4, 6, 8, 10])
        ax.set_xlim(-0.5, 11)
        ax.set_xlabel("Días desde el 16/09")
        ax.set_title(nombre, fontsize=10.5, pad=18)
        ax.text(0, 1.01, f"pendiente {E.num(p0 * 10, 3)} [{E.num(plo * 10, 3)}; {E.num(phi * 10, 3)}] cada 10 días",
                transform=ax.transAxes, fontsize=8.2, color=TINTA_2, va="bottom")
    axes[0].set_ylabel("AUC (IC 95 %)")
    fig.suptitle("Estabilidad dentro de V: AUC en cuatro ventanas cronológicas del mismo tamaño",
                 x=0.01, ha="left", fontsize=11.5, fontweight="semibold")
    return E.guardar(fig, "07_degradacion_en_v")


def seccion_robustez_test():
    """Degradación y semillas en el test, si ya existen; si no, un aviso."""
    R = D.resultados_test()
    partes = [df for nombre, df in R.items() if nombre.startswith(("degradacion", "semillas"))]
    if not partes:
        return Markdown("**Pendiente de la evaluación única:** degradación por ventanas del test y variación "
                        "entre las 5 semillas fijadas de antemano.")
    return _tabla(pd.concat(partes, keys=[n for n in R if n.startswith(("degradacion", "semillas"))]).round(4))


# =================================================================================================================
# 6. Huella de estilo y matchmaking
# =================================================================================================================

def fig_huella():
    """Reconocer a un jugador no visto por cómo juega, con campeones distintos: AUC según cuántas partidas."""
    red = pd.read_csv(D.HUELLA / "mejoras_red_habitos.csv")
    red = red[red.muestra == "campeones_distintos"].sort_values("k")
    lda = pd.read_csv(D.HUELLA / "mejoras_parte2.csv")
    lda = lda[(lda.muestra == "campeones_distintos") & (lda.variante == "estadísticas + hábitos · lda")]
    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    ax.fill_between(red.k, red.lo, red.hi, color=MODELO, alpha=0.12, lw=0)
    ax.plot(red.k, red.auc, color=MODELO, label="Red contrastiva")
    ax.plot(red.k, red.auc, "o", color=MODELO, ms=7, mec="white", mew=1.5)
    ax.errorbar(lda.k, lda.auc, yerr=[lda.auc - lda.lo, lda.hi - lda.auc], fmt="o", color=GRIS, ms=7, mec="white",
                mew=1.5, ecolor=GRIS, elinewidth=1.1, label="Discriminante lineal (referencia)")
    for _, f in red.iterrows():
        ax.annotate(E.num(f.auc, 3), (f.k, f.auc), xytext=(0, -16), textcoords="offset points", ha="center",
                    fontsize=8.8)
    ax.set_xticks(red.k)
    ax.set_xlabel("Partidas de cada lado de la comparación")
    ax.set_ylabel("AUC (IC 95 %)")
    ax.set_ylim(0.84, 1.0)
    E.eje_coma(ax, "y", 2)
    ax.legend(loc="lower right")
    E.titulos(ax, "La huella de estilo reconoce a jugadores que el modelo nunca vio",
              "¿Son la misma persona? Dos grupos de partidas con campeones distintos; IC por bootstrap de jugadores")
    return E.guardar(fig, "08_huella_de_estilo")


def fig_matchmaking():
    """Desbalance |p − 0,5| real contra el de reacomodar a los mismos 10 jugadores (simulación bajo el modelo)."""
    s = pd.read_parquet(D.SWAPS, columns=["p_a_real", "p_b_real", "p_a_indep_ab_pareto_t20",
                                          "p_b_indep_ab_pareto_t20"])
    # Cada mitad del ajuste evalúa la configuración que eligió la otra (sesgo de optimización, §9.S.67 punto 7).
    real = 100 * (np.abs(s.p_a_real - 0.5) + np.abs(s.p_b_real - 0.5)).values / 2
    nuevo = 100 * (np.abs(s.p_a_indep_ab_pareto_t20 - 0.5) + np.abs(s.p_b_indep_ab_pareto_t20 - 0.5)).values / 2
    W = D.pesos_bootstrap(len(real), 1000)
    ic = lambda x: (x.mean(), *np.percentile(W @ x / W.sum(1), [2.5, 97.5]))  # noqa: E731
    r, n = ic(real), ic(nuevo)
    bins = np.linspace(0, 30, 31)
    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    for x, color, txt in [(real, GRIS, f"Como las armó el juego: {E.ic(*r, d=2)} pp"),
                          (nuevo, MODELO, f"Reacomodadas: {E.ic(*n, d=2)} pp")]:
        h, _ = np.histogram(x, bins=bins)
        h = h / len(x)
        ax.fill_between(bins[:-1], h, step="post", color=color, alpha=0.12, lw=0)
        ax.step(bins[:-1], h, where="post", color=color, lw=2, label=txt)
    ax.set_xlim(0, 30)
    E.eje_coma(ax, "y", 0, porcentaje=True)
    ax.set_xlabel("Desbalance |P(gana azul) − 50 %|, en puntos")
    ax.set_ylabel("Partidas")
    ax.legend(loc="upper right", title="Desbalance medio [IC 95 %]", title_fontsize=9.5, alignment="left")
    E.titulos(ax, "Simulación: los mismos 10 jugadores, mejor repartidos, dan partidas más parejas",
              "V Master+; nadie juega un rol que juega menos que en la partida real; lo evalúa un modelo que no eligió")
    return E.guardar(fig, "09_matchmaking")
