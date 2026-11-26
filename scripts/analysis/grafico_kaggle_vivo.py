"""
Comparación con Kaggle en la predicción en vivo al minuto 10 (pedido del usuario, 2026-11-18, para mostrar en
clase; mismo estilo que `grafico_draftrec_puente.py`). Números provistos por el usuario (acierto e IC 95 %).

* Receta simple = logística con 6 diferencias azul − rojo al minuto 10 (oro, XP, CS, kills, muertes,
  asistencias).
* Kaggle: 9.879 partidas (Diamante I-Master, 2020), validación cruzada de 10 pliegues.
* LAS: entrenada con 117.404 partidas Master+ anteriores al 16/09 y evaluada en V Master+ (16.331 partidas).
* Nuestro modelo en vivo completo: §9.S.60, minuto 10.

Color = dónde se evalúa (verde Kaggle, azul LAS; azul oscuro para nuestro modelo completo); rayado = entrenado
en el otro dataset.

    python scripts/analysis/grafico_kaggle_vivo.py     # → data/analysis/kaggle_vivo.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[2]
SALIDA = REPO / "data" / "analysis" / "kaggle_vivo.png"

VERDE, AZUL, AZUL_OSC = "#8fd19e", "#5b9be6", "#1f5fb4"
INK, INK_2, MUTED, FONDO = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"

BARRAS = [  # (etiqueta, acierto, ± IC 95 %, color, rayado)
    ("Receta simple\nentrenada y evaluada en Kaggle", 72.7, 0.9, VERDE, False),
    ("Receta simple\nentrenada en LAS, evaluada en Kaggle", 72.6, 0.9, VERDE, True),
    ("Receta simple\nentrenada y evaluada en LAS", 68.1, 0.7, AZUL, False),
    ("Nuestro modelo en vivo\ncompleto (LAS)", 71.6, 0.7, AZUL_OSC, False),
]
SALTOS = ["−0,1 pts\nel modelo transfiere",
          "−4,5 pts\nmismo modelo: el minuto 10\nde hoy predice peor",
          "+3,5 pts [+2,9; +4,1]\nvariables por línea,\nhistorial y ensamble"]
PIE = ("Receta simple = logística con 6 diferencias azul − rojo al minuto 10 (oro, XP, CS, kills, muertes, "
       "asistencias). Kaggle: 9.879 partidas, validación cruzada de 10 pliegues.\nLAS: entrenada con 117.404 "
       "partidas Master+ anteriores al 16/09 y evaluada en V Master+ (16.331 partidas). Barras de error: IC 95 % "
       "del acierto. El eje empieza en 60 %.")


def main():
    sns.set_theme(context="notebook", style="whitegrid")
    plt.rcParams.update({
        "figure.facecolor": FONDO, "axes.facecolor": FONDO, "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 13,
        "axes.spines.top": False, "axes.spines.right": False, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
        "axes.axisbelow": True, "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
    })
    fig, ax = plt.subplots(figsize=(12.5, 7))
    x = list(range(len(BARRAS)))
    vals = [b[1] for b in BARRAS]
    semi = [b[2] for b in BARRAS]
    sns.barplot(x=x, y=vals, hue=x, palette=[b[3] for b in BARRAS], width=0.56, legend=False,
                edgecolor=FONDO, linewidth=2, ax=ax)
    for i, (_, v, h, _, rayado) in enumerate(BARRAS):
        if rayado:
            ax.patches[i].set_hatch("///")
            ax.patches[i].set_edgecolor("white")
        ax.errorbar(i, v, yerr=h, fmt="none", ecolor=INK_2, elinewidth=1.6, capsize=7, zorder=5)
        ax.text(i, v + h + 0.15, f"{v:.1f} %  (± {h:.1f})".replace(".", ","), ha="center", va="bottom",
                fontsize=11.5, fontweight="bold", color=INK)

    # escalera entre barras y saltos anotados
    for i in range(len(BARRAS) - 1):
        ax.plot([i + 0.28, i + 1 - 0.28], [vals[i], vals[i]], color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
        ax.annotate("", xy=(i + 0.72, vals[i + 1]), xytext=(i + 0.72, vals[i]),
                    arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=1.4, shrinkA=0, shrinkB=0))
        tope = max(vals[i] + semi[i], vals[i + 1] + semi[i + 1])
        ax.text(i + 0.5, tope + 0.9, SALTOS[i], ha="center", va="bottom", fontsize=9, color=INK_2,
                linespacing=1.15, zorder=6, bbox=dict(facecolor=FONDO, edgecolor="none", pad=2))

    # separación entre lo evaluado en Kaggle y lo evaluado en LAS
    ax.axvline(1.5, color="#c3c2b7", lw=1, ls=(0, (4, 3)), zorder=0)
    ax.text(1.45, 79.6, "evaluado en Kaggle", ha="right", va="top", fontsize=10, fontweight="bold", color=INK_2)
    ax.text(1.55, 79.6, "evaluado en LAS", ha="left", va="top", fontsize=10, fontweight="bold", color=INK_2)

    ax.set_ylim(60, 80)
    ax.set_yticks(range(60, 81, 2))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f} %"))
    ax.set_ylabel("Acierto (desde 60 %)")
    ax.set_xticks(x, [b[0] for b in BARRAS], fontsize=9.5)
    ax.set_xlabel("")
    ax.grid(axis="x", visible=False)
    ax.set_title("Predicción del ganador en vivo (minuto 10): Kaggle contra nuestro modelo", loc="left")
    ax.legend(handles=[Patch(facecolor=VERDE, label="evaluado en Kaggle (Diamante I–Master, 2020)"),
                       Patch(facecolor=AZUL, label="evaluado en LAS (Master+, 2026)"),
                       Patch(facecolor="white", edgecolor=INK_2, hatch="///", label="entrenado en el otro dataset")],
              loc="upper right", frameon=True, facecolor=FONDO, edgecolor="#e1e0d9", fontsize=9)
    fig.text(0.01, 0.01, PIE, fontsize=8, color=MUTED, linespacing=1.3)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SALIDA, dpi=150)
    print(SALIDA)


if __name__ == "__main__":
    main()
