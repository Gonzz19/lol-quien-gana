"""
Puente de acierto: DraftRec → nuestro mejor modelo en el dataset de DraftRec → nuestro mejor modelo con solo la
información de DraftRec en nuestros datos (estimación) → nuestro mejor modelo completo (pedido del usuario,
2026-11-18, para mostrar en clase; números de PROJECT_SUMMARY §9.S.52 y §9.S.59).

* DraftRec 55,35 %: tabla de test del paper (Lee et al., WWW 2022), en su dataset y su partición.
* Escalón 9 en el dataset de DraftRec 59,27 %: `draftrec_mejor.py`, mismo test.
* Escalón 9 en nuestra V Master+ 62,5 %: §9.S.52.
* Estimación de lo que cuesta sacarle al escalón 9 lo que DraftRec no tiene (LP / mapa, fechas, rango, W/L de
  temporada): en el híbrido costó 1,0 punto (61,1 → 60,1 %, `ablacion_draftrec.py`); el escalón 9 usa más de esas
  variables, así que se extrapola a 1,5-2,0 puntos → 60,5-61,0 % (punto medio ~60,7 %). No es una medición.

    python scripts/analysis/grafico_draftrec_puente.py     # → data/analysis/draftrec_puente.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

REPO = Path(__file__).resolve().parents[2]
SALIDA = REPO / "data" / "analysis" / "draftrec_puente.png"

AZUL, AZUL_CLARO, GRIS = "#2a78d6", "#8fb6ea", "#898781"
INK, INK_2, MUTED, FONDO = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"

EST_LO, EST_HI = 60.5, 61.0
EST = (EST_LO + EST_HI) / 2
N_DRAFTREC, N_V = 28_690, 16_428   # partidas del test de DraftRec y de V Master+
PASOS = [  # (etiqueta, acierto, color, es estimación, partidas del test)
    ("DraftRec\n(paper, su dataset)", 55.35, GRIS, False, N_DRAFTREC),
    ("Nuestro mejor modelo\nen el dataset de DraftRec", 59.27, AZUL, False, N_DRAFTREC),
    ("Nuestro mejor modelo\nsolo con la información\nde DraftRec, en nuestros datos", EST, AZUL_CLARO, True, N_V),
    ("Nuestro mejor modelo\n(V Master+, completo)", 62.5, AZUL, False, N_V),
]


def ic95(acierto, n):
    """Semiancho del IC 95 % binomial del acierto, en puntos."""
    p = acierto / 100
    return 196 * (p * (1 - p) / n) ** 0.5
SALTOS = ["+3,9 pts\nmetodología\n(mismos datos)", "~+1,5 pts\npoblación\n(LAS Master+ 2026 contra\nDiamond 2+ 2021)",
          "~+1,8 pts\nvariables que DraftRec no tiene\n(LP y mapa, fechas, rango)"]


def main():
    sns.set_theme(context="notebook", style="whitegrid")
    plt.rcParams.update({
        "figure.facecolor": FONDO, "axes.facecolor": FONDO, "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 13,
        "axes.spines.top": False, "axes.spines.right": False, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
        "axes.axisbelow": True, "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
    })
    fig, ax = plt.subplots(figsize=(12, 6.4))
    x = list(range(len(PASOS)))
    vals = [p[1] for p in PASOS]
    sns.barplot(x=x, y=vals, hue=x, palette=[p[2] for p in PASOS], width=0.56, legend=False,
                edgecolor=FONDO, linewidth=2, ax=ax)
    est = [i for i, p in enumerate(PASOS) if p[3]][0]
    ax.patches[est].set_hatch("///")
    ax.patches[est].set_edgecolor("white")
    # IC 95 % de cada barra: binomial del acierto; en la estimación, además, el rango de la extrapolación
    semi = []
    for i, (_, v, _, es_est, n) in enumerate(PASOS):
        h = ic95(v, n) + ((EST_HI - EST_LO) / 2 if es_est else 0)
        semi.append(h)
        ax.errorbar(i, v, yerr=h, fmt="none", ecolor=INK_2, elinewidth=1.6, capsize=7, zorder=5)

    # escalera entre barras y saltos anotados
    for i in range(len(PASOS) - 1):
        ax.plot([i + 0.28, i + 1 - 0.28], [vals[i], vals[i]], color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
        ax.annotate("", xy=(i + 0.72, vals[i + 1]), xytext=(i + 0.72, vals[i]),
                    arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=1.4, shrinkA=0, shrinkB=0))
        ax.text(i + 0.5, max(vals[i] + semi[i], vals[i + 1] + semi[i + 1]) + 0.45, SALTOS[i], ha="center", va="bottom", fontsize=9,
                color=INK_2, linespacing=1.15)
    for i, (_, v, _, es_est, _) in enumerate(PASOS):
        txt = (f"~{v:.1f} %" if es_est else f"{v:.1f} %").replace(".", ",")
        ax.text(i, v + semi[i] + 0.15, f"{txt}  (± {semi[i]:.1f})".replace("± 0.", "± 0,").replace("± 1.", "± 1,"),
                ha="center", va="bottom", fontsize=11.5, fontweight="bold", color=INK)
        if es_est:
            ax.text(i, 50.35, "estimación\n(60,5-61,0 % ± muestreo)", ha="center", va="bottom", fontsize=8.5,
                    color=INK, style="italic")

    ax.set_ylim(50, 66)
    ax.set_yticks(range(50, 67, 2))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f} %"))
    ax.set_ylabel("Acierto en test (desde 50 %)")
    ax.set_xticks(x, [p[0] for p in PASOS], fontsize=9.5)
    ax.set_xlabel("")
    ax.grid(axis="x", visible=False)
    ax.set_title("Predicción del ganador antes de la partida: de DraftRec a nuestro mejor modelo", loc="left")
    fig.text(0.01, 0.01, "DraftRec: Lee et al., WWW 2022 (tabla de test). Nuestro mejor modelo = escalón 9. "
             "Barras de error: IC 95 % binomial del acierto\n(test de DraftRec 28.690 partidas; V Master+ "
             "16.428). La barra rayada es una extrapolación de la ablación del híbrido (−1,0 pt), no una medición: "
             "su intervalo suma el rango de la extrapolación al muestreo.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SALIDA, dpi=150)
    print(SALIDA)


if __name__ == "__main__":
    main()
