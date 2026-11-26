"""Estilo único de las figuras del informe: paleta, tipografía, ejes y números en castellano.

Paleta (validada con el validador de la skill `dataviz`, fondo blanco):

* **AZUL / ROJO**: solo para los lados del mapa (lado azul / lado rojo). ΔE con daltonismo 21,6.
* **GRIS**: referencias y líneas base (azar, "siempre rojo", el previo solo).
* **MODELO**: violeta, la serie principal cuando no se habla de un lado. Con AZUL y ROJO pasa todos los
  chequeos (peor par adyacente ΔE 31,5 sin daltonismo).
* **RAMPA**: violeta ordinal claro → oscuro para pasos que se acumulan (pasa el chequeo ordinal; el extremo claro
  tiene contraste 2,37:1 con el fondo).

Marcas: líneas de 2 px, puntos ≥ 8 px con aro blanco, barras finas, grilla horizontal en un gris muy claro y
sólida, sin borde superior ni derecho. El texto nunca va en el color de la serie.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

AZUL = "#2a78d6"
ROJO = "#e34948"
MODELO = "#6250d6"
GRIS = "#898781"
GRIS_CLARO = "#d9d8d2"
RAMPA = ["#a79ff0", "#8174e3", "#4a3aa7"]

TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
GRILLA = "#e9e8e3"
EJE = "#c3c2b7"
FONDO = "#ffffff"

REPO = Path(__file__).resolve().parents[2]
CARPETA_PNG = REPO / "notebooks" / "figuras" / "informe"


def aplicar() -> None:
    """Fija los rcParams de matplotlib para todas las figuras del informe."""
    mpl.rcParams.update({
        "figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
        "font.size": 10.5, "axes.titlesize": 11.5, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.titlepad": 10, "axes.labelsize": 10, "axes.labelcolor": TINTA_2,
        "axes.edgecolor": EJE, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRILLA, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.axisbelow": True, "xtick.color": TINTA_2, "ytick.color": TINTA_2, "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5, "xtick.major.size": 0, "ytick.major.size": 0, "text.color": TINTA,
        "legend.frameon": False, "legend.fontsize": 9.5, "lines.linewidth": 2, "lines.solid_capstyle": "round",
        "lines.markersize": 7, "axes.formatter.use_mathtext": False,
        "mathtext.fontset": "dejavusans", "figure.constrained_layout.use": True,
    })


# --- números en castellano -------------------------------------------------------------------------------------

def num(x: float, d: int = 4) -> str:
    """0.67267 -> '0,6727' (coma decimal, punto de miles)."""
    s = f"{x:,.{d}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def pct(x: float, d: int = 1) -> str:
    """0.6241 -> '62,4 %'."""
    return f"{num(100 * x, d)} %"


def entero(n: int) -> str:
    """187817 -> '187.817'."""
    return f"{int(n):,}".replace(",", ".")


def ic(v: float, lo: float, hi: float, d: int = 4, porcentaje: bool = False) -> str:
    """'0,6727 [0,6644; 0,6810]' o '62,4 % [61,7; 63,2]'."""
    if porcentaje:
        return f"{num(100 * v, d)} % [{num(100 * lo, d)}; {num(100 * hi, d)}]"
    return f"{num(v, d)} [{num(lo, d)}; {num(hi, d)}]"


def eje_coma(ax, eje: str = "y", d: int = 2, porcentaje: bool = False) -> None:
    """Formatea las marcas del eje con coma decimal (y '%' si `porcentaje`, con valores en 0-1)."""
    if porcentaje:
        f = FuncFormatter(lambda v, _: f"{num(100 * v, d)} %")
    else:
        f = FuncFormatter(lambda v, _: num(v, d))
    (ax.yaxis if eje == "y" else ax.xaxis).set_major_formatter(f)


def titulos(ax, titulo: str, sub: str | None = None) -> None:
    """Título del panel y, debajo, una línea gris con el detalle."""
    ax.set_title(titulo, pad=20 if sub else 10)
    if sub:
        ax.text(0, 1.015, sub, transform=ax.transAxes, ha="left", va="bottom", fontsize=9.2, color=TINTA_2)


def guardar(fig, nombre: str):
    """Guarda la figura en `notebooks/figuras/informe/<nombre>.png`, la cierra y la devuelve (Jupyter la muestra
    una sola vez como valor de la celda)."""
    CARPETA_PNG.mkdir(parents=True, exist_ok=True)
    fig.savefig(CARPETA_PNG / f"{nombre}.png")
    plt.close(fig)
    return fig
