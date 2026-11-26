"""
Genera `notebooks/informe_final.ipynb`: el notebook único de la entrega (issue #6).

Poco texto, poco código a la vista: cada celda de código es una llamada a `src/informe/figuras.py`, con la entrada
oculta (metadata `jupyter.source_hidden` y tag `hide-input`). Las figuras leen resultados ya guardados en `data/`
(no se reentrena nada) y además se exportan a `notebooks/figuras/informe/*.png`. El test no se carga: sus conteos
salen del manifiesto, y sus resultados aparecen solos cuando exista `data/analysis/test_final/`
(`scripts/analysis/evaluacion_test.py`). Decisiones de diseño en `PROJECT_SUMMARY.md` §9.S.81.

    python scripts/analysis/build_informe_nb.py            # escribe el notebook
    python scripts/analysis/build_informe_nb.py --ejecutar # y lo ejecuta (~1 min)
"""

import argparse
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "notebooks" / "informe_final.ipynb"

OCULTA = {"jupyter": {"source_hidden": True}, "tags": ["hide-input"]}
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip(), metadata=OCULTA))  # noqa: E731

md(r"""
# ¿Quién gana? Antes y durante una partida de League of Legends

Partidas clasificatorias de Diamante para arriba en el servidor de Latinoamérica Sur. Se predice la probabilidad de
que gane el **lado azul** con dos modelos: uno **antes de la partida** (campeones elegidos e historial de los diez
jugadores) y otro **en vivo**, a los minutos 10, 15 y 20.

Todas las métricas se muestran con su intervalo de confianza del 95 % (IC), por bootstrap de partidas salvo que se
indique otra cosa.
""")

code(r"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path.cwd().resolve()
ROOT = ROOT.parent if ROOT.name == "notebooks" else ROOT
sys.path.insert(0, str(ROOT))
from src.informe import figuras as F
""")

# ---------------------------------------------------------------------------------------------------------------
md(r"""
## 1. La pregunta y los datos

Cada modelo se ajusta con partidas anteriores a las que predice: el tiempo es la única partición. Las variantes se
eligen con bloques que avanzan hacia adelante y con la ventana de validación **V**; el **test** queda cerrado hasta
la evaluación final.
""")
code("F.tabla_datos()")
code("F.fig_linea_tiempo()")

# ---------------------------------------------------------------------------------------------------------------
md(r"""
## 2. Antes de la partida

Una regresión logística sobre **diferencias entre equipos**: cada variable entra como azul − rojo.

$$\operatorname{logit} P(\text{gana azul}) = \beta_0 + \sum_k \beta_k \left(x_k^{\text{azul}} - x_k^{\text{rojo}}\right)$$

**Por qué así.** Si se intercambian los lados, todas las diferencias cambian de signo y la predicción pasa a
$\sigma(\beta_0 - \sum_k \beta_k \Delta x_k)$: sin $\beta_0$ el modelo sería exactamente antisimétrico,
$P(a, b) = 1 - P(b, a)$. Toda la ventaja del lado rojo queda en un solo número, $\beta_0$, en lugar de contaminar los
coeficientes de las variables. La penalización L2 mantiene estables las ~1.500 columnas (los campeones por línea son
la mayoría).
""")
code("F.respuesta_previo()")
code("F.tabla_previo()")
md(r"""
**Qué aporta cada grupo.** El salto más grande lo da conocer al jugador *con ese campeón y en esa línea*; el resto
suma de a milésimas, y cada paso se aceptó solo si mejoraba en V y en los bloques hacia adelante. Las dos últimas
filas no agregan información: corrigen datos que en el ajuste sabían algo del futuro, y por eso restan.
""")
code("F.fig_aportes_previo()")
md(r"""
**Qué tan confiable es cada probabilidad.** El AUC es la probabilidad de que el modelo ordene bien un par al azar,
$\text{AUC} = P(\hat p_i > \hat p_j \mid i \text{ ganó el azul},\ j \text{ ganó el rojo})$. Las predicciones están
calibradas (cuando dice 70 %, el azul gana cerca del 70 %), y en las partidas donde está más seguro acierta mucho más
que el promedio.
""")
code("F.fig_separacion_previo()")
code("F.fig_calibracion_cobertura()")

# ---------------------------------------------------------------------------------------------------------------
md(r"""
## 3. Durante la partida

Al minuto $m$ se suma el estado de la partida, $\mathbf x_m$ (oro, experiencia, nivel, súbditos, daño, asesinatos,
muertes y asistencias por línea, azul − rojo, y la *sorpresa* de cada línea: lo que lleva menos lo que se esperaba
antes de jugar), al logit del modelo previo, $L$:

$$\hat P_m = \sigma\!\left(\beta_0 + \beta_L\, L + \boldsymbol\beta^\top \mathbf x_m\right)$$

Se ajusta solo con partidas Master+ anteriores a V, una regresión por minuto, con las partidas que siguen vivas en
ese minuto.
""")
code("F.respuesta_en_vivo()")
code("F.tabla_en_vivo()")
code("F.fig_en_vivo_minutos()")
md(r"""
**Cuánto suma lo previo.** Aun viendo el oro y las bajas, saber quiénes juegan y qué eligieron sigue sumando, cada
vez menos a medida que la partida avanza. Los dos primeros puntos de cada minuto vienen de una medición anterior que
no guardó las predicciones, por eso no tienen IC.
""")
code("F.fig_suma_previo()")

# ---------------------------------------------------------------------------------------------------------------
md(r"""
## 4. Resultado en el test

Los dos modelos, congelados, se evalúan **una sola vez** en el test (desde el 27/09): AUC, acierto contra «siempre
rojo», log loss y error de calibración, con IC por bootstrap por bloques de días.
""")
code("F.seccion_test()")
md(r"""
**¿Se sostiene lo que se midió en V?** Con la misma definición en los dos conjuntos (las partidas que siguen vivas al
minuto 20), el test da lo mismo que V o apenas más, en los cuatro momentos: no hubo sobreajuste a V por haber elegido
el modelo mirándola. En esta cohorte el previo da 0,668; sobre todas las partidas del test, 0,6765 (la tabla de arriba).
""")
code("F.fig_test_vs_v()")
md(r"""
**El recorrido.** Cada barra es un hito del modelo previo, medido en la misma V Master+; la última, el modelo final
en el test. El salto grande es pasar de saber qué campeones se eligieron a saber quién juega (+5,8 puntos con el
stacking); de ahí en adelante cada paso suma décimas. Las dos correcciones de fuga cuestan 0,2 puntos.
""")
code("F.fig_recorrido_previo()")
md(r"""
**Contra un trabajo publicado.** DraftRec (Lee et al., WWW 2022) predice el ganador en Master+ y Diamante con
el draft y el historial de cada jugador. Corrimos nuestro modelo previo (con todo lo que su dataset
permite reconstruir: variables por línea, rating causal, contracruces y sinergias) sobre su misma partición, y en
las mismas 28.690 partidas de su test acierta 3,9 puntos más. Su dataset no tiene LP, fechas, rango ni W/L de la
temporada, así que esa comparación deja afuera parte de lo que usa nuestro modelo en nuestros datos.
""")
code("F.fig_draftrec()")

# ---------------------------------------------------------------------------------------------------------------
md(r"""
## 5. Robustez

**¿Se degrada con el tiempo?** Dentro de V, partida en cuatro ventanas cronológicas del mismo tamaño, el AUC no
muestra una tendencia distinguible de cero en ninguno de los dos modelos (la pendiente lleva su IC).
""")
code("F.fig_degradacion_v()")
md(r"""
En el test se repite la misma medición por ventanas y, además, la variación entre 5 semillas fijadas de antemano.
""")
code("F.seccion_robustez_test()")

# ---------------------------------------------------------------------------------------------------------------
md(r"""
## 6. Dos preguntas laterales

**Huella de estilo.** ¿Se puede reconocer a una persona por cómo juega, aunque cambie de campeón? Con jugadores que
el modelo nunca vio y campeones distintos de cada lado, una red contrastiva lo logra con AUC cercano a 0,9 a partir
de 10 partidas, y sube con más partidas.
""")
code("F.fig_huella()")
md(r"""
**Matchmaking.** Si se reacomodan los mismos diez jugadores entre equipos y líneas, sin que nadie pase a un rol que
juega menos, el desbalance que estima el modelo se reduce a la mitad. Es una **simulación bajo el modelo**: esas
partidas no se jugaron, y no se pueden respetar dúos ni grupos porque los datos no los registran.
""")
code("F.fig_matchmaking()")

# ---------------------------------------------------------------------------------------------------------------
md(r"""
## 7. Limitaciones

- **Elegido mirando V.** Unas 40 variantes del modelo previo se compararon en V; los bloques hacia adelante y las
  partidas de Diamante de la misma ventana sirven de control, pero el número que vale es el del test.
- **El test se mira una sola vez**, con el protocolo fijado antes de cargarlo.
- **Eventos del timeline** (dragones, primera sangre, oro sin gastar) suman en vivo en V y en los bloques, pero no se
  presentan: no hay timelines de las partidas de Diamante de la ventana para replicarlo.
- **Las partidas largas son casi impredecibles antes de jugar**: lo previo sirve sobre todo cuando la partida se
  define temprano.
- **Datos que no están**: bans, rol pedido y autofill exacto; el rol es el jugado.
- **Un servidor y una temporada**: Master+ de Latinoamérica Sur, con casi todas las partidas entre junio y
  septiembre de 2026.
""")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ejecutar", action="store_true", help="ejecutar el notebook después de escribirlo")
    args = ap.parse_args()
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb["metadata"]["language_info"] = {"name": "python"}
    if args.ejecutar:
        from nbclient import NotebookClient
        NotebookClient(nb, timeout=1200, kernel_name="python3",
                       resources={"metadata": {"path": str(OUT.parent)}}).execute()
    nbf.write(nb, OUT)
    print(f"[+] {OUT} ({len(cells)} celdas)")


if __name__ == "__main__":
    main()
