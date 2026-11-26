# Hoja de Ruta del Proyecto (ROADMAP.md)

> **Propósito de este archivo:** responder, en una sola lectura, a *qué se está
> buscando* y *en qué etapa está el trabajo*. Es el mapa; el detalle técnico y la
> justificación matemática de cada decisión viven en
> [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md), y las reglas operativas para agentes
> en [`AGENTS.md`](AGENTS.md).

---

## 1. La Pregunta de Investigación

Todo el proyecto existe para responder una única pregunta de Machine Learning:

> **¿Qué predice mejor el resultado de una partida competitiva de League of
> Legends: el _draft_ (qué campeones se eligieron y cómo se enfrentan entre sí),
> o el _factor humano_ (quiénes son los 10 jugadores y cómo juegan)?**

No es una pregunta retórica: es una hipótesis falsable. Ambos enfoques se
construyen como modelos independientes que predicen **exactamente el mismo
target** $Y$ (victoria de Blue/Red) sobre **exactamente las mismas partidas**, y
se comparan con las mismas métricas. Si uno gana, gana medido.

### Los tres competidores

| Modelo | Entrada $X$ | Hipótesis que defiende |
| :--- | :--- | :--- |
| **Modelo Campeones** (Draft/Macro) | Composición 5v5, sinergias, matchups 1v1 de línea, estado global @10/@15/@20 | Los counters y el diseño de campeones dictan la partida |
| **Modelo Jugadores** (Factor Humano/Micro) | Arquetipos históricos de los 10 invocadores: %KP, %Damage Share, agresividad temprana, visión, dispersión de pool | La habilidad y el estilo individual superan la ventaja teórica del draft |
| **Ensamble Híbrido** | Ambos espacios combinados | La interacción manda: un campeón fuerte *en manos de un especialista* no es lo mismo que en manos de un weakside |

### Por qué la respuesta no es obvia

Un campeón con ventaja teórica de matchup en manos de alguien que lo juega por
primera vez puede perder la línea contra su *counter* pilotado por un
especialista. El proyecto mide cuál de las dos fuerzas domina en SoloQ de alto
elo, y dónde exactamente se cruzan.

---

## 2. El Dominio: Por Qué Diamante+ en LAS

* **Servidor `la2` (LAS):** población acotada y densamente conectada. El grafo de
  jugadores cierra sobre sí mismo, así que el *crawling* en profundidad satura
  historiales en vez de dispersarse.
* **Diamante+ (Challenger → Diamond):** filtra el ruido de partidas donde el
  resultado se explica por errores groseros y no por draft ni por estilo. Es el
  régimen donde la pregunta de investigación tiene sentido.
* **Solo Ranked SoloQ 5v5 (`queue_id = 420`)**, con *remakes* ($\le 300$ s)
  descartados explícitamente.

---

## 3. Etapas del Trabajo

> ### Regla de avance: las etapas son secuenciales, no paralelas
>
> Cada etapa está separada de la siguiente por una **compuerta**: una condición que
> debe ser cierta antes de empezar la que sigue. Ninguna etapa arranca por
> adelantado, y el trabajo hecho antes de que su compuerta abra es **provisional**:
> hay que rehacerlo sobre los datos definitivos.
>
> ```
> volumen objetivo  →  limpieza  →  dataset armado  →  EDA  →  entrenamiento  →  memoria
>       (E2)            (E4a)          (E4b)          (E5)        (E6)          (E7)
> ```
>
> La cadena se lee así, y en ese orden estricto:
>
> * **No se limpia el dataset hasta tener la cantidad de datos que queremos.**
>   Limpiar sobre un dataset que sigue creciendo obliga a repetir la limpieza en
>   cada corrida, y peor: los umbrales de corte calculados sobre una muestra
>   parcial dejan de ser válidos cuando llegan los datos que faltaban.
> * **No se hace EDA hasta que el dataset esté armado.** Un análisis exploratorio
>   sobre datos a medio recolectar describe la muestra, no la población: las
>   distribuciones, los clusters y las correlaciones cambian cuando entra el resto.
> * **No se entrena hasta que el EDA esté cerrado.** Las decisiones de features,
>   umbrales de madurez y filtros salen del EDA; entrenar antes es elegir a ciegas.
>
> **Por qué esto importa acá en particular:** los vectores de estilo por jugador
> son promedios históricos. Un invocador con 30 partidas y el mismo invocador con
> 300 producen vectores distintos, así que todo lo que se derive de ellos —
> clusters, arquetipos, features del Modelo Jugadores — se invalida cuando sigue entrando
> volumen. Por eso la compuerta de volumen es la primera y la más dura.


### Etapa 1 — Infraestructura de Extracción ✅ COMPLETADA

Construir un crawler que pueda sostener recolección masiva contra la API de Riot
sin ser bloqueado ni corromper datos.

* Cliente HTTP con *rate limiting* estricto (techo seguro 18 req/s, 95 req/120 s).
* Pool multi-key con balanceo y detección de claves vencidas.
* SQLite con deduplicación por clave primaria y *context manager* estricto
  (evita `[WinError 32]` en Windows).
* Recolección en profundidad (`depth_first`) sobre el grafo de jugadores.

**Compuerta ✅ abierta:** el crawler sostiene corridas largas sin bloqueos ni
corrupción, con la suite en verde.

### Etapa 2 — Recolección y Enriquecimiento ✅ CERRADA (dataset congelado)

Acumular volumen suficiente para que los vectores de estilo por jugador sean
estadísticamente maduros.

* **Logrado:** 272.849 partidas de Ranked Solo/Duo de LAS (export del 01/11/2026,
  40.821 jugadores), con Timeline @10/@15/@20 en todas. Orden de llenado: 50 partidas por
  jugador en todos los tiers y después hasta ~300, más la ventana desde el 01/09 para el test.
* **Splits vigentes** (`export_to_csv.py --target splits`, `PROJECT_SUMMARY.md` §9.S.12):
  test = 20 % más reciente de Master+ con un corte único (27/09); Diamond anterior al corte en el
  train y posterior aparte. El test del 27/09 ya se usó para evaluar los baselines de draft (§11).
* **Bases aparte que siguen creciendo** (no tocan `lol_las.db` ni los 4 CSV): timelines completos
  con eventos (§9.S.64) y LP en vivo de Master+ (§9.S.22).

**Compuerta ✅ abierta:** la regla del split se fijó el 26/10/2026 y el dataset vigente es el
export del 01/11/2026; desde entonces `lol_las.db` no se recolecta.

### Etapa 3 — Identidad Canónica Multi-Proyecto ✅ COMPLETADA

Riot cifra el PUUID como HMAC(puuid_global, project_secret): **cada proyecto de
API Key devuelve un PUUID distinto para el mismo humano**. Sin resolver esto, un
mismo jugador aparece como varios y sus vectores de estilo se fragmentan.

* Registro de proyectos y mapeo PUUID↔proyecto (`scripts/identity/`).
* Resolución inversa por partida: el Match DTO devuelve los 10 PUUIDs frescos, y
  el join `(match_id, participant_id)` contra las filas locales establece enlaces
  deterministas sin gastar Account-V1 por jugador.
* Unificación en grafo de 3 capas (`src/features/identity.py`: `v_puuid_canonical` +
  `player_puuid_map` + `puuid_riot_ids`) con **invariante de disyunción de
  proyectos** ($\Pi(C_u) \cap \Pi(C_v) = \emptyset$) para bloquear colisiones por
  reutilización de Riot ID y biyección estricta componente $\leftrightarrow$
  `player_id` materializada en `player_id_map`.

**Compuerta ✅ abierta:** el 100,00 % de los PUUIDs del padrón y de las
participaciones resuelve a su componente conexa `player_id` (`0` PUUIDs huérfanos).

### Etapa 4 — Datasets de ML ✅ COMPLETADA

Cuatro CSV derivados de SQLite, con coherencia matemática verificada entre sí:

| Dataset | Alimenta a | Contenido |
| :--- | :--- | :--- |
| `full_dataset.csv` | Ambos | Participación individual + contexto de equipo + Timeline |
| `players.csv` | **Modelo Jugadores** | Feature store del padrón activo: vectores de estilo por jugador |
| `champions_by_lane.csv` | **Modelo Campeones** | Rendimiento agregado por campeón y línea |
| `matchups_1v1.csv` | **Modelo Campeones** | Duelos directos de línea con diferenciales $\Delta$Gold/CS/XP |

* **4a — Armado ✅ operativo.** `export_to_csv.py` regenera los cuatro CSV desde
  SQLite cuando se lo invoca. Esto no está bloqueado: es una proyección de lo que
  haya en la base en ese momento.
* **4b — Limpieza ✅ en el preprocesamiento de ML.** Los CSV siguen siendo el volcado
  fiel de la BD (`AGENTS.md` §3.G); los filtros (biyección de posiciones, partidas sin
  ganador) y la deuda metodológica (§4) se aplican aguas abajo,
  en el código de cada modelo, sobre el dataset congelado.

**Compuerta ✅ abierta:** 4b corre sobre el volumen definitivo.

### Etapa 5 — Análisis Exploratorio y Arquetipos ✅ HECHA (quedan decisiones de presentación)

* EDA rehecho sobre el dataset congelado el 26/10 (§9.S.19) en
  [`notebooks/eda_visualizer.ipynb`](notebooks/eda_visualizer.ipynb); el clustering y la huella, en
  [`notebooks/eda_reconocimiento_jugadores.ipynb`](notebooks/eda_reconocimiento_jugadores.ipynb), que
  reemplaza a `clustering_jugadores.ipynb` y `eda_huella.ipynb` (el k sigue a decisión del usuario: no
  hay grupos naturales, §9.S.2 y §9.S.53; la huella, §10.3-10.9). Notebooks de la etapa de
  modelos: `eda_modelos_ganador.ipynb` (§9.S.42 y siguientes) y
  `rating_jugadores.ipynb` (§9.S.21).
* *Clustering* de arquetipos tácticos puros (K-Means/GMM) con **nombres dinámicos**
  derivados de la variable más distintiva de cada centroide ($|z|$ máximo), sin
  etiquetas semánticas impuestas a mano.
* Embeddings campeón-línea y firmas de jugador
  ([`src/features/player_signatures.py`](src/features/player_signatures.py)):
  * **Variante A (macro):** una firma global por jugador.
  * **Variante B (posicional):** una firma por (jugador, línea), con umbral de
    madurez $N \ge 10$ partidas en esa línea.

**Compuerta:** se abre con el EDA cerrado sobre el dataset final y las features
del Modelo Jugadores congeladas.

### Etapa 6 — Entrenamiento y Enfrentamiento 🟡 AVANZADA (falta el test)

**Dónde está (2026-11-19). Los modelos vigentes se midieron en V Master+ con AUC y acierto.**

* *Draft contra jugadores:* el draft solo da ~0,55 / 54 %; con los jugadores, el híbrido
  0,6531 / 61,1 % (§9.S.51, §9.S.55).
* *Antes de la partida:* **escalón 9** (logística sobre draft campeón × línea + historial de
  cada jugador + nivel histórico y W/L de la foto 1 + Kalman-Elo + contracruces + modelo por
  línea; `brainstorm_final.py`): **0,6740 / 62,6 %** en V Master+ tras aislar el
  preprocesamiento de V (§9.S.68); 0,6727 / 62,4 % al enmascarar el LP de la foto 1 antes de la foto (§9.S.75,
  issue #4). El modelo incluye Kalman-Elo y nivel histórico por LP.
* *Durante la partida:* `6_Final_dEns10` a los 10 / 15 / 20 min: 0,7912 / 71,6 %, 0,8441 / 76,1 %
  y 0,8750 / 79,1 % (§9.S.60). Los eventos del timeline suman otros +0,006 a +0,007 (§9.S.66):
  falta integrarlos.
* *Matchmaking (simulación bajo el modelo):* con los mismos 10 jugadores, el desbalance que mide el modelo
  baja de ~10 a ~5 pp (4,96) sin que ningún jugador empeore su rol; no restringe dúos ni premades (§9.S.67, §9.S.77).

**Lo que falta:** integrar los eventos y congelar el protocolo y los modelos finales.
La revisión recomienda una cohorte confirmatoria independiente por el uso previo del test
en los baselines (`TAREAS.md` §3.A y `PROJECT_SUMMARY.md` §11).

* Entrenar el Modelo Campeones, el Modelo Jugadores y el Ensamble Híbrido sobre el mismo *split*.
* Estudio de ablación de representación (Jugadores vs Campeones vs Ensamble).
* Comparativa formal: Accuracy, ROC-AUC, F1, Brier Score.
* Análisis de discrepancias: *dónde* cada modelo acierta y el otro falla.
* **Regla de validación (desde 2026-11-14, `PROJECT_SUMMARY.md` §9.S.47):** toda variante se
  decide con validación temporal por bloques sobre el ajuste (`src/models/validacion_temporal.py`:
  5 bloques por fecha, ventana creciente). Entra si mejora en ≥ 3 de los 4 bloques evaluados y el
  IC 95 % de la media de los Δ es > 0. V solo confirma, y el test se abre una sola vez, al final.
* **V solo con Master+ (desde 2026-11-15, §9.S.51):** V = partidas de la ventana de validación con
  mediana del tier del lobby ≥ Master (`particion_v_master`); las de Diamond de esa ventana pasan a
  A4 para evaluarse hacia adelante, pero **el modelo que predice V se entrena solo con partidas
  anteriores al inicio de V** (sin fuga hacia adelante). Todo resultado se reporta con AUC **y** acierto.

**Compuerta:** se abre con las métricas de los tres modelos medidas sobre el mismo
*split*, listas para escribirse.

### Etapa 7 — Memoria Técnica Final ⬜ PENDIENTE

Redacción del informe académico y la carátula explicativa a partir de
`PROJECT_SUMMARY.md`, que acumula la justificación de cada decisión de diseño.

---

## 4. Deuda Metodológica

Cinco sesgos del dataset están **auditados y con corrección diseñada**. Ninguno se corrige
en la BD ni en los CSV; en el preprocesamiento de ML de los modelos de quién gana, cuatro ya
se aplican y uno sigue abierto (estado detallado en `AGENTS.md` §3.F):

1. **Sesgo de lado** — Red gana más que Blue por asignación de MMR. *Aplicada:* variables azul − rojo e intercepto.
2. **Matriz dispersa de matchups** — muchos cruces campeón vs campeón con una sola partida. *Aplicada:* encogimiento empírico-bayesiano causal y regularización.
3. **Distribución bimodal de invocadores** — mayoría de perfiles con muy pocas partidas. *Abierta* en el clustering de jugadores; en quién gana se usa encogimiento por partidas previas.
4. **Truncamiento por rendición temprana** — partidas que terminan antes del minuto 20. *Aplicada:* módulos por minuto con máscara de fin de partida.
5. **Posiciones duplicadas / vacías y etiqueta de victoria anómala por payload de Riot** — lobbies sin biyección 1:1 de roles (`posicion_duplicada`, `posicion_vacia`) más la partida `LA2_1618651762` (`ganador_incoherente`, 10 participantes con `win = 0`), que se conservan en la BD/CSV como volcado fiel. *Aplicada:* se filtran en el preprocesamiento de ML.

> Las magnitudes medidas y la **solución diseñada para cada una** (data augmentation
> simétrico, suavizado bayesiano, umbrales de madurez, módulos temporales
> escalonados, filtro de biyección) están en [`AGENTS.md`](AGENTS.md) §3.F, que es
> su única fuente de verdad.

> **Garantía anti-*data leakage*:** las estadísticas de jugador se calculan
> *a priori* sobre historial previo a la partida predicha, y los snapshots
> @10/@15/@20 son estado en vivo — ningún modelo ve minutos futuros ni métricas
> de post-partida.

---

## 5. Mapa del Repositorio

```
.
├── ROADMAP.md              # este archivo: qué se busca y en qué etapa estamos
├── TAREAS.md               # tablero del día a día: qué hay que hacer ahora
├── PROJECT_SUMMARY.md      # memoria técnica: el porqué de cada decisión
├── AGENTS.md               # reglas operativas y changelog
├── README.md               # instalación y uso
│
├── run.py                  # entrada: recolectar + validar + respaldar
├── collect_data.py         # entrada: recolección (fases sueltas)
├── export_to_csv.py        # entrada: SQLite -> 4 datasets CSV (y splits)
│
├── src/                    # paquete central
│   ├── config.py           #   configuración y rutas
│   ├── rate_limiter.py     #   rate limiting de Riot API
│   ├── riot_client.py      #   cliente HTTP multi-key
│   ├── database.py         #   modelo SQLite y deduplicación
│   ├── collector.py        #   orquestación del crawling
│   ├── features/           #   ingeniería de features derivadas (márgenes, lobby, draft, LP…)
│   └── models/             #   baselines, validación temporal por bloques, modelo en vivo
│
├── scripts/                # utilidades agrupadas por propósito
│   ├── analysis/           #   lo vigente: modelos finales, sus cachés, matchmaking, huella, notebooks (README.md)
│   ├── collection/         #   recolección, LP en vivo, timelines con eventos
│   ├── identity/           #   resolución de PUUIDs multi-proyecto (Etapa 3)
│   ├── backup/             #   sincronización y restauración vía Google Drive
│   └── ops/                #   diagnóstico: keys, monitor en vivo, lookup
│
├── historial/              # experimentos descartados: idea, resultado y por qué no entraron (README.md)
├── notebooks/              # EDA, clustering, embeddings, modelos de victoria, huella
├── tools/                  # diagnóstico manual interactivo (no automatizado)
├── tests/                  # suite pytest
└── docs/                   # BRAINSTORM_NOCHE.md y archive/ (bitácoras históricas)
```
