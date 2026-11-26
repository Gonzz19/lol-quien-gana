# Resumen Técnico del Proyecto - League of Legends ML Crawler & Pipeline

> **Documento de Contexto y Estado del Proyecto para Humanos y Agentes de IA.**  
> Este documento contiene el estado actual del repositorio, la arquitectura implementada, las decisiones de diseño, los cambios realizados, las ideas descartadas y la hoja de ruta para los modelos de Machine Learning.
> 
> **🤖 INSTRUCCIÓN OBLIGATORIA PARA AGENTES DE IA / LLMs:**
> Cualquier agente de IA (LLM/Copilot/Claude/GPT/Antigravity) que realice cambios en este proyecto **DEBE actualizar este archivo (`PROJECT_SUMMARY.md`) y registrar su sesión en [`AGENTS.md`](AGENTS.md)** con cada modificación de arquitectura, nuevo feature, dataset o modelo implementado.

---

## 1. Visión y Objetivos de Machine Learning: El Enfrentamiento de Dos Modelos

El proyecto consiste en una plataforma de extracción, estructuración y modelado de datos de alto elo (**Diamante+ en LAS / `la2`**) diseñada para responder a la pregunta central de Machine Learning:  
> **"¿Qué factor tiene mayor poder predictivo sobre el resultado de una partida de League of Legends: el Draft y Matchups teóricos de los campeones (Macro/General) o el Factor Humano y Estilo de los 10 Jugadores (Micro/Personalizado)?"**

Para ello, se enfrentan **dos modelos independientes y un ensamble híbrido**:

```
                              ┌────────────────────────────────────────────────────────┐
                              │  Partida en Tiempo Real / Live State (Min 0, 10, 15, 20) │
                              └───────────┬────────────────────────────────┬───────────┘
                                          │                                │
                     ┌────────────────────┴──────────┐   ┌─────────────────┴──────────────────┐
                     ▼                               ▼   ▼                                    ▼
       ┌───────────────────────────┐ ┌───────────────────────────┐ ┌───────────────────────────┐
       │     MODELO CAMPEONES      │ │     MODELO JUGADORES      │ │     ENSAMBLE HÍBRIDO      │
       │   (Draft & Matchups)      │ │   (Factor Humano / Perfil)│ │ (Macro Draft + Ajuste Hum)│
       ├───────────────────────────┤ ├───────────────────────────┤ ├───────────────────────────┤
       │ • 10 Campeones y Sinergias│ │ • Arquetipos de 10 players│ │ • Meta-clasificador       │
       │ • Matchups 1v1 teóricos   │ │ • %KP, %Dmg, Agresividad  │ │ • Ponderación dinámica    │
       │ • Estado general al min X │ │ • OTP vs First-time pick  │ │   Draft vs Skill          │
       └─────────────┬─────────────┘ └─────────────┬─────────────┘ └─────────────┬─────────────┘
                     └─────────────────────────────┼─────────────────────────────┘
                                                   ▼
                                  ┌─────────────────────────────────┐
                                  │       BENCHMARK & RESULTADOS    │
                                  │  • Comparación de Accuracy/AUC  │
                                  │  • Análisis de discrepancias    │
                                  └─────────────────────────────────┘
```

1. **Modelo Campeones (General / Draft & Matchups Macro):**
   - **Enfoque:** El juego en abstracto. Evalúa la composición de 5v5, sinergias de equipo, ventajas teóricas de línea (`matchups_1v1.csv`) y estado global de la partida al minuto 10/15/20.
   - **Datasets:** `champions_by_lane.csv`, `matchups_1v1.csv` y estado global de `full_dataset.csv`.

2. **Modelo Jugadores (Personalizado / Factor Humano & Arquetipos Micro):**
   - **Enfoque:** La identidad y estilo de los 10 jugadores. Evalúa sus vectores históricos de arquetipo (`players.csv`), agresividad temprana habitual, control de visión, especialización de champion pool (OTP vs Versátil) y maestría específica.
   - **Datasets:** `players.csv` y perfiles históricos de `full_dataset.csv`.

3. **Garantía Estricta Contra Data Leakage (Fuga de Datos):**
   - **Estadísticas de Jugadores:** Se calculan *a priori* de forma histórica antes de la partida que se predice.
   - **Snapshots Temporales (@10, @15, @20):** Representan el estado en vivo (*Live Game State*) en ese minuto exacto; ningún modelo accede a información de minutos futuros ni a estadísticas de post-partida (daño final, duración, ítems tardíos).


---

## 2. Arquitectura del Sistema

```
                        ┌─────────────────────────────────────┐
                        │       Riot Games API (v4 / v5)      │
                        │ (League-V4, Match-V5, Timeline-V5)  │
                        └──────────────────┬──────────────────┘
                                           │
                                           ▼
                        ┌─────────────────────────────────────┐
                        │         src/riot_client.py          │
                        │  - Token-bucket Rate Limiter        │
                        │  - 20 req/1s & 100 req/2min limits  │
                        │  - Lotes de 100 partidas por call   │
                        │  - Auto-recuperación ante HTTP 429  │
                        └──────────────────┬──────────────────┘
                                           │
                                           ▼
                        ┌─────────────────────────────────────┐
                        │          src/collector.py           │
                        │  - Sincronización Diamante+         │
                        │  - Modo Depth-First (Profundidad)   │
                        │  - Deduplicación O(1) en SQLite     │
                        │  - Timelines @10, @15, @20          │
                        └──────────────────┬──────────────────┘
                                           │
                                           ▼
                        ┌─────────────────────────────────────┐
                        │      data/lol_las.db (SQLite)       │
                        │  - players (40,612 indexados)       │
                        │  - matches                          │
                        │  - match_participants (10 por game) │
                        │  - match_timeline (10 por game)     │
                        └──────────────────┬──────────────────┘
                                           │
                                           ▼
                        ┌─────────────────────────────────────┐
                        │         export_to_csv.py            │
                        │  Exportación atómica a 4 Datasets   │
                        └──────────────────┬──────────────────┘
                                           │
     ┌──────────────────────┬──────────────┴───────┬──────────────────────┐
     ▼                      ▼                      ▼                      ▼
full_dataset.csv       players.csv        champions_by_lane.csv     matchups_1v1.csv
(Detalle + Timeline) (Arquetipos ML)     (Métricas por Rol)       (Matchups 1v1 ML)
```

### A. Arquitectura Multi-Proyecto y Aislamiento Dinámico de Rate Limits

Riot Games encripta los identificadores de jugador (*PUUID*) de manera determinista mediante una función HMAC ligada al secreto del proyecto de desarrollador:
$$\text{PUUID}_{\text{local}} = \text{HMAC}(\text{PUUID}_{\text{global}}, \text{Secret}_{\text{project}})$$
Por lo tanto, una misma persona física recibe 3 PUUIDs completamente distintos según cuál de las 3 API Keys del pool consulte la API.

Para garantizar la unicidad de las entidades sin duplicaciones en la base de datos ni fugas de información, el sistema implementa:
1. **Padrón Canónico Unificado (`player_identities`):** Clave canónica uniforme basada en `lower(gameName#tagLine)`, independiente de rotaciones de claves o proyectos.
2. **Matriz de Traducción Multi-Proyecto (`player_puuid_map`):** Relaciona biyectivamente cada par $(\text{canonical\_id}, \text{project\_index}) \to \text{puuid}_{\text{local}}$.
3. **Caché en Memoria $O(1)$:** `DataCollector` y `DatabaseManager` traducen cualquier PUUID entrante al PUUID canónico de referencia (Proyecto 0) en tiempo constante antes de la inserción.
4. **Aislamiento Dinámico de Concurrencia:** Durante fases intensivas de registro inicial de un proyecto (como la resolución de identidades de Master con Key 2 a través de Account-V1 a razón de $100\text{ req} / 120\text{s}$), el pipeline autónomo (`scripts/collection/auto_nightly_pipeline.py`) aísla la recolección de partidas a las Keys 0 y 1. Esto erradica colisiones de Rate Limit (HTTP 429) y previene la inserción de PUUIDs no traducidos, incorporando automáticamente la Key 2 al pool de 3 claves tan pronto como su padrón alcanza madurez ($\ge 8,000$ identidades).

---

## 3. Estado de los Datos y Datasets Exportados

Todos los archivos generados residen en la carpeta `data/` y se actualizan automáticamente tras cada ejecución del crawler:

| Archivo CSV | Filas / Registros | Columnas / Clave | Propósito Principal |
| :--- | :--- | :--- | :--- |
| **`data/full_dataset.csv`** | **$981.670$ filas** ($380.6\text{ MB}$) | 62 columnas | Dataset maestro con métricas finales, contexto de equipo (%KP, %Dmg Share, %Gold Share) y snapshots del Timeline al min 10, 15 y 20 (**100% cobertura**). |
| **`data/players.csv`** | **$54.328$ jugadores** ($17.7\text{ MB}$) | 43 columnas | Vector de comportamiento y arquetipos tácticos por jugador derivado de las partidas recolectadas (**Para Modelo Jugadores**), con exactamente 700 filas de Apex unificadas canónicamente. |
| **`data/champions_by_lane.csv`**| **$844$ roles/champs** ($0.14\text{ MB}$) | 22 columnas | Rendimiento promedio de cada campeón en cada línea (TOP, JUNGLE, MID, BOT, UTILITY). |
| **`data/matchups_1v1.csv`** | **$490.914$ enfrentamientos** ($165.6\text{ MB}$) | 27 columnas | Enfrentamientos directos 1v1 (Blue vs Red) con diferenciales de Oro, CS, XP y Daño al min 10, 15 y 20, diferenciales finales y aceleración de bola de nieve (**Para Modelo Campeones**). |

> **Garantía de Coherencia Matemática y Cobertura Total:**  
> $\sum \text{games\_recorded (players.csv)} = 981.670 = \text{filas (full\_dataset.csv)}$.  
> $490.914 \text{ matchups 1v1} \times 2 = 981.828 \text{ participantes evaluados}$ (ajustado por desempates e inversiones posicionales biyectivas en $98.167 \text{ partidas únicas}$).  
> **Cobertura de Timeline:** **$98.167 / 98.167$ partidas ($100.0\%$)** con snapshots completos al minuto 10, 15 y 20.
> **Padrón Apex:** Exactamente **700 cupos canónicos** (200 Challenger y 500 Grandmaster) unificados mediante `v_puuid_canonical` sin duplicaciones por cambios de Riot ID.
> **Cobertura de Invocadores ($\ge 50$ partidas):** **$5.699$ invocadores** (Challenger: $199/200$ $99.5\%$, Grandmaster: $500/500$ $100.0\%$, Master: $4.693/8.749$ $53.6\%$, Diamond: $307$); con $\ge 100$: **$2.563$**, con $\ge 200$: **$598$**, con $\ge 300$: **$158$**.


---

## 4. Decisiones de Diseño, Cambios y Enfoques Descartados

### A. Almacenamiento: SQLite como Fuente Central vs CSV plano
* **Descartado:** Escribir directamente y únicamente en archivos CSV a medida que llegaban las peticiones de red.
* **Motivo:** En Windows, el reescrito continuo de CSV bloquea archivos (`[WinError 32]`), genera condiciones de carrera y hace inviable la deduplicación en tiempo real.
* **Implementado:** Base de datos SQLite (`data/lol_las.db`) con transacciones ACID, índices en `match_id` y `puuid`, y context managers que cierran conexiones de inmediato. Los CSVs se exportan de forma atómica a partir de consultas SQL agregadas.

### B. Indexación de Jugadores: Diamante+ Completo vs Solo Apex
* **Inicial:** Se sincronizaban únicamente las ligas Apex (Challenger: 200, Grandmaster: 500, Master: ~8,259 $\approx 8,959$ jugadores).
* **Cambio:** Se integró la paginación de `League-V4 /entries/RANKED_SOLO_5x5/DIAMOND/{I,II,III,IV}`, logrando indexar a los **40,612 jugadores Diamante+** de LAS (31,653 Diamantes + 8,959 Apex).

### C. Coherencia en `players.csv` (Partidas de Muestra vs Perfil Anual)
* **Descartado:** Exportar en `players.csv` el total anual de `wins + losses` de la temporada reportado por la API de Riot (daba sumas de más de 10 millones de partidas cuando el dataset solo tenía 465 partidas).
* **Implementado:** `players.csv` se reformuló como una **agregación estricta de las partidas registradas en el dataset general**, calculando métricas de arquetipo reales (%KP, %Damage Share, Early Aggression @10) basadas en las partidas analizadas.

### D. Estrategia de Muestreo: Modo Profundidad (`Depth-First`) vs Amplitud
* **Descartado:** Iterar secuencialmente sobre los 40,000 jugadores descargando 1-2 partidas de cada uno. Esto no permitía calcular vectores de arquetipo por falta de datos por jugador.
* **Implementado:** Priorización `depth_first` que selecciona a jugadores que ya tienen historial parcial en la base de datos para completarlos hasta alcanzar **50 a 100 partidas por jugador**.

### E. Optimización de API: Lotes de Match IDs
* **Inicial:** Peticiones de `count=10` por jugador.
* **Implementado:** Peticiones de `count=100` (máximo permitido por Match-V5) con paginación automática. Esto redujo las llamadas de lista a prácticamente cero, maximizando las descargas de detalles y timelines.

### F. Captura de Match-V5 Timeline API
* **Implementado:** Para cada partida se descarga el timeline estructurado extrayendo snapshots exactos a los **minutos 10, 15 y 20**:
  - `gold_10`, `xp_10`, `level_10`, `cs_10` (minions + jungla), `damage_10`, `kills_10`, `deaths_10`, `assists_10`.
  - Lo mismo para minuto 15 y 20.
  - Esto habilita el cálculo de diferencias de fase de líneas y snowball temprano.

### G. Multi-API-Key Pool con Balanceo Dinámico
* **Problema:** El Rate Limit estricto de Riot ($20\text{ req}/1\text{s}$, $100\text{ req}/120\text{s}$) impone un techo teórico de $\approx 23.5\text{ partidas/min}$ ($\approx 1.400\text{ partidas/h}$) por Development Key.
* **Implementado:** Soporte para un pool arbitrario de $N$ claves en `RIOT_API_KEYS=key1,key2,key3...`.
* **Mecanismo:** `RiotAPIClient` gestiona un pool de `KeyContext` (cada una con su `RiotRateLimiter` y sesión HTTP persistente dedicada), despachando cada petición a la clave que presente el menor tiempo de espera (`min_wait = min(ctx.get_wait_time())`).
* **Throughput:** Escalamiento lineal ($N \times 47.5\text{ req/min}$), alcanzando $\approx 3.650+\text{ partidas/h}$ con 3 claves.
* **Tolerancia a Fallos:** Si una clave individual expira (HTTP 401/403) o se satura (HTTP 429), solo se desactiva o pausa ese contexto, permitiendo que el resto del pool continúe extrayendo sin interrupciones.

### H. Planificador Adaptativo por Tier y Solapamiento de Grafos (Meta = 300 partidas)
* **Problema del Overshooting:** En elos altos densos (Challenger/GM: 700 jugadores), pedir 300 partidas directas a cada jugador provoca que los primeros en la cola terminen con 400-500 partidas, debido a que las partidas de los jugadores posteriores les siguen sumando partidas de forma pasiva.
* **Derivación Matemática:**
  Sea $N$ la población del tier, $k = 9$ los compañeros por partida, y $p_{intra}$ la probabilidad de emparejamiento intra-tier. Cada partida descargada directamente aporta $1$ partida al jugador primario y genera $k \cdot p_{intra}$ participaciones pasivas distribuidas entre el resto de la población.
  El promedio de directas $\bar{m}$ para converger a $T = 300$ partidas finales se rige por:
  $$\bar{m} = \frac{T}{1 + k \cdot p_{intra}}$$
  Para compensar la asimetría temporal del recorrido secuencial (donde los primeros jugadores reciben pasivas de todos los posteriores y los últimos de ninguno), el umbral base directo $B_{tier}$ se parametriza como:
  $$B_{tier} = T - \frac{1}{2} \left(k \cdot p_{intra} \cdot \bar{m}\right)$$
* **Valores Óptimos Implementados:**
  - **Challenger ($N=200, p_{intra}=0.90$):** $B = 160$ partidas base $+ 147.0$ pasivas $\rightarrow \mathbf{\approx 307 \text{ partidas finales}}$.
  - **Grandmaster ($N=500, p_{intra}=0.85$):** $B = 175$ partidas base $+ 131.2$ pasivas $\rightarrow \mathbf{\approx 306 \text{ partidas finales}}$.
  - **Master ($N=8.259, p_{intra}=0.75$):** $B = 215$ partidas base $+ 89.2$ pasivas $\rightarrow \mathbf{\approx 304 \text{ partidas finales}}$.
  - **Diamond ($N=31.653, p_{intra}=0.65$):** $B = 260$ partidas base $+ 42.0$ pasivas $\rightarrow \mathbf{\approx 302 \text{ partidas finales}}$.
* **Mecanismo Top-Up:** El crawler evalúa $\text{matches\_to\_request} = \max(0, B_{tier} - \text{existing\_in\_db})$. Si un jugador ya alcanzó su umbral pasivamente, se salta automáticamente con **0 peticiones de red**.

### I. Población Cerrada y Blindaje contra Elo Drift (Deriva de Elo)
* **Decisión:** La tabla `players` es estrictamente cerrada y estática. Solo se puebla mediante `sync_players()` consultando los endpoints oficiales de ligas (`League-V4`).
* **Justificación:** Si el crawler agregara a la lista de objetivos a los otros 9 participantes de cada partida descargada, se produciría una "deriva de elo" (*elo drift*), donde partidas con smurfs o duos en placements harían que el crawler descendiera a buscar partidas de Esmeralda, Platino u Oro, arruinando la calidad del dataset competitivo.
* **Garantía:** El crawler solo itera sobre los 40.612 jugadores Diamante+ pre-verificados.

### J. Desencriptado Multi-Cuenta Resiliente de PUUIDs
* **Problema:** Riot Games encripta los `PUUID` de los jugadores utilizando una clave HMAC privada asociada a la cuenta de desarrollador que ejecutó la consulta en `League-V4`. Al usar múltiples API Keys de distintas cuentas, una clave puede recibir `HTTP 400 Bad Request - Exception decrypting puuid`.
* **Solución Implementada:**
  1. En las peticiones de lista `/by-puuid/{puuid}/ids`, si una clave falla con error de desencriptado, el dispatcher rota instantáneamente excluyendo ese contexto y prueba con las demás claves del pool.
  2. Los identificadores de partida (`match_id`, ej: `LA2_1621221364`) son universales y públicos: una vez obtenidos los IDs, la descarga de `/matches/{id}` y `/timeline` se balancea concurrentemente entre todas las claves del pool.
### K. Modelado de Premades Habituales en Diamante
* **Objetivo Académico:** Cuantificar el grado de juego coordinado (premade / DuoQ) en Diamante para modelar el impacto del juego en pareja sobre el winrate individual (Modelo Jugadores) y predecir sinergias en draft (Modelo Campeones).
* **Definición Formal de Pareja Premade:**
  Dados dos invocadores $A$ y $B$, denotamos por $G_{AB}$ la cantidad de partidas disputadas en el **mismo equipo** ($\text{team\_id}$) y por $N_A$ el total de partidas registradas del invocador $A$. Definimos que $B$ es un compañero habitual / premade de $A$ si y solo si:
  $$\text{Premade}(A, B) \iff G_{AB} \ge 50 \quad \lor \quad \left( \frac{G_{AB}}{N_A} \ge 0.10 \quad \land \quad G_{AB} \ge 3 \right)$$
* **Garantía Muestral (Elegibilidad):**
  Un invocador es elegible para este análisis solo si $N_A \ge 3$, eliminando el sesgo de aquellos con $\le 2$ partidas donde es matemáticamente imposible satisfacer la cota inferior.
* **Hallazgos Empíricos en Invocadores de Diamante ($N=211$ elegibles):**
  - **$77.3\%$ ($163$ invocadores)** juegan exclusivamente **SoloQ puro** ($0$ compañeros habituales).
  - **$20.9\%$ ($44$ invocadores)** tienen **$1$ compañero habitual** (DuoQ prioritario).
  - **$1.9\%$ ($4$ invocadores)** tienen **$2$ compañeros habituales** (rotación de dos dúos).
  - **$0.0\%$** tienen $\ge 3$ compañeros habituales.

### M. Arquitectura de Extracción Multi-Hilo, Paginación Precisa y Guardado por Lotes

Para maximizar el throughput de extracción sin violar las restricciones del Rate Limiter ni generar cuellos de botella de red o disco:
1. **Concurrencia HTTP con `ThreadPoolExecutor`:**
   * La latencia física transatlántica entre el host (LAS) y los servidores de Match-V5 (`americas.api.riotgames.com`) es de $\sim 180 - 250\text{ ms}$.
   * En ejecución monohilo, el RTT imponía un tope de $\sim 2\text{ partidas/s}$. Se implementó un pool de $4$ a $8$ trabajadores concurrentes (`workers = min(8, max(2, num_keys * 2))`).
   * La reserva de ranuras de petición se realiza de forma atómica y thread-safe mediante `threading.Lock()` en `RiotAPIClient._select_key_and_acquire`, garantizando equidad y respeto estricto del límite $20\text{ req}/1\text{s}$ y $100\text{ req}/120\text{s}$ por clave.
2. **Descarga Simultánea de Match DTO y Timeline DTO:**
   * Cada hilo despacha concurrentemente las dos peticiones necesarias por partida (`/matches/{id}` y `/matches/{id}/timeline`), solapando los tiempos de red y reduciendo la espera por partida en un **$45\%$**.
3. **Paginación Exacta (`start = existing_recorded_games`):**
   * Al consultar partidas de un jugador ya indexado, se envía el offset exacto `start=int(existing_count)`. Esto erradica el $100\%$ de llamadas a la API que devuelven IDs de partidas ya conocidas.
4. **Transacciones SQLite por Lotes (Batch Commits):**
   * Mediante `save_matches_and_timelines_batch(batch, batch_size=15)`, las partidas procesadas se acumulan en memoria y se insertan en bloques de $15$ en una única transacción atómica (`BEGIN...COMMIT`), reduciendo en un **$95\%$** las operaciones de sincronización a disco (`fsync`) en Windows NTFS.

### L. Purga de Remakes ($\le 300\text{ s}$), Cache Negativo y Ahorro del 95% en Peticiones
* **Problema Metodológico de Machine Learning:**
  Las partidas canceladas tempranamente (remakes, $\text{duración} \le 300\text{ s}$ / $5\text{ min}$) carecen de desarrollo competitivo, distorsionan los promedios de oro y CS, y generan ruido artificial en los vectores de arquetipo de los invocadores.
* **Limitación de Riot API:**
  El endpoint de listado `/matches/by-puuid/{puuid}/ids` únicamente entrega identificadores en bruto (`["LA2_...", ...]`) sin metadatos de duración o estado de rendición previa, haciendo forzosa la consulta inicial del Match DTO.
* **Optimización Dual Implementada (Ahorro del 95% de Desperdicio):**
  1. **Early Return en Timeline:** Al recibir el Match DTO, si `gameDuration <= 300`, el hilo cancela inmediatamente la petición de Timeline DTO, ahorrando el $50\%$ de las peticiones para ese remake al instante.
  2. **Cache Negativo Persistente (`discarded_matches`):** Se almacena el ID del remake en la tabla `discarded_matches (match_id PRIMARY KEY, reason)`. En `filter_unprocessed_match_ids`, se realiza una consulta `UNION` con `matches`. Cuando el crawler analiza a los otros 9 jugadores que disputaron ese remake, SQLite lo descarta en $0\text{ ms}$ en memoria local, con **cero peticiones adicionales a Riot**.
  3. **Resultado:** El costo total por remake desciende de hasta 20 peticiones a **exactamente 1**, protegiendo el cupo de la API y garantizando la pureza del dataset competitivo.

### M. Auditoría Forense de la Base de Datos y Puntos Débiles de Machine Learning
Tras auditar el dataset completo ($23,717$ partidas y $237,170$ participantes), se identificaron cinco consideraciones metodológicas críticas para el entrenamiento de los modelos:
1. **Sesgo Estructural de Lado (Red Side = $52.60\%$ vs Blue = $47.40\%$):**
   Riot asigna intencionalmente mayor MMR promedio al Lado Rojo en SoloQ para compensar la desventaja de perspectiva isométrica. Para evitar que el Modelo Campeones aprenda este sesgo como una ventaja intrínseca de los campeones, el pipeline de entrenamiento debe aplicar **Data Augmentation Simétrico** (duplicando cada partida invirtiendo Blue/Red con etiquetas invertidas) o incluir `is_blue_side` como covariable fija.
2. **Matriz Dispersa en Matchups 1v1 (Sparsity):**
   En TOP existen $5,697$ cruces únicos, pero el $41.7\%$ cuenta con $1$ sola partida registrada. Un clasificador supervisado con One-Hot Encoding puro sobreajustará severamente. Se requiere la adopción de **Embeddings de Campeones** basados en atributos de combate (rango, burst, sustain, CC) y **Suavizado Bayesiano Empírico (Laplace Smoothing con $C = 10$)** para regularizar emparejamientos infrecuentes:
   $$WR_{\text{ajustado}} = \frac{W + C \cdot WR_{\text{global}}}{N + C}$$
3. **Umbral Mínimo de Partidas en Vectores de Invocador (Modelo Jugadores):**
   De los $30,952$ perfiles en `players.csv`, el $65.0\%$ tiene $< 5$ partidas (compañeros/rivales pasivos). Para el clustering de arquetipos (K-Means/GMM), el conjunto de entrenamiento debe restringirse estrictamente a invocadores con **$\ge 15$ a $20$ partidas** ($> 3,000$ invocadores maduros).
4. **Truncamiento Temporal por Rendición Temprana (@20):**
   El $9.9\%$ de las partidas ($2,359$ partidas) concluyó antes del minuto 20 por rendición. El Modelo Campeones debe formularse en módulos temporales escalonados (Draft min 0, Early min 10 con $99.4\%$ de cobertura, Mid min 15 con $98.9\%$), sin exigir de forma obligatoria el snapshot @20 para predecir victorias tempranas.
5. **Duplicación Heurística de Posiciones por Riot ($2.53\%$):**
   En $599$ partidas, dos invocadores del mismo equipo comparten posición según la API de Riot (ej. doble TOP o doble MID por lane swaps). En el preprocesamiento de `matchups_1v1.csv`, deben aislarse para preservar biyecciones $1:1$ canónicas.

### N. Umbral Temporal de 50 Partidas en DFS y Dinámica de Parches sin Ruido Muestral (Sección 12 EDA)
* **Threshold Temporal de 50 Partidas:**
  Para evitar una concentración prematura en unos pocos jugadores con cientos de partidas, se configuró un umbral temporal plano de **$50$ partidas por jugador** (`TEMPORARY_DFS_TARGET = 50`) en todos los tiers. Esto asegura una masa crítica horizontal balanceada antes de activar los objetivos dinámicos adaptativos hacia $\approx 300$ partidas (`DYNAMIC_TIER_MATCH_TARGETS`).
* **Análisis Robusto de Metajuego y Winrate Shifts (Filtro $N \ge 80$ part/parche):**
  Para suprimir el ruido estadístico de selecciones esporádicas ($N < 40$), se aplicó un filtro estricto de **$N \ge 80$ partidas por parche evaluado** y **$\ge 400$ partidas acumuladas**, presente en al menos 4 de los 5 parches principales (`16.13` a `16.17`):
  - **Giros Críticos Robustos:** *Naafiri* (caída de $55.3\%$ a $40.3\%$, $\Delta -15.0\%$, $N=1,317$), *Swain* (caída a $40.0\%$, $\Delta -14.5\%$, $N=929$), *Leblanc* (ascenso a $55.1\%$, $\Delta +13.8\%$, $N=1,388$), *Singed* (caída a $47.9\%$, $\Delta -13.7\%$), *Shyvana* (caída de $63.3\%$ a $50.3\%$, $\Delta -13.0\%$), *Gangplank* (ascenso a $57.4\%$, $\Delta +12.5\%$).

### O. Campeones Menos Jugados por Línea: Nichos Competitivos (Sección 13 EDA)
Para aislar las selecciones reales de baja popularidad en Diamante+ de posibles lane swaps de 1 sola partida (ej. Blitzcrank Top o Yuumi Mid), se impuso la condición de **presencia de rol $\ge 20\%$**:
- **Carril Superior (Top):** *Heimerdinger* ($186$ part, $77.8\%$ en rol, $52.7\%$ WR), *Trundle* ($235$ part, $68.9\%$ en rol, $46.8\%$ WR), *Quinn* ($266$ part, $50.7\%$ en rol, $52.6\%$ WR), *Illaoi* ($273$ part, $81.0\%$ en rol, $49.8\%$ WR).
- **Jungla (Jungle):** *Rammus* ($169$ part, $93.9\%$ en rol, $48.5\%$ WR), *Amumu* ($218$ part, $86.5\%$ en rol, $53.2\%$ WR), *Sejuani* ($258$ part, $69.7\%$ en rol, $47.3\%$ WR), *Evelynn* ($263$ part, $80.2\%$ en rol, $48.7\%$ WR).
- **Carril Central (Mid):** *Aurelion Sol* ($280$ part, $35.8\%$ en rol, $48.6\%$ WR), *Annie* ($379$ part, $71.8\%$ en rol, $48.3\%$ WR), *Kassadin* ($397$ part, $91.3\%$ en rol, $50.4\%$ WR), *Malzahar* ($405$ part, $53.5\%$ en rol, $51.6\%$ WR).
- **Tirador (Bot ADC):** *Nilah* ($294$ part, $99.3\%$ en rol, $49.3\%$ WR), *Ziggs* ($390$ part, $35.7\%$ en rol, $48.2\%$ WR), *Kog'Maw* ($421$ part, $86.6\%$ en rol, $49.4\%$ WR), *Sivir* ($515$ part, $97.9\%$ en rol, $52.2\%$ WR).
- **Soporte (Support):** *Zyra* ($185$ part, $46.8\%$ en rol, $50.8\%$ WR), *Tahm Kench* ($326$ part, $54.6\%$ en rol, $48.5\%$ WR), *Renata* ($355$ part, $99.7\%$ en rol, $50.7\%$ WR), *Taric* ($362$ part, $94.3\%$ en rol, $52.2\%$ WR).
* **Visualización vigente (2026-09-04):** gráfico consolidado de barras apiladas con los **Bottom 10** campeones menos jugados y desglose proporcional por rol, con escala 100% dinámica (`xlim = max\_total \times 1.18`, offsets y umbrales relativos) para no recortar barras al crecer el dataset.

### P. Invocadores Hiper-Dominantes ($\ge 75\%$ WR) por ELO (Sección 14 EDA)
En la base de datos de $38,341$ invocadores con $> 50$ partidas de temporada disputadas, existen exactamente **$79$ jugadores ($0.21\%$)** que sostienen un winrate $\ge 75.0\%$:
- **Distribución por Tier:**
  $$\text{Diamond: } 62 \text{ jug. } (78.5\%) \quad|\quad \text{Master: } 13 \text{ jug. } (16.5\%) \quad|\quad \text{Grandmaster: } 3 \text{ jug. } (3.8\%) \quad|\quad \text{Challenger: } 1 \text{ jug. } (1.3\%)$$
- **Interpretación Competitiva:**
  El $78.5\%$ se concentra en Diamante, caracterizando el fenómeno de *smurfing* (cuentas de jugadores profesionales o elo superior escalando rápidamente). La frecuencia se reduce exponencialmente a medida que el MMR se equilibra en Master y GM, destacándose un único invocador en Challenger con **$76.7\%$ de winrate en $172$ partidas**, lo que representa una anomalía estadística de dominancia extrema en la liga.

### Q. Histograma de Distribución de Cambios en Winrate en Puntos Porcentuales (pp: 16.13 vs 16.17, Sección 15 EDA)
Se modeló la distribución del cambio de rendimiento individual $\Delta WR = WR_{16.17} - WR_{16.13}$ expresado estrictamente en **puntos porcentuales (pp)** para los $156$ campeones con muestra robusta ($N \ge 40$ partidas en ambos parches extremos):
- **Propiedades de la Distribución:**
  - **Media:** $\mu = -0.07\text{ pp}$ (suma cero macroscópica del sistema de puntuación).
  - **Mediana:** $+0.27\text{ pp}$.
  - **Desviación Estándar:** $\sigma = 6.82\text{ pp}$.
  - **Zona de Estabilidad ($|\Delta WR| \le 5\text{ pp}$):** El $60.3\%$ de los campeones experimentó fluctuaciones leves, lo que indica un núcleo del metajuego resiliente a los parches.
  - **Colas Anchas ($|\Delta WR| \ge 10\text{ pp}$):** El $13.5\%$ de los campeones sufrió desplazamientos bruscos, concentrando el impacto de buffs directos (*Quinn* $+21.8\text{ pp}$, *Aurelion Sol* $+20.5\text{ pp}$, *Kog'Maw* $+19.3\text{ pp}$, *Tahm Kench* $+16.7\text{ pp}$) y nerfs de balance (*Ziggs* $-19.8\text{ pp}$, *Maokai* $-17.1\text{ pp}$, *Naafiri* $-15.0\text{ pp}$, *Dr. Mundo* $-14.8\text{ pp}$).

### R. Garantías de Calidad del Dataset para el Modelo Jugadores (Auditoría Muse sobre 39,326 partidas, `players.csv` del 2026-09-03 01:11)
* **Vintage y coherencia:** `players.csv` con $41,048$ perfiles cumple $\sum \text{games\_recorded} = 393,260 = 10 \times 39,326$ exacto contra la BD viva post-reconciliación de alias. **Apex al 100%:** Challenger $200/200$ y Grandmaster $500/500$ con $\ge 50$ partidas ($16,756$ partidas migradas por PUUID dual, $0$ alias residuales).
* **Filtro de madurez formalizado:** Con corte $\text{games\_recorded} \ge 15$ quedan $6,893$ perfiles de entrenamiento; el $59.0\%$ ($24,285$) tiene $< 5$ partidas (confirma §3.F.3 de `AGENTS.md`: entrenar K-Means/GMM solo sobre maduros, resto solo inferencia).
* **Integridad de rangos:** $0$ nulos en las 7 métricas clave, $WR \in [0,100]$, $\%KP \in [0,100]$, $\text{damage\_share} \le 52.5\%$, $\text{gold\_share} \in [10.6, 35.3]\%$ (centrado en el $20\%$ teórico de $1/5$), $KDA \ge 0$, win global $50.0\%$ exacto.
* **Truncamiento ≠ pérdida:** Nulos de Timeline estrictamente por fin anticipado ($208$ partidas sin frame @10, $407$ sin @15, $3,134$ sin @20); $\text{AVG}_{SQL}$ ignora `NULL`, luego los vectores de agresividad temprana son insesgados.
* **Señal por tier (evidencia de poder discriminativo):** $\overline{WR}_{CHALL}=54.3\% > \overline{WR}_{GM}=53.2\% > \overline{WR}_{MST}=49.4\% > \overline{WR}_{DIA}=48.8\%$, con champion pool medio $21.4 > 19.5 > 11.6 > 9.7$ — gradiente monótono compatible con arquetipos separables.
* **Advertencia de leakage temporal (obligatoria para el entrenamiento):** Las 28 columnas son agregados sobre *todas* las partidas registradas, sin corte temporal. Para el Modelo Jugadores debe imponerse split out-of-time (vectores con partidas $< t$ predicen $\ge t$); de lo contrario el perfil "ve el futuro".

#### S. Base Estática para Embeddings de Campeones (Riot Data Dragon 16.17.1)
* **Artefacto:** `data/ddragon/16.17.1/champion.json` (159 KB, descargado directo del CDN oficial, sin API key).
* **Derivado tabular:** `data/ddragon/16.17.1/champions.csv` (20 KB, 173 filas × 29 columnas) con `champion_id` numérico (= `key`, joineable con `championId` de la BD), `tags` separados por `|`, `partype`, `info {attack, defense, magic, difficulty}` y las 20 `stats` base (se descartó solo `blurb/title/image/version`).
* **Cobertura:** 173 campeones con cruce 100% contra `SELECT DISTINCT champion_name` de la BD (173/173).
* **Atributos por campeón (matriz base $\approx 170 \times 20$):** `tags` (clase: Fighter/Tank/Mage/Assassin/Marksman/Support), `partype` (recurso), `info {attack, defense, magic, difficulty}` (escala 0–10 de Riot) y `stats` físicas (`attackrange`, `movespeed`, `armor`, `attackdamage`, `hp`, ...).
* **Uso diseñado (Fase A):** `StandardScaler` + PCA a 12D + normalización L2 → `data/champion_embeddings.csv`, representación densa compartida por el Modelo Jugadores (firma del champion pool por jugador) y el Modelo Campeones (reemplazo del One-Hot, que sobreajusta con 41.7% de cruces en $N=1$). Composición posicional: $\text{emb}(campeón, línea) = \text{emb}_{base} + \text{offset}_{línea}$.
* **Decisión de almacenamiento:** solo el JSON resumido; el tarball `dragontail-*.tgz` (~1 GB, incluye splash arts e íconos) no se necesita para ML tabular.

### T. Depuración y Estructuración de `players.csv` como Feature Store del Padrón Activo Diamante+
* **Diagnóstico de los Invocadores Extra:**
  Una auditoría sobre los $34,337$ invocadores no federados en la tabla `players` reveló que el **$99.5\%$ de sus apariciones ocurrían en lobbies de Master ($74.4\%$), Grandmaster ($19.6\%$) y Challenger ($6.6\%$)**, descartando que fuesen cuentas de ligas bajas (Platino/Esmeralda). Sin embargo, más de $10,200$ contaban con apenas $1$ partida disputada, distorsionando los vectores de estilo.
* **Adopción del Padrón Activo Diamante+:**
  En `export_to_csv.py`, se migró la consulta de `LEFT JOIN` a `INNER JOIN players pl ON p.puuid = pl.puuid`, consolidando un catálogo de **$23,583$ invocadores verificados de la liga** con al menos una partida registrada.
* **Nuevas Columnas de Especialización y Confianza:**
  1. `main_lane`: Rol dominante del jugador (`TOP`, `JUNGLE`, `MIDDLE`, `BOTTOM`, `UTILITY`), con distribución homogénea en la liga (~$4,300$ a $5,000$ por carril).
  2. `role_focus_pct`: Porcentaje de partidas disputadas en su línea principal.
  3. `confidence_tier`: Clasificación de madurez del perfil para el Modelo Jugadores:
     - `HIGH` ($N \ge 50$ part.): $2,247$ jugadores (núcleo para clustering de arquetipos).
     - `MEDIUM` ($15 \le N < 50$ part.): $5,855$ jugadores (entrenamiento y validación).
     - `LOW` ($3 \le N < 15$ part.): $9,603$ jugadores (inferencia en progreso).
     - `UNRELIABLE` ($N < 3$ part.): $5,878$ jugadores (muestra inicial).
  4. **Diferenciales Directos 1v1 vs Rival de Línea:** $\Delta\text{Gold@15}$, $\Delta\text{CS@15}$, $\Delta\text{CS End}$, $\Delta\text{Daño End}$, $\Delta\%\text{Gold Snowball}$.

### U. Cobertura Poblacional y Distribución de Partidas por ELO (Sección 2 EDA)
* **Objetivo Metodológico:**
  Cuantificar la penetración del crawling en el Padrón Activo Diamante+ ($23,583$ invocadores federados) mediante un desglose en tres escalones jerárquicos:
  1. **Apex (Grandmaster + Challenger):** $N = 1,391$ invocadores.
  2. **Master:** $N = 15,848$ invocadores.
  3. **Diamante:** $N = 6,344$ invocadores.
* **Métricas de Cobertura y Tendencia Central:**
  Para cada estrato se calculó la tasa de cobertura poblacional frente a tres umbrales de madurez estadística ($\ge 50$, $\ge 100$ y $\ge 200$ partidas) y las medidas de tendencia central sobre $G_i$ (partidas registradas por jugador):
  $$\text{Coverage}_k = \frac{\sum_{i \in \text{ELO}} \mathbb{I}(G_i \ge k)}{N_{\text{ELO}}} \times 100$$
  $$\mu_{\text{ELO}} = \frac{1}{N_{\text{ELO}}} \sum_{i \in \text{ELO}} G_i, \quad M_{\text{ELO}} = \text{mediana}(G_{\text{ELO}})$$
* **Resultados Empíricos en la Base de Datos (71,602 Partidas):**
  - **Apex:** Penetración masiva con el **$70.81\%$ ($985$ jugadores)** superando las 50 partidas, el **$19.48\%$ ($271$ jugadores)** con $\ge 100$ y $6$ jugadores con $\ge 200$. Media $\mu = 67.0$ partidas y mediana $M = 71$ partidas, confirmando saturación del escalón superior.
  - **Master:** Cobertura en expansión con **$1,252$ jugadores ($7.90\%$)** con $\ge 50$ partidas, **$155$ jugadores ($0.98\%$)** con $\ge 100$ y $4$ jugadores con $\ge 200$. Media $\mu = 19.0$ partidas y mediana $M = 12$ partidas.
  - **Diamante:** Fase inicial de muestreo horizontal sobre el padrón con **$10$ jugadores ($0.16\%$)** con $\ge 50$ partidas. Media $\mu = 3.3$ partidas y mediana $M = 2$ partidas.
* **Diseño Visual:**
  Implementación de una figura multi-panel de 3 columnas en `notebooks/eda_visualizer.ipynb` (reemplazando el Gráfico 2 anterior) con escala $Y$ adaptativa por sección para visualizar con fidelidad las colas de cada distribución, etiquetas numéricas bidimensionales ($\%\text{ y }N$) sobre cada barra y tarjetas superiores con las estadísticas descriptivas ($\text{Población}, \mu, M$).

### V. Sistema de Identidades Multi-Proyecto (PUUIDs por API Key Unificados)
* **Causa raíz formal:**
  Riot encripta el PUUID como $\text{PUUID}_{proy} = \text{HMAC}(\text{puuid\_global}, \text{project\_secret})$. Cada proyecto de Riot Developer tiene su propio `project_secret` invariante (sobrevive a la rotación de 24h de la key), luego el mismo humano presenta cadenas PUUID totalmente distintas bajo cada key del pool. Como la PK de `players` y de `match_participants` es el PUUID crudo, el `sync_players` con el Proyecto 2 duplicó el padrón (Challenger $200 \to 400$, GM $500 \to 1.000$, Master $8.259 \to 16.545$) y fragmentó historiales entre proyectos ($189.162$ filas P1 vs $226.086$ P2).
* **Identidad verdadera:**
  $\text{canonical\_id} = \text{lower}(\text{gameName}\#\text{tagLine})$ verificado vía Account-V1, único invariante estable entre proyectos. El **proyecto 0** (posición 0 del pool `RIOT_API_KEYS`) es el canónico: su PUUID es el que se persiste.
* **Esquema (tablas nuevas en `data/lol_las.db`):**
  1. `api_projects(project_index, key_hint, first_seen, last_seen)`: registro del pool; solo hint enmascarado, nunca la clave. El índice = posición en la lista (estable ante rotación).
  2. `player_identities(canonical_id PK, game_name, tag_line, tier, rank, league_points, wins, losses)`: un registro por jugador verdadero.
  3. `player_puuid_map(canonical_id, project_index, puuid UNIQUE, summoner_id)`: exactamente **un PUUID por (jugador, proyecto)**. Invariante verificable: $|\text{PUUIDs}(c)| = |\text{api\_projects}| \ \forall c$ (método `get_identity_coverage()`).
* **Costo cero en el crawler (condición del usuario):**
  La resolución costosa (Account-V1) vive exclusivamente en el script batch `scripts/identity/register_key_project.py`. El hot path (`collect_matches`, batch saves) solo hace lookup $O(1)$ en memoria (`_translation_cache` alias $\to$ canónico, cargado una vez por sesión): **cero requests extra a Riot durante el crawling**. Todo PUUID entrante se traduce al canónico antes del `INSERT`, por lo que la BD se auto-sanea hacia adelante.
* **Unificación en estadísticas y datasets:**
  `export_players` agrupa por $\text{COALESCE}(\text{canonical\_id}, \text{puuid})$ (partición exacta: $\sum \text{games\_recorded}$ sigue igualando a las filas de `full_dataset.csv`) y expone `canonical_id` + `num_puuids` (verificable contra $|\text{api\_projects}|$). `export_full_dataset` agrega columna `canonical_id`. `get_coverage_stats` y el deep crawl usan la variante canónica (`COUNT(DISTINCT match_id)` anti fan-out) con fallback legacy si el mapa está vacío. Los diferenciales vs rival de `matchups_1v1` se traducen a canónico antes de promediar por jugador.
* **Alta dinámica de keys:**
  `python scripts/identity/register_key_project.py [--api-keys K1,K2] [--tiers ...] [--include-diamond] [--dry-run]`: registra el pool, aplica fingerprint (Jaccard sobre muestra Challenger: $\approx 1$ = rotación del mismo proyecto, $\approx 0$ = proyecto nuevo, con aviso de reordenar el pool), recorre cada jugador en cada proyecto (los ya conocidos se omiten sin requests) y reporta cupos + invariante. Todo warn-only, nunca bloquea.
* **Cupos oficiales del padrón LAS** (`DatabaseManager.EXPECTED_ROSTER_SIZES`, solo advertencia): Challenger $= 200$, Grandmaster $= 500$ (exactos, sin tolerancia), Master $\approx 8.259$, Diamond $\approx 31.653$ (referenciales). Un `actual > expected` señala duplicación multi-proyecto pendiente de resolución.

### W. Política de Nube para las DBs (Drive): Escritor Único, Reemplazo Atómico, Versionados y Restore
* **Regla de oro (last-writer-wins):** Google Drive no tiene locks. Solo la máquina recolectora escribe (`backup_to_drive.py` vía pipeline o pre-push); cualquier otra máquina/agente solo descarga y abre en `mode=ro` con nombre distinto.
* **Lo que ya era seguro:** el backup usa la API `sqlite3.backup()` a un snapshot temporal y recién después copia a Drive (consistente aun con el crawler escribiendo), y copia solo `lol_las.db` (nunca `-wal`/`-shm` parciales).
* **Endurecimiento y Reemplazo Atómico (`safe_drive_copy`):**
  1. `PRAGMA quick_check` antes de subir: si falla, **no se sube nada**.
  2. **Reemplazo Atómico Anti-Truncamiento:** En filesystems virtuales (como Google Drive Desktop en Windows), escribir directamente con `shutil.copyfile` sobreescribiendo un archivo abierto en `'wb'` trunca el archivo a 0 bytes si la subida se interrumpe o si el indexador de Drive bloquea el archivo (`[WinError 32]`). `safe_drive_copy` ahora escribe primero en un archivo temporal oculto (`.{stem}_upload_{pid}.tmp`), valida que `tmp_size == src_size`, y luego ejecuta un reemplazo atómico (`os.replace`) con reintentos para sustituir el archivo en Drive sin riesgo alguno de corrupción. En `finally`, cualquier residuo temporal se destruye.
  3. Manifiesto `lol_las.manifest.json` (+ uno por snapshot) con `{matches, players, map_rows, projects, patch_min/max, db_size_mb, created_at}` como provenance mínima.
  4. Snapshots versionados inmutables `snapshots/lol_las_YYYYMMDD_HHMM_m<M>.db`: se crea uno nuevo si el último tiene $\ge 7$ días (`BACKUP_VERSIONED_DAYS`) o el padrón creció $\ge 10.000$ partidas (`BACKUP_VERSIONED_MATCHES`); se conservan los últimos $6$ (`BACKUP_VERSIONED_KEEP`) podando el resto.
  5. `scripts/backup/restore_from_drive.py`: flujo inverso con respaldo pre-restore (`local_pre_restore_backup`, parametrizable con `local_backups_dir` para que los tests corran aislados en `tmp_path` sin ensuciar el disco local) y **rollback automático** si el `quick_check` post-copia falla; flags `--list`, `--snapshot`, `--dest`, `--dry-run`.
  6. El pipeline nocturno sondea el mapa multi-proyecto en RO y activa `--require-canonical` solo cuando ya hay identidades (estricto con mapa, legacy avisado sin mapa).

### X. Firmas de Jugador por Embeddings de Campeón-Línea y Nombres Dinámicos de Cluster (Modelo Jugadores)
* **Embedding v1 (único vigente, 22D):** cada par (campeón, línea) con $\ge 100$ partidas en `champions_by_lane.csv` se vectoriza con sus 22 promedios de gameplay estandarizados (`StandardScaler`): KDA, CS/oro/XP/daño @10/@15/@20, daño total, daño recibido, oro ganado, CS total, visión y `first_tower_rate_pct` (se excluye `first_blood_rate_pct` por ruidosa según ANOVA + ablación). Validación en `notebooks/embeddings.ipynb` (5 celdas v1, sin escritura): analogía por traslación de línea $emb(champ,A) - \vec{A} + \vec{B} \approx emb(champ,B)$ con error medio $1.93$ vs $5.13$ sin trasladar y $6.12$ entre pares cualesquiera ($\approx 32\%$ de una distancia típica).
* **Variantes descartadas (celdas v2/v3 borradas del notebook):** v2 (+42 stats base de Data Dragon $\to$ 65D, error $2.75$) y v3 (+12 físicas con escalado $\to$ 35D, error $2.34$). Los atributos estáticos de campeón introducen offsets gruesos que rompen la geometría de comportamiento; el gameplay puro gana y es lo único que se mantiene.
* **Firmas (`src/features/player_signatures.py`, sin red ni BD):** la firma es el centroide (media) de los embeddings de los campeones jugados, más `pool_dispersion` (distancia media al centroide: mide amplitud **y** diversidad real del pool, reemplazando a `unique_champions_count`):
  $$\text{sig}(p, \ell) = \frac{1}{N_{p,\ell}} \sum_{g \in G_{p,\ell}} emb(\text{champ}_g, \ell), \qquad \text{disp}(p,\ell) = \frac{1}{N} \sum_g \lVert emb_g - \text{sig} \rVert_2$$
  - **Variante A (global):** todo el historial del jugador (31,356 firmas).
  - **Variante B (posicional):** solo partidas de **esa** línea, con elegibilidad estricta $N_{p,\ell} \ge 10$ (`MIN_LANE_GAMES = 10`): sin 10+ partidas en el carril no hay embedding fiable (70,471 firmas, 15,370 elegibles). La Sección 17 del EDA filtra además `pool_dispersion` no elegible antes de clusterizar.
* **Nombres de cluster 100% derivados de los datos (cero etiquetas fijas):** cada cluster se bautiza con su variable más distintiva = mayor $|z|$ del centroide respecto a la media poblacional, con asignación única greedy (los clusters se ordenan por $|z|$ máximo descendente y cada uno toma su mejor variable aún libre) y color por id:
  $$z_{cj} = \frac{\mu_{cj} - \mu_j}{\sigma_j}, \qquad \text{nombre}(c) = \text{``}C{c} \cdot \arg\max_j |z_{cj}|\ (z{:}{+}.\text{1f}\sigma)\text{''}$$
  Los nombres se regeneran en cada corrida (Secciones 16 y 17 del EDA); el heatmap de perfiles $z$ conserva la lectura táctica completa sin hardcodear arquetipos.

---

## 5. Estructura del Código Fuente

La organización sigue un criterio de **separación por responsabilidad**: `src/` es
el paquete importable con la lógica de dominio; `scripts/` agrupa utilidades
ejecutables por propósito; `tools/` aísla el diagnóstico manual interactivo (que
NO debe ser recogido por pytest); `notebooks/` contiene el análisis.

```
.
├── .env                       # Configuración local (RIOT_API_KEYS, ROUTING, PLATFORM) — no versionado
├── .env.example               # Plantilla de configuración
├── requirements.txt           # Dependencias declaradas por bloque funcional
├── pyproject.toml             # Config de pytest (testpaths = tests)
│
├── ROADMAP.md                 # Pregunta de investigación y etapas del trabajo
├── PROJECT_SUMMARY.md         # Este documento: memoria técnica
├── AGENTS.md                  # Reglas operativas y changelog de agentes
├── README.md                  # Instalación y uso
│
├── run.py                      # Punto de entrada único: crawler + validación oportunista (§9.N)
├── collect_data.py            # CLI principal de recolección
├── export_to_csv.py           # Exportador de SQLite a los 4 CSVs estructurados para ML
│
├── data/                      # Generado, no versionado
│   ├── lol_las.db             #   Base de datos SQLite (fuente de la verdad)
│   ├── full_dataset.csv       #   Dataset maestro (Partidas + Timeline + Contexto Equipo)
│   ├── players.csv            #   Feature Store de vectores de estilo del Padrón Activo
│   ├── champions_by_lane.csv  #   Rendimiento agregado por campeón en cada rol
│   └── matchups_1v1.csv       #   Enfrentamientos 1v1 con diferenciales (Modelo Campeones)
│
├── src/                       # Paquete de dominio (importable)
│   ├── config.py              #   Constantes (QUEUE_SOLOQ_ID=420, rutas, pool de keys)
│   ├── rate_limiter.py        #   Token-bucket para límites 20/1s y 100/120s
│   ├── riot_client.py         #   Cliente HTTP multi-key con auto-retry 429/5xx y paginación
│   ├── database.py            #   Schema SQLite y deduplicación estricta
│   ├── collector.py           #   Orquestador del crawler (depth_first + corte por tiempo o señal)
│   ├── validation.py          #   Auditoría incremental solo-lectura del dataset (§9.N)
│   ├── features/              #   Ingeniería de features derivadas
│   │   └── player_signatures.py   # Firmas de jugador (variantes A macro y B posicional)
│   └── models/baselines/      #   Baselines (§9.S.20): draft_rules, draft_logreg, identificacion
│
├── scripts/                   # Utilidades ejecutables, agrupadas por propósito
│   ├── collection/
│   │   └── auto_nightly_pipeline.py   # Corrida autónoma + export + commit + push
│   ├── identity/                      # Resolución de PUUIDs multi-proyecto (§8)
│   │   ├── register_key_project.py    #   Registro de proyectos y mapeo por Account-V1
│   │   ├── migrate_dead_puuids.py     #   Unión por solapamiento de historial
│   │   └── resolve_inverse_by_match.py#   Resolución inversa bottom-up por partida
│   ├── backup/
│   │   ├── backup_to_drive.py         # Snapshot atómico + versionados + manifiesto
│   │   └── restore_from_drive.py      # Flujo inverso con respaldo pre-restore
│   └── ops/                           # Diagnóstico y observabilidad
│       ├── check_api_keys.py          #   Vigencia del pool de claves (200/401/403/429)
│       ├── monitor_live.py            #   Monitor de telemetría en vivo (solo lectura)
│       ├── identity_progress.py       #   Avance de la resolución de identidades (solo lectura)
│       └── lookup_player.py           #   Consulta de jugador por PUUID o Riot ID
│
├── notebooks/                 # Análisis (versionados SIN outputs)
│   ├── eda_visualizer.ipynb   #   EDA completo
│   ├── clustering_jugadores.ipynb #   Clustering de arquetipos de jugadores
│   └── embeddings.ipynb       #   Embeddings campeón-línea (v1, 22D)
│
├── tools/                     # Diagnóstico manual interactivo — NO son tests
│   └── check_riot_api.py      #   Validación de conectividad y API Key
│
├── tests/                     # Suite pytest
│   ├── conftest.py            #   Bootstrap de sys.path + fixture temp_db compartida
│   ├── test_crawler.py        #   Deduplicación, BD, rate limiter, pool multi-key
│   ├── test_migration.py      #   Migración legacy, alias, dedup y poda
│   ├── test_identity.py       #   Union-find de identidades y player_id
│   ├── test_validation.py     #   Motor de auditoría y planificador oportunista
│   ├── test_key_pool.py       #   Rotación de claves, pool agotado y señal de corte
│   ├── test_cloud_backup.py   #   Snapshot, versionado y restauración
│   └── test_signatures.py     #   Firmas de jugador (variantes A y B)
│
└── docs/archive/              # Bitácoras históricas de coordinación multi-agente
```


---

## 6. Guía de Comandos Rápidos

### A. Ejecutar Pruebas Unitarias
```bash
python -m pytest tests/
```

### B0. Sesión Completa: Recolectar y Auditar a la Vez (recomendado)
```bash
# 4 horas de crawler; la auditoría del dataset corre dentro del cooldown del rate limit
python run.py --minutos 240

# Ciclos indefinidos (Ctrl+C corta y deja los dos reportes de cierre)
python run.py --ciclos 0 --minutos 60

# Auditar sin red lo que ya está bajado (~11 s sobre 98.000 partidas, solo lectura)
python run.py --solo-validar

# Reauditar la base entera desde cero, con cobertura de identidad incluida
python run.py --solo-validar --reiniciar-cursor --identidad
```

### B. Recolectar Partidas en Modo Profundidad (Para Modelo de Arquetipos)
```bash
# Descargar hasta 100 partidas por jugador durante 60 minutos:
python collect_data.py --collect-matches --matches-per-player 100 --duration-minutes 60

# Descargar partidas enfocándose exclusivamente en Master+:
python collect_data.py --collect-matches --matches-per-player 100 --tier MASTER --duration-minutes 60
```

### C. Regenerar Manualmente los 4 Datasets en CSV
```bash
python export_to_csv.py
```

### D. Ver Resumen del Estado de la Base de Datos
```bash
python collect_data.py --summary
```

### E. Abrir el Visualizador de Datos (Jupyter Notebook)
Abrir [`notebooks/eda_visualizer.ipynb`](notebooks/eda_visualizer.ipynb) en VS Code o JupyterLab y seleccionar el kernel **`Python 3.11 (Proyecto Final)`**.

---

## 7. Próximos Pasos: El Gran Enfrentamiento (Modelo Jugadores vs Modelo Campeones)

### A. Definición de la Hipótesis Central de Machine Learning
El objetivo del proyecto de investigación es responder una pregunta fundamental de analítica en deportes electrónicos:
$$\text{¿Qué determina con mayor precisión el resultado de una partida competitiva de League of Legends?}$$
Ambos modelos compiten prediciendo **EXACTAMENTE EL MISMO TARGET** ($Y = \text{Victoria Blue/Red}$ en Draft o ventaja $\Delta\text{Gold@15}$):

1. **Modelo Jugadores (Enfoque de Invocadores / Hand Skill & Playstyle):**
   - **Entrada ($X$):** Características históricas, perfiles de dominancia, estilo de juego y **arquetipos tácticos de los 10 jugadores humanos**, a ciegas de las sinergias abstractas de campeones.
   - **Hipótesis:** El factor humano, la especialización de rol y la agresividad individual superan las ventajas teóricas del draft en SoloQ.

2. **Modelo Campeones (Enfoque de Draft / Meta & Matchups):**
   - **Entrada ($X$):** Composición de los 10 campeones, sinergias intra-equipo, escalado, rango, y probabilidades teóricas de los duelos 1v1 en línea (`champions_by_lane.csv` y `matchups_1v1.csv`), a ciegas de quién está piloteando el teclado.
   - **Hipótesis:** Las ventajas estadísticas y los counters inherentes del diseño de campeones dictan el destino de la partida por encima de las individualidades.

3. **Ensamble Híbrido (Modelo Jugadores + Modelo Campeones):**
   - Modela la interacción real: un campeón fuerte en manos de un especialista vs un counter en manos de un jugador de weakside.

---

### B. Estudio de Ablación para la Representación en el Modelo Jugadores (Variantes A, B y A+B)
Para alimentar el clasificador supervisado del Modelo Jugadores (XGBoost / LightGBM / Red Neuronal), se implementará un estudio comparativo de representación de perfiles:

1. **Variante A (Clustering Macrotáctico General - Nivel Ecosistema):**
   - Agrupa a todos los jugadores en un espacio vectorial táctico puro de 15 dimensiones (14 stats + `pool_dispersion` del embedding: `role_focus_pct`, `%Gold Share`, `Vision Score`, $\text{GD@15}$, $\Delta\text{CS Rival End}$, $\Delta\text{Damage Rival End}$, $\Delta\%\text{Gold Snowball}$, etc.).
   - Asigna a cada invocador uno de los 5 clusters macro con **nombre dinámico** = su variable más distintiva ($|z|$ máximo del centroide, §4.X), sin arquetipos semánticos fijos.
   - Mide la **filosofía económica y de recursos** del jugador en la partida.

2. **Variante B (Clustering Posicional por Línea - Nivel Micro/Carril):**
   - Agrupa a los jugadores **exclusivamente contra sus pares de la misma línea** evaluando solo las partidas disputadas en ese carril, con madurez posicional $N \ge 10$ partidas y $P \ge 15\%$ en el carril (firma = promedio de $emb(\text{champ}, \text{línea})$ sobre esas partidas).
   - Cada rol produce 3 clusters con **nombre dinámico** = su variable más distintiva dentro del rol (p. ej. el cluster dominado por $\text{GD@15}$ se bautiza solo), sin sub-arquetipos hardcodeados.
   - Mide la **función técnica y espacial** en su carril.

3. **Variante A + B (Ensamble Jerárquico Bi-Nivel):**
   - Cada uno de los 10 jugadores en el lobby se representa combinando ambas etiquetas:
     $$\text{Vector Jugador}_i = [\dots, \text{Arquetipo General}_i, \text{Arquetipo Posicional}_i, \dots]$$
4. **Evaluación Académica:** Se presentará una comparativa formal de métricas (Accuracy, ROC-AUC, F1-Score, Brier Score) entre la Variante A, la Variante B y la Variante A+B frente al Modelo Campeones.

---

### C. Benchmark Externo de la Industria: Baseline Heurístico con OP.GG MCP

En la evaluación rigurosa de modelos de Machine Learning aplicados a deportes electrónicos, no basta con contrastar los clasificadores contra un modelo trivial (como `DummyClassifier` o predicción por clase mayoritaria). Para validar el valor agregado real de las representaciones vectoriales densas (embeddings de combate de campeones y arquetipos tácticos de invocadores), se incorporará como **Baseline Heurístico de la Industria** el servicio de analítica de **OP.GG** a través de su protocolo estandarizado **OP.GG MCP (`https://mcp-api.op.gg/mcp`)**.

#### 1. Fundamento Académico del Baseline Externo
OP.GG representa el estándar de facto comercial en la comunidad competitiva de League of Legends. Sus predicciones y evaluaciones se basan en heurísticas deterministas consolidadas:
* **Score de Rendimiento Propietario (*OP Score*):** Calificación en escala 0–10 basada en métricas finales ponderadas de KDA, daño relativo, oro y control de visión.
* **Tier Promedio del Lobby (*Average Tier Info*):** Estimación agregada del nivel del emparejamiento (ej. Master / GM).
* **Guías de Enfrentamiento de Línea (*Lane Matchup Guides*):** Probabilidades empíricas basadas en winrates globales tabulares de millones de partidas.

#### 2. Metodología de Comparación (Benchmark Tripartito)
Para una muestra de test out-of-time, se contrastarán 3 niveles de predicción:
$$\text{Baseline Heurístico (OP.GG)} \quad \text{vs} \quad \text{Modelo Campeones (Draft Embeddings)} \quad \text{vs} \quad \text{Modelo Ensamble (Draft + Arquetipos)}$$

* **Hipótesis del Benchmark:** Si el *Modelo Jugadores* y el *Ensamble Híbrido* superan tanto a la heurística de OP.GG como al *Modelo Campeones* en log-loss calibrado y ROC-AUC, se demuestra científicamente que los arquetipos de comportamiento de los 10 individuos contienen información predictiva ortogonal que las métricas superficiales de la industria omiten por completo.

#### 3. Auditoría de Integración y Aislamiento Técnico
Se evaluó empíricamente la conexión directa contra el servidor MCP de OP.GG mediante JSON-RPC 2.0 (`lol_get_summoner_game_detail`, `lol_list_summoner_matches`):
---

## 8. Arquitectura Canónica Multi-Proyecto y Padrón Activo Diamante+ (Opción A)

### A. Diagnóstico y Resolución del Desdoblamiento ("BaanShee Split")
Durante la auditoría forense sobre el dataset de invocadores (`data/players.csv`), se identificó una anomalía estructural donde jugadores federados aparecían bifurcados en múltiples filas (ej. *BaanShee* con 171 y 156 partidas en registros separados).

* **Causa Raíz 1 (Vista Canónica Asimétrica):** La vista `v_puuid_canonical` traducía aliases secundarios a su clave principal (`alias_puuid -> primary_puuid`), pero carecía de la identidad reflexiva (`primary_puuid -> primary_puuid`) para cuentas que no figuraban en `player_puuid_map` (cuentas de Diamante cargadas en `player_puuid_aliases`). En consecuencia, partidas disputadas con la clave viva caían en el PUUID en crudo, mientras partidas antiguas resolvían a su Riot ID.
* **Causa Raíz 2 (Agrupamiento por Riot ID Dinámico):** La consulta en `export_players()` agrupaba por `canonical_id` (Riot ID `game_name#tag_line`). Si un jugador cambiaba su nombre de invocador a lo largo de la temporada, la agregación SQL generaba múltiples clusters para el mismo individuo.

### B. Formalización de la Vista Biyectiva y Agrupación Canónica
Se redefinió formalmente `v_puuid_canonical` garantizando la propiedad de biyección estricta:
$$\forall p \in \text{PUUIDs}, \quad \exists ! \, c \in \text{Canonical PUUIDs} : v(p) = c$$

Incorporando el mapeo reflexivo condicional:
$$\text{SELECT } a.\text{primary\_puuid}, a.\text{primary\_puuid}, \dots \quad \text{WHERE NOT EXISTS in } player\_puuid\_map$$

Y en `export_players()`:
$$\text{GROUP BY } \text{COALESCE}(v.\text{canonical\_puuid}, p.\text{puuid})$$

### C. Implementación de la Opción A (Purificación del Padrón Activo)
Para preservar la máxima validez estadística del **Modelo Jugadores (Arquetipos)**, se implementó un `INNER JOIN players pl ON pl.puuid = COALESCE(v.canonical_puuid, p.puuid)`.
* **Propósito:** Filtrar a los rivales casuales transeúntes (`UNKNOWN`) que disputaron partidas esporádicas pero no pertenecen a la población federada de Diamante+.
* **Resultado del Padrón:**
  - Invocadores únicos en `players.csv`: **20,896** (0 duplicados de PUUID).
  - Tiers canónicos: Challenger: **200/200**, Grandmaster: **500/500**, Master: **8,296**, Diamond: **11,900**.
  - Invocadores `UNKNOWN`: **0**.
  - Cobertura $\ge 50$ partidas: **5,719 perfiles estadísticamente maduros**.

### D. Generalización de Clusters Tácticos a la Población Madura ($\ge 50$ Partidas)
Inicialmente, el análisis de clustering estilístico del **Modelo Jugadores** se restringió metodológicamente al estrato Apex (Challenger + Grandmaster, $N \approx 700$) como prueba de concepto de máxima pureza. Una vez alcanzado el volumen de 98,167 partidas y la unificación canónica de identidades, se generalizó la muestra a **toda la población madura con $\ge 50$ partidas disputadas** ($N = 5,719$ invocadores: 199 Challenger, 500 Grandmaster, 4,603 Master y 417 Diamante), sin restricción de elo.
* **Justificación:** Con $N \ge 50$, cada vector de 15 dimensiones alcanza estabilidad asintótica por la Ley de los Grandes Números, garantizando que los arquetipos tácticos descubiertos reflejen el espectro conductual completo de la alta competencia y no únicamente los sesgos de la cima de Apex.
* **Resultados de Estructura:** La estadística de Hopkins se mantiene en $H = 1.0000$, confirmando una estructura de conglomerados natural y extrema sobre los 5,719 vectores.


---

## 9. Auditoría y Depuración Integral del Repositorio (2026-09-07)

Antes de entrar en la **Etapa 6** (entrenamiento y enfrentamiento de modelos), se
ejecutó una auditoría estática completa del repositorio. El criterio rector fue
que un repositorio que va a sostener un informe académico debe ser *navegable y
verificable*: cada archivo con un propósito localizable, cero código muerto y una
suite que se pueda correr sin efectos colaterales.

### A. Metodología de la Auditoría

Se construyó un analizador basado en el AST de Python (`ast.walk`) que recorre
todos los módulos y cruza tres señales:

1. **Imports no usados:** se recolectan los nombres ligados por `Import` /
   `ImportFrom` y se contrastan contra los `ast.Name` y las raíces de
   `ast.Attribute` efectivamente referenciados en el módulo.
2. **Definiciones nunca referenciadas:** cada `FunctionDef` / `ClassDef` de nivel
   superior se busca en el corpus completo (`.py` + celdas de `.ipynb`); si el
   conteo de apariciones no supera al número de definiciones, es código muerto.
3. **Definiciones duplicadas:** mismo identificador definido en más de un módulo.

La suite (`46 passed`) se usó como *baseline* de regresión: se ejecutó antes de
la primera modificación y después de cada paso, exigiendo el mismo resultado.

### B. Defectos Corregidos

1. **Script inejecutable con ruta absoluta ajena.**
   `scripts/resolve_player_aliases.py` insertaba en `sys.path` la ruta
   `c:\Users\DarkA\OneDrive\...`, correspondiente a otra máquina. El módulo era
   incapaz de importar `src.*` en cualquier entorno distinto del original.
   Además, su propio sucesor lo declaraba metodológicamente inferior: el
   *docstring* de `resolve_inverse_by_match.py` describe el enfoque top-down
   (partir del jugador fresco y buscar solapamiento) como **"ciego con
   inactivos"**, y lo reemplaza por la resolución inversa bottom-up. El script
   fue **eliminado**; el historial de Git lo preserva.

2. **Colisión entre herramienta de diagnóstico y suite automatizada.**
   `test_riot_api.py` residía en la raíz y definía `test_platform_status()` y
   `test_account_and_summoner()`, pero **no es un test**: realiza llamadas de red
   reales y bloquea en `input()` esperando un Riot ID. Un `pytest` sin argumentos
   recogía 48 tests en lugar de 46, incorporando dos funciones interactivas.
   Solución en tres frentes: se movió a `tools/check_riot_api.py`, se renombraron
   sus funciones al prefijo `check_*`, y se fijó `testpaths = ["tests"]` en
   `pyproject.toml`. La barrera es ahora redundante por diseño.

3. **Ruta de base de datos dependiente del directorio de trabajo.**
   `monitor_live.py` definía `DB_PATH = Path("data/lol_las.db")`, relativa al CWD:
   el monitor solo funcionaba si se lo invocaba desde la raíz. Se ancló a
   `Path(__file__).resolve().parents[2]`, haciéndolo invariante al CWD.

4. **`GOOGLE_DRIVE_BACKUP_PATH` documentado como URL.**
   `.env.example` proponía una URL de `drive.google.com`. La variable se consume
   como ruta de sistema de archivos (`Path(...).exists()` y `shutil.copy2`), que
   no habla HTTP: con una URL, `backup_to_drive.py` cae siempre por la rama de
   error y **nunca sincroniza**, sin fallar ruidosamente. Se corrigió el valor y
   se documentó explícitamente la restricción.

5. **Dependencias no declaradas.** `requirements.txt` omitía `numpy`, `rich`,
   `matplotlib`, `seaborn`, `scipy`, `scikit-learn` y `jupyter`, todas importadas
   por el monitor en vivo, los notebooks o los tests. Se reescribió declarando
   cada bloque funcional y verificando la resolución de cada import.

### C. Eliminación de Duplicación

* **`fetch_account_identity` triplicada.** Tres scripts de `scripts/identity/`
  implementaban la misma resolución Account-V1 construyendo la URL a mano y
  llamando al método **privado** `client._request`, rompiendo el encapsulamiento
  del cliente. La lógica se promovió al contrato público
  `RiotAPIClient.get_account_by_puuid()`, único lugar donde vive ahora la forma
  del endpoint. Los *stubs* de test se migraron al método público, con lo que
  dejaron de acoplarse al parseo de cadenas de URL.

* **Bootstrap de `sys.path` repetido nueve veces.** Se centralizó el de los tests
  en `tests/conftest.py`.

* **Fixture `temp_db` duplicada** en `test_crawler.py` y `test_migration.py`
  (con nombres de archivo distintos, `test_lol.db` y `t.db`). Unificada en
  `conftest.py`.

* **Quince imports no usados** eliminados. El analizador reporta ahora **cero**.

### D. Reorganización por Responsabilidad

La raíz acumulaba módulos de dominio, entradas CLI, herramientas de diagnóstico y
notebooks sin distinción. La nueva jerarquía (§5) separa cuatro responsabilidades:
`src/` (dominio importable), `scripts/` (ejecutables agrupados por propósito:
`collection`, `identity`, `backup`, `ops`), `tools/` (diagnóstico manual, aislado
de pytest) y `notebooks/` (análisis). Los `ROOT_DIR` de los scripts desplazados se
recalcularon de `parent.parent` a `parents[2]`, y todo el módulo se verificó con
un *import check* explícito de los 14 módulos del paquete.

### E. Peso del Repositorio

`data_visualizer.ipynb` pesaba **6,31 MB**, de los cuales prácticamente la
totalidad eran 18 imágenes PNG embebidas en base64 dentro de los `outputs`. Cada
re-ejecución del notebook producía un *diff* de megabytes ilegible. Se removieron
los outputs y los `execution_count`, preservando íntegramente las 36 celdas de
código: **6,31 MB → 96,7 KB** (reducción del 98,5 %). Los gráficos se regeneran
ejecutando el notebook.

### F. Retiro del Protocolo Multi-Agente

El repositorio arrastraba tres bitácoras de coordinación entre dos agentes
(`TASK_DISPATCH.md`, `COMMUNICATION.md`, `COMMUNICATION_ARCHIVE.md`, 681 líneas)
que imponían lectura obligatoria por turno, pases por escrito y doble
autorización. Ese modo de trabajo ya no está vigente. Las bitácoras se movieron a
`docs/archive/` como registro histórico y `AGENTS.md` §1 declara explícitamente el
**modo agente único**, reduciendo la lectura obligatoria por turno a `ROADMAP.md`,
`AGENTS.md` y `PROJECT_SUMMARY.md`.

### G. Documento de Hoja de Ruta

Se incorporó **`ROADMAP.md`** para cubrir un vacío: la pregunta de investigación y
el estado de avance estaban dispersos entre secciones de `PROJECT_SUMMARY.md` y
`AGENTS.md`, sin una vista única. El nuevo documento enuncia la hipótesis central,
los tres modelos en competencia, la justificación del dominio (Diamante+ en LAS) y
las **siete etapas** del trabajo con su estado, más los cinco riesgos
metodológicos ya auditados.

### H. Verificación

| Control | Antes | Después |
| :--- | :--- | :--- |
| Suite `pytest` | 46 passed | **46 passed** |
| Tests recogidos por `pytest` sin argumentos | 48 (2 interactivos) | **46** |
| Imports no usados | 15 | **0** |
| Funciones duplicadas entre módulos | 1 (×3 copias) | **0** |
| Módulos que importan correctamente | no verificable (deps faltantes) | **14/14** |
| Tamaño del notebook de EDA | 6,31 MB | **96,7 KB** |

### I. Refactor de la Documentación para Consumo por Agentes (2026-09-07)

Los documentos que un agente de IA lee cada turno son infraestructura, no prosa:
lo que se pone en ellos compite por la ventana de contexto con el problema que se
intenta resolver. Se auditaron `CLAUDE.md`, `AGENTS.md` y `ROADMAP.md` bajo ese
criterio.

**Diagnóstico — carga de contexto.** `CLAUDE.md` §1 ordenaba *"DEBES LEER"* tres
documentos completos antes de cualquier intervención. Medido:

$$\text{Lectura obligatoria} = \underbrace{2{,}2}_{\text{CLAUDE}} + \underbrace{44{,}3}_{\text{AGENTS}} + \underbrace{9{,}0}_{\text{ROADMAP}} + \underbrace{71{,}1}_{\text{SUMMARY}} = 126{,}7 \text{ KB} \approx 31.700 \text{ tokens}$$

De esos, **28,7 KB (el 65 % de `AGENTS.md`) eran el changelog §7**, que no altera
ninguna decisión de ejecución. El puntero incondicional obligaba a cargarlo entero
en cada turno.

**Solución — punteros condicionados por rama.** `CLAUDE.md` se reescribió como
enrutador: una única lectura incondicional (`AGENTS.md` §3, los invariantes que
ningún cambio puede violar) más una tabla que asocia cada rama de trabajo con el
documento que la sirve. `PROJECT_SUMMARY.md` no se recortó —es la memoria
académica y su extensión es deliberada—; cambió su modo de acceso, de lectura
íntegra por turno a consulta por sección.

$$\text{Lectura obligatoria} = 2{,}3 + 7{,}9 = 10{,}2 \text{ KB} \approx 2.550 \text{ tokens} \quad (-92\%)$$

**Fuente de verdad única.** `AGENTS.md` §4 (datasets), §5 (comandos) y §6 (próximos
pasos) replicaban contenido de `ROADMAP.md`, `README.md` y este documento; se
colapsaron en una tabla de punteros. `ROADMAP.md` §4 repetía las cinco deudas
metodológicas con sus soluciones: ahora las nombra y remite a `AGENTS.md` §3.F.
Verificación automatizada: **cero frases de más de 70 caracteres compartidas entre
documentos**.

**Cachés obsoletos.** `AGENTS.md` §2 y §3.C fijaban ~15 magnitudes (98.167 partidas,
23.583 invocadores, coberturas por tier) que el crawler invalida en cada corrida, y
que ya eran mutuamente inconsistentes. Se sustituyeron por `collect_data.py
--summary`, que las produce en vivo, conservando lo que no se deduce leyendo el
código: la justificación del umbral de 50 antes de 300.

**Prohibiciones convertidas en objetivos.** Una instrucción negativa arrastra a
contexto la conducta que pretende evitar. Se reformularon tres: *"Prohibido el
bloqueo de archivos"* → *"Toda apertura de la BD pasa por el context manager"*;
*"Nunca omitir `RiotRateLimiter`"* → *"Toda petición a Riot pasa por
`RiotRateLimiter`"*; *"No aplicar filtros de timestamp restrictivos"* → *"El crawler
recolecta el historial completo de la temporada"*.

**Defecto corregido.** `AGENTS.md` §6 instruía asignar etiquetas semánticas fijas al
clustering (*Lane Dominator*, *Scaling Farmer*, *Roaming Playmaker*, *Utility
Support*), práctica que §4.X de este documento había descartado explícitamente en
favor de **nombres dinámicos** derivados del $|z|$ máximo de cada centroide. Era
sedimento con una instrucción activamente contraria al diseño vigente.

### J. Compuertas de Avance entre Etapas

Las siete etapas de `ROADMAP.md` §3 pasaron a estar separadas por **compuertas**:
condiciones explícitas y verificables que deben cumplirse antes de iniciar la
siguiente. La cadena es estricta:

$$\text{volumen objetivo} \rightarrow \text{limpieza} \rightarrow \text{dataset armado} \rightarrow \text{EDA} \rightarrow \text{entrenamiento} \rightarrow \text{memoria}$$

**Fundamento estadístico.** Los vectores de estilo de `players.csv` son promedios
sobre el historial de cada invocador. Para un jugador con $n$ partidas registradas,
cada componente $\bar{x}_n$ del vector tiene error estándar $\sigma/\sqrt{n}$: un
invocador con $n = 30$ y el mismo con $n = 300$ producen vectores distintos, y la
diferencia no es ruido despreciable sino un desplazamiento sistemático del
centroide. Por lo tanto **todo artefacto derivado —clusters, arquetipos, features
del Modelo Jugadores— se invalida mientras siga entrando volumen**. Esa es la razón
de que la compuerta de volumen sea la primera y la más rígida.

El mismo argumento aplica a la limpieza: los umbrales de corte (madurez mínima,
descarte de perfiles ruidosos) se calculan sobre la distribución observada, y una
distribución parcial produce umbrales que dejan de ser válidos cuando llega el
resto de los datos. Limpiar antes obliga a repetir la limpieza y, peor, invita a
congelar decisiones tomadas sobre una muestra sesgada.

**Consecuencias sobre el estado declarado.** Dos etapas se reclasificaron para no
contradecir la regla:

* **Etapa 4** se dividió en **4a — armado** (`export_to_csv.py`, operativo y no
  bloqueado, pues es una proyección de la base en un instante dado) y **4b —
  limpieza** (bloqueada por la compuerta de volumen, se ejecuta una sola vez sobre
  el dataset definitivo).
* **Etapa 5 (EDA y arquetipos)** pasó de *en curso* a **provisional**: los análisis
  y clusterings ya ejecutados orientan decisiones pero no constituyen resultado, y
  se rehacen íntegramente cuando abra la compuerta de 4b.

Se documentó una excepción justificada: la **Etapa 3 (identidad canónica)** corre en
paralelo a la 2, porque no consume volumen y porque sin ella un mismo humano queda
fragmentado en varios PUUID, lo que invalida cualquier vector de estilo calculado
antes de resolverla.

### K. Unificación de la Nomenclatura de Modelos

La documentación arrastraba **dos nomenclaturas simultáneas** para los mismos dos
modelos, y —críticamente— **con mapeos invertidos entre sí**:

| Nomenclatura vieja | Referente | Ocurrencias |
| :--- | :--- | :--- |
| `Modelo 1` | Arquetipos de jugador | 29 |
| `Modelo 2` | Matchups de campeones | (idem) |
| `Modelo A` | **Campeones** | 11 |
| `Modelo B` | **Jugadores** | (idem) |

Que `Modelo 1` designara a los jugadores mientras `Modelo A` designaba a los
campeones convertía cualquier sustitución automática en un riesgo de corrupción
semántica, y obligaba a todo lector —humano o agente— a resolver la ambigüedad por
contexto en cada aparición.

Se unificó en **`Modelo Jugadores`** y **`Modelo Campeones`**, nombres que ya
utilizaba §7 de este documento. La elección no es cosmética: a diferencia de un
índice arbitrario, el nombre transporta su propio referente, de modo que una
mención aislada es interpretable sin recuperar la definición. Las sustituciones se hicieron mediante un mapeo explícito revisado caso por
caso, preservando los conceptos homónimos pero distintos (`Variante A` / `Variante
B` de las firmas de jugador, `Opción A` de la arquitectura canónica) y recentrando
manualmente las cajas del diagrama ASCII de §1 para conservar su alineación.

Total sustituido: **45 referencias** en cinco archivos —los cuatro documentos más
las celdas markdown de `notebooks/eda_visualizer.ipynb`—. Las únicas menciones
supervivientes de la nomenclatura vieja son la tabla de arriba, que la documenta, y
`docs/archive/`, que es registro histórico congelado.

---

### L. Tablero Operativo de Tareas (`TAREAS.md`, 2026-09-07)

**El hueco.** Tras el refactor documental de §9.I, la documentación cubría tres de
los cuatro ejes temporales del trabajo, y dejaba uno vacío:

| Eje | Pregunta que responde | Documento |
| :--- | :--- | :--- |
| Pasado | ¿Qué se hizo? | `AGENTS.md` §7 |
| Presente estructural | ¿En qué etapa estamos? | `ROADMAP.md` §3 |
| Fundamento | ¿Por qué se decidió así? | `PROJECT_SUMMARY.md` |
| **Futuro inmediato** | **¿Qué hay que hacer ahora?** | **(ninguno)** |

Las tareas pendientes existían únicamente en la conversación de cada sesión. Como
el repositorio opera en modo agente único y sin memoria entre sesiones, eso implica
que **toda tarea identificada y no ejecutada en el mismo turno se perdía**. El
`ROADMAP.md` no cubre ese hueco por diseño: describe etapas de meses separadas por
compuertas, no acciones de una sesión.

**La decisión.** Se creó `TAREAS.md` como cuarto documento, de lectura obligatoria
junto a `AGENTS.md` §3 al abrir cualquier sesión (`CLAUDE.md` §1). Su contrato son
tres reglas: leer el tablero al abrir, anotar toda tarea **en el momento en que se
identifica** —aunque no se ejecute hoy—, y moverla a "cerradas" al terminarla.

**El invariante que lo hace sostenible.** `TAREAS.md` **no acumula historia**: cuando
un cambio se cierra formalmente según `CLAUDE.md` §2, su fila migra al changelog de
`AGENTS.md` §7 y **se borra del tablero**. Esto es deliberado y corrige empíricamente
el defecto medido en §9.I, donde el changelog había crecido hasta representar el 65 %
de los 126,7 KB que se cargaban por turno. Un tablero de tareas es la estructura con
mayor tendencia natural a ese crecimiento —cada sesión agrega filas y ninguna las
quita—, así que la regla de migración se fija en el contrato del archivo y no como
convención tácita. El tamaño de `TAREAS.md` queda acotado por el número de tareas
**abiertas**, que es estacionario, en vez de por el número de tareas **históricas**,
que es monótono creciente.

**Deslinde contra los documentos vecinos.** El riesgo de un cuarto documento es
reintroducir la duplicación que §9.I eliminó. Se fijan cuatro fronteras explícitas
en §1 del propio archivo:

* contra `ROADMAP.md`: el ROADMAP dice *dónde estamos*; `TAREAS.md` dice *qué toco hoy*;
* contra `AGENTS.md` §7: §7 es historia **cerrada**; `TAREAS.md` es lo que todavía no;
* contra `AGENTS.md` §3.F: la deuda metodológica está **bloqueada tras una compuerta**
  y solo se toca por pedido explícito del usuario, mientras que el tablero contiene
  únicamente trabajo desbloqueado;
* contra `PROJECT_SUMMARY.md`: acá va el *porqué*, allá el *qué*.

**Formato con prioridad legible sin contexto.** Cada tarea declara tres campos además
de su descripción: *Bloquea* (qué queda trabado mientras no se haga), *Costo*
(tiempo, requests, riesgo) y *Estado* (qué falta para arrancar: una decisión del
usuario, una key, volumen). El propósito es que un agente que abre el archivo en frío
pueda ordenar el trabajo sin reconstruir la conversación que lo originó — que es
exactamente la información que se perdía antes.

**Siembra inicial.** El tablero nace con cinco tareas reales, todas cuantificadas
contra la base de datos viva en modo solo-lectura, no estimadas: la resolución
inversa de 16.078 PUUIDs sin identidad (6.815 requests $\approx$ 2,4 h al 100 %, o
3.600 requests $\approx$ 76 min al 80 %, con la curva de rendimientos decrecientes
tabulada), el alta de una segunda API key (con una sola, el techo es de 47,5 req/min),
la optimización del re-scoring greedy de `resolve_inverse_by_match.py` (~34 escaneos
completos de 981.670 filas por run), la corrección del agrupamiento por `puuid` en
lugar de `canonical_id` en el notebook (`coverage>=100` pasa de 925 a 2.712 jugadores)
y la continuación de la recolección hacia la compuerta de volumen.

**Alternativas descartadas.** (1) *Una sección más dentro de `ROADMAP.md`*: mezcla dos
horizontes temporales incompatibles —etapas de meses y tareas de una sesión— y obliga
a cargar la estrategia entera para leer el pendiente del día. (2) *Issues de GitHub*:
quedan fuera del árbol del repositorio, así que un agente no los ve en la lectura
obligatoria de apertura, que es precisamente el mecanismo que se quiere garantizar.
(3) *Marcadores `TODO` en el código*: dispersan el estado del proyecto por decenas de
archivos y no admiten tareas que no son de código, como el alta de una API key.

### M. Identidad de Jugador por Componentes Conexas (`player_id`, 2026-09-07)

#### M.1. El problema: ningún identificador disponible es una clave de persona

La API de Riot no expone ningún identificador que sea a la vez estable ante
renombres y portable entre proyectos. Los dos candidatos fallan en ejes opuestos:

| Identificador | Estable ante renombre | Portable entre proyectos |
| :--- | :---: | :---: |
| `puuid` | sí | **no** (cifrado por producto de API) |
| `canonical_id` = $\text{lower}(gameName\#tagLine)$ | **no** | sí |

El PUUID está cifrado con una clave ligada al *producto*, no a la key: una key
rotada del mismo proyecto devuelve los mismos PUUIDs — invariante que
`scripts/identity/register_key_project.py` verifica por fingerprint de Jaccard.
Como el pool se armó con keys de tres cuentas de Riot distintas, hay tres
espacios de cifrado y por tanto hasta tres PUUIDs por humano.

Medición sobre la BD viva (2026-09-07, solo lectura), que descarta usar
cualquiera de las dos columnas por separado:

$$\left|\{\,c : |\text{canonical\_puuid}(c)| > 1\,\}\right| = 3.102
\qquad
\left|\{\,p : |\text{canonical\_id}(p)| > 1\,\}\right| = 158$$

Es decir: 3.102 identidades estaban partidas en varios PUUID canónicos (restos
de la migración de PUUIDs muertos) y 158 PUUIDs tenían dos `name#tag` asociados
(renombres). Agrupar por `canonical_id` habría unido los primeros pero partido
los segundos, y el conteo del padrón *sube* de 20.896 a 20.980 filas: une 3.102 y
parte 158, con saldo neto $+84$.

#### M.2. La solución: union-find sobre las dos capas simultáneamente

Se define un grafo no dirigido $G = (V, E)$ donde $V$ es el conjunto de PUUIDs
conocidos y $E = E_1 \cup E_2$ combina las dos clases de arista:

$$E_1 = \{\,(a, c) : \texttt{v\_puuid\_canonical}(a) \to c\,\}
\qquad\text{(misma persona dentro de un proyecto)}$$

$$E_2 = \{\,(p_i, p_j) : \text{canonical\_id}(p_i) = \text{canonical\_id}(p_j)
\ \wedge\ \text{es\_nombre}(\cdot)\,\}
\qquad\text{(misma persona entre proyectos)}$$

Cada **componente conexa** de $G$ es una persona, y su etiqueta es el `player_id`.
La construcción usa union-find con compresión de caminos y unión por tamaño,
$O(|E| \cdot \alpha(|V|))$ con $\alpha$ la inversa de Ackermann: 66.741 aristas
resueltas en 1,26 s sobre la base viva.

La etiqueta es el mínimo lexicográfico de los `name#tag` del componente; si el
componente no tiene ningún nombre conocido, el mínimo de sus PUUIDs. Es
determinista para un estado dado de la base — verificado por test invirtiendo el
orden de las aristas de entrada.

> **Limitación asumida.** `player_id` es una clave derivada del estado actual de
> la base, no un identificador durable: si más adelante se descubre un alias o un
> renombre nuevo, la etiqueta de ese componente puede cambiar entre
> exportaciones. Sirve para agrupar dentro de un dataset, no para citar a un
> jugador entre versiones del dataset. Un identificador durable exigiría
> persistir las observaciones nombre↔PUUID con su fecha, que es la tarea abierta
> de capturar `riotIdGameName` / `riotIdTagline` en la ingesta.

Resultado sobre la base viva: 82.820 PUUIDs colapsan en **41.576 personas**, con
la distribución de tamaños de componente $\{1: 16.083,\ 2: 9.748,\ 3: 15.742,\
4: 1,\ 5: 1,\ 6: 1\}$. Los 15.742 componentes de tamaño exactamente 3
corresponden a jugadores mapeados en los tres proyectos.

#### M.3. Alternativas descartadas

1. **Renombre puro sin cambio semántico** (`canonical_id` → `player_id`, `puuid`
   → `puuid_raw`, sin tocar la agrupación). Descartada porque el nombre habría
   prometido unicidad sobre una columna que no la tiene: 3.102 identidades
   seguirían partidas. Poner una etiqueta tranquilizadora sobre una clave
   incorrecta es un defecto peor que el original, porque lo vuelve invisible.
2. **`player_id` = `canonical_id`.** Descartada por la medición de M.1: arregla
   3.102 casos y rompe 158 renombres, que son justamente el escenario que motivó
   todo el trabajo.
3. **Mantener `puuid` como columna de compatibilidad** junto a `player_id`.
   Descartada por decisión explícita del usuario: la columna `puuid` es el
   footgun, y conservarla anula el propósito del cambio.

#### M.4. El defecto que se elimina

La columna `puuid` significaba **dos cosas distintas** según el archivo, con el
mismo nombre: en `full_dataset.csv` era el PUUID crudo de la participación
(agrupar por él fragmenta), y en `players.csv` era ya el PUUID canónico de
proyecto 0 (agrupar por él es correcto). Ninguna convención de lectura sobrevive
a esa ambigüedad.

Magnitud medida del error que inducía, sobre `full_dataset.csv`:

$$\frac{|\{\text{puuid\_raw}\}|}{|\{\text{player\_id}\}|}
= \frac{78.408}{41.576} = 1{,}89\times$$

y restringido al padrón activo, $45.740 / 20.896 = 2{,}19\times$. Coincide con lo
medido en `TAREAS.md`: contar por identidad en vez de por PUUID sube
$\text{coverage} \ge 100$ de 925 a 2.712 jugadores.

El contrato nuevo es una sola frase: **`player_id` es la identidad; todo lo que
termina en `_raw` no se agrupa nunca.** `puuid_raw` se conserva para volver a la
API o depurar, y el sufijo lleva la advertencia incorporada.

#### M.5. Verificación

* `players.csv` conserva **20.896 filas**: dentro del padrón activo `player_id` y
  la agrupación anterior por `canonical_puuid` coinciden exactamente, con **0**
  `player_id` abarcando más de un `canonical_puuid`. Los 3.102 fragmentados
  viven fuera del padrón, así que la coherencia de §3.C de `AGENTS.md` se
  preserva sin cambios.
* `full_dataset.csv` mantiene sus 981.670 filas y **0 `player_id` nulos**.
* $\sum \text{games\_played}$ de `champions_by_lane.csv` $= 981.653 \le 981.670$;
  la diferencia de 17 es preexistente y corresponde a participaciones con
  posición fuera de las cinco canónicas, que ese CSV filtra por diseño.
* `num_puuids` se redefine como el **tamaño del componente conexo**, no como el
  conteo de PUUIDs en las participaciones: el camino de escritura ya canonicaliza
  a proyecto 0 antes del `INSERT`, así que esa segunda medida daría 1 siempre.
  Distribución resultante: $\{1: 3.754,\ 2: 3.550,\ 3: 13.591,\ 6: 1\}$.
* Suite `pytest` **55/55 en verde** (46 previos + 9 nuevos en
  `tests/test_identity.py`, que cubren transitividad, cada capa por separado, el
  caso de renombre, el determinismo de la etiqueta, la ausencia de la vista en
  bases legacy y el invariante `players["player_id"].is_unique`).

### N. Validación Oportunista del Dataset dentro del Hueco del Rate Limit (`run.py`, 2026-09-07)

**Problema.** El crawler no está limitado por CPU ni por disco sino por el rate
limit de Riot, y el techo que manda es el de ventana larga: $95$ requests cada
$120$ s por key (`AGENTS.md` §3.B). Con $k$ keys el caudal sostenible es

$$R_{\max} = k \cdot \frac{95}{120\,\text{s}} \approx 0{,}79\,k \ \text{req/s},$$

mientras que la ventana corta permite ráfagas de $18$ req/s. Es decir: el pool
consume su cuota de $95$ requests en $t_b \approx 95/18 \approx 5{,}3$ s de
ráfaga y después **espera $\approx 115$ s**. En régimen de ráfaga la fracción de
tiempo de pared bloqueada es

$$\phi = 1 - \frac{t_b}{120} \approx 0{,}956,$$

y no depende de $k$: sumar claves multiplica el caudal, no reduce la proporción
de espera. Con una sola key eso son ~57 minutos de CPU ociosa por hora, con
98.167 partidas en la base que nunca había auditado nadie.

**Decisión: cosechar ese hueco.** `run.py` corre el crawler en el hilo principal
y un hilo de fondo que audita la base **solo cuando el pool está en cooldown**.
El planificador (`OpportunisticValidator._budget()`) mira la espera $w$ de la
clave más libre y concede presupuesto de validación:

$$B(w) = \begin{cases}
\min(w - \varepsilon,\ B_{\max}) & \text{si } w \ge \theta \quad (\text{stall real}) \\
B_{\text{idle}} & \text{si hace} \ge T_{\text{idle}} \text{ que no se valida} \\
0 & \text{en otro caso (la red tiene prioridad)}
\end{cases}$$

con $\theta = 0{,}35$ s, $\varepsilon = 0{,}05$ s, $B_{\max} = 5$ s,
$B_{\text{idle}} = 0{,}5$ s y $T_{\text{idle}} = 45$ s. Hay una cuarta rama, que
antecede a las dos últimas: si **ninguna clave viva registró una petición en la
ventana larga** el crawler no está pidiendo nada, y el presupuesto es $B_{\max}$.
Cubre los dos huecos largos que no son cooldown —el arranque y, sobre todo, la
exportación de los 4 CSV al cierre de cada ciclo, que son varios minutos sin
tocar la red. La rama del medio existe
para que la auditoría avance igual cuando el rate limit no aprieta (pocas keys,
red lenta, crawler en fase de paginación). Sin cliente Riot —modo
`--solo-validar`— el presupuesto es infinito.

**Por qué la validación no le roba throughput al crawler.** El trabajo está
cuantizado en rebanadas de $s$ partidas. Medido sobre la BD viva: 98.167
partidas en $10{,}9$ s con $s = 300$, o sea

$$v \approx 9.006\ \text{partidas/s}, \qquad t(s) = \frac{s}{v},\qquad t(300) \approx 33\ \text{ms}.$$

El bucle interior comprueba el deadline **antes** de arrancar cada rebanada, así
que el desborde máximo sobre el presupuesto es exactamente $t(s)$. La garantía
es entonces una desigualdad verificable:

$$t(s) < \varepsilon \iff s < \varepsilon \cdot v = 0{,}05 \cdot 9.006 \approx 450.$$

Con el valor por defecto $s = 300$ el desborde peor caso ($33$ ms) cae dentro
del margen $\varepsilon$ ($50$ ms) que se le descontó a la espera: **la rebanada
termina antes de que se libere la ranura del crawler**. Subir `--slice` por
encima de ~450 rompe esa garantía y empieza a competir con la red; es el único
parámetro del módulo que no se puede tocar sin recalcular esta cota. En el otro
extremo, auditar la base entera cuesta $11$ s, o sea **un décimo de un solo
cooldown de ventana larga**.

**Cursor incremental.** El estado vive en `data/validation_state.json` y avanza
por el `rowid` de `matches`, que crece monótonamente con la ingesta. Se eligió
el `rowid` y no `collected_at` porque el timestamp tiene resolución de segundo y
el writer commitea en micro-lotes de 5 partidas: un corte por tiempo podría
partir un lote y saltearse filas. Con el `rowid`, las partidas que el crawler
guarda mientras la auditoría corre quedan por delante del cursor y se auditan en
la pasada siguiente; nada se audita dos veces y nada se saltea.

**Aislamiento respecto del crawler.** El validador abre la base con
`sqlite3.connect("file:...?mode=ro", uri=True)` más `PRAGMA query_only = ON`, el
mismo patrón de `scripts/ops/identity_progress.py`. En WAL los lectores no
bloquean al escritor, y el modo `ro` hace *imposible* —a nivel SQLite, no por
disciplina del código— que la auditoría tome un lock de escritura y dispare el
`[WinError 32]` de `AGENTS.md` §3.A. Un test lo fija: `DELETE FROM matches` sobre
esa conexión levanta `sqlite3.OperationalError`.

**Doctrina de severidades.** Los 16 chequeos se agrupan en tres niveles, y la
distinción es metodológica, no cosmética:

* **`error`** — rompe un invariante de `AGENTS.md` §3: 10 participantes, equipos
  $5$–$5$, exactamente 5 ganadores, `participantId` único, campeón válido,
  contadores no negativos, duración presente.
* **`aviso`** — anomalía recuperable que no invalida la partida: remake
  guardado, `queueId` ≠ 420, timeline ausente o incompleta, snapshot @10 faltante
  en una partida de más de 11 minutos.
* **`info`** — **deuda metodológica conocida** de `AGENTS.md` §3.F, que se mide
  pero no se corrige acá: posiciones duplicadas (§3.F.5) y ausencia del snapshot
  @20 (§3.F.4). Marcarlas como error habría empujado a "arreglarlas" en la
  ingesta, y la limpieza está detrás de la compuerta de volumen (`ROADMAP.md` §3).

Dos umbrales evitan falsos positivos por truncamiento temporal: el snapshot @10
solo se exige a partir de los $660$ s y el @20 a partir de los $1.260$ s (un
minuto de gracia sobre el corte nominal). Sin ese margen, las rendiciones
tempranas —el $9{,}9\%$ de §3.F.4— aparecerían como datos faltantes cuando en
realidad la partida terminó antes.

**Resultado de la primera auditoría completa (98.167 partidas, 10,9 s).** La
base está sustancialmente sana: **0** participaciones huérfanas, **0** timelines
huérfanas, **0** partidas sin participantes, **0** partidas descartadas que
igual quedaron guardadas, **0** remakes, **0** timelines faltantes y **0**
snapshots @20 ausentes en partidas que llegaron al minuto 21 (los $72.160$
`gold_20` nulos corresponden todos a partidas más cortas: truncamiento legítimo,
no pérdida de dato). Los hallazgos reales son tres:

| Hallazgo | Casos | % | Lectura |
| :--- | ---: | ---: | :--- |
| `ganador_incoherente` | 1 | 0,001 % | `LA2_1618651762`: 1.490 s, los 10 participantes con `win = 0`. Anomalía del payload de Riot, no de la ingesta. |
| `posicion_duplicada` | 2.343 | 2,39 % | Deuda §3.F.5 remedida sobre el dataset completo: baja del $2{,}53\%$ auditado sobre 23.717 partidas. |
| `posicion_vacia` | 17 | 0,02 % | `individualPosition` vacío o `Invalid`; explica el hueco de 17 filas de la coherencia §3.C. |

Ese último renglón cierra una cuenta que estaba abierta: la diferencia
$981.670 - 981.653 = 17$ entre `full_dataset.csv` y la suma de
`champions_by_lane.csv` (§9.M) es exactamente el conjunto de participaciones sin
posición asignable.

**Alternativas descartadas.**

* *Auditar en una corrida aparte, después del crawler.* Es lo que se venía
  haciendo (no se hacía nunca). Cuesta 11 s de reloj propio en vez de 0, y sobre
  todo compite por el disco con el writer si se lo lanza en paralelo a ciegas.
* *Auditar en SQL con una sola consulta agregada.* Una pasada de `GROUP BY`
  sobre las 981.670 participaciones no se puede interrumpir a mitad de camino,
  así que no cabe en una ventana de espera: o bloquea el proceso varios segundos
  o no corre. La granularidad por rebanada es justamente lo que la hace
  planificable.
* *Guardar los hallazgos en la propia BD.* Habría convertido al validador en
  escritor, con lock de escritura y riesgo de `[WinError 32]`. El JSON externo
  mantiene la propiedad de "solo lectura" como invariante duro.

**Respaldo a Drive al cierre (`--respaldar`).** La documentación de
`auto_nightly_pipeline.py` y `backup_to_drive.py` afirmaba que un hook
`pre-push` sincronizaba la nube en cada `git push`. Verificado el 2026-09-07:
**ese hook no existe** —`.git/hooks/` solo tiene los `.sample` y `core.hooksPath`
está sin definir—, así que la nube dependía de acordarse de correr el script a
mano. Se descartó instalar el hook por dos razones: la BD está en `.gitignore` y
nunca viaja por git, así que el push no es el evento que cambia los datos; y
colgarle la copia de ~1,3 GB a cada `git push` castiga los commits de una línea.
En cambio `run.py --respaldar` lo dispara al cerrar la sesión, que es cuando la
BD efectivamente creció, y **solo si la corrida guardó algo** —partidas,
timelines o jugadores nuevos—: una sesión estéril o una de `--solo-validar` no
reescribe 687 MB en Drive para nada. Si el respaldo se pidió y falló, `run.py`
sale con código 1; la política de integridad de §4.W manda igual, así que un
`quick_check` fallido aborta la subida antes de pisar la copia sana de la nube.

### O. Pool Multi-Clave Resiliente: Rotación en Caliente y Agotamiento No Fatal (2026-09-07)

Las dev keys de Riot vencen cada 24 h, así que en una sesión larga es normal que
una clave del pool se muera a mitad de camino. El cliente ya la desactivaba al
primer `HTTP 401/403`, pero quedaban tres agujeros.

**1. El agotamiento del pool era un `sys.exit()` en un hilo trabajador.** Cuando
moría la última clave, `RiotAPIClient` llamaba a `sys.exit(1)`. Disparado desde
uno de los hilos del `ThreadPoolExecutor` de `collector.py`, eso **no termina el
proceso**: `SystemExit` mata solo a ese hilo y lo consume el `try/except` que
envuelve cada future. El crawler seguía recorriendo el padrón con todas las
peticiones fallando y, peor, llamando a `mark_player_crawled()` sobre jugadores
a los que no les bajó nada — corrupción silenciosa del criterio de "ya visitado".
Ahora `_select_key_and_acquire()` devuelve `None`, levanta el `threading.Event`
`pool_exhausted` y `_request()` se rinde; el orquestador decide qué hacer.

**2. No había forma de cortar el recorrido desde afuera.** `collect_matches()` y
`backfill_timelines()` aceptan ahora un `stop_event` opcional que se consulta en
los mismos tres puntos donde ya se miraba el presupuesto de tiempo (unificados en
un `_debe_parar()`). `run.py` lo levanta cuando el pool queda sin claves vivas: el
crawler cierra ordenadamente, con su reporte de cierre completo, y la validación
del dataset ya descargado **sigue corriendo** — que es justo cuando más máquina
libre hay.

**3. Renovar una clave obligaba a reiniciar la corrida.** El `KeyPoolSupervisor`
de `run.py` relee `.env` cada `--vigilar-keys` segundos (300 por defecto) y
reemplaza las claves **por ranura**, no por cadena. La distinción es la que
sostiene todo el sistema de identidades: el PUUID está cifrado por *proyecto* de
Riot, no por key (§9.V), y `RIOT_API_PROJECT_INDICES` mapea posición del pool →
proyecto. Regenerar una dev key vencida devuelve una cadena nueva **del mismo
proyecto**, así que pisar la ranura $i$ con `keys[i]` preserva la validez de
todos los PUUIDs ya guardados. Por el mismo motivo, una clave en una ranura
*nueva* no se incorpora sola: sería un proyecto sin registrar y mezclaría PUUIDs
—hace falta `allow_new_slots=True` y pasar antes por
`scripts/identity/register_key_project.py`.

**4. La penalización de un `HTTP 429` era invisible para el resto del pool.**
`RiotRateLimiter.handle_rate_limit_exceeded()` dormía los segundos que pide
`Retry-After` y **después vaciaba los dos historiales**. Durante esa pausa
cualquier otro hilo que consultara `get_wait_time()` veía cuota íntegra y volvía
a golpear la API — 429 encadenados, exactamente lo que el limiter existe para
evitar. Ahora la penalización se publica en `penalty_until` *antes* de dormir y
`get_wait_time()` la devuelve como

$$w = \max(w_{\text{corta}},\ w_{\text{larga}},\ \max(0,\ t_{\text{penalidad}} - t)),$$

con lo cual la espera queda visible para los demás hilos y, de paso, para el
planificador de la validación: una penalización de 73 s medida en la corrida de
prueba pasó de ser tiempo perdido a ser presupuesto de auditoría.

**5. El rechazo por PUUID de otro proyecto no se contaba en ninguna parte.** Un
`HTTP 400 - Exception decrypting` no es error de red ni éxito: el cliente lo
reintenta con las demás claves y, si ninguna lo descifra, devuelve `None` en
silencio. Con el pool incompleto —hoy hay una sola key, en la ranura 3, mientras
los PUUIDs del padrón son del proyecto 0 (ver `TAREAS.md`)— ese es el **modo de
falla dominante**: la corrida de prueba del 2026-09-07 terminó con 9 peticiones,
0 respuestas 200, 0 errores registrados y 0 partidas nuevas, sin una sola línea
que dijera por qué. Ahora se contabiliza como `puuid_mismatch` (global y por
clave) y el reporte de cierre lo nombra explícitamente cuando iguala o supera a
las respuestas exitosas.

Como efecto colateral el pool dejó de ser una caja negra: cada `KeyContext`
lleva su propio contador de `requests / ok / rate_limited / errors`, y el reporte
de cierre imprime el reparto real de carga por ranura junto con las bajas de la
sesión. Un pool desbalanceado —o una clave que se murió a los diez minutos—
antes era invisible dentro del agregado global.

El preflight de `AGENTS.md` §3.B quedó incorporado al arranque: `run.py` sondea
cada clave (una petición barata a League-V4) y arma el pool solo con las usables,
reutilizando `probe_key()` / `check_keys()` de `scripts/ops/check_api_keys.py`
para no duplicar el criterio. `HTTP 429` cuenta como usable: la clave es válida y
solo estaba saturada en el instante del sondeo.

---

### P. Los CSV como Volcado Fiel: Prohibición de Limpieza No Solicitada (2026-09-08)

La regla queda escrita como invariante en `AGENTS.md` §3.G: **ningún agente de IA
limpia, filtra o depura los 4 CSV de `data/` por iniciativa propia**; solo cuando
el usuario lo pide explícitamente, y solo la limpieza pedida. Esta sección
justifica *por qué* esa regla es necesaria y qué alternativas se descartaron.

#### 1. El problema: la limpieza silenciosa destruye la auditabilidad

El pipeline tiene dos artefactos con roles distintos que es fácil confundir:

| Artefacto | Rol | Propiedad que debe conservar |
| :--- | :--- | :--- |
| `data/lol_las.db` | Fuente de verdad de la ingesta | Fidelidad al payload de Riot |
| Los 4 CSV de `data/` | **Volcado** de la BD para consumo de ML | Fidelidad a la BD |
| Matrices de entrenamiento | Entrada del modelo | Adecuación al supuesto del modelo |

La propiedad que sostiene todo el trabajo de auditoría del proyecto es la
**conmutatividad del volcado**: para cualquier consulta agregada $f$,

$$f(\text{CSV}) = f(\text{BD})$$

Si el exportador aplica un filtro $\phi$ por su cuenta, lo que sale es
$f(\phi(\text{BD}))$, y la igualdad se rompe **sin dejar rastro en el archivo**.
El daño no es perder filas —eso se puede volver a exportar— sino que a partir de
ahí ninguna discrepancia entre CSV y BD es diagnosticable: un sesgo del crawler
(recolección desbalanceada por tier), un sesgo de la ingesta (timelines faltantes)
y un sesgo del exportador ($\phi$) producen la misma sintomatología en el CSV. Un
agente que "arregla" el dataset al exportar convierte un sesgo medible en un sesgo
invisible, que es estrictamente peor que el sesgo original.

Esto es exactamente lo que el proyecto no puede permitirse: las cinco entradas de
deuda metodológica de `AGENTS.md` §3.F existen porque cada sesgo fue **medido
sobre el CSV** contrastándolo con la BD. Con limpieza libre en el exportador, esa
auditoría forense de 23.717 partidas no habría sido reproducible.

#### 2. Por qué el filtro pertenece al preprocesamiento, no al export

Las cinco correcciones de §3.F no son verdades sobre los datos: son **supuestos de
un modelo concreto**. Aplicarlas en el export las volvería globales, y son
mutuamente incompatibles entre modelos:

* La augmentación simétrica de lado (§3.F.1) sirve al Modelo Campeones y
  **arruinaría** el Modelo Jugadores: duplicar cada partida invertida multiplica
  por dos las participaciones de cada invocador y corrompe `games_recorded`.
* El corte $\text{games\_recorded} \ge 15$ (§3.F.3) sanea los clusters de
  arquetipos y **destruiría** la coherencia de §3.C: al desaparecer del padrón los
  jugadores de cola baja, $\sum \text{games\_played}$ de `champions_by_lane.csv`
  dejaría de igualar las filas de `full_dataset.csv`.
* El filtro de biyección de posiciones (§3.F.5) es correcto para `matchups_1v1.csv`
  y descarta información legítima en `full_dataset.csv`, donde un lane swap es un
  dato real de la partida.

Formalmente: si $\phi_1, \dots, \phi_5$ son los filtros y $M_1, M_2$ los dos
modelos, no existe un $\phi$ único tal que $\phi(\text{BD})$ sirva a ambos. Elegir
uno en el export es imponerle a $M_2$ el supuesto de $M_1$. Aguas abajo, en cambio,
cada modelo compone los suyos ($\phi_{M_1} = \phi_1 \circ \phi_5$, etc.) sobre el
mismo volcado íntegro, la decisión queda documentada y es reversible sin volver a
tocar la BD.

#### 3. Alternativas descartadas

* **Exportar limpio y crudo a la vez (8 CSV).** Duplica ~580 MB por copia,
  multiplica por dos el invariante de §3.C y crea la pregunta "¿cuál es el bueno?"
  en cada consumidor. Descartada: el mismo resultado se obtiene aplicando el filtro
  en el notebook, con dos líneas y sin costo de disco.
* **Una bandera `--clean` en `export_to_csv.py`.** Descartada por asimetría de
  riesgo: la bandera no aporta nada que el preprocesamiento no haga mejor, y
  agrega una forma de producir un CSV limpio **indistinguible del crudo** una vez
  escrito en disco (mismo nombre, mismo esquema, misma carpeta), que es justamente
  el fallo del punto 1.
* **Marcar en vez de borrar (columna `is_outlier`).** Es la alternativa
  defendible, y sigue disponible si el usuario la pide. No se adopta de oficio
  porque una columna de criterio congela ese criterio en el artefacto: cuál es el
  umbral de outlier es una decisión de modelado que hoy no está tomada, y la
  compuerta de volumen de `ROADMAP.md` §3 dice que se toma una sola vez, sobre el
  dataset final.

#### 4. Qué hace el agente en su lugar

Ante un sesgo detectado al exportar, el rol del agente es **nombrarlo, no
corregirlo**: anotarlo en `TAREAS.md` §3 o citar la entrada de §3.F que aplique, y
dejar la decisión al usuario. Es el mismo protocolo que ya rige para la deuda
metodológica, extendido al momento del volcado.

#### 5. El defecto que hacía falsa la regla: `pytest` borraba los CSV reales

La regla del punto 1 se escribió por una limpieza *deliberada*. Al verificarla se
encontró que el repositorio ya tenía una limpieza **accidental**, y mucho peor:
correr la suite de tests dejaba los 4 CSV de `data/` con solo la cabecera.

El reporte de cierre de sesión de `src/collector.py` terminaba con

```python
export_all_datasets(self.db.db_path, DATA_DIR)
```

Los dos argumentos venían de fuentes distintas: la BD del argumento uno es la que
la sesión realmente recolectó (`self.db.db_path`), mientras que el destino del
argumento dos era el **`DATA_DIR` global del repositorio**, importado de
`src/config.py`. En producción coinciden y nada se nota. En los tests no: el
fixture `temp_db` crea una base vacía en un directorio temporal, y
`test_el_crawler_respeta_la_senal_de_corte` la lleva hasta el cierre de sesión.
El resultado era un volcado de la BD temporal —vacía— **sobre los CSV reales**.

La huella cuadra con lo observado al abrir la sesión: `players.csv` con 637 bytes
y `champions_by_lane.csv` con 394, exactamente sus cabeceras y nada más. Los dos
grandes sobrevivían por casualidad — al no haber participaciones ni duelos, la
`query` de `full_dataset` y la de `matchups_1v1` no llegaban a escribirse antes
que el proceso real terminara de regenerarlos.

**Corrección:** el destino se deriva ahora de la BD que se recolectó,
`Path(self.db.db_path).parent`, así que el argumento uno y el dos no pueden volver
a divergir. En producción `db_path` es `data/lol_las.db` y el destino sigue siendo
el mismo `data/` de siempre: cero cambio de comportamiento. Una sesión sobre una
BD temporal escribe junto a esa BD, donde no molesta a nadie.

**Test de regresión** (`test_session_export_never_touches_the_repo_data_dir`):
toma el `st_mtime_ns` de los 4 CSV de `data/` antes de correr una sesión sobre
`temp_db` y exige que ninguno cambie, y además que los 4 aparezcan junto a la BD
temporal. Verificado en las dos direcciones: pasa con la corrección y **falla al
reponer el `DATA_DIR` global**, que es lo que lo hace un test y no una
declaración de intenciones. Suite: **116/116**.

El defecto es el mismo error de fondo que motiva §3.G, en su versión no
intencional: un camino de escritura que decide por su cuenta qué hay en los CSV.
Que la regla y el defecto se hayan encontrado en la misma sesión no es
coincidencia — la regla obliga a preguntarse *quién más escribe acá*.

---

### Q. Clausura del Grafo de Identidades (Capa 3 + Invariante de Proyecto), Padrón por Componente y Guardas de Ingesta/Respaldo (2026-10-19)

Cinco defectos acoplados hacían que la resolución de identidades entre los 4
proyectos de API (`P0`–`P3`) no llegara íntegra a los CSV ni a las notebooks:

1. **Desfase entre la BD local y la de Google Drive sin guarda de regresión.**
   La BD local (`data/lol_las.db`) tenía un estado anterior de la resolución
   inversa ($26.422$ identidades, $35.173$ alias, $0$ filas en `puuid_riot_ids`),
   mientras que la copia en Google Drive (`G:\Mi unidad\Lol Database\lol_las.db`)
   contenía el avance real ($33.882$ identidades, $50.151$ alias, $17.468$
   observaciones en `puuid_riot_ids`, $4.821$ PUUIDs pendientes sobre las mismas
   $98.167$ partidas). Además, `scripts/backup/backup_to_drive.py` carecía de una
   guarda monotónica: ejecutar `backup_to_drive.py` o `run.py --respaldar` desde
   una copia atrasada habría sobrescrito silenciosamente el trabajo de Drive.
   Se restauró la copia de Drive (con backup previo en `data/backups/`), se
   restituyó el `key_hint` de `project_index = 0` (`RGAPI-84c...bc1a`) en
   `api_projects`, y se incorporó `check_no_regression(local_manifest, remote_manifest)`
   en `backup_to_drive.py`, que compara los seis contadores monotónicos
   (`matches`, `match_participants`, `match_timeline`, `player_identities`,
   `player_puuid_aliases`, `puuid_riot_ids`) contra el `backup_manifest.json`
   remoto y aborta el respaldo ante cualquier retroceso salvo `--force` explícito.

2. **Extensión del `UnionFind` a 3 capas con invariante de unicidad por proyecto y biyección componente $\leftrightarrow$ `player_id` (`src/features/identity.py`).**
   El grafo de identidades operaba solo sobre `v_puuid_canonical` (Capas 1 y 2),
   ignorando las $17.468$ observaciones $(u, \text{game\_name}, \text{tag\_line}, t_{\min}, t_{\max})$
   de `puuid_riot_ids` (donde reside el proyecto 3). Sin embargo, unir componentes
   ciegamente por `lower(game_name)#lower(tag_line)` introduce **colisiones por
   reutilización de Riot ID** cuando un jugador libera un nombre y otra cuenta del
   mismo proyecto lo adopta.

   Sea $\pi(u) \in \{0, 1, 2, 3\}$ el proyecto de cifrado conocido de un PUUID $u$
   (inferido de `players` $\to 0$, `player_puuid_map` y `puuid_riot_ids`), y sea
   $\Pi(C) = \{\pi(u) : u \in C\}$ el conjunto de proyectos presentes en la
   componente conexa $C$. Dado que Riot asigna **exactamente un PUUID por cuenta y
   por proyecto de API**, toda componente física válida debe satisfacer el
   **invariante de disyunción de proyectos**:

   $$\forall\, u \neq v \in C \text{ con } \pi(u), \pi(v) \text{ conocidos}: \quad \pi(u) \neq \pi(v)$$

   En consecuencia, una arista por nombre en las Capas 2/3 entre $u \in C_u$ y
   $v \in C_v$ ($C_u \neq C_v$) es admisible si y solo si:

   $$\Pi(C_u) \cap \Pi(C_v) = \emptyset$$

   Para emparejar correctamente al dueño vigente cuando un nombre fue reutilizado en
   el tiempo, los candidatos de cada etiqueta se ordenan en orden decreciente de
   `last_seen_game_creation` antes de intentar las uniones. Finalmente, si dos
   componentes disjuntas $C_1 \neq C_2$ comparten la misma etiqueta preferida
   $\ell$, se preserva la **biyección estricta entre componentes conexas y
   `player_id`**: primero se elige una etiqueta alternativa no disputada del
   historial de renombres de la componente (si existe), y en última instancia se
   desambigua de forma determinista mediante el sufijo del miembro lexicográficamente
   mínimo, $\ell\text{\texttt{\~}}\min(C)_{[0:8]}$.

   **Verificación sobre la BD viva ($114.864$ PUUIDs):**
   * Conflictos en Capa 1 (`canonical_puuid`): **$0$**.
   * Colisiones de nombre bloqueadas en Capas 2/3 por $\Pi(C_u) \cap \Pi(C_v) \neq \emptyset$:
     exactamente **$3$ nombres ($6$ aristas)** en todo el padrón:
     - `jugador_a#tag`: dos cuentas distintas de `P0` en `players` (Diamond II 75 LP vs. Diamond IV 70 LP).
     - `jugador_b#tag1`: una cuenta de `P3` renombró `jugador_b#tag1` $\to$ `jugador_b#tag2` y otra cuenta de `P3` tomó `jugador_b#tag1`; por preferencia de etiqueta no disputada quedan separadas como `'jugador_b#tag2'` y `'jugador_b#tag1'`.
     - `jugador_c#tag`: dos cuentas de `P3` con el mismo Riot ID en épocas distintas ($t = 1788141121286$ vs. $t = 1743603963675$); el orden temporal une la cuenta activa con su componente `P0/P1/P2` (`'jugador_c#tag'`) y aísla la cuenta antigua (`'jugador_c#tag~lMahr4xU'`).
   * Resultado global en `player_id_map`: **$114.864$ PUUIDs $\to 52.887$ personas físicas (`player_id`)**, de las cuales **$40.821$** pertenecen al padrón Diamante+.

3. **Pertenencia al padrón por componente en `export_players()` (`export_to_csv.py`).**
   Aunque `export_players()` agrupaba por `m.player_id`, hacía `INNER JOIN players pl ON pl.puuid = COALESCE(v.canonical_puuid, p.puuid)`
   fila por fila. Toda participación de un jugador del padrón registrada bajo un
   PUUID de otro proyecto unido por Capa 2 o Capa 3 (cuyo `v.canonical_puuid` no
   estaba directamente en `players`) quedaba descartada antes del `GROUP BY`.
   Definida la pertenencia al padrón a nivel de componente conexa:

   $$\text{Padrón}(C) \iff \exists\, u \in C \text{ tal que } u \in \text{players}$$

   mediante la CTE `padron_players` (`SELECT m.player_id, MAX(pl.tier) AS tier, MAX(pl.rank) AS rank FROM player_id_map m INNER JOIN players pl ON pl.puuid = m.puuid GROUP BY m.player_id`),
   `players.csv` pasa de **$18.471$ a $25.922$ jugadores del padrón con partidas** y
   de **$831.523$ a $958.520$ participaciones** (recuperando **$126.997$ participaciones**
   que ya estaban en la BD), elevando `coverage50` ($\ge 50$ partidas) de **$5.593$ a
   $6.113$ jugadores ($+520$)** sin realizar una sola petición a la API. El
   invariante `AGENTS.md` §3.C se conserva intacto ($958.520 \le 981.670$).

4. **Persistencia de `riotIdGameName` / `riotIdTagline` en la ingesta y fijación del Proyecto Canónico 0 (`src/database.py`, `src/collector.py`).**
   Tanto `save_match_and_participants()` como `save_matches_and_timelines_batch()`
   extraen ahora `(puuid, riotIdGameName, riotIdTagline)` del payload de Match-V5
   para los 10 participantes y ejecutan un upsert transaccional sobre
   `puuid_riot_ids` (manteniendo `MIN(first_seen_game_creation)` y
   `MAX(last_seen_game_creation)`), evitando perder identidades en futuras
   recolecciones. En `DataCollector`, `_canonical_client()` resuelve la ranura cuyo
   `RIOT_API_PROJECT_INDICES` es `0` (en lugar de asumir `clients[0]`), y
   `sync_players()` aborta con advertencia explícita si el Proyecto 0 no está
   presente y activo en el pool, impidiendo contaminar o podar `players` con PUUIDs
   de otro proyecto.

5. **Migración de `notebooks/eda_visualizer.ipynb` y `player_signatures_*.csv` a `player_id`.**
   Se actualizaron las celdas de `notebooks/eda_visualizer.ipynb` (celdas 7, 23, 33
   y 35) y se regeneraron `player_signatures_A.csv` y `player_signatures_B.csv`
   para operar exclusivamente sobre `player_id`, `blue_player_id` y `red_player_id`,
   eliminando todas las referencias residuales a columnas retiradas (`puuid`,
   `canonical_id`, `blue_puuid`, `red_puuid`).

6. **Cierre al 100 % de la resolución inversa de identidades (`scripts/identity/resolve_inverse_by_match.py`, `scripts/ops/identity_progress.py`).**
   Se actualizó `resolve_inverse_by_match.py` para operar con clientes etiquetados
   por `(project_index, client)` y persistir también en `player_identities` /
   `player_puuid_map` los PUUIDs vigentes del proyecto canónico (`dead == fresh`),
   y se incorporó `player_puuid_map` al sembrado del `UnionFind` y a
   `identity_progress.py`. Tras procesar las $4.057$ partidas restantes con las
   claves activas de `P0` y `P1`, la cobertura alcanzó el **100,00 %**
   (**$78.408 / 78.408$ PUUIDs** con identidad, **$0$ PUUIDs huérfanos**, **$0 / 981.670$
   participaciones huérfanas**), elevando `players.csv` a **$26.447$ jugadores** del
   padrón Diamante+.

7. **Saneamiento de `tier = 'UNKNOWN'` y conteo unificado por componente en la cola de recolección profunda (`src/database.py`).**
   Durante la resolución de identidades por partida (`resolve_identities.py`),
   `upsert_player_identity()` insertaba filas en `player_identities` con el valor por
   defecto `tier = 'UNKNOWN'`. Como `'UNKNOWN'` no es `NULL`, la expresión
   `COALESCE(id.tier, p.tier)` de `get_canonicals_for_deep_crawl()` devolvía
   `'UNKNOWN'` aun cuando `players.tier` tenía el rango real (`'DIAMOND'` o
   `'MASTER'`), haciendo que `collector.py` cayera en el fallback de `260`
   partidas/jugador en vez de la meta de `50` partidas/jugador. Se corrigieron tres
   puntos en `src/database.py`:
   * `upsert_player_identity()` preserva el `tier` y `rank` existentes cuando recibe
     `tier = 'UNKNOWN'` (`CASE WHEN excluded.tier IS NOT NULL AND excluded.tier != 'UNKNOWN' THEN excluded.tier ELSE player_identities.tier END`).
   * `get_canonicals_for_deep_crawl()` resuelve el rango con
     `COALESCE(NULLIF(id.tier, 'UNKNOWN'), NULLIF(p.tier, 'UNKNOWN'), 'DIAMOND')` y
     calcula `existing_recorded_games` agregando sobre la componente conexa
     `player_id_map` del jugador.
   * `get_player_match_count(puuid)` cuenta `COUNT(DISTINCT match_id)` sobre todos
     los PUUIDs que comparten `player_id` en `player_id_map` (con fallback a
     `v_puuid_canonical`), de modo que el freno en `50` partidas de `collector.py`
     evalúe la madurez real de la persona física y no la de un único proyecto.
   Saneadas las $12.697$ filas `'UNKNOWN'` en `data/lol_las.db`, la cola de
   recolección profunda quedó con **$0$ `'UNKNOWN'`** y **$21.076$ jugadores
   incompletos ($< 50$ partidas)**: **$17.044$ `DIAMOND` ($80{,}9\%$)**, **$4.031$
   `MASTER` ($19{,}1\%$)** y **$1$ `CHALLENGER`**.

---

### R. Hito 100.000 Partidas: EDA de Arquetipos y Evaluación de Baselines Heurísticos (6 Targets) (2026-10-20)

Al superar las **$100.671$ partidas únicas** ($1.006.710$ participaciones y timelines al 100 %), con la unificación de identidades al $100{,}00\%$ y la regeneración de `players.csv` ($27.377$ jugadores), se ejecutaron dos análisis clave para fijar las líneas de base del proyecto:

#### 1. Actualización de Arquetipos de Invocador ($N \ge 50$ partidas)
La población de jugadores con madurez estadística suficiente ($N \ge 50$) creció a **$6.299$ invocadores** ($4.715$ Master, $885$ Diamond, $500$ Grandmaster, $199$ Challenger).
El clustering $K$-Means ($K=5$) sobre las $15$ dimensiones híbridas (combate, farmeo, visión, diferenciales 1v1 y dispersión de pool de embeddings) confirmó la estabilidad de los **5 arquetipos tácticos** sin etiquetas forzadas:
1. **Cluster 4 ($16{,}4\%$, $n=1.030$): *Hyper-Dominant 1v1 Carry*** ($z_{\text{GD@15}} = +1{,}43$, $z_{\Delta \text{Damage}} = +1{,}31$, $z_{\Delta \text{CS@15}} = +1{,}30$).
2. **Cluster 2 ($21{,}7\%$, $n=1.366$): *Low-Economy Weakside Absorb*** ($z_{\Delta \text{CS@15}} = -1{,}09$, $z_{\Delta \text{CS End}} = -1{,}04$, $z_{\text{GD@15}} = -0{,}97$).
3. **Cluster 1 ($12{,}5\%$, $n=787$): *Support & Vision Facilitator*** ($z_{\text{Vision}} = +2{,}25$, $z_{\text{CS}} = -2{,}28$).
4. **Cluster 3 ($12{,}6\%$, $n=795$): *Versatile Flex Generalist*** ($z_{\text{PoolDispersion}} = +1{,}86$, $z_{\text{RoleFocus}} = -0{,}99$).
5. **Cluster 0 ($36{,}8\%$, $n=2.321$): *Disciplined Meta Specialist / Resource Farmer*** ($z_{\text{CS@15}} = +0{,}55$, $z_{\text{TotalCS}} = +0{,}54$, $z_{\text{Vision}} = -0{,}49$).

> **Corrección (2026-10-20, §9.S):** los nombres en cursiva de esta lista
> (*Hyper-Dominant 1v1 Carry*, etc.) son una lectura humana del perfil $z$, no
> salen del notebook, que nombra cada cluster por su variable de mayor $|z|$
> (§4.X). Además, la elección $K = 5$ **no** está respaldada por el silhouette,
> que es máximo en $k = 2$ ($0{,}362$) y decrece de forma monótona hasta
> $0{,}172$ en $k = 5$; y el Hopkins $H \approx 1$ citado en versiones previas
> era un artefacto de cálculo (valor corregido: $H = 0{,}842$). Ver §9.S.2.

#### 2. Evaluación de Baselines Estadísticos sobre los 6 Targets (`scripts/analysis/evaluate_baselines.py`)

> **Corrección (2026-10-20, §9.S.1):** la primera versión de esta sección
> describía un baseline con partición cronológica 80/20 y estadísticas
> recalculadas sobre el *train*. Ese código ya no existe: el script vigente
> aplica las dos reglas pedidas por el usuario directamente sobre
> `champions_by_lane.csv`, sin entrenamiento. Los números de abajo son los de
> ese script, medidos sobre las $100.671$ partidas.

Reglas, sin ningún parámetro aprendido:

* **Targets 1..5 — ganador de línea @15.** Para la línea $L$ se predice que gana
  Blue si
  $$\overline{\text{Gold}}_{15}(c_{\text{Blue}}, L) > \overline{\text{Gold}}_{15}(c_{\text{Red}}, L)$$
  con $\overline{\text{Gold}}_{15}$ = `avg_gold_at_15` de `champions_by_lane.csv`.
  La etiqueta real es quién tuvo más oro al minuto 15 en esa partida.
* **Target 0 — ganador de la partida.** Cada equipo recibe
  $$\text{Score}(E) = \frac{1}{5}\sum_{i \in E} WR(c_i, L_i)$$
  con $WR$ = `winrate_pct` del campeón **en su línea**, y se predice el equipo de
  mayor score.

Filtros de evaluación (no tocan los CSV, §3.G): se excluye `LA2_1618651762` (sin
ganador único), los duelos sin un jugador por equipo en la línea
(`posicion_duplicada`/`posicion_vacia`, 4.805) y los duelos sin oro @15 o con
empate exacto (5.302).

| Target | N | Accuracy | ROC-AUC | Referencia trivial |
| :--- | ---: | ---: | ---: | ---: |
| Partida (media WR campeón-línea) | 100.670 | **54,98 %** | 0,5718 | 52,02 % (siempre Red) |
| TOP @15 | 99.556 | 57,38 % | 0,6069 | 51,19 % |
| JUNGLE @15 | 98.268 | 58,71 % | 0,6248 | 50,68 % |
| MIDDLE @15 | 99.545 | 57,39 % | 0,6020 | 50,87 % |
| BOTTOM @15 | 98.587 | 55,94 % | 0,5872 | 50,16 % (siempre Blue) |
| UTILITY @15 | 97.292 | **60,65 %** | 0,6502 | 50,57 % |
| Líneas (global) | 493.248 | **58,01 %** | 0,6141 | 50,63 % |

**Lectura.** El oro esperado del campeón explica bastante más el resultado de
la línea (+7,4 pp sobre la referencia) que el winrate promedio explica la
partida (+3,0 pp sobre "siempre Red").

**Advertencia metodológica.** Las tablas de campeón se calculan sobre las mismas
partidas que se evalúan, así que estas cifras son **descriptivas** y no de
generalización. Con $\ge 100$ partidas por par campeón-línea, el aporte de una
sola partida a su propia media es pequeño, pero para comparar contra los modelos
entrenados de la Etapa 6 hay que repetir la evaluación con las tablas calculadas
solo sobre el conjunto de entrenamiento.

##### Relación Líneas Ganadas @ 15 $\to$ Victoria Final

Sobre las partidas con los 5 duelos limpios (oro @15 presente y sin empate en
las 5 líneas):

$$\begin{array}{c|c|r}
\text{Líneas ganadas @ 15 por Blue} & \text{Winrate Partida} & \text{Partidas} \\ \hline
0 \text{ de } 5 & 8{,}78\% & 7.019 \\
1 \text{ de } 5 & 21{,}82\% & 17.217 \\
2 \text{ de } 5 & 38{,}43\% & 25.240 \\
3 \text{ de } 5 & 57{,}63\% & 24.867 \\
4 \text{ de } 5 & 75{,}82\% & 16.431 \\
5 \text{ de } 5 & 90{,}07\% & 6.337 \\
\end{array}$$

Correlación entre ganar la línea @15 y ganar la partida: **JUNGLE
($r = +0{,}2638$, $63{,}21\%$)** y **MIDDLE ($r = +0{,}2619$, $63{,}12\%$)**,
seguidas de BOTTOM ($r = +0{,}2571$, $62{,}84\%$), UTILITY ($r = +0{,}2366$,
$61{,}84\%$) y TOP ($r = +0{,}1993$, $60{,}00\%$). El porcentaje es
$P(\text{gana la partida} \mid \text{gana la línea @15})$.

---

### S. Notebook sin Valores Fijos, Hopkins Corregido y Baseline Alineado al Pedido (2026-10-20)

#### S.1. El baseline debía ser una regla, no un modelo

El pedido del usuario es un baseline **sin aprendizaje**: comparar
`avg_gold_at_15` campeón-línea para cada línea, y la media de `winrate_pct`
campeón-línea por equipo para la partida. El script vigente
(`scripts/analysis/evaluate_baselines.py`) implementa exactamente eso, pero
`tests/test_baselines.py` seguía importando las funciones de la versión con
partición 80/20 (`evaluate_lane_baseline_vectorized`,
`compute_train_statistics`…). El `ImportError` cortaba la recolección y **`pytest`
no corría ningún test**. El test se reescribió contra las funciones vigentes y
fija las reglas del usuario: equipo con mayor WR medio gana, la media cubre las
5 líneas, se excluye la partida sin ganador único, gana la línea el campeón con
más oro medio @15, sin oro o con empate no hay etiqueta, y la posición duplicada
no se empareja cartesianamente.

*Alternativa descartada:* restaurar la versión 80/20. Es una evaluación más
honesta para comparar contra un modelo, pero no es lo que se pidió como
baseline. Queda anotada como la evaluación que corresponde en la Etapa 6.

#### S.2. Sección 16 del notebook: qué estaba fijo y por qué importaba

1. **Hopkins inflado por un error de índice.** El código hacía
   `dist_X = cdist(X[idx], X)` (matriz $m \times n$) y luego
   `np.fill_diagonal(dist_X, np.inf)`. Eso pone infinito en $(i, i)$, pero la
   fila $i$ es el punto `idx[i]`: la autodistancia, que vale $0$, queda viva en
   $(i, \texttt{idx}[i])$. Entonces $w_i = 0$ para toda la muestra y
   $$H = \frac{\sum u_i}{\sum u_i + \sum w_i} = \frac{\sum u_i}{\sum u_i + 0} = 1.$$
   El corregido excluye $(i, \texttt{idx}[i])$. Sobre los 6.299 jugadores: con
   el error, $H = 1{,}0000$; corregido, $H = 0{,}8422$. Sigue por encima del
   umbral de $0{,}70$ (hay tendencia a agruparse), pero **los $H \approx 0{,}999$–$1{,}000$
   citados en §4.X, §8.D y el changelog no son mediciones válidas**.
2. **k fijo presentado como óptimo.** El panel B se titulaba "Selección Óptima
   (k=5)" con `K_OPT = 5` escrito a mano. Barrido sobre los datos actuales:
   $$\begin{array}{c|ccccccc} k & 2 & 3 & 4 & 5 & 6 & 7 & 8 \\ \hline
   \text{silhouette} & 0{,}362 & 0{,}217 & 0{,}202 & 0{,}172 & 0{,}160 & 0{,}154 & 0{,}153
   \end{array}$$
   El dato sugiere $k = 2$, previsiblemente la separación soporte / no soporte
   que domina PC1. $k = 5$ puede sostenerse por interpretabilidad, pero es una
   **decisión de modelado** que el usuario tiene que tomar. Se dejó `K_OPT = 5`
   sin cambiar el resultado, y el panel marca a la vez el k elegido y el de
   silhouette máximo. Con `K_OPT = None` se usa el segundo.
3. **Nombres de ejes del PCA escritos a mano** ("Asignación de Recursos… vs
   Utilidad/Visión"). Ahora se derivan de las `N_TOP_LOADINGS` cargas de mayor
   $|\cdot|$, con su signo: PC1 (38,2 %) = +Total CS, +CS @15, +% Damage Share;
   PC2 (19,3 %) = +GD @15, +ΔDamage End, +ΔCS @15.
4. **Escala de flechas fija** (`3.6`): ahora es relativa al percentil 95 del
   rango de los puntos, así que el biplot sigue legible si cambian las variables.
5. **Textos con conteos fijos** ("los 5 arquetipos", "Challenger + Grandmaster"
   en un análisis que ya incluía a todos): ahora se calculan desde `k_opt`,
   `len(mature)` y `MIN_GAMES`.
6. **Variables y etiquetas en dos listas paralelas**: ahora están en un único
   diccionario `FEATURES`, así que no pueden desalinearse.

En la sección 17 se corrigieron los textos que hablaban de "700 invocadores de
Apex", la etiqueta `Pool` → `Pool Disp`, y se bajó `dpi` de 300 a 150. A 300 dpi,
la figura de 22×14 pulgadas mide 6.600×4.200 px; la última ejecución guardada
de esa celda terminó en un cuelgue del kernel.

**Lo que no se cambió y queda para el usuario:** la firma de embeddings tiene
22 dimensiones (`sig_0..sig_21`), pero ninguno de los dos gráficos la usa: solo
entra `pool_dispersion`. Llamar a la sección "con embedding" es correcto solo en
ese sentido acotado.

#### S.3. `player_id_map` se refresca al cerrar cada sesión de recolección (2026-10-21)

`player_id_map` (tabla permanente) es la versión materializada del union-find de
§9.Q.2. La leen dos funciones del camino caliente del crawler:
`get_player_match_count()` (cuántas partidas tiene ya la persona, para decidir
cuántas pedir) y `get_canonicals_for_deep_crawl()` (la cola). Hasta ahora solo la
regeneraba `resolve_inverse_by_match.py`: después de un crawl, los PUUIDs recién
vistos no pertenecían a ninguna persona en esa tabla, así que sus partidas no
sumaban al conteo de su dueño.

El efecto era acotado: no duplica a nadie (el export arma su propio mapa
temporal en cada corrida) y lo único que se pierde son requests, porque pide
partidas de más que el filtro por `match_id` descarta. Aun así, es estado
derivado que puede desfasarse de su fuente, y la corrección es barata.

**Decisión:** `collect_matches()` llama a `materialize_player_id_map()` al cierre,
**después** del `join` del hilo escritor (sin escrituras concurrentes) y **solo si**
la sesión guardó partidas. Costo medido: reconstruir el mapa de 119.134 PUUIDs
lleva ~11 s, contra sesiones de horas. Si falla, se informa y queda el mapa
anterior: los datos ya están guardados.

*Alternativas descartadas:* (a) refrescar durante la sesión, porque compite por
el lock con el escritor y el crawler ya tolera conteos levemente atrasados; (b)
eliminar la tabla permanente y recalcular en cada consulta, porque
`get_player_match_count()` se llama una vez por jugador y 11 s por llamada es
inviable; (c) actualizar el mapa de forma incremental en la ingesta, porque una
partida nueva puede fusionar dos componentes y el union-find no admite deshacer
etiquetas sin recalcular.

Test: `test_session_refreshes_player_id_map_with_new_puuids`, que falla sin la
corrección y pasa con ella. De paso apareció un caso borde **no corregido**:
`export_players()` levanta `TypeError` si todas las partidas carecen de timeline,
porque las columnas de diferenciales quedan todas en `None` y no se pueden negar.
Con los datos reales no ocurre (timeline al 100 %); quedó anotado en `TAREAS.md`.

#### S.4. Proyecto de cada PUUID como dato de primera clase (2026-10-21)

**La causa común de los problemas de keys.** Riot cifra cada PUUID con el
proyecto de la key que hace la petición, y una key solo descifra los de su
proyecto. Hasta ahora el sistema **nunca registraba de qué proyecto era cada
PUUID en el momento en que lo recibía**: lo suponía después (P0 por convención,
posición en el pool, "canónico" de la vista). Cada parche anterior corrigió una
de esas suposiciones y dejó vivas las demás. Medido en solo lectura sobre la BD
del 2026-10-21, con pool P0+P1:

1. **La cola mandaba PUUIDs ajenos.** `get_canonicals_for_deep_crawl()` devolvía
   `COALESCE(v.canonical_puuid, p.puuid)`. La resolución inversa del 07/09
   (hecha solo con la key de P3) había dejado PUUIDs de P0 del padrón como alias
   de su PUUID de P3, así que la vista los "traducía" a P3. De 21.078 jugadores
   en cola, 10.882 salían con PUUID sustituido: 3.140 de P3 (400 seguro), 5.991
   de proyecto desconocido y 1.751 de P1.
2. **Esos jugadores nunca quedaban marcados como visitados:**
   `mark_player_crawled()` hacía `UPDATE players WHERE puuid = ?` con el PUUID
   sustituido, que no está en `players`. Se reintentaban en cada sesión.
3. **La ingesta reescribía PUUIDs al guardar** con el caché de traducción, que
   para 3.224 PUUIDs de P0 y 571 de P1 apuntaba a uno de P3. Además etiquetaba
   el traducido como P0 en `puuid_riot_ids` (4.088 etiquetas falsas), lo que
   contaminaba el invariante de proyectos de §9.Q.2: 59 de las 60 personas
   separadas con `~` eran cortes falsos.
4. **El proyecto de una key salía de su posición en la lista,** que `run.py`
   filtra al descartar keys vencidas, y el recargador de `.env` reemplazaba por
   posición. Si se vencía la key de P0, la de P1 heredaba la ranura 0.
5. **Error preexistente en la cola:** `HAVING existing_recorded_games < ?`
   resolvía el nombre contra la columna homónima de la CTE (`NULL` sin partidas)
   y no contra el alias `COALESCE(…, 0)`. Todo jugador del padrón con 0 partidas
   quedaba fuera: 13.442 Diamond nunca entraban a la recolección.

**Diseño.** El proyecto pasa a ser un dato que se registra cuando Riot lo dice,
nunca una suposición:

* **Directorio `puuid_projects(puuid, project_index, source)`.** `source =
  'exact'` cuando lo dijo Riot y `'inferred'` cuando se deduce; un deducido nunca
  pisa un exacto. Fuentes exactas: `players` (sincronizado por la key de P0),
  `player_puuid_map`, las etiquetas 1 y 3 de `puuid_riot_ids` (0 contradicciones
  sobre 5.092 verificables y una corrida de una sola key, respectivamente), y
  desde ahora la ingesta. La etiqueta 0 de `puuid_riot_ids` se descarta: 107
  contradicciones contra el mapa.
* **Deducción por partida** (`src/features/puuid_projects.py`). Una partida se
  descarga con una sola key, así que sus PUUIDs crudos comparten proyecto. Un
  PUUID sin etiqueta toma el proyecto mayoritario de los PUUIDs exactos de sus
  partidas si la proporción $\ge 0{,}95$. Validación *leave-one-out* sobre
  30.149 PUUIDs exactos: cubre el 86,9 % con **98,65 % de acierto**. El error
  sale de filas que la ingesta vieja había traducido.
* **La ingesta guarda el PUUID tal cual llega** (sin traducir) y anota como
  exacto el proyecto de la key que descargó la partida. El cliente expone ese
  dato por hilo (`last_served_project()`). La unificación por persona ya era de
  lectura (`player_id_map`), así que la traducción al escribir no aportaba nada
  y era la que fabricaba PUUIDs indescifrables.
* **El proyecto de una key se busca por la key** en el `.env`
  (`project_index_for_key`), y el recargador reemplaza por proyecto
  (`refresh_pool(..., projects=...)`). Una key ajena al `.env` se usa igual,
  pero sus respuestas no etiquetan nada.
* **Ruteo de listas.** La cola devuelve, por persona, `players_puuid` (P0 del
  padrón, el que se marca como visitado) y `candidates = {proyecto: puuid}`.
  `get_match_ids_by_puuid(..., candidates=...)` elige el proyecto cuya key está
  más libre y le manda solo el PUUID de ese proyecto.
* **Autocorrección.** Cada lista de partidas confirma o refuta una etiqueta: un
  200 la vuelve `'exact'`, y un `400 Exception decrypting` la borra y prueba el
  proyecto siguiente. El crawler acumula ese feedback y lo escribe el hilo
  escritor.

**Por qué es sostenible y no pierde eficiencia.**

* *Ningún request se manda a una key que no pueda responderlo*, salvo un
  deducido erróneo (≈1,35 % de los deducidos), que se corrige en ese mismo
  request y no se repite.
* *El sistema aprende solo.* Al bajar las partidas de un jugador, cada una que
  descarga P1 trae su PUUID de P1 exacto. Desde ese momento sus listas se
  reparten entre las dos keys, sin Account-V1 ni scripts extra. Lo mismo vale
  para cualquier key nueva o renovada.
* *Capacidad.* Las descargas de partidas y timelines (2 requests por partida)
  van siempre a la key más libre; solo las listas de los jugadores con PUUID de
  un único proyecto quedan atadas a esa key. Hoy, 10.946 de los 34.521 jugadores
  de la cola ya tienen PUUID de P0 y de P1.

*Alternativas descartadas:* (a) sincronizar los PUUIDs de P1 por Account-V1
(~6–10 h de cuota de P1) en lugar de deducirlos y aprenderlos gratis; queda como
opción si el reporte de cierre muestra a P0 saturado; (b) seguir traduciendo al
escribir pero a "un P0 seguro", porque solo mueve el problema al día en que P0
no esté en el pool; (c) inferir el proyecto por posición o por hint de la key,
porque ambos cambian al renovar o reordenar keys.

**Aplicado sobre la BD viva** (respaldo previo en
`data/backups/lol_las_pre_puuid_projects_20261021.db`): 79.994 PUUIDs exactos y
23.344 deducidos en `puuid_projects`; `player_id_map` recalculado, con las
personas separadas con `~` bajando de 60 a 1 y el padrón intacto en 40.821
personas. Cola simulada con la función real: **34.521 jugadores (antes 21.078),
el 100 % con PUUID de P0 descifrable y marcable.**

Tests: `tests/test_puuid_projects.py` (16 casos, incluido el de
`jugador_d#tag`) y la reescritura de
`test_batch_save_keeps_raw_puuid_and_records_its_project`. También se corrigieron
dos detalles de `export_to_csv.py`: `player_tier` se resuelve por persona en
`full_dataset.csv` (132.032 filas del padrón sin tier), y `export_players()` ya
no levanta `TypeError` sin timelines.

#### S.5. Cuota real de Riot en el limitador (keys permanentes, 2026-10-21)

Con dos keys que no vencen (P0 y P1, verificadas por hint contra `api_projects`
y en uso desde el 04/09), el techo de la recolección es la cuota de cada key, y
la ventana que manda es la larga. El limitador tenía dos defectos:

* **Margen fijo:** `95 / 120 s` escrito a mano contra un límite real de `100`.
  Con dos keys que trabajan sin parar, el 5 % de la cuota queda sin usar de forma
  permanente.
* **Cabeceras mal usadas:** `update_from_headers()` solo **dormía el hilo** 2 s
  cuando `X-App-Rate-Limit-Count` pasaba de 96 en la ventana de 120 s. Bloqueaba
  un trabajador sin liberar cuota e ignoraba el límite informado, así que una key
  con más cuota (producción, `500:10,30000:600`) seguiría frenada a 95/120 s.

**Decisión:** el límite pasa a ser el que informa Riot en `X-App-Rate-Limit`,
menos `HEADER_MARGIN = 1` para cubrir peticiones en vuelo:

$$\text{límite efectivo} = L_{\text{Riot}} - 1 \qquad\Rightarrow\qquad 99/120\text{ s}\ \text{en vez de}\ 95/120\text{ s} \ (+4{,}2\ \%).$$

En cada respuesta se reconcilia el historial local con `X-App-Rate-Limit-Count`:
si Riot contó $c$ peticiones y el proceso registró $h < c$, se agregan $c - h$
marcas en el instante actual. Así, otro proceso que use la misma key (un
notebook, `check_api_keys.py`) no provoca un 429. Hasta recibir la primera
respuesta rigen los valores conservadores de siempre (18/1 s, 95/120 s).

*Alternativas descartadas:* subir los defaults a mano a 100, porque sin el conteo
de Riot cualquier petición no registrada produce un 429, que cuesta un
`Retry-After` completo; y quitar el margen, porque los hilos en vuelo todavía no
figuran en el conteo de Riot.

Tests: 4 casos en `tests/test_key_pool.py` (límite de una key de desarrollo, key
con más cuota, conteo de otro proceso, cabeceras ausentes o rotas).

#### S.6. Recolección por ventana de tiempo para el split temporal (2026-10-22)

**Decisión del usuario:** en lugar de completar 50 partidas por jugador en
todos los tiers, recolectar todas las partidas posibles de Master+ hasta el
2026-10-25 y hacer después el split temporal train/test con una regla fija.
Diamond queda fuera como meta de recolección; sus datos existentes no se tocan
(§3.G).

**Por qué no alcanzaba el crawler existente.** `collect_matches()` trabaja por
meta (hasta $N$ partidas por jugador) y solo recorre a quien está por debajo de
ella. Los 6.300 jugadores con $\ge 50$ partidas (Apex completo, 4.716 Master) no
se vuelven a visitar, así que sus partidas posteriores al último crawl solo
entran si aparecen en partidas de otro jugador. Un split temporal heredaría ese
sesgo: el período de test tendría huecos justo en los jugadores más maduros. Las
metas medidas tampoco entraban en el plazo: 50 partidas en todos los tiers
exige 1,39 millones de partidas-jugador de Diamond (> 600.000 partidas), y 50 en
Master, ~48.000 partidas (déficit de 82.163 / 1,71 Master incompletos por
partida), que caerían casi todas después del corte.

**Diseño** (`src/window_crawl.py`, `run.py --ventana-desde`):

* **Cobertura uniforme:** se recorre a todos los jugadores del padrón en los tiers
  elegidos (por defecto Challenger, GM y Master: 9.427 personas), sin importar
  cuántas partidas tengan, y a cada uno se le piden sus rankeadas en
  $[\text{desde}, \text{hasta})$ con `startTime`/`endTime` de Match-V5 (segundos
  epoch, 00:00 hora de Argentina, UTC−3 fijo porque no hay horario de verano).
* **Dos fases con estado en la base:** `crawl_window_players` (quién ya fue
  listado) y `crawl_window_matches` (IDs de la ventana).
  1. *Listar:* ~1 request por jugador y cada 100 IDs. Al terminar, `--fase
     estado` informa el **número exacto** de partidas pendientes y las horas
     estimadas ($2$ requests por partida, $49{,}5$ req/min por key). Con eso se
     elige la ventana antes de gastar cuota en descargas.
  2. *Descargar:* solo las listadas que no están ni en `matches` ni en
     `discarded_matches`, con la ingesta de §9.S.4 (proyecto exacto de cada
     PUUID) y `player_id_map` regenerado al final.
* **Reanudable:** un jugador queda listado solo si Riot respondió `200`
  (`last_request_ok()`), así que una caída de red no lo marca como "sin
  partidas". Una partida que no baja queda pendiente. Ctrl+C y la señal de pool
  agotado cortan ordenado, y la próxima corrida retoma.
* **Orden de descarga ascendente por `match_id`.** En LA2 el ID crece con el
  tiempo, así que si la descarga no termina, lo bajado cubre completo un período
  continuo que arranca en `desde`, pegado al train existente (que termina el
  07/09). El dataset termina donde llegó la descarga, sin huecos internos.
* **No exporta CSV al terminar:** el export es parte del cierre (§3.G y día de
  congelamiento), no de cada corrida.

*Alternativas descartadas:* (a) fijar el corte en una fecha y recolectar test
solo para los 6.164 maduros previos al corte, porque el usuario prefiere decidir
el corte sobre el dataset final con una regla, y la cobertura uniforme sirve a
ambos enfoques; (b) subir la meta por jugador a 300, porque el recorrido por
jugador llega en el plazo a una fracción del padrón con profundidad desigual y
no garantiza cobertura pareja en el tiempo; (c) descargar de la más nueva a la
más vieja, que también da un período continuo, pero deja un hueco entre el
train actual y lo nuevo si la descarga no termina.

**Pendiente de decisión antes de entrenar:** la regla del split (propuesta: el
20 % más reciente de las partidas por `game_creation`, con corte en el límite de
un día) se fija antes de mirar resultados de modelos.

Tests: `tests/test_window_crawl.py` (16 casos). Se verificó por mutación que
detectan tres errores plausibles: recorrer solo a los incompletos (9 tests
fallan), marcar como listado a un jugador cuya lista falló (1) y mandar las
fechas en milisegundos (12).

#### S.7. Listado concurrente y arranque calibrado del limitador (2026-10-22, corrida real)

Medido en la primera corrida real de `--fase listar` (ventana desde 2026-09-01,
9.427 jugadores Master+):

* **El listado iba a la mitad de la cuota.** 31 jugadores/min × 1,56 requests de
  lista por jugador ≈ 48 req/min, contra 99 de cuota con 2 keys. El reparto entre
  keys era correcto (1.562 listas confirmadas por P0 y 1.394 por P1, y el 99,6 %
  de los jugadores con PUUID de los dos proyectos). El problema era que la fase
  era **secuencial**: cada request esperaba su respuesta (~1,2 s) antes de mandar
  el siguiente. Ahora `list_phase()` usa un pool de hilos (por defecto
  $\min(16, 4 \cdot \text{keys})$) con todas las escrituras en el hilo principal.
  Medido sostenido en 5 minutos: **67,8 jugadores/min (×2,2)**.
* **429 en cadena al relanzar.** Riot cuenta la cuota por key, no por proceso. Al
  relanzar, la ventana de 120 s seguía llena por la corrida anterior, y 16 hilos
  con historial vacío salieron juntos antes de la primera respuesta: 6 errores
  429 seguidos con `Retry-After` de 42 → 17 s. El limitador de cada key arranca
  ahora en **modo calibración**: sale una sola petición hasta recibir la primera
  respuesta, que trae el conteo real (`X-App-Rate-Limit-Count`). Si esa respuesta
  no llega en 5 s, se libera igual para no bloquear.

Tests: `test_el_listado_es_concurrente` y 3 casos del arranque calibrado en
`tests/test_key_pool.py`.

#### S.8. Auditoría verificable de jugadores duplicados (2026-10-22)

`scripts/ops/audit_duplicates.py` (solo lectura, sin red, ~75 s) recalcula el
mapa de identidad en memoria y sale con código 1 si falla alguno de estos
chequeos. Cada uno corresponde a una forma distinta de duplicar o fusionar
jugadores:

| Chequeo | Qué delata | Debe dar |
| :--- | :--- | :--- |
| `padron_duplicado` | la misma persona con dos filas en `players` | 0 |
| `misma_persona_dos_veces_en_partida` | **fusión falsa** de dos personas (nadie juega dos veces la misma partida) | 0 |
| `partidas_sin_10_personas` | participaciones repetidas o fusiones | 0 |
| `personas_sin_nombre_con_partidas` | PUUIDs con partidas sin unir a nadie (mitades sueltas) | 0 |
| `listado_ausente_en_su_partida` | **jugador partido**: Riot listó la partida para él, pero al bajarla no aparece | 0 |
| nombres compartidos sin prueba | mismo `name#tag` en dos personas sin un PUUID exacto del mismo proyecto en cada una | 0 |

**Por qué el chequeo de listados es una prueba y no una heurística.** Riot
garantiza que cada partida de la lista de un jugador lo incluye. Si al bajarla
ninguno de sus 10 PUUIDs pertenece a la persona de ese jugador, su PUUID de esa
partida quedó en otra persona. Es el único caso que no deja rastro en los demás
chequeos: un cambio de nombre más un PUUID de proyecto todavía desconocido.
El registro (`crawl_window_listings`) guarda el **PUUID del padrón**, no el
`player_id`: la etiqueta puede cambiar al recalcular el mapa, y apoyarse en ella
haría que el chequeo pasara en silencio. Por lo mismo, un PUUID del padrón sin
persona en el mapa cuenta como falla (`LEFT JOIN`), y la consulta solo devuelve
0 sin evaluar cuando la tabla no existe.

**Exposición medida antes de bajar** (padrón Master+, 8.997 personas): 8.933
(99,3 %) ya tienen un PUUID exacto para P0 y para P1. Sus partidas nuevas llegan
bajo un PUUID que el sistema ya asocia con ellos, así que no pueden partirse ni
con un cambio de nombre. Las 64 restantes (33 sin P1 y 31 con P1 deducido)
quedan cubiertas por el chequeo de listados: las 15 que se habían listado antes
de que existiera el registro se volvieron a listar con `WindowCrawler.relist()`.

**Línea de base** sobre la BD del 2026-10-22: todos los chequeos en 0, y 3
nombres compartidos, todos con prueba de cuentas distintas (`jugador_c#tag`,
`jugador_e#tag`, `jugador_f#tag`: en cada uno, cada mitad tiene su propio PUUID
exacto del mismo proyecto y nunca jugaron juntas).

Tests: 4 en `tests/test_identity.py` y 4 en `tests/test_window_crawl.py`,
incluido el caso de renombre y el de cambio de etiqueta. Verificados por
mutación: si se anula la comparación de persona, fallan 2.

#### S.9. PUUIDs asignados a la persona equivocada en el mapa de P2 (2026-10-23)

**Cómo apareció.** El chequeo `listado_ausente_en_su_partida` (§9.S.8) dio 26
casos en su primera corrida, todos en partidas bajadas entre el 1 y el 7 de
septiembre, no en la descarga en curso. Eran 7 jugadores del padrón (p. ej.
`jugador_g#tag`, Challenger) a los que Riot listaba esas partidas pero que no
figuraban en ellas: uno de los 10 PUUIDs guardados estaba unido a otra persona.

**Causa.** Para esos PUUIDs había dos afirmaciones incompatibles:
`player_puuid_aliases` (resolución inversa por partida, que nombra a la persona
A) y `player_puuid_map` (mapa de proyecto 2 del 06/09, que lo asigna a la
identidad B). Mientras el PUUID tiene fila en el mapa, `v_puuid_canonical`
ignora su alias, así que quedaba unido a B. Verificado contra Riot en los 7: el
mapa era el erróneo.

**Criterio de corrección** (`scripts/identity/fix_map_contradictions.py`): para
cada PUUID en conflicto se pide a Riot, con la key de P0, una partida guardada
donde aparece. En el mismo `participantId` Riot devuelve el PUUID de P0 de quien
jugó: el dueño real.
* Si el dueño real ya es la persona a la que está unido el PUUID → **cambio de
  nombre**, no se toca.
* Si es otra persona → se borra la fila del mapa y el PUUID se vincula al dueño
  real con un alias.
* Para un PUUID que representa a dos personas según la partida (filas
  reescritas), existe la corrección fila por fila (`--filas-de-puuid`), que
  reemplaza solo las filas cuyo lugar ocupa otra persona por el PUUID de P0 que
  da Riot.

**Resultado** (respaldo previo en `data/backups/lol_las_pre_fix_map_20261023.db`):

| Lado del alias contrastado | Casos | Cambio de nombre | Mal asignado (corregido) |
| :--- | ---: | ---: | ---: |
| `alias_puuid` | 458 | 448 | 10 |
| `primary_puuid` | 377 | 376 | 0 (+1 sin respuesta: `jugador_g#tag`, ya corregido) |

El PUUID de `jugador_g#tag` (P2) se le había asignado entero a `jugador_h#tag` y
solo aparecía por el lado primario del alias, que la primera versión del
detector no miraba. Se corrigió fila por fila: 85 partidas, todas suyas según
Riot. Después de la corrección, la auditoría da **0 en los 6 chequeos**.

**Chequeo complementario sin requests:** ningún PUUID ni ninguna persona aparece
en dos partidas superpuestas en el tiempo en toda la base. Una persona no puede
jugar dos partidas a la vez, así que esto descarta fusiones de dos humanos
activos al mismo tiempo.

**Para que no se repita:** el mapa de P2 lo generó `register_key_project.py`; P2
ya no está en el pool. Si se vuelve a usar ese script con una key nueva, correr
después `fix_map_contradictions.py --dry-run` y la auditoría.

**Partidas de otros servidores (2026-10-23).** Match-V5 devuelve las partidas de
un jugador en todos los servidores de su región (`americas`: BR1, LA1, LA2,
NA1). La ventana del 01/09 trae 441 de otros servidores sobre 139.669 (0,3 %),
de jugadores de LAS que también juegan rankeds en Brasil, Norteamérica o LAN.
Como el `match_id` ordena alfabéticamente, `BR1_` y `LA1_` se bajaban antes que
`LA2_`, fuera de orden temporal. Desde esta fecha `WindowCrawler` descarga solo
partidas con el prefijo del servidor del proyecto (`DEFAULT_PLATFORM`). Las 328
que ya están en la BD (279 de esta descarga y 49 previas) no se borran (§3.G): se
filtran en el preprocesamiento por prefijo de `match_id`. Test:
`test_solo_se_descargan_partidas_del_servidor_del_proyecto`.

#### S.10. Cola del crawler normal: menos partidas primero (2026-10-23)

**Pedido del usuario:** priorizar a los jugadores con menos partidas. Sus
partidas traen a otros jugadores del padrón, así que los que están cerca de 50 se
completan de rebote, sin gastar requests de lista en ellos. Orden de
`get_canonicals_for_deep_crawl()`:

1. jugadores con alguna partida y por debajo de la meta, antes que los de 0;
2. nunca visitados antes que los ya visitados, para que quien agotó su historial
   (por ejemplo, 45 rankeds en toda la temporada) no se reintente al principio en
   cada sesión;
3. **menos partidas primero**;
4. visita más vieja primero.

Al 23/10, la cola de Master+ con menos de 50 tenía 3.024 personas: 154 con 1–9
partidas, 1.319 con 10–29 y 1.551 con 30–49. Además, `collect_matches()` refresca
`player_id_map` al arrancar y la descarga por ventana lo refresca cada 5.000
partidas: la descarga del 22–23/10 se cortó a la fuerza antes de su refresco
final, y sin esto los conteos por persona salían de menos.

Tests: `test_la_cola_prioriza_a_los_de_menos_partidas` y
`test_el_crawler_refresca_el_mapa_al_arrancar`.

#### S.11. Objetivo final del dataset y recolección desatendida (2026-10-23)

**Objetivo fijado por el usuario:** **50 partidas por persona** en todo el
dataset (train) y, aparte, **un ~20 % extra de partidas recientes repartidas en
todos los rangos** (test). "Rangos" se entiende como Master+, porque Diamond
quedó fuera de la recolección. Hoy (129.242 partidas), Challenger y GM tienen el
100 % con $\ge 50$ y Master el 66,6 % (5.526 de 8.297). Según el récord de liga y
el listado de la ventana, 2.750 de los 2.771 restantes tienen al menos 50
rankeds en la temporada; los 21 que no llegan tienen una mediana de 41.

**Defecto que lo impedía: la paginación del crawler normal.** El bucle de listas
cortaba con `page_start < target_for_player`: con meta 50, solo miraba las ~100
partidas más recientes de cada jugador. Si esas ya estaban descargadas (por
ejemplo, por la ventana), el jugador no avanzaba nunca, aunque tuviera partidas
más viejas. Ahora pagina hacia atrás de a 100 (el máximo de Riot) hasta
completar la meta o agotar el historial, con un tope de 1.000 IDs por jugador y
sesión (`MAX_HISTORY_IDS`). También descarta partidas de otros servidores, igual
que la ventana. Test `test_el_crawler_pagina_hacia_atras_hasta_completar_la_meta`,
verificado por mutación: con el corte viejo, falla.

**Secuencia desatendida** (`scripts/collection/secuencia_fin_de_semana.py --hasta
"AAAA-MM-DD HH:MM"`), con cada paso reanudable:
1. crawler normal (menos partidas primero, meta 50) mientras siga bajando la
   cantidad de Master+ con menos de 50 (pedido del usuario: completar Master antes
   que el test);
2. terminar la ventana del 01/09 (cobertura pareja de Master+, base del test);
3. ciclos de partidas recientes: listar y descargar desde el día del ciclo
   anterior, hasta la hora de corte.

Un paso se repite mientras avance y se corta tras 3 intentos seguidos sin
progreso (con espera de 60 s, por ejemplo sin red). El log queda en
`logs/secuencia_fin_de_semana.log`. Estimación con 2 keys (~2.900 partidas/h):
paso 1 ~34 h, paso 2 ~6–8 h, paso 3 ~4–5 h para los días hasta el corte.

**El split se define en el preprocesamiento, no en los CSV (§3.G):** el train
toma las 50 partidas por persona y el test, las partidas recientes extra,
estratificadas por rango. La regla exacta se fija antes de mirar resultados de
modelos.

Tests: `tests/test_secuencia.py` (4 casos). Una fixture `autouse` redirige `LOG` a
un directorio temporal en todos los casos: antes, tres de ellos escribían líneas
`t: 10 -> 6` en el log real, y en una corrida desatendida eso se confunde con
progreso del crawler. Lanzada el 2026-10-23 18:09 con corte `2026-10-26 06:00`
(punto de partida: 129.242 partidas, 3.024 Master+ con < 50, 98.506 IDs pendientes
en la ventana del 01/09).

**Relleno del paso 3 y orden suave por LP (2026-10-25, pedidos del usuario).**
Entre ciclos del paso 3, en vez de dormir hasta 3 h, la secuencia corre el crawler
normal con meta de 300 partidas por jugador: primero GM y Challenger y, si termina
antes, Master (`--partidas-por-jugador 300`). Dentro de cada grupo la cola se ordena
con un puntaje suave (`--prioridad-lp 0.7`, `priorizar_por_lp` en `src/collector.py`):

* puntaje = 0,7 × percentil de elo (tier × 10.000 + LP, alto primero) + 0,3 ×
  percentil de "menos partidas primero"; menor puntaje va antes;
* nadie sale de la cola, solo cambia el orden: un Master bajo con muy pocas
  partidas todavía puede adelantarse a un Master alto que ya tiene muchas;
* el tier pesa más que el LP (el LP se reinicia al subir de tier).

Motivo medido sobre 253.008 partidas (25/10): un jugador Master del padrón con LP
en el quintil más bajo (< 49) juega el 26,2 % de sus partidas en lobbies de rango
Diamond (3,2 Diamond del padrón por lobby); en el quintil más alto (> 474 LP), el
3,1 % (1,6 por lobby). El 95 % de las 73.357 partidas de rango Diamond tiene al
menos un Master+ del padrón: salen del historial de nuestros Master+, no de buscar
Diamond. Con el relleno limitado por tiempo, priorizar LP alto trae más partidas
Master+ y menos lobbies con jugadores que el modelo vio poco.
*Alternativa descartada:* filtrar por LP mínimo (rígido; el usuario pidió algo
suave). Tests: `tests/test_prioridad_lp.py` (5 casos, verificado por mutación).

#### S.12. Test proporcional por rango (2026-10-23)

> **Reemplazada el 2026-10-24 por §9.S.17** (corte de fecha fijo, test sin
> estratificar). Se conserva como registro de la alternativa descartada.

**Pedido del usuario:** el test tiene que tener el mismo reparto por rango que el
dataset (si Master es el 80 % de las partidas, también es el 80 % del test).
Implementado en `src/features/split.py`, como preprocesamiento (no toca los CSV):

* **Rango de una partida (regla vigente desde el 2026-10-25, pedido del usuario):**
  el tier más alto $t$ tal que el lobby tiene **al menos 5 jugadores del padrón de
  tier $t$ o superior**. Con $n_{\ge t}$ = jugadores del padrón con tier $\ge t$:
  $$\text{rango} = \max\{\, t : n_{\ge t} \ge 5 \,\},\qquad
  \text{DIAMOND} < \text{MASTER} < \text{GRANDMASTER} < \text{CHALLENGER}.$$
  Si ningún nivel junta 5 queda `SIN_RANGO`. Es acumulado: 3 Challenger + 3 GM
  es GM (6 de GM o superior), y no "sin rango" como daría contar cada tier por
  separado.
  *Regla anterior, descartada:* la mediana de los tiers de los jugadores del
  padrón presentes (empates hacia abajo). Ignoraba a los jugadores fuera del
  padrón, así que el umbral real cambiaba con cuántos había en el lobby, y un
  estrato Challenger exigía la mitad del padrón presente en Challenger. Medido
  sobre 191.439 partidas (24/10): los Challenger aparecen en 22.265 partidas
  (11,6 %), pero solo 779 tenían mediana Challenger; el lobby medio tiene 5,5
  Master, 3,0 Diamond, 0,6 GM, 0,24 Challenger y 0,6 fuera del padrón. Con la
  regla nueva, la proyección del dataset final da ~1.860 partidas Challenger
  (antes ~900) y ~11.600 GM.
* **Split:** un único corte de día $T$. Train = todo lo anterior. Para cada rango
  $r$, el test toma las $q_r = \operatorname{round}(\tfrac{f}{1-f}\,n^{\text{train}}_r)$
  partidas más recientes posteriores a $T$, así que la fracción de test es $f$ en
  cada rango y el reparto del test coincide con el del train. $T$ es el día más
  tardío que deja cuota suficiente en todos los rangos. Lo posterior a $T$ que
  sobra queda `fuera` para no romper la separación temporal.

*Alternativa descartada:* un corte por rango (el 20 % más reciente de cada uno),
porque las partidas de train de un rango podrían ser posteriores a las de test de
otro, y las features de jugador mezclarían información del futuro.

**Vista previa** sobre los datos del 23/10 (cambia con la recolección del fin de
semana): corte 2026-08-30, train 84.163, test 21.041, fuera 23.710. El reparto
del train y del test es idéntico: Master 79,4 %, Diamond 14,7 %, GM 5,2 % y
Challenger 0,7 %. Las 23.710 "fuera" son partidas recientes de lobbies con
mediana Diamond, que abundan en la ventana porque los Master juegan con Diamond.

Tests: `tests/test_split.py` (6 casos: reparto, orden temporal, retroceso del
corte, sobrante fuera del train, rango de la partida y fracción inválida). El del
rango cubre el umbral acumulado y `SIN_RANGO`; verificado por mutación (sin el
acumulado, falla).

**El test no lleva Diamond (decisión del usuario, 2026-10-24).** El test sigue
siendo temporal (un solo corte $T$) y proporcional por rango, pero solo entre los
rangos `TEST_TIERS = (MASTER, GRANDMASTER, CHALLENGER)`. Formalmente, con $E$ el
conjunto de rangos elegibles:

* la cuota es $q_r = \operatorname{round}(\frac{f}{1-f}\,n^{\text{train}}_r)$ solo
  para $r \in E$, así que el reparto del test coincide con el del train **restringido
  a $E$** (`reparto_train_rangos_test` en el manifiesto) y la fracción $f$ vale
  dentro de cada $r \in E$;
* el corte $T$ se busca mirando solo los rangos de $E$: un rango que no va al test
  no puede obligar a retroceder el corte y achicar el train sin motivo;
* las partidas de rangos fuera de $E$ (Diamond y `SIN_RANGO`) anteriores a $T$
  quedan en train; las posteriores quedan `fuera`, para no romper la separación
  temporal.

*Por qué Diamond se queda en el train por ahora:* qué hacer con Diamond en el
train es una decisión pendiente del usuario. Dejarlo etiquetado `train` no la
condiciona: `match_split.csv` trae la columna `tier`, así que el preprocesamiento
puede filtrarlo después sin regenerar el split. Sacarlo ahora sería irreversible
dentro del export.

*Alternativa descartada:* mantener la regla original y filtrar Diamond del test
a posteriori. El test quedaría por debajo del 20 % en los rangos que sí lleva, y
el corte se seguiría calculando con la escasez de Diamond reciente.

**Todo lo Master+ posterior al corte va al test (decisión del usuario,
2026-10-25; `SOBRANTE_AL_TEST`).** El corte $T$ se sigue eligiendo con las cuotas
$q_r$, así que ningún rango de $E$ queda con menos del 20 %; pero después **todas**
las partidas de $E$ posteriores a $T$ se etiquetan `test`, no solo las $q_r$ más
recientes. Motivo: el estrato Challenger (~10 lobbies por día) fija $T$ muy
temprano, y con la regla exacta las partidas Master entre $T$ y el inicio del test
Master quedaban `fuera` — ~26.000 en la proyección del dataset final —, porque no
pueden ir al train sin que el train tenga partidas posteriores a partidas del test
(los mismos jugadores aparecen en lobbies de varios rangos, así que sus features
aprenderían de su propio futuro).

Proyección (~261.000 partidas): $T$ = 19/09, train 161.287, test 55.992, fuera
43.748 (solo Diamond y `SIN_RANGO` posteriores a $T$). Reparto del test: Master
94,5 %, GM 4,8 %, Challenger 0,7 %; del train en $E$: 91,6 %, 7,2 % y 1,2 %.

*Costo aceptado:* el reparto del test deja de ser idéntico al del train (los
rangos con más partidas recientes pesan más). Se compensa reportando las métricas
también por rango, o ponderándolas por el reparto del train.

*Alternativas descartadas:* (a) un corte por rango: train de un rango posterior a
test de otro con los mismos jugadores (leakage de features); (b) fusionar GM y
Challenger: solo mueve $T$ al 21/09 y deja ~50.000 `fuera`; (c) dejar el corte a
Master con Challenger/GM por debajo del 20 %: rompe la cuota mínima por rango que
pidió el usuario. Test: `test_con_sobrante_al_test_no_queda_nada_elegible_fuera`,
verificado por mutación.

**Base de la cuota desde el 01/09 (decisión del usuario, 2026-10-25;
`BASE_CUOTA_DESDE`).** $q_r$ se calcula con el train de $r$ posterior al 01/09, no
con todo el train. Motivo: antes del 01/09 la cantidad de partidas de cada tier
depende de cuánto historial se bajó (a GM+ se lo buscó más hondo, y el relleno del
paso 3 baja todavía más historial GM+); con la base completa, ese historial viejo
subía la cuota Challenger y corría el corte hacia atrás (~10 partidas de cuota por
día de corte, porque Challenger juega ~10 lobbies por día), y cada día de corte
mandaba ~1.100 partidas Diamond a `fuera`. Desde el 01/09 la ventana cubre a todo
el padrón Master+ por igual, así que la proporción mide lobbies reales.

Proyección (~261.000): $T$ = 08/10, train 212.190, test 22.557, fuera 26.383
(contra $T$ = 20/09, test 49.581, fuera 47.783 con la base completa).

*Consecuencia que queda a decisión del usuario:* el test es el 20 % del período
reciente de cada rango, pero solo el 8,6 % del dataset. **Identidad que acota el
problema:** después de cualquier corte, las partidas Master+ y las Diamond/sin
rango son casi la misma cantidad (proyección con test = 20 % de todo Master+:
corte 28/09, test 35.650, Diamond/sin rango posteriores 38.070). Con un corte único,
test sin Diamond y sin Diamond posterior al corte en el train, cada partida de test
deja ~1 partida `fuera`; recolectar más no cambia esa proporción. La única forma
de no desperdiciarlas es usarlas fuera del entrenamiento (validación temporal o
evaluación aparte sobre Diamond). Test:
`test_la_base_de_la_cuota_ignora_el_historial_anterior`, verificado por mutación.

**REGLA VIGENTE desde el 2026-10-25 (decisión del usuario): `master_plus_temporal_split`.**
Reemplaza en el export a la proporcional por rango (que queda en el código, con sus
tests, como alternativa documentada). Pedido del usuario: el test tiene que ser
grande (el 8,6 % de la variante anterior era poco), temporal, del 20 %, y Diamond
apartado para medir cómo rinde el modelo ahí.

* **Test = el 20 % más reciente de todo Master+**, con un corte de día único $T$:
  el día más tardío con al menos el 20 % de las partidas Master+ desde $T$. La
  fracción es sobre Master+ en conjunto, no por rango: cada rango queda con lo que
  jugó después de $T$ (Master 20,6 %, GM 14,2 %, Challenger 16,1 % en la proyección).
* **Diamond aparte, en las dos mitades:** `diamante_antes` (anterior a $T$) y
  `diamante_despues` (desde $T$). No entran al train ni al test. Las posteriores
  evalúan el modelo de Master+ en Diamond; las anteriores quedan para decidir si
  vuelven al train. Exportadas en `data/splits/diamante/{antes,despues}/`.
* **`sin_rango`** (lobbies donde ningún nivel junta 5 del padrón): aparte.

Por qué no por rango: la proporción exacta por rango obligaba a correr $T$ hasta
donde alcanzara el estrato más chico (Challenger, ~10 lobbies por día) y, sin
`fuera`, el test quedaba o muy chico (8,6 %, base desde el 01/09) o desbalanceado.
Por qué Diamond aparte y no `fuera`: así ninguna partida se pierde (objetivo del
usuario: el dataset más grande posible) y las Diamond posteriores al corte dan una
segunda evaluación sin contaminar el train.

Proyección (~261.800 partidas, corte de la recolección 26/10 08:00, sin contar el
relleno GM+ del paso 3): $T$ = 29/09, train 141.190, test 35.578 (20,1 % de
Master+), diamante 44.778 + 32.641, sin rango 7.584. Reparto del test: Master
94,6 %, GM 4,5 %, Challenger 0,9 %; del train: 91,9 %, 6,9 %, 1,2 %. Test:
`test_master_plus_test_es_el_20_por_ciento_mas_reciente_y_diamond_aparte`,
verificado por mutación (meta sobre todo el dataset en vez de Master+; Diamond
posterior al corte sin etiqueta propia).

**Diamond anterior al corte al train (decisión del usuario, 2026-10-26;
`DIAMANTE_ANTES_AL_TRAIN`).** El test sigue siendo solo Master+ (20 % más reciente,
corte único $T$, que se fija **solo con Master+**: sumar Diamond al train no lo
mueve). El Diamond **anterior** a $T$ pasa a `train`; el **posterior** queda aparte
(`diamante_despues`) para evaluar en Diamond. Motivo: 1 de cada 4 participaciones
del test es de un jugador Diamond, y sin Diamond en el train esos jugadores tienen
poco perfil. No hay leakage: todo ese Diamond es anterior a todo el test.

Medido sobre la base final (272.521 partidas de LAS, 26/10;
`scripts/analysis/cobertura_perfiles_test.py`): $T$ = 27/09, test 38.013 (20,4 % de
Master+; por rango Master 21,0 %, GM 13,5 %, Challenger 15,2 %), train 147.939
Master+ + 44.165 Diamond, Diamond desde $T$ 34.701, sin rango 7.703. Participaciones
del test: Master 59,5 %, Diamond 24,2 %, fuera del padrón 7,5 %, GM 6,4 %,
Challenger 2,4 %. Jugadores del test con al menos 20 partidas en el train:

| Jugador del test | Train solo Master+ | Train + Diamond anterior a $T$ |
| :--- | ---: | ---: |
| Diamond (≥ 10 / ≥ 20 / mediana) | 75,8 % / 60,8 % / 30 | 91,1 % / 80,4 % / 53 |
| Fuera del padrón (≥ 10) | 35,5 % | 53,0 % |
| Master+ (≥ 20) | 99,5–100 % | 99,9–100 % |
| Todos (≥ 20) | 84,6 % | 90,8 % |

*Pendiente para la tabla de features:* el ~9 % de participaciones del test con menos
de 20 partidas en el train (sobre todo fuera del padrón) se cubre con el prior del
rating por tier (regularización), con historial calculado hasta el día anterior a
cada partida de test (incluye el Diamond posterior a $T$, que es pasado) y con una
feature de "partidas conocidas del jugador". `sin_rango` anterior a $T$ (1.839)
queda aparte por ahora.

*Relación con la regla de `gonzz19` (§9.S.17 en esa rama: corte fijo al 01/09, test
= todo lo posterior, ~55 % del dataset, con Diamond):* se conserva su principio de
no descartar partidas y de un corte único, pero con test del 20 % de Master+ y sin
Diamond. Test: `test_master_plus_test_es_el_20_por_ciento_mas_reciente_y_diamond_aparte`
(ampliado), verificado por mutación.

**Proyección sobre el dataset final (2026-10-25, `scripts/analysis/proyeccion_split.py`):**
~261.000 partidas de LAS (206.907 bajadas + ~45.500 pendientes de la ventana + ~8.650
del paso 3). Con la regla vigente: corte $T$ = 18/09, train 158.907, test 30.964,
fuera 71.165 (40.806 Diamond y 26.123 Master posteriores a $T$). Test Master desde
el 05/10, GM desde el 23/09, Challenger desde el 19/09. **El estrato Challenger fija
el corte:** juega ~10 partidas de lobby Challenger por día, y su cuota del 20 % solo
se llena retrocediendo hasta el 18/09. No se resuelve recolectando: la ventana del
01/09 ya listó a todo el padrón Master+ (9.427 jugadores), así que todas las
partidas Challenger recientes ya están bajadas o pendientes; mover $T$ al 05/10
exigiría el doble de lobbies Challenger de los que se jugaron. Decisión pendiente
del usuario. *Nota de medición:* `crawl_window_listings` solo cubre a ~4.600
jugadores (se creó después de listar al resto); la lista completa de la ventana
está en `crawl_window_matches`, y es la que usa la proyección para contar.

Tests nuevos: `test_el_test_no_lleva_diamond_y_es_proporcional_en_el_resto` y
`test_un_rango_fuera_del_test_no_hace_retroceder_el_corte`, verificados por
mutación (quitar el filtro de rangos en la cuota rompe el primero; quitarlo en la
búsqueda del corte rompe el segundo).

#### S.13. Baseline sin leakage (2026-10-23)

**El defecto:** el baseline de §9.R calculaba las tablas de campeón (winrate y oro
medio @15 por campeón y línea) con `champions_by_lane.csv`, que agrega **todas**
las partidas, y después evaluaba esas mismas partidas. Cada partida contribuía al
winrate de sus propios campeones en la dirección de su resultado: leakage.

**La corrección** (`scripts/analysis/evaluate_baselines.py`, modo por defecto
`sin-leakage`): split temporal con el test proporcional por rango (§9.S.12);
tablas calculadas con `stats_from_participants(train)` y evaluación solo sobre el
test. El modo original queda como `--modo descriptivo`. Las reglas son las del
usuario, sin cambios.

**Resultado** (datos del 23/10: corte 2026-08-30, train 84.163, test 21.041):

| Target | Con leakage | Sin leakage | Referencia trivial (test) |
| :--- | ---: | ---: | ---: |
| Partida | 54,98 % | **52,64 %** (ROC-AUC 0,5381) | 51,29 % (siempre Red) |
| Líneas @15 (global) | 58,01 % | **57,97 %** (ROC-AUC 0,6140) | 50,34 % |

**Lectura:** el leakage inflaba el baseline de partida en 2,3 pp: sin él, supera a
"siempre Red" por solo 1,35 pp. El de línea casi no cambia, porque el oro medio
@15 de un campeón es estable con miles de partidas y no depende de una sola. Estos
son los números de referencia para los modelos de la Etapa 6. Se recalculan sobre
el dataset final del 26/10.

**Recalculado el 26/10** (dataset final; split vigente: corte 27/09, train 192.104,
test 38.013; `data/analysis/baseline_metrics_20261026.json`):

| Target | Baseline | Referencia trivial (test) |
| :--- | ---: | ---: |
| Partida | **52,83 %** (ROC-AUC 0,5416; Brier 0,2496) | 52,51 % (siempre Red) |
| Líneas @15 (global) | **57,47 %** (ROC-AUC 0,6087) | 50,60 % |

Con el test actual, la ventaja del baseline de partida sobre "siempre Red" baja a
0,32 pp, dentro del error muestral (el error estándar de una accuracy con
$n = 37.989$ es ≈ 0,26 pp). El AUC de 0,54 indica que el ordenamiento sí tiene algo
de señal. Pero la probabilidad que da la regla está peor calibrada que una
constante: predecir siempre $p = 0{,}5251$ da un Brier de
$0{,}5251 \cdot 0{,}4749 = 0{,}2494$, contra 0,2496 de la regla. Un modelo del
escalón 2 tiene que ganarle a los dos, con intervalos de confianza.

Tests: `test_las_tablas_salen_solo_de_las_partidas_que_se_le_pasan` y
`test_sin_leakage_el_test_no_cambia_las_tablas`. El segundo arma un test que
contradice al train: si el test se colara en las tablas, la regla cambiaría de
lado.

#### S.14. Train y test exportados por separado (2026-10-23)

**Pedido del usuario:** que las partidas del test vayan por separado, para no
tener problemas. `python export_to_csv.py --target splits` genera, sin tocar los
CSV de `data/` (§3.G):

```
data/splits/match_split.csv      match_id, split (train/test/fuera), tier
data/splits/split_manifest.json  corte, cantidades, reparto por rango, regla
data/splits/train/               full_dataset, matchups_1v1, players, champions_by_lane
data/splits/test/                full_dataset, matchups_1v1
```

**Por qué los agregados van solo en train:** `players.csv` (vectores de estilo
por jugador) y `champions_by_lane.csv` (winrate y oro por campeón) son promedios.
Calculados sobre todo el dataset, meterían información del test en las features
del train: el mismo leakage del baseline original (§9.S.13). En `train/` se
agregan solo con partidas de train, y el test no trae agregados: el modelo usa los
del train. `match_split.csv` es la fuente única de la asignación.

**Mecanismo:** las cuatro funciones de export aceptan `match_ids` (una tabla TEMP,
que funciona en modo solo lectura) y filtran la consulta. Sin `match_ids`, el
export es el de siempre. Las partidas de otros servidores quedan fuera de los dos
splits.

Tests: `tests/test_splits_export.py`. Verificado por mutación: sin el filtro en
`champions_by_lane`, un campeón que solo jugó en el test aparece en el train y el
test falla.

**Orden de la secuencia (2026-10-23, pedido del usuario):** primero el crawler
normal hasta 50 partidas por persona, y recién después la ventana del 01/09 y los
ciclos recientes que alimentan el test. Si algo se atrasa, la prioridad es
completar Master. Las partidas más recientes que baja el crawler normal son de la
ventana, así que también achican el paso 2.

#### S.15. Día y hora de cada partida (`match_times.csv`, 2026-10-23)

`python export_to_csv.py --target match_times` escribe `data/match_times.csv`, con
una fila por partida y columnas `match_id`, `servidor`, `game_creation`,
`inicio_art`, `fin_estimado_art`, `fecha`, `dia_semana`, `hora` y
`duracion_min`. Está en hora de Argentina (UTC−3 fijo) y se cruza con los demás
CSV por `match_id`. Con `player_id` de `full_dataset.csv` da el historial
cronológico de cada jugador (sesiones, descanso entre partidas, horarios).

`fin_estimado_art` = `game_creation + game_duration` puede quedar 1–2 min antes
del fin real: `game_creation` es cuando el servidor crea la partida, antes de la
pantalla de carga. *Alternativa descartada:* guardar `gameStartTimestamp` y
`gameEndTimestamp` exactos. Para las partidas nuevas no cuesta requests (vienen en
el DTO), pero para las 129.242 existentes exigiría volver a pedirlas, y para
sesiones y horarios la precisión de minutos alcanza (decisión del usuario).

#### S.16. Forma del día por jugador y partida (`daily_form.csv`, 2026-10-23)

`python scripts/analysis/build_daily_form.py` escribe `data/daily_form.csv`, con una
fila por jugador-partida de LAS (se cruza con `full_dataset.csv` por `match_id` +
`player_id`). Responde a cómo le venía yendo a ese jugador **ese día, antes de esa
partida**. La lógica está en `src/features/daily_form.py`.

**Día de juego de 08:00 a 08:00 ART:** `dia_juego = fecha(inicio_ART − 8 h)`. Una
partida a las 03:00 del martes cuenta en la sesión del lunes. Cortar a medianoche
partiría en dos la sesión nocturna típica.

**Columnas:**
- `n_previas_dia`, `ganadas_previas_dia`, `perdidas_previas_dia`.
- `racha`: con signo, +3 si viene de 3 victorias seguidas ese día.
- `primera_del_dia` y `minutos_desde_ultima`: el descanso desde el fin de su partida
  anterior **del mismo día**. En la primera del día queda vacío y `primera_del_dia`
  vale True. *Alternativa descartada:* medirlo desde la última partida de cualquier
  día. Mezcla dos cosas distintas, la pausa dentro de una sesión (minutos) y el
  tiempo entre sesiones (horas o días), en una misma escala muy asimétrica. La
  diferencia entre sesiones queda en el indicador.
- `winrate_habitual`, `indice_forma_dia`, `dia_completo` y `split`.

**Sin leakage, por construcción:** todo sale de las partidas *anteriores* del mismo
día. Por ejemplo, `ganadas_previas = cumsum(win) − win`, así que el resultado propio
nunca entra. Un test de mutación lo verifica: si se quita el `− win`, fallan 3 tests.

**Índice resumen:** es un winrate del día suavizado hacia el nivel habitual del
jugador y centrado en ese nivel:

$$\text{indice} = \frac{G + C\cdot WR_j}{n + C} - WR_j,\qquad C = 3$$

Vale 0 sin partidas previas y tiende a $G/n - WR_j$ cuando el jugador acumula
partidas. *Alternativa descartada:* usar el winrate del día crudo $G/n$. Con 1 sola
partida previa salta entre 0 y 1, y además no distingue a un jugador de 60 % que va
1/2 de uno de 40 % que va 1/2.

**$WR_j$ sale solo del train** (split de §9.S.12; desde el 2026-10-24, §9.S.17) y se suaviza hacia el winrate
global del train con $k = 10$:

$$WR_j = \frac{W_j + k\,\overline{WR}}{N_j + k}$$

Sin ese suavizado, un jugador con 1–2 partidas en train tendría $WR_j \in \{0, 1\}$.
Calcularlo con todas las partidas metería el resultado del test por la puerta de atrás.

**`dia_completo`:** dice si el jugador tiene bajadas *todas* las partidas de ese día.
- Solo se puede afirmar dentro de la ventana listada (`crawl_window_listings`, del
  2026-09-01 al último día de juego entero antes del listado).
- Solo vale para los jugadores con listado propio (4.612). Para el resto queda vacío.
- Vale False si al jugador le falta cualquier partida listada: una partida no bajada
  no tiene fecha, así que no se sabe en qué día cae.

Primera corrida, con la descarga de la ventana a medias: 1.289.140 filas; vacío
1.160.064, False 126.939, True 2.137. El 64,4 % de las filas tiene al menos una
previa en el día. El desvío del índice es 0,11. **Se regenera el día del freeze**,
después del export final.

#### S.17. Split temporal con corte de fecha fijo y test sin estratificar (2026-10-24) — ALTERNATIVA DESCARTADA

> **Estado (2026-10-26, decisión del usuario): descartada; la regla vigente es la de
> §9.S.12 (`master_plus_temporal_split`).** Se escribió en la rama del compañero
> (`gonzz19`) en paralelo a la recolección del fin de semana; el código vive en esa
> rama y en `gonchi` se revirtió (`git revert`), conservando su análisis de la huella
> de estilo (§10). Se deja el texto original abajo como registro de la alternativa.
>
> **Por qué se descartó.** Con la base final (272.521 partidas de LAS), el corte fijo
> al 01/09 deja **145.018 partidas de test (53,2 %)** y 127.503 de train: la ventana
> del 01/09 terminó bajando mucho más de lo previsto el 24/10 (40.144 de test). El
> usuario pidió un test del 20 % y sin Diamond (para medir aparte cómo rinde en
> Diamond); esta regla da más de la mitad de test y con un tercio de lobbies Diamond.
>
> **Qué se conserva de esta propuesta en la regla vigente:** no descartar partidas,
> un corte temporal único, informar el reparto por rango y reportar las métricas por
> rango (con repesado al train, $w_r = \pi^{\text{train}}_r / \pi^{\text{test}}_r$,
> si hace falta una cifra global). Su argumento del **train congelado** se cumple con
> la recolección cerrada: el corte vigente se calcula sobre la base congelada del
> 26/10 (27/09) y ya no se mueve; si se vuelve a recolectar, hay que fijarlo a mano.

**Pedido del usuario:** el train son las partidas viejas (~90k) y el test, todo lo
posterior, que es lo que baja la recolección por ventana del compañero. Reemplaza
la regla de §9.S.12. Implementado en `src/features/split.py` (`temporal_split`),
como preprocesamiento (no toca los CSV, §3.G):

$$\text{split}(m) = \begin{cases} \text{train} & t_m < T \ \text{test} & t_m \ge T \end{cases}
\qquad T = \texttt{TEST\_DESDE} = \text{2026-09-01 00:00 ART}$$

**Por qué $T$ es una fecha fija y por qué esa.** $T$ coincide con el inicio de la
ventana de recolección (`run.py --ventana-desde 2026-09-01`, §9.S.6), que es la
fuente del test.
* Un corte posterior (el 03/09 daba ~97k de train) haría que la ventana siguiera
  metiendo partidas en el train: el train nunca quedaría congelado, y el EDA hecho
  sobre él se invalidaría con cada corrida.
* Un corte definido como fracción del total (la regla anterior) se mueve con cada
  recolección, por el mismo motivo.

El costo es quedar en 88.770 partidas de train, apenas debajo de las 90k pedidas.
El crawler normal todavía puede sumar partidas viejas al train hasta el freeze
(pagina hacia atrás, §9.S.11): la regla queda fija; el contenido, hasta el 26/10.

**Por qué el test no se estratifica por rango.** La ventana baja solo Master+
(§9.S.6). Forzar en el test el reparto por rango del train obliga a una de dos
cosas: descartar partidas recientes (la regla anterior dejaba **23.710** `fuera`,
el 18 % de la base) o quedarse sin cupo en los rangos que la ventana no cubre. En
lugar de esconder el corrimiento de distribución, se mide y se informa:

| Rango (mediana del lobby) | train | test |
| :--- | ---: | ---: |
| Master | 70.491 (79,4 %) | 25.650 (63,9 %) |
| Diamond | 13.083 (14,7 %) | 13.081 (32,6 %) |
| Grandmaster | 4.606 (5,2 %) | 1.287 (3,2 %) |
| Challenger | 590 (0,7 %) | 126 (0,3 %) |
| **Total** | **88.770** | **40.144** |

(Datos del 24/10, 128.914 partidas de LAS; las 328 de otros servidores quedan en
ninguno.) El test tiene más del doble de lobbies con mediana Diamond: son partidas
de jugadores Master con compañeros Diamond, que la ventana baja enteras. Las
métricas se reportan **por rango** y, si hace falta una cifra global comparable,
repesadas al reparto del train ($w_r = \pi^{\text{train}}_r / \pi^{\text{test}}_r$).

*Alternativas descartadas:* la regla anterior (§9.S.12), por las partidas que tira
y porque supone que el test sale de la misma distribución por rango, cosa que la
recolección no garantiza; y un corte por rango, por la mezcla de información del
futuro que ya documentaba §9.S.12.

Tests: `tests/test_split.py` (6 casos: corte a las 00:00 ART, corte por defecto =
inicio de la ventana, test sin estratificar ni descartes, train congelado ante
partidas nuevas, corte que vacía un lado y mediana de rango). Verificados por
mutación: con `<=` en lugar de `<` fallan 3; con el corte por defecto movido al
03/09 falla 1. `test_splits_export.py` y `test_baselines.py` pasan el corte
explícito.

#### S.18. Clustering fuera del notebook de EDA (2026-10-26)

Las secciones 16 y 17 de `eda_visualizer.ipynb` (clustering global de jugadores y
clustering por línea con jerarquía de inexperiencia) pasan a
`notebooks/clustering_jugadores.ipynb`, sin cambios en el código.

*Por qué:* son modelado, no análisis exploratorio. Ajustan K-Means, eligen k y
nombran clusters. El EDA describe los datos que alimentan a los modelos, y mezclar
ahí resultados de un modelo cuyo k sigue sin decidir (§9.S.2: silhouette máximo
en k = 2) haría pasar una elección de modelado por un hecho del dataset. Las dos
celdas ya cargaban sus propios CSV (`players.csv`, `player_signatures_A/B.csv`,
`matchups_1v1.csv`) y la BD en solo lectura, así que el notebook nuevo corre solo.

*Alternativa descartada:* dejarlas al final del EDA con una advertencia. Obliga a
correr el EDA entero para ver el clustering y viceversa, y la presentación del EDA
quedaría atada a una decisión pendiente.

#### S.19. EDA rehecho: todo el dataset, sin números fijos (2026-10-26)

`notebooks/eda_visualizer.ipynb` se reescribió con las secciones acordadas con el
usuario y se ejecutó de punta a punta sobre los datos congelados del 26/10 (unos 3
minutos).

*Sobre qué datos.* Originalmente sobre **todo el dataset**, porque el test todavía no
estaba fijado. **Desde el 2026-10-27, las Partes B y C (secciones 7 a 19) usan solo el
train**: el test ya está fijado (§9.S.12) y la regla es que nada de lo que se vea en él
puede orientar una decisión del modelo. Con el EDA sobre todo el dataset, el meta de los
parches del test (16.19 a 16.21), el winrate de sus campeones y el perfil de sus
jugadores entraban en gráficos que sí orientan decisiones (qué normalizar, qué
variables usar). La Parte A sigue sobre todo el dataset, porque describe el reparto
entre particiones (secciones 1, 3 y 5) y la calidad de datos, y eso no orienta el
modelo.

Cómo se restringe: una celda al inicio de la Parte B filtra `full_dataset` y
`matchups_1v1` a las partidas con `split == "train"` (`match_split.csv`) y reemplaza
`players.csv` y `champions_by_lane.csv` por los de `data/splits/train/`, agregados
solo con partidas de train (§9.S.14). Filtrar las filas no alcanzaba: los agregados por
jugador del volcado completo promedian también sus partidas del test. Quedan afuera
también el Diamond posterior al corte, las partidas sin rango asignable y las de fuera
de LAS. Se descartó dejar el Diamond posterior: no es test, pero comparte período y
parches con el test y mostraría su meta por otro camino. La sección 19 lee el W/L de
la temporada de la tabla `players` de la BD, una foto del 04/09, anterior al corte,
así que no contiene partidas del período de test.

Los hallazgos 5, 7 y 8 se recalcularon sobre el train. Los demás que citan las Partes B
y C (4 y 6) cambian en décimas.

*Qué se sacó.* Bottom 10 de campeones, enfrentamientos 2v2 de bot (no aportan a ningún
modelo) y premades en Diamond (en Master+ no existe la cola dúo). El clustering pasó a
`clustering_jugadores.ipynb` (§9.S.18).

*Qué se agregó.* Descripción y particiones, calidad de datos, partidas por jugador en
cada partición (Master+ y Diamond por separado), el target por lado, el reparto de
varianza por rol y campeón ($\eta^2$) y las correlaciones dentro del rol.

*Sin números fijos.* Todo lo que se cita sale de las celdas: los parches de la meta
(los que tienen al menos 5 % de las partidas), el primer y el último parche, los
límites de los ejes, el umbral de las etiquetas y el conjunto de ADC (al menos 50 %
de las partidas del campeón en `BOTTOM`, en lugar de una lista escrita a mano). Los
parches se ordenan como números (16.9 < 16.10).

*Memoria.* `full_dataset.csv` ocupa unos 2,8 GB cargado y la máquina tiene 8 GB. Se
carga sin `puuid_raw` ni los ítems, con los textos repetidos como categoría y los
números en float32, y sin copias filtradas del dataset completo. Las 55 filas con una
posición fuera de las 5 líneas se cuentan y se sacan al cargar.

*Hallazgos que condicionan el modelado.*

1. **54 partidas sin ganador**: las 10 filas tienen `win = 0`. 52 son del 13/10/2026
   (probablemente partidas anuladas por una caída del servidor) y 45 caen en el test o
   en Diamond posterior. Como target cuentan como "ganó rojo", así que hay que
   filtrarlas en el preprocesamiento. El EDA ya las excluye del análisis del target.
2. **El lado rojo gana el 51,64 %** (azul 48,36 % ± 0,19, $n = 272.795$). Es el piso de
   "siempre rojo". En la sección 6 la ventaja crece en las partidas de más de 25 minutos.
3. **El faltante de `@20` es estructural**: el 100 % de las filas sin dato son partidas
   que terminaron antes del minuto. Usar `@20` sesga hacia las partidas largas.
4. **El rol domina las variables de farm y visión**: explica el 91 % de la varianza de
   monstruos neutrales, el 83 % de CS@15 y el 63 % de la visión. Sumarle el campeón
   agrega poco en esas variables y mucho en daño recibido (13,7 % → 28,5 %). Cualquier
   comparación entre jugadores tiene que normalizar por rol.
5. **El cambio de meta entre parches es casi todo ruido**: en el train, el desvío de
   $\Delta WR$ entre 16.13 y 16.18 es 2,66 pp, contra 2,53 pp esperados solo por ruido
   muestral (con todo el dataset, entre 16.13 y 16.20: 2,63 contra 2,47)
   ($\sqrt{\overline{p(1-p)/n_1 + p(1-p)/n_2}}$ con $p = 0{,}5$).
6. **Redundancia baja**: dentro del rol, solo daño a campeones y oro total superan
   $|\rho| \ge 0{,}8$ (0,83).
7. **Los jugadores son de un rol, pero no de un campeón** (sección 12, 11.146 jugadores
   con ≥ 50 partidas en el train). Con umbral del 20 % de las partidas (2026-10-27, antes
   10 %), el 56,1 % tiene un solo rol significativo y el 39,2 % dos; en campeones, el
   29,5 % no tiene ninguno que llegue al 20 %. El reparto ordenado por rol —para cada
   jugador, $s_{(1)} \ge \dots \ge s_{(5)}$ con $s_{(k)}$ la fracción de sus partidas en
   su $k$-ésimo rol más jugado, promediado entre jugadores— da 68,9 / 18,6 / 7,3 / 3,6 /
   1,6 %. El rol principal es una variable estable del jugador; el campeón no.
8. **Variables del jugador contra su winrate** (sección 18, agregada el 2026-10-27;
   la de WR ≥ 75 % pasa a ser la 19). Spearman sobre los promedios de `players.csv`
   del train (10.979 jugadores con ≥ 50 partidas), con cada variable pasada a percentil
   dentro del rol principal (mismo motivo que la sección 17). Las más asociadas son de
   resultado: KDA 0,38, diferencia de oro @15 0,34, diferencia de daño final 0,32,
   muertes −0,23. Las tempranas y de estilo quedan bajas: agresividad @10 0,07,
   campeones distintos −0,07, % en el rol principal 0,12; visión (−0,015) no se
   distingue de 0 (banda $|\rho| < 1{,}96/\sqrt{n-3} \approx 0{,}019$) y la cantidad de
   partidas apenas (0,04).
   Con 50 partidas el winrate tiene un error estándar de $0{,}5/\sqrt{50} \approx 7$ pp,
   que atenúa toda correlación con él: los valores son cotas inferiores.

*Barras de las secciones 13 y 14 (2026-10-27).* Antes mostraban el P25–P75 entre
partidas, que se leía como margen de error y tapaba las medianas: en la diferencia
contra el rival el P25 es negativo en casi todas las líneas (oro @15 en top: mediana
+431, P25–P75 de −709 a +1.577). Ahora la barra es el IC 95 % de la mediana por
estadísticos de orden, entre los cuantiles $\tfrac12 \pm 1{,}96\cdot\tfrac{1}{2\sqrt n}$
(aproximación normal a la binomial del rango de la mediana, sin suponer normalidad).
Con cientos de miles de filas por rol queda más chico que el marcador. A pedido del
usuario las secciones 13 y 14 quedan solo con el gráfico (sin tablas de medianas,
P25–P75 ni % adelante); las cifras de dispersión citadas aquí salen de esa versión
intermedia. En CS la variable es entera y el IC puede
quedar asimétrico o nulo. El daño se verificó: es acumulado, no decrece entre minutos y
a los 20 minutos coincide con el daño final (cociente mediano 1,00); su IQR es ancho
(±25 %) porque depende del campeón (rol + campeón explica bastante más que el rol en
daño @15, 33,7 % contra 25,0 %). Se descartó el IC por bootstrap: da lo mismo con
este $n$ y cuesta minutos sobre 2,7 M filas; y el error estándar de la media, porque
el daño y las diferencias son asimétricos y con colas.

*Por qué 20 % y el reparto ordenado.* Con 10 % entraban como "rol significativo" 5
partidas de autofill sobre 50, que es justo el ruido que el umbral debía filtrar. El
umbral sigue siendo arbitrario, así que se agregó el reparto ordenado, que no depende
de ninguno: muestra cuánto pesa el main frente al secundario de forma continua. Se
descartó promediar la fracción por rol fijo (TOP, JUNGLE…): mezcla jugadores de
distintos mains y da un reparto casi uniforme que no dice nada de la especialización.

*Alternativas descartadas.* Hacer el EDA solo sobre el train: no hay test fijado y
la parte descriptiva no orienta decisiones. Promedios globales en el timeline: mezclan
roles con niveles muy distintos. Mostrar ganadores y perdedores por separado en la
diferencia contra el rival: una distribución es la otra con el signo invertido.

#### S.20. Baselines en `src/models/baselines/` (2026-10-26)

Los baselines estaban repartidos entre un script (la regla), `fingerprint.py`
(`pool`) y el script del nivel 1 (`elo_linea`). Pasan a un paquete propio:

* `draft_rules.py`: la regla sin entrenamiento (§9.S.1), movida desde
  `evaluate_baselines.py`, que queda como comando fino. Se separó el cálculo de
  los puntajes (`match_rule_scores`, `lane_rule_scores`) de las métricas, para
  que otros modelos puedan usar los mismos puntajes.
* `draft_logreg.py`: regresión logística sobre el draft (escalón 2).
* `identificacion.py`: baselines del escalón 1. Entran `elo_linea` y el nuevo
  `embedding`. **`pool` se queda en `fingerprint.py` y solo se reexporta**, a
  pedido del usuario, para no generar conflictos con la rama del compañero.

Se eligió `src/models/` en lugar de una carpeta `models/` en la raíz porque así
el código se importa y se testea como el resto. La raíz queda libre para guardar
modelos entrenados más adelante.

**Logística del draft.** Usa una codificación antisimétrica (tipo Bradley-Terry):
una columna por (campeón, línea) que vale +1 en Blue y −1 en Red.

$$P(\text{gana Blue}) = \sigma\Big(b + \sum_{i \in \text{Blue}} w_i - \sum_{i \in \text{Red}} w_i\Big)$$

$w$ es la fuerza del campeón en esa línea y $b$ la ventaja de lado. La
regularización es L2, y $C$ se elige por log loss sobre el último 20 % del train
en orden temporal; después se reentrena con todo el train. Para las líneas hay un
modelo por línea, con el target "Blue tiene más oro @15". Se prueban dos entradas:
solo el duelo o el draft completo. Se evalúan solo partidas completas, así que
todos los modelos ven las mismas partidas.

Resultados en el test (`scripts/analysis/evaluate_draft_logreg.py`,
`data/analysis/modelos_draft.json`; 37.196 partidas y 184.072 duelos):

| Modelo | Partida: accuracy / AUC | Líneas juntas: accuracy / AUC |
| :--- | :--- | :--- |
| Constante | 52,51 % / 0,500 | 50,69 % / 0,505 |
| Regla | 52,86 % / 0,542 | 57,45 % / 0,610 |
| Logística del duelo | — | 58,32 % / 0,618 |
| **Logística del draft** | **53,52 % / 0,543** | **59,02 % / 0,629** |
| Logística con lados separados | 53,38 % / 0,540 | — |

Lectura:

1. **Partida.** La mejora sobre la regla está en la calibración y no en el
   ordenamiento: el AUC es casi el mismo, pero el log loss baja de 0,6924 a 0,6891.
   Separar los lados no aporta, así que la ventaja de lado es una sola constante
   ($b = -0{,}07$).
2. **Líneas.** El draft completo le gana al duelo donde la línea no es 1 contra 1:
   +1,6 pp en bot (el soporte decide buena parte del oro del ADC) y alrededor de
   +1 pp en jungla y soporte. En top y mid no agrega nada.
3. **El techo de un modelo aditivo de campeones** es ~53,5 % / 0,54 en la partida
   y ~59 % / 0,63 en las líneas. No ve interacciones (matchups ni sinergias) ni
   jugadores.

**Baseline `embedding` (escalón 1).** El perfil de cada muestra es el promedio de
emb(campeón, línea) sobre sus $k$ partidas. Usa el embedding de
`player_signatures.py` (22 estadísticas medias estandarizadas, pares con ≥ 100
partidas), construido con el `champions_by_lane.csv` del train. Se verificó que
esa tabla suma exactamente las 1.921.003 filas de línea del train, así que no hay
leakage. Un par sin vector usa el vector medio de su línea. Describe al jugador
solo por el *tipo* de campeones que elige. A diferencia de `pool`, dos campeones
que se juegan parecido quedan cerca, así que sigue funcionando con campeones
disjuntos. Los resultados están en §10.2.

*Alternativa descartada:* reutilizar `player_signatures_A.csv`. Promedia todo el
historial del jugador (incluido el test) y la prueba necesita un perfil por
muestra.

Tests: `tests/test_draft_logreg.py` (5) y `tests/test_identificacion.py` (4). Fijan
la antisimetría y el vocabulario del train; que el modelo aprenda al campeón
fuerte; que la validación sea el tramo más reciente del train; que el embedding
reconozca campeones parecidos donde `pool` no puede, y que no vea el estilo con
los mismos campeones.

#### S.21. Rating por jugador y partida: TrueSkill con la foto como medición (2026-10-27)

**El problema.** Riot no da el rango de un jugador al momento de cada partida. Las
partidas (match-v5) no traen el rango de los participantes, y la API de ligas
(league-v4) solo da el rango **actual**, sin historial. Lo único que tenemos es una
foto del padrón (`players`, 04/09/2026 22:51 UTC) con LP, victorias y derrotas de
40.821 jugadores Diamond+. Esa foto envejece: las filas sin tier del dataset pasan
del ~4 % (marzo a julio) al 9,7 % en septiembre y al 21 % en octubre, justo el
período del test.

**Fuentes descartadas** (investigadas el 2026-10-27). Ningún sitio de terceros
(OP.GG, u.gg, League of Graphs, Mobalytics, LoLDB, DPM.LOL) publica el historial de
LP por API; todos lo arman consultando el rango actual cada vez que alguien abre un
perfil, con huecos en los jugadores que nadie busca. Scrapearlos viola sus términos y
la política de Riot pide usar sus servicios oficiales. Blitz, Porofessor y similares
leen el cliente local: solo sirven para la cuenta propia. Los datasets públicos
(Kaggle, Zenodo) tienen el rango por partida o liga, nunca por jugador al jugar.
Reconstruir el LP partida a partida desde la foto tampoco funciona: antes del 04/09
tenemos solo una parte de las partidas reales de cada jugador (mediana desde enero:
Challenger 57 %, Grandmaster 43 %, Master 26 %, Diamond 3 %), y el error de sumar y
restar LP se acumula. Una propuesta intermedia (imputar las partidas faltantes con el
winrate del perfil restante, recorriendo hacia atrás desde la foto) se descartó
porque ataba las partidas no vistas a las observadas y suponía que la primera
partida observada era cercana al inicio del split.

**La solución: un sistema de rating que aprende de los resultados**
(`src/features/rating.py`). TrueSkill de dos equipos, sin empates. Se eligió sobre
Elo y Glicko porque esos son 1 contra 1; TrueSkill modela al equipo como suma de
habilidades y reparte la actualización según la incertidumbre de cada jugador.

* **Escala en LP absolutos:** Diamond IV 0 LP = 0, cada división +100, Master 0 LP =
  400 (Master, Grandmaster y Challenger comparten una escala de LP). Solo importan
  las diferencias, así que $\mu$ puede ser negativa (Emerald en partidas Diamond).
* **Predicción previa**, con $c^2 = \sum_i \sigma_i^2 + 10\beta^2$ y $\delta$ la
  ventaja del lado rojo:
  $$P(\text{gana azul}) = \Phi\Big(\frac{\sum_{azul}\mu - \sum_{rojo}\mu - \delta}{c}\Big)$$
* **Actualización** con $t$ el mismo cociente visto desde el ganador,
  $v = \varphi(t)/\Phi(t)$ y $w = v(v + t)$:
  $\mu_i \mathrel{{+}{=}} \pm(\sigma_i^2/c)\,v$,
  $\sigma_i^2 \mathrel{{\times}{=}} 1 - (\sigma_i^2/c^2)\,w$.
* **Dinámica por partidas no vistas.** Antes de cada partida,
  $\sigma^2 \mathrel{{+}{=}} \tau^2(1 + h)$, con $h = (N - n)/n$ las partidas no vistas
  por partida observada ($N = W + L$ de la foto, $n$ las observadas desde el inicio
  supuesto del split, 08/01/2026, tope 30). Solo antes de la foto: después, la
  recolección por ventana es casi completa. Así, un jugador del que vemos poco tiene
  un rating más incierto y se actualiza más fuerte con cada partida que sí vemos.
* **La foto como medición, en su fecha**, como un filtro de Kalman:
  $K = \sigma^2/(\sigma^2 + \sigma_{foto}^2)$, $\mu \leftarrow \mu + K(z - \mu)$,
  $\sigma^2 \leftarrow (1-K)\sigma^2$. Se descartó usar la foto como prior desde el
  inicio: para las partidas anteriores al 04/09 sería información del futuro.
* **Ventaja roja** $\delta = \Phi^{-1}(p_{rojo})\sqrt{10}\,\beta$, con $p_{rojo}$ la
  tasa del train: reproduce esa tasa con habilidades iguales.
* **Jugadores fuera del padrón** arrancan en $\mu_0 = 0$ (el piso del padrón) con
  $\sigma_0$ amplia y se ubican solos por contra quién juegan.

**Sin fuga del target.** Cada fila guarda el rating **previo** a la partida. Las
partidas del test reciben el rating con todo lo anterior (la foto y las partidas
previas, incluidas otras del test ya jugadas): es lo que se sabría en producción. Lo
que queda fijo antes del test son los **parámetros**, ajustados solo con el train.
Se descartó TrueSkill Through Time (suaviza con partidas posteriores): el rating de
una partida incluiría su resultado, y en el test solo existe la versión hacia
adelante. Una segunda foto posterior al test **no** puede usarse para ajustar
parámetros, porque sus W/L incluyen resultados del test; sí para validar.

**Fuera del modelo, como diagnóstico.** La cobertura y el sesgo de muestreo
$z = (WR_{obs} - WR_{foto})/\sqrt{WR_{foto}(1-WR_{foto})/n}$ se miran en el notebook.
No entran como variables del modelo de draft antes del 04/09, porque el winrate de la
foto incluye el resultado de cada partida anterior a ella.

**Estado: sin evaluar, por decisión del usuario (2026-10-27).** El rating se evalúa
recién con la **recorrida de la foto del 04/09 a una segunda foto**: cada jugador del
padrón arranca con su LP real, se recorren las partidas posteriores (cobertura casi
completa) y en la fecha de la segunda foto se compara el rating con el LP real. Esa
es la pérdida para ajustar $\beta$, $\tau$ y $\sigma_0$ y para medir el error según
la distancia en días. Hasta entonces, `data/rating_partida.csv` es **provisorio**.

Por qué no se evalúa antes, con lo que ya se midió:

* **Sin ancla, la escala absoluta no se identifica.** Un sistema de rating mide
  habilidad relativa a los rivales: si todos arrancan en $\mu_0$, un grupo de
  Challengers que juega entre sí no tiene forma de llegar a 2.400 LP. Evaluar antes
  de la foto mide sobre todo esa falta de ancla.
* **Elegir por log-loss no sirve para el elo.** Grilla de 24 combinaciones
  ($\beta \in \{800, 1600, 3200\}$, $\tau \in \{5, 20\}$, $\sigma_0 \in \{200, 400\}$,
  $\sigma_{foto} \in \{200, 800\}$) sobre la ventana del train desde el 15/08: la
  mejor (β = 1600, τ = 5, σ₀ = 400, σ_foto = 800) da log-loss 0,6919 contra 0,6926 de
  una constante, una mejora mínima: el matchmaking iguala a los equipos. Y esa
  configuración, que casi ignora la foto, ordena a los jugadores **peor** contra el
  LP real (Spearman 0,45, dentro de cada rango entre 0,06 y 0,30) que los parámetros
  por defecto (0,58). Cuanto más pesa la foto, peor predice: Riot arma las partidas
  por MMR, no por LP.
* **Alternativa descartada por ahora:** validación cruzada por grupos de jugadores
  (anclar con la foto a 4/5 del padrón y medir al 1/5 restante). Da una pérdida sin
  segunda foto, pero el usuario prefirió esperar la recorrida entre fotos, que mide
  el uso real (seguir a los jugadores después de la foto, el período del test).

**Hallazgo del diagnóstico: sesgo de muestreo de winrate.** Con
$z = (WR_{obs} - WR_{foto})/\sqrt{WR_{foto}(1-WR_{foto})/n}$ en jugadores con ≥ 20
partidas observadas antes de la foto, la media de $z$ es negativa en todos los
rangos: Challenger −0,20, Grandmaster −0,14, Master −0,19, Diamond −0,65. En nuestros
datos los jugadores ganan menos que en su registro real. La explicación más probable
es la recolección: a los Diamond los vemos sobre todo en partidas de Master+, donde
son los más débiles del lobby. Cualquier variable de winrate calculada con nuestras
partidas hereda ese sesgo.

**Segunda foto y recorrida entre fotos (2026-10-28).** La foto nueva se guarda
**aparte**, en la tabla `ladder_snapshots` (`snapshot_at`, `puuid`, `project_index`,
tier, división, LP, W, L, `fuente`), para no pisar `players`, que se sobrescribe en
cada sync. Script: `scripts/collection/snapshot_ladder.py`.

* **Fase `ladder`**, con la key de P0 (los PUUID del padrón son de P0): foto del
  28/10/2026 00:49:59 UTC, 43.526 jugadores (200 Challenger, 500 Grandmaster, 8.542
  Master, 34.284 Diamond) en 177 requests, todos 200.
* **Fase `sin_rango`**: los 31.418 jugadores de nuestras partidas de LAS que no estaban
  en el padrón del 04/09 ni están en el ladder de hoy, consultados uno por uno
  (league-v4 `entries/by-puuid`, método nuevo `get_soloq_entry_by_puuid`) con la key
  del proyecto de su PUUID, de más a menos partidas. Quien no tiene ranked solo queda
  como `UNRANKED`.
* La escala de LP absolutos se extendió hacia abajo (Emerald I 0 LP = −100, Emerald
  IV 0 LP = −400, Platinum I 0 LP = −500…), para ubicar a esos jugadores.

**Recorrida (exploratoria, sin archivo de salida).** Cada jugador del padrón arranca
en su LP exacto del 04/09 (incertidumbre inicial $\sigma_{ini}$); los demás en 0 con
$\sigma_0 = 400$. Se recorren las 134.695 partidas desde la foto 1 (las últimas son del
26/10 05:36 UTC: faltan ~2 días hasta la foto 2) y se compara el rating final con el LP
de la foto 2, en jugadores con ≥ 10 partidas en el período. Grilla de 18 combinaciones
($\beta \in \{200, 400, 800\}$, $\tau \in \{5, 15, 30\}$,
$\sigma_{ini} \in \{50, 150\}$); la mejor por MAE: β = 200, τ = 5, σ_ini = 150.
Referencia: "el LP no cambió" (el LP del 04/09 como predicción del de hoy).

| Rango hoy | n | MAE rating | MAE "no cambió" | ρ rating | ρ "no cambió" |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Todos (padrón) | 16.681 | **133** | 185 | **0,83** | 0,61 |
| Challenger | 197 | 345 | **332** | **0,83** | 0,67 |
| Grandmaster | 482 | **315** | 386 | **0,58** | 0,28 |
| Master | 7.622 | **145** | 217 | **0,72** | 0,43 |
| Diamond | 8.380 | **107** | 140 | **0,77** | 0,54 |

1. **El rating le gana a la foto vieja**: error medio de 133 LP contra 185, y se acerca
   más al LP de hoy en el 63,7 % de los jugadores. Ordena mucho mejor dentro de cada
   rango.
2. **No sigue la inflación de LP de Master+.** Entre fotos, el LP real subió (mediana)
   +170 en Challenger, +229 en Grandmaster y +115 en Master; el rating bajó −146 en
   Challenger y −59 en Grandmaster, y subió solo +31 en Master. TrueSkill es de suma
   cero y el LP de Master+ se acumula con las partidas: por eso falla justo en
   Challenger. Candidatas: una deriva de LP por partida jugada en Master+, o una
   recalibración por rango.
3. **Jugadores nuevos**: los 3.711 que no estaban en el padrón y hoy están en el ladder
   quedan con ρ = 0,63 y MAE de 164 LP, partiendo de 0 sin ninguna información.
4. **Sin fuga, el rating casi no predice partidas**: con LP exacto de arranque, el AUC
   en las partidas del train entre fotos (04/09 a 27/09) es 0,513. El 0,65 de la prueba
   anterior (tier del 04/09 aplicado desde enero) era casi todo fuga del target.

*Cuidado:* la foto 2 es posterior al test. Los parámetros elegidos con ella valen para
el **elo**, no para variables del modelo de draft evaluado en el test.

**Split propio del problema del elo y modelo "al revés" (decisión del usuario,
2026-10-28).** El objetivo es reconstruir el LP de cada jugador en cada partida
**hacia atrás**: partir de una foto, recorrer las partidas en orden inverso y deshacer
en cada una el cambio de LP que el modelo dice que produjo. **Train:** de la foto 1
(04/09) a la foto 2 (28/10), con LP real en las dos puntas: se recorre desde la foto 2
y se compara con la foto 1. **Test:** antes del 04/09; no se carga ni se evalúa
todavía. Mini EDA del train: `notebooks/elo_entre_fotos.ipynb` (importa solo partidas
posteriores a la foto 1, filtradas al leer y verificadas).

* **Cobertura entre fotos** ($\Delta N = N_2 - N_1$ de Riot contra partidas
  observadas): mediana 97,5 % en Challenger, 97,2 % en Grandmaster, 96,1 % en Master
  y 50,0 % en Diamond. Ningún jugador con $\Delta N < 0$: no hubo reinicio de ranked.
* **Regla de LP medida** ($\Delta LP \approx a\,\Delta W - b\,\Delta L$, mínimos
  cuadrados por rango de hoy): Challenger 24,7 / 23,9, Grandmaster 23,3 / 22,4,
  Master 20,4 / 19,3, Diamond 16,6 / 16,4 LP por victoria / derrota (R² 0,37–0,61).
  Casi simétrica: la inflación de Master+ viene sobre todo de ganar más de lo que se
  pierde, no de que la victoria valga mucho más.
* **Cuánto explican nuestras partidas el $\Delta LP$:** con cobertura ≥ 95 %, la
  correlación con las victorias netas observadas (0,65) iguala a la de las reales
  (0,66); con cobertura < 50 %, cae a 0,36 contra 0,89. El resto de la varianza con
  cobertura completa es el LP por partida que varía según el MMR: lo que un modelo de
  habilidad podría capturar.
* **Referencias para el modelo al revés** (26.534 jugadores, objetivo $z_1$ desde la
  foto 2): "no cambió" MAE 145 LP, acierto de rango 73,3 %; promedio del rango 160 LP;
  **deshacer con la regla y nuestras partidas: MAE 111 LP, R² del cambio 0,38,
  acierto de rango 79,1 %**; techo con las W/L reales de Riot (no disponible en el
  test): 89 LP, R² 0,50.
* **Rotación de rangos:** del 04/09 al 28/10, el 42,5 % de los Master y el 17,2 % de
  los Grandmaster bajaron a Diamond; 9.169 personas del padrón no están en la foto 2
  (bajaron de Diamond y la fase `sin_rango` no las consulta, porque estaban en el
  padrón).

**Informe de modelos al revés (2026-10-28).** `notebooks/elo_entre_fotos.ipynb` dejó de
ser una descripción del problema: el usuario pidió que fuera un **informe de los
modelos**. Protocolo (decisión del usuario): **todo el período 04/09 → 28/10 es train,
sin separar jugadores**; el test es lo anterior al 04/09 y no se carga. La regla $a, b$,
los hiperparámetros y los modelos aprendidos se ajustan y se miden sobre los 26.534
jugadores evaluables. Una primera versión separaba 80/20 por jugador y el usuario la
descartó. Consecuencia: en los modelos que aprenden (corrección lineal y gradient
boosting), el error reportado es dentro de la muestra y optimista. Con el 80/20, el
gradient boosting daba 72 LP en la parte ajustada contra 80 en la reservada. Los cortes
de Grandmaster y Challenger para pasar LP a rango salen de la foto 2, que es la que se
conoce al arrancar; antes se usaban los de la foto 1, que es el target.

| Modelo (train completo) | MAE | Mediana | R² del cambio | ρ dentro del rango | Acierto de rango | Cobertura del IC 95 % |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| No cambió | 145 | 94 | −0,01 | 0,61 | 73,9 % | 95,0 % |
| Promedio del rango | 160 | 110 | −0,12 | — | 74,3 % | 95,0 % |
| Regla al revés ($a, b$ según el rango en cada paso) | 111 | 73 | 0,38 | 0,73 | 79,2 % | 95,0 % |
| Elo al revés (mejor $\beta = 3200$) | 114 | 73 | 0,34 | 0,72 | 78,6 % | 95,0 % |
| TrueSkill al revés (mejor $\beta = 1600$, $\sigma_{foto} = 200$) | 135 | 86 | 0,11 | 0,67 | 75,3 % | 93,9 % |
| Regla + corrección lineal (gap, obs, ocultas) | 108 | 75 | 0,49 | 0,79 | 74,8 % | 95,0 % |
| **Regla + gradient boosting** | **71** | **44** | **0,70** | **0,88** | **87,4 %** | 95,1 % |
| **Regla + gradient boosting, sin `ocultas`** | **73** | **47** | **0,70** | **0,87** | **86,9 %** | 95,2 % |
| Techo (W/L reales de Riot) | 89 | 49 | 0,50 | 0,83 | 80,2 % | 95,0 % |

* **Por qué fallan Elo y TrueSkill al revés.** La actualización hacia adelante es una
  contracción: un jugador sobrevalorado pierde más de lo esperado y el rating lo
  corrige. Su inversa es una expansión. Deshacer la victoria de un favorito resta poco
  y la de un no favorito resta mucho, así que cada error de estimación se agranda
  partida a partida. Con las probabilidades calculadas sobre las estimaciones vigentes,
  el Elo diverge. Con la fuerza congelada en la foto 2 ya no diverge, pero empeora a
  medida que $\beta$ baja: con $\beta = 100$ da MAE 216, contra 114 con $\beta = 3200$
  (≈ la regla). TrueSkill con $\beta = 400$, o con $\sigma$ creciendo por $\tau^2(1+h)$
  al alejarse de la foto, da MAE de millones de LP. Tiene además una razón de dominio:
  Riot no reparte el LP según la sorpresa del resultado, porque el emparejamiento ya
  equilibra las partidas, sino según la brecha entre el MMR y el LP propio. Se probó
  también modular el LP por partida con esa brecha (media del lobby − LP propio),
  $\Delta = a + \gamma \cdot \text{brecha}$. Con $\gamma = \pm 0{,}005$ empeora la regla
  (MAE 123–132) y con $|\gamma|$ mayor diverge. Se descartó.
* **Por qué gana el gradient boosting.** No toca la recorrida: predice el residuo de la
  regla, $z_1 - \hat z_1^{regla}$, con `HistGradientBoostingRegressor` (pérdida
  absoluta, 400 iteraciones, tasa 0,05, 31 hojas, 50 muestras mínimas por hoja). Las
  variables solo usan lo que también existe en el test: la foto de partida, nuestras
  partidas y el LP de la foto de partida de los otros 9 jugadores de cada partida.
  Importancia por permutación (medida con el 20 % reservado de la primera versión), en aumento del MAE: `lobby_viejo` (LP
  medio de los otros jugadores en sus 10 partidas más viejas) 45 LP, `pred_regla` 41,
  `partidas2` (W + L del split en la foto) 23 y `gap` (lobby − LP propio) 16. Riot
  empareja por MMR, así que el lobby de las partidas más cercanas al target indica
  dónde estaba el jugador entonces. Por eso le gana al techo, que no usa esa
  información.
* **`ocultas`** (partidas según Riot que no tenemos) usa el W/L del extremo target. En
  el test solo se puede calcular hasta el inicio del split, así que la versión sin esa
  variable es la referencia honesta.
* **Pendiente.** El error del gradient boosting es dentro de la muestra; la prueba
  real es el test. Challenger y Grandmaster siguen en 125–130 LP de MAE. El modelo
  predice el LP en la fecha de una foto, no en cada partida: para tenerlo partida a
  partida, hay que anclar la regla al revés en las dos puntas y repartir la diferencia.

**Reestructura del notebook, modelos descartados y GRU (2026-10-28).** Pedido del
usuario: que el notebook sea un mini análisis del problema y, de ahí, la comparación de
modelos, sin los que no sirven.

* **Parte A, mini análisis:** rotación de rangos, cobertura, regla $a, b$ y por qué
  informa el LP de los rivales. $\rho(z_1 - z_2,\ \text{lobby}_{viejo} - z_2) = 0{,}63$
  en las 10 partidas más viejas, contra 0,43 en las 10 más nuevas: el lobby cercano al
  target dice dónde estaba el jugador.
* **Parte B, modelos:** no cambió, regla al revés, Elo al revés, regla + gradient
  boosting (sin `ocultas`), regla + GRU y techo. Las métricas principales son MAE,
  RMSE, R² del cambio y ρ dentro del rango; como secundarias, la mediana del error, ρ
  global y el acierto de rango y de división. Se sacó el intervalo del 95 %.
* **Descartados de la comparación:**
  * promedio del rango: erra más que "no cambió";
  * TrueSkill al revés: diverge, y en su mejor configuración queda en 135 LP;
  * corrección lineal: baja el MAE apenas, de 111 a 108, y empeora el acierto de rango.
* **Código compartido:** `src/features/elo_reverso.py` concentra carga del train,
  jugadores evaluables, regla $a, b$, recorrida al revés (regla y Elo), lobby por partida
  y variables estáticas, con 7 tests en `tests/test_elo_reverso.py`. El generador del
  notebook pasó al repo: `scripts/analysis/build_elo_fotos_nb.py`.
* **GRU (`scripts/analysis/elo_gru.py`).** Predice el mismo residuo de la regla que el
  gradient boosting, pero leyendo la secuencia de partidas de cada jugador de la más
  nueva a la más vieja, en el sentido de la recorrida al revés. Así no depende de
  resúmenes fijos como `lobby_viejo`.
  * **Por paso:** el resultado; el LP medio de aliados y de rivales menos el propio; el
    LP medio de los otros 9; la fracción de los 9 con foto; los días antes de la foto; y
    el lado.
  * **Arquitectura:** el estado final (48 unidades) se concatena con 6 variables
    estáticas (z2, regla, W + L y winrate del split en la foto, partidas y victorias
    observadas) y pasa por una red 64 → 1.
  * **Entrenamiento:** pérdida L1, Adam con tasa 2·10⁻³, lotes de 256 de largo
    parecido y 15 épocas fijas. Sin conjunto de validación (protocolo del usuario), el
    número de épocas no se elige por la pérdida.
  * **Salida:** `data/analysis/elo_gru_pred.csv`, que el notebook lee si existe.
    Necesita PyTorch, agregado a `requirements.txt`.
  * *Riesgo:* con 26.534 secuencias y error dentro de la muestra, una GRU puede
    memorizar; su MAE en el train no la compara de forma justa contra la regla.

**EDA entre fotos (2026-10-28).** Pedido del usuario: `notebooks/eda_entre_fotos.ipynb`
reemplaza a `elo_entre_fotos.ipynb`, conciso, en dos partes. Generador:
`scripts/analysis/build_eda_fotos_nb.py`.

* **Qué cambió entre fotos**, sobre los 31.652 jugadores que están en las dos.
  * *Rangos:* 10,8 % subió, 77,0 % quedó igual y 12,2 % bajó. De Challenger se quedó
    el 59 %; Grandmaster es el rango más inestable (37 % se quedó, 35 % bajó a Master).
  * *Cortes, en LP de Master:* Grandmaster pasó de 645 a 732 (+87) y Challenger de
    1189 a 1260 (+71). Los cupos son los mismos (500 y 200): sube el LP necesario,
    no la cantidad de lugares.
  * *LP total:* +244.628 LP (+7,7 por jugador; mediana 0). Por rango del 04/09, el
    Δ medio es −322 en Challenger, −259 en Grandmaster, −61 en Master y +42 en
    Diamond: los de arriba bajan, porque la cima es angosta.
  * *Sesgo:* el ladder de Riot solo lista Diamond+. 9.129 jugadores del padrón
    (Diamond el 04/09) no están en la foto 2: muy probablemente bajaron a Emerald o
    menos. Las bajadas de Diamond están subestimadas y su +42 está sesgado hacia
    arriba. Para cerrarlo hay que consultarlos por PUUID (como la fase `sin_rango`).
* **Modelos:** solo no cambió (baseline), Elo al revés (β = 3200, el mejor de la
  grilla), gradient boosting y GRU, con MAE, RMSE, R² del cambio y ρ dentro del rango.
  La regla y el techo se usan por dentro pero no se reportan. Dentro de la muestra,
  sobre 26.534 jugadores (MAE / RMSE / R² del cambio / ρ dentro del rango): no cambió
  145,2 / 223,7 / −0,007 / 0,606; Elo al revés 113,7 / 181,5 / 0,337 / 0,719; gradient
  boosting 74,1 / 122,9 / 0,696 / 0,867. La GRU queda sin fila hasta que se entrene.
* **LP partida a partida.** Los modelos predicen el LP de la foto 1, no el de cada
  partida. Para dibujar el camino (`elo_reverso.camino`):
  1. se deshace la regla desde $z_2$ (LP antes de cada partida, $r_k$, con $r_0$ el
     de la más vieja);
  2. la diferencia con lo que predice el modelo, $\hat z_1 - r_0$, se reparte en
     proporción a las partidas deshechas: $c_k = r_k + (\hat z_1 - r_0)(1 - k/n)$.

  El camino arranca en $\hat z_1$ y termina en $z_2$. Entre gradient boosting y GRU
  solo cambia la inclinación de esa corrección: la forma partida a partida es la de
  la regla. El notebook la compara, para 3 jugadores al azar con ≥ 20 partidas
  observadas, con la recta entre el LP real de las dos fotos. Alternativa
  descartada por ahora: una GRU que dé el LP en cada paso, que no tiene target por
  partida.

**EDA entre fotos, segunda versión (2026-10-28).** Pedido del usuario: gráficos con
seaborn y la paleta del EDA del dataset, error por rango, error según partidas hacia
atrás, grupos de los que más se movieron, deriva de LP, NGBoost y suavizador de Kalman.
Resultados, todos dentro de la muestra sobre 26.534 jugadores:

| Modelo | MAE | RMSE | R² del cambio | ρ dentro del rango |
| :--- | ---: | ---: | ---: | ---: |
| No cambió | 145,2 | 223,7 | −0,007 | 0,606 |
| Elo al revés | 113,7 | 181,5 | 0,337 | 0,719 |
| Kalman (no observadas por tasa del rango) | 101,5 | 155,4 | 0,514 | 0,795 |
| Gradient boosting | 74,1 | 122,9 | 0,696 | 0,867 |
| NGBoost | 73,2 | 102,8 | 0,787 | 0,873 |

* **NGBoost contra el gradient boosting.**
  * *Qué ajusta:* el mismo residuo con las mismas variables, pero NGBoost ajusta
    $\mu$ y $\sigma$ de una Normal con el gradiente natural. Configuración: 800 árboles
    de profundidad 5, tasa 0,05 y hojas de ≥ 50; residuo en cientos de LP; ~7 min en CPU.
  * *Resultado:* MAE casi igual (73 contra 74), pero mejor R² (0,79 contra 0,70). El
    boosting de sklearn minimiza el error absoluto, que apunta a la mediana; NGBoost
    estima la media y castiga más los errores grandes.
  * *Incertidumbre:* el intervalo $\mu \pm 1{,}96\sigma$ cubre al 96,4 % de los
    jugadores (96–98,5 % en cada rango) y el error real crece con $\sigma$. Es el único
    modelo con incertidumbre calibrada.
* **Suavizador de Kalman** (`elo_reverso.kalman_al_reves`, RTS).
  * *Estado:* el LP antes de cada partida. Arranca en $z_2$ con varianza 0 y recorre
    hacia atrás.
  * *Partidas observadas:* se deshacen con la regla, más un ruido $q = 25$.
  * *Partidas no observadas:* se reparten entre las observadas. Cada una mueve la
    media en $p\,a - (1 - p)\,b$ y suma $p(1 - p)(a + b)^2 + q$ de varianza, con $p$ el
    winrate observado suavizado, $(W + 10)/(N + 20)$.
  * *Medición:* el lobby de cada partida, $\text{lobby} \approx 183 + 0{,}76\,z$ con
    $\sigma = 218$ LP por partida, calibrado con las 3 partidas más nuevas de cada
    jugador (`calibrar_lobby`).
  * *Versiones:* solo observadas, 106,6 LP; con las no observadas estimadas por la tasa
    del rango, 101,5. La tasa es de no observadas por cada observada, del train: 0,03 en
    Challenger y Grandmaster, 0,13 en Master y 1,09 en Diamond. Con el número real de
    Riot (referencia, no existe en el test), 102,7. Las no observadas ayudan en Master
    y Diamond, donde hay muchas, y empeoran Challenger, donde casi no hay.
  * *Dónde falla:* Challenger queda en 288 LP de MAE, peor que el Elo. Una sola
    calibración lineal del lobby no sirve en la cima, donde el lobby tiene pendiente
    0,40 y no 0,76.
  * *Cobertura:* el IC 95 % cubre solo el 61 %. El modelo lineal-gaussiano subestima
    la varianza, porque los errores tienen colas pesadas.
  * *Lo que aporta:* el LP en cada partida con incertidumbre, que ningún otro modelo da
    sin repartir la corrección a mano.
  * *Parámetros:* $q$ y $R$ salieron de una grilla chica sobre el MAE (q ∈ {25, 100,
    400, 900}); un $q$ más alto mejora la cobertura y empeora el MAE.
* **Grupos de los que más se movieron.** El grupo de los que cambiaron de rango tiene
  6.818 jugadores (25,7 %). Comparte el 56,3 % con el de los 6.818 de mayor $|\Delta
  LP|$, que arranca en 185 LP. El de mayor $|\Delta LP|$ sin cambiar de rango
  arranca en 103 LP.
  * *Sesgo en la dirección del cambio:* se mide como $(\hat z_1 - z_1)\,\mathrm{sign}(z_1
    - z_2)$. Todos los modelos se quedan cortos: el boosting en el 69–79 % de los
    jugadores, con −48 LP de media en todos y −112 en los de más cambio de LP.
  * *Por qué:* predicen cerca de $z_2$ porque la mayoría de los jugadores se mueve poco.
* **Deriva de LP.**
  * *Tamaño:* $a - b$ da +38, +44, +53 y +8 LP cada 100 partidas al 50 % (Challenger,
    Grandmaster, Master y Diamond).
  * *Efecto en el boosting:* sacarla (regla de suma cero, `sin_deriva`) lo lleva de
    74,1 a 74,9 de MAE. El boosting compensa casi toda la deriva con `obs` y `obs_w`.
  * *Efecto en NGBoost:* nulo en el total (73,2 con y sin deriva; RMSE 102,8 contra
    103,0). Solo Grandmaster empeora sin deriva, de 112 a 115.

**Winrate, lobby y decay (2026-10-28).** Tres pruebas pedidas por el usuario, todas
dentro de la muestra (MAE al predecir la foto 1).

* **Ablación del winrate y el lobby.** El boosting y NGBoost ya los tenían como
  variables desde el principio; la ablación mide cuánto pesa cada uno al sacarlo.
  * *Boostings:* con las variables del modelo.
  * *Kalman:* el lobby es su medición y el winrate define el resultado esperado de
    las no observadas.

  | Modelo | Ninguno | Solo winrate | Solo lobby | Los dos |
  | :--- | ---: | ---: | ---: | ---: |
  | Kalman | 111,3 | 116,2 | 101,2 | 101,5 |
  | Gradient boosting | 95,0 | 92,2 | 76,2 | 74,1 |
  | NGBoost | 95,0 | 91,9 | 76,1 | 73,2 |

  El lobby es lo que más pesa (Riot empareja por MMR); el winrate suma 2–3 LP.
* **Ajuste de $a, b$ por partida en la regla:** $a' = a + \delta$ y $b' = b - \delta$,
  con $\delta = k_{wr}(wr - 0{,}5) + k_{lobby}(\text{lobby} - z_0)/100$ (`al_reves`,
  parámetros `k_wr`, `k_lobby`). No mejora: el mejor $k$ es 0 en los dos casos.
  * *Lobby contra el LP del paso:* diverge (MAE de miles), porque un término que
    depende del LP reconstruido amplifica el error hacia atrás, como invertir el Elo.
  * *Con el LP de la foto congelado, crudo o corregido por su sesgo:* empeora en las
    dos direcciones. El lobby de una sola partida es muy ruidoso (σ ≈ 218 LP), y un
    ajuste chico repetido en cientos de partidas se acumula como deriva.
* **Decay de Master+ (regla dada por el usuario: 75 LP por día pasados 14 días sin
  jugar).** Implementado en `decaimiento`, `decay_total` y en el parámetro `fotos` de
  `al_reves`, `camino` y `kalman_al_reves`. Hacia atrás se devuelve el LP perdido en
  cada hueco sin jugar si el jugador está en Master+. `ajustar_regla(decay=...)`
  descuenta el decay antes de ajustar $a, b$.

  | Modelo | Sin decay | Con decay |
  | :--- | ---: | ---: |
  | Regla | 111,3 | 137,0 |
  | Elo al revés | 113,7 | 136,4 |
  | Kalman | 101,5 | 131,6 |

  Para los boostings, que ya podían haber aprendido el decay del residuo, hay tres
  versiones:

  | Versión | Gradient boosting | NGBoost |
  | :--- | ---: | ---: |
  | A: como estaba | 74,1 | 73,2 |
  | B: regla base con decay | 74,4 | 73,6 |
  | C: B + decay total como variable | 74,2 | 72,7 |

  * *Por qué empeora:* la regla tal cual no ajusta a LAS. Entre los 1.384 Master+ de
    las dos fotos sin partidas no observadas, 401 tienen huecos de más de 14 días. A
    ellos el decay calculado les quita una mediana de 525 LP, pero el LP que las
    partidas no explican es de solo −85 en la mediana (−173 de media).
  * *Calibración con los datos:* el LP por día que mejor ajusta es de 13 con 14 días
    de gracia, 18 con 21 y 29 con 28. Puede haber un *banco* de días (cada partida
    suma días de gracia) o un tope; no se pudo confirmar.
  * *Decisión:* los modelos siguen sin decay en la regla. Como variable para NGBoost
    (versión C) mejora apenas, de 73,2 a 72,7, y más en Challenger (105 → 102) y
    Grandmaster (112 → 110). Queda pendiente calibrar el decay con los datos.

**Lobby con el LP predicho y Kalman como variable (2026-10-28).** Pedido del usuario:
calcular el MMR del lobby con el LP *predicho* de cada jugador en esa partida (no con
su foto 2) y usar el Kalman como variable de los boostings. Además, NGBoost (no el
gradient boosting) lleva el decay de Master+ como variable, por decisión del usuario.
Todo dentro de la muestra.

* **Lobby con el LP de cada uno en esa partida.** `al_reves(..., devolver_camino=True)`
  da el LP de cada jugador antes de cada partida, deshaciendo la regla desde su foto 2
  (los que no tienen foto arrancan en el LP medio de los conocidos).
  `lobby_por_fila(..., por_fila=True)` arma el lobby con esos LP. Hay tres versiones:

  | Lobby del boosting | MAE GB |
  | :--- | ---: |
  | LP de la foto 2 (como estaba) | 74,1 |
  | LP predicho por el camino de la regla (sin target) | 73,7 |
  | Camino anclado en la predicción fuera de fold (5 folds) | 75,0 |

  El anclaje fuera de fold no ayuda: suma el ruido de esas predicciones (81 LP de MAE)
  y tiene una fuga de segundo orden, porque el modelo de un fold se entrenó con
  jugadores que aparecen en el lobby de otros.
* **Kalman calibrado por rango.**
  * *Cómo se eligió:* una curva del lobby por tramos según el rango
    (`calibrar_lobby_por_grupo`), y $q$ y la escala del ruido por rango por máxima
    verosimilitud gaussiana de $z_1$. Grilla: $q$ ∈ {25, 100, 400, 1600}, escala ∈
    {1, 2, 4}, decay ∈ {0, 13 LP/día}.
  * *Resultado:* todos los rangos eligieron el borde ($q = 1600$, ×4, sin decay). La
    cobertura del IC 95 % sube de 55 % a 92 %, pero el MAE empeora de 108 a 113.
  * *Lectura:* calibrar la varianza y bajar el MAE tiran en direcciones opuestas.
    El decay no entra ni a 13 LP/día.
* **Kalman como variable** (media − $z_2$ y desvío), con gradient boosting:

  | Lobby | Sin Kalman | Kalman original | Kalman con lobby predicho | Kalman calibrado |
  | :--- | ---: | ---: | ---: | ---: |
  | Foto 2 | 74,1 | 71,7 | 73,8 | 72,2 |
  | Predicho por la regla | 73,7 | **70,8** | 73,2 | 73,5 |
  | Anclado fuera de fold | 75,0 | 72,1 | 71,1 | 71,2 |

  "Kalman original" es el de la sección 2.2: lobby de la foto 2, una sola recta,
  $q = 25$, 101,5 LP de MAE por sí solo. Rinde más como variable que el calibrado,
  aunque solo sea peor: aporta una señal distinta de la del lobby predicho que ya
  tiene el boosting.
* **Configuración elegida:** lobby predicho por la regla + Kalman original como
  variable, en los dos boostings. Es la mejor y la más limpia: ninguna de las dos
  partes usa el target para armarse.
  * *Fuga menor:* las 12 combinaciones se compararon dentro de la muestra, así que
    la elección tiene un poco de optimismo.

  | Modelo | MAE | RMSE | R² del cambio | ρ dentro del rango |
  | :--- | ---: | ---: | ---: | ---: |
  | NGBoost con decay (antes) | 72,0 | 100,6 | 0,796 | 0,877 |
  | NGBoost con decay + lobby predicho + Kalman | **70,3** | **97,5** | **0,809** | **0,882** |

  La mejora se concentra en Challenger (108 → 103), Grandmaster (111 → 105) y Master
  (86 → 83).
* **Pendiente:**
  * validación con folds: hoy todo es dentro de la muestra;
  * un lobby iterado (predecir, recalcular el lobby con esas predicciones, volver a
    predecir), que solo tiene sentido con folds bien anidados.

**Niveles, folds y stacking (2026-10-28).** Pedido del usuario: ordenar la parte 2 de
`eda_entre_fotos.ipynb` en niveles de mejoras, entrenar la GRU con las mismas mejoras que
los boostings, medir fuera de la muestra y probar el stacking de la GRU con NGBoost.

* **Niveles** (`elo_reverso.NIVELES`, `matrices_por_nivel`; la GRU los arma igual en
  `elo_gru.py --nivel`). Se acumulan:
  * *sin mejoras:* se predice el cambio $z_1 - z_2$ con z2, W + L del split y partidas
    y victorias observadas. La GRU lee además, por partida, resultado, días antes de la
    foto y lado.
  * *simple:* residuo de la regla al revés **con deriva** ($a \ne b$), más la regla, el
    winrate (observado y de la temporada) y el lobby con la foto 2 (medio, viejo,
    reciente, gap). La GRU lee por partida aliados, rivales, lobby y cuántos tienen foto.
  * *completo:* lobby con el LP de cada uno en esa partida (camino de la regla), media y
    desvío del Kalman original y decay total. **El gradient boosting ahora también lleva
    el decay** (antes solo NGBoost), por pedido del usuario. La GRU compara aliados y
    rivales con su LP del paso (no el de la foto 2) y lee además los días hasta la partida
    siguiente.
* **Medición fuera de la muestra.** `folds_por_jugador` reparte a los jugadores en 5
  folds por un hash de su id (el mismo en todos los scripts) y a cada uno lo predice un
  modelo que no lo vio. Los pocos parámetros que no se aprenden ($a, b$ de la regla, la
  recta del lobby y la tasa de no observadas del Kalman) se ajustan con todos: 15
  números contra 26.534 jugadores.
* **Bug del *early stopping*.** `HistGradientBoostingRegressor` trae
  `early_stopping="auto"`, que se activa con más de 10.000 filas y separa un 10 % interno
  para cortar. Con scikit-learn 1.9.1 cortaba en 191 de 400 iteraciones: 74,5 de MAE en
  vez de 69,7, con el mismo código. Ahora va `early_stopping=False`. Las comparaciones
  entre gradient boosting y NGBoost de las secciones anteriores pueden depender de la
  versión de scikit-learn con que se corrieron.
* **GRU.**
  * *Épocas:* con el 20 % de los jugadores fuera (`elo_gru.py --validacion`), la
    validación del nivel simple se estanca en ~77 LP entre las épocas 35 y 45; con 15
    épocas (lo de antes) quedaba en ~81. Con estado 96 sobreajusta pasada la época 40
    (66,5 en train contra 78,8 en validación a la 60). Se fijaron 40 épocas y estado 48.
  * *Pérdida* (nivel completo, validación a la época 40):

    | Pérdida | MAE | RMSE |
    | :--- | ---: | ---: |
    | L1 (la de siempre) | 76,6 | 125,7 |
    | Normal ($\mu$, $\sigma$, como NGBoost) | 77,3 | 120,6 |
    | Huber ($\delta$ = 100 LP) | 76,5 | 123,1 |

    La normal baja el RMSE 5 LP y sube el MAE 0,7. Queda L1 por defecto
    (`--perdida`).
* **Resultados** (MAE / RMSE / R² del cambio / ρ dentro del rango):

  | Nivel | Modelo | Dentro de la muestra | 5 folds |
  | :--- | :--- | :--- | :--- |
  | Sin mejoras | Gradient boosting | 94,0 / 156,0 / 0,511 / 0,799 | 103,2 / 164,6 / 0,455 / 0,768 |
  | | NGBoost | 96,5 / 144,8 / 0,578 / 0,805 | 105,9 / 160,8 / 0,480 / 0,768 |
  | | GRU | 97,3 / 155,3 / 0,515 / 0,813 | 97,1 / 155,0 / 0,516 / 0,800 |
  | Simple | Gradient boosting | 70,6 / 118,6 / 0,717 / 0,874 | 80,9 / 130,2 / 0,659 / 0,850 |
  | | NGBoost | 73,2 / 102,8 / 0,787 / 0,873 | 84,0 / 128,1 / 0,670 / 0,847 |
  | | GRU | 72,7 / 116,7 / 0,726 / 0,875 | 76,2 / 124,0 / 0,691 / 0,866 |
  | Completo | Gradient boosting | 69,0 / 117,1 / 0,724 / 0,880 | 79,7 / 128,6 / 0,667 / 0,855 |
  | | NGBoost | 70,4 / 98,2 / 0,806 / 0,882 | 83,0 / 126,4 / 0,679 / 0,851 |
  | | **GRU** | 70,2 / 115,1 / 0,733 / 0,880 | **74,6 / 122,4 / 0,698 / 0,870** |

  * *Dentro de la muestra empatan; afuera gana la GRU.* Los boostings pierden 10–13 LP
    al salir de la muestra y la GRU 4–5. La ventaja de NGBoost en RMSE (98) era casi
    toda memoria: fuera de fold queda en 126 y su IC 95 % cubre al ~89 %, no al 96 %.
    Es la configuración de NGBoost (800 árboles de profundidad 5, sin *early
    stopping*), no el algoritmo: por pedido del usuario no se regularizó todavía.
  * *Qué aporta cada nivel:* el salto grande es de "sin mejoras" a "simple" (−20 a −25
    LP en los tres); el nivel completo suma 1–2 LP fuera de fold.
  * *El protocolo dentro de la muestra elegía mal:* daba NGBoost como el mejor, y fuera
    de fold es el peor de los tres en MAE.
* **Stacking: la GRU con la $\mu$ y $\sigma$ de NGBoost** (`elo_gru.py --con-ngb`).
  * *Sin fuga, con folds anidados* (`elo_boostings.py` escribe
    `data/analysis/elo_ngb_stack.csv`). Para el fold externo $k$, la $\mu$ y $\sigma$ de
    los jugadores de los otros folds salen de NGBoost que no vieron ni a $k$ ni a su
    propio fold (20 ajustes), y las del fold $k$, de NGBoost que no vieron $k$. La GRU con
    todos los jugadores usa la $\mu$ fuera de fold simple. Sin esto, la GRU aprendería a
    confiar en una $\mu$ que en train erra 70 y afuera 83.
  * *Resultado (5 folds):* GRU + NGBoost 74,4 / 121,6 / 0,702 / 0,871, contra 74,6 /
    122,4 / 0,698 / 0,870 de la GRU sola. El promedio simple de las dos da 75,4 / 117,3 /
    0,723 / 0,871: mejor RMSE, peor MAE.
  * *Lectura:* NGBoost no le agrega a la GRU información que no tenga; el promedio solo
    suaviza los errores grandes. **Modelo de referencia: la GRU completa**; el stacking no
    justifica el costo (20 ajustes de NGBoost más).
* **Error por rango** (MAE, 5 folds): la GRU es la mejor en Grandmaster (127), Master
  (88) y Diamond (67); en Challenger gana el gradient boosting (138 contra 142). NGBoost
  es el peor en la cima (185 en Challenger).
* **Cómo se corre.** Los modelos que aprenden ya no se entrenan en el notebook: lee los
  CSV de `scripts/analysis/elo_boostings.py` (gradient boosting y NGBoost, los tres
  niveles, dentro de la muestra, 5 folds y folds anidados; 38 ajustes de NGBoost en 10
  procesos con joblib, ~14 min: NGBoost no corre en GPU porque sus árboles son de
  scikit-learn) y de `scripts/analysis/elo_gru.py` (`--nivel`, `--folds`, `--con-ngb`;
  ~1 min por corrida en una RTX 3070, ~2–4 con `--folds`). El notebook pasó de ~11 a ~2
  minutos.
* **Secciones que salieron del notebook** (pedido del usuario; siguen documentadas
  arriba): la calibración de NGBoost, las versiones del Kalman, la ablación de winrate y
  lobby, el ajuste de $a, b$ por partida y la deriva.

**Objetivo de métricas: experimentos (2026-10-28).** Pedido del usuario, trabajando sin
supervisión: acercar el elo al revés a MAE ~50, RMSE ~70, R² del cambio ~0,85 y ρ ~0,9,
**sin sobreajustar**, probando lo que haga falta. No se tocó el notebook ni el código del
repo: los scripts quedaron fuera del repo (scratchpad de la sesión). Lo que sirve, para
llevarlo al pipeline, está descrito acá.

* **Protocolo anti-sobreajuste.**
  * *Caja cerrada:* 15 % de los jugadores (3.908) por un hash distinto del de los folds,
    $\text{crc32}(\texttt{"caja-"} + id) \bmod 100 < 15$. No se miró hasta el final.
  * *Desarrollo:* el 85 % restante, con los 5 folds de `folds_por_jugador`. $a, b$ de la
    regla se ajustaron solo con desarrollo.
  * *Selección:* dos candidatas fijadas **antes** de abrir la caja y evaluadas una sola vez.
* **De dónde viene el error** (gradient boosting fuera de fold y, después, la mejor
  combinación en desarrollo):
  * *Decay con piso.* De 1.286 Master+ que estuvieron más de 20 días sin jugar antes de la
    foto 2 (sin no observadas), 1.271 terminaron en Diamond en valores fijos: 375 =
    Diamond I 75 LP (633), 275 = Diamond II 75 LP (449), 325 (33). El decay baja de a una
    división y cae en 75 LP de la de abajo. Hacia atrás borra la información: desde el piso
    no se sabe si venía de Master 450 o de Challenger 2700. Explica por qué la regla del
    decay "sobrestimaba 6×" (sección "Winrate, lobby y decay"): un Master 450 solo puede
    perder 75 antes de tocar el piso. Sin piso, la regla de 75 LP/día pasados 14 días
    calza (un Challenger 2760 inactivo 44 días: 2175 calculado contra 2160 real).
  * *MMR distinto del LP.* Riot empareja por MMR, así que **el lobby mide el MMR, no el
    LP**, y el winrate tiende a 50 % sea cual sea el LP. Con MMR ≫ LP se gana +30–40 y se
    pierde −10 (un Master 455 → 1164 con 45 V y 58 D), y el LP **converge** al MMR: varios
    $z_1$ terminan en el mismo $z_2$. Invertir esa convergencia amplifica el error (la misma
    razón por la que diverge el Elo al revés). Las sumas $\sum_t (\text{lobby}_t - LP_t)$
    por victorias y derrotas explican poco (regla 111 → 107, con coeficientes de signo
    inconsistente): el lobby de una partida es un indicador muy ruidoso del MMR.
  * *Partidas no observadas* en Diamond (~1 por cada observada): su resultado es
    desconocido.
  * *Pesos* (mejor combinación, desarrollo):

    | Grupo | Jugadores | MAE | RMSE | % del error² |
    | :--- | ---: | ---: | ---: | ---: |
    | Diamond (resto) | 13.008 | 55,8 | 75,2 | 22,6 |
    | Master (resto) | 5.720 | 73,9 | 108,9 | 20,9 |
    | GM/Challenger (resto) | 378 | 71,9 | 106,7 | 1,3 |
    | Piso del decay + inactivo > 14 días al final | 2.228 | 106,4 | 175,1 | 21,0 |
    | $\lvert z_1 - z_2 \rvert > 400$ (resto) | 1.292 | 191,5 | 293,6 | 34,2 |

    Hasta los casos limpios (0 no observadas y huecos ≤ 10 días, 833 jugadores) tienen
    MAE 88 con el gradient boosting: la ganancia por partida varía mucho entre jugadores.
* **Lo que funcionó** (desarrollo, 5 folds, MAE / RMSE / R² / ρ):
  * *Variables de piso e inactividad:* `piso` ($z_2 \in \{275, 325, 375\}$), días sin
    jugar al final y al principio, hueco máximo entre partidas, rango por la foto 2,
    lobby viejo y reciente. Gradient boosting 80,3 → 76,4; GRU 75,9 → 74,8. Con el decay
    fino (excesos sobre 14 días del hueco final, inicial y de los del medio, cantidad de
    huecos largos, partidas en los 14 días antes del hueco final, pisos 375 y 275 por
    separado) el gradient boosting baja a 75,8 / 121,7 / 0,704 / 0,868 (800
    iteraciones, tasa 0,03); a la GRU no le suma (75,2).
  * *Promedio de 5 semillas de la GRU:* 74,8 → 73,9 / 122,5 / 0,700 / 0,873.
  * *Combinación GRU + gradient boosting* con pesos ajustados sin el fold evaluado: 0,65 y
    0,35 → 73,0 / 119,8 / 0,713 / 0,876. El promedio simple con las versiones de pérdida
    cuadrática da 73,6 / 117,3 / 0,725 / 0,876 (mejor RMSE y R²).
* **Lo que no funcionó:**
  * *Bradley-Terry* sobre los resultados (fuerza por jugador, en 1 o 3 épocas): log-loss
    0,692 en partidas guardadas contra 0,693 de una moneda. El matchmaking deja las
    partidas 50/50: los resultados no dicen quién es más fuerte.
  * *Variables por partida extra en la GRU* (gap, lobby con la foto 2, LP del paso,
    racha) y *rendimiento por partida* (KDA, participación en kills, % de daño y de oro,
    muertes, duración): 76,1–76,6, peor que sin ellas. Como agregados por jugador en el
    gradient boosting, tampoco suman.
  * *Arquitecturas:* LSTM 77,7, GRU bidireccional 77,6, GRU con atención 75,9, DeepSets
    77,3, Transformer de 2 capas 82,1, contra 76,5 de la GRU con las mismas variables.
  * *Más capacidad o regularización en la GRU:* estado 96 con dropout 0,2 79,0; 2 capas
    79,3; 60 épocas con dropout 0,1 77,4; tasa 10⁻³ 75,2; estado 32 75,2; 30 épocas
    75,7. Las 40 épocas y el estado 48 siguen siendo lo mejor.
  * *Base del Kalman en vez de la regla:* 76,8 contra 76,5.
  * *Regularizar el gradient boosting* (hojas de 100–200, 15 nodos, L2, *early stopping*
    interno, submuestreo de variables): 76,3–77,3, nada mejor que sin regularizar.
  * *Pérdida cuadrática:* mejora el RMSE (GRU 120,0, gradient boosting 120,1) y empeora
    el MAE (77,2 y 78,9). Sirve solo dentro de un promedio.
* **Descartado por fuga o por protocolo:**
  * `player_tier`/`player_rank` de `full_dataset.csv` salen de la tabla `players`: son la
    foto 1.
  * `player_identities` del 05–07/09 (28.000 filas): son casi la foto 1.
  * `player_identities` del 19/10 (7.355 jugadores, 9 días antes de la foto 2): no hay
    una medición así en el test. *Corrección (2026-10-29):* tampoco es una medición;
    son valores por defecto (ver "Interpolación entre fotos: validación").
  * El lobby de las partidas justo **antes** del 04/09 mediría el LP cerca de la foto 1,
    pero esas partidas son del test, que no se carga.
* **Resultado en la caja cerrada** (3.908 jugadores, modelos entrenados con todo el
  desarrollo):

  | Modelo | MAE | RMSE | R² del cambio | ρ dentro del rango |
  | :--- | ---: | ---: | ---: | ---: |
  | Regla al revés | 111,8 | 174,6 | 0,365 | 0,733 |
  | Gradient boosting L1 (decay fino) | 74,1 | 116,2 | 0,719 | 0,871 |
  | GRU + piso e inactividad, 5 semillas | 71,9 | 113,7 | 0,731 | 0,876 |
  | **Principal: 0,65 GRU + 0,35 gradient boosting** | **71,0** | 112,0 | 0,738 | **0,878** |
  | **Alternativa: promedio de 5** (2 GRU L1, GRU L2, GB L1, GB L2) | 71,5 | **109,3** | **0,751** | **0,878** |
  | *Objetivo del usuario* | *~50* | *~70* | *~0,85* | *~0,9* |

  La caja da algo mejor que el desarrollo (73,0), porque entrena con el 100 % y no con el
  80 % del desarrollo, y la regla rinde igual en los dos (111,8 contra 111,2): la
  selección no metió optimismo. Por rango (principal): Diamond 62,9, Master 87,7,
  GM/Challenger 101,9.
* **Por qué no se llega al objetivo con estos datos.** El MAE ~50 pide bajar a ese nivel
  incluso a Diamond, que ya está en 56–63 por sus partidas no observadas. El RMSE y el R²
  los domina el 5 % que se mueve más de 400 LP (34 % del error²): escaladores y caedores
  que convergen a su MMR y jugadores en el piso del decay, donde el LP de partida no
  queda escrito ni en $z_2$ ni en el lobby. Para R² 0,85 haría falta un RMSE ~87.
  Información que sí acercaría: una foto intermedia o una foto 3.
* **Pendiente:** llevar al pipeline (`elo_reverso`, `elo_boostings.py`, `elo_gru.py`) las
  variables de piso e inactividad, el promedio de semillas y la combinación.

**Interpolación entre fotos: validación (2026-10-29).** Pregunta del usuario: para el LP
en una partida entre las dos fotos, ¿rinde más una pasada desde una sola foto o combinar
las dos, pesando más la foto más cercana según cuánto se puede confiar en cada lado?

* **Cómo se combina.** Dos estimaciones del LP en la partida $t$: $\hat z_t^{\rightarrow}$
  desde la foto 1 y $\hat z_t^{\leftarrow}$ desde la foto 2, cada una con varianza
  $\sigma^2(n) = \alpha + \beta n$, con $n$ las partidas recorridas desde su foto. La
  combinación de mínima varianza es
  $$\hat z_t = \frac{\hat z_t^{\rightarrow}/\sigma_{\rightarrow}^2 + \hat z_t^{\leftarrow}/\sigma_{\leftarrow}^2}{1/\sigma_{\rightarrow}^2 + 1/\sigma_{\leftarrow}^2}.$$
  Con $\alpha = 0$ y el mismo $\beta$ en los dos lados es el camino de la regla anclado
  (`camino`): la diferencia de cierre se reparte en proporción a las partidas, como un
  puente browniano. Con $\alpha > 0$, o con $\alpha, \beta$ distintos por lado, el peso
  cambia. La versión aprendida estima $\sigma$ de cada lado con un gradient boosting sobre
  el error absoluto de cada pasada.
* **Los dos lados no son igual de confiables.** El gradient boosting del nivel completo
  en las dos direcciones (la de adelante con los datos espejados: tiempo negado, W/L
  invertidos, $a \leftrightarrow b$, ancla en la foto 1), 5 folds, ventana entera:

  | Dirección | MAE | $\sigma^2(n) \approx \alpha + \beta n$ |
  | :--- | ---: | :--- |
  | Hacia atrás (foto 2 → foto 1) | 79,6 | 13.242 + 78,6·n |
  | Hacia adelante (foto 1 → foto 2) | 57,9 | 5.551 + 45,9·n |

  Hacia adelante el decay es determinístico (se sabe dónde cae el piso); hacia atrás, un
  jugador en el piso pudo venir de muchos LP. En los dos lados $\alpha$ domina: el error
  casi no crece con las partidas recorridas, así que para validar alcanza un punto
  intermedio por jugador en muchos jugadores.
* **No hay medición intermedia en LAS.** Las `player_identities` del 19/10 tienen todas
  (5.619 con tier) LP 0, rango "I" y 0 W/L: son los valores por defecto de las filas que
  creó la reconstrucción del padrón de ese día (§Q). Solo el tier es real. `ladder_snapshots`
  tiene únicamente la foto 2 y `players`, la foto 1.
* **Fuente elegida: lolpros.gg.** Se buscaron datasets públicos con LP por partida
  (Kaggle, Hugging Face, GitHub) y no hay: la API de Riot no da historial de LP, y los
  datasets traen a lo sumo el rango del día en que se bajaron. Mobalytics registra LP por
  partida, pero sin API pública, solo para jugadores que sigue y con un certificado que
  no valida. lolpros.gg guarda ~1 foto por día de cada cuenta que sigue (tier, LP, W, L) y
  la expone en `/lol/riot-accounts/{uuid}/rankings`. Son pros y semipros, casi todo
  Challenger/GM de EUW: mide el *método* de interpolación, no el modelo de LAS. Como
  entrenamiento sirve para la dinámica de LP por partida ($a, b$ por jugador), no para el
  modelo de partidas (no trae partidas ni lobby).
  * *Descarga* (`scripts/collection/lolpros_historial.py`): un pedido cada 4 s con espera
    creciente ante 429; la API permite ~4 pedidos cada 10–15 min, así que 300 cuentas
    llevan 10–20 h. Se respeta el límite (sin rotar IP ni simular un navegador). Python
    necesita `truststore` en la notebook: certifi no reconoce la cadena que ve Windows.
  * *Validación* (`scripts/analysis/elo_val_lolpros.py`): ventanas de 14, 30 y 54 días
    por cuenta, puntas = fotos de los extremos, medición = las fotos del medio; los
    estimadores solo ven las puntas y el W/L acumulado de cada día. Compara sin cambio,
    una pasada de la regla desde cada punta, la recta por tiempo y por partidas, el
    camino anclado y el suavizado doble (lineal y aprendido); $a, b$ y la confiabilidad
    se ajustan con 5 folds por cuenta.
* **Pendiente:** completar la descarga y correr la validación.

**Gráfico de mejoras con R² y ρ (2026-10-30).** MAE y RMSE no alcanzan para ver la
progresión, porque el R² del cambio y la ρ dentro del rango responden a otras preguntas:
cuánta varianza del cambio se explica y qué tan bien se ordena a los jugadores de un rango.
Se agregó una fila al gráfico de `eda_entre_fotos`, sin compartir eje entre las dos, porque
sus escalas son distintas (R² 0,45–0,81, ρ 0,77–0,88). Con un eje común, la ρ quedaría casi
plana. La imagen guardada se redibujó con las tablas del notebook, redondeadas a 3 decimales,
porque las predicciones no están en esta máquina; se descartó correr `--ejecutar` por ese
mismo motivo. La próxima ejecución completa la recalcula con los valores exactos.

**Seguimiento de LP propio (2026-10-30) — reemplaza a lolpros como fuente principal.**
Al retomar la descarga de lolpros (pedido del usuario), la API devolvió 429 a los ~7
historiales con `Retry-After: 298`, y la espera se renovaba con cada reintento: unos 10
pedidos cada 5 min, 8 h o más para 1.000 cuentas. Se descartaron:

* *El HTML de lolpros.gg:* la página del jugador trae el cierre y el pico de cada season,
  no el historial diario. El gráfico lo pide el navegador al mismo endpoint limitado.
* *OP.GG, Mobalytics, League of Graphs:* tienen LAS (lolpros no), pero sus términos
  prohíben el acceso automatizado, el `robots.txt` de League of Graphs solo habilita bots
  de publicidad, y las páginas se arman con APIs internas detrás de anti-bot. Además, sus
  datos salen de la misma Riot API.

**Diseño.** La Riot API no guarda historial de LP, pero `league-v4` devuelve el estado
actual de todo Master+ de una región en 3 pedidos (`challengerleagues`,
`grandmasterleagues`, `masterleagues`). Medirlo seguido arma el historial que falta, con
LP real en el medio de cualquier ventana (`scripts/collection/seguimiento_lp.py`, en
`data/external/lp_seguimiento/`):

* **Regiones: LA2, EUW1 y NA1.** LAS valida en la población del modelo (ritmo de juego,
  LP por partida, inflación de Master+). EUW y NA aportan volumen: ventanas largas en
  pocos días. Cada región tiene su cuota en Riot, así que EUW y NA no le quitan cuota al
  crawler. Si las curvas de error difieren entre regiones, manda LAS.
* **Ladder completo cada 60 min (`apex.csv`).** Solo se guardan los cambios de estado
  (tier, división, LP, W, L) y una fila `FUERA` al salir de Master+. Guardar todo serían
  ~20.000 filas por hora y la mayoría repetidas. `pasadas.csv` registra cada pedido: el
  estado entre dos filas es el de la anterior *si* hubo pasadas en el medio.
* **Alto volumen cada 10 min (`alto_volumen.csv`).** Las 50 cuentas por región que más
  juegan se consultan una por una (`entries/by-puuid`). Una partida dura ~30 min más la
  cola, así que con fotos cada 10 min casi siempre $\Delta W + \Delta L \le 1$ entre fotos y
  el cambio de LP es el de una partida: se mide la dinámica $a, b$ por partida
  directamente. Se las sigue aunque bajen de Master. La elección (`seguidos.csv`) se
  rehace cada 6 h: con 24 h de historia, por partidas por día en los últimos 3 días,
  $\big(n(t) - n(t - 3\,\text{d})\big)/3$ con $n = W + L$ y el estado arrastrado; antes,
  por $W + L$ de la temporada (proxy grueso: premia a quien jugó mucho y dejó).
* **Una sola key, la de P0.** Los PUUID vienen cifrados por proyecto: rotar P0 y P1
  partiría cada cuenta en dos identidades.
* **Costo:** 9 pedidos por hora de ladder y 150 cada 10 min de seguidos (50 por
  región). En LA2 son ~10 pedidos cada 120 s compartidos con el crawler, contra 95 de
  techo por key.

**Cuánto juntar.** El error de la regla casi no crece con las partidas recorridas
($\alpha$ domina en $\sigma^2 \approx \alpha + \beta n$, tabla de arriba), así que
importan más los jugadores con un punto intermedio que las ventanas largas. Con 3–4 días
alcanza para una primera comparación de métodos (ventanas de hasta 30–50 partidas). Con
~2 semanas hay ventanas de 100+ partidas en buena parte de los activos. Solo la
inflación de LP de Master+ depende del calendario: con 2 semanas se estima su tasa y se
extrapola a las 7,5 semanas entre fotos. Diamante queda afuera (paginarlo son decenas de
pedidos por pasada); se suma si la medición en Master+ deja dudas.

**Registro de LP por partida (2026-10-28) — no generado, queda para el modelo final.**
Objetivo del usuario: saber, más o menos, qué LP tenía cada jugador al jugar cada partida.
`scripts/analysis/build_lp_partida.py` escribiría `data/lp_por_partida.csv`, pero el
CSV se generó antes de tiempo y **se borró por pedido del usuario**. El registro se
arma con el modelo final, después de evaluar en el test (antes del 04/09), que todavía
no se tocó. Lo que sigue describe el diseño del script y la corrida que se borró.

* **Partidas no observadas, en proporción (pedido del usuario).** Al ir hacia atrás,
  por cada partida observada que se deshace se deshacen también las no observadas que
  le tocan: con proporción 1:6, 1 + 6 partidas por paso. Cada no observada mueve el LP
  en lo esperado según el winrate total del jugador, $wr\,a - (1 - wr)\,b$
  (`elo_reverso.cambio_oculto`; parámetros `no_observadas` y `wr` de `al_reves`,
  `camino` y `adelante`).
  * *Prueba (MAE al predecir la foto 1):*

    | No observadas | Regla | Gradient boosting |
    | :--- | ---: | ---: |
    | No se descartan | 111,3 | 74,1 |
    | Tasa media del rango | 111,8 | 74,5 |
    | Proporción real del jugador | 107,2 | 71,4 |

  * *Por qué la tasa del rango no ayuda:* reparte un promedio que no se parece al
    jugador.
  * *Cuándo se puede usar la proporción real:* necesita las partidas de Riot entre
    fotos. Es legítima cuando las dos fotos se conocen (el puente), no cuando la foto
    1 es lo que se predice.
* **Winrate total como variable.** El winrate de la temporada en la foto 2
  ($W_2/N_2$, `wr_temporada`) ya estaba entre las variables del boosting, de NGBoost y
  de la GRU. Ahora también define el resultado esperado de las no observadas.
* **Métodos, según las fotos del jugador** (1.346.950 filas):

  | Método | Filas | % | Cómo |
  | :--- | ---: | ---: | :--- |
  | `puente` | 1.105.805 | 82,1 | Regla al revés desde $z_2$ con la proporción real de no observadas; la diferencia con $z_1$ real se reparte entre las partidas (`camino`). Arranca exactamente en $z_1$ y termina en $z_2$ |
  | `desde_foto2` | 211.364 | 15,7 | NGBoost (entrenado con los del puente) predice $z_1$ y el camino se ancla ahí; no observadas con la tasa del rango |
  | `desde_foto1` | 29.226 | 2,2 | Regla hacia adelante desde $z_1$ (`adelante`), tasa del rango. Son sobre todo los Diamond que no están en la foto 2 |
  | `lobby` | 555 | 0,0 | Sin foto, o UNRANKED en la foto 2: LP medio de los otros 9 de la partida |

* **Controles:** sin filas sin LP y sin duplicados. El puente arranca exactamente en
  la foto 1. El salto entre partidas seguidas de un jugador tiene mediana de 20 LP y
  percentil 99 de 36.
* **Límites:**
  * *Error medido:* solo en las puntas; en el medio del camino hace falta una foto 3.
  * *Uso descriptivo:* `puente` y `desde_foto2` miran partidas posteriores (incluida
    la propia). No sirven como variable de un modelo que prediga el resultado.
  * *Debajo de Diamond:* `desde_foto2` extrapola, porque la regla y NGBoost se
    ajustaron con Diamond+. El 16,7 % de esas filas queda por debajo de Diamond.
  * *Antes del 04/09 (test):* no se generó.
  * *GRU:* entrenada después de esta sección (ver "Niveles, folds y stacking").

Archivos: `src/features/rating.py` (11 tests en `tests/test_rating.py`),
`scripts/analysis/build_rating.py` (grilla y salida, ~12 min),
`scripts/analysis/build_rating_nb.py` → `notebooks/rating_jugadores.ipynb` (la foto,
la cobertura, el sesgo de winrate y el envejecimiento de la foto; las secciones de
evaluación se generan con `--evaluacion` cuando exista la segunda foto).

#### S.22. LP por partida medido en vivo, Master+ de LAS, BR, NA y EUW (2026-10-30)

**Problema.** Riot no da historial de LP: entre dos fotos solo hay una resta, de la que
salen $a, b$ promedio. Tampoco hay una medición intermedia para validar la interpolación.

**Método.** Se lee el ladder de Master+ cada 2 minutos y se compara cada jugador con su
lectura anterior. Si $\Delta W + \Delta L = 1$, jugó exactamente una partida y

$$\Delta LP = z_{\text{después}} - z_{\text{antes}}$$

es lo que esa partida movió. La partida se busca en Match-V5 y se enlaza si terminó dentro
de la ventana y el resultado coincide con el W/L; una sola respuesta enlaza a todos los
jugadores medidos del lobby. Base aparte (`data/lp_vivo/lp_vivo.db`), código en
`src/lp_vivo.py` y `scripts/collection/lp_vivo.py`.

**Alternativas descartadas.**

* *Spectator-V5 y League-V4 por jugador (idea original):* no hay un endpoint con todas las
  partidas en curso, así que hay que preguntar por PUUID para encontrarlas, y después hacen
  falta ~20 pedidos por partida para leer el LP. El ladder trae a todos con 3 pedidos.
* *Sumar los datos a `lol_las.db`:* pedido explícito de mantenerlos aparte. Además mezcla
  servidores en una base cuya clave de población es LAS.
* *Usar las dos keys:* Riot cifra los PUUID por proyecto y Match-V5 devuelve los del proyecto
  de la key, así que el cruce con el ladder se rompería. Se usa solo P1.

**Lo que mostró la prueba de 8 minutos (descartada, no está en la base).**

* *Réplicas atrasadas:* el ladder alterna entre respuestas nuevas y viejas. 189 jugadores
  pasaron de un estado a uno con W + L menor y volvieron, y cada vuelta contaba como una partida
  falsa. Se ignora una lectura con W + L menor en hasta 10 (`atrasada`). Una baja mayor se
  registra como `reinicio`, porque sería un cambio de temporada.
* *Atraso:* entre el fin de la partida y la lectura que muestra el cambio pasaron de 2 a
  ~15 minutos, por eso la ventana de enlace mira 25 minutos para atrás.
* *Tope de Master:* la liga viene con 10.000 jugadores exactos en BR, NA y EUW (LP mínimo
  visible en EUW: 451); LAS, con ~8.570, entra entera. **En EUW y NA no se ven los Master de
  LP bajo**, así que esa población no es comparable con LAS sin condicionar por LP.
* *Magnitud:* ±~20 LP por partida en LAS contra ±~30 en los otros tres, pero con muy pocos
  casos sin rebote en LAS (11). Es un primer indicio, no una medición.

**Validación (decisión del usuario, 2026-10-30).** Se valida con los 4 servidores y no solo
con LAS: el modelo recibe el LP y el lobby, y la diferencia de ganancia entre servidores no
se considera un impedimento. El lobby del CSV es el $z$ de antes medio de los otros jugadores
*medidos*, sin el propio: no usa nada posterior a la partida, y queda sesgado hacia Master+
porque los Diamond no se miden.

**Primera medición del ΔLP por partida (2026-10-30, exploratoria, ~1,5 h de datos).** 4.975
eventos, 765 partidas. El principal no predice por partida, así que se evaluó el paso que le
implica su curva (`camino`): $\pm a, b - (\hat z_1 - \hat z_1^{regla})/n$, con la corrección
del período 04/09–28/10 aplicada a partidas posteriores. Es una aproximación, no el modelo.

| LAS (618 eventos con curva) | MAE | RMSE | R² del ΔLP |
| :--- | ---: | ---: | ---: |
| Regla de las fotos ($a, b$ de LAS) | 3,02 | 5,78 | 0,924 |
| Curva del principal | 3,83 | 6,43 | 0,906 |
| Regla medida por servidor (5 folds) | 2,62 | 5,99 | 0,918 |
| GB por partida con servidor (5 folds) | 2,45 | 5,73 | 0,925 |

$a, b$ de las fotos (Master 20,4/19,3) coinciden con lo medido en vivo (21,6/18,6). Fuera de LAS
no sirven (MAE 11–12), y el GB sin variable de servidor erra 7,9 en LAS: el servidor pesa. El R²
dentro de victoria o derrota es ≈ 0: lobby y winrate casi no agregan al resultado y al servidor.
Queda sin explicar una cola de ΔLP de hasta ±70 con $\Delta W + \Delta L = 1$. Falta repetirlo
con más horarios.

*Corrección (2026-11-02, §9.S.40):* con 2,4 días de datos esta primera medición se corrige en
tres puntos. El R² por partida engaña: lo domina el signo del resultado, y la regla, con R² 0,92,
acumula 111 LP de error entre fotos. "Lobby y winrate casi no agregan" valía por partida y con un
lobby de pocos jugadores medidos; por jugador, el gap con el lobby explica la ganancia que se
acumula. La cola de ±70 tiene explicación (Aegis of Valor, devoluciones de LP y pisos), y lo que
parecía un efecto del servidor es un régimen de LP distinto desde el reinicio del 29/04.

**Límites.** No recupera el pasado. Quien entra por primera vez desde Diamond no tiene LP de
antes (`entro`), y las partidas jugadas durante un corte quedan como `varias`, sin enlazar.

#### S.23. Quién gana, antes de la partida: modelos simples sobre el train (2026-11-02/03)

Pedido del usuario: ver cómo está el target con modelos simples, **sin tocar el test**,
partiendo solo `data/splits/train`. Target: gana el azul. Solo información previa a la
partida. Scripts: `historial/ganador_primeros/ganador_features.py` + `ganador_modelos.py` (con LP) y
`ganador_features_grande.py` + `ganador_red.py` (sin LP, más partidas de ajuste).

**Protocolo.** Bloques temporales dentro del train, por partida completa (5 líneas por lado):

| Bloque | Período | Uso |
| :--- | :--- | :--- |
| H | antes de la foto 1 (04/09) | huella, y matchups/sinergias en la versión con LP |
| F | 04/09 → mediana del período (≈ 16/09) | ajuste (26.934 partidas) |
| V | ≈ 16/09 → 27/09 | validación (26.934 partidas; azul gana 48,8 %) |

Todo lo que describe a un jugador sale de partidas anteriores a la que se predice. La
partición es temporal porque el test también lo es; una aleatoria deja pasar el futuro (ver
la fuga del rango). Métricas: AUC (principal: no depende del umbral) y acierto con umbral
0,5; IC por bootstrap de V (±0,007 de AUC, ±0,6 puntos de acierto) y diferencias contra la
base con bootstrap pareado.

**Variables** (cada una como azul − rojo, por línea y total de equipo):

* *Campeones:* one-hot con signo (+1 azul, −1 rojo).
* *Huella:* media por jugador de los $z$ de `fingerprint.residualize` (desvío respecto del
  mismo campeón, línea y resultado), con ≥ 10 partidas en H.
* *LP:* foto 1 más la regla hacia adelante, $z_{k+1} = z_k + a_g$ si gana y $z_k - b_g$ si
  pierde ($a, b$ de `elo_reverso.ajustar_regla`). Solo existe desde el 04/09. La foto 1 es la
  más cercana a todo F y V. La foto 2 no se usa: llegar a ella exige recorrer el test, y la
  pasada hacia atrás deshace el resultado de la misma partida que se predice.
* *Experiencia con el campeón:* con $n_c$ partidas previas con ese campeón, $n$ en total y
  $n_\ell$ en esa línea: $\log(1+n_c)$, estreno ($n_c = 0$), $n_c/(n+1)$, $(n_\ell+1)/(n+2)$,
  fuera de línea ($n \ge 10$ y $n_\ell/n < 0{,}2$) y winrate con el campeón menos el propio,
  suavizado: $\frac{w_c + 5\,\hat p}{n_c + 5} - \hat p$, con $\hat p = \frac{w+5}{n+10}$.
* *Cobertura:* $\log(1+n)$, para separar experiencia de "cuántas partidas suyas tenemos".
* *Dificultad de Riot:* `info.difficulty` de Data Dragon (1–10) y su cruce con estreno y con
  $\log(1+n_c)$.
* *Matchups* (campeón contra rival, por línea) y *sinergias* (pares de compañeros: bot,
  jungla + mid, jungla + top): winrate del par suavizado hacia lo esperado por cada campeón
  solo, $r = \frac{w + K e}{n + K} - e$ con $K = 20$. En la versión con LP se estiman con H;
  sin LP, acumulados.

**Resultados en V** (logística, $C = 0{,}05$, salvo que se diga):

| Modelo | Acierto | AUC |
| :--- | :---: | :---: |
| Tasa base | 51,2 % | 0,500 |
| Campeones | 52,9 % | 0,542 |
| LP (foto 1 o foto 1 + regla: igual) | 52,5 % | 0,529 |
| Huella | 54,0 % | 0,556 |
| Experiencia con el campeón | 55,7 % | 0,579 |
| Campeones + LP + huella | 55,2 % | 0,575 |
| **Base:** campeones + LP + huella + experiencia | 58,0 % | 0,610 |
| **Todo** (+ cobertura, dificultad, matchups, sinergias) | **58,1 %** | **0,612** |

Sobre "todo", espejado (ver abajo): red de 64 unidades 57,6 % / 0,607; LightGBM 0,601;
XGBoost 0,601; CatBoost 0,600; HistGradientBoosting 0,599; NGBoost 0,597 (40 min). Contra
la base, todos los boostings pierden de 0,009 a 0,013 con el IC pareado entero por debajo de
cero. El promedio logística + CatBoost da 0,612 y baja el log-loss de 0,6747 a 0,6730.

**Hallazgos.**

1. **Fuga del rango.** El tier de la foto da AUC 0,600 en partidas anteriores a la foto y
   0,490 después: es una foto tomada después de esas partidas, así que contiene sus
   resultados. Usado de forma honesta (LP desde la foto en adelante) cae a 0,529.
2. **El LP aporta poco.** El emparejamiento iguala el LP de los equipos.
3. **La experiencia con el campeón es real, no cobertura** (chequeo pedido): la cobertura
   sola da 0,523, y experiencia + cobertura 0,586 contra 0,579. Por rango del lobby da 0,575
   en Diamond y 0,580 en Master+. Dentro de cada jugador (≥ 100 partidas, restando su
   winrate), estrenar campeón cuesta de 3,4 a 4,1 puntos y con 20+ partidas suma 1,6, con
   una curva monótona en los dos rangos.
4. **Las interacciones no aparecen.** Los árboles y la red quedan por debajo de la
   logística, y matchups/sinergias solos dan 0,51.
5. **Espejado.** Cada partida de ajuste se agrega con los lados invertidos ($X \to -X$,
   $y \to 1 - y$) y una columna de lado. En la logística no cambia nada (con variables
   azul − rojo es casi el mismo modelo). En los árboles suma +0,003 y en la red +0,003 (media
   de 3 semillas: 0,604 → 0,607).
6. **Más partidas no alcanzan; la huella reciente sí** (versión sin LP, misma V):

   | Huella | Ajuste | Logística | Red espejada |
   | :--- | ---: | :---: | :---: |
   | hasta el 04/09 | 26.934 | **0,610** | 0,606 |
   | hasta el 09/08 | 26.934 | 0,600 | 0,594 |
   | hasta el 09/08 | 85.757 | 0,603 | 0,601 |
   | sin huella | 26.934 | 0,596 | 0,592 |
   | sin huella | 160.885 | 0,597 | 0,594 |

   Sacar el LP cuesta 0,002. Las partidas extra le sirven más a la red que a la logística,
   pero no la alcanzan. Lo que cae al agrandar el ajuste es la huella, que hay que armar con
   datos más viejos.

**Alternativas descartadas.**

* *Rating TrueSkill (`rating_partida.csv`) para ajustar con todo el train:* desde el 04/09
  rinde como el LP (0,528). Con el ajuste sobre todo el train cae a 0,508; la hipótesis, sin
  verificar, es que antes de la foto el rating no tiene ancla y su escala cambia.
* *Embeddings de campeón-línea:* dan lo mismo que el one-hot (0,539 contra 0,539).
* *Firma del pool:* 0,512.
* *Curva de aprendizaje por campeón:* el usuario prefirió la dificultad de Riot, que no
  aportó (−0,0004).

**Red con codificador compartido por jugador (2026-11-12, `ganador_red_jugador.py`, GPU).**
La red de arriba recibe las variables ya restadas (azul − rojo por línea), así que no puede
combinar dos variables *del mismo jugador* (por ejemplo, campeón difícil × estreno, o huella ×
línea). La alternativa es pasar a cada jugador $i$ por la misma red $\varphi$, con entrada
$x_i$ = [embedding del campeón (16), embedding de la línea (4), experiencia, cobertura,
dificultad, huella + indicador de "tiene huella"], dos capas de 64 y dropout 0,3. Dos salidas:

* *escalar:* $\varphi(x_i) \in \mathbb{R}$ y
  $z = \sum_{i \in A}\varphi(x_i) - \sum_{i \in R}\varphi(x_i) + b + w^\top m$;
* *equipo:* $\varphi(x_i) \in \mathbb{R}^{64}$, $h_T = \sum_{i \in T}\varphi(x_i)$ y
  $z = g(h_A, h_R) - g(h_R, h_A) + b + w^\top m$, que deja combinar jugadores del mismo equipo
  y del rival.

$m$ son matchups y sinergias (ya azul − rojo) y $b$ la ventaja de lado. Las dos cumplen
$z(\text{lados invertidos}) = -z + 2b$, así que el espejado está en la arquitectura y no hace
falta duplicar partidas. Parada temprana con el 15 % más nuevo del ajuste, como antes. AUC en
V; entre corchetes, la diferencia pareada contra la logística (IC95, bootstrap de V):

| Config | Logística | Red agregada espejada | Escalar (3 semillas) | Equipo (3 semillas) |
| :--- | :---: | :---: | :---: | :---: |
| A: huella hasta 04/09, ajuste 26.934 | **0,610** | 0,606 | 0,607 [−0,006; +0,001] | 0,606 [−0,008; −0,001] |
| B: huella hasta 09/08, ajuste 85.757 | **0,603** | 0,601 | 0,599 [−0,007; −0,001] | 0,601 [−0,005; +0,000] |
| C: sin huella, ajuste 160.885 | 0,597 | 0,594 | 0,597 [−0,002; +0,003] | 0,596 [−0,003; +0,002] |

Cada semilla sola queda de 0,007 a 0,010 por debajo de la logística; el promedio de 3
recupera ~0,005, o sea que buena parte de lo que falta es varianza de la red y no estructura
que la logística no vea. Con todo el train (C) la red por jugador alcanza a la logística, y la
salida "equipo" no le gana a la "escalar": **no hay interacciones dentro del jugador ni dentro
del equipo que se puedan aprender con estos datos.** Queda descartada como modelo principal;
se confirma el hallazgo 4.

**¿Sirve un MoE? Chequeo por grupos (2026-11-12, `ganador_por_grupos.py`, config A).** Una
mezcla de expertos con compuerta, $p = \sigma\big(\sum_k g_k(c)\,z_k\big)$ con
$g = \mathrm{softmax}(\text{red}(c))$, solo le gana al mejor experto si el orden entre expertos
cambia según el caso $c$. Antes de construirla se midió el AUC de cada experto por grupo de
partida. Los descriptores son simétricos entre lados, para que la compuerta no rompa la
antisimetría: tier del lobby, cobertura mínima del lobby (partidas previas del jugador menos
visto, en terciles), estrenos en el lobby y jugadores sin huella. Entre corchetes, diferencia
pareada contra la logística (IC95, bootstrap dentro del grupo):

| Grupo | n | Logística | LightGBM | CatBoost | Red escalar | Promedio |
| :--- | ---: | :---: | :---: | :---: | :---: | :---: |
| todo V | 26.934 | **0,610** | 0,600 [−0,014; −0,006] | 0,597 | 0,607 [−0,006; +0,001] | 0,610 |
| Diamond | 9.853 | **0,594** | 0,584 | 0,582 | 0,590 | 0,593 |
| Master+ | 17.081 | 0,618 | 0,608 | 0,605 | 0,615 | 0,619 |
| cobertura mínima < 8 | 8.748 | **0,595** | 0,585 | 0,584 | 0,593 | 0,595 |
| cobertura mínima ≥ 29 | 9.084 | **0,618** | 0,607 | 0,606 | 0,613 | 0,618 |
| sin estrenos | 5.830 | 0,601 | 0,593 | 0,594 | 0,602 [−0,007; +0,009] | 0,604 |
| 3+ sin huella | 10.077 | **0,594** | 0,584 | 0,581 | 0,593 | 0,594 |
| 0 sin huella | 6.670 | **0,620** | 0,608 | 0,606 | 0,613 | 0,618 |

**El orden no cambia en ningún grupo:** la logística es la mejor o empata en los 11; los
boostings pierden en todos. Lo que sí cambia es la *dificultad*: con lobbies de Diamond, poca
cobertura o 3+ jugadores sin huella, **todos** los expertos caen a ≈ 0,59, y con buena
información suben a ≈ 0,62. Falta información, y ningún experto entrenado sobre las mismas
variables la compensa. Vista previa del stacking (ajuste en una mitad de V por fecha,
evaluación en la otra, las dos direcciones): sin compuerta suma +0,0014 y +0,0033 de AUC y
baja el log-loss de 0,6764 a 0,6740 y de 0,6739 a 0,6714. Con compuerta por grupo da lo mismo
o un poco menos (+0,0007 y +0,0026).

**Conclusión.** Con expertos que ven las mismas variables, un MoE no tiene de dónde ganar: la
compuerta no encuentra casos donde convenga cambiar de experto. Un stacking lineal simple sí
mejora un poco la calibración. Un MoE solo puede rendir si se suman expertos con
**información nueva** (secuencia temporal del jugador, rating dinámico, contexto de la sesión,
duos), y su compuerta puede usar justamente estos descriptores de cobertura.

**Pendiente.**

* Huella acumulada partida a partida: es la única forma de tener a la vez huella reciente
  y muchas partidas de ajuste (decisión del usuario).
* Forma del momento (`daily_form.csv`): medida el 2026-11-12, sin señal (§9.S.24).

**Huella progresiva (2026-11-03).** Pedido del usuario: huella acumulada partida a partida, con
logística, y con cuántas partidas se midió como variable. `historial/ganador_primeros/ganador_huella_progresiva.py`
(~2 min, después de `ganador_features_grande.py`).

* *Huella:* para cada partida, el promedio de los $z$ de estilo de las partidas **anteriores** del
  jugador (la propia no), encogido hacia 0: $h = rac{\sum_j w_j z_j}{\sum_j w_j + k}$, con
  $w_j = 2^{-	ext{edad}_j/v}$ (edad en partidas del jugador). Las medias y desvíos por campeón,
  línea y resultado salen solo de partidas anteriores a V.
* *Cantidad:* $\log(1+n)$ por línea y total, con $n$ las partidas que entraron en la huella.
* *Elección:* $k \in \{0, 5, 10, 20\}$ y $v \in \{30, 100, \infty\}$ con el 15 % más nuevo del
  ajuste, sin mirar V. Ganó $v = 100$, $k = 10$; todas las variantes con $k \ge 5$ quedan a ±0,001.

| Modelo (logística, misma V) | Ajuste | Acierto | AUC | Contra la referencia (IC 95 % pareado) |
| :--- | ---: | :---: | :---: | :---: |
| Referencia: huella congelada al 04/09 | 26.934 | 57,7 % | 0,610 | — |
| Progresiva + $n$, mismo ajuste | 26.934 | 59,0 % | 0,630 | +0,021 [+0,017, +0,024] |
| Progresiva sin $n$, todo el train previo | 160.885 | 59,2 % | 0,634 | +0,024 [+0,020, +0,029] |
| **Progresiva + $n$, todo el train previo** | 160.885 | **59,2 %** | **0,634** | **+0,025 [+0,020, +0,030]** |
| Sin huella, todo el train previo | 160.885 | 57,0 % | 0,597 | −0,013 [−0,018, −0,008] |

La mayor parte de la ganancia es la huella al día (+0,021 con el mismo ajuste); las partidas extra
suman +0,004. $n$ casi no aporta (+0,0001): el encogimiento y la cobertura ya cargan esa información.
Mejora en todos los tramos de partidas previas del lobby, salvo con 100 o más (0,616 → 0,619).

**MLP más grande con la huella progresiva (2026-11-03).** Pregunta del usuario: con 160.885
partidas, ¿ahora sí rinde una red más grande? `historial/ganador_primeros/ganador_mlp.py`: mismas 362
variables, espejada, parada temprana con el 15 % más nuevo del ajuste, 3 semillas.

| Modelo (misma V) | AUC | Contra la logística (IC 95 % pareado) |
| :--- | :---: | :---: |
| Logística | **0,634** | — |
| MLP 64 | 0,632 | −0,003 [−0,004, −0,002] |
| MLP 256-128 | 0,630 | −0,004 [−0,005, −0,003] |
| MLP 512-256-128 | 0,630 | −0,004 [−0,006, −0,003] |
| MLP 1024-512-256 | 0,630 | −0,005 [−0,006, −0,003] |

Curva de aprendizaje (las N partidas más nuevas): la logística da 0,630 / 0,634 / 0,634 / 0,634
con 27, 55, 110 y 161 mil; la MLP 256-128 queda siempre 0,004 abajo y la brecha no se cierra. La
logística se satura con ~55 mil partidas: con estas variables la señal es aditiva, y agrandar la
red no encuentra interacciones. Más datos o más capacidad no son el cuello de botella; lo son las
variables.

**Variables por jugador en vez de azul − rojo (2026-11-03).** Pregunta del usuario: ¿por qué
restar los equipos y pasar el campeón por equipo, y no cada jugador con su campeón, línea, huella y
demás? En una logística, $w_a x_{azul} + w_r x_{rojo}$ con un juego simétrico da $w_r = -w_a$, así
que restar impone esa simetría con la mitad de parámetros. Pero una red que ve
$f(x_{azul} - x_{rojo})$ no puede aprender $f(x_{azul}) - f(x_{rojo})$, así que "no hay
interacciones" podía ser un artefacto de la entrada. `historial/ganador_primeros/ganador_por_jugador.py`:
34 variables por jugador (experiencia, cobertura, dificultad, huella progresiva y su $n$) más su
campeón y su línea; matchups y sinergias quedan restados.

| Modelo (misma V) | AUC | Contra la referencia (IC 95 % pareado) |
| :--- | :---: | :---: |
| Referencia: logística azul − rojo, campeón por equipo | 0,634 | — |
| Logística, los dos equipos por separado | 0,633 | −0,002 [−0,003, −0,000] |
| Logística azul − rojo, campeón × línea | 0,634 | −0,000 [−0,002, +0,002] |
| MLP con los dos equipos concatenados (256-128 o 512-256) | 0,629 | −0,005 [−0,008, −0,003] |
| Codificador compartido por jugador, 128-32 | 0,635 | +0,000 [−0,003, +0,003] |
| Codificador compartido por jugador, 256-64 | 0,635 | +0,001 [−0,002, +0,004] |
| **Promedio de la referencia y el codificador 256-64** | **0,637** | **+0,003 [+0,002, +0,004]** |

El codificador compartido es la primera red que empata con la logística: misma red para los 10
jugadores, suma por equipo y $g(azul) - g(rojo)$. Promediada con la logística, mejora con el IC
entero positivo: sus errores no son los mismos. Concatenar los equipos en una MLP plana empeora:
tiene que aprender diez veces lo mismo y además la simetría.

**Habilidad contra estilo y matchup de estilos (2026-11-03).** Propuesta del usuario: separar la
huella en habilidad y estilo, comparar estilos teniendo en cuenta la habilidad y sumar al matchup
el tipo de jugador. `historial/ganador_primeros/ganador_estilo_habilidad.py`:

* *Habilidad:* $h \cdot w$, con $w$ la dirección de la huella (azul − rojo, sumada por línea) que
  mejor predice ganar, ajustada solo con el ajuste. Pesa sobre todo % del daño (+0,48), muertes por
  minuto (−0,40), oro por minuto (+0,37) y agresividad temprana (+0,36).
* *Estilo:* $h - (h \cdot w)\,w$ en 4 ejes de PCA (76 % de la varianza): farm y oro, muertes contra
  farm, visión, y daño recibido. Ejes continuos y no clusters, porque no hay grupos naturales.
* *Matchup de estilos por línea:* $s^{azul}_i s^{rojo}_j - s^{rojo}_i s^{azul}_j$ para $i < j$,
  antisimétrico; y estilo propio × habilidad del rival de línea.

| Modelo (misma V) | Acierto | AUC | Contra la huella completa (IC 95 %) |
| :--- | :---: | :---: | :---: |
| Huella completa (23 por línea), logística | 59,2 % | 0,634 | — |
| Habilidad (1) + estilo (4) por línea | 59,4 % | 0,635 | +0,000 [−0,001, +0,001] |
| + matchup de estilos | 59,4 % | 0,634 | −0,000 [−0,001, +0,001] |
| + estilo × habilidad del rival | 59,3 % | 0,634 | −0,000 [−0,001, +0,001] |
| Huella completa + matchup + estilo × habilidad | 59,2 % | 0,633 | −0,001 [−0,001, −0,000] |
| Red por jugador con habilidad + estilo | 59,3 % | 0,633 | −0,002 [−0,003, +0,000] |
| La misma red + línea contra línea | 59,2 % | 0,632 | −0,002 [−0,004, −0,001] |

Cinco números por jugador rinden lo mismo que 23: la huella se comprime sin pérdida. El cruce de
estilos entre rivales de línea no aporta, ni lineal ni en la red: con estos datos, qué tipo de
jugador enfrenta a cuál no cambia quién gana más allá de lo que cada uno aporta por separado.

#### S.24. Quién gana: margen previo del jugador, duos y sesión (2026-11-12)

Después del chequeo del MoE (§9.S.23) quedó claro que hacía falta **información nueva**, no
otro modelo sobre las mismas variables. Se midieron tres fuentes, todas internas (decisión del
usuario: no buscar datos de afuera por ahora) y todas solo con `data/splits/train`. Script:
`historial/ganador_primeros/ganador_margen_duos.py`; lógica en `src/features/margin.py`.

**1. Margen previo (idea del usuario: puntajes de ataque y defensa, como los modelos de
Poisson del fútbol).** Hasta acá cada partida aportaba un bit sobre la fuerza del jugador
(ganó o perdió), y la huella descuenta el resultado a propósito porque mide estilo. El margen
usa *por cuánto* le fue bien. Para cada estadística $x$ de una fila (jugador, partida):

$$r = \frac{x - \hat\mu_{c\ell}}{\sigma_\ell},\qquad
\hat\mu_{c\ell} = \frac{\sum_{\text{previas}} x + 20\,\mu_\ell}{n_{c\ell} + 20}$$

donde $\hat\mu_{c\ell}$ es la media del campeón $c$ en la línea $\ell$ con **solo partidas
anteriores**, encogida hacia la media de la línea. No se descuenta el resultado. El puntaje
del jugador antes de cada partida es $S/(N+5)$, con $S$ la suma de sus $r$ previos y $N$ la
cantidad: sobre todas sus partidas anteriores (`acum`) y sobre las últimas 20 (`v20`). Entra
al modelo como azul − rojo por línea y total. Estadísticas, por familia:

| Familia | Estadísticas | Aporte sola sobre la base (A) |
| :--- | :--- | :---: |
| Ataque | kills, asistencias y daño por minuto, % del daño del equipo, participación en kills | +0,019 |
| Economía | oro y CS por minuto, % del oro del equipo | +0,013 |
| Línea | oro a los 10 y 15, XP y CS a los 15, **menos el rival directo de línea** | +0,011 |
| Defensa | muertes y daño recibido por minuto | +0,004 |
| Visión | visión por minuto | −0,001 |

Las familias se solapan: todas juntas dan +0,020, casi lo mismo que ataque sola.

**Resultados en V** (logística, $C = 0{,}05$; diferencia pareada contra la base, IC95):

| Config | Base | + margen y winrate previo | + duos | Acierto |
| :--- | :---: | :---: | :---: | :---: |
| A: huella hasta 04/09, ajuste 26.934 | 0,610 | 0,630 [+0,015; +0,024] | 0,630 | 59,3 % |
| B: huella hasta 09/08, ajuste 85.757 | 0,603 | 0,637 [+0,028; +0,038] | **0,637** | **59,7 %** |
| C: sin huella, ajuste 160.885 | 0,597 | 0,636 [+0,034; +0,044] | 0,636 | 59,7 % |

Es la mejora más grande desde la experiencia con el campeón: de 0,612 (mejor modelo de
§9.S.23, con LP) a 0,637, y de 58,1 % a 59,7 % de acierto.

* **El margen reemplaza a la huella.** En A, todo lo nuevo *sin* huella da 0,632 contra 0,630
  con huella; en B, 0,637 en los dos casos. Como el margen es acumulado partida a partida, no
  necesita reservar un bloque de partidas viejas: se ajusta con todo el train y deja de valer
  el hallazgo 6 de §9.S.23 ("más partidas no alcanzan"). Resuelve la decisión pendiente de la
  huella acumulada.
* **No es el winrate.** El winrate previo solo suma de −0,000 a +0,002. La señal está en el
  margen: dos jugadores con el mismo 50 % no rinden igual, y el emparejamiento no lo compensa.
* **Acumulado contra últimas 20:** `acum` solo +0,019 (A), `v20` solo +0,017, juntos +0,020.
  La forma reciente agrega poco sobre el nivel general.
* **Solo, sin campeones ni experiencia,** el bloque nuevo da 0,595–0,599: se complementa con
  la base, no la reemplaza.

**Verificación de que no hay leakage.** El salto era grande, así que se revisó antes de
creerlo. `tests/test_margin.py` (6 tests): el resultado coincide con un cálculo fila por fila
que solo mira filas anteriores; cambiar los datos y el resultado de una partida y de todas
las posteriores no mueve su puntaje; y si se quita el "− valor propio" de la suma acumulada
fallan 4 de los 6 (test de mutación). Lo único calculado con todo el train son la media y el
desvío por línea de cada estadística, que son una escala y no contienen resultados.

**2. Duos.** Par del mismo equipo con ≥ 3 partidas previas juntos y **ninguna** enfrentados
(por azar se enfrentarían más veces de las que comparten equipo: 25 pares cruzados contra 20
del mismo lado). El 46 % de las partidas tiene alguno. Winrate del equipo con duo: 51,6 % en
lobbies Diamond (49,5 % sin duo) y 48,4 % / 47,8 % / 47,4 % en Master / Grandmaster /
Challenger (50,6–50,8 % sin duo). Como variable (duos azul − rojo, cruzado con Diamond /
Master+) suma +0,001 (IC que toca el cero en A; [+0,000; +0,002] en B y C). Señal real pero
chica. *Sin explicar:* por qué el signo se invierte en Master+, donde no debería haber cola
dúo; hay que mirarlo antes de citarlo. *Resuelto en §9.S.27:* en Master sí hay cola dúo, y el
efecto negativo sobrevive al control por los jugadores. *Aviso:* el tier del lobby sale de la foto del padrón
(§9.S.23, fuga del rango); acá es simétrico entre lados y solo elige el signo del efecto.

**3. Sesión: descartada.** La racha del día parece predecir (52,9 % con ≥ 4 victorias seguidas
contra 48,9 % con ≥ 4 derrotas), pero el exceso sobre el winrate previo del propio jugador es
plano (de −0,4 a −1,6 puntos sin tendencia, en jugadores con ≥ 20 partidas previas): los que
vienen ganando son los que ganan más siempre. Partidas jugadas en el día y descanso entre
partidas, igual (diferencias ≤ 0,5 puntos).

**Alternativas pendientes de comparar.** El modelo de Poisson completo (tasa del equipo =
suma de ataques − suma de defensas rivales, ajustado por máxima verosimilitud) es la versión
estructurada de esto; el promedio previo es su aproximación barata. Conviene construirlo solo
si mejora la estimación con pocas partidas, que es donde el promedio es más ruidoso.
Medido en §9.S.25.

#### S.25. Quién gana: redes y boostings con el margen, y el modelo de Poisson (2026-11-12)

Todo sobre la config C (ajuste con las 160.885 partidas del train anteriores a V, sin huella),
con el margen previo y los duos de §9.S.24. Solo `data/splits/train`.

**1. Los modelos de §9.S.23 con el margen adentro** (`ganador_red_jugador.py --margen
--sin-huella`, `ganador_por_grupos.py --margen --sin-huella`). La red por jugador recibe además
los 32 puntajes de margen de cada jugador; los boostings, las mismas columnas que la logística.

| Modelo | AUC en V | Diferencia contra la logística (IC95) |
| :--- | :---: | :---: |
| Logística | 0,636 | — |
| LightGBM (espejado) | 0,631 | [−0,008; −0,003] |
| CatBoost (espejado) | 0,630 | [−0,009; −0,004] |
| Red por jugador, escalar (3 semillas) | 0,638 | [−0,001; +0,004] |
| **Red por jugador, equipo (3 semillas)** | **0,640** | **[+0,001; +0,006]** |

Es la primera red que le gana a la logística (60,1 % de acierto). En §9.S.23 la salida
"equipo" no aportaba sobre la "escalar"; con el margen sí, o sea que ahora hay combinaciones
entre jugadores del mismo equipo y del rival que antes no tenían con qué armarse. Cada semilla
sola empata con la logística (0,635–0,637): la ventaja sale de promediar. Los boostings
pierden en 10 de los 11 grupos de §9.S.23 y empatan en uno. La red gana en lobbies de Master+
(0,648 contra 0,643) y empata en Diamond y con cobertura baja. Stacking (ajuste en una mitad de
V, evaluación en la otra): +0,005 y +0,002 de AUC sobre la logística sola (antes +0,001 y
+0,003); la compuerta por grupos sigue sin sumar. *Conclusión sobre el MoE:* la mezcla útil es
logística + red por jugador con pesos fijos; no hay evidencia de que convenga cambiar el peso
según el caso.

**2. Modelo de ataque y defensa completo** (`src/features/attack_defense.py`,
`ganador_poisson.py`). Es el modelo de Poisson del fútbol (Maher; Dixon y Coles) sin equipos
fijos: la tasa de un equipo es la suma de sus cinco jugadores,

$$\eta_{\text{azul}} = \mu + \text{lado} + \sum_{i \in \text{azul}} (a_i + A_{c_i \ell_i})
  - \sum_{j \in \text{rojo}} (d_j + D_{c_j \ell_j}),$$

con $a_i$ el ataque del jugador, $d_j$ su defensa y $A$, $D$ los efectos del campeón en la
línea; lo mismo para el rojo con los papeles cambiados. Tres medidas del equipo: kills
($y \sim \text{Poisson}(\text{minutos}\cdot e^{\eta})$), oro por minuto y daño por minuto
($y \sim \mathcal{N}(\eta, 1)$, estandarizados). Se estima todo junto por máxima verosimilitud
con penalización $\tfrac{\lambda}{2}\sum\theta^2$, que deja cerca de 0 a los jugadores con pocas
partidas. Para no mirar el futuro, el train se parte en 40 bloques por fecha y cada partida
usa los parámetros ajustados con los bloques **anteriores** (V cae en los bloques 34–39).
Variables por medida: suma de ataques, suma de defensas y $\eta$, las tres como azul − rojo.

| Sobre la base (0,597) | AUC | Diferencia (IC95) |
| :--- | :---: | :---: |
| + kills (Poisson), λ = 300 | 0,604 | [+0,005; +0,010] |
| + daño, λ = 5 | 0,606 | [+0,007; +0,011] |
| + oro, λ = 5 | 0,609 | [+0,010; +0,015] |
| + las tres medidas | 0,610 | [+0,011; +0,016] |
| + margen previo y duos (§9.S.24) | **0,636** | [+0,034; +0,044] |
| + margen, duos y las tres medidas | **0,639** | +0,0024 sobre el margen [+0,0016; +0,0033] |

El modelo completo tiene señal, pero **rinde un tercio de lo que da el margen** (+0,014 contra
+0,040), y la hipótesis de que ayudaría con pocas partidas no se cumple: con cobertura mínima
< 8 da 0,596 contra 0,626 del margen, y sumado al margen aporta lo mismo en los tres terciles
(+0,002 a +0,003). Por qué pierde:

* *Reparte el crédito del equipo.* La medida es del equipo y se la atribuye a cinco jugadores
  a la vez; el margen usa la estadística **individual** de cada uno (sus kills, su oro, su
  línea contra el rival directo), que dice mucho más por partida.
* *Parámetros atrasados.* Dentro de un bloque no se actualizan (en V, unos dos días); el
  margen se actualiza partida a partida.
* *Un número por jugador,* sin distinguir línea ni lo reciente.

La penalización casi no importa (λ = 50 contra 300 en kills, 5 contra 25 en oro y daño:
diferencias ≤ 0,001; no se eligió λ con V, entran todas juntas).

**Decisión.** El experto de fuerza es el margen previo individual. El modelo de ataque y
defensa queda como variable complementaria (+0,002). Mejor resultado medido en V: red por
jugador "equipo" 0,640; logística con margen, duos y las tres medidas 0,639.

**Tests.** `tests/test_attack_defense.py` (6): signos del diseño, recupera ataque y defensa
simulados (correlación > 0,9), la penalización encoge hacia cero, y los puntajes de un bloque
no cambian si se alteran ese bloque y los posteriores (Poisson y normal).

**Pendiente.** Red por jugador con las tres medidas adentro; ataque y defensa por jugador con
la estadística individual como medida (kills propias ~ Poisson) y actualización partida a
partida; stacking final logística + red sobre predicciones fuera de muestra de todo el train.
Los dos primeros, medidos en §9.S.26.

#### S.26. Quién gana: red con las medidas de Poisson y stacking fuera de muestra (2026-11-12)

Las dos continuaciones que eligió el usuario al cerrar §9.S.25: (b) meter el modelo de ataque
y defensa dentro de la red por jugador, y (a) un stacking de logística + red cuyos pesos no se
ajusten con V. Config C (ajuste: las 160.885 partidas del train anteriores a V, sin huella),
con margen y duos de §9.S.24. Solo `data/splits/train`; el test sigue sin tocarse.

**(b) Red con Poisson** (`ganador_red_jugador.py --poisson`). `block_scores`
(`src/features/attack_defense.py`) ahora devuelve también $a_i$ y $d_i$ de cada uno de los 10
jugadores, no solo las sumas por equipo. Cada jugador entra a $\phi$ con su ataque y su defensa
en las tres medidas y los dos λ (12 números más); las 18 diferencias por partida (ataque,
defensa y $\eta$, azul − rojo) entran a la parte lineal de la red y a la logística de
referencia. Siguen siendo parámetros de bloques temporales **anteriores**: el test de
no-leakage existente cubre las salidas nuevas porque salen del mismo $\theta$.

| Modelo (3 semillas en las redes) | Sin Poisson | Con Poisson |
| :--- | :---: | :---: |
| Logística | 0,6362 | 0,6387 |
| Red por jugador "equipo" | 0,6397 | **0,6416** (60,2 %) |
| Red − logística (IC95) | +0,0035 [+0,0009; +0,0060] | +0,0029 [+0,0002; +0,0055] |
| Red con Poisson − red sin Poisson | — | +0,0019 [+0,0002; +0,0034] |

Poisson suma lo mismo a la red que a la logística (+0,002): es información nueva, pero
pequeña, y la red no le saca más partido que una combinación lineal.

**(a) Stacking fuera de muestra** (`ganador_red_jugador.py --oos 5`, `ganador_stacking.py`).
El problema del stacking de §9.S.25 era que los pesos se ajustaban con una mitad de V. Acá el
ajuste se parte en $K = 5$ bloques por fecha y el bloque $k \ge 1$ se predice con una
logística y una red (3 semillas) entrenadas **solo con los bloques $< k$** (ventana que
crece; la parada temprana de la red usa el 15 % más nuevo de esos bloques). Eso da 128.708
predicciones fuera de muestra y en orden temporal por experto, el requisito de la tarea del
MoE. El meta-modelo es una logística sin penalización sobre los logits de los expertos,

$$\hat p = \sigma\Big(c + \sum_e w_e \,\operatorname{logit} p_e\Big),$$

ajustada con esas predicciones y aplicada a las predicciones en V de los expertos entrenados
con todo el ajuste. V no interviene ni en los expertos ni en los pesos. *Desfasaje conocido:*
los expertos del bloque 1 vieron 32 mil partidas y los de V 161 mil, así que sus predicciones
son menos seguras; por eso también se ajustó el meta con los 2 últimos bloques solos. Los
pesos cambian poco y el AUC en V no cambia.

AUC fuera de muestra por bloque: la red le gana a la logística en los 4 bloques (+0,001 a
+0,005), con y sin Poisson. El resultado de V no es suerte de un período.

| Combinación (en V) | AUC | Log-loss | Pesos | Contra el mejor experto solo (IC95) |
| :--- | :---: | :---: | :--- | :---: |
| Logística + red (margen) | 0,6404 | 0,6614 | 0,38 / 0,71 | +0,0007 [−0,0004; +0,0017] |
| Logística + red (con Poisson) | 0,6423 | 0,6604 | 0,38 / 0,69 | +0,0007 [−0,0002; +0,0018] |
| Las dos redes | 0,6418 | 0,6608 | 0,45 / 0,67 | +0,0002 |
| **Los 4 expertos** | **0,6428** | **0,6602** | −0,21 / 0,35 / 0,55 / 0,40 | **+0,0012 [+0,0002; +0,0023]** |
| Promedio simple de los 4 logits | 0,6414 | 0,6609 | (iguales) | −0,0002 |

Lectura:

* **El stacking casi no suma sobre la mejor red.** Contra la logística sí (+0,004 a +0,007,
  IC lejos del cero), pero eso lo da la red sola. Sobre la red con Poisson, el mejor stacking
  suma +0,001; mejora más el log-loss (0,6611 → 0,6602) que el orden. Los pesos suman ≈ 1,1:
  el meta corrige un poco la falta de confianza de los expertos.
* **El peso negativo de la logística con margen** no es un error: está anidada en la logística
  con Poisson (mismas variables más 18), así que el meta usa la diferencia entre las dos para
  subir el peso de Poisson. Sin ella (3 expertos) da 0,6424, con IC que toca el cero
  [−0,0002; +0,0020] contra la red con Poisson.
* **Selección sobre V.** Se miraron 7 combinaciones en V; la mejor está elegida mirando V, así
  que su +0,001 es optimista. Lo robusto es: red ≥ logística, y el stacking ≥ cada experto,
  en todas las combinaciones.

**Decisión.** Modelo de quién gana para llevar al test: **stacking de los 4 expertos** (red
y logística, con y sin Poisson; pesos ajustados fuera de muestra), AUC 0,643 y 60,1 % en V.
La alternativa más simple, que pierde 0,001, es la red con Poisson sola (0,642). La mezcla
por grupos (MoE) queda descartada (§9.S.25). Lo que sigue no es otro modelo: es información
nueva o pasar al test.

**Alternativas descartadas en esta sección.** Ajustar el meta con una mitad de V (§9.S.25):
usa V dos veces y deja la mitad para evaluar. Folds aleatorios en lugar de bloques por fecha:
el margen y el Poisson se calculan con el pasado, y un fold aleatorio dejaría al experto
entrenar con partidas posteriores a las que predice. Promedio simple: no necesita ajuste,
pero no puede corregir la confianza ni bajar el peso de un experto redundante (0,6414 contra
0,6428).

#### S.27. Quién gana: los duos de Master+ (2026-11-12)

Pendiente de §9.S.24: el equipo con duo gana más en Diamond (51,6 %) y menos en Master+
(47–48 %). Pregunta del usuario: ¿en Master+ se puede jugar en duo, o son coincidencias?
Scripts: `historial/ganador_primeros/duos_roles.py` y `duos_parche.py`. Solo `data/splits/train`.

**1. El nulo correcto.** Para separar duos reales de coincidencias se compara cuántos pares
caen *siempre juntos* contra cuántos caen *siempre enfrentados*. El emparejamiento es por
roles, así que dos jugadores de la **misma** línea son rivales directos y nunca van juntos. Un
primer intento con $P(\text{juntos}) = 4/9$ para todo par daba un nulo equivocado, y quedó
descartado. Restringido a pares de líneas distintas en el cruce, el compañero del otro lado
ocupa 4 de los 8 puestos con otra línea, así que

$$P(\text{juntos}) = \tfrac12,\qquad \frac{\#\text{siempre juntos}}{\#\text{siempre enfrentados}}
\xrightarrow{\text{azar}} 1\ \text{para cualquier } k.$$

En el primer cruce de cada par, $P(\text{juntos})$ medida da 0,491–0,497: el nulo se cumple.

**2. Hay duos reales en Master; en GM pocos; en Challenger ninguno.** Contando pares con
≥ 4 cruces **dentro del mismo parche** (los duos reales juegan muchas partidas seguidas):

| Parche | Diamond | Master | Grandmaster | Challenger |
| :--- | :---: | :---: | :---: | :---: |
| 16.13 | 82 / 0 | 724 / 13 (56) | 84 / 8 | 10 / 1 |
| 16.15 | 186 / 0 | 1.001 / 29 (35) | 27 / 11 (2,5) | 3 / 1 |
| 16.17 | 766 / 0 | 1.478 / 54 (27) | 61 / 17 (3,6) | 7 / 8 |

(siempre juntos / siempre enfrentados; entre paréntesis el cociente, que por azar vale 1).
Pasa en todos los parches de 2026 con datos (16.1–16.19). La regla publicada que encontramos
(guías y prensa, no las notas oficiales de Riot, y contradictorias entre sí sobre las fechas)
dice lo mismo: GM y Challenger no pueden jugar en duo; Master sí, con un compañero de rango
cercano. Los pocos duos de lobbies GM serían jugadores de Master en un lobby que sube de tier
(el tier del lobby es el más alto con ≥ 5 jugadores, §9.S.12). Contando sobre todo el train en
lugar de por parche, con $k = 3$ cruces Master parecía no tener duos (cociente 0,98): los pares
con pocos cruces son casi todos coincidencias, y los duos reales están en $k$ alto. La
variable de §9.S.24 (≥ 3 partidas juntos y ninguna enfrentados), en cambio, cuenta cada
partida, y la mayoría de sus filas (66 mil de 80 mil en Master) son pares con ≥ 5 partidas
juntos: mide sobre todo duos reales.

**3. El duo de Master+ pierde, aun controlando por los jugadores.** Logística con todas las
variables de la config C (campeones, experiencia, margen, winrate previo) más los duos,
ajustada sobre el train y V juntos (es una medición descriptiva del efecto, no una evaluación;
IC por bootstrap de partidas, 20 réplicas):

| Lobby | Sin controles | Con el modelo completo |
| :--- | :---: | :---: |
| Diamond | +1,9 pp | +0,6 pp [−0,2; +1,2] |
| Master+ | −2,4 pp | **−2,8 pp [−3,1; −2,2]** |

En Diamond la ventaja del duo se explica casi entera por quiénes juegan en duo (jugadores con
mejor margen previo). En Master+ el efecto es real y negativo: lo que el modelo sabe de los
jugadores por separado no lo explica.

**Pistas sobre el mecanismo** (descriptivas; ninguna lo cierra):
* *El par es más débil que su lobby.* Tier de la foto (0 = Diamond … 3 = Challenger): en
  lobbies Master el par promedia 0,78 contra 0,89 del lobby; en GM, 1,43 contra 1,67. Con la
  regla de "rango cercano", el de abajo del par entra a lobbies más altos.
* *Reencolar juntos después de perder.* Par que viene de jugar la partida anterior juntos: en
  Master gana 50,1 % si la ganó y 47,8 % si la perdió (Diamond: 52,8 % y 49,7 %).
* *Compensación del emparejamiento por premade* (hipótesis): si Riot le suma MMR al duo para
  emparejarlo, el rival es más fuerte de lo que el duo aporta. No se puede medir sin el MMR.
* *Coincidencias en GM:* pares que el emparejador puso del mismo lado ≥ 3 veces sin ser duo
  ganan 47,5 % (2.255 filas). Sin explicación; puede ser cómo el emparejador reparte a los
  jugadores con MMR parecido.

**Decisión.** La variable de duos queda como está (separada Diamond / Master+, que es justo lo
que el signo pide). Se puede citar: "en Master+ el equipo con duo gana 2,8 puntos menos de lo
esperable por sus jugadores". Pendiente, si se quiere afinar: separar duos reales de
coincidencias con la regla por parche (≥ 4 cruces, todos juntos) y medir si la variable mejora.

#### S.28. El parche desde el contenido de la partida, y la fuerza del campeón por parche (2026-11-12)

Idea del usuario: predecir el parche como información nueva. El parche es una función de la
fecha (cada versión ocupa unas dos semanas sin solaparse), así que la pregunta útil es
cuánto lo delata el **contenido** de la partida: eso mide cuánto cambia el meta. Scripts:
`historial/ganador_primeros/parche_predecir.py` y `parche_ganador.py`. Solo `data/splits/train`.

**1. Clasificador del parche** (logística multinomial, partición aleatoria estratificada 80/20,
8 parches con ≥ 5.000 partidas, 16.12–16.19, 169.168 partidas). La partición es aleatoria a
propósito: se pregunta si los parches se distinguen, no se pronostica el próximo.

| Variables | Acierto | Balanceado | A ±1 parche |
| :--- | :---: | :---: | :---: |
| Azar (clase mayoritaria / balanceado) | 0,221 | 0,125 | — |
| Campeones por línea (antes de la partida) | 0,306 | 0,249 | 0,646 |
| Ítems finales | 0,311 | 0,237 | 0,688 |
| Estadísticas de juego | 0,222 | 0,129 | 0,587 |
| Todo | 0,359 | 0,302 | 0,724 |

* El meta **deriva de a poco, sin saltos**: los errores caen en el parche vecino (la matriz de
  confusión es una banda alrededor de la diagonal) y el último parche (16.19, 9.591 partidas)
  se confunde casi siempre con los dos anteriores.
* Las estadísticas de juego (duración, kills, oro, visión por minuto) **no** cambian con el
  parche: el ritmo de las partidas es estable en este período.
* Cambios visibles (tasa de aparición por partida): Locke, campeón nuevo en 16.13, pasa de
  0 a 0,11 en mid; Senna ADC cae de 0,21 a 0,09 en 16.13 y a 0,03 en 16.15; Camille support
  sube a 0,09 y cae a 0,03 en 16.16 (probable nerf); Nasus top de 0,04 a 0,10 en 16.16;
  magos en bot (Syndra, Viktor, Ziggs, Mel) suben en 16.14–16.15 y bajan en 16.16.

**2. ¿Sirve para quién gana?** La forma de usar el parche antes de la partida es la fuerza del
campeón en el parche en curso. Para cada (campeón $c$, línea $\ell$), solo con partidas
anteriores:

$$x_{\text{parche}} = \frac{G_p + K\,e_{c\ell}}{N_p + K} - e_{c\ell},$$

con $N_p$, $G_p$ las partidas y victorias previas de $(c,\ell)$ **en el mismo parche**,
$e_{c\ell}$ el winrate acumulado de toda la historia y $K = 50$. Vale 0 al empezar cada parche.
Como control, $x_{\text{reciente}}$ es lo mismo con olvido exponencial (vida media de 300
partidas de $(c,\ell)$), que no sabe dónde empieza el parche; y la popularidad en el parche,
elecciones previas de $(c,\ell)$ sobre partidas previas del parche. Entran azul − rojo por
línea y total, sumadas a la logística de la config C con margen y duos:

| Sobre la base (0,6362) | AUC | Diferencia (IC95) |
| :--- | :---: | :---: |
| + fuerza en el parche | 0,6364 | +0,0001 [−0,0000; +0,0004] |
| + fuerza reciente (olvido) | 0,6366 | +0,0003 [+0,0000; +0,0006] |
| + popularidad en el parche | 0,6372 | +0,0009 [+0,0002; +0,0015] |
| + $e_{c\ell}$ directo | 0,6370 | +0,0007 [+0,0005; +0,0009] |
| + las tres de parche/reciente/popularidad | 0,6375 | +0,0012 [+0,0005; +0,0019] |

**Lectura.** Que el campeón esté fuerte *en este parche* no agrega casi nada sobre su fuerza
acumulada: el one-hot de campeones y $e_{c\ell}$ ya capturan casi todo, y dentro de un parche
los winrates de campeón se mueven poco comparados con lo que pesan los jugadores (el margen
sumó +0,040). La popularidad suma un poco más que el winrate; una lectura posible es que
mide lo que el meta adopta antes de que el winrate lo refleje. Lo que se aprendió es
descriptivo: el meta cambia de a poco y por campeones puntuales, y el resultado de la partida
depende de quién juega mucho más que de en qué parche.

**Decisión.** No se agregan variables de parche al modelo de quién gana: +0,001 en la
logística no justifica tres bloques más. Si se quisiera, la popularidad en el parche es la
única que vale la pena probar en la red.

#### S.29. Quién gana: duos reales, margen por campeón y la meta del 65 % (2026-11-12)

Pedido del usuario: separar los duos reales de las coincidencias y probarlos, y después ver
qué afinar para llevar el acierto de 60,1 % a 65 %. Config C, solo `data/splits/train`.

**1. Duos reales contra coincidencias** (`ganador_duos_real.py`). Con el nulo de §9.S.27 (una
coincidencia cae del mismo lado con $P = 1/2$), un par con $p_j \ge 5$ partidas juntos y
ninguna enfrentado es duo casi seguro ($2^{-5}$ por azar), y $p_j \in \{3, 4\}$ es dudoso.
Variables por equipo (azul − rojo, separadas Diamond / Master+): duos seguros, dudosos,
"vienen de jugar juntos la partida anterior", "y la perdieron", y el registro conjunto del duo
seguro, $(g_j + 5)/(p_j + 10) - 0{,}5$. Sobre la logística con margen y sin duos (0,6357),
ninguna versión suma más de +0,0006 (IC que toca el cero), igual que la variable vieja
(+0,0005). El efecto de un duo es grande (−2,8 pp en Master+), pero en la mayoría de las
partidas los dos equipos tienen los mismos duos y la diferencia es 0: no ordena partidas.
**Decisión:** la variable de duos queda como estaba; separar reales no aporta.

**2. Qué haría falta para el 65 %.** Con logits normales y separación $d'$ (modelo binormal),
acierto y AUC van juntos: AUC 0,637 ↔ 59,8 %, 0,664 ↔ 61,7 %, 0,688 ↔ 63,6 %, **0,715 ↔
65,5 %**. Llegar al 65 % en todas las partidas pide un AUC ≈ 0,71, +0,07 sobre lo que había;
todo el camino desde la base de §9.S.23 fueron +0,046. El techo lo pone el emparejamiento:
arma partidas parejas con el MMR, y antes de la partida solo se puede aprovechar lo que el MMR
no ve.

**3. Margen por grupo** (`src/features/margin.py: prior_margin_by`,
`ganador_margen_grupos.py`). El margen de §9.S.24 es un número por jugador y estadística.
Ahora también dentro de un grupo $g$ (su campeón, su línea), como desvío del margen general,
encogido hacia él:

$$m_{pg} = \frac{S_{pg} + k_g\, m_p}{N_{pg} + k_g},\qquad \text{variable} = m_{pg} - m_p,$$

con $S_{pg}$, $N_{pg}$ las sumas previas de los residuos $r$ (§9.S.24) del jugador en el grupo,
$m_p$ su margen acumulado y $k_g = 10$. Sin partidas previas en el grupo vale 0. Tests
(`tests/test_margin.py`, 2 nuevos): coincide con un cálculo fila por fila que solo mira filas
anteriores y no cambia si se alteran la partida y las posteriores; si la suma del grupo incluye
la fila propia, fallan los dos (mutación).

| Sobre la base (0,6362, margen y duos) | AUC | Acierto | Diferencia (IC95) |
| :--- | :---: | :---: | :---: |
| + margen por campeón | 0,6432 | 60,4 % | +0,0069 [+0,0045; +0,0093] |
| + margen por línea | 0,6395 | 60,1 % | +0,0033 [+0,0015; +0,0051] |
| + forma muy reciente (últimas 5) | 0,6370 | 59,8 % | +0,0008 [−0,0001; +0,0016] |
| + los tres | **0,6446** | **60,6 %** | +0,0083 [+0,0056; +0,0113] |

El margen por campeón es la mejora más grande desde el margen mismo: rendir bien con *ese*
campeón dice más que rendir bien en general. Con $k_g$ = 3, 10 y 30 el resultado es el mismo
dentro del ruido (+0,0068 / +0,0069 / +0,0060 por campeón), así que no hay que afinar $k_g$.

**4. Modelo final en V.** Con los márgenes por grupo, Poisson, margen y duos: logística
0,6469 (60,7 %), red por jugador 0,6459 (con tantas entradas cada semilla corta en ~20
épocas y ya no le gana a la logística). Stacking fuera de muestra de los 6 expertos (logística
y red, en las tres variantes; pesos con los 5 bloques temporales de §9.S.26): **AUC 0,649,
60,7 %**. Desde el inicio de la sesión: 0,636 / 59,7 % → 0,649 / 60,7 %.

**Acierto según cobertura** (el modelo solo opina cuando $|p - 0{,}5|$ supera un umbral):

| Partidas más seguras | Umbral $\lvert p - 0{,}5\rvert$ | Acierto |
| :--- | :---: | :---: |
| 10 % | 0,220 | 77,2 % |
| 30 % | 0,140 | 70,4 % |
| 50 % | 0,090 | 66,6 % |
| **60 %** | **0,069** | **65,3 %** |
| 75 % | 0,042 | 63,5 % |
| 100 % | 0 | 60,7 % |

**Decisión.** El 65 % sobre todas las partidas no es alcanzable con lo que hay. El 65 % se
cumple en el 60 % de las partidas donde el modelo está más seguro, y así conviene presentarlo:
un predictor que dice cuándo sabe. *Ojo:* el umbral se leyó en V; para el test hay que fijarlo
antes (por ejemplo con las predicciones fuera de muestra del ajuste). Modelo candidato para el
test: el stacking de los 6 expertos.

#### S.30. Quién gana: cambio reciente del jugador y encaje del equipo (2026-11-13)

Propuesta del usuario: armar clusters para sumar información, empezando por las **cuentas
boosteadas** (el jugador se vuelve OTP de un campeón y empieza a ganar; validar con League of
Graphs; cluster y vecinos más cercanos), y clusters de daño y de economía.

**Por qué se midió antes de clusterizar.** Lo que puede predecir no es la etiqueta
"boosteada" sino que el historial dejó de describir a quien juega. Eso se mide por partida y
sin etiquetas; si las medidas no suman, un cluster armado con ellas tampoco (un cluster es una
función de sus variables, con menos información). Además: (a) OTP + buenos resultados no
separa una cuenta boosteada de un jugador que encontró un campeón fuerte; (b) el perfil de hoy
en una página externa incluye el período del test, así que sirve para validar casos, no para
construir la variable; (c) los clusters por jugador de daño y economía ya se midieron y no
tienen grupos naturales (§9.S.2), y los promedios de daño y oro son fuerza, que el margen ya
captura.

**Medidas** (`src/features/change.py`, `ganador_cambio.py`). Para cada jugador y partida se
comparan sus últimas $W = 10$ partidas previas (ventana reciente) contra todas las anteriores
a esas ($n_{ant} \ge 5$; si falta alguna ventana la medida vale 0, 76,1 % de las filas tiene
las dos):

* **Pool.** Concentración $c = \frac{\sum_k n_k^2 - n}{n(n-1)}$, la probabilidad de que dos
  partidas distintas de la ventana sean con el mismo campeón ($n_k$ partidas con el campeón
  $k$). Se usa en vez de Herfindahl $\sum_k (n_k/n)^2$ porque este crece al achicarse la
  ventana y compararía mal 10 partidas contra 80. Variables: $c_{rec} - c_{ant}$ y
  `otp_nuevo` = fracción de la ventana reciente con el campeón de la partida menos la misma
  fracción en la anterior.
* **Resultados.** $\Delta w$ = winrate reciente − anterior, y la conjunción de la hipótesis,
  $\max(\text{otp\_nuevo}, 0)\cdot\max(\Delta w, 0)$.
* **Huella.** Para cada estadística de `STYLE_FEATURES`, el residuo
  $r = (x - \bar x^{prev}_{c,l,w})/\sigma_{l,w}$ (media previa del campeón $c$ en la línea $l$
  con el resultado $w$, encogida hacia la de $(l, w)$ con peso 20; descontar el resultado saca
  la fuerza, como en §10), y
  $z = (\bar r_{rec} - \bar r_{ant})/\sqrt{1/n_{rec} + 1/n_{ant}}$. La distancia es la media
  de $z^2$. También su producto con $\Delta w$. $\sigma_{l,w}$ sale solo del período de ajuste.
* **Hábitos.** Horario: distancia de variación total entre las proporciones de 6 franjas de
  4 h (hora de Argentina, suavizado 0,5) de cada ventana. Ritmo:
  $\log\frac{n_{48h} + 1}{n\cdot 48h/(t - t_0) + 1}$, partidas previas en las últimas 48 h
  contra las esperadas por el ritmo histórico del jugador ($\ge 10$ previas y $\ge 7$ días).
* **Encaje del equipo.** Con el margen acumulado de porción de oro y de daño de los 5
  (§9.S.24): $|\sum_i m_i|$ y el desvío entre los 5. La suma sola ya está en la base; lo nuevo
  es lo no lineal (todos piden recursos, o nadie).

Las distancias (huella, horario) no tienen signo y su valor sin cambio depende de $n_{ant}$:
se les resta la media de las filas de ajuste con parecida cantidad de partidas previas (20
cuantiles). Todo entra como azul − rojo, por línea y total. Tests (`tests/test_change.py`, 9):
cada medida coincide con un cálculo fila por fila que solo mira filas anteriores, y ninguna
cambia si se alteran la partida actual y las posteriores.

**Resultado** (config C, sobre la logística de §9.S.29; diferencia pareada de AUC en V, IC95
con 200 remuestreos):

| Sobre la base (0,6446) | AUC | Diferencia (IC95) |
| :--- | :---: | :---: |
| + pool | 0,6446 | +0,0000 [−0,0005; +0,0005] |
| + resultados (con la conjunción OTP y gana) | 0,6447 | +0,0001 [−0,0001; +0,0004] |
| + huella | 0,6447 | +0,0001 [−0,0002; +0,0005] |
| + horario | 0,6448 | +0,0002 [−0,0001; +0,0005] |
| + **ritmo** (una sola columna, total del equipo) | **0,6459** | **+0,0013 [+0,0007; +0,0020]** |
| + encaje | 0,6448 | +0,0003 [−0,0002; +0,0007] |
| + todo | 0,6460 | +0,0014 [+0,0003; +0,0024] |

Solas, sin la base: pool 0,510, resultados 0,514, huella 0,502, hábitos 0,518, encaje 0,528.

* **La hipótesis de la cuenta boosteada no suma.** Ni el cambio de pool, ni "se volvió OTP y
  empezó a ganar", ni la distancia de huella mueven el AUC. No dice que no haya cuentas
  boosteadas: dice que, a 10 partidas contra el resto, la señal no ordena partidas más allá de
  lo que ya ordenan el margen por campeón y el margen reciente (que ya contienen "le va mejor
  que antes con este campeón"). **Decisión:** no se etiqueta ni se arma el cluster de
  boosteadas para el modelo de quién gana. La detección de cambio de persona sigue siendo el
  objetivo de la huella (§10, nivel 3), con su propia validación.
* **El encaje no suma** (+0,0003): la composición de estilos de oro y daño no agrega sobre la
  suma de márgenes.
* **El ritmo suma poco pero es estable.** Coeficiente positivo en las 5 líneas (el equipo cuyos
  jugadores vienen jugando más que lo habitual gana más), y la mejora aparece en las dos
  mitades de V (0,6458 → 0,6471 y 0,6434 → 0,6445). Es del tamaño de los duos y del parche
  (+0,001), que no entraron. *Reserva:* el ritmo depende de qué partidas bajó el crawler, no
  solo de cuánto jugó la persona; antes de usarlo hay que verificar que no sea un efecto de la
  cobertura. **Decisión:** queda como candidato, no entra al stacking todavía.

**MMR de páginas externas (consulta del usuario).** Las páginas de MMR no tienen el MMR de
Riot: lo estiman con el historial público (rango de los rivales, LP ganado y perdido). Lo que
muestran hoy es posterior al train y al test, el mismo problema que el rango de la foto
(§9.S.23). **Decisión:** no se scrapea como variable. Alternativa propia, pendiente: nivel
del lobby previo de cada jugador (promedio del rating de los otros 9 en sus partidas
anteriores) y margen ajustado por ese nivel.

#### S.31. Quién gana: MMR propio con el nivel del lobby, y comparación con un medidor web (2026-11-13)

Pedido del usuario: correr el MMR propio de §9.S.30 y comparar con 10 perfiles de una página
de MMR. Config C, solo `data/splits/train`.

**Idea.** El emparejamiento junta jugadores de MMR parecido, así que el nivel de los
compañeros y rivales con los que el juego te pone es una medición indirecta de tu MMR, más
rápida que el LP (que se mueve de a ±20 por partida y tiene pisos de *decay*). Es lo mismo que
hacen los medidores web que miran "el rango promedio de tus partidas".

**Fórmula** (`src/features/lobby.py: prior_lobby_level`). Con $x_q$ una medida de nivel de cada
jugador, el nivel de una partida para el jugador $p$ es la media de los **otros** 9 (el propio
$x_p$ nunca entra):
$$L(p, m) = \frac{\sum_{q \in m,\, q \ne p} x_q}{\#\{q \ne p \text{ con } x_q\}},$$
y el MMR propio antes de una partida es la media de $L$ en las partidas **anteriores**,
encogida hacia $\mu_0$ (media de $L$ en el ajuste) con peso $k = 5$:
$$\text{nivel\_acum} = \frac{\sum_{\text{previas}} L + k\mu_0}{n + k},$$
y lo mismo sobre las últimas 10 previas con $L$ conocido. Dos versiones de $x$:

* **foto:** LP absoluto de la foto del padrón del 04/09 (`lp_absoluto`, Diamond IV 0 = 0,
  Master 0 = 400). Conocido el 95,4 % de las filas. *Ojo:* para partidas del ajuste anteriores
  a la foto, $x$ de los otros es posterior a la partida. V es toda posterior a la foto, y el
  test también: ahí no hay información del futuro. Se controló ajustando solo con partidas
  posteriores a la foto (abajo).
* **rating:** $\mu$ de TrueSkill previo a cada partida (`data/rating_partida.csv`, §9.S.21):
  solo información pasada.

**Margen ajustado por el rival** (`src/features/margin.py: prior_margin_vs_rival`). Para las
estadísticas contra el rival de línea (diferencias de oro a 10 y 15, XP y CS a 15), el residuo
de cada partida suma el margen previo del rival: $r' = r + m_{rival}$. Sacarle +1 a un rival
que suele sacar +1 vale +2. Si la línea no tiene exactamente un rival, $r' = r$.

Tests (`tests/test_lobby.py`, 4): las dos funciones coinciden con un cálculo fila por fila
que solo mira filas anteriores; el nivel no cambia si se altera el $x$ propio, ni si se
alteran la partida actual y las posteriores; ídem el margen contra el rival.

**¿Mide el MMR?** Por jugador con ≥ 20 partidas en el train (17.521), el MMR propio con la foto
contra el LP real: Spearman 0,627 con la foto 1 y 0,410 con la foto 2 (28/10); MAE 182 y 234
LP. Referencia: la foto 1 contra la foto 2 da 0,520. O sea, el nivel del lobby ordena a los
jugadores casi como su LP del mismo momento, pero predice el LP de un mes y medio después
peor que el LP mismo (la foto 2 tiene mucho *decay*).

**Resultado** (sobre la logística de §9.S.29, 0,6446; diferencia pareada, IC95 con 200
remuestreos):

| Sobre la base | AUC | Acierto | Diferencia (IC95) |
| :--- | :---: | :---: | :---: |
| + nivel del lobby (foto) | **0,6474** | 60,7 % | **+0,0028 [+0,0019; +0,0038]** |
| + nivel del lobby (rating) | 0,6461 | 60,6 % | +0,0015 [+0,0005; +0,0027] |
| + margen ajustado por rival | 0,6453 | 60,6 % | +0,0007 [+0,0000; +0,0014] |
| + rating + rival | 0,6464 | 60,6 % | +0,0018 [+0,0007; +0,0031] |
| + todo | 0,6474 | 60,6 % | +0,0028 [+0,0018; +0,0041] |
| Ajustando solo con partidas posteriores a la foto (26.934): + nivel (foto) | 0,6393 | 60,0 % | +0,0033 [+0,0020; +0,0048] sobre 0,6360 |

* **El nivel del lobby con la foto es la mejora más grande de las ideas nuevas desde el margen
  por campeón** (+0,0028), y no viene del LP del futuro: ajustando solo con partidas posteriores
  a la foto la mejora es igual o mayor (+0,0033). El LP de la foto por sí solo ya estaba
  medido (§9.S.23: −0,002 al sacarlo); lo nuevo es el nivel de los lobbies **del jugador**, un
  promedio de muchos LP ajenos, menos ruidoso que el LP propio.
* El rating TrueSkill da la mitad (+0,0015): con las partidas que bajamos, el rating de los
  otros 9 mide peor el nivel que su LP real.
* El margen ajustado por rival casi no suma (+0,0007): el margen por campeón y por línea ya
  descuenta casi todo.
* **Decisión:** `nivel (foto)` entra como experto candidato para el stacking (§9.S.29). Para
  el test se usa la misma foto del 04/09, anterior a todo el test.

**Comparación con un medidor web (NoobHours, 2026-11-13).** Se eligieron 10 jugadores del
padrón con ≥ 45 partidas, LP en las dos fotos y alguna partida después del 20/09, repartidos
por percentil del MMR propio. Dos no aparecen (cambiaron de Riot ID). De los 8:

| Jugador | MMR propio (LP, 09/2026) | LP foto 1 | LP foto 2 | Rango en la web (11/2026) | MMR de la web |
| :--- | :---: | :---: | :---: | :---: | :---: |
| iwc bearcente | 300 | 412 | 275 | Diamond II 75 | 3363 |
| herolorencab | 413 | 415 | 375 | Diamond II 75 | 3342 |
| eltopoquegira | 472 | 320 | 468 | Diamond I 75 | 3490 |
| lee acc | 546 | 633 | 296 | Diamond II 96 | 3342 |
| t1 yrag | 588 | 406 | 351 | Diamond II 75 | 3333 |
| waver velvet | 645 | 731 | 375 | Diamond I 75 | 3502 |
| calcium | 736 | 935 | 375 | Diamond I 75 | 3467 |
| pascai | 1179 | 1718 | 1630 | Grandmaster 1288 | 4805 |

Spearman con el MMR de la web: MMR propio 0,50 (0,25 sin `pascai`), LP de la foto 1 0,51
(0,27). El MMR de la web contra el rango de la web: 0,88 (0,82).

* **El "MMR" de la web es el rango actual más un ajuste por winrate.** La propia página dice
  que calcula con rango, LP y winrate; los números lo confirman: los tres Diamond II 75 con
  winrate 51–53 % dan 3333–3342, y el de 56 % da 3363 (≈ 3000 + 1,25 × LP absoluto, ±30
  por winrate). No usa el nivel de los lobbies ni información que no tengamos.
* **6 de los 8 están en un piso de *decay*** (Diamond II 75 = 275, Diamond I 75 = 375, §9.S.21):
  cuentas que dejaron de jugar. Su "MMR" en la web es el de una cuenta decaída, no el de
  septiembre.
* **Mismo servidor (verificado a pedido del usuario: el MMR es por servidor).** Las partidas
  del train de los 8 son todas `LA2_` (LAS), y la búsqueda en la web fue en LAS. Las victorias
  y derrotas de la web coinciden **exactas** con las de la foto 2 del ladder de LAS en 7 de 8
  (`pascai` siguió jugando: 282-244 → 323-288). Es la misma cuenta, en el mismo servidor. Y
  confirma el *decay*: con las mismas W/L, tres bajaron de rango entre el 28/10 y el 13/11
  (`eltopoquegira` Master 68 → Diamond I 75, `herolorencab` Diamond I 75 → Diamond II 75,
  `t1 yrag` Diamond I 51 → Diamond II 75) sin jugar una partida.
* **La fórmula de la web, aplicada a septiembre (idea del usuario: sacarla con muchos
  perfiles y aplicarla en nuestro tiempo).** Con los 7 de Diamond, una regresión lineal la
  reproduce con error máximo de 2,4 puntos:
  $$\text{MMR}_{web} \approx 2660 + 1{,}241\cdot \text{LP abs} + 6{,}44\cdot \text{WR}\,(\%)$$
  (≈ 125 por división, ≈ 6 por punto de winrate; en Grandmaster comprime: `pascai` da 4805 y
  la recta daría ≈ 5100). No hacen falta más perfiles para Diamond. Aplicada a la foto del
  04/09 de cada jugador, correlaciona 0,996 con el LP de la foto. En el modelo (con el nivel
  del lobby ya adentro): ajustando con todo el train **baja** 0,021 porque el winrate de la
  temporada en la foto incluye el resultado de las partidas del ajuste anteriores a la foto
  (*leakage*, el mismo de §9.S.23); ajustando solo después de la foto, −0,0004
  [−0,0010; +0,0003]. **Decisión:** no se usa ni se scrapea más; es el LP y el winrate de la
  temporada, que el modelo ya tiene por otras vías.

#### S.32. Quién gana: el nivel del lobby en el stacking (2026-11-13)

Pedido del usuario: sumar el nivel del lobby (§9.S.31) al stacking de §9.S.29. Config C, solo
`data/splits/train`; el test no se tocó.

**Cómo.** La notebook no tiene `torch`, así que el nivel del lobby entra como **experto
logístico** (`ganador_logistica_oos.py`): la logística de §9.S.29 más los bloques que se
pidan, con predicciones fuera de muestra en los **mismos 5 bloques por fecha** que
`ganador_red_jugador.py --oos 5` (el bloque $k$ predicho con lo ajustado en los bloques
$< k$) y el mismo formato de salida, así que `ganador_stacking.py` lo mezcla con los 6
expertos ya guardados. Dos expertos nuevos: `lobby` (grupos + nivel del lobby) y `lobby+P`
(además las diferencias de ataque y defensa de Poisson, §9.S.25). La red por jugador acepta
ahora `--lobby data/analysis/ganador_mmr.pkl` (en la logística de referencia, el bloque por
partida; en $\varphi$, `nivel_acum` y `nivel_v10` de cada jugador) para correrla en la PC con
GPU; todavía no se corrió.

**El nivel del lobby fuera de muestra.** AUC por bloque de la logística sin y con él:
0,6184 / 0,6362 / 0,6439 / 0,6428 contra 0,6189 / 0,6358 / 0,6445 / 0,6450. Solo suma en el
bloque 4 (desde el 02/09), el único que llega a la foto del 04/09. En los bloques anteriores,
donde el LP de la foto es posterior a las partidas, no suma: la foto no le pasa información del
futuro al meta-modelo. Logística sola con Poisson y lobby: **0,6493** en V, igual al stacking
de 6 expertos de §9.S.29 con un solo modelo.

**Stacking** (diferencia pareada contra el stacking de 6 expertos de §9.S.29, IC95 con 200
remuestreos de V; meta ajustado con las predicciones fuera de muestra de los bloques 1-4):

| Meta-modelo | AUC | Acierto | vs 6 expertos (IC95) |
| :--- | :---: | :---: | :---: |
| 6 expertos (§9.S.29) | 0,6489 | 60,7 % | — |
| **7 expertos (+ lobby)** | **0,6513** | **61,0 %** | **+0,0023 [+0,0015; +0,0032]** |
| 8 expertos (+ lobby y lobby+P) | 0,6513 | 61,0 % | +0,0024 [+0,0017; +0,0033] |
| 3 redes + logística lobby+P | 0,6502 | 60,8 % | +0,0013 [+0,0010; +0,0016] |
| red grupos + logística lobby | 0,6491 | 60,8 % | +0,0001 [−0,0004; +0,0007] |
| 7 expertos, meta con los últimos 2 bloques | 0,6523 | 61,0 % | +0,0033 [+0,0025; +0,0045] |

* **Por qué gana el de 7 y no el chico.** Los pesos del de 7 son −0,80 / 0,11 / 0,81 / 0,07 /
  −0,51 / 0,61 / 0,81 (logística margen, red margen, logística Poisson, red Poisson, logística
  grupos, red grupos, logística lobby): se restan logísticas casi iguales. Restar la logística
  de grupos a la de lobby deja, en esencia, la parte del nivel del lobby, con más peso del que
  le da la logística sola (su $C = 0{,}05$ lo encoge junto con las ~1.400 columnas restantes).
  El meta deshace ese encogimiento. Es legítimo (los pesos salen de predicciones fuera de
  muestra y V no interviene), pero frágil: depende de diferencias chicas entre expertos muy
  correlacionados. El de "últimos 2 bloques" agranda esos pesos (−1,23 / … / 1,02) y por eso no
  se elige, aunque dé más en V.
* **Cobertura**, con el umbral fijado **sin mirar V**: meta ajustado con los bloques 1-3, umbral
  = cuantil de $|p - 0{,}5|$ en el bloque 4; después se aplica a V.

| Cobertura buscada | Umbral | Cobertura real en V | Acierto 6 expertos | Acierto 7 expertos |
| :---: | :---: | :---: | :---: | :---: |
| 30 % | 0,150 | 26-27 % | 71,4 % | 71,6 % |
| 50 % | 0,099 | 46 % | 67,4 % | 67,7 % |
| 60 % | 0,077 | 56 % | 65,7 % | **66,0 %** |
| 75 % | 0,047 | 72 % | 63,8 % | 64,0 % |
| 100 % | 0 | 100 % | 60,7 % | 61,0 % |

  El umbral del bloque 4 cubre un poco menos en V (56 % en vez de 60 %): V tiene menos
  partidas con predicción extrema. Así fijado, el 65 % se cumple con margen en el 56 % de las
  partidas.

**Decisión.** Candidato para el test: **stacking de 7 expertos con el meta ajustado en todos los
bloques** (0,6513 / 61,0 %; 66,0 % en el 56 % más seguro, con umbral 0,078 fijado en el bloque
4). Alternativa más robusta si en el test los pesos con signo alternado no se sostienen: 3 redes
+ logística lobby+P (0,6502), sin expertos redundantes. Pendiente en la PC con GPU: la red por
jugador con `--lobby`, que podría reemplazar a las logísticas casi iguales.

*Revisado en §9.S.33:* el candidato pasa a ser el stacking de 4 expertos elegidos sin V.

#### S.33. Quién gana: qué experto del stacking aporta y cuál mete ruido (2026-11-13)

Pedido del usuario. `ganador_stacking_aporte.py`, sobre los 8 expertos (logística y red de
margen, Poisson y grupos; logísticas lobby y lobby+P). La regla para no elegir con V: el
criterio principal es el **meta evaluado bloque a bloque**, ajustado con los bloques $< k$ y
medido en el bloque $k$ ($k = 2, 3, 4$, log loss medio), que nunca ve V. V se mira después.

**1. Los expertos se repiten.** Correlación de logits fuera de muestra: las logísticas entre sí
0,91–0,99 (margen–Poisson 0,993, grupos–lobby+P 0,994), cada red con su logística 0,85–0,91,
las redes entre sí 0,91–0,97. Lo que diferencia de verdad es **logística contra red** y
**margen contra grupos/lobby**.

**2. Solos.** El mejor en los bloques es la red de grupos (log loss 0,65815, AUC 0,6457); en V
el mejor es la logística lobby+P (0,6493). Los de margen y Poisson quedan atrás en las dos
miradas: sus variables ya están dentro de los expertos de grupos y lobby.

**3. Sacando uno a la vez** (los 8 dan 0,65696 en los bloques y 0,6513 en V):

| Sin… | Δ log loss bloques (×10⁻⁴) | Δ AUC en V al sacarlo |
| :--- | :---: | :---: |
| red grupos | **+11,1** | −0,0009 [−0,0020; +0,0002] |
| logística grupos | +0,7 | +0,0009 [+0,0004; +0,0014] |
| cualquiera de los otros 6 | ≤ +0,2 | ≤ +0,0003 |

Solo la red de grupos es imprescindible en los bloques. Los demás son intercambiables: sacar
cualquiera lo cubre otro casi igual.

**4. Pesos.** Ajustando el meta en cada bloque por separado y en 200 remuestreos: logística de
margen y de grupos siempre **negativas** (IC95 [−0,33; −0,09] y [−0,75; −0,38]); red de grupos,
logística de lobby+P y de lobby siempre positivas; la red de Poisson cambia de signo (ruido).
El patrón estable es "lobby+P menos grupos": el meta agranda la diferencia entre la logística
con Poisson y lobby y la logística sin ellas. *Hipótesis probada y descartada:* que la
logística encogiera de más esas 30 columnas; con su penalización dividida por 4, 9, 25 y 100
(escalándolas antes de la regularización, $C = 0{,}05$) el AUC en V queda en 0,6493–0,6494 y
el log loss en los bloques no se mueve. La diferencia no es un problema de regularización
de una sola logística: es una corrección que solo aparece al combinar.

**5. Selección hacia adelante con el criterio de bloques (sin V):**

| Paso | Experto agregado | Log loss bloques | AUC V | vs los 8 (IC95) |
| :---: | :--- | :---: | :---: | :---: |
| 1 | red grupos | 0,65815 | 0,6459 | −0,0054 [−0,0068; −0,0043] |
| 2 | logística lobby+P | 0,65700 | 0,6496 | −0,0017 [−0,0024; −0,0010] |
| 3 | logística grupos (peso −0,51) | 0,65692 | 0,6506 | −0,0007 [−0,0012; −0,0002] |
| 4 | red Poisson | **0,65687** | **0,6512** | −0,0001 [−0,0003; +0,0001] |

Con pesos $\ge 0$ se frena en 3 (red grupos, logística lobby+P, red Poisson): 0,6501 en V. Los
8 con pesos $\ge 0$ dejan 4 en cero y dan 0,6502.

**Qué aporta y qué es ruido.**

* **Aportan:** la red de grupos (la única que no se puede sacar: aporta la no linealidad por
  jugador) y la logística con Poisson y lobby (lo mejor de las variables nuevas). Con las dos,
  0,6496 en V.
* **Aporta poco, pero de forma estable:** la logística de grupos con peso negativo y la red de
  Poisson: +0,0016 en V entre las dos, con muy poca mejora en los bloques (−0,13 × 10⁻⁴ de log
  loss). Se eligieron sin mirar V, así que la mejora en V es una medición honesta, pero chica.
* **Redundantes (ruido en el stacking):** logística y red de margen, logística de Poisson y
  logística de lobby. Sacarlos no cambia nada en los bloques (≤ 0,2 × 10⁻⁴) ni en V
  (−0,0001 [−0,0003; +0,0001]). Sus variables ya están en los expertos que quedan.

**Decisión.** Candidato para el test: **stacking de 4 expertos** (red grupos, logística
lobby+P, logística grupos, red Poisson), elegidos sin V: **0,6512 / 61,0 %**, igual que los 8
con la mitad de expertos. Respaldo si en el test el peso negativo no se sostiene: **2 expertos**
(red grupos + logística lobby+P), 0,6496, pesos positivos. Los otros 4 se descartan.

**Decisión final del usuario (2026-11-13), fijada antes de abrir el test:** el modelo principal es el
**stacking de 2 expertos** (red de grupos + logística con Poisson y nivel del lobby). Razones: la mitad
de entrenamiento (una red en GPU en vez de dos), pesos positivos con un rol distinto cada experto
(no lineal por jugador / variables nuevas), y la ventaja de la de 4 (+0,0016 en V, −0,13 × 10⁻⁴ de log
loss en los bloques) depende de restar dos logísticas con correlación 0,994, frágil ante cambios de
período. La de 4 se reporta solo como comparación; no se elige entre las dos con el test.

#### S.34. Quién gana: diagnóstico del modelo de 2 expertos (2026-11-13)

Pedido del usuario: ver qué hace el modelo elegido y qué se puede mejorar.
`ganador_dos_expertos.py`. Meta: $\text{logit } p = 0{,}711\, z_{red} + 0{,}353\, z_{log} + 0{,}013$
(ajustado con las predicciones fuera de muestra de los bloques 1-4). V: AUC 0,6496, acierto
60,8 %, log loss 0,6569.

**1. Está calibrado.** Por decil, la probabilidad media y la frecuencia real difieren ≤ 0,025
(ECE 0,012); la pendiente de recalibrar en V es 1,006. No hay nada que ganar recalibrando.

**2. Dónde rinde y dónde no** (AUC en V; los descriptores usan solo partidas anteriores):

| Grupo | n | AUC | Acierto | Acierto en el 56 % más seguro |
| :--- | :---: | :---: | :---: | :---: |
| Lobby Master+ | 17.081 | 0,660 | 61,9 % | 66,9 % |
| Lobby Diamond | 9.853 | **0,630** | 58,8 % | 63,9 % |
| 0 jugadores con < 10 partidas previas | 16.928 | 0,657 | 61,5 % | 66,6 % |
| 3+ jugadores con < 10 partidas previas | 3.967 | **0,617** | 57,9 % | 62,5 % (solo el 49 % pasa el umbral) |
| Cobertura mínima < 10 | 10.006 | 0,637 | 59,6 % | 64,5 % |

Por día de V, el AUC va de 0,637 a 0,670 sin tendencia: no se degrada con el tiempo dentro de
V. Por hora, estrenos de campeón y fuera de rol las diferencias son chicas y sin un patrón que
se pueda usar. **El punto débil son los jugadores de los que sabemos poco**, que se concentran
en los lobbies de Diamond (el crawler siguió a Master+, §9.S.12).

**3. Los dos expertos casi siempre coinciden.** Eligen el mismo ganador en el 88 % de V
(acierto 62,2 % los dos); en el 12 % restante la red acierta 48,3 % y la logística 51,7 %: es
una moneda. Que el desacuerdo sea grande no ayuda a ninguno.

**4. Pesos que dependen del caso (compuerta) no mejoran.** Meta con interacciones logit ×
grupo, elegido con el log loss de los bloques 2-4 sin V: ningún grupo baja el log loss más de
0,1 × 10⁻⁴ (la hora lo empeora +1,7 × 10⁻⁴); en V, +0,0003 como mucho. Coincide con §9.S.23 y
§9.S.33.

**Conclusión.** La combinación está saturada: calibrada, sin compuerta útil y con expertos que
coinciden casi siempre. Lo que falta mejorar está **dentro de los expertos**, y el lugar con
más margen son las partidas con jugadores poco conocidos (lobbies de Diamond, 3+ jugadores con
< 10 previas: AUC 0,617 contra 0,657). Caminos, por orden de costo:
1. *Prior para jugadores con poco historial* (sin red): hoy su margen previo vale 0 (la
   media). Reemplazarlo por uno esperado según su nivel (LP de la foto, nivel del lobby de esa
   partida) y encogerlo hacia él con las partidas que tenga.
2. *Completar el historial* de los jugadores con < 10 partidas previas que aparecen en el train
   (crawler, cuota de las 2 keys).
3. *La red por jugador con `--lobby`* en la PC con GPU (§9.S.32).

#### S.35. Quién gana: valor de arranque, rasgos condicionales del jugador e investigación (2026-11-13)

Pedidos del usuario: probar el camino 1 de §9.S.34 y buscar relaciones de juego que no estemos
usando (su ejemplo: "hay jugadores que si les das una kill antes del minuto 10 te hacen la
partida entera"), con dos agentes de investigación en paralelo. Base: la logística con Poisson y
nivel del lobby, experto del modelo de 2 (0,6493 en V). Config C, solo `data/splits/train`.

**1. Valor de arranque del margen** (`src/features/margin.py: prior_shift`,
`ganador_prior.py`). El margen previo es $m = S/(N + k)$ y vale 0 sin historial. Encogido hacia
un valor esperado $\mu$: $m^* = (S + k\mu)/(N + k) = m + k\mu/(N + k)$, así que basta con agregar
el término $k\mu/(N + k)$, que pesa con pocas partidas y se apaga con muchas. $\mu$ por
estadística: regresión lineal del residuo de cada partida contra el nivel previo (LP propio de
la foto, MMR propio, nivel del lobby de esa partida), ajustada con las filas de ajuste. Test
nuevo: coincide con el cálculo fila por fila y no usa la partida actual ni las futuras.

| Sobre la base | Ajuste con todo el train | Ajuste solo después de la foto |
| :--- | :---: | :---: |
| + arranque con LP propio, $k = 5$ | −0,0134 (*leakage*: el LP propio es posterior a las partidas viejas) | +0,0015 [+0,0002; +0,0032] |
| + arranque sin LP propio, $k = 5$ | −0,0001 [−0,0017; +0,0016] | +0,0010 [−0,0001; +0,0022] |

En las partidas con 3+ jugadores de < 10 previas no mejora (0,6084 → 0,6051). El nivel explica
~1 % del residuo de cada partida ($R^2 \le 0{,}012$): saber el nivel de alguien no dice cómo va
a jugar. **Decisión:** se descarta.

**2. Investigación** (dos agentes; fuentes en el informe de cada uno, resumen):
* *Cuánto se puede esperar.* Lo publicado con metodología limpia está en nuestro rango:
  DraftRec (WWW'22, 280 mil partidas de LoL, corte temporal) llega a 55,4 % de acierto;
  NeuralAC (AAAI'21, Dota 2) a AUC 0,66. Las cifras de 70-75 % (Do et al. 2021, White & Romano
  2020) usan pocas partidas, elos bajos o winrates de temporada que probablemente incluyen la
  partida que se predice. Mejoras realistas: +0,005 a +0,015 de AUC.
* *Rasgos condicionales.* No hay estudios de que sean estables por jugador; existen a nivel
  campeón (conversión de ventaja, escalado). Tilt y rachas: efecto real pero chico (−0,6 pp
  después de una derrota en Master). Recomendación de los dos: medir la confiabilidad
  (partidas pares contra impares) antes de modelar.
* *Arquitectura.* Lo que más rinde en la literatura es la interacción explícita entre rivales:
  NeuralAC (vectores de "fortaleza" y "debilidad" con producto entre rivales; sin el módulo de
  competencia pierde ~0,017 de AUC) y Blade-Chest. En una red que suma a los 5 jugadores
  ($\sum\varphi$), los pares se pierden.

**3. Rasgos condicionales y cruce con el rival** (`src/features/conditional.py`,
`ganador_snowball.py`). Por jugador, con partidas previas y encogido hacia la media ($k = 10$):
$\text{rasgo} = (\sum_{\text{previas con } c} v + k\mu)/(\#\text{previas con } c + k)$. Rasgos:
P(adelante en oro a los 10 > +300), P(atrás < −300), P(kill antes del 10), P(muerte antes del
10), **conversión** = media de $(\text{ganó} - \hat p)$ en las partidas en que iba > +500 a los 15,
**remontada** = lo mismo yendo < −500, dependencia de la duración (partidas > 32 min contra
< 25) y volatilidad (desvío de la diferencia de oro a los 15). $\hat p = P(\text{gana} \mid
\text{oro a los 15, línea})$ ajustado con las filas de ajuste: el rasgo mide cuánto gana de más
respecto de lo que su ventaja predice. Cruces antisimétricos con el rival de línea:
conversión × P(muerte antes del 10) del rival, adelante × atrás del rival, kill × muerte del
rival, remontada × adelante del rival. Tests nuevos (3).

*Confiabilidad* (pares contra impares, 11.587 jugadores con ≥ 40 partidas de ajuste):

| Rasgo | r | Rasgo | r |
| :--- | :---: | :--- | :---: |
| kill antes del 10 | **0,727** | conversión de ventaja | **0,058** |
| volatilidad | 0,684 | remontada | 0,029 |
| muere antes del 10 | 0,590 | partidas largas | 0,049 |
| adelante a los 10 | 0,579 | partidas cortas | 0,071 |
| atrás a los 10 | 0,453 | | |

*Aporte:* rasgos +0,0006 [−0,0002; +0,0014]; cruces −0,0002; solo "adelante a los 10"
+0,0005 [+0,0001; +0,0009]; el resto ≤ +0,0002.

**Conclusión.** Cómo arranca cada jugador es muy estable (r 0,45-0,73), pero el margen previo
ya lo contiene (diferencias de oro, XP y CS a los 10 y 15, kills por minuto). Lo que el
usuario describe, convertir una ventaja en la partida, **no es un rasgo estable del jugador** con
nuestros datos: r = 0,06 entre las dos mitades de sus partidas, casi todo ruido (la conversión
depende del campeón, del equipo y del rival, no de la persona). Por eso los cruces tampoco
suman. **Decisión:** se descartan rasgos y cruces; se guardan los arreglos por jugador
(`data/analysis/ganador_snowball.pkl`, clave `jug`) por si la red con interacción entre rivales
los aprovecha. Lo que queda con evidencia: (a) un término de interacción entre rivales de línea
dentro de la red por jugador (tipo NeuralAC/Blade-Chest, antisimétrico), en la PC con GPU; (b)
la red con `--lobby`.

#### S.36. Quién gana: modelo por línea y término de interacción entre rivales (2026-11-13)

Pedidos del usuario: escribir el término de interacción entre rivales en la red, y "hacer el
modelo de cada línea, que también ayuda al puntaje global del equipo". Config C, solo
`data/splits/train`.

**1. Modelo por línea** (`ganador_lineas.py`). Para cada línea $l$ se predice, con información
previa a la partida, la diferencia de oro a los 15 entre los dos jugadores de esa línea,
$y_l = \text{oro15}_{azul} - \text{oro15}_{rojo}$ (en miles; el 99 % de las partidas la tiene).
Entrada: las 101 variables por jugador de los dos (experiencia, margen previo, márgenes por
campeón y por línea, Poisson, nivel del lobby) y el matchup de campeones de esa línea. Dos
modelos, los dos antisimétricos (invertir los lados invierte la predicción):
* *ridge:* lineal sobre (azul − rojo, matchup), espejado y sin constante ($\alpha = 10$);
* *HGB:* `HistGradientBoostingRegressor` sobre (azul, rojo, matchup), entrenado con cada
  partida también espejada ($y \to -y$); la predicción es $(f(a, r) - f(r, a))/2$.

Sin leakage hacia el modelo del equipo: en el ajuste, cada partida se predice con un modelo
entrenado en los **otros** 4 tramos por fecha (*cross-fitting*); V, con todo el ajuste.
*Salvedad:* para las partidas viejas del ajuste, el modelo de línea vio partidas posteriores.
Es un mapeo de población hacia la diferencia de oro, no hacia el resultado, y V queda limpia.

| Línea (V) | Correlación ridge | Correlación HGB | AUC "azul gana la línea" (HGB) |
| :--- | :---: | :---: | :---: |
| Top | 0,355 | 0,373 | 0,672 |
| Jungla | 0,297 | 0,332 | 0,648 |
| Mid | 0,317 | 0,329 | 0,648 |
| Bot | 0,251 | 0,255 | 0,612 |
| Soporte | 0,258 | 0,321 | 0,638 |

La línea es mucho más predecible que la partida (AUC 0,61-0,67 por línea contra 0,65 para
ganar, con una sola dupla de jugadores), y top la que más.

*Aporte al modelo del equipo* (las 5 predicciones y su suma, sobre la logística con Poisson y
nivel del lobby, 0,6493): ridge −0,0001 (es lineal en las mismas variables que la logística ya
tiene); **HGB +0,0018 [+0,0006; +0,0032]** (0,6511). Lo que aporta es la no linealidad dentro de
cada línea: cómo se combinan las variables de los dos jugadores de esa línea.

*En el modelo de 2 expertos* (la logística pasa a tener también las líneas;
`ganador_logistica_oos.py --extra ... "data/analysis/ganador_lineas.pkl:hgb"`): AUC por bloque
fuera de muestra 0,6211 / 0,6393 / 0,6483 / 0,6482, contra 0,6198 / 0,6365 / 0,6453 / 0,6461 sin
ellas: mejora en los 4. Meta $0{,}653\, z_{red} + 0{,}407\, z_{log}$. **V: 0,6496 → 0,6508
(+0,0012 [+0,0008; +0,0018]), acierto 60,9 %**; log loss del meta en los bloques 2-4 −5,0 × 10⁻⁴,
la mejora más grande en los bloques de toda la sesión. **Decisión:** el experto logístico del
modelo de 2 pasa a incluir las líneas (HGB).

**2. Término de interacción entre rivales** (`ganador_red_jugador.py --rivales d`). De la salida
$h_i$ de $\varphi$ salen dos vectores de dimensión $d$ por jugador, fortaleza $f_i = A h_i$ y
debilidad $w_i = B h_i$, y el logit suma, sobre las 5 líneas,
$$\sum_l \langle f_{azul,l},\, w_{rojo,l}\rangle - \langle f_{rojo,l},\, w_{azul,l}\rangle,$$
antisimétrico por construcción. Es la forma de NeuralAC/Blade-Chest: un rasgo del jugador que
solo se activa según un rasgo de su rival directo, que la suma $\sum \varphi$ por equipo pierde.

Corrido en la notebook (RTX 4050, `torch` 2.14.1 + CUDA 12.6, instalado en `.conda` el
2026-11-13), con las mismas entradas que la red de grupos (margen, Poisson, márgenes por grupo,
sin huella), variante equipo, 3 semillas y 5 bloques fuera de muestra, $d = 16$:

| Red | AUC sola en V | AUC por bloque (1-4) | Modelo de 2 (con la logística de líneas) |
| :--- | :---: | :---: | :---: |
| De la PC de escritorio (§9.S.29) | 0,6459 | 0,6298 / 0,6432 / 0,6488 / 0,6453 | 0,6508 |
| Base, re-entrenada en la notebook | 0,6463 | 0,6300 / 0,6445 / 0,6490 / 0,6447 | 0,6510 |
| Con rivales, $d = 16$ | 0,6450 | 0,6279 / 0,6421 / 0,6474 / 0,6440 | 0,6507, −0,0002 [−0,0010; +0,0007] |

* La red re-entrenada en otra GPU reproduce la de escritorio (diferencia en el modelo de 2:
  −0,0001 [−0,0005; +0,0004]).
* **El término de rivales no suma**: queda por debajo en los 4 bloques y en V. Coincide con
  §9.S.35: las variables por jugador no tienen un rasgo estable que se active según el rival, y
  lo que sí depende del rival de línea ya lo capturan los matchups de campeones y el modelo por
  línea. Lo que NeuralAC gana en Dota (+0,017) viene de aprender el par campeón-campeón, que acá
  ya está en los matchups. **Decisión:** se descarta; `--rivales` queda en el script.
* La corrida con rivales y nivel del lobby se cortó en la parte fuera de muestra porque el
  sistema se quedó sin memoria (Claude Code detuvo el proceso en segundo plano). La parte en V
  alcanzó a terminar: 0,645 (3 semillas), sin mejora sobre la base.
* **Red con nivel del lobby, sin rivales** (`--lobby`, relanzada con 8 GB libres): sola 0,6450
  en V; por bloque 0,6299 / 0,6442 / 0,6494 / 0,6459 (mejor que la base solo en el bloque 4,
  +0,0012). En el modelo de 2: log loss del meta en los bloques 0,65636 contra 0,65653
  (−1,7 × 10⁻⁴), pero en V **−0,0008 [−0,0015; −0,0001]** (0,6501). **Decisión:** la red queda
  sin el nivel del lobby: ya entra por la logística, y en la red lo repite.

**Modelo vigente (2 expertos):** red de grupos + logística con Poisson, nivel del lobby y
modelo por línea. **V: 0,6508-0,6510, acierto 60,9-61,0 %.**
* **Conclusión:** no da parecido, y no es un problema del MMR propio: la web mide otra cosa
  (el rango de hoy) en otro momento (noviembre). Con 8 jugadores, el acuerdo de 0,5 lo sostiene
  `pascai`, el único fuera de Diamond. Para validar el MMR propio sirve el LP de las fotos
  (Spearman 0,627 sobre 17.521 jugadores), no la web. Se descarta el scraping masivo.

#### S.37. Quién gana: fuentes externas, composición del draft y runas (2026-11-14)

Pedido del usuario: "scrapear" fuentes externas (Data Dragon, CommunityDragon, Meraki,
Lolalytics, notas de parche…) para sumar información del draft y probarla. Config C, solo
`data/splits/train`.

**1. Qué fuentes se usan y cuáles no.**

| Fuente | Qué da | Decisión |
| :--- | :--- | :--- |
| CommunityDragon (`champions/{id}.json` del cliente) | Tipo de daño (físico / mágico / mixto), `playstyleInfo` 0-3 (daño, durabilidad, control, movilidad, utilidad) | **Se usa** |
| Meraki Analytics (`champions.json`) | Roles finos (TANK, VANGUARD, ENCHANTER, ARTILLERY…), cuerpo a cuerpo / rango, rango de ataque | **Se usa** |
| Data Dragon | Tags gruesos y dificultad 1-10 | Ya estaba (dificultad, §9.S.23); sus tags no se actualizan con los reworks |
| match-v5 (Riot) | Runas, hechizos de invocador (D = `summoner1Id`, F = `summoner2Id`) y nivel de cuenta de cada jugador en cada partida | **Se baja una muestra** (punto 3) |
| Lolalytics, u.gg, op.gg, League of Graphs, Mobalytics | Winrates y matchups por parche y elo | **Descartadas:** sus términos prohíben el scraping; incluyen partidas posteriores a las nuestras (fuga); los matchups ya se calculan con nuestras partidas de Diamante+ en LAS, acumulados (§9.S.23) |
| champion-mastery-v4, league-v4 (Riot) | Maestría de por vida, W/L de la temporada | Pendiente: son fotos de hoy, con la misma fuga que el rango de la foto (§9.S.23) |
| Leaguepedia / Oracle's Elixir | Orden de picks y bans | Solo competitivo, no se cruza con SoloQ |

Los atributos de campeón son fijos y no salen de resultados: no hay fuga temporal. Se usa la
versión "latest" (16.22); casi no cambian entre parches.

*Nota técnica:* en la notebook la ruta IPv6 hacia Cloudflare (donde están CommunityDragon y
Meraki) queda colgada y cada pedido de Python esperaba ~20 s antes de caer a IPv4: la
descarga de 257 campeones llevó más de una hora sin terminar. Forzando IPv4 en
`bajar_campeones_externos.py` tarda 35 s. Riot API no está afectada: la descarga de runas va
al techo de la key (48 partidas/min con una key de desarrollo).

**2. Composición del draft** (`ganador_composicion.py`). Por equipo, con los atributos de sus 5
campeones: cantidad de daño mágico $n_{AP} = \sum_i a_i$ ($a_i = 1$ mágico, $0{,}5$ mixto, $0$
físico), "todo AD" ($n_{AP} \le 0{,}5$), "todo AP" ($n_{AP} \ge 4$), desbalance
$|n_{AP} - 2{,}5|$; sumas de durabilidad, control, movilidad, utilidad y daño; cantidad de
frontline (TANK, VANGUARD, WARDEN, JUGGERNAUT), engage (VANGUARD, CATCHER, DIVER), poke
(ARTILLERY), enchanters, asesinos, tiradores y de rango; rango de ataque medio; e indicadores
"sin frontline", "sin engage", "sin control" y "poca durabilidad". Cuatro cruces con el rival,
antisimétricos (p. ej. $\text{poke}_{az}\cdot\text{sinEngage}_{ro} - \text{poke}_{ro}\cdot\text{sinEngage}_{az}$).
Entran como azul − rojo: 25 columnas.

*Por qué puede sumar algo sobre el one-hot:* la logística con one-hot por lado es aditiva en
los campeones, así que ya tiene la fuerza de cada uno. Lo que no puede representar son los
efectos de conjunto: "todo AD", "cero engage" o "poke contra un equipo sin engage" no son suma
de efectos por campeón. Por eso se espera poco: es solo la parte no aditiva del draft.

| Prueba (AUC en V) | Base | + composición |
| :--- | :---: | :---: |
| Solo draft: one-hot | 0,5461 | 0,5495 |
| Solo draft: one-hot + matchups + sinergias | 0,5496 | 0,5525 |
| Logística del modelo de 2 (con Poisson, lobby y líneas) | 0,6511 | 0,6518 |
| **Modelo de 2 expertos** | **0,6508** | **0,6517** (+0,0009 [+0,0004; +0,0014]) |

Bloques fuera de muestra de la logística: 0,6231 / 0,6407 / 0,6496 / 0,6496 contra
0,6211 / 0,6393 / 0,6483 / 0,6482: mejora en los 4. Solo con Meraki (sin CommunityDragon) el
modelo de 2 da 0,6515; CommunityDragon suma +0,0002 [−0,0001; +0,0007] sobre eso, dentro del
ruido. Los coeficientes más grandes (composición sola, estandarizados) son el rango de ataque
medio (−), el desbalance de daño (−), el daño mágico (+), la frontline (+) y el control (+).
"Todo AD" y "sin frontline" casi no pesan: son raros (0,3 % y 21 % de los equipos) o ya están
en otras columnas.

**Decisión del usuario (2026-11-14): la composición entra en la logística del modelo de 2.**
Es una mejora chica pero consistente (los 4 bloques y V), y es información disponible en el
momento de predecir: los campeones se conocen al cerrar la selección. Experto logístico vigente:
`ganador_logistica_oos.py --poisson data/analysis/ganador_poisson.pkl --extra
"data/analysis/ganador_mmr.pkl:nivel (foto)" "data/analysis/ganador_lineas.pkl:hgb"
"data/analysis/ganador_composicion.pkl:composicion" --guardar
data/analysis/ganador_logistica_composicion` (antes: `ganador_composicion.py`, ~40 s), predeterminado
en `ganador_dos_expertos.py`. Modelo de 2: meta $0{,}615\, z_{red} + 0{,}444\, z_{log}$; **V 0,6517,
acierto 60,9 %, 66,2 % en el 56 % más seguro**; calibrado (ECE 0,0085, pendiente 1,006).

*La composición en la red de grupos* (pedido del usuario: "la logística es una red de una
neurona"). Para que la red pueda aprender interacciones que la logística no ve, entra en dos
lugares (`ganador_red_jugador.py --composicion data/analysis/ganador_composicion.pkl`): los 15
atributos del campeón de cada jugador, estandarizados, se suman a su entrada de $arphi$ (de ahí a
la suma del equipo y a $g$, no lineal: puede armar "todo AD" sola), y las 25 columnas de
composición a la parte lineal por partida. Mismos argumentos que la red vigente (margen,
Poisson, grupos, sin huella, variante equipo, 3 semillas, 5 bloques).

| | Red sola en V | Por bloque (1-4) | Modelo de 2 (con la logística con composición) |
| :--- | :---: | :---: | :---: |
| Red de grupos vigente | 0,6459 | 0,630 / 0,643 / 0,649 / 0,645 | 0,6517, acierto 60,9 % |
| Red con composición | 0,6478 | 0,634 / 0,645 / 0,651 / 0,648 | 0,6519, acierto 60,6 %; +0,0002 [−0,0010; +0,0013] |

La red sola mejora en los 4 bloques (+0,002 en V; re-entrenar la red con otras semillas mueve
~0,0004, §9.S.36), pero en el modelo de 2 no suma: lo que gana es lo mismo que la logística ya
toma de la composición, y el meta le pasa peso a la red (0,71 contra 0,62) sin cambiar el AUC.
La red tampoco encontró interacciones de composición que las variables armadas no tuvieran.
**Decisión:** la red del modelo de 2 queda como estaba; `--composicion` queda en el script.

**3. Runas, hechizos y nivel de cuenta** (`runas_hechizos.py`, `ganador_runas.py`). La BD no
guarda esos campos del payload de match-v5, así que se vuelven a pedir las partidas (Riot
conserva el detalle ~2 años: las del train, de julio a septiembre de 2026, están). Una muestra
de 10.000 partidas al azar (semilla fija) del bloque 4 del ajuste y de V, en una base aparte
(`data/runas/runas.db`). Todo se elige en la selección de campeones: información previa a la
partida: las runas y hechizos se ven en la pantalla de carga, y el usuario las acepta como
información del momento de predecir (2026-11-14). Variables por jugador: nivel de cuenta (log y < 100), sin Flash, Flash en la F, Smite
fuera de la jungla, TP, Ignite, Exhaust, Ghost, Cleanse, y la rareza del combo de hechizos y de
la runa clave para ese campeón en esa línea (frecuencia suavizada, contada solo en la parte de
ajuste y sin la propia fila). Se mide sobre la predicción del modelo de 2: una logística chica
con [logit de la base, variables nuevas] se ajusta en la parte de la muestra del bloque 4 y se
evalúa en la de V, contra la misma logística solo con la base.

*Resultado* (10.000 partidas: 5.481 del bloque 4 para ajustar, 4.519 de V; base 0,6501 en esa
muestra de V):

| + variables | AUC | Diferencia | IC95 |
| :--- | :---: | :---: | :---: |
| Nivel de cuenta | 0,6502 | +0,0001 | [−0,0006; +0,0008] |
| Hechizos (incluye Flash en la F) | 0,6501 | +0,0000 | [−0,0023; +0,0022] |
| Rareza del combo de hechizos y de la runa clave | 0,6519 | +0,0017 | [−0,0009; +0,0043] |
| Todo | 0,6521 | +0,0019 | [−0,0017; +0,0055] |

* El **nivel de cuenta** no suma: en Diamante+ el 10 % de las cuentas tiene nivel < 100, pero
  lo que eso dice del jugador ya está en su margen previo y su cobertura.
* Los **hechizos** no suman. El Flash en la F lo usa el 40 % de los jugadores: es una costumbre,
  no algo que cambie la partida.
* La **rareza** es lo único con señal: quien lleva una runa clave rara para su campeón y línea
  pierde más (coeficiente −0,061, el más grande después de la base), igual que con un combo de
  hechizos raro (−0,038). El intervalo todavía incluye el 0: con 4.519 partidas de V el ancho es
  ±0,0026.
* **Siguiente paso posible:** completar las runas de toda V para llevar el intervalo a ±0,001
  (unas 22.400 partidas más, ~7 h 45 con una key de desarrollo, agregando al script una opción
  para bajar solo V), o el bloque 4 entero además (~49.000 más, ~17 h), que también estima mejor
  la frecuencia de cada runa por campeón. A decisión del usuario.

Nota operativa: la descarga se cortó una vez en 9.564 partidas porque Claude Code detuvo los
procesos en segundo plano por falta de memoria del sistema; se retomó con el mismo comando y
terminó sin pérdidas (cada partida se guarda al bajarla).


#### S.38. Quién gana: techo del margen con un oráculo de historial (2026-11-14)

Pedido del usuario: medir si queda información en el historial de los jugadores antes de gastar
horas de crawler en completarlo (camino (2) de §9.S.34). Script: `ganador_oraculo.py` (~4 min en
tres pasos con caché, 3 GB de RAM). Es una **cota superior tramposa a propósito**: no es un modelo
usable y no entra al modelo de 2.

*Diseño.* El mismo margen de §9.S.24 (15 estadísticas, residuo contra la media del campeón en la
línea, encogido $S/(N+k_j)$ con $k_j=5$, $k_{cl}=20$), el winrate encogido ($k=10$) y
$\log(1+N)$, calculados de seis formas, **todas sin la partida que se predice**. Con $c_i$ la suma
acumulada (incluida la fila $i$) del jugador y $T$ su total en el train:

| Modo | Suma que usa | Qué representa |
| :--- | :--- | :--- |
| previo | $c_i - x_i$ | lo que usa el modelo (todas las anteriores) |
| futuro | $T - c_i$ | todas las posteriores |
| oráculo | $T - x_i$ | todas las demás (leave-one-out) |
| previo_v20 / futuro_v20 | 20 anteriores / 20 posteriores | ventana cercana a un lado |
| cercano_v20 | 10 anteriores + 10 posteriores | ventana cercana centrada |

La media del campeón en la línea sigue la misma regla que el modo. Por partida, azul − rojo por
línea y total del equipo; la logística de siempre ($C=0{,}05$, estandarizada). Ajuste 04/09–16/09
(29.049 partidas), V desde el 16/09 (27.210). En V cada jugador tiene en promedio 124 partidas
previas y 17 posteriores dentro del train. El modelo es liviano (sin campeones, matchups, lobby ni
red): sirve para comparar modos entre sí, no contra el 0,6517 del modelo de 2.

*Resultados* (AUC en V; IC 95 % bootstrap pareado de la diferencia contra "previo"):

| Modo | Completo | Sin conteo ($\log N$) | Solo margen |
| :--- | :---: | :---: | :---: |
| previo | 0,5913 | 0,5917 | 0,5908 |
| oráculo | +0,0009 [−0,0035; +0,0047] | +0,0001 | −0,0002 [−0,0035; +0,0030] |
| previo_v20 | −0,0035 | −0,0035 | −0,0025 |
| cercano_v20 | +0,0126 | −0,0082 | −0,0072 |
| futuro | +0,0170 [+0,0102; +0,0255] | −0,0365 | −0,0368 |
| futuro_v20 | +0,0241 | −0,0326 | −0,0327 |

Por cobertura del lobby (solo margen, oráculo − previo): 0 jugadores con < 10 previas −0,0014
[−0,0042; +0,0019] (n = 17.099); 1–2: +0,0027 [−0,0053; +0,0092]; 3+: +0,0013 [−0,0085; +0,0116]
(n = 4.010).

*Lectura.*

1. **El historial está saturado.** Conocer *todas* las demás partidas del jugador, incluidas las
   futuras (~140 en vez de ~124), no mueve el AUC (+0,0001 sin conteo). Tampoco en los lobbies con
   3 o más jugadores con poca historia, que eran el punto débil de §9.S.34. Completar el historial
   con el crawler (~30–60 h) no tiene de dónde ganar, al menos por la vía del rendimiento previo.
2. **Lo que "gana" el futuro es leakage de conteo, no fuerza.** Con $\log N$ adentro, los modos
   futuros suben hasta +0,024; sin él caen 0,033–0,037 (tienen menos partidas). La cantidad de
   partidas que el jugador juega *después* depende del resultado de esta: quien gana sigue
   jugando, quien pierde corta (o el crawler lo sigue menos). Es el mismo fenómeno que el
   **ritmo de juego** de §9.S.30, visto desde el otro lado, y una advertencia para cualquier
   variable contada con todo el período (los agregados de `players.csv`, §4).
3. **La ventana reciente no le gana al acumulado** a ningún lado: el margen es un rasgo estable,
   coherente con la deriva lenta de §10.

*Conclusión.* Con este dataset, la información del jugador antes de la partida está agotada: lo
que falta para pasar de ~61 % es variación propia de cada partida, no historial. Lo que queda por
el lado de la información previa es lo que todavía no está en la base (runas, hechizos y nivel de
cuenta, §9.S.37). Para el informe, el oráculo da el argumento de techo: ni siquiera viendo el
futuro del jugador el margen mejora.

*Alternativas descartadas.* Correr el oráculo dentro del modelo de 2 completo: necesita los
pickles de ~3 GB y la red en GPU; como la comparación es entre modos de la misma variable, el
modelo liviano alcanza para la pregunta. Un rating suavizado hacia atrás y adelante (TrueSkill
*smoothing*): misma pregunta con más supuestos; el LOO es su versión sin modelo.


#### S.39. Quién gana: destilación con los snapshots y revisión de los componentes (2026-11-14)

Pedido del usuario: probar la destilación con información privilegiada que propusieron los agentes
de investigación y revisar si los componentes ya armados (huella, margen, Poisson) tienen margen.

*Destilación.* Un **maestro** ve lo mismo que el alumno más las diferencias azul − rojo por línea de
los snapshots @10/@15 (oro, xp, cs, daño, kills, muertes); el **alumno** solo ve lo pre-partida y se
entrena con la etiqueta suave $q = a\,y + (1-a)\,p_{maestro}$. Para no filtrar el resultado, cada
ajuste del alumno (bloques 1-4 de `--oos 5` y el ajuste completo para V) recibe predicciones del
maestro hechas con 3 pliegues por fecha *dentro de ese mismo ajuste*. Idea: la victoria es una
etiqueta muy ruidosa y el maestro dice "esta partida venía favorecida" con menos ruido (en CTR de
Taobao, *Privileged Features Distillation*, KDD 2020, sumó +0,012 de AUC). Prueba liviana (onehot de
campeones, experiencia, cobertura, margen y winrate previos; 407 columnas), mismo orden de partidas
y mismos bloques que el modelo de 2 (verificado contra sus `_oos.npz`):

| | $a = 1$ (sin maestro) | $a = 0{,}5$ | $a = 0{,}25$ | $a = 0$ |
| :--- | :---: | :---: | :---: | :---: |
| Maestro, AUC fuera de pliegue | | 0,837 | | |
| Logística alumno, V | **0,6346** | 0,6346 | 0,6345 | 0,6345 |
| LightGBM alumno, V | 0,6274 | 0,6296 | **0,6303** | |

(En la logística los 4 bloques dan lo mismo: diferencias ≤ 0,0012 en el bloque 1 y ≤ 0,0001 en el
resto. LightGBM: `cross_entropy`, parada temprana con la etiqueta real.)

*Lectura.* En un modelo lineal la destilación no cambia nada: con 160 mil partidas la logística ya
estima bien sus coeficientes y el maestro le enseña la misma dirección. En un modelo flexible sí ayuda
(+0,003 a LightGBM), porque el ruido de la etiqueta es lo que lo hace sobreajustar, pero no alcanza a
la logística. El candidato natural es la **red de grupos**, el experto más fuerte del modelo de 2:
`ganador_red_jugador.py` ahora acepta `--maestro data/analysis/ganador_maestro.npz --alfa 0.5` (solo
config C). Pendiente correrla en la PC con GPU.

*Agregados del equipo.* Máximo, mínimo y desvío dentro del equipo (azul − rojo) de 8 variables previas
del jugador (margen de oro @15, oro, daño, muertes y participación acumulados, winrate, winrate con el
campeón, partidas con el campeón), contra la suma por línea que ya usa el modelo. Logística liviana:
el **máximo** suma **+0,0011 [+0,0004; +0,0018]**; mínimo +0,0003 (no significativo); desvío +0,0009;
los tres juntos +0,0010. Pesa más "el que carga" que "el eslabón débil" (en la línea de
arXiv 2106.11397). Pendiente medirlo en la logística del modelo de 2 con `--extra
data/analysis/ganador_agregados.pkl:máx`.

*Revisión de los componentes.*

1. **Huella.** Mide estilo, no fuerza: se residualiza *por resultado* a propósito (§10), así que por
   construcción no sabe quién gana. Ya la reemplazó el margen (§9.S.24) y el encaje de estilos no sumó
   (§9.S.30). Para quién gana no tiene margen; queda para reidentificación y cuentas compartidas.
2. **Margen.** El oráculo (§9.S.38) muestra que ni más historial ni una ventana reciente lo mejoran;
   tocar el encogimiento ($k_j$ por estadística según su ICC) mueve el mismo número de partidas y se
   espera ≈ 0. El único hueco medible fue la agregación (arriba).
3. **Poisson.** Los objetivos de oro y daño son por minuto de *toda* la partida: el equipo que gana
   termina con más oro, así que el "ataque" en oro mezcla fuerza con resultado y se parece mucho al
   winrate. Una variante más limpia sería ajustar ataque y defensa sobre el **oro @15 por línea**
   (antes de que la partida se decida), pero el margen ya incluye `d_gold_15` por línea y el Poisson
   entero sumó +0,002 sobre el margen: se espera ≤ +0,001. Los parámetros son constantes dentro de
   cada uno de los 40 bloques; pasarlos a un filtro de Kalman (fuerza que deriva) choca con lo mismo
   que el oráculo: no falta historia.

*Alternativas descartadas.* Destilar en la logística con más peso al maestro ($a < 0{,}25$): la tabla
ya muestra que no cambia. Maestro con LightGBM: la logística con snapshots ya llega a 0,837 y es
reproducible en segundos. Multitarea (predecir la diferencia de oro como tarea auxiliar) en la
logística: es lineal en las mismas variables, equivale a cambiar pesos; tiene sentido solo dentro de
la red, junto con el maestro.

Archivos: `historial/ganador_primeros/ganador_maestro.py` (escribe `ganador_maestro.npz` y
`ganador_agregados.pkl`, ~2 min), `--maestro/--alfa` en `ganador_red_jugador.py`. Los dos archivos de
`data/analysis/` se generaron con el mismo código fuera de la PC (pandas 3; el único empate de
`game_creation` que ordena distinto se corrige y se verifica contra los `_oos.npz`).

#### S.40. Mecánica del LP medida en vivo y diseño del mapa de elo (2026-11-02)

**Pregunta del usuario.** Para un mapa del elo de todos los jugadores de LAS, partida a partida:
¿conviene un modelo entrenado con el seguimiento en vivo, o usarlo solo para validar el principal?
¿Cómo se tratan las partidas que no están en el dataset?

**Decisión propuesta, pendiente de que la apruebe el usuario.** El en vivo entrena un **modelo del
paso** (cuánto LP mueve una partida). El principal se queda como generador de anclas donde falta una
foto. El mapa se infiere por tramos, con un suavizado que nunca invierte la dinámica.

**Datos y cuidados.** `lp_vivo.db` (copia de Drive) del 30/10 14:00 al 01/11 23:10 UTC, con un corte
de 28,8 h: 128.557 eventos enlazados en 22.778 partidas, 22.720 de ellos en LAS. Todo se midió sobre
los 22.626 jugadores de desarrollo: la caja cerrada no se abrió. Antes de la foto 1 solo se contaron
filas, sin leer resultados. Los scripts, exploratorios y sin tests, quedan en
`historial/mecanica_lp/`, porque los del 29/10 se perdieron en un temporal.

**El ΔLP no es una regla con ruido: es casi determinista por partes.** En LAS:

$$\Delta z = \begin{cases} m\,(K + \delta_i) & \text{victoria, con } m = 2 \text{ si hay Aegis of Valor y } m = 1 \text{ si no} \\ -(K - \delta_i) & \text{derrota; } 0 \text{ si hay devolución} \end{cases}$$

| Pieza | Medido en LAS | Lo que dice Riot | ¿Se ve en el dataset? |
| :--- | :--- | :--- | :--- |
| $K = 20$ | Cuartiles +19/+21 y −21/−19. En BR, NA y EUW, ~30 | ±30 solo en las regiones reiniciadas (abajo) | Constante |
| $\delta_i$, propio del jugador | Ganancia + \|pérdida\| ≈ 40; corr(δ al ganar, δ al perder) = 0,92; ICC 0,83; autocorrelación 0,84 a 1 y a 5 partidas; de −5 a +10 (+30/−10 es el tope) | Con MMR sobre el rango, "greater LP rewards upon victory and smaller deductions upon defeat" | Por el gap con el lobby |
| Victoria doble | 6,7 % de las victorias; 64 % en un rol raro (< 5 % de sus partidas) de TOP, JG o SUP; ~0 en MID | Aegis of Valor: "doubles the LP you gain upon winning" | Rol y rareza: AUC 0,87 |
| Derrota en 0, por equipo | 4,5 % de los equipos perdedores; cae en 4 de 5 jugadores; 92 % en partidas de menos de 15 min | Consolation LP: AFK, abandono o "intentional feeding" verificados, fuera de la premade | Solo duración y rendimiento: AUC 0,73 |
| Pisos | En Master 0, el 63 % se queda; si baja, cae a Diamond I con 75, 50 o 25 LP | 3 partidas de protección; "25, 50, or 75 LP" | Por el estado |
| Decay | Es un banco de días. De 322 que decayeron, el 61 % había jugado hace menos de 14 días; banco ≤ 1 en el 97,5 % (control: 38 %) | Master+: 1 día por partida, tope 14, −75 por día. Diamond: 7 por partida, tope 28, −50 por día | Por las horas de juego |

El anuncio de Aegis promete también protección al perder, pero en LAS no se ve: el 0 individual es
del 0,1–0,5 %, sin relación con la rareza del rol. Reemplaza a la regla "75 LP pasados 14 días sin
jugar" de §9.S.21.

Varianza por paso: 37,9 LP² con la regla, 34,6 con el rol, 30,0 con el δ propio y **0,87 en las
partidas normales con el δ propio**. Casi todo lo que queda son dobles y devoluciones sin identificar.

$\delta$ contra el gap (media del lobby − LP propio), por jugador con ≥ 8 partidas. El lobby usa el
LP medido o, si no hay, la foto 2 (97,6 % de los jugadores, contra 60,6 % de antes):

| Gap | < −200 | −200 a −100 | −100 a 0 | 0 a 100 | 100 a 200 | 200 a 400 | 400 a 800 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| δ medio (LP) | −1,2 | −0,9 | −0,3 | +0,5 | +2,7 | +5,1 | +9,6 |
| Jugadores | 125 | 186 | 254 | 159 | 45 | 15 | 7 |

Es asimétrico: quien está por debajo de su MMR sube rápido, y quien está por encima casi no baja por
esta vía. Con gap, lobby, LP, winrate y partidas, δ medio tiene R² 0,55 fuera de fold. Explica los
escaladores que §9.S.21 no podía explicar: 45 V y 58 D a +30/−10 dan +770, y el caso real subió 709.

**LAS es un régimen aparte.** El 29/04/2026 (parche 26.9) Riot llevó a Master 0 LP el rango y el MMR
de Master+ en NA, EUW, EUNE, BR, LAN y TR, y subió la ganancia base a ±30, con un rango de +35/−25 a
+25/−35. LAS no figura en ninguna lista. Riot dice que ninguna otra región tuvo ese nivel de
inestabilidad y que "for all other players, nothing will change". Que LAS no se reinició es una
inferencia, apoyada en el ±20 medido. lolpros lo confirma del otro lado: caídas de hasta 2.600 LP el
30/04 en cuentas de EUW, con las W/L intactas. Consecuencias:

* BR, NA y EUW sirven para medir Aegis y devoluciones, no $K$ ni $\delta$ de LAS. Revisa la decisión
  del 30/10 de validar con los 4 servidores.
* lolpros mide el método, no la dinámica de LAS, y solo después del 30/04.
* **Inicio de la temporada:** en lolpros las W/L se reinician entre el 08/01 06:31 y el 09/01 06:00
  UTC y no vuelven a reiniciarse hasta el 30/10 (8 cuentas). `INICIO_SPLIT` = 08/01 queda bien.
* **Arranque:** Riot pide 5 partidas de colocación, con techo en Diamond III, y una derrota de
  colocación da 0 LP. En lolpros, tras 5–8 partidas, $z$ = 18–52.

**Prueba hacia adelante: de la foto 1 a la foto 2.** Solo W/L observadas. δ, dobles y devoluciones
salen solo del en vivo, que es posterior y nunca vio estas fotos. Lo único que no sale de ahí es el
decay de Riot y $K$ de Diamond (18,75 por L1 sobre fotos; entre 16,5 y 20 no cambia nada).

| Modelo (22.626 jugadores) | MAE | RMSE | R² del cambio | ρ dentro del rango |
| :--- | ---: | ---: | ---: | ---: |
| No cambió | 145,5 | 224,4 | −0,007 | 0,604 |
| Regla $a, b$, ajustada a estas mismas fotos | 111,2 | 175,8 | 0,382 | 0,728 |
| GB sin mecánica, 5 folds | 63,2 | 92,8 | 0,828 | 0,858 |
| **Mecánica medida en vivo, sin etiquetas** | **64,7** | **87,7** | **0,846** | 0,817 |
| **Mecánica + GB, 5 folds** (promedio L1 y L2) | **48,7** | **68,3** | **0,907** | **0,889** |
| *Objetivo del usuario (planteado hacia atrás)* | *~50* | *~70* | *~0,85* | *~0,9* |

δ y el decay hacen casi todo: sin δ, R² 0,62 (MAE 92); sin decay, R² 0,54 (MAE 90). El banco contra
"14 días sin jugar": RMSE 87,7 contra 101,7. Dobles y devoluciones casi no mueven el total, pero en
Challenger y Grandmaster sacarlos lleva de 98 a 131 y 113 LP de MAE. Queda un sesgo de −13 LP, sin
calibrar.

**Prueba hacia atrás: predecir la foto 1 con la foto 2 como única ancla.**

| Modelo | MAE | RMSE | R² del cambio | ρ dentro del rango |
| :--- | ---: | ---: | ---: | ---: |
| Regla | 111,2 | 175,8 | 0,382 | 0,733 |
| Inversión determinista de la mecánica | 402,6 | 753,5 | −10,4 | 0,493 |
| Suavizado en grilla con la mecánica (prototipo; 6.000 jugadores) | 112,7 | 166,9 | 0,474 | 0,769 |
| **Principal** | **71,5** | **115,6** | **0,733** | **0,880** |
| Principal + mecánica como variables, 5 folds | 71,3 | 116,2 | 0,730 | 0,880 |

**Por qué la asimetría.** La dinámica contrae: 300 LP de error de arranque quedan en 136 tras ~94
partidas. Hacia adelante eso corrige. Hacia atrás destruye información, porque varios $z_1$ llevan al
mismo $z_2$, igual que el piso del decay. Es la misma razón por la que diverge el Elo al revés. El
principal ya aprovecha lo que dicen una foto y el lobby, y la mecánica no le agrega.

**Partidas que faltan.**

| Período | Partidas en el dataset | Participaciones ocultas | Cobertura mediana |
| :--- | ---: | ---: | :--- |
| Entre fotos | 134.695 | 468.617 (Diamond 384.314) | C 97,5 %, GM 97,2 %, M 96,1 %, D 50,0 % |
| Antes de la foto 1 | 137.826 (69 % en julio y agosto) | 9.505.078 (Diamond 7.080.183, Master 2.239.328) | C 57 %, GM 43 %, M 26 %, D 2,7 % |

Prueba de la punta, ocultando partidas al azar a 3.415 jugadores de Master+ casi completos (94
partidas en la mediana). MAE; las ocultas entran por su paso esperado:

| Cobertura | Adelante, con saldo W/L conocido | Adelante, sin saldo | Atrás, GB con lobby |
| :--- | ---: | ---: | ---: |
| 100 % | 56,0 | 59,6 | 100,4 |
| 75 % | 62,6 | 89,9 | 118,7 |
| 50 % | 72,9 | 114,1 | 125,3 |
| 25 % | 98,2 | 147,4 | 134,7 |
| 10 % | 165,8 | 211,3 | 149,7 |
| 5 % | 239,3 | 272,5 | 155,1 |
| 3 % | 289,2 | 313,5 | 161,2 |

Las fotos dan el total de W y L, así que **el saldo de las ocultas se conoce y solo falta su orden**.
Por debajo de ~25 % la mecánica se rompe: el lobby de pocas partidas es ruidoso y δ(gap) es convexa,
así que el ruido empuja para arriba, y las ocultas entran en bloque. Falta encoger el lobby según su
incertidumbre. El modelo aprendido se degrada suave.

Prueba del camino, en 319 tramos en vivo (14 partidas en la mediana), con anclas en las puntas. Error
en las partidas vistas: 4,8 LP con todo a la vista, 11,1 al 75 %, 17,2 al 50 %, 21,4 al 25 %, 23,9 al
10 % y 25,2 al 5 %. Una recta por partidas erra 27,7. Con $q$ la varianza por partida, un puente
browniano da $\mathbb{E}|e| \approx 0{,}31\sqrt{qn}$, con $q \approx 30$ LP² si la partida se ve y
$q = (2K)^2 p(1-p) = 400$ si no: calza 6,4 (medido 4,8) y 23,4 (medido 25,2). Extrapolado a 100
partidas, 17 LP con todas vistas y 62 con todas ocultas. Una partida oculta aporta 13 veces más
varianza que una vista.

Decisión: **bajar lo que rinde, a una base aparte y sin timeline**, para no tocar `lol_las.db`. Entre
fotos son ~56.000 partidas (468.617 / 8,43 jugadores del padrón por partida); en julio y agosto,
~240.000, suponiendo partidas parejas en el tiempo. Con el listado, ~1 y ~2 días con las dos keys, a
68.400 pedidos por key y por día. Listar los IDs primero da la posición exacta de las ocultas y
cuántos jugadores resuelve cada partida, para bajar primero las que más resuelven. Antes de julio
(31 % de esas filas) no se baja: se usa lo que hay, con menos confianza.

**Diseño del mapa.**

| Tramo | Filas | Anclas | Método | Error esperado |
| :--- | ---: | :--- | :--- | :--- |
| Entre fotos, con las dos | 1.105.805 | Exactas en las dos puntas | Suavizado con el modelo del paso; el cierre fija el δ medio del jugador | ~20 LP (extrapolado) |
| Entre fotos, solo foto 1 | 29.226 | Una, al inicio | Adelante + GB | ~49 en la punta |
| Entre fotos, solo foto 2 | 211.364 | Una, al final | El principal da la foto 1 que falta; después, como el primero | ~72 en la punta; debajo de Diamond, sin medir |
| Antes de la foto 1, últimos 54 días | 972.250 | Foto 1 exacta | El principal da el LP 54 días antes; el modelo del paso une ese punto con la foto | Sin medir: ~40 LP si el principal rinde igual (estimado) |
| Antes de esos 54 días | 406.010 | Ese punto + arranque de temporada (W = L = 0, techo Diamond III) | Adelante desde enero, anclado en ese punto | Sin medir; confianza baja |
| Sin ninguna foto | 3.330 | — | Lobby | Alto |

* **Modelo del paso:** $K$, $\delta(\text{gap})$, P(doble | rol, rareza, LP), P(devolución | duración,
  equipo), pisos y banco de días, ajustados con LAS. Su métrica es el error acumulado a $n$ partidas y
  el cierre entre fotos, nunca el R² por partida.
* **Inferencia:** grilla de LP, como el filtro del 29/10, pero con el núcleo hacia adelante y el mensaje
  hacia atrás, sin invertir: $\beta_t(z) = \sum_c \pi_c(z)\,\beta_{t+1}(z + \Delta_c(z))$. Dos pasadas,
  porque el lobby depende del LP de los otros.
* **Antes de la foto 1 manda la cercanía.** El 70,5 % de esas partidas está a menos de 54 días de la
  foto 1 y a más de 6 meses de enero, y 54 días es justo lo que el principal aprendió a retroceder. Ahí
  fija un punto y el modelo del paso dibuja el camino. La recorrida desde enero queda para lo anterior:
  la contracción olvida el arranque, y Riot dice que el modo de recuperación "will almost always happen
  at the start of each new ranked season". La primera versión de este diseño iba desde enero en todo el
  tramo; se corrigió porque desaprovechaba el ancla más cercana.
* **Salida por fila:** `z_media`, `z_p10`, `z_p90`, `metodo`. Hay una versión descriptiva, que mira el
  futuro, y otra causal, solo con el pasado, para el modelo de draft: esta última no puede usar la foto
  1 antes del 04/09.

**El principal hay que volver a entrenarlo.** `elo_principal_pred.csv` solo tiene el LP del 04/09 de los
26.534 jugadores con las dos fotos, justo los que no lo necesitan, y los modelos no se guardaron. Los
scripts del 29/10, que se daban por perdidos, estaban en el temporal de la PC de escritorio y quedan en
`historial/elo_principal_oct/`, con la receta y el resultado en la caja (`bitacora.md`; segunda
apertura: 69,5 / 107,6 / 0,759 / 0,882). Falta parametrizar el ancla y la ventana, hoy fijas en `prep.py`.
Aprendió un solo horizonte (54 días) con cobertura de 50–97 %: para usarlo antes de la foto 1 hay que
bajar esas partidas o reentrenarlo enmascarando. Su error casi no depende de las partidas recorridas: MAE
67 con hasta 5 partidas y 74 con más de 200.

**Alternativas descartadas.**

* *Principal + `camino` (lo de hoy):* solo tiene dato en una punta, reparte la corrección parejo y no
  sabe de decay, pisos ni del tope de δ. Se entrenó con 54 días y cobertura de 50–97 %; antes de la
  foto 1 son 240 días y 3–57 %. Se queda donde falta un ancla.
* *Modelo de ΔLP entrenado en vivo y recorrido solo:* hacia atrás explota (MAE 403).
* *Que el principal prediga el LP tras cada partida (tarea abierta):* el paso casi no necesita aprendizaje
  profundo, y el R² por partida no permite elegir.
* *Mecánica como variable del principal:* 71,3 contra 71,5 de MAE y peor RMSE. No aporta.
* *Entrenar con los 4 servidores:* otro régimen de LP.
* *GRU por partida con pérdida en el cierre y en los ΔLP en vivo:* no se probó. Queda para una segunda
  etapa, si δ(gap) se queda corta: el 28/10 las arquitecturas no aportaron y el paso es casi determinista.

**Validación.** La caja cerrada ya se usó en la selección del principal, así que la prueba limpia es
una **foto 3** a 2–3 semanas de la foto 2 (~177 pedidos), con las partidas del período. Nadie vio ese
tramo, y el en vivo da LP real en cada partida de Master+: es la primera medición del error en el medio
del camino. Antes del 04/09 no hay LP real salvo la foto 1, así que el test es una sola corrida desde
enero sin esa foto, comparada al final; un padrón del compañero de otra fecha serviría de control. Para
probar al principal en su horizonte hace falta una foto a 54 días de la foto 2 (~21/12); una a 2–3 semanas
lo prueba a menor distancia.

**Límites.**

* Son 2,4 días. Los nodos altos de δ(gap) salen de 22 jugadores.
* Diamond (78 % del padrón) no se mide en vivo: δ se extrapola, y agregado funciona (sin δ en Diamond,
  MAE 80 contra 64), pero $K$ sale de las fotos. Debajo de Diamond no hay nada medido.
* La transferencia a antes de septiembre no está probada. Riot admite "a number of mistakes" en la
  configuración de inicio de 2026, las regiones reiniciadas alternaron +10/−30 y ±20 según el parche, y
  la fecha de inicio de Aegis no está confirmada.
* Falta probar el suavizado en tramos largos. En 14 partidas la mecánica con cierre casi no le gana a la
  regla con cierre (4,8 contra 5,2): su valor esperado está en los tramos largos, los escaladores y el
  decay, y sigue sin medirse.
* El 89 % de los jugadores con decay muestra un +75 de rebote: `lp_vivo` no filtra réplicas atrasadas
  con el mismo W + L. Hay que limpiarlo antes de calibrar decay.

**Fuentes (Riot).** [Reinicio de Master+](https://www.leagueoflegends.com/en-us/news/dev/dev-apex-tier-ranked-reset/),
[notas del parche 26.9](https://www.leagueoflegends.com/en-us/news/game-updates/league-of-legends-patch-26-9-notes/),
[cambios de 2026](https://www.leagueoflegends.com/en-us/news/dev/dev-ranked-2026),
[MMR, rango y LP](https://support.riotgames.com/en-us/league-of-legends/gameplay/mmr-rank-and-lp),
[colocación y decay](https://support.riotgames.com/en-us/league-of-legends/gameplay/placements-promotions-series-demotions-and-decay),
[autofill](https://support.riotgames.com/en-us/league-of-legends/gameplay/matchmaking-and-autofill),
[Master+](https://support.riotgames.com/league-of-legends/gameplay/master-grandmaster-and-challenger-the-apex-tiers).

#### S.41. Embeddings y clustering de partidas (2026-11-14)

**Pedido del usuario.** Embeddings de las partidas con toda la información disponible (historial
reciente de los jugadores, campeones, etc.) y clustering con K-Means y C-means, para ver si
aparece algo interesante. `scripts/analysis/embeddings_partidas.py` (~4 min con la caché de
variables; `--rehacer` la reconstruye en ~1 min más; `--grupo todo|historial|campeon`). Solo
`data/splits/train`; el test no se toca. Clusters, autoencoder y escalas se ajustan con las
160.883 partidas anteriores a V (la mitad posterior a la foto del 04/09, la misma de siempre);
V (26.934 partidas) solo se asigna y se mide. Se excluyen las partidas sin biyección de líneas
y las que no tienen ganador.

**Representación.** Por jugador y partida, 85 variables, todas con partidas **anteriores**:

* *Historial (45):* margen previo acumulado y de las últimas 20 sobre las 15 estadísticas de
  §9.S.24 (`prior_margin`), winrate previo acumulado y de 20, experiencia (partidas totales, con
  el campeón, estreno, porción del campeón y de la línea, fuera de rol, winrate con el campeón
  encogido), cambio de pool (`pool_change`), ritmo (`play_rate`) y $\log(1 + \text{horas desde la
  partida anterior})$.
* *Campeón (40):* tags, `info` y 7 estadísticas base de ddragon 16.17.1, y el perfil de estilo
  típico del campeón: media de los $z$ por línea de las 23 variables de `STYLE_FEATURES`,
  encogida $\bar z = \sum z / (n + 30)$ y calculada solo con las partidas de ajuste.

Equipo = los 5 jugadores en orden de línea (425 números). Un **autoencoder** denoising
(425 → 512 → 128 → 16 → simétrico, dropout 0,1 en la entrada, MSE, 30 épocas en GPU) da el
embedding de equipo $e \in \mathbb{R}^{16}$. Partida antes de jugarse, simétrica (no depende
de qué equipo es azul): $\big[(e_A + e_R)/2,\ |e_A - e_R|\big]$. Partida por su
**desarrollo** (descriptiva, posterior): duración, ritmo de kills, daño y visión totales por
minuto, $|\Delta\text{oro}|$ a 10/15/20 y final, $|\Delta\text{kills}|$ a 15 y final, remontada
(el que iba adelante en oro a los 15 perdió) y rendición antes del 20.

**Clustering.** K-Means (MiniBatch, $k = 2 \ldots 10$, silhouette en 20.000 y BIC de un GMM
diagonal), fuzzy C-means (Bezdek, implementado en el script; pertenencia
$u_{ik} = 1 / \sum_j (d_{ik}/d_{ij})^{2/(m-1)}$) y PCA de 16 componentes como referencia lineal.

**Resultados.**

| Embedding | R² reconstr. (V) | Silhouette k=2 / k=6 | Solo el embedding, AUC quién gana | Suma a la base |
| :--- | :--- | :--- | :--- | :--- |
| Todo (85) | 0,646 | 0,099 / 0,080 | 0,591 | −0,0001 [−0,0005; +0,0003] |
| Historial (45) | 0,595 | 0,274 / 0,109 | 0,599 | **+0,0011 [+0,0004; +0,0017]** |
| Campeón (40) | 0,985 | 0,188 / 0,088 | 0,520 | +0,0003 [−0,0001; +0,0007] |

Base: logística con las 85 variables azul − rojo por línea, AUC 0,6397 en V (más débil que el
modelo de 2 de §9.S.37, 0,6517, porque no tiene Poisson, lobby ni modelo por línea).

1. **No hay tipos de equipo naturales.** Silhouette ≤ 0,27 en todas las variantes (≥ 0,5 sería
   estructura clara) y el BIC del GMM baja hasta $k = 10$ sin codo: una nube continua. Con todas
   las variables, el embedding queda dominado por la composición (lo que mejor se reconstruye es
   el campeón, R² 0,985). El corte en $k = 2$ del historial (81 % / 19 %) separa equipos con
   medias casi iguales (partidas totales 3,61 contra 3,80 en log, estrenos 26 % contra 19 %;
   ganan 0,499 y 0,504).
2. **El C-means no encuentra pertenencias difusas con estructura.** Con $m = 2$ y $m = 1{,}5$ las
   pertenencias colapsan a $1/k$ (coeficiente de partición 0,167 = sin estructura; en 16
   dimensiones sin grupos, el C-means converge al centroide). Con $m = 1{,}2$ o $1{,}1$ se
   vuelve un K-Means duro y las pertenencias altas son automáticas. Lo que dice algo es el
   acuerdo con K-Means sobre los mismos datos: NMI 0,45-0,71: dos particiones razonables de la
   misma nube no coinciden.
3. **Los tipos de equipo descriptivos tienen sentido, pero no cambian quién gana.** Con $k = 6$
   en el historial: equipos de cuentas nuevas y estrenos (20 %, ganan 0,493), equipos de
   especialistas con pool concentrado (12 %, 0,517; 0,522 en V), equipos de soportes con
   muchas asistencias y buen winrate reciente (12 %). El cruce de tipos (36 celdas,
   antisimétrico) no suma (−0,0001): la ventaja de los especialistas ya está en la experiencia
   con el campeón de la base.
4. **El desarrollo sí tiene tipos (silhouette 0,25 en $k = 2$), y lo previo no los anticipa.**
   Cinco tipos: ventaja moderada que se cierra a los 27 min (26 %), **remontadas** (26 %: el
   que iba adelante a los 15 perdió), partidas parejas y largas (23 %, 33 min), aplastadas
   (18 %, 6.000 de oro a los 15) y rendiciones antes del 20 (7 %). El reparto de esos tipos es
   el mismo en todos los tipos previos: NMI 0,0002-0,0012, y la tasa de remontada a los 15 queda
   entre 27 y 29 % en todos los grupos. Antes de la partida no se puede saber si va a ser una
   aplastada o una remontada, más allá de quién es favorito.
5. **El único aporte medible es chico y no es del clustering.** El embedding del historial suma
   +0,0011 sobre la base: es una no linealidad sobre variables que la logística ya tiene, y es
   menor que lo que agregaron Poisson y el lobby al modelo de 2. Ni los tipos de K-Means ni las
   pertenencias del C-means suman (±0,0001). La base acierta algo mejor en los lobbies de más
   tier (tipo previo con tier medio 0,80: AUC 0,654 contra 0,625-0,647 en el resto).

**Decisión.** No se agregan tipos de partida ni de equipo al modelo de quién gana. El embedding
del historial queda como candidato menor para el stacking (medirlo en `ganador_dos_expertos.py`
antes de incluirlo). Los tipos de desarrollo sirven para el informe: muestran el techo (una de
cada cuatro partidas se da vuelta después del minuto 15, y lo previo no la anticipa).

*Alternativas descartadas.* (a) K-Means sobre las 850 variables crudas de la partida: con
silhouette de 0,04-0,07 en PCA de 16 (que ya captura lo lineal) no hay más estructura que la
del autoencoder. (b) Embeddings supervisados (una red entrenada para predecir quién gana): ya
es la red de grupos del modelo de 2; los clusters sobre ese espacio serían niveles de
probabilidad, no tipos. (c) Elegir $k$ por silhouette: da $k = 2$ en todos los casos, sin
contenido; los perfiles se describen con $k = 6$ (equipos y partida previa) y $k = 5$
(desarrollo), y la conclusión no cambia con otro $k$ porque el cruce y las pertenencias se
miden contra V.

Archivos: `scripts/analysis/embeddings_partidas.py`; salidas en
`data/analysis/embeddings_partidas/` (`variables.npz` = caché, `embeddings.npz`, `resumen.json`,
`log.txt`, `tsne_equipos.png`, `matriz_tipos.png`; subcarpetas `historial/` y `campeon/`).

#### S.42. EDA de los modelos de victoria: draft contra jugadores, el desarrollo y qué es significativo (2026-11-14)

**Pedido del usuario.** Un EDA general de todo lo que se hizo para predecir victorias, desde el
primer baseline, en el formato del EDA del LP (`eda_entre_fotos`). El resultado es
`notebooks/eda_modelos_ganador.ipynb`, que genera `scripts/analysis/build_eda_modelos_nb.py
--ejecutar` (~8 min). Se ordena alrededor de la pregunta central de §1 y §7.A (¿draft o
jugadores?) y de la segunda que el proyecto planteó desde el principio (¿cuánto se decide
durante la partida?). Solo se usa el train, con la partición F/V de §9.S.23: lo que se calcula en
vivo se ajusta con las partidas anteriores a V y se mide en V, con IC por bootstrap pareado
(B = 400).

**Fuentes.**

* *En vivo:* la caché de §9.S.41 (85 variables previas por jugador, campeones y estado a
  10/15/20).
* *Registro:* las 76 variantes de §9.R a §9.S.41, copiadas a `scripts/analysis/ganador_registro.tsv`
  (versionado; `*.csv` está en `.gitignore`). Cada una tiene su referencia, ΔAUC, IC si existe,
  decisión y el conjunto donde se midió. Las de octubre se midieron en todo el dataset o en el
  test, antes de fijar la regla de no tocarlo: entran al resumen, pero no a las pruebas.

*Alternativa descartada:* recalcular las variantes de §9.S.24-§9.S.39. Sus resultados intermedios
están en otra máquina y rehacerlos lleva horas de GPU. El registro es auditable contra este
documento.

**1. Draft contra jugadores (en vivo, misma logística $C = 0{,}05$, azul − rojo por línea).**
Bloques: *Campeones* = campeón × línea con signo (Bradley-Terry, §9.S.20) + atributos de Data
Dragon + perfil de estilo típico del campeón. *Jugadores* = margen previo, winrate, partidas,
línea, rol, pool, ritmo y descanso. *Jugador × campeón* = partidas con el campeón, estreno,
porción y winrate con él (la interacción que §7.A llamaba híbrido).

| Modelo | AUC en V [IC 95 %] | Acierto |
| :--- | :---: | :---: |
| Lado (constante) | 0,500 | 51,2 % |
| Campeones: solo draft | 0,547 [0,540; 0,554] | 53,3 % |
| Campeones: draft + atributos + perfil | 0,548 [0,541; 0,555] | 53,4 % |
| Jugador × campeón (4 variables) | 0,568 [0,562; 0,575] | 55,2 % |
| Jugadores | 0,611 [0,604; 0,617] | 57,8 % |
| Jugadores + jugador × campeón | 0,632 [0,626; 0,639] | 59,2 % |
| **Híbrido: todo** | **0,641** [0,635; 0,648] | **59,9 %** |

Diferencias pareadas: jugadores (los dos bloques) − campeones **+0,084 [+0,075; +0,093]**;
híbrido − jugadores +0,009 [+0,006; +0,012]; atributos y perfil sobre el draft solo +0,0007
[−0,0002; +0,0017]. **Respuesta a §7.A:** el factor humano supera al draft por mucho. Las cuatro
variables de jugador × campeón predicen más que el draft entero, y el draft agrega poco (aunque
significativo) sobre los jugadores.

**2. Cuánto se decide durante la partida** (logística sobre Δoro, Δxp y Δkills del equipo, sola
y con el logit del híbrido previo; las 25.081 partidas de V que llegaron al minuto 20):

| Minuto | Solo el estado | Estado + lo previo | Acierto |
| :--- | :---: | :---: | :---: |
| Antes | — | 0,634 | 59,4 % |
| 10 | 0,721 | 0,745 | 67,7 % |
| 15 | 0,793 | 0,805 | 72,8 % |
| 20 | 0,858 | 0,863 | 77,7 % |

Lo previo suma cada vez menos al estado (+0,024 al 10, +0,005 al 20). Se dan vuelta muchas
ventajas: el equipo que va adelante en oro pierde el 32,5 % al minuto 10, el 27,0 % al 15 y el
23,3 % al 20. Con 5.000 de oro de ventaja a los 15, pierde el 7,0 %. Las rendiciones antes del 20
(6,3 %) son lo contrario: el que iba adelante a los 15 gana el 98,7 %. Primera sangre y primera
torre no entran, porque el dataset no tiene el minuto en que ocurrieron. *Techo práctico:* aun
sabiendo cómo va a estar la partida a los 10 minutos, el modelo erra una de cada tres. El
maestro de §9.S.39 (snapshots @10/@15, 0,837 fuera de pliegue) es consistente con esta curva.

**3. Familias** (sacar una dentro del híbrido sin el one-hot del draft, AUC 0,6397). La
experiencia con el campeón es lo irreemplazable: −0,0194 [−0,0230; −0,0158], unas 4 veces más
que la siguiente (pool −0,0045). El margen es lo que más predice solo (0,593), pero el acumulado y
el de 20 se tapan entre sí (−0,0040 y −0,0016 al sacarlos). El winrate previo sobra: −0,0004
[−0,0010; +0,0002]. Partidas, línea y rol (−0,0039), perfil del campeón (−0,0033), Data Dragon
(−0,0016) y ritmo y descanso (−0,0012) son chicos pero significativos.

**4. Registro.** 19 variantes mejoran, 9 empeoran, 29 no son significativas y 19 no tienen IC.
Con Holm sobreviven 21 de las 32 significativas sin corregir, y con BH 30. Las que caen con Holm
son diferencias de *modelo* con IC anchos (red por jugador, red con Poisson, stacking de 4); los
efectos chicos de variables sobreviven. Selección (máximo de $N(0, \mathrm{EE}_i^2)$ por tanda):
en 10 de las 11 tandas la mejor variante supera el p95 del azar; la excepción es S.37, donde la
mejor (la rareza de runas) tampoco era significativa. Que la V sea la misma en todas las pruebas
y que las bases se encadenen no lo corrige ningún método: el AUC elegido en V es algo optimista.

**5. Dónde falla.** Lobbies de Diamond: AUC 0,624, contra 0,650 en Master y GM+. Lobbies donde
todos tienen ≥ 50 partidas previas: 0,637, contra 0,650 en el tramo de 10 a 49 (son los más
parejos). ECE por deciles: 0,0092.

**Conclusión para el informe.** El resultado de una partida de SoloQ de alto elo depende antes
de la partida mucho más de los jugadores que del draft, y dentro de los jugadores, de dos cosas:
si conocen su campeón y cuánto mejor que lo esperado vienen rindiendo. Ningún modelo más complejo
le ganó con claridad a una logística sobre las mismas variables: cada salto del camino fue una
variable mejor pensada. Y buena parte del resultado se decide jugando.

#### S.43. Planteamiento básico del mapa de LP partida a partida (`lp_mecanica.py`) (2026-11-14)

**Pedido del usuario.** Empezar un planteamiento básico del mapa de LP partida a partida para todo el
tramo de entrenamiento sin esperar a reentrenar `principal` ni generar aún `data/lp_por_partida.csv`
(reservado para después del cierre del modelo). Implementado en `src/features/lp_mecanica.py`
con 9 tests unitarios en `tests/test_lp_mecanica.py` y el benchmark `scripts/analysis/mapa_lp_basico.py`,
sin tocar `data/lol_las.db`, los CSV de `data/`, el split de test (< 04/09) ni la caja cerrada
del 15 % (`~en_caja`).

**Arquitectura e invariantes.** El planteamiento combina dos capas sin mezclar regímenes de servidores
ni violar causalidad:

1. **Parámetros ajustados únicamente con LAS (`la2`):** `cargar_eventos_vivo_las` +
   `ajustar_parametros_las` estiman el paso solo sobre los 22.720 eventos `enlazada` de LAS en
   `lp_vivo.db` (con roles históricos de `lol_las.db` anteriores al inicio de `lp_vivo`), y
   `ajustar_k_diamond` ajusta `k_diamond` en L1 sobre desarrollo (`~en_caja`): `k_master = 20,00`,
   `k_diamond = 18,75`, `perdon = 0,0406`, `piso_quieto = 0,6257`, `piso_baja = 30,80` (en LAS,
   114/143 descensos desde Master 0 caen a D1 75 LP = −25 y 24/143 a −50), `doble_base = 0,0666`
   y $\delta(\text{gap})$ monótona por tramos de `[−1,11; +10,00]` entre gaps `−700,5` y `+722,0`,
   más tabla `doble_por_rol` suavizada por Bayes empírico ($C = 10$). Además,
   `filtrar_rebotes_sin_partida` elimina rebotes de réplica atrás/adelante en eventos `sin_partida`.
2. **Paso hacia adelante (`recorrer_adelante`) y banco de días:** avanza por partidas manteniendo el
   saldo del banco (`avanzar_banco`, `recargar_banco`; tope 14 días / +1 día por partida en
   Master+ con decay de 75/día y piso D1 75 = 375, tope 28 días / +7 en Diamond con decay de 50/día
   y piso D4 75 = 75). Si se pasan `dW_total` y `dL_total`, reparte las ocultas en proporción
   temporal; si no, admite una tasa global `tasa_oc` por partida observada.
3. **Suavizado exacto en grilla 1D sin invertir la dinámica (`suavizar_en_grilla`):** sobre una grilla
   discreta `[−400, 3400]` con paso 10 LP, aplica operadores lineales exactamente adjuntos
   (`_pushforward_lineal` / `_pullback_lineal`, verificados en test con producto interno exacto)
   entre una densidad inicial `(z1, sigma1)` y una final `(z2, sigma2)`. Permite anclas exactas
   (`sigma = 0`), anclas ruidosas de `principal` (`sigma_ancla_z1 > 0`), o extremo abierto
   (`z2 = None`, caso estrictamente causal).
4. **Modos `descriptivo` vs. `causal` (`estimar_mapa_tramo`):** en `modo="descriptivo"`, usa ambas
   fotos, interpola lobbies iniciales y refina con 2 pasadas (`suavizar_lobby_jugador` centrado);
   en `modo="causal"`, prohíbe `foto2`, prohíbe `foto1` si hay partidas anteriores a `t1` (lanza
   `ValueError`), usa fracciones de rol acumuladas previas y suavizado de lobby estrictamente causal
   (`<= k`). Acepta `jugadores_objetivo` para evaluar muestras manteniendo los 10 participantes en los
   lobbies.

**Resultados medidos (`python scripts/analysis/mapa_lp_basico.py`).**

*Tramos contiguos reales de LAS en `lp_vivo.db` ($\ge 10$ partidas seguidas; 510 tramos, 6.314
puntos intermedios):*

| Método | MAE | RMSE | MedAE | p90 | Cobertura [p10, p90] |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Recta temporal | 29,61 | 37,67 | 22,41 | 60,27 | — |
| Camino regla (a, b) | 5,73 | 8,57 | 3,54 | 14,48 | — |
| Paso adelante (causal, solo `z1`) | 10,64 | 16,13 | 7,00 | 24,10 | — |
| **Suavizado grilla (2 anclas)** | **4,60** | **7,01** | **2,51** | **12,06** | **0,960** (ancho med. 30,0 LP) |

Paso 1-a-1 sin acumular ($n = 6.824$): MAE = **2,37 LP**.

*Cierre Foto 1 → Foto 2 en desarrollo ($n = 22.626$, `~en_caja`; subconjunto `ocultas == 0`,
$n = 5.586$):*

| Método | MAE todo | RMSE todo | MAE `ocultas == 0` |
| :--- | :---: | :---: | :---: |
| No cambió (`z1`) | 145,45 | 224,40 | 160,01 |
| Regla (a, b) solo observadas | 115,99 | 182,51 | 135,18 |
| **Paso mecánico solo observadas (`tasa_oc` global)** | **71,49** | **98,42** | — |
| **Paso mecánico + saldo `dW`/`dL`** | **63,36** | **89,22** | **61,18** |

Desglose por `tier1` (`MAE z1` → `Regla` → `Mecánico obs` → `Mecánico + saldo`): Challenger (172)
`457,99 → 395,60 → 116,13 → 89,50`; Grandmaster (405) `371,87 → 308,24 → 98,20 → 70,08`;
Master (6.678) `190,05 → 133,87 → 73,98 → 69,48`; Diamond (15.371)
`116,61 → 100,03 → 69,21 → 60,23`.

*Robustez del lobby ante baja cobertura del pool en `foto1` ($n = 1.500$ activos de desarrollo sin
ocultas; enmascarando el 50 %–95 % de los jugadores):* sin suavizado temporal (`ventana=1, c=0,0`),
el MAE salta de `72,06` (100 %) a `75,51` (25 %), `83,21` (10 %) y `88,25` (5 %). Al usar
`suavizar_lobby_jugador(ventana=31)`, el salto desaparece (`74,64 / 76,91 / 80,59` con `c_lobby=0,5`
y `71,38 / 71,86 / 73,89` con `c_lobby=0,0`). Hallazgo empírico clave para la iteración siguiente:
al principio probamos `c_lobby = 4,0`, lo que encogía de más el `gap` porque `n_eff` en
`suavizar_lobby_jugador` ya está en unidades de partidas equivalentes (`sum(n_lob / 9)`, no en
co-jugadores); bajarlo a `c_lobby = 0,5` mejora los tramos cortos de `lp_vivo.db` (`14,08 → 13,76`
MAE frente a `0,0`, mínimo en `1,0` con `13,66`), pero en el cierre largo Foto 1 → Foto 2 la ventana
móvil ya promedia el ruido y encoger el `gap` hacia 0 subestima a quienes suben (`73,89` con `c=0,0`
vs. `80,59` con `c=0,5` al 5 %). Queda abierto usar `c_lobby` solo cuando `ventana` es chica o
reemplazar el encogimiento de `gap` por un aumento de incertidumbre `sigma_lobby`.

*Muestra de `estimar_mapa_tramo` (1.000 jugadores dev, 47.480 filas, 50 % con `foto1` reemplazada
por `principal` con `sigma_ancla_z1 = 80`):* tiempos `12,30 s` (1 pasada), `22,04 s` (2 pasadas),
`15,36 s` (causal). La pasada 2 ajusta poco sobre la pasada 1 (`|Pass2 − Pass1|` media `0,68 LP`,
p90 `1,81 LP`). Nota pendiente: en las filas `ancla_principal_grilla` (`23.406`, ancho mediano
`60,0 LP`) no se pasó `tasa_oc_defecto`, mientras `puente_grilla` (`24.074`, `100,0 LP`) sí incluyó
la varianza de partidas ocultas por `dW_total`/`dL_total`; al generar el mapa completo convendrá
estimar `tasa_oc_defecto` cuando falte una de las fotos.

#### S.44. Qué pesa en las partidas fáciles y en las difíciles de predecir (2026-11-14)

**Pedido del usuario.** Medir qué afecta más cuando las partidas son fáciles de predecir y cuando
son difíciles. Está en la sección 7 de `notebooks/eda_modelos_ganador.ipynb`, con el mismo
generador y los mismos datos que §9.S.42.

**Definición.** La dificultad se mide con el híbrido de §9.S.42 (logística con draft y las 85
variables, ajustada sin V). En V: *fácil* = el 20 % con mayor |logit|, *difícil* = el 20 % con
menor |logit|, *medio* = el resto. *Alternativa descartada:* definir la dificultad con el
resultado (partidas que el modelo acertó o no). Mezcla dificultad con suerte, y no se puede saber
antes de jugar.

**Descomposición del logit.** En la logística,
$\text{logit} = b + \sum_f c_f$, con $c_f = \sum_{j \in f} w_j x_j$, de forma exacta (se verifica
contra las probabilidades). El *aporte alineado* $c_f \cdot \operatorname{signo}(\text{logit})$ es
cuánto empuja la familia $f$ hacia el favorito, y el *conflicto* es
$1 - |\sum_f c_f| / \sum_f |c_f|$ (0 = todas hacia el mismo lado, 1 = se cancelan). *Alternativa
descartada:* SHAP sobre un boosting. En una logística, la descomposición lineal ya es exacta y
aditiva, y el boosting rinde menos (§9.S.25).

**Resultados (V, 26.934 partidas).**

| Grupo | Prob. media del favorito | Acierto | Conflicto | Familias a favor (de 10) |
| :--- | :---: | :---: | :---: | :---: |
| Difícil | 51,8 % | 50,6 % | 0,93 | 5,1 |
| Medio | 59,7 % | 58,8 % | 0,72 | 5,7 |
| Fácil | 72,8 % | 72,2 % | 0,43 | 6,5 |

1. **Difícil = información que se contradice**, no falta de información. Con conflicto 0,93, el
   acierto es el de una moneda.
2. **Pesan las mismas familias en los dos extremos.** Del empuje a favor del favorito,
   experiencia con el campeón 34-36 %, margen acumulado 29-30 % y margen de las últimas 20
   13-14 %, en los tres grupos. Las familias de campeones (draft, Data Dragon, perfil) se mueven
   lo mismo en los tres ($|c_f|$ casi igual), pero su empuje neto es chico: se cancelan entre
   equipos.
3. **Winrate previo y pool empujan en contra del favorito** en los tres grupos (−0,022 y −0,025 de
   logit en las fáciles). Dado el margen, corrigen: un winrate alto con el mismo margen anticipa
   una vuelta a la media, y un pool concentrado frena al especialista que el margen sobrestima.
   Esto explica que sacar el pool cueste AUC (§9.S.42, punto 3) aunque solo no prediga nada.
4. **De dónde viene la confianza importa algo.** Entre las fáciles, cuando domina el margen
   acumulado se acierta el 74,0 % [72,0; 76,0]; cuando domina la experiencia, el 71,7 % [69,9;
   73,5]; cuando dominan los atributos del campeón, el 67,8 % [63,0; 72,2]. Con experiencia y
   margen de acuerdo, 72,8 %; en desacuerdo, 69-71 %.
5. **Lo que hace fácil un lobby es la asimetría entre equipos.** En las fáciles, la diferencia de
   margen entre equipos es ×1,46 y la de estrenos ×1,24 respecto de las difíciles. En cambio,
   tier (×1,03), historial (×0,99 el mínimo) y estrenos totales (×1,01) son iguales.
6. **Las fáciles se juegan más desparejas** (solo descriptivo; el modelo no lo ve). El favorito va
   adelante a los 15 en el 64,3 %, contra 50,1 % en las difíciles. Duran 27,8 min contra 28,8, y
   las rendiciones antes del 20 son 8,4 % contra 6,5 %.
7. **Las sorpresas se deciden temprano.** En las fáciles que el modelo falla, el favorito va
   adelante a los 10 solo en el 35,1 % (mediana −654 de oro); en las que acierta, en el 71,7 %. A
   los 15: 32,0 % contra 77,7 %. El favorito no tira la partida al final: pierde la fase de
   líneas.

**Implicaciones.** (a) Para mejorar las difíciles, el cuello de botella es separar señales
contradictorias, no agregar señales nuevas: son partidas parejas en todo lo que se mide. (b) Las
sorpresas aparecen en la fase de líneas, lo que apoya que el modelo por línea de §9.S.36 haya
sido de lo poco que sumó al final. (c) La primera versión de esta sección proponía filtrar las
partidas claras por acuerdo entre familias. El punto 9 lo descarta: fuera de la muestra, ese
filtro no le gana a la probabilidad sola.

**8. Regla «si duda, rojo» (pedido del usuario; sección 7.5).** El rojo gana el 51,9 % en el ajuste
y el 51,2 % en V. Propuesta: si $\max(p, 1-p) < t$, predecir rojo en lugar de seguir al modelo,
con $t = 0{,}52$ como caso pedido. El umbral se evalúa con el logit fuera de pliegue del ajuste
(KFold 5, mismo híbrido) y se confirma en V, con IC bootstrap pareado de la diferencia de acierto:

| $t$ | % dudosas (V) | Gana rojo en las dudosas (ajuste / V) | Δ acierto en V [IC 95 %] |
| :---: | :---: | :---: | :---: |
| 0,51 | 5,6 % | 50,3 % / 51,1 % | +0,07 pp [−0,14; +0,24] |
| 0,52 | 11,2 % | 49,8 % / 49,1 % | −0,17 pp [−0,45; +0,07] |
| 0,55 | 27,7 % | 50,0 % / 49,7 % | −0,61 pp [−1,09; −0,23] |
| 0,60 | 52,9 % | 50,5 % / 49,7 % | −2,25 pp [−2,86; −1,73] |

**Se descarta.** El intercepto de la logística ya es la ventaja del rojo. Una partida con
$p \approx 0{,}5$ es una en la que los jugadores favorecen al azul lo justo para compensarla, y en
esas partidas el rojo gana la mitad, no el 51-52 %. Con $t = 0{,}52$ la regla cambia moneda por
moneda (no significativa); con umbrales mayores empeora, porque reemplaza predicciones que sí
saben algo. Cerca del 50 % el modelo está calibrado (bins de 0,02 entre 0,44 y 0,56: desvío
máximo 0,02). *Alternativa adoptada para las difíciles:* abstenerse y reportar el acierto por
cobertura, en lugar de forzar un lado.

**9. Qué hacer con las fáciles (sección 7.6).** Se probaron tres ideas fuera de la muestra:

* *Meta-modelo de acierto.* Es una logística que predice si el híbrido va a acertar a partir de
  |logit|, conflicto, familias a favor, acuerdo entre experiencia y margen y el aporte alineado de
  cada familia, ajustada con el logit fuera de pliegue. En V, el AUC de «va a acertar» da 0,5896,
  contra 0,5898 con |logit| solo. En el 5, 10, 20 y 30 % más seguro, el acierto es 77,1, 74,8,
  72,2 y 69,9 % por |logit|, y el meta-modelo suma menos de 0,6 pp. *Descartado:* el conflicto ya
  está absorbido en el |logit|, porque mucho conflicto implica un logit chico.
* *Recalibración por grupo* (logística sobre el logit fuera de pliegue). La pendiente es 0,957 en
  las fáciles, 0,883 en las medias y 1,190 en las difíciles. Las fáciles están calibradas, y en las
  difíciles la corrección no cambia ninguna decisión. *Descartado.*
* *Confirmar en vivo.* A los 10 minutos, el favorito de una fácil va adelante en oro en el 61,5 %
  de los casos, y ahí se acierta el 84,2 %. En el 38,5 % restante se acierta el 53,1 %, una moneda.
  A los 15 minutos, si el favorito va atrás, se acierta el 46,1 %: peor que cambiar al que va
  adelante. *Adoptado como recomendación:* predicción en dos tiempos, previa más confirmación al
  minuto 10, para la demo y el informe. Antes de la partida no queda nada que exprimir de las
  fáciles.

#### S.45. Quién gana: la forma del día medida con el margen (2026-11-14)

**Pedido del usuario.** Medir cuánto mejor o peor que lo esperado viene jugando hoy cada
jugador, con todas las estadísticas de habilidad relevantes, en lugar de con victorias. La racha
de victorias del día ya se había descartado (§9.S.24, punto 3), pero con la señal más ruidosa:
el margen fue lo que llevó el modelo de 0,610 a 0,637. Script:
`historial/ganador_primeros/ganador_forma_dia.py` (~6 min, solo `data/splits/train`, misma partición F/V
y caché de §9.S.41).

**Definición.**

* *Día de juego:* de 06:00 a 06:00, hora de Argentina (`--corte`, pedido del usuario). La
  franja de 06:00 a 08:00, que distingue este corte del de `daily_form` (08:00), tiene el 1,8 %
  de las partidas, y el valle de actividad cae entre las 6 y las 8.
* *Residuo* de cada estadística, como en el margen:
  $r = (x - \hat\mu_{c\ell}) / \sigma_\ell$, con la media previa del campeón en la línea.
* *Forma de hoy:* $\text{hoy} = (S_d + k\,m)/(N_d + k) - m$, donde $S_d$ y $N_d$ son la suma y la
  cantidad de $r$ de las partidas previas del jugador en el día, $m$ es su margen acumulado y
  $k = 3$ (`prior_margin_by` con el día como grupo). Vale 0 en la primera partida del día, así que
  mide el desvío de hoy respecto de su nivel, no el nivel.
* *Estadísticas (19):* las 15 del margen (kills, asistencias y daño por minuto; % del daño;
  participación en kills; muertes y daño recibido por minuto; oro y CS por minuto; % del oro;
  visión por minuto; oro a los 10 y a los 15, XP y CS a los 15 contra el rival de línea) más XP
  y CS a los 10 y daño a los 15 contra el rival, y la agresividad temprana.
* *Variantes:* victorias de hoy con el mismo encogimiento (la racha de §9.S.24); el margen de la
  partida anterior del día, sin encoger (tilt); las dos juntas con el margen de hoy. Las tres
  llevan el log de las partidas previas del día.

El 61,5 % de las filas tiene alguna partida previa en el día. *Sin leakage* por construcción:
toda suma es cumsum − valor propio.

**Resultados en V** (híbrido del EDA, draft + 85 variables por línea, AUC 0,6413; ΔAUC con IC
bootstrap pareado, B = 400):

| Variante | Columnas | ΔAUC [IC 95 %] | Δ log loss |
| :--- | :---: | :---: | :---: |
| Victorias de hoy | 10 | +0,0002 [−0,0003; +0,0006] | −0,00006 |
| Margen de hoy (19 estadísticas) | 100 | −0,0001 [−0,0011; +0,0010] | +0,00024 |
| Margen de la partida anterior del día | 100 | +0,0002 [−0,0009; +0,0011] | +0,00001 |
| Las tres juntas | 200 | −0,0001 [−0,0014; +0,0011] | +0,00017 |

**A nivel jugador, el efecto existe.** Se tomaron las filas del ajuste con dos o más partidas
previas en el día y un compuesto de habilidad: la media de los $r$ de kills, asistencias, daño,
participación, oro, CS y las tres diferencias a los 15, menos las muertes. El winrate sube del
49,3 % al 52,2 % del peor al mejor quintil de la forma de hoy. En una logística por jugador,
la forma de hoy suma +0,036 de logit por desvío controlando el nivel acumulado, y +0,029
controlando además las últimas 20 (el nivel da +0,071). Es real y no es solo el nivel, pero es
chico: ~0,7 pp de winrate por desvío.

**Por qué no llega al modelo.** (a) Las partidas de hoy ya están dentro del margen de las
últimas 20, que el híbrido tiene estadística por estadística. (b) Un efecto de 0,03 de logit
por jugador, sumado entre 5 contra 5, es chico al lado de la dispersión del logit del modelo.
(c) Con 100 columnas nuevas, el ruido que agregan compensa lo poco que suman (la log loss empeora
apenas).

**Sin las últimas 20 (pedido del usuario: «no siempre tenemos v20»).** Con menos de 20
partidas previas, `v20_` es idéntico a `acum_` (la ventana abarca todo el historial). Solo aporta
información propia para el 71,0 % de las filas con 20 o más previas; el 26,0 % tiene de 1 a 19 y
el 3,0 % ninguna. Se sacaron del híbrido las 16 variables `v20_*` y `wr_v20` por línea y se probó
si la forma de hoy las reemplaza. «WR de hoy» son las victorias de hoy encogidas; el winrate
acumulado (`wr_acum`) queda en todas las variantes. ΔAUC contra «sin v20», IC bootstrap pareado:

| Variante | Todas (26.934) | Lobby con alguien < 20 previas (14.756) | Los 10 con ≥ 20 previas (12.178) |
| :--- | :---: | :---: | :---: |
| Con v20 (base) | **+0,0015** [+0,0005; +0,0026] | +0,0008 [−0,0005; +0,0022] | **+0,0025** [+0,0006; +0,0047] |
| Sin v20 + margen de hoy | +0,0009 [−0,0003; +0,0021] | +0,0009 [−0,0008; +0,0025] | +0,0009 [−0,0011; +0,0028] |
| Sin v20 + margen de hoy + WR de hoy | +0,0007 [−0,0005; +0,0020] | +0,0009 [−0,0006; +0,0026] | +0,0005 [−0,0015; +0,0025] |
| Con v20 + margen de hoy + WR de hoy | +0,0014 [+0,0000; +0,0029] | +0,0010 [−0,0010; +0,0029] | +0,0020 [−0,0007; +0,0046] |

Sin v20, la forma de hoy recupera alrededor del 60 % de lo que v20 aporta, pero ninguna
diferencia es significativa. En los lobbies con historial corto valen lo mismo, y v20 solo se
despega donde los 10 tienen 20 o más partidas, que es justamente donde existe como variable
propia. El WR de hoy no suma nada sobre el margen de hoy. **Decisión:** v20 se queda. La forma
de hoy sería un reemplazo razonable solo si v20 no se pudiera calcular (por ejemplo, en vivo
para un jugador sin historial en la base), y aun así recupera poco.

**Decisión: se descarta** como variable del modelo de minuto 0. Confirma §9.S.24 con la señal
buena: la forma del día existe pero ya está contenida en la forma reciente. *Alternativas
descartadas:* encoger menos ($k$ menor), que agrega varianza a un efecto ya cubierto. *No
probado:* definir la sesión por pausas (p. ej., < 2 h entre partidas) en lugar del día de juego.

#### S.46. Evaluación causal del mapa de LP mecánico en `quién gana` (2026-11-15)

**Pedido del usuario.** Construir el mapa causal de LP partida a partida (`src/features/lp_mecanica.py`,
§9.S.43) sobre el tramo posterior a la Foto 1 (`t1 = 2026-09-04 22:51:22 UTC`, 40.821 jugadores en
`lol_las.db`) e incorporarlo al modelo de predicción de victoria (`quién gana`), comparándolo con
la `foto1` estática y el `prior_lobby_level` del compañero (`src/features/lobby.py`, §9.S.31).
Script: `scripts/analysis/ganador_mapa_lp.py` (caché en `data/analysis/mapa_lp_causal_v.pkl`; no
lee `foto2`, test `< 04/09` ni `en_caja`, ni genera `data/lp_por_partida.csv`).

**Corrección de causalidad en `lp_mecanica.py`.** Al preparar la evaluación causal se detectaron dos
fugas en `src/features/lp_mecanica.py` que se corrigieron y cubrieron en `tests/test_lp_mecanica.py`:

1. En `estimar_mapa_tramo`, los jugadores sin foto (`causal_lobby`) usaban `.ffill().bfill()`,
   propagando hacia atrás el lobby de partidas futuras; ahora usan `suavizar_lobby_jugador(...,
   modo="causal")` y quedan marcados como `"sin_dato"` antes de su primera observación válida.
2. En `suavizar_en_grilla`, aunque su docstring prometía filtro causal puro con `z2 is None`, seguía
   ejecutando el paso hacia atrás (`beta_pre_match`), filtrando hacia atrás lobbies futuros
   `lik_lob[k+1..]` y el resultado `w_arr[k]` cuando `tau_lobby > 0`. Ahora omite la pasada
   hacia atrás cuando `z2 is None`, garantizando invariancia causal estricta y reduciendo a la mitad
   el cómputo en `modo="causal"`.

**Cobertura y sesgo de faltantes en V** (`data/analysis/embeddings_partidas/variables.npz`, 187.817
partidas limpias; `fit_all = 160.883`, `fit_post(A) = 26.934`, `V = 26.934` del 16/09 al 27/09;
mapa corrido sobre las 55.020 partidas LAS `>= t1`, 550.200 filas en `90,2 s` con
`ParametrosPaso` de `lp_vivo.db`):

* El `89,05 %` de las filas de V (`239.859`) tiene `foto1` (`causal_desde_foto1`, con
  `|lp_causal - lp_foto|` media `115,2` LP, mediana `78,7` y p90 `268,7`), y el `10,95 %` restante
  (`29.481`) entra por `causal_lobby`. Cobertura del mapa `100,00 %`; `prior_lobby_level >= 1`:
  `97,11 %`.
* **Asimetría de quienes faltan en `foto1`:** las cuentas sin `foto1` (`tiene_foto1 = 0`, asignadas
  por `causal_lobby` alrededor de `328,5` LP de media contra `549,6` LP de `causal_desde_foto1`) son
  cuentas nuevas o en ascenso (`logn_tot = 2,38` y `wr_acum = +0,0283` vs. `4,46` y `+0,0064`), con
  winrate observado de **`52,25 %`** en V frente a **`49,72 %`** en `tiene_foto1 = 1` (correlación
  univariada de `tiene_foto1` = `−0,0522`, AUC `0,47260`). Si se mezcla `causal_lobby` crudo con
  `causal_desde_foto1` sin separar `tiene_foto1` ni imputar con `nivel_v10`, las `15.206` partidas
  con algún jugador sin foto invierten la pendiente lineal (AUC univariado `0,49219`), ocultando
  que en las `11.728` partidas con los 10 jugadores en `foto1` el seguimiento mecánico sube el AUC
  univariado de **`0,50130`** (`lp_foto`) a **`0,51811`** (`lp_causal`). Imputando los faltantes con
  `nivel_v10` encogido (`k = 5`), el AUC univariado en todo V pasa de `0,50416` (`foto`) a `0,51247`.

**Señal intrínseca sobre Draft solo** (`fit = A`, `26.934` -> V, regresión logística `C = 0,05` sin
`Xd_85`; IC bootstrap pareado `B = 400`):

* Con la `foto1` imputada (`lp_foto (fill niv)`): `+0,00002` en `equipo` y `+0,00023` en `por_linea`
  (la foto congelada del 04/09 ya no discrimina equipos en V: en `10f1`, `+0,00003`).
* Con el mapa actualizado (`lp_causal (fill niv)`): **`+0,00127 [+0,00018; +0,00253]`** (`equipo`,
  `0,54429 -> 0,54555`; en `10f1` `+0,00239`) y **`+0,00197 [+0,00077; +0,00341]`** (`por_linea`,
  `0,54626`; en `10f1` `0,54247 -> 0,54554`, `+0,00307`).
* Combinando `+ niv(mapa) + mapa_lp`: **`0,55549 (+0,01121 [+0,00724; +0,01518])`** (`por_linea`,
  `+0,00767 [+0,00428; +0,01122]` sobre `nivel(foto)` solo).

**Aporte marginal sobre el Híbrido completo (`Draft + Xd_85`, `fit = ~val = 160.883`, features `>= t1`
centradas y escaladas en `fit_post = A`) en V (`26.934` partidas):**

| Variante | Repr. | AUC (V) | ΔAUC vs. base [IC 95 %] | Acc | Log Loss | `10f1` (`n=11.728`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Base (`Draft + Xd_85`) | — | 0,64128 | — | 59,84 % | 0,66103 | 0,64433 |
| `+ lp_foto (fill niv)` | `equipo` | 0,64138 | +0,00010 [−0,00007; +0,00026] | 59,90 % | 0,66100 | 0,64433 |
| `+ lp_causal (fill niv)` | `equipo` | 0,64153 | +0,00025 [+0,00006; +0,00047] | 59,86 % | 0,66093 | 0,64465 |
| `+ lp_causal (crudo lobby)` | `por_linea` | 0,64132 | +0,00004 [−0,00029; +0,00033] | 59,82 % | 0,66104 | 0,64437 |
| `+ actividad (tf1 + n_niv)` | `equipo` | 0,64433 | +0,00305 [+0,00164; +0,00435] | 60,26 % | 0,65972 | 0,64603 |
| `+ nivel(foto) [lobby.py]` | `equipo` | 0,64412 | +0,00283 [+0,00158; +0,00399] | 60,14 % | 0,65980 | 0,64719 |
| `+ nivel(foto) [lobby.py]` | `por_linea` | 0,64463 | +0,00335 [+0,00213; +0,00479] | 60,06 % | 0,65950 | 0,64771 |
| `+ nivel(mapa_lp)` | `por_linea` | 0,64460 | +0,00332 [+0,00203; +0,00478] | 60,05 % | 0,65951 | 0,64773 |
| **`+ niv(foto) + mapa_lp`** | `equipo` | **0,64547** | **+0,00419 [+0,00277; +0,00568]** | **60,35 %** | **0,65919** | **0,64789** |
| **`+ niv(mapa) + mapa_lp`** | `equipo` | **0,64540** | **+0,00412 [+0,00268; +0,00558]** | 60,34 % | 0,65922 | **0,64791** |

**Lectura y mecanismos (sin exageraciones):**

1. **Qué explica el salto de `+0,00335` de `lobby.py`:** dos tercios no provienen del nivel de MMR
   del lobby, sino de `n_nivel` (`log1p` de partidas jugadas desde el `04/09`, que por sí sola ya
   aporta **`+0,00210`** en `equipo` con peso negativo `−0,0877` dado `Xd_85`) y de `tiene_foto1`
   (señal de cuentas nuevas/escaladoras). El nivel promedio `nivel_acum`/`nivel_v10` aporta ~`+0,001`
   adicionales al identificar el nivel real del lobby por encima del margen relativo al campeón.
2. **Por qué el LP causal solo (`+0,00025`) aporta poco sobre `Xd_85`, y por qué junto con `niv` sí
   gana (`+0,00136 [+0,00051; +0,00218]` vs. `niv(foto) equipo`):** la evolución $\Delta LP$ desde la
   foto depende casi linealmente del `W − L` reciente, señal ya parcialmente capturada por `wr_v20`,
   `wr_acum` y los márgenes en `Xd_85`. Cuando el mapa se resume por equipo (`lp_causal`, `dl_desde_f1`,
   `lp_incert`, `tiene_foto1`) sobre `nivel(foto)` (`0,64547`, Acierto `60,35 %`, LL `0,65919`), suma
   **`+0,00136`** sobre `nivel(foto) equipo` (`0,64412`) y **`+0,00084`** sobre `nivel(foto) por_linea`
   (`0,64463`), reduciendo además la log loss (`0,65950 -> 0,65919`). Por el contrario, expandir las
   variables del mapa a 42–48 columnas `por_linea` sobreajusta con `C = 0,05` (`0,64448–0,64455`).
3. **Dónde se concentra la mejora:** en lobbies **Master+** (`n = 18.239`, LP ≥ 400): `base 0,64738`
   -> `nivel(foto) 0,65093 (+0,00355)` -> `+ niv(foto) + mapa_lp` **`0,65266 (+0,00528 [+0,00386;
   +0,00683])`** (LL `0,65523` vs `0,65622`), y también cuando falta alguna foto (`15.206`: `+0,00446`
   vs. `+0,00261`) y en `10f1` (`0,64791` vs. `0,64719`). En **Diamond** (`n = 8.695`) ni el lobby ni
   el mapa despegan del cero (`+0,00117 [−0,00232; +0,00490]`). Una capa no lineal (`HistGradientBoosting`
   en 2 etapas sobre logit OOF) rinde menos (`0,64282`, LL `0,66095`) que la logística (`0,64498`, LL
   `0,65926`), y `lp_incert` solo en `10f1` no suma (`−0,00030`): actúa junto con `tiene_foto1` y
   `dl_desde_f1`. En el régimen estricto `fit = A` (`26.934` -> V), los resultados replican la misma
   relación (`equipo`: base `0,63493`, `nivel(foto) 0,63630`, `+ niv(foto) + mapa_lp` **`0,63713`**,
   `+0,00083 [+0,00023; +0,00148]` sobre `nivel(foto)`).

**Decisión práctica.** El seguimiento mecánico causal sustituye con ventaja a la foto congelada del
04/09 cuando ya pasaron 2–3 semanas (en `Draft solo` y en `10f1`), pero sobre el híbrido de 160k
partidas solo agrega valor claro si entra **compactado por equipo** junto a `nivel(foto)` (`0,64547`,
`+0,00136` vs `nivel(foto) equipo`), **imputando con `nivel_v10`** y separando `tiene_foto1` (nunca
mezclando `causal_lobby` crudo) — y con la cautela abierta de que parte del salto de `actividad`
(`n_nivel`, `tiene_foto1`) comparte sensibilidad a la cobertura del crawler con §9.S.30.

#### S.47. Validación temporal por bloques: la regla de decisión y lo que cambia (2026-11-14)

**Pedido del usuario.** Validar con bloques temporales en lugar de usar solo V. Hasta acá, cada
idea se decidía con una única ventana fija (V, ≈16/09 → 27/09), con IC por bootstrap. Ese IC mide
el ruido de muestreo de V, pero no el sesgo de elegir la mejor de más de 70 variantes mirando V
(sección 5 de `eda_modelos_ganador.ipynb`), ni si el efecto se sostiene en otra época. La
validación cruzada existía solo dentro del stacking y del modelo por línea, para producir
predicciones fuera de muestra.

**Herramienta.** `src/models/validacion_temporal.py` (5 tests en `tests/test_validacion_temporal.py`):

* `bloques_por_fecha(t, k)` parte el ajuste en $k$ bloques de igual cantidad de partidas, por
  fecha. Son los mismos cortes que `ganador_logistica_oos.py` y `ganador_red_jugador.py --oos`.
* `predecir_por_bloques` predice el bloque $b \ge 1$ con lo entrenado en los bloques $< b$
  (ventana creciente, *forward-chaining*). Un test verifica que nunca entrena con partidas
  posteriores a las que predice.
* `comparar_por_bloques` da el ΔAUC por bloque con IC bootstrap pareado, y la media de los Δ con
  IC bootstrap estratificado por bloque.
* `decidir` aplica la regla.

**Regla de decisión (vigente desde hoy).** Una variante **entra** si mejora en al menos 3 de los
4 bloques evaluados **y** el IC 95 % de la media de los Δ queda entero arriba de 0. **Empeora**
si el IC de la media queda entero debajo de 0. Lo demás es **no concluyente**. V solo confirma.

*Alternativas descartadas:*

* *k-fold al azar:* las variables son causales, así que no filtra el resultado, pero mide
  "predecir partidas de la misma época". El juego cambia con los parches, así que es optimista
  para la pregunta real, que es predecir partidas futuras.
* *Ventana deslizante de tamaño fijo:* descarta historia que el modelo sí usa. Además, el bloque
  0 abarca 2024-11 → 2026-07 y es el que más pesa en el entrenamiento del bloque 1.
* *Exigir 4 de 4 bloques:* con efectos de milésimas, un bloque en contra es normal aunque el
  efecto sea real. *Decidir solo por la media:* una media positiva podría venir de un único
  bloque.

**Aplicación** (`scripts/analysis/validacion_bloques.py`, ~12 min con 4 hilos). Modelo: el
híbrido de §9.S.42, con la estandarización ajustada solo con las filas de entrenamiento de cada
bloque. Bloques del ajuste:

| Bloque | Fechas | AUC del híbrido |
| :--- | :--- | :---: |
| 1 | 02/07 → 03/08 | 0,6225 |
| 2 | 03/08 → 20/08 | 0,6340 |
| 3 | 20/08 → 02/09 | 0,6422 |
| 4 | 02/09 → 16/09 | 0,6367 |
| V | 16/09 → 27/09 (entrenado con los 5 bloques) | 0,6413 |

El AUC sube con el tiempo porque los jugadores acumulan historial en la base.

ΔAUC **al sacar** cada parte del híbrido, o al agregar algo (signo +: el modelo queda mejor):

| Variante | b1 | b2 | b3 | b4 | Media [IC 95 %] | V [IC 95 %] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Sin experiencia con el campeón | −0,0194 | −0,0199 | −0,0173 | −0,0209 | **−0,0194** [−0,0211; −0,0179] | −0,0188 |
| Sin pool | −0,0043 | −0,0045 | −0,0042 | −0,0032 | **−0,0041** [−0,0049; −0,0032] | −0,0046 |
| Sin margen acumulado | −0,0004 | −0,0036 | −0,0052 | −0,0050 | **−0,0036** [−0,0044; −0,0028] | −0,0040 |
| Sin partidas, línea y rol | −0,0021 | −0,0013 | −0,0013 | −0,0010 | **−0,0014** [−0,0021; −0,0007] | −0,0040 |
| Sin ritmo y descanso | −0,0007 | −0,0004 | −0,0008 | −0,0015 | **−0,0009** [−0,0012; −0,0006] | −0,0013 |
| Sin winrate previo | −0,0001 | +0,0001 | −0,0008 | −0,0009 | −0,0004 [−0,0007; −0,0002] | −0,0005 (n. s.) |
| Sin últimas 20 (margen y WR) | +0,0021 | +0,0017 | −0,0005 | −0,0006 | +0,0007 [+0,0001; +0,0013] | **−0,0014** [−0,0026; −0,0002] |
| Sin el draft (campeón × línea) | +0,0012 | +0,0005 | +0,0003 | −0,0007 | +0,0003 [−0,0004; +0,0009] | **−0,0015** [−0,0029; −0,0002] |
| Sin Data Dragon | −0,0003 | +0,0002 | +0,0000 | +0,0001 | +0,0000 [−0,0002; +0,0003] | −0,0004 |
| Sin perfil típico del campeón | +0,0012 | +0,0004 | −0,0000 | +0,0001 | **+0,0004** [+0,0002; +0,0006] | −0,0002 (n. s.) |
| + margen de hoy (§9.S.45) | −0,0028 | −0,0008 | −0,0008 | +0,0012 | **−0,0008** [−0,0015; −0,0002] | −0,0001 |
| + WR de hoy | −0,0006 | +0,0001 | −0,0006 | +0,0004 | −0,0001 [−0,0004; +0,0001] | +0,0002 |
| Sin últimas 20 + margen de hoy | −0,0008 | +0,0012 | −0,0000 | +0,0012 | +0,0004 [−0,0005; +0,0013] | −0,0006 |

**Lectura.**

1. **Lo que se sostiene en todas las épocas:** experiencia con el campeón, pool, margen
   acumulado, partidas, línea y rol, y ritmo y descanso. Hacen falta en los 4 bloques y en V,
   con el mismo tamaño. El margen acumulado casi no vale en el bloque 1 (−0,0004), cuando la base
   tenía poco historial, y pesa desde el 2.
2. **Las últimas 20 dependen de la época.** En los bloques 1-2 sobran: con poco historial en la
   base son casi iguales al acumulado (con menos de 20 previas son idénticas, §9.S.45) y solo
   agregan ruido. Desde el bloque 3 empiezan a sumar, y en V suman de forma significativa. Se
   quedan: la tendencia apunta a que valen más cuanto más historial hay, que es la situación del
   test y de la demo.
3. **El draft (campeón × línea) solo vale en V.** En los bloques no suma (media +0,0003 al
   sacarlo, n. s.); en V suma 0,0015. Es consistente con un efecto que depende del meta del
   momento: los pesos campeón × línea aprendidos con partidas viejas sirven poco en el bloque
   siguiente si cambió el parche. La conclusión de §9.S.42 de que el draft «agrega poco aunque
   significativo» sobre los jugadores pasa a ser: **agrega poco y no en todas las épocas**.
4. **Perfil típico del campeón y Data Dragon sobran cuando está el draft.** El perfil cumple la
   regla para sacarlo (mejora en 3 de 4 bloques, media +0,0004 [+0,0002; +0,0006]), y en V da lo
   mismo. El −0,0033 de la sección 4 del EDA se midió sin el one-hot del draft, y ahí el perfil
   lo reemplaza. **Decisión:** en un modelo con draft, el perfil y Data Dragon se pueden sacar
   (no queda aplicado a ningún modelo).
5. **La forma del día queda confirmada como descartada,** ahora con evidencia en contra: el margen
   de hoy empeora en 3 de 4 bloques.

**Alcance.** Se revalidó lo que se puede reproducir en esta máquina con la caché de §9.S.41. Las
decisiones del modelo de 2 expertos (red de grupos, Poisson, margen por campeón, lobby, modelo
por línea, composición) necesitan las salidas de la PC con GPU. El meta del modelo de 2 ya se
eligió con bloques (`ganador_dos_expertos.py`), pero las variables de sus expertos se aceptaron
mirando V. Hay que repasarlas con esta regla cuando se corra allá.

#### S.48. Curva de aprendizaje: cuántas partidas necesita el modelo (2026-11-15)

**Pedido del usuario.** Ver cómo cambia el modelo según cuántas partidas de train se le dan.
Script: `scripts/analysis/curva_aprendizaje.py` (~4 min con 4 hilos; salidas en
`data/analysis/curva_aprendizaje/`).

**Diseño.** Las variables de cada jugador (margen, experiencia, pool…) se calculan siempre con
todo su historial previo (caché de §9.S.41). Lo que varía es cuántas partidas del ajuste entran a
la logística (C = 0,05, estandarización con esas filas). Se compara con dos criterios de
selección:

* *Recientes:* las N inmediatamente anteriores a V. Es lo que se haría en la práctica.
* *Al azar:* N al azar del ajuste, con 3 semillas. Separa cantidad de antigüedad.

Se evalúan dos modelos: el híbrido (draft campeón × línea + 85 variables por línea) y solo
jugadores (sin el draft). La métrica es el AUC en V, con IC bootstrap pareado (B = 200) de la
diferencia contra usar todo el ajuste. *Alternativa descartada:* medir con los bloques de
§9.S.47. Los bloques tempranos tienen menos historial y mezclarían dos efectos: menos partidas
para ajustar y variables más pobres.

| Partidas (recientes) | Desde | Híbrido: AUC | Δ vs todo [IC 95 %] | Solo jugadores: AUC | Híbrido al azar (media) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 2.000 | 15/09 | 0,5766 | −0,0647 [−0,0706; −0,0577] | 0,5764 | 0,5783 |
| 5.000 | 14/09 | 0,5983 | −0,0430 [−0,0485; −0,0374] | 0,5995 | 0,6050 |
| 10.000 | 12/09 | 0,6172 | −0,0241 [−0,0287; −0,0190] | 0,6183 | 0,6155 |
| 20.000 | 07/09 | 0,6320 | −0,0093 [−0,0127; −0,0060] | 0,6319 | 0,6272 |
| 40.000 | 30/08 | 0,6383 | −0,0030 [−0,0054; −0,0004] | 0,6370 | 0,6329 |
| 80.000 | 13/08 | 0,6410 | −0,0003 [−0,0020; +0,0017] | 0,6404 | 0,6379 |
| 120.000 | 14/07 | 0,6411 | −0,0002 [−0,0013; +0,0009] | 0,6398 | 0,6399 |
| 160.883 (todo) | 06/11/2024 | 0,6413 | — | 0,6397 | — |

Con todo el ajuste, C ∈ {0,01; 0,05; 0,2} da 0,6414 / 0,6413 / 0,6408 en el híbrido: la
regularización no limita.

**Lectura.**

1. **La logística se satura en ~80.000 partidas.** Con las 80.000 más recientes, desde
   mediados de agosto, rinde igual que con todo. Desde ahí, más partidas para ajustar no mejoran
   nada. Lo que limita son las variables, no la cantidad de datos, y es consistente con que cada
   salto del camino haya venido de una variable nueva (§9.S.42).
2. **Importa lo reciente.** Entre 10.000 y 120.000 partidas, las recientes le ganan a las mismas
   cantidades al azar (+0,003 a +0,005 entre 20.000 y 40.000). El juego cambia: pesos aprendidos
   con partidas del mismo meta que V sirven más. Por debajo de 10.000 se invierte: 2.000-5.000
   partidas de uno o dos días son menos variadas que las mismas al azar.
3. **El draft necesita muchos datos.** Hasta 40.000 partidas, el híbrido y solo jugadores son
   iguales. El draft recién suma con 80.000 o más (+0,0006 a +0,0016), porque son 851 columnas
   campeón × línea que necesitan muchas partidas por par. Encaja con §9.S.47, donde el draft no
   sumaba en los bloques tempranos, entrenados con menos partidas.
4. **Para reentrenar** (demo en vivo, datos posteriores al 27/09): alcanza con las últimas ~80.000
   partidas, unos 40 días con el ritmo de recolección de agosto y septiembre. Esto vale para la
   logística. La red de grupos del modelo de 2 expertos no se midió y podría necesitar más.

#### S.49. El paso del mapa de LP con modelos aprendidos (2026-11-15)

**Pedido del usuario.** Probar si modelos mejores mejoran el mapa de LP de §9.S.43. Script:
`scripts/analysis/mapa_lp_modelos.py --lp-vivo RUTA` (~5 min). No modifica
`src/features/lp_mecanica.py`: usa su paso (`_paso_esperado_escalar`) y su ajuste
(`ajustar_parametros_las`) como referencia.

**Qué se mejora.** El mapa avanza partida a partida, y en cada una suma un paso:
$\pm k\,(1 \pm p_{\text{doble}}) + \delta(\text{gap})$, con
$\text{gap} = \text{LP del lobby} - \text{LP propio}$ como proxy de MMR − LP. Acá se predice el
cambio real de LP de cada partida ($dz$) con modelos aprendidos. Las variables son todas
disponibles en el mapa en el momento del paso:

* LP propio, resultado, gap, conocidos del lobby, rol, duración, si estaba en el piso de Master;
* victorias y derrotas de la temporada antes de la partida (en el mapa salen de la foto 1 más las
  partidas observadas);
* racha y neto de las últimas 3 y 10 partidas.

**Datos.** Los eventos `enlazada` de `lp_vivo.db`, en la copia de Drive que llega hasta el 01/11:
22.720 de LAS (5.171 jugadores) y 105.837 de BR, EUW y NA, todos del 30/10 al 01/11 y todos
Master+. Para evaluar con jugadores no vistos se usa GroupKFold por jugador (5 pliegues): con 2,5
días de datos, un corte temporal no deja nada. El paso mecánico se ajusta con los mismos
pliegues.

**Paso a paso (LAS, jugadores no vistos).**

| Modelo | MAE | RMSE | MAE en partidas normales (92,7 %) | MAE fuera de lo normal | Sesgo |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Constante ±20 | 2,85 | 6,33 | 1,44 | 20,84 | −1,48 |
| Paso mecánico | 3,04 | 5,84 | 1,94 | 17,12 | +0,42 |
| HGB, pérdida absoluta (mediana) | **2,25** | 5,82 | 1,19 | 15,82 | −0,88 |
| HGB, pérdida cuadrática (media) | 2,75 | **5,52** | 1,72 | 15,92 | **+0,02** |
| Mecánico + HGB (media) del residuo | 2,74 | 5,51 | 1,70 | 16,11 | +0,01 |
| HGB (media) con los 4 servidores | 2,76 | 5,53 | 1,72 | 16,11 | −0,04 |

*Normal* es una victoria de +10 a +31 o una derrota de −5 a −30. El resto son victorias dobles,
el piso de Master, perdones y descensos, y concentran casi todo el error.

**Acumulado.** Sobre 500 tramos contiguos de ≥ 10 partidas de jugadores de test, se recorre
desde el primer LP real con el paso de cada modelo, como hace el mapa causal:

| Modelo | MAE | RMSE | p90 | Δ MAE por tramo vs mecánico [IC 95 %] |
| :--- | :---: | :---: | :---: | :---: |
| Paso mecánico | 12,18 | 17,11 | 26,15 | — |
| HGB (mediana) | 11,63 | 19,65 | 27,90 | −0,76 [−1,68; +0,15] |
| HGB (media) | 11,66 | 17,16 | 25,16 | **−0,58 [−1,13; −0,06]** |
| Mecánico + HGB (media) del residuo | 11,65 | 17,19 | 24,96 | **−0,63 [−1,13; −0,13]** |
| HGB (media) con los 4 servidores | 11,72 | 17,30 | 25,81 | −0,55 [−1,07; −0,03] |

**Lectura.**

1. **Paso a paso, el modelo aprendido le gana a la fórmula, pero conviene que prediga la media.**
   El HGB con pérdida absoluta tiene el menor error por partida, pero predice la mediana. Como la
   distribución es asimétrica (victorias dobles), queda sesgado en −0,9 LP por partida, y al
   acumular el RMSE empeora. Para un mapa que suma pasos, lo que importa es no tener sesgo, por lo
   que el criterio correcto es la media (pérdida cuadrática). El paso mecánico pierde incluso
   contra la constante ±20 en MAE por partida porque suma el valor esperado de la victoria doble
   en todas las victorias, pero es casi insesgado.
2. **Al acumular, la mejora es real pero chica:** −0,6 LP de MAE por tramo (−5 %), significativa,
   sin cambio en el RMSE. La mejor versión es *mecánico + corrección aprendida*: conserva la
   estructura (piso, doble, banco) y el HGB corrige lo que la fórmula no ve.
3. **Qué aprende el modelo:** después del resultado, el LP propio (0,80 LP de aumento del MAE al
   permutarlo), el gap (0,28), la duración (0,18) y las partidas de la temporada (0,15). Racha,
   neto reciente y winrate de la temporada casi no aportan: el gap ya resume el MMR − LP.
4. **Otros servidores no ayudan a LAS** (−0,55 contra −0,58). Confirma el criterio de §9.S.43 de
   ajustar con LAS.
5. **Por qué no cambia mucho el mapa:** el error que queda está en el 7,3 % de partidas fuera de
   lo normal, que ningún modelo anticipa. En el cierre foto 1 → foto 2 (MAE 63 LP, §9.S.43) pesan
   mucho más las partidas no observadas y el decay que la precisión del paso: 0,6 LP en 10
   partidas es poco frente a 63.

**¿Cabe en el mapa tal como está? (pedido del usuario: probar quién gana con el mapa nuevo).** El
suavizado en grilla de `lp_mecanica.py` (`suavizar_en_grilla`) solo recibe, por partida, el LP
(la grilla), el gap, el resultado, el rol y los conocidos del lobby. No recibe la duración ni las
victorias y derrotas de la temporada. Las correcciones que caben ahí no mejoran el mapa:

| Corrección | Δ MAE por tramo vs mecánico [IC 95 %] |
| :--- | :---: |
| Simétrica $c(	ext{LP}, 	ext{gap})$ (mismo corrimiento al ganar y al perder, como el δ de Riot) | −0,22 [−0,57; +0,15] |
| $c(	ext{LP}, 	ext{gap}, 	ext{resultado}, 	ext{rol}, 	ext{conocidos})$ | −0,11 [−0,54; +0,31] |
| Con todas las variables (duración, temporada, racha) | −0,63 [−1,13; −0,13] |

La parte útil de la corrección está en la duración y en las partidas de la temporada. Para
usarla, `estimar_mapa_tramo` y `suavizar_en_grilla` tendrían que recibir esas columnas. Por eso
no se rehízo la evaluación de quién gana de §9.S.46: con las correcciones que caben, el mapa no
cambia de forma medible. Además, en §9.S.46 el LP del mapa suma solo +0,0003 sobre el híbrido;
lo que suma es la actividad y tener foto.

**Decisión: no se aplica a `lp_mecanica.py`** (es un módulo del otro agente, y la ganancia es
chica). Queda registrada la versión que convendría: mecánico + HGB (media) del residuo. *Mayor
ganancia esperable:* modelar las partidas no observadas por jugador (hoy `tasa_oc` es global) y
su resultado. *Limitaciones:* 2,5 días de datos y solo Master+, así que no se evalúa el paso en
Diamond (66 % de los lobbies de `lol_las` es Master, 28,5 % Diamond).

#### S.50. Validación por bloques y curva de aprendizaje con el mapa de LP (2026-11-15)

**Pedido del usuario.** Repetir la validación por bloques (§9.S.47) y la curva de aprendizaje
(§9.S.48) sumando al híbrido las variables del mapa de LP de §9.S.46. Script:
`historial/ganador_primeros/ganador_con_mapa.py` (~5 min). Lee la caché del mapa causal
(`data/analysis/mapa_lp_causal_v.pkl`) y arma las variables igual que `ganador_mapa_lp.py`, en
representación "equipo" (media azul − rojo de las 5 líneas). Antes de la foto 1 valen 0, y se
estandarizan con las filas de entrenamiento posteriores a ella.

**Variantes.**

* *Actividad:* si tiene foto y el log de partidas con nivel desde el 04/09.
* *Nivel (foto):* el nivel del lobby partida a partida con el LP de la foto (`lobby.py`).
* *Nivel (mapa) + mapa:* el nivel del lobby con el LP del mapa, más el LP del mapa, el cambio
  desde la foto, la incertidumbre, si tiene foto y la brecha nivel − LP. Es la mejor variante de
  §9.S.46.

**Bloques.** El mapa existe solo desde el 04/09, y en los bloques 1-3 de §9.S.47 sus variables
valdrían 0. Por eso se parte el tramo de ajuste posterior a la foto (A, 04/09 → 16/09, 26.934
partidas) en 4 sub-bloques por fecha. A2, A3 y A4 (6.733 partidas cada uno) se predicen hacia
adelante, cada uno entrenado con todo el ajuste anterior (antes del 04/09 más los sub-bloques
previos). Se aplica la regla de §9.S.47, que con 3 bloques exige mejorar en los 3.

| Variante | A2 (07-10/09) | A3 (10-13/09) | A4 (13-16/09) | Media [IC 95 %] | Decisión | V [IC 95 %] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| + actividad | −0,0006 | +0,0025 | +0,0035 | +0,0018 [+0,0008; +0,0026] | no concluyente | +0,0031 [+0,0017; +0,0044] |
| + nivel (foto) | −0,0009 | +0,0019 | +0,0034 | +0,0015 [+0,0007; +0,0022] | no concluyente | +0,0028 [+0,0015; +0,0039] |
| + nivel (mapa) + mapa | −0,0020 | +0,0024 | +0,0052 | +0,0018 [+0,0005; +0,0030] | no concluyente | **+0,0041** [+0,0026; +0,0056] |

El híbrido sin mapa da 0,6358 / 0,6417 / 0,6266 en A2-A4 y 0,6413 en V.

**Curva de aprendizaje** (las N partidas más recientes antes de V; V con y sin nivel (mapa) +
mapa):

| Partidas | Desde | % posterior a la foto | Sin mapa | Con mapa | Δ [IC 95 %] |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 5.000 | 14/09 | 100 % | 0,5983 | 0,6002 | +0,0018 [+0,0004; +0,0031] |
| 10.000 | 12/09 | 100 % | 0,6172 | 0,6187 | +0,0015 [+0,0003; +0,0027] |
| 20.000 | 07/09 | 100 % | 0,6320 | 0,6336 | +0,0017 [+0,0005; +0,0028] |
| 26.934 | 04/09 | 100 % | 0,6349 | 0,6370 | +0,0021 [+0,0009; +0,0032] |
| 40.000 | 30/08 | 67 % | 0,6383 | 0,6410 | +0,0026 [+0,0013; +0,0039] |
| 80.000 | 13/08 | 34 % | 0,6410 | 0,6441 | +0,0031 [+0,0018; +0,0043] |
| 160.883 | 06/11/2024 | 17 % | 0,6413 | 0,6454 | +0,0041 [+0,0026; +0,0056] |

**Lectura.**

1. **El mapa suma, pero su aporte crece con el tiempo desde la foto.** Falla solo en A2, de 3 a 6
   días después de la foto, y sube en A3 y A4 y en V. Hay dos causas, que van juntas: (a) en A2
   los pesos del mapa se aprendieron con solo 6.733 partidas posteriores a la foto (A1); (b) las
   variables del mapa son "desde el 04/09", así que pocos días después casi todos tienen pocas
   partidas con nivel y el LP del mapa todavía es casi la foto. Por la regla estricta queda **no
   concluyente**; la media es significativa en las tres variantes, y la falla es la esperable en
   el primer tramo.
2. **En los bloques, lo que suma es la actividad.** Tener foto y cuántas partidas jugó desde el
   04/09 dan lo mismo que el mapa entero (media +0,0018 las dos). El LP del mapa y la brecha
   agregan algo solo en V (+0,0041 contra +0,0031). Coincide con §9.S.46: dos tercios del aporte
   son actividad, no LP.
3. **La curva con mapa mantiene la forma de §9.S.48.** El aporte del mapa crece con más partidas
   de ajuste (+0,0018 con 5.000, +0,0041 con todo), aunque la proporción posterior a la foto baje.
   Los pesos del resto del modelo mejoran y el mapa corrige mejor lo que queda.
4. **Para la demo en vivo,** las variables dependen de cuánto pasó desde la última foto. En vivo
   se tiene el LP real de league-v4 en el momento, que reemplaza a foto + mapa. La actividad
   reciente se puede calcular siempre.

**Decisión.** La actividad y el nivel del lobby entran como candidatos fuertes. Fallan solo en el
primer sub-bloque, por una causa identificada, y el modelo de 2 expertos ya usa el nivel del lobby.
El LP del mapa agrega poco sobre la actividad y queda como opcional. Para el test conviene
confirmarlo en el modelo de 2 expertos, en la PC con GPU.

#### S.51. V solo con partidas de Master+, y acierto además de AUC (2026-11-15)

**Pedido del usuario (regla desde hoy).** La V de quién gana queda solo con partidas de Master+.
Las de Diamond de la ventana de V pasan al último sub-bloque del ajuste (A4). En cada resultado
se reporta el acierto además del AUC. Se implementa en `src/models/validacion_temporal.py`
(`es_master_plus`, `particion_v_master`, con test), y queda anotado en `ROADMAP.md`, Etapa 6.

**Definición de partida de Master+.** La mediana del tier de los 10 jugadores es ≥ Master, es
decir, al menos 6 de 10 son Master, Grandmaster o Challenger (los jugadores sin dato no cuentan).
*Alternativa descartada:* el tier más común del lobby. Con 5 y 5 queda indefinido, y la mediana
es más estricta en el borde. *Aviso:* el tier sale de la foto del padrón (§9.S.23). Se usa solo
para elegir qué partidas se evalúan, no como variable, así que no filtra el resultado al
modelo, pero el corte refleja el rango de la foto y no el del momento de la partida.

**Partición resultante.** De las 26.934 partidas de la ventana de V (16/09 → 27/09), quedan
**16.428 de Master+ en V** y **10.506 de Diamond pasan a A4**. A4 queda con 17.240 partidas
(13/09 → 27/09). *Consecuencia:* el modelo que se evalúa en V se entrena también con las partidas
de Diamond de los mismos días. No hay fuga del resultado de las partidas de V (las variables son
causales y las filas son otras), pero sí información de la misma época, que antes no tenía.

**§9.S.50 con la nueva partición** (`ganador_con_mapa.py`, mismas variantes; Δ AUC y acierto):

| Variante | Δ A2 / A3 / A4 | Media [IC 95 %] | Acierto A2 / A3 / A4 | **AUC V** | **Acierto V** | Δ V [IC 95 %] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Híbrido (referencia) | — | — | 60,0 / 60,1 / 58,9 % | **0,6500** | **60,5 %** | — |
| + actividad | −0,0006 / +0,0025 / +0,0033 | +0,0017 [+0,0007; +0,0026] | 59,8 / 59,9 / 59,3 % | 0,6526 | 61,1 % | +0,0026 [+0,0015; +0,0036] |
| + nivel (foto) | −0,0010 / +0,0022 / +0,0030 | +0,0014 [+0,0008; +0,0021] | 59,7 / 60,0 / 59,2 % | 0,6521 | 60,9 % | +0,0021 [+0,0005; +0,0035] |
| + nivel (mapa) + mapa | −0,0022 / +0,0024 / +0,0042 | +0,0015 [+0,0004; +0,0025] | 59,6 / 60,2 / 59,3 % | **0,6533** | **61,1 %** | +0,0033 [+0,0018; +0,0047] |

Las tres variantes quedan **no concluyentes** por la regla (fallan en A2). Las conclusiones de
§9.S.50 no cambian.

**Curva de aprendizaje (V Master+).** Ahora "las N más recientes" incluyen las partidas de Diamond
de la ventana de V:

| Partidas | Desde | AUC sin / con mapa | Acierto sin / con mapa | Δ AUC mapa [IC 95 %] |
| :---: | :--- | :---: | :---: | :---: |
| 5.000 | 22/09 | 0,6030 / 0,6008 | 57,2 / 57,1 % | −0,0022 [−0,0044; −0,0004] |
| 10.000 | 16/09 | 0,6191 / 0,6179 | 58,1 / 58,1 % | −0,0012 [−0,0026; +0,0004] |
| 20.000 | 12/09 | 0,6347 / 0,6353 | 59,9 / 59,8 % | +0,0006 [−0,0008; +0,0020] |
| 40.000 | 03/09 | 0,6450 / 0,6466 | 60,4 / 60,6 % | +0,0015 [+0,0001; +0,0029] |
| 80.000 | 17/08 | 0,6503 / 0,6523 | 60,7 / 60,7 % | +0,0020 [+0,0005; +0,0033] |
| 171.389 (todo) | 06/11/2024 | 0,6500 / 0,6533 | 60,5 / 61,1 % | +0,0033 [+0,0018; +0,0047] |

**Lectura.**

1. **Master+ se predice mejor:** el híbrido da 0,6500 / 60,5 % en V Master+, contra 0,6413 /
   59,9 % en la V completa. Coincide con §9.S.42, punto 5 (los lobbies de Diamond son los más
   difíciles).
2. **Con el mapa, el mejor modelo de esta máquina llega a 0,6533 de AUC y 61,1 % de acierto en
   Master+.** El aporte del mapa en V baja un poco (+0,0033 contra +0,0041), y los bloques dan lo
   mismo.
3. **La saturación sigue en ~80.000 partidas** (0,6503 contra 0,6500 con todo, sin mapa).
4. **Con pocas partidas recientes, el mapa empeora.** Las 5.000-10.000 más recientes son casi todas
   de Diamond, y aprender los pesos del mapa con Diamond para predecir Master+ no sirve. Es un
   motivo más para entrenar con al menos ~40.000 partidas.

Los resultados anteriores a esta sección (§9.S.41 a §9.S.50) se midieron con la V completa y no
se rehicieron, salvo §9.S.50.

**Corrección de fuga (mismo día, pregunta del usuario: «¿hay leakage con la nueva
configuración?»).**

* *El tier que define Master+:* sin fuga. El padrón (`players.updated_at`) es entero la foto 1
  (04/09 22:51), anterior a toda V (16/09 → 27/09).
* *Las partidas de Diamond de la ventana en el entrenamiento de V:* **fuga leve.** El modelo que
  predecía V se entrenaba también con partidas posteriores a muchas de las que predice. El 94,7 %
  de las partidas de V Master+ tiene algún jugador que después juega una de esas partidas. Las
  variables son causales, así que no se filtra el resultado de ninguna partida de V, pero sí
  información del mismo meta y de días posteriores. En el híbrido, sacarlas del entrenamiento
  cambia V de 0,6500 / 60,5 % a 0,6488 / 60,4 % (Δ −0,0012 [−0,0018; −0,0006]); en solo draft,
  −0,0009 (n. s.).

**Regla corregida.** Las partidas de Diamond de la ventana de V quedan en A4 y se evalúan hacia
adelante, entrenando solo con lo anterior. **El modelo que predice V se entrena solo con
partidas anteriores al inicio de V (16/09).** `ganador_con_mapa.py` ya lo aplica, y en la curva
de aprendizaje "las N más recientes" vuelven a ser anteriores a V. Números sin fuga (reemplazan a
las tablas de arriba en la columna V):

| Modelo (V Master+, 16.428 partidas) | AUC | Acierto | Δ AUC vs híbrido [IC 95 %] |
| :--- | :---: | :---: | :---: |
| Híbrido (referencia) | 0,6488 | 60,4 % | — |
| + actividad | 0,6518 | 61,0 % | +0,0031 [+0,0018; +0,0042] |
| + nivel (foto) | 0,6517 | 60,8 % | +0,0029 [+0,0014; +0,0042] |
| **+ nivel (mapa) + mapa** | **0,6533** | **61,1 %** | **+0,0046 [+0,0029; +0,0060]** |

Los bloques A2-A4 no cambian, porque ya se predecían hacia adelante. La curva sin fuga: sin mapa,
0,6027 / 57,4 % con 5.000 partidas, 0,6409 / 60,3 % con 26.934, 0,6488 / 60,5 % con 80.000 y
0,6488 / 60,4 % con todo. Con mapa: 0,6046 / 57,6 %, 0,6436 / 60,4 %, 0,6523 / 60,6 % y
0,6533 / 61,1 %. El aporte del mapa es positivo en todos los tamaños (+0,0017 a +0,0046), y
desaparece el resultado negativo con pocas partidas, que venía de entrenar con Diamond de la
ventana. **El 61,1 % con mapa se sostiene sin la fuga;** el híbrido solo pierde 0,1 punto.

#### S.52. Brainstorming nocturno Claude ↔ Gemini: quién gana en Master+ (2026-11-15)

**Pedido del usuario.** Trabajar toda la noche con Gemini (Antigravity) buscando mejoras del modelo de
quién gana con señales significativas en Master+. Claude es el agente principal y hace la revisión
final; los dos revisan lo del otro. El canal de trabajo es `docs/BRAINSTORM_NOCHE.md`, con reglas,
bitácora y revisiones cruzadas. Todo con la partición de §9.S.51: V = 16.428 partidas de Master+; el
modelo de V se entrena solo con partidas anteriores al 16/09; los sub-bloques A2-A4 (Master+) se
predicen hacia adelante.

**Herramientas.** `scripts/analysis/brainstorm_base.py` (`cargar`, `ajustar_predecir`, `evaluar`,
`comparar`, `extras_referencia(D, escalon)`) y un script por idea: `brainstorm_NN_*.py` (Claude) y
`brainstorm_g_NN_*.py` (Gemini).

**Escalera aceptada** (logística C = 0,05 sobre draft campeón × línea + 85 variables por jugador + mapa
de LP; cada escalón medido contra el anterior en V Master+, con IC por bootstrap pareado y los 3
sub-bloques):

| Escalón | Quién | V AUC | Acierto | Δ [IC 95 %] | Bloques |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Híbrido + mapa, todo el entrenamiento (§9.S.51) | — | 0,6531 | 61,1 % | — | — |
| 1. Entrenar solo con Master+ | Claude | 0,6557 | 61,3 % | +0,0026 [+0,0012; +0,0038] | 3/3 |
| 2. + margen por (jugador, campeón) y por (jugador, línea) (§9.S.29) | Claude | 0,6614 | 61,6 % | +0,0058 [+0,0028; +0,0085] | 3/3 |
| 3a. + nivel histórico (emparejamiento con el LP de la foto 1, todo el historial, §9.S.31) | Claude | 0,6642 | 61,8 % | +0,0028 [+0,0017; +0,0039] | 3/3 |
| 3b. + W/L de temporada de la foto 1, actualizado con las partidas observadas | Gemini | 0,6663 | 62,0 % | +0,0021 [+0,0004; +0,0034] | 3/3 |
| **4. + contracruces de campeones por línea y sinergias causales** | Gemini (verificado por Claude) | **0,6674** | **61,9 %** | +0,0011 [+0,0006; +0,0017] | 3/3 |

| 4b. + winrate por línea / campeón-línea y caída por autofill (ROL_WR) | Gemini (verificado: Diamond-V +0,0014 [+0,0001; +0,0025]; IC en V toca 0) | 0,6682 | 61,9 % | +0,0009 [+0,0000; +0,0016] | 3/3 |
| **5. Ensamble: 0,65 × logit de la logística del escalón 4 + 0,35 × red NeuralAC en GPU** (Wide & Deep por jugador + duelo directo por línea + cruce 5 × 5; 3 semillas) | Gemini (verificado por Claude) | **0,6696** | **62,0 %** | +0,0022 [+0,0014; +0,0030] | 3/3 |
| **6. Logística: 4b + Kalman-Elo causal (resultado 70 % + probabilidad @15 30 %, con tirón del lobby), tempo y objetivos (primera sangre/torre, cambio de trinket, sesgo temprano/tardío por jugador y campeón-línea) y economía 15→20** | Gemini (verificado por Claude) | **0,6707** | **62,2 %** | +0,0024 [+0,0010; +0,0040] (bloques IC [+0,0004; +0,0035]) | 3/3 |
| **7. Logística: 6 (sin mapa) + modelo por línea: oro @15 de cada línea predicho antes de la partida con HGB causal (hacia adelante por tramos; §9.S.36 rehecho)** | Gemini (verificado por Claude: placebo 0, Diamond-V +0,0012 [+0,0005; +0,0020]) | **0,6722** | **62,5 %** | +0,0012 [+0,0006; +0,0019] (bloques [+0,0009; +0,0024]) | 3/3 |
| **8. Ensamble sobre el escalón 7: logística + red NeuralAC (6 semillas) + red de atención (5 semillas) + CatBoost residual, logits 45/25/15/15** | Gemini (verificado por Claude: robusto a los pesos; con pesos iguales +0,0014 [+0,0001; +0,0027], Diamond-V +0,0028) | **0,6735** | 62,4 % | +0,0014 [+0,0003; +0,0024] (bloques [+0,0003; +0,0029]) | 3/3 |
| **9. Logística: 7 + margen esperado @10/@15 por par de campeones en cada línea (oro, XP, CS; empírico-bayesiano causal)** | Gemini (verificado por Claude: placebo ≈ 0, Diamond-V +0,0019 [+0,0007; +0,0030]; bloques [−0,0001; +0,0018]) | **0,6739** | **62,5 %** | +0,0018 [+0,0008; +0,0027] | 3/3 |
| 10. Ensamble sobre el escalón 9 (logística + red NeuralAC + red de atención + CatBoost residual reentrenados con 18 canales por jugador; logits 50/25/25 o 60/20/10/10) | Gemini (verificado por Claude: en V +0,0008 a +0,0010 con cualquier peso, IC > 0 en 2 de 5; Diamond-V +0,0016 a +0,0025, todos IC > 0) | 0,6749 | 62,7 % | +0,0010 [+0,0001; +0,0019] (marginal) | 3/3 |

Mejora total de la noche: **+0,0165 de AUC y +0,9 puntos de acierto** sobre la referencia de §9.S.51
(+0,0143 con la versión simple, solo logística, del escalón 4). Escalón 5, verificación: el peso 0,35
estaba fijado desde antes (`brainstorm_g_05`). Es robusto al peso (0,15 a 0,50: +0,0012 a +0,0026,
todos con IC > 0) y se replica en las Diamond de la ventana de V (+0,0013 [+0,0003; +0,0023]). La red
sola no es significativa (+0,0016); suma por diversidad con la logística. No mueve el acierto,
mejora el ordenamiento. Escalón 6, verificación (Claude): estados leídos antes de actualizar con la partida (orden (t, id)), el modelo
@15 usado como objetivo blando se ajustó solo con partidas anteriores al 04/09, y todo lo de la partida actual
(resultado, duración, oro @15/@20) entra solo en la actualización posterior. Placebo (filas permutadas): −0,0002 en
V y negativo en bloques; Diamond-V +0,0022 [−0,0001; +0,0042] (misma dirección). El sub-bloque CLUTCH (rendimiento
contra lo esperado a los 15) no aporta y se deja afuera. La logística del escalón 6 supera al ensamble del 5 sin GPU.
*Lección metodológica (placebo de ROL_WR):* columnas al azar también mejoraron
en los 3 bloques (+0,0002). Con efectos de ~0,001, "3 de 3 bloques" solo no alcanza; para aceptar algo así
se exige además el IC en V y la réplica en las Diamond de la ventana.
Con el escalón 2, el 20 % de partidas más seguras se acierta al 75,0 % [73,5; 76,5] y el 10 % al
76,9 %, con calibración ECE 0,006.

**Ganancia total y candidato final.** El candidato final es el escalón 9 (logística, sin el mapa de LP;
`scripts/analysis/brainstorm_final.py`, que reproduce todo; 118.064 partidas de entrenamiento y 1.504 columnas):
V Master+ **0,6739 / 62,5 %**, log loss 0,6443, ECE 0,0071, cobertura 5 % → 80,9 %, 10 % → 78,9 %, 20 % → 76,3 %. La comparación
que sigue (con IC) es la del escalón 6.
En V Master+ da **AUC 0,6709 y acierto 62,33 %** (log loss 0,6458, ECE 0,0069). En la fracción más segura acierta
81,1 % en el 5 %, 78,7 % en el 10 % y 76,2 % en el 20 %. Contra la referencia de §9.S.51: **ΔAUC +0,0178
[+0,0137; +0,0218]** y **Δacierto +1,26 pp [+0,68; +1,82]** (bootstrap pareado, B = 1000). McNemar da p < 0,0001:
1.269 partidas solo las acierta el final y 1.062 solo las acertaba la referencia. El mapa de LP deja de aportar
(0,6707 con mapa contra 0,6709 sin mapa).

**Por qué funcionan.**

* *Solo Master+:* los pesos se aprenden con la población que se predice. Ponderar Master+ ×2 o ×4 da
  la mitad de la mejora, y entrenar con cortes más laxos (tier medio ≥ 0,3-0,9) o con todo empeora.
* *Margen por campeón y línea:* rendir con este campeón y en esta línea no es lo mismo que rendir en
  general. Estaba en el modelo del compañero y no en el de esta máquina.
* *Nivel histórico:* el emparejamiento revela el MMR oculto. La media del LP de la foto 1 de los otros
  9 en todas las partidas previas lo estima mejor que el mapa, que solo usa partidas desde el 04/09.
* *W/L de la foto:* a igual LP, el winrate y el volumen de la temporada miden el rezago entre el MMR
  oculto y el LP visible (escaladores contra estancados). Gemini lo confirmó en vivo con `lp_vivo.db`
  en 4 servidores: el LP real solo da AUC 0,505, y con W/L de temporada 0,549 (+0,045 [+0,013; +0,078]).
* *Matchups:* contracruces campeón contra campeón por línea y sinergias (ADC + soporte, top + jungla,
  mid + jungla), empírico-bayesianos y causales. Suman sobre todo en los lobbies que el modelo
  predecía peor: sin estrenos +0,0019 y de veteranos +0,0016. Un placebo (matchups de otra partida al
  azar) da +0,0000.

**Chequeo de selección (Claude).** Esta noche se probaron unas 40 variantes contra la misma V Master+.
Las 10.506 partidas de Diamond de la ventana de V nunca se usaron para decidir, así que se
reevaluó ahí cada escalón. El margen por campeón y línea (+0,0045 [+0,0015; +0,0075]) y los matchups
(+0,0009 [+0,0000; +0,0016]) se replican; el W/L de la foto va en la misma dirección (+0,0021
[−0,0004; +0,0045]). El nivel histórico no se replica en Diamond (−0,0002), aunque en Master+ se
sostiene en los 3 sub-bloques, que son independientes de V: parece propio de la cima. Entrenar solo con
Master+ perjudica a Diamond, como era de esperar. **Las mejoras de información no son un artefacto
de haber elegido mirando V.**

**Probado y descartado** (Δ en V sobre la referencia de su momento):

| Idea | Quién | Resultado |
| :--- | :--- | :--- |
| Asimetría de LP entre equipos; hora × habilidad; duos causales | Claude | −0,0010; +0,0002; −0,0006 (los duos mejoran en bloques pero no en V) |
| Margen en lobbies del mismo nivel; fuerza por jugador (one-hot) | Claude | −0,0015; −0,0076 |
| Peso por antigüedad (τ = 15/30/60 d); C; mapa por línea | Claude | −0,0038…−0,0002; ±0,0002; −0,0029 |
| W/L de la foto como máx/mín/desvío o por línea | Claude | −0,0004; −0,0005 |
| Modelos por segmento (estreno / veteranos) | Claude | −0,0024 a −0,0119 |
| Forma reciente con el campeón y "óxido" | Claude | +0,0004 (n. s.) |
| Encogimiento del margen por campeón (k = 5 / 20); margen por (campeón, línea) | Claude | −0,0002; −0,0003; −0,0007 |
| Sacar del entrenamiento rendiciones antes del 20 / ponderar por duración | Claude | −0,0016; −0,0021 |
| Historial cara a cara del rival de línea | Claude | demasiado escaso: solo el 7,5 % de los pares de línea de V se cruzó antes |
| Agregados de habilidad max/min/std; autofill y smurf; lado × nivel del lobby | Gemini | +0,0008 (n. s.) sobre el escalón 3 |
| Timeline @10/@20; margen ajustado por rival; volatilidad; rendición temprana | Gemini | −0,0007 a +0,0005 |
| Red Wide & Deep en GPU, sola y en ensamble 35/65 | Gemini | +0,0007 sola; +0,0005 sobre la logística |
| Composición del draft; Poisson; parche; reencuentros en ≤ 90 min | Gemini | +0,0002; −0,0002; 0,0000; +0,0003 |

**Más descartes de la madrugada** (sobre el escalón 6, regla de 04:33: IC de bloques y de V, o réplica en Diamond-V):
L2 diferencial por bloque (+0,0003, ajuste fino opcional); cruce de tempo entre equipos E·T (≈ 0); parche exacto,
Elo individual @15 y cambio de rol (≈ 0; `participant_id` ya es la posición); F1/nivel no lineales (empeoran); Elo
por rol (V +0,0005 pero bloques con IC en 0 y Diamond −0,0006); composición ampliada y draft factorizado (≈ 0);
draft por recencia (−0,0016 a −0,0023); pool reciente (bloques −0,0010); ítems y trinket (≈ 0). *Hallazgo
descriptivo:* los equipos casi full-AD contra tanques rinden peor de lo que dice el draft aditivo (`full_ad_risk`
negativo en todos los cortes), pero suma solo +0,0004 y no entra.

**Techo de la información pre-partida.** Desde el escalón 9, once ideas seguidas dan ≈ 0 o no pasan la regla:
maestría por campeón, contexto @15 y post-15 en HGB, heterocedasticidad, residuo económico, matchups 2 contra 2, Kalman por
línea @15 (V +0,0010, pero bloques y Diamond-V con IC en 0), Kalman post-15 y de combate, reencuentros, tilt y sesión,
grafo LSQR y progresión 15→20. La información previa a la partida que se puede extraer de este dataset para Master+
parece cerca del techo con el escalón 9. Lo que queda son milésimas que V ya no distingue del ruido de selección.
El próximo salto probablemente requiere datos nuevos (runas de toda V, LP y W/L en vivo de league-v4, maestría de
campeón), no más transformaciones de los mismos datos.

**Diagnóstico de residuos (cierre).** Sobre A2-A4 Master+ (13.860 partidas, sin tocar V), de ~75 sondas contra el
residuo $y - p$ del escalón 9 la mayor da $|z| = 2{,}96$, lo esperable por azar con esa cantidad de pruebas: no queda
estructura explotable. En esos bloques el modelo es algo sobreconfiado en los extremos, y bajar C a 0,01 los mejora
(+0,0009), pero en V empeora (−0,0003): C = 0,05 se queda. *Descriptivo:* el AUC es 0,761 en partidas de menos de
25 min, 0,658 entre 25 y 32 y 0,566 en las de 32 o más. Las partidas largas son casi impredecibles antes de jugar.
Los atributos estáticos del campeón (perfil, Data Dragon) funcionan como prior de los one-hot raros del draft: sacarlos
todos cuesta −0,0010 [−0,0017; −0,0004], así que no se podan.

**Predicción en dos tiempos sobre el escalón 9** (`historial/en_vivo/brainstorm_39_en_vivo.py`). Segunda etapa:
logística sobre el logit previo del escalón 9 y el estado del equipo al minuto $m$ (Δoro, Δxp, Δkills), ajustada con las
predicciones hacia adelante de A2-A4 Master+ y medida en V Master+ (partidas que llegaron al minuto $m$):

| Momento | Solo el estado (AUC / acierto) | Previo + estado | Aporte del previo [IC 95 %] | Acierto en el 10 % / 20 % más seguro |
| :--- | :---: | :---: | :---: | :---: |
| Antes de la partida | — | 0,6739 / 62,5 % | — | 78,9 % / 76,3 % |
| Minuto 10 | 0,7470 / 68,3 % | **0,7774 / 70,4 %** | +0,0304 [+0,0272; +0,0342] | 93,3 % / 88,9 % |
| Minuto 15 | 0,8138 / 73,4 % | **0,8288 / 74,9 %** | +0,0151 [+0,0126; +0,0177] | 97,1 % / 94,0 % |
| Minuto 20 | 0,8584 / 77,4 % | **0,8656 / 78,0 %** | +0,0072 [+0,0057; +0,0088] | 98,9 % / 97,0 % |

El modelo previo suma sobre el estado en los tres momentos, cada vez menos. En el 20 % de partidas más seguras antes de
jugar, si el favorito va adelante en oro al minuto 10 (64,8 % de las veces) se acierta el 85,6 %; si va atrás, el 59,0 %
(a los 15: 87,9 % y 49,6 %). Con el modelo de §9.S.42 el previo sumaba +0,024 al minuto 10; con el escalón 9, +0,030.

*Segunda etapa por líneas* (`historial/en_vivo/brainstorm_40_en_vivo_lineas.py`). Reemplazar el Δoro y Δxp del equipo
por los de cada una de las 5 líneas suma **+0,0022 [+0,0012; +0,0034]** al minuto 10 (0,7796 / 70,7 %) y **+0,0020
[+0,0009; +0,0029]** al 15 (0,8308 / 75,1 %). Al minuto 15, sumar además el oro @15 que el modelo por línea predecía antes
de jugar (equivale a la "sorpresa" de cada línea: oro real − esperado) lleva a **0,8377 / 75,8 % (+0,0088 [+0,0073;
+0,0106] sobre la versión por equipo)**, y al minuto 10 a **0,7842 / 71,2 % (+0,0068 [+0,0051; +0,0085])**: importa cuánto se esperaba de cada línea, no solo cuánto oro tiene. Un HGB sobre
las mismas variables rinde menos que la logística (−0,0031 a −0,0042). Peso del oro por línea (por 1.000 de oro, minuto
15): bot 0,43, jungla 0,42, mid 0,39, top 0,33, soporte 0,15. El oro del soporte vale la tercera parte que el de un carry.
Cobertura y calibración del mejor modelo en vivo (por líneas + esperado): al minuto 10, el 5 % / 10 % / 20 % / 50 % más
seguro se acierta 96,2 / 93,8 / 90,1 / 82,2 % (ECE 0,0106); al minuto 15, 99,3 / 97,4 / 95,0 / 87,7 % (ECE 0,0094).

**Cuánto aportan las predicciones del modelo por línea** (pedido del usuario; `historial/en_vivo/brainstorm_41_aporte_lineas.py`,
V Master+). Las 6 columnas (oro @15 predicho de cada línea y su suma) **solas** dan AUC 0,6019 / 57,7 %. Sobre el modelo
**base** (draft + 85 variables por jugador, 0,6515 / 60,8 %) suman **+0,0108 [+0,0088; +0,0129]** (0,6622 / 61,5 %), con
los 3 bloques a favor (+0,0110). Dentro del modelo **final**, sacarlas cuesta **−0,0009 [−0,0014; −0,0002]** (bloques
−0,0015 [−0,0021; −0,0009]). La diferencia entre +0,0108 y +0,0009 se explica porque el modelo por línea se alimenta del
margen por campeón y por línea y del nivel histórico: sobre la base trae esa información de contrabando, y en el
modelo final esas variables ya están. Su aporte propio es la no linealidad dentro de cada línea, ~0,001. Por línea sola,
para el resultado de la partida: mid 0,555, jungla 0,545, top 0,544, bot 0,541, soporte 0,500.

**Poda del escalón 4.** Sin el one-hot del draft: −0,0028; sin perfil ni Data Dragon: −0,0008.
**Sin el mapa de LP: −0,0002 (n. s.)**: con nivel histórico y W/L de la foto, el mapa ya no aporta. Se
puede sacar para simplificar la demo.

**Dónde sigue fallando** (escalón 3, V Master+): lobbies sin estrenos (AUC 0,6455) y de veteranos
con 50+ partidas previas (0,6533), contra 0,6746-0,6759 en el resto; horario 18-24 h (0,6571) contra
madrugada (0,6853). Cuando nadie estrena y todos están establecidos se pierde la asimetría de
experiencia, y la información que falta es de draft fino y forma.

**Pendiente.** Correr el modelo de 2 expertos del compañero sobre V Master+, con las variables nuevas
(PC con GPU). Recalcular la caché del nivel histórico con `mu0` de `~val0` (efecto −0,0001). Antes del
test, fijar el modelo final en el escalón 4 sin el mapa.
#### S.53. e-means contra K-Means en el clustering de jugadores (2026-11-13)

**Pedido del usuario.** Probar e-means en lugar de K-Means. e-means (Bessa, Douze, Jégou
et al., FAIR, arXiv 2611.09501, noviembre de 2026) viene en FAISS 1.16.0 como
`faiss.contrib.e_means`. Se instaló `faiss-cpu` 1.16.0.

**Qué cambia respecto de Lloyd.** K-Means (Lloyd) reasigna todos los puntos y recalcula
cada centroide como la media exacta en cada iteración. e-means guarda por cluster una suma
$S_j$ y un conteo $N_j$, y los actualiza con lotes de $B$ puntos y una media móvil
exponencial:

$$N_j \leftarrow (1-\alpha_t)\,N_j + \alpha_t\,n_j,\qquad S_j \leftarrow (1-\alpha_t)\,S_j + \alpha_t\,s_j,\qquad c_j = S_j/N_j,$$

donde $n_j$ y $s_j$ son el conteo y la suma de los puntos del lote asignados a $j$. El paso
sigue un recocido coseno, $\alpha_t = \alpha_0\,(0{,}1 + 0{,}9 \cdot \tfrac12(1+\cos(\pi t/M)))$,
y $B$ y $\alpha_0$ salen de leyes de escala en $k$, $n$ y el número de pasadas: no hay que
ajustar nada más que $k$. Con $B = n$ y $\alpha = 1$, cada paso es exactamente una iteración
de Lloyd. Cada 5 lotes, un cluster con $N_j < 0{,}1 \cdot \bar N$ se revive partiendo uno
grande, y al final cada centroide es la media de los puntos que se le asignaron en la última
pasada. Arranca de $k$ puntos al azar, sin k-means++. Los autores reportan la pérdida de 25
pasadas de Lloyd en 1,4–2,6 pasadas, con aceleraciones de 10–45× en GPU, sobre corpus de
millones de vectores y $k$ de miles.

**Comparación (`scripts/analysis/cluster_emeans.py`, `src/models/clustering.py`).** Mismos
datos y variables que la sección 16 de `clustering_jugadores.ipynb`: 14.249 jugadores con
50 partidas o más y firma (eran 6.299 en §9.R: entró volumen), con las 15 variables
estandarizadas. Hay tres escenarios: global, centrado por rol, y uno por cada rol. Se barre
$k = 2\ldots8$ con 10 semillas. Los métodos son `kmeans` (sklearn, 10 arranques k-means++
por semilla), `kmeans1` (un solo arranque, para comparar una corrida contra otra) y
`emeans` (5 pasadas, lote 32, $\alpha_0 \approx 0{,}15$ por las leyes). En los tres, las
etiquetas son el centroide más cercano al final y la SSE se calcula con esas etiquetas.
Resultados en `data/analysis/cluster_emeans.csv`. Es exploratorio: `players.csv` mezcla
train y test, igual que el notebook.

| Escenario, k = 5 | SSE e-means / K-Means | Silhouette (e / K) | ARI e-means contra K-Means | Estabilidad e-means / kmeans1 / kmeans |
| :--- | ---: | ---: | ---: | ---: |
| Global | 1,00002 | 0,174 / 0,174 | 0,99 | 0,98 / 0,99 / 1,00 |
| Centrado por rol | 1,00002 | 0,120 / 0,120 | 0,99 | 0,83 / 0,69 / 0,99 |
| TOP | 1,00008 | 0,141 / 0,141 | 0,97 | 0,96 / 0,90 / 0,97 |
| UTILITY | 1,00011 | 0,152 / 0,152 | 0,99 | 0,73 / 0,68 / 0,99 |

La estabilidad es el ARI medio entre los pares de corridas con distinta semilla.

* **Misma solución.** En los 7 escenarios y todos los $k$, la mejor SSE de e-means queda a
  menos de 0,1 % de la de K-Means, y la partición coincide: ARI ≥ 0,91, salvo TOP con k = 6
  (0,65, dos soluciones casi empatadas en SSE). El silhouette es el mismo a tres decimales.
* **La elección de k no cambia.** Con los dos métodos el silhouette es máximo en $k = 2$
  (0,355 global) y baja de forma monótona (0,174 en $k = 5$). e-means no resuelve la tarea
  abierta de elegir $k$.
* **Corrida contra corrida, e-means es más estable que un K-Means de un arranque** (global
  con k = 7: 0,95 contra 0,74; centrado con k = 5: 0,83 contra 0,69), pero menos que K-Means
  con 10 arranques, que ya es el mejor de 10 corridas.
* **Tiempo.** Con $n \approx 14$ mil, $d = 15$ y $k \le 8$, una corrida de e-means tarda
  0,1–0,7 s, más que un arranque de sklearn (0,01–0,1 s) y lo mismo que 10 arranques. La
  ventaja del paper aparece con millones de puntos y miles de clusters. Acá Lloyd converge en
  pocas iteraciones baratas.
* **Mínimos locales con $k$ chico.** En el test con 4 grupos bien separados, 2 de 5 semillas
  de e-means quedaron con dos centroides en el mismo grupo: arranca de puntos al azar, sin
  k-means++. Por eso `comparar` se queda con la mejor de varias semillas.

**Decisión.** En esta escala e-means es equivalente a K-Means: no mejora la SSE ni la
separación, ni ayuda a elegir $k$. Queda disponible en `src/models/clustering.py`
(`ajustar(X, k, "emeans")`) por si el clustering pasa a nivel de partida (≈2,7 M filas) o a
muchos clusters, donde sí debería rendir. El notebook sigue con K-Means.

*Alternativa descartada:* implementar e-means en numpy a partir del paper. La versión de
FAISS es la de los autores e incluye detalles que el texto no fija del todo (la revivida
partiendo un cluster y el índice del recocido).

#### S.54. Modelo del LP partida a partida con el LP medido en vivo, LAS (2026-11-13)

**Pedido del usuario.** Entrenar un modelo que calcule el LP partida a partida con los datos
de `lp_vivo` (§9.S.22). Por ahora solo LAS. Primero la infraestructura; el uso se decide
después. Es el paso previo al mapa de LP de los jugadores: medir cuánto se equivoca el LP
reconstruido a mitad de camino, algo que con dos fotos no se puede medir (§9.S.21,
"Registro de LP por partida").

**Datos.** `data/lp_vivo/lp_vivo.db`, leída en solo lectura con `src.lp_vivo.cargar_enlazadas`
(la misma función que ahora usa `lp_vivo.py --exportar`). La base tiene 128.557 filas
enlazadas, contra 79.804 del CSV exportado el 01/11. LAS (`la2`): 22.720 filas, 3.654
partidas, 5.171 jugadores, del 30/10 13:49 al 01/11 23:08 (2,4 días). Objetivo:
$\Delta LP = z_{\text{después}} - z_{\text{antes}}$, en la escala continua de `lp_absoluto`
(un ascenso o descenso de tier no es un salto). Son datos posteriores a la foto 2 y de una
base aparte, así que no tocan el test del modelo de draft (antes del 04/09).

**Variables, por niveles acumulados** (`src/features/lp_partida.py`, `NIVELES`):

| Nivel | Agrega | ¿Existe en `lol_las.db`? |
| :--- | :--- | :--- |
| `resultado` | victoria, rango (Challenger, GM, Master, Diamond) | sí |
| `partida` | $z$ de antes; distancia al lobby, a los compañeros y a los rivales medidos ($\bar z_{\text{grupo}} - z$, sin el propio); W/L y winrate de la temporada; duración; posición | sí (el lobby, con LP estimado) |
| `historia` | racha, partidas previas observadas, horas desde la anterior | sí |
| `lp_previo` | último ΔLP de una victoria y de una derrota anteriores | **no**: es un techo |

`lp_previo` usa LP real anterior. Muestra cuánto falta explicar (Riot ajusta la ganancia por
la distancia entre el MMR oculto y el LP, y la ganancia reciente es su mejor reflejo), pero
fuera de `lp_vivo` ese dato no existe, así que no entra al recorrido.

**Modelos.** `regla`: ΔLP medio por (rango, resultado), la $a, b$ de las fotos medida en vivo.
`lineal`: Ridge por separado para victorias y derrotas. `gb`: LightGBM con pérdida de Huber
($\alpha = 8$ LP). `gb_l2`: el mismo con pérdida cuadrática.

**Evaluación** (`evaluar`, 5 folds por jugador con `folds_por_jugador`):

* *Por partida:* MAE, RMSE, $R^2$ y $R^2$ dentro del resultado,
  $R^2_{\text{dentro}} = 1 - \text{SSE} / \sum_{w}\sum_{i: win_i = w}(y_i - \bar y_w)^2$:
  cuánto explica el modelo además de saber si ganó.
* *Recorrido:* una cadena es una racha de partidas seguidas de un jugador en la que W + L
  sube de a 1 y el LP empalma ($z_{\text{antes},i+1} = z_{\text{después},i}$). Se parte del
  $z$ real de la primera partida y se encadenan las predicciones,
  $\hat z_{i+1} = \hat z_i + \hat\Delta(x_i;\hat z_i)$, recalculando con $\hat z$ las
  variables que dependen del LP propio. El error a $s$ partidas de la última medición es el
  error del mapa de LP en ese punto. En LAS hay 4.081 cadenas de 2 o más partidas (21.169
  filas, la más larga de 30).

**Resultados** (`python scripts/analysis/lp_partida_modelo.py`, 41 s; CSV en `data/analysis/lp_partida_*.csv`):

| Modelo | MAE | RMSE | $R^2$ | $R^2_{\text{dentro}}$ | Recorrido: 1 | 2–3 | 4–7 | 8–15 | 16+ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Regla | 3,46 | 6,15 | 0,914 | 0,004 | 3,9 | 7,6 | 12,2 | 18,3 | 27,5 |
| Lineal, `partida` | 3,27 | 5,91 | 0,921 | 0,081 | 3,7 | 6,9 | 10,9 | 16,0 | 23,6 |
| GB, `partida` | **2,43** | 5,59 | 0,929 | 0,176 | **2,8** | **5,2** | **8,8** | **13,6** | 21,9 |
| GB L2, `partida` | 2,76 | 5,52 | 0,931 | 0,196 | 3,2 | 5,8 | 9,5 | 14,1 | **21,6** |
| GB, `historia` | 2,43 | 5,56 | 0,930 | 0,184 | 2,8 | 5,2 | 8,9 | 13,7 | 22,1 |
| GB, `lp_previo` (techo) | 2,00 | 5,37 | 0,934 | 0,239 | — | — | — | — | — |

* **Las variables de la partida sirven, a diferencia de la primera medición de §9.S.22**
  ($R^2_{\text{dentro}} \approx 0$ con 618 eventos). Con 22.720 filas, el GB explica el 18 %
  de lo que el resultado no explica, y baja el MAE de 3,46 a 2,43 LP por partida.
* **La historia observada casi no suma** (2,43 contra 2,43). El LP previo real sí (2,00): la
  parte del ΔLP que depende del MMR oculto no se ve en las variables de la partida.
* **El error del mapa crece con la distancia a la última medición:** 2,8 LP a una partida,
  ~9 a 4–7 y ~14 a 8–15. Crece más rápido que $\sqrt{s}$, así que no es solo ruido
  independiente que se suma: hay un componente persistente por jugador (el MMR oculto). Como
  referencia, la recorrida entre fotos de §9.S.21 erra 133 LP en la foto 1.
* **La pérdida importa para encadenar.** Con L1 (la primera versión) el GB predecía la
  mediana (+20 al ganar), pero la media es +21,6 por la cola de ganancias grandes (hasta +60).
  Eso da un sesgo de −0,86 LP por partida, que en el recorrido suma −10 LP a las 8–15 partidas
  y −23 a las 16 o más (26,4 de MAE ahí). L2 no tiene sesgo (−0,002), pero erra más por
  partida (2,76). Huber queda en el medio: sesgo −0,35, MAE 2,43 y el menor error total en el
  recorrido (7,09 contra 7,32 con L1 y 7,62 con L2).

**Límites.**

* *Pocos días y solo Master+.* 2,4 días de LAS, sin Diamond (el ladder solo se lee para
  Master+). Solo 40 jugadores de LAS tienen 20 partidas o más.
* *La cola sin explicar sigue ahí:* 1,1 % de filas con $|\Delta LP| > 40$ (§9.S.22).
  `mae_tipico` excluye esas filas.
* *El lobby está sesgado hacia Master+,* porque solo se miden jugadores de Master+ (§9.S.22).
  En `lol_las.db`, el lobby vendría del LP estimado.
* *Validación interna de LAS.* Ver si un modelo de LAS sirve en otros servidores, o si suma
  entrenar con los cuatro (§9.S.22 midió que el servidor pesa), queda para después:
  `--plataformas la2,br1,na1,euw1`.

**Alternativas descartadas.**

* *Predecir $z_{\text{después}}$ directamente:* es casi $z_{\text{antes}}$, así que el
  $R^2$ engaña y el modelo puede copiar la entrada. Predecir la diferencia aísla lo que la
  partida mueve.
* *Folds por partida:* el mismo jugador quedaría en train y validación, y su MMR oculto se
  filtraría. Se usan folds por jugador, como en el elo al revés.
* *Usar el CSV exportado:* tiene un 38 % menos de filas que la base. Leer la base evita
  regenerar un archivo de `data/`.

**Para usarlo.** `--guardar gb:partida` entrena con todas las filas y guarda
`data/models/lp_partida_gb_partida.joblib`. Siguientes pasos posibles: más días de
`lp_vivo` (el colector de la PC de escritorio), los otros servidores, y llevar el modelo al
puente entre fotos de `build_lp_partida.py` en lugar de la regla.

#### S.55. Los ocho modelos del camino, en Train y en V Master+ (2026-11-16/17)

**Pedido del usuario (16/11).** Todos los modelos de "quién gana" medidos igual, en Train y en V, sin tocar el
test: hasta acá la Regla y la Logística Draft solo tenían números en el test (§9.S.20, de antes de la regla de no
tocarlo) y los demás solo tenían V, que además cambió de definición el 15/11 (§9.S.51).
`scripts/analysis/modelos_train_v.py` (~3 min) escribe `data/analysis/modelos_train_v.csv`, y la sección 1 de
`eda_modelos_ganador.ipynb` lo grafica.

* **V** = V Master+ (§9.S.51): las 16.428 partidas de Master+ del 16/09 al 27/09, las mismas para todos. Cada
  modelo predice V entrenado solo con partidas anteriores al 16/09.
* **Train** = dentro de la muestra: el mismo modelo que predice V, evaluado sobre las partidas con que se
  ajustó. En los stackings, los expertos ajustados con todo el ajuste predicen el ajuste y el meta (ajustado
  fuera de muestra, como siempre) los combina.

| Modelo | Train: AUC / acierto (n) | V Master+: AUC / acierto | Train − V |
| :--- | :---: | :---: | :---: |
| Baseline (siempre el rojo) | 0,500 / 51,9 % (160.883) | 0,500 / 52,2 % | — |
| Regla (§9.S.13) | 0,566 / 54,7 % (160.883) | 0,549 / 53,7 % | +0,017 |
| Logística Draft (§9.S.20) | 0,563 / 54,8 % (160.883) | 0,550 / 53,8 % | +0,014 |
| Campeones (§9.S.23) | 0,568 / 55,3 % (26.934) | 0,545 / 53,4 % | +0,024 |
| Campeones + Jugador (§9.S.23) | 0,611 / 58,0 % (26.934) | 0,582 / 55,6 % | +0,030 |
| Stacking de 6 expertos (§9.S.29) | 0,662 / 61,5 % (160.885) | 0,657 / 61,4 % | +0,004 |
| MOE: 2 expertos + composición + por línea (§9.S.37) | 0,663 / 61,6 % (160.885) | 0,660 / 61,7 % | +0,002 |
| **Logística Master+ (escalón 9, §9.S.52)** | 0,671 / 62,3 % (118.064) | **0,674 / 62,5 %** | −0,003 |

**Lectura.** (1) Todo lo que es solo draft queda en ~0,55 de AUC y ~54 % de acierto en V, apenas arriba del
lado (52,2 %): lo que mueve la aguja es saber quién juega. (2) El Stacking y el MOE, medidos en V Master+, dan
0,657 y 0,660 (en la V de todas las ligas daban 0,649 y 0,652, §9.S.29 y §9.S.37): rinden mejor en la cima,
como todos los modelos. El modelo final les saca +0,014 a +0,017 de AUC y +0,8 a +1,1 puntos. (3) Sobreajuste:
la brecha Train − V es de 0,014-0,017 en los modelos de draft y de 0,024-0,030 en Campeones y Campeones + Jugador
(ajustados solo con las 26.934 partidas posteriores a la foto, porque necesitan su LP); en el Stacking y el MOE es
de 0,002-0,004, porque las variables por jugador vienen encogidas y el meta se ajusta fuera de muestra. En el
modelo final V supera a Train; la explicación más probable (no medida) es que el ajuste arranca antes, con
historiales más cortos, y en V cada jugador ya acumuló más partidas previas.

**Reproducción (orden de las cachés).** En esta PC faltaban tres archivos que la cadena lee y que no están en
git; se traen de `G:\Mi unidad\Lol Database\data_completa\`: `splits/match_split.csv` → `data/splits/`,
`rating_partida.csv` → `data/`, y la carpeta `externos/` → `data/externos/` (Meraki y CommunityDragon, para la
composición). Después, en serie (~40 min de CPU y ~16 min de GPU en la PC de escritorio):

```bash
python historial/ganador_primeros/ganador_features.py                  # Campeones y Campeones + Jugador (~30 s)
python historial/ganador_primeros/ganador_features_grande.py           # ~40 s
python historial/ganador_primeros/ganador_margen_duos.py --sin-evaluar # ~1 min
python historial/ganador_primeros/ganador_margen_grupos.py             # ~2 min
python historial/ganador_primeros/ganador_mmr.py                       # ~2 min
python historial/ganador_primeros/ganador_poisson.py                   # ~21 min (PYTHONIOENCODING=utf-8 si se redirige)
python historial/ganador_primeros/ganador_lineas.py                    # ~11 min
python scripts/analysis/ganador_composicion.py               # ~15 s
python historial/ganador_primeros/ganador_logistica_oos.py --poisson data/analysis/ganador_poisson.pkl \
  --extra "data/analysis/ganador_mmr.pkl:nivel (foto)" "data/analysis/ganador_lineas.pkl:hgb" \
  "data/analysis/ganador_composicion.pkl:composicion" --guardar data/analysis/ganador_logistica_composicion
# GPU, ~5 min cada una (comunes: --sin-huella --configs C --variantes equipo --oos 5)
python historial/ganador_primeros/ganador_red_jugador.py --margen data/analysis/ganador_margen_duos.pkl ... --guardar data/analysis/ganador_red_jugador_margen.npz
python historial/ganador_primeros/ganador_red_jugador.py --margen ... --poisson data/analysis/ganador_poisson.pkl ... --guardar data/analysis/ganador_red_jugador_poisson.npz
python historial/ganador_primeros/ganador_red_jugador.py --margen ... --poisson ... --grupos data/analysis/ganador_margen_grupos.pkl ... --guardar data/analysis/ganador_red_jugador_grupos.npz
python scripts/analysis/modelos_train_v.py
python scripts/analysis/build_eda_modelos_nb.py --ejecutar
```

El escalón 9 usa las cachés de `data/analysis/brainstorm/` (orden en §9.S.69).

#### S.56. Qué tan parejas son las partidas difíciles de predecir (`eda_modelos_ganador.ipynb` §6.3) (2026-11-16)

> **Numeración:** `§9.S.55` queda reservada para `scripts/analysis/modelos_train_v.py`
> (ya referenciada en `TAREAS.md`, `modelos_train_v.py` y el encabezado de
> `build_eda_modelos_nb.py`); esta subsección se registra como **`S.56`**.

**Pregunta.** En la validación temporal $V$ ($N = 26\,934$ partidas), los modelos previos de
*quién gana* dejan un 20 % de partidas cerca del $50\,\%$ (`difícil`: $N = 5\,387$, menor
$|\text{logit}|$ del modelo híbrido). Se agrega la subsección **6.3** al notebook
`notebooks/eda_modelos_ganador.ipynb` (a través de su generador canónico
`scripts/analysis/build_eda_modelos_nb.py`, renumerando las antiguas 6.3–6.5 a 6.4–6.6)
para medir si esas partidas difíciles son efectivamente más parejas que las medias (`medio`,
60 % central, $N = 16\,160$), las fáciles (`fácil`, 20 % con mayor $|\text{logit}|$,
$N = 5\,387$) y el total de $V$ (`en general`, $N = 26\,934$).

**Definición y decisiones de diseño.** Se distinguen explícitamente dos planos porque responden
cosas distintas:
- **Paridad antes de jugar:** diferencia absoluta entre promedios por equipo (azul menos rojo)
  sobre cuatro variables previas del `variables.npz` (`|Δ margen previo|`, `|Δ exp. campeón|`,
  `|Δ estrenos|`, `|Δ tier|`), resumidas con media por ser acotadas y poco sesgadas.
- **Paridad en la cancha:** diferencia absoluta de oro acumulado (`|Δoro|` al 10, 15, 20 y
  final) y de kills finales (`|Δkills final|`), resumidas con **mediana** por la asimetría hacia
  snowballs grandes; duración media en minutos; y cinco proporciones de desenlace: pareja al 15
  ($|\Delta\text{oro}_{15}| < 2\,000$), paliza al 15 ($|\Delta\text{oro}_{15}| \ge 5\,000$),
  remontada desde el 15, corta ($< 25$ min) y larga ($\ge 32$ min).
- **Filtro de `@15` válido en remontadas:** todas las métricas `@15` (incluida la tasa de
  remontadas) se calculan restringiendo a partidas con snapshot `@15` presente ($98{,}9\,\%$ de
  $V$) y ventaja de oro no nula (`ok15 & (g15 != 0)`). En la subsección preexistente **6.2** el
  conteo rápido `(np.sign(g15) != gana)` incluía `NaN` como distinto de $1$ (inflando la tasa
  del grupo difícil en $\approx 1$ pp, de $28{,}2\,\%$ a $28{,}9\,\%$); en **6.3** se aplicó el
  filtro consistente con **3.2** sin alterar **6.2**.
- **Alternativas descartadas:** usar un único gráfico resumen se descartó porque mezclaba escalas
  incompatibles (unidades de margen previo, oro, minutos y porcentajes) y ocultaba la dispersión
  de `@15`. Se diseñó una grilla `2×2` en `seaborn`: (1) desviación porcentual respecto del
  promedio general comparando variables previas vs. variables en la cancha, (2) mediana de
  `|Δoro|` a los 10, 15, 20 min y final con curva de referencia de `en general`, (3) boxplot
  `whis=(10, 90)` sin outliers de `|Δoro al 15|` con medianas anotadas, y (4) tasas de
  finalización y paridad en `%`.

**Resultados medidos en $V$ ($N = 26\,934$).**
- **Antes de jugar**, el 20 % difícil llega claramente más equilibrado que el resto:
  - `|Δ margen previo|` (media): `0.060` (`difícil`), `0.065` (`medio`), `0.087` (`fácil`), `0.068` (`en general`).
  - `|Δ exp. campeón|` (media): `0.626`, `0.684`, `0.920`, `0.720`.
  - `|Δ estrenos|` (media): `0.760`, `0.792`, `0.943`, `0.816`.
  - `|Δ tier|` (media): `0.261`, `0.266`, `0.261`, `0.264` (plano en los cuatro grupos por efecto del matchmaking de MMR).
- **En la cancha**, la diferencia entre `difícil`, `medio` y `en general` prácticamente
  desaparece hasta el minuto 20:
  - `|Δoro al 10|` (mediana): `1,362` (`difícil`) vs `1,377` (`medio`), `1,430` (`fácil`), `1,385` (`en general`); Mann-Whitney `difícil vs medio p = 0.641`, `difícil vs fácil p = 6.6e-03`.
  - `|Δoro al 15|` (mediana): `2,590` vs `2,587`, `2,808`, `2,633`; Mann-Whitney `difícil vs medio p = 0.613`, `difícil vs fácil p = 2.7e-05`.
  - `|Δoro al 20|` (mediana): `3,655` vs `3,652`, `4,026`, `3,710`; Mann-Whitney `difícil vs medio p = 0.241`, `difícil vs fácil p = 3.1e-07`.
  - `|Δoro final|` (mediana): `8,485` (`difícil`), `8,614` (`medio`), `9,063` (`fácil`), `8,667` (`en general`). `|Δkills final|` (mediana): `12`, `12`, `13`, `12`. Duración media: `28.8`, `28.6`, `27.8`, `28.5` min.
  - Proporciones: pareja al 15 ($< 2\,000$ oro): `40.5%`, `40.4%`, `38.4%`, `40.0%`; paliza al 15 ($\ge 5\,000$ oro): `20.3%`, `20.6%`, `23.5%`, `21.1%`; remontada desde el 15: `28.2%`, `27.3%`, `25.0%`, `27.0%`; corta ($< 25$ min): `24.2%`, `25.3%`, `31.0%`, `26.2%`; larga ($\ge 32$ min): `28.9%`, `27.6%`, `23.9%`, `27.1%`.
- **Lectura y relación inversa.** En una partida `difícil` el juego hace snowball casi igual que en
  una partida típica (el $20{,}3\,\%$ igual son palizas de $\ge 5\,000$ de oro al min 15), solo
  que antes de empezar no se distingue de qué lado caerá la moneda. Al revés, al condicionar por
  cómo *terminó* siendo la partida, el AUC del híbrido cae fuerte en las partidas efectivamente
  parejas/largas (`< 2k oro @15`: `0.605` vs `≥ 5k`: `0.707`; `≥ 32 min`: `0.558` vs `< 25 min`: `0.728`).

#### S.57. Pipeline in-game (@10, @15, @20): corrección de fin de partida, caché causal y arquitectura (2026-11-16)

**Motivación y hallazgos de diagnóstico.** Sobre los modelos en vivo de §9.S.52 (`brainstorm_39_en_vivo.py`
y `brainstorm_40_en_vivo_lineas.py`), se detectaron cuatro brechas estructurales antes de entrenar:
1. **Fuga de último frame en Riot Timeline-V5 (`dur <= 60*m`):** `src/database.py` guarda `frames[m]` también
   cuando la partida terminó entre `m-1` y `m` (el frame de caída del Nexo). En `variables.npz` hay
   `215` partidas con `gold_10` finito y `dur <= 10 min` (`92,1%` gana el líder), `216` con `gold_15` y
   `dur <= 15 min` (`96,3%`) y `2.995` con `gold_20` y `dur <= 20 min` (`99,7%`). Para evaluar y entrenar
   sin fuga en el minuto `m`, es obligatorio enmascarar `game_duration <= 60 * m` (`mask_finished_matches`, §3.F.4).
2. **Volumen de entrenamiento restringido a A2-A4 (`13.860` Master+):** `brainstorm_39/40` entrenaban la
   etapa 2 solo sobre `bloque >= 1`, dejando afuera `104.204` partidas Master+ de `bloque 0`. Partiendo
   `bloque 0` en 4 tramos cronológicos con predicciones `L_oof` *forward-chaining*, el entrenamiento
   causal de etapa 2 pasa a **~92.000+ partidas Master+** (`m=10`: `94.359` válidas en train, `16.331` en
   `V Master+`, `10.422` en `Diamond-V`).
3. **Cinco estadísticas del timeline sin usar + trayectorias:** `match_timeline` guarda además `cs`,
   `damage`, `level`, `deaths` y `assists` por jugador, de donde salen el **oro residual causal**
   (`gold ~ 1 + cs + kills + assists` por carril, proxy de placas/torres/bounties), picos de nivel
   (`>=6/8/11/13/16`), coordinación (`assists/kills`), dispersión intra-equipo (`max + min`) y
   trayectorias (`10→15→20`: velocidad, aceleración, cambios de líder y drawdown).
4. **Antisimetría exacta con sesgo de lado (§3.F.1) y composición/jugadores:** `L_sym = L_oof - L_0`
   (descontando la tasa base del lado antes de espejar), escalado RMS sin restar media en `X_anti`,
   y cruces estrictamente antisimétricos de composición (`full_ad_vs_tank`, `poke_vs_sin_engage`,
   `asesino_vs_poco_dur`, `hypercarry_comp`, `cbias`, crecimiento por nivel de `champions.csv`),
   ritmo (`Δoro × total_kills`), sinergias 2v2 (`sign(a+b)(a+b)^2 - sign(a)a^2 - sign(b)b^2`) y rasgos
   causales de jugador (`pbias`, `pvol`, `estreno`/`offrole`/`skill × Δoro`).

**Implementación y resultados parciales ejecutados (`ingame_preparar.py`, `2203.8s`).**
* `src/models/ingame.py` + `tests/test_ingame.py` (7 tests de enmascaramiento, causalidad sin fuga,
  oro residual y antisimetría exacta de features y de `InGameNet` en PyTorch con cabezas auxiliares
  futuras `@15/@20/fin` enmascaradas por duración).
* `scripts/analysis/ingame_preparar.py` generó `data/analysis/ingame/ingame_cache.npz` (`269.3 MB`) y
  las cachés auxiliares `gemini_g11.npz` y `gemini_g23.npz`:
  * **Stage-1 `L_oof` causal (7 tramos):** `AUC A2-A4 Master+ = 0.6561`, `AUC V Master+ = 0.6676`.
  * **Expectativas causales por carril (`Ridge DRAFT+comp + LightGBM residual`, correlación en `V`):**
    - `gold_10`: `0.326 / 0.338 / 0.379 / 0.406 / 0.355` (`xp_10`: `0.342 / 0.419 / 0.312 / 0.347 / 0.304`)
    - `gold_15`: `0.331 / 0.396 / 0.389 / 0.417 / 0.410` (`xp_15`: `0.350 / 0.422 / 0.324 / 0.362 / 0.319`)
    - `gold_20`: `0.341 / 0.431 / 0.406 / 0.427 / 0.456` (`xp_20`: `0.340 / 0.419 / 0.329 / 0.344 / 0.324`)
  * **Expectativas causales de transición:** `@10→@15` y `@10+@15→@20` calculadas por carril.
* `scripts/analysis/ingame_evaluar.py` quedó listo y optimizado para CPU (`LogReg`, `LightGBM_Standalone`,
  `LightGBM_Residual`, `CatBoost_Residual`, `InGameNet_NoAux`, `InGameNet_MultiTask` y `Ensemble_NoLin`),
  pendiente de corrida completa a pedido del usuario (interrumpido al arrancar `m = 10`).

#### S.58. UMAP del estado en vivo, de las partidas y de los jugadores (2026-11-17)

**Pedido.** Recomendación del profesor: probar UMAP. Tres vistas, cada una con una pregunta que se puede
contestar con números: (1) estado al minuto 15, ¿hay una región de partidas que siguen abiertas a pesar del
oro y que el modelo en vivo no ve?; (2) partidas antes de jugarse, ¿hay un tipo de partida donde el escalón 9
acierte más o menos de lo que promete?; (3) jugadores, ¿hay subestilos dentro de cada rol que K-Means corta
mal (§9.S.53)? `scripts/analysis/umap_vistas.py --vista vivo|partidas|jugadores` (2, 4 y 2 min); salidas en
`data/analysis/umap/<vista>/` (`grupos.csv`, `prueba.csv`, `perfiles.csv`, figuras `umap_*.png`).

**Por qué no alcanza con mirar el dibujo.** UMAP arma un grafo difuso de $k$ vecinos ($k$ = `n_neighbors` =
30) y busca una disposición en baja dimensión que conserve esos vecindarios. Conserva lo local, no las
distancias globales ni las densidades: en 2D el tamaño y la separación de las manchas no significan nada, y
cambian con la semilla, con `n_neighbors` y con `min_dist`. Por eso el protocolo es:

1. Cada vista se dibuja con tres semillas y con PCA al lado (lo que aparece solo en una semilla no se lee).
2. Los grupos se buscan con **HDBSCAN sobre un UMAP de 10 dimensiones** (`min_dist` = 0), que conserva más
   estructura que el de 2. HDBSCAN no fija $k$: arma la jerarquía de densidad y se queda con los grupos
   estables (`min_cluster_size` = 150, `min_samples` = 30); lo que no pertenece a ninguna región densa
   queda como ruido (etiqueta −1).
3. Se descubre en un conjunto y se confirma en otro: el UMAP y HDBSCAN se ajustan en los sub-bloques A2-A4
   Master+ (o en el ajuste) y V Master+ se proyecta con `transform` y se asigna con `approximate_predict`.
4. Un grupo interesa solo si el modelo se equivoca **sistemáticamente** en él. En el marco que corresponde
   (el líder en oro, o el favorito), con $p_i$ la probabilidad del modelo e $y_i$ el resultado:
   $$\bar r_g = \frac{1}{n_g}\sum_{i \in g}(y_i - p_i), \qquad
     z_g = \frac{\bar r_g}{\sqrt{\sum_{i \in g} p_i(1-p_i)}\,/\,n_g},$$
   que bajo calibración es ≈ N(0, 1). Además, el grupo se suma al modelo (apilado sobre su logit) y se mide en
   V con IC pareado, junto a un **placebo**: los mismos grupos asignados a filas permutadas.

*Vista 1: estado al minuto 15.* Estado orientado al equipo que va adelante en oro,
$s = \operatorname{sign}\big(\sum_\ell \Delta\text{oro}_\ell\big)$:
$s \cdot [\Delta\text{oro}_\ell, \Delta\text{xp}_\ell, \Delta\text{kills}, \text{sorpresa}_\ell, L_9]$, con
$\text{sorpresa}_\ell = \Delta\text{oro}_\ell - \widehat{\Delta\text{oro}}_\ell$ (el oro de la línea que el
modelo por línea predecía antes de jugar) y $L_9$ el logit previo del escalón 9, estandarizado con los bloques.
Solo partidas vivas al minuto 15 (`game_duration > 900`, §3.F.4): la primera corrida no las filtraba y
Antigravity lo marcó en la revisión cruzada (20 partidas de los bloques y 17 de V; inflaba +0,0003 el AUC del
modelo en vivo). Modelo en vivo de referencia: la logística "líneas + esperado completo" de `brainstorm_40` con el
escalón 9 de previo, **V 0,8370 / 75,6 %**. El líder en oro gana el 72,4 % (bloques) y el 73,0 % (V). En los
bloques, cada uno se predice con los otros dos (en las dos direcciones del tiempo): es solo descubrimiento; la
decisión se toma en V, que se predice con los tres bloques, todos anteriores.

| Vista 1 (minuto 15) | Bloques | V Master+ |
| :--- | :---: | :---: |
| Grupos de HDBSCAN / ruido | 7 / 72 % | 7 / 82 % |
| Mayor $\lvert z_g \rvert$ | 2,43 (grupo 4, 2 % de las partidas) | 1,63 (otro grupo); el grupo 4 da −0,66 |
| + líder × grupo, ΔAUC | −0,0000 (1 de 3 bloques) | −0,0001 [−0,0003; +0,0002] |
| + líder × grupo + logit × grupo | −0,0005 | −0,0002 [−0,0004; +0,0001] |
| Placebo | −0,0008 | −0,0003 [−0,0006; −0,0001] |

UMAP ordena las partidas casi solo por $\lvert\Delta\text{oro}\rvert$: la región de ventajas de más de 6.000 de
oro (el líder gana más del 90 %) queda en un extremo, y el residuo del modelo en vivo es ruido sal y pimienta en
bloques y en V. Con 14 grupos-conjunto mirados, un $\lvert z \rvert$ = 2,4 que no se replica es lo esperable por
azar. **No hay una región de "partidas abiertas a pesar del oro" que la logística en vivo no capture**, lo que
coincide con que el HGB sobre las mismas variables rinde menos que la logística (§9.S.52).

*Vista 2: partidas antes de jugarse.* Contexto simétrico de la idea 45 (259 variables: media, dispersión y
$\lvert$azul − rojo$\rvert$ de las 85 por jugador, nivel y tier del lobby); no dice quién gana, describe la partida.
UMAP ajustado con 30.000 partidas Master+ del ajuste que no están en los bloques; $p$ = escalón 9 (`po_ref9`
hacia adelante en A2-A4, `pv_ref9` en V). HDBSCAN da 2 grupos y casi nada de ruido: uno chico (3-4 %), con
dispersión extrema de la visión por minuto entre los 10 y perfil de soporte, donde el escalón 9 acierta algo
menos de lo que promete (exceso −0,017 en bloques y −0,022 en V; $z$ = −0,9 y −1,0, sin significancia), y el
resto. Recalibrar por grupo ($y \sim L_9 + L_9 \times \text{grupo}$; el grupo es simétrico, así que solo
puede escalar la confianza) da **ΔAUC V +0,0000 [−0,0001; +0,0001]**, igual que el placebo. En el dibujo se ve
el gradiente de tier del lobby y una isla de lobbies muy desparejos ($\lvert L_9 \rvert$ ≈ 1, el "grupo fácil"
de la idea 45); el acierto menos lo esperado es ruido. Confirma la idea 45 con otro espacio.

*Vista 3: jugadores.* 14.249 jugadores con 50 partidas o más; las 15 variables de estilo de §9.S.53,
estandarizadas dentro del rol principal (así el rol no domina). `players.csv` mezcla train y test: es
descriptivo, igual que §9.S.53. El UMAP global no muestra tipos discretos: es un continuo con un **eje de
habilidad** (winrate de 46 % a 58 % de punta a punta) y dos lóbulos de soportes con estilos distintos
(clase principal Support contra Tank). Por rol, HDBSCAN sobre 10D:

| Rol | Jugadores | Grupos | Ruido | ARI con K-Means (mismo $k$) | Silhouette HDBSCAN / K-Means, mismos jugadores (sin ruido) | K-Means con todos (no comparable) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| TOP | 2.520 | 5 | 43 % | 0,50 | 0,155 / 0,186 | 0,141 |
| JUNGLE | 2.856 | 5 | 47 % | 0,47 | 0,161 / 0,178 | 0,136 |
| MIDDLE | 3.323 | 5 | 61 % | 0,55 | 0,171 / 0,178 | 0,134 |
| BOTTOM | 2.900 | 3 | 32 % | 0,32 | 0,167 / 0,222 | 0,189 |
| UTILITY | 2.650 | 3 | 30 % | 0,45 | 0,205 / 0,258 | 0,228 |

*Corrección (2026-11-23, §9.S.77).* La primera versión de la tabla comparaba la silhouette de HDBSCAN, calculada
sin su ruido, con la de K-Means sobre todos los jugadores: conjuntos distintos, y el de HDBSCAN más fácil (saca
justo los puntos de frontera). Sobre los mismos jugadores, **K-Means queda por encima en los cinco roles** (y
más todavía si se reajusta sin el ruido: 0,196-0,250). No cambia la lectura: todas quedan entre 0,15 y 0,26.

Dos grupos se repiten en casi todos los roles: "flex con perfil de soporte" (pool disperso, poco CS, mucha
visión: el rol principal está mal asignado porque juega mucho de soporte) y "dominantes" (ventaja de oro y CS
a los 15 y al final; en mid son asesinos con **57,2 %** de winrate contra ~49 % del resto). Silhouette de 0,15
a 0,23 y 30-60 % de ruido dicen que no hay estilos separados: el estilo es un continuo, y K-Means lo corta en
lugares arbitrarios (por eso coincide solo parcialmente con HDBSCAN).

**Decisión.** Nada de UMAP entra al modelo principal ni al de en vivo: en las tres vistas los grupos no
predicen el error del modelo, ni en el conjunto donde se descubrieron ni en V. El valor de UMAP para el informe
es visual (las figuras de las tres vistas) y el resultado negativo, que con este protocolo es creíble.

*Alternativas descartadas.* (a) Buscar grupos en el UMAP de 2D: dependen del dibujo y de la semilla. (b)
t-SNE: el mismo problema con las distancias y, además, no proyecta datos nuevos (V), así que no permite
descubrir en un conjunto y confirmar en otro (§9.S.41 lo usó solo para mirar). (c) Elegir $k$ por silhouette:
da $k$ = 2 sin contenido (§9.S.41); HDBSCAN deja que la densidad decida y declara ruido lo que no es grupo.

#### S.59. Nuestra metodología sobre el dataset de DraftRec, y el baseline de su recomendador (2026-11-16/17)

**Pedido.** (16/11) Compararnos contra un trabajo publicado, en **sus** datos. (17/11) Llevar a esos datos el
mejor modelo (escalón 9, §9.S.52) y medir el baseline del recomendador de picks (tarea "Predictor y
recomendador de picks", paso 1).

**El dataset.** DraftRec (Lee et al., WWW 2022): 279.893 partidas de cola solo/dúo, Challenger a Diamond 1-2,
junio-septiembre de 2021, con split temporal 85/5/10 (train 236.408, val 14.795, test 28.690). Cada partida
da los 10 jugadores en orden de pick y, para cada uno, el índice de esa partida en su historial;
`user_history.npy` (62.466 jugadores × 1.145 partidas × 57) guarda por partida el campeón, el rol, el equipo,
los 10 bans, si ganó y 43 estadísticas de final de partida ya estandarizadas. El historial está ordenado por
fecha, así que lo previo a la partida $h$ es `historial[:h]`: todo lo que sigue es causal. **No tiene** línea
de tiempo, rango, LP ni fecha de cada partida del historial. Su tabla de test (solo acierto): logística
52,55 %, NeuralAC 52,66 %, OptMatch 54,11 %, **DraftRec 55,35 %**.

*Paso 1: el híbrido en sus datos* (`draftrec_comparacion.py`, 16/11). Las variables por jugador de §9.S.23
(experiencia, pool, winrate con el campeón encogido, rol) y un margen de final de partida (estadística − media
del campeón en el train: acumulado, de las últimas 20 y con ese campeón), todo como media de los 5 azul − media
de los 5 rojo, en una logística con C elegido en val:

| Modelo (test de DraftRec) | AUC | Acierto |
| :--- | :---: | :---: |
| Baseline (tasa del train) | 0,500 | 50,7 % |
| Logística draft (campeón × rol) | 0,553 | 53,8 % |
| Jugadores sin margen | 0,582 | 55,8 % |
| Jugadores + margen | 0,612 | 58,0 % |
| **Híbrido: draft + jugadores + margen** | **0,623** | **58,8 %** |

*Paso 2: la ablación en nuestra V Master+* (`ablacion_draftrec.py`). Para separar dataset de metodología, se
le sacó a nuestro híbrido + mapa (V 0,6531 / 61,1 %) lo que DraftRec no tiene: sin mapa −0,0043 [−0,0059;
−0,0026]; sin mapa ni línea de tiempo −0,0047; **compatible con DraftRec** (sin mapa, línea de tiempo ni fechas)
**0,6408 / 60,1 %** (−0,0123 [−0,0151; −0,0096]). La línea de tiempo no suma sobre las estadísticas finales; lo
que pesa son las variables con fecha y el mapa. Con la misma información, en nuestros datos se acierta 60,1 % y
en los suyos 58,8 %: la diferencia que queda es de población (Master+ de LAS en 2026 contra Diamond 2 y más
en 2021, de un servidor que el dataset no documenta) y no se puede atribuir a la implementación.

*Paso 3: el escalón 9 en sus datos* (`draftrec_mejor.py`, 17/11). Una sola pasada por los 16 GB guarda en caché
lo previo de cada (jugador, partida) (`--solo-pasadas`, 5,5 min); después se suman uno por uno los ingredientes
del escalón 9 que se pueden reconstruir. Azul y rojo se toman del campo de equipo del historial (no del orden de
pick); el 100 % de las partidas tiene los 5 roles en los dos equipos.

1. **Por línea:** $x_{\ell} = v_{\text{azul},\ell} - v_{\text{rojo},\ell}$ para cada rol $\ell$ en vez de la
   media de los 5 (como `Xd` del modelo principal).
2. **Margen y winrate por (jugador, rol):** $m^{\text{rol}} = \big(\sum_{j<h,\ \text{rol}_j = r} \text{marg}_j
   + K\, m^{\text{acum}}\big) / (n_r + K)$ con $K$ = 5, encogido hacia el margen total; ídem el winrate.
3. **Elo causal** (análogo del Kalman-Elo del escalón 6, solo con resultados): las 279.893 partidas en orden de
   tiempo; cada una se describe con los ratings de antes y después
   $R_u \leftarrow R_u \pm K\,\big(y - \sigma(\bar R_{\text{azul}} - \bar R_{\text{rojo}})\big)$. Diferencia
   por línea, del equipo y partidas vistas. $K$ se eligió en val (0,15); con $K$ = 0,6 el final da 0,6288 en
   vez de 0,6294 (val empatado).
4. **Contracruces por línea y sinergias:** winrate empírico-bayesiano causal del par (campeón azul, campeón
   rojo) en la línea, $\text{WR} = (w + K/2)/(n + K)$ con $K$ = 20, orientado al azul y menos 1/2; ídem los pares
   de equipo CARRY + SUPPORT, TOP + JUNGLE y MIDDLE + JUNGLE (azul − rojo).
5. **Margen esperado por par en la línea:** $\mathbb{E}[\Delta s \mid \ell, c_A, c_B] = \sum \Delta s / (n + 10)$,
   encogido a 0, para el oro, el CS y el daño a campeones de final de partida.

Los acumulados 3-5 recorren train → val → test en orden: el test se describe con todo lo anterior a cada
partida, igual que el modelo principal, pero la logística se ajusta solo con train.

| Escalón (test de DraftRec) | AUC | Acierto | Δ AUC contra el anterior [IC 95 %] |
| :--- | :---: | :---: | :---: |
| Híbrido (media de los 5) | 0,6227 | 58,77 % | — |
| 1. + por línea | 0,6260 | 59,16 % | +0,0033 [+0,0018; +0,0048] |
| 2. + margen y winrate por rol | 0,6263 | 59,14 % | +0,0003 [−0,0009; +0,0015] |
| 3. + Elo causal | 0,6279 | 59,19 % | +0,0017 [+0,0008; +0,0026] |
| 4. + contracruces y sinergias | 0,6290 | 59,16 % | +0,0011 [+0,0004; +0,0019] |
| **5. + margen esperado por par (≈ escalón 9)** | **0,6294** | **59,27 %** | +0,0003 [−0,0006; +0,0014] |

Total sobre el híbrido: **ΔAUC +0,0066 [+0,0044; +0,0089]** y **Δacierto +0,50 pp [+0,16; +0,92]**
(bootstrap pareado en el test, B = 500); contra DraftRec, **+3,9 puntos de acierto**. Lectura: los tres
ingredientes que suman en DraftRec (por línea, rating causal, contracruces y sinergias) son los mismos que
sumaron en nuestros datos (§9.S.52), así que **se replican en un dataset independiente, de otra población y otro
año**. Los dos que no suman tienen explicación: el margen por rol es redundante con el margen por campeón (el
campeón casi determina el rol), y el margen esperado por par usaba en nuestros datos el timeline @10/@15, mientras
que acá solo hay estadísticas de final de partida, que son consecuencia del resultado (el que gana termina con
más oro) y repiten la información del contracruce.

*Paso 4: el baseline del recomendador.* DraftRec mide su recomendador con HR@10 (88,0-89,9 % según la
configuración): si el campeón que el jugador eligió está entre los 10 recomendados. Baseline: los campeones más
jugados por ese jugador antes de la partida, con peso $0{,}5^{(h-1-j)/50}$ a la partida $j$ (vida media de 50
partidas), sin los ya elegidos ni los baneados en ese draft y completando con la popularidad del rol en el
train. En los 286.900 picks del test:

| Ranking | HR@1 | HR@5 | HR@10 |
| :--- | :---: | :---: | :---: |
| Más jugados en el rol de la partida | 48,8 % | 81,8 % | **90,5 %** |
| Más jugados en general | 43,7 % | 76,8 % | 87,8 % |
| Popularidad del campeón en el rol | 15,5 % | 42,9 % | 63,7 % |
| DraftRec (paper) | — | — | 88,0-89,9 % |

Un conteo de frecuencias con decaimiento empata (sin rol) o supera (con rol) a DraftRec. HR@10 mide **imitación**:
si se adivina lo que el jugador va a elegir, no si le conviene. El valor de un recomendador se mide por cuánto
cambia la probabilidad de ganar (paso 3 de la tarea de picks: evaluación fuera de política). *Advertencias:* el
baseline "en el rol" conoce el rol de la partida, que en el draft real se intuye pero no se sabe; y no sabemos si
DraftRec excluye los campeones ya elegidos y baneados, cosa que facilita un poco el baseline. Aun sin rol (87,8 %),
la conclusión se sostiene.

**Decisión.** Para nuestros modelos no cambia nada: todos estos ingredientes ya están en el escalón 9. Para el
informe quedan tres resultados: (1) con nuestra metodología, en sus datos y su partición, 59,3 % contra 55,4 %;
(2) los ingredientes que funcionaron se replican fuera de nuestro dataset; (3) la métrica de su recomendador la
alcanza un conteo de frecuencias.

*Alternativas descartadas.* (a) Elegir el $K$ del Elo por el AUC del Elo solo en val: crece con $K$ (0,5128 con
0,03 → 0,5219 con 0,6), porque un $K$ grande sigue la forma reciente, pero en el modelo completo no mejora. (b)
Entrenar solo con la cima (escalón 1): DraftRec no trae el rango. (c) El mapa de LP, el nivel histórico y el W/L de
temporada (escalones 3a/3b): no hay LP ni W/L en el dataset.

**Puente de acierto para la presentación (2026-11-18, `grafico_draftrec_puente.py`).** El gráfico va de DraftRec
a nuestro mejor modelo en cuatro barras. La tercera es una **estimación**: el escalón 9 con solo la información de
DraftRec, en nuestros datos. Al híbrido sacarle eso le costó 1,0 pt; el escalón 9 usa más de esas variables (mapa
de LP, nivel histórico, W/L de temporada, rango), así que se extrapola a 1,5-2,0 pts, es decir 60,5-61,0 %. El
gráfico la marca como extrapolación, con su rango. Así la diferencia de 7 puntos se reparte en:

| Barra | Acierto | Salto |
| :--- | ---: | :--- |
| DraftRec (paper, su dataset) | 55,4 % | |
| Nuestro mejor modelo en el dataset de DraftRec | 59,3 % | +3,9 pts de metodología, con los mismos datos |
| Nuestro mejor modelo solo con la información de DraftRec, en nuestros datos (estimación) | ~60,8 % | ~+1,5 pts de población |
| Nuestro mejor modelo completo | 62,5 % | ~+1,8 pts de las variables que DraftRec no tiene |

Salida: `data/analysis/draftrec_puente.png`.

Archivos: `scripts/analysis/draftrec_comparacion.py`, `ablacion_draftrec.py` y `draftrec_mejor.py`; resultados en
`data/analysis/draftrec_comparacion.csv`, `ablacion_draftrec.csv`, `draftrec_mejor.csv` (K = 0,15),
`draftrec_mejor_k06.csv` y `draftrec_hr.csv`; caché de las pasadas en `draftrec_mejor_pasadas.npz` (628 MB).

#### S.60. Modelo en vivo: evaluación completa, el previo del escalón 9 en dos etapas y señales nuevas (2026-11-17)

Sigue a §9.S.57 (pipeline, máscara de partidas terminadas, caché causal). Todo en V Master+ con el ajuste
anterior al 16/09, sub-bloques A2-A4 hacia adelante y Diamond-V como réplica; solo partidas vivas en el minuto
$m$ (`game_duration > 60m`). La parte de esta noche se hizo con Antigravity (Gemini 3) en la PC de escritorio
(scripts `ingame_g_0*`, de Antigravity, revisados por Claude).

**1. Evaluación completa** (`ingame_evaluar.py`, laptop, 16-17/11; `data/analysis/ingame/resultados_ingame.csv`):

| Variante (V Master+) | Minuto 10 | Minuto 15 | Minuto 20 |
| :--- | :---: | :---: | :---: |
| Base: previo + Δoro/Δxp/Δkills del equipo (13,8k) | 0,7779 / 70,5 % | 0,8301 / 75,0 % | 0,8627 / 77,8 % |
| + por línea y sorpresa (13,8k) | 0,7846 / 71,1 % | 0,8398 / 75,9 % | 0,8712 / 78,7 % |
| A1: + más datos (92k, mismas variables) | 0,7845 / 71,0 % | 0,8396 / 75,9 % | 0,8711 / 78,8 % |
| **A2: + 8 estadísticas, trayectorias y expectativas por línea** | **0,7879 / 71,4 %** | 0,8412 / 76,0 % | 0,8729 / 78,9 % |
| A3: + composición y cruces | 0,7876 / 71,3 % | 0,8413 / 75,9 % | 0,8727 / 78,9 % |
| A4: + rasgos de jugador | 0,7868 / 71,3 % | 0,8412 / 75,9 % | 0,8726 / 78,9 % |
| A4 entrenado con Master+ y Diamond | 0,7866 / 71,2 % | 0,8409 / 75,9 % | 0,8727 / 78,7 % |
| LightGBM / CatBoost residual | 0,7868 / 71,2 % | 0,8409-0,8413 | 0,8724-0,8726 |
| **InGameNet multitarea** (cabezas auxiliares del futuro) | 0,7874 / 71,2 % | **0,8428 / 75,9 %** | **0,8739 / 78,8 %** |
| Ensamble no lineal | 0,7878 / 71,2 % | 0,8427 / 76,0 % | 0,8737 / 78,9 % |

El paso que suma es A2 (+0,0033 al 10, +0,0014 al 15 y +0,0017 al 20 sobre líneas + sorpresa); composición,
jugadores, árboles, Diamond y ensambles ≈ 0. La red multitarea le gana a A2 al 15 (+0,0016) y al 20 (+0,0010). La fuga del último frame no inflaba los
números de `brainstorm_39/40` al 10 y al 15 (0,7774 / 0,8288 de antes; con la máscara, −0,0003 en V, §9.S.58). Al
20 sí: el mismo estado básico (Δoro, Δxp y Δkills del equipo) da 0,8584 en `brainstorm_39` y 0,8537 con la máscara
(`ingame_estado_puro.py`); al 10 y al 15 coincide (0,7470 / 0,7465 y 0,8138 / 0,8136).
*Convierte o tira* (`ingame_convertir.py`, idea del usuario): residuo causal de cada jugador según su equipo o
su línea iba arriba o abajo en oro (ganó − P(gana | Δoro al minuto $m$)), activado según el estado: +0,0002
[−0,0001; +0,0004] al 10 y al 15, 1 de 3 bloques; ≈ 0, no entra.

**2. El previo del escalón 9 en una etapa** (`ingame_g_01_prior_y_gpu.py`, `ingame_g_02_mejoras_vivo.py`). La
etapa 1 de `ingame_preparar.py` usa una logística previa simplificada ($L_{\text{orig}}$, V 0,6677); el escalón 9
da 0,6739 y su ensamble del escalón 10 ($L_{\text{ens10}}$ = 0,50 logística + 0,25 red NeuralAC + 0,25 CatBoost
residual, en logits) 0,6749. Reemplazar el previo en la logística A2:

| Ajuste de la etapa 2 | Minuto 10: Δ AUC V contra $L_{\text{orig}}$ | Minuto 15 | Minuto 20 |
| :--- | :---: | :---: | :---: |
| 13,8k (A2-A4, todo posterior a la foto), $L_{\text{ens10}}$ | **+0,0015 [+0,0002; +0,0027]** | +0,0006 [−0,0004; +0,0015] | +0,0002 |
| 92k (chunks 1-6), $L_{\text{esc9}}$ empalmado | +0,0007 [−0,0005; +0,0019] | +0,0000 | +0,0000 |

Con más datos el previo mejor casi deja de sumar. El diagnóstico (Antigravity) es que el previo tiene **dos
regímenes**: en las ~78k partidas anteriores a la foto 1 el escalón 9 no tiene sus columnas de foto (W/L de
temporada, nivel histórico) y es más débil (AUC ≈ 0,63-0,66), y en las ~14k posteriores es el completo (≈ 0,674),
como en V. La logística aprende un solo peso para $L$, dominado por el régimen débil. Agregar $L \times
\text{post}$ dentro de la logística de 92k no alcanza: la regularización ($C$ = 0,01-0,05) aplasta una columna
que es 0 en el 85 % de las filas.

**3. El previo en dos etapas** (`ingame_g_03_regimen_y_senales.py`, variante 5b). Se desacopla el estado del
previo nuevo:

$$z = \operatorname{logit}\hat P_{92k}(y \mid \text{estado}_m, L_{\text{orig}}), \qquad
\operatorname{logit} P(y) = a + b\, z + c\,(L_{\text{ens10}} - L_{\text{orig}}),$$

donde la primera etapa es la logística A2 entrenada con las 92k (todo el volumen para aprender el estado) y la
segunda se ajusta **solo con partidas posteriores a la foto**, hacia adelante (A3 ← A2, A4 ← A2-A3, V ← A2-A4),
con las predicciones de la primera etapa hacia adelante como $z$. Con $z$ sola la segunda etapa es una
recalibración monótona (mismo AUC): la ganancia viene de la diferencia de previos, con su peso aprendido en el
régimen correcto. Verificado por Claude en el código; placebo ($d$ permutada dentro de cada conjunto) +0,0000.

| Minuto | V Master+ (AUC / acierto) | Δ contra $L_{\text{orig}}$ en V | A3-A4 | Diamond-V |
| :---: | :---: | :---: | :---: | :---: |
| 10 | **0,7894 / 71,4 %** | **+0,0016 [+0,0006; +0,0025]** | +0,0017 | +0,0011 [−0,0001; +0,0022] |
| 15 | 0,8419 / 76,0 % | **+0,0007 [+0,0002; +0,0012]** | +0,0005 | +0,0006 [−0,0000; +0,0012] |
| 20 | 0,8732 / 78,8 % | +0,0003 [+0,0002; +0,0005] | +0,0001 | +0,0002 |

Al minuto 10 queda +0,0015 [+0,0005; +0,0025] sobre la mejor logística de la laptop, con acierto en el 5/10/20 %
más seguro de 96,4 / 94,5 / 91,0 %. El aporte del previo baja con el minuto, como en §9.S.52: al 20 el estado ya
dice casi todo.

**4. Señales nuevas del estado** (sobre la logística A2 de 92k; cada una con placebo de filas permutadas).
Interacciones no lineales (decaimiento del previo con $\lvert\Delta\text{oro}\rvert$, saturación, cruces 2 contra 2,
ritmo × oro) y duelos @15 con HGB: iguales a sus placebos. Concentración del oro y del daño en los carries
(5 columnas del timeline: porción del oro de ADC + mid, eficiencia daño/oro, máximo y mínimo por jugador, muertes
máximas): +0,0005 [+0,0002; +0,0008] al 15 y +0,0005 [+0,0001; +0,0008] al 20, 3 de 3 bloques, sin réplica en
Diamond-V. Predicción previa del vuelco después del 15 (`HGB_POST15`) con decaimiento por la ventaja: al 20
+0,0007 [+0,0002; +0,0012], 3 de 3 bloques y Diamond-V +0,0007. Aceleración 10 → 15 → 20: ≈ 0.

Las dos que pasaron se volvieron a medir **encima del previo en dos etapas** y con el IC de la media de los tres
bloques, no solo el conteo 3 de 3 (`ingame_g_04_mt_a4_gpu.py`). `HGB_POST15` queda **descartada**: sobre 5b la
media de bloques cruza el cero en los tres minutos (al 20, +0,0004 [−0,0001; +0,0008]) y sobre el modelo final
de abajo da V +0,0002 [−0,0002; +0,0006]. La diferencia de previos $d$ ya trae lo que predecía del vuelco. La
concentración en los carries **se sostiene al 15 y al 20** sobre 5b: media de bloques +0,0005 [+0,0002; +0,0009]
y +0,0007 [+0,0002; +0,0010], V +0,0006 [+0,0003; +0,0009] y +0,0005 [+0,0002; +0,0009], placebo 0; al 10 no
pasa (bloques +0,0001 [−0,0002; +0,0005]). Entra como variable de tres de los cuatro miembros del ensamble.

**5. Modelo en vivo vigente** (`ingame_g_04_mt_a4_gpu.py`, Antigravity, RTX 3070, ~54 min). Un ensamble de
cuatro miembros, todos entrenados con las 92k y causales, más la segunda etapa del previo:

1. la logística A2 con la concentración en los carries;
2. la red multitarea de la laptop (InGameNet: una cabeza para el ganador y cabezas auxiliares, solo en
   entrenamiento, que predicen el futuro de la partida: Δoro y Δxp de los minutos siguientes, oro y kills del
   final) sobre A4;
3. la misma red sobre A4 con la concentración;
4. la misma red como residuo: arranca del logit hacia adelante de la logística 1 y aprende lo que le falta.

Las redes entrenan **hasta** 16 épocas con tasa de aprendizaje coseno y se quedan con la época de menor log loss
sobre el 12 % más reciente de su propio entrenamiento (`_fit_mt_gpu` en `ingame_g_04_mt_a4_gpu.py`: corte
cronológico `int(0,88 n)`, `best_state`). Es parada temprana por selección del mejor punto, sin paciencia; la
versión anterior de este párrafo decía "16 épocas fijas, sin parada temprana", y no es lo que hace el código
(corrección del 2026-11-23, §9.S.78). Ningún hiperparámetro mira V: el 12 % sale de las partidas de ajuste. Los
lotes son de 512 en vez de los 2048 de la laptop (el diseño fijado en el encabezado del script antes de correr:
con 16 épocas son 4 veces más pasos de gradiente). Cada una es el promedio de 2 semillas en los bloques y de 4 en el ajuste final. La combinación es

$$\operatorname{logit} P_{\text{ens}} = \beta_0 + \sum_{k=1}^{4} \beta_k z_k, \qquad
\operatorname{logit} P = a + b\,\operatorname{logit} P_{\text{ens}} + c\, d, \quad
d = \begin{cases} L_{\text{ens10}} - L_{\text{orig}} & \text{después de la foto}\\ 0 & \text{antes}\end{cases}$$

con $z_k$ el logit del miembro $k$. Las dos capas se ajustan hacia adelante: en A2 el promedio simple de los logits
y la segunda etapa neutra, A3 ← A2, A4 ← A2-A3, y V ← A2-A4 con las predicciones de bloque de los miembros
(entrenados con los chunks anteriores a cada bloque). Para V los miembros se reentrenan con los chunks 1-6. Los
pesos de la segunda etapa caen solos con el minuto: $c$ = 0,611 al 10, 0,431 al 15 y 0,236 al 20.

| Minuto ($n_V$ / $n_{DV}$) | V Master+ (AUC / acierto) | Log loss | ECE | Media de bloques A2-A4 contra A2 | Δ V contra A2 | Δ V contra lo mejor de la laptop | Diamond-V (Δ contra A2) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 10 (16.331 / 10.422) | **0,7912 / 71,6 %** | 0,5501 | 0,0085 | 3/3, +0,0027 [+0,0013; +0,0040] | +0,0033 [+0,0020; +0,0046] | +0,0033 [+0,0020; +0,0046] (A2) | 0,7801 / 70,4 % (+0,0022 [+0,0005; +0,0038]) |
| 15 (16.238 / 10.369) | **0,8441 / 76,1 %** | 0,4855 | 0,0152 | 3/3, +0,0022 [+0,0011; +0,0032] | +0,0029 [+0,0020; +0,0038] | +0,0013 [+0,0005; +0,0021] (red) | 0,8416 / 75,6 % (+0,0018 [+0,0007; +0,0030]) |
| 20 (14.971 / 9.691) | **0,8750 / 79,1 %** | 0,4400 | 0,0062 | 3/3, +0,0021 [+0,0012; +0,0030] | +0,0021 [+0,0014; +0,0028] | +0,0011 [+0,0002; +0,0018] (red) | 0,8747 / 78,7 % (+0,0014 [+0,0004; +0,0024]) |

Pasa la regla completa en los tres minutos: IC en V > 0, 3 de 3 bloques con el IC de la media > 0 y réplica en
Diamond-V con IC > 0. El placebo de la segunda etapa ($d$ permutada dentro de cada conjunto) devuelve
exactamente el ensamble sin $d$. Es estable dentro de V: cortando V por la mediana de la fecha (21/09), la ventaja
sobre A2 es +0,0025 / +0,0040 al 10, +0,0020 / +0,0038 al 15 y +0,0013 / +0,0029 al 20 (primera / segunda mitad). De dónde sale la mejora, al 10: cada miembro solo suma +0,0002 a +0,0006 sobre A2 y ninguno con IC > 0,
el ensamble +0,0017 [+0,0008; +0,0026] (la logística y las redes se equivocan en partidas distintas) y la
segunda etapa otros +0,0016 [+0,0006; +0,0026]. Al 15 y al 20 las redes ya ganaban solas, y el ensamble y $d$
agregan entre +0,0002 y +0,0007 cada uno.

Tres salvedades. (a) Al 20, el paso de $d$ solo (0,8746 → 0,8750, V +0,0004 [+0,0001; +0,0006]) no replica por
separado: bloques +0,0001 [−0,0000; +0,0003], Diamond-V +0,0002 [−0,0001; +0,0005]. A2-A4 tiene pocas partidas
posteriores a la foto para medirlo. Se deja igual porque es el mismo mecanismo aceptado al 10 y al 15, se
ajusta sin V y su peso ya se achica solo. El valor conservador al 20 es el del ensamble sin $d$: 0,8746 /
79,0 %. (b) Al 15 el ECE sube de 0,0116 a 0,0152 con mejor log loss (0,4865 → 0,4855): la tasa de victorias de
azul baja de 0,4833 en A2-A4 a 0,4781 en V, y la recalibración ajustada en A2-A4 la arrastra en los deciles
centrales. (c) Descartados en el camino: una sola red con $d$ (al 10, +0,0020 con IC [+0,0001; +0,0038] y sin
réplica en Diamond-V) y `HGB_POST15` encima del modelo final.

Reproducción: `python -u scripts/analysis/ingame_g_04_mt_a4_gpu.py` (GPU; necesita `ingame_cache.npz`, las
predicciones del escalón 9 y 10 que lee `_cargar_priors` de `ingame_g_01_prior_y_gpu.py` y
`data/analysis/brainstorm/gemini_hgb_ctx_p15.npz`). Salidas: `data/analysis/ingame/mt_a4_gpu_92k.csv` y
`pred_mt_a4_gpu_92k.npz`.

**6. Cuánto suma lo previo en vivo** (sección 3 de `eda_modelos_ganador`). Para que las dos barras salgan de
las mismas partidas se midieron dos modelos sin previo. `ingame_g_05_sin_previo.py` (Antigravity, GPU, ~43 min)
es el vigente con $L_{	ext{orig}} = 0$ en los cuatro miembros y sin segunda etapa. Esa versión todavía tiene
información anterior a la partida dentro de las variables: las expectativas por línea (Ridge + LightGBM sobre
draft, composición y jugadores), la composición y los rasgos de jugador. `ingame_estado_puro.py` (CPU, ~30 s)
apaga también eso: logística con las 8 estadísticas por línea, el oro residual (el que no explican CS, kills y
asistencias en ese minuto; es estado, no previo) y las trayectorias, nada más.

| V Master+ (AUC) | Minuto 10 | Minuto 15 | Minuto 20 |
| :--- | :---: | :---: | :---: |
| Estado básico: Δoro, Δxp, Δkills del equipo | 0,7465 / 68,2 % | 0,8136 / 73,4 % | 0,8537 / 77,1 % |
| **Solo el estado** (A2 sin nada previo) | **0,7517 / 68,7 %** | **0,8189 / 73,7 %** | **0,8586 / 77,4 %** |
| + expectativas por línea (A2 sin $L$) | 0,7681 / 69,5 % | 0,8295 / 75,0 % | 0,8660 / 78,2 % |
| + composición, jugadores y redes (ensamble sin $L$) | 0,7805 / 70,8 % | 0,8387 / 75,8 % | 0,8715 / 78,6 % |
| **+ el logit previo y la segunda etapa (vigente)** | **0,7912 / 71,6 %** | **0,8441 / 76,1 %** | **0,8750 / 79,1 %** |

Lo previo suma +0,040 / +0,025 / +0,016 de AUC y +3,0 / +2,3 / +1,7 puntos de acierto. Baja con el minuto,
pero al 20 sigue en +0,016. Las expectativas por línea son el escalón que más suma en los tres minutos: con el
draft y los jugadores se sabe cuánto oro "debería" llevar cada línea, y el oro observado se lee contra eso (la
"sorpresa" de cada línea, §9.S.52). Parte de la brecha es capacidad y no información: con el previo, el ensamble le saca +0,002 a +0,003 a
la logística (punto 5). Todas las filas pasan sus bloques A2-A4 y Diamond-V en la misma dirección
(`sin_previo_92k.csv`, `estado_puro.csv`).

**7. Exploración después del vigente: elegir en A2-A4 y mirar V una vez** (Antigravity, `ingame_g_06*_a24.py` e
`ingame_g_07_final_v.py`). Regla de la madrugada: toda la selección en A2-A4 sin leer V, y V una sola vez por
candidato final, como incremento sobre el vigente con la segunda etapa incluida. Probado en A2-A4:

* árboles residuales (CatBoost en GPU y LightGBM) con parada temprana sobre un 12 % cronológico interno: solos
  pierden −0,0008 a −0,0023 contra el ensamble y el ensamble les da peso negativo. Descartados (la laptop los
  había corrido sin parada temprana; con ella da lo mismo);
* el escalado de los campeones (`cbias`) por la cercanía de la partida, $\Delta c \cdot e^{-\lvert g\rvert/3}$ y
  $\Delta c \cdot e^{-\lvert z\rvert}$ ($z$ el logit de la etapa 1): suma en la logística solo al 20 (+0,0006
  [+0,0002; +0,0010]);
* una red con atención 5 × 5 entre líneas (InGameNetV2): con el atajo lineal duplicado empeora (−0,0010 a
  −0,0021); sin él, +0,0002 en bloques;
* cuatro bloques nuevos del estado (saturación de la ventaja, recompensas por kills, "afinidad" de quién está
  alimentado (el escalado o el daño del campeón por su Δoro estandarizado) y divergencia entre XP y oro): solo el
  tercero se sostiene, +0,0002 al 15 y al 20.

El candidato final al 15 (la afinidad metida en las variables por línea de la red con atención, dentro del
ensamble, más la segunda etapa) le ganó al vigente en A2-A4 (+0,00034 [+0,00004; +0,00064], 3 de 3) y en V:

| Minuto 15 | V Master+ (AUC / acierto) | Δ V | Diamond-V (Δ) | Placebo (variables permutadas), Δ V |
| :--- | :---: | :---: | :---: | :---: |
| Vigente | 0,8441 / 76,1 % | — | 0,8416 | — |
| Candidato | 0,8445 / 76,2 % | +0,00032 [+0,00003; +0,00059] | 0,8418 (+0,00019 [−0,00016; +0,00055]) | +0,00006 [−0,00016; +0,00028] |

Al 20 el candidato no pasa (V +0,00015 [−0,00006; +0,00036]). **No se promueve**, por tres razones. (a) No hay
réplica independiente: los bloques A2-A4 fueron el conjunto de selección entre unas 20 variantes, así que su IC ya no
cuenta como réplica, y la única réplica limpia, Diamond-V, cruza el cero. (b) V se miró dos veces (15 y 20), y una
cota inferior de +0,00003 no sobrevive a esa corrección. (c) El costo es un modelo distinto solo al 15, con una
arquitectura nueva, por +0,0004 de AUC y +0,1 puntos, con el ECE algo peor (0,0152 → 0,0159). Queda documentado como el
mejor candidato si llegan datos nuevos (tarea de los timelines), para volver a medirlo ahí.

**UMAP del estado al minuto 15** (§9.S.58): no hay una región de partidas que la logística en vivo no vea. Junto
con lo anterior, el modelo en vivo está cerca de su techo con los datos que hay: el modelo vigente suma
+0,0021 a +0,0033 sobre A2, y los ~25 intentos que no entraron se quedaron en ±0,001. El salto siguiente
necesita información nueva del minuto $m$: objetivos (dragones, Heraldo, Barón, torres, placas), oro sin gastar
y estadísticas reales del campeón, que el crawler no guardó (tarea "Volver a bajar los timelines").

#### S.61. Ideas 43-45 del brainstorming: estado esperado, builds raras y partidas difíciles (2026-11-16)

Tres ideas sobre el híbrido + mapa de §9.S.51 (V Master+ 0,6531 / 61,1 %), con la regla de §9.S.52: IC de la
media de los bloques A2-A4 **y** IC en V. Ninguna entra.

*Idea 43: estado esperado a los 10/15/20 con los promedios del jugador* (`brainstorm_43_estado_esperado.py`). Si
el historial dice cómo suele estar cada jugador a los 10/15/20 minutos (oro, XP, CS, kills, contra su rival de
línea), se puede armar el estado esperado de la partida antes de jugarla y pasarlo por el modelo en vivo
($P_{\text{en vivo}}(\widehat{\text{estado}})$). Con la historia acumulada o las últimas 20, la $P$ pasa los
bloques (3 de 3, +0,0006 a +0,0008, IC > 0), pero en V da +0,0003 a +0,0005 con IC que toca 0; con la sesión del
día, ≈ 0. El estado esperado crudo (sin pasar por el modelo en vivo) no pasa los bloques. Candidato marginal: es
una transformación no lineal de márgenes que el modelo ya tiene.

*Idea 44: builds raras y conjuntos de ítems* (`brainstorm_44_builds.py`). Por jugador, con sus partidas
anteriores: rareza de sus ítems, pares, tríos y combinación completa, y "calidad" de esos conjuntos (winrate
empírico-bayesiano), en general y con el campeón elegido. Los ítems completos se definen con los datos (ítems con
frecuencia ≥ 0,7 veces más alta en partidas de más de 32 min que en las de menos de 22), porque no se pudo bajar
`item.json`. Sobre el modelo: ≈ 0 (la mejor, "todo", V +0,0002 con IC en 0; acierto 61,1 % en todas). En lo
descriptivo, la calidad de los pares de ítems va de 41 % a 62 % de victoria entre quintiles extremos (partidas de
20 a 25 min con los mismos ítems), pero está confundida por el estado de la partida: la "rareza" de una build
refleja sobre todo ir atrasado (menos ítems completos), y la combinación rara en sí no dice nada.

*Idea 45: partidas difíciles y fáciles* (`brainstorm_45_dificultad.py`). "Difícil" no puede ser "el modelo
duda" (circular); la pregunta es si, **a igual confianza**, hay partidas donde el modelo acierta más o menos de
lo que promete (exceso = acierto − media de $\max(p, 1-p)$). Contexto simétrico de 259 variables; PCA de 20,
K-Means con $k$ = 8 y un meta-modelo HGB que predice si el modelo acierta. K-Means encuentra un grupo fácil
(equipos muy desparejos; 67,7 % de acierto en V) que el modelo ya conoce (exceso ≈ 0 en los bloques) y uno caótico
(asesinos y agresión temprana; exceso −1,6 y −0,8 puntos, algo sobreconfiado). Ninguno sirve: el meta-modelo con
contexto predice el acierto **peor** que $\lvert\text{logit}\rvert$ solo (AUC 0,582-0,584 contra 0,591) y la
recalibración por componentes o grupos empeora V (0,6485-0,6518 contra 0,6531). El UMAP de §9.S.58 llega a lo
mismo con otro espacio.

Archivos: `historial/ideas_posteriores/brainstorm_43_estado_esperado.py`, `brainstorm_44_builds.py` y
`brainstorm_45_dificultad.py`; resultados en `data/analysis/brainstorm/` (`estado_esperado.csv`, `builds.csv`,
`builds_descriptivo.csv`, `dificultad_*.csv`).

#### S.62. Experto tabular TabM para el escalón 9 y rasgos que no se pueden construir (2026-11-17)

Idea 4 de las "ideas abiertas" de `TAREAS.md`: una red tabular moderna como experto extra del stacking previo a la
partida, distinta de las redes por jugador (Wide & Deep, NeuralAC) ya probadas. Hecho por Antigravity en la GPU
(`historial/ideas_posteriores/tabm_esc9_gpu.py`, ~15 min).

**Modelo.** TabM (Gorishniy et al., 2024): un MLP con *BatchEnsemble*, $k = 32$ submodelos que comparten los pesos
y se diferencian por vectores multiplicativos de rango 1 por capa,

$$h^{(i)}_{\ell+1} = \sigma\big(((h^{(i)}_\ell \odot r^{(i)}_\ell)\, W_\ell) \odot s^{(i)}_\ell + b^{(i)}_\ell\big), \qquad
\hat p = \tfrac{1}{k}\textstyle\sum_i \sigma(f^{(i)}(x)),$$

con embeddings periódicos PLR de cada numérica ($x \mapsto \text{ReLU}(W[\sin 2\pi c x, \cos 2\pi c x])$, 12
frecuencias, dimensión 4) sobre las 653 numéricas del escalón 9, una rama aparte para las 851 columnas del draft y
la indicadora de foto como condicionamiento. Antisimetría exacta como en el resto del proyecto:
$f(x) = \tfrac12(g(x) - g(-x)) + b_{\text{lado}}$. Parada temprana con un holdout cronológico interno del train
(épocas 6-11), 2 semillas. El stacking se ajusta en A2-A4 hacia adelante, sin V.

| V Master+ | AUC / acierto | Δ V contra el escalón 9 | Diamond-V |
| :--- | :---: | :---: | :---: |
| Escalón 9 (logística) | 0,6739 / 62,5 % | — | 0,6437 / 60,3 % |
| Escalón 10 ($L_{\text{ens10}}$) | 0,6749 / 62,7 % | +0,0010 [+0,0001; +0,0018] | 0,6452 / 60,3 % |
| TabM solo | 0,6693 / 62,4 % | −0,0046 [−0,0068; −0,0028] | 0,6410 / 60,0 % |
| Stacking escalón 9 + TabM (peso de TabM 0,014) | 0,6739 / 62,6 % | +0,0000 | 0,6437 / 60,5 % |
| Stacking escalón 10 + TabM (peso −0,119) | 0,6749 / 62,6 % | +0,0010 (≈ escalón 10) | 0,6451 / 60,3 % |

**Descartado.** Sobre estas variables la red no llega a la logística y el stacking no le da peso. Coincide con
el techo de la información previa de §9.S.52: con ~1.500 columnas ya diseñadas a mano (márgenes, Elo, contracruces),
lo que una red podría aprender de las interacciones ya está en las variables o es ruido con unas 100k partidas de entrenamiento. La
alternativa RealMLP no se probó; se espera lo mismo.

**Rasgos que no se pueden construir con los datos actuales.** El punto 3 de las ideas abiertas (historial de AFK,
abandonos y rendiciones del jugador, y nivel en otras colas, como en TrueSkill 2) necesita columnas que el crawler
no guardó: `match_participants` no tiene `timePlayed` ni flags de rendición, y la BD solo tiene la cola 420
(272.849 partidas de SoloQ, sin flex ni normales). Haría falta volver a bajar Match-V5.

**Ritmo de juego** (`play_rate`, partidas por día del jugador antes de la partida; Antigravity, solo en A2-A4, sin
mirar V): 0 de 3 bloques con cualquiera de las tres versiones (media, últimas 48 h, las 8 juntas: −0,0001 a
−0,0002). Descartado sin llegar a V.

#### S.63. Oro equivalente de kills, CS y objetivos (primera sangre, primera torre, dragones, vacuolarvas, heraldo) (`eda_modelos_ganador.ipynb` §3.1) (2026-11-17)

**Pregunta.** ¿Cuánto oro vale una ventaja de la partida? Se mide como la ventaja de oro del equipo a los 15 que da
la misma probabilidad de ganar. Script: `scripts/analysis/oro_equivalente_roles.py`, que escribe
`data/analysis/oro_equivalente_roles.csv`.

**Método.** Primero se ajusta una curva de victoria espejada y sin intercepto, $\Pr(\text{gana azul}) =
\sigma(b_{oro}\,\Delta g_{15})$, donde $\Delta g_{15}$ es la diferencia de oro azul − rojo en miles. Cada partida
entra dos veces, una con los lados invertidos, así que no hay ventaja de lado. Para una ventaja codificada como
+1 (azul), −1 (rojo) o 0 (nadie) se reportan dos números:

* *equivale a* $= 1000\,\mathrm{logit}(p)/b_{oro}$, donde $p$ es la fracción de partidas que gana quien tiene la
  ventaja. Es el efecto total, e incluye todo lo que viene con la ventaja: el oro que da y el hecho de que quien
  la tiene suele ir ganando. No es causal.
* *a igual oro* $= 1000\,\beta_e/\beta_g$, a partir de una logística espejada con $\Delta g_{15}$ y la ventaja
  juntas. Es lo que la ventaja agrega cuando ya se conoce el oro.

Las kills y el CS se miden contra el rival directo: la diferencia de una línea contra la misma línea rival a los
15, por tramos, sin el soporte y con el ADC aparte. La primera sangre y la primera torre salen de los 16.000
timelines del piloto (§3.3 del notebook). Ahí se ve quién mata a quién y en qué línea cae la torre, así que se
puede sacar al soporte. Resultado: una kill vale ~470 y 10 CS valen ~460-540; la primera sangre sin soporte vale
~780 y la primera torre ~1.930. *A igual oro*, todo eso queda en ~0: esas ventajas valen lo que valen porque dan
oro. La única excepción es tener 3 o más kills de ventaja en una línea, que da +140 a +190.

**Descartado: medir la primera sangre y la primera torre con el dataset grande (185.871 partidas).** Ese dataset
guarda `first_tower_kill` y `first_blood_kill` como marcas por jugador. En el 3,7 % de las partidas ningún jugador
tiene la marca de primera torre, porque la tiraron los súbditos. Con la primera sangre pasa en el 0,3 %. La versión
anterior codificaba "azul = algún jugador azul tiene la marca", así que esas partidas se contaban como del rojo.
Por eso la tasa de primera torre del azul daba 47,8 % en vez de ~50 %. El dataset grande tampoco dice la línea ni
el rol. Se usa solo el piloto, y `embeddings_partidas.py` pasa a codificar `fb_azul`/`ft_azul` como 1 / 0 / 0,5
(0,5 = nadie tiene la marca). La caché `D_*` toma el cambio la próxima vez que se corra ese script. Con el sesgo,
la primera torre daba 1.932 en el dataset grande y 1.928 en el piloto, y *a igual oro* +42 contra −89: lo mismo
dentro del ruido.

**Presentación (pedido del usuario, 2026-11-17).** La §3.1 muestra una tabla (WR de quien tiene la ventaja y oro equivalente) y dos gráficos de barras en seaborn, uno con el WR y otro con el oro. Las ventajas del mismo tipo van superpuestas en una sola barra, la más grande atrás, y cada familia tiene su color (kills, CS, primera sangre, primera torre), con tonos más oscuros para los niveles más altos. Se agrega la primera sangre **con soporte**, que incluye todas las primeras sangres del piloto (15.847): vale ~700 de oro, contra ~780 sin soporte (11.564 casos). Del notebook se quitan los casos y *a igual oro*, que siguen en el CSV.

**Ampliación: por rol, con la otra ventaja pareja y con monstruos (pedido del usuario, 2026-11-17).** Las barras pasan a agruparse por rol (TOP, Jungla, Mid, ADC), y cada rol tiene cuatro: kills, kills con el CS parejo, CS, y CS con las mismas kills. "Parejo" para el CS es $|\Delta CS|\le 5$. Se eligió así porque la diferencia de CS por línea a los 15 tiene un desvío de 23 a 30, así que ±5 es una fracción chica de la variación, más chica que el ancho de los tramos (10 CS), y deja de 20.000 a 30.000 casos por rol. Una tolerancia de ±3 era más estricta pero dejaba un 35 % menos de casos, y ±10 ya es un tramo entero de CS. Para las kills, "las mismas" es $\Delta K = 0$, porque son enteros chicos. Con la otra ventaja pareja, cada una vale bastante menos. En ADC, +1 kill pasa de 472 a 338 de oro y +10 CS de 540 a 258: buena parte del valor de cada una era que venía junto con la otra (la correlación entre $\Delta K$ y $\Delta CS$ es 0,50 en ADC y 0,57 en top). El top es el rol que menos vale en todo, y el CS de la jungla el que más (+40 o más: 75 % de WR, ~2.900 de oro).

Los monstruos salen de los eventos `ELITE_MONSTER_KILL` del piloto, que ya estaban guardados y no se usaban; no se bajó nada nuevo. Para cada monstruo se toma el lado de `killerTeamId`, o el de `killerId` si falta. Se miden:

* dragones: el primero, los dos primeros del mismo equipo y los tres primeros del mismo equipo;
* vacuolarvas: son 3, en un solo spawn a los ~9 min, así que el reparto es 2-1 o 3-0;
* heraldo: uno, a los ~16 min.

Resultados (WR de quien la tiene, y oro equivalente):

| Ventaja | WR | Oro |
| :--- | ---: | ---: |
| Primer dragón | 62,6 % | 1.314 |
| Dos primeros dragones | 75,0 % | 2.800 |
| Tres primeros dragones | 85,2 % | 4.460 |
| Heraldo | 66,8 % | 1.783 |
| Vacuolarvas 3-0 | 56,8 % | 702 |
| Vacuolarvas 2-1 | 51,8 % | 185 |

El heraldo y la mayoría de los dragones (mediana de ~17 min) caen después del minuto 15, así que en parte miden cómo sigue la partida, y por eso sus *a igual oro* son grandes (dragón +755, heraldo +612). Las vacuolarvas caen antes del 15, y su *a igual oro* da ~0, como el de la primera torre. Más tarde se sacó la tabla del notebook a pedido del usuario; los números siguen en el CSV.

**Versión final del gráfico (2026-11-17).** Quedan solo las variantes con la otra ventaja pareja (kills con CS
±5 y CS con las mismas kills), por línea. Las ventajas "solas" se sacan del gráfico porque mezclan las dos, pero
siguen en el CSV. Cada tramo tiene entre 2.400 y 14.400 casos. El IC 95 % del oro sale por el método delta:
como oro $=1000\,\mathrm{logit}(p)/b$, el error es $\mathrm{ee}=1000/(b\sqrt{n\,p(1-p)})$, que se puede escribir
como $	ext{oro}/(\mathrm{logit}(p)\sqrt{n\,p(1-p)})$ y así no hace falta guardar $b$. Da ±90 a ±220 de oro y
±0,8 a ±1,9 puntos de WR. Alcanza para los saltos grandes (CS de jungla contra top), pero no para tramos vecinos.
Se agrega una **escala de oro real** en grises: el promedio de la diferencia de oro a los 15 del jugador con la
ventaja contra su rival de línea, con los 4 roles juntos. +1 kill con CS parejo da ~340, +3 o más ~1.390, +10 CS
con las mismas kills ~250 y +40 o más ~1.230.

**¿Está sesgada la conversión a oro?** Se discutió si es circular, ya que el oro sale en parte de lo mismo que se
mide. *Equivale a* no usa el oro de las partidas con la ventaja: usa solo su WR y la pendiente $b$ de la curva
ajustada sobre todas las partidas. No es circular. Que una kill valga más o menos su oro real es justamente lo
que se espera si la kill importa a través del oro que da. La circularidad sí afectaba a *a igual oro*: al
controlar por el oro del equipo se le saca a la ventaja el oro que ella misma produjo, así que ~0 era esperable.
Esa es una razón más para no mostrarlo. Quedan dos sesgos reales:

* *Confusión.* Quien tiene la ventaja suele ir ganando por otras cosas, así que *equivale a* es una asociación
  total, no un efecto causal.
* *Atenuación.* $\mathrm{logit}(ar p)$ no es el promedio de los logits. Si el efecto varía entre partidas,
  *equivale a* tiende a quedarse corto, y la curva supone además que el WR es lineal en logit del oro.

La coincidencia con el oro real (mid y ADC casi iguales, jungla arriba, top abajo) indica que los dos sesgos son
chicos, o que se compensan.

**Advertencia de lectura.** El 79 % de las primeras torres del piloto cae antes del minuto 15, y entonces su oro ya
está en $\Delta g_{15}$. El 21 % restante cae después. *A igual oro* mezcla una ventaja ya absorbida por el oro con
una que todavía no pasó.


#### S.64. Timelines completos con eventos: piloto de 16.000 y descarga de toda la base (2026-11-17/18)

**Qué faltaba.** De cada timeline de Match-V5, el crawler guardó solo una foto por jugador en los minutos 10, 15
y 20 (`totalGold`, XP, nivel, CS, daño y K/D/A contado a partir de `CHAMPION_KILL`; `src/database.py`,
`save_match_timeline`). Todo lo demás se descartaba y el JSON no se guardaba:

* los demás minutos;
* los eventos de monstruos (dragones y alma, heraldo, vacuolarvas, barón);
* los edificios (torres, placas, inhibidores);
* ítems, wards y habilidades;
* el detalle de cada kill;
* `currentGold` (oro sin gastar, y con él el oro invertido), `championStats` y las posiciones.

**Piloto.** `scripts/collection/timelines_eventos.py` guarda el **JSON crudo completo comprimido con zlib** en
una base aparte, `data/timelines_eventos/timelines.db`, que no toca `lol_las.db` ni los CSV (AGENTS.md §3.G).
Toda petición pasa por `RiotAPIClient` y su rate limiter (§3.B). La muestra fue de 8.000 partidas Master+ del
ajuste y 8.000 de V, intercaladas. Bajaron las 16.000 en 166 min, 1,1 GB, sin ninguna partida sin timeline.
Sobre la logística en vivo, con los eventos en una segunda etapa, el AUC en V Master+ subió +0,006 a +0,008 en
los minutos 10, 15 y 20, con IC 95 % por encima de 0 (`ingame_eventos.py`; detalle en TAREAS). Fue la mayor
mejora del modelo en vivo desde §9.S.52. Lo que pesa son los objetivos (dragones), la primera sangre, el oro
sin gastar y los jugadores muertos en el minuto. La §3.1 de `eda_modelos_ganador` (§9.S.63) lee estos mismos
timelines.

**Extensión a toda la base (`--todas`, 2026-11-17).** Agrega a la selección las 272.849 partidas de
`lol_las.db`, por orden de prioridad:

1. V: 16.428;
2. Master+ del ajuste (chunks 1-6 de `ingame_cache.npz`): 94.927;
3. resto de la caché en vivo: 76.462;
4. resto de la base: 85.032.

Dentro de cada grupo el orden es al azar con semilla fija. Así, cualquier corte deja completo lo que más usa el
modelo en vivo.

**Tamaño.** Cada timeline pesa ~67 KB comprimido y ~750 KB como JSON, así que la base entera son **~18 GB**
(sin comprimir serían ~200 GB). El crudo va aparte y se procesa **una sola vez** a tablas compactas:

* objetivos con su minuto: ~100 MB;
* foto por jugador minuto a minuto hasta el 20: ~2 GB.

Una pasada sobre el crudo cuesta ~3 min cada 16.000 partidas con un núcleo, es decir ~50-60 min para la base
entera, y los notebooks no deberían pagar eso cada vez. Alternativas descartadas:

* guardar solo los campos que se usan hoy: obligaría a volver a pedir todo a la API ante cualquier pregunta
  nueva;
* JSON sin comprimir: ~11 veces más disco sin ninguna ventaja;
* meterlo en `lol_las.db`: la base principal crecería de 2,2 a ~20 GB y se volverían lentos los respaldos a
  Drive y las auditorías de `run.py --solo-validar`.

**Primera noche (17/11 22:47 a 18/11 06:40).** Bajaron 46.706 timelines en 472 min (~5.940 por hora netas,
con 2 keys). Hubo 15 respuestas 429, 5 timeouts y 0 errores. **V queda completa** (16.428) y las Master+ del
ajuste van 46.278 de 94.927. La base ocupa 4,36 GB. Faltan 210.143, unas ~35 h a ese ritmo. Para retomar:
`python scripts/collection/timelines_eventos.py --todas`, que saltea lo que ya está guardado.

**Respaldo a Drive (2026-11-18).** `backup_to_drive.py` (y por lo tanto `run.py --respaldar`) sube también
`timelines.db` a `<Drive>/timelines_eventos/`, con un manifiesto que guarda la cantidad de timelines. La base de
timelines no es como `lol_las.db`, y por eso tiene tres reglas propias:

1. **Se sube solo si creció.** Si la cantidad de timelines y el tamaño coinciden con el manifiesto de Drive, se
   saltea. Subir 18 GB en cada cierre de recolección no tiene sentido.
2. **No hay snapshot local intermedio.** Para `lol_las.db` se usa la API de backup de SQLite hacia un archivo
   temporal, pero una copia temporal de 18 GB no entra en el disco (quedan ~30 GB). En su lugar se copia el
   archivo mientras una conexión de lectura mantiene un lock compartido, así nadie puede confirmar escrituras a
   mitad de la copia.
3. **No se copia con la descarga corriendo.** Si el descargador escribiera, tendría que esperar todo lo que dura
   la copia, y con su espera de 60 s se caería. Por eso el descargador renueva un latido en
   `data/timelines_eventos/descargando.lock` cada vez que guarda (~cada 15 s) y lo borra al terminar. El respaldo
   saltea la base si el latido tiene menos de 5 minutos. Si el descargador muere sin borrarlo, el latido queda
   viejo y el respaldo vuelve a funcionar solo.

Alternativas descartadas:

* detectar el proceso con `psutil`: no está en `requirements.txt`, y en Windows `os.kill(pid, 0)` termina el
  proceso en vez de consultarlo;
* bloquear la base con `BEGIN EXCLUSIVE` durante la copia: tira abajo al descargador.

Se mantiene la misma guarda contra retrocesos que en `lol_las.db`: si la base local tiene menos timelines que
Drive, no se sube salvo con `--force`. Primera copia: 62.706 timelines, 4,36 GB.

**Mediciones repetidas con 62.706 timelines (2026-11-18).** Con V entera y 46.278 partidas Master+ del ajuste,
se volvieron a correr las dos mediciones que dependen de los timelines.

* *Aporte de los eventos al modelo en vivo* (`ingame_eventos.py`; §3.3 de `eda_modelos_ganador`, que ahora
  lee el CSV en vez de tener los números escritos a mano). Al sumar la prioridad 1 a la logística en vivo, el
  ΔAUC en V Master+ (~16.300 partidas) da:

  | Minuto | ΔAUC | IC 95 % | Tramos por fecha que mejora | En el piloto |
  | ---: | ---: | :--- | :---: | :---: |
  | 10 | +0,0056 | [+0,0039; +0,0073] | 3/3 | 2/3 |
  | 15 | +0,0060 | [+0,0046; +0,0075] | 3/3 | 3/3 |
  | 20 | +0,0070 | [+0,0054; +0,0084] | 3/3 | 2/3 |

  Las magnitudes son las del piloto (+0,006 / +0,006 / +0,008), con la mitad de margen, y ahora mejoran los 3
  tramos en todos los minutos. La base es algo más floja (0,787 al 10 contra 0,790), porque se ajusta con las
  ~48.000 Master+ sin timeline completo. Los dragones son la variable de más peso en los tres minutos.
* *Oro equivalente de los objetivos* (§3.1). Los márgenes bajan de ±80-190 a ±41-98, y los valores apenas
  cambian:

  | Ventaja | Oro equivalente |
  | :--- | ---: |
  | Primera sangre con soporte | 628 ± 41 |
  | Primera sangre sin soporte | 722 ± 49 |
  | Primera torre | 1.983 ± 44 |
  | Primer dragón | 1.281 ± 42 |
  | Dos primeros dragones | 2.785 ± 60 |
  | Tres primeros dragones | 4.441 ± 93 |
  | Vacuolarvas 3-0 | 775 ± 46 |
  | Vacuolarvas 2-1 | 348 ± 98 |
  | Heraldo | 1.856 ± 46 |

  Lo único que cambia de lectura son las vacuolarvas 2-1: con el piloto (185 ± 190) no se distinguían de 0, y
  ahora sí. Las kills y el CS no ganan precisión, porque ya usaban las ~185.000 partidas de la caché en vivo.
  El margen del CS de top (±110 sobre ~160) sale de que todo oro equivalente arrastra ±100 por cada punto de
  error del WR.

**Qué grupo de eventos aporta (2026-11-18, `eventos_ablacion.py`).** Es una estimación rápida, pedida para saber
cuánto conviene agregar al modelo en vivo. Usa la misma base y la misma segunda etapa que `ingame_eventos.py`, y
mide cada grupo de dos formas: *solo* (base + ese grupo, contra la base) y *arriba de los demás* (todos los grupos,
contra todos menos ese). Los IC son por bootstrap pareado en V Master+.

| Grupo | ΔAUC al 10 / 15 / 20 | Δacierto |
| :--- | :--- | :--- |
| **Dragones (y punto de alma), solo** | **+0,0044 / +0,0049 / +0,0056** (todos con IC por encima de 0) | +0,3 a +0,5 pts arriba de los demás |
| Todos los grupos juntos | +0,0055 / +0,0059 / +0,0069 | +0,32 / +0,23 / +0,61 pts |
| Muertos en el minuto y shutdowns, arriba de los demás | ≈0 / +0,0004 / +0,0009 | |
| Oro sin gastar / invertido, arriba de los demás | ≈0 / +0,0005 / +0,0003 | |
| Vacuolarvas, arriba de los demás | +0,0008 / ≈0 / ≈0 | |
| Heraldo | 0 / 0 / ≈0 | |
| Torres, placas e inhibidores | ≈0 | |
| Primera sangre | ≈0 | |

Los dragones solos explican ~80-90 % de la mejora. El heraldo da 0 al 10 y al 15 por construcción, porque aparece
a los ~15 min. Torres, placas y primera sangre ya están en el oro y en las kills. Esto coincide con la §3.1 de
`eda_modelos_ganador`: el dragón es la única ventaja cuyo valor no pasa por el oro (buffs permanentes; *a igual
oro*, +755).

El acierto se mueve poco porque solo cambia cuando la probabilidad cruza el 50 %. Con ~16.000 partidas, su margen
pareado es de ±0,2-0,3 pts.

*Salvedad:* la base es la logística en vivo ajustada con las ~48.000 Master+ sin timeline completo (0,8726 / 78,7 %
al 20), no el ensamble vigente (0,8750 / 79,1 %). Si la mejora se trasladara entera, el vigente quedaría en
~0,880 / ~79,5 % al 20. Lo más probable es que sea algo menos, porque las redes ya captan parte de lo mismo por el
oro residual y las trayectorias.

`ingame_eventos.py` ahora lee los timelines en paralelo y guarda las variables en `variables_eventos.npz`. Se
reusa mientras no cambie el conjunto de partidas, y la lectura bajó de 26 min a ~6.

**Limitación: copiar a `G:` no es lo mismo que subir a la nube.** `backup_to_drive.py` copia a la carpeta local de
Drive para escritorio y ahí termina; la subida real la hace Drive en segundo plano, y el script no la espera. La
primera copia de `timelines.db` (4,36 GB) quedó en `G:` a las 08:26, pero llegó a la nube recién ~30-45 min
después (~3 MB/s de subida). Mientras tanto, desde otra PC se veían la carpeta y el manifiesto, pero no la base.
Con `lol_las.db` (2 GB) pasa lo mismo, aunque se nota menos. Hay que dejar la PC prendida con Drive abierto
hasta que el archivo tenga identificador de nube. Para saberlo, en la base interna de Drive
(`%LOCALAPPDATA%/Google/DriveFS/<cuenta>/metadata_sqlite_db`, tabla `items`) un archivo pendiente tiene un `id`
que empieza con `local-`, y uno ya subido tiene el identificador real de Drive.

**Descarga por lotes con `--ids` en lugar de `--todas` (2026-11-24, pedido del usuario).** `--todas` mete en
`seleccion` las ~273.000 partidas de `lol_las.db` y baja todo lo pendiente. Eso incluye ~150.000 que el modelo en
vivo con eventos no usa: chunk 0, Diamond viejo de los chunks 1-6, `diamante_despues`, `sin_rango` e inválidas.
Son ~25 h de cuota de API sin retorno. Se reemplaza por tres lotes explícitos, armados cruzando solo
`ingame_cache.npz` y `data/splits/match_split.csv` (sin abrir `test/full_dataset.csv`), y bajados en este orden:

| Lote (`--grupo`) | Criterio | Partidas | Ya en la base al armarlo | Pendientes |
| :--- | :--- | ---: | ---: | ---: |
| `diam_v` | `chunk == 8` (Diamond-V + las 887 Master+ que entran a V con la regla unificada del split) | 10.506 | 7.911 | 2.595 |
| `a24_master` | `chunk ∈ {4,5,6}` y (`master` en la caché **o** `tier ∈ {MASTER, GRANDMASTER, CHALLENGER}` en el split) | 14.513 | 14.370 | 143 |
| `test_master` | `split == 'test'` en el split | 38.013 | 0 | 38.013 |

Los tres son disjuntos. Diamond-V va primero porque es lo que hace falta para validar la réplica. El pedido
estimaba ~7.800 pendientes en `a24_master`. Quedaron 143 porque los Master+ de la caché en los chunks 4-6 ya
estaban completos en el grupo `train` de §9.S.64, y casi todos los que entran solo por el `tier` del split habían
caído en el grupo `cache`, que ya se había bajado en ~75 %. Las partidas pendientes suman 40.751 (~6,7 h a
~6.100/h neto con P0+P1).

*Diseño.* `--ids ARCHIVO --grupo NOMBRE` anota el lote en una tabla nueva `lote(grupo, match_id)` con PK
compuesta y suma a `seleccion` solo las que no estaban, con `orden` a continuación del existente. Baja únicamente
las pendientes **de ese archivo**, en el orden del archivo. Dos alternativas se descartaron:

1. Reusar `pendientes()`: devuelve todo lo pendiente de `seleccion`, y ahí siguen anotadas las ~104.000 de los
   grupos `cache`/`resto` que dejó `--todas`. Correr un lote habría arrastrado la descarga entera.
2. Reescribir `seleccion.grupo`: cambiaría la historia de §9.S.64 (qué se bajó por qué criterio) y no admite que
   una partida pertenezca a dos lotes.

`--resumen` muestra, además de los grupos, el avance por lote (bajadas / total / sin timeline). Tests:
`tests/test_timelines_eventos.py`.

*Cuidado:* correr el script **sin** `--ids` (o con `--todas`) retoma todo lo pendiente de `seleccion`, residuo
de `cache`/`resto` incluido.

#### S.65. Rendimiento y corrección de la recolección, el cierre y el enlace de LP (2026-11-18)

**Padrón y cuota.** League-V4 cifra cada PUUID con el proyecto de la clave que
atiende la petición. `RiotAPIClient` ya identifica el proyecto por clave contra
la configuración original, incluso después del preflight. `DataCollector` usa
esa identidad para sincronizar únicamente con P0 y comparte su limitador con el
cliente dedicado. Si P0 falta, está inactivo o la clave tiene proyecto desconocido,
omite la sincronización. Se descartó resolver el proyecto por posición porque
un pool filtrado con solo P1 tiene a P1 en la posición 0. La recarga en caliente
asocia proyectos antes de descartar placeholders y descarta configuraciones de
índices inválidas. El diagnóstico de claves usa el mismo transporte limitado
que las peticiones normales, conserva el código HTTP del primer sondeo y, en
429, espera el `Retry-After` antes de continuar. Las ventanas y penalizaciones
del limitador usan un reloj monotónico para conservar sus plazos cuando cambia
la hora del sistema. Los timestamps de SQLite siguen en UTC.

**Consultas de cobertura.** El latido consulta `COUNT(*) FROM matches` cada
60 s y reutiliza cobertura durante 300 s. Antes ejecutaba dos cálculos de
cobertura por latido y un tercero al vencer la caché. El incondicional y el
del resumen se ejecutaban siempre. El reporte de cierre usa el cálculo
del propio resumen. Se mantuvo el cálculo en el despachador para conservar
un único ciclo de vida y evitar lecturas de cobertura simultáneas por sesión.
El monitor toma los datos lentos una vez por minuto y entre esas lecturas solo
consulta total de partidas y último latido. Usa el cálculo oficial por persona
y considera activo un latido de hasta 360 s de antigüedad, que también cubre
los intervalos de 4–5 min observados en corridas anteriores.

Para una persona $i$, con conjunto $P_i$ de PUUIDs en el mapa, el conteo es
$g_i = \left|\{m \mid \exists p \in P_i,\ (m,p) \in \text{match\_participants}\}\right|$.
La cobertura a umbral $k$ en el padrón $R$ es
$\sum_{i\in R}\mathbf 1[g_i\ge k]$, por tier y en total. En el conteo individual se incluye
el PUUID consultado para que siga encontrando sus partidas si falta en el mapa.
Con `player_id_map` materializado se consulta solo su componente, reconstruido
al iniciar y al cerrar una sesión con partidas nuevas. Se descartó unir el
resultado con la vista legacy porque esa rama vuelve a unir por nombre y puede
sumar cuentas distintas que el Union-Find separó al detectar proyectos
superpuestos. La vista sigue disponible para bases con el mapa vacío.
`_init_tables` compara su SQL con `sqlite_master` y migra la vista de forma
atómica si su definición cambió. Un resumen sobre una base ya migrada evita
ese DDL. La inicialización sigue pudiendo completar el esquema de una base antigua.

**SQLite y mediciones.** `cache_size=-262144` fija un máximo sugerido de 256 MiB
por conexión, que SQLite ocupa según las páginas consultadas. `temp_store=MEMORY`
evita archivos temporales en las agrupaciones de cobertura. Se aplican a las
conexiones del gestor y a las de lectura del monitor, la validación y la
auditoría. El export conserva el almacenamiento temporal de SQLite por defecto
debido al pico simultáneo de memoria de pandas. Las siguientes mediciones
usaron la base real en `mode=ro`, sin inicializar `DatabaseManager` sobre ella.

| Comparación | Antes | Después | Comprobación |
| :--- | ---: | ---: | :--- |
| Cobertura oficial | 20,91 s | 11,53 s con los PRAGMA y 10,29 s al omitir la vista cuando hay mapa | Misma salida por tier y total. Padrón de 40.821 personas, 14.249 en coverage50 y 1.462 en coverage300. |
| Conteo por jugador, cuatro mediciones | 2,24–2,59 s | 0,14–2,03 ms | 300 jugadores cotejados, 75 por tier, sin diferencias. El caso sintético de nombre reutilizado confirma que la vista anterior puede sobrecontar. |
| Monitor | 9,05 s por actualización | 0,30 ms entre refrescos y 11,56 s para el refresco completo | Las dos consultas rápidas evitan barrer las participaciones. El refresco completo usa la definición oficial y bloquea el refresco de pantalla mientras dura. |

**Reserva, corte y export.** Un ID queda reservado dentro de la sesión desde
el despacho hasta el commit o el fallo. El filtro de cada página consulta la
reserva y `filter_unprocessed_match_ids()` bajo el mismo candado, evitando que
el escritor libere un ID entre ambos chequeos. Se conserva la deduplicación en
SQLite para cubrir otros productores. Al vencer el presupuesto o llegar una
señal de corte, el despachador cancela las tareas que siguen en la cola, recoge
las listas y concede hasta 5 s para recoger el resto de las que ya estaban en
ejecución. Los resultados fuera de ese plazo se pueden solicitar en otra
sesión. La espera acota la cosecha de descargas. Una consulta de historial
síncrona ya iniciada, el escritor, el resumen y el export pueden alargar el
cierre. Una petición HTTP en curso puede terminar en segundo plano y su
respuesta queda fuera de la ingestión de esa sesión.

El final de la cola del escritor marca que todas las respuestas admitidas ya
fueron entregadas. Se espera su vaciado real para completar el mapa y tomar
el resumen y la foto de export. Un `join` con timeout permitía sacar esa foto
con partidas del mismo ciclo todavía pendientes. Los cuatro CSV leen una sola
transacción de solo lectura y construyen el mapa TEMP una vez en lugar de
tres. Se usa el mismo SQL y formato de salida. Ante un commit externo entre
el primer CSV y el resto, los cuatro conservan la foto inicial. La comparación
contra `origin/gonchi` produjo bytes iguales para los cuatro archivos de una
base sintética de 32 partidas y 320 participaciones, que incluyó alias,
timeline faltante, posición vacía y duplicada, y la etiqueta de victoria
anómala. Hubo 159 cruces, 10 perfiles y 18 combinaciones campeón/línea. El
export sintético tomó 41,4 ms antes y 31,2 ms después, una sola observación.
Quedan pendientes el procesamiento por bloques y la publicación atómica de
los archivos. Se difiere `--sin-exportar` hasta resolver qué CSV incluye el
respaldo cuando la última foto se omitió.

El escritor devuelve el número de filas nuevas en `matches` y el número de
esas partidas con filas de timeline confirmadas. Consulta los IDs ya guardados
y escribe dentro de la misma transacción para que un commit concurrente no
infle el contador. El porcentaje de sesión es
$100\,N_{\text{nuevas con timeline}}/N_{\text{nuevas}}$ y vale cero si no
se agregó ninguna partida. `players_crawled` cuenta jugadores marcados como
visitados durante la sesión.

**LP por partida.** Una candidata debe ser SoloQ de la misma plataforma,
tener al jugador, durar más de 300 s y terminar en
$(t_{\text{antes}}-25\,\mathrm{min},\,t_{\text{después}}+1\,\mathrm{min}]$.
Se enlaza cuando hay una sola candidata conocida. Con dos o más, o con
detalles faltantes en la lista pedida para el evento, el evento queda
pendiente y conserva el reintento programado. Evaluar una partida por vez
podía convertir el orden de la lista en evidencia inexistente. La consulta
agrupa las candidatas locales antes de actualizar los eventos. Para diez
participantes pasó de 21 `SELECT` emitidos por Python a una lectura y un
`executemany`, en la misma transacción. Los compañeros se resuelven con las
candidatas disponibles en SQLite, lo que evita consultar diez historiales
por partida. Una alternativa que Riot todavía no publicó puede faltar en
ese caso. Las filas históricas ya enlazadas requieren una auditoría aparte.

**Consultas de ventana y matriz de draft.** La secuencia y el crawler
comparten el predicado que cuenta cada ID local de la ventana ausente de
`matches` y de `discarded_matches`. El estado usa `COUNT(*)` sin
materializar la lista. Para el draft, la columna $(c,l)$ vale
$\mathbf 1[\text{azul}_l=c]-\mathbf 1[\text{rojo}_l=c]$.
La codificación vectorizada conserva el orden lexicográfico de los pares,
el tipo `float64` y los ceros explícitos cuando ambos lados cancelan.
Brainstorm, curva, validación por bloques y mapa conservan el vocabulario
completo. Forma del día y el EDA conservan el vocabulario restringido al
ajuste y omiten pares desconocidos. En 270.000 partidas sintéticas, los
arreglos `indptr`, `indices` y `data` fueron idénticos. Con vocabulario
completo tomó 2,226 → 0,335 s. Con el vocabulario del ajuste tomó 2,036 → 0,338 s.

#### S.66. `eda_modelos_ganador` sobre el escalón 9, y eventos con 62.706 timelines (2026-11-18)

Pedido del usuario: que el notebook esté corrido entero con el mejor modelo previo y sincronizado con lo
que se bajó en la laptop. Hasta acá, las secciones 2, 4, 5 y 6 usaban el *híbrido* (logística con draft +
85 variables, ajustada con todo lo anterior a V y medida en la V de todas las ligas, 26.934 partidas:
0,6412 / 59,8 %); solo la sección 1 (tabla) y la 3 (números fijos) mencionaban el escalón 9.

**Qué cambia.** Todo se mide en V Master+ (16.428) y el modelo de las secciones 2, 4, 5 y 6 es el escalón
9 (§9.S.52), armado igual que `brainstorm_final.py`: logística $C = 0{,}05$ sobre
[draft | 85 variables × 5 líneas | señales del brainstorming], estandarizada con las 118.064 partidas de
Master+ anteriores al 16/09. Reproduce 0,6739 / 62,5 % exacto, también en la PC de escritorio y después de
regenerar `embeddings_partidas` con el arreglo de `fb_azul`/`ft_azul` de §9.S.63 (no lo mueve).

* *Sección 2 (bloques).* Las columnas extra del escalón 9 se reparten por bloque: márgenes jugador-línea,
  nivel histórico, W/L de la foto, winrate por línea, Kalman-Elo, tempo y economía → jugadores; margen
  jugador-campeón → jugador × campeón; contracruces y duelo esperado → campeones; el modelo por línea (oro
  a los 15 predicho con jugadores y campeones a la vez) queda como *mixto* y entra solo en el modelo
  completo. Campeones 0,559, jugadores 0,639, jugadores + jugador × campeón 0,662, todo 0,674. Los cruces
  suman +0,010 al draft solo; en el híbrido, atributos y estilo sumaban +0,0007.
* *Sección 4 (familias).* 21 familias (el draft es una más), cada una sacada y reentrenada. Champion
  Mastery sigue primera (−0,0068; en el híbrido −0,019, porque el margen jugador-campeón la cubre en
  parte), después pool, nivel histórico, draft, duelo esperado y W/L de la foto. Las familias de margen se
  tapan entre sí (≤ 0,001 cada una).
* *Sección 5.* Los grupos de tier se pasaron a < 1, 1-1,5 y ≥ 1,5: V Master+ no tiene lobbies con tier
  medio < 0,5 (el corte del híbrido dejaba un grupo vacío y rompía el `roc_auc_score`). Rinde parejo en
  los tres (0,673-0,679). Las curvas de LP y duración salen de
  `acierto_por_lp_duracion.py --modelo escalon9` (`acierto_lp_duracion_escalon9.csv`).
* *Sección 6.* Descomposición exacta del logit por las 21 familias (`assert` contra la predicción) y el
  logit fuera de pliegue con el mismo escalón 9 (5 pliegues sobre el ajuste). Conclusiones iguales:
  difíciles = familias que se cancelan (conflicto 0,95, 52 % de acierto), fáciles = 76 %; |logit| solo
  ordena igual que el meta-modelo (0,6076 contra 0,6078); "si duda, rojo" no suma.
* *Sección 3.* §3.1 y §3.3 se recalcularon con los 62.706 timelines que había al copiar `timelines.db`
  (toda V Master+ y 46.278 de Master+ del ajuste; §9.S.64). Oro equivalente: primer dragón ~1.280, tres
  dragones ~4.440, primera torre ~1.980, primera sangre ~630 con soporte. Eventos sobre la logística en
  vivo: +0,0055 / +0,0060 / +0,0070 de AUC a los 10 / 15 / 20, con IC > 0 y 3 de 3 tramos (el piloto daba
  +0,0058 / +0,0061 / +0,0076). La §3.3 ya no tiene números fijos: lee `eventos_aporte.csv`.

*Alternativa descartada.* Dejar el híbrido en las secciones descriptivas porque "las conclusiones no
cambian": cambian los números (≈ +0,03 de AUC) y algunas lecturas (Diamond ya no es el grupo difícil; los
cruces de campeones sí suman), y la sección 1 ya presenta el escalón 9 como el modelo de hoy.

#### S.67. Matchmaking más parejo reacomodando a los mismos 10 jugadores (2026-11-19)

**Pregunta (pedido del usuario).** Con el mismo grupo de 10 jugadores de una partida Master+, ¿se podría
armar una partida más pareja (p del modelo previo más cerca de 0,5) cambiando equipos y roles sin afectar
mucho los roles? Trabajo orquestado por Claude: Sol 6.2 (Codex) hizo la compuerta de calibración, Gemini 4
Hydrogen el perfil de roles y Gemini 4 Neon la búsqueda; revisión y verificación de números, Claude.
Todo en V Master+ (16.428 partidas), con el modelo ajustado solo con Master+ anteriores al 16/09.

> **Alcance (corrección del 2026-11-23, §9.S.77).** Es una **simulación bajo el modelo**: las configuraciones
> nuevas no se jugaron y su "qué tan pareja" es la p que les asigna la logística. **No restringe dúos ni
> premades** (la base no los registra): un reacomodo puede separar a dos jugadores que entraron juntos. La
> variante principal es la que **protege a cada jugador** ($\Delta d_i \le 0$ para los 10, 4,96 pp); el tope
> $D \le D_\text{real}$ protege solo el total y deja peor al 3,3 % de los jugadores.

**1. Compuerta: el modelo previo está calibrado** (`matchmaking_calibracion.py`, `REPORTE_calibracion.md`).
Antes de usar p como medida de "qué tan pareja" es una partida hay que verificar que una p de 0,60 signifique
un 60 % de victorias. Se miden el diagrama de confiabilidad por tramos (IC bootstrap), el ECE y la recta de
calibración logit(P(y=1)) = a + b·logit(p): b = 1 es calibrado, b < 1 sobreconfiado, b > 1 subconfiado.

| Variante | AUC / acierto | ECE (deciles) | a | b [IC 95 %] | favorito con p 57,5-62,5 % gana |
|---|---:|---:|---:|---:|---:|
| Escalón 9 corregido (con draft, §9.S.68) | 0,6740 / 62,6 % | 1,04 % | −0,011 | 0,952 [0,902; 1,003] | 59,4 % (n = 3.269) |
| Sin draft (320 columnas) | 0,6369 / 59,6 % | 0,75 % | +0,004 | 1,028 [0,961; 1,095] | 60,0 % (n = 3.918) |

Los tres bloques temporales de V tienen b con IC que incluye 1. En Diamond-V (A4) los dos modelos quedan
sobreconfiados (b 0,83-0,84): la calibración vale para Master+, que es la población del estudio. Platt
ajustado fuera de V (10.920 predicciones fuera de muestra anteriores a V) empeora el ECE en V (0,75 → 0,82 %
sin draft), así que **se usa la p cruda**. Claude reprodujo AUC, a, b y el tramo de 60 % desde los parquet.

*Hallazgo y corrección.* La primera réplica del escalón 9 estimaba escalas por línea, imputaciones y el
prior del nivel histórico con agregados que alcanzaban V. La variante sin draft ya congelaba sus
agregados antes del cuantil 90 % temporal del ajuste. El §9.S.68 mide la corrección del escalón 9
sobre las mismas partidas. En la prueba de Platt con draft, los agregados incluyen su bloque
interno de calibración. La decisión para matchmaking usa la variante sin draft, con un corte previo a ese bloque.

**2. Por qué se puntúa con la variante sin draft.** El emparejamiento ocurre antes del draft: si un jugador
pasa de mid a top no puede llevarse su campeón de mid. Se sacan las 851 columnas del draft y todo lo que
depende del campeón elegido (experiencia y winrate con el campeón, tags y perfil del campeón, margen
jugador×campeón, contracruces y sinergias, exceso de WR con el campeón en la línea, tempo del campeón, HGB
por línea): quedan 320 columnas de jugador y jugador×línea (lista en `calib_columnas_sin_draft.json`). Cuesta
0,037 de AUC, pero es la única versión cuyo contrafáctico está definido. *Descartada:* puntuar con el escalón 9
llevando cada jugador su campeón; se reporta solo como chequeo para intercambios en la misma línea.

**3. Perfil de roles y desvío** (`matchmaking_perfil_roles.py`, `perfil_roles.parquet`). Para cada
participación, la distribución causal de roles del jugador (solo partidas anteriores, unificando cuentas con
`player_id_map`), suavizada hacia la distribución global q con α = 5:
π_i(r) = (n_r + α·q_r) / (n_prev + α). El desvío del jugador i en la posición r es
**d_i(r) = max_s π_i(s) − π_i(r) ∈ [0, 1]**: 0 si juega su rol principal, y bajo naturalmente para quien
juega de todo (distribución plana), sin tratarlo aparte. El costo de una partida es D = Σ_i d_i. Posición =
`individual_position` (la base no guarda `team_position`, que sería la asignada por la cola). En 2,73 M
participaciones: 65,8 % juega su rol principal, d medio 0,115 (p90 0,48); con n_prev ≥ 20, autofill
(π < 0,15) 14,7 %. *Descartadas:* 1 − π_i(r) (castiga a los flexibles aunque jueguen su rol más común) y
una divergencia KL entre distribuciones (no se lee como "cuánto juega esa línea").

**4. Contrafáctico exacto** (`matchmaking_swaps.py`). El logit se factoriza como
z = B0 + Σ_l C[jugador azul en l, l] − Σ_l C[jugador rojo en l, l] + términos no lineales, donde C[i, l] es el
aporte lineal del jugador i jugando la línea l. Las columnas que dependen solo del jugador se mueven con él;
las de jugador×línea (`share_lane`, `offrole`, margen previo por línea `ML`, WR por línea `ROL_WR`) se
recalculan causalmente para la línea nueva, y los agregados por equipo no lineales (mínimos, máximos, desvío,
media de los dos mayores, brecha absoluta del lobby) se recalculan exactos. Ninguna de las 320 columnas se
aproxima. Validación: en la asignación real reproduce la p del modelo con |Δp| máx 2,7·10⁻⁷.

Movimientos: (a) cambiar de equipo a los dos jugadores de una línea (2⁵ = 32 orientaciones, no simétricas
por el efecto del lado; no cambia ningún rol); (b) intercambiar roles entre jugadores que juegan ambos roles,
con π_i(rol nuevo) ≥ τ, τ ∈ {0,20; 0,35}; (a+b) todas las combinaciones, por enumeración exhaustiva (media 126
configuraciones por partida con τ = 0,20, máx 4.896).

**5. Resultados en V Master+** (p cruda, modelo sin draft):

| Escenario | \|p−0,5\| media | en [0,45; 0,55] | D medio | % jugadores fuera de su rol principal |
|---|---:|---:|---:|---:|
| Riot (real) | 9,74 pp | 30,6 % | 1,252 | 31,8 % |
| (a) misma línea | 1,17 pp | 95,8 % | 1,252 | 31,8 % |
| (b) un cambio de rol, τ 0,20, D ≤ D_real | 8,24 pp | 40,4 % | 1,136 | 29,8 % |
| (a+b) τ 0,20, D ≤ D_real | 0,73 pp | 97,7 % | 1,073 | 29,0 % |
| (a+b) τ 0,20, primero mínimo D | 1,14 pp | 96,0 % | 0,900 | 25,8 % |
| **(a+b) τ 0,20, nadie empeora (Δd_i ≤ 0)** | 0,96 pp | 96,8 % | 1,131 | 28,7 % |

Casi todo el equilibrio sale de (a), que no mueve a nadie de rol; los cambios de rol suman poco equilibrio
pero permiten **bajar** el desvío respecto de Riot (devolver a su rol a dos jugadores en autofill que
juegan el rol del otro). Las partidas más desparejas tienen apenas más autofill (r = 0,06 entre |p−0,5| y
D), pero la asimetría de autofill entre equipos explica buena parte de la dirección:
corr(p_azul − 0,5, D_azul − D_rojo) = −0,38. Chequeo con draft, solo (a), con la caché de §9.S.68: 13,04 → 1,50 pp.

**6. Sesgo de optimización (maldición del ganador).** Elegir el mínimo de |p − 0,5| entre decenas de
configuraciones con un modelo estimado empuja hacia 0,5 aunque la partida no sea más pareja. Se ajustan dos
modelos con mitades disjuntas del entrenamiento (intercaladas por fecha, 59.032 partidas cada una); uno elige
la configuración y el otro la evalúa, y al revés:

| Escenario | real | elegido | evaluado por el otro modelo | mejora retenida | indep. en [0,45; 0,55] |
|---|---:|---:|---:|---:|---:|
| (a) misma línea | 10,03 pp | 1,06 pp | **5,06 pp** | 55 % | 56,8 % |
| (a+b) τ 0,20, D ≤ D_real | 10,03 pp | 0,66 pp | 4,87 pp | 55 % | 58,6 % |
| **(a+b) τ 0,20, nadie empeora (Δd_i ≤ 0)** | 10,03 pp | 0,87 pp | **4,96 pp** | 55 % | 57,7 % |

La lectura honesta es la columna independiente: **el desbalance que mide el modelo se puede reducir a la
mitad** (de ~10 a ~5 pp) sin que ningún jugador quede más lejos de su rol (4,96 pp con Δd_i ≤ 0), y las partidas
en [0,45; 0,55] pasan de 31 % a ~58 %. *Corrección (2026-11-23, §9.S.77):* la versión anterior decía "sin empeorar
roles" de la fila D ≤ D_real, que topa solo el total: ahí el 3,3 % de los jugadores (en el 24,6 % de las partidas)
queda más lejos de su rol, con Δd medio +0,21 entre ellos. Ese número
sigue siendo optimista: las dos mitades comparten las variables (estimaciones ruidosas de la habilidad de cada
jugador) y solo se descuenta el ruido de los coeficientes, no el de las variables. A favor de que la señal
sea real: el modelo está calibrado justamente en los extremos (p > 0,65 → 71 % de victorias), así que los
desbalances grandes que ve en las partidas reales no son ruido. En contra: las configuraciones nuevas no
se jugaron, así que la mejora no se puede verificar con resultados.

*Alternativas descartadas.* Reportar el |p − 0,5| elegido (0,7 pp) como resultado: es casi todo sesgo de
selección. Búsqueda greedy o beam: la exhaustiva entra en ~150 s para V entera.

*Reproducir:* `python scripts/analysis/matchmaking_calibracion.py` (~100 s),
`python scripts/analysis/matchmaking_perfil_roles.py` (~45 s), `python -u scripts/analysis/matchmaking_swaps.py`
(~150 s; `--muestra 500` para una prueba). Salidas en `data/analysis/matchmaking/`.

**7. Auditoría: ¿qué sabe el reacomodo al elegir? (pedido del usuario, 2026-11-19).** La única información
admitida es quiénes son los 10 jugadores y sus historiales. Revisión de las 320 columnas y de la búsqueda:
* *Historial del jugador* (`Xj`: márgenes acumulados y de las últimas 20, winrate, volumen, `share_lane`,
  `offrole`, horas desde la partida anterior, ritmo, concentración del pool): todo con `_previo` (cumsum menos la
  fila actual) o leído antes de actualizar; la media campeón×línea del residuo también es previa.
* *Jugador × línea* (`ML`, `ROL_WR`, `share_lane`, `offrole`): se recalculan para cada línea candidata con
  `_query_cumsum_por_linea`, que suma solo filas con `row < inicio de la partida`.
* *Kalman* (`KALMAN_OUT`, `KALMAN_MRG`, `TEMPO_OBJ`, `ECON_MID`): estados leídos antes de actualizar con la partida.
* *Foto 1* (`FOTO1_DIN`, `MMR_HIST`): ladder del 04/09 22:51 UTC, anterior a V, más las partidas posteriores ya
  jugadas; `MMR_HIST` usa el LP de la foto de los rivales de partidas previas.
* *Perfil de roles π*: partidas estrictamente anteriores. *Escalas e imputación*: congeladas antes de V.
* *Resultado de la partida, duración, campeones, stats o timeline de la partida*: no entran.
Lo único de la partida misma que se usa es la **hora** (para cortar el historial; se conoce al entrar a la cola) y
la **asignación de Riot** (`individual_position` y lado), que es la *salida* del emparejamiento contra la que se
compara: sirve de configuración de referencia, de tope D ≤ D_real y para admitir que cada jugador conserve su rol
real aunque π < τ. No entra como variable: cada configuración se puntúa solo con jugadores, líneas candidatas e
historiales. Salvedad: `individual_position` se deduce de lo jugado, así que es una aproximación de la asignación.

**8. En el EDA (pedido del usuario, 2026-11-19).** La sección 7 de `eda_modelos_ganador.ipynb`
(`build_eda_modelos_nb.py`) lo muestra leyendo esos parquet, sin reentrenar: diagrama de confiabilidad de
las dos variantes (7.1), distribución del desvío d (7.2), tabla de reacomodos (7.3), elegido contra evaluado
por el otro modelo con el histograma de |p − 0,5| (7.4), desvío por jugador y Δd de quienes cambian de rol
(7.5), un ejemplo anonimizado (7.7) y, con resultados reales, la asimetría de autofill (7.6): por tramos de
D_azul − D_rojo el win rate observado acompaña a la probabilidad, y en el extremo el modelo se queda corto
(D_azul − D_rojo ≤ −1: p 57,8 %, gana el azul 63,4 %, 574 partidas). Es la única pieza del estudio que se
verifica con resultados jugados. Las respuestas pasan a la sección 8.


---

#### S.68. Aislamiento de V en el preprocesamiento del escalón 9 (2026-11-19)

**Problema y decisión.** Un historial que excluye la partida actual puede seguir dependiendo de
estadísticas del futuro si la transformación de cada observación se estima sobre toda la serie.
Se usa como referencia $F=\{i:t_i<t_V\}$, con $t_V$ el corte de V (16/09 UTC)
y `F = ~val0`. Así quedan fuera tanto V Master+ como Diamond-V. Una máscara que excluye solo
V Master+ deja Diamond-V en la estimación. La regresión final entrenada con $F$ usa solo Master+;
los agregados de preprocesamiento preservan su definición sobre el conjunto completo anterior a V.

Para cada estadística $x$ y línea $\ell$, se estiman $\hat\mu_{\ell,x}$ y $\hat\sigma_{\ell,x}$
como media y desvío de $x_i$ para $i\in F$ con $\ell_i=\ell$. En el residuo de cada fila,
$r_i=(x_i-\hat m_{c_i,\ell_i,<i})/\hat\sigma_{\ell_i,x}$, la media del campeón–línea se encoge
hacia $\hat\mu_{\ell_i,x}$ con los registros anteriores a $i$. El historial general, jugador–campeón
y jugador–línea agrega solo residuos previos. Para imputar tiempo desde la última partida se usa
$\operatorname{mediana}_{i\in F}(\log(1+\Delta h_i))$ entre tiempos observados y para atributos
faltantes del campeón se usa la media de $F$. Para el nivel histórico se aplica
$\mu_0=\operatorname{media}_{i\in F}(z_i)$ entre participantes con LP en la foto 1. El HGB
por línea usa además atributos escalados sobre el catálogo de campeones, cuya ponderación es
independiente del número de partidas jugadas por cada campeón en V.

La foto 1, del 04/09/2026 22:51:22 UTC, aporta LP y W/L del padrón y ancla el corte de V.
Se verifica su fecha en `players` antes de reconstruir las variables que la usan, porque una
sincronización del padrón pisa LP y W/L. El modelo conserva Kalman-Elo causal, nivel histórico por
emparejamiento y W/L actualizado desde esa foto. El mapa mecánico de LP es otro bloque: reconstruye
el saldo de LP partida a partida (§9.S.46) y se activa con `con_mapa=True`. Se mantiene la variante
sin ese bloque, que obtuvo mejor log loss en V y un ΔAUC al agregarlo de −0,00030
[−0,00136; +0,00074].

**Reconstrucción.** En orden, se ejecutan
`embeddings_partidas.py --rehacer --solo-variables`, `brainstorm_05_jugador.py --rehacer --solo-cache`,
`brainstorm_07_mmr_historico.py --rehacer --solo-cache`,
`brainstorm_g_09_hgb_lineas_causal.py --rehacer --solo-cache` y `brainstorm_final.py`,
todos bajo `scripts/analysis/`. El HGB depende de las tres primeras cachés, `gemini_mpr.npz`
y el catálogo de atributos. La carga compartida rechaza un HGB más antiguo que sus insumos
presentes. La hora de modificación es una comprobación local y puede cambiar al copiar archivos.
La reconstrucción trabaja con los CSV del train congelado y conserva el test cerrado.

**Validación pareada.** Cada diferencia de AUC usa las mismas partidas antes y después.
Se conservaron las predicciones anteriores en `data/analysis/brainstorm/_pre_fix_esc9_ref.npz`.
Los extremos del IC 95 % son los percentiles 2,5 y 97,5 de 2.000 remuestras pareadas con reemplazo
dentro de cada conjunto.

| Conjunto | Partidas | AUC anterior | AUC corregida | ΔAUC [IC 95 %] |
| :--- | ---: | ---: | ---: | :--- |
| A2 Master+ | 4.749 | 0,67390 | 0,67387 | −0,00002 [−0,00047; +0,00045] |
| A3 Master+ | 4.607 | 0,66653 | 0,66687 | +0,00034 [−0,00012; +0,00080] |
| A4 Master+ | 4.504 | 0,65119 | 0,65080 | −0,00039 [−0,00088; +0,00011] |
| V Master+ | 16.428 | 0,67390 | 0,67403 | +0,00013 [−0,00013; +0,00038] |
| Diamond-V | 10.506 | 0,64365 | 0,64325 | −0,00040 [−0,00072; −0,00008] |

Con la corrección, V Master+ obtiene 62,59 % de acierto frente a 62,52 % (Δ +0,07 pp
[−0,07; +0,21]), con log loss 0,64431 → 0,64425. La diferencia absoluta de probabilidad por
partida promedia 0,34 pp y llega a 1,91 pp. En el tramo con favorito alrededor de 60 %, la p
media es 59,90 % y gana 59,41 % [57,72; 61,12], sobre 3.269 partidas. En Diamond-V el AUC baja
0,00040 [−0,00072; −0,00008]. La decisión de corregir se apoya en el aislamiento temporal.

Los resultados por A2–A4 sirven para sensibilidad al cambio. Esos cortes ocurren antes de
$t_V$ y comparten los agregados de $F$, que contiene fechas posteriores a cada corte interno.
Por ello la evaluación de esos bloques no certifica que todo el preprocesamiento se haya
ajustado dentro de cada prefijo. Los resultados de V sí tienen la frontera aplicada a los
agregados de la corrección. Hay una asimetría histórica adicional: el nivel por emparejamiento
usa LP de la foto 1 incluso para reconstruir el nivel de lobbies anteriores a la foto. La W/L
dinámica se activa a partir de la fecha de la foto. La foto ya existía al comienzo de V, pero
queda pendiente medir el efecto de enmascarar ese LP en las filas de ajuste anteriores al 04/09.

#### S.69. Reproducción del escalón 9 en otra máquina y poda con L1 (2026-11-15)

**Pedido del usuario.** Verificar el número del compañero y medir si el modelo final se puede achicar.

**Reproducción.** En la notebook del usuario (rama `gonza`, cachés regeneradas desde cero con el mismo
`data/splits/train`), `brainstorm_final.py` da **V Master+ AUC 0,6740 / acierto 62,6 %**, log loss 0,6443,
ECE 0,0065, cobertura 5 % → 80,4 %, 10 % → 79,0 %, 20 % → 76,3 % (el compañero reporta 0,6739 / 62,5 %). El
resultado no depende de la máquina. *Orden real de las cachés* (la nota "se arman solas si faltan" no era
cierta para el mapa): `embeddings_partidas.py` → `ganador_mapa_lp.py` (calibra el paso con
`data/lp_vivo/lp_vivo.db` y escribe `mapa_lp_causal_v.pkl`; sin él `brainstorm_base._armar` falla con
`params=None`) → `brainstorm_05` → `07` → `g_07` → `g_10` → `g_11` → `g_09` → `g_23`. Unos 90 min en CPU.
`brainstorm_g_11` guarda su caché y después falla al imprimir una comparación (`KeyError` de la referencia
del escalón 3); la caché es válida.

**Por qué no destilar.** El modelo ya es una logística de 1.504 coeficientes. Lo costoso es calcular las
variables (estados por jugador partida a partida y un HGB por línea), y un alumno necesita las mismas
entradas. Además §9.S.39 ya mostró que destilar hacia una logística no cambia nada. Lo único que achica es
quitar columnas.

**Poda con L1** (`historial/ideas_posteriores/brainstorm_43_poda_l1.py`, ~9 min). Misma matriz que el modelo final.
Para cada $C_1$ se ajusta la logística con penalización $\ell_1$,

$$\hat\beta(C_1) = \arg\min_\beta \; \lVert\beta\rVert_1 + C_1 \sum_i \log\!\left(1 + e^{-\tilde y_i\, x_i^\top \beta}\right),$$

que lleva coeficientes exactamente a cero. Se miden dos variantes: la L1 tal cual y el *reajuste* (logística
L2 con $C = 0{,}05$ solo sobre las columnas vivas, para quitar el encogimiento que la L1 les impone a las que
quedan). Δ contra la L2 completa, IC por bootstrap pareado en V (B = 500), bloques A2-A4 hacia adelante.

| $C_1$ | Columnas | L1: AUC V / acierto | L1 → reajuste L2: AUC V / acierto | Δ V del reajuste [IC 95 %] | Δ bloques |
| :---: | :---: | :---: | :---: | :---: | :---: |
| — (L2 completa) | 1.504 | — | 0,6740 / 62,6 % | — | — |
| 0,05 | 575 | 0,6732 / 62,4 % | 0,6728 / 62,3 % | −0,0012 [−0,0022; −0,0001] | +0,0002 |
| 0,02 | 424 | 0,6725 / 62,3 % | 0,6720 / 62,4 % | −0,0019 [−0,0035; −0,0003] | +0,0008 |
| 0,008 | 301 | 0,6709 / 62,3 % | 0,6715 / 62,1 % | −0,0025 [−0,0046; −0,0003] | −0,0001 |
| 0,004 | 222 | 0,6681 / 62,2 % | 0,6705 / 62,4 % | −0,0035 [−0,0063; −0,0010] | −0,0027 |
| 0,002 | 152 | 0,6623 / 61,8 % | 0,6680 / 62,1 % | −0,0060 [−0,0088; −0,0032] | −0,0042 |
| 0,001 | 84 | 0,6483 / 61,1 % | 0,6592 / 61,6 % | −0,0148 [−0,0191; −0,0105] | −0,0142 |
| 0,0005 | 24 | 0,6284 / 59,6 % | 0,6371 / 60,1 % | −0,0368 [−0,0425; −0,0313] | −0,0335 |

**Lectura.**

1. **No hay poda gratis, pero es barata hasta ~300-400 columnas.** Con 424 columnas (28 %) se pierde 0,0019
   de AUC y 0,2 puntos de acierto; con 301 (20 %), 0,0025. El IC de V excluye el cero en todos los casos,
   así que la L2 completa sigue siendo el modelo a reportar. En los bloques, 424 y 575 columnas no empeoran.
2. **Lo primero que se va es el one-hot del draft.** De sus 851 columnas quedan 0 hasta $C_1 = 0{,}004$, 3 con
   301 columnas y 31 con 424. El draft sobrevive por sus resúmenes (matchups, margen esperado por par,
   atributos del campeón), no por la identidad del campeón. Coincide con la poda del escalón 4 (sin el
   one-hot: −0,0028).
3. **El núcleo son ~24 columnas que dan 0,637 / 60,1 %:** experiencia con el campeón (`logn_champ` en 3
   líneas), participación en kills y reparto de daño acumulados, margen por campeón/línea y nivel histórico (4),
   Kalman (7), winrate por rol (2), una del HGB y una del margen por par.
4. **El reajuste ayuda solo con poda fuerte** (+0,006 a +0,011 con menos de 152 columnas). Con 424 o más, la
   L1 tal cual rinde igual o mejor: ahí el encogimiento ya no sesga.

**Alternativas descartadas.** *Destilación* (arriba). *Cortar bloques enteros* es la otra forma de achicar y
está en la escalera de §9.S.52 (escalón 3b: 0,6663 / 62,0 % sin Kalman, HGB ni matchups): pierde más AUC
(−0,0076) que la L1 con 301 columnas (−0,0025), pero elimina cachés completas, que es lo que simplifica el
cálculo; la L1 saca columnas sueltas y casi todas las cachés siguen haciendo falta (con 301 columnas quedan
vivas columnas de los 8 bloques). *Elastic net y selección por estabilidad:* no se probaron; el resultado
(poda barata pero no gratis) no cambia la decisión.

**Decisión.** El modelo final sigue siendo la L2 completa. Si hace falta una versión chica para la demo, la
recomendación es cortar por bloques (escalón 3b) y no por columnas.

#### S.70. Qué errores del emparejamiento explota el modelo (2026-11-15)

**Pregunta del usuario.** Riot busca que cada partida sea pareja. Si el modelo acierta 62,6 %, ¿qué errores
del emparejamiento está explotando, y queda alguno sin explotar? Script:
`historial/ideas_posteriores/brainstorm_44_errores_emparejamiento.py` (~3 min; `data/analysis/brainstorm/errores_*.csv`).
Todo en V Master+ (16.428 partidas) con el modelo final.

**Punto de partida.** El emparejamiento iguala bien lo que mira: con el LP real de los 10 jugadores el AUC es
0,505 (§9.S.52, `lp_vivo.db`), y acá el signo de la diferencia de LP medio entre equipos no predice nada (azul
gana 47,8 % tanto si tiene más LP como si tiene menos). Lo que el modelo predice es lo que el emparejamiento
no mira.

**A. Descomposición por tipo de error.** El logit de la logística es aditivo,
$\operatorname{logit}(p) = \beta_0 + \sum_g c_g$ con $c_g = x_g^\top \beta_g$, así que se parte por grupo de
columnas sin reajustar. El peso de un grupo es $\operatorname{cov}(c_g, \sum_h c_h) / \operatorname{var}(\sum_h c_h)$
(suma 1). "AUC sin él" resta $c_g$ del logit (sin reajuste: es una cota de lo que se pierde).

| Tipo de error (lo que el emparejamiento no mira) | Columnas | Peso en el logit | AUC solo | AUC sin él | Domina (todas / 20 % seguro) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Nivel real del jugador ≠ MMR (rendimiento y winrate acumulados, nivel histórico, W/L de la foto, Kalman) | 186 | 29,0 % | 0,5957 | 0,6468 | 33,8 % / 40,9 % |
| Experiencia con el campeón (partidas con él, estreno, pool, margen por campeón) | 115 | 29,2 % | 0,5980 | 0,6508 | 31,4 % / 39,5 % |
| Draft (one-hot, atributos del campeón, matchups, margen esperado por par) | 1.073 | 12,6 % | 0,5494 | 0,6586 | 17,9 % / 13,6 % |
| Modelo por línea (oro @15 predicho; mezcla nivel, campeón y línea) | 6 | 13,3 % | 0,5985 | 0,6684 | 4,3 % / 2,3 % |
| Estilo (tempo, objetivos, economía 15→20) | 14 | 7,6 % | 0,5680 | 0,6699 | 2,8 % / 0,9 % |
| Línea / autofill (fuera de rol, margen por línea, winrate por rol) | 95 | 5,2 % | 0,5259 | 0,6668 | 7,7 % / 2,6 % |
| Actividad (partidas totales, horas desde la anterior, ritmo) | 15 | 3,1 % | 0,5292 | 0,6688 | 2,0 % / 0,1 % |

*Lectura.* Casi el 60 % de la señal son dos errores del mismo tamaño: el MMR va atrasado respecto del nivel
real, y el emparejamiento ignora con qué campeón juega cada uno. El draft pesa 13 % y el autofill 5 %: en
Master+ casi nadie juega fuera de rol. **Las partidas seguras no son de un tipo distinto:** en el 20 % más
seguro (|logit| medio 1,23 contra 0,56; acierto 76,3 %) los pesos son los mismos (29 / 29 / 13 %); lo que
cambia es que más grupos empujan para el mismo lado (5,3 de 7 contra 4,4). Un lobby desparejo es una
acumulación de errores chicos, no un error grande. No aparece un tipo de error subexplotado.

**B. Cola (poca gente buscando partida).** Medida con las partidas del dataset que empiezan en ±30 min.

| Partidas simultáneas | Mediana | Dispersión de LP en el lobby | \|logit\| medio | AUC | Acierto |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Q1 (menos gente) | 66 | 178 | 0,558 | 0,6793 | 63,3 % |
| Q2 | 117 | 164 | 0,553 | 0,6878 | 63,6 % |
| Q3 | 146 | 161 | 0,569 | 0,6798 | 63,2 % |
| Q4 | 164 | 158 | 0,551 | 0,6609 | 61,6 % |
| Q5 (más gente) | 182 | 158 | 0,556 | 0,6618 | 61,1 % |

Con poca gente el lobby se arma con LP más disperso (Spearman −0,12) y la partida es más predecible: AUC
0,6823 en Q1-Q3 contra 0,6611 en Q4-Q5, **diferencia +0,021 [+0,004; +0,037]** (bootstrap, B = 1000). Por
franja (ART): 00-06 h 0,690, 12-18 h 0,680, 06-12 h y 18-24 h 0,664. **Pero el modelo no lo sabe ni lo puede
usar:** su confianza media es la misma en los cinco quintiles (Spearman con |logit| −0,005), y agregar la
interacción logit × cola (ajustada en la primera mitad de V, medida en la segunda) da 0,6634 contra 0,6633. Lo
que cambia es la pendiente de calibración: 1,00 [0,94; 1,06] con poca gente y 0,88 [0,80; 0,95] con mucha. Con
la cola llena el modelo es sobreconfiado. Sirve para elegir qué partidas mostrar en la demo (madrugada y
tarde), no para subir el AUC. Coincide con la interacción hora × habilidad descartada en §9.S.52.

**C. Lado.** Azul gana 47,8 % en V (48,0 % en el ajuste Master+). El equipo rojo tiene más LP medio en el
68,4 % de las partidas (media −36,5 LP, EE 0,7) y más nivel histórico en el 65,4 %. Es consistente con que el
emparejamiento mande al equipo de más MMR al lado rojo para compensar la ventaja del azul, y con que esa
compensación se pase: gana más el rojo. El modelo ya lo usa (intercepto −0,091, elige rojo el 55 % de las
veces y acierta 63,4 % ahí contra 61,5 % eligiendo azul). En el 20 % de lobbies de tier más alto azul gana
46,4 % contra 48,2 % en el resto (diferencia 1,8 puntos, EE ≈ 1,0: no significativa).

**D. "Cola de perdedores".** Hipótesis popular: al que viene perdiendo le tocan peores compañeros. Para cada
jugador-partida (Master+, desde la foto 1; 349.339 filas con 3 partidas previas) se compara la media de sus 4
compañeros menos la de los 5 rivales según la racha previa.

| Racha previa | Filas | Gana la siguiente | Δ nivel histórico (z) | Δ LP (z) | Δ winrate acumulado (z) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Perdió las 3 anteriores | 38.861 | 52,1 % | −0,11 (−0,3) | −0,46 (−0,7) | 0,0000 (0,0) |
| Ganó las 3 anteriores | 48.494 | 52,0 % | −0,60 (−1,8) | −0,56 (−0,9) | −0,0001 (−0,6) |
| Mezcla | 261.984 | 51,9 % | ref. | ref. | ref. |

(Δ contra "mezcla"; desvío del nivel 68 y del LP 120.) **No existe:** ninguna diferencia llega a |z| = 2, las
magnitudes son menos del 1 % de un desvío, y el que perdió 3 seguidas gana la siguiente igual que el resto.
Restringiendo a rachas dentro de las últimas 6 horas, lo mismo (|z| < 1,4). El emparejamiento no castiga ni
compensa las rachas.

**Conclusión.** El emparejamiento falla por construcción en lo que no mira (nivel real atrasado y campeón),
y el modelo ya explota esos dos errores en partes iguales. Las tres hipótesis de un "error lógico" adicional
dan: la cola es real pero no explotable para el AUC, el lado ya está en el modelo, y la cola de perdedores no
existe. Subir más requiere información nueva sobre el MMR (LP partida por partida, `seguimiento_lp.py`), no
otra lectura de las mismas variables.

*Alternativas descartadas.* Medir el aporte de cada grupo reajustando sin él (más limpio, pero ya está en la
escalera y las podas de §9.S.52); medir la cola con partidas de todo el servidor (no las tenemos: el proxy es
el propio dataset, que cubre todo Master+ en la ventana).

---

#### S.71. Medias por jugador y línea, mejoras en vivo, relación entre variables, ítems y qué trae la respuesta cruda de Riot (2026-11-16)

**Pedidos del usuario.** (1) Sumar al modelo previo una media por jugador de su estado a los 10/15/20 minutos en partidas
anteriores, por línea. (2) Ver qué se puede mejorar de los modelos en vivo. (3) PCA o un clustering que no sea K-Means para
buscar relación entre variables, arrancando con todas y cortando. (4) Bajar la información de ítems: build general, outliers y
cómo varía el winrate, con suavizado. (5) Si el techo son los datos, qué datos faltan; verificar si el crawler guarda los bans.
Todo sobre el escalón 9 (§9.S.52) con la partición de §9.S.51; en esta máquina el escalón 9 da V Master+ 0,6740 / 62,6 %.

**1. Media por (jugador, línea) a 10/15/20** (`historial/ideas_posteriores/brainstorm_45_medias_jugador_linea.py`). Para cada jugador y
línea, con sus $n$ partidas anteriores en esa línea (la actual no entra), la media encogida hacia 0 de la diferencia contra el
rival de línea, $\hat m = \sum_i d_i / (n + K)$ con $K = 5$, para oro, XP, CS, daño, nivel, kills, muertes y asistencias a los
10, 15 y 20 minutos (24 números); la media de oro, XP y CS propios centrada en la media de la línea (9, con la media de la
línea calculada antes del 16/09); y $\log(1 + n)$. El valor a 15 (20) solo cuenta si la partida duró más de 15 (20) minutos.
Al modelo entra azul − rojo por línea.

| Variante sobre el escalón 9 | V AUC | Δ V [IC 95 %] | Δ bloques [IC 95 %] |
| :--- | :---: | :---: | :---: |
| A. oro/XP/CS contra el rival por línea (45 columnas) | 0,6741 | +0,0001 [−0,0005; +0,0008] | −0,0003 [−0,0010; +0,0004] |
| B. todo por línea (170) | 0,6736 | −0,0004 [−0,0015; +0,0008] | +0,0003 [−0,0011; +0,0016] |
| C. todo, media del equipo (34) | 0,6739 | −0,0001 [−0,0008; +0,0006] | +0,0001 [−0,0006; +0,0008] |

**No suma.** La información ya estaba: las 85 variables por jugador incluyen la diferencia contra el rival de oro @10/@15, XP
@15 y CS @15 (acumulada y de las últimas 20), el margen por (jugador, línea) del escalón 2 resume 15 estadísticas, y el Kalman
del escalón 6 usa la probabilidad @15. Gemini había probado el timeline @10/@20 sin separar por línea con el mismo resultado
(§9.S.52). *Alternativa descartada:* ventana de las últimas $N$ en vez de la media acumulada: la curva de §9.S.48 y el oráculo
de §9.S.38 ya muestran que más o distinto historial del mismo tipo no agrega.

**2. Mejoras de la segunda etapa en vivo** (`historial/en_vivo/brainstorm_46_en_vivo_mejoras.py`). Punto de partida: el modelo
por líneas de §9.S.52 (logit previo, Δoro y ΔXP de cada línea, Δkills del equipo y lo que el modelo por línea esperaba antes de
jugar: oro @15, y XP @15 u oro y XP @10 de `brainstorm_42_xp_lineas.py`). Ajuste con las predicciones hacia adelante de A2-A4
Master+, medición en V Master+, y una réplica que no mira V (ajuste con A2-A3, medición en A4 Master+).

| Momento | Por equipo (§9.S.52) | Por líneas + esperado (vigente) | Δ [IC 95 %] | Acierto en el 5 / 10 / 20 / 50 % más seguro |
| :--- | :---: | :---: | :---: | :---: |
| Minuto 10 | 0,7769 / 70,4 % | 0,7842 / 71,0 % | +0,0073 [+0,0052; +0,0094] | 96,2 / 94,4 / 90,0 / 82,1 % |
| Minuto 15 | 0,8286 / 74,9 % | 0,8371 / 75,7 % | +0,0085 [+0,0067; +0,0105] | 99,1 / 97,5 / 95,1 / 87,8 % |
| **Minuto 20 (nuevo)** | 0,8611 / 77,6 % | **0,8695 / 78,6 %** | **+0,0084 [+0,0067; +0,0104]** | 99,5 / 98,7 / 97,4 / 91,3 % |

(La cobertura es la del vigente ajustado también con Diamond.) Lo que suma es llevar la versión por líneas al minuto 20, que no
existía (`brainstorm_40` solo cubría 10 y 15); en A4 se replica (+0,0090 [+0,0057; +0,0125]). Agregados sobre el vigente, Δ en V
a los 10 / 15 / 20 minutos:

| Agregado | Min 10 | Min 15 | Min 20 |
| :--- | :---: | :---: | :---: |
| CS, daño, nivel, kills, muertes y asistencias por línea | −0,0004 | −0,0003 | +0,0001 |
| + trayectoria (oro y XP por línea de los minutos anteriores) | — | −0,0005 | −0,0003 |
| + medias del jugador en la línea (punto 1) | −0,0006 | −0,0011 | −0,0007 |
| + forma (logit × Δoro, mayor y menor ventaja de línea, desvío entre líneas, raíz del Δoro) | −0,0010 | −0,0014 | −0,0010 |
| **Vigente ajustado también con las Diamond de A2-A4** (≈ 30.000 partidas en vez de 13.700) | +0,0005 [+0,0001; +0,0010] | +0,0006 [+0,0000; +0,0010] | +0,0005 [+0,0002; +0,0009] |
| HGB espejado solo | −0,0055 | −0,0049 | −0,0049 |
| Mezcla 0,7 logística + 0,3 HGB | +0,0003 | +0,0002 | +0,0004 |

Las variables nuevas no suman: ninguna tiene el IC por encima de 0, ni en V ni en A4. El oro y la XP por línea ya resumen el
estado; el resto es redundante o ruido con 13.700 partidas de ajuste, y la relación es lineal en el logit (el HGB pierde 0,005).
Ajustar la segunda etapa también con las Diamond de A2-A4 da una mejora chica y del mismo signo en los tres minutos (+0,0005 en
V, con el IC apenas sobre 0; en A4 +0,0010, +0,0002 y +0,0003, con el IC tocando 0): no cuesta nada y se adopta para la demo,
sin contarlo como mejora demostrada. **El techo en vivo es de datos:**
la base guarda fotos de oro, XP, CS y KDA a 10/15/20, pero no objetivos por minuto (dragones, torres, heraldo) ni ítems
comprados a esa altura; eso está en los eventos de la línea de tiempo, que el crawler no guarda.

**3. Relación entre variables y poda** (`historial/ideas_posteriores/brainstorm_47_relacion_features.py`). Sobre las 653 columnas densas
del escalón 9 (todo menos las 851 del one-hot del draft, que queda siempre), estandarizadas con el entrenamiento.

*PCA.* Para el 50 / 80 / 90 / 95 / 99 % de la varianza hacen falta 46 / 156 / 241 / 321 / 458 componentes; la dimensión
efectiva (razón de participación, $(\sum \lambda_i)^2 / \sum \lambda_i^2$) es 119. Logística sobre draft + las primeras $k$
componentes:

| Componentes | Varianza | V AUC | Δ [IC 95 %] |
| :---: | :---: | :---: | :---: |
| 653 (todas) | 100 % | 0,6736 | −0,0003 [−0,0005; −0,0001] |
| 400 | 97,9 % | 0,6749 | +0,0009 [−0,0003; +0,0024] |
| 300 | 94,0 % | 0,6738 | −0,0002 [−0,0021; +0,0017] |
| 200 | 86,1 % | 0,6683 | −0,0057 [−0,0086; −0,0027] |
| 100 | 69,0 % | 0,6569 | −0,0170 [−0,0217; −0,0134] |
| 30 | 41,0 % | 0,6408 | −0,0332 [−0,0401; −0,0271] |
| 5 | 12,7 % | 0,6241 | −0,0498 [−0,0576; −0,0428] |

*Clustering jerárquico de variables* (distancia $1 - |\rho|$, enlace promedio; cada grupo se reemplaza por la media de sus
variables estandarizadas con el signo alineado):

| Grupos | V AUC | Δ [IC 95 %] |
| :---: | :---: | :---: |
| 500 | 0,6747 | +0,0008 [−0,0002; +0,0019] |
| 400 | 0,6753 | +0,0013 [−0,0002; +0,0032] |
| 300 | 0,6697 | −0,0043 [−0,0073; −0,0013] |
| 200 | 0,6628 | −0,0112 [−0,0148; −0,0079] |
| 100 | 0,6540 | −0,0200 [−0,0252; −0,0161] |
| 30 | 0,6373 | −0,0367 [−0,0428; −0,0308] |
| 15 | 0,6322 | −0,0418 [−0,0490; −0,0344] |

Lectura: hay redundancia para bajar de 653 a unas 300-400 columnas sin perder (con 400 la estimación es levemente mejor, +0,0009
con PCA y +0,0013 con grupos, las dos con el IC tocando 0: no se adopta mirando solo V), pero no hay una estructura de pocas
dimensiones: por debajo de 300 se pierde rápido. La señal no está solo en las direcciones de más varianza (por AUC univariado,
tras la componente 1 siguen la 11, la 12 y la 44), así que cortar por varianza no es la mejor poda; coincide con la poda L1 de
§9.S.69 (no hay poda gratis fuerte). Con un corte grueso en 15 grupos, las variables se agrupan por tipo de medida más que por
bloque de origen: el rendimiento del jugador, el margen por campeón y el margen por línea caen en los mismos grupos. Sacar
el grupo de rendimiento + experiencia + Kalman cuesta −0,0091 [−0,0114; −0,0061]; los de rendimiento y margen, hasta −0,0044;
los de atributos estáticos del campeón, −0,0001 a −0,0012. Entre familias la correlación media es baja (máximo 0,17,
margen esperado por par ↔ modelo por línea): los bloques de la escalera dicen cosas distintas. *Alternativas descartadas:*
K-Means sobre variables (pedido explícito de no usarlo; además exige fijar $k$ y supone grupos esféricos) y clustering de
partidas (ya hecho en §9.S.41, sin tipos naturales). Tablas en `data/analysis/brainstorm/relacion_*.csv`.

**4. Ítems** (`historial/ideas_posteriores/brainstorm_48_items.py`; metadatos en `data/ddragon/16.17.1/item.json`, bajado de Data
Dragon). La base guarda el inventario final (`item0..item5`), sin orden de compra. Ítem completo: sin mejora posterior, ≥ 1.600
de oro, ni consumible ni trinket ni botas. El inventario final depende del resultado (winrate 35 % con 0-1 ítems completos, 54 %
con 3, 58 % con 5), así que el winrate se mide como residuo: $r = y - \hat p(\text{línea}, \text{n.º de ítems completos},
\text{duración})$, y para el efecto de un ítem se resta además la media previa del residuo de su campeón-línea.

* *Build general.* 349 campeón-línea con ≥ 300 partidas; los 3 ítems más comunes aparecen en promedio en el 55 % de las
  partidas, y la mediana de ítems completos distintos por campeón-línea es 41. Tabla en `items_build_general.csv`.
* *Rareza y outliers.* Rareza de una build = media de $-\log_2 \frac{n_{\text{ítem}} + 1}{n_{\text{campeón-línea}} + V}$ sobre
  sus ítems completos, con conteos de partidas anteriores. Por percentil dentro del campeón-línea, el residuo va de −0,26 %
  (0-20) a +0,28 % (60-80) y cae a **−0,73 % ± 0,19 % en el 5 % más raro**. Depende de la experiencia: con menos de 5 partidas
  previas con el campeón, −1,53 % ± 0,27 %; con 5-29, −0,30 % ± 0,33 %; con 30 o más, +0,77 % ± 0,47 %. La build rara de alguien
  que no conoce al campeón es un error; la de un veterano no.
* *Efecto por ítem, suavizado.* $\hat e = \sum r / (n + K)$ con $K = 100$. De 2.618 pares (campeón-línea, ítem) con ≥ 200 usos,
  212 tienen $|z| > 3$ (por azar se esperan 7). Pero los más altos son los que se compran yendo adelante (Guardian Angel +6,2 %
  en general, +9 a +12 % en junglas asesinos) y los más bajos los que se compran yendo atrás (Serpent's Fang −3,8 %,
  Morellonomicon −3,7 %, Randuin's Omen −2,9 %, Thornmail −2,2 %): el residuo no saca del todo la confusión con el estado de
  la partida. **Es descriptivo, no causal.**
* *En el modelo previo.* Media de las partidas anteriores del jugador (y del jugador con ese campeón) de la rareza y del
  efecto de sus builds, azul − rojo: por línea (20 columnas) +0,0002 [−0,0003; +0,0007]; media del equipo (4) +0,0002
  [−0,0001; +0,0006]; placebo −0,0001. **No suma**, igual que el perfil de build de Gemini (§9.S.52).

Para separar "ítem bueno" de "ítem que se compra ganando" hace falta el orden de compra. La respuesta cruda de Riot trae
`challenges.legendaryItemUsed`, una lista de los ítems legendarios del jugador (ver punto 5); falta comprobar que esté en orden
de compra.

**5. Qué datos faltan y qué trae la respuesta cruda de match-v5.** Verificado: el crawler no guarda bans (`matches` tiene id,
fecha, duración, versión y cola; `match_participants` no tiene bans, runas, hechizos ni `challenges`), no queda el JSON crudo,
y `runas.db` guardó solo runas y hechizos. Se bajaron dos partidas de V (82 KB cada una, 10 KB comprimidas) para ver qué viene:

* *Previo a la partida:* los 10 bans con campeón y turno de baneo (−1 si no baneó); **`selectedRolePreferences`** por jugador
  (por ejemplo `JUNGLE.AUTOFILL.BOTTOM.TOP.FILL`: rol asignado, si fue primario / secundario / autofill, y roles pedidos) y
  `positionAssignedByMatchmaking`; runas y hechizos completos. El autofill hoy se estima por el historial (5 % del logit en
  §9.S.70); esto lo da exacto. El orden de pick **no viene** (el turno es de baneo).
* *Como historial del jugador:* 127 métricas de `challenges` (solo kills, placas, ventaja de línea, CS de jungla antes del 10,
  skillshots esquivados y acertados, visión contra el rival, tiempos del primer dragón y la primera torre), pings por tipo,
  tiempo muerto, `wasAfk` y marcas de conducta.
* *Por equipo:* objetivos totales al final (no por minuto: no sirven para el modelo en vivo).

`scripts/collection/partidas_crudas.py` (con tests) baja la respuesta entera comprimida a `data/crudas/partidas.db`, sin tocar
`lol_las.db`, en un orden fijo que mezcla A2-A4 Master+ y V (≈ 30.300 partidas, ~8 h 45 con una key de desarrollo). **La
descarga no se lanzó: espera la decisión del usuario.** *Alternativa descartada:* bajar solo V (16.428): sin A2-A4 no hay
partidas con logit previo fuera de muestra para ajustar el peso de las variables nuevas.

Otros datos que el emparejamiento no usa y no tenemos: maestría de campeón y LP / racha del momento (la API da solo el valor de
hoy: sirven en vivo, en el histórico son fuga), partidas en otras colas, y los eventos de la línea de tiempo (objetivos por
minuto y orden de compra).

#### S.72. Reorganización: lo vigente en `scripts/analysis/` y los experimentos descartados en `historial/` (2026-11-22)

**Pedido del usuario.** El repo quedó grande para presentarlo: `scripts/analysis/` tenía 274 archivos (~48.500
líneas de Python) y la mayoría eran experimentos que ningún modelo final usa. Se pidió una parte del repo que
sea el historial de pruebas, bien organizado (idea → resultado → por qué se descartó), y lo vigente con la forma
habitual (`src/`, `notebooks/`, `tests/`).

**Criterio para decidir qué es vigente.** Un script queda en `scripts/analysis/` si cumple alguna de estas tres
condiciones; el resto pasa a `historial/`:

1. es una raíz vigente: los modelos finales (escalón 9 y `6_Final_dEns10`), matchmaking, clustering, huella,
   rating, validación, la comparación con DraftRec y los generadores de notebooks;
2. lo importa (de forma transitiva) una raíz vigente: el grafo se armó con los imports reales (`ast`), no con
   menciones en el texto;
3. escribe una caché que lee un script vigente. Esta tercera condición no la ve el grafo de imports, y fue la
   que hizo falta: el modelo en vivo lee `mc_ajuste.npz`, `conoce_matchup.npz`, `margen_grupos_escala_sin_v.npz`,
   `gemini_g24.npz`, `gemini_hgb_ctx_p15.npz`, `gemini_g30_lk15.npz`, `gemini_g20.npz` y `L_esc9_92k.npz`, que
   arman scripts de la etapa de brainstorming descartados como modelos sueltos. Esos 9 scripts (y `g_29`, que
   importa `g_30`) siguen en `scripts/analysis/`. El chequeo se repitió hasta que ya no apareció ninguna
   dependencia nueva.

Resultado: 78 scripts vigentes y 111 movidos (más las carpetas `principal_29oct/` y `mecanica_lp/`), en 7
grupos por etapa: `ganador_primeros/` (§9.S.23-§9.S.39), `brainstorming_noche/` (§9.S.52), `en_vivo/`,
`ideas_posteriores/`, `clustering_huella/`, `elo_principal_oct/` (§9.S.21) y `mecanica_lp/` (§9.S.40).
`historial/README.md` tiene una tabla por grupo con la idea, el resultado (con los números de esta memoria
o de `docs/BRAINSTORM_NOCHE.md`) y el motivo del descarte. `scripts/analysis/README.md` ordena lo vigente por
área, con el orden de las cachés del escalón 9.

**Cómo se movió sin romper nada.** Los scripts importan con rutas absolutas (`scripts.analysis.X`) y calculan
la raíz del repo con `Path(__file__).resolve().parents[2]`. `historial/<grupo>/x.py` tiene la misma profundidad
que `scripts/analysis/x.py`, así que la raíz no cambia. Se reescribieron los imports
(`scripts.analysis.X` → `historial.<grupo>.X`) y las rutas citadas en el código y en la documentación. Los
scripts de `historial/` pueden importar de `scripts/analysis/` (por ejemplo `brainstorm_base`), pero no al
revés.

**Guarda nueva** (`tests/test_scripts_importables.py`). Los scripts de análisis no se ejecutan en los tests
(necesitan los datos y tardan minutos). La guarda comprueba, sin ejecutarlos, lo que una reorganización suele
romper: que cada script compile; que cada import entre scripts apunte a un archivo existente; que `parents[k]`
dé la raíz del repo; y que nada de `scripts/` importe de `historial/`. Detectó dos errores durante el
movimiento: un cambio de `parents[2]` a `parents[3]` que era innecesario, y un import de `g_29` desde `g_30`.
Tiene un caso por script, así que la suite pasa de 410 a 669 tests.

**Alternativas descartadas.**

* *Borrar los experimentos.* Esta memoria los cita como evidencia de cada descarte, y la sección 1 de
  `eda_modelos_ganador` reproduce el camino con `historial/ganador_primeros/`. Borrarlos deja afirmaciones sin
  respaldo.
* *Una rama aparte (`entrega`) sin los experimentos.* Obliga a mantener dos ramas sincronizadas con el
  compañero. Una carpeta en la misma rama los deja a un clic y la guarda la protege.
* *Pasar los modelos finales a módulos de `src/` y renombrar los scripts `brainstorm_*`.* Es lo más limpio para
  leer, pero cambia los imports de ~100 scripts y de los notebooks, choca con el trabajo en curso en `gonchi`, y
  para validarlo hay que regenerar toda la cadena de cachés (~90 min de CPU más GPU) y comprobar que el AUC no
  cambie. Queda como paso siguiente, si hace falta para la entrega.
* *Clasificar por menciones en el texto en vez de por imports.* Sobreestima: las docstrings citan scripts viejos
  como contexto ("como en `brainstorm_g_01`").

#### S.73. El modelo previo (escalón 9) en `src/ganador/`, con cachés que se invalidan solas (2026-11-22)

**Pedido del usuario (issue #2).** Para la entrega, el modelo previo vigente (escalón 9, §9.S.52 y §9.S.68) tenía que
correrse con un solo comando y sin depender de la cadena de ~12 scripts `brainstorm_*` de la noche de brainstorming.
Esa cadena tenía tres problemas: (1) cada caché se reutilizaba **solo porque el archivo existía**, así que si cambiaba
el código o los datos se podía entrenar con variables viejas sin darse cuenta (la guarda por fecha de
`cargar_hgb_por_linea` cubría un solo caso); (2) `brainstorm_base.cargar` exigía el mapa de LP y, con él,
`lp_vivo.db`, aunque el modelo final no usa el mapa desde §9.S.68; (3) había rutas absolutas a `G:\`.

**Qué hay ahora.**

| Módulo | Qué hace |
| :--- | :--- |
| `src/ganador/rutas.py` | Rutas relativas a la raíz del repo (`data/splits/train`, `lol_las.db`, atributos, Data Dragon) |
| `src/ganador/_constructores.py` | Las funciones que calculan cada grupo de variables, **copiadas tal cual** de los scripts originales (las generó `tools/generar_constructores_ganador.py`, que verifica con `assert` que el cuerpo copiado coincida) |
| `src/ganador/cache.py` | Caché con huella (abajo) |
| `src/ganador/datos.py` | Partición V / `fi_v` / Diamond-V y las matrices base (`Xd`, `DRAFT`), sin mapa de LP |
| `src/ganador/variables.py` | `calcular_todas()` (9 grupos con nombres claros: `kalman_elo`, `contracruces`, `oro_linea_hgb`...) y `matriz_extras()` |
| `src/ganador/modelo_previo.py` | `ajustar`, `predecir`, `metricas` y `evaluar_v` |
| `scripts/analysis/modelo_previo.py` | Un solo comando: `python scripts/analysis/modelo_previo.py [--rehacer] [--comparar]` |

**El modelo** es el mismo de §9.S.52: logística L2 con C = 0,05 sobre $[\,\text{DRAFT}\;|\;z(X_d)\;|\;z(E)\,]$, donde
DRAFT es campeón × línea con signo (851 columnas dispersas), $X_d$ son las 85 variables por jugador azul − rojo por
línea (425) y $E$ los extras del escalón 9 (228). La estandarización $z(x) = (x - \mu_{tr}) / (\sigma_{tr} + 10^{-6})$
usa **solo las filas de entrenamiento** (Master+ anteriores al 16/09, `fi_v` ∩ `master`, 118.064 partidas). El
acierto se compara con la línea base de predecir siempre el lado más frecuente de V ($\max(\bar y, 1-\bar y)$ = 52,2 %).

**Caché que se invalida sola.** Cada grupo se guarda en `data/cache/ganador/<grupo>.npz` junto con una huella

$$h = \mathrm{SHA256}\big(\,\mathrm{src}(f) \,\|\, \{\text{nombre, tamaño, mtime}\}_{\text{archivos}} \,\|\, h_{\text{dependencias}}\big)$$

donde $\mathrm{src}(f)$ es el código fuente de la función que lo calcula y de las funciones del mismo módulo que
llama (recursivamente). Si la huella guardada no coincide, el grupo se recalcula. Como la huella de un grupo
incluye las de los grupos de los que depende (por ejemplo, `oro_linea_hgb` depende de `margen_campeon_linea`,
`nivel_historico` y `contracruces`), un cambio aguas arriba se propaga solo.

**Verificación.** Desde cero (sin ninguna caché), `modelo_previo.py --comparar` tarda ~22 min (8 de ellos el HGB
por línea); con la caché vigente, ~45 s. En V Master+ (16.428 partidas): AUC 0,6740, acierto 62,6 %, log loss
0,6442, ECE 0,0104, igual que la referencia regenerada con `brainstorm_final.py`, y las 16.428 probabilidades son
**idénticas bit a bit** a `pred_final.npz` ($\max|\Delta p| = 0$). Cobertura: 80,4 % de acierto en el 5 % más seguro,
76,4 % en el 20 % y 69,9 % en el 50 %.

**Tests** (`tests/test_ganador_modelo_previo.py`): con datos sintéticos de señal conocida, el modelo la aprende
(AUC > 0,75 y el coeficiente mayor en la variable con señal); entrena solo con Master+ anterior a V; **no-fuga**:
multiplicar las variables de V por 100 e invertir su resultado no cambia ni las escalas ni los coeficientes, y la
predicción de una partida no depende de las demás de V; la caché recalcula cuando cambia la huella, y la huella
cambia con el código, con los datos y con las dependencias.

**Alternativas descartadas.**

* *Reescribir los constructores con código limpio.* Hubiera sido más legible, pero cualquier diferencia mínima
  (orden de un `sort`, un `fillna`) cambia el AUC y obliga a justificar un número nuevo. Copiarlos tal cual y
  comprobar que las predicciones son idénticas deja el número de §9.S.68 respaldado sin discusión.
* *Invalidar por fecha de modificación (como `cargar_hgb_por_linea`).* No detecta un cambio de código y obliga a
  listar a mano los insumos de cada caché; la huella cubre las dos cosas.
* *Mover ya los scripts `brainstorm_*` a `historial/`.* Todavía los importan scripts vigentes: el modelo en vivo
  (`ingame_preparar.py`), matchmaking, `build_eda_modelos_nb.py`, `modelos_train_v.py`, `umap_vistas.py` y otros.
  Se moverán cuando esos usen `src.ganador`; mientras tanto, `brainstorm_final.py` queda como referencia.

#### S.74. Reproducibilidad: rutas relativas, lecturas solo del train y dependencias (2026-11-23, issue #8)

**Qué se buscaba.** Que cualquier script vigente corra en otra máquina con solo clonar el repo y poner `data/`
en su lugar, y que ningún script de ajuste pueda leer el test aunque la base crezca.

**1. Rutas.** Cuatro scripts vigentes tenían rutas absolutas a la unidad de Drive de una sola PC
(`G:\Mi unidad\Lol Database\...`): `brainstorm_g_06_comp_poisson.py` y `ingame_preparar.py` (atributos de
campeones, con la ruta local como respaldo), `mapa_lp_basico.py` (`lp_vivo.db`) y `umap_vistas.py` (valores por
defecto de `--players` y `--firmas`). Ahora todas salen de `src/ganador/rutas.py`, relativas a la raíz del repo:
`ATRIB_CSV` (ya existía), y tres constantes nuevas, `PLAYERS_CSV`, `FIRMAS_A_CSV` y `LP_VIVO_DB`. Una búsqueda de
`[A-Z]:\` y `G:/` en `src/`, `scripts/` y la raíz no encuentra otras. En otra máquina, el que tenga los datos en
otro lado pasa `--lp-vivo`, `--players` o `--firmas`, o enlaza `data/`. *Descartado:* una variable de entorno
`LOL_DATA` (agrega configuración que nadie más usa; la convención del repo ya es `data/` en la raíz) y dejar el
respaldo a `G:` (en otra PC no existe y tapa que falta el archivo local).

**2. Lecturas del CSV global.** `huella_mejoras.py` (parte 2), `huella_niveles.py` (`datos_timeline`) y
`huella_previo.py` (`habito_progresivo`) leían `data/full_dataset.csv`, que incluye el test, y lo filtraban por
las partidas con timeline de `data/timelines_eventos/timelines.db`. Pasan a `rutas.TRAIN_CSV`
(`data/splits/train/full_dataset.csv`). Tamaños hoy: `full_dataset.csv` tiene 272.849 partidas, el train 192.104
(todas contenidas en el global) y la diferencia, el test, 80.745 (80.417 de LA2). Las partidas con timeline usadas
hasta ahora (100.049, §10.4) son V y Master+ del ajuste, seleccionadas desde `ingame_cache.npz`, que se arma con el
train: el cambio no altera los resultados publicados. Lo que evita es que, cuando
`timelines_eventos.py --todas` baje el grupo `resto` (el resto de `lol_las.db`, test incluido), esas partidas
entren solas a la parte 2 de la huella o a la media progresiva de hábitos de `huella_previo` (que sí alimenta un
modelo). La base de timelines y `habitos.pkl` no estaban en la máquina donde se hizo el cambio, así que el conteo
exacto "antes = después" queda por confirmar donde estén (un `nunique` de `match_id` antes y después basta).

**3. Dependencias.** `requirements.txt` decía que `lightgbm`, `catboost` y `xgboost` eran "solo para
`ganador_modelos.py`", que ya está en `historial/`. Uso real en lo vigente: `lightgbm` en `ingame_preparar.py`,
`ingame_evaluar.py` y `src/features/lp_partida.py`; `catboost` en `ingame_evaluar.py` y
`brainstorm_g_08/13/22/28`; `xgboost` solo en `historial/` (opcional). También se corrigió el comentario de
`torch` (lo usan el modelo en vivo, `huella_*`, `embeddings_partidas` y redes `brainstorm_g_*`, no solo
`elo_gru.py`) y el de `ngboost` (`elo_boostings.py`, `build_lp_partida.py`). Las versiones no cambian.

**4. README de `scripts/analysis/`.** La sección del modelo en vivo listaba `brainstorm_g_20`, `g_29`, `g_30` e
`ingame_g_02` como entradas del ensamble; volvieron a `historial/` (commit `ea930bd`) porque
`ingame_g_04_mt_a4_gpu.py` no los lee. De `ingame_g_01` e `ingame_g_03` el paso final solo importa funciones;
`ingame_g_03` corrido como script todavía lee la caché de `g_30` para una variante que no entra al paso final.

#### S.75. Nivel histórico sin la foto 1 antes de la foto: máscara temporal (2026-11-23, issue #4)

**El problema.** `nivel_historico` (`brainstorm_07_mmr_historico.py::construir_cache`, copiado en
`src/ganador/_constructores.py`) mide el nivel de un jugador por emparejamiento: el promedio del LP de la foto 1
($x_j$, LP absoluto del ranking tomado en $t_1$ = 04/09) de los otros 9 en sus partidas previas, encogido hacia
$\mu_0$ (§9.S.31):

$$\text{nivel}_{\text{acum}}(i) = \frac{\sum_{p < i} L_p + k\,\mu_0}{n_i + k}, \qquad L_p = \frac{1}{|O_p|}\sum_{j \in O_p} x_j,\; k = 5$$

(más la versión de las últimas 10 y $\log(1 + n_i)$; el escalón 9 usa esas 3 columnas, azul − rojo y media por
equipo; el HGB por línea, `oro_linea_hgb`, las usa por jugador, `A[..., :3]`). `df["x"] = df.player_id.map(foto1["z"])`
no tenía máscara temporal: en una partida jugada en $t < t_1$ la variable usaba un LP medido **después** de la
partida, que ya incorpora su resultado. Son el **84,2 %** de las 118.064 partidas con que se ajusta el modelo de V
(78,2 % de `fi`). V es posterior a $t_1$, así que no se inflaba V; lo que se sesgaba eran los pesos (de la logística
y del HGB), aprendidos sobre todo en filas donde la variable sabía algo del futuro.

**La corrección.** Para $t < t_1$ las 5 columnas de `A` valen el **prior**, 0: nivel $= \mu_0$ (centrado, 0),
$n = 0$, brecha 0 y "sin foto". Para $t \ge t_1$ no cambia nada: la foto ya se conocía, también el LP de los
rivales de partidas previas a $t_1$, así que esas filas siguen usando todo el historial (verificado: `A` idéntica
bit a bit en las filas posteriores). Es el mismo criterio que el W/L de la foto en `brainstorm_g_01` (`wl_foto1`:
`has_f1 = has_raw & post`, 0 si no). El HGB por línea queda corregido por el mismo cambio, porque recibe `A` del
grupo `nivel_historico` (su huella depende de la de ese grupo, así que se recalcula solo).

**Por qué prior 0 y no NaN.** (1) La logística no admite NaN: habría que imputar, y la imputación natural de una
diferencia azul − rojo es 0, el mismo valor. (2) 0 es exactamente lo que la cuenta da para un jugador sin
historial ni foto, así que no introduce un valor nuevo. (3) Es la convención de `wl_foto1`, la otra variable de la
foto ya enmascarada. Para el HGB NaN sería posible (lo maneja de forma nativa), pero las filas enmascaradas tienen
exactamente 0,0 en las tres columnas y las demás casi nunca: el árbol puede separarlas igual.

**Alternativa descartada: enmascarar la entrada** ($x_j =$ NaN en las filas con $t < t_1$, antes de
`prior_lobby_level`). Eso también borra, para las partidas posteriores a $t_1$, el aporte de los rivales de sus
partidas anteriores a $t_1$, que es información legítima en ese momento. La fuga está en *cuándo se consulta* la
variable, no en *qué partidas previas* se promedian: por eso la máscara va sobre la salida.

**Resultado (V Master+, 16.428 partidas; IC 95 % por bootstrap pareado, 1.000 réplicas).** Empeora levemente:

| | AUC | acierto | log loss | ECE |
| :--- | ---: | ---: | ---: | ---: |
| Antes (con la fuga; = `pred_final.npz` bit a bit) | 0,6740 | 62,64 % | 0,6442 | 0,0104 |
| Después (máscara $t < t_1$) | 0,6727 | 62,41 % | 0,6451 | 0,0109 |
| Δ | −0,0013 [−0,0025; −0,0002] | −0,23 pp [−0,56; +0,10] | | |

Por partes (mismo bootstrap): solo la logística con `A` enmascarada y el HGB viejo, ΔAUC −0,0007 [−0,0018;
+0,0004]; solo el HGB nuevo, −0,0001 [−0,0005; +0,0002]; las dos juntas, −0,0013 (no es aditivo). El HGB nuevo
predice casi lo mismo en V (correlación con el viejo 0,986-0,993 por línea).

En la validación por bloques (A2-A4 Master+, cada uno con lo anterior, regla de §9.S.47): ΔAUC A2 −0,0010,
A3 −0,0014, A4 +0,0001; media −0,0008 [−0,0022; +0,0008], mejora en 1 de 3: **no concluyente** por la regla.
Acierto A2 62,94 → 62,43 %, A3 62,41 → 62,01 %, A4 60,99 → 61,03 %.

**Lectura.** La pérdida es esperable y no indica que la máscara esté mal: con la fuga, el coeficiente del nivel se
aprendía con 118 mil partidas en las que la variable era un proxy *mejor* del MMR (el LP de la foto, medido después,
ya refleja la partida); ahora se aprende solo con el 15,8 % posterior a $t_1$, y el resto aporta 0. V no tiene fuga
en ninguno de los dos casos, pero el número viejo venía de pesos estimados con una variable que, en las partidas
de entrenamiento, no significaba lo mismo que en V o en el test (donde siempre es previa). **La corrección se
aplica igual**, por razones metodológicas: el modelo previo vigente pasa a **AUC 0,6727 / acierto 62,4 %** en
V Master+.

**Pendiente.** (1) `data/cache/ganador/nivel_historico.npz` y `oro_linea_hgb.npz` se recalculan solos la próxima vez
que se corra `modelo_previo.py` (~16 min, casi todo el HGB), porque cambió la huella. Las versiones medidas acá
están en `data/cache/ganador_issue4/` (con `pred_v_issue4.npz`). (2) `data/analysis/brainstorm/mmr_historico.npz`,
que leen `brainstorm_final.py`, el modelo en vivo (`ingame_preparar.py`), matchmaking y varios `brainstorm_g_*`,
**sigue con la fuga** hasta correr `brainstorm_07_mmr_historico.py --solo-cache --rehacer` (esa caché se reusa solo
porque existe). Al regenerarla cambian `pred_final.npz` y todo lo que la usa, así que queda para cuando el usuario
lo decida. (3) `MMR_V2` de `brainstorm_g_10` (no entra al modelo final) usa la foto 1 con un esquema propio; no se
auditó acá.

**Resuelto el 2026-11-23 (pedido del usuario).** (1) Se copiaron a `data/cache/ganador/` las dos entradas de
`ganador_issue4/` (las anteriores, en `data/cache/ganador_pre_issue4/`). (2) Se regeneraron `mmr_historico.npz`
(`brainstorm_07 --solo-cache --rehacer`) y la caché del HGB por línea que la lee (`gemini_hgb_lineas.npz`,
`brainstorm_g_09 --solo-cache --rehacer`), y se volvió a correr `brainstorm_final.py`: V Master+ 0,6727 / 62,4 %, y
`pred_final.npz` es idéntico bit a bit (máx $|\Delta p| = 0$, mismos índices) a `data/cache/ganador/pred_v.npz` de
`modelo_previo.py`. La cadena vieja y `src/ganador` vuelven a coincidir. Las versiones con la fuga quedaron en
`data/analysis/_respaldo_pre_issue4_20261123/`. Matchmaking y el ensamble del anexo leen la caché nueva a partir de
su próxima corrida; sus números publicados son anteriores a la corrección.

**Test** (`tests/test_ganador_nivel_historico.py`): con datos sintéticos y la foto simulada, las filas con
$t < t_1$ no cambian al cambiar los LP de la foto y valen 0 (falla con el código viejo); las posteriores sí usan
el LP, cuentan las partidas previas a $t_1$ y coinciden con la cuenta sin máscara.

#### S.76. Crawler y base: cierre con tope, tier en orden de liga y partidas de 10 (2026-11-23, issue #9)

**Qué se buscaba.** La auditoría de datos dio la base sana (272.849 partidas, todas con 10 participantes, sin
duplicados, train y test sin solaparse), pero dejó cinco puntos del código que podían romperla o colgar una
sesión si se vuelve a recolectar. Ninguno cambia una fila de la base ni de los 4 CSV actuales (§3.G).

**1. El cierre espera al escritor con tope.** `collect_matches` esperaba al hilo escritor con `join()` sin
límite: un lock ajeno sobre la base o un disco lleno colgaban el cierre para siempre. Ahora espera
`WRITER_JOIN_TIMEOUT_SECONDS = 300`. La cola guarda hasta 200 partidas, un lote de 5 se escribe en
milisegundos y cada conexión espera hasta 60 s un lock (`busy_timeout`), así que 300 s cubren varios
reintentos sin acercarse a un cierre normal. Si no termina: aviso, no se refresca `player_id_map` ni se
exportan los CSV (leerían una base a medio escribir), el reporte lleva `writer_timed_out = True` y `run.py` no
encadena otro ciclo (sumaría un segundo escritor sobre una base trabada). El hilo es *daemon*: no impide que
el proceso termine. *Descartado:* matar el hilo (Python no puede, y cortar un commit a mitad es peor) y
exportar igual (el export dejaría de ser el volcado fiel de la base).

**2. Tier de una persona en orden de liga.** `MAX(tier)` compara texto: $\text{DIAMOND} > \text{CHALLENGER} >
\text{MASTER}$ alfabéticamente. Una persona con dos PUUIDs en el padrón, uno DIAMOND y otro CHALLENGER, salía
DIAMOND en `full_dataset.csv`, `players.csv`, el split y la cobertura. Se define
`TIER_ORDER_LIGA = (IRON, …, DIAMOND, MASTER, GRANDMASTER, CHALLENGER)` en `src/config.py` y
`sql_tier_maximo(col)` en `src/database.py`, que calcula
$\operatorname{tier}^\* = \text{TIER\_ORDER\_LIGA}[\max_i r(\text{tier}_i)]$ con $r$ = posición en la lista
(0 para `NULL`, `UNKNOWN` o valores desconocidos); si ningún valor del grupo es un tier conocido devuelve
`MAX(col)` como antes, así `NULLIF(…, 'UNKNOWN')` sigue funcionando. Se usa en los tres `padron_players` /
`padron` de `export_to_csv.py`, en `player_tier` de `players.csv` y en la cobertura de `database.py`. Los
órdenes que ya había (`src/features/split.py`, `fingerprint.py`, `_TIER_BASE` del collector) cubren solo
Diamante+ y ordenan en pandas; no se tocan. Verificación de solo lectura sobre `data/lol_las.db`, comparando
la expresión vieja con la nueva: 0 de 40.821 personas cambian de tier, en el export y en la cobertura (el
mapa de identidad no une dos PUUIDs del padrón por Riot ID; solo el alias legacy lo haría). El rango
(`MAX(pl.rank)`) sigue comparando texto (`IV > I`), pero hoy tampoco hay personas con dos divisiones; se anota.

**3. Contadores aparte.** Un lote cuyo commit fallaba sumaba sus partidas a "descartadas (remakes /
errores)", mezclando un fallo transitorio con un descarte por criterio. Ahora van a `write_errors`, que no
suma en descartadas (esas partidas no quedan en `discarded_matches` y se vuelven a pedir). Las rechazadas por
participantes (punto 4) sí son descartes y suman en descartadas, con su propia línea. Los dos contadores
salen en el reporte de cierre, en el diccionario de la sesión y en el latido (`crawler_metrics_timeline`
gana las columnas `write_errors` y `rejected_participants`, migración idempotente con `ALTER TABLE` al abrir
la base).

**4. Partidas de 10.** `_write_match_batch` y `save_match_and_participants` rechazan la partida si no trae
exactamente 10 participantes con 10 PUUIDs distintos (con uno repetido el `INSERT OR IGNORE` dejaba 9 filas).
Registra la partida en el log, suma `partidas_rechazadas_participantes` y sigue con el lote: no lanza
excepción. No se anota en `discarded_matches`: un payload truncado puede ser transitorio y conviene volver a
pedirlo. *Descartado:* guardarla igual y filtrar en el preprocesamiento (rompe la invariante de 10 filas que
suponen los exports y los modelos, §3.F.5) y lanzar excepción (perdía el lote entero). La fixture `temp_db`
de los tests apaga el control (`participantes_esperados = None`) porque sus partidas de juguete traen 1 o 2
participantes; los tests del control lo vuelven a encender.

**5. `export_to_csv.py`.** Leer `matchups_1v1.csv` para los diferenciales de `players.csv` tenía
`except Exception: pass`. Pasa a `except (OSError, ValueError)` (archivo bloqueado, vacío o corrupto: las
excepciones de pandas son `ValueError`) con un aviso que dice por qué `players.csv` sale sin diferenciales.

**6. Corrida sobre una copia (`run.py --db RUTA`).** Todo lo que escribe una corrida sale de la ruta de la
base: `validation_state.json` y los CSV del cierre van junto a ella. `--db` no crea una base si la ruta no
existe, y no se combina con `--respaldar` (que sube `data/lol_las.db`). `--solo-validar` sobre una copia de la
base real dio la auditoría completa sin novedades (272.849 partidas; los 54 `ganador_incoherente` ya conocidos,
§9.S.19) y `data/` quedó sin cambios. **Pendiente para el usuario** (usa la Riot API y las keys del `.env`),
desde la raíz del repo principal:

```powershell
mkdir C:\tmp\copia_lol
python -c "import sqlite3; s=sqlite3.connect('file:data/lol_las.db?mode=ro', uri=True); d=sqlite3.connect(r'C:\tmp\copia_lol\lol_las.db'); s.backup(d); d.close(); s.close()"
copy data\validation_state.json C:\tmp\copia_lol\
python run.py --db C:\tmp\copia_lol\lol_las.db --minutos 10
```

Mirar en el reporte de cierre: que termine sin el aviso del escritor, `errores de escritura = 0`, rechazadas
por participantes = 0 y que los 4 CSV aparezcan en `C:\tmp\copia_lol\` (no en `data/`). Después borrar la
copia (~2,1 GB).

**Hecho el 2026-11-23** con una key nueva: la corrida de 10 minutos sobre la copia cerró sin el aviso del escritor,
con 0 errores de escritura, 0 rechazadas por participantes y 0 errores HTTP; guardó 4 partidas nuevas (la
auditoría pasó de 272.849 a 272.853 sin novedades: los mismos 54 `ganador_incoherente`). Los 4 CSV se
escribieron en la carpeta de la copia y los de `data/` quedaron sin tocar; la copia se borró. Aviso aparte: 15
pedidos rebotaron con `Exception decrypting` porque la key es de otro proyecto de API que los PUUIDs guardados;
para recolectar de verdad hace falta el pool permanente (P0/P1). La recolección la hace el compañero.

#### S.77. Matchmaking como simulación, silhouette comparable y residuo de la huella sin fuga (2026-11-23, issue #11)

**Qué se buscaba.** La revisión de los modelos (`docs/REVISION_MODELOS_20261120.md`) marcó tres problemas en
análisis secundarios: una afirmación del matchmaking que el código no sostenía, una comparación de silhouettes
entre conjuntos distintos y una fuga leve en la normalización de la huella. Los tres se corrigieron y se volvieron
a medir.

**1. Matchmaking (§9.S.67): simulación, protección individual y dúos.** Se volvieron a correr los tres
`matchmaking_*.py` (calibración 246 s, perfil de roles 150 s, reacomodos 356 s). La calibración reproduce §9.S.67
(sin draft: AUC 0,6369, b = 1,027; ECE 0,78 % contra 0,75 % documentado, diferencia de redondeo en los deciles).

*Qué decía mal.* "Sin empeorar roles" describía la fila $D \le D_\text{real}$, que topa solo la **suma**
$D = \sum_i d_i$. La suma puede bajar aunque un jugador quede peor, si otro mejora más. Medido en
`swaps_jugadores.parquet` (164.280 jugador-partida de V Master+), con $\Delta d_i = d_i(\text{nuevo}) - d_i(\text{Riot})$:

| Variante, τ = 0,20 (V Master+) | Jugadores con $\Delta d_i > 0$ | Partidas con alguno | Evaluado por el otro modelo | En [0,45; 0,55] | D medio | Cambian de rol |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Riot (real) | — | — | 10,03 pp | 31 % | 1,252 | — |
| (a) solo misma línea | 0 % | 0 % | 5,06 pp | 56,9 % | 1,252 | 0 % |
| **(a+b), $\Delta d_i \le 0$ para los 10 (principal)** | **0 %** | **0 %** | **4,96 pp** | **57,7 %** | 1,131 | 3,8 % |
| (a+b), $D \le D_\text{real}$ | 3,3 % (3,25 %) | 24,6 % | 4,88 pp | 58,5 % | 1,072 | 10,2 % |
| (a+b), primero el menor D | 4,1 % | 33,1 % | 4,95 pp | 58,1 % | 0,900 | 15,4 % |

Entre los que empeoran con el tope total, $\Delta d$ medio es +0,21: no es ruido de redondeo, es pasar a una línea
que juegan bastante menos. *Decisión:* la variante principal pasa a ser la que protege a cada jugador
(`ab_pareto_t20`, $\max_i \Delta d_i \le 0$, ya calculada por `matchmaking_swaps.py`). Cuesta 0,08 pp de
desbalance contra el tope total y nadie queda peor que con Riot. Es la única lectura en la que "sin empeorar el rol
de nadie" es literal.

*Simulación y dúos.* Las configuraciones nuevas no se jugaron: "más pareja" significa que la logística les asigna
una p más cerca de 0,5. La única verificación con resultados reales es la asimetría de autofill de §9.S.67.8. Tampoco
se restringen dúos ni premades: la base no registra quién entró junto a la cola, así que hasta el movimiento (a),
que no cambia ningún rol, puede separar a dos jugadores de un dúo. Esto se declaró en §9.S.67, en
`REPORTE_swaps.md` (lo escribe `matchmaking_swaps.py`), en la sección 7 de `build_eda_modelos_nb.py`, en
`scripts/analysis/README.md` y en `ROADMAP.md`.

*Alternativas descartadas.* Mantener $D \le D_\text{real}$ como principal y solo reportar el 3,3 %: la ganancia
(0,08 pp) no compensa dejar peor a uno de cada treinta jugadores. Restringir dúos con una heurística (dos jugadores
del mismo equipo con muchas partidas juntos): sin el dato de la cola sería adivinar, y es lo que la simulación no
puede ver.

**2. Clustering (§9.S.58): silhouette sobre el mismo subconjunto.** En `umap_vistas.py` la silhouette de HDBSCAN
se calculaba sin su ruido, $s(X_{\text{ok}}, g_{\text{ok}})$, y la de K-Means con todos los jugadores, $s(X, k)$.
Sacar el ruido elimina los puntos de frontera, que son los que tienen silhouette baja, así que la comparación
favorecía a HDBSCAN. Ahora la vista jugadores guarda tres columnas:

* `silhouette K-Means (sin ruido)`: el mismo K-Means, evaluado solo en los jugadores que HDBSCAN asigna (la comparable);
* `silhouette K-Means (todos)`: la de antes;
* `silhouette K-Means reajustado sin ruido`: K-Means ajustado de nuevo solo en esos jugadores.

| Rol | HDBSCAN (sin ruido) | K-Means, mismos jugadores | K-Means reajustado | K-Means, todos (antes) |
| :--- | :---: | :---: | :---: | :---: |
| TOP | 0,155 | 0,186 | 0,196 | 0,141 |
| JUNGLE | 0,161 | 0,178 | 0,201 | 0,136 |
| MIDDLE | 0,171 | 0,178 | 0,222 | 0,134 |
| BOTTOM | 0,167 | 0,222 | 0,225 | 0,189 |
| UTILITY | 0,205 | 0,258 | 0,250 | 0,228 |

La corrida reprodujo exactamente las silhouettes anteriores (HDBSCAN y K-Means con todos), así que la diferencia es
solo el conjunto. **Sobre los mismos jugadores, K-Means queda por encima en los cinco roles.** Antes parecía que
HDBSCAN ganaba en top, jungla y mid. La conclusión no cambia: todas están entre 0,15 y 0,26, el estilo es un
continuo, y HDBSCAN no encuentra grupos más separados que K-Means. Se corrigieron la tabla de §9.S.58 y los valores
de `build_eda_reconocimiento_nb.py` §1.3. Comando: `python scripts/analysis/umap_vistas.py --vista jugadores`
(~3 min). Para esta medición, la salida fue a una carpeta temporal y no se pisó `data/analysis/umap/`.

**3. Huella (§10.4): residuo ajustado solo con los jugadores de ajuste.** `residualize()` reemplaza cada estadística
por $z = (x - \mu_{c,l,w}) / \sigma_{c,l,w}$ dentro de campeón, línea y resultado. `huella_mejoras.py` estimaba
$\mu$ y $\sigma$ con todas las filas y después partía a los jugadores en ajuste y evaluación: las partidas de los
jugadores evaluados entraban en la normalización. Es una fuga leve, porque cada grupo tiene cientos de partidas de
muchos jugadores, pero rompe la regla de §10.4 de que la evaluación "nunca se vio".

*Cambio.* `residualize(df, ajuste=pids)` estima $\mu$, $\sigma$ y el tamaño del grupo ($\ge 30$) solo con las
filas de `pids` y los aplica a todas. Sin `ajuste`, se comporta como antes (lo usan los análisis descriptivos).
`huella_mejoras.py` parte la población **antes** de residualizar, sobre los jugadores crudos, y pasa la misma
partición a las partes 1, 3 y 4. Antes se re-partía dentro de cada parte, sobre los jugadores que sobrevivían al
residuo. `--residuo global` reproduce la versión anterior con la misma partición, para que la comparación sea
pareada. La parte 2 (hábitos de los timelines) recibió el mismo cambio en su normalización.

| Campeones distintos, jugadores no vistos (k = 10 / 20) | Residuo con todos | Residuo solo con ajuste | Publicado en §10.4 |
| :--- | :---: | :---: | :---: |
| Euclídea | 0,730 / 0,816 | 0,735 / 0,819 | 0,729 / 0,815 |
| Mahalanobis | 0,775 / 0,865 | 0,772 / 0,866 | 0,773 / 0,865 |
| LDA | 0,817 / 0,886 | 0,816 / 0,885 | 0,816 / 0,883 |
| Red contrastiva | 0,835 / 0,903 | 0,831 / 0,904 | 0,833 / 0,901 |
| Recencia (k = 10): todas iguales / vida media 20 | 0,869 / 0,873 | 0,872 / 0,876 | 0,869 / 0,873 |

Filas residualizadas: 1.225.439 con todos y 1.217.330 solo con ajuste (−0,7 %: grupos que con la mitad de los
jugadores no llegan a 30 partidas); 8.997 jugadores en los dos casos. Todas las diferencias son de ±0,005 o menos,
dentro de los IC (±0,006-0,008), y en las dos direcciones. **La fuga no inflaba la huella**, y las conclusiones de
§10.4 siguen en pie. El AUC difícil se mueve igual. Salidas: `data/analysis/huella/mejoras_parte134.csv` y
`mejoras_parte134_residuo_global.csv`.

**4. Huella con hábitos (§10.4 parte 2, `--parte red_habitos`, §10.5): la misma corrección.** El residuo de las
partidas con timeline (23 estadísticas + 18 hábitos) se calculaba aparte, en dos lugares: dentro de la parte 2 y en
`huella_niveles.datos_timeline()`, que alimenta la red contrastiva de `--parte red_habitos` y de los niveles 2 y 3.
`datos_timeline()` normalizaba con todos los jugadores y recién después partía.

*Cambio.* Hay una sola función de residuo para los timelines, `huella_mejoras.residuo_habitos(d, feats, ajuste)`.
Hace lo mismo que `residualize(ajuste=…)`: $\mu$, $\sigma$ y el tamaño mínimo del grupo ($\ge 30$) salen solo de
los jugadores de ajuste. Conserva una diferencia de antes: un hábito constante en el grupo vale 0 y no se descarta la
fila. `cargar_timeline(residuo)` parte a los jugadores crudos **antes** de residualizar, igual que `cargar_resid`.
`datos_timeline(residuo="ajuste")` devuelve ahora `(resid, ajuste, evalu)` y lo usan la parte 2, `red_habitos` y
`huella_niveles.py`, que también recibe `--residuo`. `--residuo global` reproduce la versión anterior con la misma
partición; sus salidas llevan el sufijo `_residuo_global`. Hay dos análisis 2-fold, `huella_patrones_cuenta.py`
(§10.7) y `historial/clustering_huella/boosting_en_victoria.py`, donde cada mitad hace de ajuste una vez. Esos dos
piden explícitamente el residuo global (`resid_timeline.pkl`, como antes), porque un ajuste con una sola mitad
dejaría la fuga en el otro pliegue. Desde el punto 5 usan un residuo por pliegue. De paso se corrigió `datos_train()` de
`huella_niveles.py`: desde §9.S.77 recibía la tupla de `cargar_resid()` y la guardaba como si fuera un DataFrame.
Test: `tests/test_huella_timeline.py`. Comprueba dos cosas. Si se cambian las partidas de un jugador de evaluación,
el residuo de los de ajuste no se mueve. Con `global`, ese mismo cambio sí se filtra. Con el código anterior, el test
falla.

*Datos de esta corrida.* `timelines.db` del Drive (62.706 partidas; hábitos en 62.065), solo lectura, por
`--timelines` o la variable `HUELLA_TIMELINES`. Hay 491.127 filas residualizadas con todos y 483.798 solo con ajuste
(−1,5 %). Quedan 8.991 y 8.989 jugadores. §10.4 se había medido con una base de timelines más grande, en la PC de
escritorio: 100.049 partidas y 803.480 filas. Por eso la comparación que vale es la pareada (misma base, misma
partición). La diferencia con lo publicado mezcla las dos cosas.

| Campeones distintos, jugadores no vistos, AUC [IC 95 %] | Residuo con todos | Residuo solo con ajuste | Δ | Publicado (§10.4-10.6) |
| :--- | :---: | :---: | :---: | :---: |
| 23 estadísticas · LDA, k = 10 | 0,830 [0,824; 0,837] | 0,826 [0,819; 0,833] | −0,004 | 0,827 |
| 23 estadísticas · LDA, k = 20 | 0,893 [0,886; 0,900] | 0,886 [0,879; 0,893] | −0,007 | 0,891 |
| Solo hábitos · LDA, k = 10 | 0,812 [0,805; 0,819] | 0,805 [0,798; 0,812] | −0,007 | 0,806 |
| Solo hábitos · LDA, k = 20 | 0,860 [0,852; 0,867] | 0,864 [0,856; 0,872] | +0,004 | 0,862 |
| Estadísticas + hábitos · euclídea, k = 10 | 0,795 [0,787; 0,803] | 0,789 [0,780; 0,798] | −0,006 | — |
| Estadísticas + hábitos · Mahalanobis, k = 10 | 0,831 [0,824; 0,838] | 0,820 [0,811; 0,828] | −0,011 | — |
| **Estadísticas + hábitos · LDA, k = 10** | 0,874 [0,867; 0,880] | **0,867** [0,860; 0,873] | −0,007 | 0,866 |
| **Estadísticas + hábitos · LDA, k = 20** | 0,921 [0,914; 0,927] | **0,920** [0,914; 0,927] | −0,001 | 0,921 |
| Estadísticas + hábitos · red, k = 10 | 0,897 [0,892; 0,903] | **0,892** [0,887; 0,898] | −0,005 | 0,890 |
| Estadísticas + hábitos · red, k = 20 | 0,945 [0,940; 0,950] | **0,943** [0,938; 0,948] | −0,002 | 0,941 |
| Estadísticas + hábitos · red, k = 30 / 40 / 50 | 0,958 / 0,970 / 0,973 | 0,959 / 0,963 / 0,975 | +0,001 / −0,007 / +0,002 | 0,959 / 0,966 / 0,971 |

En el AUC difícil (mismo tier y misma línea), los cambios van en la misma dirección y con el mismo tamaño. Por
ejemplo, LDA con estadísticas + hábitos pasa de 0,870 a 0,864 con k = 10, y la red de 0,940 a 0,937 con k = 20. El
top-1 de la red con k = 20 es 10,0 % con todos y 9,3 % solo con ajuste, contra 8,7 % publicado. Los ICC de los
hábitos casi no se mueven: control wards antes del 15, 0,434 → 0,431; wards antes del 10, 0,232 → 0,233.

*Lectura.* Acá, a diferencia de las partes 1, 3 y 4, casi todas las diferencias son negativas. La fuga inflaba
apenas la huella con hábitos, entre 0,001 y 0,007 en las filas principales. El caso más grande es Mahalanobis con
k = 10: −0,011, en el borde del IC. Tiene sentido: con la mitad de las filas (~480.000 contra 1,2 millones), cada
grupo campeón-línea-resultado tiene menos partidas y la media de ajuste es más ruidosa. Todas las diferencias de las
filas principales caen dentro de los IC (±0,005-0,007). Los números corregidos quedan a ±0,002 de los publicados. La
ganancia de los hábitos sobre LDA con 23 estadísticas no cambia: +0,041 / +0,034 corregido, contra +0,039 / +0,030
en §10.4. Las conclusiones de §10.4-10.6 siguen en pie. La red con k = 50 sigue siendo la mejor huella: 0,975.
Salidas: `data/analysis/huella/mejoras_parte2.csv`, `mejoras_red_habitos.csv`, `icc_habitos.json` y sus versiones
`_residuo_global`. Tiempos: parte 2 global, 9 min la primera vez (lee 62.706 timelines del Drive y arma
`habitos.pkl` y `timeline_train.pkl`); con caché, 1,5 min. `red_habitos`, 2,5-3 min cada una.

*Niveles 2 y 3 (2026-11-23).* `huella_niveles.py` corrido con el residuo de ajuste sobre la base de timelines del
Drive (62.706 partidas; 483.798 filas, 8.989 jugadores; §10.5 se midió con 100.049 en otra máquina), ~8 min. Nivel 2,
estadísticas + hábitos, ROC-AUC con todos los impostores para k = 5 / 10 / 20 / 40:

| Método | Publicado (§10.5) | Residuo de ajuste |
| :--- | :---: | :---: |
| LDA | 0,764 / 0,834 / 0,892 / 0,925 | 0,762 / 0,832 / 0,886 / 0,915 |
| Red contrastiva | 0,808 / 0,892 / 0,941 / 0,963 | 0,815 / 0,895 / 0,942 / 0,960 |

Con impostores de mismo tier y línea cae entre 0,003 y 0,008, como antes. Nivel 3 (1.448 cuentas, falsas alarmas
1/100, red contrastiva): el boosting se detecta en el 47 % antes de 5 partidas, 66-70 % antes de 10 y 78-80 % antes
de 20 (publicado: 81 % antes de 20); la cuenta compartida, 11 % / 21-29 % / 32-43 %. Las diferencias son del orden de
0,01 y en los dos sentidos, mezcladas con el cambio de base: **las conclusiones de §10.5 no cambian**. Sin
bootstrap: los n son los de la tabla (3.844 / 3.034 / 1.885 / 696 pares).

**5. Patrones intra-cuenta (§10.7) y boosting en victoria (§10.8): residuo por pliegue.** Estos dos análisis son
2-fold: la red contrastiva entrenada con una mitad de los jugadores proyecta a la otra, y al revés. Un solo
residuo ajustado con una mitad (el del punto 4) dejaría la fuga en el pliegue que proyecta a esa mitad, así que
seguían con el residuo global: los jugadores analizados entraban en $\mu_{c,l,w}$ y $\sigma_{c,l,w}$.

*Cambio.* Cada pliegue tiene su residuo. Si el pliegue $k$ entrena su red con la mitad $J_k$ y proyecta a la mitad
$\bar J_k$, sus filas son
$$z^{(k)} = \frac{x - \mu^{(k)}_{c,l,w}}{\sigma^{(k)}_{c,l,w}}, \qquad \mu^{(k)}, \sigma^{(k)} \text{ y } n^{(k)} \ge 30
\text{ estimados solo con las filas de } J_k,$$
con `hm.residuo_habitos(d, feats, ajuste=J_k)`, el mismo residuo del punto 4 (en los timelines no se usa
`residualize()`, que descarta la fila de un hábito constante en el grupo). La red $k$ se entrena con $z^{(k)}$ de
$J_k$, y cada jugador de $\bar J_k$ se analiza con sus filas $z^{(k)}$. Ni la red ni la normalización vieron a
quien se analiza. `huella_patrones_cuenta.datos_2fold(residuo)` arma los dos pliegues y la tabla que se analiza;
`redes_2fold(residuo)` entrena o carga las dos redes y devuelve, por jugador, la red que no lo vio.
`boosting_en_victoria.py` usa las mismas funciones. La partición es la de `cargar_timeline` (jugadores crudos).
`--residuo global` reproduce la versión anterior con esa misma partición; sus salidas llevan el sufijo
`_residuo_global` (también la caché de redes, que ahora guarda una firma de los datos y se reentrena si no coincide).
`--semilla N` cambia la semilla de las dos redes, para medir el ruido del entrenamiento. Test:
`tests/test_huella_patrones_pliegue.py`. Si se cambian las partidas de un jugador que un pliegue proyecta, no se
mueven ni las filas con que se entrena esa red ni las de los demás jugadores que proyecta. Con `global` ese mismo
test falla, y el test comprueba que falle.

*Medición.* Misma base que el punto 4 (`timelines.db` del Drive, copia local; 62.065 partidas con hábitos) y misma
partición. Con el residuo global quedan 491.127 filas y 8.991 jugadores; por pliegue, 483.798 y 484.658 filas por
pliegue, y se analizan 482.859 filas de 8.989 jugadores. Cada corrida tarda ~4 min (entrena las dos redes en
~2,5 min, CPU). Para separar el efecto del residuo del ruido de reentrenar las redes, se corrió además el global con
otra semilla (`--semilla 10`). IC 95 %: Wilson para el % de cuentas, bootstrap por cuenta para las medias de cada
cuadrante, bootstrap pareado por jugador para el cociente y z de Fisher para las correlaciones. Todos los roles:

| Patrones intra-cuenta (§10.7), todos los roles | Residuo global | Residuo por pliegue | Global, otra semilla |
| :--- | :---: | :---: | :---: |
| Cuentas con ≥ 50 partidas | 3.684 | 3.628 | 3.684 |
| Ortogonalidad: distancia victorias–derrotas / aleatoria | 1,033 [1,027; 1,039] | 1,045 [1,038; 1,051] | 1,033 |
| Racha buena (misma huella), % | 3,07 [2,56; 3,67] (113) | 3,14 [2,62; 3,76] (114) | 3,07 (113) |
| … $\Delta$WR / cambio horario / contigüidad | +26,5 pp / 0,24 / 0,50 | +26,4 pp / 0,22 / 0,52 | +25,9 pp / 0,22 / 0,54 |
| Candidata a compartida, % | 1,14 [0,84; 1,54] (42) | 0,88 [0,63; 1,24] (32) | 1,33 (49) |
| … $\Delta$WR / cambio horario / contigüidad | +0,5 pp / 0,29 / 0,72 [0,63; 0,81] | −0,3 pp / 0,27 / 0,68 [0,57; 0,78] | +0,5 pp / 0,27 / 0,68 |
| Candidata a boosting | 3 (0,08 %) | 3 (0,08 %) | 2 |
| Normal: cambio horario / contigüidad | 0,22 / 0,51 | 0,22 / 0,50 | 0,22 / 0,51 |
| $r$(`d_estilo`, $\Delta$WR) | −0,012 [−0,044; +0,021] | +0,011 [−0,022; +0,043] | — |
| $r$(`d_estilo`, cambio horario) | 0,139 [0,107; 0,170] | 0,119 [0,087; 0,151] | — |

En el rol principal (2.271 y 2.228 cuentas): rachas 2,47 % [1,90; 3,19] → 3,46 % [2,77; 4,30], compartidas
1,10 % [0,75; 1,62] → 1,08 % [0,72; 1,60], ninguna candidata a boosting en los dos casos; $r$(`d_estilo`,
$\Delta$WR) −0,012 → −0,001 y $r$(`d_estilo`, horario) 0,099 [0,059; 0,140] → 0,092 [0,051; 0,133].

*Lectura.* Todas las cifras agregadas caen dentro de los IC de la otra versión. La única que se mueve algo más que
su IC es el cociente de ortogonalidad (1,033 → 1,045), que sigue a 4,5 % de una partición aleatoria: ganar o perder
casi no desplaza la huella, como antes. Lo que no se sostiene es la lista de cuentas. Entre global y por pliegue,
el 94,8 % de las cuentas cae en la misma categoría, y las compartidas coinciden con Jaccard 0,38 y las rachas con
0,18. Pero reentrenar el global con otra semilla da casi lo mismo: 95,2 %, Jaccard 0,42 y 0,23. **El cambio que
introduce el residuo por pliegue no se distingue del ruido de entrenar la red de nuevo.** El hallazgo de fondo es
otro: qué cuentas puntuales quedan como candidatas depende de la semilla. Lo estable son las proporciones y los
perfiles de cada cuadrante (las compartidas más contiguas y con más cambio horario que las normales; las rachas, con
el mismo horario y la misma contigüidad que las normales). Por eso §10.7 debe leerse como un filtro de
priorización por cuadrante, no como una lista fija de cuentas. Una lista estable pediría promediar varias semillas
o exigir que la cuenta salga en la mayoría.

*Salidas.* `data/analysis/huella/patrones_*` (por pliegue), `patrones_*_residuo_global*` y los de la otra semilla
(`_residuo_global_semilla10`), más `patrones_resumen*.json` con los IC. `boosting_en_victoria.py` solo se adaptó y se
verificó que siga importable; no se volvió a medir (§10.8 sigue con el residuo global y la base de escritorio).

*Alternativas descartadas.* Mantener el residuo global y declararlo: los números no cambian, pero la regla de que la
evaluación "nunca se vio" se rompería justo en el análisis que se presenta como fuera de muestra. Un residuo con la
mitad de ajuste para los dos pliegues: el pliegue que proyecta a esa mitad la normalizaría con sus propios datos.
Validación cruzada con más pliegues: cada red vería más jugadores, pero se perdería la equivalencia con §10.7 (dos
redes) y el costo crece con el número de pliegues sin cambiar la pregunta.

*Alternativas descartadas.* Partir sobre los jugadores que sobreviven al residuo, como antes: el residuo
dependería de la partición y la partición del residuo, y ya no se comparan las dos versiones con los mismos
jugadores. Normalizar con la media y el desvío globales del campeón, sin condicionar por línea y resultado: cambia
la representación (§10.0 justifica el condicionamiento) y no corrige la fuga.

*Entorno de la corrida.* El `.conda` de la notebook no tiene `pyarrow` (necesario para los parquet del matchmaking),
`umap-learn` ni `hdbscan`. Se instalaron en una carpeta aparte, con `pip install --target`, y se agregaron solo vía
`PYTHONPATH` para estas corridas. No se instalaron en el entorno compartido porque, con pandas 3, `pyarrow` cambia
el tipo por defecto de las columnas de texto para todos los scripts. Si se agregan a `requirements.txt`, conviene
verificar antes ese efecto.

#### S.78. El modelo en vivo que se presenta: logística en dos etapas en `src/ganador/` (2026-11-23, issue #3)

**Decisión del usuario (2026-11-23).** Se presenta como modelo en vivo principal la **logística en dos etapas**
(CPU; variante 5b de `ingame_g_03_regimen_y_senales.py`, §9.S.60 punto 3). El ensamble `6_Final_dEns10`
(§9.S.60 punto 5: dos logísticas, tres redes multitarea en GPU y la segunda etapa con $L_{\text{ens10}}$) queda
como **anexo**, con sus números de §9.S.60, sin rehacerlo. Motivos: la auditoría del 22/11 encontró que el
ensamble le gana a la logística por ~+0,002 de AUC y +0,1 a +0,3 puntos de acierto; para eso necesita GPU,
~54 min, el previo del escalón 10 (red + CatBoost, `pred_g28_esc9.npz`, alineado por posición) y cachés de
variantes descartadas (`gemini_hgb_ctx_p15`, `pred_ingame`), y se eligió entre más de 50 configuraciones
mirando V. La logística se explica en dos fórmulas y corre en CPU.

**Qué hay ahora.**

| Pieza | Qué hace |
| :--- | :--- |
| `src/ganador/modelo_en_vivo.py` | Tramos, pliegues hacia adelante, previo hacia adelante, estado del timeline, expectativas por línea, las dos etapas y las métricas con IC |
| `scripts/analysis/modelo_en_vivo.py` | Un solo comando: `python scripts/analysis/modelo_en_vivo.py [--rehacer]` |
| `scripts/analysis/modelo_en_vivo_eventos.py` | Candidato: eventos del timeline encima (abajo; no promovido) |
| `tests/test_ganador_modelo_en_vivo.py` | Funcional con datos sintéticos, no-fuga de las variables del minuto y del resultado de V, pliegues causales, `mids` |

No depende de `ingame_cache.npz`, `pred_g28_esc9.npz`, `L_esc9_92k.npz`, `gemini_*`, `pred_ingame.npz`,
`mmr_historico.npz` (que sigue con la fuga de §9.S.75) ni de `brainstorm_base.cargar` (y con él de `lp_vivo.db`):
todo sale de `src.ganador.variables.calcular_todas()` y del CSV del train. Tampoco necesita `lightgbm` (abajo).

**El modelo.** Tramos cronológicos: lo anterior a A2 partido en 4 por fecha (tramos 0-3), A2-A4 (4-6), V Master+
(7) y Diamond de la ventana de V (8). Para el minuto $m \in \{10, 15, 20\}$, solo partidas vivas en $m$
(`game_duration > 60 m`, §9.S.57) con el timeline completo hasta $m$:

1. *Previo hacia adelante.* $L$ = logit del escalón 9 de `src.ganador.modelo_previo` (corregido en §9.S.75),
   ajustado para cada tramo $b = 1..6$ con las Master+ de los tramos anteriores y, para V y Diamond-V, con las
   118.064 Master+ anteriores al 16/09: en V es exactamente el previo de 0,6727 (verificado contra
   `pred_v.npz` de `modelo_previo.py`: $\max|\Delta p| = 1{,}1 \cdot 10^{-16}$).
2. *Estado al minuto $m$* (bloque A2 de §9.S.60): las 8 estadísticas por línea (oro, XP, nivel, CS, daño, kills,
   muertes, asistencias), oro residual (lo que no explican CS, kills y asistencias, OLS por línea ajustado hacia
   adelante), umbrales de nivel, agregados y extremos por equipo, trayectorias 10 → 15 → 20 y la *sorpresa*
   $\Delta x_{\ell} - \widehat{\Delta x}_{\ell}$ contra lo esperado antes de jugar en cada línea $\ell$ (Ridge sobre
   draft + composición; para oro y XP, más un árbol de gradiente espejado sobre el residuo con jugadores, campeones
   y duelo esperado; y la transición del minuto anterior). Todo azul − rojo.
3. *Etapa 1* (las ~92k Master+ de los tramos 1-6; 94.359 / 93.823 / 86.571 a los 10 / 15 / 20):
   $$z = \beta_0 + \beta_L L + \boldsymbol\beta^\top \mathbf x_m, \qquad \hat P_1 = \sigma(z),$$
   logística L2 con $C \in \{0{,}01; 0{,}03; 0{,}05\}$ elegido por log loss en A2-A4 hacia adelante (quedó
   0,01 / 0,03 / 0,03).
4. *Etapa 2* (solo partidas posteriores a la foto, hacia adelante: A2 neutra, A3 ← A2, A4 ← A2-A3, V ← A2-A4):
   $$\operatorname{logit} P = a + b\,z + c\,L, \qquad C = 0{,}5 .$$
   En la etapa 1, $\beta_L$ lo fijan sobre todo las ~78k partidas anteriores a la foto, donde el escalón 9 no
   tiene sus columnas de foto (W/L de temporada, nivel histórico) y es más débil; $c$ le devuelve a $L$ el peso
   que tiene en el régimen de V. Ajustada con A2-A4: $(a, b, c)$ = (0,040; 0,966; 0,109) al 10, (0,042; 0,967;
   0,093) al 15 y (0,026; 0,967; 0,078) al 20.

*Por qué $c\,L$ y no $c\,(L_{\text{ens10}} - L_{\text{orig}})$ como en 5b.* En 5b la etapa 1 usaba un previo
simplificado ($L_{\text{orig}}$: el de `ingame_preparar.py`, con el mapa de LP y sin el escalón 9 entero) y la
etapa 2 sumaba la diferencia con el previo mejor. Acá la etapa 1 ya usa el escalón 9, así que $L_{\text{orig}}$
sobra y la diferencia de previos se reduce a $c\,L$: misma información, una pieza menos (y sin `lp_vivo.db`).
$L_{\text{ens10}}$ (0,50 logística + 0,25 red NeuralAC + 0,25 CatBoost, §9.S.60 punto 2) no se puede
reproducir desde `src.ganador` sin rehacer la red y el CatBoost del escalón 10 en GPU; como previo solo le gana
al escalón 9 por +0,0010 de AUC (0,6749 contra 0,6739, ambos con la fuga de §9.S.75). Se reemplazó por el escalón
9. La comparación directa contra 5b no se puede rehacer en esta máquina (no están `pred_g28_esc9.npz` ni
`ingame_cache.npz`); contra sus números publicados, la versión nueva queda −0,0006 / −0,0005 / −0,0005 en V
(tabla de abajo), que es la suma de tres cambios: escalón 9 en vez del ensamble del escalón 10 (−0,0010 en el
previo), la corrección de la fuga del nivel histórico (§9.S.75: −0,0013 en el previo) y el árbol de las
expectativas (abajo). Se prefirió lo simple.

*`lightgbm` fuera del camino.* Las expectativas por línea usaban `lightgbm.LGBMRegressor`, que no está en el
entorno de la notebook. Se cambió por `sklearn.ensemble.HistGradientBoostingRegressor` con los mismos
hiperparámetros (90 árboles, 31 hojas, tasa 0,06, 63 bins, 150 partidas por hoja, 80 % de las columnas por
división; sin el submuestreo de filas del 80 %, que HGB no tiene). Es el mismo algoritmo (histogramas y
crecimiento por hojas); su efecto sobre el AUC final no se puede aislar sin `lightgbm`, y queda dentro del
−0,0005 de arriba.

**Resultados** (`python scripts/analysis/modelo_en_vivo.py`; `data/analysis/en_vivo/metricas.csv`; IC 95 % por
bootstrap de partidas, 1.000 réplicas). Nada se eligió mirando V: el único hiperparámetro elegido ($C$ de la
etapa 1) sale de A2-A4, y V se miró una vez.

*Misma cohorte en los tres minutos* (las 14.971 partidas de V Master+ vivas a los 20; "siempre rojo" acierta el
52,3 %):

| Minuto | AUC | Acierto | Log loss | ECE | Ensamble del anexo (AUC) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Solo el previo | 0,6610 [0,6522; 0,6695] | 61,6 % [60,8; 62,4] | 0,6517 | 0,0154 | — |
| 10 | **0,7628** [0,7554; 0,7705] | **69,5 %** [68,8; 70,2] | 0,5789 [0,5713; 0,5861] | 0,0124 | 0,7652 |
| 15 | **0,8172** [0,8105; 0,8238] | **74,0 %** [73,3; 74,7] | 0,5217 [0,5135; 0,5302] | 0,0161 | 0,8202 |
| 20 | **0,8727** [0,8668; 0,8782] | **78,8 %** [78,2; 79,5] | 0,4438 [0,4352; 0,4531] | 0,0078 | 0,8750 |

*Cada minuto con todas las partidas vivas en él* (para comparar con §9.S.60, que reporta así):

| Minuto ($n_V$ / $n_{DV}$) | V Master+: AUC | Acierto (siempre rojo) | Log loss | ECE | Diamond-V: AUC / acierto | 5b (§9.S.60) | Ensamble |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 10 (16.331 / 10.422) | **0,7888** [0,7818; 0,7957] | **71,5 %** [70,8; 72,2] (52,2 %) | 0,5524 | 0,0081 | 0,7781 / 70,3 % | 0,7894 / 71,4 % | 0,7912 / 71,6 % |
| 15 (16.238 / 10.369) | **0,8414** [0,8352; 0,8470] | **76,0 %** [75,3; 76,6] (52,2 %) | 0,4893 | 0,0107 | 0,8395 / 75,4 % | 0,8419 / 76,0 % | 0,8441 / 76,1 % |
| 20 (14.971 / 9.691) | **0,8727** [0,8668; 0,8782] | **78,8 %** [78,2; 79,5] (52,3 %) | 0,4438 | 0,0078 | 0,8727 / 78,7 % | 0,8732 / 78,8 % | 0,8750 / 79,1 % |

La brecha con el ensamble es −0,0024 / −0,0030 / −0,0023 de AUC en la misma cohorte (−0,0024 / −0,0027 /
−0,0023 minuto a minuto) y −0,1 a −0,3 puntos de acierto: lo que costaban las redes en GPU. Contra "siempre
rojo", el modelo acierta +17,2 / +21,8 / +26,6 puntos en la cohorte.

*Qué aporta la segunda etapa* (Δ AUC dos etapas − etapa 1 sola; placebo = $L$ permutada dentro de cada conjunto
en la etapa 2):

| Minuto | Bloques A3-A4 (media, IC) | V | Diamond-V | Placebo en V |
| :---: | :---: | :---: | :---: | :---: |
| 10 | +0,0002 [−0,0005; +0,0009] | +0,0005 [+0,0001; +0,0009] | −0,0002 [−0,0007; +0,0004] | −0,0002 |
| 15 | −0,0000 [−0,0006; +0,0005] | +0,0003 [+0,0001; +0,0006] | −0,0002 [−0,0005; +0,0002] | −0,0000 |
| 20 | −0,0003 [−0,0009; +0,0002] | +0,0001 [−0,0001; +0,0003] | −0,0001 [−0,0003; +0,0001] | +0,0000 |

Con el escalón 9 ya dentro de la etapa 1, la segunda etapa casi no suma: en 5b daba +0,0016 / +0,0007 / +0,0003
porque corregía un previo de etapa 1 más débil ($L_{\text{orig}}$, 0,6677 en V). Hoy no pasa la regla de
aceptación (los bloques cruzan el cero y Diamond-V no replica). Se mantiene porque es la estructura que decidió
el usuario, se ajusta sin V, su placebo es 0 y no empeora nada (en V el log loss baja apenas en los tres
minutos). La alternativa más simple, **la etapa 1 sola**, da lo mismo dentro del IC (0,7619 / 0,8168 / 0,8726
en la cohorte); queda a decisión del usuario.

**Decisión (2026-11-23): el modelo en vivo que se presenta es la etapa 1 sola.** La segunda etapa es una
variante y, como cualquier variante del proyecto, entra solo si pasa la regla de aceptación de §9.S.47 y §9.S.52:
mejora en la validación temporal por bloques (A3-A4 hacia adelante; A2 es neutra por construcción) con IC > 0,
IC > 0 en V mirada una vez, y réplica en Diamond-V, que nunca se usó para decidir. Con la tabla de arriba:

| Condición | 10 | 15 | 20 | ¿Pasa? |
| :--- | :---: | :---: | :---: | :---: |
| Bloques A3-A4, media con IC | +0,0002 [−0,0005; +0,0009] | −0,0000 [−0,0006; +0,0005] | −0,0003 [−0,0009; +0,0002] | No (IC cruza 0 en los tres) |
| V Master+ | +0,0005 [+0,0001; +0,0009] | +0,0003 [+0,0001; +0,0006] | +0,0001 [−0,0001; +0,0003] | Solo a los 10 y 15 |
| Diamond-V | −0,0002 [−0,0007; +0,0004] | −0,0002 [−0,0005; +0,0002] | −0,0001 [−0,0003; +0,0001] | No (signo opuesto) |

Falla al menos dos de las tres condiciones en cada minuto, y lo único positivo (V) es del orden de lo que la regla ya
vio aparecer con columnas al azar (§9.S.52: el placebo de `ROL_WR` mejoró 3 de 3 bloques). Por la regla, la carga
de la prueba la tiene la pieza que se agrega: se queda la etapa 1, que además se explica con una sola fórmula,
$\hat P = \sigma(\beta_0 + \beta_L L + \boldsymbol\beta^\top \mathbf x_m)$. El argumento de la etapa 2 (devolverle a
$L$ el peso que tiene después de la foto) era válido con el previo débil de 5b ($L_{\text{orig}}$, 0,6677 en V,
donde sumaba +0,0016 / +0,0007 / +0,0003); con el escalón 9 dentro de la etapa 1 ya no queda qué corregir.

**Números del modelo presentado** (de `data/analysis/en_vivo/metricas.csv`; la etapa 1 ya se calculaba y reportaba
en la corrida de arriba, y la corrida con el código nuevo (13,8 min con la caché) dio los mismos valores con la
columna `rol`; IC 95 % por bootstrap, 1.000 réplicas):

| Minuto | Misma cohorte (14.971): AUC | Acierto | Log loss | ECE | Cada minuto en su cohorte: AUC / acierto | Diamond-V: AUC / acierto |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 10 | **0,7619** [0,7546; 0,7696] | **69,3 %** [68,6; 70,1] | 0,5797 | 0,0141 | 0,7883 [0,7814; 0,7952] / 71,4 % | 0,7782 / 70,4 % |
| 15 | **0,8168** [0,8099; 0,8233] | **74,1 %** [73,4; 74,8] | 0,5224 | 0,0150 | 0,8411 [0,8350; 0,8467] / 76,1 % | 0,8397 / 75,5 % |
| 20 | **0,8726** [0,8666; 0,8782] | **78,7 %** [78,1; 79,4] | 0,4440 | 0,0101 | 0,8726 [0,8666; 0,8782] / 78,7 % | 0,8728 / 78,7 % |

Contra "siempre rojo" (52,3 % en la cohorte) acierta +17,1 / +21,9 / +26,5 puntos. Contra el ensamble del anexo,
−0,0033 / −0,0034 / −0,0024 de AUC en la misma cohorte (−0,0029 / −0,0030 / −0,0024 minuto a minuto). Lo que se
resigna al sacar la etapa 2 es una baja de log loss de 0,0008 / 0,0007 / 0,0002 en la cohorte, dentro del IC.

**Qué cambió en el código.** `src/ganador/modelo_en_vivo.py` declara `MODELO_PRINCIPAL = "etapa1"` y
`ajustar_predecir` devuelve, además de `z1_*` y `z2_*`, `z_fin` y `z_a24` = las de la etapa 1.
`scripts/analysis/modelo_en_vivo.py` marca cada fila de `metricas.csv` con `rol` (`principal`, `descartada` o
`referencia` para el previo solo) y guarda en `pred_en_vivo.npz` la etapa 1 como `p_{m}` / `p_a24_{m}` y la
variante descartada como `p_dos_etapas_{m}` / `p_a24_dos_etapas_{m}`, todas con `mids_{m}` / `mids_a24_{m}` y
`modelo_principal`. La etapa 2 se sigue ajustando y reportando (con su placebo) como comparación: no se borró.
`modelo_en_vivo_eventos.py` compara ahora "contra el modelo presentado" con la etapa 1; la línea de abajo
se volvió a medir contra la etapa 1 el 2026-11-23 (+0,0059 / +0,0059 / +0,0064). Test nuevo
en `tests/test_ganador_modelo_en_vivo.py`: el modelo principal es la etapa 1 y la etapa 2 se sigue calculando.

*Salvedad sobre el ECE.* El IC del ECE por bootstrap no contiene a veces al valor puntual (p. ej. 0,0081
[0,0075; 0,0174]): el ECE de una muestra remuestreada está sesgado hacia arriba (las repeticiones agregan ruido
en cada decil). El IC de AUC, acierto y log loss no tiene ese problema. Se informa el ECE puntual; su IC sirve
solo como orden de magnitud.

**Tiempos** (notebook, CPU): desde cero ~33 min (estado del timeline y expectativas ~20 min, de ellos ~16 el
árbol por línea; previo hacia adelante ~2,5 min; ajuste y bootstrap de los tres minutos ~9 min), más las
variables previas si no están (§9.S.73). Con las cachés vigentes, ~12 min (casi todo el bootstrap).

**Cachés con `mids` y huella.** `data/cache/ganador/en_vivo_estado.npz` (timeline, oro residual y expectativas;
su huella depende del código de `construir_estado` y de las funciones de `src/models/ingame.py` que usa, del CSV
del train, de los atributos de campeones, de `champions.csv` y de las huellas de `variables_partida` y
`margen_esperado_par`) y `en_vivo_previo.npz` (depende de las huellas de los 9 grupos de variables, así que la
corrección de §9.S.75 lo recalculó sola). Las dos guardan `mids` y se verifican al cargar
(`verificar_mids`): ninguna se alinea por posición. `pred_en_vivo.npz` guarda las predicciones de V, Diamond-V y
A2-A4 con sus `mids`.

**Tests** (`tests/test_ganador_modelo_en_vivo.py`): con insumos sintéticos donde el previo es débil antes de la
foto y fuerte después, la etapa 1 aprende (AUC > 0,7), la etapa 2 le da $c > 0$ y mejora, y en A2 queda neutra;
**no-fuga**: cambiar el timeline y el oro residual de los minutos posteriores a $m$ y el resultado no cambia las
variables del minuto $m$, y la de una partida no depende de las demás; invertir el resultado de V no cambia sus
predicciones; los pliegues entrenan solo con Master+ anterior y el oro residual de un tramo no cambia al
modificar los tramos posteriores; `verificar_mids` rechaza otro orden o la falta de `mids`.

**Corrección en el anexo.** §9.S.60 punto 5 decía que las redes del ensamble entrenaban "16 épocas fijas, sin
parada temprana". El código (`_fit_mt_gpu` de `ingame_g_04_mt_a4_gpu.py`) entrena hasta 16 épocas y se queda
con la de menor log loss sobre el 12 % más reciente de su propio entrenamiento: es parada temprana por selección
del mejor punto. Quedó corregido ahí. No cambia ningún número (V no entra en esa selección).

**Candidato: los eventos del timeline encima** (`modelo_en_vivo_eventos.py`; §9.S.64 y §9.S.66 daban +0,0055 /
+0,0060 / +0,0070 sobre una logística en vivo). Los eventos de prioridad 1 de `ingame_eventos.py` (dragones y punto de alma, larvas,
heraldo, torres por nivel, placas, inhibidores, oro sin gastar e invertido, muertos en el minuto y shutdowns; azul
− rojo, solo lo ocurrido antes del minuto) entran en la etapa 2, escalados por su RMS en el ajuste:
$$\operatorname{logit} P = a + b\,z + c\,L + \boldsymbol\gamma^\top \mathbf e_m .$$
Hay timeline con eventos para toda V Master+ y para ~la mitad de las Master+ del ajuste (62.706, de la copia del
18/11 de `timelines.db`, copiada fuera del repo a `C:\Users\gonza\datos_lol\timelines_eventos\`), así que se
compara contra la misma etapa 2 sin $\mathbf e$ ajustada con **las mismas** partidas de A2-A4 que tienen timeline
(~2.200 por bloque), hacia adelante. Regla de aceptación: media de bloques A3-A4 con IC > 0, V una vez, réplica en
Diamond-V. Se probó una sola variante (prioridad 1, la medida aceptada en §9.S.64).

| Minuto | Bloques A3, A4 (media, IC) | Δ AUC en V | V sin → con eventos (AUC / acierto) | Log loss | ECE | Placebo ($\mathbf e$ permutada) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 10 | +0,0032, +0,0035; **+0,0033 [+0,0003; +0,0065]** | **+0,0054 [+0,0040; +0,0070]** | 0,7887 / 71,5 % → 0,7942 / 72,0 % | 0,5524 → 0,5466 | 0,0077 → 0,0136 | −0,0005 |
| 15 | +0,0028, +0,0056; **+0,0042 [+0,0013; +0,0073]** | **+0,0055 [+0,0040; +0,0071]** | 0,8415 / 75,9 % → 0,8470 / 76,4 % | 0,4892 → 0,4819 | 0,0092 → 0,0075 | −0,0012 |
| 20 | +0,0011, +0,0078; **+0,0045 [+0,0014; +0,0075]** | **+0,0064 [+0,0047; +0,0079]** | 0,8727 / 78,9 % → 0,8791 / 79,5 % | 0,4438 → 0,4331 | 0,0080 → 0,0073 | −0,0005 |

Contra el modelo presentado (la etapa 1, rehecho el 2026-11-23 con `modelo_en_vivo_eventos.py`) la diferencia
en V es +0,0059 [+0,0043; +0,0075] / +0,0059 [+0,0043; +0,0074] / +0,0064 [+0,0048; +0,0080]; contra la etapa 2,
+0,0054 / +0,0055 / +0,0064. La descarga de Diamond-V para la réplica queda en el issue #13 (a cargo del compañero). Al 20 la V del minuto es la cohorte común, así que 0,8791 / 79,5 % le gana al ensamble del anexo
(0,8750 / 79,1 %). El dragón es el coeficiente más grande en los tres minutos (+0,22 / +0,30 / +0,34 en escala
RMS), como en §9.S.64. Pasa dos de las tres condiciones (bloques con IC > 0 y V con IC > 0; placebo ≤ 0); la
tercera no se puede medir: **Diamond-V no tiene timelines** (la descarga priorizó V y Master+ del ajuste). Al 10
el ECE empeora (0,0077 → 0,0136). **No se promueve**: queda para el usuario, con dos caminos para la réplica que
falta (bajar los timelines de Diamond-V, ~10.500 partidas, ~2 h con P0 + P1; o aceptar como réplica los bloques,
que acá no fueron conjunto de selección porque no se probó ninguna otra variante). Tiempo: ~4,5 min la lectura
de los timelines (4 procesos; después se reusa `data/analysis/en_vivo/eventos.npz`, con `mids`) y ~8 min el resto.

**Alternativas descartadas.**

* *Rehacer el ensamble en `src/`.* Es el que más acierta, pero necesita GPU, el previo del escalón 10 y ~54 min,
  por +0,002 de AUC. Queda como anexo con sus números de §9.S.60.
* *Mantener $L_{\text{orig}}$ en la etapa 1 (5b literal).* Obligaría a portar el previo simplificado con el mapa
  de LP (y `lp_vivo.db`) solo para restarlo en la etapa 2.
* *Instalar `lightgbm`.* Otra dependencia para un paso que `scikit-learn` hace con el mismo algoritmo.

#### S.79. Los `brainstorm_*` del escalón 9 y el anexo en vivo pasan a `historial/` (2026-11-23, issue #2)

**Qué se buscaba.** Cerrar el issue #2: con el modelo previo (§9.S.73) y el modelo en vivo (§9.S.78) en
`src/ganador/`, en `scripts/analysis/` tenía que quedar solo lo vigente. Quedaban 25 `brainstorm_*.py` y la cadena
del ensamble en vivo. §9.S.73 había postergado el movimiento ("todavía los importan scripts vigentes"); esta
sección lo hace y deja escrito qué dependencia quedó y por qué. No reescribe §9.S.72 ni §9.S.73: los actualiza.

**Cómo se decidió qué mover.** Se armó el grafo de imports reales (`ast`, incluidos los imports dentro de funciones
y los `from scripts.analysis import X`), más las lecturas de cachés (`*.npz`). La guarda de §9.S.72
(`tests/test_scripts_importables.py`) prohíbe que algo de `scripts/` importe de `historial/`, así que cada
consumidor vigente tenía tres salidas: (a) pasarlo a `src.ganador`, si el cambio da lo mismo y se puede comprobar;
(b) moverlo también, si es histórico; (c) dejar en `scripts/analysis/` lo que importa. Leer una caché que arma un
script de `historial/` no viola la guarda; es la única dependencia que queda hacia allá (abajo).

| Qué | Destino | Por qué |
| :--- | :--- | :--- |
| `brainstorm_05`, `07`, `08`, `g_01`, `g_03`, `g_06`, `g_07`, `g_09`, `g_10`, `g_11`, `g_23`, `brainstorm_final` | **Movidos** a `historial/escalon9/` | Arman el escalón 9; su código está copiado tal cual en `src/ganador/_constructores.py`. `brainstorm_final.py` queda como referencia histórica (`modelo_previo.py --comparar` lee su `pred_final.npz`) |
| `ingame_preparar`, `ingame_g_01`, `g_03`, `g_04`, `g_05` y `brainstorm_18`, `24`, `27`, `g_05`, `g_08`, `g_13`, `g_22`, `g_24`, `g_26`, `g_28` | **Movidos** a `historial/ensamble_en_vivo/` | Son el anexo `6_Final_dEns10` (§9.S.78) y las cachés que solo él lee. Ningún vigente los importa |
| `brainstorm_42_xp_lineas` | **Movido** a `historial/ideas_posteriores/` | Idea 42; su caché (`hgb_lineas_extra.npz`) la lee `umap_vistas` |
| `brainstorm_base.cargar_hgb_por_linea` (importaba `ATRIB_CSV` de `g_06`) | **Migrado** a `src.ganador.rutas.ATRIB_CSV` | Es la misma constante (`g_06` hacía `ATRIB_CSV = rutas.ATRIB_CSV`) |
| `brainstorm_base.extras_referencia` (importaba `construir_extras` de `g_01` para el W/L de la foto 1) | **Migrado** a `src.ganador._constructores.wl_foto1` | Copia textual de `construir_extras` hasta el W/L; solo cambia `D["B"]["player"]` por `D["player"]`, que se le pasa. Verificado con los datos: `FOTO1_DIN` idéntico bit a bit en las 187.817 partidas ($\max\lvert\Delta\rvert = 0$) |
| `tools/generar_constructores_ganador.py` | Lee los `brainstorm_*` desde `historial/escalon9/` | Regenerado, `_constructores.py` sale idéntico (solo difería el fin de línea) |
| `tests/test_preprocesamiento_causal.py` | Rutas a `historial/escalon9/`; el test del HGB parchea `src.ganador.rutas.ATRIB_CSV` | Mismas aserciones |
| `brainstorm_base.py` | **Queda** en `scripts/analysis/` | Lo importan 8 vigentes (matchmaking ×2, `modelos_train_v`, `acierto_por_lp_duracion`, `diamante_conocidos`, `huella_previo`, `umap_vistas`, `partidas_crudas`), `build_eda_modelos_nb.py` y el notebook `eda_modelos_ganador`, que no se toca (lo importa con ese nombre). Ya no importa nada de los demás `brainstorm_*` |
| `brainstorm_45_dificultad.py` | **Queda** | Análisis vigente (partidas difíciles, §9.S.61) y `umap_vistas` importa su `contexto` |
| `ingame_evaluar`, `ingame_eventos`, `eventos_ablacion`, `ingame_estado_puro` | **Quedan** | `modelo_en_vivo_eventos.py` (vigente) importa `ingame_eventos`, que importa `ingame_evaluar`; `ingame_estado_puro` da la barra de la sección 3 del notebook. Ninguno importa un `brainstorm_*`; leen `ingame_cache.npz` |
| 8 scripts de `historial/en_vivo/` y 3 de otras carpetas que importaban los movidos | Imports reescritos (`scripts.analysis.X` → `historial.<carpeta>.X`) | Siguen importables |

`historial/<carpeta>/x.py` tiene la misma profundidad que `scripts/analysis/x.py`, así que `parents[2]` sigue dando
la raíz (la guarda lo comprueba). Los mensajes con rutas ("correr antes: python …") se actualizaron.

**La dependencia que queda: cachés, no imports.** `brainstorm_base.extras_referencia` lee `margen_grupos`,
`mmr_historico`, `gemini_mpr`, `gemini_nuevas_v2`, `gemini_g11`, `gemini_hgb_lineas` y `gemini_g23` de
`data/analysis/brainstorm/`, que arman los scripts de `historial/escalon9/`; `umap_vistas` lee además
`pred_g28_esc9.npz` (anexo) y `hgb_lineas_extra.npz` (idea 42); `ingame_evaluar` y los suyos leen
`ingame_cache.npz` (anexo). Esto relaja la tercera condición de §9.S.72 ("escribe una caché que lee un script
vigente"): se aceptó porque los dos modelos que se presentan no dependen de ninguna de esas cachés (salen de
`src/ganador/` con cachés que se invalidan solas) y lo que sí depende son análisis complementarios (matchmaking,
validación, UMAP, el notebook) que se regeneran corriendo los scripts desde su nueva ubicación. Pasar esos
consumidores a `src.ganador` es el paso pendiente del issue #2 (tarea en `TAREAS.md`).

**Alternativas descartadas.**

* *Migrar ya todos los consumidores de `brainstorm_base` a `src.ganador`.* Es el destino final, pero no se puede
  comprobar "mismo resultado" ahora: `src.ganador` aplica la máscara del nivel histórico de §9.S.75 y
  `mmr_historico.npz` se está regenerando con esa corrección; hasta que termine, las dos fuentes difieren por
  construcción y una diferencia en matchmaking no distinguiría error de migración de corrección. Además
  `matchmaking_swaps` replica por dentro `g_01`, `g_11` y `g_23` para los reacomodos (§9.S.67) y el notebook
  importa `brainstorm_base`.
* *Dejar en `scripts/analysis/` un módulo que reexporte de `historial/`.* Viola la guarda y esconde la dependencia.
* *Renombrar `brainstorm_base.py` y `brainstorm_45_dificultad.py`.* El notebook (que no se toca) importa el primero
  por nombre, y esta memoria cita los dos en una docena de secciones.
* *Mover también `ingame_evaluar` y compañía.* Rompe `modelo_en_vivo_eventos.py`, que es vigente (candidato).

**Tests.** La suite pasa con 710 (708 y 2 omitidos sin `faiss` ni `lightgbm`): los mismos 262 casos de
`test_scripts_importables.py` (los scripts movidos siguen dentro de las carpetas que recorre) y uno nuevo del
modelo en vivo (§9.S.78, "Decisión").

#### S.80. La evaluación única en el test: protocolo, modo `test` y simulacro (2026-11-23, issue #5)

**Qué se buscaba.** Un único número final por modelo (previo y en vivo a 10/15/20) medido sobre el test, que nunca se
miró, con intervalos de confianza honestos, la degradación en el tiempo y la sensibilidad a la semilla. Se evalúa una
sola vez y queda probado en un log versionado (`docs/evaluacion_test/LOG_EVALUACION_TEST.md`).

**Protocolo (aprobado por el usuario, `docs/evaluacion_test/PROTOCOLO.md`).**

* *Universo:* train ∪ test ∪ Diamante posterior al corte; las partidas de Diamante solo alimentan historiales.
* *Ajuste final:* sobre ajuste ∪ V (todo Master+ anterior al 27/09); los hiperparámetros quedan congelados
  (escalón 9 con C = 0,05; etapa 1 con C = 0,01 / 0,03 / 0,03). Nada se elige mirando el test.
* *Master+:* la regla `split` (≥ 5 jugadores Master+ en la partida). En V, la regla 2×2 (ajuste × V) da mediana/mediana
  0,6727 y split/split 0,6732 / 62,6 %: la regla no mueve el número.
* *IC:* bootstrap por bloques de días, B = 1000. Las partidas del mismo día comparten parche, jugadores y horario, así
  que remuestrear partidas sueltas subestima la varianza; se remuestrean días $d_1^*,\dots,d_D^*$ con reposición y
  $\widehat{\mathrm{AUC}}^{*}=\mathrm{AUC}\big(\bigcup_j \mathcal{P}(d_j^*)\big)$, IC = percentiles 2,5 y 97,5.
* *Degradación:* ventanas cronológicas de 2.500 partidas y la pendiente de $\mathrm{AUC}_w \sim t_w$ (por semana), con su
  IC por bootstrap de ventanas.
* *Semillas:* principal 42 (fijada antes), más 0, 1, 2 y 3. La logística con lbfgs es determinista (verificado); la
  semilla solo mueve el HGB por línea (`oro_linea_hgb`), el HGB de las expectativas en vivo y el bootstrap.
* *Integridad:* SHA-256 de `match_split.csv` y del train congelados en el protocolo; el script se niega a correr si el
  log ya existe, si el árbol de git está sucio o si un hash no coincide, y los chequeos (entrenamiento anterior a la
  evaluación, todo Master+, sin NaN, previo alineado con el en vivo) corren antes de cualquier métrica; si uno falla
  escribe `LOG_..._FALLIDA` y no muestra números.

**Modo `test` en `src/ganador/`.** `rutas.configurar("v" | "simulacro" | "test")` cambia qué es ajuste y qué es
evaluación sin tocar el código de los modelos; las cachés llevan el modo en su hash. En modo V se reprodujeron el previo
y el en vivo bit a bit.

**Corte y reanudación.** La primera corrida del simulacro la mató la presión de memoria tras dos semillas sin escribir
nada. `evaluacion_test.py` ahora guarda las predicciones de cada semilla en `parciales/semilla_S.npz` y las retoma solo
si coinciden commit, modo y hash del protocolo; las métricas se calculan una vez, al final. Test:
`test_parciales_por_semilla_se_retoman_solo_con_la_misma_firma`.

**Simulacro (V hace de test; commit 311967c, 119 min).** Todos los chequeos pasan (99,4 % de partidas vivas a 10).
Semilla 42, IC 95 % por días (12 días):

| Modelo | n | AUC | Acierto (siempre rojo 52,2 %) | Log loss | ECE |
| :--- | ---: | :--- | :--- | :--- | :--- |
| Previo | 17.081 | 0,6731 [0,6640; 0,6820] | 62,5 % [61,8; 63,1] | 0,6449 | 0,010 |
| En vivo @10 | 16.984 | 0,7871 [0,7794; 0,7964] | 71,2 % [70,5; 72,1] | 0,5539 | 0,008 |
| En vivo @15 | 16.891 | 0,8402 [0,8334; 0,8475] | 75,9 % [75,0; 76,7] | 0,4909 | 0,010 |
| En vivo @20 | 15.580 | 0,8728 [0,8679; 0,8774] | 78,8 % [78,2; 79,3] | 0,4437 | 0,007 |

Misma cohorte (vivas a 20): 0,7609 / 0,8159 / 0,8728, contra 0,6614 del previo solo. El IC por días es algo más ancho
que el de partidas (previo: ±0,0090 contra ±0,0080), como se esperaba. Diamante (chequeo, fuera del modelo): previo
0,6405, en vivo 0,7793 / 0,8412 / 0,8725.

*Semillas:* el desvío del AUC entre las 5 semillas es ≤ 6,4·10⁻⁵ (previo) y ≤ 5,1·10⁻⁵ (en vivo), dos órdenes por
debajo del ancho del IC; la correlación mínima entre predicciones es 0,9996 (previo) y 0,9999 (en vivo), y el ganador
predicho coincide con la semilla 42 en ≥ 99,0 % de las partidas (diferencia máxima de probabilidad 0,023). La semilla no
es una fuente de incertidumbre relevante: se informa la 42 y el rango.

*Degradación:* pendientes por semana del AUC, previo −0,009 [−0,027; 0,009], @10 −0,002 [−0,017; 0,012], @15 −0,001
[−0,014; 0,012], @20 +0,008 [−0,004; 0,020]: ninguna distinta de cero en las ~2 semanas de V. El previo baja en la
ventana 4 (0,647) y se recupera (0,677); el en vivo es plano. El test cubre un período más largo y lo dirá con más fuerza.

**Alternativas descartadas.** Bootstrap por partidas como IC principal (ignora la correlación dentro del día; queda como
comparación); reajustar C en el ajuste final (sería elegir con más datos sin validación); promediar las 5 semillas como
número principal (cambia el modelo después de fijarlo; con este desvío daría lo mismo).

**Resultado de la corrida única (2026-11-25, commit `b1edaee`; log `docs/evaluacion_test/LOG_EVALUACION_TEST.md`).**
Todos los chequeos del protocolo en verde: cobertura del test 97,9 %, 99,4 % de las partidas con frame al minuto 10,
todo lo ajustado anterior al corte, los 4 hashes coinciden. Semilla 42, IC 95 % por bootstrap de días (30 días, B = 1000):

| Modelo | n | AUC | Acierto | Siempre rojo | Log loss | ECE |
| :--- | ---: | :--- | :--- | ---: | ---: | ---: |
| Previo | 37.196 | 0,6765 [0,6721; 0,6811] | 63,0 % [62,6; 63,4] | 52,5 % | 0,642 | 0,006 |
| En vivo @10 | 36.983 | 0,7916 [0,7880; 0,7953] | 71,6 % | 52,5 % | 0,549 | 0,004 |
| En vivo @15 | 36.798 | 0,8438 [0,8404; 0,8470] | 76,0 % | 52,5 % | 0,485 | 0,007 |
| En vivo @20 | 33.821 | 0,8732 [0,8694; 0,8768] | 78,7 % | 52,6 % | 0,442 | 0,007 |

*Contra V:* el previo da 0,6765 / 63,0 % contra 0,6727 / 62,4 % en V; el en vivo, en la cohorte del minuto 20 (la misma
definición que en V), 0,7659 / 0,8193 / 0,8732 contra 0,7619 / 0,8168 / 0,8726. El test no da peor que V: no hay
sobreajuste a V por las decisiones tomadas mirándola (el ajuste final incluye V, así que tiene ~11 días más de datos).
Por rango, el previo da 0,678 en Master (35.133), 0,645 en GM (1.724) y 0,696 en Challenger (339, IC ancho);
en Diamante (chequeo, fuera de la población objetivo) 0,647. *Degradación:* pendientes del AUC por semana +0,0000
(previo), −0,0018 / +0,0007 / +0,0028 (@10/15/20), todas con IC que contiene el 0 en las ~4 semanas del test.
*Semillas:* desvío del AUC ≤ 0,0001 entre las 5; ganador predicho idéntico en ≥ 99,2 % de las partidas.

*Cómo se corrió.* La corrida se cortó por memoria tras dos semillas y se completó con una semilla por proceso, con las
mismas guardas y la firma commit + modo + protocolo; lo único del test que se vio antes del final fue el diagnóstico
del HGB del oro por línea que el protocolo admite. Queda declarado en el log. El anexo `--sin-v` (modelo ajustado sin
V, sobre el test) está implementado pero **no se corrió**, por decisión del usuario (no volver a tocar el test).

**Merge de `gonchi` y la regla de Master+ (2026-11-25, decisión del usuario).** La rama del compañero (0ce856a,
ebda672) cambia `es_master_plus` de la mediana del tier a "≥ 5 jugadores del padrón Master o más", la misma regla que
`es_master_plus_split` y que el split. Se mergeó tal cual: es un cambio menor y deja una sola regla en el código. **No
se recalculó nada.** El test no cambia (ya usaba la regla del split). Los números de V del informe y de §9.S.52-78 se
calcularon con la mediana (V = 16.428 partidas; con la regla del split serían 17.081, §3 del protocolo) y quedan así.
Si algo se recalcula en modo `v`, V pasa a 17.081 y sus números se mueven poco. En los conflictos se priorizó `gonza`:
`main()` de `timelines_eventos.py` (con `--db` y el latido; se conservan `leer_ids` y `registrar_lote` del compañero, con
sus tests), la máscara de la foto 1 de `brainstorm_07` (los dos la habían agregado), y `build_eda_modelos_nb.py` y su
notebook enteros de `gonza` (ese notebook queda como está). En `huella_patrones_cuenta.py` se integró el FDR y la
categoría "Huella distinta + caída WR" del compañero dentro de `clasificar()`. Se aceptan los borrados del elo al revés
entre fotos (`elo_boostings`, `elo_gru`, `elo_val_lolpros`, `eda_entre_fotos`) y la quita del recomendador de
`draftrec_mejor.py` (no cambia el 59,27 %).

#### S.81. Notebook único de la entrega: primer borrador (2026-11-23, issue #6)

**Pedido del usuario.** Un solo notebook para la facultad, prolijo, con poco texto y poco código a la vista, gráficos
de buena calidad, algunas fórmulas que expliquen el porqué y el cómo, fondo claro, todo en castellano y sin jerga
interna (ni "escalón", ni "brainstorm", ni nombres de scripts o cachés). Este es el primer borrador, para iterar.

**Estructura.** `python scripts/analysis/build_informe_nb.py --ejecutar` arma y ejecuta `notebooks/informe_final.ipynb`
(~35 s; el patrón de los `build_eda_*_nb.py`). Todo lo que dibuja vive en `src/informe/`:

| Módulo | Qué hace |
| :--- | :--- |
| `estilo.py` | rcParams únicos, paleta, números con coma decimal (`num`, `pct`, `ic`, `entero`), `guardar` (PNG + cierre) |
| `datos.py` | Lee lo ya guardado (`variables_partida.npz`, `pred_v.npz`, `pred_en_vivo.npz` con `mids`, `en_vivo/metricas.csv`, huella, matchmaking, el manifiesto del test) y el bootstrap |
| `figuras.py` | Una función por figura o tabla; las celdas del notebook son una línea cada una, con la entrada oculta |

Siete secciones: (1) pregunta, datos y línea de tiempo de la partición; (2) modelo previo: fórmula, respuesta, tabla,
aporte por grupo de variables, separación por lado ganador, calibración y acierto según la confianza; (3) en vivo:
fórmula de la etapa 1, tabla, AUC y acierto por minuto en la misma cohorte, cuánto suma lo previo; (4) test y
(5) robustez con lugares que se llenan solos; (6) huella y matchmaking; (7) limitaciones. Las figuras se exportan a
`notebooks/figuras/informe/*.png` para revisarlas sin abrir el notebook.

**Qué no se lee.** De `data/splits/test/` solo el manifiesto (conteos). La sección 4 llama a
`datos.resultados_test()`: si no existe `data/analysis/test_final/` (lo escribirá `evaluacion_test.py`), muestra
"pendiente de la evaluación única"; si existe, toma `metricas*.csv` (sección 4) y `degradacion*.csv` /
`semillas*.csv` (sección 5) y los muestra como tabla. El formato definitivo de esos archivos queda para cuando exista
el script; hoy la lectura es genérica a propósito.

**IC en todo lo que se muestra.** Bootstrap de partidas con $B = 1000$ (500 en las curvas). El AUC de cada réplica se
calcula en $O(n)$ con pesos, sin reordenar: con las partidas ordenadas por $\hat p$ y $w_i$ las veces que la partida
$i$ entra en la réplica,
$$\text{AUC}_w = \frac{\sum_{i:\,y_i=1} w_i \left(\sum_{j:\,y_j=0,\ \hat p_j<\hat p_i} w_j + \tfrac12 \sum_{j:\,y_j=0,\ \hat p_j=\hat p_i} w_j\right)}{\big(\sum_{y_i=1} w_i\big)\big(\sum_{y_j=0} w_j\big)},$$
que coincide con `roc_auc_score` sobre el remuestreo explícito (test `tests/test_informe.py`, también con empates). La
calibración usa IC de Wilson por decil; la pendiente de degradación es la de MCO del AUC por ventana contra el día,
con IC por bootstrap dentro de cada ventana. El ECE se informa sin IC (su bootstrap está sesgado hacia arriba,
§9.S.78). Lo único sin IC son dos puntos de la figura "cuánto suma lo previo" (abajo).

**De dónde sale cada número.** Calculados al ejecutar: métricas del previo en V (0,6727 [0,6644; 0,6807], 62,4 %
[61,7; 63,2] contra 52,2 % de "siempre rojo"; de `pred_v.npz` y el resultado de `variables_partida.npz`, verificado
`idx_v` = V), en vivo (de `metricas.csv`: misma cohorte de 14.971, 0,7619 / 0,8168 / 0,8726), degradación en V,
huella (`mejoras_red_habitos.csv`, 0,892 con k = 10 y 0,975 con k = 50) y matchmaking (`swaps_resultados.parquet`,
10,03 → 4,96 pp, reproduce §9.S.77). Copiados del texto, con la sección citada junto a la constante en
`figuras.py`: la escalera de grupos de variables (`APORTES_PREVIO`, §9.S.52 escalones 1-4b, 6, 7 y 9 más las
correcciones de §9.S.68 y §9.S.75) y los dos pasos intermedios de lo previo en vivo (`SUMA_PREVIO_DOC`, §9.S.60
punto 6, sin predicciones guardadas). Los textos con números ("Respuesta") se generan en celdas ocultas a partir de
los mismos archivos: no hay números escritos a mano en el Markdown (issue #6).

**Decisiones de estilo.** Paleta validada con el validador de la skill `dataviz` sobre fondo blanco: azul `#2a78d6` y
rojo `#e34948` solo para los lados del mapa (separación con daltonismo ΔE 21,6); violeta `#6250d6` para el modelo
cuando no se habla de un lado (peor par sin daltonismo ΔE 31,5); gris `#898781` para referencias (azar, "siempre
rojo", el previo solo); una rampa violeta ordinal (`#a79ff0` → `#4a3aa7`, pasa el chequeo ordinal) para pasos que
se acumulan. Líneas de 2 px, puntos con aro blanco, grilla sólida muy clara, sin bordes superior y derecho, título
alineado a la izquierda con una línea gris de detalle debajo, Segoe UI. El texto nunca va en el color de la serie.

**Alternativas descartadas.**

* *Barras para "cuánto suma lo previo" con el eje desde 0,70.* Barras truncadas exageran la diferencia; quedó como
  puntos conectados por minuto, con la ganancia total anotada.
* *Un eje doble para AUC y acierto.* Dos paneles con su propio eje (small multiples).
* *Recalcular "solo el estado" para darle IC.* Pide rearmar los insumos del modelo en vivo (minutos y memoria) en el
  notebook; se dejó el número documentado y se dice en el subtítulo que no tiene IC.
* *Escribir los números en el Markdown.* Se desactualizan solos (issue #6); se generan al ejecutar.
* *Usar el mismo azul para "el modelo".* El azul significa "lado azul" en todo el informe.

**Tests.** `tests/test_informe.py` (4): formato castellano de los números, AUC ponderado contra `roc_auc_score` con y
sin réplicas (y con empates), IC que contienen al valor puntual, y el aviso cuando no existe la evaluación del test.
La suite pasa con 720 (718 y 2 omitidos); `test_scripts_importables.py` recoge el generador nuevo.

**Figura propia del test (2026-11-25, `fig_test_vs_v`, `04a_test_contra_v.png`).** Compara V con el test con la
misma definición en los dos: las partidas vivas al minuto 20 de cada conjunto (14.971 en V, 33.821 en el test), para el
previo y el en vivo a 10/15/20, en AUC y acierto. Se eligió la cohorte y no "todas las partidas" porque es la única
definición común a los cuatro momentos (al minuto 10 hay más partidas vivas que al 20), y así cada par de puntos
compara lo mismo. Por eso el previo aparece con 0,668 y no con el 0,6765 de la tabla (todas las partidas del test).
Puntos y no barras: la pregunta es si los dos coinciden, y los IC (bootstrap de días en el test, de partidas en V,
que es lo que guardó cada script) son casi siempre más chicos que el punto. Se descartó graficar la diferencia
test − V con IC: los dos conjuntos son disjuntos y los IC salen de esquemas de bootstrap distintos, así que un IC de
la diferencia exigiría recalcular sobre las predicciones de V, y no cambia la lectura (todas las diferencias son
≤ 0,007 a favor del test).

**Cierre (2026-11-25, decisión del usuario).** `informe_final` queda con el texto y el estilo actuales: es para el equipo, y la idea era que fuera corto y conciso. Los otros notebooks no se archivan ni se editan: siguen como registro del trabajo, así que los arreglos que el issue #6 pedía en `eda_modelos_ganador` (números escritos a mano en la celda 13, "algo menos en el test", "jugadores solos" 0,6369) no se aplican. El número que vale es el del test, en `informe_final` y en el log de la evaluación.

#### S.82. El recorrido del modelo previo y DraftRec en los notebooks (2026-11-24, issue #6)

**Pedido del usuario.** Completar el gráfico de modelos probados de `eda_modelos_ganador` (AUC y acierto) con los
que faltaban, y sumar al notebook de la entrega el recorrido del modelo previo de punta a punta (opción "puente",
no línea de tiempo del proyecto) y el puente contra DraftRec del compañero (`grafico_draftrec_puente.py`),
con la última barra en el test cuando exista.

**`eda_modelos_ganador`, sección 1.** El panel de V suma 14 modelos medidos en **la misma V Master+** (16.428
partidas): la escalera de §9.S.52 (híbrido + mapa y escalones 1-8 y 10), TabM (§9.S.62) y el modelo previo vigente
sin la fuga de la foto 1 (§9.S.75, 0,6727 / 62,4 %). Sus números se copian de este documento a la constante
`EXTRA_V`, con la sección al lado, porque esas corridas no guardaron predicciones; por eso no tienen Train y el
panel de Train queda con los 8 de `modelos_train_v.csv`. El IC sigue siendo Hanley-McNeil para el AUC y Wilson
para el acierto: dependen solo de (AUC, $n_1$, $n_0$) y de (acierto, $n$), y $n_1, n_0$ son los mismos en todos
porque V es la misma. La línea une solo lo adoptado, en dos tramos (el camino de §9.S.55 y la escalera, que arranca
de una logística más simple que el MOE y por eso empieza más abajo); los ensambles 5, 8 y 10 y TabM van huecos,
"probados y no adoptados". *Descartado:* sumar los modelos de §9.S.24-§9.S.39 (redes y boostings con el margen,
Poisson, 7 expertos, red con el nivel del lobby): se midieron en la V vieja con todas las ligas (26.934 partidas) y
mezclarlos movería el AUC por la población, no por el modelo. *Descartado también:* recalcularlos ahora, porque exige
rearmar las cachés de `historial/` con la evaluación del test corriendo y la RAM al límite.

**Notebook de la entrega, sección 4.** `fig_recorrido_previo`: barras de acierto en V Master+ de 8 hitos (siempre
rojo → solo draft → draft + LP y huella → stacking → MOE → logística Master+ con contracruces → + Elo, oro esperado
y duelos → sin fugas), con el AUC dentro de cada barra, el Δ de acierto entre barras y una novena barra con el
test (IC por bloques de días) que aparece sola cuando `data/analysis/test_final/metricas.csv` existe
(`_previo_test`: fila `previo` / `evaluacion` / `dias`). Se eligió el acierto como eje porque es la métrica que
también tiene DraftRec y la que se entiende sin explicar; el AUC va escrito. El escalón 2, que retrocede en
acierto (61,6 % < MOE 61,7 %) aunque gana en AUC, se salta para no mostrar como pérdida lo que es ganancia; la
escalera completa está en `fig_aportes_previo`. `fig_draftrec`: el puente del compañero con la última barra en el
test (o en V mientras no exista) y la barra estimada recalculada como final − [1,5; 2,0] puntos, rayada y sin barra
de error (es un rango, no un IC). *Descartado:* dejar la última barra en V (62,5 %, el escalón 9 con la fuga): el
informe compara contra el número que se reporta, que es el del test.

*Revisión del 2026-11-25 (decisión del usuario):* `fig_draftrec` queda con **solo las dos barras medidas sobre los
mismos datos**: DraftRec 55,35 % y nuestro escalón 9 reconstruido 59,27 %, las dos en las mismas 28.690 partidas
del test de DraftRec (§9.S.59), con IC binomial. Se sacan la barra estimada (61,0-61,5 %), porque es una
extrapolación y no una medición, y la de nuestro modelo completo en nuestro test (63,0 %), porque es otro dataset,
otra población y otro año, y ponerla al lado sugiere una comparación que no es pareja. El 63,0 % ya está en
`fig_recorrido_previo` y en `fig_test_vs_v`. El modelo de la barra de DraftRec es el final en lo que se puede
portar: los cambios posteriores al 17/11 (§9.S.68 aislamiento de V, §9.S.75 máscara de la foto 1) tocan solo
variables con LP o foto, que el dataset de DraftRec no tiene.

**Huella (punto 2 del pedido).** Queda `fig_huella` como está: ya compara la red contrastiva con el discriminante
lineal según $k$; la escalera completa (Euclídea → LDA → red → + hábitos) vive en `eda_reconocimiento_jugadores`.

**Tests.** `tests/test_informe.py` suma 2: `_previo_test` toma solo la fila principal, y las dos figuras se dibujan
con y sin test (8 y 9 barras) y se guardan con su nombre.

#### S.83. Datos publicados en Zenodo, seudonimizados (2026-11-25)

**Pedido del usuario.** Que el trabajo sea reproducible con nuestros datos: subir a Zenodo la base del crawler, los
splits, el LP en vivo y los timelines; no los resultados, las cachés ni los pesos (se regeneran con el repo).

**El problema.** El ID de jugador del proyecto es el Riot ID en minúsculas: `player_id` en los CSV y `canonical_id`
en la base valen `nombre#tag` en el 100 % de las filas (84.304 jugadores), y la base tiene además `game_name` y
`tag_line` en tres tablas. El PUUID, en cambio, Riot lo cifra por proyecto de API, y queda.

**La solución** (`scripts/ops/empaquetar_zenodo.py`). Cada ID se reemplaza por
$\text{seud}(x) = \texttt{"j"} \,\|\, \text{HMAC-SHA256}(k, x)_{[0:16]}$ (16 hex, $2^{64}$ valores: la probabilidad de
una colisión entre $n = 84.304$ IDs es $\approx n^2 / 2^{65} \approx 2 \cdot 10^{-10}$). Es determinista, así que el
mismo jugador tiene el mismo seudónimo en todos los CSV y en todas las tablas, y los joins y los modelos no cambian
(ningún código depende del formato `nombre#tag`, verificado). En la base, las columnas con IDs se detectan por
contenido (texto con `#` que no sea JSON), no por nombre: así entran `player_id_map.player_id` y
`crawl_window_players.player_id`, que no se llaman `canonical_id`. `game_name` y `tag_line` quedan vacíos ('' donde la
columna es `NOT NULL`). Se verifica que no quede ningún `#` fuera de JSON ni ningún nombre.

*Por qué HMAC con clave y no un hash simple.* Un SHA-256 de `nombre#tag` sin clave se revierte probando nombres: quien
conozca un Riot ID puede calcular su hash y encontrarlo. Con la clave secreta $k$ (32 bytes en
`data/zenodo_clave.txt`, fuera del repo y del paquete) eso no se puede; con la clave, el mismo seudónimo se reproduce
en un paquete nuevo. *Descartado:* numerar los jugadores (1, 2, ...) depende del orden y cambia entre paquetes;
sacar los PUUID, porque ya son cifrados por proyecto.

**Paquete publicado (2026-11-26).** DOI [10.5281/zenodo.24571527](https://doi.org/10.5281/zenodo.24571527). Los
timelines son los completos del compañero (209.817 partidas con timeline y 34 sin él; copia del Drive verificada con
`quick_check` y con el conteo de su manifiesto), empaquetados con la misma clave, así que los seudónimos de la base
y de los splits no cambian respecto del paquete del 25/11. `timelines.db` no tenía columnas con Riot ID; en una
muestra de 300 JSON descomprimidos tampoco aparecen nombres (Timeline-V5 trae solo PUUID). La integridad de la
subida se comprobó contra el MD5 que muestra Zenodo de cada archivo.

## 10. Huella de Estilo: Reidentificación y Detección de Cambio de Persona (2026-10-24)

### 10.0. Pregunta y motivación

**Decisión del usuario (2026-10-24), pendiente de acordar con el compañero:**
reorientar el proyecto. La predicción de victoria antes de la partida ya la venden
varias apps (Porofessor, iTero, Champ Select Coach) y queda cerca del azar en
SoloQ (baseline sin leakage 52,64 % contra 51,29 % de "siempre Red", §9.S.13). La
pregunta nueva:

> **¿Qué parte de cómo juega una persona es suya y qué parte es del campeón?** Si
> existe una huella de estilo que sobrevive al cambio de campeón, sirve para
> reconocer a un jugador por cómo juega y para detectar cuándo una cuenta pasa a
> jugarla otra persona (cuentas compartidas, *boosting*).

Las dos partes se necesitan: para afirmar "cambió la persona" hay que poder
descartar "cambió el campeón". Por eso la prueba central compara a la misma
persona con otros campeones contra personas distintas con el mismo campeón.

**Las firmas existentes no sirven para esto.** `src/features/player_signatures.py`
promedia los embeddings de los campeones que juega cada uno: codifica el **pool**,
no el estilo. *Corrección del 26/10:* no es lo mismo que el baseline `pool`. `pool`
cuenta campeones por nombre, y el embedding ubica cada campeón según cómo se juega,
así que sigue funcionando con campeones disjuntos. Se agregó como baseline
`embedding` (§9.S.20); los resultados están en §10.2.

### 10.1. Protocolo de validación

El problema de partida es que no hay cuentas compartidas etiquetadas. Se resuelve
con cinco niveles, cada uno probando algo que el anterior no puede:

| Nivel | Pregunta | Etiquetas | Métrica principal |
| :--- | :--- | :--- | :--- |
| 1 | ¿Existe la huella? | identidad (`player_id`) | AUC de reidentificación + permutación |
| 2 | ¿Reconoce a jugadores no vistos? | identidad, jugadores fuera del entrenamiento | ROC-AUC, EER, TAR@FAR=1 %, en función de $k$ |
| 3 | ¿Detecta un cambio de persona? | sintéticas: secuencias empalmadas de dos jugadores reales | detección, falsas alarmas/100 partidas, demora |
| 4 | ¿Funciona en casos reales? | ~11 fusiones erróneas confirmadas contra Riot (§9.S.9) + 824 cambios de nombre | caso por caso |
| 5 | ¿Qué aparece en el resto de las cuentas? | sin etiquetas | candidatos + corroboración independiente |

**Reglas comunes.**
* **Criterios de éxito fijados antes de mirar el test** (los fija el usuario
  después del nivel 1, antes de entrenar). Si no, la vara se ajusta al resultado.
* **Dos ejes de separación.** Por jugador (la huella se aprende con unos y se
  evalúa con otros: evita memorizar personas) y por tiempo (train antes del 01/09,
  §9.S.17; test después: estabilidad de la huella).
* **Los casos reales del nivel 4 no se miran durante el desarrollo.**
* **Rivales difíciles.** Los pares de personas distintas se arman con el mismo
  tier y la misma línea principal; si no, el modelo "reconoce" elo o rol.
* **Intervalos por bootstrap de jugadores**, no de partidas: las partidas de una
  persona no son independientes. Varias semillas en todo lo aleatorio.

**Baselines obligatorios** (la huella tiene que superarlos):

| Baseline | Si la huella no le gana, significa que… |
| :--- | :--- |
| `pool` (frecuencia de campeones) | reconoce gustos, no forma de jugar |
| `elo_linea` (tier + reparto de líneas) | reconoce nivel o rol |
| `crudo` (sin descontar el campeón) | el ajuste por campeón no aporta |

**Fuera de la huella, a propósito: el horario de juego.** Identifica mucho
(`match_times.csv`), pero no es estilo. Se reserva como segunda señal
independiente del nivel 5: un candidato es fuerte si el cambio de estilo coincide
con un cambio de horario o un salto brusco de elo/winrate, que el detector no usó.

**Nivel 3 en detalle.** Secuencias armadas con jugadores no vistos: *boosting*
(el segundo jugador de tier más alto y la misma línea), cuenta compartida
(partidas alternadas) y controles negativos sin empalme pero con cambios
naturales de campeón (miden las falsas alarmas). Resultados estratificados por la
distancia de estilo entre los dos jugadores: la salida honesta es una curva
"detecta a partir de tal distancia", no un número.

**Nivel 4 en detalle.** Necesita el respaldo previo a la corrección del 23/10
(`lol_las_pre_fix_map_20261023.db`), que está en la máquina del compañero. Positivos:
p. ej. las 85 partidas de `jugador_g#tag` asignadas a `jugador_h#tag`. Negativos: los
cambios de nombre verificados (misma persona, no debe marcarse). Con ~11
positivos no hay tasa: se informa caso por caso.

**Nivel 5:** candidatos ordenados por sospecha, revisión manual de los primeros N,
resultados agregados y sin exponer cuentas con nombre.

**Ablación:** quitar un grupo de variables por vez (oro, daño, visión, estado a los
10/15) para ver de dónde sale la huella.

*Alternativas descartadas.* Validar solo con el escaneo de cuentas reales: sin
etiquetas, cualquier resultado es especulación sobre trayectorias. Usar las firmas
de `player_signatures.py`: miden el pool (ver 10.0). Incluir el horario en la
huella: subiría el AUC, pero mezclaría estilo con rutina y dejaría al nivel 5 sin
señal independiente.

### 10.2. Nivel 1: ¿existe la huella?

**Representación** (`src/features/fingerprint.py`). Por partida, 23 estadísticas
por minuto o en los minutos 10/15 (`STYLE_FEATURES`: kills, muertes, asistencias,
participación en kills, cuota de daño y de oro, daño hecho y recibido, oro, farm,
visión, y el estado a los 10 y 15). Cada una se expresa como desvío respecto de
las partidas del mismo campeón, línea y resultado:

$$z = \frac{x - \mu_{c,l,w}}{\sigma_{c,l,w}}, \qquad n_{c,l,w} \ge 30$$

Condicionar por el resultado $w$ evita que la huella sea "gana mucho" (un ganador
siempre tiene más oro). La media de grupo incluye las partidas del propio jugador;
en campeones dominados por un *one-trick* eso acerca la media al jugador y achica
su desvío, así que el sesgo va en contra de encontrar huella (conservador).

**Prueba.** Por jugador, dos muestras disjuntas de $k$ partidas (A y B). Con
$D_{ij}$ la distancia euclídea entre el perfil A del jugador $i$ y el perfil B del
jugador $j$:

$$\text{AUC} = \frac{1}{n}\sum_i \frac{1}{n-1}\sum_{j\ne i}\Big(\mathbb{1}[D_{ij} > D_{ii}] + \tfrac{1}{2}\mathbb{1}[D_{ij} = D_{ii}]\Big)$$

0,5 es azar y 1 es reconocimiento perfecto. El p-valor sale de 1.000
permutaciones que reasignan qué B corresponde a cada A (con los rangos por fila
precalculados, cada permutación cuesta $O(n)$). `auc_dificil` restringe los $j$ a
jugadores del mismo tier y línea principal. Muestras: `aleatoria`,
`campeones_distintos` (conjuntos de campeones disjuntos entre A y B) y `temporal`
(las $k$ más viejas contra las $k$ más nuevas). Con más de 3.000 jugadores
elegibles se toma una muestra al azar (la matriz es $n \times n$).

Tests: `tests/test_fingerprint.py` (7 casos sintéticos: con huella se reconoce,
sin huella da azar, el ajuste quita el efecto del campeón, los campeones son
disjuntos y la huella sobrevive, orden temporal, AUC por fila y máscara, reparto
de varianza por jugador). Verificados por mutación: si se residualiza solo por
línea, fallan 3; si el ICC no descuenta $\sigma^2_w/\bar n$, falla el del reparto.

**Resultados** (`data/analysis/huella_nivel1.json`, `scripts/analysis/fingerprint_level1.py`,
410 s). Train al 01/09 (§9.S.17), solo LAS y líneas canónicas: 887.686 filas
jugador-partida, 88.770 partidas, 33.725 jugadores; 872.317 filas (98,3 %)
sobreviven a la residualización. Muestra de 3.000 jugadores por corrida (elegibles:
de 14.081 con $k=5$ a 5.369 con $k=20$ en campeones distintos). Media de 3
semillas (1 en `temporal`, que es determinística):

| Muestra | $k$ | `huella` AUC / difícil | `crudo` AUC / difícil | `pool` AUC / difícil | `elo_linea` AUC / difícil |
| :--- | ---: | :--- | :--- | :--- | :--- |
| aleatoria | 5 | 0,701 / 0,696 | 0,739 / 0,730 | 0,917 / 0,890 | 0,947 / 0,639 |
| aleatoria | 10 | 0,791 / 0,786 | 0,834 / 0,825 | 0,980 / 0,959 | 0,963 / 0,726 |
| aleatoria | 20 | 0,887 / 0,882 | 0,923 / 0,916 | 0,995 / 0,987 | 0,974 / 0,795 |
| **campeones distintos** | 5 | 0,660 / 0,655 | 0,654 / 0,647 | 0,466 / 0,440 | 0,886 / 0,544 |
| **campeones distintos** | 10 | 0,743 / 0,739 | 0,722 / 0,715 | 0,423 / 0,377 | 0,897 / 0,571 |
| **campeones distintos** | 20 | **0,824 / 0,819** | 0,791 / 0,780 | 0,345 / 0,277 | 0,910 / 0,595 |
| temporal | 5 | 0,665 / 0,662 | 0,701 / 0,692 | 0,851 / 0,822 | 0,915 / 0,584 |
| temporal | 10 | 0,748 / 0,745 | 0,791 / 0,783 | 0,931 / 0,896 | 0,929 / 0,635 |
| temporal | 20 | 0,832 / 0,828 | 0,860 / 0,852 | 0,963 / 0,930 | 0,937 / 0,670 |

IC95 por bootstrap de jugadores de ±0,005 a ±0,010 en `huella`; p-valor de
permutación 0,001 (el mínimo con 1.000 permutaciones) en todas las corridas de
`huella`, `crudo` y `elo_linea`.

**Lectura.**

1. **La huella existe y sobrevive al cambio de campeón.** Con conjuntos de
   campeones disjuntos y $k=20$, el AUC es 0,824 (azar 0,5), y crece con $k$ como
   se espera de una señal que se promedia (0,660 → 0,743 → 0,824).
2. **No es elo ni rol.** El AUC difícil (mismo tier y línea principal) cae apenas
   0,005 respecto del global en `huella`; en `elo_linea` cae de 0,910 a 0,595.
   El AUC global de `elo_linea` es alto porque el tier del padrón es una constante
   por jugador, idéntica en A y B por construcción: por eso la comparación justa
   contra ese baseline es la difícil, donde la huella le gana por más de 0,2 en
   todas las muestras.
3. **El ajuste por campeón aporta justo donde importa.** En `aleatoria` y
   `temporal`, `crudo` le gana a `huella` (0,923 contra 0,887 con $k=20$): sin
   descontar el campeón, el perfil también lleva el pool. Con campeones
   distintos se invierte: 0,824 contra 0,791 con $k=20$, IC95 [0,814; 0,835]
   contra [0,782; 0,801], sin superposición; con $k=10$ la diferencia (0,021) ya
   tiene los intervalos tocándose, y con $k=5$ no hay diferencia. Es decir: el
   residuo es la parte del perfil que se mantiene cuando cambia el campeón.
4. **`pool` cae por debajo de 0,5 con campeones distintos.** Es lo esperado y no
   un error: si A y B no comparten campeones, la distancia coseno entre los dos
   perfiles del mismo jugador es la máxima posible (1), mientras que con otros
   jugadores suele haber algún campeón en común. Confirma que las firmas
   actuales (§10.0) no sirven para la pregunta.
5. **Hay deriva temporal.** Con $k=20$, `temporal` (las más viejas contra las más
   nuevas) da 0,832 contra 0,887 de `aleatoria`: el estilo cambia con el tiempo
   (parches, aprendizaje, cambio de rol). El nivel 3 tiene que distinguir esta
   deriva gradual de un cambio brusco.
6. **Reconocer no es identificar.** El top-1 entre 3.000 es 2–5 % (azar 0,03 %):
   la huella ordena bien, pero no alcanza para decir *quién* es entre miles. Para
   la detección de cambio (nivel 3) alcanza con comparar la cuenta consigo misma,
   que es la pregunta que el AUC mide.

**Recalculado el 26/10 con el split vigente y el baseline `embedding`** (§9.S.20;
train: Master+ hasta el 27/09 más Diamond anterior, 1.921.003 filas, 1.894.593
residualizadas; 3.000 jugadores por corrida; media de 3 semillas). La muestra que
decide es la de campeones distintos:

| $k$ | `huella` AUC / difícil | `embedding` | `crudo` | `pool` | `elo_linea` |
| ---: | :--- | :--- | :--- | :--- | :--- |
| 5 | 0,661 / 0,656 | **0,722** / 0,553 | 0,655 / 0,646 | 0,468 / 0,444 | 0,891 / 0,546 |
| 10 | 0,736 / 0,732 | **0,759** / 0,575 | 0,719 / 0,711 | 0,425 / 0,382 | 0,911 / 0,570 |
| 20 | **0,823** / **0,819** | 0,791 / 0,602 | 0,784 / 0,775 | 0,347 / 0,283 | 0,919 / 0,603 |

IC95 de ±0,008 a ±0,010 en todas las celdas. Los números de la huella coinciden con
los del train anterior (0,824 contra 0,823 con $k=20$).

1. **El embedding es el rival "solo campeones" que corresponde, y es fuerte.** Con
   pocas partidas le gana a la huella en el AUC global (0,722 contra 0,661 con
   $k=5$), y la huella recién lo supera con $k=20$ (0,823 [0,815; 0,831] contra
   0,791 [0,783; 0,800]).
2. **La fuerza del embedding es el rol.** En el AUC difícil (mismo tier y misma
   línea principal) cae a 0,55–0,60, lo mismo que `elo_linea`: el vector de un
   soporte y el de un ADC están lejos, así que reconoce sobre todo de qué juega
   la persona. La huella casi no cae (0,819 contra 0,823). **En la comparación
   justa, entre jugadores del mismo rol, la huella le gana al embedding con todos
   los $k$**: 0,656 contra 0,553, 0,732 contra 0,575 y 0,819 contra 0,602.
3. **`pool` no sirve de baseline con campeones distintos**: da menos que el azar
   por construcción (punto 4 de arriba). Para "solo campeones" se reporta el
   embedding.
4. **En `aleatoria` y `temporal` los baselines de campeones le ganan a la huella**
   (embedding 0,969 y `pool` 0,995 con $k=20$ en `aleatoria`). Es esperable: A y B
   comparten campeones y la huella descuenta el campeón a propósito. Esas muestras
   miden otra cosa, no si existe la huella.


*Qué no prueba el nivel 1.* Los jugadores son los mismos con los que se calcularon
las medias de grupo, y la huella no se "aprendió" (es una distancia euclídea sin
pesos): el nivel 2 recién mide si una métrica aprendida generaliza a jugadores no
vistos. La media de grupo incluye al propio jugador (sesgo conservador, ver arriba).

**¿Qué estadísticas delatan a la persona?** El AUC dice que la huella existe; no
dice de dónde sale. Para cada estadística ya residualizada se mide qué parte de su
variación entre partidas se explica por quién juega (correlación intraclase,
`player_variance_share`). Con $\sigma^2_b$ la varianza de las medias por jugador,
$\sigma^2_w$ la media de las varianzas dentro de cada jugador y $\bar n$ las
partidas medias por jugador:

$$\text{ICC} = \frac{\sigma^2_b - \sigma^2_w/\bar n}{\sigma^2_b - \sigma^2_w/\bar n + \sigma^2_w}$$

Restar $\sigma^2_w/\bar n$ descuenta el ruido que tienen las medias por estar
hechas con un número finito de partidas: sin esa corrección, una estadística sin
ninguna huella daría ICC > 0 (con 10 partidas por jugador, ~0,09). Sobre los 6.542
jugadores del train con al menos 40 partidas
(`data/analysis/huella_reparto_varianza.json`):

| Estadística | ICC | | Estadística | ICC |
| :--- | ---: | :--- | :--- | ---: |
| `vision_pm` | **0,252** | | `gold_share_pct` | 0,070 |
| `cs_pm` | 0,159 | | `xp_15` | 0,065 |
| `deaths_pm` | 0,129 | | `xp_10` | 0,061 |
| `cs_15` | 0,126 | | `deaths_10` | 0,060 |
| `damage_taken_pm` | 0,115 | | `gold_15` | 0,059 |
| `cs_10` | 0,106 | | `early_aggression_10` | 0,055 |
| `damage_share_pct` | 0,090 | | `kills_pm` | 0,047 |
| `damage_15` | 0,088 | | `gold_10` | 0,046 |
| `damage_pm` | 0,081 | | `kill_participation_pct` | 0,032 |
| `gold_pm` | 0,079 | | `kills_10` | 0,028 |
| `damage_10` | 0,077 | | `assists_pm` | 0,025 |
| | | | `assists_10` | 0,014 |

*Lectura.* Lo más personal son los **hábitos que el jugador controla solo**: la
visión (un cuarto de la variación entre partidas es de la persona, con cualquier
campeón), el farm, las muertes y el daño recibido (cuánto se expone). Lo menos
personal es lo que depende de la partida y del equipo: kills, asistencias y
participación en kills. Ninguna estadística sola identifica (la mayor es 0,25):
la huella sale de combinar 23 señales débiles y de promediar varias partidas, lo
que explica que el AUC crezca con $k$.

*Uso para lo que sigue.* Es una primera ablación (§10.1): si en el nivel 2 una
métrica aprendida le da más peso a visión y farm, es coherente con esto. También
marca un riesgo: la visión depende del rol (un support wardea más), y aunque la
residualización es por campeón y línea, un jugador que cambia de línea principal
cambia mucho su visión; el nivel 3 tiene que tenerlo en cuenta en sus controles
negativos.

*Alternativa descartada.* Importancia por permutación de features sobre el AUC:
depende de la distancia elegida y de $k$, y con 23 features correlacionadas
(oro, farm y XP se mueven juntos) reparte mal el crédito. El ICC es una propiedad
de cada estadística, independiente de la prueba.

### 10.3. EDA de la huella (2026-10-26; números actualizados el 2026-10-27)

**Qué es.** `notebooks/eda_huella.ipynb` (generado por
`scripts/analysis/build_eda_huella_nb.py`) mira por dentro la representación del
nivel 1, sobre el mismo train que el nivel 1: `data/splits/train/full_dataset.csv`
del split vigente (§9.S.12, corte 27/09), solo LAS, las 5 líneas canónicas y
jugadores Master+ del padrón (`keep_min_tier`, `--min-tier MASTER` por defecto en
`fingerprint_level1.py`). Reutiliza las funciones de `src/features/fingerprint.py`
y de `scripts/analysis/fingerprint_level1.py`, así que no hay una segunda
implementación que pueda divergir. Recalcula el ICC y coincide con
`huella_reparto_varianza.json` (diferencia máxima 0,0000). No toca el test.

*Origen y actualización.* Lo escribió el compañero en la rama `gonza` (PR #1,
commit `e1664d7`) y se incorporó a `gonchi` por cherry-pick el 2026-10-27. Sus
números originales venían de su split de corte fijo al 01/09 (§9.S.17, 88.770
partidas, 33.725 jugadores sin filtro de tier) y una celda (la deriva) fallaba. Se
volvió a ejecutar sobre el train vigente con el filtro Master+, y se regeneró
`huella_reparto_varianza.json` (`fingerprint_level1.py --solo-reparto`), que era
anterior al filtro. Los hallazgos de abajo son los de esa ejecución. Las lecturas
cualitativas se sostienen; cambian la cobertura (por el filtro Master+) y el orden de
las líneas en el AUC.

**Reparto completo de la varianza.** El ICC del §10.2 es una fracción *del
residuo*. Para saber de quién es cada estadística en total, se descompone su
varianza con $R^2$ incrementales de las medias de grupo, sobre las filas que
sobreviven a la residualización:

$$1 = \underbrace{R^2_{l}}_{\text{línea}} + \underbrace{R^2_{c,l} - R^2_{l}}_{\text{campeón}} + \underbrace{R^2_{c,l,w} - R^2_{c,l}}_{\text{resultado}} + \underbrace{(1 - R^2_{c,l,w})\,\text{ICC}}_{\text{jugador}} + \underbrace{(1 - R^2_{c,l,w})(1-\text{ICC})}_{\text{partida}}$$

con $R^2_G = 1 - \operatorname{Var}(x - \mu_G)/\operatorname{Var}(x)$. La porción
"jugador" es aproximada: el ICC se mide en los jugadores con ≥ 40 partidas.

**Hallazgos** (detalle y figuras en el §10 del notebook; train vigente, Master+):

1. **Cobertura.** 189.076 partidas y 8.997 jugadores Master+, con una mediana de
   109 partidas por jugador. 8.683 tienen ≥ 40 y aportan el 99 % de las filas. Con
   el split viejo y sin filtro de tier, la mediana era 6: el filtro Master+ deja a
   los jugadores del padrón, que son los que se siguieron partida a partida.
2. **Calidad.** Solo faltan estadísticas del timeline (0,49 % a los 10 min, 1,03 %
   a los 15); el umbral $n_{c,l,w} \ge 30$ descarta entre 0,44 % y 0,59 % de las
   filas según la línea. Queda residualizado el 98,5 % del train.
3. **La línea domina** farm, XP y visión (75–84 % de la varianza); el campeón pesa
   sobre todo en el daño recibido (28 %) y el resultado en muertes (21 %),
   asistencias, oro y kills.
4. **ICC y porción absoluta no ordenan igual.** Visión: ICC 0,25, pero 4,5 % de la
   varianza total (la línea se lleva el 80 %). En absoluto lo más personal es
   `deaths_pm` (9,3 %). Para la huella manda el ICC, porque cada residuo se
   reescala a desvío 1 antes de medir distancias.
5. **Redundancia.** La dimensión efectiva (*participation ratio*,
   $1/\sum_i \lambda_i^2$ con $\lambda_i$ la fracción de varianza de cada
   componente) es 6,2 por partida y 4,1 en los perfiles de jugador (de 23
   variables). Hay pares con correlación > 0,8 (`gold_share_pct`–`gold_pm` 0,86,
   `cs_10`–`cs_15` 0,84, `gold_10`–`kills_10` 0,82). La distancia euclídea sin pesos
   cuenta esos bloques varias veces. *Hipótesis para el nivel 2:* blanquear
   (Mahalanobis) o aprender la métrica.
6. **Poco elo o línea residual.** $R^2$ del tier sobre los perfiles ≤ 0,037
   (`kill_participation_pct`); el de la línea principal ≤ 0,012. Con solo Master+
   el rango de tiers es más angosto, así que era esperable que bajara (antes 0,056).
7. **Reconocimiento desparejo.** Con campeones distintos y $k=10$ (3.000 jugadores
   muestreados), el AUC medio es 0,736, pero el 19,7 % de los jugadores queda debajo
   de 0,5. MIDDLE y BOTTOM se reconocen mejor (0,753 y 0,748) que TOP y JUNGLE (0,711
   y 0,717).
8. **La deriva es gradual.** En bloques consecutivos de 10 partidas (6.763
   jugadores con ≥ 60), la distancia al propio perfil va de 2,05 (~8 días) a 2,14
   (~45 días), contra 2,76 a otro jugador. La separación (otro − mismo) conserva ~85 %
   a los 45 días y ~83 % a los 71: el nivel 3 tiene que distinguir esta pendiente de
   un salto.

*Alternativas descartadas.* Usar un modelo lineal mixto (jugador como efecto
aleatorio, campeón/línea/resultado fijos) para la descomposición: es lo exacto,
pero con miles de jugadores y ~1.000 grupos es lento y, para una EDA, los $R^2$
incrementales más el ICC corregido del §10.2 dan la misma lectura. Proyectar con
t-SNE/UMAP: deforman distancias, y la prueba es justamente sobre distancias
euclídeas; PCA las preserva en el subespacio que muestra.

### 10.4. Mejoras de la huella y su aporte al modelo previo a la partida (2026-11-18)

**Pedido.** Probar en orden las mejoras propuestas sobre el nivel 1. Cuando una no se pueda completar (la distancia
aprendida es en realidad el nivel 2), estimarla con lo que haya. Después, medir cuánto le suma la huella mejorada al
modelo previo a la partida. Script: `scripts/analysis/huella_mejoras.py --parte 1|2|3|4`, salidas en
`data/analysis/huella/`.

**Cómo se mide, igual para todas.** La parte que se aprende (pesos, red) se ajusta con la mitad de los jugadores,
y el AUC de reidentificación se mide en la otra mitad, que el ajuste nunca vio. Los IC son por bootstrap de
jugadores. La prueba que decide es la de campeones distintos. Partes 1, 3 y 4: train Master+, 1.225.439 filas
residualizadas, 8.997 jugadores. Parte 2: las partidas con timeline completo (100.049; V entera y las Master+ del
ajuste ya bajadas), 803.480 filas, 8.996 jugadores. La parte 2 incluye partidas de V: es descriptiva, como el nivel
1.

**Mejora 1: pesar las variables.** La distancia euclídea pesa igual las 23 estadísticas, aunque varias están muy
correlacionadas (§10.3) y su parte "personal" (ICC) va de 0,02 a 0,25. Se probaron dos transformaciones lineales,
aprendidas con los jugadores de ajuste:

* *Mahalanobis*, $d^2 = (a-b)^\top S_w^{-1} (a-b)$, con $S_w$ la covarianza **dentro** del jugador (sus partidas
  centradas en su media): descuenta el ruido partida a partida y la redundancia entre variables.
* *LDA*, que usa los autovectores de $S_w^{-1/2} S_b S_w^{-1/2}$ ($S_b$ = covarianza entre las medias de los
  jugadores) y pesa cada dirección por $\sqrt{\lambda}$: más peso a lo que separa personas que a lo que varía
  entre partidas de la misma.

**Mejora 2: hábitos de los timelines.** Son 18 variables por partida que el crawler no guardaba:

* orden de habilidades (cuál sube primero al máximo, cuál toma en el nivel 1);
* minuto de la primera vuelta a la base y visitas a la tienda antes del 10;
* control wards comprados antes del 15, wards puestos antes del 10 y rotos antes del 15;
* minuto del nivel 6 y de la primera muerte;
* oro sin gastar medio entre los minutos 3 y 15;
* movimiento entre los minutos 2 y 10 y avance hacia la base rival entre el 5 y el 10;
* placas antes del 14 y control a los 15.

Se residualizan igual que las 23 estadísticas, por campeón, línea y resultado. Un hábito constante dentro del grupo
(un campeón que siempre sube Q primero) vale 0, en vez de descartar la fila.

**Mejora 3: distancia aprendida (estimación del nivel 2).** Es una red de 23 → 128 → 128 → 32 sobre el perfil medio
de k partidas, entrenada con pérdida contrastiva (InfoNCE, temperatura 0,1, 3.000 pasos de 256 jugadores, con k al
azar entre 5 y 20) con los jugadores de ajuste, y evaluada con los no vistos.

**Mejora 4: recencia.** El perfil se arma con las 60 partidas anteriores, pesadas por $2^{-\text{edad}/v}$, para
reconocer las 10 siguientes.

| Huella (campeones distintos, jugadores no vistos) | k = 10 | k = 20 |
| :--- | :---: | :---: |
| 23 estadísticas, euclídea (nivel 1) | 0,729 [0,721; 0,737] | 0,815 [0,808; 0,823] |
| 23 estadísticas, Mahalanobis | 0,773 | 0,865 |
| 23 estadísticas, LDA | 0,816 [0,810; 0,822] | 0,883 [0,878; 0,888] |
| 23 estadísticas, red contrastiva | 0,833 [0,826; 0,838] | 0,901 [0,896; 0,905] |
| Solo hábitos, LDA (partidas con timeline) | 0,806 | 0,862 |
| **23 estadísticas + hábitos, LDA** (partidas con timeline) | **0,866** [0,860; 0,872] | **0,921** [0,915; 0,926] |

*Corrección (2026-11-23, §9.S.77).* Hasta esa fecha, el residuo por campeón, línea y resultado se estimaba con todos
los jugadores, incluidos los de evaluación (fuga leve). Ahora se estima solo con los de ajuste. Re-medido con la
misma partición, las partes 1, 3 y 4 cambian ±0,005 o menos, dentro de los IC: euclídea 0,735 / 0,819, LDA
0,816 / 0,885 y red 0,831 / 0,904. Con hábitos (re-medido en una base de timelines más chica, 62.065 partidas, ver
§9.S.77 punto 4): estadísticas + hábitos con LDA 0,867 [0,860; 0,873] / 0,920 [0,914; 0,927] (antes 0,874 / 0,921 en
esa base) y solo hábitos con LDA 0,805 / 0,864. Todo dentro de los IC.

En las partidas con timeline, la base de 23 estadísticas da 0,741 / 0,822 con la euclídea y 0,827 / 0,891 con LDA,
así que la ganancia de los hábitos es +0,039 / +0,030 sobre LDA. El AUC difícil (mismo tier y misma línea) cae
apenas 0,003-0,007 en todas las variantes: nada de esto es reconocer elo ni rol. La recencia no suma: 0,869 con
todas iguales contra 0,873 con vida media de 20-40 partidas, dentro del IC. La deriva es lenta (§10.3, hallazgo 8)
y el promedio de 60 partidas ya la absorbe. El acierto en el primer intento entre miles sigue bajo (2-8 %):
reconocer no es identificar.

**El hábito más personal es comprar control wards antes del 15** (ICC 0,44, más que cualquiera de las 23
estadísticas; la visión por minuto da 0,25). Le siguen los wards antes del 10 (0,23) y el orden de las habilidades
(0,09-0,12). Movimiento y avance casi no son del jugador (0,02): dependen del campeón y de la partida.

**Lectura.** Las mejoras 1 y 3 confirman que la distancia sin pesos desperdiciaba señal: LDA iguala con 10
partidas lo que antes necesitaba 20, y la red generaliza a jugadores no vistos (la pregunta del nivel 2). Los hábitos
son la mejor fuente: solos rinden casi como las 23 estadísticas, y juntos suman 3-4 puntos de AUC. No se probó la
red con estadísticas + hábitos; por lo visto en la parte 3, debería quedar algo por encima de 0,866 / 0,921.

**Aporte al modelo previo a la partida (`scripts/analysis/huella_previo.py`).** Las 85 variables del modelo base
ya incluyen la huella progresiva de las 23 estadísticas (`perf_*`, §9.S.23). Por eso no se probaron las mejoras 1 y
3 en el modelo:

* los pesos LDA o Mahalanobis son una transformación lineal de variables que la logística ya recibe y ya pesa;
* la red está entrenada para distinguir personas, no para predecir quién gana.

Lo único con información nueva son los hábitos. El hábito progresivo de cada jugador se arma con sus partidas
anteriores con timeline, $h = \sum_j 2^{-\text{edad}_j/100} z_j / (\sum_j 2^{-\text{edad}_j/100} + 10)$, más
$\log(1+n)$, con medias de grupo solo de antes de V. Entra como azul − rojo por línea, encima del modelo base del
brainstorming (draft + 85 variables + mapa; `brainstorm_base.py`). Cobertura: el 99 % de los jugadores de V y el
70 % de los del ajuste tienen historial de hábitos.

| Modelo previo (logística, misma partición) | Bloques A2-A4, Δ media [IC] | V Master+ (AUC / acierto) | Δ V [IC 95 %] |
| :--- | :---: | :---: | :---: |
| Base (draft + 85 variables + mapa) | — | 0,6531 / 61,1 % | — |
| + hábitos progresivos | −0,0003 [−0,0017; +0,0009], 2/3 | 0,6530 / 61,1 % | −0,0001 [−0,0012; +0,0014] |

**Decisión.** Los hábitos no se suman al modelo de victoria: no aportan nada, con el IC centrado en 0. Repite lo de
§9.S.30: saber *cómo* juega alguien no dice si va a *ganar*. Lo que predice es *qué tan bien* le viene yendo, y eso
ya está en el margen y en `perf_*`. La huella mejorada sirve para lo que fue pensada, reconocer personas y
detectar cambios de mano. Sigue el nivel 3 (detección de cambio sobre secuencias empalmadas), con LDA o la red sobre
estadísticas + hábitos.

*Alternativas descartadas.* Probar LDA o la red como entrada del modelo de victoria: equivalen a variables que ya
están, o a una representación entrenada para otra tarea. Integrarlo al escalón 9 completo: en esta máquina faltan 5
de sus 7 cachés (se armaron en la de escritorio). Igual, si sobre la base da −0,0001, no hay razón para esperar otra
cosa encima del escalón 9, que agrega todavía más información del jugador.

### 10.5. Niveles 2 y 3: verificación con jugadores no vistos y detección de cambio de persona (2026-11-19)

**Pedido.** Correr los niveles 2 y 3 del protocolo (§10.1) con las representaciones de §10.4; el nivel 4 queda para
después. Script: `scripts/analysis/huella_niveles.py` (~27 min), salidas en `data/analysis/huella/nivel2.csv` y
`nivel3.csv`. Como en §10.4, la mitad de los jugadores aprende los pesos o la red y la otra mitad se evalúa. Son dos
conjuntos de datos: las 23 estadísticas del train Master+ (8.997 jugadores) y estadísticas + hábitos en las
partidas con timeline completo (8.996 jugadores). La red se entrena aparte para cada conjunto (la de §10.4, con k
al azar entre 3 y 20). Todo es descriptivo, como el nivel 1.

**Nivel 2: verificación.** En el nivel 1, el AUC por fila compara a cada jugador solo contra los demás, sin umbral
común. Acá la pregunta es la de un sistema de verificación real: dado un par (perfil A del jugador $i$, perfil B del
jugador $j$), ¿es la misma persona? Hay un **único umbral** sobre la distancia para todos los pares. Los pares
genuinos son los $n$ de la diagonal y los impostores los $n(n-1)$ de afuera. Con campeones distintos entre A y B se
reportan:

* el ROC-AUC sobre todos los pares;
* el EER, la tasa en la que las falsas aceptaciones igualan a los falsos rechazos;
* el TAR@FAR = 1 %, la fracción de genuinos aceptados cuando se acepta solo 1 % de impostores.

| k | Euclídea, 23 estadísticas | LDA, 23 est. | Red, 23 est. | LDA, est. + hábitos | **Red, est. + hábitos** |
| ---: | :--- | :--- | :--- | :--- | :--- |
| 5 | 0,636 · EER 40 % · 3 % | 0,716 · 35 % · 4 % | 0,741 · 33 % · 8 % | 0,764 · 30 % · 7 % | **0,808 · 27 % · 14 %** |
| 10 | 0,709 · 35 % · 5 % | 0,797 · 28 % · 7 % | 0,837 · 24 % · 15 % | 0,834 · 24 % · 15 % | **0,892 · 19 % · 26 %** |
| 20 | 0,795 · 28 % · 10 % | 0,867 · 21 % · 14 % | 0,905 · 18 % · 26 % | 0,892 · 19 % · 28 % | **0,941 · 13 % · 42 %** |
| 40 | 0,867 · 21 % · 20 % | 0,919 · 16 % · 26 % | 0,949 · 12 % · 42 % | 0,925 · 15 % · 45 % | **0,963 · 10 % · 58 %** |

Cada celda es ROC-AUC · EER · TAR@FAR=1 %. Con impostores de mismo tier y misma línea, el ROC-AUC cae entre 0,003 y
0,012 en todas las variantes. **La red generaliza a jugadores no vistos** (la pregunta del nivel 2), y con hábitos
es la mejor en todos los $k$.

**Nivel 3: detección de cambio de persona sobre secuencias armadas.** Se arma una secuencia por cuenta con
jugadores no vistos que tengan 60 partidas o más. El perfil de la cuenta es la media de sus primeras 40 partidas.
Después siguen 20 partidas de uno de tres tipos:

* *control*: las siguientes del mismo jugador, con sus cambios naturales de campeón;
* *boosting*: las de otro jugador no visto de la misma línea principal y tier igual o mayor, desde un punto al azar
  de su historia;
* *compartida*: alternadas entre el dueño y ese otro jugador.

El puntaje en el paso $t$ es la distancia entre la media de las últimas $\min(t, w)$ partidas nuevas y el perfil,
en la representación elegida. El umbral $\tau_t$ de cada paso es el cuantil 99 % de los controles en ese mismo paso,
así que hay **1 falsa alarma cada 100 partidas** por construcción. Ventanas de $w$ = 5 y 10. Hay 3.399 cuentas con
las 23 estadísticas y 2.608 con estadísticas + hábitos.

| Representación ($w$ = 10 salvo que se indique) | Boosting en 5 / 10 / 20 partidas | Demora mediana | Boosting por tercil de distancia de estilo (cerca / medio / lejos) | Compartida en 20 |
| :--- | :--- | :---: | :--- | :---: |
| Euclídea, 23 estadísticas | 11 / 24 / 38 % | 8 | 10 / 28 / 75 % | 11 % |
| LDA, 23 estadísticas | 29 / 44 / 59 % | 6 | 17 / 63 / 98 % | 18 % |
| Red, 23 estadísticas | 33 / 52 / 68 % | 6 | 31 / 75 / 99 % | 24 % ($w$ = 5: 30 %) |
| LDA, estadísticas + hábitos | 23 / 37 / 47 % | 6 | 6 / 38 / 97 % | 13 % |
| **Red, estadísticas + hábitos** | **46 / 67 / 81 %** | **5** | **51 / 92 / 100 %** | 31 % (**$w$ = 5: 43 %**) |

La distancia de estilo entre el dueño y el que entra se mide entre sus perfiles completos en la misma
representación.

**Lectura.**

1. **El boosting se detecta.** Con la red sobre estadísticas + hábitos, el 81 % se detecta antes de 20 partidas y
   la mitad antes de la quinta, con 1 falsa alarma cada 100 partidas de controles. La huella vieja detectaba el 38 %.
2. **La salida honesta es la curva por distancia** (§10.1): si el que entra juega distinto, se detecta casi siempre
   (100 %). Si juega parecido, la mitad de las veces. Ningún detector de estilo puede ver un cambio entre dos
   personas que juegan igual.
3. **La cuenta compartida es más difícil.** Con partidas alternadas, la media de la ventana queda a mitad de camino
   entre los dos estilos, y la ventana corta ($w$ = 5) rinde mejor (43 % contra 31 %). Un detector que mire la
   dispersión entre partidas, y no solo la media, debería servir más para ese caso. Queda propuesto.
4. **La LDA con hábitos rinde peor que la LDA sola en el nivel 3** (47 % contra 59 %), aunque en los niveles 1 y 2
   rinde mejor. La LDA se ajusta para separar perfiles de muchas partidas. En ventanas de 5-10, los hábitos
   binarios (orden de habilidades) tienen mucha varianza partida a partida, y con pesos lineales grandes inflan el
   ruido de los controles, lo que sube el umbral. La red, entrenada con $k$ desde 3, aprende a no confiar en ellos
   con pocas partidas.

**Decisión.** El detector para el nivel 4 (casos reales) es la red contrastiva sobre estadísticas + hábitos, con
ventana de 5 partidas y umbral calibrado en los controles. Salvedades para el informe:

* las secuencias son armadas: un booster real puede imitar el pool o el estilo del dueño;
* el umbral se calibró con controles de la misma población;
* los hábitos necesitan timelines completos, que hoy cubren el 39 % de la base.

*Alternativas descartadas.* Umbral por cuenta en vez de global: con 40 partidas de historia no alcanza para estimar
la variabilidad propia de cada una. CUSUM sobre los puntajes: equivale a una ventana creciente; con estos datos, la
ventana fija de 5-10 ya da la demora mínima (mediana de 5 partidas), y la prioridad era comparar representaciones.

### 10.6. EDA unificado de reconocimiento de jugadores (2026-11-19)

**Pedido.** Unificar el EDA de la huella y el del clustering en uno solo, de "reconocimiento de jugadores", con todo
lo visto en cada parte; el usuario acota después. Resultado: `notebooks/eda_reconocimiento_jugadores.ipynb`,
generado por `scripts/analysis/build_eda_reconocimiento_nb.py --ejecutar` (65 celdas, ~10 min).

**Estructura.** Siete partes, ordenadas por pregunta y no por cuándo se hizo cada cosa:

1. ¿Hay tipos de jugadores? Clustering global y por línea, e-means (§9.S.53), UMAP + HDBSCAN (§9.S.58).
2. ¿Cada uno tiene un estilo propio? El EDA de la huella (§10.2-10.3).
3. Mejoras (§10.4).
4. Nivel 2 (§10.5).
5. Nivel 3 (§10.5).
6. ¿Sirve para predecir quién gana? (§9.S.23, §9.S.30, §10.4).
7. Qué se puede afirmar y qué falta, con la redacción sobre boosting y la tasa base.

**Una sola fuente.** En la primera versión, las partes I y II se copiaban de `clustering_jugadores.ipynb` y
`eda_huella.ipynb` en cada corrida. El mismo día, a pedido del usuario, esos dos notebooks y el generador
`build_eda_huella_nb.py` se borraron porque ya no se usaban. Antes de borrarlos, sus celdas se embebieron en
`build_eda_reconocimiento_nb.py`, con los títulos renumerados (I.x, II.x), y se verificó que el notebook generado
quedara idéntico, celda por celda (65 de 65). El generador nuevo es ahora la única fuente. Los originales siguen en
el historial de git. Las partes III a VI no recalculan nada: leen los CSV de `huella_mejoras.py`,
`huella_niveles.py` y `huella_previo.py`, para que el notebook tarde minutos y no horas.

*Alternativas descartadas.*

* Reescribir las celdas de origen en el generador nuevo: duplica código que ya diverge (§9.S.18 separó el
  clustering justamente para no mezclar fuentes).
* Recalcular las mejoras y los niveles dentro del notebook: más de 40 minutos de CPU por corrida.
* Incluir los CSV del UMAP: se corrieron en la PC de escritorio y no están en esta máquina. La sección I.4 muestra
  la tabla documentada y lee los CSV si aparecen.

**Revisión editorial y adaptación para presentación (2026-11-19).** La versión original (65 celdas, 4.473 palabras en
markdown, 23 figuras) arrastraba tres estilos visuales distintos (tema oscuro heredado en el clustering, matplotlib crudo
en la huella y seaborn en las partes nuevas), doble introducción, tablas sin formato (`DataFrame.round()`), notas
internas de desarrollo y cifras desactualizadas del split viejo del 01/09 en las conclusiones de la huella (887k filas /
33k jugadores / mediana 6 frente al train Master+ vigente de 1,24 M filas / 8.997 jugadores / mediana 109), además de
inconsistencias menores (10 vs. 15 variables en §1.1, escapes literales `\u00ed`/`H  pprox 1` y celda vacía de UMAP).

Se reescribió `scripts/analysis/build_eda_reconocimiento_nb.py` para alinear el notebook al estándar visual y narrativo
de `eda_modelos_ganador.ipynb` y `eda_entre_fotos.ipynb`:

* **Reducción de verbosidad y foco narrativo:** de 65 a **35 celdas** (21 markdown, 14 código), de 4.473 a **2.294
  palabras** (−49 %) y de 23 figuras heterogéneas a **12 figuras compuestas** de alta densidad informativa, con
  numeración unificada `1.` a `7.` y sin notas internas ni duplicaciones.
* **Identidad visual unificada:** una sola celda inicial fija el lienzo claro `#fcfcfb`, paleta categórica
  `CATEGORICAL` apta para daltonismo, títulos a la izquierda y etiquetas en español legible para las 23 estadísticas y
  18 hábitos. Se corrigieron todas las superposiciones visuales (repulsión iterativa en los biplots PCA, ejes y
  anotaciones de cobertura, espaciado de la matriz de correlación y ubicación de leyendas).
* **Mejoras analíticas clave para presentación:**
  1. En §1.1, el panel D reemplaza un scatter genérico por la **composición de cada cluster global por rol principal**,
     mostrando visualmente que K-Means global ($k = 2$, silhouette 0,483) solo separa **Soporte (94 %)** del resto
     (98,6 %). En §1.2, el panel 6 grafica las curvas de silhouette $k = 2..6$ de los 5 carriles (todos $< 0{,}25$ y
     decrecientes).
  2. En §2.2, la descomposición de varianza total y el ICC en el residuo comparten eje vertical ordenados por ICC
     (`Visión / min` 0,245 en Master+ vs. 4,5 % en varianza bruta; `Muertes / min` 9,3 % absoluto).
  3. En §2.4 se incorpora el baseline **`embedding`** (además de `crudo`, `elo_linea` y `pool`) en el gráfico y tabla
     del Nivel 1, mostrando que cae al nivel del rol (0,602) cuando cambian los campeones.
  4. En §§4 y 5 se usa una gramática visual fija (color = método, trazo discontinuo = 23 estadísticas, continuo =
     estadísticas + hábitos) y el Nivel 3 muestra la ventana elegida para el Nivel 4 (**$w = 5$**, boosting 81,1 % y
     cuenta compartida 43,3 %) junto con la comparación contra $w = 10$ (31,2 %).
* **Tablas estilizadas:** todas las tablas usan `Styler` con formato numérico/porcentual, título (`caption`), índices
  limpios y resaltado `RESALTE` (`#d8e8f8`), reemplazando los volcados crudos de 24 filas del Nivel 3 por un resumen de
  6 filas sin saltos horizontales.

**Segunda ronda de ajuste para presentación y red contrastiva con hábitos en §3 (2026-11-19).**
* Se agregó `huella_mejoras.py --parte red_habitos` (`data/analysis/huella/mejoras_red_habitos.csv`), que evalúa la
  misma red contrastiva de `huella_niveles.py` sobre **estadísticas + hábitos** (41 variables, `resid_timeline.pkl`,
  misma partición por jugador y campeones distintos): **AUC 0,890** ($k=10$, IC `[0,885; 0,895]`), **0,941** ($k=20$,
  IC `[0,937; 0,945]`, difícil `0,936`, Top-1 `8,7 %`), **0,959** ($k=30$), **0,966** ($k=40$) y **0,971** ($k=50$,
  Top-1 `28,5 %`).
  *Corrección (2026-11-23, §9.S.77 punto 4):* con el residuo ajustado solo con los jugadores de entrenamiento
  (y en la base de timelines del Drive, 62.065 partidas) da 0,892 [0,887; 0,898], 0,943 [0,938; 0,948], 0,959,
  0,963 y 0,975 (Top-1 39,9 % con $k=50$); con el residuo global en esa misma base, 0,897 / 0,945 / 0,958 / 0,970 / 0,973.
* En `build_eda_reconocimiento_nb.py`: se quitaron Mahalanobis y los LDAs intermedios de timelines (`23 est. · LDA` y
  `solo hábitos · LDA`), dejando en §3 la escalera completa `Euclídea → LDA → Red contrastiva → Est. + hábitos (LDA) →
  Est. + hábitos (Red contrastiva)` tanto en la tabla como en los paneles A y B; en §4 se explicitaron ROC-AUC, EER y
  TAR @ FAR = 1 % en títulos y texto, y se simplificó la tabla a $k = 20$ y $k = 40$ (sin `k=5`, `k=10` ni `Máx. ΔAUC
  dif.`); en §5 se enmarcó la detección como **reconocimiento de patrones de comportamiento** con una tabla esquemática
  de las secuencias de 60 partidas (`Control`, `Boosting`, `Cuenta compartida`).

*Alternativas descartadas.* Conservar los 25 histogramas crudos de estadísticas por línea y la grilla de 5 hexbins PCA
por línea principal: duplican información ya resumida de forma más directa por la descomposición de varianza (§2.2) y el
barras de $R^2$ por tier y línea (§2.3), alargando innecesariamente la presentación. Mantener el tema oscuro en el
clustering: quiebra la continuidad visual al presentar junto con `eda_modelos_ganador.ipynb` y `eda_entre_fotos.ipynb`.

### 10.7. Reconocimiento de patrones intra-cuenta: clustering no supervisado de huella + winrate (2026-11-19)

**Pedido.** Reemplazar la Sección 5 del notebook (secuencias sintéticas empalmadas 40 + 20 del Nivel 3, §10.5) por el
enfoque propuesto por el usuario: buscar de forma **no supervisada dentro de cada cuenta real** subgrupos de partidas
con huella muy distinta al resto del perfil y cruzar esa separación estilística con el **winrate** (distinguiendo
candidatas a *boosting* de *cuentas compartidas* y descartando rachas buenas del mismo jugador).

**Implementación (`scripts/analysis/huella_patrones_cuenta.py`).** Sobre las partidas Master+ con timeline completo
(`data/analysis/huella/resid_timeline.pkl`, 803.480 filas, 8.996 jugadores, 41 variables residualizadas por
`(campeón, línea, resultado)`):

1. **Embeddings fuera de muestra (2-fold por jugador):** se entrenan dos redes contrastivas (`41 → 128 → 128 → 32`,
   InfoNCE, 3.000 pasos) en mitades disjuntas de jugadores (`ajuste` y `evalu`). Cada cuenta se proyecta exclusivamente
   con la red que nunca la vio durante el entrenamiento.
2. **Chequeo de ortogonalidad al resultado (`win`):** en los 7.068 jugadores con $\ge 15$ victorias y $\ge 15$ derrotas,
   la distancia euclídea entre $\text{emb}(\bar x_{\text{win}})$ y $\text{emb}(\bar x_{\text{loss}})$ tiene media
   **0,6294** frente a **0,6060** en una partición aleatoria del mismo tamaño (**ratio 1,039**). Como las variables
   están residualizadas dentro de `(campeón, línea, resultado)`, ganar o perder no desplaza la huella; por tanto, un
   subgrupo separado solo por estilo y con winrate elevado no es un artefacto de "partidas ganadas".
3. **Clustering intra-cuenta no supervisado:** en cada cuenta con $\ge 50$ partidas ($N = 5.843$ en todos los roles;
   $N = 4.287$ restringiendo solo al rol principal), sus partidas cronológicas se agrupan en bloques no superpuestos de
   5 partidas y se aplica **K-Means ($k = 2$, mínimo 2 bloques = 10 partidas por subgrupo)** únicamente sobre los
   embeddings de 32D (sin ver `win`, fechas ni horarios). Se mide la distancia entre los perfiles agregados de ambos
   subgrupos (`d_estilo`), el silhouette medio entre bloques (`silhouette`) y el salto de winrate del subgrupo
   minoritario contra el mayoritario ($\Delta\text{WR}$ y test exacto de Fisher unilateral $p_{\text{WR}}$).
4. **Matriz 2×2 y corroboración independiente:** clasificando con `estilo_distinto` (`d_estilo ≥ p95` y
   `silhouette ≥ p90`) y `wr_alto` ($\Delta\text{WR} \ge +15\ \text{pp}$ y $p_{\text{WR}} < 0{,}05$), y midiendo dos
   señales que el clustering nunca vio —distancia de variación total de franja horaria en ART (`TVD horario`, 6 franjas
   de 4 h) y `contigüidad` temporal del subgrupo— sobre $N = 5.843$ cuentas reales:

| Cuadrante | Cuentas (%) | `d_estilo` | Silhouette | WR base → subg. ($\Delta\text{WR}$) | Cambio horario (TVD) | Contigüidad |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Normal / resto | 5.576 (95,43 %) | 0,99 | 0,12 | 51,2 % → 50,0 % (−1,2 pp) | 0,21 | 0,51 |
| **Racha buena (misma huella)** | **188 (3,22 %)** | 0,98 | 0,13 | 42,4 % → 67,5 % (+25,1 pp) | **0,22** | **0,48** |
| **Candidata a compartida** | **71 (1,22 %)** | 1,44 | 0,24 | 49,7 % → 48,9 % (−0,8 pp) | **0,31** | **0,71** |
| **Candidata a boosting** | **8 (0,14 %)** | 1,45 | 0,25 | 36,5 % → 71,6 % (+35,1 pp) | **0,28** | **0,77** |

*Corrección (2026-11-23, §9.S.77 punto 5).* La tabla anterior usa el residuo global: la media y el desvío de cada
grupo campeón-línea-resultado incluían a los jugadores analizados. Ahora `huella_patrones_cuenta.py` normaliza cada
pliegue solo con los jugadores con que se entrena su red; `--residuo global` reproduce la versión anterior. En la
base de timelines del Drive (62.065 partidas, otra base que la de esta tabla), con la misma partición y todos los
roles: 3.684 → 3.628 cuentas; rachas 3,07 % [2,56; 3,67] → 3,14 % [2,62; 3,76]; compartidas 1,14 % [0,84; 1,54] →
0,88 % [0,63; 1,24]; boosting 3 → 3; cociente de ortogonalidad 1,033 [1,027; 1,039] → 1,045 [1,038; 1,051];
$r$(`d_estilo`, $\Delta$WR) −0,012 → +0,011, ambos con IC que cruzan 0. Los perfiles de cada cuadrante (horario y
contigüidad) no cambian. El cambio por cuenta es del mismo tamaño que reentrenar con otra semilla (94,8 % contra
95,2 % de cuentas en la misma categoría). La consecuencia: las proporciones y perfiles son estables, pero la lista
puntual de candidatas depende de la semilla de la red (las compartidas coinciden con Jaccard ~0,4 entre semillas).
Las conclusiones de esta sección siguen en pie como filtro por cuadrante, no como lista fija de cuentas.

**Lectura y control de rol.**
* La correlación global entre `d_estilo` y $\Delta\text{WR}$ es prácticamente nula ($-0{,}028$ en todos los roles;
  $-0{,}023$ en rol principal), confirmando la independencia estadística entre estilo y resultado. Las 188 rachas buenas
  tienen el mismo horario (0,22) y contigüidad (0,48) que las cuentas normales (0,21 / 0,51) y se descartan limpiamente.
* En contraste, las candidatas a compartida y a boosting presentan saltos claros en horario (0,31 y 0,28) y contigüidad
  (0,71 y 0,77), con correlación positiva `d_estilo` vs. cambio de horario incluso dentro del mismo rol principal
  ($r = +0{,}163$).
* Cambiar de rol aumenta moderadamente `d_estilo` (0,95 cuando el subgrupo mantiene el rol vs. 1,18 cuando cambia,
  $r = 0{,}304$), por lo que los resultados informan `% mismo rol` y verifican la persistencia del fenómeno al 100 %
  en el mismo rol (ej. Caso A: salto escalón de `1,42–1,52` con 75 % WR en los primeros 4 bloques a `0,39–0,60` con
  43 % WR en los 6 siguientes, `silhouette = 0,41`, 100 % mismo rol).
* Salidas guardadas en `data/analysis/huella/{patrones_cuentas,patrones_cuentas_todos_roles,patrones_casos}.csv` y
  `patrones_ortogonalidad.json`. Todas las cuentas se presentan anonimizadas en la Sección 5 de
  `notebooks/eda_reconocimiento_jugadores.ipynb` (36 celdas, 0 errores).

*Alternativas descartadas.*
* Mantener las secuencias sintéticas 40 + 20 del Nivel 3 en la Sección 5 del notebook: un jugador B tomado de su propio
  elo juega alrededor de 50 % WR y la ventana móvil sobre media fija no separa rachas de boosting real ni captura
  subgrupos no contiguos (descartado a pedido del usuario; `huella_niveles.py` se conserva porque también genera
  `nivel2.csv`).
* Clustering sobre partidas individuales ($k = 1$): el 60–80 % de la varianza por partida es ruido y la red contrastiva
  fue entrenada sobre perfiles de $k \ge 3$; agrupar en bloques de 5 partidas ubica al modelo en su régimen operativo.
* Restringir el escaneo exclusivamente al rol principal descartando cambios de rol: en boosting real es frecuente que
  el booster juegue su propio rol de carry en la cuenta del cliente (7 de las 8 candidatas con WR 63–80 % combinan
  cambio de rol parcial/total y horario distinto); por eso se presenta el escaneo completo acompañado del control
  explícito de `% mismo rol` y los casos 100 % mismo rol.

### 10.8. Perfil del tramo boosteado en el modelo de predicción de victoria (2026-11-19)

**Pedido.** Evaluar si la predicción del modelo previo de victoria mejora cuando, ante una partida con sospecha de
*boosting* (huella estilística muy distinta al historial previo + winrate reciente elevado), se reemplaza el perfil
histórico completo de ese jugador por **únicamente** las partidas previas de su tramo boosteado.

**Diseño estrictamente causal (`historial/clustering_huella/boosting_en_victoria.py`).** Sin fuga del resultado de la partida
evaluada $t$ y sin tocar `data/splits/test`:

1. **Modelo base congelado:** logística híbrida de `brainstorm_base.py` (draft + 85 variables por carril + mapa de LP,
   y variante sin mapa), entrenada y evaluada estrictamente fuera de muestra en `A2`, `A3`, `A4` y `V Master+`
   ($30.288$ partidas Master+ fuera de muestra). Al sustituir el perfil de un jugador en inferencia no se reajustan
   pesos; se aplica la actualización exacta del logit:
   $$\Delta\text{logit}_m = \left(\frac{x^{\text{mod}}_m - x^{\text{base}}_m}{\sigma}\right)^\top w_{x}$$
2. **Disparador causal en línea (solo partidas $< t$):** usando las redes contrastivas 2-fold fuera de muestra sobre
   41 variables (`data/analysis/huella/red_2fold_timelines.pt`), para cada jugador con $\ge W + 15$ partidas previas
   con timeline ($W \in \{8, 10, 15\}$) se mide $d_{\text{estilo}}(t) = \|\text{emb}(\bar x_{t-W:t-1}) -
   \text{emb}(\bar x_{1:t-W-1})\|_2$, $\text{WR}_{\text{rec}}$ y $\Delta\text{WR} = \text{WR}_{\text{rec}} -
   \text{WR}_{\text{ant}}$. Antes de empezar $t$ se comparan cuatro regímenes:
   * *Sospecha de boosting:* $d \ge p_{90}$ (más variantes $p_{95}$ y cambio de rol $\ge 50\ \%$),
     $\text{WR}_{\text{rec}} \ge 0{,}70$ y $\Delta\text{WR} \ge +0{,}20$.
   * *Control racha buena:* $d < p_{50}$, $\text{WR}_{\text{rec}} \ge 0{,}70$ y $\Delta\text{WR} \ge +0{,}20$.
   * *Control cambio de huella sin WR alto:* $d \ge p_{90}$ y $|\Delta\text{WR}| < 0{,}12$.
   * *Control normal:* $d < p_{50}$ y $|\Delta\text{WR}| < 0{,}12$.
3. **Reemplazo de perfil:** las 45 variables históricas del jugador (`acum_*`, `v20_*`, `wr_acum`, `wr_v20`,
   `logn_tot`, `logn_champ`, `estreno`, `wr_champ`, `share_champ`, `share_lane`, `offrole`) se recalculan
   exclusivamente con sus últimas $W$ partidas previas (`perfil_desde_indices`).

**Resultados causales en línea ($W = 15$, modelo con mapa; $W = 8$ y $W = 10$ como sensibilidad).**
Con ventanas cortas ($W = 8, 10$), la varianza muestral anula el efecto en boosting ($\Delta\text{LL} = -0{,}0002$ a
$+0{,}0005$) y perjudica los controles ($\Delta\text{LL} = +0{,}0017$ a $+0{,}0040$). Con **$W = 15$** surge una
progresión monótona y consistente entre `A2–A4` y `V Master+`:

| Régimen causal ($W = 15$) | Partidas (`A2–A4` / `V`) | $\Delta\text{LL}$ [IC 95 %] | $\Delta\text{AUC}$ (`A2–A4` / `V`) | Acierto base → reset |
| :--- | :---: | :---: | :---: | :---: |
| **Sospecha boosting ($p_{90}$, WR $\ge 70\ \%$)** | **962** (443 / 519) | **$-0{,}0039$ $[-0{,}0086;\ +0{,}0002]$** | **$+0{,}0100$** ($+0{,}0136$ / $+0{,}0076$) | **62,9 % → 63,9 %** |
| Sospecha boosting (sin mapa de LP) | 962 (443 / 519) | **$-0{,}0042$ $[-0{,}0087;\ -0{,}0002]$** | **$+0{,}0096$** | **62,3 % → 64,0 %** |
| Control: Racha buena ($d < p_{50}$, WR $\ge 70\ \%$) | 4.285 (1.802 / 2.483) | $-0{,}0001$ $[-0{,}0023;\ +0{,}0020]$ | $+0{,}0009$ ($+0{,}0042$ / $-0{,}0016$) | 60,9 % → 61,4 % |
| Control: Estilo sin WR alto ($d \ge p_{90}$) | 9.852 (4.224 / 5.628) | $+0{,}0018$ $[+0{,}0002;\ +0{,}0035]$ | $-0{,}0039$ ($-0{,}0030$ / $-0{,}0043$) | 60,7 % → 60,6 % |
| Control: Normal ($d < p_{50}$, $|\Delta\text{WR}| < 12\ \text{pp}$) | 25.580 (11.051 / 14.529) | $+0{,}0056$ $[+0{,}0041;\ +0{,}0071]$ | $-0{,}0106$ ($-0{,}0100$ / $-0{,}0111$) | 60,2 % → 59,9 % |

**Mecanismo (ablación por grupo de variables en $W = 15$):**
* **No es un efecto genérico de recencia:** resetear `Solo margen` en jugadores normales **empeora** significativamente
  ($\Delta\text{LL} = +0{,}0014\ [+0{,}0007;\ +0{,}0022]$, $\Delta\text{AUC} = -0{,}0029$, igual en `A2–A4` y `V`),
  mientras que bajo *Sospecha boosting* sube `P(equipo jug)` en $+1{,}25\ \text{pp}$ y aporta la mayor parte de la
  mejora ($\Delta\text{LL} = -0{,}0029$, $\Delta\text{AUC} = +0{,}0070$).
* **Experiencia con el campeón (`logn_champ`, peso logístico $+0{,}130$):** truncar la historia de campeones castiga
  fuertemente a los controles normales ($\Delta\text{AUC}$ cae de $-0{,}0037$ en *reset limpio* a $-0{,}0106$ en *reset
  total*) y a las rachas ($-0{,}0066$), porque sus partidas antiguas reflejan maestría real. Bajo *Sospecha boosting*,
  en cambio, el *reset total* supera al *reset limpio* ($\Delta\text{AUC} = +0{,}0100$ vs. $+0{,}0075$, acierto
  `63,9 %` vs. `62,8 %`), coherente con que el historial previo del dueño original deja de representar al piloto nuevo.
* **Impacto global sobre todo `V Master+` ($N = 16.428$):** la regla causal $W = 15$ solo se dispara en **519 partidas
  (3,16 %)**. Promediado sobre el total de `V`, el cambio global es despreciable ($\Delta\text{LL} = -0{,}00008\
  [-0{,}00025;\ +0{,}00010]$, $\Delta\text{AUC} = +0{,}00018$, de `0,65142` a `0,65160`). Esto explica por qué las
  variables de estilo/cambio no movían el AUC agregado (§6 y §9.S.30): la mejora existe *condicional a los lobbies con
  sospecha de boosting*, pero la baja prevalencia del fenómeno en Master+ la diluye en el conjunto completo.

**Estudio de caso retrospectivo sobre las 8 candidatas de §5.** Dos de las 8 cuentas detectadas en §10.7 caen dentro
de `A2–A4 + V` con $\ge 5$ partidas previas dentro de su tramo boosteado ($n = 20$ partidas, WR real `80,0 %`):
*Cuenta #2* (TOP base 38,3 % → MID 80,0 % en `V`) y *Cuenta #5* (MID/TOP/SUPP base 33,3 % → Jungla/Lee Sin 80,0 % en
`A2–A4`). Al evaluar cada partida usando solo las previas ($< t$) dentro del tramo boosteado:
* **Con mapa de LP:** `P(equipo jug)` sube de **46,1 % → 53,8 % (+7,7 pp)** en ambas cuentas por igual (`+7,7 pp`),
  $\text{LL}$ cae de **0,6702 → 0,5788** ($\Delta\text{LL} = -0{,}0914\ [-0{,}1449;\ -0{,}0330]$) y el acierto pasa de
  **60,0 % → 65,0 %**.
* **Sin mapa de LP:** `P(equipo jug)` sube de **45,4 % → 53,3 % (+7,9 pp)**, $\text{LL}$ cae de **0,6847 → 0,5898**
  ($\Delta\text{LL} = -0{,}0950\ [-0{,}1484;\ -0{,}0358]$) y el acierto pasa de **60,0 % → 70,0 %**.

*Salvedad:* este estudio de caso constituye una **cota superior retrospectiva** (las 8 cuentas fueron elegidas con el
clustering de toda la trayectoria, usando WR del tramo completo), mientras que la prueba con $W = 15$ es 100 % causal.
Salidas en `data/analysis/huella/boosting_en_victoria*.{csv,json}`.

*Alternativas descartadas.*
* Reentrenar pesos de la logística sobre las filas sustituidas: con $< 1.000$ partidas disparadas sobreajusta y mezcla
  el efecto de limpiar el perfil con una recalibración de pesos.
* Alimentar el clustering intra-cuenta de §5 como variable global del modelo de victoria: introduce fuga del futuro al
  agrupar toda la trayectoria de la cuenta.

### 10.9. Anatomía de la huella por subgrupo (§5) y clustering exploratorio sobre cuentas anómalas (2026-11-19)

**Pedido.** (1) Reemplazar el segundo gráfico y tabla de la Sección 5 de `notebooks/eda_reconocimiento_jugadores.ipynb`
(trayectorias bloque a bloque y tabla de 8 IDs) por la comparación directa de **cómo es la huella de cada parte del
perfil** (`Perfil base` vs. `Subgrupo alternativo`) en una **cuenta boosteada**, una **cuenta compartida** y una
**cuenta normal**; (2) explorar fuera del notebook si aplicar clustering (K-Means, Ward, GMM, HDBSCAN) sobre las cuentas
con sospecha de boosting / múltiples usuarios revela subestructura adicional.

**1. Huellas por parte del perfil en Sección 5 (`huella_patrones_cuenta.py` + `build_eda_reconocimiento_nb.py`).**
Se actualizaron las salidas de `huella_patrones_cuenta.py` (reutilizando `red_2fold_timelines.pt` para reproducibilidad
determinista entre scripts; conteos vigentes en $N = 5.843$ cuentas: 8 candidatas a boosting, 68 a compartida, 197
rachas buenas descartadas y 5.570 normales) agregando `patrones_huellas_ejemplos.csv` y `patrones_huellas_padron.csv`.
Para eliminar cualquier confusión por cambio de rol, los tres casos representativos son **100 % ADC** seleccionados por
reglas objetivas sobre `patrones_cuentas.csv` (única candidata a boosting en el mismo rol; compartida del mismo rol con
mayor volumen y WR parejo; normal más cercana a la mediana de $d_{\text{estilo}}$), comparados en 12 variables
residualizadas (7 de combate/economía + 5 de tienda/visión) mediante un dumbbell horizontal de 3 paneles (que muestra
`Base` y `Alt.`) y una tabla `Styler` compacta únicamente con los deltas ($\Delta = \text{Alt.} - \text{Base}$) más el
resumen del padrón completo:

* **Cuenta boosteada (ADC, 30 / 20 part., WR `43 % → 75 %`, `d = 1,44`, `TVD = 0,30`):** ruptura en bloque en
  **combate, economía y ritmo** (`Daño @15 +2,0σ`, `% Daño +2,0σ`, `Oro @15 +1,5σ`, `Visitas tienda +1,4σ`,
  `CS/min +1,3σ`, `Agresión @10 +1,2σ`, `Muertes/min +1,1σ`, `Visión/min −0,7σ`), con **media $|\Delta| = 1{,}06\sigma$**
  (9 de 12 variables con $|\Delta| \ge 0{,}60\sigma$). Incluso descontando `(campeón, línea, resultado)`, el piloto
  boosteado domina la fase de líneas (en las 8 candidatas del padrón, el signo del cambio es positivo en **7/8** para
  `Oro @15` $+0{,}64\sigma$ y `Placas @14` $+0{,}37\sigma$, y **6/8** para `Oro/min`, `Daño @15`, `CS @15` y `Kills/min`).
* **Cuenta compartida (ADC, 160 / 40 part., WR `56 % → 57 %`, `d = 1,44`, `TVD = 0,36`):** economía y combate casi
  equivalentes (0 de 7 variables llegan a $0{,}60\sigma$, coherente con el mismo WR), pero **hábitos opuestos de visión
  y gestión de tienda** (`Visión/min −1,3σ`, `Control wards @15 −1,3σ`, `Oro sin gastar +0,7σ`, `Visitas tienda −0,7σ`;
  **media $|\Delta| = 0{,}48\sigma$**). En las 68–71 compartidas del padrón, `Control wards @15` es la variable con mayor
  salto relativo respecto a lo normal ($0{,}65\sigma$ vs. $0{,}30\sigma$, $2{,}2\times$).
* **Cuenta normal (ADC, 65 / 65 part., WR `51 % → 54 %`, `d = 0,94`, `sil = 0,10`):** incluso aplicando el mismo
  K-Means $k = 2$, ambas mitades quedan prácticamente superpuestas (**media $|\Delta| = 0{,}22\sigma$**, **0 de 12**
  variables con $|\Delta| \ge 0{,}60\sigma$).

**2. Clustering exploratorio sobre cuentas con boosting / múltiples usuarios (fuera del notebook).**
Se aplicaron K-Means ($k = 2..6$), Ward, GMM y HDBSCAN sobre las cuentas anómalas (79 candidatas: 8 boosting + 71
compartidas; y muestra ampliada $p_{95}$ de $N = 293$) tanto **entre cuentas** (variables estructurales: `tvd_hora`,
`contiguidad`, `pct_main_alt`, `champ_jaccard`, `frac_alt`, `d_estilo`, `silhouette`, con y sin `delta_wr`, además de
$|\Delta|$ de las 41 variables de huella) como **dentro de cada cuenta** ($k = 2, 3, 4$):

* **Sin islas rígidas, pero con un eje conductual muy claro:** al igual que en el padrón completo (§1), el silhouette
  global entre cuentas anómalas es bajo ($0{,}15\text{–}0{,}22$ en K-Means/Ward) y HDBSCAN marca **40 % (32/79) como
  ruido**. Sin embargo, **K-Means ($k = 2$, sin ver `delta_wr`, sil. $0{,}181$)** separa dos regímenes netos:
  1. *Compartida intercalada en mismo rol* ($n = 30$, solo 1 boosting): baja contigüidad (`0,49`), mismo rol (`75 %`),
     mayor solapamiento de campeones (`Jaccard 0,39`) y $\Delta\text{WR} = -0{,}3\ \text{pp}$.
  2. *Ruptura continua con cambio de rol/pool* ($n = 49$, concentra **7 de las 8 boosting sin haber visto `delta_wr`**):
     alta contigüidad (`0,85`), cambio de rol (`mismo_rol = 28 %`), bajo solapamiento (`Jaccard 0,21`) y mayor
     separación (`d = 1,47`).
* **Subtipos en $k = 4$ sin `delta_wr` (sil. $0{,}179$):** distingue además *compartidas intercaladas en mismo rol*
  ($n = 20$, `contig 0,35`, `rol 82 %`, `TVD 0,34`), *préstamos continuos en horario opuesto* ($n = 12$, `contig 0,89`,
  `TVD 0,49`), *rupturas extremas con cambio de rol* ($n = 19$, `rol 16 %`, `d 1,56`, $\Delta\text{WR} +6{,}6\ \text{pp}$)
  y un bloque de **posibles falsos positivos / cambios del mismo dueño** ($n = 28$, `TVD 0,21` idéntico al de una cuenta
  normal, $\Delta\text{WR} +0{,}8\ \text{pp}$). Este cuarto subgrupo muestra que **exigir además un mínimo de cambio
  horario o baja superposición de pool** purificaría las alertas de cuenta compartida.
* **¿Hay evidencia de cuentas con 3 o más usuarios?** Evaluando $k = 2, 3, 4$ dentro de las 64 cuentas anómalas con
  $\ge 12$ bloques, Silhouette elige **$k = 2$ en 58/64 cuentas (91 %)** y en **0/64** $k \ge 3$ supera a $k = 2$ por
  más de $+0{,}04$: con bloques de 5 partidas no hay evidencia sólida de cuentas compartidas entre 3+ personas.

**3. Corroboración por cambio de rol y tipo de campeón en Sección 5 (`huella_patrones_cuenta.py`, `build_eda_reconocimiento_nb.py`).**
A pedido del usuario, se agregaron dos señales independientes adicionales al panel B y tabla resumen de §5 —ambas en
escala TVD $[0, 1]$ y nunca vistas por el clustering de huella—: `tvd_rol` (distribución sobre los 5 carriles) y
`tvd_tipo_camp` (distribución sobre **10 arquetipos de campeón** obtenidos con K-Means sobre 13 estadísticas de
`data/splits/train/champions_by_lane.csv`, `games_played >= 50`, más `tvd_subtipo_rol` con 3 subtipos por carril).
Sobre las 5.843 cuentas Master+:
* **Normal:** `Horario 0,21`, `Rol 0,19`, `Tipo camp. 0,23`, `Contig. 0,51`.
* **Racha buena (misma huella):** `0,21 / 0,20 / 0,24 / 0,48` (idéntica a normal en las 4 señales).
* **Candidata a compartida:** `0,30 / 0,48 / 0,45 / 0,70` (rol $2{,}5\times$, tipo de campeón $2{,}0\times$).
* **Candidata a boosting:** `0,25 / 0,58 / 0,51 / 0,76` (rol $3{,}0\times$, tipo de campeón $2{,}2\times$).
* **Control solo rol principal ($N = 4.287$, `tvd_rol = 0`):** el salto en tipo de campeón persiste incluso sin cambiar
  de carril (`Normal 0,15`, `Racha buena 0,15`, `Compartida 0,24`, `Boosting 0,33`; $r = +0{,}235$ con `d_estilo`).
* **Por qué se mantienen separadas de la huella y no mezcladas adentro:** incorporar el tipo de campeón dentro del
  embedding de huella haría disparar falsas alarmas cada vez que un mismo jugador prueba campeones o roles nuevos;
  mantenerlos como capas separadas y combinadas distingue entre cambio de meta del mismo jugador (cambia pool/rol pero
  no huella ni horario), cambio de persona en el mismo rol/pool (evidencia más fuerte) y cambio simultáneo de rol +
  tipo de campeón + huella + WR (ej. Cuenta #3: 274 partidas de Soporte Enchanter → 25 Skarner Jungla).
* Además se fijó `OMP_NUM_THREADS=1` en `huella_patrones_cuenta.py`, reduciendo su tiempo de ejecución de ~15 min a
  ~4 min con resultados idénticos bit a bit.

*Alternativas descartadas.*
* Graficar radar plots (*spider charts*) para las 12 variables de huella en §5: distorsionan áreas según el orden de los
  ejes y dificultan comparar `Base` vs. `Alt.` en tres paneles simultáneos; el dumbbell horizontal con eje $\sigma$
  compartido y separador temático permite lectura inmediata.
* Incluir el sub-clustering de las 79 cuentas anómalas en el notebook de presentación: los silhouettes ($0{,}15\text{–}0{,}22$)
  y el 40 % de ruido en HDBSCAN confirman que no son clusters duros, por lo que queda documentado como análisis
  exploratorio en la memoria técnica y no como hallazgo cerrado en la presentación.

## 11. Revisión metodológica previa a la entrega (2026-11-20)

La [revisión de los modelos del TP](docs/REVISION_MODELOS_20261120.md) contrasta las afirmaciones de los notebooks con su implementación y con predicciones guardadas. En la comparación central, el modelo de 320 columnas que puede prescindir del campeón elegido (§9.S.67) obtiene AUC 0,6369, el draft con atributos y cruces 0,5592 y el escalón 9 completo 0,6740 sobre las mismas 16.428 partidas de validación. El bloque «Jugadores» de §2 del EDA también incorpora variables que dependen del campeón actual.

El informe distingue dos usos históricos de datos reservados. Las reglas y logísticas de draft ya se evaluaron sobre el test de septiembre (§9.S.13 y §9.S.20). La caja por jugador del problema de reconstrucción de LP tuvo dos aperturas (§9.S.40). Son evaluaciones distintas. Los controles de esta revisión utilizaron el conjunto de entrenamiento, su ventana de validación y resultados ya publicados.

**Cómo se eligió el escalón 9.** La regla aplicada en §9.S.52 exigió una mejora puntual en V y permitió usar Diamond-V cuando uno de los intervalos incluía cero. Al pasar del escalón 7 al 9, los bloques anteriores dieron un IC de [−0,0001; +0,0018], V mejoró +0,0018 y Diamond-V +0,0019 en AUC. Ambas ventanas intervinieron en la decisión. La comparación entre ellas describe la sensibilidad al tier. Una confirmación independiente necesita otro conjunto reservado antes de elegir. [Regla y registro de la decisión](docs/BRAINSTORM_NOCHE.md)

**Disponibilidad de la foto en el modelo histórico de campeones y jugadores (§9.S.23).** Para generar `ganador_feats.pkl`, el código parte de la foto del 04/09 y actualiza el LP con cada resultado previo. Los coeficientes de esa actualización, sin embargo, se estiman con los cambios de LP y los conteos de victorias y derrotas entre el 04/09 y el 28/10. Por ello, reproducir así la fila «Campeones + Jugador» de la comparación de septiembre requiere estadísticas posteriores a la predicción. Un control temporal puede usar la foto fija o coeficientes conocidos antes de predecir. El escalón 9 publicado tiene su propia receta sin mapa mecánico. [Variables del modelo histórico](historial/ganador_primeros/ganador_features.py) · [Ajuste de coeficientes](src/features/elo_reverso.py) · [Tabla comparativa](scripts/analysis/modelos_train_v.py)

**Sensibilidad del escalón 9 al reloj de los cruces.** Para las diez columnas `MATCHUPS` se reconstruyeron en memoria los mismos conteos de campeón, contracruce y sinergia. En la variante A se actualizan al inicio de cada partida y en B cuando llega su fin estimado, $t_{fin}=t_{creación}+1000\,d_{segundos}$. En 7.569 de 16.428 partidas de V había un cruce idéntico todavía en juego en algún carril. Se reajustó la misma logística en 118.064 partidas, cambiando solo las columnas `157:167` de la matriz de variables adicionales. La variante A reprodujo exactamente esas columnas y las probabilidades guardadas. En V, AUC A/B fue 0,674033/0,673983, con $\Delta=-0{,}000050$ e IC 95 % pareado por partida [−0,000127; +0,000022] (300 réplicas). Las variables derivadas, incluido el HGB por línea, conservaron sus valores originales en este control.

**Sensibilidad a la población.** En la ventana de validación de 26.934 partidas, la mediana de tier disponible y la regla oficial de cinco jugadores del padrón admiten 16.194 partidas en común. La mediana incluye otras 234 y la regla oficial otras 887. Al evaluar el mismo ajuste A en las 17.081 partidas oficiales, el AUC fue 0,675101 y el acierto 62,68 %. La diferencia de población cambia el denominador y debe explicitarse al comparar validación y test.

**Predicción en vivo sobre las mismas partidas.** La intersección de los índices de validación de `pred_final.npz` y `pred_mt_a4_gpu_92k.npz` a los 10, 15 y 20 minutos contiene las 14.971 partidas elegibles al minuto 20 (91,13 % de V). En esas partidas los AUC son 0,661820 antes de jugar, 0,765189 al minuto 10, 0,820243 al 15 y 0,874981 al 20. Los aciertos respectivos son 61,69 %, 69,60 %, 74,10 % y 79,12 %. Las 748 predicciones más seguras al minuto 20 cubren el 4,55 % de toda V. Aciertan el 99,60 % con una mediana de 2,90 minutos por jugar, frente a 8,88 minutos entre todas las elegibles. Se usaron las predicciones en vivo archivadas, anteriores a la revisión del previo en §9.S.68.

**Instante observable.** En 800 partidas de V elegidas con semilla 20261120 se contrastó el timestamp del frame con el segundo nominal. La mediana del retraso del frame fue de 0,230 / 0,315 / 0,399 segundos a los 10 / 15 / 20 minutos. Hubo al menos una baja posterior al corte nominal e incluida por índice de frame en 6/794, 8/791 y 14/729 partidas activas, respectivamente. Estas frecuencias miden disponibilidad al instante de predecir. La base local de eventos tenía 62.706 IDs, todos del split oficial de entrenamiento, incluidos los 16.428 de V.

**Alcance de los arquetipos de campeón de §10.9.** Al cruzar las trece variables solicitadas por `arquetipos_campeon()` con el esquema que genera el exportador de entrenamiento, solo coinciden los promedios de bajas, muertes y asistencias. La función omite los nombres ausentes y calcula los grupos con las columnas restantes. La afirmación de §10.9 de que se usaron trece requiere corregir el esquema y repetir la comparación. La tabla de categorías de cuentas también agrupa como «resto» casos de cambio de estilo con caída de winrate. Para estimar la frecuencia de cada patrón hacen falta ambos signos del cambio y un control por las miles de pruebas simultáneas. [Selección de variables y categorías](scripts/analysis/huella_patrones_cuenta.py) · [Esquema del exportador](export_to_csv.py)

### Intervalos de confianza en las figuras

Los intervalos del 95 % son puntuales y condicionales a las predicciones archivadas o a los perfiles calculados. Para acierto, tasas de victoria y proporciones con conteos conocidos, se usa Wilson con centro $(\hat p+z^2/(2n))/(1+z^2/n)$ y semiancho $z\sqrt{\hat p(1-\hat p)/n+z^2/(4n^2)}/(1+z^2/n)$, con $z=1{,}96$. Para AUC y métricas de regresión con observaciones disponibles, el bootstrap toma los percentiles 2,5 y 97,5 de la métrica recalculada. Se remuestrean partidas en victoria y matchmaking, jugadores en LP y huella, y cuentas en los patrones intra-cuenta. Los contrastes pareados comparten los índices del remuestreo para cada variante. Cuando solo se dispone del AUC y del conteo de cada clase de la cohorte se usa la aproximación de Hanley–McNeil. El piloto de eventos conserva el IC pareado archivado de la diferencia de AUC.

Las bandas de una curva con ventanas solapadas se leen punto a punto. El remuestreo por partida aproxima las partidas como independientes, aunque puedan repetir jugadores. Para medias agregadas se usa $\bar x\pm1{,}96s/\sqrt{n}$ y para la mediana se aproximan los estadísticos de orden con cuantiles en $0{,}5\pm1{,}96/(2\sqrt{n})$. En el EDA, el IC de Fisher para la correlación transforma los límites de $\operatorname{arctanh}(r)\pm1{,}96/\sqrt{n-3}$ con la tangente hiperbólica.

En LP, las barras para el error agregado se calculan por jugador con las fotos del padrón y las predicciones guardadas. El intervalo individual de NGBoost es predictivo ($\mu\pm1{,}96\sigma$). Para «Elo al revés» se conserva el error histórico por rango sin agregar barras, porque ese modelo carece de predicciones archivadas por jugador. En huella se recuperaron los límites de AUC publicados en la tabla de mejoras de $k=10$ y $k=20$. El AUC difícil del Nivel 1 y el gráfico de verificación del Nivel 2 conservan sus puntos publicados, porque sus artefactos disponibles carecen de los intervalos correspondientes.

**Forma gráfica del intervalo: banda en las curvas, bigote en barras y puntos.** Cuando la figura une estimaciones con una línea (un eje ordenado: $k$, días, quintiles, deciles de probabilidad, momentos de la partida, niveles de mejora o escalones del modelo), el IC 95 % se dibuja como una banda sombreada del color de la serie entre $\ell(x)$ y $u(x)$ (`fill_between`, opacidad 0,12–0,16), interpolada linealmente entre puntos igual que la línea. Las barras de error en cada punto se descartaron para las curvas porque, con dos o más series cercanas, los bigotes se superponen y obligaban a desplazar las series en $x$ (como los ±0,06 que tenía el gráfico de desvío de rol), lo que falsea la abscisa; la banda conserva el $x$ exacto y deja ver de un vistazo dónde las series se solapan. La interpolación de la banda entre puntos es solo visual: el intervalo vale en cada punto estimado, en coherencia con la lectura punto a punto de arriba. Las barras, los gráficos de puntos sin línea (calibración del matchmaking por deciles, *forest plots*) y las estimaciones aisladas (intervalo predictivo de NGBoost en un instante) siguen con bigotes, porque ahí no hay un eje continuo que justifique interpolar. Se aplicó a la deriva temporal de la huella (§2.4 de `eda_reconocimiento_jugadores`), la progresión por nivel de mejoras de LP (`eda_entre_fotos`), la calibración del rating (`build_rating_nb.py --evaluacion`) y, en `eda_modelos_ganador`, a la progresión de modelos, la calibración del escalón 9, la facilidad y el acierto por quintil, el oro por momento y el desvío de rol.

### Correcciones aplicadas tras la revisión (2026-11-20 / 2026-11-24)

Por decisión explícita del usuario tras revisar `docs/REVISION_MODELOS_20261120.md`, se implementaron las siguientes correcciones metodológicas y editoriales (dejando para una corrida posterior en GPU la re-evaluación de eventos sobre el ensamble `6_Final_dEns10`):

1. **Definición única de Master+ ($\ge 5$ empadronados).** `es_master_plus()` en `src/models/validacion_temporal.py` pasó de la mediana de tier disponible a exigir `(tnum >= 1.0).sum(axis=1) >= 5`, igualando exactamente la regla oficial del split (`src/features/split.py`). Se actualizó `tests/test_validacion_temporal.py`.
2. **Bloque «Jugadores» limpio (320 columnas), tamaños de entrenamiento y nombre del Elo.** En `scripts/analysis/build_eda_modelos_nb.py`, `otp_nuevo` pasó a `JUGADOR_CAMPEON` y las 5 columnas dependientes del campeón dentro de `ROL_WR` (índices 2 y 9) y `TEMPO_OBJ` (índices 7–9) pasaron a `"Jugador × campeón"`, dejando exactamente las 320 variables puras de jugador y rol (idénticas a `matchmaking_calibracion.py`). Se explicitaron las diferencias de `n Train` en §1 y se reemplazó la denominación `"Kalman-Elo"` por `"Elo decreciente"` (conservando las claves internas de caché por compatibilidad).
3. **Reloj causal de cruces y eliminación de fugas de fotos de LP.** `construir_mpr()` en `scripts/analysis/brainstorm_g_07_matchups_parche_rematch.py` usa una cola de prioridad (`heapq`) indexada por fin estimado $t_{\text{fin}} = t_{\text{creación}} + \operatorname{round}(60\,000\,d_{\text{min}})$, volcando cada partida al estado solo cuando $t_{\text{fin}} \le t_{\text{creación}}(m)$. En `scripts/analysis/ganador_features.py` se eliminó el uso de la foto 2 (28/10) fijando paso simétrico previo $a = b = 20$ LP, y en `scripts/analysis/brainstorm_07_mmr_historico.py` se anuló el LP de la foto 1 en las filas estrictamente anteriores al 04/09 (`A[~D["post"]] = 0.0`).
4. **En vivo y remontadas `@15`.** Se eliminó la columna «Acierto en el 5/10/20 % más seguro» de la tabla en vivo de §9.S.60 (la cohorte por minuto y el índice de frame se mantuvieron por decisión del usuario). En §6.2 de `build_eda_modelos_nb.py` se enmascaró el oro faltante/cero a los 15 minutos (`ok15_nz_62 = np.isfinite(g15) & (g15 != 0)`) para que `NaN` no cuente como remontada.
5. **Arquetipos de campeón (17 columnas), ambos signos de $\Delta\text{WR}$ y FDR en §5 de la huella.** `arquetipos_campeon()` en `scripts/analysis/huella_patrones_cuenta.py` ahora exige con `assert` las 17 columnas reales de `data/splits/train/champions_by_lane.csv` (antes matcheaba solo 3). Se añadió la categoría explícita `Huella distinta + caída WR` y el ajuste Benjamini–Hochberg (`fdr_bh`): sobre $N = 5.843$ cuentas Master+, aparecen 8 candidatas a boosting ($\Delta\text{WR} = +30{,}7\ \text{pp}$, `TVD rol 0,58 / tipo 0,53 / contig 0,76`), 8 cuentas con huella distinta + caída de WR ($\Delta\text{WR} = -33{,}6\ \text{pp}$, `TVD hora 0,37 / tipo 0,56 / contig 0,82`), 68 candidatas a compartida y 197 rachas buenas (0/8 pasan FDR global $q < 0{,}05$, reforzando su rol como filtro de priorización). `notebooks/eda_reconocimiento_jugadores.ipynb` quedó recalculado y ejecutado.
6. **Matchmaking orientado a laxitud y protección individual.** `scripts/analysis/matchmaking_swaps.py` evalúa con las mitades independientes A ↔ B todos los puntos de la frontera de laxitud ($\Delta D_{\max} \in \{0,\ 0{,}05,\ 0{,}10,\ 0{,}20,\ 0{,}35,\ 0{,}50,\ \infty\}$) y los guarda en `swaps_frontera.parquet`. En `build_eda_modelos_nb.py` §7.2–7.4 se destaca la curva según laxitud de rol y el punto de **protección individual** (`ab_pareto_t20`, $\Delta d_i \le 0$ para los 10 jugadores) junto a `ab_minD_t20` (priorizar mejorar el rol).
7. **Retiro del modelo de LP entre fotos, quita de `HR@10` y fix de `embeddings.ipynb`.** Se eliminaron `notebooks/eda_entre_fotos.ipynb`, `scripts/analysis/build_eda_fotos_nb.py`, `elo_boostings.py`, `elo_gru.py` y `elo_val_lolpros.py` junto con sus referencias (conservando `src/features/elo_reverso.py` por sus utilidades compartidas). Se quitó el bloque del recomendador `HR@10` de `scripts/analysis/draftrec_mejor.py` y se reemplazaron las cifras fijas de `notebooks/embeddings.ipynb` por cálculos dinámicos (ejecutado).






