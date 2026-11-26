# Protocolo de la evaluación única en el test

Fijado el 2026-11-23, **antes de cargar el test** (issue #5; decisión y alternativas en `PROJECT_SUMMARY.md`
§9.S.80). La evaluación la corre `scripts/analysis/evaluacion_test.py` **una sola vez**; el log
(`docs/evaluacion_test/LOG_EVALUACION_TEST.md`) guarda el hash SHA-256 de este archivo, así que cualquier cambio
posterior al protocolo queda a la vista.

**Aprobado por el usuario el 2026-11-23, tal cual:** Diamond posterior al corte en el universo solo para los
historiales; ajuste final con ajuste ∪ V; regla del split (≥ 5) en todo el modo test; C de la etapa 1 congelado en
0,01 / 0,03 / 0,03; bootstrap por bloques de días con B = 1000; semilla principal 42 más 0, 1, 2 y 3; ventanas de
2.500 partidas.

## 1. Qué se evalúa y por qué hace falta

Dos modelos que predicen si gana el equipo azul en partidas de League of Legends de Master+ (servidor LAS):

* **Modelo previo** (antes de que empiece la partida): el "escalón 9" de `src/ganador/modelo_previo.py`.
* **Modelo en vivo** (a los minutos 10, 15 y 20): la "etapa 1" de `src/ganador/modelo_en_vivo.py`.

Los dos se eligieron entre decenas de variantes mirando la ventana de validación V (16/09 → 27/09): el número de V
(0,6727 de AUC el previo) es optimista. El número que va al informe es el del test: partidas posteriores al 27/09
que ningún modelo final vio.

**Uso previo del test (declaración).** El test del corte 27/09 ya se usó una vez, para los baselines del draft:
la regla sin entrenamiento y la logística campeón × línea (`PROJECT_SUMMARY.md` §9.S.13 y §9.S.20; 37.196 partidas
completas). Después el split se regeneró con la regla vigente (38.013 partidas de test, `split_manifest.json`).
Ningún modelo del escalón 9 ni del en vivo, ni ninguna de sus variantes, se evaluó en el test. Las decisiones
tomadas con esos baselines (que el draft solo no alcanza) no eligieron nada de lo que se evalúa acá.

## 2. Datos congelados

* **Split:** `data/splits/match_split.csv`, regla de `src/features/split.py` (§9.S.12/§9.S.17): corte el
  **27/09/2026 00:00 hora de Argentina**; test = partidas de Master+ (abajo) desde el corte.
* Hashes SHA-256 (el script los recalcula y **no carga el test si alguno no coincide**):

  | Archivo | SHA-256 |
  | :--- | :--- |
  | `data/splits/match_split.csv` | `ceb5a0bbfb5d131bb82571c4691113b8c9d410d9ac214ba1932f5bcdae1f037e` |
  | `data/splits/train/full_dataset.csv` | `effcd4af654b6693f55f75e8857075b9127acaa0043fbc573ed611c23387006a` |
  | `data/splits/test/full_dataset.csv` | `c895559c6234ad3de9ce58fbfd23adabe8e2d80e42838d43a7bf81e43055b230` |
  | `data/splits/diamante/despues/full_dataset.csv` | `85cbdba315719edd55f265436bec30f9892e7512f20a60e071880683509a4f4c` |

  <!-- sha256 match_split ceb5a0bbfb5d131bb82571c4691113b8c9d410d9ac214ba1932f5bcdae1f037e -->
  <!-- sha256 train effcd4af654b6693f55f75e8857075b9127acaa0043fbc573ed611c23387006a -->
  <!-- sha256 test c895559c6234ad3de9ce58fbfd23adabe8e2d80e42838d43a7bf81e43055b230 -->
  <!-- sha256 diamante_despues 85cbdba315719edd55f265436bec30f9892e7512f20a60e071880683509a4f4c -->

  Calcular el hash del archivo entero no es mirarlo: no da conteos, fechas ni resultados.
* **Universo de partidas** (modo `test` de `src/ganador/rutas.py`): train ∪ test ∪ Diamond posterior al corte
  (`data/splits/diamante/despues/`), en orden de fecha. Las Diamond posteriores al corte **no se evalúan** como test;
  están para que el historial de los jugadores del test incluya también sus partidas en lobbies Diamond, igual que en
  V (donde las Diamond de la ventana entran al historial y se reportan aparte como "Diamond-V"). Se reportan como
  chequeo, igual que Diamond-V.
* **Frames por minuto del modelo en vivo:** salen de las columnas `{oro, xp, cs, daño, kills, muertes, asistencias,
  nivel}_{10,15,20}` del CSV, que `export_to_csv.py` arma con
  `LEFT JOIN match_timeline t ON p.match_id = t.match_id AND p.participant_id = t.participant_id` sobre
  `lol_las.db`. El CSV del test salió del mismo export (mismo join), así que no hace falta otra consulta. Una partida
  entra al minuto $m$ si duró más de $m$ minutos y tiene los 10 jugadores con frame hasta $m$ (`validas`). La
  cobertura (fracción de partidas del test con frame al minuto 10) se informa y se exige ≥ 90 % (§7).

## 3. Regla de Master+ (unificada)

Una partida es Master+ si **al menos 5 de sus 10 jugadores están en el padrón con tier ≥ Master** (la regla del
split, `es_master_plus_split`). Es la que define el test. Sobre las 187.817 partidas del train, contar los tiers de
`player_tier` da exactamente el rango de `match_split.csv` (0 discrepancias), así que el pipeline la reproduce.

La validación usaba la mediana del tier de los jugadores del padrón (`es_master_plus`). En V las dos reglas
difieren en 1.121 de 26.934 partidas: V tiene 16.428 partidas con la mediana y 17.081 con la regla del split. El
efecto en V se reporta en `PROJECT_SUMMARY.md` §9.S.80. En la evaluación se usa **solo la regla del split**: para
las filas de ajuste, para las partidas evaluadas y para todo lo aprendido dentro de las variables que filtra por
Master+ (el HGB por línea y los pliegues del modelo en vivo).

## 4. Qué se ajusta, con qué

Todo lo que se aprende usa **solo partidas anteriores al corte del 27/09** (`~val0` en el código):

| Pieza | Con qué se ajusta |
| :--- | :--- |
| Escalas, imputaciones y medias de las variables previas | Todas las partidas anteriores al corte |
| Kalman-Elo: logística del estado a los 15 | Partidas anteriores a la foto 1 (04/09) |
| HGB del oro por línea (`oro_linea_hgb`) | Hacia adelante dentro del ajuste; las partidas del test, con todo el Master+ anterior al corte |
| **Modelo previo** (logística L2, C = 0,05) | **Todas las partidas Master+ anteriores al corte** (ajuste ∪ V) |
| Oro residual, expectativas por línea (Ridge + HGB) | Hacia adelante por tramos; las del test, con el Master+ anterior al corte |
| Previo hacia adelante $L$ (entrada del en vivo) | Por tramos; en el test, el modelo previo de arriba |
| **Modelo en vivo, etapa 1** (logística sobre el estado del minuto y $L$) | Tramos 1 a A4 = todo el Master+ anterior al corte salvo el primer cuarto del historial viejo, igual que en V |

*Por qué ajuste ∪ V y no solo hasta el 16/09:* es lo estándar (el modelo final se reentrena con todo lo disponible
antes de predecir), y la distancia entre lo último que se ve y lo que se predice queda igual que en V (cero días).
Entrenar solo hasta el 16/09 dejaría un hueco de 11 días que en producción no existe y castigaría al modelo por la
deriva de ese hueco. El precio: el número del test no es estrictamente "el modelo de V"; los hiperparámetros sí son
los de V (abajo).

**Las variables de una partida del test usan partidas anteriores a ella**, incluidas otras del test: el historial de
cada jugador, los contracruces, el Kalman-Elo, etc. se calculan recorriendo las partidas en orden de fecha y leyendo
el estado **antes** de actualizarlo con el resultado de la partida (es lo que ya hacía V; está verificado por los
tests de no-fuga de `tests/test_ganador_*.py`). Así se usaría en producción: el resultado de una partida de ayer se
conoce hoy. Lo que **no** se hace es ajustar ningún parámetro con partidas del test.

**Hiperparámetros congelados** (los de V, código del commit que figura en el log): previo C = 0,05; HGB por línea
y de expectativas con los parámetros del código; etapa 1 con C = 0,01 / 0,03 / 0,03 a los 10 / 15 / 20 (los que
eligió la grilla en V, `data/analysis/en_vivo/metricas.csv`; en el test **no se vuelve a elegir**).

## 5. Semillas

La logística (lbfgs) es determinista: dos ajustes dan los mismos coeficientes bit a bit (verificado). Dependen de
una semilla el HGB del oro por línea (división de validación del corte temprano) y el HGB de las expectativas del
modelo en vivo (80 % de columnas por división), y el bootstrap.

* **Semillas de los modelos: 42 (la principal, la de siempre), 0, 1, 2, 3**, fijadas el 2026-11-23. La semilla
  cambia los dos HGB a la vez. **El número principal es el de la semilla 42.** Se reportan media, desvío, mínimo y
  máximo de cada métrica entre las 5, y la correlación de las predicciones con la principal.
* **Bootstrap:** semilla 20260927 para los IC; se reportan también los IC con las semillas 1, 2, 3 y 4 para mostrar
  cuánto se mueven sus extremos.

## 6. Métricas, IC y ventanas

* **Métricas:** AUC, acierto (umbral 0,5), log loss, ECE (10 deciles de $p$) y la **ventaja del acierto sobre
  "siempre rojo"** (predecir siempre que gana el equipo rojo). La línea base es "siempre rojo" y no "el lado más
  frecuente del test": elegir el lado mirando los resultados ya sería usar el test. En V el rojo gana el 52,2 %.
* **Conjuntos:** previo en todo el test, por rango (Master, Grandmaster, Challenger; con la regla del split) y en
  Diamond posterior al corte (chequeo). En vivo, en cada minuto: las partidas del test vivas en ese minuto, la
  **cohorte común** (vivas a los 20, la misma en los tres minutos, con el previo $L$ como referencia) y Diamond
  posterior al corte (chequeo).
* **IC 95 % por bootstrap por bloques de días**, B = 1000: se remuestrean días calendario (00:00 ART) enteros con
  reposición y se toman los percentiles 2,5 y 97,5. Las partidas de un mismo día comparten parche y jugadores (el
  mismo jugador juega decenas de partidas por día), así que no son independientes; remuestrear partidas sueltas da un
  IC demasiado angosto. Por jugador no se puede remuestrear limpio: cada partida tiene 10 jugadores y los grupos se
  solapan. En el conjunto principal se informa también el IC por partidas, para comparar.
* **Degradación:** las mismas métricas en **ventanas cronológicas de 2.500 partidas** consecutivas (la última, si
  queda con menos de 1.250, se une a la anterior), con IC por bootstrap de partidas dentro de cada ventana, y la
  **pendiente del AUC por semana** (mínimos cuadrados del AUC de cada ventana contra su fecha media) con IC por
  bootstrap. Con 2.500 partidas el error estándar del AUC de una ventana es ≈ 0,011.

## 7. Chequeos y qué se hace si uno falla

| Momento | Chequeo | Si falla |
| :--- | :--- | :--- |
| Antes de cargar nada | El log no existe; el árbol de git está limpio; los 4 hashes coinciden | No corre. Se corrige y se vuelve a lanzar (no cuenta como evaluación: no se leyó el test) |
| Antes de cargar el test | El CSV del test tiene las mismas columnas que el del train | No corre (no se leyó ninguna partida) |
| Con el test cargado, **antes de calcular métricas** | Todo lo ajustado es anterior al corte; las partidas evaluadas son exactamente las del test de `match_split.csv`; las Diamond, las `diamante_despues`; todas las evaluadas son Master+ con la regla del split; cobertura del test ≥ 95 % (partidas con las 10 filas válidas); predicciones sin NaN; ≥ 90 % de las partidas del test con frame al minuto 10 | **No se calcula ninguna métrica.** Se escribe `LOG_EVALUACION_TEST_FALLIDA_<fecha>.md` con el chequeo que falló. Corregir el código y volver a correr requiere el OK del usuario y queda registrado; como no se vio ninguna métrica, sigue siendo una evaluación única |
| Durante las métricas | (ninguno) | Si el script se corta después de calcular métricas, no se repite: se reporta lo que quedó en pantalla y en disco |

Una sola salida intermedia del pipeline mira el test antes de terminar: el HGB del oro por línea imprime su AUC
por línea en las partidas evaluadas (un diagnóstico del código original). No es una métrica del modelo; si la corrida
se cae después de esa línea y hay que relanzarla, se declara en el log.

Repetir la evaluación una vez que existe el log solo se puede con `--excepcion "motivo"` (con el OK del usuario): el
log anterior se conserva con fecha y el nuevo dice que es una excepción al protocolo.

## 8. Cómo se corre

```bash
python scripts/analysis/evaluacion_test.py --simulacro     # ensayo con V como test (no toca el test)
python scripts/analysis/evaluacion_test.py                 # la evaluación única (con el OK del usuario)
```

Salidas: `data/analysis/test_final/` (`metricas.csv`, `ventanas.csv`, `pendientes.csv`, `semillas.csv`,
`ic_bootstrap_semillas.csv`, `resumen.json`, `chequeos.json`, `predicciones.npz`) y el log. Se commitean tal cual
salieron. Cachés en `data/cache/ganador_test/` (no pisan las de V).
