# Scripts de análisis vigentes

Acá queda solo lo que producen o necesitan los resultados finales. Los experimentos descartados, con la
idea, el resultado y por qué no entraron, están en [`historial/`](../../historial/README.md).

Todos se corren desde la raíz del repo (`python scripts/analysis/<script>.py`). Los dos modelos de quién gana
viven en `src/ganador/` y se corren con un solo comando cada uno. La cadena de scripts `brainstorm_*` que armaba el
escalón 9 y el ensamble en vivo del anexo pasaron a `historial/escalon9/` y `historial/ensamble_en_vivo/`
(§9.S.79). Acá quedan solo dos nombres de esa etapa, porque son vigentes: `brainstorm_base.py` (carga y partición
que usan matchmaking, validación, `umap_vistas`, `partidas_crudas` y el notebook `eda_modelos_ganador`, que lo
importa con ese nombre) y `brainstorm_45_dificultad.py` (partidas difíciles; `umap_vistas` importa su `contexto`).

`tests/test_scripts_importables.py` comprueba que todo compile, que los imports entre scripts apunten a
archivos existentes y que nada de acá importe del historial.

---

## 1. Quién gana, antes de la partida: escalón 9 (V Master+ 0,6727 / 62,4 % con la máscara de la foto 1)

**Un solo comando:** `python scripts/analysis/modelo_previo.py` (usa `src/ganador/`; ~22 min desde cero, ~45 s
con la caché vigente en `data/cache/ganador/`, que se recalcula sola si cambia el código o los datos; §9.S.73).
Logística sobre el draft campeón × línea, el historial de cada jugador, el nivel histórico y el W/L de la
foto 1, Kalman-Elo, contracruces y el modelo por línea. Decisión y fórmulas en `PROJECT_SUMMARY.md`
§9.S.52 y §9.S.68; la máscara del nivel histórico en §9.S.75.

Lo comparten con otros análisis: `embeddings_partidas.py` (`variables.npz`: draft y las 85 variables por jugador,
§9.S.41), `ganador_mapa_lp.py` (mapa de LP causal) y `ganador_composicion.py` (atributos de campeones de Meraki y
CommunityDragon).

**La cadena original** (`brainstorm_05` → … → `brainstorm_final.py`, 0,6740 / 62,6 % sin la máscara) está en
`historial/escalon9/`, con el orden de las cachés. Sigue siendo la que arma `data/analysis/brainstorm/*.npz`, que leen
`brainstorm_base.extras_referencia` y sus usuarios (matchmaking, `modelos_train_v`, `acierto_por_lp_duracion`,
`diamante_conocidos`, `umap_vistas` y el notebook). Desde el issue #4 (§9.S.75) coincide con
`src/ganador` solo tras `historial/escalon9/brainstorm_07_mmr_historico.py --solo-cache --rehacer`.

## 2. Quién gana, durante la partida: logística con el previo del escalón 9 (§9.S.78)

**Un solo comando:** `python scripts/analysis/modelo_en_vivo.py` (usa `src/ganador/modelo_en_vivo.py`; CPU, sin
GPU ni `lightgbm`; cachés `en_vivo_estado` y `en_vivo_previo` en `data/cache/ganador/` que se invalidan solas).
El modelo que se presenta es **la etapa 1**: logística sobre el estado del minuto con las ~92k partidas Master+ y el
logit previo del escalón 9 hacia adelante. V Master+: 0,7883 / 0,8411 / 0,8726 de AUC a los 10 / 15 / 20 (cada
minuto con sus partidas vivas); 0,7619 / 0,8168 / 0,8726 en la misma cohorte. La etapa 2 ($z$ y $L$ solo con
partidas posteriores a la foto) se sigue calculando y reportando como **variante descartada**: no pasa la regla
de aceptación (§9.S.47; "Decisión" en §9.S.78). Reporta con IC por bootstrap y guarda `pred_en_vivo.npz` con `mids`.
`modelo_en_vivo_eventos.py` mide los eventos del timeline encima (candidato, no promovido).

| Script | Qué hace |
| :--- | :--- |
| `ingame_evaluar.py` | Evaluación común de los modelos en vivo de la etapa anterior (lee `ingame_cache.npz`, que arma `historial/ensamble_en_vivo/ingame_preparar.py`) |
| `ingame_eventos.py`, `eventos_ablacion.py` | Aporte de los eventos del timeline (§9.S.64, §9.S.66); `modelo_en_vivo_eventos.py` usa sus variables |
| `ingame_estado_puro.py` | "Solo el estado", sin nada previo: barra de la sección 3 de `eda_modelos_ganador` |

**Anexo: el ensamble `6_Final_dEns10`** (0,7912 / 0,8441 / 0,8750 a los 10 / 15 / 20, cada minuto en su cohorte;
0,7652 / 0,8202 / 0,8750 en la misma). Ensamble en dos etapas con redes en GPU y el previo del escalón 10
(§9.S.57, §9.S.60); no se rehízo. Sus scripts (`ingame_preparar`, `ingame_g_01/03/04/05` y los `brainstorm_*` cuyas
cachés lee) están en `historial/ensamble_en_vivo/`: `python -u historial/ensamble_en_vivo/ingame_g_04_mt_a4_gpu.py`.

## 3. Validación y comparación

| Script | Qué hace |
| :--- | :--- |
| `evaluacion_test.py` | **Evaluación única en el test** de los dos modelos según `docs/evaluacion_test/PROTOCOLO.md` (§9.S.80). Se niega a correr si ya existe el log o el árbol no está limpio; `--simulacro` ensaya todo con V como test |
| `validacion_bloques.py` | Validación temporal por bloques: la regla de decisión de variantes (§9.S.47) |
| `curva_aprendizaje.py` | AUC según cuántas partidas ajustan el modelo (§9.S.48) |
| `modelos_train_v.py` | Los 8 modelos del camino en Train y V (§9.S.55). Lee cachés de `historial/ganador_primeros/` |
| `evaluate_baselines.py`, `evaluate_draft_logreg.py` | Baselines y modelos del draft (§9.S.13, §9.S.20) |
| `draftrec_comparacion.py`, `draftrec_mejor.py`, `grafico_draftrec_puente.py` | Comparación con DraftRec (WWW 2022) en sus datos (§9.S.59) |
| `grafico_kaggle_vivo.py` | Comparación con Kaggle en vivo al minuto 10 |
| `acierto_por_lp_duracion.py`, `diamante_conocidos.py`, `brainstorm_45_dificultad.py` | Acierto por LP del lobby, duración, Diamond y partidas difíciles (con `brainstorm_base` y las cachés de `historial/escalon9/`) |
| `cobertura_perfiles_test.py`, `proyeccion_split.py` | Cobertura de jugadores del test y proyección del split |

## 4. Matchmaking (§9.S.67)

`matchmaking_perfil_roles.py` → `matchmaking_calibracion.py` → `matchmaking_swaps.py`: reacomodar a los
mismos 10 jugadores baja el desbalance que mide el modelo de ~10 a ~5 puntos (4,96) sin que ningún jugador
empeore su rol. Es una simulación bajo el modelo y no restringe dúos ni premades (§9.S.77).

## 5. Jugadores: clustering, huella de estilo y rating

| Script | Qué hace |
| :--- | :--- |
| `cluster_emeans.py`, `umap_vistas.py` | K-Means contra e-means; UMAP + HDBSCAN (§9.S.53, §9.S.58). `umap_vistas` lee `hgb_lineas_extra.npz` (`historial/ideas_posteriores/brainstorm_42_xp_lineas.py`) y `pred_g28_esc9.npz` (anexo) |
| `fingerprint_level1.py`, `huella_niveles.py`, `huella_mejoras.py`, `huella_patrones_cuenta.py`, `huella_previo.py` | Huella de estilo, niveles 1-3 y patrones dentro de cada cuenta (§10) |
| `build_rating.py` | Rating TrueSkill por partida (§9.S.21; el elo al revés entre las dos fotos se retiró el 2026-11-24) |
| `build_lp_partida.py`, `build_daily_form.py`, `lp_partida_modelo.py`, `mapa_lp_basico.py`, `mapa_lp_modelos.py` | LP por partida y mapa de LP (§9.S.21, §9.S.43, §9.S.49, §9.S.54) |
| `oro_equivalente_roles.py` | Oro equivalente de kills, CS y objetivos por rol (§9.S.63) |

## 6. Notebooks

| Script | Genera |
| :--- | :--- |
| `build_informe_nb.py` | `notebooks/informe_final.ipynb`, el notebook único de la entrega (figuras de `src/informe/`, PNG en `notebooks/figuras/informe/`; §9.S.81) |
| `build_eda_modelos_nb.py` | `notebooks/eda_modelos_ganador.ipynb` |
| `build_eda_reconocimiento_nb.py` | `notebooks/eda_reconocimiento_jugadores.ipynb` |
| `build_rating_nb.py` | `notebooks/rating_jugadores.ipynb` |

`ganador_registro.tsv` es el registro histórico de corridas de los modelos de quién gana.
