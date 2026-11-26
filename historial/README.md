# Historial de pruebas

Acá están los experimentos que **se probaron y no forman parte de los modelos finales**. No es código
muerto: cada script es una idea que se midió con la misma regla que los modelos vigentes. Casi todos se
descartaron por la misma razón: **apareció algo mejor**, o la idea no superó la regla de aceptación.

Lo vigente vive en otro lado:

| Qué | Dónde |
| :--- | :--- |
| Librería (crawler, base, features, modelos) | `src/` |
| Modelos finales y lo que los alimenta | `scripts/analysis/` (ver su `README.md`) |
| EDA y resultados | `notebooks/` |
| Tests | `tests/` |

**Reglas de esta carpeta.**

* Nada vigente importa de acá. Lo verifica `tests/test_scripts_importables.py`, que además compila cada
  script y comprueba que sus imports y su ruta a la raíz del repo sigan siendo válidos.
* Los scripts siguen funcionando y se corren desde la raíz, por ejemplo
  `python historial/brainstorming_noche/brainstorm_16_poda.py`. Varios leen cachés de `data/analysis/` que
  arman los scripts vigentes.
* Al revés, hay una sola excepción, por cachés y no por imports: `scripts/analysis/brainstorm_base.py`
  (`extras_referencia`) y los scripts vigentes que lo usan (matchmaking, `modelos_train_v`, `umap_vistas`,
  `acierto_por_lp_duracion`, `diamante_conocidos`, el notebook `eda_modelos_ganador`...) leen las cachés de
  `data/analysis/brainstorm/` que arman `escalon9/` y `ensamble_en_vivo/`. Para regenerarlas se corren esos
  scripts desde acá (orden en `escalon9/` abajo y en §9.S.69). El modelo previo y el modelo en vivo vigentes no
  las usan: salen de `src/ganador/` (§9.S.73, §9.S.78, §9.S.79).
* Los números de abajo son AUC / acierto en **V Master+** (16.428 partidas del 16/09 al 27/09, §9.S.51),
  salvo que se indique otra cosa. "Δ" es la diferencia contra el modelo de referencia de ese momento,
  con IC 95 % por bootstrap pareado. El detalle de cada prueba está en la sección de `PROJECT_SUMMARY.md`
  que se cita (§9.S.N).

**La regla de aceptación** (§9.S.47 y §9.S.52). Una variante entra solo si mejora en la validación
temporal por bloques (A2-A4, hacia adelante), con IC > 0 en V y, desde la madrugada del brainstorming,
con réplica en las partidas Diamond de la ventana de V, que nunca se usaron para decidir. Con efectos de
~0,001, "3 de 3 bloques" solo no alcanza: unas columnas al azar también mejoraron en los 3 bloques.

---

## El camino en una tabla

Por qué el modelo final es el que es. Cada fila se descartó al aparecer la siguiente (§9.S.55):

| Modelo | V Master+ AUC / acierto | Por qué se dejó |
| :--- | :---: | :--- |
| Siempre el rojo | 0,500 / 52,2 % | Referencia |
| Regla con winrate de campeón por línea (§9.S.13) | 0,549 / 53,7 % | Solo draft: techo de ~0,55 |
| Logística del draft (§9.S.20) | 0,550 / 53,8 % | Ídem |
| Campeones (§9.S.23) | 0,545 / 53,4 % | Ídem |
| Campeones + jugador (§9.S.23) | 0,582 / 55,6 % | Pocas partidas de ajuste (necesitaba el LP de la foto) |
| Stacking de 6 expertos (§9.S.29) | 0,657 / 61,4 % | Lo superó la logística con más información |
| 2 expertos + composición + por línea (§9.S.37) | 0,660 / 61,7 % | Ídem |
| **Escalón 9: logística Master+ (§9.S.52, §9.S.68)** | **0,674 / 62,6 %** (0,6727 / 62,4 % con la máscara de §9.S.75) | **Vigente**: `src/ganador/` y `scripts/analysis/modelo_previo.py` (la cadena original quedó en `escalon9/`) |

En vivo, la predicción en dos etapas por líneas de `en_vivo/` (0,7842 / 0,8377 a los minutos 10 / 15) quedó
por debajo del ensamble `6_Final_dEns10` (0,7912 / 0,8441 / 0,8750 a los 10 / 15 / 20, §9.S.60), que hoy es el
anexo (`ensamble_en_vivo/`). El modelo en vivo que se presenta es la logística de `src/ganador/modelo_en_vivo.py`
con el previo del escalón 9 (etapa 1 sola: 0,7883 / 0,8411 / 0,8726, §9.S.78).

---

## `ganador_primeros/` — primeros modelos de quién gana (12-14/11, §9.S.23-§9.S.39, §9.S.45, §9.S.50)

Antes del brainstorming. Se medían sobre la V de todas las ligas, así que sus números no se comparan
directamente con los de V Master+. Los modelos de esta etapa (stacking de 6 y de 2 expertos) los
reemplazó el escalón 9. La sección 1 del notebook `eda_modelos_ganador` todavía los reproduce desde acá
(orden de comandos en §9.S.55).

| Script | Idea | Resultado | Por qué se descartó |
| :--- | :--- | :--- | :--- |
| `ganador_features`, `ganador_features_grande` | Variables por partida: huella, matchups, sinergias, experiencia; después con todo el train | Base de "Campeones" y "Campeones + jugador" (0,545 y 0,582) | Las reemplazó `brainstorm_base` |
| `ganador_modelos` | Ablación de variables y comparación logística / árboles / red, con y sin espejado | Árboles y red por debajo de la logística: no hay interacciones que aprender (§9.S.23) | Gana la logística |
| `ganador_red`, `ganador_mlp` | Red chica espejada; MLP más grande con 160.000 partidas | La MLP queda 0,004 abajo con cualquier tamaño de ajuste (§9.S.23) | Gana la logística |
| `ganador_red_jugador` | Red con un codificador compartido por jugador | Experto del stacking (§9.S.25-§9.S.26) | El stacking lo superó el escalón 9 |
| `ganador_logistica_oos`, `ganador_stacking` | Expertos con predicciones fuera de muestra y meta-modelo | Stacking de 6 expertos: 0,657 / 61,4 % (V Master+) | Lo superó el escalón 9 |
| `ganador_stacking_aporte`, `ganador_dos_expertos` | Qué experto aporta y diagnóstico del modelo de 2 | De 7 expertos alcanzan 2; la combinación está saturada y calibrada (§9.S.33-§9.S.34) | Diagnóstico; el modelo de 2 lo superó el escalón 9 |
| `ganador_por_grupos`, `ganador_por_jugador` | ¿Mezcla de expertos (MoE) con compuerta? ¿Variables por jugador sin restar azul − rojo? | Expertos que ven las mismas variables no tienen de dónde ganar (§9.S.23, §9.S.25) | No suma |
| `ganador_margen_duos`, `ganador_margen_grupos` | Margen previo del jugador (por cuánto gana, no solo si gana); por campeón y por línea | El margen fue la mejor señal de la etapa (§9.S.24, §9.S.29) | **Pasó al modelo final** dentro de `brainstorm_base`; estos scripts quedaron viejos |
| `ganador_duos_real`, `duos_parche`, `duos_roles` | Duos reales contra coincidencias, por parche y por rol | Separar duos reales no aporta (§9.S.27, §9.S.29) | No suma |
| `parche_ganador`, `parche_predecir` | Fuerza del campeón en el parche en curso; predecir el parche desde la partida | +0,001 en V (§9.S.28) | No alcanza la regla |
| `ganador_poisson` | Ataque y defensa por jugador (Poisson) para kills, oro y daño | +0,002 como complemento del margen (§9.S.25) | Superado por el margen por campeón y línea; en el brainstorming, −0,0002 (§9.S.52) |
| `ganador_mmr` | MMR propio: nivel del lobby y margen ajustado por el rival | El margen ajustado suma +0,0007; el nivel del lobby fue experto candidato (§9.S.31) | La idea siguió como "nivel histórico" (`brainstorm_07`, en el modelo final) |
| `ganador_cambio` | ¿Cambió el jugador (pool, OTP, cuenta boosteada)? ¿Encaja el equipo? | La cuenta boosteada no suma; el encaje +0,0003 (§9.S.30) | No suma |
| `ganador_lineas` | Modelo por línea como experto del equipo | Mejora en bloques; el término entre rivales no suma (§9.S.36) | Rehecho de forma causal en `brainstorm_g_09`, que **sí está en el modelo final** |
| `ganador_prior` | Valor de arranque del margen para jugadores con poco historial | En partidas con 3 o más jugadores nuevos: 0,6084 → 0,6051 (§9.S.35) | Empeora |
| `ganador_snowball` | "Si le das una kill antes del 10 te hace la partida": rasgos condicionales | Cómo arranca cada jugador es estable, pero no predice más que el margen (§9.S.35) | No suma |
| `ganador_oraculo` | Cota superior, tramposa a propósito: ¿y si conociéramos más historial del jugador? | Ni con el futuro del jugador sube: el historial está agotado (§9.S.38) | Experimento de techo, no es un modelo |
| `ganador_maestro` | Destilación: un maestro con información de la partida enseña al modelo previo | La destilación no cambia la logística (§9.S.39) | No suma |
| `ganador_runas` | Runas, hechizos y nivel de cuenta (muestra de 10.000) | Solo la rareza de la runa clave: +0,0017 [−0,0009; +0,0043] (§9.S.37) | IC toca 0; faltan las runas de toda V |
| `ganador_huella_progresiva`, `ganador_estilo_habilidad` | Huella de estilo calculada con las partidas anteriores; habilidad contra estilo | Sin margen sobre las variables por jugador (§9.S.39) | No suma |
| `ganador_forma_dia` | Forma del día como margen: ¿juega mejor o peor que lo esperado hoy? | ΔAUC −0,0001 a +0,0002, todo n. s. (§9.S.45) | No suma |
| `ganador_con_mapa` | Validación por bloques con el mapa de LP | Suma en A3, A4 y V (+0,0041), falla en A2 (§9.S.50) | Con nivel histórico y W/L de la foto, el mapa dejó de aportar (−0,0002, §9.S.52) |

---

## `brainstorming_noche/` — brainstorming Claude ↔ Gemini (15/11, §9.S.52)

Una noche buscando mejoras sobre el híbrido + mapa (0,6531 / 61,1 %). Hay un script por idea:
`brainstorm_NN_*` es de Claude y `brainstorm_g_NN_*` de Gemini. Lo que se aceptó es la **escalera**; los
scripts de sus escalones (`brainstorm_05`, `07`, `g_01`, `g_06`, `g_07`, `g_09`, `g_10`, `g_11`, `g_23`...) están
en `escalon9/` y su código, copiado tal cual, en `src/ganador/` (§9.S.73, §9.S.79). Acá queda lo que no entró,
más las verificaciones independientes de cada escalón. Bitácora completa en `docs/BRAINSTORM_NOCHE.md`.

**Escalera aceptada** (como referencia: explica contra qué se midió cada descarte):

| Escalón | V AUC / acierto |
| :--- | :---: |
| Híbrido + mapa (§9.S.51) | 0,6531 / 61,1 % |
| 1. Entrenar solo con Master+ | 0,6557 / 61,3 % |
| 2. + margen por (jugador, campeón) y (jugador, línea) | 0,6614 / 61,6 % |
| 3. + nivel histórico y W/L de la foto 1 | 0,6663 / 62,0 % |
| 4. + contracruces y sinergias de campeones | 0,6674 / 61,9 % |
| 6. + Kalman-Elo, tempo, objetivos y economía | 0,6707 / 62,2 % |
| 7. + modelo por línea (oro @15 predicho, HGB causal) | 0,6722 / 62,5 % |
| **9. + margen esperado por par de campeones** | **0,6739 / 62,5 %** |

| Script | Idea | Resultado | Por qué se descartó |
| :--- | :--- | :--- | :--- |
| `brainstorm_01_master` | Pesos propios de Master+: entrenar solo con Master+ o ponderarlo | Solo Master+: +0,0026 | **Entró** (escalón 1); el script quedó como prueba |
| `brainstorm_02_lobby` | Asimetría de LP entre equipos; hora × habilidad; duos causales | −0,0010; +0,0002; −0,0006 | Empeora o n. s. |
| `brainstorm_03_ajustes` | Peso por antigüedad (τ = 15/30/60 d), C de la logística, mapa por línea | −0,0038 a −0,0002; ±0,0002; −0,0029 | Empeora |
| `brainstorm_04_margen_nivel` | Margen del jugador en lobbies de su mismo nivel | −0,0015 | Empeora |
| `brainstorm_06_cobertura` | Acierto en la fracción más segura y calibración | Diagnóstico (20 % más seguro: ~75 %) | No es un modelo |
| `brainstorm_09_foto_agregados` | W/L de la foto como máx / mín / desvío del equipo, o por línea | −0,0004; −0,0005 | Empeora |
| `brainstorm_10_umbral` | ¿Con qué lobbies entrenar? Corte por tier medio | Cortes más laxos (0,3-0,9) o todo empeoran | Se queda "solo Master+" |
| `brainstorm_11_correcciones` | Correcciones de la revisión de Gemini a `07` y `08` | Efecto −0,0001 | Ajuste menor |
| `brainstorm_12_errores` | ¿Dónde falla el escalón 3? AUC por subgrupos | Lobbies sin estrenos 0,6455 y de veteranos 0,6533 | Diagnóstico: guió las ideas 15-20 |
| `brainstorm_13_segmentos` | Un modelo por segmento de lobby | −0,0024 a −0,0119 | Partir los datos reduce N |
| `brainstorm_14_verifica_escalon4` | Verificación independiente del escalón 4 de Gemini | Confirmado con placebo ≈ 0 | Verificación (el escalón **entró**) |
| `brainstorm_15_campeon_reciente` | Forma reciente con el campeón y "óxido" | +0,0004 (n. s.) | No suma |
| `brainstorm_16_poda`, `brainstorm_32_poda6` | Poda de familias del escalón 4 / 6 | Sin one-hot del draft −0,0028; sin perfil −0,0008; sin mapa −0,0002 | No se poda, salvo el mapa |
| `brainstorm_17_chequeo_diamond` | Reevaluar cada escalón en las 10.506 Diamond de V, nunca usadas para decidir | El margen por campeón y los matchups se replican | Chequeo de selección: las mejoras no son un artefacto de mirar V |
| `brainstorm_19_ruido_etiqueta` | Sacar remakes y rendiciones tempranas del entrenamiento | −0,0016; −0,0021 | Empeora |
| `brainstorm_20_candidato_final`, `brainstorm_21_importancia`, `brainstorm_31_ganancia_total` | Métricas, importancia por bloques y ganancia total de los candidatos intermedios | Escalón 6: ΔAUC +0,0178 [+0,0137; +0,0218] contra la referencia | Los reemplazó `brainstorm_final` (escalón 9) |
| `brainstorm_22_verifica_escalon5`, `brainstorm_g_15_semillas_escalon5` | Ensamble logística 65 % + red NeuralAC 35 % (escalón 5) | +0,0022, robusto a semillas y pesos | La logística del escalón 6 lo supera sin GPU |
| `brainstorm_23_interacciones_lobby` | Pesos que dependen del tipo de lobby, sin partir los datos | × veteranía: bloques +0,0009 pero V −0,0003; con 4 rasgos, V +0,0001 | No suma: lo que la red gana en veteranos no es una interacción lineal |
| `brainstorm_25_mezcla_poblaciones`, `brainstorm_26_acuerdo` | Mezclar modelos entrenados con poblaciones distintas; para la demo, elegir partidas donde logística y red coinciden | Mezcla −0,0001 a −0,0003; acuerdo: 76,0 % contra 75,7 % en el 20 % más seguro | No suma (ruido) |
| `brainstorm_28_verifica_rolwr` | Verificación de ROL_WR (winrate por línea y caída por autofill) | +0,0009, IC en V toca 0; columnas al azar también mejoran 3 de 3 bloques | De acá salió la regla de réplica en Diamond |
| `brainstorm_29_draft_jerarquico` | Draft jerárquico: campeón además de campeón × línea | V −0,0002; bloques −0,0004 [−0,0007; −0,0000] | Empeora |
| `brainstorm_30_verifica_g11`, `brainstorm_33_verifica_g09`, `brainstorm_35_verifica_g23` | Verificación (placebo, Diamond-V) de los escalones 6, 7 y 9 | Confirmados | Verificaciones: los tres escalones **entraron** |
| `brainstorm_34_verifica_escalon8`, `brainstorm_36_verifica_escalon10`, `brainstorm_g_25_stack_y_perjug`, `brainstorm_g_33_lk15_y_esc10_gpu_blend` | Ensambles con redes y CatBoost sobre los escalones 7 y 9 | Escalón 10: 0,6749, +0,0010 [+0,0001; +0,0019] (marginal) | Marginal y requiere GPU: se prefirió la logística |
| `brainstorm_37_verifica_g30`, `brainstorm_g_31_macro_y_combat_kalman` | Kalman 1v1 por línea @15 y su descomposición | V +0,0010, pero bloques y Diamond-V con IC en 0 | No pasa la regla |
| `brainstorm_38_c_escalon9`, `brainstorm_g_39_diagnostico_residuos` | ¿Bajar C? ¿Queda estructura en el residuo? | C = 0,01: +0,0009 en bloques, −0,0003 en V; la mayor de ~75 sondas da \|z\| = 2,96 (azar) | Se queda C = 0,05: no queda nada explotable |
| `brainstorm_g_02_lp_vivo` | ¿Cuánto predice el LP real en vivo (4 servidores)? | LP solo 0,505; con W/L de temporada 0,549 | Diagnóstico: motivó el W/L de la foto (escalón 3) |
| `brainstorm_g_04_timeline_ext` | Timeline @10/@20, margen ajustado por rival, volatilidad, rendición temprana | −0,0007 a +0,0005 | No suma |
| `brainstorm_g_12_comp_draft_bilineal` | Subroles finos de Meraki y draft bilineal | ≈ 0 | No suma |
| `brainstorm_g_14_side_y_regularizacion` | L2 distinta por bloque de columnas; asimetría azul / rojo por el counterpick | L2 por bloque: bloques +0,0005; counterpick: V +0,0001 | Ajuste fino opcional; el counterpick no suma |
| `brainstorm_g_16_refina_g11` | Ablación de los sub-bloques del escalón 6 | CLUTCH no aporta | Se dejó afuera ese sub-bloque |
| `brainstorm_g_17_parche_y_skill15` | Parche exacto, Elo individual @15, cambio de rol | ≈ 0 | No suma |
| `brainstorm_g_18_cruce_tempo` | Choque de tempo entre equipos | ≈ 0 | No suma |
| `brainstorm_g_19_foto1_nonlin_y_kalman_rol` | Interacciones no lineales de la foto; Elo por rol | Empeoran; Elo por rol V +0,0005 pero Diamond −0,0006 | No pasa la regla |
| `brainstorm_g_20b_ablate_build`, `brainstorm_g_20c_full_ad_y_comp` | Ítems, trinket y riesgo "full AD contra tanques" | ≈ 0; full AD +0,0004 | No suma (hallazgo descriptivo real, pero chico) |
| `brainstorm_g_21_draft_y_pool_fresco` | Draft por recencia; pool reciente del jugador | −0,0016 a −0,0023; bloques −0,0010 | Empeora |
| `brainstorm_g_27_heteroscedasticidad` | Residuo económico y varianza por jugador | ≈ 0 | Techo de información previa |
| `brainstorm_g_32_rematch_tilt_y_sesion` | Reencuentros en menos de 75 min, tilt y sesión | ≈ 0 | Techo de información previa |
| `brainstorm_g_34_mid_eff_y_grafo_lsqr`, `brainstorm_g_35_svd_matchups_15` | Grafo LSQR; factorización bilineal de matchups 1v1 y 2v2 @15 | Descartadas en la bitácora | Techo de información previa |
| `brainstorm_g_36_lobby_y_estilo_fit`, `brainstorm_g_37_mismatches_tempo_neutral` | W/L del lobby, encaje estilo-campeón, choques en la misma línea, tempo | ≈ ±0,0002 | Techo de información previa |
| `brainstorm_g_38_reg_bloques_y_stack` | Podar las 200 columnas estáticas del campeón; stacking por bloques con meta HGB | Descartada. Hallazgo: las columnas estáticas sirven de prior de los one-hot raros (sacarlas: −0,0010) | No se podan |

**Conclusión de la noche.** Desde el escalón 9, once ideas seguidas dan ≈ 0. La información previa a la
partida que se puede sacar de este dataset para Master+ parece cerca del techo. El próximo salto necesita
datos nuevos (runas de toda V, bans y rol pedido de la respuesta cruda de match-v5), no transformaciones
de los mismos datos.

---

## `escalon9/` — la cadena original del escalón 9 (15/11, §9.S.52, §9.S.68; movida el 23/11, §9.S.79)

**Idea.** El modelo previo vigente se armaba con una cadena de scripts de la noche de brainstorming: cada uno
calculaba un grupo de variables, lo guardaba en `data/analysis/brainstorm/` y `brainstorm_final.py` ajustaba la
logística final. **Resultado:** 0,6740 / 62,6 % en V Master+ (0,6727 / 62,4 % con la máscara del nivel histórico,
§9.S.75). **Por qué quedan acá:** desde §9.S.73 el mismo modelo se calcula con `src/ganador/` (las funciones de
estos scripts copiadas tal cual por `tools/generar_constructores_ganador.py`, que lee estos archivos), con cachés
que se invalidan solas, y daba las mismas 16.428 probabilidades bit a bit. Siguen corriendo y siguen armando las
cachés de `data/analysis/brainstorm/` que leen `brainstorm_base.extras_referencia` y sus usuarios (matchmaking,
`modelos_train_v`, `umap_vistas`, el notebook). `brainstorm_final.py` queda como **referencia histórica** del
escalón 9 (`modelo_previo.py --comparar` lo usa como control).

Orden para rehacer las cachés (§9.S.69): `embeddings_partidas` y `ganador_mapa_lp` (en `scripts/analysis/`) y después:

| Paso | Script | Caché | Qué arma | En `src/ganador/` |
| :---: | :--- | :--- | :--- | :--- |
| 3 | `brainstorm_05_jugador` | `margen_grupos.npz` | Margen por (jugador, campeón) y (jugador, línea): escalón 2 | `margen_campeon_linea` |
| 4 | `brainstorm_07_mmr_historico` | `mmr_historico.npz` | Nivel histórico por emparejamiento: escalón 3a | `nivel_historico` |
| — | `brainstorm_g_01_foto1_wl` | (se calcula al vuelo) | W/L de la foto 1 dinámico: escalón 3b | `wl_foto1` |
| — | `brainstorm_08_combinado`, `brainstorm_g_03_agregados` | `pred_combinado.npz` | Escalera combinada de la noche (la leen `g_08` y siguientes) | — |
| 5 | `brainstorm_g_07_matchups_parche_rematch` | `gemini_mpr.npz` | Contracruces y sinergias: escalón 4 | `contracruces` |
| 6 | `brainstorm_g_10_nuevas_senales` | `gemini_nuevas_v2.npz` | Winrate por línea y caída por autofill: escalón 4b | `winrate_linea` |
| 7 | `brainstorm_g_11_elo_clutch_tempo` | `gemini_g11.npz` | Kalman-Elo, tempo, objetivos y economía: escalón 6 | `kalman_elo` |
| — | `brainstorm_g_06_comp_poisson` | — | Composición del draft (atributos de Meraki y CommunityDragon) | `composicion_campeones` |
| 8 | `brainstorm_g_09_hgb_lineas_causal` | `gemini_hgb_lineas.npz` | Modelo por línea (oro @15 predicho, HGB causal): escalón 7 | `oro_linea_hgb` |
| 9 | `brainstorm_g_23_duelo_res_y_per_jug` | `gemini_g23.npz` | Margen esperado por par de campeones: escalón 9 | `margen_esperado_par` |
| — | `brainstorm_final` | `pred_final.npz` | Ajusta y evalúa el modelo final | `modelo_previo` |

---

## `ensamble_en_vivo/` — el anexo: ensamble `6_Final_dEns10` (16-17/11, §9.S.57, §9.S.60; movido el 23/11, §9.S.79)

**Idea.** Predecir a los 10 / 15 / 20 minutos con un ensamble en dos etapas: dos logísticas y tres redes
multitarea en GPU sobre el estado de la partida, con el previo del escalón 10 (red + CatBoost sobre el escalón 9).
**Resultado:** 0,7912 / 0,8441 / 0,8750 en V Master+, cada minuto en su cohorte (0,7652 / 0,8202 / 0,8750 en la
misma cohorte). **Por qué quedan acá:** por decisión del usuario (§9.S.78) se presenta la logística de
`src/ganador/modelo_en_vivo.py`, que corre en CPU y sin estas cachés; el ensamble le gana por ~+0,002 de AUC pero
necesita GPU, ~54 min, el previo del escalón 10 y cachés de variantes descartadas, y se eligió entre más de 50
configuraciones mirando V. Queda como anexo con sus números de §9.S.60, sin rehacer.

| Script | Rol en el ensamble |
| :--- | :--- |
| `ingame_preparar` | Caché causal del estado a los 10/15/20 (`data/analysis/ingame/ingame_cache.npz`, con `L_oof`). La siguen leyendo `ingame_evaluar`, `ingame_eventos`, `ingame_estado_puro`, `eventos_ablacion` y `oro_equivalente_roles` (vigentes) |
| `ingame_g_01_prior_y_gpu`, `ingame_g_03_regimen_y_senales` | Funciones que importa el paso final; `g_03` es también la variante 5b (§9.S.60 punto 3), origen del modelo en vivo vigente |
| `ingame_g_04_mt_a4_gpu` | El ensamble `6_Final_dEns10` (GPU, ~54 min) → `pred_mt_a4_gpu_92k.npz`, que lee el notebook `eda_modelos_ganador` |
| `ingame_g_05_sin_previo` | El mismo ensamble sin previo (comparación "solo el estado") |
| `brainstorm_g_28_redes_esc9_perjug` | Redes y CatBoost sobre el escalón 9 → `pred_g28_esc9.npz` (previo del escalón 10; lo lee también `umap_vistas`) |
| `brainstorm_18_mc_ajuste`, `brainstorm_24_escala_sin_v`, `brainstorm_27_conoce_matchup`, `brainstorm_g_24_maestria_y_per_jug`, `brainstorm_g_26_hgb_ctx_y_post15` | Cachés de entrada (`mc_ajuste`, `margen_grupos_escala_sin_v`, `conoce_matchup`, `gemini_g24`, `gemini_hgb_ctx_p15`). Como modelos sueltos se descartaron en la noche (`docs/BRAINSTORM_NOCHE.md`) |
| `brainstorm_g_05_gpu_master`, `brainstorm_g_08_destilacion_gpu`, `brainstorm_g_13_attn_gpu_catboost`, `brainstorm_g_22_red_y_catboost_esc6` | Redes y CatBoost en GPU de los escalones 5-7 cuyas clases y funciones importa `g_28` |

---

## `en_vivo/` — predicción durante la partida (§9.S.52, §9.S.60, §9.S.71)

| Script | Idea | Resultado | Por qué se descartó |
| :--- | :--- | :--- | :--- |
| `brainstorm_39_en_vivo` | Dos etapas: logit previo del escalón 9 + estado del equipo (Δoro, Δxp, Δkills) | 0,7774 / 0,8288 / 0,8656 a los 10 / 15 / 20 | Lo superó la versión por líneas |
| `brainstorm_40_en_vivo_lineas` | Ídem, con el oro de cada línea y el oro que se esperaba de ella | 0,7842 / 0,8377 a los 10 / 15 | Lo superó `6_Final_dEns10` (0,7912 / 0,8441 / 0,8750) |
| `brainstorm_41_aporte_lineas` | ¿Cuánto aporta el modelo por línea? | +0,0108 sobre la base; en el final, sacarlo cuesta −0,0009 | Diagnóstico: el modelo por línea sigue en el final |
| `brainstorm_46_en_vivo_mejoras` | Agregados nuevos sobre la segunda etapa por líneas | Ninguna variable con IC > 0, ni en V ni en A4 (§9.S.71) | No suma |
| `ingame_convertir` | ¿El jugador convierte la ventaja o la tira? | +0,0002 [−0,0001; +0,0004], 1 de 3 bloques (§9.S.60) | No suma |
| `ingame_g_06_a24_explore` … `ingame_g_06f_combo_a24`, `ingame_g_07_final_v` | Red v2 con atención entre líneas y señales nuevas, elegidas solo en A2-A4 y con V mirada una vez | −0,0008 a −0,0023 contra el ensamble, que les da peso negativo (§9.S.60 punto 7) | No supera al vigente |

---

## `ideas_posteriores/` — ideas y diagnósticos sobre el escalón 9 (16-17/11)

| Script | Idea | Resultado | Por qué se descartó |
| :--- | :--- | :--- | :--- |
| `brainstorm_42_xp_lineas` | Modelo por línea (el HGB causal de `escalon9/brainstorm_g_09`) para la XP @15 y el oro y la XP @10 | `hgb_lineas_extra.npz`, entrada de la segunda etapa en vivo por líneas (`en_vivo/brainstorm_40`) | Ese modelo en vivo lo superó el anexo y después la logística de `src/ganador/`; la caché la sigue leyendo `umap_vistas` (vista en vivo) |
| `brainstorm_43_estado_esperado` | Pasar por el modelo en vivo el estado *esperado* a los 10/15/20, armado con los promedios del jugador | Bloques +0,0006 a +0,0008; V +0,0003 a +0,0005 con IC en 0 (§9.S.61) | Marginal: es una transformación de márgenes que el modelo ya tiene |
| `brainstorm_44_builds` | Rareza y calidad de las builds del jugador | V +0,0002, IC en 0 (§9.S.61) | No suma; la "rareza" refleja ir atrasado |
| `brainstorm_45_medias_jugador_linea` | Media del estado del jugador en esa línea en sus partidas anteriores | −0,0004 a +0,0001 (§9.S.71) | No suma: ya está en las 85 variables por jugador |
| `brainstorm_48_items` | Build general por campeón, builds raras y winrate suavizado | [−0,0001; +0,0006], placebo −0,0001 (§9.S.71) | No suma |
| `brainstorm_43_poda_l1` | Achicar el modelo final con L1 | 575 columnas: −0,0012 [−0,0022; −0,0001] (§9.S.69) | Toda poda cuesta AUC: se quedan las 1.504 columnas |
| `brainstorm_47_relacion_features` | PCA y clustering jerárquico de las variables, poda por redundancia | No hay poda gratis fuerte (§9.S.71) | Diagnóstico |
| `brainstorm_44_errores_emparejamiento` | Qué errores del emparejamiento de Riot explota el modelo | Lo que predice es lo que el emparejamiento no mira (§9.S.70) | Diagnóstico, no un modelo |
| `tabm_esc9_gpu` | TabM (MLP con BatchEnsemble) como experto tabular | Solo 0,6693 (−0,0046); en stacking con peso ≈ 0 (§9.S.62) | La red no llega a la logística |
| `ablacion_draftrec` | ¿Cuánto pierde el modelo con solo la información del dataset de DraftRec? | Estimación para el puente de la comparación (§9.S.59) | Paso intermedio de la comparación con DraftRec |

---

## `clustering_huella/` — clustering de jugadores y huella de estilo (§9.S.2, §10)

| Script | Idea | Resultado | Por qué se descartó |
| :--- | :--- | :--- | :--- |
| `cluster_global_vs_rol`, `cluster_escalado` | Clustering global contra por rol; ventaja a los 15 contra escalado contra estilo | Silueta débil (máxima en k = 2): no hay grupos naturales (§9.S.2) | La versión vigente está en `cluster_emeans` y el notebook `eda_reconocimiento_jugadores` |
| `boosting_en_victoria` | Ante sospecha de boosting, usar solo el tramo boosteado del historial | Mejora mucho en el subgrupo sospechoso, pero en toda V ΔAUC +0,00018 (§10.8) | Efecto global despreciable |

---

## Carpetas de etapas anteriores

* **`elo_principal_oct/`** — el "modelo principal" del elo al revés (28-29/10, §9.S.21): reconstruir el LP
  entre las dos fotos del ranking. Son las corridas tal como se hicieron, con su `bitacora.md` y su
  `registro.tsv`. Lo vigente para ese problema es `scripts/analysis/elo_boostings.py` y `elo_gru.py`, con
  el notebook `rating_jugadores`. Ver el `README.md` de la carpeta antes de correr nada: escriben cachés
  grandes al lado del script.
* **`mecanica_lp/`** — pruebas exploratorias de la mecánica del LP medida en vivo (02/11, §9.S.40). Lo que
  sirvió pasó a `src/features/lp_mecanica.py` (§9.S.43). El mapa de LP que salió de ahí dejó de aportar al
  modelo final cuando entraron el nivel histórico y el W/L de la foto (§9.S.52).
