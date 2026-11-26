# TAREAS.md — Tablero Operativo del Día a Día

> **Lectura obligatoria al inicio de cada sesión, antes de tocar nada.**
> Este archivo responde una sola pregunta: **¿qué hay que hacer ahora?**
> Es el único lugar donde viven las tareas pendientes del día a día.

---

## 1. Cómo se usa

Tres reglas, y ninguna es opcional:

1. **Al abrir sesión:** leer §2 (En curso) y §3 (Abiertas). Lo que esté "En curso" de una
   sesión anterior se retoma o se cierra antes de abrir una tarea nueva.
2. **Al aparecer una tarea:** entra en §3 en el momento en que se identifica, aunque no se vaya
   a hacer hoy. Una tarea que solo existe en la conversación no existe.
3. **Al terminar una tarea:** pasa a §4 con fecha ISO. Cuando el cambio se cierra formalmente
   (ver [`README.md`](README.md), "Para contribuir"), su fila pasa al changelog de [`AGENTS.md`](AGENTS.md) §7 y
   **se borra de acá**. Este archivo no acumula historia.

| Tipo de cosa | Dónde vive |
| :--- | :--- |
| Tarea concreta, accionable, del día a día | **acá**, §3 |
| En qué etapa está el proyecto y qué compuerta falta abrir | [`ROADMAP.md`](ROADMAP.md) §3 |
| Qué se hizo y por qué (historia cerrada) | [`AGENTS.md`](AGENTS.md) §7 |
| Justificación matemática o de diseño de una decisión | [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md) |
| Sesgos del dataset y su corrección diseñada | [`AGENTS.md`](AGENTS.md) §3.F |

Formato de una tarea:

```markdown
- [ ] **Título corto.** Qué hay que hacer, en una o dos frases.
      *Bloquea:* qué queda trabado mientras esto no se haga (o `—`).
      *Costo:* estimación real (tiempo, requests, riesgo).
      *Estado:* qué falta para poder arrancar (decisión del usuario, key, volumen…).
```

> **Depuración del 2026-11-19.** Se sacaron del tablero, por decisión del usuario, el
> recomendador de picks y las runas/hechizos (sus resultados quedan en `PROJECT_SUMMARY.md`
> §9.S.37 y §9.S.59), y todas las tareas ya cerradas o superadas por el escalón 9: la recolección
> y el congelamiento del 26/10, la tabla de features (hoy `brainstorm_base.py`), el modelo de 2
> expertos y sus variantes (§9.S.23-39, superados por §9.S.52), `seguimiento_lp.py` y lolpros
> (reemplazados por `lp_vivo`, §9.S.22/40). La historia de cada una está en `AGENTS.md` §7.

---

## 2. En curso

- [ ] **Quién gana: bajar la respuesta cruda de match-v5 (bans, rol pedido y autofill, `challenges`).**
      El crawler no guarda nada de eso (§9.S.71). `python scripts/collection/partidas_crudas.py --n 30300` baja A2-A4 Master+ y V
      a `data/crudas/partidas.db` (JSON entero comprimido; corta y retoma). Después: segunda etapa sobre el logit del escalón 9 con
      bans (¿le banearon el campeón principal?), autofill exacto y rareza de runas en toda V; comprobar si
      `challenges.legendaryItemUsed` está en orden de compra (ítems) y probar `challenges` como historial del jugador.
      *Bloquea:* las únicas variables previas nuevas que quedan por probar.
      *Costo:* ~8 h 45 con una key de desarrollo (vence a las 24 h; la del 16/11 está en el `.env` de la notebook); ~300 MB.
      *Estado:* script y tests listos, 2 partidas de prueba revisadas; **esperando que el usuario confirme la descarga.**

- [ ] **Timelines con eventos de toda la base y entrada al modelo en vivo (`PROJECT_SUMMARY.md` §9.S.64 y §9.S.66).**
      Con 62.706 timelines (V completa + 46.278 Master+ del ajuste) los eventos suman al modelo en
      vivo +0,0055 / +0,0060 / +0,0070 de AUC a los 10 / 15 / 20 (IC > 0, 3/3 tramos); pesan sobre
      todo los dragones, la primera sangre, el oro sin gastar y los muertos en el minuto.
      *Falta:* (1) terminar la descarga **por lotes**, sin `--todas` (decisión del usuario 2026-11-24:
      `--todas` bajaba ~150.000 que no se usan). Log `logs/timelines_ids_20261124.log`. `diam_v` (10.506) y
      `a24_master` (14.513) **completas**; `test_master` (38.001 / 38.013, 12 sin timeline) **completas: descarga
      terminada el 2026-11-25 15:58**, 14,4 GB comprimidos, 0 errores HTTP. Retoma con
      `timelines_eventos.py --ids data/timelines_eventos/ids_N_*.txt --grupo X` (§9.S.64); estado en `--resumen`; (2) procesar
      el crudo una vez a tablas compactas; (3) decidir si los eventos entran al modelo en vivo que se
      presenta (desde la decisión de §9.S.78, la etapa 1 sola; los eventos entran como etapa 2): dan +0,0054 / +0,0055 / +0,0064 en V
      y bloques A3-A4 con IC > 0 en los tres minutos (`modelo_en_vivo_eventos.py`), pero falta la
      réplica en Diamond-V, que no tiene timelines (~10.500 partidas, ~2 h con P0+P1). **Esperando la
      decisión del usuario:** bajarlos y medir la réplica, o aceptar los bloques como réplica.
      *Bloquea:* el modelo en vivo final que va al test.
      *Costo:* ~35 h de descarga desatendida + ~1 día de procesamiento y evaluación.

---

## 3. Abiertas

### A. Camino a la entrega

- [ ] **Terminar de sacar `brainstorm_base` de lo vigente (issue #2, `PROJECT_SUMMARY.md` §9.S.79).** Los
      `brainstorm_*` del escalón 9 y del anexo ya están en `historial/`, pero `scripts/analysis/brainstorm_base.py`
      (`cargar`, `ajustar_predecir`, `extras_referencia`) sigue en uso por matchmaking, `modelos_train_v`,
      `acierto_por_lp_duracion`, `diamante_conocidos`, `huella_previo`, `umap_vistas`, `partidas_crudas` y el notebook
      `eda_modelos_ganador`, y `extras_referencia` lee las cachés que arma `historial/escalon9/`. Pasar cada uno a
      `src.ganador` (`variables.calcular_todas` + `modelo_previo`) y comprobar que dé lo mismo.
      *Bloquea:* que lo vigente no dependa de cachés armadas desde `historial/`. *Costo:* ~1 h por consumidor más una
      corrida con datos para comparar; el notebook hay que regenerarlo. *Estado:* después de regenerar `mmr_historico.npz`
      (la tarea de abajo), para que la cadena vieja y `src.ganador` coincidan y la comparación sea exacta.

- [ ] **Preparar la evaluación final con los hallazgos de la [revisión de modelos](docs/REVISION_MODELOS_20261120.md).** Unificar la definición de lobby Master+ entre V y el split oficial, fechar los resultados históricos por fin de partida y usar la selección de variables sin campeón en la comparación jugadores–draft. Registrar el uso previo del test por los baselines (§9.S.13 y §9.S.20) al fijar el protocolo final.
      *Bloquea:* la comparación principal de la memoria. *Estado:* revisión documentada; cambios de modelado pendientes.

- [ ] **Medir la disponibilidad de LP anterior a la foto 1 en el ajuste pre-partida (§9.S.68).**
      Probar un nivel histórico que desconozca el LP del 04/09 para partidas anteriores a la foto.
      Reportar el delta pareado en V Master+ y Diamond-V. Para usar A2–A4 como evidencia hacia
      adelante, estimar escalas e imputaciones con el prefijo de cada corte.

- [ ] **Congelar los modelos finales y el protocolo de evaluación.** Antes de evaluar los
      modelos finales con `data/splits/test/`, dejar escrito en `PROJECT_SUMMARY.md` qué se evalúa, con qué
      entrenamiento y qué métricas (AUC, acierto, log loss, ECE y cobertura): pre-partida =
      escalón 9 sin el mapa mecánico de LP (§9.S.52 y §9.S.68), en vivo =
      `6_Final_dEns10` (con eventos si pasan la regla), y el reacomodo de matchmaking (§9.S.67).
      Con ~40 variantes elegidas mirando V, esperar algo menos en el test.
      *Bloquea:* la Etapa 6 y la memoria.
      *Costo:* una corrida por modelo.
      *Estado:* después de incorporar los timelines al modelo en vivo. Decisión del usuario: con qué datos se
      reentrena para el test (solo hasta el 16/09 como en V, o todo el train hasta el 27/09).

- [ ] **Memoria técnica y cierre de notebooks (Etapa 7).** Redactar el informe desde
      `PROJECT_SUMMARY.md` y dejar los notebooks sin números viejos:
      - `eda_visualizer.ipynb`: tres decisiones pendientes desde el 26/10 (§9.S.19): (a) varianza
        dentro y entre jugadores y poder univariado sobre la victoria; (b) si la sección 19 (WR ≥ 75 %)
        se queda; (c) cómo cambiar las secciones 10-11 (el ΔWR entre parches es casi todo ruido).
      - `eda_reconocimiento_jugadores.ipynb` (§1.1, ex `clustering_jugadores.ipynb`): decidir el k
        (`K_OPT = 5`, `None` o gap / estabilidad por
        bootstrap). El silhouette es máximo en k = 2 y no hay grupos naturales (§9.S.2, §9.S.53).
      - `embeddings.ipynb`: quitar los valores escritos a mano (`1.9343`, `analogía 1.93`, `azar ~6.8`).
      *Bloquea:* la entrega.
      *Costo:* medio-alto.
      *Estado:* las decisiones de los notebooks esperan al usuario; la redacción, a que cierre el test.

- [ ] **Pregunta de investigación en `ROADMAP.md` §1.** El trabajo real terminó respondiendo
      "quién gana" (draft contra jugadores, antes y durante la partida), con la huella de estilo como
      rama aparte. Acordar con el compañero la formulación final y reescribir §1 con ella.
      *Bloquea:* la introducción de la memoria.
      *Costo:* una conversación.
      *Estado:* esperando al usuario y al compañero.

- [ ] **Ordenar git antes de commitear.** La rama local `gonchi` (`3d863b4` + cambios sin
      commitear de matchmaking y del fix de la fuga) divergió de `origin/gonchi` (`a34b4c6`, del
      18/11: timelines en el respaldo a Drive, 4 tests nuevos, §3.1/§3.3 del EDA): hay que integrar
      las dos antes de pushear. `COORDINACION.md` quedó archivado en `docs/archive/`.
      *Hook `pre-push`:* está instalado aunque `PROJECT_SUMMARY.md` documenta la decisión de no
      tenerlo; sube la BD de 2 GB en cada push (4-7 min, el 16/11 se cortó por timeout).
      *Avance (2026-11-20):* integrado. `origin/gonchi` (`a34b4c6` + `23cbe0e`, EDA de reconocimiento de
      jugadores y boosting en victoria) se mergeó con `3d863b4` (commit `f6941c8`) y los cambios sin
      commitear se reaplicaron encima, todavía sin commitear (`AGENTS.md` §7, fila del 2026-11-20).
      *Estado:* falta commitear el trabajo local y pushear; esperando decisión del usuario (cuándo y si
      el hook queda).

### B. Infraestructura (sin apuro)

- [ ] **Export por bloques y escritura atómica de los CSV.** `export_to_csv.py --target splits`
      se muere en silencio sin RAM (PC de 8 GB con todo abierto) y escribe con `df.to_csv(destino)`
      directo: un corte deja el CSV a medias. Escribir por bloques con igualdad de bytes, a
      `destino.tmp` + `os.replace`, con tests. Mientras tanto, exportar con todo cerrado.
      *Costo:* bajo-medio. *Estado:* listo; no tocar mientras corre el crawler (§3.G).

- [ ] **Pendientes de la revisión del crawler del 18/11 (§9.S.65).** Auditar los enlaces de LP
      previos al cambio de "candidata única"; evaluar `idx_timeline_match` en una copia; probar
      paralelizar `build_rating.py`, vectorizar el AUC y medir cachés de CSV.
      *Costo:* bajo cada una. *Estado:* listo.

- [ ] **Si se vuelve a recolectar.** Medir con el monitor abierto minutos entre latidos,
      requests/min por clave, partidas nuevas/min, memoria de SQLite, el reparto P0/P1 y cuánto
      agregan al cierre la consulta de historial, el resumen y el export. Marcar a los jugadores con
      historial agotado para que `get_canonicals_for_deep_crawl` no los repita (§3.A, con tests).
      *Estado:* solo si se reabre la recolección (el dataset está congelado en 272.849 partidas).

### C. Ramas opcionales (esperan decisión del usuario)

- [ ] **Huella de estilo, niveles 2 a 5 (`PROJECT_SUMMARY.md` §10).** El nivel 1 está cerrado
      (AUC 0,824, §10.2). Para seguir hacen falta, en orden: (1) decidir qué hacer con las partidas
      de Master+ con Diamantes en el lobby (dejarlas, condicionar o solo lobbies puros; números en
      `eda_huella.ipynb`, borrado el 2026-11-19: `eda_reconocimiento_jugadores.ipynb` ya no trae esa
      sección, se recupera con `git show cd07270:notebooks/eda_huella.ipynb`); (2) fijar los criterios de éxito antes de entrenar (p. ej.
      ganarle a `pool`, `elo_linea` y `crudo` con IC95 sin solaparse, y un máximo de falsas alarmas
      cada 100 partidas); (3) verificar que `data/backups/lol_las_pre_fix_map_20261023.db` contenga
      las ~11 fusiones reales para la prueba ciega del nivel 4; (4) niveles 2-5.
      *Avance (2026-11-18/20, rama remota):* mejoras del nivel 1 y su aporte al modelo previo (§10.4),
      niveles 2 y 3 con jugadores no vistos (§10.5, `huella_niveles.py`), EDA unificado
      `eda_reconocimiento_jugadores.ipynb` (§10.6), patrones intra-cuenta (§10.7), perfil del tramo
      boosteado en el modelo de victoria (§10.8) y anatomía de la huella por subgrupo (§10.9). Falta el
      nivel 4 (y el 5); los puntos (1) a (3) siguen pendientes.
      *Costo:* alto.

- [ ] **Mapa de LP partida a partida como capítulo propio (§9.S.40, 43, 46, 49, 54).** Al agregarlo
      al escalón 9 corregido, el AUC de V cambia −0,00030 [−0,00136; +0,00074] (§9.S.68).
      Si el usuario lo quiere como resultado en sí, lo pendiente es:
      decidir el plan de `lp_vivo` (sacar BR/NA, sumar Diamond de LAS, filtrar rebotes, guardar
      `timePlayed` y rendiciones); sondear si Match-V5 delata Aegis o la devolución (~1.400 pedidos);
      bajar a una base aparte las partidas entre fotos y de julio-agosto; foto 3 del ladder;
      reentrenar el principal con ancla y ventana (`historial/elo_principal_oct/`); calibrar
      `c_lobby` / `sigma_lobby` y generar el mapa por tramos (versión descriptiva y causal, el CSV
      después del test). Del elo al revés quedan: regularizar NGBoost, piso/inactividad y promedio
      GRU + GB, lobby iterado, la foto 2 de los 9.129 Diamond fuera del ladder y la inflación de LP de Master+ en la recorrida.
      *Costo:* alto. *Estado:* esperando decisión del usuario.

- [ ] **Demo en vivo (idea, sin decisión).** Predecir en la pantalla de carga vía Spectator-v5,
      nunca en la selección de campeones (Riot prohíbe des-anonimizar ahí), saltear `puuid` nulo,
      no mostrar MMR ni conectar apuestas, registrar el producto en el Developer Portal, y antes
      verificar que el pipeline en vivo da las mismas variables que el offline y reentrenar con
      datos recientes (el train termina el 27/09).

---

## 4. Cerradas recientemente

> Se vacía cuando la fila correspondiente entra en [`AGENTS.md`](AGENTS.md) §7.

| Fecha (ISO) | Tarea | Resultado |
| :--- | :--- | :--- |
| 2026-11-22 | Merge de `origin/gonchi` y reorganización del repo para la entrega | Lo vigente queda en `scripts/analysis/` (78 scripts, con `README.md`); los experimentos descartados, en `historial/` con idea, resultado y por qué se descartaron. Guarda de imports nueva. [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md) §9.S.72 |
| 2026-11-16 | Medias por jugador y línea a 10/15/20, mejoras en vivo, PCA / clustering de variables e ítems | Medias: no suman (−0,0004 a +0,0001). En vivo: versión por líneas al minuto 20, 0,8695 / 78,6 % (+0,0084); ajustar también con Diamond +0,0005; el resto no suma. Variables: 653 → 300-400 sin perder. Ítems: builds raras −0,73 % (−1,53 % sin experiencia con el campeón); no suman al modelo. [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md) §9.S.71 y [`AGENTS.md`](AGENTS.md) §7. |
| 2026-11-12 | Reemplazar la BD local corrupta (PC del usuario, rama `gonchi`) | La copia del 26/10 daba `database disk image is malformed`; quedó en `data/backups/lol_las_corrupta_20261026.db`. Nueva `data/lol_las.db` = export del 01/11 en Drive (2.169.212.928 bytes, `quick_check` ok, 272.849 partidas, 40.821 jugadores). Splits viejos apartados en `data/backups/splits_viejo_20261024/`; se usaron los de `data_completa/splits` de Drive porque `export_to_csv.py --target splits` murió en silencio (ver §3.B). Solo datos locales: no cambia código. |
