# ¿Quién gana? Partidas Diamante+ de League of Legends (LAS)

Proyecto final de Machine Learning. Recolectamos con la API de Riot las partidas de Ranked Solo/Duo
de los jugadores de Diamante o superior del servidor LAS y estudiamos qué predice mejor quién gana:
el draft (los campeones elegidos) o los jugadores, antes de la partida y durante ella.

**Resultado en el test** (Master+, desde el 27/09, evaluado una sola vez):

| Modelo | AUC | Acierto |
| :--- | :---: | :---: |
| Antes de la partida | 0,677 | 63,0 % (siempre rojo: 52,5 %) |
| En vivo, minuto 10 / 15 / 20 | 0,792 / 0,844 / 0,873 | 71,6 / 76,0 / 78,7 % |

El recorrido completo está en [`notebooks/informe_final.ipynb`](notebooks/informe_final.ipynb).

## Autores

- Gonzalo Malaspina
- Gonzalo Lopez

## Arquitectura del repositorio

```
run.py, collect_data.py      recolección desde la API de Riot (crawler con SQLite)
export_to_csv.py             exporta la base a los 4 CSV de data/ y a los splits train/test

src/
  collector.py, riot_client.py, rate_limiter.py, database.py   crawler, cliente, cuota y base
  features/                  variables: draft, rating, nivel del lobby, huella de estilo, split
  ganador/                   modelos finales: previo (antes de la partida) y en vivo (10/15/20)
  informe/                   figuras y tablas del notebook del informe
  models/                    validación temporal, clustering, en vivo, baselines

scripts/
  analysis/                  lo vigente: modelos, evaluación del test y builders de notebooks
  collection/                recolecciones auxiliares (timelines con eventos, LP en vivo)
  ops/, identity/, backup/   diagnóstico, identidad de jugadores, respaldo a Drive

notebooks/                   informe_final (la entrega) y los EDA de cada tema
historial/                   experimentos descartados: idea, resultado y por qué
docs/evaluacion_test/        protocolo y log de la evaluación única en el test
tests/                       pytest
data/                        generado, no versionado (base, CSV, splits, cachés)
```

La documentación técnica completa, con cada decisión y su justificación, está en
[`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md).

## Librerías

Python 3.12. Todo está en [`requirements.txt`](requirements.txt):

```bash
pip install -r requirements.txt
```

- **Base:** `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `jupyter`, `pytest`.
- **Recolección:** `requests`, `python-dotenv`, `tqdm`, `rich`.
- **Algunos modelos y análisis:** `torch` (GPU opcional), `lightgbm`, `catboost`, `ngboost`, `pyarrow`,
  `umap-learn`, `hdbscan`. Con `faiss` además corre el clustering e-means.

## Manual de uso

**Datos.** No están en el repo: se bajan de Zenodo (DOI: _pendiente_) y se descomprimen en `data/`:

| Archivo | Va en | Qué es |
| :--- | :--- | :--- |
| `splits.zip` | `data/splits/` | Train, test y Diamante: con esto corren los modelos, la evaluación y el informe |
| `lol_las_db.zip` | `data/lol_las.db` | La base completa del crawler (regenera los CSV con `export_to_csv.py`) |
| `lp_vivo.zip` | `data/lp_vivo/` | LP partida a partida en vivo |
| `timelines.db` | `data/timelines_eventos/` | Timelines completos con eventos |

Los jugadores están seudonimizados: el ID de jugador (que en la recolección es el Riot ID, `nombre#tag`) se
reemplazó por un código fijo (`j` + 16 hex) igual en todos los archivos, y no hay nombres. El paquete se arma con
`python scripts/ops/empaquetar_zenodo.py`. Resultados, cachés y pesos de modelos no se publican: se regeneran
(el informe necesita antes `modelo_previo.py`, `modelo_en_vivo.py`, `huella_mejoras.py` y `matchmaking_swaps.py`;
la evaluación del test ya está versionada en `data/analysis/test_final/`).

```bash
# El informe: arma y ejecuta notebooks/informe_final.ipynb (~35 s)
python scripts/analysis/build_informe_nb.py --ejecutar

# Los modelos finales, medidos en la validación (V)
python scripts/analysis/modelo_previo.py      # antes de la partida (~45 s con caché)
python scripts/analysis/modelo_en_vivo.py     # en vivo a los 10/15/20 minutos (~12 min con caché)

# Los tests
python -m pytest
```

La evaluación en el test ya se hizo y no se repite (protocolo y log en `docs/evaluacion_test/`).
Para ensayarla sin tocar el test: `python scripts/analysis/evaluacion_test.py --simulacro`.

**Recolectar partidas** (hace falta una API key de Riot en `.env`, a partir de `.env.example`):

```bash
python run.py --minutos 240        # recolectar 4 horas y validar la base
python run.py --solo-validar       # auditar la base sin red
python export_to_csv.py            # regenerar los 4 CSV de data/
```

## Comandos útiles

```bash
# Estado y diagnóstico (sin red, salvo las keys)
python collect_data.py --summary                     # estado de la base de datos
python scripts/ops/check_api_keys.py                 # vigencia de las API keys
python scripts/ops/audit_duplicates.py               # jugadores duplicados o fusionados (~75 s)
python export_to_csv.py --target splits              # train/ y test/ por separado en data/splits/

# Recolecciones auxiliares (bases aparte, retoman solas)
python scripts/collection/timelines_eventos.py --ids ARCHIVO --grupo NOMBRE   # timelines con eventos por lotes
python scripts/collection/timelines_eventos.py --resumen                      # estado de esa base
python scripts/collection/lp_vivo.py --intervalo 120                          # LP por partida en vivo, Master+

# Modelos y análisis
python scripts/analysis/matchmaking_calibracion.py   # calibración del modelo previo en V (~100 s)
python -u scripts/analysis/matchmaking_swaps.py      # reacomodar equipos para partidas más parejas (~150 s)
python scripts/analysis/modelo_en_vivo_eventos.py --db RUTA/timelines.db      # eventos del timeline en el modelo en vivo
python scripts/analysis/draftrec_mejor.py            # nuestra metodología en el dataset de DraftRec (~10 min)
python scripts/analysis/validacion_bloques.py --hilos 4    # validación temporal por bloques (~12 min)
python scripts/analysis/curva_aprendizaje.py         # AUC según cuántas partidas ajustan el modelo (~4 min)
python scripts/analysis/umap_vistas.py --vista partidas    # UMAP + HDBSCAN (también: vivo, jugadores)
python scripts/analysis/lp_partida_modelo.py         # modelo del LP partida a partida (~40 s)
python scripts/analysis/build_rating.py              # rating TrueSkill por partida (~12 min)

# Los otros notebooks
python scripts/analysis/build_eda_modelos_nb.py --ejecutar          # eda_modelos_ganador
python scripts/analysis/build_eda_reconocimiento_nb.py --ejecutar   # eda_reconocimiento_jugadores (~8 min)
```

Cada script de `scripts/analysis/` está descripto en su [README](scripts/analysis/README.md), y los experimentos
descartados en el de [`historial/`](historial/README.md).

## Para contribuir

Un cambio está terminado cuando `python -m pytest` da verde, la decisión está explicada en
[`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md) (por qué se eligió y qué se descartó) y hay una fila nueva en el changelog
de [`AGENTS.md`](AGENTS.md) §7. Las reglas que ningún cambio puede romper (SQLite en Windows, rate limiting de Riot,
deduplicación, coherencia de los 4 CSV) están en [`AGENTS.md`](AGENTS.md) §3.

## Licencia

Código bajo licencia [MIT](LICENSE): se puede usar, copiar y modificar libremente, sin garantía de ningún tipo.
Los autores no se hacen responsables por ningún uso del código ni de los datos. Los datos vienen de la API de Riot
Games; League of Legends es una marca de Riot Games, Inc., que no avala ni patrocina este proyecto.
