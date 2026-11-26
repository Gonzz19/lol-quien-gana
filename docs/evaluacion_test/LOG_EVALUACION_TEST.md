# Evaluación única en el test

* Fecha y hora: 2026-11-25T19:14:51-03:00 → 2026-11-25T19:57:47-03:00 (43 min)
* Commit: `b1edaee0a5decc8655cede3810c2cfed168bf218`
* Protocolo: `docs/evaluacion_test/PROTOCOLO.md`, SHA-256 `a4026173bfa2144d55ce3b83e37410aab81694bab0b3bd5864a78374166a7b7a`
* SHA-256 `match_split`: `ceb5a0bbfb5d131bb82571c4691113b8c9d410d9ac214ba1932f5bcdae1f037e` (coincide con el protocolo)
* SHA-256 `train`: `effcd4af654b6693f55f75e8857075b9127acaa0043fbc573ed611c23387006a` (coincide con el protocolo)
* SHA-256 `test`: `c895559c6234ad3de9ce58fbfd23adabe8e2d80e42838d43a7bf81e43055b230` (coincide con el protocolo)
* SHA-256 `diamante_despues`: `85cbdba315719edd55f265436bec30f9892e7512f20a60e071880683509a4f4c` (coincide con el protocolo)
* Modo: `test`; regla de Master+: `split`; semillas: [42, 0, 1, 2, 3] (principal 42); B = 1000; ventana = 2500 partidas
* Versiones: python 3.12.14, numpy 2.5.3, pandas 3.0.5, scikit-learn 1.9.0, scipy 1.18.1, plataforma Windows-11-10.0.26200-SP0

## Declaración: cortes y relanzamientos (PROTOCOLO §7)

La corrida única (OK del usuario del 2026-11-23) no se hizo en un solo proceso. Ningún paso cambió el código, los
datos ni los hiperparámetros: todos los procesos corrieron sobre el commit `b1edaee` con el mismo protocolo, y las
predicciones de cada semilla se guardaron con la firma commit + modo + hash del protocolo para que el último proceso
las retomara.

1. **2026-11-24, proceso original** (`evaluacion_test.py`): calculó y guardó las semillas 42 (00:28) y 0 (01:04) y se
   cortó por falta de memoria durante la semilla 1. No había calculado ninguna métrica.
2. **2026-11-24, una semilla por proceso** (pedido del usuario, para liberar RAM entre semillas): un driver externo al
   repo llamó `predecir` + `guardar_parcial` del script, con las mismas guardas (log inexistente, árbol limpio,
   hashes). Guardó la semilla 1 (15:12) y la 2 (15:52). El intento de la semilla 3 (15:53) se frenó en la guarda
   del árbol de git, porque `notebooks/informe_final.ipynb` tenía dos `execution_count` cambiados, antes de
   cargar el test.
3. **2026-11-25**: se apartó ese cambio (`git stash`), el driver guardó la semilla 3 (19:14) y el script principal
   retomó las 5 semillas y calculó las métricas una sola vez (este log).

**Lo único que se vio del test antes del final** es el diagnóstico que el protocolo admite: cada proceso que ajustó
predicciones imprimió el AUC por línea del HGB del oro en las partidas evaluadas (por ejemplo, semilla 1: BOTTOM
0,6490, JUNGLE 0,6789, MIDDLE 0,6929, TOP 0,7105, UTILITY 0,6690). No es una métrica del modelo y no se usó para
cambiar nada. Ninguna métrica del previo ni del modelo en vivo se calculó ni se vio antes de este log.

## Chequeos

* `entrenamiento_anterior_a_la_evaluacion`: True
* `evaluacion_toda_master_regla_split`: True
* `previo_sin_nan`: True
* `previo_alineado_con_en_vivo`: True
* `en_vivo_10_sin_nan`: True
* `en_vivo_15_sin_nan`: True
* `en_vivo_20_sin_nan`: True
* `fraccion_vivas_10`: 0.994273577804065
* `vivas_10_suficientes`: True
* `inicio_es_el_corte`: True
* `evaluacion_son_las_del_test`: True
* `diamante_son_las_diamante_despues`: True
* `cobertura_test`: 0.9785073527477441
* `cobertura_test_suficiente`: True

## Métricas (semilla principal, IC 95 % por bootstrap de días)

| Modelo | Minuto | Conjunto | IC | n | días | AUC | Acierto | Siempre rojo | Ventaja | Log loss | ECE |
| :--- | :---: | :--- | :---: | ---: | ---: | :--- | :--- | ---: | :--- | :--- | :--- |
| previo |  | evaluacion | dias | 37,196 | 30 | 0.6765 [0.6721; 0.6811] | 0.6299 [0.6260; 0.6337] | 0.5251 | 0.1048 [0.0983; 0.1114] | 0.6419 [0.6395; 0.6442] | 0.0060 [0.0053; 0.0127] |
| previo |  | evaluacion | partidas | 37,196 |  | 0.6765 [0.6710; 0.6819] | 0.6299 [0.6251; 0.6349] | 0.5251 | 0.1048 [0.0982; 0.1116] | 0.6419 [0.6388; 0.6451] | 0.0060 [0.0056; 0.0129] |
| previo |  | evaluacion MASTER | dias | 35,133 | 30 | 0.6778 [0.6735; 0.6824] | 0.6304 [0.6265; 0.6342] | 0.5259 | 0.1045 [0.0980; 0.1113] | 0.6410 [0.6384; 0.6435] | 0.0062 [0.0055; 0.0126] |
| previo |  | evaluacion GRANDMASTER | dias | 1,724 | 30 | 0.6446 [0.6182; 0.6666] | 0.6125 [0.5934; 0.6326] | 0.5128 | 0.0998 [0.0714; 0.1304] | 0.6614 [0.6477; 0.6770] | 0.0335 [0.0280; 0.0600] |
| previo |  | evaluacion CHALLENGER | dias | 339 | 30 | 0.6955 [0.6368; 0.7499] | 0.6578 [0.5976; 0.7148] | 0.4985 | 0.1593 [0.0659; 0.2454] | 0.6350 [0.6056; 0.6684] | 0.0914 [0.0718; 0.1598] |
| previo |  | diamante (chequeo) | dias | 33,998 | 30 | 0.6473 [0.6414; 0.6534] | 0.6047 [0.5994; 0.6100] | 0.5014 | 0.1032 [0.0972; 0.1091] | 0.6602 [0.6567; 0.6637] | 0.0243 [0.0200; 0.0303] |
| en_vivo | 10 | evaluacion | dias | 36,983 | 30 | 0.7916 [0.7880; 0.7953] | 0.7160 [0.7123; 0.7198] | 0.5253 | 0.1907 [0.1851; 0.1963] | 0.5486 [0.5446; 0.5523] | 0.0041 [0.0043; 0.0106] |
| en_vivo | 10 | evaluacion | partidas | 36,983 |  | 0.7916 [0.7871; 0.7960] | 0.7160 [0.7113; 0.7203] | 0.5253 | 0.1907 [0.1839; 0.1977] | 0.5486 [0.5442; 0.5530] | 0.0041 [0.0044; 0.0104] |
| en_vivo | 10 | cohorte 20 | dias | 33,821 | 30 | 0.7659 [0.7615; 0.7706] | 0.6961 [0.6916; 0.7010] | 0.5264 | 0.1697 [0.1638; 0.1760] | 0.5752 [0.5706; 0.5796] | 0.0106 [0.0079; 0.0155] |
| previo_hacia_adelante | 10 | cohorte 20 | dias | 33,821 | 30 | 0.6681 [0.6637; 0.6726] | 0.6245 [0.6205; 0.6285] | 0.5264 | 0.0981 [0.0915; 0.1046] | 0.6467 [0.6443; 0.6492] | 0.0082 [0.0071; 0.0147] |
| en_vivo | 10 | diamante (chequeo) | dias | 33,790 | 30 | 0.7799 [0.7744; 0.7849] | 0.7055 [0.7001; 0.7105] | 0.5015 | 0.2041 [0.1986; 0.2093] | 0.5621 [0.5569; 0.5675] | 0.0086 [0.0063; 0.0140] |
| en_vivo | 15 | evaluacion | dias | 36,798 | 30 | 0.8438 [0.8404; 0.8470] | 0.7603 [0.7572; 0.7635] | 0.5251 | 0.2352 [0.2290; 0.2419] | 0.4848 [0.4804; 0.4893] | 0.0067 [0.0054; 0.0115] |
| en_vivo | 15 | evaluacion | partidas | 36,798 |  | 0.8438 [0.8400; 0.8478] | 0.7603 [0.7560; 0.7648] | 0.5251 | 0.2352 [0.2288; 0.2420] | 0.4848 [0.4794; 0.4898] | 0.0067 [0.0053; 0.0120] |
| en_vivo | 15 | cohorte 20 | dias | 33,821 | 30 | 0.8193 [0.8147; 0.8235] | 0.7407 [0.7369; 0.7445] | 0.5264 | 0.2143 [0.2082; 0.2211] | 0.5177 [0.5123; 0.5234] | 0.0115 [0.0088; 0.0164] |
| previo_hacia_adelante | 15 | cohorte 20 | dias | 33,821 | 30 | 0.6681 [0.6637; 0.6726] | 0.6245 [0.6205; 0.6285] | 0.5264 | 0.0981 [0.0915; 0.1046] | 0.6467 [0.6443; 0.6492] | 0.0082 [0.0071; 0.0147] |
| en_vivo | 15 | diamante (chequeo) | dias | 33,580 | 30 | 0.8337 [0.8289; 0.8379] | 0.7506 [0.7456; 0.7551] | 0.5016 | 0.2490 [0.2440; 0.2541] | 0.5001 [0.4946; 0.5065] | 0.0076 [0.0059; 0.0125] |
| en_vivo | 20 | evaluacion | dias | 33,821 | 30 | 0.8732 [0.8694; 0.8768] | 0.7870 [0.7831; 0.7912] | 0.5264 | 0.2606 [0.2534; 0.2678] | 0.4424 [0.4365; 0.4485] | 0.0065 [0.0054; 0.0121] |
| en_vivo | 20 | evaluacion | partidas | 33,821 |  | 0.8732 [0.8694; 0.8767] | 0.7870 [0.7825; 0.7913] | 0.5264 | 0.2606 [0.2534; 0.2674] | 0.4424 [0.4369; 0.4483] | 0.0065 [0.0058; 0.0119] |
| en_vivo | 20 | cohorte 20 | dias | 33,821 | 30 | 0.8732 [0.8694; 0.8768] | 0.7870 [0.7831; 0.7912] | 0.5264 | 0.2606 [0.2534; 0.2678] | 0.4424 [0.4365; 0.4485] | 0.0065 [0.0054; 0.0121] |
| previo_hacia_adelante | 20 | cohorte 20 | dias | 33,821 | 30 | 0.6681 [0.6637; 0.6726] | 0.6245 [0.6205; 0.6285] | 0.5264 | 0.0981 [0.0915; 0.1046] | 0.6467 [0.6443; 0.6492] | 0.0082 [0.0071; 0.0147] |
| en_vivo | 20 | diamante (chequeo) | dias | 31,416 | 30 | 0.8688 [0.8654; 0.8721] | 0.7808 [0.7759; 0.7850] | 0.5033 | 0.2775 [0.2712; 0.2834] | 0.4500 [0.4451; 0.4551] | 0.0075 [0.0062; 0.0129] |

## Degradación: pendiente del AUC por semana (ventanas de 2.500 partidas)

| Modelo | Minuto | Ventanas | Pendiente (AUC/semana) | IC 95 % |
| :--- | :---: | ---: | ---: | :--- |
| previo |  | 15 | +0.0000 | [-0.0046; +0.0043] |
| en_vivo | 10 | 15 | -0.0018 | [-0.0053; +0.0020] |
| en_vivo | 15 | 15 | +0.0007 | [-0.0026; +0.0039] |
| en_vivo | 20 | 14 | +0.0028 | [-0.0003; +0.0059] |

## Semillas (conjunto de evaluación)

| Modelo | Métrica | Principal | Media | Desvío | Mín | Máx | Corr. mín. de p | Mismo ganador (mín.) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| previo | auc | 0.6765 | 0.6764 | 0.00011 | 0.6763 | 0.6766 | 0.99957 | 0.9917 |
| previo | acierto | 0.6299 | 0.6298 | 0.00019 | 0.6296 | 0.6301 |  |  |
| previo | log_loss | 0.6419 | 0.6419 | 0.00006 | 0.6418 | 0.6420 |  |  |
| previo | ece | 0.0060 | 0.0062 | 0.00032 | 0.0057 | 0.0065 |  |  |
| en_vivo10 | auc | 0.7916 | 0.7916 | 0.00002 | 0.7915 | 0.7916 | 0.99986 | 0.9953 |
| en_vivo10 | acierto | 0.7160 | 0.7157 | 0.00027 | 0.7155 | 0.7160 |  |  |
| en_vivo10 | log_loss | 0.5486 | 0.5487 | 0.00003 | 0.5486 | 0.5487 |  |  |
| en_vivo10 | ece | 0.0041 | 0.0044 | 0.00026 | 0.0041 | 0.0048 |  |  |
| en_vivo15 | auc | 0.8438 | 0.8438 | 0.00004 | 0.8437 | 0.8438 | 0.99990 | 0.9962 |
| en_vivo15 | acierto | 0.7603 | 0.7599 | 0.00032 | 0.7595 | 0.7603 |  |  |
| en_vivo15 | log_loss | 0.4848 | 0.4848 | 0.00004 | 0.4847 | 0.4848 |  |  |
| en_vivo15 | ece | 0.0067 | 0.0067 | 0.00008 | 0.0066 | 0.0068 |  |  |
| en_vivo20 | auc | 0.8732 | 0.8731 | 0.00003 | 0.8731 | 0.8732 | 0.99994 | 0.9971 |
| en_vivo20 | acierto | 0.7870 | 0.7873 | 0.00020 | 0.7870 | 0.7875 |  |  |
| en_vivo20 | log_loss | 0.4424 | 0.4424 | 0.00004 | 0.4424 | 0.4425 |  |  |
| en_vivo20 | ece | 0.0065 | 0.0066 | 0.00011 | 0.0065 | 0.0067 |  |  |

Tablas completas en `data/analysis/test_final` (`metricas.csv`, `ventanas.csv`, `pendientes.csv`, `semillas.csv`, `ic_bootstrap_semillas.csv`, `resumen.json`).
