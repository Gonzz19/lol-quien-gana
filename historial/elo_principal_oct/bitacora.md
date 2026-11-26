# Bitácora — segunda tanda del objetivo de métricas (noche del 29/10/2026)

Protocolo: igual que la primera tanda. Desarrollo = 85 % (5 folds `folds_por_jugador`), caja cerrada = 15 %
(crc32("caja-"+id) % 100 < 15), ya abierta una vez con la primera tanda. Métricas: MAE, RMSE, R² del cambio,
ρ de Spearman ponderado por tier. Todo queda en `registro.tsv` (una fila por corrida) y `preds/<tag>.npy`.

Referencias de desarrollo (primera tanda): GRU est_plus 5 semillas 73,9 / 122,5; GB final L1 75,8 / 121,7;
mezcla fija 0,65/0,35 73,0 / 119,8.

## Oráculo (usa la foto 1, solo para medir techos)
* GB + ocultas totales exactas: 72,8 / 119,2. GB + victorias y derrotas ocultas exactas: 63,7 / 114,2
  (Diamond 52, Master 84). Aun sabiendo el resultado de cada partida oculta, el RMSE casi no baja: lo domina
  el decay censurado y los grandes movimientos.
* Las ocultas de Diamond ganan 49,5 % y su winrate NO se correlaciona con el observado (r = 0,04); las
  observadas ganan 46,6 % (sesgo del crawler). Hay que usar un winrate fijo para las ocultas.

## Decay (decay_curva.py)
* Master+: 75 LP/día después de 14 días, hasta 375 (D1 75).
* Diamond I (y D2 76-99): 50 LP/día después de ~28 días, hasta 275 (D2 75). Un ex-Master sigue a 275 a los
  ~32-35 días. De D2 75 para abajo casi no hay decay.
* El `decay` del pipeline (75/día para todos, sin pisos) está mal calibrado, pero el GB ya lo compensa con
  días sin jugar y pisos: el decay calibrado (decay_cal.py) no sumó (gb_dc_* 75,9-76,9).

## Lo que probé (dev, 5 folds)
| Idea | Script | Resultado |
| --- | --- | --- |
| MiniRocket (6073 var, ridge LOO) | minirocket.py | solo 88,1/132,6; como variable del GB 75,4/121,5 (−0,4) |
| LightGBM / XGBoost | zoo.py | lgb_l1 75,5/120,9; xgb_l1 75,4/120,6 |
| **CatBoost** | zoo.py | cat_huber 74,9/118,1; cat_bag5 74,7/118,0; el mejor boosting |
| Cuantiles (LGB q25/q75) | zoo.py | como contexto de la compuerta |
| Filtro de grilla (HMM 1D, decay con pisos, prior foto 2) | grilla.py | solo 119,2 (peor que la regla); como variables de CatBoost 74,9/117,7 (≈0) |
| GRU cabeza de deltas | gru_exp.py --delta | 75,8/122,9 (peor) |
| GRU multitarea ocultas | --aux 0.3 | 75,1/124,0 (≈0, peor) |
| GRU mezcla de 2 Laplace (mediana) | --mdn 2 | 74,6/126,5 (MAE ≈, RMSE peor) |
| Compuerta (pesos por contexto) con 5 expertos | compuerta.py | 72,8/118,9 |
| **Compuerta con 11 expertos + cuantiles** | compuerta.py --nombre comp_11 | **72,3/117,5/0,724/0,878** |
| Conformal Mondrian por tier | conformal.py | cobertura exacta; ancho 80 %: D 187, M 242, GM 362 LP |
| TabPFN v2 | tabpfn_exp.py | (la versión nueva pide login en el navegador; v2 pública anda) |

La compuerta aprende lo esperable: con 1-5 partidas observadas confía en los boostings, con 60+ en la GRU.

## Descartado sin correr (y por qué)
* STL: necesita serie regular con estacionalidad; no hay LP por partida ni estacionalidad por jugador.
* Filtro de partículas: en 1D la grilla es exacta; con la grilla sin ganancia, no hay nada que ganar.
* Dos estados LP + MMR / bayesiano jerárquico: la grilla 1D ya no superó a la regla; el MMR entra por el lobby.
* Propagación de etiquetas: usaría z1 de vecinos, que no existe al aplicar el modelo (fuga respecto del uso).
* GNN / EM sobre lobbies: requiere recomputar lobbies con predicciones fuera de fold de todos los vecinos;
  la ganancia del lobby con LP predicho ya fue chica. Ver si queda tiempo.

## Segunda parte (04:00)
* Lobby: el error tiene un offset fijo por jugador (MMR − LP): sd entre jugadores 142 (D) / 143 (M) / 88 (GM+),
  dentro del jugador 109 / 156 / 257 por partida. Por eso hay que templar la verosimilitud (τ) en la grilla 1D.
* **Grilla como variables, con lobby**: τ 0,1 → cat_huber 74,7/117,7; **τ 0,5 → 74,4/117,1**; τ 1 → 74,3/117,4.
  Con bolsa: cat_bag5_rsm+grid_l0.5 **73,9/116,5/0,729/0,875**; cat_bag5_h100+grid_l0.5 74,2/**115,5**/0,733.
  La grilla mejora a todos los boostings (~−0,5 MAE, −1 a −2 RMSE). Como experto solo sigue peor que la regla.
* EM sobre lobbies (1 vuelta, con CatBoost anidado): ≈0 (74,7/118,2). Con la predicción anidada como variable,
  74,3/119,0 (es el autostacking, no los lobbies). Autostacking solo: 74,7/119,1.
* Especialistas por subgrupo (piso+inactivo, inactivo, ≤5 partidas): peores que el global en los tres.
* Regla lineal aprendida (L1): 95,2/146,4.
* GRU + CatBoost anidado como estática (g_cat): 74,4/120,4 (vs 74,8/123,9 de la misma GRU sin eso).
* GRU con aumento (borrar 10 % de partidas observadas): 76,7/125,2 (peor).
* GRU mezcla 3 Laplace: 75,4/128,2; auxiliar λ 1: 75,5/124,0.
* Compuertas: comp_12 71,9/116,2; comp_12h (Huber) 72,0/115,6/0,733/0,880; comp_13 (boostings con grilla)
  71,9/115,9; con contexto de la grilla (p10/p90/sd) igual. La compuerta está saturada en ~71,9/115,6.

## Tercera parte (04:40)
* **Stacking al revés: CatBoost + predicción anidada de la GRU** (anid_gru, 15 ajustes de GRU, pares de folds):
  cat_huber 74,9 → **73,2**/118,1. Con grilla τ .5: 73,0/117,6; bolsa rsm: **72,7/117,1/0,726/0,878**;
  bolsa h100: 72,9/116,3/0,730.
* Grilla de dos estados (LP × offset fijo del lobby, 11 valores; grilla2d.py): sola 130,1; como variables
  74,8/117,8 → peor que la 1D. El offset no es fijo (el LP converge al MMR).
* Grilla 1D, más peso al lobby: τ 2 → cat_huber+grid_l2.0 **74,0/116,9/0,727** (mejor que τ .5). Probando τ 4.
* Aumento de la GRU 25 %: 77,4/125,7 (peor, como el 10 %).
* comp_15s (con los CatBoost apilados): **71,7/116,2/0,730/0,879**; con Huber 72,0/115,7/0,732/0,880.

## Cuarta parte (05:15)
* Grilla τ 2 como variables: cat_bag5_rsm+grid_l2.0 **73,5/116,5**; cat_bag5_h100+grid_l2.0 73,9/**115,6**/0,733;
  cat_huber100+grid_l2.0 74,1/115,8; xgb 74,3/119,8; lgb 74,5/120,2. Con GRU anidada: rsm 72,6/117,4; h100 73,0/116,7.
* GRU: + grilla τ .5 como estáticas 75,4/123,7 (peor); + CatBoost anidado + grilla 74,1/119,1 (mejor GRU de 1 semilla);
  Huber 75,2/122,5.
* GB con objetivo agregado (LightGBM, gradiente propio sobre la suma o el promedio por jugador): diverge.
* comp_t2 (τ 2): 71,8/116,8/0,727/0,880; Huber 72,0/116,2/0,730/0,880.
* **Promedio simple (sin pesos) de 4**: est_plus_s5 + g_cat + cat_bag5_rsm+an_gru+grid_l2.0 + cat_bag5_h100+grid_l2.0
  → **71,9/116,1/0,731/0,880**. Igual que la compuerta: la ganancia está en los expertos, no en el combinador.

## Candidatos para la caja (fijados antes de abrirla)
* Principal: promedio simple de 4 (arriba). Si g_cg2_s5 (GRU 5 semillas + CatBoost anidado + grilla τ 2) mejora a
  g_cat en desarrollo, la reemplaza (decisión solo con desarrollo).
* Alternativa: compuerta L1 sobre los 14 expertos de comp_t2 (+ cuantiles como contexto).

## Quinta parte (05:35) — antes de abrir la caja
* **TabPFN v2** (pesos públicos, 8000 filas de contexto, 4 estimadores, sin ajustar nada): mediana **73,8/121,6/0,704/0,872**,
  media 75,5/119,9. Familia distinta → suma diversidad.
* Promedio simple de 5 (los 4 de antes + tabpfn_mediana): **71,6/115,7/0,732/0,880** (la mezcla anidada le da 0,30 a TabPFN).
* Candidato principal revisado (solo con desarrollo, la caja sigue cerrada): promedio simple de 5. Si TabPFN con grilla τ 2
  (tabpfn_g2_mediana) o la GRU g_cg2_s5 mejoran a sus versiones en desarrollo, las reemplazan.

## Candidatos FINALES (fijados a las 06:03, antes de abrir la caja en esta tanda)
* PRINCIPAL: promedio simple de est_plus_s5, g_cg2_s5, cat_bag5_rsm+an_gru+grid_l2.0, cat_bag5_h100+grid_l2.0,
  tabpfn_g2_mediana. Desarrollo: 71,5/115,6/0,733/0,880.
* ALTERNATIVA: compuerta comp_t3 (17 expertos + cuantiles de contexto). Desarrollo: 71,4/116,5/0,728/0,880.
* Referencias: regla al revés; principal de la primera tanda (0,65 GRU + 0,35 GB: caja 71,0/112,0/0,738/0,878).

## RESULTADO EN LA CAJA (segunda apertura, 06:04)
| Modelo | MAE | RMSE | R² | ρ | Diamond | Master | GM | CH |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regla al revés | 111,8 | 174,6 | 0,365 | 0,733 | 96,6 | 145,3 | 143,8 | 140,0 |
| principal 1.ª tanda (0,65 GRU + 0,35 GB) | 71,0 | 112,0 | 0,738 | 0,878 | 62,9 | 87,7 | 97,7 | 110,0 |
| **PRINCIPAL: promedio de 5** | **69,5** | **107,6** | **0,759** | 0,882 | 61,6 | 85,4 | 97,3 | 113,3 |
| ALTERNATIVA: compuerta comp_t3 | 69,2 | 108,9 | 0,753 | 0,883 | 61,3 | 84,8 | 101,8 | 117,6 |

Mejora en la caja = mejora en desarrollo (−1,5 MAE en ambos): la selección no metió optimismo.
Conformal Mondrian del principal calibrado en desarrollo, cobertura en la caja: 80 % → D 80,9 / M 79,0 / GM 86,8 /
CH 88,6; 90 % → 90,5 / 89,8 / 94,1 / 94,3. Ancho total al 80 %: D 186, M 239, GM 330, CH 357 LP.

### Receta del principal (para llevar al pipeline)
1. GRU est_plus (E1, 8 variables por partida P0, L1, 40 épocas, oculta 48), promedio de 5 semillas.
2. GRU est_plus + CatBoost Huber anidado como estática + 6 variables de la grilla τ 2, 5 semillas (g_cg2_s5).
3. CatBoost Huber δ 60, 5 semillas con rsm .7 y Bernoulli .8, sobre gbF + grilla τ 2 + GRU anidada (1 semilla).
4. CatBoost Huber δ 100, 5 semillas, sobre gbF + grilla τ 2.
5. TabPFN v2 (pesos públicos Prior-Labs/TabPFN-v2-reg), 8000 filas de contexto, 4 estimadores, mediana, gbF + grilla τ 2.
Promedio simple de los 5. La grilla: grilla.py --tau 2 (decay calibrado con pisos, ocultas con winrate fijo del grupo,
mezcla π 0,3/0,7 de partidas ocultas en los huecos, lobby t de Student templado τ, prior = histograma de la foto 2).

## Después de la caja (solo desarrollo, para el registro)
* TabPFN 12000 filas × 8 estimadores + grilla: mediana 72,5/120,4 (mejor individual en MAE); en el promedio de 5
  71,4/115,6 vs 71,5/115,6 → no vale el triple de cómputo (90 min).
* Grilla τ 4: cat_huber 73,6/117,2 (τ 2: 74,0/116,9). Meseta.
* ExtraTrees 82,9 / RandomForest 79,4 con grilla: débiles.
