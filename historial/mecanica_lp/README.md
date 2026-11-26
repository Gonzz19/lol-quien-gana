# Mecánica del LP: pruebas exploratorias (2026-11-02)

Respaldo de `PROJECT_SUMMARY.md` §9.S.40. Son pruebas, sin tests, y nada de `src/` las importa. Lo que
sirva pasa a `src/features/lp_mecanica.py`.

Todas leen `lol_las.db` y `lp_vivo.db` en modo solo lectura. Usan una carpeta de trabajo, fuera del repo:
la copia de la base va en `<trabajo>/vivo/lp_vivo.db`, y ahí mismo se guardan los resultados intermedios.

```bash
python historial/mecanica_lp/v1_panorama.py <trabajo>          # estado de la base en vivo
python historial/mecanica_lp/c1_cobertura.py .                 # cuántas partidas faltan (antes de la foto 1 solo cuenta filas)
```

Las demás reciben `<trabajo> .` (`e2` y `e12`, solo `<trabajo>`), y este orden importa porque cada una usa lo
que guardó la anterior:

| Script | Qué mide |
| :--- | :--- |
| `v2_estructura`, `v3_cola` | Forma del ΔLP; si los saltos son reales o réplicas viejas |
| `v4_mecanicas` | Dobles por rol y devoluciones por equipo |
| `v5_delta` | δ por jugador, δ contra el gap, rareza del rol con la historia del dataset |
| `e0_datos` | Train del elo (foto 1 → foto 2) con el rol de cada partida |
| `e1_adelante`, `e2_ablacion` | Recorrida hacia adelante, ablación y puente en vivo |
| `e3_atras` | Suavizado en grilla hacia atrás (`e3_atras.py <trabajo> . 6000`) |
| `e4_temporada`, `e5_lolpros` | Cierre de temporada y reinicios en lolpros |
| `e7_reinicio_banco`, `e8_regimen` | Reinicio del 29/04, banco de días y Aegis al perder |
| `e9_principal` | Adelante con banco, inversión y mecánica como variable del principal |
| `e10_cobertura`, `e11_adelante_ocultas` | Error según la cobertura, en la punta y en el camino |
| `e12_adelante_gb` | Mecánica + GB hacia adelante |

`mec.py` tiene los parámetros que salen del en vivo, y `mapa.py`, la recorrida vectorial.

La caja cerrada queda afuera (`crc32("caja-" + id) % 100 < 15`). `e3_atras`, `e9_principal` y
`e12_adelante_gb` leen `elo_principal_pred.csv`.
