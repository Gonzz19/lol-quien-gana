"""
Baselines del proyecto (PROJECT_SUMMARY.md §9.S.20).

Escalón 2, predicción con el draft (ganador de la partida y de cada línea @15):
    draft_rules   Regla sin entrenamiento: media de WR campeón-línea / oro medio @15.
    draft_logreg  Regresión logística sobre los campeones (duelo o draft completo).

Escalón 1, identificación del jugador (prueba de `src/features/fingerprint.py`):
    identificacion  Perfiles `pool`, `elo_linea` y `embedding` (solo campeones).
"""
