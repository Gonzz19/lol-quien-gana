# Modelo principal del elo al revés: scripts del 28 y 29/10

Recuperados el 2026-11-02 del temporal de la PC de escritorio, donde se daban por perdidos. Son los
experimentos tal como se corrieron, sin tests. El nombre del modelo y su receta están en `bitacora.md`
("Receta del principal"), y cada corrida, en `registro.tsv`. `exportar_principal.py` escribió
`data/analysis/elo_principal_pred.csv`.

**No correrlos desde esta carpeta.** Escriben sus cachés al lado del script (`cache.pkl` pesa 167 MB, y
además `seq.pkl`, `grid_*.pkl` y `preds/`). Hay que copiarlos a una carpeta fuera del repo y arrancar por
`prep.py`, `prep_seq.py` y `grilla.py --tag grid_l2.0 --tau 2`. La ruta del repo está fija en `prep.py`.
`tabpfn_exp.py` espera los pesos públicos en la caché de Hugging Face de la PC de escritorio.

Las cachés originales (344 MB) siguen en el temporal, en la sesión `71199607`, y se pueden regenerar.

`previos/` tiene las pruebas del 28/10 para el notebook.
