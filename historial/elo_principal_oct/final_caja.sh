# Producción de las predicciones de la caja para los expertos con grilla τ 2 (todo el desarrollo → caja).
# No imprime métricas de la caja: eso lo hace una sola vez compuerta.py --caja con el candidato final.
for m in cat_bag5_rsm cat_bag5_h100 cat_rmse cat_huber100 xgb_l1 lgb_l1 lgb_q25 lgb_q75 cat_bag5_rsm+an_gru cat_bag5_h100+an_gru; do
  PYTHONIOENCODING=utf-8 python gb_grid.py grid_l2.0 $m --caja 2>&1 | grep -v Warn
done
