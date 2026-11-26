E1=z2,pred_regla,partidas2,wr_temporada,obs,obs_w,kal_media,kal_sd,decay,piso,dias_fin,dias_ini,hueco_max,g_ch,g_gm,g_ma,lobby_viejo,lobby_reciente
run() { PYTHONIOENCODING=utf-8 python -u gru_exp.py "$@" 2>&1 | grep -v "^par" | tail -1; }
run --tag g_cat --est $E1 --extra anid_cat_huber.npy --nota "est_plus + predicción anidada de CatBoost Huber como estática"
run --tag g_aug10 --est $E1 --aug 0.1 --nota "est_plus + aumento: borra 10 % de las partidas observadas en la mitad de los lotes"
run --tag gru --est $E1 --anidado
PYTHONIOENCODING=utf-8 python -u zoo.py cat_huber+an_gru 2>&1 | grep -v Warn | tail -1
run --tag g_aug25 --est $E1 --aug 0.25 --nota "est_plus + aumento 25 %"
run --tag g_grid05 --est $E1 --grid grid_l0.5 --nota "est_plus + variables del filtro de grilla τ .5 como estáticas"
run --tag g_cat_grid --est $E1 --extra anid_cat_huber.npy --grid grid_l0.5 --nota "est_plus + CatBoost anidado + grilla τ .5 como estáticas"
run --tag g_huber --est $E1 --perdida huber --nota "est_plus con pérdida Huber (δ = 100 LP)"
run --tag gru --est $E1 --caja --nota "GRU est_plus 1 semilla, para la variable anidada en la caja"
run --tag g_cg2_s5 --est $E1 --extra anid_cat_huber.npy --grid grid_l2.0 --semillas 5 --nota "est_plus + CatBoost anidado + grilla τ 2, 5 semillas"
run --tag g_cg2_s5 --est $E1 --caja --extra-caja cat_huber --grid grid_l2.0 --semillas 5
