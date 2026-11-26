# Espera a que termine tanda4 (GPU) y sigue con la cola de la GPU.
OUT="C:/Users/DarkA/AppData/Local/Temp/claude/c--Users-DarkA-OneDrive-Escritorio-Facultad-3er-a-o-ML-TPF/71199607-81f8-4196-9578-7c6e3b376925/tasks/bbvoh8w1y.output"
until grep -q "^d_delta_aux" "$OUT" 2>/dev/null; do sleep 20; done
E1=z2,pred_regla,partidas2,wr_temporada,obs,obs_w,kal_media,kal_sd,decay,piso,dias_fin,dias_ini,hueco_max,g_ch,g_gm,g_ma,lobby_viejo,lobby_reciente
run() { PYTHONIOENCODING=utf-8 python -u gru_exp.py "$@" 2>&1 | tail -1; }
run --tag g_cat --est $E1 --extra anid_cat_huber.npy --nota "est_plus + predicción anidada de CatBoost Huber como estática"
run --tag g_aug10 --est $E1 --aug 0.1 --nota "est_plus + aumento: borra 10 % de las partidas observadas en la mitad de los lotes"
run --tag g_aug25 --est $E1 --aug 0.25 --nota "est_plus + aumento 25 %"
PYTHONIOENCODING=utf-8 python -u tabpfn_exp.py --max-train 8000 < /dev/null 2>&1 | grep -v -i warn | tail -3
run --tag gru --est $E1 --anidado --nota "GRU est_plus anidada (1 semilla) para stacking"
cd "$(dirname "$0")"; PYTHONIOENCODING=utf-8 python -u zoo.py cat_huber+an_gru cat_bag5+an_gru 2>&1 | grep -v Warn | tail -2
