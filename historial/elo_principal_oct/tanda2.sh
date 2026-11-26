E1=z2,pred_regla,partidas2,wr_temporada,obs,obs_w,kal_media,kal_sd,decay,piso,dias_fin,dias_ini,hueco_max,g_ch,g_gm,g_ma,lobby_viejo,lobby_reciente
run() { PYTHONIOENCODING=utf-8 python -u gru_exp.py "$@" 2>&1 | tail -1; }
run --tag est_plus_s1 --est $E1 --semillas 1 --nota "est_plus, otra corrida (ruido)" 
run --tag est_plus_h96 --est $E1 --oculta 96 --dropout 0.2 --wd 1e-4 --nota "estado 96, dropout .2, wd 1e-4"
run --tag est_plus_e60 --est $E1 --epocas 60 --dropout 0.1 --nota "60 épocas, dropout .1"
run --tag est_plus_2c --est $E1 --capas 2 --dropout 0.2 --nota "2 capas, dropout .2"
run --tag est_plus_lr1 --est $E1 --lr 1e-3 --epocas 60 --nota "lr 1e-3, 60 épocas"
run --tag est_plus_s5 --est $E1 --semillas 5 --nota "est_plus, promedio de 5 semillas"
