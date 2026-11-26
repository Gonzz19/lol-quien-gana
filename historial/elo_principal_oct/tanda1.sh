P0=resultado,aliados,rivales,lobby,conocidos,dias,hueco,azul
E0=z2,pred_regla,partidas2,wr_temporada,obs,obs_w,kal_media,kal_sd,decay
EP=piso,dias_fin,dias_ini,hueco_max,g_ch,g_gm,g_ma,lobby_viejo,lobby_reciente
PP=gap,lobby2_rel,lp_paso,racha
PF=kda,kp,dmg,oro,muertes,duracion
run() { PYTHONIOENCODING=utf-8 python -u gru_exp.py "$@" 2>&1 | tail -1; }
run --tag est_plus --est $E0,$EP --nota "estáticas + piso, días sin jugar, rango, lobby viejo/reciente"
run --tag paso_plus --paso $P0,$PP --nota "por partida + gap, lobby foto2, lp del paso, racha"
run --tag perf --paso $P0,$PF --nota "por partida + rendimiento (kda, kp, daño, oro, muertes, duración)"
run --tag todo --paso $P0,$PP,$PF --est $E0,$EP --nota "todas las variables"
run --tag todo_kalbase --paso $P0,$PP,$PF --est $E0,$EP --base kalman --nota "todas, base Kalman"
run --tag todo_lstm --arq lstm --paso $P0,$PP,$PF --est $E0,$EP --nota "todas, LSTM"
run --tag todo_bigru --arq bigru --paso $P0,$PP,$PF --est $E0,$EP --nota "todas, GRU bidireccional"
run --tag todo_attn --arq attn --paso $P0,$PP,$PF --est $E0,$EP --nota "todas, GRU + atención"
run --tag todo_tf --arq transformer --capas 2 --oculta 64 --dropout 0.1 --paso $P0,$PP,$PF --est $E0,$EP --nota "todas, Transformer 2 capas"
run --tag todo_ds --arq deepsets --oculta 64 --paso $P0,$PP,$PF --est $E0,$EP --nota "todas, DeepSets"
