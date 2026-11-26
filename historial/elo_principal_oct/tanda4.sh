E1=z2,pred_regla,partidas2,wr_temporada,obs,obs_w,kal_media,kal_sd,decay,piso,dias_fin,dias_ini,hueco_max,g_ch,g_gm,g_ma,lobby_viejo,lobby_reciente
run() { PYTHONIOENCODING=utf-8 python -u gru_exp.py "$@" 2>&1 | tail -1; }
run --tag d_delta --est $E1 --delta --nota "est_plus + cabeza de deltas (suma por partida)"
run --tag d_aux03 --est $E1 --aux 0.3 --nota "est_plus + auxiliar: victorias/derrotas ocultas (λ .3)"
run --tag d_mdn2 --est $E1 --mdn 2 --nota "est_plus + mezcla de 2 Laplace, mediana"
run --tag d_bidelta --est $E1 --arq bigru --delta --nota "biGRU + cabeza de deltas"
run --tag d_aux1 --est $E1 --aux 1.0 --nota "est_plus + auxiliar ocultas (λ 1)"
run --tag d_mdn3 --est $E1 --mdn 3 --nota "est_plus + mezcla de 3 Laplace, mediana"
run --tag d_delta_aux --est $E1 --delta --aux 0.3 --nota "deltas + auxiliar"
