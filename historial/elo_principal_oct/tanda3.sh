E1=z2,pred_regla,partidas2,wr_temporada,obs,obs_w,kal_media,kal_sd,decay,piso,dias_fin,dias_ini,hueco_max,g_ch,g_gm,g_ma,lobby_viejo,lobby_reciente
E2=$E1,exc_fin,exc_ini,exc_medio,n_huecos14,antes_fin,piso375,piso275
run() { PYTHONIOENCODING=utf-8 python -u gru_exp.py "$@" 2>&1 | tail -1; }
run --tag est2 --est $E2 --nota "est_plus + decay fino, 1 semilla"
run --tag est2_s5 --est $E2 --semillas 5 --nota "est_plus + decay fino, 5 semillas"
run --tag est2_l2_s5 --est $E2 --semillas 5 --perdida l2 --nota "decay fino, pérdida cuadrática, 5 semillas"
