"""
Candidato final del brainstorming nocturno (PROJECT_SUMMARY.md §9.S.52, docs/BRAINSTORM_NOCHE.md): escalón 9 sin el
mapa de LP. Reproduce el modelo de punta a punta y reporta AUC, acierto, log loss, calibración y cobertura en V Master+.

Modelo: logística (C = 0,05) entrenada solo con partidas de Master+ anteriores al 16/09, sobre
  draft campeón × línea con signo + 85 variables previas por jugador (azul − rojo por línea)
  + margen por (jugador, campeón) y por (jugador, línea)            [brainstorm_05_jugador.py]
  + nivel histórico por emparejamiento (LP de la foto 1)            [brainstorm_07_mmr_historico.py]
  + W/L de temporada de la foto 1, actualizado causalmente           [brainstorm_g_01_foto1_wl.py, se calcula al vuelo]
  + contracruces de campeones y sinergias causales (MATCHUPS)       [brainstorm_g_07_matchups_parche_rematch.py]
  + winrate por línea / campeón-línea y autofill (ROL_WR)            [brainstorm_g_10_nuevas_senales.py]
  + Kalman-Elo causal, tempo/objetivos y economía 15→20 (g_11)      [brainstorm_g_11_elo_clutch_tempo.py]
  + modelo por línea: oro @15 predicho con HGB causal (5 líneas + suma) [brainstorm_g_09_hgb_lineas_causal.py]
  + margen esperado @10/@15 por par de campeones en cada línea          [brainstorm_g_23_duelo_res_y_per_jug.py]

Cada caché la genera el script indicado (todos causales, solo `data/splits/train`). Antes hay que correr
`embeddings_partidas.py` y `ganador_mapa_lp.py` (escribe `mapa_lp_causal_v.pkl`; sin él `brainstorm_base.cargar` falla).
Orden completo en PROJECT_SUMMARY.md §9.S.69.

    python historial/escalon9/brainstorm_final.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
B = REPO / "data" / "analysis" / "brainstorm"
NECESARIAS = {
    "margen_grupos.npz": "brainstorm_05_jugador.py",
    "mmr_historico.npz": "brainstorm_07_mmr_historico.py",
    "gemini_mpr.npz": "brainstorm_g_07_matchups_parche_rematch.py",
    "gemini_nuevas_v2.npz": "brainstorm_g_10_nuevas_senales.py",
    "gemini_g11.npz": "brainstorm_g_11_elo_clutch_tempo.py",
    "gemini_hgb_lineas.npz": "brainstorm_g_09_hgb_lineas_causal.py",
    "gemini_g23.npz": "brainstorm_g_23_duelo_res_y_per_jug.py",
}
faltan = {k: v for k, v in NECESARIAS.items() if not (B / k).exists()}
if faltan:
    sys.exit("Faltan cachés; correr antes:\n" + "\n".join(f"  python historial/escalon9/{v}   (→ {k})" for k, v in faltan.items()))

from scripts.analysis.brainstorm_base import ajustar_predecir, cargar, extras_referencia  # noqa: E402

D = cargar()
X = extras_referencia(D, escalon=9)
tr = D["fi_v"][D["master"][D["fi_v"]]]
vi = np.flatnonzero(D["val"])
p, lr = ajustar_predecir(D, tr, vi, extra=X, extra_post=False, con_mapa=False, devolver_modelo=True)
y = D["y"][vi]
ok = (p > 0.5) == y
dec = pd.qcut(p, 10, labels=False)
g = pd.DataFrame({"p": p, "y": y, "d": dec}).groupby("d").agg(p=("p", "mean"), r=("y", "mean"), n=("y", "size"))
ece = (np.abs(g.p - g.r) * g.n).sum() / g.n.sum()
print(f"Entrenamiento: {len(tr):,} partidas de Master+ anteriores al 16/09; columnas: {lr.coef_.shape[1]:,}")
print(f"V Master+ ({len(vi):,}): AUC {roc_auc_score(y, p):.4f} | acierto {ok.mean():.1%} | log loss {log_loss(y, p):.4f} | ECE {ece:.4f}")
o = np.argsort(-np.abs(p - 0.5))
print("Cobertura (acierto en la fracción más segura): " +
      "  ".join(f"{c:.0%} → {ok[o[:int(c * len(vi))]].mean():.1%}" for c in (0.05, 0.10, 0.20, 0.30, 0.50)))
np.savez_compressed(B / "pred_final.npz", p_v=p, idx_v=vi)
print(f"[+] {B / 'pred_final.npz'}")
