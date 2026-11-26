"""Idea G36 (propuesta por Claude en bitácora 04:47 sobre DUR_EARLY_LATE de g_11):
Probar el **choque de tempo entre equipos y por línea** con arrays estrictamente causales `< t`:
- Para cada `(campeón, línea)` y para cada `jugador`: sesgo temprano (`wr <26m`, `dec/bayesian`),
  sesgo tardío (`wr >=29m`) y balance `bias = T - E` (`E_pos = max(0, -bias)`, `T_pos = max(0, +bias)`).
- Variables antisimétricas de choque:
  1. `CH_TEAM`: `E_azul · T_rojo - E_rojo · T_azul` (tanto en versión campeón-línea como jugador,
     usando sesiones `wr_e/wr_l` y partes positivas `E_pos/T_pos`).
  2. `CH_LANE`: suma por los 5 carriles directos `E_{azul,l} · T_{rojo,l} - E_{rojo,l} · T_{azul,l}`
     más choque Jungla vs carriles enemigos y Botlane (ADC+SUP) vs rival.
  3. `COHERENCIA`: dispersión interna del tempo de la composición (`std(cbias_B) - std(cbias_R)`),
     desacople jungla-early vs líneas-late dentro del equipo, y cruce `pbias_jugador * cbias_rival`.

Evaluado sobre la referencia vigente **Escalón 6 (sin mapa, 0,6709)** + chequeo ciego en **Diamond-V**.
Caché guardada en `data/analysis/brainstorm/gemini_g18_tempo.npz`.
"""
import math
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.brainstorm_base import ajustar_predecir, cargar, comparar, evaluar, extras_referencia, mostrar  # noqa: E402

CACHE = REPO / "data" / "analysis" / "brainstorm" / "gemini_g18_tempo.npz"


def construir_cache_tempo(D: dict, rehacer: bool = False) -> dict[str, np.ndarray]:
    if CACHE.exists() and not rehacer:
        z = np.load(CACHE)
        if int(z["nm"][0]) == D["NM"]:
            return {k: z[k] for k in z.files if k != "nm"}

    t0 = time.time()
    print("Construyendo caché gemini_g18_tempo.npz...", flush=True)
    NM = D["NM"]
    Z = D["Z"]
    mids = Z["mids"].astype(str)
    t_arr = D["t"].astype(np.int64)
    y_arr = D["y"].astype(np.int8)
    dur_min = Z["D_dur_min"].astype(np.float32)
    champ = Z["champ"].reshape(NM, 10).astype(np.int32)

    players = D["B"]["player"].reshape(NM, 10).astype(str)
    _, p_inv = np.unique(players.reshape(-1), return_inverse=True)
    pid_mat = p_inv.reshape(NM, 10).tolist()
    n_p = int(p_inv.max()) + 1

    cl_pairs = champ * 5 + np.tile(np.arange(5), 2)[None, :]
    _, cl_inv = np.unique(cl_pairs.reshape(-1), return_inverse=True)
    cl_mat = cl_inv.reshape(NM, 10).tolist()
    n_cl = int(cl_inv.max()) + 1

    # Arrays por jugador (NM, 10) para reutilización abierta y trazabilidad
    p_early_wr = np.zeros((NM, 10), dtype=np.float32)
    p_late_wr = np.zeros((NM, 10), dtype=np.float32)
    p_bias = np.zeros((NM, 10), dtype=np.float32)
    c_early_wr = np.zeros((NM, 10), dtype=np.float32)
    c_late_wr = np.zeros((NM, 10), dtype=np.float32)
    c_bias = np.zeros((NM, 10), dtype=np.float32)

    # Bloques candidatos de choque de tempo
    CH_CAMP_TEAM = np.zeros((NM, 6), dtype=np.float32)
    CH_JUG_TEAM = np.zeros((NM, 6), dtype=np.float32)
    CH_LANE = np.zeros((NM, 8), dtype=np.float32)
    COHERENCIA = np.zeros((NM, 6), dtype=np.float32)

    wr_e_sum = [0.0] * n_p
    wr_e_n = [0.0] * n_p
    wr_l_sum = [0.0] * n_p
    wr_l_n = [0.0] * n_p

    cl_e_win = [0.0] * n_cl
    cl_e_n = [0.0] * n_cl
    cl_l_win = [0.0] * n_cl
    cl_l_n = [0.0] * n_cl
    dec30 = math.exp(-math.log(2.0) / 30.0)

    order = np.lexsort((mids, t_arr))
    for m in order:
        ps = pid_mat[m]
        cls = cl_mat[m]
        yb = float(y_arr[m])
        dm = float(dur_min[m])

        pe = [((wr_e_sum[ps[i]] + 6.0 * 0.5) / (wr_e_n[ps[i]] + 12.0)) - 0.5 for i in range(10)]
        pl = [((wr_l_sum[ps[i]] + 6.0 * 0.5) / (wr_l_n[ps[i]] + 12.0)) - 0.5 for i in range(10)]
        pb = [pl[i] - pe[i] for i in range(10)]

        ce = [((cl_e_win[cls[i]] + 10.0 * 0.5) / (cl_e_n[cls[i]] + 20.0)) - 0.5 for i in range(10)]
        cl = [((cl_l_win[cls[i]] + 10.0 * 0.5) / (cl_l_n[cls[i]] + 20.0)) - 0.5 for i in range(10)]
        cb = [cl[i] - ce[i] for i in range(10)]

        p_early_wr[m] = pe
        p_late_wr[m] = pl
        p_bias[m] = pb
        c_early_wr[m] = ce
        c_late_wr[m] = cl
        c_bias[m] = cb

        # Partes positivas de sesgo early (E_pos > 0 si prefiere <26m) y late (T_pos > 0 si prefiere >=29m)
        ec_b = sum(max(0.0, -cb[i]) for i in range(5)) * 0.2
        tc_b = sum(max(0.0, cb[i]) for i in range(5)) * 0.2
        ec_r = sum(max(0.0, -cb[5 + i]) for i in range(5)) * 0.2
        tc_r = sum(max(0.0, cb[5 + i]) for i in range(5)) * 0.2

        ep_b = sum(max(0.0, -pb[i]) for i in range(5)) * 0.2
        tp_b = sum(max(0.0, pb[i]) for i in range(5)) * 0.2
        ep_r = sum(max(0.0, -pb[5 + i]) for i in range(5)) * 0.2
        tp_r = sum(max(0.0, pb[5 + i]) for i in range(5)) * 0.2

        # Medias firmadas de wr_e y wr_l
        ce_b_m, ct_b_m = sum(ce[:5]) * 0.2, sum(cl[:5]) * 0.2
        ce_r_m, ct_r_m = sum(ce[5:]) * 0.2, sum(cl[5:]) * 0.2
        pe_b_m, pt_b_m = sum(pe[:5]) * 0.2, sum(pl[:5]) * 0.2
        pe_r_m, pt_r_m = sum(pe[5:]) * 0.2, sum(pl[5:]) * 0.2

        # 1. CH_CAMP_TEAM: E_azul * T_rojo - E_rojo * T_azul + interacciones cruzadas
        CH_CAMP_TEAM[m, 0] = (ec_b * tc_r) - (ec_r * tc_b)
        CH_CAMP_TEAM[m, 1] = (ce_b_m * ct_r_m) - (ce_r_m * ct_b_m)
        CH_CAMP_TEAM[m, 2] = (max(max(0.0, -cb[i]) for i in range(5)) * tc_r) - (max(max(0.0, -cb[5 + i]) for i in range(5)) * tc_b)
        CH_CAMP_TEAM[m, 3] = (ec_b * max(max(0.0, cb[5 + i]) for i in range(5))) - (ec_r * max(max(0.0, cb[i]) for i in range(5)))
        CH_CAMP_TEAM[m, 4] = (sum(ce[:5]) * sum(cb[5:])) * 0.04 - (sum(ce[5:]) * sum(cb[:5])) * 0.04
        CH_CAMP_TEAM[m, 5] = (sum(cb[:5]) * abs(sum(cb[5:]))) * 0.04 - (sum(cb[5:]) * abs(sum(cb[:5]))) * 0.04

        # 2. CH_JUG_TEAM: mismo choque con pbias / pe / pl de jugadores
        CH_JUG_TEAM[m, 0] = (ep_b * tp_r) - (ep_r * tp_b)
        CH_JUG_TEAM[m, 1] = (pe_b_m * pt_r_m) - (pe_r_m * pt_b_m)
        CH_JUG_TEAM[m, 2] = (ep_b * tc_r) - (ep_r * tc_b)  # jugadores early B vs champs scaling R
        CH_JUG_TEAM[m, 3] = (ec_b * tp_r) - (ec_r * tp_b)  # champs early B vs jugadores scaling R
        CH_JUG_TEAM[m, 4] = (max(max(0.0, -pb[i]) for i in range(5)) * tp_r) - (max(max(0.0, -pb[5 + i]) for i in range(5)) * tp_b)
        CH_JUG_TEAM[m, 5] = (sum(pb[:5]) * abs(sum(pb[5:]))) * 0.04 - (sum(pb[5:]) * abs(sum(pb[:5]))) * 0.04

        # 3. CH_LANE: choque directo contra rival de linea l (0=BOT, 1=JG, 2=MID, 3=TOP, 4=SUP)
        #    E_{b,l}*T_{r,l} - E_{r,l}*T_{b,l}
        ch_cl = [max(0.0, -cb[l]) * max(0.0, cb[5 + l]) - max(0.0, -cb[5 + l]) * max(0.0, cb[l]) for l in range(5)]
        ch_pl = [max(0.0, -pb[l]) * max(0.0, pb[5 + l]) - max(0.0, -pb[5 + l]) * max(0.0, pb[l]) for l in range(5)]
        ch_cr = [ce[l] * cl[5 + l] - ce[5 + l] * cl[l] for l in range(5)]
        CH_LANE[m, 0] = sum(ch_cl)
        CH_LANE[m, 1] = sum(ch_pl)
        CH_LANE[m, 2] = sum(ch_cr)
        CH_LANE[m, 3] = ch_cl[1]  # JG vs JG
        CH_LANE[m, 4] = ch_cl[0] + ch_cl[4]  # BOT + SUP
        CH_LANE[m, 5] = ch_cl[2] + ch_cl[3]  # SOLO LANES
        # Jungla early B vs scaling de lineas R
        CH_LANE[m, 6] = (max(0.0, -cb[1]) * (cb[5] + cb[7] + cb[8])) - (max(0.0, -cb[6]) * (cb[0] + cb[2] + cb[3]))
        # Jugador early B vs campeon scaling R en su linea
        CH_LANE[m, 7] = sum(max(0.0, -pb[l]) * max(0.0, cb[5 + l]) - max(0.0, -pb[5 + l]) * max(0.0, cb[l]) for l in range(5))

        # 4. COHERENCIA interna de equipo + ritmo conjunto
        mc_b, mc_r = sum(cb[:5]) * 0.2, sum(cb[5:]) * 0.2
        mp_b, mp_r = sum(pb[:5]) * 0.2, sum(pb[5:]) * 0.2
        std_cb = math.sqrt(sum((cb[i] - mc_b) ** 2 for i in range(5)) * 0.2)
        std_cr = math.sqrt(sum((cb[5 + i] - mc_r) ** 2 for i in range(5)) * 0.2)
        std_pb = math.sqrt(sum((pb[i] - mp_b) ** 2 for i in range(5)) * 0.2)
        std_pr = math.sqrt(sum((pb[5 + i] - mp_r) ** 2 for i in range(5)) * 0.2)
        COHERENCIA[m, 0] = std_cb - std_cr
        COHERENCIA[m, 1] = std_pb - std_pr
        # Jungla early + lineas late dentro del MISMO equipo (desacople o 1-3-1 / seguro de escala)
        COHERENCIA[m, 2] = (ce[1] * (cb[0] + cb[2] + cb[3]) * 0.33) - (ce[6] * (cb[5] + cb[7] + cb[8]) * 0.33)
        COHERENCIA[m, 3] = (cb[1] * (cb[0] + cb[2] + cb[3]) * 0.33) - (cb[6] * (cb[5] + cb[7] + cb[8]) * 0.33)
        COHERENCIA[m, 4] = (pb[1] * cb[1]) - (pb[6] * cb[6])
        COHERENCIA[m, 5] = (sum(ce[i] * max(0.0, pe[i]) for i in range(5)) - sum(ce[5 + i] * max(0.0, pe[5 + i]) for i in range(5))) * 0.2

        # Actualizar despues
        for i in range(10):
            p = ps[i]
            cls_i = cls[i]
            win_p = yb if i < 5 else (1.0 - yb)
            if dm < 26.0:
                wr_e_sum[p] = wr_e_sum[p] * dec30 + win_p
                wr_e_n[p] = wr_e_n[p] * dec30 + 1.0
                cl_e_win[cls_i] += win_p
                cl_e_n[cls_i] += 1.0
            elif dm >= 29.0:
                wr_l_sum[p] = wr_l_sum[p] * dec30 + win_p
                wr_l_n[p] = wr_l_n[p] * dec30 + 1.0
                cl_l_win[cls_i] += win_p
                cl_l_n[cls_i] += 1.0

    out = {
        "p_early_wr": p_early_wr, "p_late_wr": p_late_wr, "p_bias": p_bias,
        "c_early_wr": c_early_wr, "c_late_wr": c_late_wr, "c_bias": c_bias,
        "CH_CAMP_TEAM": CH_CAMP_TEAM,
        "CH_JUG_TEAM": CH_JUG_TEAM,
        "CH_LANE": CH_LANE,
        "COHERENCIA": COHERENCIA,
    }
    np.savez_compressed(CACHE, nm=np.array([NM]), **out)
    print(f"Caché gemini_g18_tempo.npz lista en {time.time() - t0:.1f}s", flush=True)
    return out


def main():
    t0 = time.time()
    D = cargar()
    ms = D["master"]
    R6 = extras_referencia(D, escalon=6)
    G = construir_cache_tempo(D)

    CH_TEAM = np.c_[G["CH_CAMP_TEAM"], G["CH_JUG_TEAM"]].astype(np.float32)
    ALL_G18 = np.c_[G["CH_CAMP_TEAM"], G["CH_JUG_TEAM"], G["CH_LANE"], G["COHERENCIA"]].astype(np.float32)
    VAR = {
        "ref: Escalón 6 (sin mapa)": R6,
        "+ CH_CAMP_TEAM (6)": np.c_[R6, G["CH_CAMP_TEAM"]],
        "+ CH_JUG_TEAM (6)": np.c_[R6, G["CH_JUG_TEAM"]],
        "+ CH_TEAM juntos (12)": np.c_[R6, CH_TEAM],
        "+ CH_LANE directo (8)": np.c_[R6, G["CH_LANE"]],
        "+ COHERENCIA tempo (6)": np.c_[R6, G["COHERENCIA"]],
        "+ TODO g18 tempo (26)": np.c_[R6, ALL_G18],
    }

    def run_one(nom, X):
        X32 = np.asarray(X, dtype=np.float32)
        return nom, evaluar(D, lambda tr, te, M=X32: ajustar_predecir(D, tr[ms[tr]], te, extra=M, extra_post=False, con_mapa=False))

    P = dict(Parallel(n_jobs=2, backend="threading")(delayed(run_one)(n, X) for n, X in VAR.items()))
    mostrar(comparar(D, P, "ref: Escalón 6 (sin mapa)"))

    di = np.flatnonzero(D["diam_v"])
    yd = D["y"][di]
    B = np.random.default_rng(1).integers(0, len(di), size=(300, len(di)))
    b0 = P["ref: Escalón 6 (sin mapa)"][0][di]
    print(f"\nChequeo Diamond-V ({len(di):,} partidas):")
    for nom in [k for k in VAR if not k.startswith("ref:")]:
        p = P[nom][0][di]
        d = np.array([roc_auc_score(yd[b], p[b]) - roc_auc_score(yd[b], b0[b]) for b in B])
        print(f"  {nom:28s} AUC {roc_auc_score(yd, p):.4f}  Δ {roc_auc_score(yd, p) - roc_auc_score(yd, b0):+.4f} "
              f"[{np.percentile(d, 2.5):+.4f}; {np.percentile(d, 97.5):+.4f}]")
    print(f"Listo en {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
