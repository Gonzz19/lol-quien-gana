"""Idea G42 (`docs/BRAINSTORM_NOCHE.md`): **Rematch inmediato (`< 75 min`) y Tilt Mecánico en Sesión `< t`**:
A diferencia de `sesión` de §9.S.24 (racha binaria de W/L, con 75 % ruido) y de `duos` globales,
exploita marcas exactas de fin de partida (`t_prev + dur_prev * 1000`) y rendimiento mecánico `@15`:

  1) `REMATCH_MENTAL (4)`:
     - `ally_lost_75m`: pares del mismo equipo que perdieron JUNTOS hace `< 75 min` (mental/fricción)
     - `ally_won_75m`: pares del mismo equipo que ganaron JUNTOS hace `< 75 min`
     - `opp_dom_75m`: saldo previo Blue vs Red en enfrentamientos directos hace `< 75 min`
     - `insta_duo_30m`: jugadores que vienen de compartir lobby hace `< 30 min` (en mismo equipo vs rival)
  2) `MECH_TILT_SESION (5)`:
     - `insta_requeue_loss`: jugadores que perdieron su partida anterior y entraron a esta `< 8 min` tras explotar el nexo
     - `stomp_ff_tilt`: derrota rápida anterior (`dur <= 21 min`) con re-cola `< 25 min` (#6 de `fit`: `p=1.79e-5`)
     - `marathon_4h`: exceso de fatiga (`max(0, n_juegos_ultimas_4h - 2)`)
     - `losses_4h`: derrotas acumuladas en las últimas `4 horas` (tilt acumulado de sesión)
     - `mech_drop_15`: caída mecánica reciente: promedio `dg15_res` de sus últimas 2 partidas (`< 4h`) menos su media previa larga (captura manos frías / tilt jugando peor en línea hoy)

Todo 100 % causal (`< m` en orden `(t, m)`), evaluado sobre **Escalón 9 oficial (`0,6739`)**, con placebo y `Diamond-V`.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.brainstorm_base import (  # noqa: E402
    ajustar_predecir,
    cargar,
    comparar,
    evaluar,
    extras_referencia,
    mostrar,
)
from historial.brainstorming_noche.brainstorm_g_29_matchup_2v2_15 import _extraer_15_por_carril  # noqa: E402

CACHE = REPO / "data" / "analysis" / "brainstorm" / "gemini_g32_sesion_tilt.npz"


def construir_sesion_y_rematch(D: dict, rehacer: bool = False) -> dict[str, np.ndarray]:
    if CACHE.exists() and not rehacer:
        z = np.load(CACHE)
        if int(z["nm"][0]) == D["NM"]:
            return {k: z[k] for k in z.files if k != "nm"}

    t0 = time.time()
    print("[g_32] Construyendo REMATCH_MENTAL y MECH_TILT_SESION causales (< m)...", flush=True)
    NM = D["NM"]
    Z = D["Z"]
    mids = Z["mids"].astype(str)
    t_arr = D["t"].astype(np.int64)
    y_arr = D["y"].astype(np.int8)
    dur_ms = (Z["D_dur_min"].astype(np.float64) * 60_000.0).astype(np.int64)
    champ = Z["champ"].reshape(NM, 2, 5)
    dg, _, ok15 = _extraer_15_por_carril(D)

    players = D["B"]["player"].reshape(NM, 10).astype(str)
    uniq_p, p_inv = np.unique(players.reshape(-1), return_inverse=True)
    pid_mat = p_inv.reshape(NM, 10).tolist()
    n_p = len(uniq_p)

    uc, c_inv = np.unique(champ.reshape(-1), return_inverse=True)
    c_mat = c_inv.reshape(NM, 2, 5).tolist()
    nc = len(uc)
    cl_g_s = np.zeros((nc, 5), dtype=np.float64)
    cl_g_n = np.zeros((nc, 5), dtype=np.float64)

    # Historial reciente por jugador (para ventanas exactas de 75m / 4h y tilt mecánico)
    # Guardamos última partida y cola de últimas 4 horas
    last_end_t = [-1] * n_p
    last_win = [0] * n_p
    last_dur = [0] * n_p
    hist_4h = [deque() for _ in range(n_p)]          # lista de (t_start, win_p)
    hist_g15_4h = [deque() for _ in range(n_p)]      # lista de (t_start, g15_res_signed)
    # Memoria corta de últimos 20 minutos/partidas por jugador para baseline @15
    ew_g15 = [0.0] * n_p
    ew_n15 = [0.0] * n_p
    # Últimas partidas recientes (t_end, win_p, other_player -> side_relation): usamos últimas ~6 partidas en 90m por jugador
    recent_mates = [deque() for _ in range(n_p)]     # (t_start, team_players_frozenset, opp_players_frozenset, win_p)

    W75_MS = 75 * 60_000
    W30_MS = 30 * 60_000
    W4H_MS = 4 * 3600_000

    REMATCH_MENTAL = np.zeros((NM, 4), dtype=np.float32)
    MECH_TILT = np.zeros((NM, 5), dtype=np.float32)

    order = np.lexsort((mids, t_arr))
    dg_l = dg.tolist()
    ok_l = ok15.tolist()

    for m in order:
        tm = int(t_arr[m])
        dm = int(dur_ms[m])
        yb = float(y_arr[m])
        ps = pid_mat[m]
        cb, cr = c_mat[m][0], c_mat[m][1]

        # ---- 1. Leer estado previo causal (< m) ----
        # Rematch en últimas 75m / 30m usando historial reciente de los 10 jugadores
        b_set = set(ps[:5])
        r_set = set(ps[5:])

        def team_rematch_feats(team_ps, my_set, op_set):
            lost_same = 0.0
            won_same = 0.0
            opp_dom = 0.0
            same_30 = 0.0
            for idx_i, pi in enumerate(team_ps):
                q = recent_mates[pi]
                while q and (tm - q[0][0]) > W75_MS:
                    q.popleft()
                for t_k, same_k, opp_k, wk in q:
                    dt_k = tm - t_k
                    # Pares dentro del mismo equipo actual (evitar doble conteo usando id menor)
                    for pj in team_ps[idx_i + 1:]:
                        if pj in same_k:
                            if wk < 0.5:
                                lost_same += 1.0
                            else:
                                won_same += 1.0
                            if dt_k <= W30_MS:
                                same_30 += 1.0
                    # Saldo directo contra rivales actuales en últimos 75m
                    for po in op_set:
                        if po in opp_k:
                            opp_dom += (1.0 if wk > 0.5 else -1.0)
            return lost_same, won_same, opp_dom, same_30

        b_ls, b_ws, b_od, b_30 = team_rematch_feats(ps[:5], b_set, r_set)
        r_ls, r_ws, r_od, r_30 = team_rematch_feats(ps[5:], r_set, b_set)
        REMATCH_MENTAL[m, 0] = b_ls - r_ls
        REMATCH_MENTAL[m, 1] = b_ws - r_ws
        REMATCH_MENTAL[m, 2] = b_od - r_od
        REMATCH_MENTAL[m, 3] = b_30 - r_30

        # Sesión / Tilt físico por jugador
        def team_tilt_feats(team_ps):
            insta_loss = 0.0
            stomp_ff = 0.0
            mar_4h = 0.0
            loss_4h = 0.0
            mdrop = 0.0
            for p in team_ps:
                te_p = last_end_t[p]
                if te_p > 0 and last_win[p] == 0:
                    rest_m = (tm - te_p) / 60_000.0
                    if 0.0 <= rest_m < 8.0:
                        insta_loss += 1.0
                    if last_dur[p] <= 21 * 60_000 and 0.0 <= rest_m < 25.0:
                        stomp_ff += 1.0

                q4 = hist_4h[p]
                while q4 and (tm - q4[0][0]) > W4H_MS:
                    q4.popleft()
                n4 = len(q4)
                if n4 > 2:
                    mar_4h += float(n4 - 2)
                loss_4h += sum(1.0 - w_k for _, w_k in q4)

                qg = hist_g15_4h[p]
                while qg and (tm - qg[0][0]) > W4H_MS:
                    qg.popleft()
                if len(qg) >= 2 and ew_n15[p] >= 5.0:
                    recent_g = (qg[-1][1] + qg[-2][1]) * 0.5
                    base_g = ew_g15[p] / ew_n15[p]
                    mdrop += max(-3.0, min(3.0, recent_g - base_g))
            return insta_loss, stomp_ff, mar_4h, loss_4h, mdrop

        bt = team_tilt_feats(ps[:5])
        rt = team_tilt_feats(ps[5:])
        for k in range(5):
            MECH_TILT[m, k] = bt[k] - rt[k]

        # ---- 2. Actualizar estado DESPUÉS ----
        dgm = dg_l[m]
        okm = ok_l[m]
        for slot in range(10):
            p = ps[slot]
            l = slot % 5
            is_b = slot < 5
            wp = yb if is_b else (1.0 - yb)
            my_team = b_set if is_b else r_set
            op_team = r_set if is_b else b_set

            last_end_t[p] = tm + dm
            last_win[p] = 1 if wp > 0.5 else 0
            last_dur[p] = dm
            hist_4h[p].append((tm, wp))
            recent_mates[p].append((tm, my_team, op_team, wp))

            if okm[l]:
                g_signed = dgm[l] if is_b else -dgm[l]
                c_p = cb[l] if is_b else cr[l]
                c_o = cr[l] if is_b else cb[l]
                exp_g = (cl_g_s[c_p, l] / (cl_g_n[c_p, l] + 20.0)) - (cl_g_s[c_o, l] / (cl_g_n[c_o, l] + 20.0))
                res_g = g_signed - exp_g
                hist_g15_4h[p].append((tm, res_g))
                ew_g15[p] = ew_g15[p] * 0.95 + res_g
                ew_n15[p] = ew_n15[p] * 0.95 + 1.0

        for l in range(5):
            if okm[l]:
                cl_g_s[cb[l], l] += dgm[l]
                cl_g_n[cb[l], l] += 1.0
                cl_g_s[cr[l], l] -= dgm[l]
                cl_g_n[cr[l], l] += 1.0

    np.savez_compressed(CACHE, nm=np.array([NM]), REMATCH_MENTAL=REMATCH_MENTAL, MECH_TILT=MECH_TILT)
    print(f"[g_32] Caché guardada en {time.time() - t0:.1f}s", flush=True)
    return {"REMATCH_MENTAL": REMATCH_MENTAL, "MECH_TILT": MECH_TILT}


if __name__ == "__main__":
    D = cargar()
    NM, ms = D["NM"], D["master"]
    X9 = extras_referencia(D, escalon=9)
    res = construir_sesion_y_rematch(D)
    RM, MT = res["REMATCH_MENTAL"], res["MECH_TILT"]
    TODO = np.c_[RM, MT].astype(np.float32)
    PL = TODO[np.random.default_rng(31).permutation(NM)]

    def ml(X):
        return lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=X, extra_post=False, con_mapa=False)

    VAR = {
        "Escalón 9 oficial": ml(X9),
        "+ REMATCH_MENTAL (4)": ml(np.c_[X9, RM]),
        "+ MECH_TILT (5)": ml(np.c_[X9, MT]),
        "+ TODO g32 (9)": ml(np.c_[X9, TODO]),
        "placebo g32": ml(np.c_[X9, PL]),
    }
    P = dict(Parallel(n_jobs=2, backend="threading")(delayed(lambda n: (n, evaluar(D, VAR[n])))(n) for n in VAR))
    mostrar(comparar(D, P, "Escalón 9 oficial"))

    di = np.flatnonzero(D["diam_v"])
    yd = D["y"][di]
    B = np.random.default_rng(1).integers(0, len(di), size=(300, len(di)))
    qd = P["Escalón 9 oficial"][0][di]
    for n in VAR:
        if n == "Escalón 9 oficial":
            continue
        pd_ = P[n][0][di]
        d = np.array([roc_auc_score(yd[b], pd_[b]) - roc_auc_score(yd[b], qd[b]) for b in B])
        print(f"  Diamond-V {n:20s}: {roc_auc_score(yd, pd_):.4f} (Δ {roc_auc_score(yd, pd_) - roc_auc_score(yd, qd):+.4f} [{np.percentile(d, 2.5):+.4f}; {np.percentile(d, 97.5):+.4f}])")
