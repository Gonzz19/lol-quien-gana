"""Ideas 4b, 17 y 18 del brainstorming (docs/BRAINSTORM_NOCHE.md): señales causales (< t)
por fuera de los márgenes simples, evaluadas sobre `solo Master+ + MC + ML` (ref 0,6614):

1. `MATCHUPS`: residual causal bayesiano empirico (< t) de cada cruce directo de campeones
   en la misma línea (`c_azul vs c_rojo`, K=25, estricto respecto a la fuerza individual `wr_cl`)
   + sinergias causales (`ADC+SUP`, `TOP+JG`, `MID+JG`, K=30). Antisimétrico por construcción.
2. `PARCHE`: fuerza y popularidad del campeón `(c, l)` dentro del parche vigente (`Z["parche"]`,
   K=50) + EWMA reciente (vida media ~200 partidas de `(c, l)`) respecto al acumulado histórico
   (§9.S.28 dejó +0,0012 IC>0 fuera del híbrido; acá además corrige pesos viejos en Master+).
3. `REMATCH`: reencuentros en cola en ventana corta (`<= 90 min` de `last_game` por jugador)
   separando compañero tras derrota, compañero tras victoria, ex-rival tras derrota y ex-compañero
   ahora enfrente (corrige la heurística `juntos>=3` que daba 46 % falsos duos).

Sin leakage: cada partida `m` solo ve partidas cuyo fin estimado (`t + dur_min * 60_000`) es
`<= t[m]`, nunca partidas simultáneas en curso ni `tnum` / `val` futuros.
"""
import heapq
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.brainstorm_base import ajustar_predecir, cargar, comparar, evaluar, mostrar  # noqa: E402
from historial.escalon9.brainstorm_g_06_comp_poisson import cargar_foto1_eq_din  # noqa: E402

CACHE = REPO / "data" / "analysis" / "brainstorm" / "gemini_mpr.npz"


def construir_mpr(D: dict) -> dict[str, np.ndarray]:
    t0 = time.time()
    NM = D["NM"]
    t = D["t"]
    y = D["y"].astype(float)
    champ = D["Z"]["champ"].reshape(NM, 2, 5).astype(int)
    parche_str = pd_codes = None
    _, p_codes = np.unique(D["Z"]["parche"].astype(str), return_inverse=True)

    # Mapeo entero de jugadores para loops O(N) en ~3-5s
    _, pl_codes = np.unique(D["B"]["player"].reshape(-1), return_inverse=True)
    pl_codes = pl_codes.reshape(NM, 2, 5)

    ord_idx = np.lexsort((np.arange(NM), t))
    dur_ms = np.rint(np.nan_to_num(D["Z"]["D_dur_min"].astype(float), nan=30.0) * 60_000.0).astype(np.int64)
    t_fin = t.astype(np.int64) + dur_ms

    # Contadores causales (actualizados estrictamente al fin estimado t_fin <= t[m])
    g_cl = defaultdict(float)
    n_cl = defaultdict(int)
    ew_cl = defaultdict(lambda: 0.5)
    decay = 2.0 ** (-1.0 / 200.0)

    g_p = defaultdict(float)
    n_p = defaultdict(int)
    n_p_tot = defaultdict(int)
    n_tot_fin = 0

    # Counters (min_c, max_c, lane) y sinergias (slot_pair, min_c, max_c)
    g_cnt = defaultdict(float)
    n_cnt = defaultdict(int)
    g_syn = defaultdict(float)
    n_syn = defaultdict(int)
    SYN_PAIRS = ((0, 4), (3, 1), (2, 1))

    MATCHUPS = np.zeros((NM, 10), dtype=np.float32)  # 5 counters + sum_cnt + 3 syn + sum_syn
    PARCHE = np.zeros((NM, 18), dtype=np.float32)    # 5 x_parche + 5 x_rec + 5 pop + 3 sumas
    REMATCH = np.zeros((NM, 6), dtype=np.float32)    # tm_loss, tm_win, opp_loss, opp_win, rev_ex_tm_loss, ally_ex_opp_loss

    last_game = {}  # pl_id -> (t_fin_ms, win_int, team_fset, enemy_fset)
    WIN_90M = 5_400_000
    pendientes: list[tuple[int, int]] = []  # min-heap de (t_fin_ms, m)

    def _aplicar(mf: int) -> None:
        nonlocal n_tot_fin
        ym = y[mf]
        pm = int(p_codes[mf])
        tf = int(t_fin[mf])
        teams_f = [frozenset(int(x) for x in pl_codes[mf, 0]), frozenset(int(x) for x in pl_codes[mf, 1])]
        n_tot_fin += 1
        n_p_tot[pm] += 1
        for s in (0, 1):
            ws = ym if s == 0 else (1.0 - ym)
            for l in range(5):
                c = int(champ[mf, s, l])
                key_cl = (c, l)
                g_cl[key_cl] += ws
                n_cl[key_cl] += 1
                ew_cl[key_cl] = decay * ew_cl[key_cl] + (1.0 - decay) * ws
                key_pcl = (pm, c, l)
                g_p[key_pcl] += ws
                n_p[key_pcl] += 1
                p_id = int(pl_codes[mf, s, l])
                last_game[p_id] = (tf, int(round(ws)), teams_f[s], teams_f[1 - s])
            for idx_p, (l1, l2) in enumerate(SYN_PAIRS):
                c1, c2 = int(champ[mf, s, l1]), int(champ[mf, s, l2])
                key_s = (idx_p, c1, c2) if c1 <= c2 else (idx_p, c2, c1)
                g_syn[key_s] += ws
                n_syn[key_s] += 1
        for l in range(5):
            cb, cr = int(champ[mf, 0, l]), int(champ[mf, 1, l])
            if cb == cr:
                continue
            if cb < cr:
                key_c = (cb, cr, l)
                g_cnt[key_c] += ym
            else:
                key_c = (cr, cb, l)
                g_cnt[key_c] += (1.0 - ym)
            n_cnt[key_c] += 1

    for m in ord_idx:
        tm = int(t[m])
        # Incorporar solo partidas cuyo fin estimado ya ocurrió antes del inicio de `m`
        while pendientes and pendientes[0][0] <= tm:
            _, mf = heapq.heappop(pendientes)
            _aplicar(mf)

        pm = int(p_codes[m])
        n_games_tot = max(1, n_tot_fin)
        n_games_pat = max(1, n_p_tot[pm])

        # ---- 1. Calcular features ANTES de actualizar estadísticas con la partida m
        wr_side = np.empty((2, 5), dtype=float)
        xp_side = np.empty((2, 5), dtype=float)
        xr_side = np.empty((2, 5), dtype=float)
        pp_side = np.empty((2, 5), dtype=float)
        for s in (0, 1):
            for l in range(5):
                c = int(champ[m, s, l])
                key_cl = (c, l)
                ng, gg = n_cl[key_cl], g_cl[key_cl]
                e_cl = (gg + 15.0) / (ng + 30.0)
                wr_side[s, l] = e_cl
                ew = ew_cl[key_cl] if ng > 0 else e_cl
                key_pcl = (pm, c, l)
                np_c, gp_c = n_p[key_pcl], g_p[key_pcl]
                xp_side[s, l] = (gp_c + 50.0 * e_cl) / (np_c + 50.0) - e_cl
                xr_side[s, l] = ew - e_cl
                pop_s = ng / (2.0 * n_games_tot)
                pp_side[s, l] = (np_c + 20.0 * pop_s) / (2.0 * n_games_pat + 20.0) - pop_s

        dxp = xp_side[0] - xp_side[1]
        dxr = xr_side[0] - xr_side[1]
        dpp = pp_side[0] - pp_side[1]
        PARCHE[m, :5] = dxp
        PARCHE[m, 5:10] = dxr
        PARCHE[m, 10:15] = dpp
        PARCHE[m, 15] = dxp.sum()
        PARCHE[m, 16] = dxr.sum()
        PARCHE[m, 17] = dpp.sum()

        # Counters por línea
        for l in range(5):
            cb, cr = int(champ[m, 0, l]), int(champ[m, 1, l])
            if cb == cr:
                continue
            p_exp = float(np.clip(0.5 + (wr_side[0, l] - wr_side[1, l]), 0.1, 0.9))
            if cb < cr:
                key_c = (cb, cr, l)
                res = (g_cnt[key_c] + 25.0 * p_exp) / (n_cnt[key_c] + 25.0) - p_exp
            else:
                key_c = (cr, cb, l)
                p_min = 1.0 - p_exp
                res = -((g_cnt[key_c] + 25.0 * p_min) / (n_cnt[key_c] + 25.0) - p_min)
            MATCHUPS[m, l] = res
        MATCHUPS[m, 5] = MATCHUPS[m, :5].sum()

        # Sinergias clave por equipo (Azul - Rojo)
        for idx_p, (l1, l2) in enumerate(SYN_PAIRS):
            res_s = [0.0, 0.0]
            for s in (0, 1):
                c1, c2 = int(champ[m, s, l1]), int(champ[m, s, l2])
                if c1 <= c2:
                    key_s = (idx_p, c1, c2)
                else:
                    key_s = (idx_p, c2, c1)
                p_e = float(np.clip(0.5 + (wr_side[s, l1] - 0.5) + (wr_side[s, l2] - 0.5), 0.1, 0.9))
                res_s[s] = (g_syn[key_s] + 30.0 * p_e) / (n_syn[key_s] + 30.0) - p_e
            MATCHUPS[m, 6 + idx_p] = res_s[0] - res_s[1]
        MATCHUPS[m, 9] = MATCHUPS[m, 6:9].sum()

        # Rematch <= 90 min desde el fin de la partida previa por equipo (Azul - Rojo)
        teams = [frozenset(int(x) for x in pl_codes[m, 0]), frozenset(int(x) for x in pl_codes[m, 1])]
        rem_eq = np.zeros((2, 6), dtype=float)
        for s in (0, 1):
            t_me, t_op = teams[s], teams[1 - s]
            for p in t_me:
                prev = last_game.get(p)
                if prev is None or (tm - prev[0]) > WIN_90M:
                    continue
                _, w_prev, p_tm_prev, p_op_prev = prev
                n_tm = len(p_tm_prev & t_me) - 1  # -1 por si p está incluido
                n_op = len(p_op_prev & t_op)
                n_cross_rev = len(p_tm_prev & t_op) - 1 if p in p_tm_prev else len(p_tm_prev & t_op)
                n_cross_ally = len(p_op_prev & t_me)
                if n_tm > 0:
                    rem_eq[s, 0 if w_prev == 0 else 1] += n_tm * 0.5
                if n_op > 0:
                    rem_eq[s, 2 if w_prev == 0 else 3] += n_op * 0.5
                if w_prev == 0 and n_cross_rev > 0:
                    rem_eq[s, 4] += n_cross_rev
                if w_prev == 0 and n_cross_ally > 0:
                    rem_eq[s, 5] += n_cross_ally
        REMATCH[m] = (rem_eq[0] - rem_eq[1]).astype(np.float32)

        # ---- 2. Encolar el resultado de `m` para su fin estimado t_fin[m]
        heapq.heappush(pendientes, (int(t_fin[m]), int(m)))

    print(f"MPR causal construido en {time.time() - t0:.1f}s | rematch!=0 frac={(np.abs(REMATCH).sum(1) > 0).mean():.2%}", flush=True)
    return {"MATCHUPS": MATCHUPS, "PARCHE": PARCHE, "REMATCH": REMATCH}


def main():
    D = cargar()
    ms = D["master"]
    if not CACHE.exists():
        out = construir_mpr(D)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, **out)
    M = np.load(CACHE)
    A_mmr = np.load(REPO / "data" / "analysis" / "brainstorm" / "mmr_historico.npz")["A"]
    NIV_EQ = (A_mmr[:, 0] - A_mmr[:, 1])[..., :3].mean(axis=1).astype(np.float32)
    C = np.load(REPO / "data" / "analysis" / "brainstorm" / "margen_grupos.npz")
    MC_ML_NIV = np.c_[
        (C["MC"][:, 0] - C["MC"][:, 1]).reshape(D["NM"], -1),
        (C["ML"][:, 0] - C["ML"][:, 1]).reshape(D["NM"], -1),
        NIV_EQ,
    ].astype(np.float32)
    F1_EQ = cargar_foto1_eq_din(D)

    if "--solo-cache" in sys.argv:
        return

    def solo_m(extra_pre=None, extra_post=None):
        def pred(tr_raw, te):
            tr = tr_raw[ms[tr_raw]]
            if extra_post is None:
                tot = MC_ML_NIV if extra_pre is None else np.c_[MC_ML_NIV, extra_pre].astype(np.float32)
                return ajustar_predecir(D, tr, te, extra=tot, extra_post=False)
            Xd, post, mapa = D["Xd"], D["post"], D["mapa"]
            Xtr_p = np.c_[Xd[tr], MC_ML_NIV[tr]] if extra_pre is None else np.c_[Xd[tr], MC_ML_NIV[tr], extra_pre[tr]]
            Xte_p = np.c_[Xd[te], MC_ML_NIV[te]] if extra_pre is None else np.c_[Xd[te], MC_ML_NIV[te], extra_pre[te]]
            mu, sd = Xtr_p.mean(0), Xtr_p.std(0) + 1e-6
            pf = np.c_[mapa, extra_post].astype(np.float32)
            ptr = post[tr]
            me, se = pf[tr][ptr].mean(0), pf[tr][ptr].std(0) + 1e-6
            Mtr = sparse.hstack([
                D["DRAFT"][tr],
                sparse.csr_matrix(((Xtr_p - mu) / sd).astype(np.float32)),
                sparse.csr_matrix(np.where(ptr[:, None], (pf[tr] - me) / se, 0.0).astype(np.float32)),
            ]).tocsr()
            Mte = sparse.hstack([
                D["DRAFT"][te],
                sparse.csr_matrix(((Xte_p - mu) / sd).astype(np.float32)),
                sparse.csr_matrix(np.where(post[te][:, None], (pf[te] - me) / se, 0.0).astype(np.float32)),
            ]).tocsr()
            lr = LogisticRegression(C=0.05, max_iter=4000).fit(Mtr, D["y"][tr])
            return lr.predict_proba(Mte)[:, 1]
        return pred

    VAR = {
        "ref: solo Master+ + MC/ML + niv_hist": solo_m(),
        "+ counters + sinergias causales": solo_m(M["MATCHUPS"]),
        "+ parche (fuerza/pop/EWMA)": solo_m(M["PARCHE"]),
        "+ rematch <= 90m real": solo_m(M["REMATCH"]),
        "+ Foto1 + matchups + parche + rematch": solo_m(np.c_[M["MATCHUPS"], M["PARCHE"], M["REMATCH"]], F1_EQ),
    }
    P = dict(Parallel(n_jobs=1, backend="threading")(delayed(lambda n: (n, evaluar(D, VAR[n])))(n) for n in VAR))
    mostrar(comparar(D, P, "ref: solo Master+ + MC/ML + niv_hist"))


if __name__ == "__main__":
    main()
