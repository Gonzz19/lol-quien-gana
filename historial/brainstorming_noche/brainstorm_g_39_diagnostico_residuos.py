"""
Brainstorming Noche — Gemini G39 (`brainstorm_g_39_diagnostico_residuos.py`):
Diagnóstico EXPLORATORIO exclusivamente sobre `A2..A4` Master+ (`13.860` partidas
anteriores al `16/09`; `V` y `Diamond-V` quedan 100 % INTACTOS para evitar cualquier
fuga/sobreajuste de selección).

Mide:
1. Calibración, AUC por subgrupo en `A2..A4` del `Escalón 9 oficial` (`con_mapa=False`).
2. Correlaciones residuales `r = y - p_esc9` en `A2..A4` (con p-valor z = r*sqrt(N))
   frente a sondas estructurales no probadas aún o dispersas:
   - Dispersión por bloque (`std/max/min` en los 5 carriles de `HGB`, `MEM`, `MC`, `A`).
   - Interacciones cruzadas de rol (`JG` × `Top/Mid/Bot`, `Sup` × `ADC/Mid`, `Solo_lanes` vs `Botlane`).
   - Interacciones no lineales del propio logit / calibración (`logit^3`, `sign(z)*sqrt|z|`).
   - Interacciones simétricas de composición DDragon (`ATQ/MAG/TANK/RANGE` × enemigo).
   - Interacciones con contexto de `A2..A4` (`estrenos`, `offrole`, `tier_lobby`, `sin_foto`, `hora`).
3. Efecto de escala **conjunta** (`C` y pesos relativos `DRAFT` / `Xd` / `Esc9`) medidos solo en `A2..A4`.

Uso:
  python historial/brainstorming_noche/brainstorm_g_39_diagnostico_residuos.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.analysis.brainstorm_base import (  # noqa: E402
    ajustar_predecir,
    cargar,
    extras_referencia,
)


def main():
    t0 = time.time()
    D = cargar()
    ref9 = extras_referencia(D, escalon=9)
    NM = D["NM"]
    ms = D["master"]
    y = D["y"]
    fi = D["fi"]
    bl = D["bloque"]
    cols = D["cols"]
    Xj = D["Xj"]
    print(f"Cargado en {time.time() - t0:.1f}s")

    # Predicciones OOS exclusivamente en A2..A4 Master+
    po = np.full(NM, np.nan)
    for b in D["EV"]:
        tr = fi[bl[fi] < b]
        te = np.flatnonzero(bl == b)
        po[te] = ajustar_predecir(D, tr[ms[tr]], te, extra=ref9, extra_post=False, con_mapa=False)

    idx_a = np.flatnonzero((bl >= 1) & (bl <= 3) & ms & np.isfinite(po))
    ya = y[idx_a]
    pa = po[idx_a]
    ra = ya - pa
    N = len(idx_a)
    print(f"Muestra diagnóstica A2..A4 Master+: N={N:,} | AUC={roc_auc_score(ya, pa):.4f} | LL={log_loss(ya, pa):.5f}")

    # 1) Chequeo de calibración por decil en A2..A4
    dec = pd.qcut(pa, 10, labels=False)
    cal = pd.DataFrame({"p": pa, "y": ya, "d": dec}).groupby("d").agg(n=("y", "size"), p=("p", "mean"), y=("y", "mean"))
    cal["err"] = cal["y"] - cal["p"]
    print("\n=== Calibración A2..A4 por decil ===")
    print(cal.round(4).to_string())

    # 2) Subgrupos en A2..A4
    ci = {c: i for i, c in enumerate(cols)}
    Xja = Xj[idx_a]
    n_prev = np.expm1(Xja[..., ci["logn_tot"]]).reshape(N, 10)
    estr = Xja[..., ci["estreno"]].reshape(N, 10).sum(1)
    off = Xja[..., ci["offrole"]].reshape(N, 10).sum(1)
    tl = np.nan_to_num(D["Z"]["tnum"][idx_a], nan=0.0)
    gm_cnt = (tl >= 2).sum(1)
    dur = np.nan_to_num(D["Z"]["D_dur_min"][idx_a], nan=28.0)

    print("\n=== AUC / LL por subgrupos en A2..A4 ===")
    segs = {
        "estrenos=0": estr == 0,
        "estrenos>=1": estr >= 1,
        "offrole=0": off == 0,
        "offrole>=1": off >= 1,
        "min_n<10": n_prev.min(1) < 10,
        "min_n>=30": n_prev.min(1) >= 30,
        "sin_GM": gm_cnt == 0,
        "con_GM+": gm_cnt >= 1,
        "dur<25m": dur < 25.0,
        "dur 25-32m": (dur >= 25.0) & (dur < 32.0),
        "dur>=32m": dur >= 32.0,
    }
    for sn, sm in segs.items():
        if sm.sum() >= 300:
            print(f"  {sn:14s}: n={sm.sum():5d} | AUC={roc_auc_score(ya[sm], pa[sm]):.4f} | LL={log_loss(ya[sm], pa[sm]):.5f}")

    # 3) Sondas masivas sobre el residuo `ra = ya - pa` EN A2..A4
    probes = {}
    C_mg = np.load(REPO / "data" / "analysis" / "brainstorm" / "margen_grupos.npz")
    MC_l = (C_mg["MC"][idx_a, 0] - C_mg["MC"][idx_a, 1]).mean(-1)  # (N, 5)
    MC_team = C_mg["MC"][idx_a].mean(-1)  # (N, 2, 5)
    A_mmr = np.load(REPO / "data" / "analysis" / "brainstorm" / "mmr_historico.npz")["A"][idx_a]  # (N, 2, 5, 5)
    HGB = np.load(REPO / "data" / "analysis" / "brainstorm" / "gemini_hgb_lineas.npz")["HGB"][idx_a]
    MEM = np.load(REPO / "data" / "analysis" / "brainstorm" / "gemini_g23.npz")["MATCHUP_EXP_MARGIN"][idx_a]
    g11 = np.load(REPO / "data" / "analysis" / "brainstorm" / "gemini_g11.npz")
    KOUT = g11["KALMAN_OUT"][idx_a]

    # Logit y no-linealidades del logit
    logit_a = np.log(np.clip(pa, 1e-5, 1 - 1e-5) / (1 - np.clip(pa, 1e-5, 1 - 1e-5)))
    probes["logit_sq_signed"] = logit_a * np.abs(logit_a)
    probes["logit_cu"] = logit_a**3
    probes["logit_tanh"] = np.tanh(logit_a * 2.0)
    probes["logit_x_minlogn"] = logit_a * (Xja[..., ci["logn_tot"]].reshape(N, 10).min(1) - 3.0)
    probes["logit_x_estr"] = logit_a * (estr - estr.mean())
    probes["logit_x_gm"] = logit_a * (gm_cnt - gm_cnt.mean())
    probes["logit_x_off"] = logit_a * (off - off.mean())
    probes["logit_x_post_dur_proxy_sinfoto"] = logit_a * ((D["J"][idx_a, :, :, 6] < 0.5).reshape(N, 10).sum(1) - 1.0)

    # Dispersión interna (std, max, min) en HGB, MEM, MC, A
    for name, arr in [("HGB_5", np.stack([HGB[:, :5], -HGB[:, :5]], 1)),
                      ("MEM_5", np.stack([MEM[:, 4:9], -MEM[:, 4:9]], 1)),
                      ("MC_5", MC_team),
                      ("MMR_5", A_mmr[..., 0])]:
        probes[f"{name}_max"] = arr[:, 0].max(1) - arr[:, 1].max(1)
        probes[f"{name}_min"] = arr[:, 0].min(1) - arr[:, 1].min(1)
        probes[f"{name}_std"] = arr[:, 0].std(1) - arr[:, 1].std(1)

    # Slots: 0=BOT, 1=JG, 2=MID, 3=TOP, 4=SUP
    # Sinergias / choques cruzados entre roles en la misma partida
    # JG(1) × Solos(2,3) y Bot(0)+Sup(4) en stats reales de jugador (no solo champ)
    d_kp = Xja[:, 0, :, ci["acum_kills_pm"]] - Xja[:, 1, :, ci["acum_kills_pm"]]
    d_dmg = Xja[:, 0, :, ci["acum_damage_share_pct"]] - Xja[:, 1, :, ci["acum_damage_share_pct"]]
    d_g15 = Xja[:, 0, :, ci["acum_d_gold_15"]] - Xja[:, 1, :, ci["acum_d_gold_15"]]
    jg_b, jg_r = MC_team[:, 0, 1], MC_team[:, 1, 1]
    mid_b, mid_r = MC_team[:, 0, 2], MC_team[:, 1, 2]
    top_b, top_r = MC_team[:, 0, 3], MC_team[:, 1, 3]
    bot_b, bot_r = MC_team[:, 0, 0], MC_team[:, 1, 0]
    sup_b, sup_r = MC_team[:, 0, 4], MC_team[:, 1, 4]

    probes["syn_JG_x_MID_MC"] = jg_b * mid_b - jg_r * mid_r
    probes["syn_BOT_x_SUP_MC"] = bot_b * sup_b - bot_r * sup_r
    probes["syn_JG_x_TOP_MC"] = jg_b * top_b - jg_r * top_r
    probes["syn_HGB_JG_x_MID"] = HGB[:, 1] * (HGB[:, 2])  # simetrico? cuidado! no usar producto de 2 diffs!
    del probes["syn_HGB_JG_x_MID"]
    # Sinergia simétrica por equipo en atributos DataDragon (Blue vs Red)
    # Sumas por equipo de tag_Tank, tag_Mage, tag_Assassin, tag_Marksman, tag_Support, info_atk, info_mag, st_attackrange
    tags = {k: Xja[:, :, :, ci[k]] for k in cols if k.startswith(("tag_", "info_", "st_"))}
    tk_b, tk_r = tags["tag_Tank"][:, 0].sum(1), tags["tag_Tank"][:, 1].sum(1)
    mg_b, mg_r = tags["tag_Mage"][:, 0].sum(1), tags["tag_Mage"][:, 1].sum(1)
    as_b, as_r = tags["tag_Assassin"][:, 0].sum(1), tags["tag_Assassin"][:, 1].sum(1)
    mk_b, mk_r = tags["tag_Marksman"][:, 0].sum(1), tags["tag_Marksman"][:, 1].sum(1)
    rg_b, rg_r = tags["st_attackrange"][:, 0].mean(1), tags["st_attackrange"][:, 1].mean(1)
    iatk_b, iatk_r = tags["info_atk"][:, 0].sum(1), tags["info_atk"][:, 1].sum(1)
    imag_b, imag_r = tags["info_mag"][:, 0].sum(1), tags["info_mag"][:, 1].sum(1)
    idef_b, idef_r = tags["info_def"][:, 0].sum(1), tags["info_def"][:, 1].sum(1)

    # Productos inter-equipo estrictamente antisimétricos: f(B,R) = -f(R,B)
    probes["comp_range_x_enemy_ass"] = rg_b * as_r - rg_r * as_b
    probes["comp_range_x_enemy_tank"] = rg_b * tk_r - rg_r * tk_b
    probes["comp_atk_x_enemy_def"] = iatk_b * idef_r - iatk_r * idef_b
    probes["comp_mag_x_enemy_def"] = imag_b * idef_r - imag_r * idef_b
    probes["comp_atk_ratio_sq"] = (iatk_b / (iatk_b + imag_b + 1.0))**2 - (iatk_r / (iatk_r + imag_r + 1.0))**2
    probes["comp_tank_sat"] = (tk_b >= 2).astype(float) - (tk_r >= 2).astype(float)
    probes["comp_no_tank"] = (tk_b == 0).astype(float) - (tk_r == 0).astype(float)
    probes["comp_double_ad"] = (mk_b >= 2).astype(float) - (mk_r >= 2).astype(float)
    probes["comp_mage_bot_or_no_mk"] = (mk_b == 0).astype(float) - (mk_r == 0).astype(float)
    probes["comp_hgb_x_range_adv"] = (HGB[:, 5]) * (rg_b + rg_r - 650.0) / 100.0  # HGB * sum(range) antisimétrico
    probes["comp_hgb_x_tank_tot"] = (HGB[:, 5]) * (tk_b + tk_r - 2.0)

    # Diferencia entre KALMAN_OUT y Xd/MC, horas/parche
    # Desacuerdo entre DRAFT/MEM (campeones) y MC/ELO (jugadores)
    for k in range(5):
        probes[f"d_kills_lane{k}"] = d_kp[:, k]
        probes[f"d_dmg_lane{k}"] = d_dmg[:, k]
        probes[f"d_g15_lane{k}"] = d_g15[:, k]
    # Hora del día y día de semana interacción con favoritos
    hora = ((D["t"][idx_a] / 3.6e6) - 3.0) % 24.0
    probes["logit_x_madrugada"] = logit_a * ((hora >= 1) & (hora < 7)).astype(float)
    # Diferencias por rol de participación / utilidad del último partido (`log_horas_desde`, `ritmo`, `conc_rec`)
    for fnm in ("log_horas_desde", "ritmo", "conc_rec", "d_pool", "otp_nuevo", "wr_champ", "share_champ"):
        arr = Xja[:, :, :, ci[fnm]]
        probes[f"agg_max_{fnm}"] = arr[:, 0].max(1) - arr[:, 1].max(1)
        probes[f"agg_min_{fnm}"] = arr[:, 0].min(1) - arr[:, 1].min(1)

    rows = []
    for k, v in probes.items():
        v = np.nan_to_num(v.astype(np.float64), nan=0.0)
        if v.std() < 1e-8:
            continue
        r = float(np.corrcoef(ra, v)[0, 1])
        z = r * np.sqrt(N)
        rows.append((abs(z), z, r, k))

    rows.sort(reverse=True)
    print(f"\n=== TOP 25 SONDAS RESIDUALES (r con y - p_esc9 en A2..A4, N={N:,}) ===")
    for _, z, r, k in rows[:25]:
        print(f"  {k:32s}: r={r:+.5f}  | z={z:+.2f}")

    # 4) Escaneo de regularización CONJUNTA por escala en A2..A4 (ZERO V!)
    print("\n=== Barrido de C y pesos relativos de bloques conjuntos SOLO en A2..A4 ===")
    def eval_a24(C=0.05, s_draft=1.0, s_xd=1.0, s_ref=1.0):
        po_x = np.full(NM, np.nan)
        for b in D["EV"]:
            tr = fi[bl[fi] < b]
            tr_m = tr[ms[tr]]
            te = np.flatnonzero(bl == b)
            mu, sd = D["Xd"][tr_m].mean(0), D["Xd"][tr_m].std(0) + 1e-6
            me, se = ref9[tr_m].mean(0), ref9[tr_m].std(0) + 1e-6
            Xtr = sparse.hstack([
                D["DRAFT"][tr_m] * s_draft,
                sparse.csr_matrix(((D["Xd"][tr_m] - mu) / sd * s_xd).astype(np.float32)),
                sparse.csr_matrix(((ref9[tr_m] - me) / se * s_ref).astype(np.float32)),
            ]).tocsr()
            Xte = sparse.hstack([
                D["DRAFT"][te] * s_draft,
                sparse.csr_matrix(((D["Xd"][te] - mu) / sd * s_xd).astype(np.float32)),
                sparse.csr_matrix(((ref9[te] - me) / se * s_ref).astype(np.float32)),
            ]).tocsr()
            lr = LogisticRegression(C=C, max_iter=4000).fit(Xtr, y[tr_m])
            po_x[te] = lr.predict_proba(Xte)[:, 1]
        aucs = []
        for b in D["EV"]:
            m = (bl == b) & ms
            aucs.append(roc_auc_score(y[m], po_x[m]))
        ll_a = log_loss(ya, po_x[idx_a])
        print(f"  C={C:4.2f} d={s_draft:.1f} x={s_xd:.1f} r={s_ref:.1f} -> A2={aucs[0]:.4f} A3={aucs[1]:.4f} A4={aucs[2]:.4f} mean={np.mean(aucs):.4f} LL_A={ll_a:.5f}")

    for cfg in [
        (0.05, 1.0, 1.0, 1.0),
        (0.02, 1.0, 1.0, 1.0),
        (0.01, 1.0, 1.0, 1.0),
        (0.10, 1.0, 1.0, 1.0),
        (0.05, 1.5, 0.5, 1.5),
        (0.05, 1.0, 0.4, 1.5),
        (0.02, 1.5, 0.5, 2.0),
        (0.05, 2.0, 0.5, 1.5),
    ]:
        eval_a24(*cfg)


if __name__ == "__main__":
    main()
