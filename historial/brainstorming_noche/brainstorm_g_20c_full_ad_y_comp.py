"""Aislamiento limpio de `full_ad_risk` (`B[:, 3:4]`) y 3 cruces estructurales puros de composición
(`no_AP × tanque_enemigo`, `sin_frontline × asesinos_enemigos`, `bajo_engage × rango_enemigo`) +
`BOOT_ADAPT (5:7)`, evaluados s/Escalón 6 sin mapa y con L2 diferencial (`g14`)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.brainstorm_base import ajustar_predecir, cargar, comparar, evaluar, extras_referencia, mostrar  # noqa: E402
from historial.brainstorming_noche.brainstorm_g_20_items_jungla_trueskill import _eval_l2_dif  # noqa: E402

CACHE20 = REPO / "data" / "analysis" / "brainstorm" / "gemini_g20.npz"
CHAMP_CSV = REPO / "data" / "ddragon" / "16.17.1" / "champions.csv"
TAGS = ["Fighter", "Tank", "Mage", "Assassin", "Marksman", "Support"]


def construir_cruces_puros(D: dict) -> dict:
    ch = pd.read_csv(CHAMP_CSV, usecols=["champion_id", "tags", "attackrange"])
    c2v = {}
    c2rng = {}
    for cid, tg, rng in zip(ch["champion_id"].astype(int), ch["tags"].astype(str), ch["attackrange"].astype(float)):
        v = np.zeros(6, dtype=np.float32)
        for i, tname in enumerate(TAGS):
            if tname in tg:
                v[i] = 1.0
        s = v.sum()
        if s > 0:
            v /= s
        c2v[cid] = v
        c2rng[cid] = 1.0 if rng >= 475 else 0.0

    champ = D["Z"]["champ"]
    NM = D["NM"]
    T = np.zeros((NM, 2, 5, 6), dtype=np.float32)
    R = np.zeros((NM, 2, 5), dtype=np.float32)
    for cid, v in c2v.items():
        msk = champ == cid
        if msk.any():
            T[msk] = v
            R[msk] = c2rng[cid]

    t_mag = (T[:, :, :, 2] + 0.5 * T[:, :, :, 5]).sum(axis=2)
    t_tank = (T[:, :, :, 1] + 0.35 * T[:, :, :, 0]).sum(axis=2)
    t_ass = (T[:, :, :, 3]).sum(axis=2)
    t_rng = R.sum(axis=2)

    full_ad_puro = np.maximum(0.0, 1.3 - t_mag) * t_tank[:, ::-1]
    squishy_vs_ass = np.maximum(0.0, 0.9 - t_tank) * t_ass[:, ::-1]
    melee_vs_poke = np.maximum(0.0, 2.0 - t_rng) * np.maximum(0.0, t_rng[:, ::-1] - 2.0)

    return {
        "FULL_AD_PURO": (full_ad_puro[:, 0] - full_ad_puro[:, 1])[:, None].astype(np.float32),
        "COMP_TRIO": np.c_[
            full_ad_puro[:, 0] - full_ad_puro[:, 1],
            squishy_vs_ass[:, 0] - squishy_vs_ass[:, 1],
            melee_vs_poke[:, 0] - melee_vs_poke[:, 1],
        ].astype(np.float32),
    }


def main():
    t0 = time.time()
    D = cargar()
    B = np.load(CACHE20)["BUILD_ARCH_ADAPT"]
    CP = construir_cruces_puros(D)
    REF6 = extras_referencia(D, escalon=6)
    ms = D["master"]

    def solo_m(extra=None):
        tot = REF6 if extra is None else np.c_[REF6, extra].astype(np.float32)
        return lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=tot, extra_post=False, con_mapa=False)

    full_ad_b = B[:, 3:4]
    ad_y_boot = B[:, [3, 5, 6]]

    P = {"ref: Escalón 6 (sin mapa)": evaluar(D, solo_m())}
    P["+ full_ad_risk (B[3], 1 col)"] = evaluar(D, solo_m(full_ad_b))
    P["+ FULL_AD_PURO (tags, 1 col)"] = evaluar(D, solo_m(CP["FULL_AD_PURO"]))
    P["+ COMP_TRIO (tags, 3 cols)"] = evaluar(D, solo_m(CP["COMP_TRIO"]))
    P["+ B[3] + BOOT_ADAPT (3 cols)"] = evaluar(D, solo_m(ad_y_boot))
    P["+ B[3] c/L2 dif"] = evaluar(D, _eval_l2_dif(D, REF6, full_ad_b))
    P["+ B[3]+BOOT c/L2 dif"] = evaluar(D, _eval_l2_dif(D, REF6, ad_y_boot))
    mostrar(comparar(D, P, "ref: Escalón 6 (sin mapa)"))

    fi_v, dv = D["fi_v"], D["diam_v"]
    ydv = D["y"][dv]
    p_dv_ref = solo_m()(fi_v, dv)
    auc_ref = roc_auc_score(ydv, p_dv_ref)
    rng = np.random.default_rng(77)
    boots = rng.integers(0, len(ydv), size=(1500, len(ydv)))
    print(f"\nChequeo Diamond-V ({len(ydv):,} partidas): ref={auc_ref:.4f}")
    for nom, fn in [
        ("+ full_ad_risk (B[3])", solo_m(full_ad_b)),
        ("+ FULL_AD_PURO (tags)", solo_m(CP["FULL_AD_PURO"])),
        ("+ COMP_TRIO (tags)", solo_m(CP["COMP_TRIO"])),
        ("+ B[3] + BOOT_ADAPT", solo_m(ad_y_boot)),
        ("+ B[3] c/L2 dif", _eval_l2_dif(D, REF6, full_ad_b)),
        ("+ B[3]+BOOT c/L2 dif", _eval_l2_dif(D, REF6, ad_y_boot)),
    ]:
        p_dv = fn(fi_v, dv)
        auc_p = roc_auc_score(ydv, p_dv)
        ds = np.array([roc_auc_score(ydv[b], p_dv[b]) - roc_auc_score(ydv[b], p_dv_ref[b]) for b in boots])
        print(f"  {nom:24s} AUC {auc_p:.4f}  Δ {auc_p - auc_ref:+.4f} [{np.percentile(ds, 2.5):+.4f}; {np.percentile(ds, 97.5):+.4f}]")
    print(f"Listo en {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
