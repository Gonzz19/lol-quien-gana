"""Ablación rápida de los sub-componentes de `BUILD_ARCH_ADAPT` (cacheados en `gemini_g20.npz`)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.brainstorm_base import ajustar_predecir, cargar, comparar, evaluar, extras_referencia, mostrar  # noqa: E402
from historial.brainstorming_noche.brainstorm_g_20_items_jungla_trueskill import _eval_l2_dif  # noqa: E402

CACHE = REPO / "data" / "analysis" / "brainstorm" / "gemini_g20.npz"


def main():
    t0 = time.time()
    D = cargar()
    C20 = np.load(CACHE)
    B = C20["BUILD_ARCH_ADAPT"]
    T = C20["BAYES_TRUESKILL"]
    REF6 = extras_referencia(D, escalon=6)
    ms = D["master"]

    def solo_m(extra=None):
        tot = REF6 if extra is None else np.c_[REF6, extra].astype(np.float32)
        return lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=tot, extra_post=False, con_mapa=False)

    P = {"ref: Escalón 6 (sin mapa)": evaluar(D, solo_m())}
    P["+ BUILD_DEV (cols 0:3)"] = evaluar(D, solo_m(B[:, :3]))
    P["+ FULL_AD_AP_RISK (3:5)"] = evaluar(D, solo_m(B[:, 3:5]))
    P["+ BOOT_ADAPT (5:7)"] = evaluar(D, solo_m(B[:, 5:7]))
    P["+ MEJAI_GREED (7:8)"] = evaluar(D, solo_m(B[:, 7:8]))
    P["+ BOOT_MEJAI (5:8)"] = evaluar(D, solo_m(B[:, 5:8]))
    P["+ DEV + BOOT (0:3,5:8)"] = evaluar(D, solo_m(np.c_[B[:, :3], B[:, 5:8]]))
    P["+ BUILD(8) c/L2 dif"] = evaluar(D, _eval_l2_dif(D, REF6, B))
    P["+ DEV+BOOT+TRUE c/L2 dif"] = evaluar(D, _eval_l2_dif(D, REF6, np.c_[B[:, :3], B[:, 5:8], T]))
    mostrar(comparar(D, P, "ref: Escalón 6 (sin mapa)"))

    fi_v, dv = D["fi_v"], D["diam_v"]
    ydv = D["y"][dv]
    p_dv_ref = solo_m()(fi_v, dv)
    auc_ref = roc_auc_score(ydv, p_dv_ref)
    rng = np.random.default_rng(77)
    boots = rng.integers(0, len(ydv), size=(1500, len(ydv)))
    print(f"\nChequeo Diamond-V ({len(ydv):,} partidas): ref={auc_ref:.4f}")
    for nom, fn in [
        ("+ BUILD_DEV (0:3)", solo_m(B[:, :3])),
        ("+ FULL_AD_AP_RISK (3:5)", solo_m(B[:, 3:5])),
        ("+ BOOT_ADAPT (5:7)", solo_m(B[:, 5:7])),
        ("+ MEJAI_GREED (7:8)", solo_m(B[:, 7:8])),
        ("+ BOOT_MEJAI (5:8)", solo_m(B[:, 5:8])),
        ("+ BUILD(8) c/L2 dif", _eval_l2_dif(D, REF6, B)),
        ("+ DEV+BOOT+TRUE c/L2", _eval_l2_dif(D, REF6, np.c_[B[:, :3], B[:, 5:8], T])),
    ]:
        p_dv = fn(fi_v, dv)
        auc_p = roc_auc_score(ydv, p_dv)
        ds = np.array([roc_auc_score(ydv[b], p_dv[b]) - roc_auc_score(ydv[b], p_dv_ref[b]) for b in boots])
        print(f"  {nom:24s} AUC {auc_p:.4f}  Δ {auc_p - auc_ref:+.4f} [{np.percentile(ds, 2.5):+.4f}; {np.percentile(ds, 97.5):+.4f}]")
    print(f"Listo en {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
