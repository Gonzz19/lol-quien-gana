"""Ideas 7 y 13 del brainstorming (docs/BRAINSTORM_NOCHE.md): agregados de habilidad del equipo,
interacciones de autofill/smurf y asimetría de lado según nivel del lobby, sobre la referencia
vigente (solo Master+ + MC + ML = 0,6614).

A diferencia de `brainstorm_02_lobby.py` (que probó max/min/std solo de LP/nivel), acá se prueban
agregados de habilidad real (como §9.S.39, que sumó +0,0011 en el modelo liviano pero no se probó
con mapa + MC/ML + solo Master+):
* `AGREGADOS`: max, min y std dentro de cada equipo (azul − rojo) del margen medio, margen por
  campeón (MC), margen por línea (ML), winrate previo, winrate con el campeón y d_gold_15.
* `AUTOFILL_SMURF`: interacciones no lineales ausentes en `Xd`: `sum(offrole) >= 2`,
  `offrole * estreno`, `offrole * (1 - share_lane)`, `JG_offrole * (logn_champ < 1)`, y
  perfil smurf/baja cobertura (`baja_cob * wr_acum`, `baja_cob * margen_medio`).
* `LADO_LOBBY`: nivel/ritmo promedio del lobby `(Azul + Rojo)/2` (permite ajustar el sesgo de
  lado Azul —que cae a 47,3 % en GM/Chal— e interacción con la diferencia de habilidad).

Todo es causal y se mide con `extra_post=False` sobre la referencia `MC + ML`.
"""
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.brainstorm_base import ajustar_predecir, cargar, comparar, evaluar, mostrar  # noqa: E402


def construir_bloques(D: dict, MC_raw: np.ndarray, ML_raw: np.ndarray) -> dict[str, np.ndarray]:
    NM = D["NM"]
    Xj, cols = D["Xj"], D["cols"]

    idx_acum = [i for i, c in enumerate(cols) if c.startswith("acum_")]
    cidx = {c: i for i, c in enumerate(cols)}

    marg_p = Xj[:, :, :, idx_acum].mean(axis=3)          # (NM, 2, 5)
    mc_p = MC_raw.mean(axis=3)                            # (NM, 2, 5)
    ml_p = ML_raw.mean(axis=3)                            # (NM, 2, 5)
    wr_p = Xj[:, :, :, cidx["wr_acum"]]
    wrc_p = Xj[:, :, :, cidx["wr_champ"]]
    lnc_p = Xj[:, :, :, cidx["logn_champ"]]
    dg15_p = Xj[:, :, :, cidx["acum_d_gold_15"]]

    def agg_equipo(arr: np.ndarray) -> np.ndarray:
        return np.c_[
            arr[:, 0].max(1) - arr[:, 1].max(1),
            arr[:, 0].min(1) - arr[:, 1].min(1),
            arr[:, 0].std(1) - arr[:, 1].std(1),
        ].astype(np.float32)

    AGREGADOS = np.hstack([agg_equipo(a) for a in (marg_p, mc_p, ml_p, wr_p, wrc_p, lnc_p, dg15_p)])

    # Autofill compuesto & Smurf / baja cobertura en ascenso (Idea 7)
    off = Xj[:, :, :, cidx["offrole"]]
    est = Xj[:, :, :, cidx["estreno"]]
    shl = Xj[:, :, :, cidx["share_lane"]]
    lnt = Xj[:, :, :, cidx["logn_tot"]]
    dg10 = Xj[:, :, :, cidx["acum_d_gold_10"]]

    def diff_sum(v: np.ndarray) -> np.ndarray:
        return (v[:, 0].sum(1) - v[:, 1].sum(1)).astype(np.float32)

    doble_off = ((off[:, 0].sum(1) >= 2).astype(np.float32) - (off[:, 1].sum(1) >= 2).astype(np.float32))
    off_est = diff_sum(off * est)
    off_rare = diff_sum(off * (1.0 - shl))
    jg_off_est = ((off[:, 0, 1] * (lnc_p[:, 0, 1] < 1.0)).astype(np.float32) -
                  (off[:, 1, 1] * (lnc_p[:, 1, 1] < 1.0)).astype(np.float32))

    # Cuentas con poca historia en BD (log1p(15) ~ 2.77) pero winrate / margen positivo
    baja_cob = np.maximum(0.0, 2.77 - lnt)
    smurf_wr = diff_sum(baja_cob * wr_p)
    smurf_mg = diff_sum(baja_cob * marg_p)
    smurf_dg = diff_sum(baja_cob * dg10)
    has_foto = D["J"][:, :, :, 6]  # 0 antes de 04/09 o si falta foto; usar post solo en LADO si aplica

    AUTOFILL_SMURF = np.c_[doble_off, off_est, off_rare, jg_off_est, smurf_wr, smurf_mg, smurf_dg].astype(np.float32)

    # Nivel/ritmo del lobby (Azul + Rojo)/2 para capturar sesgo Blue WR en GM/Chal
    lob_marg = (marg_p[:, 0].mean(1) + marg_p[:, 1].mean(1)) * 0.5
    lob_lnt = (lnt[:, 0].mean(1) + lnt[:, 1].mean(1)) * 0.5
    lob_aggr = (Xj[:, 0, :, cidx["perf_early_aggression_10"]].mean(1) +
                Xj[:, 1, :, cidx["perf_early_aggression_10"]].mean(1)) * 0.5
    diff_marg = marg_p[:, 0].mean(1) - marg_p[:, 1].mean(1)
    LADO_LOBBY = np.c_[
        lob_marg, lob_lnt, lob_aggr, lob_marg * diff_marg, lob_lnt * diff_marg
    ].astype(np.float32)

    # También versión de smurf post-t1 sin foto: (1 - tiene_foto) * (wr_acum y margen) cuando post
    # (sin foto es indicador en mapa, esto es interacción causal desde t1)
    sin_f = 1.0 - has_foto
    SMURF_POST = np.c_[
        diff_sum(sin_f * wr_p),
        diff_sum(sin_f * marg_p),
        diff_sum(sin_f * dg10),
    ].astype(np.float32)

    return {
        "AGREGADOS": AGREGADOS,
        "AUTOFILL_SMURF": AUTOFILL_SMURF,
        "LADO_LOBBY": LADO_LOBBY,
        "SMURF_POST": SMURF_POST,
    }


def main():
    D = cargar()
    ms = D["master"]
    C = np.load(REPO / "data" / "analysis" / "brainstorm" / "margen_grupos.npz")
    MC_raw, ML_raw = C["MC"], C["ML"]
    MC_ML = np.c_[
        (MC_raw[:, 0] - MC_raw[:, 1]).reshape(D["NM"], -1),
        (ML_raw[:, 0] - ML_raw[:, 1]).reshape(D["NM"], -1),
    ].astype(np.float32)

    BL = construir_bloques(D, MC_raw, ML_raw)

    from scipy import sparse
    from sklearn.linear_model import LogisticRegression

    def solo_m(extra_pre=None, extra_post=None):
        def pred(tr_raw, te):
            tr = tr_raw[ms[tr_raw]]
            if extra_post is None:
                tot = MC_ML if extra_pre is None else np.c_[MC_ML, extra_pre].astype(np.float32)
                return ajustar_predecir(D, tr, te, extra=tot, extra_post=False)
            Xd, post, mapa = D["Xd"], D["post"], D["mapa"]
            Xtr_p = np.c_[Xd[tr], MC_ML[tr]] if extra_pre is None else np.c_[Xd[tr], MC_ML[tr], extra_pre[tr]]
            Xte_p = np.c_[Xd[te], MC_ML[te]] if extra_pre is None else np.c_[Xd[te], MC_ML[te], extra_pre[te]]
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
        "ref: solo Master+ + MC/ML": solo_m(),
        "+ agregados equipo (max/min/std)": solo_m(BL["AGREGADOS"]),
        "+ autofill compuesto + smurf pre": solo_m(BL["AUTOFILL_SMURF"]),
        "+ lado × nivel/ritmo del lobby": solo_m(BL["LADO_LOBBY"]),
        "+ smurf sin-foto post": solo_m(None, BL["SMURF_POST"]),
        "+ agregados + autofill/smurf + sin-foto": solo_m(np.c_[BL["AGREGADOS"], BL["AUTOFILL_SMURF"]], BL["SMURF_POST"]),
        "+ todas (agg + autofill + lado + sin-foto)": solo_m(np.c_[BL["AGREGADOS"], BL["AUTOFILL_SMURF"], BL["LADO_LOBBY"]], BL["SMURF_POST"]),
    }
    P = dict(Parallel(n_jobs=1, backend="threading")(delayed(lambda n: (n, evaluar(D, VAR[n])))(n) for n in VAR))
    mostrar(comparar(D, P, "ref: solo Master+ + MC/ML"))


if __name__ == "__main__":
    main()
