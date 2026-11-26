"""
Base común del brainstorming de Master+ (docs/BRAINSTORM_NOCHE.md; PROJECT_SUMMARY.md §9.S.52+).

`cargar()` devuelve un diccionario con todo lo necesario para probar ideas sobre el mejor modelo
de esta máquina (híbrido + nivel (mapa) + mapa, §9.S.50-51), con la partición vigente:

* V = ventana 16/09 → 27/09 con mediana de tier ≥ Master (`particion_v_master`);
* el modelo de V se entrena solo con partidas anteriores al 16/09 (`fi_v`);
* sub-bloques A2-A4 del tramo 04/09 → 16/09 (más las Diamond de la ventana en A4) para evaluar
  hacia adelante (`bloque_fit`, `EV`).

`ajustar_predecir(D, tr, te, extra=None, peso=None, C=0.05)` ajusta la logística del híbrido
(draft + 85 variables estandarizadas con las filas de entrenamiento) con columnas extra
opcionales (estandarizadas con las filas de entrenamiento posteriores a la foto 1 si
`extra_post=True`, o con todas si no) y pesos por fila.

`evaluar(D, predictor, nombre)` corre la variante en los sub-bloques y en V y devuelve AUC y
acierto. `comparar(D, P, ref)` arma la tabla con Δ e IC.

La primera llamada arma la caché `data/analysis/brainstorm/base.npz` (~1 min); después carga en
segundos.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.features.draft import draft_matrix_by_lane  # noqa: E402
from src.models.validacion_temporal import bloques_por_fecha, particion_v_master  # noqa: E402

CACHE = REPO / "data" / "analysis" / "brainstorm" / "base.npz"


def _armar():
    import scripts.analysis.ganador_mapa_lp as gm
    from src.features.lobby import prior_lobby_level
    Z = np.load(gm.RUTA_NPZ, allow_pickle=True)
    mids, Xj, champ, t, val0, y = Z["mids"], Z["Xj"], Z["champ"], Z["t"], Z["val"].astype(bool), Z["y"].astype(int)
    NM = len(mids)
    foto1, t1 = gm._cargar_foto1(gm.RUTA_DB)
    # sin la caché del mapa hay que calibrar los parámetros del paso con lp_vivo (antes se pasaba None y fallaba)
    params = None if gm.RUTA_CACHE.exists() else gm._calibrar_params(gm.RUTA_DB, t1, None)[0]
    cache = gm._cargar_o_calcular_mapa(mids, t, y, t1, foto1, params, rehacer=False)
    post = t >= t1
    fit_post = post & ~val0
    dpc, keys = cache["df_post_cron"], cache["df_npz_keys"].copy()
    en_fp = dpc.match_id.isin(set(mids[fit_post]))
    mu_f, mu_m = float(dpc.loc[en_fp, "lp_foto"].mean()), float(dpc.loc[en_fp, "lp_causal"].mean())
    nf = prior_lobby_level(dpc, "lp_foto", k=5, mu0=mu_f, window=10, min_otros=3)
    nm_ = prior_lobby_level(dpc, "lp_causal", k=5, mu0=mu_m, window=10, min_otros=3)
    F = dpc.copy()
    for p, n in (("foto", nf), ("mapa", nm_)):
        F[f"{p}_acum"], F[f"{p}_v10"] = n["nivel_acum"].to_numpy(float), n["nivel_v10"].to_numpy(float)
        F[f"{p}_n"] = np.log1p(n["n_nivel"].to_numpy(float))
    keys["_pos"] = np.arange(len(keys))
    M = keys.merge(F.drop(columns=["game_creation", "team_id", "lane"]), on=["match_id", "player_id"], how="left") \
            .sort_values("_pos", kind="mergesort").reset_index(drop=True)
    pr = np.repeat(post, 10)
    col = lambda c, mu=0.0: np.where(pr, M[c].fillna(mu).to_numpy(np.float32) - mu, 0.0)  # noqa: E731
    tiene = (M.lp_foto.notna() & pr).to_numpy(np.float32)
    lp_m = np.where(pr, np.where(M.metodo.to_numpy() == "causal_lobby", M.mapa_v10.fillna(mu_m),
                                 M.lp_causal.fillna(mu_m)).astype(np.float32) - mu_m, 0.0)
    dl = np.where(tiene > 0.5, lp_m - col("lp_foto", mu_f), 0.0)
    inc = M.lp_incert.to_numpy(np.float32)
    inc_mu = float(np.nanmean(inc[np.repeat(fit_post, 10)]))
    lp_inc = np.where(pr, np.nan_to_num(inc, nan=inc_mu) - inc_mu, 0.0)
    # por jugador (partidas, 2 equipos, 5 líneas): variables del mapa sin agregar, para ideas nuevas
    J = np.stack([col("mapa_acum", mu_m), col("mapa_v10", mu_m), col("mapa_n"), lp_m, dl, lp_inc, tiene],
                 axis=-1).reshape(NM, 2, 5, -1).astype(np.float32)
    lp_abs = np.where(pr, M.lp_causal.to_numpy(np.float32), np.nan).reshape(NM, 2, 5)
    np.savez_compressed(CACHE, J=J, J_cols=np.array(["niv_acum", "niv_v10", "niv_n", "lp", "dl_foto", "incert", "tiene_foto"]),
                        lp_abs=lp_abs, post=post, t1=t1, player=M.player_id.to_numpy().reshape(NM, 2, 5).astype(str))


def cargar():
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    if not CACHE.exists():
        _armar()
    B = np.load(CACHE, allow_pickle=True)
    Z = np.load(REPO / "data" / "analysis" / "embeddings_partidas" / "variables.npz", allow_pickle=True)
    Xj, champ, t, val0, y = Z["Xj"], Z["champ"], Z["t"], Z["val"].astype(bool), Z["y"].astype(int)
    NM = len(y)
    val, diam_v = particion_v_master(val0, Z["tnum"])
    fit = ~val
    J, post = B["J"], B["post"]
    eq = (J[:, 0] - J[:, 1]).mean(axis=1)
    nmv, lpm = eq[:, 1], eq[:, 3]
    mapa = np.c_[eq, nmv - lpm].astype(np.float32)          # = "+ niv(mapa) + mapa" de §9.S.46/50
    Xd = (Xj[:, 0] - Xj[:, 1]).reshape(NM, -1).astype(np.float32)
    DRAFT = draft_matrix_by_lane(champ)
    fi = np.flatnonzero(fit)
    ai = np.flatnonzero(post & ~val0)
    bloque = np.full(NM, -1)
    bloque[fi] = 0
    bloque[ai] = bloques_por_fecha(t[ai], 4)
    bloque[diam_v] = 3
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            tier_lobby = np.nanmean(Z["tnum"], axis=1)
    return dict(Z=Z, B=B, Xj=Xj, cols=list(Z["cols"]), y=y, t=t, NM=NM, val=val, val0=val0, diam_v=diam_v,
                fit=fit, fi=fi, fi_v=fi[~diam_v[fi]], post=post, Xd=Xd, DRAFT=DRAFT, mapa=mapa, J=J,
                bloque=bloque, EV=[1, 2, 3], tier_lobby=tier_lobby, master=particion_v_master(np.ones(NM, bool), Z["tnum"])[0])


def ajustar_predecir(D, tr, te, extra=None, extra_post=True, peso=None, C=0.05, con_mapa=True, con_draft=True, devolver_modelo=False):
    Xd, post = D["Xd"], D["post"]
    mu, sd = Xd[tr].mean(0), Xd[tr].std(0) + 1e-6

    def armar(f):
        P = ([D["DRAFT"][f]] if con_draft else []) + [sparse.csr_matrix((Xd[f] - mu) / sd)]
        return P

    Ptr, Pte = armar(tr), armar(te)
    # `extra` puede ser un array (se estandariza según `extra_post`) o una lista de (array, post: bool)
    if extra is None:
        bloques = []
    elif isinstance(extra, (list, tuple)):
        bloques = list(extra)
    else:
        bloques = [(extra, extra_post)]
    extras = ([(D["mapa"], True)] if con_mapa else []) + bloques
    for e, es_post in extras:
        ref = tr[post[tr]] if es_post else tr
        me, se = e[ref].mean(0), e[ref].std(0) + 1e-6
        f = lambda idx: (np.where(post[idx][:, None], (e[idx] - me) / se, 0) if es_post else (e[idx] - me) / se)  # noqa: E731
        Ptr.append(sparse.csr_matrix(f(tr).astype(np.float32)))
        Pte.append(sparse.csr_matrix(f(te).astype(np.float32)))
    lr = LogisticRegression(C=C, max_iter=4000).fit(sparse.hstack(Ptr).tocsr(), D["y"][tr], sample_weight=peso)
    p = lr.predict_proba(sparse.hstack(Pte).tocsr())[:, 1]
    return (p, lr) if devolver_modelo else p


def evaluar(D, predictor):
    """`predictor(tr, te) -> p`. Devuelve (p fuera de muestra por sub-bloque, p en V)."""
    po = np.full(D["NM"], np.nan)
    fi, bl = D["fi"], D["bloque"]
    for b in D["EV"]:
        tr = fi[bl[fi] < b]
        te = np.flatnonzero(bl == b)
        po[te] = predictor(tr, te)
    pv = predictor(D["fi_v"], np.flatnonzero(D["val"]))
    return po, pv


def comparar(D, P, ref, B=200, solo_master_bloques=True):
    """Tabla AUC/acierto por variante: sub-bloques (solo Master+ si se pide) y V, Δ vs `ref` con IC."""
    y, bl, val = D["y"], D["bloque"], D["val"]
    yv = y[val]
    rng = np.random.default_rng(1)
    BOOT = rng.integers(0, len(yv), size=(B, len(yv)))
    po_r, pv_r = P[ref]
    filas = []
    for n, (po, pv) in P.items():
        f = {"variante": n}
        dif_b, reps = [], []
        for b in D["EV"]:
            m = (bl == b) & (D["master"] if solo_master_bloques else True)
            f[f"AUC A{b + 1}"] = roc_auc_score(y[m], po[m])
            f[f"ac A{b + 1}"] = ((po[m] > 0.5) == y[m]).mean()
            dif_b.append(f[f"AUC A{b + 1}"] - roc_auc_score(y[m], po_r[m]))
            if n != ref:   # bootstrap pareado dentro del bloque (para el IC de la media de bloques)
                im = np.flatnonzero(m)
                bb = np.random.default_rng(2 + b).integers(0, len(im), size=(B // 2, len(im)))
                reps.append([roc_auc_score(y[im[s]], po[im[s]]) - roc_auc_score(y[im[s]], po_r[im[s]]) for s in bb])
        f["Δ bloques (media)"] = np.mean(dif_b)
        f["bloques que mejora"] = int(sum(d > 0 for d in dif_b))
        if reps:   # desde 04:35: IC de la media de bloques (la lección del placebo de ROL_WR)
            mb = np.mean(reps, axis=0)
            f["Δ bloq lo"], f["Δ bloq hi"] = np.percentile(mb, 2.5), np.percentile(mb, 97.5)
        else:
            f["Δ bloq lo"] = f["Δ bloq hi"] = 0.0
        f["AUC V"] = roc_auc_score(yv, pv)
        f["acierto V"] = ((pv > 0.5) == yv).mean()
        d = np.array([roc_auc_score(yv[s], pv[s]) - roc_auc_score(yv[s], pv_r[s]) for s in BOOT])
        f["Δ V"] = f["AUC V"] - roc_auc_score(yv, pv_r)
        f["Δ V lo"], f["Δ V hi"] = np.percentile(d, 2.5), np.percentile(d, 97.5)
        filas.append(f)
    return pd.DataFrame(filas)


def mostrar(T):
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 30)
    fmt = {c: (lambda x: f"{x:.1%}") for c in T.columns if c.startswith("ac")}
    out = T.copy()
    for c in out.columns:
        if c.startswith("ac"):
            out[c] = out[c].map(lambda x: f"{x:.1%}")
        elif c.startswith(("AUC", "Δ")) and out[c].dtype != object:
            out[c] = out[c].map(lambda x: f"{x:+.4f}" if c.startswith("Δ") else f"{x:.4f}")
    print(out.to_string(index=False))


def cargar_hgb_por_linea():
    """Carga el HGB y rechaza una caché anterior a cualquiera de sus insumos."""
    from src.ganador.rutas import ATRIB_CSV   # la misma ruta que usaba brainstorm_g_06_comp_poisson

    ruta = CACHE.with_name("gemini_hgb_lineas.npz")
    insumos = (
        REPO / "data" / "analysis" / "embeddings_partidas" / "variables.npz",
        CACHE.with_name("margen_grupos.npz"),
        CACHE.with_name("mmr_historico.npz"),
        CACHE.with_name("gemini_mpr.npz"),
        ATRIB_CSV,
    )
    fecha_hgb = ruta.stat().st_mtime_ns
    posteriores = [p.name for p in insumos if p.is_file() and p.stat().st_mtime_ns > fecha_hgb]
    if posteriores:
        raise ValueError(
            f"{ruta.name} es anterior a {', '.join(posteriores)}. "
            "Regenerar con python historial/escalon9/brainstorm_g_09_hgb_lineas_causal.py --rehacer --solo-cache."
        )
    with np.load(ruta) as datos:
        return datos["HGB"]


def extras_referencia(D, escalon=3, rol_wr=False):
    """Columnas extra de la referencia vigente de la noche (escalón 3, V 0,6663 / 62,0 %):
    margen por campeón y línea + nivel histórico (equipo) + W/L de la foto 1 dinámico.
    Uso: `ajustar_predecir(D, tr[D["master"][tr]], te, extra=np.c_[extras_referencia(D), NUEVO], extra_post=False)`.
    Necesita las cachés de historial/escalon9/ (brainstorm_05_jugador.py, brainstorm_07_mmr_historico.py...).
    El W/L de la foto 1 sale de `src.ganador` (`wl_foto1`, copia textual de `construir_extras` de
    brainstorm_g_01_foto1_wl.py; solo cambia `D["B"]["player"]` por `D["player"]`, §9.S.73 y §9.S.79)."""
    from src.ganador._constructores import wl_foto1
    NM = D["NM"]
    if "_ref" not in D:
        C = np.load(REPO / "data" / "analysis" / "brainstorm" / "margen_grupos.npz")
        mcml = np.c_[(C["MC"][:, 0] - C["MC"][:, 1]).reshape(NM, -1), (C["ML"][:, 0] - C["ML"][:, 1]).reshape(NM, -1)]
        A = np.load(REPO / "data" / "analysis" / "brainstorm" / "mmr_historico.npz")["A"]
        niv = (A[:, 0] - A[:, 1])[..., :3].mean(axis=1)
        foto1_din = wl_foto1({**D, "player": D["B"]["player"]})["FOTO1_DIN"]
        D["_ref"] = np.c_[mcml, niv, foto1_din].astype(np.float32)
    if escalon >= 4:   # + MATCHUPS de Gemini (verificado por Claude 03:25: +0,0011 [+0,0006; +0,0017], 3/3)
        mat = np.load(REPO / "data" / "analysis" / "brainstorm" / "gemini_mpr.npz")["MATCHUPS"]
        out = np.c_[D["_ref"], mat]
        if escalon >= 6:
            rol_wr = True
        if rol_wr:     # escalón 4b: + ROL_WR de Gemini (verificado por Claude 04:35: V +0,0009, Diamond-V +0,0014)
            out = np.c_[out, np.load(REPO / "data" / "analysis" / "brainstorm" / "gemini_nuevas_v2.npz")["ROL_WR"]]
        if escalon >= 6:   # + g_11 de Gemini sin CLUTCH_15 (verificado por Claude 04:45: V +0,0024 sobre 4b, bloques IC > 0)
            g = np.load(REPO / "data" / "analysis" / "brainstorm" / "gemini_g11.npz")
            out = np.c_[out, g["KALMAN_OUT"], g["KALMAN_MRG"], g["TEMPO_OBJ"], g["ECON_MID"]]
        if escalon >= 7:   # + modelo por línea @15 con HGB causal (g_09 de Gemini; verificado por Claude 08:15)
            out = np.c_[out, cargar_hgb_por_linea()]
        if escalon >= 9:   # + margen esperado por par de campeones (g_23 de Gemini; verificado por Claude 09:50)
            out = np.c_[out, np.load(REPO / "data" / "analysis" / "brainstorm" / "gemini_g23.npz")["MATCHUP_EXP_MARGIN"]]
        return out.astype(np.float32)
    return D["_ref"]
