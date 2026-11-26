"""UMAP del estado al minuto 15, de las partidas antes de jugarse y de los jugadores
(recomendación del profesor, 2026-11-17; PROJECT_SUMMARY.md §9.S.58).

UMAP no es una prueba: en 2D las distancias y los tamaños no significan nada y el dibujo depende de la
semilla, de `n_neighbors` y de `min_dist`. Por eso cada vista: (1) se dibuja con varias semillas y con PCA
al lado; (2) busca grupos con HDBSCAN sobre un UMAP de 10 dimensiones, no sobre el de 2; (3) valida todo
grupo fuera de donde se descubrió. Se descubre en los sub-bloques A2-A4 Master+ (o en el ajuste) y se
confirma en V Master+ con la regla de la noche: IC pareado al sumarlo al modelo, réplica en los bloques y
placebo (los mismos grupos asignados al azar).

Vistas:
* `vivo`: estado al minuto 15 orientado al equipo que va adelante en oro: Δoro y Δxp por línea, Δkills,
  sorpresa por línea (oro real − el que predecía el modelo por línea antes de jugar) y el logit previo del
  escalón 9. Color: si ganó el que iba adelante, la P del modelo en vivo y el residuo. Pregunta: ¿hay una
  región de partidas que siguen abiertas a pesar del oro, y el modelo en vivo la ve?
* `partidas`: contexto simétrico previo de la idea 45 (259 variables; no dice quién gana, describe la
  partida). Color: confianza y acierto del escalón 9, tier. Pregunta: ¿hay un tipo de partida donde el
  escalón 9 acierte más o menos de lo que promete?
* `jugadores`: las 15 variables de estilo de §9.S.53 (≥ 50 partidas), centradas por rol. Color: rol, tier,
  winrate, clase del campeón principal. HDBSCAN por rol contra K-Means con el mismo k.

    python scripts/analysis/umap_vistas.py --vista vivo          # ~5 min
    python scripts/analysis/umap_vistas.py --vista partidas      # ~15 min
    python scripts/analysis/umap_vistas.py --vista jugadores     # ~3 min

Salidas en `data/analysis/umap/<vista>/`: `emb.npz`, `grupos.csv`, `prueba.csv`, `*.png`, `resumen.json`.
Solo lectura de `data/`; el test no se toca.
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import adjusted_rand_score, roc_auc_score, silhouette_score  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
BR = REPO / "data" / "analysis" / "brainstorm"
SAL = REPO / "data" / "analysis" / "umap"
from src.ganador import rutas  # noqa: E402
T0 = time.time()
log = lambda s: print(f"[{time.time() - T0:6.0f}s] {s}", flush=True)  # noqa: E731
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
lg = lambda p: np.log(p / (1 - p))  # noqa: E731
SEMILLAS = (0, 1, 2)


# ------------------------------------------------------------------ utilidades comunes
def umap_ajustar(X_fit, otros, dim, semilla, nn=30, md=0.1):
    import umap
    r = umap.UMAP(n_components=dim, n_neighbors=nn, min_dist=md if dim == 2 else 0.0, random_state=semilla,
                  low_memory=True)
    E = r.fit_transform(X_fit).astype(np.float32)
    return E, [r.transform(X).astype(np.float32) for X in otros]


def hdbscan_grupos(E_fit, otros, min_tam, min_muestras):
    import hdbscan
    h = hdbscan.HDBSCAN(min_cluster_size=min_tam, min_samples=min_muestras, prediction_data=True).fit(E_fit)
    return h.labels_, [hdbscan.approximate_predict(h, E)[0] for E in otros]


def tabla_grupos(g, p, y, etiqueta):
    """Por grupo: frecuencia, P media, tasa observada, residuo medio (y − p) con su z bajo calibración,
    acierto contra esperado (exceso) y AUC. `p` e `y` ya están en el marco que interesa (líder / favorito)."""
    filas = []
    for k in np.unique(g):
        m = g == k
        pk, yk = p[m], y[m]
        se = np.sqrt((pk * (1 - pk)).sum()) / m.sum()
        ac, esp = ((pk > 0.5) == yk).mean(), np.maximum(pk, 1 - pk).mean()
        auc = roc_auc_score(yk, pk) if (m.sum() > 50 and 0 < yk.mean() < 1) else np.nan
        filas.append({"conjunto": etiqueta, "grupo": int(k), "partidas": int(m.sum()), "fraccion": m.mean(),
                      "p_media": pk.mean(), "observado": yk.mean(), "residuo": (yk - pk).mean(),
                      "z": (yk - pk).mean() / se, "acierto": ac, "esperado": esp, "exceso": ac - esp, "AUC": auc})
    return pd.DataFrame(filas)


def onehot(g, k):
    G = np.zeros((len(g), k), np.float32)
    ok = g >= 0
    G[np.flatnonzero(ok), g[ok]] = 1.0
    return G


def comparar_v(yv, pv, p_ref, B=500, semilla=1):
    rng = np.random.default_rng(semilla)
    idx = rng.integers(0, len(yv), size=(B, len(yv)))
    d = np.array([roc_auc_score(yv[s], pv[s]) - roc_auc_score(yv[s], p_ref[s]) for s in idx])
    return {"AUC V": roc_auc_score(yv, pv), "acierto V": ((pv > 0.5) == yv).mean(),
            "Δ V": roc_auc_score(yv, pv) - roc_auc_score(yv, p_ref),
            "Δ V lo": np.percentile(d, 2.5), "Δ V hi": np.percentile(d, 97.5)}


def apilar(F_tr, y_tr, F_te, C=1.0):
    mu, sd = F_tr.mean(0), F_tr.std(0) + 1e-9
    lr = LogisticRegression(C=C, max_iter=3000).fit((F_tr - mu) / sd, y_tr)
    return lr.predict_proba((F_te - mu) / sd)[:, 1]


def prueba_grupos(variantes, y, bl, A, V, C=1.0):
    """`variantes`: nombre → matriz F (todas las filas). La primera es la referencia. Bloques: cada bloque se
    predice con los otros dos (cruzado, en las dos direcciones del tiempo: es solo el conjunto de
    descubrimiento); V se predice con los tres bloques, todos anteriores, y es donde se decide. Tabla con IC en V."""
    ref = None
    filas = []
    for nombre, F in variantes.items():
        po = np.full(len(y), np.nan)
        for b in (1, 2, 3):
            tr, te = A[bl[A] != b], A[bl[A] == b]
            po[te] = apilar(F[tr], y[tr], F[te], C)
        pv = apilar(F[A], y[A], F[V], C)
        f = {"variante": nombre}
        for b in (1, 2, 3):
            m = A[bl[A] == b]
            f[f"AUC A{b + 1}"] = roc_auc_score(y[m], po[m])
        if ref is None:
            ref = (po, pv)
        dif = [roc_auc_score(y[A[bl[A] == b]], po[A[bl[A] == b]]) -
               roc_auc_score(y[A[bl[A] == b]], ref[0][A[bl[A] == b]]) for b in (1, 2, 3)]
        f["Δ bloques"] = float(np.mean(dif))
        f["bloques que mejora"] = int(sum(d > 0 for d in dif))
        f.update(comparar_v(y[V], pv, ref[1]))
        filas.append(f)
    return pd.DataFrame(filas)


def dibujar(ax, E, c, titulo, tipo="hex", cmap="coolwarm", vmin=None, vmax=None, max_puntos=20000, semilla=0):
    if tipo == "hex":
        hb = ax.hexbin(E[:, 0], E[:, 1], C=c, reduce_C_function=np.mean, gridsize=32, mincnt=6, cmap=cmap,
                       vmin=vmin, vmax=vmax, linewidths=0)
        plt.colorbar(hb, ax=ax, fraction=0.046, pad=0.02)
    else:   # categorías
        rng = np.random.default_rng(semilla)
        s = rng.permutation(len(E))[:max_puntos]
        cats = pd.Series(c[s]).astype(str)
        for i, k in enumerate(sorted(cats.unique())):
            m = (cats == k).to_numpy()
            ax.scatter(E[s][m, 0], E[s][m, 1], s=2, alpha=0.5, label=k, color=plt.cm.tab10(i % 10))
        ax.legend(markerscale=5, fontsize=7, loc="best", frameon=False)
    ax.set_title(titulo, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def guardar_fig(fig, ruta):
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)
    log(f"  figura {ruta.name}")


def cargar_previo_esc9(D):
    """Logit previo del escalón 9: hacia adelante en A2-A4 (po_ref9) y modelo previo al 16/09 en V (pv_ref9)."""
    P = np.load(BR / "pred_g28_esc9.npz")
    p = np.asarray(P["po_ref9"], np.float64).copy()
    V = np.flatnonzero(D["val"])
    p[V] = P["pv_ref9"]
    return np.clip(p, 1e-4, 1 - 1e-4)


# ------------------------------------------------------------------ vista: estado al minuto 15
def vista_vivo(args):
    from scripts.analysis.brainstorm_base import cargar
    out = SAL / "vivo"
    out.mkdir(parents=True, exist_ok=True)
    D = cargar()
    y, NM, Z, bl, ms = D["y"], D["NM"], D["Z"], D["bloque"], D["master"]
    p0 = cargar_previo_esc9(D)
    L = lg(p0)
    df = pd.read_csv(REPO / "data" / "splits" / "train" / "full_dataset.csv",
                     usecols=["match_id", "game_creation", "game_duration", "team_id", "lane", "win", "gold_15", "xp_15"])
    df = df[df.lane.isin(["BOTTOM", "JUNGLE", "MIDDLE", "TOP", "UTILITY"])]
    df = df[df.groupby(["match_id", "team_id"]).lane.transform("nunique") == 5]
    df = df[df.groupby("match_id").lane.transform("size") == 10]
    df = df[df.groupby("match_id").win.transform("sum") == 5]
    df = df.sort_values(["game_creation", "match_id", "team_id", "lane"]).reset_index(drop=True)
    assert (df.match_id.to_numpy()[::10] == Z["mids"]).all()
    lane = lambda c: (lambda a: (a[:, 0] - a[:, 1]) / 1000.0)(df[c].to_numpy(float).reshape(NM, 2, 5))  # noqa: E731
    g, x = lane("gold_15"), lane("xp_15")
    # solo partidas vivas al minuto 15: si terminó antes, el "frame 15" es el final (Nexo caído) y se filtra
    # el resultado (§3.F.4, `mask_finished_matches`; revisión de Antigravity, 2026-11-17)
    viva15 = df.game_duration.to_numpy(float)[::10] > 900
    del df
    k15 = Z["D_d_kills_15"].astype(float)
    hgbl = np.load(BR / "gemini_hgb_lineas.npz")["HGB"][:, :5]
    ex = np.load(BR / "hgb_lineas_extra.npz")["xp_15"]
    sorpresa = g - hgbl
    # modelo en vivo de referencia al 15 (brainstorm_40, "líneas + esperado completo", previo = escalón 9)
    F_vivo = np.c_[L, g, x, k15, sorpresa, ex]
    ok = np.isfinite(F_vivo).all(1) & np.isfinite(L) & viva15
    log(f"partidas que terminaron antes del minuto 15 (excluidas): {(~viva15).sum():,}")
    s = np.sign(g.sum(1))
    ok &= s != 0
    A = np.flatnonzero((bl >= 1) & ms & ok)
    V = np.flatnonzero(D["val"] & ok)
    log(f"minuto 15: bloques A2-A4 Master+ {len(A):,}, V Master+ {len(V):,}")
    p_vivo = np.full(NM, np.nan)
    for b in (1, 2, 3):
        tr, te = A[bl[A] != b], A[bl[A] == b]
        p_vivo[te] = apilar(F_vivo[tr], y[tr], F_vivo[te])
    p_vivo[V] = apilar(F_vivo[A], y[A], F_vivo[V])
    log(f"modelo en vivo (ref.): V AUC {roc_auc_score(y[V], p_vivo[V]):.4f} / "
        f"{((p_vivo[V] > .5) == y[V]).mean():.1%}; bloques (cruzado) {roc_auc_score(y[A], p_vivo[A]):.4f}")

    # estado orientado al líder en oro
    nombres = ([f"oro_{l}" for l in ("bot", "jg", "mid", "top", "sup")] + [f"xp_{l}" for l in ("bot", "jg", "mid", "top", "sup")]
               + ["kills"] + [f"sorpresa_{l}" for l in ("bot", "jg", "mid", "top", "sup")] + ["previo"])
    S = s[:, None] * np.c_[g, x, k15, sorpresa, L]
    mu, sd = S[A].mean(0), S[A].std(0) + 1e-9
    Sz = ((S - mu) / sd).astype(np.float32)
    yl = np.where(s > 0, y, 1 - y).astype(int)               # ganó el que iba adelante en oro
    pl = np.where(s > 0, p_vivo, 1 - p_vivo)                 # P del modelo en vivo de que gane el líder
    log(f"el líder en oro gana: bloques {yl[A].mean():.1%}, V {yl[V].mean():.1%}")

    E10, (E10v,) = umap_ajustar(Sz[A], [Sz[V]], 10, 0, nn=args.vecinos)
    log("UMAP 10D listo")
    gA, (gV,) = hdbscan_grupos(E10, [E10v], args.min_grupo, args.min_muestras)
    k = int(gA.max()) + 1
    log(f"HDBSCAN: {k} grupos, ruido {np.mean(gA < 0):.1%} en bloques / {np.mean(gV < 0):.1%} en V")
    T = pd.concat([tabla_grupos(gA, pl[A], yl[A], "bloques"), tabla_grupos(gV, pl[V], yl[V], "V")])
    perfiles = pd.DataFrame(Sz[A]).groupby(gA).mean()
    perfiles.columns = nombres
    T.to_csv(out / "grupos.csv", index=False)
    perfiles.to_csv(out / "perfiles.csv")
    print(T.round(4).to_string(index=False))

    # ¿sumar el grupo al modelo en vivo? (marco azul: s × grupo es antisimétrico)
    G = onehot(np.r_[gA, gV][np.argsort(np.r_[A, V])], k)    # en el orden de np.sort(r_[A, V])
    filas = np.sort(np.r_[A, V])
    Gt = np.zeros((NM, k), np.float32)
    Gt[filas] = G
    rng = np.random.default_rng(7)
    Gp = np.zeros_like(Gt)
    Gp[filas] = G[rng.permutation(len(filas))]
    lv = np.where(np.isfinite(p_vivo), lg(np.clip(np.nan_to_num(p_vivo, nan=0.5), 1e-4, 1 - 1e-4)), 0.0)
    var = {"en vivo (recalibrado)": lv[:, None],
           "+ líder × grupo": np.c_[lv, s[:, None] * Gt],
           "+ líder × grupo + logit × grupo": np.c_[lv, s[:, None] * Gt, lv[:, None] * Gt],
           "placebo: grupos al azar": np.c_[lv, s[:, None] * Gp, lv[:, None] * Gp]}
    P = prueba_grupos(var, y, bl, A, V)
    P.to_csv(out / "prueba.csv", index=False)
    print(P.round(4).to_string(index=False))

    # figuras: UMAP 2D (3 semillas) y PCA
    E2 = {}
    for sem in SEMILLAS:
        E2[sem], (E2v,) = umap_ajustar(Sz[A], [Sz[V]], 2, sem, nn=args.vecinos)
        E2[f"v{sem}"] = E2v
    pca = PCA(2, random_state=0).fit(Sz[A])
    Ep, Epv = pca.transform(Sz[A]), pca.transform(Sz[V])
    res = yl - pl
    fig, ax = plt.subplots(2, 4, figsize=(19, 9))
    for fila, (E, Ev, nom) in enumerate(((E2[0], E2["v0"], "UMAP (semilla 0)"), (Ep, Epv, "PCA"))):
        dibujar(ax[fila, 0], E, np.abs(g[A].sum(1)), f"{nom}: |Δoro| del equipo @15 (miles)", cmap="viridis")
        dibujar(ax[fila, 1], E, yl[A], f"{nom}: ganó el líder en oro (bloques)", cmap="RdYlGn", vmin=0.3, vmax=1)
        dibujar(ax[fila, 2], E, res[A], f"{nom}: residuo del modelo en vivo (bloques)", vmin=-0.12, vmax=0.12)
        dibujar(ax[fila, 3], Ev, res[V], f"{nom}: residuo del modelo en vivo (V Master+)", vmin=-0.12, vmax=0.12)
    guardar_fig(fig, out / "umap_vivo.png")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for i, sem in enumerate(SEMILLAS):
        dibujar(ax[i], E2[sem], yl[A], f"UMAP semilla {sem}: ganó el líder", cmap="RdYlGn", vmin=0.3, vmax=1)
    guardar_fig(fig, out / "umap_vivo_semillas.png")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
    dibujar(ax[0], E2[0], gA, "UMAP 2D coloreado por HDBSCAN (10D)", tipo="cat")
    dibujar(ax[1], E2[0], S[A][:, nombres.index("previo")], "UMAP: logit previo del líder", vmin=-1, vmax=1)
    guardar_fig(fig, out / "umap_vivo_grupos.png")
    np.savez_compressed(out / "emb.npz", A=A, V=V, E10=E10, E10v=E10v, gA=gA, gV=gV, E2=E2[0], E2v=E2["v0"],
                        p_vivo=p_vivo, s=s)
    resumen = {"bloques": len(A), "V": len(V), "grupos": k, "ruido_bloques": float(np.mean(gA < 0)),
               "lider_gana_V": float(yl[V].mean()), "prueba": P.to_dict(orient="records")}
    (out / "resumen.json").write_text(json.dumps(resumen, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    log("vista vivo lista")


# ------------------------------------------------------------------ vista: partidas antes de jugarse
def vista_partidas(args):
    from scripts.analysis.brainstorm_45_dificultad import contexto
    from scripts.analysis.brainstorm_base import cargar
    out = SAL / "partidas"
    out.mkdir(parents=True, exist_ok=True)
    D = cargar()
    y, NM, bl, ms = D["y"], D["NM"], D["bloque"], D["master"]
    p = cargar_previo_esc9(D)
    L = lg(p)
    C, nombres = contexto(D)
    C = C.astype(np.float32)
    A = np.flatnonzero((bl >= 1) & ms & np.isfinite(L))
    V = np.flatnonzero(D["val"])
    fit_pool = np.setdiff1d(np.flatnonzero(D["fit"] & ms & ~D["diam_v"]), A)
    rng = np.random.default_rng(0)
    Fi = np.sort(rng.choice(fit_pool, size=min(args.n_ajuste, len(fit_pool)), replace=False))
    log(f"contexto {C.shape[1]} variables; UMAP ajustado con {len(Fi):,} partidas Master+ previas a A2 o fuera de los "
        f"bloques; bloques {len(A):,}, V {len(V):,}")
    E10, (E10a, E10v) = umap_ajustar(C[Fi], [C[A], C[V]], 10, 0, nn=args.vecinos)
    log("UMAP 10D listo")
    gF, (gA, gV) = hdbscan_grupos(E10, [E10a, E10v], args.min_grupo, args.min_muestras)
    k = int(gF.max()) + 1
    log(f"HDBSCAN: {k} grupos, ruido {np.mean(gF < 0):.1%} (ajuste) / {np.mean(gA < 0):.1%} (bloques) / {np.mean(gV < 0):.1%} (V)")
    fav_y = np.where(p >= 0.5, y, 1 - y).astype(int)
    fav_p = np.maximum(p, 1 - p)
    T = pd.concat([tabla_grupos(gA, fav_p[A], fav_y[A], "bloques"), tabla_grupos(gV, fav_p[V], fav_y[V], "V")])
    perf = pd.DataFrame(C[Fi]).groupby(gF).mean()
    perf.columns = nombres
    top = {int(gr): ", ".join(f"{c} ({fila[c]:+.2f})" for c in fila.abs().sort_values(ascending=False).index[:6])
           for gr, fila in perf.iterrows()}
    T["rasgos"] = T.grupo.map(top)
    T.to_csv(out / "grupos.csv", index=False)
    print(T.drop(columns="rasgos").round(4).to_string(index=False))
    for gr, r in top.items():
        print(f"  grupo {gr}: {r}")

    # recalibración por grupo (el grupo es simétrico: solo puede escalar la confianza, logit × grupo)
    filas = np.r_[A, V]
    Gt = np.zeros((NM, k), np.float32)
    Gt[filas] = onehot(np.r_[gA, gV], k)
    Gp = np.zeros_like(Gt)
    Gp[filas] = Gt[filas][np.random.default_rng(7).permutation(len(filas))]
    var = {"escalón 9 (recalibrado)": L[:, None],
           "+ logit × grupo": np.c_[L, L[:, None] * Gt],
           "placebo: grupos al azar": np.c_[L, L[:, None] * Gp]}
    P = prueba_grupos(var, y, bl, A, V)
    P.to_csv(out / "prueba.csv", index=False)
    print(P.round(4).to_string(index=False))

    E2 = {}
    for sem in SEMILLAS:
        E2[sem], (E2a, E2v) = umap_ajustar(C[Fi], [C[A], C[V]], 2, sem, nn=args.vecinos)
        E2[f"a{sem}"], E2[f"v{sem}"] = E2a, E2v
    pca = PCA(2, random_state=0).fit(C[Fi])
    Epa, Epv = pca.transform(C[A]), pca.transform(C[V])
    acierto = (fav_y == 1).astype(float)
    fig, ax = plt.subplots(2, 4, figsize=(19, 9))
    for fila, (Ea, Ev, nom) in enumerate(((E2["a0"], E2["v0"], "UMAP"), (Epa, Epv, "PCA"))):
        dibujar(ax[fila, 0], Ea, D["tier_lobby"][A], f"{nom}: tier medio del lobby", cmap="viridis")
        dibujar(ax[fila, 1], Ea, np.abs(L[A]), f"{nom}: |logit| del escalón 9 (confianza)", cmap="viridis")
        dibujar(ax[fila, 2], Ea, acierto[A] - fav_p[A], f"{nom}: acierto − esperado (bloques)", vmin=-0.1, vmax=0.1)
        dibujar(ax[fila, 3], Ev, acierto[V] - fav_p[V], f"{nom}: acierto − esperado (V Master+)", vmin=-0.1, vmax=0.1)
    guardar_fig(fig, out / "umap_partidas.png")
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.6))
    for i, sem in enumerate(SEMILLAS):
        dibujar(ax[i], E2[f"a{sem}"], np.abs(L[A]), f"UMAP semilla {sem}: |logit|", cmap="viridis")
    dibujar(ax[3], E2[0], gF, "UMAP 2D (ajuste) por HDBSCAN (10D)", tipo="cat")
    guardar_fig(fig, out / "umap_partidas_semillas.png")
    np.savez_compressed(out / "emb.npz", Fi=Fi, A=A, V=V, gF=gF, gA=gA, gV=gV, E2=E2[0], E2a=E2["a0"], E2v=E2["v0"])
    resumen = {"ajuste": len(Fi), "bloques": len(A), "V": len(V), "grupos": k, "prueba": P.to_dict(orient="records")}
    (out / "resumen.json").write_text(json.dumps(resumen, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    log("vista partidas lista")


# ------------------------------------------------------------------ vista: jugadores
F_JUG = ["role_focus_pct", "pool_dispersion", "avg_cs_at_15", "avg_total_cs", "avg_gold_share_pct",
         "avg_damage_share_pct", "avg_damage_taken", "avg_early_aggression_10", "avg_kill_participation_pct",
         "avg_vision_score", "avg_gold_diff_at_15", "avg_cs_diff_at_15", "avg_cs_diff_end",
         "avg_damage_diff_end", "avg_gold_pct_diff_delta"]
ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]


def vista_jugadores(args):
    out = SAL / "jugadores"
    out.mkdir(parents=True, exist_ok=True)
    jug = pd.read_csv(args.players)
    sig = pd.read_csv(args.firmas, usecols=["player_id", "pool_dispersion"])
    m = jug[jug.games_recorded >= 50].merge(sig, on="player_id", how="inner").dropna(subset=F_JUG + ["main_lane"])
    m = m[m.main_lane.isin(ROLES)].reset_index(drop=True)
    # clase del campeón principal (más jugado en el train) según los tags de Data Dragon
    fd = pd.read_csv(REPO / "data" / "splits" / "train" / "full_dataset.csv", usecols=["player_id", "champion_id"])
    princ = fd.groupby(["player_id", "champion_id"]).size().reset_index(name="n").sort_values("n").drop_duplicates(
        "player_id", keep="last")
    dd = pd.read_csv(REPO / "data" / "ddragon" / "16.17.1" / "champions.csv", usecols=["champion_id", "tags", "name"])
    dd["clase"] = dd.tags.str.split("|").str[0]
    m = m.merge(princ[["player_id", "champion_id"]], on="player_id", how="left").merge(dd, on="champion_id", how="left")
    m["clase"] = m.clase.fillna("?")
    log(f"jugadores con ≥ 50 partidas: {len(m):,}")
    Xc = m[F_JUG].groupby(m.main_lane).transform(lambda s_: (s_ - s_.mean()) / s_.std()).to_numpy(np.float32)
    E2 = {sem: umap_ajustar(Xc, [], 2, sem, nn=args.vecinos)[0] for sem in SEMILLAS}
    Ep = PCA(2, random_state=0).fit_transform(Xc)
    tier = m.player_tier.fillna("?").to_numpy()
    fig, ax = plt.subplots(2, 4, figsize=(20, 9))
    for fila, (E, nom) in enumerate(((E2[0], "UMAP"), (Ep, "PCA"))):
        dibujar(ax[fila, 0], E, m.main_lane.to_numpy(), f"{nom}: rol principal (variables centradas por rol)", tipo="cat")
        dibujar(ax[fila, 1], E, tier, f"{nom}: tier", tipo="cat")
        dibujar(ax[fila, 2], E, m.winrate_pct.to_numpy(), f"{nom}: winrate (%)", cmap="RdYlGn", vmin=45, vmax=58)
        dibujar(ax[fila, 3], E, m.clase.to_numpy(), f"{nom}: clase del campeón principal", tipo="cat")
    guardar_fig(fig, out / "umap_jugadores.png")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for i, sem in enumerate(SEMILLAS):
        dibujar(ax[i], E2[sem], m.clase.to_numpy(), f"UMAP semilla {sem}: clase", tipo="cat")
    guardar_fig(fig, out / "umap_jugadores_semillas.png")

    filas, perfiles = [], []
    fig, ax = plt.subplots(2, 5, figsize=(24, 9))
    for i, rol in enumerate(ROLES):
        mr = (m.main_lane == rol).to_numpy()
        X = Xc[mr]
        E10 = umap_ajustar(X, [], 10, 0, nn=args.vecinos)[0]
        e2 = umap_ajustar(X, [], 2, 0, nn=args.vecinos)[0]
        mt = max(30, int(0.03 * len(X)))
        g, _ = hdbscan_grupos(E10, [], mt, max(10, mt // 3))
        k = int(g.max()) + 1
        f = {"rol": rol, "jugadores": int(mr.sum()), "grupos_hdbscan": k, "ruido": float(np.mean(g < 0))}
        if k >= 2:
            ok = g >= 0
            km = KMeans(k, n_init=10, random_state=0).fit(X)
            f["ARI vs K-Means (sin ruido)"] = adjusted_rand_score(g[ok], km.labels_[ok])
            f["silhouette HDBSCAN (espacio original)"] = silhouette_score(X[ok], g[ok])
            # mismo subconjunto que HDBSCAN (sin su ruido), para que las dos siluetas se comparen (§9.S.77);
            # la de todos los jugadores queda aparte, y la de un K-Means ajustado solo sin el ruido también
            f["silhouette K-Means (sin ruido)"] = silhouette_score(X[ok], km.labels_[ok])
            f["silhouette K-Means (todos)"] = silhouette_score(X, km.labels_)
            f["silhouette K-Means reajustado sin ruido"] = silhouette_score(
                X[ok], KMeans(k, n_init=10, random_state=0).fit_predict(X[ok]))
            wr = m.winrate_pct.to_numpy()[mr]
            for gr in range(k):
                mg = g == gr
                z = pd.Series(X[mg].mean(0), index=F_JUG)
                perfiles.append({"rol": rol, "grupo": gr, "jugadores": int(mg.sum()), "winrate": wr[mg].mean(),
                                 "clase_top": pd.Series(m.clase.to_numpy()[mr][mg]).value_counts().index[0],
                                 "rasgos": ", ".join(f"{c} ({v:+.2f})" for c, v in
                                                     z.reindex(z.abs().sort_values(ascending=False).index).head(4).items())})
        filas.append(f)
        dibujar(ax[0, i], e2, g, f"{rol}: HDBSCAN en 10D ({k} grupos, ruido {np.mean(g < 0):.0%})", tipo="cat")
        dibujar(ax[1, i], e2, m.clase.to_numpy()[mr], f"{rol}: clase del campeón principal", tipo="cat")
        log(f"  {rol}: {f}")
    guardar_fig(fig, out / "umap_jugadores_por_rol.png")
    R = pd.DataFrame(filas)
    Pf = pd.DataFrame(perfiles)
    R.to_csv(out / "grupos.csv", index=False)
    Pf.to_csv(out / "perfiles.csv", index=False)
    print(R.round(3).to_string(index=False))
    print(Pf.round(2).to_string(index=False))
    (out / "resumen.json").write_text(json.dumps({"jugadores": len(m), "por_rol": filas}, indent=1, ensure_ascii=False,
                                                 default=float), encoding="utf-8")
    log("vista jugadores lista")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vista", choices=["vivo", "partidas", "jugadores", "todas"], default="vivo")
    ap.add_argument("--vecinos", type=int, default=30)
    ap.add_argument("--min-grupo", type=int, default=150)
    ap.add_argument("--min-muestras", type=int, default=30)
    ap.add_argument("--n-ajuste", type=int, default=30000)
    ap.add_argument("--players", default=str(rutas.PLAYERS_CSV))
    ap.add_argument("--firmas", default=str(rutas.FIRMAS_A_CSV))
    args = ap.parse_args()
    vistas = ["vivo", "partidas", "jugadores"] if args.vista == "todas" else [args.vista]
    for v in vistas:
        log(f"=== vista {v} ===")
        {"vivo": vista_vivo, "partidas": vista_partidas, "jugadores": vista_jugadores}[v](args)


if __name__ == "__main__":
    main()
