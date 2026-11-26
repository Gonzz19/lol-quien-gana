r"""Modelo en vivo de quién gana a los minutos 10, 15 y 20: logística con el previo del escalón 9 (`PROJECT_SUMMARY.md` §9.S.78).

Nace de la variante 5b de `ingame_g_03_regimen_y_senales.py` (§9.S.60 punto 3; hoy en `historial/ensamble_en_vivo/`),
rehecha sin cachés de variantes descartadas y con el previo del escalón 9 de `src.ganador.modelo_previo`:

* **Etapa 1, el modelo que se presenta** (las ~92k partidas Master+ de los tramos 1-6): logística sobre el
  estado al minuto $m$ (las 8 estadísticas por línea del timeline, oro residual, niveles, trayectorias y la
  "sorpresa" contra lo esperado por línea; bloque A2 de §9.S.60) más el logit previo $L$ del escalón 9,
  calculado hacia adelante. $z = \operatorname{logit} \hat P_1(y \mid \text{estado}_m, L)$.
* **Etapa 2, variante descartada** (solo partidas posteriores a la foto 1, hacia adelante: A3 ← A2,
  A4 ← A2-A3, V ← A2-A4): $\operatorname{logit} P = a + b\,z + c\,L$. Le devuelve a $L$ el peso que tiene
  después de la foto (en la etapa 1 lo fijan sobre todo las ~78k partidas anteriores a la foto), pero no pasa
  la regla de aceptación de §9.S.47: suma +0,0005 / +0,0003 / +0,0001 en V y en los bloques A3-A4 y en
  Diamond-V el IC cruza el cero. Por eso el modelo principal (`MODELO_PRINCIPAL`) es la etapa 1 sola
  ("Decisión" en §9.S.78). La etapa 2 se sigue calculando y reportando como comparación, no se borró.

Tramos (`tramos`): el bloque 0 de `datos.armar` (todo lo anterior a A2) partido en 4 por fecha (0-3), A2 = 4,
A3 = 5, A4 = 6, V Master+ = 7 y Diamond de la ventana de V = 8. Toda pieza aprendida (previo, oro residual,
expectativas por línea, etapas 1 y 2) se ajusta para cada tramo solo con partidas Master+ de tramos
anteriores; para V y Diamond-V, con todo lo anterior a la ventana de V. El test no se carga nunca.

Todo `.npz` guarda `mids` y se verifica contra las partidas de `D` al cargarlo (nada se alinea por posición).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, roc_auc_score

from src.ganador import cache, modelo_previo, rutas
from src.models import ingame as IG
from src.models.ingame import MINUTES, TIMELINE_STATS

MINUTOS = MINUTES                     # (10, 15, 20)
C_ETAPA1 = (0.01, 0.03, 0.05)         # grilla de la etapa 1; se elige por log loss en A2-A4 (nunca en V)
C_ETAPA2 = 0.5
MODELO_PRINCIPAL = "etapa1"           # §9.S.78 "Decisión": la etapa 2 no pasa la regla de §9.S.47
A2, A3, A4, TV, TDV = 4, 5, 6, 7, 8   # tramos
ST = {s: i for i, s in enumerate(TIMELINE_STATS)}
COLUMNAS_JUGADOR = [                  # variables por jugador que usan las expectativas por línea
    "acum_d_gold_15", "v20_d_gold_15", "acum_d_gold_10", "v20_d_gold_10", "acum_d_xp_15", "acum_d_cs_15",
    "wr_acum", "wr_v20", "estreno", "offrole", "logn_tot", "logn_champ", "share_lane",
    "acum_deaths_pm", "acum_vision_pm", "acum_damage_pm", "acum_gold_pm",
]
PARAM_HGB = dict(max_iter=90, max_leaf_nodes=31, learning_rate=0.06, max_bins=63, min_samples_leaf=150,
                 max_features=0.8, early_stopping=False, random_state=42)
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigm(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


# --------------------------------------------------------------------------------------------- tramos

def tramos(D: dict) -> np.ndarray:
    """Tramo cronológico de cada partida (0-3 antes de A2, 4-6 = A2-A4, 7 = V Master+, 8 = Diamond-V)."""
    from src.models.validacion_temporal import bloques_por_fecha
    bloque, t = D["bloque"], D["t"]
    chunk = np.full(D["NM"], -1, dtype=np.int32)
    b0 = bloque == 0
    chunk[b0] = bloques_por_fecha(t[b0], 4)
    chunk[bloque == 1], chunk[bloque == 2], chunk[bloque == 3] = A2, A3, A4
    chunk[D["val"]] = TV
    chunk[D["diam_v"]] = TDV
    return chunk


def pliegues(chunk: np.ndarray, t: np.ndarray, master: np.ndarray, val0: np.ndarray) -> list:
    """Pares (te, tr) hacia adelante para las piezas previas: tramos 1-6, y V + Diamond-V juntos.

    Para el tramo b se entrena con Master+ de los tramos < b y anteriores a su primera partida; para la ventana de
    V, con Master+ anterior a la ventana (lo mismo que `modelo_previo.filas_entrenamiento_v`).
    """
    out = []
    for b in range(1, 7):
        te = np.flatnonzero(chunk == b)
        tr = np.flatnonzero((chunk >= 0) & (chunk < b) & (t < t[te].min()) & master)
        out.append((te, tr))
    out.append((np.flatnonzero((chunk == TV) | (chunk == TDV)), np.flatnonzero(~val0 & master)))
    return out


# --------------------------------------------------------------------------------------------- previo

def previo_hacia_adelante(D: dict, extras: np.ndarray, chunk: np.ndarray) -> np.ndarray:
    """Logit del escalón 9 (`modelo_previo`) de cada partida, ajustado solo con partidas anteriores.

    Tramos 1-6: Master+ de los tramos anteriores. V y Diamond-V: el modelo de `modelo_previo.evaluar_v`
    (Master+ anterior al 16/09), así que en V es exactamente el previo vigente (0,6727, §9.S.75). Tramo 0: NaN.
    """
    L = np.full(D["NM"], np.nan, dtype=np.float64)
    for te, tr in pliegues(chunk, D["t"], D["master"], D["val0"]):
        if te.size and chunk[te[0]] in (TV, TDV):
            tr = modelo_previo.filas_entrenamiento_v(D)
        log(f"  previo: tramo {chunk[te[0]]} ({len(te):,}) con {len(tr):,} partidas")
        L[te] = _logit(modelo_previo.ajustar(D, extras, tr).predecir(D, extras, te))
    return L


# --------------------------------------------------------------------------------------------- estado

def leer_timeline(mids: np.ndarray) -> dict:
    """Las 8 estadísticas por jugador a los 10/15/20 del train, alineadas con `mids` (se verifica).

    `tl_raw` (NM, 2, 5, 3, 8): oro, XP y daño en miles; con NaN si la partida terminó antes del minuto
    (`game_duration <= 60 m`, el frame final del Nexo; §9.S.57).
    """
    from src.features.margin import ORDEN
    usecols = ["match_id", "game_creation", "game_duration", "player_id", "team_id", "lane", "win"] + \
              [f"{s}_{m}" for m in MINUTOS for s in TIMELINE_STATS]
    df = rutas.leer_partidas(usecols=usecols)
    df = df[df["lane"].isin(IG.LANES)]
    df = df[df.groupby(["match_id", "team_id"])["lane"].transform("nunique") == 5]
    df = df[df.groupby("match_id")["lane"].transform("size") == 10]
    df = df[df.groupby("match_id")["win"].transform("sum") == 5]
    df = df.sort_values(ORDEN).reset_index(drop=True)
    NM = len(mids)
    if len(df) != 10 * NM or not (df["match_id"].to_numpy()[::10] == mids).all():
        raise ValueError("Las partidas del CSV no coinciden con las de variables_partida")
    dur = df["game_duration"].to_numpy(np.float32)[::10]
    tl = np.empty((NM, 2, 5, len(MINUTOS), len(TIMELINE_STATS)), dtype=np.float32)
    for mi, m in enumerate(MINUTOS):
        for si, s in enumerate(TIMELINE_STATS):
            a = df[f"{s}_{m}"].to_numpy(np.float32).reshape(NM, 2, 5)
            if s in ("gold", "xp", "damage"):
                a = a / 1000.0
            tl[:, :, :, mi, si] = IG.mask_finished_matches(a, dur, m)
    return {"tl_raw": tl, "dur_sec": dur, "player_ids": df["player_id"].to_numpy().astype(str).reshape(NM, 2, 5)}


def oro_residual(tl: np.ndarray, folds: list) -> np.ndarray:
    """Oro que no explican CS, kills y asistencias (placas, torres, recompensas), con OLS por línea hacia adelante."""
    out = np.full(tl.shape[:3] + (len(MINUTOS),), np.nan, dtype=np.float32)
    for mi in range(len(MINUTOS)):
        g, c, k, a = (tl[:, :, :, mi, ST[s]] for s in ("gold", "cs", "kills", "assists"))
        for te, tr in folds:
            fit = np.zeros(len(tl), bool)
            fit[tr] = True
            coef = IG.fit_residual_gold_coeffs(g, c, k, a, fit)
            out[te, :, :, mi] = IG.apply_residual_gold(g[te], c[te], k[te], a[te], coef)
    return out


def _d(tl, mi, s):
    return (tl[:, 0, :, mi, ST[s]] - tl[:, 1, :, mi, ST[s]]).astype(np.float32)


def expectativas(tl, folds, draft, comp, cbias, p_mat, mem, semilla: int = 42) -> tuple[dict, dict]:
    """Δ esperado por línea (azul − rojo) antes de jugar, y transiciones 10→15 y 15→20 (§9.S.57).

    * `exp[stat_m]` (NM, 5) para oro, XP, CS y daño: Ridge sobre draft + composición; para oro y XP, más un
      HGB sobre el residuo con los jugadores, los campeones y el duelo esperado (espejado: se entrena con el
      partido y su espejo, y se predice ½(f(x) − f(x̄)), así que es antisimétrico).
    * `trans[trans_gold_m]` (NM, 5): oro de la línea en $m$ esperado con lo que se vio en el minuto anterior.

    Antes había un LightGBM donde hoy está `HistGradientBoostingRegressor` (mismos hiperparámetros:
    90 árboles, 31 hojas, tasa 0,06, 63 bins, 150 por hoja, 80 % de columnas); así no hace falta `lightgbm`.
    `semilla` es la del HGB (elige el 80 % de columnas de cada división); 42 es la principal (§9.S.80).
    """
    param = {**PARAM_HGB, "random_state": semilla}
    NM = len(tl)
    Xr = sparse.hstack([draft, sparse.csr_matrix(comp["comp_anti"])]).tocsr()
    exp = {}
    for mi, m in enumerate(MINUTOS):
        for s in ("gold", "xp", "cs", "damage"):
            obs = _d(tl, mi, s)
            pred = np.zeros((NM, 5), np.float32)
            for l in range(5):
                ok = np.isfinite(obs[:, l])
                for te, tr in folds:
                    tr = tr[ok[tr]]
                    if len(tr) >= 200:
                        rg = Ridge(alpha=300.0, fit_intercept=False, solver="lsqr").fit(Xr[tr], obs[tr, l])
                        pred[te, l] = rg.predict(Xr[te])
            exp[f"{s}_{m}"] = pred
    log("  expectativas: Ridge listo")
    slot = comp["slot_static"]
    for mi, m in enumerate(MINUTOS):
        for s in ("gold", "xp"):
            key = f"{s}_{m}"
            res = _d(tl, mi, s) - exp[key]
            for l in range(5):
                lado = [p_mat[:, 0, l], p_mat[:, 1, l], slot[:, 0, l], slot[:, 1, l],
                        cbias[:, 0, l:l + 1], cbias[:, 1, l:l + 1], mem[:, 4 + l:5 + l], comp["comp_anti"]]
                espejo = [p_mat[:, 1, l], p_mat[:, 0, l], slot[:, 1, l], slot[:, 0, l],
                          cbias[:, 1, l:l + 1], cbias[:, 0, l:l + 1], -mem[:, 4 + l:5 + l], -comp["comp_anti"]]
                Xa, Xb = np.c_[tuple(lado)].astype(np.float32), np.c_[tuple(espejo)].astype(np.float32)
                ok = np.isfinite(res[:, l])
                for te, tr in folds:
                    tr = tr[ok[tr]]
                    if len(tr) < 500:
                        continue
                    h = HistGradientBoostingRegressor(**param).fit(np.r_[Xa[tr], Xb[tr]], np.r_[res[tr, l], -res[tr, l]])
                    exp[key][te, l] += 0.5 * (h.predict(Xa[te]) - h.predict(Xb[te])).astype(np.float32)
            log(f"  expectativas: {key} con HGB")
    trans = {}
    for mi, m in ((1, 15), (2, 20)):
        cur = _d(tl, mi, "gold")
        dg, dx, dk = _d(tl, mi - 1, "gold"), _d(tl, mi - 1, "xp"), _d(tl, mi - 1, "kills")
        pred = np.zeros((NM, 5), np.float32)
        for l in range(5):
            f = [dg, dx[:, l:l + 1], dk[:, l:l + 1], exp[f"gold_{m}"][:, l:l + 1],
                 cbias[:, 0, l:l + 1] - cbias[:, 1, l:l + 1], comp["comp_anti"]]
            if m == 20:
                f.append(dg - _d(tl, 0, "gold"))
            Xa = np.column_stack(f).astype(np.float32)
            fin = np.isfinite(Xa).all(axis=1)
            ok = np.isfinite(cur[:, l]) & fin
            for te, tr in folds:
                tr = tr[ok[tr]]
                te = te[fin[te]]
                if len(tr) < 500 or len(te) == 0:
                    continue
                h = HistGradientBoostingRegressor(**param).fit(np.r_[Xa[tr], -Xa[tr]], np.r_[cur[tr, l], -cur[tr, l]])
                pred[te, l] = 0.5 * (h.predict(Xa[te]) - h.predict(-Xa[te]))
        trans[f"trans_gold_{m}"] = pred
    log("  expectativas: transiciones listas")
    return exp, trans


def construir_estado(D: dict, mem: np.ndarray, semilla: int = 42) -> dict:
    """Timeline, oro residual y expectativas por línea de todas las partidas (no usa el resultado de ninguna)."""
    mids = D["Z"]["mids"].astype(str)
    chunk = tramos(D)
    folds = pliegues(chunk, D["t"], D["master"], D["val0"])
    log("estado: timeline")
    T = leer_timeline(mids)
    tl, dur = T["tl_raw"], T["dur_sec"]
    log("estado: escalado de campeones (causal)")
    tempo = IG.causal_champ_player_tempo(D["t"], mids, D["Z"]["champ"], T["player_ids"], dur / 60.0, D["y"],
                                         _d(tl, 0, "gold"), _d(tl, 1, "gold"), _d(tl, 2, "gold"))
    atrib = pd.read_csv(rutas.ATRIB_CSV) if rutas.ATRIB_CSV.exists() else None
    ddr = pd.read_csv(rutas.DDRAGON_CSV) if rutas.DDRAGON_CSV.exists() else None
    comp = IG.build_composition_features(D["Z"]["champ"], atrib, ddr)
    cols = list(D["cols"])
    p_mat = D["Xj"][:, :, :, [cols.index(c) for c in COLUMNAS_JUGADOR]].astype(np.float32)
    log("estado: oro residual")
    gres = oro_residual(tl, folds)
    log("estado: expectativas por línea")
    exp, trans = expectativas(tl, folds, D["DRAFT"], comp, tempo["cbias"], p_mat, mem, semilla)
    return dict(mids=mids, chunk=chunk, dur_sec=dur, tl_raw=tl, gold_resid=gres,
                **{f"exp_{k}": v for k, v in exp.items()}, **{f"tr_{k}": v for k, v in trans.items()})


# --------------------------------------------------------------------------------------------- insumos con caché

def _huella_grupo(nombre: str) -> str:
    with np.load(rutas.CACHE / f"{nombre}.npz", allow_pickle=True) as z:
        return str(z[cache.CLAVE])


def verificar_mids(datos: dict, mids: np.ndarray, que: str) -> None:
    if "mids" not in datos or not np.array_equal(np.asarray(datos["mids"]).astype(str), np.asarray(mids).astype(str)):
        raise ValueError(f"{que}: las partidas guardadas no coinciden con las actuales (no se alinea por posición)")


def insumos(D: dict, V: dict, rehacer: bool = False, semilla: int = 42) -> dict:
    """Estado y previo de todas las partidas, desde `data/cache/ganador/` si la huella sigue vigente.

    Llamar después de `variables.calcular_todas()` (que deja vigentes los grupos de variables). Dos cachés:
    `en_vivo_estado` (depende de `variables_partida` y `margen_esperado_par`) y `en_vivo_previo` (depende de
    todos los grupos: si cambia el previo, solo se rehace él). Con una `semilla` distinta de la principal (42, la de
    los dos HGB: el de las expectativas y el del oro por línea del previo) van a `en_vivo_estado_s{n}` y
    `en_vivo_previo_s{n}`; `V` tiene que venir de `variables.calcular_todas(semilla=n)`.
    """
    from src.ganador import variables
    suf = "" if semilla == variables.SEMILLA else f"_s{semilla}"
    modo = f"modo={rutas.MODO}"
    mids = D["Z"]["mids"].astype(str)
    codigo_ig = [cache.firma_codigo(f) for f in (IG.mask_finished_matches, IG.fit_residual_gold_coeffs,
                                                  IG.apply_residual_gold, IG.causal_champ_player_tempo,
                                                  IG.build_composition_features)]
    h_est = cache.huella(construir_estado, [*rutas.FUENTES, rutas.ATRIB_CSV, rutas.DDRAGON_CSV],
                         [_huella_grupo("variables_partida"), _huella_grupo("margen_esperado_par"), *codigo_ig,
                          repr(sorted(PARAM_HGB.items())), repr(COLUMNAS_JUGADOR), modo, f"semilla={semilla}"])
    E = cache.obtener("en_vivo_estado" + suf,
                      lambda: construir_estado(D, V["margen_esperado_par"]["MATCHUP_EXP_MARGIN"], semilla), h_est, rehacer)
    verificar_mids(E, mids, "en_vivo_estado")
    grupos = ["variables_partida", "margen_campeon_linea", "nivel_historico", "wl_foto1", "contracruces",
              "winrate_linea", "kalman_elo", variables.nombre_oro(semilla), "margen_esperado_par"]
    h_prev = cache.huella(previo_hacia_adelante, [], [*map(_huella_grupo, grupos),
                                                      cache.firma_codigo(modelo_previo.ajustar),
                                                      cache.firma_codigo(variables.matriz_extras), modo])

    def _previo():
        log("previo: escalón 9 hacia adelante")
        return {"mids": mids, "L": previo_hacia_adelante(D, variables.matriz_extras(D, V), E["chunk"])}
    P = cache.obtener("en_vivo_previo" + suf, _previo, h_prev, rehacer)
    verificar_mids(P, mids, "en_vivo_previo")
    return dict(E, L=P["L"], y=D["y"].astype(np.int8), master=D["master"], val=D["val"], diam_v=D["diam_v"],
                post=D["post"])


# --------------------------------------------------------------------------------------------- modelo

def variables_minuto(I: dict, m: int, filas: np.ndarray) -> np.ndarray:
    """Matriz antisimétrica de la etapa 1 al minuto `m` para las partidas `filas` (bloque A2 de §9.S.60).

    Solo lee el timeline hasta el minuto `m`, el oro residual de `m`, las expectativas previas a la partida y $L$.
    """
    exp = {k[4:]: v[filas] for k, v in I.items() if k.startswith("exp_")}
    tr = {k[3:]: v[filas] for k, v in I.items() if k.startswith("tr_")}
    mi = MINUTOS.index(m)
    tl = I["tl_raw"][filas].copy()
    tl[:, :, :, mi + 1:, :] = np.nan            # nada posterior al minuto m entra, ni por error
    f = IG.extract_minute_features(m, I["L"][filas].astype(np.float32), tl, I["gold_resid"][filas], exp_lane=exp,
                                   exp_trans=tr, include_all_stats=True, include_traj=True, include_expected=True,
                                   include_comp_cross=False, include_player=False)
    return f["X_anti"]


def validas(I: dict, m: int) -> np.ndarray:
    """Partidas vivas en el minuto `m` (duran más de 60 m s), con timeline completo hasta `m` y previo."""
    mi = MINUTOS.index(m)
    ok_tl = np.isfinite(I["tl_raw"][:, :, :, :mi + 1, :]).all(axis=(1, 2, 3, 4))
    return (I["dur_sec"] > 60.0 * m) & ok_tl & np.isfinite(I["L"])


def _ajustar_lr(X, y, C):
    return LogisticRegression(C=C, max_iter=3000).fit(X, y)


def ajustar_predecir(I: dict, m: int, filas_final: np.ndarray, c_etapa1=C_ETAPA1) -> dict:
    """Las dos etapas al minuto `m`.

    Devuelve las predicciones hacia adelante en A2-A4 (`a24`, etapa 1 y dos etapas), las de `filas_final`
    (ajustadas con todo lo anterior a V) y los parámetros elegidos. Nada se elige mirando `filas_final`.
    `z_a24` y `z_fin` son las del modelo principal (`MODELO_PRINCIPAL`: la etapa 1, igual a `z1_*`);
    `z2_*` son las de la variante descartada de dos etapas.
    """
    y, chunk, master = I["y"].astype(int), I["chunk"], I["master"]
    ok = validas(I, m)
    bloques = [np.flatnonzero((chunk == b) & master & ok) for b in (A2, A3, A4)]
    a24 = np.concatenate(bloques)
    tr_b = [np.flatnonzero((chunk >= 1) & (chunk < b) & master & ok) for b in (A2, A3, A4)]
    tr_full = np.flatnonzero((chunk >= 1) & (chunk <= A4) & master & ok)
    if not ok[filas_final].all():
        raise ValueError("hay partidas sin estado válido en el minuto pedido")
    usadas = np.unique(np.r_[tr_full, filas_final])
    pos = {k: np.searchsorted(usadas, v) for k, v in (("full", tr_full), ("fin", filas_final))}
    X = variables_minuto(I, m, usadas)

    def fila(idx):
        return X[np.searchsorted(usadas, idx)]

    # Etapa 1: C por log loss en A2-A4 hacia adelante
    z_a24, mejor = {}, (np.inf, None)
    for c in c_etapa1:
        z = np.concatenate([_ajustar_lr(fila(tr), y[tr], c).decision_function(fila(te)) for tr, te in zip(tr_b, bloques)])
        ll = log_loss(y[a24], _sigm(z))
        z_a24[c] = z
        if ll < mejor[0]:
            mejor = (ll, c)
    c1 = mejor[1]
    z1 = z_a24[c1]
    lr1 = _ajustar_lr(X[pos["full"]], y[tr_full], c1)
    z_fin = lr1.decision_function(X[pos["fin"]])

    corte = np.cumsum([0, *map(len, bloques)])
    z2, z2_fin, coef = etapa2(z1, I["L"][a24], y[a24], corte, z_fin, I["L"][filas_final])
    return dict(a24=a24, bloques=bloques, z1_a24=z1, z2_a24=z2, z1_fin=z_fin, z2_fin=z2_fin, C1=c1,
                etapa2=coef, n_entrenamiento=len(tr_full), z_a24=z1, z_fin=z_fin)


def etapa2(z1, L, y, corte, z_fin, L_fin, E=None, E_fin=None):
    r"""$\operatorname{logit} P = a + b z + c L\;[+\,\gamma^\top e]$, hacia adelante en A2-A4 y con A2-A4 entero para V.

    `z1`, `L`, `y` (y `E`, columnas opcionales que se suman, ya escaladas) son las partidas de A2-A4 en orden
    (A2 = `corte[0]:corte[1]`, ...). En A2 queda neutra (z1). Devuelve (z en A2-A4, z en las finales, (a, b, c, ...)).
    """
    X = np.c_[z1, L] if E is None else np.c_[z1, L, E]
    X_fin = np.c_[z_fin, L_fin] if E is None else np.c_[z_fin, L_fin, E_fin]
    z2 = np.asarray(z1, dtype=np.float64).copy()
    for k in (1, 2):
        prev, sl = slice(0, corte[k]), slice(corte[k], corte[k + 1])
        z2[sl] = _ajustar_lr(X[prev], y[prev], C_ETAPA2).decision_function(X[sl])
    lr = _ajustar_lr(X, y, C_ETAPA2)
    return z2, lr.decision_function(X_fin), (float(lr.intercept_[0]), *map(float, lr.coef_[0]))


# --------------------------------------------------------------------------------------------- métricas

def ece(y, p, n=10) -> float:
    g = pd.DataFrame({"p": p, "y": y, "d": pd.qcut(p, n, labels=False, duplicates="drop")}).groupby("d") \
          .agg(p=("p", "mean"), r=("y", "mean"), n=("y", "size"))
    return float((np.abs(g.p - g.r) * g.n).sum() / g.n.sum())


def metricas(y, p) -> dict:
    y = np.asarray(y).astype(int)
    return {"auc": roc_auc_score(y, p), "acierto": float(((p > 0.5) == y).mean()),
            "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]), "ece": ece(y, p),
            "siempre_rojo": float((y == 0).mean())}


def metricas_ic(y, p, B: int = 1000, semilla: int = 0) -> dict:
    """Métricas con IC 95 % por bootstrap de partidas (percentiles 2,5 y 97,5)."""
    y = np.asarray(y).astype(int)
    m = metricas(y, p)
    rng = np.random.default_rng(semilla)
    reps = [metricas(y[b], p[b]) for b in rng.integers(0, len(y), size=(B, len(y)))]
    out = {"n": len(y)}
    for k, v in m.items():
        r = np.array([x[k] for x in reps])
        out[k] = (v, float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5)))
    return out


def delta_auc_ic(y, p1, p0, B: int = 1000, semilla: int = 0) -> tuple[float, float, float]:
    """AUC(p1) − AUC(p0) con IC 95 % por bootstrap pareado."""
    y = np.asarray(y).astype(int)
    d = roc_auc_score(y, p1) - roc_auc_score(y, p0)
    rng = np.random.default_rng(semilla)
    ds = [roc_auc_score(y[b], p1[b]) - roc_auc_score(y[b], p0[b]) for b in rng.integers(0, len(y), size=(B, len(y)))]
    return float(d), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))
