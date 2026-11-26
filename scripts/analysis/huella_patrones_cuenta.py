"""
Detección no supervisada de subgrupos de estilo + winrate dentro de cada cuenta (pedido del usuario,
2026-11-19; PROJECT_SUMMARY §10.7).

En lugar de evaluar ventanas deslizantes sobre secuencias empalmadas sintéticas, analiza directamente
cada cuenta real de LAS con suficiente historial en las partidas Master+ con timeline completo
(`hm.timeline_train()`, 41 variables residualizadas por campeón, línea y resultado):

1. **Embeddings fuera de muestra (2-fold por jugador):** se entrenan dos redes contrastivas
   (41 → 128 → 128 → 32, InfoNCE) en mitades disjuntas de jugadores. Cada jugador se proyecta
   exclusivamente con la red que **nunca lo vio** durante el entrenamiento (cero memorización).
   El residuo es **por pliegue** (default, §9.S.77 punto 5): el pliegue que proyecta a una mitad normaliza
   (media, desvío y tamaño mínimo de cada grupo campeón-línea-resultado) solo con la otra mitad, la misma con
   la que se entrena su red. `--residuo global` reproduce la versión anterior (normaliza con todos los
   jugadores, `resid_timeline.pkl`) con la misma partición; sus salidas llevan el sufijo `_residuo_global`.
2. **Chequeo de ortogonalidad al resultado (`win`):** verifica que las victorias y derrotas de una
   misma persona caen en la misma región del espacio de huella (distancia victorias vs. derrotas
   equivalente a una partición aleatoria), garantizando que buscar un subgrupo con huella distinta y
   winrate alto no recupera simplemente "partidas ganadas".
3. **Clustering no supervisado intra-cuenta (sobre bloques de `BLOQUE` partidas):** para cada cuenta
   con al menos `MIN_PARTIDAS` partidas (y opcionalmente controlado por rol principal), se agrupan sus
   bloques de partidas con K-Means esférico ($k = 2$, exigiendo $\\ge 2$ bloques por subgrupo) usando
   **solo** los embeddings de huella (sin ver `win`, ni horario, ni fecha).
4. **Cruce con winrate y corroboración independiente:** sobre los dos subgrupos encontrados
   ($C_{\\text{base}}$ mayoritario vs. $C_{\\text{alt}}$ minoritario) se calculan:
   * `d_estilo`: distancia euclídea entre los embeddings de los perfiles agregados de $C_{\\text{base}}$
     y $C_{\\text{alt}}$ (cada uno con $\\ge 10$ partidas).
   * `sil`: silhouette medio entre los bloques de la cuenta.
   * `wr_base`, `wr_alt`, `delta_wr` y `p_wr` (test exacto de Fisher sobre victorias/derrotas).
   * Señales independientes (no usadas en el clustering): `tvd_hora` (distancia de variación total
     entre la distribución horaria ART de ambos subgrupos), `contiguidad` (fracción de transiciones
     que permanecen en el mismo subgrupo: alta = racha continua tipo boosting; baja = alternancia tipo
     cuenta compartida), `% rol principal` en $C_{\\text{alt}}$ y solapamiento de campeones.

Salidas en `data/analysis/huella/`:
* `patrones_cuentas.csv` — una fila por cuenta analizada (todo el padrón elegible).
* `patrones_ortogonalidad.json` — chequeo de ortogonalidad al winrate y umbrales de referencia.
* `patrones_casos.csv` — trayectoria bloque a bloque de ejemplos representativos (candidato a boosting,
  candidato a cuenta compartida y racha buena con misma huella) para graficar en el notebook.
* `patrones_resumen.json` — los números principales de §10.7 con IC 95 %: Wilson para las proporciones,
  bootstrap por cuenta para las medias de cada cuadrante, bootstrap por jugador para la ortogonalidad y z de
  Fisher para las correlaciones.

    python scripts/analysis/huella_patrones_cuenta.py                    # residuo por pliegue
    python scripts/analysis/huella_patrones_cuenta.py --residuo global   # versión anterior, misma partición
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.features.fingerprint import STYLE_FEATURES  # noqa: E402
from scripts.analysis import huella_mejoras as hm  # noqa: E402
from scripts.analysis.huella_niveles import datos_timeline, entrenar_red  # noqa: E402
from scripts.analysis.fingerprint_level1 import player_info  # noqa: E402

SALIDA = hm.SALIDA
FEATS = STYLE_FEATURES + hm.HABITOS
T0 = time.time()
log = lambda s: print(f"[{time.time() - T0:6.1f}s] {s}", flush=True)  # noqa: E731
CATEGORIAS = ["Normal / sin señal conjunta", "Racha buena (misma huella)",
              "Candidata a compartida", "Huella distinta + caída WR", "Candidata a boosting"]


# --------------------------------------------------------------------------------------------- datos 2-fold
def datos_2fold(residuo="pliegue", crudo=None):
    """Partidas con timeline residualizadas para el 2-fold, y los dos pliegues.

    Devuelve `(df, pliegues)`. Cada pliegue es `(resid, entrena, proyecta)`: su red se entrena con las filas de
    `entrena` en `resid` y proyecta a los jugadores de `proyecta`. `df` junta, para cada jugador, las filas con
    las que lo proyecta su pliegue: es lo que se analiza después.

    * `residuo="pliegue"` (§9.S.77 punto 5): cada pliegue normaliza (media, desvío y tamaño mínimo del grupo
      campeón-línea-resultado) solo con sus jugadores de entrenamiento, `hm.residuo_habitos(ajuste=entrena)`. Ni
      la red ni la normalización vieron a los jugadores que proyecta.
    * `residuo="global"`: la versión anterior, una sola normalización con todos los jugadores para los dos
      pliegues (`datos_timeline("global")`, caché `resid_timeline.pkl`).

    La partición es siempre la de `hm.cargar_timeline`, sobre los jugadores crudos, así que las dos opciones se
    comparan pareadas. `crudo` reemplaza a `hm.timeline_train()` (tests).
    """
    if residuo == "global":
        r, ajuste, evalu = datos_timeline("global", crudo=crudo)
        r = r.reset_index(drop=True)
        return r, [(r, ajuste, evalu), (r, evalu, ajuste)]
    if residuo != "pliegue":
        raise ValueError(f"residuo desconocido: {residuo!r}")
    d = hm.timeline_train() if crudo is None else crudo
    ajuste, evalu = hm.partir_jugadores(d["player_id"].unique())
    pliegues = []
    for entrena, proyecta in ((ajuste, evalu), (evalu, ajuste)):
        r = hm.residuo_habitos(d, FEATS, ajuste=entrena).reset_index(drop=True)
        pliegues.append((r, entrena, proyecta))
    df = pd.concat([r[r["player_id"].isin(proy)] for r, _, proy in pliegues], ignore_index=True)
    log(f"residuo por pliegue: {len(pliegues[0][0]):,} y {len(pliegues[1][0]):,} filas por pliegue; "
        f"{len(df):,} analizadas ({df['player_id'].nunique():,} jugadores)")
    return df, pliegues


def _red_desde_estado(sd):
    import torch
    m = torch.nn.Sequential(
        torch.nn.Linear(len(FEATS), 128), torch.nn.GELU(),
        torch.nn.Linear(128, 128), torch.nn.GELU(),
        torch.nn.Linear(128, 32),
    )
    m.load_state_dict(sd)
    m.eval()

    def emb(M):
        with torch.no_grad():
            return torch.nn.functional.normalize(m(torch.tensor(M, dtype=torch.float32)), dim=1).numpy()
    return emb


def _estado_de(emb):
    """state_dict de la red que encierra la función de `entrenar_red` (para la caché)."""
    import torch
    return next(c.cell_contents.state_dict() for c in emb.__closure__
                if isinstance(c.cell_contents, torch.nn.Module))


def redes_2fold(residuo="pliegue", pasos=3000, crudo=None, cache=True, semilla=0):
    """`(df, emb_por_jug)`: los datos de `datos_2fold` y, por jugador, el embedding de la red que nunca lo vio.

    Caché: `red_2fold_timelines.pt` con el residuo global (el nombre de antes) y `red_2fold_timelines_pliegue.pt`
    por pliegue (con `_semillaN` si `semilla` no es 0; la red del pliegue i usa `semilla + i`). Guarda una firma
    (filas y jugadores de entrenamiento de cada pliegue) y los pasos; si no coinciden con los datos actuales, por
    ejemplo con otra base de timelines, se reentrena.
    """
    import torch
    df, pliegues = datos_2fold(residuo, crudo=crudo)
    firma = [[int(r["player_id"].isin(ent).sum()), len(ent)] for r, ent, _ in pliegues]
    nombre = "red_2fold_timelines" + ("" if residuo == "global" else "_pliegue") + (
        f"_semilla{semilla}" if semilla else "")
    ruta = SALIDA / f"{nombre}.pt"
    ck = torch.load(ruta, map_location="cpu", weights_only=True) if cache and ruta.exists() else None
    if ck is not None and ck.get("firma") == firma and ck.get("pasos") == pasos:
        log(f"cargando redes contrastivas 2-fold desde {ruta.name}")
        embs = [_red_desde_estado(ck["eval"]), _red_desde_estado(ck["ajus"])]
    else:
        embs = []
        for i, (r, ent, proy) in enumerate(pliegues):
            log(f"entrenando red del pliegue {i + 1} ({len(ent):,} jug.) → proyecta {len(proy):,} jug.")
            embs.append(entrenar_red(r, FEATS, ent, pasos=pasos, semilla=semilla + i))
        if cache:
            SALIDA.mkdir(parents=True, exist_ok=True)
            torch.save({"eval": _estado_de(embs[0]), "ajus": _estado_de(embs[1]), "firma": firma, "pasos": pasos},
                       ruta)
    emb_por_jug = {}
    for emb, (_, _, proy) in zip(embs, pliegues):
        emb_por_jug |= {p: emb for p in proy}
    return df, emb_por_jug


# --------------------------------------------------------------------------------------------- IC
def ic_wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    """IC 95 % de Wilson para k / n."""
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(c - h), float(c + h)]


def ic_media(x, n_boot: int = 2000, semilla: int = 0) -> list[float]:
    """[media, IC 95 % inferior, superior] por bootstrap de percentiles."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return [float("nan")] * 3
    rng = np.random.default_rng(semilla)
    b = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    return [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def ic_corr(x, y) -> list[float]:
    """[r de Pearson, IC 95 % inferior, superior] por la transformación z de Fisher."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    n = int(ok.sum())
    r = float(np.corrcoef(x[ok], y[ok])[0, 1])
    z, se = np.arctanh(np.clip(r, -0.999999, 0.999999)), 1 / np.sqrt(max(n - 3, 1))
    return [r, float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))]


def clasificar(res: pd.DataFrame) -> tuple[float, float]:
    """Cuadrantes de §10.7 (p95 de d_estilo y p90 de silhouette + significancia de WR, en los dos signos) y FDR
    de Benjamini-Hochberg sobre los p-valores del WR (global y condicional al cribado por estilo, que es ciego al
    resultado). Agrega las columnas a `res` y devuelve los umbrales."""
    thr_d = float(res["d_estilo"].quantile(0.95))
    thr_sil = float(res["silhouette"].quantile(0.90))
    res["estilo_distinto"] = (res["d_estilo"] >= thr_d) & (res["silhouette"] >= thr_sil)
    res["q_wr_two"] = fdr_bh(res["p_wr_two"].to_numpy())
    res["q_wr_gt"] = fdr_bh(res["p_wr_gt"].to_numpy())
    res["q_wr_lt"] = fdr_bh(res["p_wr_lt"].to_numpy())
    res["q_wr_two_en_est"] = np.nan
    m_est = res["estilo_distinto"].to_numpy()
    if m_est.any():
        res.loc[m_est, "q_wr_two_en_est"] = fdr_bh(res.loc[m_est, "p_wr_two"].to_numpy())
    res["wr_alto"] = (res["delta_wr"] >= 0.15) & (res["p_wr_gt"] < 0.05)
    res["wr_bajo"] = (res["delta_wr"] <= -0.15) & (res["p_wr_lt"] < 0.05)
    res["categoria"] = np.select(
        [
            res["estilo_distinto"] & res["wr_alto"],
            res["estilo_distinto"] & res["wr_bajo"],
            res["estilo_distinto"] & (res["delta_wr"].abs() < 0.12),
            (~res["estilo_distinto"]) & res["wr_alto"],
        ],
        ["Candidata a boosting", "Huella distinta + caída WR", "Candidata a compartida",
         "Racha buena (misma huella)"],
        default="Normal / sin señal conjunta",
    )
    return thr_d, thr_sil


def resumen_cuadrantes(res: pd.DataFrame) -> dict:
    """Tabla de cuadrantes con IC: % de cuentas (Wilson) y medias por cuadrante (bootstrap por cuenta)."""
    n = len(res)
    out = {"cuentas": n,
           "corr_d_estilo_delta_wr": ic_corr(res["d_estilo"], res["delta_wr"]),
           "corr_d_estilo_tvd_hora": ic_corr(res["d_estilo"], res["tvd_hora"]),
           "cuadrantes": {}}
    for cat in CATEGORIAS:
        s = res[res["categoria"] == cat]
        out["cuadrantes"][cat] = {
            "n": int(len(s)), "pct": len(s) / n if n else float("nan"), "pct_ic": ic_wilson(len(s), n),
            **{c: ic_media(s[c]) for c in ["d_estilo", "silhouette", "wr_base", "wr_alt", "delta_wr",
                                            "tvd_hora", "contiguidad"]},
        }
    return out


def kmeans_2_min_size(E: np.ndarray, min_blocks: int = 2, n_init: int = 12, seed: int = 0):
    """K-Means k=2 sobre embeddings normalizados; garantiza >= min_blocks en cada cluster."""
    n = len(E)
    km = KMeans(n_clusters=2, random_state=seed, n_init=n_init)
    lbl = km.fit_predict(E)
    counts = np.bincount(lbl, minlength=2)
    if counts.min() >= min_blocks:
        return lbl
    # Si un cluster quedó con < min_blocks, reasignamos los puntos más cercanos al centroide minoritario
    minor = int(np.argmin(counts))
    major = 1 - minor
    dists = np.linalg.norm(E - km.cluster_centers_[minor], axis=1)
    lbl = np.full(n, major, dtype=int)
    lbl[np.argsort(dists)[:min_blocks]] = minor
    return lbl


def tvd_horas(ts_a: np.ndarray, ts_b: np.ndarray) -> float:
    """Distancia de variación total (0 a 1) entre distribuciones de franja horaria (6 bloques de 4 h, ART)."""
    ha = ((pd.to_datetime(ts_a, unit="ms", utc=True).tz_convert("America/Argentina/Buenos_Aires").hour) // 4).to_numpy()
    hb = ((pd.to_datetime(ts_b, unit="ms", utc=True).tz_convert("America/Argentina/Buenos_Aires").hour) // 4).to_numpy()
    pa = np.bincount(ha, minlength=6) / max(len(ha), 1)
    pb = np.bincount(hb, minlength=6) / max(len(hb), 1)
    return float(0.5 * np.abs(pa - pb).sum())


def tvd_categorias(a: np.ndarray, b: np.ndarray, n_cat: int) -> float:
    """Distancia de variación total (0 a 1) entre dos vectores enteros en [0, n_cat)."""
    pa = np.bincount(a, minlength=n_cat) / max(len(a), 1)
    pb = np.bincount(b, minlength=n_cat) / max(len(b), 1)
    return float(0.5 * np.abs(pa - pb).sum())


def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    """Ajuste de Benjamini-Hochberg (FDR) para un vector de p-valores."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p.copy()
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q_ord = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n, dtype=float)
    q[order] = np.clip(q_ord, 0.0, 1.0)
    return q


def arquetipos_campeon() -> tuple[dict, dict]:
    """Agrupa pares (campeón, línea) del train en 10 arquetipos globales y 3 subtipos por carril."""
    cbl = pd.read_csv(ROOT / "data" / "splits" / "train" / "champions_by_lane.csv")
    num_cols = [
        "avg_kills", "avg_deaths", "avg_assists", "avg_kda",
        "avg_cs_at_10", "avg_gold_at_10", "avg_damage_at_10",
        "avg_cs_at_15", "avg_gold_at_15", "avg_damage_at_15",
        "avg_total_damage", "avg_damage_taken", "avg_gold_earned", "avg_total_cs",
        "avg_vision_score", "first_blood_rate_pct", "first_tower_rate_pct",
    ]
    faltan = [c for c in num_cols if c not in cbl.columns]
    assert not faltan, f"Faltan columnas en champions_by_lane.csv: {faltan}"
    c_ok = cbl[cbl["games_played"] >= 50].copy()
    z_all = StandardScaler().fit_transform(c_ok[num_cols].fillna(0.0).to_numpy(float))
    km_g = KMeans(n_clusters=10, n_init=20, random_state=0).fit_predict(z_all)
    arch_glob = {(ch, ln): int(k) for ch, ln, k in zip(c_ok["champion_name"], c_ok["lane"], km_g)}

    arch_sub = {}
    for ln, g in c_ok.groupby("lane"):
        zw = StandardScaler().fit_transform(g[num_cols].fillna(0.0).to_numpy(float))
        km_s = KMeans(n_clusters=3, n_init=20, random_state=0).fit_predict(zw)
        for ch, k in zip(g["champion_name"], km_s):
            arch_sub[(ch, ln)] = int(k)
    return arch_glob, arch_sub


def chequeo_ortogonalidad(df: pd.DataFrame, emb_por_jug: dict, min_por_lado: int = 15, seed: int = 0) -> dict:
    """Compara distancia emb(victorias) vs emb(derrotas) contra una partición aleatoria del mismo tamaño."""
    rng = np.random.default_rng(seed)
    X = df[FEATS].to_numpy(np.float32)
    W = df["win"].to_numpy(int)
    d_wl, d_rand = [], []
    for pid, ix in df.groupby("player_id").indices.items():
        if pid not in emb_por_jug:
            continue
        w_ix = ix[W[ix] == 1]
        l_ix = ix[W[ix] == 0]
        k = min(len(w_ix), len(l_ix))
        if k < min_por_lado:
            continue
        f = emb_por_jug[pid]
        kw = rng.choice(w_ix, k, replace=False)
        kl = rng.choice(l_ix, k, replace=False)
        perm = rng.permutation(np.concatenate([kw, kl]))
        vecs = np.stack([X[kw].mean(0), X[kl].mean(0), X[perm[:k]].mean(0), X[perm[k:]].mean(0)])
        E = f(vecs)
        d_wl.append(float(np.linalg.norm(E[0] - E[1])))
        d_rand.append(float(np.linalg.norm(E[2] - E[3])))
    d_wl, d_rand = np.array(d_wl), np.array(d_rand)
    # IC 95 % del cociente de medias: bootstrap por jugador, pareado (el mismo jugador en los dos lados)
    jb = np.random.default_rng(seed + 1).integers(0, len(d_wl), (2000, len(d_wl)))
    ratio_b = d_wl[jb].mean(1) / d_rand[jb].mean(1)
    return {
        "dist_win_vs_loss_ic95": ic_media(d_wl)[1:],
        "dist_aleatoria_ic95": ic_media(d_rand)[1:],
        "ratio_media_ic95": [round(float(np.percentile(ratio_b, q)), 4) for q in (2.5, 97.5)],
        "jugadores": int(len(d_wl)),
        "dist_win_vs_loss_media": round(float(d_wl.mean()), 4),
        "dist_win_vs_loss_mediana": round(float(np.median(d_wl)), 4),
        "dist_aleatoria_media": round(float(d_rand.mean()), 4),
        "dist_aleatoria_mediana": round(float(np.median(d_rand)), 4),
        "ratio_media": round(float(d_wl.mean() / d_rand.mean()), 4),
    }


def analizar_cuentas(df: pd.DataFrame, emb_por_jug: dict, info: pd.DataFrame,
                     arch_glob: dict | None = None, arch_sub: dict | None = None,
                     bloque: int = 5, min_partidas: int = 50, min_bloques_alt: int = 2,
                     solo_rol_principal: bool = False):
    """Ejecuta el clustering no supervisado intra-cuenta y cruza el subgrupo con winrate y señales externas."""
    d = df.sort_values(["player_id", "game_creation"]).reset_index(drop=True)
    if solo_rol_principal:
        d = d[d["lane"] == d["player_id"].map(info["main_lane"])].reset_index(drop=True)
    if arch_glob is None or arch_sub is None:
        arch_glob, arch_sub = arquetipos_campeon()

    X = d[FEATS].to_numpy(np.float32)
    W = d["win"].to_numpy(int)
    TS = d["game_creation"].to_numpy(np.int64)
    LANES = d["lane"].to_numpy()
    CHAMPS = d["champion_name"].to_numpy()
    lane_to_i = {"TOP": 0, "JUNGLE": 1, "MIDDLE": 2, "BOTTOM": 3, "UTILITY": 4}
    LANE_I = np.array([lane_to_i.get(ln, 0) for ln in LANES], dtype=int)
    ARCH_G = np.array([arch_glob.get((ch, ln), 0) for ch, ln in zip(CHAMPS, LANES)], dtype=int)
    ARCH_S = np.array([arch_sub.get((ch, ln), 0) for ch, ln in zip(CHAMPS, LANES)], dtype=int)

    filas = []
    bloques_por_cuenta = {}
    for pid, ix in d.groupby("player_id").indices.items():
        if len(ix) < min_partidas or pid not in emb_por_jug:
            continue
        f = emb_por_jug[pid]
        n_b = len(ix) // bloque
        if n_b < 2 * min_bloques_alt + 2:
            continue
        # Bloques cronológicos no superpuestos de `bloque` partidas
        ix_use = ix[: n_b * bloque]
        b_idx = [ix_use[k * bloque:(k + 1) * bloque] for k in range(n_b)]
        M = np.stack([X[bi].mean(0) for bi in b_idx])
        E = f(M)

        lbl = kmeans_2_min_size(E, min_blocks=min_bloques_alt, seed=0)
        # Cluster 0 = mayoritario (base del perfil), cluster 1 = minoritario (subgrupo alternativo)
        c0_n, c1_n = int((lbl == 0).sum()), int((lbl == 1).sum())
        if c1_n > c0_n:
            lbl = 1 - lbl
            c0_n, c1_n = c1_n, c0_n

        sil = float(silhouette_score(E, lbl))
        idx_base = np.concatenate([b_idx[k] for k in range(n_b) if lbl[k] == 0])
        idx_alt = np.concatenate([b_idx[k] for k in range(n_b) if lbl[k] == 1])

        # Distancia entre los perfiles agregados de ambos subgrupos en el espacio de la red
        E_sub = f(np.stack([X[idx_base].mean(0), X[idx_alt].mean(0)]))
        d_estilo = float(np.linalg.norm(E_sub[0] - E_sub[1]))

        # Cruce con winrate (100% fuera del clustering)
        w_base, n_base = int(W[idx_base].sum()), len(idx_base)
        w_alt, n_alt = int(W[idx_alt].sum()), len(idx_alt)
        wr_base, wr_alt = w_base / n_base, w_alt / n_alt
        delta_wr = wr_alt - wr_base
        _, p_wr_two = fisher_exact([[w_alt, n_alt - w_alt], [w_base, n_base - w_base]], alternative="two-sided")
        _, p_wr_gt = fisher_exact([[w_alt, n_alt - w_alt], [w_base, n_base - w_base]], alternative="greater")
        _, p_wr_lt = fisher_exact([[w_alt, n_alt - w_alt], [w_base, n_base - w_base]], alternative="less")

        # Corroboración independiente (horario, continuidad temporal, rol y tipo de campeón)
        tvd_h = tvd_horas(TS[idx_base], TS[idx_alt])
        tvd_rol = tvd_categorias(LANE_I[idx_base], LANE_I[idx_alt], 5)
        tvd_tipo = tvd_categorias(ARCH_G[idx_base], ARCH_G[idx_alt], 10)
        tvd_sub = tvd_categorias(ARCH_S[idx_base], ARCH_S[idx_alt], 3)
        # Contigüidad temporal del subgrupo alternativo: 1 = aparece en un solo bloque continuo; 0 = todo intercalado
        trans = int((lbl[1:] != lbl[:-1]).sum())
        # Mínimo teórico de transiciones (1 si está en un extremo, 2 si está en el medio) vs máximo
        max_trans = min(2 * c1_n, n_b - 1)
        contiguidad = 1.0 - max(0, trans - 1) / max(max_trans - 1, 1)

        main_lane = info.at[pid, "main_lane"]
        tier = info.at[pid, "player_tier"]
        pct_main_base = float((LANES[idx_base] == main_lane).mean())
        pct_main_alt = float((LANES[idx_alt] == main_lane).mean())
        ch_base, ch_alt = set(CHAMPS[idx_base]), set(CHAMPS[idx_alt])
        champ_jacc = len(ch_base & ch_alt) / max(len(ch_base | ch_alt), 1)
        # Distancia de cada bloque al centroide del perfil base (para graficar trayectorias de casos)
        dist_al_base = np.linalg.norm(E - E_sub[0], axis=1)

        filas.append({
            "player_id": pid, "tier": tier, "main_lane": main_lane,
            "partidas": int(len(ix_use)), "bloques": n_b,
            "partidas_base": n_base, "partidas_alt": n_alt,
            "frac_alt": n_alt / len(ix_use),
            "d_estilo": d_estilo, "silhouette": sil,
            "wr_base": wr_base, "wr_alt": wr_alt, "delta_wr": delta_wr,
            "p_wr_two": float(p_wr_two), "p_wr_gt": float(p_wr_gt), "p_wr_lt": float(p_wr_lt),
            "tvd_hora": tvd_h, "contiguidad": float(contiguidad),
            "tvd_rol": tvd_rol, "tvd_tipo_camp": tvd_tipo, "tvd_subtipo_rol": tvd_sub,
            "pct_main_base": pct_main_base, "pct_main_alt": pct_main_alt,
            "champ_jaccard": float(champ_jacc),
        })
        bloques_por_cuenta[pid] = pd.DataFrame({
            "player_id": pid,
            "bloque": np.arange(1, n_b + 1),
            "subgrupo": np.where(lbl == 1, "Subgrupo alternativo", "Perfil base"),
            "dist_al_base": dist_al_base,
            "wr_bloque": [float(W[bi].mean()) for bi in b_idx],
            "fecha_ms": [int(np.median(TS[bi])) for bi in b_idx],
            "pct_main": [float((LANES[bi] == main_lane).mean()) for bi in b_idx],
        })

    return pd.DataFrame(filas), bloques_por_cuenta


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bloque", type=int, default=5)
    ap.add_argument("--min-partidas", type=int, default=50)
    ap.add_argument("--pasos", type=int, default=3000)
    ap.add_argument("--residuo", choices=["pliegue", "global"], default="pliegue",
                    help="normalizar cada pliegue solo con los jugadores con que se entrena su red (default) o con "
                         "todos, como antes del §9.S.77 punto 5 (las salidas llevan el sufijo _residuo_global)")
    ap.add_argument("--timelines", type=Path, default=None,
                    help="timelines.db a usar si hay que rearmar las cachés de hábitos (por defecto la de "
                         "huella_mejoras o la variable HUELLA_TIMELINES). Se abre solo en lectura")
    ap.add_argument("--semilla", type=int, default=0,
                    help="semilla de las redes (pliegue i usa semilla + i); distinta de 0 mide el ruido del "
                         "entrenamiento y agrega el sufijo _semillaN a salidas y caché")
    args = ap.parse_args()
    if args.timelines is not None:
        hm.RUTA_TL = args.timelines
    sufijo = ("_residuo_global" if args.residuo == "global" else "") + (
        f"_semilla{args.semilla}" if args.semilla else "")

    SALIDA.mkdir(parents=True, exist_ok=True)
    # 2-fold por jugador: ninguna cuenta fue vista por la red que la proyecta ni, con el residuo por pliegue,
    # por la normalización de las filas con que se la proyecta
    df, emb_por_jug = redes_2fold(args.residuo, pasos=args.pasos, semilla=args.semilla)
    info = player_info(df)
    log(f"timelines completos ({args.residuo}): {len(df):,} filas, {df['player_id'].nunique():,} jugadores")

    # 1. Chequeo de ortogonalidad al resultado (win)
    orto = chequeo_ortogonalidad(df, emb_por_jug)
    log(f"ortogonalidad a win (n={orto['jugadores']:,}): dist W vs L = {orto['dist_win_vs_loss_media']:.4f} "
        f"vs aleatoria = {orto['dist_aleatoria_media']:.4f} (ratio {orto['ratio_media']:.3f})")

    # 2. Escaneo intra-cuenta (todas las partidas + versión restringida al rol principal para controlar off-role)
    arch_glob, arch_sub = arquetipos_campeon()
    log("analizando subgrupos intra-cuenta (todas las partidas)...")
    res_all, blq_all = analizar_cuentas(df, emb_por_jug, info, arch_glob=arch_glob, arch_sub=arch_sub,
                                        bloque=args.bloque, min_partidas=args.min_partidas, solo_rol_principal=False)
    log("analizando subgrupos intra-cuenta (solo rol principal)...")
    res_rol, blq_rol = analizar_cuentas(df, emb_por_jug, info, arch_glob=arch_glob, arch_sub=arch_sub,
                                        bloque=args.bloque, min_partidas=args.min_partidas, solo_rol_principal=True)

    # Clasificación con ambos signos de ΔWR y ajuste FDR (Benjamini-Hochberg)
    for nombre, res in [("todas", res_all), ("rol_principal", res_rol)]:
        thr_d, thr_sil = clasificar(res)
        n_fdr_glob = int((res["estilo_distinto"] & res["wr_alto"] & (res["q_wr_gt"] < 0.05)).sum())
        n_fdr_cond = int((res["estilo_distinto"] & res["wr_alto"] & (res["q_wr_two_en_est"] < 0.10)).sum())
        log(f"[{nombre}] N={len(res):,} | umbral d_estilo p95={thr_d:.3f} (sil p90={thr_sil:.3f}) | "
            f"boosting FDR global<0.05={n_fdr_glob}, FDR cond.<0.10={n_fdr_cond} | "
            f"conteos: {res['categoria'].value_counts().to_dict()}")
        for cat in CATEGORIAS:
            s = res[res["categoria"] == cat]
            if len(s):
                log(f"   {cat:<28} n={len(s):>4} | d={s['d_estilo'].mean():.2f} sil={s['silhouette'].mean():.2f} "
                    f"ΔWR={s['delta_wr'].mean():+.1%} hora={s['tvd_hora'].mean():.2f} rol={s['tvd_rol'].mean():.2f} "
                    f"tipo={s['tvd_tipo_camp'].mean():.2f} contig={s['contiguidad'].mean():.2f}")

    # Usar como principal el análisis restringido al **rol principal** (elimina falsos positivos por autofill/off-role)
    thr_d_rol = float(res_rol["d_estilo"].quantile(0.95))
    meta = {
        **orto,
        "bloque_partidas": args.bloque,
        "min_partidas": args.min_partidas,
        "cuentas_todas": int(len(res_all)),
        "cuentas_rol_principal": int(len(res_rol)),
        "umbral_d_estilo_p95": round(thr_d_rol, 4),
        "umbral_silhouette": 0.25,
        "umbral_delta_wr": 0.15,
        "umbral_p_fisher": 0.05,
        "corr_d_estilo_delta_wr": round(float(res_rol["d_estilo"].corr(res_rol["delta_wr"])), 4),
        "corr_d_estilo_tvd_hora": round(float(res_rol["d_estilo"].corr(res_rol["tvd_hora"])), 4),
        "residuo": args.residuo,
    }
    resumen = {"residuo": args.residuo, "filas": int(len(df)), "jugadores": int(df["player_id"].nunique()),
               "ortogonalidad": orto, "todas": resumen_cuadrantes(res_all),
               "rol_principal": resumen_cuadrantes(res_rol)}
    # las salidas principales se escriben antes de elegir los ejemplos (que pueden fallar con una base chica)
    res_rol.to_csv(SALIDA / f"patrones_cuentas{sufijo}.csv", index=False)
    res_all.to_csv(SALIDA / f"patrones_cuentas_todos_roles{sufijo}.csv", index=False)
    (SALIDA / f"patrones_ortogonalidad{sufijo}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                                                 encoding="utf-8")
    (SALIDA / f"patrones_resumen{sufijo}.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False),
                                                           encoding="utf-8")

    # Guardar 3 casos ilustrativos anonimizados del análisis en rol principal (boosting, compartida, racha)
    casos = []
    # 1) Candidata a boosting: mayor d_estilo entre las candidatas con >= 15 partidas en el subgrupo y alta contigüidad
    cand_b = res_rol[(res_rol["categoria"] == "Candidata a boosting") & (res_rol["partidas_alt"] >= 15)]
    if cand_b.empty:
        cand_b = res_rol[res_rol["categoria"] == "Candidata a boosting"]
    if not cand_b.empty:
        pid_b = cand_b.sort_values(["d_estilo", "delta_wr"], ascending=False).iloc[0]["player_id"]
        casos.append(blq_rol[pid_b].assign(caso="Candidata a boosting"))

    # 2) Candidata a cuenta compartida: alta d_estilo, WR parejo, baja contigüidad (intercalada)
    cand_c = res_rol[(res_rol["categoria"] == "Candidata a compartida") & (res_rol["partidas_alt"] >= 15)]
    if cand_c.empty:
        cand_c = res_rol[res_rol["categoria"] == "Candidata a compartida"]
    if not cand_c.empty:
        pid_c = cand_c.sort_values(["d_estilo", "contiguidad"], ascending=[False, True]).iloc[0]["player_id"]
        casos.append(blq_rol[pid_c].assign(caso="Candidata a compartida"))

    # 3) Racha buena (misma huella): WR alto pero baja d_estilo
    cand_r = res_rol[(res_rol["categoria"] == "Racha buena (misma huella)") & (res_rol["partidas_alt"] >= 15)]
    if cand_r.empty:
        cand_r = res_rol[res_rol["categoria"] == "Racha buena (misma huella)"]
    if not cand_r.empty:
        pid_r = cand_r.sort_values(["delta_wr", "d_estilo"], ascending=[False, True]).iloc[0]["player_id"]
        casos.append(blq_rol[pid_r].assign(caso="Racha buena (misma huella)"))

    # 4. Guardar huellas por subgrupo (base vs. alternativo) para 3 casos representativos en el mismo rol
    #    (cuenta boosteada, cuenta compartida y cuenta normal) + promedio |Δ| del padrón por categoría
    def _perfil_partes(pid: str, solo_rol: bool) -> tuple[np.ndarray, np.ndarray]:
        sub = df[df["player_id"] == pid].sort_values("game_creation")
        if solo_rol:
            sub = sub[sub["lane"] == info.at[pid, "main_lane"]]
        sub = sub.reset_index(drop=True)
        n_b = len(sub) // args.bloque
        X_p = sub[FEATS].to_numpy(np.float32)
        b_idx = [np.arange(k * args.bloque, (k + 1) * args.bloque) for k in range(n_b)]
        E_p = emb_por_jug[pid](np.stack([X_p[bi].mean(0) for bi in b_idx]))
        lbl_p = kmeans_2_min_size(E_p, min_blocks=2, seed=0)
        if (lbl_p == 1).sum() > (lbl_p == 0).sum():
            lbl_p = 1 - lbl_p
        ix0 = np.concatenate([b_idx[k] for k in range(n_b) if lbl_p[k] == 0])
        ix1 = np.concatenate([b_idx[k] for k in range(n_b) if lbl_p[k] == 1])
        return X_p[ix0].mean(0), X_p[ix1].mean(0)

    ejemplos_rows = []
    if not cand_b.empty:
        row_b = cand_b.sort_values(["d_estilo", "delta_wr"], ascending=False).iloc[0]
        lane_ref = row_b["main_lane"]
        # Compartida en el mismo rol con mayor cantidad de partidas (estimación más precisa y WR parejo)
        cand_c_ej = res_rol[(res_rol["categoria"] == "Candidata a compartida")
                            & (res_rol["main_lane"] == lane_ref)
                            & (res_rol["partidas_alt"] >= 30)
                            & (res_rol["delta_wr"].abs() < 0.05)]
        if cand_c_ej.empty:
            cand_c_ej = res_rol[(res_rol["categoria"] == "Candidata a compartida") & (res_rol["main_lane"] == lane_ref)]
        row_c = cand_c_ej.sort_values(["partidas", "d_estilo"], ascending=False).iloc[0]
        # Normal típica en el mismo rol (cercana a la mediana de d_estilo normal, ~80-140 partidas, WR ~50 %)
        med_d_norm = float(res_rol[res_rol["categoria"] == "Normal / sin señal conjunta"]["d_estilo"].median())
        cand_n_ej = res_rol[(res_rol["categoria"] == "Normal / sin señal conjunta")
                            & (res_rol["main_lane"] == lane_ref)
                            & (res_rol["partidas"].between(80, 140))
                            & (res_rol["partidas_alt"] >= 25)
                            & (res_rol["delta_wr"].abs() < 0.04)
                            & (res_rol["wr_base"].between(0.45, 0.55))]
        row_n = cand_n_ej.assign(
            err=(cand_n_ej["d_estilo"] - med_d_norm).abs() + (cand_n_ej["silhouette"] - 0.10).abs()
        ).sort_values("err").iloc[0]

        for tipo, r_ej in [("Cuenta boosteada", row_b), ("Cuenta compartida", row_c), ("Cuenta normal", row_n)]:
            m0, m1 = _perfil_partes(r_ej["player_id"], solo_rol=True)
            for j, f in enumerate(FEATS):
                ejemplos_rows.append({
                    "tipo": tipo, "main_lane": r_ej["main_lane"],
                    "partidas": int(r_ej["partidas"]),
                    "partidas_base": int(r_ej["partidas_base"]), "partidas_alt": int(r_ej["partidas_alt"]),
                    "wr_base": float(r_ej["wr_base"]), "wr_alt": float(r_ej["wr_alt"]),
                    "d_estilo": float(r_ej["d_estilo"]), "silhouette": float(r_ej["silhouette"]),
                    "tvd_hora": float(r_ej["tvd_hora"]), "contiguidad": float(r_ej["contiguidad"]),
                    "variable": f, "base": float(m0[j]), "alt": float(m1[j]), "delta": float(m1[j] - m0[j]),
                })

    # Resumen del padrón completo: media de |Δ| por variable en las 3 categorías (todos los roles)
    rng_p = np.random.default_rng(42)
    pids_b_all = res_all[res_all["categoria"] == "Candidata a boosting"]["player_id"].tolist()
    pids_c_all = res_all[res_all["categoria"] == "Candidata a compartida"]["player_id"].tolist()
    pids_n_all = rng_p.choice(res_all[res_all["categoria"] == "Normal / sin señal conjunta"]["player_id"].to_numpy(),
                              min(500, int((res_all["categoria"] == "Normal / sin señal conjunta").sum())),
                              replace=False).tolist()
    padron_cols = {}
    for col_name, pids_g in [("abs_delta_boosting", pids_b_all),
                             ("abs_delta_compartida", pids_c_all),
                             ("abs_delta_normal", pids_n_all)]:
        ds = [np.abs(_perfil_partes(p, solo_rol=False)[1] - _perfil_partes(p, solo_rol=False)[0]) for p in pids_g]
        padron_cols[col_name] = np.mean(ds, axis=0) if ds else np.full(len(FEATS), np.nan)
    df_padron = pd.DataFrame({"variable": FEATS, **padron_cols})

    if casos:
        pd.concat(casos, ignore_index=True).to_csv(SALIDA / f"patrones_casos{sufijo}.csv", index=False)
    if ejemplos_rows:
        pd.DataFrame(ejemplos_rows).to_csv(SALIDA / f"patrones_huellas_ejemplos{sufijo}.csv", index=False)
    df_padron.to_csv(SALIDA / f"patrones_huellas_padron{sufijo}.csv", index=False)
    log(f"listo: {SALIDA / f'patrones_cuentas{sufijo}.csv'} (+ resumen con IC, huellas ejemplos y padrón)")


if __name__ == "__main__":
    main()
