"""
Idea 44 del brainstorming (pedido del usuario, 2026-11-16): builds raras y conjuntos de ítems.

Lo ya probado con ítems (`g_20`, `g_20b`, `g_20c`) miraba el *estilo* del build (AP / AD / tanque), las botas
y el trinket, y dio ≈ 0. Acá se miran los **ítems concretos y sus combinaciones**:

* Ítems completos: sin `item.json` (la red de esta máquina intercepta TLS) se definen con los datos: un ítem
  es componente si su frecuencia en el inventario final cae con la duración (frecuencia en partidas > 32 min
  / frecuencia en < 22 min < 0,7: se transformó en otro). Quedan botas tier 2 y legendarios.
* Por cada fila (jugador en una partida), contra las partidas **anteriores** del mismo campeón en la misma
  línea (conteos en línea, partida por partida; las 10 filas de una partida no se ven entre sí):
  - rareza de ítems, pares y tríos: media de −log P(conjunto | campeón, línea);
  - combinación rara: media de log P(i)P(j) / P(ij) sobre los pares (ítems comunes juntados de forma rara);
  - calidad de ítems, pares y tríos: winrate encogido del conjunto con ese campeón − winrate del campeón.
  Estas métricas usan el build final, así que de la partida actual **no se usan nunca**: solo entran a los
  promedios previos del jugador.
* Variables previas del jugador (causales): media de su historia y de sus últimas 20; con el campeón que
  eligió (encogida hacia su media); y si le rinde la build rara (cov. encogida entre victoria y rareza).
* Equipo: media azul − media rojo. Evaluación contra el híbrido + mapa: bloques A2-A4 y V Master+.
* Descriptivo (para el notebook, no para el modelo): victoria según la rareza y la calidad del build de la
  propia partida, por tramos de duración → `builds_descriptivo.csv`.

    python historial/ideas_posteriores/brainstorm_44_builds.py      # ~10 min
"""
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.brainstorm_base import ajustar_predecir, cargar, comparar, evaluar, mostrar  # noqa: E402

LANES = ["BOTTOM", "JUNGLE", "MIDDLE", "TOP", "UTILITY"]
ITEMS = [f"item{i}" for i in range(6)]
SALIDA = REPO / "data" / "analysis" / "brainstorm"
CACHE = SALIDA / "builds.npz"
K_WR, K_CAMP, K_COV, MIN_CL = 20.0, 5.0, 10.0, 30
METRICAS = ["rare_item", "rare_pair", "rare_tri", "combo", "q_item", "q_pair", "q_tri"]
T0 = time.time()
log = lambda s: print(f"[{time.time() - T0:5.0f}s] {s}", flush=True)  # noqa: E731


def leer(D):
    cols = ["match_id", "game_creation", "game_duration", "player_id", "team_id", "lane", "win",
            "champion_name"] + ITEMS
    df = pd.read_csv(REPO / "data/splits/train/full_dataset.csv", usecols=cols)
    df = df[df.lane.isin(LANES)]
    df = df[df.groupby(["match_id", "team_id"]).lane.transform("nunique") == 5]
    df = df[df.groupby("match_id").lane.transform("size") == 10]
    df = df[df.groupby("match_id").win.transform("sum") == 5]
    df = df.sort_values(["game_creation", "match_id", "team_id", "lane"]).reset_index(drop=True)
    mids = df.match_id.to_numpy()[::10]
    assert len(mids) == D["NM"] and (mids == D["Z"]["mids"]).all(), "el orden de las partidas no coincide con D"
    return df


def completos(df):
    L = df[["game_duration"] + ITEMS].melt(id_vars="game_duration").query("value != 0")
    c, l = df.game_duration < 22 * 60, df.game_duration > 32 * 60
    fc = L[L.game_duration < 22 * 60].value.value_counts() / c.sum()
    fl = L[L.game_duration > 32 * 60].value.value_counts() / l.sum()
    n = L.value.value_counts()
    T = pd.DataFrame({"n": n, "fc": fc, "fl": fl}).fillna(0)
    ok = T[(T.n >= 300) & (T.fl / (T.fc + 1e-9) >= 0.7)].index
    return {int(i): k for k, i in enumerate(sorted(ok))}


def metricas_por_fila(df, idx):
    """Métricas del build de cada fila contra los conteos previos de su campeón y línea (en línea)."""
    K = len(idx)
    cl_cod = pd.factorize(df.champion_name.astype(str) + "|" + df.lane)[0].astype(np.int64)
    items = df[ITEMS].to_numpy()
    win = df.win.to_numpy().astype(float)
    n_cl, w_cl = {}, {}
    cnt, wins = {}, {}
    M = np.full((len(df), len(METRICAS)), np.nan)
    n_it = np.zeros(len(df), int)
    sets = []
    for r in range(len(df)):
        s = sorted({idx[i] for i in items[r] if i in idx})
        n_it[r] = len(s)
        sets.append(s)
    for a in range(0, len(df), 10):
        filas = range(a, a + 10)
        for r in filas:                                    # medir antes de actualizar
            cl, s = int(cl_cod[r]), sets[r]
            n = n_cl.get(cl, 0)
            if n < MIN_CL or not s:
                continue
            wr = w_cl[cl] / n
            def P(key):
                return (cnt.get(key, 0) + 0.5) / (n + 1.0)
            def Q(key):
                return (wins.get(key, 0.0) + K_WR * wr) / (cnt.get(key, 0) + K_WR) - wr
            k1 = [(cl, i) for i in s]
            k2 = [(cl, i, j) for i, j in combinations(s, 2)]
            k3 = [(cl, i, j, k) for i, j, k in combinations(s, 3)]
            M[r, 0] = np.mean([-np.log(P(k)) for k in k1])
            M[r, 4] = np.mean([Q(k) for k in k1])
            if k2:
                M[r, 1] = np.mean([-np.log(P(k)) for k in k2])
                M[r, 3] = np.mean([np.log(P((cl, k[1]))) + np.log(P((cl, k[2]))) - np.log(P(k)) for k in k2])
                M[r, 5] = np.mean([Q(k) for k in k2])
            if k3:
                M[r, 2] = np.mean([-np.log(P(k)) for k in k3])
                M[r, 6] = np.mean([Q(k) for k in k3])
        for r in filas:                                    # actualizar con la partida
            cl, s, w = int(cl_cod[r]), sets[r], win[r]
            n_cl[cl] = n_cl.get(cl, 0) + 1
            w_cl[cl] = w_cl.get(cl, 0.0) + w
            ks = [(cl, i) for i in s] + [(cl, i, j) for i, j in combinations(s, 2)] + \
                 [(cl, i, j, k) for i, j, k in combinations(s, 3)]
            for k in ks:
                cnt[k] = cnt.get(k, 0) + 1
                wins[k] = wins.get(k, 0.0) + w
        if a % 200000 == 0:
            log(f"  filas {a:,}/{len(df):,}")
    return M, n_it


def previas(df, M):
    """Variables previas de cada fila a partir de las métricas de las partidas anteriores del jugador."""
    o = np.lexsort((df.game_creation.to_numpy(), df.player_id.to_numpy()))
    F = pd.DataFrame(M[o], columns=METRICAS)
    pid = df.player_id.to_numpy()[o]
    pc = (pd.Series(pid).astype(str) + "|" + pd.Series(df.champion_name.astype(str).to_numpy()[o])).to_numpy()
    win = df.win.to_numpy()[o].astype(float)

    def media_previa(G, clave):
        hay, Gz = G.notna().astype(float), G.fillna(0.0)
        s = Gz.groupby(clave).cumsum() - Gz
        n = hay.groupby(clave).cumsum() - hay
        return s, n

    s, n = media_previa(F, pid)
    hist = s / n.replace(0, np.nan)
    ult20 = F.groupby(pid).shift(1).groupby(pid).rolling(20, min_periods=1).mean() \
             .reset_index(level=0, drop=True).reindex(F.index)
    sc, nc = media_previa(F[["rare_pair", "q_pair"]], pc)
    camp = (sc + K_CAMP * hist[["rare_pair", "q_pair"]].fillna(F[["rare_pair", "q_pair"]].mean())) / (nc + K_CAMP)
    # ¿le rinde la build rara? cov. encogida entre (victoria − 0,5) y la rareza de pares centrada
    z = (F.rare_pair - F.rare_pair.mean()) * (win - 0.5)
    sz, nz = media_previa(z.to_frame("c"), pid)
    cov = sz.c / (nz.c + K_COV)
    X = pd.concat([hist.add_suffix("_hist"), ult20.add_suffix("_u20"), camp.add_suffix("_camp"),
                   cov.rename("rinde_rara")], axis=1)
    inv = np.empty_like(o)
    inv[o] = np.arange(len(o))
    return X.iloc[inv].reset_index(drop=True)


def construir(D):
    if CACHE.exists():
        C = np.load(CACHE, allow_pickle=True)
        return C["E"], list(C["cols"])
    df = leer(D)
    idx = completos(df)
    log(f"ítems completos: {len(idx)}")
    cache_m = SALIDA / "builds_metricas.npz"            # el conteo en línea tarda ~11 min
    if cache_m.exists():
        C = np.load(cache_m)
        M, n_it = C["M"], C["n_it"]
    else:
        M, n_it = metricas_por_fila(df, idx)
        np.savez_compressed(cache_m, M=M, n_it=n_it)
    log("métricas por fila listas")
    # descriptivo: victoria según el build de la propia partida, por duración
    d = pd.DataFrame(M, columns=METRICAS).assign(win=df.win.to_numpy(), n_items=n_it,
                                                 dur=pd.cut(df.game_duration / 60, [0, 20, 25, 30, 35, 70]))
    filas = []
    for met in ("rare_pair", "q_pair", "combo"):
        ok = d[met].notna()
        q = d.loc[ok].groupby("dur", observed=True)[met].transform(lambda x: pd.qcut(x.rank(method="first"), 5, labels=False))
        t = d.loc[ok].assign(quintil=q + 1).groupby(["dur", "quintil"], observed=True).agg(
            filas=("win", "size"), winrate=("win", "mean"), items=("n_items", "mean")).reset_index()
        filas.append(t.assign(metrica=met))
    pd.concat(filas).to_csv(SALIDA / "builds_descriptivo.csv", index=False)
    X = previas(df, M)
    cols = list(X.columns)
    A = X.to_numpy(np.float32)
    mu = np.nanmean(A, axis=0)
    A = np.where(np.isnan(A), mu, A).reshape(D["NM"], 2, 5, -1)
    E = A[:, 0].mean(1) - A[:, 1].mean(1)
    np.savez_compressed(CACHE, E=E, cols=np.array(cols))
    return E, cols


def main():
    D = cargar()
    log(f"partidas {D['NM']:,}; V Master+ {int(D['val'].sum()):,}")
    E, cols = construir(D)
    log(f"variables previas listas: {cols}")
    sel = lambda pref: [i for i, c in enumerate(cols) if any(c.startswith(p) for p in pref)]  # noqa: E731
    grupos = {
        "rareza (ítems, pares, tríos, combinación)": [i for i, c in enumerate(cols)
                                                      if c.split("_")[0] in ("rare", "combo") and not c.endswith("_camp")],
        "calidad de conjuntos (ítems, pares, tríos)": [i for i, c in enumerate(cols)
                                                       if c.startswith("q_") and not c.endswith("_camp")],
        "con el campeón elegido (rareza y calidad de pares)": [i for i, c in enumerate(cols) if c.endswith("_camp")],
        "le rinde la build rara": sel(["rinde_rara"]),
        "todo": list(range(len(cols))),
    }
    P = {"Híbrido + mapa (referencia)": evaluar(D, lambda tr, te: ajustar_predecir(D, tr, te))}
    for nombre, ii in grupos.items():
        Z = E[:, ii]
        P[f"+ {nombre}"] = evaluar(D, lambda tr, te, Z=Z: ajustar_predecir(D, tr, te, extra=Z, extra_post=False))
        log(f"listo: {nombre} ({len(ii)} cols)")
    T = comparar(D, P, "Híbrido + mapa (referencia)")
    mostrar(T)
    T.to_csv(SALIDA / "builds.csv", index=False)


if __name__ == "__main__":
    main()
