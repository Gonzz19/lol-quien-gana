"""Ideas 16 y 17 del brainstorming (docs/BRAINSTORM_NOCHE.md): Composición del draft (§9.S.37) +
Poisson ataque/defensa causal (§9.S.25, + variante oro @15) + Foto 1 W/L dinámico sobre
`solo Master+ + MC + ML`.

Ninguno de los dos bloques está todavía en `variables.npz` ni en `brainstorm_base.py`.
Cuidado de fuga:
* `atributos_campeones.csv` son atributos fijos del campeón (tipo de daño, roles Meraki, rango).
* `block_scores` usa 40 bloques cronológicos estrictamente anteriores (`< b`), y la estandarización
  `z` de oro/daño/oro15 se calcula únicamente sobre `~val0` (`< 16/09`).
"""
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.brainstorm_base import ajustar_predecir, cargar, comparar, evaluar, mostrar  # noqa: E402
from src.features.attack_defense import block_scores  # noqa: E402
from src.ganador import rutas  # noqa: E402

CACHE = REPO / "data" / "analysis" / "brainstorm" / "gemini_comp_poisson.npz"
ATRIB_CSV = rutas.ATRIB_CSV


def construir_composicion(D: dict) -> tuple[np.ndarray, np.ndarray]:
    NM = D["NM"]
    C = pd.read_csv(ATRIB_CSV).set_index("champion_id").astype(float)
    C = C.fillna(C.mean())
    cols = list(C.columns)
    champ = D["Z"]["champ"].reshape(NM, 10)  # IDs reales de campeón por slot
    unicos = np.unique(champ)
    A = C.reindex(unicos).fillna(C.mean())
    cmap = {int(cid): i for i, cid in enumerate(unicos)}
    cidx = np.vectorize(cmap.__getitem__, otypes=[np.int64])(champ)
    V = A.to_numpy(np.float32)[cidx]  # (NM, 10, 14)

    def equipo(E):
        g = {k: E[:, :, i] for i, k in enumerate(cols)}
        ap_ = g["ap"].sum(1)
        return {
            "n_ap": ap_, "todo_ad": (ap_ <= 0.5) * 1.0, "todo_ap": (ap_ >= 4.0) * 1.0,
            "desbalance_dano": np.abs(ap_ - 2.5),
            "dur": g["ps_dur"].sum(1), "cc": g["ps_cc"].sum(1), "mov": g["ps_mov"].sum(1),
            "util": g["ps_util"].sum(1), "dano": g["ps_dano"].sum(1),
            "n_front": g["front"].sum(1), "sin_front": (g["front"].sum(1) == 0) * 1.0,
            "n_engage": g["engage"].sum(1), "sin_engage": (g["engage"].sum(1) == 0) * 1.0,
            "n_rango": g["rango"].sum(1), "alcance": g["alcance"].mean(1) / 100.0,
            "n_poke": g["poke"].sum(1), "n_enchanter": g["enchanter"].sum(1),
            "n_asesino": g["asesino"].sum(1), "n_tirador": g["tirador"].sum(1),
            "sin_cc": (g["ps_cc"].sum(1) <= 3.0) * 1.0, "poco_dur": (g["ps_dur"].sum(1) <= 5.0) * 1.0,
        }

    az, ro = equipo(V[:, :5]), equipo(V[:, 5:])
    X = np.c_[tuple(az[k] - ro[k] for k in az)]
    cruces = [
        az["n_poke"] * ro["sin_engage"] - ro["n_poke"] * az["sin_engage"],
        az["n_asesino"] * ro["poco_dur"] - ro["n_asesino"] * az["poco_dur"],
        az["n_ap"] * ro["dur"] - ro["n_ap"] * az["dur"],
        az["n_engage"] * ro["n_rango"] - ro["n_engage"] * az["n_rango"],
    ]
    COMP = np.c_[X, np.c_[tuple(cruces)]].astype(np.float32)
    Vz = ((V - C.mean().to_numpy(np.float32)) / (C.std().to_numpy(np.float32) + 1e-6)).reshape(NM, 2, 5, -1).astype(np.float32)
    return COMP, Vz


def construir_poisson(D: dict, n_bloques: int = 12) -> tuple[np.ndarray, np.ndarray]:
    NM = D["NM"]
    mids, t, fit0 = D["Z"]["mids"], D["t"], ~D["val0"]
    LANES = ["BOTTOM", "JUNGLE", "MIDDLE", "TOP", "UTILITY"]
    df = pd.read_csv(
        REPO / "data" / "splits" / "train" / "full_dataset.csv",
        usecols=[
            "match_id", "game_creation", "team_id", "lane", "player_id", "champion_id",
            "win", "kills", "gold_earned", "total_damage_dealt_to_champions", "game_duration", "gold_15",
        ],
    )
    df = df[df.lane.isin(LANES)]
    df = df[df.groupby(["match_id", "team_id"]).lane.transform("nunique") == 5]
    df = df[df.groupby("match_id").lane.transform("size") == 10]
    df = df[df.groupby("match_id").win.transform("sum") == 5].copy()
    df = df.sort_values(["game_creation", "match_id", "team_id", "lane"]).reset_index(drop=True)
    assert (df.match_id.to_numpy()[::10] == mids).all()

    jug = df["player_id"].astype("category").cat.codes.to_numpy(np.int64).reshape(NM, 10)
    cl = (df["champion_id"].astype(str) + df["lane"]).astype("category").cat.codes.to_numpy(np.int64).reshape(NM, 10)
    minutos = np.maximum(1.0, df["game_duration"].to_numpy(np.float64)[::10] / 60.0)

    def eq_sum(col: str) -> np.ndarray:
        a = df[col].fillna(0.0).to_numpy(np.float64).reshape(NM, 2, 5).sum(axis=2)
        return a

    kills = eq_sum("kills")
    oro = eq_sum("gold_earned") / minutos[:, None]
    dano = eq_sum("total_damage_dealt_to_champions") / minutos[:, None]
    # Para oro @15 imputar solo filas >900s con media de fit0 para evitar sesgo en remakes tempranos
    g15_raw = df["gold_15"].to_numpy(np.float64).reshape(NM, 2, 5).sum(axis=2)
    ok15 = (minutos > 15.0) & np.isfinite(g15_raw).all(axis=1)
    g15_mean = g15_raw[fit0 & ok15].mean()
    g15 = np.where(ok15[:, None], g15_raw, g15_mean)

    def z_fit(mat: np.ndarray) -> np.ndarray:
        mu = mat[fit0].mean()
        sd = mat[fit0].std() + 1e-6
        return (mat - mu) / sd

    bloque = np.searchsorted(np.quantile(t, np.linspace(0, 1, n_bloques + 1)[1:-1]), t, side="right")
    MEDIDAS = {
        "kills": (kills, "poisson", (150.0,)),
        "oro": (z_fit(oro), "gauss", (15.0,)),
        "dano": (z_fit(dano), "gauss", (15.0,)),
        "g15": (np.where(ok15[:, None], z_fit(g15), 0.0), "gauss", (25.0,)),
    }
    P_list, PJ_list = [], []
    for nombre, (v, fam, lams) in MEDIDAS.items():
        for lam in lams:
            t1 = time.time()
            s = block_scores(jug, cl, v[:, 0], v[:, 1], bloque, lam, 1.0, fam, minutos if fam == "poisson" else None)
            P_list.append(np.c_[s["ataque"], s["defensa"], s["eta"]].astype(np.float32))
            PJ_list.append(np.stack([s["ataque_jug"], s["defensa_jug"]], axis=2).reshape(NM, 2, 5, 2).astype(np.float32))
            print(f"  Poisson {nombre} lam={lam:g} ({time.time() - t1:.1f}s)", flush=True)
    return np.hstack(P_list).astype(np.float32), np.concatenate(PJ_list, axis=-1).astype(np.float32)


def cargar_foto1_eq_din(D: dict) -> np.ndarray:
    NM = D["NM"]
    pl = D["B"]["player"].reshape(-1)
    pr = np.repeat(D["post"], 10)
    with sqlite3.connect(f"file:{REPO / 'data' / 'lol_las.db'}?mode=ro&immutable=1", uri=True) as con:
        foto = pd.read_sql_query(
            "SELECT m.player_id, p.tier, p.league_points, p.wins, p.losses "
            "FROM players p JOIN player_id_map m ON p.puuid = m.puuid",
            con,
        ).drop_duplicates("player_id").set_index("player_id")
    has_f1 = pd.Series(pl).isin(foto.index).to_numpy() & pr
    al = foto.reindex(pl)
    W0 = al["wins"].fillna(0.0).to_numpy(np.float64)
    L0 = al["losses"].fillna(0.0).to_numpy(np.float64)
    zm = (has_f1 & (al["tier"].fillna("").to_numpy() == "MASTER") & (al["league_points"].fillna(-1).to_numpy() == 0)).astype(np.float32)
    y_jug = np.repeat(D["y"], 10).astype(np.float64)
    es_azul = np.tile(np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=bool), NM)
    win_slot = np.where(es_azul, y_jug, 1.0 - y_jug)
    df_post = pd.DataFrame({"_pos": np.arange(NM * 10), "player_id": pl, "t": np.repeat(D["t"], 10),
                            "m": np.repeat(np.arange(NM), 10), "post": pr, "win": win_slot})
    sub_p = df_post[df_post["post"]].sort_values(["t", "m"], kind="mergesort").copy()
    sub_p["dw"] = sub_p.groupby("player_id", sort=False)["win"].cumsum() - sub_p["win"]
    sub_p["dn"] = sub_p.groupby("player_id", sort=False).cumcount().astype(np.float64)
    sub_p["dl"] = sub_p["dn"] - sub_p["dw"]
    dw, dl = np.zeros(NM * 10), np.zeros(NM * 10)
    dw[sub_p["_pos"].to_numpy()] = sub_p["dw"].to_numpy()
    dl[sub_p["_pos"].to_numpy()] = sub_p["dl"].to_numpy()
    W, L = W0 + dw, L0 + dl
    tot = W + L
    wr = np.where(has_f1, (W + 5.0) / (tot + 20.0) - 0.5, 0.0)
    ln = np.where(has_f1, np.log1p(tot) - 5.3, 0.0)
    nw = np.where(has_f1, W - L, 0.0)
    a = np.stack([wr, ln, nw, zm], axis=-1).reshape(NM, 2, 5, 4)
    return (a[:, 0] - a[:, 1]).mean(axis=1).astype(np.float32)


def main():
    D = cargar()
    ms = D["master"]
    if not CACHE.exists():
        print("Construyendo caché gemini_comp_poisson.npz...", flush=True)
        COMP, J_CAMP = construir_composicion(D)
        POIS, J_POIS = construir_poisson(D, n_bloques=12)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, COMP=COMP, J_CAMP=J_CAMP, POIS=POIS, J_POIS=J_POIS)
    G = np.load(CACHE)
    MPR = np.load(REPO / "data" / "analysis" / "brainstorm" / "gemini_mpr.npz")

    from scripts.analysis.brainstorm_base import extras_referencia
    REF = extras_referencia(D)
    pc_path = REPO / "data" / "analysis" / "brainstorm" / "pred_combinado.npz"
    PC = np.load(pc_path) if pc_path.exists() else None

    def esc3(extra=None):
        tot = REF if extra is None else np.c_[REF, extra].astype(np.float32)
        return lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=tot, extra_post=False)

    P = {}
    if PC is not None:
        P["ref: Escalón 3 (0,6663)"] = (PC["o2"], PC["v2"])
    else:
        P["ref: Escalón 3 (0,6663)"] = evaluar(D, esc3())

    VAR = {
        "+ counters + sinergias (MATCHUPS)": esc3(MPR["MATCHUPS"]),
        "+ composición draft (25 cols)": esc3(G["COMP"]),
        "+ Poisson (kills/oro/daño/g15)": esc3(G["POIS"]),
        "+ MATCHUPS + composición": esc3(np.c_[MPR["MATCHUPS"], G["COMP"]]),
        "+ MATCHUPS + composición + Poisson": esc3(np.c_[MPR["MATCHUPS"], G["COMP"], G["POIS"]]),
    }
    for n in VAR:
        P[n] = evaluar(D, VAR[n])
    mostrar(comparar(D, P, "ref: Escalón 3 (0,6663)"))
    np.savez_compressed(
        REPO / "data" / "analysis" / "brainstorm" / "pred_g06.npz",
        **{f"o_{i}": P[n][0] for i, n in enumerate(P)},
        **{f"v_{i}": P[n][1] for i, n in enumerate(P)},
        names=np.array(list(P)),
    )


if __name__ == "__main__":
    main()
