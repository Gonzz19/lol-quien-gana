"""Ideas 6 y 14 del brainstorming (docs/BRAINSTORM_NOCHE.md): señales causales faltantes en `Xj`
desde `full_dataset.csv` (timeline @10/@20, escalado 10->20, volatilidad, early-FF/tilt, y margen
ajustado por rival de línea `prior_margin_vs_rival`), sobre la referencia vigente
(solo Master+ + MC + ML = 0,6614).

En `variables.npz` (`Xj`), `prior_margin` solo se calculó sobre 15 stats: no tiene `d_xp_10`,
`d_damage_10`, `d_cs_10`, `d_kda_10`, los diffs @20 (`d_gold_20`, `d_xp_20`, `d_damage_20`,
`scale_20_10 = d_gold_20 - d_gold_10`), ni la volatilidad del jugador en `d_gold_15` (r = 0,684 en
§9.S.35), ni historial de rendición temprana <= 960s (`p = 1,8e-5` en fit), ni `first_blood`/
`first_tower`, ni `prior_margin_vs_rival` (`margin.py`).

Cuidado de fuga: `lane_scale` se estima EXCLUSIVAMENTE sobre partidas `< 16/09` (`~val0`).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.brainstorm_base import ajustar_predecir, cargar, comparar, evaluar, mostrar  # noqa: E402
from src.features.margin import _previo, _residuo, lane_diffs, lane_scale, prior_margin, prior_margin_vs_rival  # noqa: E402

CACHE = REPO / "data" / "analysis" / "brainstorm" / "gemini_timeline.npz"


def construir_cache(D: dict) -> None:
    NM = D["NM"]
    fit0_mids = set(D["Z"]["mids"][~D["val0"]])
    USE = [
        "match_id", "game_creation", "game_duration", "player_id", "team_id", "win",
        "champion_id", "lane", "first_blood_kill", "first_tower_kill",
        "gold_10", "xp_10", "cs_10", "damage_10", "kills_10", "deaths_10", "assists_10",
        "gold_15", "xp_15", "cs_15", "damage_15",
        "gold_20", "xp_20", "cs_20", "damage_20",
    ]
    df = pd.read_csv(REPO / "data" / "splits" / "train" / "full_dataset.csv", usecols=USE)
    LANES = ["BOTTOM", "JUNGLE", "MIDDLE", "TOP", "UTILITY"]
    df = df[df.lane.isin(LANES)]
    df = df[df.groupby(["match_id", "team_id"]).lane.transform("nunique") == 5]
    df = df[df.groupby("match_id").lane.transform("size") == 10]
    df = df[df.groupby("match_id").win.transform("sum") == 5].copy()
    df = df.sort_values(["game_creation", "match_id", "team_id", "lane"]).reset_index(drop=True)
    assert (df.match_id.to_numpy()[::10] == D["Z"]["mids"]).all()

    df["kda_10"] = df["kills_10"] + 0.5 * df["assists_10"] - df["deaths_10"]
    df["damage_15_clean"] = np.where(df["game_duration"] > 900, df["damage_15"], np.nan)
    for c in ("gold_20", "xp_20", "cs_20", "damage_20"):
        df[c] = np.where(df["game_duration"] > 1200, df[c], np.nan)

    diff_cols = ["xp_10", "damage_10", "cs_10", "kda_10", "gold_10", "gold_15", "xp_15", "damage_15_clean",
                 "gold_20", "xp_20", "cs_20", "damage_20"]
    df = lane_diffs(df, diff_cols)
    df["scale_20_10"] = df["d_gold_20"] - df["d_gold_10"]
    df["eff_loss"] = ((df["game_duration"] <= 960) & (df["win"] == 0)).astype(float)
    df["fb"] = df["first_blood_kill"].fillna(0).astype(float)
    df["ft"] = df["first_tower_kill"].fillna(0).astype(float)

    NEW_STATS = ["d_xp_10", "d_damage_10", "d_cs_10", "d_kda_10", "d_damage_15_clean",
                 "d_gold_20", "d_xp_20", "d_cs_20", "d_damage_20", "scale_20_10"]
    ADJ_STATS = ["d_gold_15", "d_gold_10", "d_xp_15", "d_damage_10"]
    FIT_MASK = df["match_id"].isin(fit0_mids)

    esc_new = lane_scale(df[FIT_MASK], NEW_STATS)
    esc_adj = lane_scale(df[FIT_MASK], ADJ_STATS)

    PM = prior_margin(df, NEW_STATS, k_cl=20, k_j=5, window=20, escala=esc_new)
    ADJ = prior_margin_vs_rival(df, ADJ_STATS, k_cl=20, k_j=5, window=20, escala=esc_adj)

    # Volatilidad causal de d_gold_15 (std previa encogida hacia el prior empírico de fit)
    r_g15, v_g15 = _residuo(df, "d_gold_15", 20, esc_adj)
    pj = df["player_id"]
    S1 = _previo(r_g15, pj)
    S2 = _previo(r_g15 * r_g15, pj)
    N1 = _previo(v_g15, pj)
    var0 = float(r_g15[FIT_MASK & (v_g15 > 0.5)].var())
    m_prev = (S1) / (N1 + 5.0)
    # Exceso cuadrático respecto a media previa encogida con k=10 hacia var0
    var_g15 = np.maximum(0.0, (S2 + 10.0 * var0) / (N1 + 10.0) - (S1 / (N1 + 10.0)) ** 2)
    vol_g15 = np.sqrt(var_g15).astype(np.float32)

    # Tasas causales encogidas de early_ff_loss, first_blood, first_tower
    n_tot = df.groupby("player_id", sort=False).cumcount().astype(float)
    eff_r = ((_previo(df["eff_loss"], pj) + 10.0 * 0.0235) / (n_tot + 10.0)).to_numpy(np.float32)
    fb_r = ((_previo(df["fb"], pj) + 10.0 * 0.10) / (n_tot + 10.0)).to_numpy(np.float32)
    ft_r = ((_previo(df["ft"], pj) + 10.0 * 0.10) / (n_tot + 10.0)).to_numpy(np.float32)
    BEH = np.c_[vol_g15, eff_r, fb_r, ft_r].astype(np.float32)

    def to_diff(mat: np.ndarray) -> np.ndarray:
        a = mat.reshape(NM, 2, 5, -1)
        return (a[:, 0] - a[:, 1]).astype(np.float32)

    np.savez_compressed(
        CACHE,
        PM_lin=to_diff(PM.to_numpy(np.float32)).reshape(NM, -1),
        PM_eq=to_diff(PM.to_numpy(np.float32)).mean(axis=1),
        ADJ_lin=to_diff(ADJ.to_numpy(np.float32)).reshape(NM, -1),
        ADJ_eq=to_diff(ADJ.to_numpy(np.float32)).mean(axis=1),
        BEH_lin=to_diff(BEH).reshape(NM, -1),
        BEH_eq=to_diff(BEH).mean(axis=1),
    )


def main():
    D = cargar()
    ms = D["master"]
    if not CACHE.exists():
        print("Construyendo caché gemini_timeline.npz...")
        construir_cache(D)
    T = np.load(CACHE)
    A_mmr = np.load(REPO / "data" / "analysis" / "brainstorm" / "mmr_historico.npz")["A"]
    NIV_EQ = (A_mmr[:, 0] - A_mmr[:, 1])[..., :3].mean(axis=1).astype(np.float32)
    C = np.load(REPO / "data" / "analysis" / "brainstorm" / "margen_grupos.npz")
    MC_ML_NIV = np.c_[
        (C["MC"][:, 0] - C["MC"][:, 1]).reshape(D["NM"], -1),
        (C["ML"][:, 0] - C["ML"][:, 1]).reshape(D["NM"], -1),
        NIV_EQ,
    ].astype(np.float32)

    def solo_m(extra=None):
        tot = MC_ML_NIV if extra is None else np.c_[MC_ML_NIV, extra].astype(np.float32)
        return lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=tot, extra_post=False)

    VAR = {
        "ref: solo Master+ + MC/ML + niv_hist": solo_m(),
        "+ timeline @10/@20 (equipo)": solo_m(T["PM_eq"]),
        "+ timeline @10/@20 (línea)": solo_m(T["PM_lin"]),
        "+ rival adj (equipo)": solo_m(T["ADJ_eq"]),
        "+ vol + earlyFF + FB/FT (línea)": solo_m(T["BEH_lin"]),
        "+ vol + earlyFF + @10/@20 (eq+beh)": solo_m(np.c_[T["PM_eq"], T["BEH_lin"], T["ADJ_eq"]]),
    }
    P = dict(Parallel(n_jobs=1, backend="threading")(delayed(lambda n: (n, evaluar(D, VAR[n])))(n) for n in VAR))
    mostrar(comparar(D, P, "ref: solo Master+ + MC/ML + niv_hist"))


if __name__ == "__main__":
    main()
