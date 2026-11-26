"""Maestro con información privilegiada y agregados de equipo para quién gana (§9.S.39).

1. MAESTRO: logística con las variables pre-partida livianas (campeones, experiencia, cobertura, margen
   y winrate previos) MÁS las diferencias azul - rojo por línea de los snapshots @10/@15 (oro, xp, cs, daño,
   kills, muertes). Sus probabilidades son la etiqueta suave del alumno (`ganador_red_jugador.py
   --maestro`): q = a*y + (1-a)*p. Para no filtrar el resultado, cada ajuste del alumno (bloques 1-4
   de --oos 5 y el ajuste completo para V) recibe predicciones del maestro hechas con validación
   cruzada en 3 pliegues por fecha DENTRO de ese mismo ajuste: el maestro nunca vio la fila que
   etiqueta ni las partidas que el alumno va a predecir.
   Salida: ganador_maestro.npz con `idx`, `bloque` (los de config C) y `p_V`, `p_1`..`p_4`
   (largo de idx; NaN fuera del ajuste de cada tarea).
2. AGREGADOS: máximo, mínimo y desvío dentro de cada equipo (azul - rojo) de 8 variables previas del
   jugador. Salida: ganador_agregados.pkl en el formato `nuevo` de --extra de ganador_logistica_oos.py.

El orden de partidas se reconstruye como en ganador_features_grande.py y se verifica contra
--referencia (un _oos.npz de config C). Solo data/splits/train."""
import argparse, pickle, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
AN = REPO / "data" / "analysis"
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default=str(REPO / "data" / "splits" / "train" / "full_dataset.csv"))
ap.add_argument("--referencia", default=str(AN / "ganador_logistica_composicion_oos.npz"))
ap.add_argument("--salida", default=str(AN))
ap.add_argument("--src", default=str(REPO), help="raíz desde donde importar src/")
args = ap.parse_args()
sys.path.insert(0, args.src)
from src.features.fingerprint import add_rates
from src.features.margin import lane_diffs, prior_margin, prior_winrate

COLS = ["match_id", "game_creation", "game_duration", "player_id", "team_id", "win", "champion_id", "lane",
        "kills", "deaths", "assists", "kill_participation_pct", "damage_share_pct", "gold_share_pct",
        "total_damage_dealt_to_champions", "total_damage_taken", "gold_earned", "total_minions_killed",
        "neutral_minions_killed", "vision_score"] + [f"{c}_{m}" for m in (10, 15)
                                                     for c in ("gold", "xp", "cs", "damage", "kills", "deaths")]
LANES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
df = pd.read_csv(args.dataset, usecols=COLS)
df = df[df.lane.isin(LANES)]
df = df[df.groupby(["match_id", "team_id"]).lane.transform("nunique") == 5]
df = df[df.groupby("match_id").lane.transform("size") == 10].copy()
df["s"] = np.where(df.team_id == 100, 1, -1)
df = add_rates(df).sort_values(["game_creation", "match_id", "team_id", "lane"]).reset_index(drop=True)
df = lane_diffs(df, ["gold_10", "gold_15", "xp_10", "xp_15", "cs_10", "cs_15", "damage_10", "damage_15",
                     "kills_10", "kills_15", "deaths_10", "deaths_15"])

# ---- pre-partida (sin futuro): experiencia, cobertura, margen y winrate previos
gp = df.groupby("player_id")
df["n_tot"] = gp.cumcount()
gc = df.groupby(["player_id", "champion_id"])
df["n_champ"] = gc.cumcount()
df["w_champ"] = gc.win.cumsum() - df.win
df["w_tot"] = gp.win.cumsum() - df.win
df["n_lane"] = df.groupby(["player_id", "lane"]).cumcount()
df["x_logn_champ"] = np.log1p(df.n_champ)
df["x_estreno"] = (df.n_champ == 0).astype(float)
df["x_wr_champ"] = (df.w_champ + 5 * (df.w_tot + 5) / (df.n_tot + 10)) / (df.n_champ + 5) - (df.w_tot + 5) / (df.n_tot + 10)
df["x_share_champ"] = df.n_champ / (df.n_tot + 1)
df["x_share_lane"] = (df.n_lane + 1) / (df.n_tot + 2)
df["x_offrole"] = ((df.n_tot >= 10) & (df.n_lane / df.n_tot.clip(lower=1) < 0.2)).astype(float)
df["x_logn_tot"] = np.log1p(df.n_tot)
PRE = ["x_logn_champ", "x_estreno", "x_wr_champ", "x_share_champ", "x_share_lane", "x_offrole", "x_logn_tot"]
STATS = ["kills_pm", "assists_pm", "damage_pm", "damage_share_pct", "kill_participation_pct", "deaths_pm",
         "damage_taken_pm", "gold_pm", "cs_pm", "gold_share_pct", "vision_pm", "d_gold_10", "d_gold_15", "d_xp_15", "d_cs_15"]
J = pd.concat([prior_margin(df, STATS, 20, 5, 20), prior_winrate(df, 20)], axis=1)
MARG = list(J.columns)
df = pd.concat([df, J], axis=1)
SNAP = [f"d_{c}_{m}" for m in (10, 15) for c in ("gold", "xp", "cs", "damage", "kills", "deaths")]
for c in SNAP:
    df[c] = df[c].fillna(0)

# ---- orden de partidas como ganador_features_grande.py
mt = df.groupby("match_id").game_creation.first()
mids = mt.sort_values().index.to_numpy().copy()
df = df.sort_values(["match_id", "lane"]).reset_index(drop=True)
row = pd.Series(np.arange(len(mids)), index=mids)[df.match_id].to_numpy()
lane_i = df.lane.map({l: i for i, l in enumerate(LANES)}).to_numpy()
s = df.s.to_numpy()


def por_linea(X):
    d = X.shape[1]
    out = np.zeros((len(mids), 6 * d), dtype=np.float32)
    for li in range(5):
        sel = lane_i == li
        np.add.at(out[:, li * d:(li + 1) * d], row[sel], X[sel] * s[sel, None])
    np.add.at(out[:, 5 * d:], row, X * s[:, None])
    return out


champs = {c: i for i, c in enumerate(sorted(df.champion_id.unique()))}
onehot = np.zeros((len(mids), len(champs)), dtype=np.float32)
np.add.at(onehot, (row, df.champion_id.map(champs).to_numpy()), s)
Xpre = np.hstack([onehot, por_linea(df[PRE].to_numpy()), por_linea(df[MARG].to_numpy())])
Xsnap = por_linea(df[SNAP].to_numpy())
y = df[df.s == 1].groupby("match_id").win.first().reindex(mids).to_numpy().copy()
t = mt.reindex(mids).to_numpy().copy()

f = np.load(args.referencia)
vi, oi = f["C_val_idx"], f["C_oos_idx"]
assert (y[vi] == f["C_val_y"]).all() and (t[vi] == f["C_val_t"]).all(), "V no coincide"
mm = oi[np.where(y[oi] != f["C_oos_y"])[0]]
for a, b in zip(mm[::2], mm[1::2]):          # empates de game_creation: el sort de pandas 2 vs 3 los ordena distinto
    assert b == a + 1 and t[a] == t[b]
    for M in (Xpre, Xsnap, y): M[[a, b]] = M[[b, a]]
assert (y[oi] == f["C_oos_y"]).all()
print("empates corregidos:", len(mm) // 2)
print("orden verificado:", len(mids), "partidas; Xpre", Xpre.shape, "Xsnap", Xsnap.shape)
AGG = ["acum_d_gold_15", "acum_gold_pm", "acum_damage_pm", "acum_deaths_pm", "acum_kill_participation_pct",
       "wr_acum", "x_wr_champ", "x_logn_champ"]
P10 = np.zeros((len(mids), 10, len(AGG)))
P10[row, 5 * (s == -1) + lane_i] = df[AGG].to_numpy()
for a, b in zip(mm[::2], mm[1::2]):
    P10[[a, b]] = P10[[b, a]]
    mids[[a, b]] = mids[[b, a]]
az, ro = P10[:, :5], P10[:, 5:]
nuevo = {"máx": az.max(1) - ro.max(1), "mín": az.min(1) - ro.min(1), "desvío": az.std(1) - ro.std(1)}
nuevo = {k: v.astype(np.float32) for k, v in nuevo.items()}
pickle.dump({"mids": mids, "nuevo": nuevo, "columnas": AGG}, open(Path(args.salida) / "ganador_agregados.pkl", "wb"), protocol=4)

# ---------- maestro
Xt = np.hstack([Xpre, Xsnap]).astype(np.float64)
vi = f["C_val_idx"]
val = np.zeros(len(y), bool); val[vi] = True
idx = np.where(~val)[0]
bloque = np.searchsorted(np.quantile(t[idx], np.linspace(0, 1, 6)[1:-1]), t[idx], side="right")
assert (idx[bloque >= 1] == f["C_oos_idx"]).all() and (bloque[bloque >= 1] == f["C_oos_bloque"]).all()


def oof(sel):
    """predicción del maestro para las filas idx[sel], con 3 pliegues por fecha dentro de idx[sel]"""
    fit = idx[sel]
    p = np.full(len(idx), np.nan)
    pos = np.where(sel)[0]
    for pl in np.array_split(np.arange(len(fit)), 3):
        resto = np.setdiff1d(np.arange(len(fit)), pl)
        sc = StandardScaler(with_mean=False).fit(Xt[fit[resto]])
        p[pos[pl]] = LogisticRegression(C=0.05, max_iter=3000).fit(sc.transform(Xt[fit[resto]]), y[fit[resto]]) \
            .predict_proba(sc.transform(Xt[fit[pl]]))[:, 1]
    return p


t0 = time.time()
salida = {"idx": idx, "bloque": bloque}
for k in range(1, 5):
    salida[f"p_{k}"] = oof(bloque < k)
    print(f"maestro bloque {k}: AUC OOF {roc_auc_score(y[idx[bloque < k]], salida[f'p_{k}'][bloque < k]):.4f} ({time.time() - t0:.0f}s)", flush=True)
salida["p_V"] = oof(np.ones(len(idx), bool))
print(f"maestro V: AUC OOF {roc_auc_score(y[idx], salida['p_V']):.4f} ({time.time() - t0:.0f}s)")
np.savez(Path(args.salida) / "ganador_maestro.npz", **salida)
print("escritos ganador_maestro.npz y ganador_agregados.pkl")
