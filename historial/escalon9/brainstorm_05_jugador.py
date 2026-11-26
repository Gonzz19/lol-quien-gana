"""Brainstorming (docs/BRAINSTORM_NOCHE.md), ideas 10 y 11, sobre híbrido + mapa entrenado solo con Master+.

10. Margen por (jugador, campeón) y por (jugador, línea) (§9.S.29: +0,0069 y +0,0033 en el modelo
    del compañero; no está en el híbrido de esta máquina). `prior_margin_by`, k_g = 10, 15
    estadísticas, azul − rojo por línea.
11. Fuerza propia de cada jugador aprendida de los resultados: one-hot de jugador con signo
    (+1 azul, −1 rojo), estilo Bradley-Terry, vocabulario = jugadores del entrenamiento con
    ≥ 20 partidas. La regularización L2 de la logística la encoge. Es un "rating" que se ajusta
    con todo el entrenamiento a la vez: para las filas de entrenamiento ve sus propios resultados,
    pero para V solo usa partidas anteriores (sin fuga hacia V).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.brainstorm_base import ajustar_predecir, cargar, comparar, evaluar, mostrar  # noqa: E402
from src.features.fingerprint import add_rates  # noqa: E402
from src.features.margin import lane_diffs, lane_scale, prior_margin_by  # noqa: E402

CACHE = REPO / "data" / "analysis" / "brainstorm" / "margen_grupos.npz"
STATS = ["kills_pm", "assists_pm", "damage_pm", "damage_share_pct", "kill_participation_pct",
         "deaths_pm", "damage_taken_pm", "gold_pm", "cs_pm", "gold_share_pct", "vision_pm",
         "d_gold_10", "d_gold_15", "d_xp_15", "d_cs_15"]


def construir_cache(D: dict, rehacer: bool = False) -> None:
    if CACHE.exists() and not rehacer:
        return
    NM = D["NM"]
    USE = ["match_id", "game_creation", "game_duration", "player_id", "team_id", "win", "champion_id", "lane",
           "kills", "deaths", "assists", "kill_participation_pct", "damage_share_pct", "gold_share_pct",
           "total_damage_dealt_to_champions", "total_damage_taken", "gold_earned", "total_minions_killed",
           "neutral_minions_killed", "vision_score", "gold_10", "gold_15", "xp_15", "cs_15"]
    df = pd.read_csv(REPO / "data" / "splits" / "train" / "full_dataset.csv", usecols=USE)
    df = df[df.lane.isin(["BOTTOM", "JUNGLE", "MIDDLE", "TOP", "UTILITY"])]
    df = df[df.groupby(["match_id", "team_id"]).lane.transform("nunique") == 5]
    df = df[df.groupby("match_id").lane.transform("size") == 10]
    df = df[df.groupby("match_id").win.transform("sum") == 5].copy()
    df = add_rates(df).sort_values(["game_creation", "match_id", "team_id", "lane"]).reset_index(drop=True)
    df = lane_diffs(df, ["gold_10", "gold_15", "xp_15", "cs_15"])
    assert (df.match_id.to_numpy()[::10] == D["Z"]["mids"]).all()
    # Escala por línea congelada estrictamente antes de V (~val0: sin V Master+ ni Diamond-V)
    esc = lane_scale(df[np.repeat(~D["val0"], 10)], STATS)
    MC = prior_margin_by(df, STATS, ["champion_id"], "mc", k_g=10, escala=esc)
    ML = prior_margin_by(df, STATS, ["lane"], "ml", k_g=10, escala=esc)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, MC=MC.to_numpy(np.float32).reshape(NM, 2, 5, -1),
                        ML=ML.to_numpy(np.float32).reshape(NM, 2, 5, -1))


def main():
    rehacer = "--rehacer" in sys.argv
    solo_cache = "--solo-cache" in sys.argv
    D = cargar()
    NM, ms = D["NM"], D["master"]
    construir_cache(D, rehacer=rehacer)
    if solo_cache:
        print(f"[+] {CACHE}")
        return
    C = np.load(CACHE)
    MC = (C["MC"][:, 0] - C["MC"][:, 1]).reshape(NM, -1)
    ML = (C["ML"][:, 0] - C["ML"][:, 1]).reshape(NM, -1)

    # ---- one-hot de jugador con signo
    pl = D["B"]["player"]
    cods, inv = np.unique(pl.reshape(-1), return_inverse=True)
    inv = inv.reshape(NM, 2, 5)

    def con_jugadores(tr, te, extra=None, min_n=20, C_j=0.05):
        tr = tr[ms[tr]]
        cnt = np.bincount(inv[tr].reshape(-1), minlength=len(cods))
        keep = np.flatnonzero(cnt >= min_n)
        pos = -np.ones(len(cods), int)
        pos[keep] = np.arange(len(keep))

        def onehot(idx):
            r, c, v = [], [], []
            for s, sg in ((0, 1.0), (1, -1.0)):
                p = pos[inv[idx, s]]                         # (n, 5)
                rr, ll = np.nonzero(p >= 0)
                r.append(rr); c.append(p[rr, ll]); v.append(np.full(len(rr), sg))
            return sparse.csr_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=(len(idx), len(keep)))
        # el one-hot de jugadores entra como "extra" disperso: se arma la matriz base y se apila
        from scripts.analysis.brainstorm_base import ajustar_predecir as _ap  # noqa
        Xd, post, mapa = D["Xd"], D["post"], D["mapa"]
        mu, sd = Xd[tr].mean(0), Xd[tr].std(0) + 1e-6
        ref = tr[post[tr]]
        me, se = mapa[ref].mean(0), mapa[ref].std(0) + 1e-6

        def armar(idx):
            P = [D["DRAFT"][idx], sparse.csr_matrix((Xd[idx] - mu) / sd),
                 sparse.csr_matrix(np.where(post[idx][:, None], (mapa[idx] - me) / se, 0).astype(np.float32)), onehot(idx)]
            if extra is not None:
                ee = extra
                m2, s2 = ee[tr].mean(0), ee[tr].std(0) + 1e-6
                P.append(sparse.csr_matrix(((ee[idx] - m2) / s2).astype(np.float32)))
            return sparse.hstack(P).tocsr()
        lr = LogisticRegression(C=C_j, max_iter=4000).fit(armar(tr), D["y"][tr])
        return lr.predict_proba(armar(te))[:, 1]

    def solo_m(extra=None):
        return lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=extra, extra_post=False)

    VAR = {
        "referencia: solo Master+": solo_m(),
        "+ margen por campeón": solo_m(MC),
        "+ margen por línea": solo_m(ML),
        "+ margen por campeón y por línea": solo_m(np.c_[MC, ML]),
        "+ fuerza por jugador (one-hot, ≥ 20 partidas)": lambda tr, te: con_jugadores(tr, te),
    }
    P = dict(Parallel(n_jobs=3, backend="threading")(delayed(lambda n: (n, evaluar(D, VAR[n])))(n) for n in VAR))
    mostrar(comparar(D, P, "referencia: solo Master+"))


if __name__ == "__main__":
    main()
