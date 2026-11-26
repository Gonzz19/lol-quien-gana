"""Idea 16 del brainstorming (docs/BRAINSTORM_NOCHE.md): MMR por emparejamiento con TODO el historial.

`prior_lobby_level` (§9.S.31, en el modelo del compañero; no en el de esta máquina) con x = LP de la
foto 1 de los otros 9, sobre todas las partidas previas del jugador (no solo desde el 04/09, como
el "nivel (mapa)"). Para V es causal (la foto es anterior a V y solo se usan partidas previas).
Las partidas anteriores a la foto (t < t1, el 84 % del entrenamiento) no pueden usar ese LP, que se
midió después: ahí las columnas valen el prior (0 = sin foto ni historial), como el W/L de la foto en
`brainstorm_g_01` (issue #4, `PROJECT_SUMMARY.md` §9.S.75).

Variantes sobre la referencia (híbrido + mapa + margen por campeón y línea, solo Master+):
* + nivel histórico (acumulado, últimas 10, log n; azul − rojo, media por equipo y por línea);
* + nivel histórico + brecha con el LP de la foto propio (rezago del LP respecto del MMR).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.brainstorm_base import REPO, ajustar_predecir, cargar, comparar, evaluar, mostrar  # noqa: E402
import scripts.analysis.ganador_mapa_lp as gm  # noqa: E402
from src.features.lobby import prior_lobby_level  # noqa: E402

CACHE = REPO / "data" / "analysis" / "brainstorm" / "mmr_historico.npz"


def construir_cache(D: dict, rehacer: bool = False) -> None:
    if CACHE.exists() and not rehacer:
        return
    NM = D["NM"]
    foto1, t1 = gm._cargar_foto1(gm.RUTA_DB)
    pl = D["B"]["player"].reshape(-1)
    df = pd.DataFrame({"match_id": np.repeat(D["Z"]["mids"], 10), "game_creation": np.repeat(D["t"], 10),
                       "team_id": np.tile(np.repeat([100, 200], 5), NM), "lane": np.tile(np.arange(5), 2 * NM),
                       "player_id": pl})
    df["x"] = df.player_id.map(foto1["z"].to_dict())
    # Prior mu0 congelado estrictamente antes de V (~val0: sin V Master+ ni Diamond-V)
    mu0 = float(df.loc[np.repeat(~D["val0"], 10), "x"].mean())
    N = prior_lobby_level(df, "x", k=5, mu0=mu0, window=10, min_otros=3)
    propio = df.x.fillna(mu0).to_numpy()
    A = np.c_[N.nivel_acum - mu0, N.nivel_v10 - mu0, np.log1p(N.n_nivel), N.nivel_v10.to_numpy() - propio,
              df.x.notna().astype(float)].astype(np.float32)
    # Máscara temporal (issue #4): antes de la foto 1 su LP es posterior a la partida. Prior = 0 en
    # todas las columnas (nivel = mu0, n = 0, sin foto); para t >= t1 el LP de los rivales de partidas
    # previas, aun anteriores a t1, ya se conocía: esas filas no cambian.
    A[np.repeat(D["t"] < t1, 10)] = 0.0
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, A=A)


def main():
    rehacer = "--rehacer" in sys.argv
    solo_cache = "--solo-cache" in sys.argv
    D = cargar()
    NM, ms = D["NM"], D["master"]
    construir_cache(D, rehacer=rehacer)
    if solo_cache:
        print(f"[+] {CACHE}")
        return
    C = np.load(REPO / "data" / "analysis" / "brainstorm" / "margen_grupos.npz")
    MCML = np.c_[(C["MC"][:, 0] - C["MC"][:, 1]).reshape(NM, -1), (C["ML"][:, 0] - C["ML"][:, 1]).reshape(NM, -1)]
    A = np.load(CACHE)["A"]
    dif = A[:, 0] - A[:, 1]
    NIV_EQ = dif[..., :3].mean(axis=1)
    NIV_BR_EQ = dif.mean(axis=1)
    NIV_LIN = np.c_[dif[..., :3].reshape(NM, -1), NIV_EQ]

    def ref(extra=None):
        X = MCML if extra is None else np.c_[MCML, extra]
        return lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=X.astype(np.float32), extra_post=False)

    VAR = {"referencia: + margen por campeón y línea": ref(), "+ nivel histórico (equipo)": ref(NIV_EQ),
           "+ nivel histórico + brecha (equipo)": ref(NIV_BR_EQ), "+ nivel histórico (por línea)": ref(NIV_LIN)}
    P = dict(Parallel(n_jobs=2, backend="threading")(delayed(lambda n: (n, evaluar(D, VAR[n])))(n) for n in VAR))
    mostrar(comparar(D, P, "referencia: + margen por campeón y línea"))


if __name__ == "__main__":
    main()
