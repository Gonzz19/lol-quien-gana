"""Baselines de identificación (src/models/baselines/identificacion.py, PROJECT_SUMMARY.md §9.S.20)."""

import numpy as np
import pandas as pd

from src.features.fingerprint import evaluate, make_halves, profiles, residualize
from src.models.baselines.identificacion import embedding_profiles, load_embeddings, pool_profiles

FEATS = ["f1", "f2", "f3"]


def _tipos(n_tipos=6, dim=4, seed=0):
    """Pares de campeones casi idénticos en el embedding (como Jinx y Caitlyn)."""
    rng = np.random.default_rng(seed)
    vecs = {}
    for t in range(n_tipos):
        base = rng.normal(0, 3, dim)
        vecs[(f"T{t}a", "BOTTOM")] = base
        vecs[(f"T{t}b", "BOTTOM")] = base + rng.normal(0, 0.05, dim)
    return vecs, {"BOTTOM": np.zeros(dim)}


def test_embedding_reconoce_campeones_parecidos_y_pool_no():
    # Cada jugador elige un "tipo" de campeón: en A juega uno del par y en B el otro.
    vecs, lane_vecs = _tipos()
    rows, halves = [], {}
    for p in range(60):
        t = p % 6
        a = [len(rows) + i for i in range(10)]
        rows += [{"player_id": f"p{p}", "champion_name": f"T{t}a", "lane": "BOTTOM"}] * 10
        b = [len(rows) + i for i in range(10)]
        rows += [{"player_id": f"p{p}", "champion_name": f"T{t}b", "lane": "BOTTOM"}] * 10
        halves[f"p{p}"] = {"A": np.array(a), "B": np.array(b)}
    df = pd.DataFrame(rows)

    _, A, B = embedding_profiles(df, halves, vecs, lane_vecs)
    assert evaluate(A, B, n_perm=100, n_boot=20)["auc"] > 0.85
    # Sin campeones en común, la frecuencia de campeones no tiene nada que comparar.
    _, A, B = pool_profiles(df, halves)
    assert evaluate(A, B, metric="cosine", n_perm=100, n_boot=20)["auc"] < 0.6


def test_embedding_no_ve_el_estilo_con_los_mismos_campeones():
    # Todos juegan el mismo pool; solo cambia el estilo. La huella lo ve, el embedding no.
    rng = np.random.default_rng(1)
    champs = ["C0", "C1", "C2", "C3"]
    vecs = {(c, "MIDDLE"): rng.normal(0, 3, 4) for c in champs}
    rows = []
    for p in range(60):
        s = rng.normal(0, 1, len(FEATS))
        for g in range(60):
            c = champs[rng.integers(len(champs))]
            rows.append({"player_id": f"p{p}", "champion_name": c, "lane": "MIDDLE", "win": g % 2,
                         "game_creation": g, **dict(zip(FEATS, s + rng.normal(0, 1, len(FEATS))))})
    df = residualize(pd.DataFrame(rows), FEATS, min_group=10)
    halves = make_halves(df, k=15, mode="aleatoria", seed=2)

    _, A, B = embedding_profiles(df, halves, vecs, {"MIDDLE": np.zeros(4)})
    assert abs(evaluate(A, B, n_perm=100, n_boot=20)["auc"] - 0.5) < 0.1
    _, A, B = profiles(df, halves, FEATS)
    assert evaluate(A, B, n_perm=100, n_boot=20)["auc"] > 0.9


def test_par_sin_embedding_usa_el_vector_de_su_linea():
    vecs = {("Jinx", "BOTTOM"): np.array([1.0, 1.0])}
    lane_vecs = {"BOTTOM": np.array([5.0, -5.0])}
    df = pd.DataFrame({"champion_name": ["Jinx", "Raro"], "lane": ["BOTTOM", "BOTTOM"]})
    _, A, B = embedding_profiles(df, {"p": {"A": np.array([0]), "B": np.array([1])}}, vecs, lane_vecs)
    np.testing.assert_allclose(A[0], [1.0, 1.0])
    np.testing.assert_allclose(B[0], [5.0, -5.0])


def test_el_embedding_sale_solo_de_la_tabla_que_se_le_pasa(tmp_path):
    # Solo entran los pares con >= 100 partidas de esa tabla (la del train, sin leakage).
    from src.features.player_signatures import EMB_FEATURES
    rng = np.random.default_rng(0)
    filas = []
    for i, (champ, n) in enumerate([("Ahri", 500), ("Zed", 300), ("Lux", 50)]):
        filas.append({"champion_name": champ, "lane": "MIDDLE", "games_played": n,
                      **{f: rng.normal(i, 1) for f in EMB_FEATURES}})
    path = tmp_path / "champions_by_lane.csv"
    pd.DataFrame(filas).to_csv(path, index=False)
    vecs, lane_vecs = load_embeddings(path)
    assert set(vecs) == {("Ahri", "MIDDLE"), ("Zed", "MIDDLE")}
    assert set(lane_vecs) == {"MIDDLE"}
