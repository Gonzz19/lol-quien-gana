"""Regresión logística del draft (src/models/baselines/draft_logreg.py, PROJECT_SUMMARY.md §9.S.20)."""

import numpy as np
import pandas as pd

from src.models.baselines.draft_logreg import (complete_match_ids, draft_matrix, duel_matrix,
                                               fit_temporal_C, lane_duels, match_target,
                                               prob_metrics, temporal_holdout)

LANES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]


def _match(mid, blue, red, blue_win, blue_gold=6000, red_gold=5000):
    rows = []
    for team, champs, gold in ((100, blue, blue_gold), (200, red, red_gold)):
        win = blue_win if team == 100 else 1 - blue_win
        for lane in LANES:
            rows.append({"match_id": mid, "team_id": team, "win": win,
                         "champion_name": champs[lane], "lane": lane, "gold_15": gold})
    return rows


def _resto(top):
    return {"TOP": top, "JUNGLE": "J", "MIDDLE": "M", "BOTTOM": "B", "UTILITY": "U"}


def test_matriz_antisimetrica_y_vocabulario_del_train():
    parts = pd.DataFrame(_match("M1", _resto("Darius"), _resto("Garen"), 1)
                         + _match("M2", _resto("Garen"), _resto("Darius"), 0)
                         + _match("M3", _resto("Nuevo"), _resto("Garen"), 1))
    X, vocab = draft_matrix(parts, ["M1", "M2"])
    assert "Nuevo|TOP" not in vocab
    X = X.toarray()
    np.testing.assert_allclose(X[0], -X[1])                   # cambiar de lado invierte el signo
    X3, _ = draft_matrix(parts, ["M3"], vocab=vocab)
    assert X3.toarray()[0, vocab.get_loc("Garen|TOP")] == -1   # el campeón no visto queda en cero
    # Las otras 4 líneas tienen el mismo campeón de los dos lados: +1 y −1 se anulan.
    assert np.count_nonzero(X3.toarray()) == 1
    Xs, _ = draft_matrix(parts, ["M1"], vocab=vocab, sides=True)
    assert Xs.shape[1] == 2 * len(vocab) and Xs.toarray().min() == 0


def test_aprende_al_campeon_fuerte():
    rng = np.random.default_rng(0)
    rows, creation = [], []
    for i in range(400):
        fuerte_azul = rng.random() < 0.5
        gana_fuerte = rng.random() < 0.8
        blue, red = (_resto("Fuerte"), _resto("Debil")) if fuerte_azul else (_resto("Debil"), _resto("Fuerte"))
        rows += _match(f"M{i}", blue, red, int(fuerte_azul == gana_fuerte))
        creation.append(i)
    parts = pd.DataFrame(rows)
    ids = pd.Index([f"M{i}" for i in range(400)])
    X, vocab = draft_matrix(parts, ids)
    y = match_target(parts, ids)
    tr = np.arange(300)
    m, C, curva = fit_temporal_C(X, y, tr, np.array(creation))
    assert set(curva) and C in curva
    w = pd.Series(m.coef_[0], index=vocab)
    assert w["Fuerte|TOP"] > 0 > w["Debil|TOP"]
    assert prob_metrics(y[300:], m.predict_proba(X[300:])[:, 1], n_boot=50, n_boot_auc=20)["accuracy"]["valor"] > 0.7


def test_la_validacion_es_el_tramo_mas_reciente_del_train():
    creation = np.array([5, 1, 4, 2, 3, 9, 8])
    fit, val = temporal_holdout(np.arange(5), creation, val_frac=0.4)
    assert set(val) == {0, 2} and set(fit) == {1, 3, 4}     # las dos más nuevas del train


def test_duelos_y_partidas_completas():
    ok = _match("M1", _resto("A"), _resto("B"), 1, blue_gold=6000, red_gold=6000)  # empate de oro
    rota = _match("M2", _resto("A"), _resto("B"), 1)
    rota[1]["lane"] = "TOP"                                   # Blue con doble TOP
    parts = pd.DataFrame(ok + rota)
    assert list(complete_match_ids(parts)) == ["M1"]
    d = lane_duels(parts, ["M1"], "TOP")
    assert np.isnan(d.loc["M1", "y"])                         # sin ganador de línea
    X, vocab = duel_matrix(["A", "C"], ["B", "A"])
    np.testing.assert_allclose(X.toarray(), [[1, -1, 0], [-1, 0, 1]])
    assert list(vocab) == ["A", "B", "C"]


def test_metricas_de_probabilidad():
    y = np.array([0, 1, 0, 1])
    perfecto = prob_metrics(y, np.array([0.1, 0.9, 0.2, 0.8]), n_boot=20, n_boot_auc=20)
    assert perfecto["accuracy"]["valor"] == 1.0 and perfecto["auc"]["valor"] == 1.0
    constante = prob_metrics(y, np.full(4, 0.5), n_boot=20, n_boot_auc=20)
    assert constante["auc"]["valor"] == 0.5
    np.testing.assert_allclose(constante["brier"]["valor"], 0.25)
