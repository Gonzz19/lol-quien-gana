"""
Pruebas unitarias para src/models/baselines/draft_rules.py (y el split de evaluate_baselines.py).

Fijan las dos reglas pedidas por el usuario (sin entrenamiento):
* Línea: gana quien tiene el campeón con mayor `avg_gold_at_15` en esa línea.
* Partida: gana el equipo con mayor media de `winrate_pct` campeón-línea.
"""

import pandas as pd
import pytest

from src.models.baselines.draft_rules import (
    build_lane_duels,
    coherent_match_ids,
    lane_baseline,
    load_champion_stats,
    match_baseline,
)

LANES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]


def _champions_csv(tmp_path, rows):
    path = tmp_path / "champions_by_lane.csv"
    pd.DataFrame(rows, columns=["champion_name", "lane", "games_played", "wins",
                                "winrate_pct", "avg_gold_at_15"]).to_csv(path, index=False)
    return path


def _match(match_id, blue, red, blue_win, blue_gold=6000, red_gold=5000):
    """10 filas de una partida; `blue`/`red` son {lane: campeón}."""
    rows = []
    for team, champs, gold in ((100, blue, blue_gold), (200, red, red_gold)):
        win = blue_win if team == 100 else 1 - blue_win
        for lane in LANES:
            rows.append({"match_id": match_id, "team_id": team, "win": win,
                         "champion_name": champs[lane], "lane": lane, "gold_15": gold})
    return rows


@pytest.fixture
def stats(tmp_path):
    rows = []
    for lane in LANES:
        rows.append(("Fuerte", lane, 100, 55, 55.0, 6000.0))
        rows.append(("Debil", lane, 100, 45, 45.0, 5000.0))
    return load_champion_stats(_champions_csv(tmp_path, rows))


def test_partida_gana_el_equipo_con_mayor_winrate_medio(stats):
    fuerte = {lane: "Fuerte" for lane in LANES}
    debil = {lane: "Debil" for lane in LANES}
    parts = pd.DataFrame(_match("M1", fuerte, debil, blue_win=1)
                         + _match("M2", debil, fuerte, blue_win=0))
    res = match_baseline(parts, stats)
    assert res["n"] == 2
    assert res["accuracy"] == 1.0


def test_partida_promedia_las_cinco_lineas(stats):
    # Blue: 3 fuertes + 2 débiles (media 0,51) vs Red: 2 fuertes + 3 débiles (0,49).
    blue = {"TOP": "Fuerte", "JUNGLE": "Fuerte", "MIDDLE": "Fuerte",
            "BOTTOM": "Debil", "UTILITY": "Debil"}
    red = {"TOP": "Debil", "JUNGLE": "Debil", "MIDDLE": "Debil",
           "BOTTOM": "Fuerte", "UTILITY": "Fuerte"}
    parts = pd.DataFrame(_match("M1", blue, red, blue_win=0))
    res = match_baseline(parts, stats)
    assert res["accuracy"] == 0.0  # predijo Blue (mayor media) y ganó Red


def test_partida_sin_ganador_unico_se_excluye(stats):
    fuerte = {lane: "Fuerte" for lane in LANES}
    debil = {lane: "Debil" for lane in LANES}
    ok = _match("M1", fuerte, debil, blue_win=1)
    rota = _match("M2", fuerte, debil, blue_win=1)
    for r in rota:
        r["win"] = 0  # como LA2_1618651762: los 10 con win = 0
    parts = pd.DataFrame(ok + rota)
    assert list(coherent_match_ids(parts)) == ["M1"]
    assert match_baseline(parts, stats)["excluded_incoherent"] == 1


def test_linea_gana_el_campeon_con_mas_oro_medio_a_los_15(stats):
    fuerte = {lane: "Fuerte" for lane in LANES}
    debil = {lane: "Debil" for lane in LANES}
    # M1: Blue lleva Fuerte (6000 esperado) y realmente saca más oro -> acierto.
    # M2: Blue lleva Débil (5000 esperado) pero saca más oro -> error.
    parts = pd.DataFrame(_match("M1", fuerte, debil, 1, blue_gold=6100, red_gold=5900)
                         + _match("M2", debil, fuerte, 1, blue_gold=6100, red_gold=5900))
    res = lane_baseline(parts, stats)
    for lane in LANES:
        assert res[lane]["n"] == 2
        assert res[lane]["accuracy"] == 0.5
    assert res["GLOBAL"]["n"] == 10


def test_linea_sin_oro_o_con_empate_no_tiene_ganador(stats):
    fuerte = {lane: "Fuerte" for lane in LANES}
    debil = {lane: "Debil" for lane in LANES}
    empate = _match("M1", fuerte, debil, 1, blue_gold=6000, red_gold=6000)
    sin_oro = _match("M2", fuerte, debil, 1, blue_gold=None, red_gold=5000)
    res = lane_baseline(pd.DataFrame(empate + sin_oro), stats)
    assert res["_dropped_no_label"] == 10
    assert "GLOBAL" not in res


def test_posicion_duplicada_no_se_empareja_cartesianamente():
    rows = _match("M1", {l: "A" for l in LANES}, {l: "B" for l in LANES}, 1)
    rows[1]["lane"] = "TOP"  # Blue queda con doble TOP y sin JUNGLE
    duels, dropped = build_lane_duels(pd.DataFrame(rows))
    assert set(duels["lane"]) == {"MIDDLE", "BOTTOM", "UTILITY"}
    assert dropped == 2


# --------------------------------------------------------------------------- #
# Sin leakage: tablas de campeón solo con el train                            #
# --------------------------------------------------------------------------- #
def test_las_tablas_salen_solo_de_las_partidas_que_se_le_pasan():
    from src.models.baselines.draft_rules import stats_from_participants
    train = pd.DataFrame([
        {"match_id": "T1", "team_id": 100, "win": 1, "champion_name": "Ahri", "lane": "MIDDLE", "gold_15": 6000},
        {"match_id": "T2", "team_id": 100, "win": 0, "champion_name": "Ahri", "lane": "MIDDLE", "gold_15": 5000},
        {"match_id": "T1", "team_id": 200, "win": 0, "champion_name": "Zed", "lane": "MIDDLE", "gold_15": 4000},
    ])
    st = stats_from_participants(train)
    assert st["wr_lane"][("Ahri", "MIDDLE")] == 0.5
    assert st["gold_lane"][("Ahri", "MIDDLE")] == 5500
    assert st["wr_champ"]["Zed"] == 0.0


def _csvs(tmp_path):
    """Train viejo: Fuerte gana y saca más oro. Test reciente: al revés.

    Si el test se colara en las tablas, la regla cambiaría de lado; sin leakage,
    sigue prediciendo con lo que dijo el train y falla en el test.
    """
    lanes = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
    t0 = 1_788_231_600_000
    rows = []
    for k in range(40):                     # 40 días: los últimos van al test
        reciente = k >= 32
        mid = f"LA2_{k}"
        for team, champ in ((100, "Fuerte"), (200, "Debil")):
            gana_fuerte = not reciente
            win = int((team == 100) == gana_fuerte)
            for i, lane in enumerate(lanes):
                gold = (6000 if champ == "Fuerte" else 5000) if not reciente else (4000 if champ == "Fuerte" else 7000)
                rows.append({"match_id": mid, "team_id": team, "win": win, "champion_name": champ,
                             "lane": lane, "gold_15": gold, "player_id": f"p{team}{i}",
                             "game_creation": t0 + k * 86_400_000})
    full = tmp_path / "full.csv"
    pd.DataFrame(rows).to_csv(full, index=False)
    players = tmp_path / "players.csv"
    pd.DataFrame({"player_id": [f"p{t}{i}" for t in (100, 200) for i in range(5)],
                  "player_tier": ["MASTER"] * 10}).to_csv(players, index=False)
    return full, players


def test_sin_leakage_el_test_no_cambia_las_tablas(tmp_path):
    from scripts.analysis.evaluate_baselines import split_participants
    from src.models.baselines.draft_rules import stats_from_participants
    full, players = _csvs(tmp_path)
    train, test, res = split_participants(full, players, test_frac=0.2)
    assert res["test"] == 8 and res["train"] == 32
    assert set(train["match_id"]).isdisjoint(test["match_id"])
    st = stats_from_participants(train)
    assert st["gold_lane"][("Fuerte", "TOP")] == 6000       # solo train: no vio el 4000 del test
    m = match_baseline(test, st)
    assert m["accuracy"] == 0.0                              # el train dice "Fuerte", el test dice lo contrario
