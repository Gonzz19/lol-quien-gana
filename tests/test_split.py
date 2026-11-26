"""Split temporal con test proporcional por rango (src/features/split.py)."""

from datetime import datetime

import pandas as pd
import pytest

from src.features.split import ART, SIN_RANGO, match_tiers, proportional_temporal_split

DAY = 86_400_000
T0 = 1_788_231_600_000  # 2026-09-01 00:00 ART


def _matches(spec):
    """spec: [(tier, día, cantidad)] -> una fila por partida."""
    rows, i = [], 0
    for tier, day, n in spec:
        for k in range(n):
            rows.append({"match_id": f"M{i}", "game_creation": T0 + day * DAY + k * 1000, "tier": tier})
            i += 1
    return pd.DataFrame(rows)


def test_el_test_tiene_el_mismo_reparto_por_rango_que_el_train():
    # 80 % Master / 20 % GM en todo el período; el GM reciente es más escaso.
    spec = [("MASTER", d, 8) for d in range(20)] + [("GRANDMASTER", d, 2) for d in range(20)]
    labels, res = proportional_temporal_split(_matches(spec), test_frac=0.2)
    assert res["reparto_test"] == pytest.approx(res["reparto_train"], abs=0.02)
    assert res["test"] / (res["train"] + res["test"]) == pytest.approx(0.2, abs=0.02)


def test_el_test_es_posterior_al_train():
    m = _matches([("MASTER", d, 5) for d in range(15)] + [("CHALLENGER", d, 1) for d in range(15)])
    labels, _ = proportional_temporal_split(m, test_frac=0.2)
    t = m.set_index("match_id")["game_creation"]
    assert t[labels[labels == "train"].index].max() < t[labels[labels == "test"].index].min()


def test_el_corte_retrocede_si_un_rango_no_alcanza():
    # Challenger solo jugó al principio: no hay test posible para ese rango sin retroceder.
    spec = [("MASTER", d, 10) for d in range(20)] + [("CHALLENGER", d, 3) for d in range(5)]
    labels, res = proportional_temporal_split(_matches(spec), test_frac=0.2)
    assert res["reparto_test"].get("CHALLENGER", 0) == pytest.approx(res["reparto_train"]["CHALLENGER"], abs=0.03)


def test_lo_que_sobra_despues_del_corte_no_va_al_train():
    # GM juega mucho más en los últimos días: sobra GM reciente, que queda 'fuera'.
    spec = ([("MASTER", d, 10) for d in range(20)] + [("GRANDMASTER", d, 1) for d in range(15)]
            + [("GRANDMASTER", d, 6) for d in range(15, 20)])
    m = _matches(spec)
    labels, res = proportional_temporal_split(m, test_frac=0.2)
    assert res["fuera"] > 0
    assert res["reparto_test"] == pytest.approx(res["reparto_train"], abs=0.02)
    t = m.set_index("match_id")["game_creation"]
    assert (t[labels[labels == "fuera"].index] > t[labels[labels == "train"].index].max()).all()


def test_el_rango_es_el_tier_mas_alto_con_5_jugadores_o_superior():
    tiers = {"c": "CHALLENGER", "g": "GRANDMASTER", "m": "MASTER", "d": "DIAMOND"}

    def lobby(mid, spec):  # spec: "ccgmm..." una letra por jugador del padrón; "x" = fuera del padrón
        return [(mid, f"{mid}_{i}_{ch}") for i, ch in enumerate(spec)]

    rows = (lobby("A", "cccccmmmmm")     # 5 Challenger -> CHALLENGER
            + lobby("B", "ccccgmmmmm")   # 4 Chall + 1 GM = 5 GM o más -> GRANDMASTER
            + lobby("C", "cccgggmmmm")   # 3 + 3: ningún tier junta 5 solo, pero 6 GM o más
            + lobby("D", "ggggmddddd")   # 4 GM + 1 Master = 5 Master o más -> MASTER
            + lobby("E", "ggggdddddd")   # 4 GM: no llega a 5 Master o más -> DIAMOND
            + lobby("F", "mmmmxxxxxx"))  # 4 del padrón: ningún nivel junta 5
    full = pd.DataFrame(rows, columns=["match_id", "player_id"])
    padron = full[~full["player_id"].str.endswith("x")].assign(
        player_tier=lambda d: d["player_id"].str[-1].map(tiers))[["player_id", "player_tier"]]
    t = match_tiers(full, padron)
    assert t.to_dict() == {"A": "CHALLENGER", "B": "GRANDMASTER", "C": "GRANDMASTER",
                           "D": "MASTER", "E": "DIAMOND", "F": SIN_RANGO}


def test_fraccion_invalida():
    with pytest.raises(ValueError):
        proportional_temporal_split(_matches([("MASTER", 0, 5)]), test_frac=1.2)


def test_el_test_no_lleva_diamond_y_es_proporcional_en_el_resto():
    # Diamond reciente abunda (los Master juegan con Diamond), pero no puede ir al test.
    spec = ([("MASTER", d, 8) for d in range(20)] + [("GRANDMASTER", d, 2) for d in range(20)]
            + [("DIAMOND", d, 3) for d in range(20)] + [("DIAMOND", d, 10) for d in range(17, 20)])
    m = _matches(spec)
    labels, res = proportional_temporal_split(m, test_frac=0.2, test_tiers=("MASTER", "GRANDMASTER", "CHALLENGER"))
    tier = m.set_index("match_id")["tier"]
    t = m.set_index("match_id")["game_creation"]
    assert "DIAMOND" not in set(tier[labels[labels == "test"].index])
    assert res["reparto_test"] == pytest.approx(res["reparto_train_rangos_test"], abs=0.02)
    # 20 % del test en los rangos elegibles; Diamond anterior al corte sigue en train,
    # el posterior queda fuera.
    eleg = tier != "DIAMOND"
    assert (labels[eleg] == "test").sum() / (labels[eleg] != "fuera").sum() == pytest.approx(0.2, abs=0.02)
    corte = t[labels[labels == "test"].index].min()
    assert (labels[(tier == "DIAMOND") & (t < corte)] == "train").all()
    assert (labels[(tier == "DIAMOND") & (t >= corte)] == "fuera").all()


def test_un_rango_fuera_del_test_no_hace_retroceder_el_corte():
    # Diamond solo jugó al principio: como no va al test, no tiene que mover el corte.
    base = [("MASTER", d, 10) for d in range(20)]
    sin_diamond = proportional_temporal_split(_matches(base), test_frac=0.2)[1]["corte"]
    con_diamond = proportional_temporal_split(
        _matches(base + [("DIAMOND", d, 5) for d in range(5)]), test_frac=0.2,
        test_tiers=("MASTER", "GRANDMASTER", "CHALLENGER"))[1]["corte"]
    assert con_diamond == sin_diamond


def test_con_sobrante_al_test_no_queda_nada_elegible_fuera():
    # GM juega mucho al final: sin la opción, el GM reciente que sobra queda 'fuera'.
    spec = ([("MASTER", d, 10) for d in range(20)] + [("GRANDMASTER", d, 1) for d in range(15)]
            + [("GRANDMASTER", d, 6) for d in range(15, 20)] + [("DIAMOND", d, 4) for d in range(20)])
    m = _matches(spec)
    eleg = ("MASTER", "GRANDMASTER", "CHALLENGER")
    _, base = proportional_temporal_split(m, test_frac=0.2, test_tiers=eleg)
    labels, res = proportional_temporal_split(m, test_frac=0.2, test_tiers=eleg, sobrante_al_test=True)
    tier = m.set_index("match_id")["tier"]
    t = m.set_index("match_id")["game_creation"]
    assert res["corte"] == base["corte"]                       # el corte no cambia
    assert res["test"] > base["test"]                          # el sobrante entra al test
    assert not ((labels == "fuera") & tier.isin(eleg)).any()   # nada elegible queda fuera
    assert (labels[tier == "DIAMOND"][t >= t[labels == "test"].min()] == "fuera").all()
    assert t[labels[labels == "train"].index].max() < t[labels[labels == "test"].index].min()


def test_la_base_de_la_cuota_ignora_el_historial_anterior():
    # Challenger con mucho historial viejo (días 0-9) y poco reciente: con la base
    # completa su cuota obliga a retroceder el corte; con la base desde el día 10, no.
    spec = ([("MASTER", d, 10) for d in range(30)] + [("CHALLENGER", d, 6) for d in range(10)]
            + [("CHALLENGER", d, 1) for d in range(10, 30)])
    m = _matches(spec)
    eleg = ("MASTER", "GRANDMASTER", "CHALLENGER")
    base = datetime.fromtimestamp((T0 + 10 * DAY) / 1000, ART).strftime("%Y-%m-%d")
    _, completa = proportional_temporal_split(m, test_frac=0.2, test_tiers=eleg)
    _, reciente = proportional_temporal_split(m, test_frac=0.2, test_tiers=eleg, base_desde=base)
    assert reciente["corte"] > completa["corte"]
    assert reciente["train"] > completa["train"]


def test_master_plus_test_es_el_20_por_ciento_mas_reciente_y_diamond_aparte():
    from src.features.split import master_plus_temporal_split
    spec = ([("MASTER", d, 8) for d in range(20)] + [("GRANDMASTER", d, 2) for d in range(20)]
            + [("CHALLENGER", d, 1) for d in range(5)]          # Challenger solo al principio
            + [("DIAMOND", d, 5) for d in range(20)] + [(SIN_RANGO, d, 1) for d in range(20)])
    m = _matches(spec)
    labels, res = master_plus_temporal_split(m, test_frac=0.2)
    tier = m.set_index("match_id")["tier"]
    t = m.set_index("match_id")["game_creation"]
    eleg = tier.isin(["MASTER", "GRANDMASTER", "CHALLENGER"])
    # 20 % de todo Master+ (corte por día: puede pasarse un poco, nunca quedarse corto)
    assert 0.2 <= res["test_sobre_master_plus"] <= 0.26
    # temporal: todo el train antes que todo el test
    assert t[labels == "train"].max() < t[labels == "test"].min()
    # un rango escaso no corre el corte: Challenger queda sin test y está bien
    assert res["test_por_rango"]["CHALLENGER"] == 0
    # Diamond nunca entra al test: el anterior al corte va al train, el posterior aparte
    corte = t[labels == "test"].min()
    assert (labels[(tier == "DIAMOND") & (t < corte)] == "train").all()
    assert (labels[(tier == "DIAMOND") & (t >= corte)] == "diamante_despues").all()
    assert res["train_diamante"] == int(((tier == "DIAMOND") & (t < corte)).sum()) > 0
    assert (labels[tier == SIN_RANGO] == "sin_rango").all()
    assert set(labels[eleg]) == {"train", "test"}
    # el Diamond del train no mueve el corte: se fija solo con Master+
    _, sin = master_plus_temporal_split(m, test_frac=0.2, diamante_antes_al_train=False)
    assert sin["corte"] == res["corte"] and sin["test"] == res["test"]
    assert sin["diamante_antes"] == res["train_diamante"]
