"""Pruebas de firmas de jugador (variantes A y B). Sin red ni BD viva."""

import pandas as pd


from src.features.player_signatures import (
    EMB_FEATURES,
    MIN_LANE_GAMES,
    build_champion_embeddings,
    signature_a,
    signature_b,
)


def _fake_champs():
    # 2 campeones x 2 líneas con stats separables
    rows = []
    for champ, lane, kda, cs in [("Zed", "MIDDLE", 3.0, 150), ("Zed", "TOP", 2.0, 140),
                                 ("Malphite", "TOP", 2.5, 145), ("Jinx", "BOTTOM", 3.5, 200)]:
        r = {"champion_name": champ, "lane": lane, "games_played": 500,
             "avg_kills": kda, "avg_deaths": 5.0, "avg_assists": 5.0, "avg_kda": kda}
        for c in EMB_FEATURES:
            r.setdefault(c, 1.0)
        r["avg_kills"], r["avg_kda"] = kda, kda
        r["avg_total_cs"] = cs
        rows.append(r)
    return pd.DataFrame(rows)


def test_emb_excludes_first_blood():
    assert "first_blood_rate_pct" not in EMB_FEATURES
    assert len(EMB_FEATURES) == 22


def test_build_embeddings_shapes():
    vecs, lanes, scaler = build_champion_embeddings(_fake_champs())
    assert len(vecs) == 4
    assert set(lanes) == {"MIDDLE", "TOP", "BOTTOM"}
    assert next(iter(vecs.values())).shape == (22,)


def _games():
    return pd.DataFrame([
        # OTP Zed mid (canon1): 3 partidas iguales -> dispersión 0
        {"player_id": "canon1", "champion_name": "Zed", "lane": "MIDDLE"},
        {"player_id": "canon1", "champion_name": "Zed", "lane": "MIDDLE"},
        {"player_id": "canon1", "champion_name": "Zed", "lane": "MIDDLE"},
        # Flexible (canon2): Zed mid + Jinx bot -> dispersión > 0
        {"player_id": "canon2", "champion_name": "Zed", "lane": "MIDDLE"},
        {"player_id": "canon2", "champion_name": "Jinx", "lane": "BOTTOM"},
        # Identidad sin nombre conocido (player_id = PUUID) + línea inválida (se excluye)
        {"player_id": "raw9", "champion_name": "Zed", "lane": "MIDDLE"},
        {"player_id": "raw9", "champion_name": "Zed", "lane": "Invalid"},
    ])


def test_signature_a_otp_dispersion_zero():
    vecs, _, _ = build_champion_embeddings(_fake_champs())
    sig = signature_a(_games(), vecs)
    assert set(sig["player_id"]) == {"canon1", "canon2", "raw9"}
    otp = sig[sig["player_id"] == "canon1"].iloc[0]
    flex = sig[sig["player_id"] == "canon2"].iloc[0]
    assert otp["n_games"] == 3
    assert otp["pool_dispersion"] == 0.0
    assert flex["pool_dispersion"] > 0.0
    assert sig[sig["player_id"] == "raw9"].iloc[0]["n_games"] == 1  # Invalid excluida


def test_signature_b_eligibility():
    vecs, _, _ = build_champion_embeddings(_fake_champs())
    sig = signature_b(_games(), vecs, min_lane_games=2)
    canon1_mid = sig[(sig["player_id"] == "canon1") & (sig["lane"] == "MIDDLE")].iloc[0]
    assert canon1_mid["n_games"] == 3 and bool(canon1_mid["eligible"]) is True
    canon2_bot = sig[(sig["player_id"] == "canon2") & (sig["lane"] == "BOTTOM")].iloc[0]
    assert canon2_bot["n_games"] == 1 and bool(canon2_bot["eligible"]) is False


def test_min_lane_games_default_ten():
    # Elegibilidad posicional: solo (jugador, linea) con >=10 partidas
    assert MIN_LANE_GAMES == 10
    vecs, _, _ = build_champion_embeddings(_fake_champs())
    sig = signature_b(_games(), vecs)  # default
    assert bool(sig[sig["player_id"] == "canon1"].iloc[0]["eligible"]) is False
