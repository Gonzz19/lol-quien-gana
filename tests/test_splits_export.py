"""Export de train y test por separado (export_to_csv.export_splits, §9.S.14)."""

import json

import pandas as pd

LANES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
T0 = 1_788_231_600_000
DAY = 86_400_000


def _match(match_id, day, champ_blue, puuids):
    return {"metadata": {"matchId": match_id},
            "info": {"gameCreation": T0 + day * DAY, "gameDuration": 1800, "gameVersion": "16.20",
                     "queueId": 420,
                     "participants": [{"participantId": i + 1, "puuid": p, "teamId": 100 if i < 5 else 200,
                                       "win": i < 5, "championName": champ_blue if i < 5 else "Zed",
                                       "championId": 1, "individualPosition": LANES[i % 5],
                                       "riotIdGameName": f"n{p}", "riotIdTagline": "LAS"}
                                      for i, p in enumerate(puuids)]}}


def _db(temp_db):
    puuids = [f"m{i}" for i in range(10)]
    temp_db.save_players([{"puuid": p, "summoner_id": "s", "tier": "MASTER", "rank": "I"} for p in puuids])
    # 30 días: el campeón azul de las partidas viejas es "Viejo" y el de las recientes "Nuevo".
    batch = [(f"LA2_{d}", _match(f"LA2_{d}", d, "Viejo" if d < 24 else "Nuevo", puuids), None, 0)
             for d in range(30)]
    batch.append(("BR1_X", _match("BR1_X", 29, "Nuevo", puuids), None, 0))  # otro servidor
    temp_db.save_matches_and_timelines_batch(batch)


def test_train_y_test_van_separados_y_sin_leakage(temp_db, tmp_path):
    from export_to_csv import export_splits

    _db(temp_db)
    out = tmp_path / "splits"
    man = export_splits(temp_db.db_path, out, test_frac=0.2)

    split = pd.read_csv(out / "match_split.csv").set_index("match_id")["split"]
    tr_full = pd.read_csv(out / "train" / "full_dataset.csv")
    te_full = pd.read_csv(out / "test" / "full_dataset.csv")
    tr_ids, te_ids = set(tr_full.match_id), set(te_full.match_id)

    assert tr_ids.isdisjoint(te_ids)
    assert tr_ids == set(split[split == "train"].index)
    assert te_ids == set(split[split == "test"].index)
    assert "BR1_X" not in tr_ids | te_ids                       # otro servidor: en ninguno
    fuera = set(split[split == "fuera"].index)
    assert fuera.isdisjoint(tr_ids | te_ids)
    assert man["train"] == len(tr_ids) and man["test"] == len(te_ids)

    # Agregados del train: solo con partidas de train.
    champs = pd.read_csv(out / "train" / "champions_by_lane.csv")
    assert "Nuevo" not in set(champs.champion_name)            # solo jugó en el período reciente
    players = pd.read_csv(out / "train" / "players.csv")
    assert players.games_recorded.max() == len(tr_ids)
    mu = pd.read_csv(out / "train" / "matchups_1v1.csv")
    assert set(mu.match_id) <= tr_ids
    assert json.loads((out / "split_manifest.json").read_text(encoding="utf-8"))["corte"] == man["corte"]


def test_sin_filtro_el_export_es_el_de_siempre(temp_db, tmp_path):
    from export_to_csv import export_full_dataset

    _db(temp_db)
    df = export_full_dataset(temp_db.db_path, tmp_path / "full.csv")
    assert df.match_id.nunique() == 31                         # el volcado completo no cambia (§3.G)


def test_export_completo_usa_una_foto_y_un_mapa_sin_cambiar_bytes(temp_db, tmp_path, monkeypatch):
    import export_to_csv as export

    _db(temp_db)
    expected = tmp_path / "individual"
    actual = tmp_path / "shared"
    expected.mkdir()
    db_path = temp_db.db_path
    export.export_full_dataset(db_path, expected / "full_dataset.csv")
    matchups = export.export_matchups_1v1(db_path, expected / "matchups_1v1.csv")
    export.export_players(db_path, expected / "players.csv", matchups_df=matchups)
    export.export_champions_by_lane(db_path, expected / "champions_by_lane.csv")

    attach = export.attach_for_participants
    attached = []

    def track_attach(conn):
        attached.append(conn)
        return attach(conn)

    to_csv = pd.DataFrame.to_csv

    def write_with_concurrent_commit(df, path, **kwargs):
        to_csv(df, path, **kwargs)
        if path.name == "full_dataset.csv":
            mid = "LA2_LATE"
            temp_db.save_matches_and_timelines_batch(
                [(mid, _match(mid, 31, "Tarde", [f"m{i}" for i in range(10)]), None)])

    monkeypatch.setattr(export, "attach_for_participants", track_attach)
    monkeypatch.setattr(pd.DataFrame, "to_csv", write_with_concurrent_commit)
    export.export_all_datasets(db_path, actual)
    assert temp_db.is_match_saved("LA2_LATE")
    assert len(attached) == 1
    for csv in expected.glob("*.csv"):
        assert (actual / csv.name).read_bytes() == csv.read_bytes()


def test_match_times_da_dia_y_hora_en_argentina(temp_db, tmp_path):
    from export_to_csv import export_match_times

    _db(temp_db)
    df = export_match_times(temp_db.db_path, tmp_path / "t.csv").set_index("match_id")
    # T0 = 2026-09-01 00:00 ART (martes); LA2_0 dura 1800 s.
    assert df.loc["LA2_0", "inicio_art"] == "2026-09-01 00:00:00"
    assert df.loc["LA2_0", "fin_estimado_art"] == "2026-09-01 00:30:00"
    assert df.loc["LA2_0", "dia_semana"] == "martes" and df.loc["LA2_0", "hora"] == 0
    assert df.loc["BR1_X", "servidor"] == "BR1"
    assert len(df) == 31
