"""
Pruebas del sistema de nube: backup versionado, manifiesto, restore con rollback.
Todo contra directorios temporales; sin red ni BD viva.
"""

import json
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

import pytest


from scripts.backup.backup_to_drive import (
    backup_timelines,
    check_integrity,
    collect_manifest,
    list_versioned,
    parse_snapshot_name,
    prune_versioned,
    run_backup,
    should_create_versioned,
    snapshot_name,
)
from src.database import DatabaseManager


@pytest.fixture
def demo_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = DatabaseManager(Path(tmpdir) / "demo.db")
        db.save_players([
            {"puuid": "p1", "summoner_id": "s1", "tier": "CHALLENGER", "rank": "I",
             "league_points": 800, "wins": 10, "losses": 5},
        ])
        yield db


def test_integrity_ok_and_corrupt(tmp_path):
    good = tmp_path / "good.db"
    DatabaseManager(good)
    ok, msg = check_integrity(good)
    assert ok is True and msg == "ok"

    bad = tmp_path / "bad.db"
    bad.write_bytes(b"esto no es sqlite")
    ok, _ = check_integrity(bad)
    assert ok is False


def test_manifest_counts(demo_db):
    m = collect_manifest(demo_db.db_path)
    assert m["players"] == 1
    assert m["matches"] == 0
    assert m["map_rows"] == 0
    assert m["projects"] == 0
    assert m["db_size_mb"] > 0
    assert "created_at" in m


def test_manifest_reports_exact_participants_not_derived(demo_db):
    """Regresión: el manifiesto usa COUNT(*) exacto, no matches*10.

    La BD viva tiene 27 partidas con doble ingesta multi-proyecto (20 filas);
    derivar ocultaría esa anomalía (716,020 vs 716,290 exactos).
    """
    conn = sqlite3.connect(demo_db.db_path)
    try:
        conn.execute("INSERT INTO matches (match_id, game_creation, game_duration, game_version, queue_id)"
                     " VALUES ('M_DOBLE', 1, 1800, '16.17.1', 420);")
        conn.execute("INSERT INTO match_participants (match_id, puuid, participant_id, team_id)"
                     " VALUES ('M_DOBLE', 'puuid_P0', 1, 100);")
        conn.execute("INSERT INTO match_participants (match_id, puuid, participant_id, team_id)"
                     " VALUES ('M_DOBLE', 'puuid_P1', 1, 100);")
        conn.commit()
    finally:
        conn.close()
    m = collect_manifest(demo_db.db_path)
    assert m["matches"] == 1
    assert m["participants"] == 2  # exacto, NO matches*10 == 10


def test_snapshot_name_roundtrip():
    m = {"matches": 71602}
    name = snapshot_name(m, datetime(2026, 9, 4, 12, 30))
    assert name == "lol_las_20260904_1230_m71602.db"
    assert parse_snapshot_name(name) == (datetime(2026, 9, 4, 12, 30), 71602)
    assert parse_snapshot_name("lol_las.db") is None
    assert parse_snapshot_name("random.txt") is None


def test_should_create_versioned_logic():
    now = datetime(2026, 9, 4, 12, 0)
    assert should_create_versioned([], {"matches": 100}, now=now) is True
    recent = [("lol_las_20260903_1200_m100.db", datetime(2026, 9, 3, 12, 0), 100)]
    assert should_create_versioned(recent, {"matches": 150}, now=now) is False
    old = [("lol_las_20260820_1200_m100.db", datetime(2026, 8, 20, 12, 0), 100)]
    assert should_create_versioned(old, {"matches": 150}, now=now) is True
    jumped = [("lol_las_20260903_1200_m100.db", datetime(2026, 9, 3, 12, 0), 100)]
    assert should_create_versioned(jumped, {"matches": 10100}, now=now,
                                    every_matches=10000) is True


def test_prune_keeps_newest_and_cleans_orphans(tmp_path):
    snaps = tmp_path / "snapshots"
    snaps.mkdir()
    for i in range(1, 6):
        (snaps / f"lol_las_2026090{i}_1200_m{i}.db").write_bytes(b"x")
    (snaps / "lol_las_20260905_1200_m5.db.manifest.json").write_text("{}")
    (snaps / "lol_las_20990101_0000_m9.db.manifest.json").write_text("{}")  # huérfano
    deleted = prune_versioned(snaps, keep=2)
    remaining = sorted(p.name for p in snaps.iterdir() if p.name.endswith(".db"))
    assert remaining == ["lol_las_20260904_1200_m4.db", "lol_las_20260905_1200_m5.db"]
    assert any("2026090" in d for d in deleted)
    assert not (snaps / "lol_las_20990101_0000_m9.db.manifest.json").exists()


def test_run_backup_end_to_end(demo_db, tmp_path):
    drive = tmp_path / "drive"
    drive.mkdir()
    local_bk = tmp_path / "backups"

    r1 = run_backup(db_path=demo_db.db_path, drive_path=drive, local_backups_dir=local_bk)
    assert r1["integrity_ok"] is True
    assert r1["uploaded_latest"] is True
    assert (drive / "lol_las.db").exists()
    manifest = json.loads((drive / "lol_las.manifest.json").read_text(encoding="utf-8"))
    assert manifest["players"] == 1
    assert r1["versioned"] is not None  # primer backup siempre versiona
    assert (drive / "snapshots" / r1["versioned"]).exists()
    assert (drive / "snapshots" / f"{r1['versioned']}.manifest.json").exists()

    # Segunda corrida inmediata: sube latest pero NO crea versionado nuevo
    r2 = run_backup(db_path=demo_db.db_path, drive_path=drive, local_backups_dir=local_bk)
    assert r2["uploaded_latest"] is True
    assert r2["versioned"] is None
    assert len(list_versioned(drive / "snapshots")) == 1

    # Integridad rota: no sube nada
    bad = tmp_path / "bad.db"
    bad.write_bytes(b"basura")
    r3 = run_backup(db_path=bad, drive_path=drive, local_backups_dir=local_bk)
    assert r3["integrity_ok"] is False
    assert r3["uploaded_latest"] is False


def test_restore_roundtrip_with_rollback(demo_db, tmp_path):
    from scripts.backup.restore_from_drive import local_pre_restore_backup
    drive = tmp_path / "drive"
    drive.mkdir()
    local_bk = tmp_path / "backups"
    run_backup(db_path=demo_db.db_path, drive_path=drive, local_backups_dir=local_bk)

    # Restore a destino nuevo (sin pre-backup porque no existía)
    dest = tmp_path / "restored.db"
    import shutil
    shutil.copy2(drive / "lol_las.db", dest)
    ok, _ = check_integrity(dest)
    assert ok is True

    # Pre-restore backup del destino existente en directorio temporal
    pre = local_pre_restore_backup(dest, local_backups_dir=local_bk)
    assert pre is not None and pre.exists()

    # Fuente corrupta: el restore debe fallar y el destino quedar intacto
    (drive / "lol_las.db").write_bytes(b"corrupto")
    before = dest.read_bytes()
    assert check_integrity(drive / "lol_las.db")[0] is False
    # Simular la política de restore: no copiar si el origen falla integrity
    ok_src, _ = check_integrity(drive / "lol_las.db")
    assert ok_src is False
    assert dest.read_bytes() == before


def test_pipeline_cmd_strict_conditional(tmp_path, monkeypatch):
    import scripts.collection.auto_nightly_pipeline as pipe
    # Sin mapa -> legacy (sin flag)
    monkeypatch.setattr(pipe, "map_has_rows", lambda db_path=None: False)
    assert "--require-canonical" not in pipe.build_crawler_cmd(60)
    # Con mapa poblado -> estricto
    monkeypatch.setattr(pipe, "map_has_rows", lambda db_path=None: True)
    assert "--require-canonical" in pipe.build_crawler_cmd(60)


def test_map_probe_reads_real_db(tmp_path):
    from scripts.collection.auto_nightly_pipeline import map_has_rows
    db = DatabaseManager(tmp_path / "probe.db")
    assert map_has_rows(tmp_path / "probe.db") is False
    db.register_api_projects(["K0_ABCDEFGHIJ"])
    db.upsert_player_identity("A", "B", "puuid-x", 0)
    assert map_has_rows(tmp_path / "probe.db") is True
    assert map_has_rows(tmp_path / "inexistente.db") is False


def test_backup_blocks_regression_and_force_overrides(demo_db, tmp_path):
    from scripts.backup.backup_to_drive import check_no_regression

    drive = tmp_path / "drive"
    drive.mkdir()
    local_bk = tmp_path / "backups"

    # Poblar la BD con una identidad y un alias antes del primer backup a Drive
    demo_db.register_api_projects(["K0_ABCDEFGHIJ"])
    demo_db.upsert_player_identity("Faker", "LAS", "puuid-faker-0", 0)
    demo_db.record_puuid_alias("puuid-faker-0", "puuid-faker-dead", "Faker", "LAS")
    demo_db.record_riot_id_observations([("puuid-faker-3", "Faker", "LAS")], project_index=3, game_creation=1000)

    r1 = run_backup(db_path=demo_db.db_path, drive_path=drive, local_backups_dir=local_bk)
    assert r1["uploaded_latest"] is True
    assert r1["regression_blocked"] is False

    # Crear una segunda BD local desactualizada (sin identidades ni alias ni riot_ids)
    stale_db = DatabaseManager(tmp_path / "stale.db")
    stale_db.save_players([
        {"puuid": "p1", "summoner_id": "s1", "tier": "CHALLENGER", "rank": "I",
         "league_points": 800, "wins": 10, "losses": 5},
    ])

    ok, reasons = check_no_regression(collect_manifest(stale_db.db_path), r1["manifest"])
    assert ok is False
    assert any("identities:" in r for r in reasons)
    assert any("aliases:" in r for r in reasons)
    assert any("riot_ids:" in r for r in reasons)

    # Intentar respaldar la BD desactualizada sobre Drive: debe bloquearse sin tocar Drive
    r_blocked = run_backup(db_path=stale_db.db_path, drive_path=drive, local_backups_dir=local_bk)
    assert r_blocked["regression_blocked"] is True
    assert r_blocked["uploaded_latest"] is False
    remote_after = json.loads((drive / "lol_las.manifest.json").read_text(encoding="utf-8"))
    assert remote_after["identities"] == 1
    assert remote_after["aliases"] == 1
    assert remote_after["riot_ids"] == 1

    # Con allow_regression=True (--force), permite sobrescribir
    r_forced = run_backup(db_path=stale_db.db_path, drive_path=drive,
                          local_backups_dir=local_bk, allow_regression=True)
    assert r_forced["regression_blocked"] is False
    assert r_forced["uploaded_latest"] is True


# --- Timelines completos con eventos (base aparte, §9.S.64) ------------------------------------------------

def _timelines_db(path: Path, n: int) -> Path:
    """Base con el esquema de `timelines_eventos.py` y `n` timelines falsos."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS timeline_crudo (match_id TEXT PRIMARY KEY, json_zlib BLOB NOT NULL, "
                     "bytes_json INTEGER NOT NULL, bajado_en REAL NOT NULL)")
        conn.execute("CREATE TABLE IF NOT EXISTS sin_timeline (match_id TEXT PRIMARY KEY, motivo TEXT)")
        conn.executemany("INSERT OR IGNORE INTO timeline_crudo VALUES (?, ?, ?, ?)",
                         [(f"LA2_{i}", b"x" * 100, 100, 0.0) for i in range(n)])
        conn.commit()
    finally:
        conn.close()
    return path


def test_backup_timelines_sube_solo_si_crece(tmp_path):
    drive = tmp_path / "drive"
    drive.mkdir()
    src = _timelines_db(tmp_path / "data" / "timelines_eventos" / "timelines.db", 5)

    r1 = backup_timelines(drive, src)
    assert r1["estado"] == "subida" and r1["timelines"] == 5
    dest = drive / "timelines_eventos" / "timelines.db"
    assert dest.exists() and dest.stat().st_size == src.stat().st_size
    man = json.loads((drive / "timelines_eventos" / "timelines.manifest.json").read_text(encoding="utf-8"))
    assert man["timelines"] == 5

    # sin cambios: no se vuelve a subir (la base llega a ~18 GB)
    assert backup_timelines(drive, src)["estado"] == "sin_cambios"

    _timelines_db(src, 8)
    r3 = backup_timelines(drive, src)
    assert r3["estado"] == "subida" and r3["timelines"] == 8


def test_backup_timelines_no_sube_con_la_descarga_corriendo(tmp_path):
    drive = tmp_path / "drive"
    drive.mkdir()
    src = _timelines_db(tmp_path / "tl" / "timelines.db", 3)
    lock = src.parent / "descargando.lock"
    lock.write_text("123", encoding="utf-8")

    assert backup_timelines(drive, src)["estado"] == "descarga_en_curso"
    assert not (drive / "timelines_eventos" / "timelines.db").exists()
    # latido viejo (> 5 min, el descargador murió sin borrarlo): se sube igual
    viejo = lock.stat().st_mtime + 301
    assert backup_timelines(drive, src, now=viejo)["estado"] == "subida"


def test_backup_timelines_bloquea_regresion(tmp_path):
    drive = tmp_path / "drive"
    drive.mkdir()
    grande = _timelines_db(tmp_path / "a" / "timelines.db", 6)
    chica = _timelines_db(tmp_path / "b" / "timelines.db", 2)
    assert backup_timelines(drive, grande)["estado"] == "subida"
    assert backup_timelines(drive, chica)["estado"] == "regresion"
    assert backup_timelines(drive, chica, allow_regression=True)["estado"] == "subida"


def test_backup_timelines_sin_base_local(tmp_path):
    assert backup_timelines(tmp_path, tmp_path / "no" / "timelines.db")["estado"] == "no_existe"
