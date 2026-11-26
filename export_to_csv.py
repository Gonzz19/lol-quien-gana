"""
Script de utilidad para exportar datos desde SQLite a archivos CSV para entrenamiento de Machine Learning.
Genera los datasets optimizados para:
1. data/full_dataset.csv: Partidas, participantes, contexto de equipo y timeline @10, 15, 20.
2. data/players.csv: Vectores de estilo y arquetipos por jugador (KP%, Damage Share, Gold Share, Agresividad @10, etc.).
3. data/champions_by_lane.csv: Rendimiento agregado de cada campeón por línea (Winrates, CS/Gold/Damage @10, 15, 20).
4. data/matchups_1v1.csv: Dataset de enfrentamientos directos de línea (Blue vs Red) con diferenciales temporales @10, 15, 20.
"""

import sqlite3
import argparse
import json
from contextlib import closing, contextmanager
from typing import Iterable, Optional
import pandas as pd
from pathlib import Path
from src.config import DEFAULT_DB_PATH, DATA_DIR
from src.database import sql_tier_maximo
from src.features.identity import attach_for_participants

# Tier más alto de la persona en orden de liga. `MAX(pl.tier)` compara texto y
# pondría DIAMOND por encima de CHALLENGER (§9.S.76).
_TIER_PADRON = sql_tier_maximo("pl.tier")
_TIER_JUGADOR = sql_tier_maximo("COALESCE(pp.tier, NULLIF(pid.tier, 'UNKNOWN'), 'UNKNOWN')")


# Filtro opcional por lista de partidas (para exportar los splits train/test por
# separado). Vive en una tabla TEMP: funciona con la conexión en modo solo lectura.
_FILTRO = "<<FILTRO>>"


def _match_filter(conn, match_ids: Optional[Iterable[str]], col: str, joiner: str) -> str:
    """Crea `temp.split_matches` y devuelve la cláusula SQL ('' si no hay filtro)."""
    if match_ids is None:
        return ""
    conn.execute("DROP TABLE IF EXISTS temp.split_matches;")
    conn.execute("CREATE TEMP TABLE split_matches (match_id TEXT PRIMARY KEY);")
    conn.executemany("INSERT OR IGNORE INTO split_matches (match_id) VALUES (?);",
                     ((m,) for m in match_ids))
    return f" {joiner} {col} IN (SELECT match_id FROM temp.split_matches) "


@contextmanager
def _export_snapshot(db_path: Path, *, identity: bool):
    # El export ya materializa DataFrames grandes. SQLite conserva el almac?n
    # temporal por defecto para que pueda usar disco en sus ordenamientos.
    with closing(sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro",
                                 uri=True, timeout=60.0)) as conn:
        conn.execute("BEGIN")
        if identity:
            attach_for_participants(conn)
        yield conn


def export_full_dataset(db_path: Path, output_csv: Path,
                        match_ids: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Exporta el dataset general unificando partidas, participantes, métricas de equipo, timeline y rango del jugador.
    """
    with _export_snapshot(db_path, identity=True) as conn:
        return _export_full_dataset(conn, output_csv, match_ids)


def _export_full_dataset(conn, output_csv: Path, match_ids):
    print(f"[*] Generando CSV 1: Dataset completo con Timeline y Métricas de Equipo...")
    filtro = _match_filter(conn, match_ids, "p.match_id", "WHERE")
    query = f"""
        WITH padron_players AS (
            -- Tier/rango por persona: cualquier PUUID suyo que esté en el padrón
            -- (mismo criterio que export_players). Cruzarlo fila por fila dejaba
            -- sin tier las partidas guardadas bajo PUUIDs de otros proyectos.
            SELECT m.player_id, {_TIER_PADRON} AS tier, MAX(pl.rank) AS rank
            FROM player_id_map m
            INNER JOIN players pl ON pl.puuid = m.puuid
            GROUP BY m.player_id
        ),
        match_teams AS (
            SELECT
                match_id, team_id,
                SUM(kills) AS team_total_kills,
                SUM(total_damage_dealt_to_champions) AS team_total_damage,
                SUM(gold_earned) AS team_total_gold
            FROM match_participants
            GROUP BY match_id, team_id
        )
        SELECT
            p.match_id,
            m.game_creation,
            m.game_duration,
            m.game_version,
            -- Identidad de persona: la ÚNICA columna válida para agrupar por jugador.
            pim.player_id,
            -- PUUID crudo de la participación: cifrado por proyecto de API, así que
            -- el mismo humano aparece con hasta un PUUID distinto por proyecto.
            -- Sirve para volver a la API o depurar; NUNCA para agrupar.
            p.puuid AS puuid_raw,
            p.participant_id,
            p.team_id,
            p.win,
            p.champion_id,
            p.champion_name,
            p.individual_position AS lane,
            p.kills,
            p.deaths,
            p.assists,
            p.kda,
            -- Métricas de Participación en el Equipo
            ROUND(100.0 * (p.kills + p.assists) / MAX(1, mt.team_total_kills), 1) AS kill_participation_pct,
            ROUND(100.0 * p.total_damage_dealt_to_champions / MAX(1, mt.team_total_damage), 1) AS damage_share_pct,
            ROUND(100.0 * p.gold_earned / MAX(1, mt.team_total_gold), 1) AS gold_share_pct,
            p.total_damage_dealt_to_champions,
            p.total_damage_taken,
            p.gold_earned,
            p.total_minions_killed,
            p.neutral_minions_killed,
            p.vision_score,
            p.champ_level,
            p.item0, p.item1, p.item2, p.item3, p.item4, p.item5, p.item6,
            p.first_blood_kill,
            p.first_tower_kill,
            pp.tier AS player_tier,
            pp.rank AS player_rank,
            -- Métricas Timeline Minuto 10
            t.gold_10,
            t.xp_10,
            t.level_10,
            t.cs_10,
            t.damage_10,
            t.kills_10,
            t.deaths_10,
            t.assists_10,
            (t.kills_10 + t.deaths_10 + t.assists_10) AS early_aggression_10,
            -- Métricas Timeline Minuto 15
            t.gold_15,
            t.xp_15,
            t.level_15,
            t.cs_15,
            t.damage_15,
            t.kills_15,
            t.deaths_15,
            t.assists_15,
            -- Métricas Timeline Minuto 20
            t.gold_20,
            t.xp_20,
            t.level_20,
            t.cs_20,
            t.damage_20,
            t.kills_20,
            t.deaths_20,
            t.assists_20
        FROM match_participants p
        INNER JOIN matches m ON p.match_id = m.match_id
        INNER JOIN match_teams mt ON p.match_id = mt.match_id AND p.team_id = mt.team_id
        INNER JOIN player_id_map pim ON pim.puuid = p.puuid
        LEFT JOIN padron_players pp ON pp.player_id = pim.player_id
        LEFT JOIN match_timeline t ON p.match_id = t.match_id AND p.participant_id = t.participant_id
        <<FILTRO>>;
    """
    df = pd.read_sql_query(query.replace(_FILTRO, filtro), conn)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"[+] CSV 1 exportado con exito: '{output_csv.name}' ({len(df)} filas, {len(df.columns)} columnas, {output_csv.stat().st_size / (1024*1024):.2f} MB)")
    return df


def export_players(db_path: Path, output_csv: Path, matchups_df: Optional[pd.DataFrame] = None,
                   match_ids: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Exporta vectores de estilo y arquetipos por jugador a partir de las partidas recolectadas,
    enriquecido con diferenciales frente a su rival directo de línea (GD@15, CS@15, CS Final).
    """
    with _export_snapshot(db_path, identity=True) as conn:
        return _export_players(conn, output_csv, matchups_df, match_ids)


def _export_players(conn, output_csv: Path, matchups_df, match_ids):
    print(f"[*] Generando CSV 2: Vectores de Arquetipos y Estilo de Jugadores...")
    filtro = _match_filter(conn, match_ids, "p.match_id", "WHERE")
    query = f"""
        WITH padron_players AS (
            SELECT
                m.player_id,
                {_TIER_PADRON} AS tier,
                MAX(pl.rank) AS rank
            FROM player_id_map m
            INNER JOIN players pl ON pl.puuid = m.puuid
            GROUP BY m.player_id
        ),
        match_teams AS (
            SELECT
                match_id, team_id,
                SUM(kills) AS team_total_kills,
                SUM(total_damage_dealt_to_champions) AS team_total_damage,
                SUM(gold_earned) AS team_total_gold
            FROM match_participants
            GROUP BY match_id, team_id
        )
        SELECT
            -- Identidad de persona (componente conexa). Una fila por jugador real.
            pim.player_id,
            -- PUUID representante para volver a la API. NO es la identidad: el mismo
            -- humano tiene un PUUID por proyecto, y acá se elige uno determinista.
            MIN(p.puuid) AS puuid_raw,
            -- PUUIDs crudos bajo los que se conoce a este humano (uno por proyecto de
            -- API, más los que sumen renombres). No se cuenta sobre las participaciones:
            -- el camino de escritura ya canonicaliza a proyecto 0 antes del INSERT, así
            -- que ahí siempre daría 1. Se cuenta sobre el componente conexo completo.
            (SELECT COUNT(*) FROM player_id_map m2 WHERE m2.player_id = pim.player_id) AS num_puuids,
            {_TIER_JUGADOR} AS player_tier,
            MAX(COALESCE(pp.rank, NULLIF(pid.rank, ''), 'I')) AS player_rank,
            COUNT(*) AS games_recorded,
            SUM(p.win) AS wins,
            COUNT(*) - SUM(p.win) AS losses,
            ROUND(100.0 * SUM(p.win) / COUNT(*), 2) AS winrate_pct,
            -- Rendimiento y KDA
            ROUND(AVG(p.kills), 2) AS avg_kills,
            ROUND(AVG(p.deaths), 2) AS avg_deaths,
            ROUND(AVG(p.assists), 2) AS avg_assists,
            ROUND(AVG(p.kda), 2) AS avg_kda,
            -- Características de Arquetipo de Jugador
            ROUND(AVG(100.0 * (p.kills + p.assists) / MAX(1, mt.team_total_kills)), 1) AS avg_kill_participation_pct,
            ROUND(AVG(100.0 * p.total_damage_dealt_to_champions / MAX(1, mt.team_total_damage)), 1) AS avg_damage_share_pct,
            ROUND(AVG(100.0 * p.gold_earned / MAX(1, mt.team_total_gold)), 1) AS avg_gold_share_pct,
            COUNT(DISTINCT p.champion_name) AS unique_champions_count,
            -- Distribución Porcentual de Posición / Línea
            ROUND(100.0 * SUM(CASE WHEN p.individual_position = 'TOP' THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_games_top,
            ROUND(100.0 * SUM(CASE WHEN p.individual_position = 'JUNGLE' THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_games_jungle,
            ROUND(100.0 * SUM(CASE WHEN p.individual_position = 'MIDDLE' THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_games_middle,
            ROUND(100.0 * SUM(CASE WHEN p.individual_position = 'BOTTOM' THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_games_bottom,
            ROUND(100.0 * SUM(CASE WHEN p.individual_position = 'UTILITY' THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_games_utility,
            -- Agresividad y Farm Temprano (@10 y @15)
            ROUND(AVG(t.kills_10 + t.deaths_10 + t.assists_10), 1) AS avg_early_aggression_10,
            ROUND(AVG(t.cs_10), 1) AS avg_cs_at_10,
            ROUND(AVG(t.gold_10), 0) AS avg_gold_at_10,
            ROUND(AVG(t.damage_10), 0) AS avg_damage_at_10,
            ROUND(AVG(t.cs_15), 1) AS avg_cs_at_15,
            ROUND(AVG(t.gold_15), 0) AS avg_gold_at_15,
            ROUND(AVG(t.damage_15), 0) AS avg_damage_at_15,
            -- Métricas Finales
            ROUND(AVG(p.total_damage_dealt_to_champions), 0) AS avg_total_damage,
            ROUND(AVG(p.total_damage_taken), 0) AS avg_damage_taken,
            ROUND(AVG(p.gold_earned), 0) AS avg_gold_earned,
            ROUND(AVG(p.total_minions_killed + p.neutral_minions_killed), 1) AS avg_total_cs,
            ROUND(AVG(p.vision_score), 1) AS avg_vision_score,
            ROUND(100.0 * AVG(p.first_blood_kill), 2) AS first_blood_rate_pct
        FROM match_participants p
        INNER JOIN match_teams mt ON p.match_id = mt.match_id AND p.team_id = mt.team_id
        INNER JOIN player_id_map pim ON pim.puuid = p.puuid
        INNER JOIN padron_players pp ON pp.player_id = pim.player_id
        LEFT JOIN player_identities pid ON pid.canonical_id = pim.player_id
        LEFT JOIN match_timeline t ON p.match_id = t.match_id AND p.participant_id = t.participant_id
        <<FILTRO>>
        GROUP BY pim.player_id
        ORDER BY games_recorded DESC, winrate_pct DESC;
    """
    df = pd.read_sql_query(query.replace(_FILTRO, filtro), conn)
    # Calcular especialización de rol (% en su posición más jugada)
    lane_cols = ['pct_games_top', 'pct_games_jungle', 'pct_games_middle', 'pct_games_bottom', 'pct_games_utility']
    df['role_focus_pct'] = df[lane_cols].max(axis=1)
    if len(df) > 0:
        # 1. Carril principal dominante del jugador
        df['main_lane'] = df[lane_cols].idxmax(axis=1).str.replace('pct_games_', '').str.upper()

        # 2. Tier de confianza estadística según volumen de partidas acumuladas
        df['confidence_tier'] = pd.cut(
            df['games_recorded'],
            bins=[-1, 2, 14, 49, 999999],
            labels=['UNRELIABLE', 'LOW', 'MEDIUM', 'HIGH']
        )
    # Enriquecer con diferenciales de línea frente al rival (GD@15, CS@15, CS Final, Daño Final, Delta Gold %)
    cols_needed = {'blue_player_id', 'red_player_id', 'diff_gold_15', 'diff_cs_15', 'diff_cs_end', 'diff_damage_end', 'diff_gold_pct_snowball'}
    if matchups_df is None and match_ids is None:
        # Con filtro no se lee el matchups del disco: podría ser de otro split.
        matchups_csv = output_csv.parent / "matchups_1v1.csv"
        if matchups_csv.exists():
            try:
                matchups_df = pd.read_csv(matchups_csv)
            except (OSError, ValueError) as err:
                # Archivo bloqueado, vacío o corrupto (EmptyDataError, ParserError y
                # UnicodeDecodeError son ValueError). El CSV de jugadores sale igual,
                # sin los diferenciales de línea, pero queda dicho por qué.
                print(f"[!] No se pudo leer '{matchups_csv.name}' ({type(err).__name__}: {err}); "
                      "players.csv sale sin los diferenciales de línea.")
    if matchups_df is not None and cols_needed.issubset(matchups_df.columns):
        # `matchups_1v1.csv` ya trae `player_id` por lado, así que los diferenciales
        # caen directo en la fila unificada del jugador: no hace falta traducir PUUIDs.
        diff_cols = ['diff_gold_15', 'diff_cs_15', 'diff_cs_end', 'diff_damage_end', 'diff_gold_pct_snowball']
        b = matchups_df[['blue_player_id'] + diff_cols].rename(columns={'blue_player_id': 'player_id'})
        r = matchups_df[['red_player_id'] + diff_cols].copy()
        for col in diff_cols:
            # Sin timeline las columnas llegan todas en None (dtype object) y
            # negarlas levanta TypeError; como número quedan NaN y se ignoran.
            b[col] = pd.to_numeric(b[col], errors='coerce')
            r[col] = -pd.to_numeric(r[col], errors='coerce')
        r = r.rename(columns={'red_player_id': 'player_id'})
        all_diffs = pd.concat([b, r], ignore_index=True)
        stats = all_diffs.groupby('player_id').agg({
            'diff_gold_15': 'mean',
            'diff_cs_15': 'mean',
            'diff_cs_end': 'mean',
            'diff_damage_end': 'mean',
            'diff_gold_pct_snowball': 'mean'
        }).round(1)
        stats.columns = ['avg_gold_diff_at_15', 'avg_cs_diff_at_15', 'avg_cs_diff_end', 'avg_damage_diff_end', 'avg_gold_pct_diff_delta']
        df = df.merge(stats, on='player_id', how='left')
        for col in stats.columns:
            df[col] = df[col].fillna(0.0)

    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"[+] CSV 2 exportado con exito: '{output_csv.name}' ({len(df)} jugadores únicos, {len(df.columns)} columnas, {output_csv.stat().st_size / (1024*1024):.2f} MB)")
    return df


def export_champions_by_lane(db_path: Path, output_csv: Path,
                             match_ids: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Exporta estadísticas agregadas por cada campeón en cada posición (TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY).
    """
    with _export_snapshot(db_path, identity=False) as conn:
        return _export_champions_by_lane(conn, output_csv, match_ids)


def _export_champions_by_lane(conn, output_csv: Path, match_ids):
    print(f"[*] Generando CSV 3: Campeones por Posición/Línea...")
    filtro = _match_filter(conn, match_ids, "p.match_id", "AND")
    query = """
        SELECT 
            p.champion_name,
            p.individual_position AS lane,
            COUNT(*) AS games_played,
            SUM(p.win) AS wins,
            COUNT(*) - SUM(p.win) AS losses,
            ROUND(100.0 * SUM(p.win) / COUNT(*), 2) AS winrate_pct,
            ROUND(AVG(p.kills), 2) AS avg_kills,
            ROUND(AVG(p.deaths), 2) AS avg_deaths,
            ROUND(AVG(p.assists), 2) AS avg_assists,
            ROUND(AVG(p.kda), 2) AS avg_kda,
            -- Promedios Timeline Minuto 10
            ROUND(AVG(t.cs_10), 1) AS avg_cs_at_10,
            ROUND(AVG(t.gold_10), 0) AS avg_gold_at_10,
            ROUND(AVG(t.xp_10), 0) AS avg_xp_at_10,
            ROUND(AVG(t.damage_10), 0) AS avg_damage_at_10,
            -- Promedios Timeline Minuto 15
            ROUND(AVG(t.cs_15), 1) AS avg_cs_at_15,
            ROUND(AVG(t.gold_15), 0) AS avg_gold_at_15,
            ROUND(AVG(t.xp_15), 0) AS avg_xp_at_15,
            ROUND(AVG(t.damage_15), 0) AS avg_damage_at_15,
            -- Promedios Timeline Minuto 20
            ROUND(AVG(t.cs_20), 1) AS avg_cs_at_20,
            ROUND(AVG(t.gold_20), 0) AS avg_gold_at_20,
            ROUND(AVG(t.xp_20), 0) AS avg_xp_at_20,
            ROUND(AVG(t.damage_20), 0) AS avg_damage_at_20,
            -- Métricas Generales Finales
            ROUND(AVG(p.total_damage_dealt_to_champions), 0) AS avg_total_damage,
            ROUND(AVG(p.total_damage_taken), 0) AS avg_damage_taken,
            ROUND(AVG(p.gold_earned), 0) AS avg_gold_earned,
            ROUND(AVG(p.total_minions_killed + p.neutral_minions_killed), 1) AS avg_total_cs,
            ROUND(AVG(p.vision_score), 1) AS avg_vision_score,
            ROUND(100.0 * AVG(p.first_blood_kill), 2) AS first_blood_rate_pct,
            ROUND(100.0 * AVG(p.first_tower_kill), 2) AS first_tower_rate_pct
        FROM match_participants p
        LEFT JOIN match_timeline t ON p.match_id = t.match_id AND p.participant_id = t.participant_id
        WHERE p.individual_position IN ('TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY')
        <<FILTRO>>
        GROUP BY p.champion_name, p.individual_position
        HAVING COUNT(*) >= 1
        ORDER BY games_played DESC, winrate_pct DESC;
    """
    df = pd.read_sql_query(query.replace(_FILTRO, filtro), conn)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"[+] CSV 3 exportado con exito: '{output_csv.name}' ({len(df)} combinaciones campeón/línea, {output_csv.stat().st_size / (1024*1024):.2f} MB)")
    return df


def export_matchups_1v1(db_path: Path, output_csv: Path,
                        match_ids: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Exporta enfrentamientos directos de línea (Blue vs Red) con diferenciales de Oro, CS, Daño y resultado.
    """
    with _export_snapshot(db_path, identity=True) as conn:
        return _export_matchups_1v1(conn, output_csv, match_ids)


def _export_matchups_1v1(conn, output_csv: Path, match_ids):
    print(f"[*] Generando CSV 4: Enfrentamientos Directos de Línea (Matchups 1v1)...")
    filtro = _match_filter(conn, match_ids, "b.match_id", "AND")
    query = """
        SELECT 
            m.match_id,
            m.game_duration,
            m.game_version,
            b.individual_position AS lane,
            -- Blue Team (Team 100)
            b.champion_name AS blue_champion,
            pb.player_id AS blue_player_id,
            b.puuid AS blue_puuid_raw,
            b.win AS blue_win,
            b.kills AS blue_kills,
            b.deaths AS blue_deaths,
            b.assists AS blue_assists,
            b.total_damage_dealt_to_champions AS blue_total_damage,
            b.gold_earned AS blue_total_gold,
            -- Red Team (Team 200)
            r.champion_name AS red_champion,
            pr.player_id AS red_player_id,
            r.puuid AS red_puuid_raw,
            r.win AS red_win,
            r.kills AS red_kills,
            r.deaths AS red_deaths,
            r.assists AS red_assists,
            r.total_damage_dealt_to_champions AS red_total_damage,
            r.gold_earned AS red_total_gold,
            -- Diferenciales Minuto 10 (Azul - Rojo)
            (tb.gold_10 - tr.gold_10) AS diff_gold_10,
            (tb.cs_10 - tr.cs_10) AS diff_cs_10,
            (tb.xp_10 - tr.xp_10) AS diff_xp_10,
            (tb.damage_10 - tr.damage_10) AS diff_damage_10,
            (tb.kills_10 - tr.kills_10) AS diff_kills_10,
            -- Diferenciales Minuto 15 (Azul - Rojo)
            (tb.gold_15 - tr.gold_15) AS diff_gold_15,
            (tb.cs_15 - tr.cs_15) AS diff_cs_15,
            (tb.xp_15 - tr.xp_15) AS diff_xp_15,
            (tb.damage_15 - tr.damage_15) AS diff_damage_15,
            (tb.kills_15 - tr.kills_15) AS diff_kills_15,
            -- Diferenciales Minuto 20 (Azul - Rojo)
            (tb.gold_20 - tr.gold_20) AS diff_gold_20,
            (tb.cs_20 - tr.cs_20) AS diff_cs_20,
            (tb.xp_20 - tr.xp_20) AS diff_xp_20,
            (tb.damage_20 - tr.damage_20) AS diff_damage_20,
            -- Diferencial Final de CS y Daño a Campeones (Azul - Rojo)
            ((b.total_minions_killed + b.neutral_minions_killed) - (r.total_minions_killed + r.neutral_minions_killed)) AS diff_cs_end,
            (b.total_damage_dealt_to_champions - r.total_damage_dealt_to_champions) AS diff_damage_end,
            -- Aceleración de Ventaja de Oro / Bola de Nieve (% Oro End - % Oro 15 vs Rival)
            ROUND(
                (100.0 * (b.gold_earned - r.gold_earned) / MAX(1, b.gold_earned + r.gold_earned))
                -
                (100.0 * (COALESCE(tb.gold_15, 0) - COALESCE(tr.gold_15, 0)) / MAX(1, COALESCE(tb.gold_15, 0) + COALESCE(tr.gold_15, 0))),
                2
            ) AS diff_gold_pct_snowball
        FROM match_participants b
        INNER JOIN match_participants r 
            ON b.match_id = r.match_id 
            AND b.individual_position = r.individual_position 
            AND b.team_id = 100 
            AND r.team_id = 200
        INNER JOIN matches m ON b.match_id = m.match_id
        INNER JOIN player_id_map pb ON pb.puuid = b.puuid
        INNER JOIN player_id_map pr ON pr.puuid = r.puuid
        LEFT JOIN match_timeline tb ON b.match_id = tb.match_id AND b.participant_id = tb.participant_id
        LEFT JOIN match_timeline tr ON r.match_id = tr.match_id AND r.participant_id = tr.participant_id
        WHERE b.individual_position IN ('TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY')
        <<FILTRO>>
        ORDER BY m.match_id, b.individual_position;
    """
    df = pd.read_sql_query(query.replace(_FILTRO, filtro), conn)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"[+] CSV 4 exportado con exito: '{output_csv.name}' ({len(df)} enfrentamientos 1v1, {output_csv.stat().st_size / (1024*1024):.2f} MB)")
    return df


def export_all_datasets(db_path: Path = DEFAULT_DB_PATH, data_dir: Path = DATA_DIR) -> None:
    """Exporta todos los datasets requeridos para Machine Learning."""
    data_dir.mkdir(parents=True, exist_ok=True)
    csv1 = data_dir / "full_dataset.csv"
    csv2 = data_dir / "players.csv"
    csv3 = data_dir / "champions_by_lane.csv"
    csv4 = data_dir / "matchups_1v1.csv"

    print("\n" + "="*60)
    print("  EXPORTACIÓN AUTOMÁTICA DE DATASETS PARA MACHINE LEARNING")
    print("="*60)
    with _export_snapshot(db_path, identity=True) as conn:
        _export_full_dataset(conn, csv1, None)
        df_matchups = _export_matchups_1v1(conn, csv4, None)
        _export_players(conn, csv2, df_matchups, None)
        del df_matchups
        _export_champions_by_lane(conn, csv3, None)
    print("="*60 + "\n")


DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]


def export_match_times(db_path: Path, output_csv: Path) -> pd.DataFrame:
    """Día y hora de cada partida, en hora de Argentina (UTC-3, sin horario de verano).

    Una fila por partida; se cruza con los demás CSV por `match_id`. `inicio_art` es
    `game_creation` (cuando el servidor crea la partida). `fin_estimado_art` es
    `game_creation + game_duration`: puede quedar 1-2 min antes del fin real,
    porque `game_creation` es anterior al arranque por la pantalla de carga.
    """
    print("[*] Generando match_times.csv: día y hora de cada partida...")
    conn = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True, timeout=60.0)
    try:
        m = pd.read_sql_query("SELECT match_id, game_creation, game_duration FROM matches;", conn)
    finally:
        conn.close()
    inicio = pd.to_datetime(m["game_creation"], unit="ms") - pd.Timedelta(hours=3)
    fin = inicio + pd.to_timedelta(m["game_duration"], unit="s")
    df = pd.DataFrame({
        "match_id": m["match_id"],
        "servidor": m["match_id"].str.split("_").str[0],
        "game_creation": m["game_creation"],
        "inicio_art": inicio.dt.strftime("%Y-%m-%d %H:%M:%S"),
        "fin_estimado_art": fin.dt.strftime("%Y-%m-%d %H:%M:%S"),
        "fecha": inicio.dt.strftime("%Y-%m-%d"),
        "dia_semana": inicio.dt.dayofweek.map(dict(enumerate(DIAS))),
        "hora": inicio.dt.hour,
        "duracion_min": (m["game_duration"] / 60).round(2),
    }).sort_values("game_creation")
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"[+] match_times.csv exportado: {len(df):,} partidas "
          f"({df['fecha'].min()} -> {df['fecha'].max()}).")
    return df


def compute_split_labels(db_path: Path, test_frac: float = 0.2):
    """Etiquetas por partida (`master_plus_temporal_split`, src/features/split.py), leídas de la BD.

    Solo partidas de LAS: las de otros servidores quedan fuera de los dos splits.
    """
    from src.features.split import master_plus_temporal_split, match_tiers

    conn = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True, timeout=60.0)
    try:
        attach_for_participants(conn)
        parts = pd.read_sql_query("""
            SELECT p.match_id, pim.player_id, m.game_creation
            FROM match_participants p
            JOIN player_id_map pim ON pim.puuid = p.puuid
            JOIN matches m ON m.match_id = p.match_id
            WHERE substr(p.match_id, 1, 4) = 'LA2_'
        """, conn)
        padron = pd.read_sql_query(f"""
            SELECT pim.player_id, {_TIER_PADRON} AS player_tier
            FROM players pl JOIN player_id_map pim ON pim.puuid = pl.puuid
            GROUP BY pim.player_id
        """, conn)
    finally:
        conn.close()
    tiers = match_tiers(parts, padron)
    m = (parts.drop_duplicates("match_id")[["match_id", "game_creation"]]
         .assign(tier=lambda d: d["match_id"].map(tiers)))
    labels, resumen = master_plus_temporal_split(m, test_frac)
    return labels, m.set_index("match_id")["tier"], resumen


def export_splits(db_path: Path = DEFAULT_DB_PATH, out_dir: Path = DATA_DIR / "splits",
                  test_frac: float = 0.2) -> dict:
    """Exporta train y test por separado, sin leakage en los agregados (§9.S.14).

    * `match_split.csv`: la asignación de cada partida (train / test / diamante_antes /
      diamante_despues / sin_rango). Es la fuente única: cualquier análisis que use el
      split la lee de acá.
    * `train/`: los 4 CSV, con `players.csv` y `champions_by_lane.csv` agregados
      **solo con partidas de train**.
    * `test/`: `full_dataset.csv` y `matchups_1v1.csv` del test, filas crudas. No se
      exportan agregados del test: el modelo usa los del train.
    * `diamante/despues/`: lo mismo para las partidas de rango Diamond desde el corte,
      apartadas del test (evalúan el modelo de Master+ en Diamond). El Diamond anterior
      al corte va a `train/` (`DIAMANTE_ANTES_AL_TRAIN`); si se desactiva, va a
      `diamante/antes/`.

    Los CSV de `data/` (el volcado completo) no se tocan (AGENTS.md §3.G).
    """
    labels, tiers, resumen = compute_split_labels(db_path, test_frac)
    out_dir = Path(out_dir)
    for sub in ("train", "test", "diamante/antes", "diamante/despues"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"match_id": labels.index, "split": labels.values,
                  "tier": tiers.reindex(labels.index).values}).to_csv(out_dir / "match_split.csv", index=False)
    train_ids = list(labels[labels == "train"].index)
    test_ids = list(labels[labels == "test"].index)

    print("\n" + "=" * 60)
    print(f"  SPLITS: corte {resumen['corte']} | train {resumen['train']:,} | test {resumen['test']:,} "
          f"| diamante {resumen['diamante_antes']:,} + {resumen['diamante_despues']:,} "
          f"| sin rango {resumen['sin_rango']:,}")
    print("=" * 60)
    export_full_dataset(db_path, out_dir / "train" / "full_dataset.csv", match_ids=train_ids)
    mu = export_matchups_1v1(db_path, out_dir / "train" / "matchups_1v1.csv", match_ids=train_ids)
    export_players(db_path, out_dir / "train" / "players.csv", matchups_df=mu, match_ids=train_ids)
    export_champions_by_lane(db_path, out_dir / "train" / "champions_by_lane.csv", match_ids=train_ids)
    export_full_dataset(db_path, out_dir / "test" / "full_dataset.csv", match_ids=test_ids)
    export_matchups_1v1(db_path, out_dir / "test" / "matchups_1v1.csv", match_ids=test_ids)
    for etiqueta, sub in (("diamante_antes", "diamante/antes"), ("diamante_despues", "diamante/despues")):
        ids = list(labels[labels == etiqueta].index)
        if not ids:  # con DIAMANTE_ANTES_AL_TRAIN el Diamond viejo ya está en train/
            continue
        export_full_dataset(db_path, out_dir / sub / "full_dataset.csv", match_ids=ids)
        export_matchups_1v1(db_path, out_dir / sub / "matchups_1v1.csv", match_ids=ids)

    manifest = {**resumen, "test_frac": test_frac, "regla": "src/features/split.py (§9.S.12)"}
    (out_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                                 encoding="utf-8")
    print(f"[+] Splits en {out_dir}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Exporta los datasets en CSV desde SQLite")
    parser.add_argument(
        "--target",
        type=str,
        default="all",
        choices=["all", "full_dataset", "players", "champions_by_lane", "matchups_1v1", "splits",
                 "match_times"],
        help="Dataset a exportar (default: all)"
    )
    args = parser.parse_args()

    db_path = DEFAULT_DB_PATH
    if not db_path.exists():
        print(f"[X] No se encontró la base de datos en {db_path}. Ejecuta collect_data.py primero.")
        return

    if args.target == "all":
        export_all_datasets(db_path, DATA_DIR)
    elif args.target == "full_dataset":
        export_full_dataset(db_path, DATA_DIR / "full_dataset.csv")
    elif args.target == "players":
        export_players(db_path, DATA_DIR / "players.csv")
    elif args.target == "champions_by_lane":
        export_champions_by_lane(db_path, DATA_DIR / "champions_by_lane.csv")
    elif args.target == "matchups_1v1":
        export_matchups_1v1(db_path, DATA_DIR / "matchups_1v1.csv")
    elif args.target == "splits":
        export_splits(db_path, DATA_DIR / "splits")
    elif args.target == "match_times":
        export_match_times(db_path, DATA_DIR / "match_times.csv")


if __name__ == "__main__":
    main()
