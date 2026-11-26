"""Cálculo del perfil de roles causal por participación y perfil global por jugador.

Construye el insumo de roles para el estudio de matchmaking (intercambio de jugadores
entre posiciones/equipos sin afectar significativamente los roles):
1. Lee en solo lectura `data/lol_las.db` (cola 420 Ranked Solo/Duo), unificando
   identidades vía `player_id_map` y tomando `individual_position` como el rol
   asignado por Riot.
2. Calcula de forma estrictamente causal (partidas estrictamente anteriores por
   `player_id`, ordenadas por `game_creation`, `match_id`) los conteos previos por rol
   y la distribución suavizada de Dirichlet/Laplace:
       pi_i(r) = (n_r + alpha * prior_r) / (n_prev + alpha), con alpha = 5.
3. Calcula:
   - `rol_principal`: argmax_r pi_i(r)
   - `d`: pi_i(rol_principal) - pi_i(rol_asignado) in [0, 1]
   - `entropia`: entropía normalizada de pi_i en [0, 1] (0 = mono-rol, 1 = uniforme)
   - `autofill`: (pi_i(rol_asignado) < 0.15) & (n_prev >= 20)
4. Exporta:
   - `data/analysis/matchmaking/perfil_roles.parquet` (una fila por match_id, puuid)
   - `data/analysis/matchmaking/perfil_roles_jugador.parquet` (perfil global por player_id)
   - `data/analysis/matchmaking/REPORTE_roles.md` (resumen descriptivo completo)

Uso:
    python scripts/analysis/matchmaking_perfil_roles.py
"""
from contextlib import closing
from datetime import datetime
from pathlib import Path
import sqlite3
import time

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DB_PATH = REPO / "data" / "lol_las.db"
OUT_DIR = REPO / "data" / "analysis" / "matchmaking"
LOCK_FILE = OUT_DIR / "RECURSO_EN_USO.txt"

ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
ALPHA = 5.0
AUTOFILL_PI_THRESHOLD = 0.15
AUTOFILL_MIN_PREV = 20

TIER_RANK = {
    "DIAMOND": 1,
    "MASTER": 2,
    "GRANDMASTER": 3,
    "CHALLENGER": 4,
}
RANK_TO_TIER = {v: k for k, v in TIER_RANK.items()}


def _esperar_y_tomar_lock() -> None:
    """Respeta el lock de máquina compartida en data/analysis/matchmaking/RECURSO_EN_USO.txt."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    while LOCK_FILE.exists():
        print("Esperando liberación de RECURSO_EN_USO.txt...", flush=True)
        time.sleep(5)
    LOCK_FILE.write_text(
        f"hydrogen-roles | {datetime.now().isoformat(timespec='seconds')}\n",
        encoding="utf-8",
    )


def _liberar_lock() -> None:
    if LOCK_FILE.exists():
        LOCK_FILE.unlink()


def _cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lee participaciones (cola 420) y tiers del padrón en solo lectura."""
    uri = f"file:{DB_PATH.as_posix()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as conn:
        query_mp = """
            SELECT
                mp.match_id,
                mp.puuid,
                m.player_id,
                mt.game_creation,
                mp.team_id,
                mp.individual_position AS rol_asignado
            FROM match_participants mp
            JOIN matches mt ON mp.match_id = mt.match_id
            JOIN player_id_map m ON mp.puuid = m.puuid
            WHERE mt.queue_id = 420
        """
        df = pd.read_sql_query(query_mp, conn)

        query_tiers = """
            SELECT m.player_id, p.tier
            FROM player_id_map m
            JOIN players p ON m.puuid = p.puuid
            WHERE p.tier IS NOT NULL
        """
        tiers_df = pd.read_sql_query(query_tiers, conn)

    return df, tiers_df


def _resolver_tiers(tiers_df: pd.DataFrame) -> pd.Series:
    """Devuelve el tier más alto observado en `players` por player_id."""
    tiers_df = tiers_df.copy()
    tiers_df["tier_num"] = tiers_df["tier"].map(TIER_RANK).fillna(0).astype(np.int8)
    best = tiers_df.groupby("player_id")["tier_num"].max()
    return best.map(RANK_TO_TIER)


def _calcular_entropia_normalizada(pi_mat: np.ndarray) -> np.ndarray:
    """Entropía de Shannon normalizada por log(5), en [0, 1]."""
    safe_pi = np.clip(pi_mat, 1e-12, 1.0)
    ent = -np.sum(safe_pi * np.log(safe_pi), axis=1) / np.log(len(ROLES))
    return np.clip(ent, 0.0, 1.0).astype(np.float32)


def _resumir_subconjunto(sub: pd.DataFrame, nombre: str) -> dict:
    """Calcula métricas descriptivas para un DataFrame de participaciones."""
    d_vals = sub["d"]
    qs = d_vals.quantile([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).to_dict()
    en_principal = (sub["rol_asignado"] == sub["rol_principal"]).mean() * 100.0
    tasa_autofill = sub["autofill"].mean() * 100.0
    por_rol = sub.groupby("rol_asignado")["d"].agg(["count", "mean", "median"]).reindex(ROLES)
    por_tier = (
        sub[sub["lobby_tier"].isin(["DIAMOND", "MASTER+"])]
        .groupby("lobby_tier")["d"]
        .agg(["count", "mean", "median"])
    )
    autofill_por_tier = (
        sub[sub["lobby_tier"].isin(["DIAMOND", "MASTER+"])]
        .groupby("lobby_tier")["autofill"]
        .mean()
        * 100.0
    )
    return {
        "nombre": nombre,
        "n_filas": len(sub),
        "d_mean": float(d_vals.mean()),
        "d_std": float(d_vals.std()),
        "d_quantiles": {k: float(v) for k, v in qs.items()},
        "pct_rol_principal": float(en_principal),
        "pct_autofill": float(tasa_autofill),
        "por_rol": por_rol,
        "por_tier": por_tier,
        "autofill_por_tier": autofill_por_tier,
    }


def _formatear_reporte(
    meta: dict,
    res_all: dict,
    res_maduro: dict,
    prior: np.ndarray,
    n_jugadores: int,
    n_jugadores_20: int,
) -> str:
    """Construye el markdown de REPORTE_roles.md."""
    lines = [
        "# Reporte Descriptivo: Perfil Causal de Roles (`matchmaking`)",
        "",
        f"- **Fecha de generación:** {meta['timestamp']}",
        f"- **Tiempo de ejecución:** {meta['elapsed_s']:.2f} s",
        f"- **Fuente:** `data/lol_las.db` (cola `420` Ranked Solo/Duo, unificado por `player_id_map`)",
        f"- **Posición utilizada:** `individual_position` (nota: la columna `team_position` no existe en `match_participants` de `lol_las.db`)",
        f"- **Participaciones totales en DB (cola 420):** {meta['total_raw']:,}",
        f"- **Participaciones excluidas por `individual_position == 'Invalid'`:** {meta['n_invalid']:,}",
        f"- **Participaciones válidas exportadas (`perfil_roles.parquet`):** {meta['total_valid']:,}",
        f"- **Jugadores únicos (`perfil_roles_jugador.parquet`):** {n_jugadores:,} (con `n_total >= 20`: {n_jugadores_20:,})",
        "",
        "## Prior Global de Roles y Parámetros",
        "",
        f"- Suavizado: $\\pi_i(r) = (n_r + \\alpha \\cdot \\text{{prior}}_r) / (n_{{\\text{{prev}}}} + \\alpha)$ con $\\alpha = {ALPHA:.0f}$.",
        f"- Definición de `autofill`: $\\pi_i(r_{{\\text{{asignado}}}}) < {AUTOFILL_PI_THRESHOLD}$ y $n_{{\\text{{prev}}}} \\ge {AUTOFILL_MIN_PREV}$.",
        "- Clasificación `lobby_tier`: para cada partida, según los jugadores con tier conocido en `players` (vía `player_id`), `MASTER+` si al menos la mitad son Master/Grandmaster/Challenger, `DIAMOND` si la mayoría son Diamond, `UNKNOWN` si ningún jugador tiene tier en `players`.",
        "",
        "| Rol | Prior global |",
        "| :--- | ---: |",
    ]
    for r, p in zip(ROLES, prior):
        lines.append(f"| {r} | {p:.6f} |")

    for res in [res_all, res_maduro]:
        q = res["d_quantiles"]
        lines.extend([
            "",
            f"## {res['nombre']} ($N = {res['n_filas']:,}$)",
            "",
            f"- **Media de $d$:** {res['d_mean']:.4f} (desvío estándar: {res['d_std']:.4f})",
            f"- **% en su rol principal (`rol_asignado == rol_principal`):** {res['pct_rol_principal']:.2f}%",
            f"- **Tasa de autofill (`pi < 0.15` y `n_prev >= 20`):** {res['pct_autofill']:.2f}%",
            "",
            "### Cuantiles de $d$",
            "",
            "| p10 | p25 | p50 (mediana) | p75 | p90 | p95 | p99 |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| {q[0.10]:.4f} | {q[0.25]:.4f} | {q[0.50]:.4f} | {q[0.75]:.4f} | {q[0.90]:.4f} | {q[0.95]:.4f} | {q[0.99]:.4f} |",
            "",
            "### Desvío medio $d$ por rol asignado",
            "",
            "| Rol asignado | Participaciones | Media $d$ | Mediana $d$ |",
            "| :--- | ---: | ---: | ---: |",
        ])
        for r, row in res["por_rol"].iterrows():
            lines.append(f"| {r} | {int(row['count']):,} | {row['mean']:.4f} | {row['median']:.4f} |")

        lines.extend([
            "",
            "### Desvío medio $d$ y autofill por tier del lobby",
            "",
            "| Lobby Tier | Participaciones | Media $d$ | Mediana $d$ | Tasa autofill (%) |",
            "| :--- | ---: | ---: | ---: | ---: |",
        ])
        for tier_name, row in res["por_tier"].iterrows():
            af = res["autofill_por_tier"].get(tier_name, float("nan"))
            lines.append(
                f"| {tier_name} | {int(row['count']):,} | {row['mean']:.4f} | {row['median']:.4f} | {af:.2f}% |"
            )

    lines.extend([
        "",
        "## Esquema de Archivos Generados",
        "",
        "### `data/analysis/matchmaking/perfil_roles.parquet`",
        "- `match_id` (`string`): ID de la partida.",
        "- `puuid` (`string`): PUUID del participante en la partida.",
        "- `player_id` (`string`, nombre#tag): ID unificado del jugador vía `player_id_map`.",
        "- `game_creation` (`int64`): Timestamp Unix en milisegundos de inicio de la partida.",
        "- `team_id` (`int32`): `100` (Blue) o `200` (Red).",
        "- `lado` (`string`): `'BLUE'` o `'RED'`.",
        "- `lobby_tier` (`string`): `'DIAMOND'`, `'MASTER+'` o `'UNKNOWN'`.",
        "- `rol_asignado` (`string`): Posición jugada/asignada (`TOP`, `JUNGLE`, `MIDDLE`, `BOTTOM`, `UTILITY`).",
        "- `n_prev` (`int32`): Cantidad de partidas válidas estrictamente anteriores del `player_id`.",
        "- `pi_TOP`, `pi_JUNGLE`, `pi_MIDDLE`, `pi_BOTTOM`, `pi_UTILITY` (`float32`): Probabilidades suavizadas causales.",
        "- `rol_principal` (`string`): Rol con mayor $\\pi_i(r)$ previo.",
        "- `d` (`float32`): $\\pi_i(\\text{rol\\_principal}) - \\pi_i(\\text{rol\\_asignado}) \\in [0, 1]$.",
        "- `entropia` (`float32`): Entropía normalizada de $\\pi_i$ en $[0, 1]$.",
        "- `autofill` (`bool`): `True` si $\\pi_i(\\text{rol\\_asignado}) < 0.15$ y $n_{\\text{prev}} \\ge 20$.",
        "",
        "### `data/analysis/matchmaking/perfil_roles_jugador.parquet`",
        "- `player_id` (`string`, nombre#tag): ID unificado del jugador.",
        "- `tier` (`string`, nullable): Tier más alto del jugador en `players` (`DIAMOND`, `MASTER`, `GRANDMASTER`, `CHALLENGER`).",
        "- `n_total` (`int32`): Total de partidas válidas del jugador en la base.",
        "- `n_TOP`, `n_JUNGLE`, `n_MIDDLE`, `n_BOTTOM`, `n_UTILITY` (`int32`): Conteos totales por rol.",
        "- `pi_TOP`, `pi_JUNGLE`, `pi_MIDDLE`, `pi_BOTTOM`, `pi_UTILITY` (`float32`): Distribución global suavizada ($\\alpha = 5$).",
        "- `rol_principal` (`string`): Rol principal global del jugador.",
        "- `entropia` (`float32`): Entropía normalizada global en $[0, 1]$.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    t0 = time.time()
    _esperar_y_tomar_lock()
    try:
        print("1/5 Cargando datos desde data/lol_las.db...", flush=True)
        df, tiers_df = _cargar_datos()
        total_raw = len(df)
        valid_mask = df["rol_asignado"].isin(ROLES)
        n_invalid = int((~valid_mask).sum())
        df = df[valid_mask].copy()
        total_valid = len(df)

        # Prior global sobre todas las participaciones válidas
        role_counts = df["rol_asignado"].value_counts().reindex(ROLES, fill_value=0)
        prior = (role_counts / role_counts.sum()).to_numpy(dtype=np.float64)

        print("2/5 Calculando lobby_tier...", flush=True)
        pid_to_tier = _resolver_tiers(tiers_df)
        df["player_tier"] = df["player_id"].map(pid_to_tier)
        is_mplus = df["player_tier"].isin(["MASTER", "GRANDMASTER", "CHALLENGER"]).astype(np.float32)
        has_tier = df["player_tier"].notna().astype(np.float32)
        df["_mplus"] = np.where(has_tier > 0, is_mplus, np.nan)
        match_mplus_mean = df.groupby("match_id")["_mplus"].transform("mean")
        df["lobby_tier"] = np.where(
            match_mplus_mean.isna(),
            "UNKNOWN",
            np.where(match_mplus_mean >= 0.5, "MASTER+", "DIAMOND"),
        )
        df.drop(columns=["_mplus", "player_tier"], inplace=True)

        print("3/5 Calculando perfil causal vectorizado por (player_id, game_creation, match_id)...", flush=True)
        df.sort_values(["player_id", "game_creation", "match_id"], kind="mergesort", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Indicadores one-hot por rol en int32
        role_idx = pd.Categorical(df["rol_asignado"], categories=ROLES).codes
        n_rows = len(df)
        ind = np.zeros((n_rows, len(ROLES)), dtype=np.int32)
        ind[np.arange(n_rows), role_idx] = 1

        # Cumsum vectorizado por player_id menos la fila actual
        ind_df = pd.DataFrame(ind, columns=ROLES)
        cum_df = ind_df.groupby(df["player_id"], sort=False).cumsum()
        prev_counts = (cum_df.to_numpy(dtype=np.int32) - ind)
        n_prev = prev_counts.sum(axis=1).astype(np.int32)

        # Distribución suavizada causal
        denom = (n_prev.astype(np.float64)[:, None] + ALPHA)
        pi_mat = (prev_counts.astype(np.float64) + ALPHA * prior[None, :]) / denom

        max_idx = np.argmax(pi_mat, axis=1)
        pi_max = pi_mat[np.arange(n_rows), max_idx]
        pi_assigned = pi_mat[np.arange(n_rows), role_idx]

        rol_principal = np.array(ROLES)[max_idx]
        d_score = np.clip(pi_max - pi_assigned, 0.0, 1.0).astype(np.float32)
        entropia = _calcular_entropia_normalizada(pi_mat)
        autofill = (pi_assigned < AUTOFILL_PI_THRESHOLD) & (n_prev >= AUTOFILL_MIN_PREV)

        out_df = pd.DataFrame({
            "match_id": df["match_id"].astype(str),
            "puuid": df["puuid"].astype(str),
            "player_id": df["player_id"].astype(str),
            "game_creation": df["game_creation"].astype(np.int64),
            "team_id": df["team_id"].astype(np.int32),
            "lado": np.where(df["team_id"].to_numpy() == 100, "BLUE", "RED"),
            "lobby_tier": df["lobby_tier"].astype(str),
            "rol_asignado": df["rol_asignado"].astype(str),
            "n_prev": n_prev,
            "pi_TOP": pi_mat[:, 0].astype(np.float32),
            "pi_JUNGLE": pi_mat[:, 1].astype(np.float32),
            "pi_MIDDLE": pi_mat[:, 2].astype(np.float32),
            "pi_BOTTOM": pi_mat[:, 3].astype(np.float32),
            "pi_UTILITY": pi_mat[:, 4].astype(np.float32),
            "rol_principal": rol_principal,
            "d": d_score,
            "entropia": entropia,
            "autofill": autofill,
        })

        print("4/5 Calculando perfil global por jugador y exportando parquets...", flush=True)
        # Perfil global no causal por player_id
        ind_df["player_id"] = df["player_id"].to_numpy()
        glob_counts = ind_df.groupby("player_id", sort=True)[ROLES].sum()
        g_mat = glob_counts.to_numpy(dtype=np.int32)
        g_tot = g_mat.sum(axis=1).astype(np.int32)
        g_pi = (g_mat.astype(np.float64) + ALPHA * prior[None, :]) / (g_tot.astype(np.float64)[:, None] + ALPHA)
        g_max_idx = np.argmax(g_pi, axis=1)

        player_out = pd.DataFrame({
            "player_id": glob_counts.index.astype(str),
            "tier": pd.Series(glob_counts.index).map(pid_to_tier).to_numpy(),
            "n_total": g_tot,
            "n_TOP": g_mat[:, 0],
            "n_JUNGLE": g_mat[:, 1],
            "n_MIDDLE": g_mat[:, 2],
            "n_BOTTOM": g_mat[:, 3],
            "n_UTILITY": g_mat[:, 4],
            "pi_TOP": g_pi[:, 0].astype(np.float32),
            "pi_JUNGLE": g_pi[:, 1].astype(np.float32),
            "pi_MIDDLE": g_pi[:, 2].astype(np.float32),
            "pi_BOTTOM": g_pi[:, 3].astype(np.float32),
            "pi_UTILITY": g_pi[:, 4].astype(np.float32),
            "rol_principal": np.array(ROLES)[g_max_idx],
            "entropia": _calcular_entropia_normalizada(g_pi),
        })

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_df.to_parquet(OUT_DIR / "perfil_roles.parquet", index=False)
        player_out.to_parquet(OUT_DIR / "perfil_roles_jugador.parquet", index=False)

        print("5/5 Generando resumen descriptivo y REPORTE_roles.md...", flush=True)
        res_all = _resumir_subconjunto(out_df, "Todas las participaciones")
        res_maduro = _resumir_subconjunto(out_df[out_df["n_prev"] >= 20], "Participaciones maduras (`n_prev >= 20`)")

        elapsed_s = time.time() - t0
        meta = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "elapsed_s": elapsed_s,
            "total_raw": total_raw,
            "n_invalid": n_invalid,
            "total_valid": total_valid,
        }
        report_md = _formatear_reporte(
            meta=meta,
            res_all=res_all,
            res_maduro=res_maduro,
            prior=prior,
            n_jugadores=len(player_out),
            n_jugadores_20=int((player_out["n_total"] >= 20).sum()),
        )
        (OUT_DIR / "REPORTE_roles.md").write_text(report_md, encoding="utf-8")
        print(f"Completado en {elapsed_s:.2f}s.", flush=True)
    finally:
        _liberar_lock()


if __name__ == "__main__":
    main()
