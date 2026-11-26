"""Verificación anti-fuga del preprocesamiento del escalón 9 (PROJECT_SUMMARY.md §9.S.68).

Comprueba dos propiedades:
1. Funcional (mutación de V): congelar `lane_scale`, la mediana de `log_horas_desde` y `mu0`
   de `prior_lobby_level` estrictamente en `~val0` vuelve las filas de entrenamiento (`~val0`)
   y la primera partida de `val0` exactamente invariantes ante perturbaciones arbitrarias en `val0`.
   Además comprueba el contraejemplo: sin congelar en `~val0`, mutar `val0` altera filas de `~val0`.
2. Estructural sobre los scripts que generan las cachés del escalón 9: asegura que no vuelva a
   introducirse `lane_scale(df, ...)`, `.median()` global ni `D["fit"]` (que incluye `diam_v`).
"""
import os
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.analysis import brainstorm_base
from src.ganador import rutas
from src.features.lobby import prior_lobby_level
from src.features.margin import ORDEN, lane_scale, prior_margin, prior_margin_by
from src.features.padron import PadronCambiado, T_FOTO1_MS, T_FOTO1_TXT, verificar_foto1

REPO = Path(__file__).resolve().parents[1]
LANES = ["BOTTOM", "JUNGLE", "MIDDLE", "TOP", "UTILITY"]


def _datos_con_val(n_fit: int = 35, n_val: int = 15, seed: int = 7) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_tot = n_fit + n_val
    lp_jug = {f"P{p}": (float(rng.normal(200, 180)) if p < 14 else np.nan) for p in range(25)}
    filas = []
    t0 = 1_750_000_000_000
    for m in range(n_tot):
        jug = rng.choice(25, 10, replace=False)
        gana = int(rng.integers(0, 2))
        tc = t0 + m * 3_600_000
        for i, p in enumerate(jug):
            team = 100 if i < 5 else 200
            pid = f"P{p}"
            filas.append({
                "match_id": f"M{m:03d}",
                "game_creation": tc,
                "team_id": team,
                "lane": LANES[i % 5],
                "player_id": pid,
                "champion_id": int(rng.integers(0, 6)),
                "win": int(gana == (team == 100)),
                "x": float(rng.normal(10.0, 2.0)),
                "lp": lp_jug[pid],
            })
    df = pd.DataFrame(filas).sort_values(ORDEN).reset_index(drop=True)
    val_fila = np.arange(len(df)) >= n_fit * 10
    return df, val_fila


def _pipeline_preproc(df: pd.DataFrame, val_fila: np.ndarray, congelar: bool) -> pd.DataFrame:
    ref = ~val_fila if congelar else np.ones(len(df), dtype=bool)
    esc = lane_scale(df.loc[ref], ["x"])
    m = prior_margin(df, ["x"], k_cl=5, k_j=3, window=5, escala=esc)
    mc = prior_margin_by(df, ["x"], ["champion_id"], "mc", k_cl=5, k_j=3, k_g=5, escala=esc)
    ml = prior_margin_by(df, ["x"], ["lane"], "ml", k_cl=5, k_j=3, k_g=5, escala=esc)
    horas = (df["game_creation"] - df.groupby("player_id")["game_creation"].shift(1)) / 3.6e6
    log_h = np.log1p(horas)
    log_h_imp = log_h.fillna(log_h.loc[ref].median()).rename("log_horas_desde")
    mu0 = float(df.loc[ref, "lp"].mean())
    niv = prior_lobby_level(df, "lp", k=5, mu0=mu0, window=5, min_otros=3)
    out = pd.concat([
        m, mc, ml, log_h_imp,
        (niv["nivel_acum"] - mu0).rename("niv_acum"),
        (niv["nivel_v5"] - mu0).rename("niv_v5"),
    ], axis=1)
    return out


def test_preprocesamiento_congelado_es_invariante_a_mutaciones_en_val():
    df, val_fila = _datos_con_val()
    corte = int(np.flatnonzero(val_fila)[0])

    # Mutar agresivamente todas las columnas futuras en val0 (estadísticas, espaciado temporal y LP)
    mut = df.copy()
    mut.loc[val_fila, "x"] = mut.loc[val_fila, "x"] * 25.0 + 500.0
    pasos_val = np.repeat(np.arange(1, int(val_fila.sum()) // 10 + 1, dtype=np.int64), 10)
    mut.loc[val_fila, "game_creation"] = mut.loc[val_fila, "game_creation"] + pasos_val * 86_400_000 * 5
    mut.loc[val_fila, "lp"] = 5000.0

    limpio_a = _pipeline_preproc(df, val_fila, congelar=True)
    limpio_b = _pipeline_preproc(mut, val_fila, congelar=True)
    # 1) Todo el entrenamiento (~val0) queda idéntico bit a bit
    pd.testing.assert_frame_equal(limpio_a.iloc[:corte], limpio_b.iloc[:corte], rtol=0, atol=0)
    # 2) La primera partida de val0 tampoco cambia en sus features de historial/nivel
    #    (salvo log_horas_desde de jugadores que ya jugaron antes y cuya fecha se desplazó;
    #    tolerancia 1e-11 por cancelación flotante de `(cumsum + r_mut) - r_mut`)
    cols_hist = [c for c in limpio_a.columns if c != "log_horas_desde"]
    pd.testing.assert_frame_equal(
        limpio_a.loc[corte:corte + 9, cols_hist],
        limpio_b.loc[corte:corte + 9, cols_hist],
        rtol=1e-11, atol=1e-11,
    )
    # 3) Contraejemplo: sin congelar en ~val0, mutar val0 contamina filas de entrenamiento
    fuga_a = _pipeline_preproc(df, val_fila, congelar=False)
    fuga_b = _pipeline_preproc(mut, val_fila, congelar=False)
    assert not np.allclose(fuga_a["acum_x"].iloc[:corte], fuga_b["acum_x"].iloc[:corte])
    assert not np.allclose(fuga_a["mc_x"].iloc[:corte], fuga_b["mc_x"].iloc[:corte])
    assert not np.allclose(fuga_a["ml_x"].iloc[:corte], fuga_b["ml_x"].iloc[:corte])
    assert not np.allclose(fuga_a["log_horas_desde"].iloc[:corte], fuga_b["log_horas_desde"].iloc[:corte])
    assert not np.allclose(fuga_a["niv_acum"].iloc[:corte], fuga_b["niv_acum"].iloc[:corte])


def test_scripts_escalon9_congelan_preprocesamiento_antes_de_v():
    emb = (REPO / "scripts" / "analysis" / "embeddings_partidas.py").read_text(encoding="utf-8")
    assert "lane_scale(df[~val_fila], STATS)" in emb
    assert "prior_margin(df, STATS, 20, 5, 20, escala=escala)" in emb
    assert "log_horas[~val_fila].median()" in emb
    assert "CA[~val_fila].mean()" in emb

    b05 = (REPO / "historial" / "escalon9" / "brainstorm_05_jugador.py").read_text(encoding="utf-8")
    assert 'lane_scale(df[np.repeat(~D["val0"], 10)], STATS)' in b05
    assert "lane_scale(df, STATS)" not in b05

    b07 = (REPO / "historial" / "escalon9" / "brainstorm_07_mmr_historico.py").read_text(encoding="utf-8")
    assert 'np.repeat(~D["val0"], 10)' in b07
    assert 'np.repeat(D["fit"], 10)' not in b07

    g06 = (REPO / "historial" / "escalon9" / "brainstorm_g_06_comp_poisson.py").read_text(encoding="utf-8")
    assert "C.mean().to_numpy(np.float32)" in g06
    assert "C.std().to_numpy(np.float32)" in g06


def test_foto1_rechaza_un_padron_con_fechas_distintas():
    with closing(sqlite3.connect(":memory:")) as con:
        con.execute("CREATE TABLE players (updated_at TIMESTAMP)")
        con.execute("INSERT INTO players VALUES (?)", (T_FOTO1_TXT,))
        assert verificar_foto1(con) == T_FOTO1_MS
        con.execute("INSERT INTO players VALUES (NULL), ('2026-11-19 18:00:00')")
        with pytest.raises(PadronCambiado):
            verificar_foto1(con)


def test_hgb_rechaza_un_insumo_mas_nuevo(tmp_path, monkeypatch):
    analysis = tmp_path / "data" / "analysis"
    carpeta = analysis / "brainstorm"
    carpeta.mkdir(parents=True)
    variables = analysis / "embeddings_partidas" / "variables.npz"
    variables.parent.mkdir()
    atributos = tmp_path / "atributos_campeones.csv"
    cruces = carpeta / "gemini_mpr.npz"
    for insumo in (variables, atributos, cruces, carpeta / "margen_grupos.npz", carpeta / "mmr_historico.npz"):
        insumo.touch()
        os.utime(insumo, (1_750_000_000, 1_750_000_000))
    ruta = carpeta / "gemini_hgb_lineas.npz"
    hgb = np.ones((2, 6))
    np.savez(ruta, HGB=hgb)
    monkeypatch.setattr(brainstorm_base, "REPO", tmp_path)
    monkeypatch.setattr(brainstorm_base, "CACHE", carpeta / "base.npz")
    monkeypatch.setattr(rutas, "ATRIB_CSV", atributos)
    np.testing.assert_array_equal(brainstorm_base.cargar_hgb_por_linea(), hgb)
    posterior = ruta.stat().st_mtime + 60
    os.utime(cruces, (posterior, posterior))
    with pytest.raises(ValueError, match="gemini_mpr.npz"):
        brainstorm_base.cargar_hgb_por_linea()
