"""Tests de src/features/lp_partida.py (modelo del LP partida a partida) y de cargar_enlazadas."""
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.features import lp_partida as L
from src.lp_vivo import ESQUEMA, cargar_enlazadas


def _fila(match, puuid, t, w, l, lp, win, team=100, tier="MASTER", delta=None, pos="TOP"):
    delta = (20 if win else -20) if delta is None else delta
    return {"match_id": match, "platform": "la2", "puuid": puuid, "game_end": t, "tier_antes": tier,
            "wins_antes": w, "losses_antes": l, "z_antes": 400.0 + lp, "z_despues": 400.0 + lp + delta,
            "delta_lp": float(delta), "win": win, "team_id": team, "duracion_s": 1800,
            "team_position": pos, "lobby_lp": np.nan, "lobby_conocidos": 0}


def _jugador(puuid, resultados, lp0=100, w0=50, l0=50, t0=0, match0=0, deltas=None):
    filas, lp, w, l = [], lp0, w0, l0
    for i, r in enumerate(resultados):
        dl = None if deltas is None else deltas[i]
        f = _fila(f"M{match0 + i}", puuid, t0 + i * 3_600_000, w, l, lp, r, delta=dl)
        filas.append(f)
        lp += f["delta_lp"]
        w, l = w + r, l + (1 - r)
    return filas


def test_racha_y_previas():
    d = L.preparar(pd.DataFrame(_jugador("a", [1, 1, 0, 0, 0, 1])))
    assert d["racha"].tolist() == [0, 1, 2, -1, -2, -3]
    assert d["previas_obs"].tolist() == list(range(6))
    assert d["horas_desde_previa"].iloc[1] == pytest.approx(1.0)


def test_ganancia_y_perdida_previa_son_estrictamente_anteriores():
    d = L.preparar(pd.DataFrame(_jugador("a", [1, 0, 1, 1], deltas=[22, -18, 25, 19])))
    assert d["ganancia_previa"].tolist()[1:] == [22, 22, 25]
    assert np.isnan(d["ganancia_previa"].iloc[0])
    assert d["perdida_previa"].tolist()[2:] == [-18, -18]


def test_cadena_se_corta_con_un_hueco_de_partidas():
    filas = _jugador("a", [1, 0, 1])
    filas[2]["wins_antes"] += 1          # jugó una partida que no se midió
    d = L.preparar(pd.DataFrame(filas + _jugador("b", [1, 1], match0=10)))
    assert d.groupby("puuid")["cadena"].nunique().to_dict() == {"a": 2, "b": 1}
    assert d.loc[d["puuid"] == "a", "paso"].tolist() == [0, 1, 0]


def test_cadena_se_corta_si_el_lp_no_empalma():
    filas = _jugador("a", [1, 1])
    filas[1]["z_antes"] += 5
    assert L.preparar(pd.DataFrame(filas))["cadena"].nunique() == 2


def test_companeros_y_rivales_sin_el_propio():
    filas = [_fila("M", "a", 0, 10, 10, 100, 1, team=100), _fila("M", "b", 0, 10, 10, 200, 1, team=100),
             _fila("M", "c", 0, 10, 10, 300, 0, team=200), _fila("M", "d", 0, 10, 10, 500, 0, team=200)]
    d = L.preparar(pd.DataFrame(filas)).set_index("puuid")
    assert d.loc["a", "equipo_lp"] == 600 and d.loc["a", "rival_lp"] == 800
    assert d.loc["c", "equipo_lp"] == 900 and d.loc["c", "rival_lp"] == 550
    assert d.loc["a", "gap_rival"] == 800 - 500


def test_regla_aprende_a_y_b_por_rango():
    filas = (_jugador("a", [1, 0] * 10, deltas=[21, -19] * 10)
             + [dict(f, tier_antes="GRANDMASTER") for f in _jugador("b", [1, 0] * 10, deltas=[15, -25] * 10, match0=100)])
    d = L.preparar(pd.DataFrame(filas))
    m = L.Regla().fit(d, d["delta_lp"])
    p = pd.Series(m.predict(d), index=d.index)
    assert p[(d["puuid"] == "a") & (d["win"] == 1)].unique().tolist() == [21]
    assert p[(d["puuid"] == "b") & (d["win"] == 0)].unique().tolist() == [-25]


def test_metricas_r2_dentro_del_resultado():
    y, win = np.array([20, 22, -18, -20.0]), np.array([1, 1, 0, 0])
    m = L.metricas(y, np.array([21, 21, -19, -19.0]), win)
    assert m["mae"] == pytest.approx(1.0)
    assert m["r2_dentro"] == pytest.approx(0.0)
    assert m["r2"] > 0.99


class _Exacto:
    """Predice el ΔLP real: el recorrido tiene que reproducir el LP real en cada paso."""

    def predict(self, d):
        return d["delta_lp"].to_numpy()


class _Constante:
    def predict(self, d):
        return np.full(len(d), 10.0)


def test_recorrido_con_modelo_exacto_reproduce_el_lp():
    d = L.preparar(pd.DataFrame(_jugador("a", [1, 0, 1, 1, 0]) + _jugador("b", [0, 0, 1], match0=20)))
    r = L.recorrido(d, _Exacto())
    assert np.allclose(r["z_hat_despues"], d.loc[r.index, "z_despues"])


def test_recorrido_encadena_desde_el_primer_lp_real():
    d = L.preparar(pd.DataFrame(_jugador("a", [1, 1, 1], lp0=100)))
    r = L.recorrido(d, _Constante()).loc[d.index]
    assert r["z_hat_antes"].tolist() == [500, 510, 520]
    assert r["z_hat_despues"].tolist() == [510, 520, 530]


def test_recorrido_recalcula_las_distancias_con_el_lp_predicho():
    vistos = []

    class _Espia(_Constante):
        def predict(self, d):
            vistos.append(d[["z_antes", "gap_lobby"]].copy())
            return super().predict(d)

    filas = _jugador("a", [1, 1], lp0=100)
    for f in filas:
        f["lobby_lp"] = 600.0
    d = L.preparar(pd.DataFrame(filas))
    L.recorrido(d, _Espia())
    assert vistos[1]["z_antes"].iloc[0] == 510 and vistos[1]["gap_lobby"].iloc[0] == 90


def _datos_sinteticos(n_jug=60, seed=0):
    rng = np.random.default_rng(seed)
    filas = []
    for j in range(n_jug):
        res = rng.integers(0, 2, 8).tolist()
        sesgo = rng.normal(0, 3)
        deltas = [(20 + sesgo if r else -20 + sesgo) for r in res]
        filas += _jugador(f"p{j}", res, lp0=int(rng.integers(0, 600)), match0=j * 100, deltas=deltas)
    return L.preparar(pd.DataFrame(filas))


def test_evaluar_fuera_de_fold_y_tablas():
    d = _datos_sinteticos()
    oof = L.evaluar(d, niveles=["resultado", "lp_previo"], modelos=["regla", "lineal"], k=3)["oof"]
    assert {"pred_regla", "pred_lineal_resultado", "pred_lineal_lp_previo"} <= set(oof.columns)
    assert oof["pred_regla"].notna().all()
    assert oof["zhat_regla"].notna().all()
    assert oof["zhat_lineal_lp_previo"].isna().all()      # el techo no se recorre
    tp = L.tabla_partida(oof)
    # El sesgo por jugador solo lo ve el nivel con el LP previo.
    assert tp.loc["lineal_lp_previo", "mae"] < tp.loc["regla", "mae"]
    tr = L.tabla_recorrido(oof)
    assert list(tr.columns) == ["1", "2-3", "4-7", "8-15", "16+", "todas"]


def test_folds_no_mezclan_jugadores():
    from src.features.elo_reverso import folds_por_jugador
    d = _datos_sinteticos(n_jug=30)
    f = pd.Series(folds_por_jugador(d["puuid"], 5), index=d.index)
    assert (f.groupby(d["puuid"]).nunique() == 1).all()


def test_gb_entrena_y_predice():
    pytest.importorskip("lightgbm")
    d = _datos_sinteticos(n_jug=40)
    m = L.GB(L.NIVELES["partida"], n_estimators=20).fit(d, d["delta_lp"])
    assert m.predict(d).shape == (len(d),)


def test_cargar_enlazadas_desde_la_base(tmp_path):
    ruta = tmp_path / "lp.db"
    conn = sqlite3.connect(ruta)
    try:
        conn.executescript(ESQUEMA)
        def insertar(tabla, v):
            conn.execute(f"INSERT INTO {tabla} ({','.join(v)}) VALUES ({','.join('?' * len(v))})", list(v.values()))

        for plat, m in (("la2", "LA2_1"), ("br1", "BR1_1")):
            insertar("partidas", {"match_id": m, "platform": plat, "game_end": 1, "duracion_s": 1800})
            for p, (tier, lp, lp2, win) in {"x": ("MASTER", 50, 70, 1), "y": ("GRANDMASTER", 300, 280, 0)}.items():
                ev = {"match_id": m, "platform": plat, "puuid": p + plat, "t_antes": 0, "t_despues": 1,
                      "tier_antes": tier, "rank_antes": "I", "lp_antes": lp, "wins_antes": 1, "losses_antes": 1,
                      "tier_despues": tier, "rank_despues": "I", "lp_despues": lp2, "tipo": "partida", "estado": "enlazada"}
                insertar("eventos", ev)
                insertar("participantes", {"match_id": m, "puuid": p + plat, "team_id": 100, "win": win})
        conn.commit()
    finally:
        conn.close()
    d = cargar_enlazadas(ruta, ["la2"])
    assert len(d) == 2 and set(d["platform"]) == {"la2"}
    x = d.set_index("puuid").loc["xla2"]
    assert x["z_antes"] == 450 and x["delta_lp"] == 20 and x["lobby_lp"] == 700
