"""Historial de LP propio: registro de cambios y elección de alto volumen (§9.S.21)."""

import pandas as pd

from scripts.collection import seguimiento_lp as sl


def fila(puuid, lp=100, w=10, l=5, tier="MASTER"):
    return {"puuid": puuid, "tier": tier, "rank": "I", "league_points": lp, "wins": w, "losses": l}


def test_solo_escribe_cambios_y_retoma_sin_duplicar(tmp_path):
    f = tmp_path / "apex.csv"
    r = sl.Registro(f)
    assert r.anotar("la2", [fila("a"), fila("b")], "2026-10-30 00:00:00") == 2
    assert r.anotar("la2", [fila("a"), fila("b", lp=118, w=11)], "2026-10-30 01:00:00") == 1
    # Al reabrir lee el último estado del CSV: la misma foto no agrega nada.
    r2 = sl.Registro(f)
    assert r2.anotar("la2", [fila("a"), fila("b", lp=118, w=11)], "2026-10-30 02:00:00") == 0
    # La misma cuenta en otra región es otra clave.
    assert r2.anotar("euw1", [fila("a")], "2026-10-30 02:00:00") == 1
    d = pd.read_csv(f)
    assert list(d["puuid"]) == ["a", "b", "b", "a"] and list(d["league_points"]) == [100, 100, 118, 100]


def test_salida_del_ladder_se_anota_una_vez(tmp_path):
    r = sl.Registro(tmp_path / "apex.csv")
    r.anotar("la2", [fila("a"), fila("b")], "t0")
    assert r.anotar_salidas("la2", {"a"}, "t1") == 1
    assert r.anotar_salidas("la2", {"a"}, "t2") == 0          # ya figura FUERA
    assert r.anotar("la2", [fila("b", lp=0)], "t3") == 1      # vuelve a entrar
    d = pd.read_csv(tmp_path / "apex.csv")
    assert list(d["tier"]) == ["MASTER", "MASTER", sl.FUERA, "MASTER"]


def cambios(filas):
    return pd.DataFrame([{"region": r, "puuid": p, "visto_at": t, "tier": "MASTER", "rank": "I",
                          "league_points": 0, "wins": w, "losses": 0} for r, p, t, w in filas])


def test_partidas_por_dia_arrastra_el_estado_previo():
    c = cambios([("la2", "a", "2026-10-29 00:00", 100),   # antes de la ventana
                 ("la2", "a", "2026-10-30 12:00", 110),
                 ("la2", "b", "2026-10-30 06:00", 50),    # entra a Master en la ventana
                 ("la2", "b", "2026-10-30 18:00", 53)])
    v = sl.partidas_por_dia(c, pd.Timestamp("2026-10-30"), pd.Timestamp("2026-10-31"))
    assert v[("la2", "a")] == 10 and v[("la2", "b")] == 3


def test_eleccion_por_temporada_y_despues_por_ritmo():
    # "a" tiene más partidas en la temporada; "b" es la que más juega ahora.
    c = cambios([("la2", "a", "2026-10-28 00:00", 900), ("la2", "b", "2026-10-28 00:00", 300),
                 ("la2", "b", "2026-10-29 06:00", 330), ("la2", "a", "2026-10-29 06:00", 901)])
    inicio = {"la2": pd.Timestamp("2026-10-28 00:00")}
    pronto = sl.elegir_alto_volumen(c[c["visto_at"] < "2026-10-28 01:00"], inicio, pd.Timestamp("2026-10-28 01:00"), 1)
    assert list(pronto["puuid"]) == ["a"] and pronto["criterio"].iat[0] == "partidas_temporada"
    luego = sl.elegir_alto_volumen(c, inicio, pd.Timestamp("2026-10-29 06:00"), 1)
    assert list(luego["puuid"]) == ["b"] and luego["criterio"].iat[0] == "partidas_dia"
