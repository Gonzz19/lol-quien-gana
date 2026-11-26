"""No-fuga del nivel histórico por emparejamiento (issue #4, PROJECT_SUMMARY.md §9.S.75).

`nivel_historico` usa el LP de la foto 1 del ranking (tomada en t1). En las partidas anteriores a t1
ese LP es posterior a la partida: las columnas tienen que valer el prior (0) y no depender de la foto.
En las partidas posteriores sí la usan, incluido el LP de los rivales de partidas previas a t1.
"""
import numpy as np
import pandas as pd
import pytest

from src.ganador import _constructores as K

T1 = 1_000


def _datos(n_pre=4, n_post=4, n_jug=14, semilla=0):
    rng = np.random.default_rng(semilla)
    NM = n_pre + n_post
    t = np.r_[np.arange(n_pre) * 10, T1 + np.arange(n_post) * 10]
    jugadores = np.array([f"p{i}" for i in range(n_jug)])
    player = np.stack([rng.choice(jugadores, 10, replace=False) for _ in range(NM)]).reshape(NM, 2, 5)
    D = dict(NM=NM, t=t, player=player, val0=np.zeros(NM, bool), Z={"mids": np.array([f"m{i}" for i in range(NM)])})
    return D, jugadores


def _nivel(monkeypatch, D, jugadores, lp):
    foto1 = pd.DataFrame({"z": lp}, index=pd.Index(jugadores, name="player_id"))
    monkeypatch.setattr(K, "cargar_foto1", lambda: (foto1, T1))
    return K.nivel_historico(D)["A"]


def test_antes_de_la_foto_no_depende_del_lp(monkeypatch):
    D, jug = _datos()
    pre = D["t"] < T1
    A1 = _nivel(monkeypatch, D, jug, np.linspace(0, 1300, len(jug)))
    A2 = _nivel(monkeypatch, D, jug, np.linspace(1300, 0, len(jug)) ** 1.1)
    assert np.array_equal(A1[pre], A2[pre])
    assert np.all(A1[pre] == 0.0)


def test_despues_de_la_foto_usa_el_lp(monkeypatch):
    D, jug = _datos()
    post = D["t"] >= T1
    A1 = _nivel(monkeypatch, D, jug, np.linspace(0, 1300, len(jug)))
    A2 = _nivel(monkeypatch, D, jug, np.linspace(1300, 0, len(jug)) ** 1.1)
    assert not np.allclose(A1[post][..., :2], A2[post][..., :2])
    assert np.all(A1[post][..., 2] > 0)          # cuenta las partidas previas, también las anteriores a t1


@pytest.mark.parametrize("semilla", [1, 2])
def test_filas_posteriores_iguales_a_la_cuenta_sin_mascara(monkeypatch, semilla):
    """La máscara solo toca t < t1: para t >= t1 el valor es el de la cuenta original."""
    D, jug = _datos(semilla=semilla)
    lp = np.random.default_rng(semilla).uniform(0, 1500, len(jug))
    A = _nivel(monkeypatch, D, jug, lp)
    # misma cuenta con la foto "tomada al principio" (t1 = -inf): sin máscara
    foto1 = pd.DataFrame({"z": lp}, index=pd.Index(jug, name="player_id"))
    monkeypatch.setattr(K, "cargar_foto1", lambda: (foto1, -1))
    A_sin = K.nivel_historico(D)["A"]
    post = D["t"] >= T1
    np.testing.assert_allclose(A[post], A_sin[post])
