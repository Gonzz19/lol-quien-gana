"""
Oro equivalente de kills, CS, primera sangre y primera torre, por rol y sin el soporte (pedido del usuario,
2026-11-17; amplía la §3.1 de eda_modelos_ganador).

Mismo método que la §3.1: curva de victoria según el Δ oro del equipo a los 15 (logística espejada, sin
intercepto, sin ventaja de lado), con dos lecturas para cada evento:

* *equivale a:* P(gana quien tiene la ventaja) pasada a oro con esa curva;
* *a igual oro:* lo que suma el evento además del Δ oro del equipo a los 15, en una logística con los dos
  (cociente de coeficientes).

Datos:
* Kills y CS de una línea a los 15 (`ingame_cache.npz`, `tl_raw`; partidas que duraron más de 15 min).
  Ventaja de kills de la línea: 1, 2, 3+; de CS: 10 (5-14), 20 (15-24), 30 (25-34), 40+. Codificación
  antisimétrica: +1 si la ventaja es del azul, −1 si es del rojo. Grupos: ADC (BOTTOM), resto sin soporte
  (TOP, JUNGLE y MIDDLE juntos: una fila por partida y línea) y cada línea por separado.
* Primera sangre, primera torre, dragones, vacuolarvas y heraldo con los 16.000 timelines del piloto
  (`data/timelines_eventos/timelines.db`): rol del que mata y del que muere (participantes 1/6 TOP, 2/7 JUNGLE,
  3/8 MIDDLE, 4/9 BOTTOM, 5/10 UTILITY; coincide con la línea en > 99,5 %), y línea de la primera torre. Se
  excluyen las primeras sangres con el soporte como asesino o como víctima. El Δ oro a los 15 sale del frame
  15 del propio timeline.

    python scripts/analysis/oro_equivalente_roles.py      # ~3 min
"""
import json
import sqlite3
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.models.ingame import LANES, TIMELINE_STATS  # noqa: E402

SALIDA = REPO / "data" / "analysis" / "oro_equivalente_roles.csv"
ROL = {1: "TOP", 2: "JUNGLE", 3: "MIDDLE", 4: "BOTTOM", 5: "UTILITY"}
espejo = lambda X: np.r_[X, -X]   # noqa: E731
CORTES_K, ETQ_K = [(1, 2), (2, 3), (3, 99)], ["+1 kill", "+2 kills", "+3 o más kills"]
CORTES_C, ETQ_C = [(5, 15), (15, 25), (25, 35), (35, 999)], ["+10 CS", "+20 CS", "+30 CS", "+40 o más CS"]
CS_PAREJO = 5
ESCALA_ORO = 1000   # el oro de la caché en vivo está en miles


def curva_oro(g, y):
    m = LogisticRegression(C=1e6, fit_intercept=False).fit(espejo(g[:, None]), np.r_[y, 1 - y])
    return m.coef_[0, 0]


def oro_equivalente(g, y, X, nombres, b_oro):
    """g: Δ oro del equipo (miles); X: columnas antisimétricas (+1 azul, −1 rojo, 0 nada)."""
    ambos = LogisticRegression(C=1e6, fit_intercept=False).fit(espejo(np.c_[g, X]), np.r_[y, 1 - y])
    filas = []
    for k, nom in enumerate(nombres):
        e = X[:, k]
        m = e != 0
        p = np.mean(np.where(e[m] > 0, y[m], 1 - y[m]))
        filas.append({"evento": nom, "casos": int(m.sum()), "gana quien lo tiene": p,
                      "equivale a": 1000 * np.log(p / (1 - p)) / b_oro,
                      "a igual oro": 1000 * ambos.coef_[0, k + 1] / ambos.coef_[0, 0]})
    return filas


def dummies(d, cortes, etiquetas):
    """d: diferencia azul − rojo; columnas antisimétricas por tramo de |d|."""
    a, s = np.abs(d), np.sign(d)
    return np.column_stack([np.where((a >= lo) & (a < hi), s, 0.0) for lo, hi in cortes]), etiquetas


def kills_y_cs():
    C = np.load(REPO / "data" / "analysis" / "ingame" / "ingame_cache.npz", allow_pickle=True)
    y, dur, tl = C["y"].astype(int), C["dur_sec"], C["tl_raw"]
    gi, ki, ci = (TIMELINE_STATS.index(s) for s in ("gold", "kills", "cs"))
    d = tl[:, 0, :, 1, :] - tl[:, 1, :, 1, :]                      # (NM, 5, stats) azul − rojo al 15
    ok = (dur > 900) & np.isfinite(d).all(axis=(1, 2))
    d, y = d[ok], y[ok]
    g = d[:, :, gi].sum(1)                                         # Δ oro del equipo, miles
    b_oro = curva_oro(g, y)
    print(f"partidas: {ok.sum():,}; 1.000 de oro a los 15: 50 % → {1 / (1 + np.exp(-b_oro)):.1%}")
    grupos = {"ADC": ["BOTTOM"], "Top + jungla + mid": ["TOP", "JUNGLE", "MIDDLE"],
              "TOP": ["TOP"], "JUNGLE": ["JUNGLE"], "MIDDLE": ["MIDDLE"], "BOTTOM (ADC)": ["BOTTOM"]}
    filas = []
    for grupo, lineas in grupos.items():
        idx = [LANES.index(l) for l in lineas]
        G = np.concatenate([g for _ in idx])
        Y = np.concatenate([y for _ in idx])
        dk = np.concatenate([d[:, i, ki] for i in idx])
        dc = np.concatenate([d[:, i, ci] for i in idx])
        # "parejo": el otro stat casi igual. CS: |Δ| ≤ CS_PAREJO (la Δ CS por línea tiene desvío 23-30);
        # kills: Δ = 0 exacto.
        for stat, dd, cortes, etq, filtro in (
                ("kills", dk, CORTES_K, ETQ_K, None),
                ("cs", dc, CORTES_C, ETQ_C, None),
                ("kills_cs_parejo", dk, CORTES_K, ETQ_K, np.abs(dc) <= CS_PAREJO),
                ("cs_kills_parejas", dc, CORTES_C, ETQ_C, dk == 0)):
            X, nombres = dummies(dd, cortes, etq)
            if filtro is not None:
                X = X * filtro[:, None]
            for f in oro_equivalente(G, Y, X, nombres, b_oro):
                filas.append({"grupo": grupo, "variable": stat, **f})
    # Escala de referencia: el oro que de verdad tiene de más a los 15 el jugador con la ventaja (promedio de
    # su Δ oro contra el rival de línea), con los 4 roles sin soporte juntos. No es un WR: solo oro.
    idx = [LANES.index(l) for l in ("TOP", "JUNGLE", "MIDDLE", "BOTTOM")]
    dk, dc, dg = (np.concatenate([d[:, i, s] for i in idx]) for s in (ki, ci, gi))
    for stat, dd, cortes, etq, filtro in (("kills_cs_parejo", dk, CORTES_K, ETQ_K, np.abs(dc) <= CS_PAREJO),
                                          ("cs_kills_parejas", dc, CORTES_C, ETQ_C, dk == 0)):
        X, nombres = dummies(dd, cortes, etq)
        X = X * filtro[:, None]
        for k, nom in enumerate(nombres):
            m = X[:, k] != 0
            filas.append({"grupo": "Oro real", "variable": stat, "evento": nom, "casos": int(m.sum()),
                          "gana quien lo tiene": np.nan,
                          "equivale a": ESCALA_ORO * float(np.mean(dg[m] * X[m, k])), "a igual oro": np.nan})
    return filas


def fb_y_ft():
    con = sqlite3.connect(REPO / "data" / "timelines_eventos" / "timelines.db")
    try:
        crudos = con.execute("SELECT json_zlib FROM timeline_crudo").fetchall()
    finally:
        con.close()
    reg = []
    for (b,) in crudos:
        t = json.loads(zlib.decompress(b))
        fr = t["info"]["frames"]
        if len(fr) <= 15:
            continue
        ev = sorted((e for f in fr for e in f["events"]), key=lambda e: e["timestamp"])
        fin = next((e for e in ev if e["type"] == "GAME_END"), None)
        if fin is None or "winningTeam" not in fin:
            continue
        y = int(fin["winningTeam"] == 100)
        pf = fr[15]["participantFrames"]
        g = sum(pf[str(p)]["totalGold"] * (1 if p <= 5 else -1) for p in range(1, 11)) / 1000
        fb = next((e for e in ev if e["type"] == "CHAMPION_SPECIAL_KILL" and e.get("killType") == "KILL_FIRST_BLOOD"), None)
        ck = next((e for e in ev if e["type"] == "CHAMPION_KILL"), None)   # el kill de la primera sangre (víctima)
        ft = next((e for e in ev if e["type"] == "BUILDING_KILL" and e.get("buildingType") == "TOWER_BUILDING"), None)
        r = {"y": y, "g": g}
        # Objetivos neutrales: lado (+1 azul, −1 rojo) del que los mata
        mon = []
        for e in ev:
            if e["type"] != "ELITE_MONSTER_KILL":
                continue
            eq = e.get("killerTeamId")
            if eq not in (100, 200):
                k = e.get("killerId", 0)
                eq = (100 if k <= 5 else 200) if 1 <= k <= 10 else None
            if eq is not None:
                mon.append((e["monsterType"], 1.0 if eq == 100 else -1.0))
        dr = [s for m, s in mon if m == "DRAGON"]
        for n in (1, 2, 3):          # los n primeros dragones del mismo equipo
            r[f"dr{n}"] = dr[0] if len(dr) >= n and len(set(dr[:n])) == 1 else 0.0
        r["grubs"] = sum(s for m, s in mon if m == "HORDE")                # azul − rojo (3 en total)
        her = [s for m, s in mon if m == "RIFTHERALD"]
        r["heraldo"] = her[0] if her else 0.0
        if fb is not None and ck is not None and 1 <= fb.get("killerId", 0) <= 10:
            k, v = fb["killerId"], ck.get("victimId", 0)
            r["fb_lado"] = 1.0 if k <= 5 else -1.0
            r["fb_asesino"] = ROL[(k - 1) % 5 + 1]
            r["fb_victima"] = ROL[(v - 1) % 5 + 1] if 1 <= v <= 10 else None
        if ft is not None:
            r["ft_lado"] = 1.0 if ft.get("teamId") == 200 else -1.0            # teamId = dueño de la torre
            r["ft_linea"] = {"TOP_LANE": "TOP", "MID_LANE": "MIDDLE", "BOT_LANE": "BOTTOM"}.get(ft.get("laneType"))
        reg.append(r)
    D = pd.DataFrame(reg)
    print(f"timelines del piloto: {len(D):,}")
    y, g = D.y.to_numpy(), D.g.to_numpy()
    b_oro = curva_oro(g, y)
    filas = []
    sin_sop = D.fb_asesino.ne("UTILITY") & D.fb_victima.ne("UTILITY") & D.fb_lado.notna()
    casos = {"Primera sangre (con soporte)": D.fb_lado.fillna(0).to_numpy(),      # todas, incluido el soporte
             "Primera sangre (sin soporte)": np.where(sin_sop, D.fb_lado.fillna(0), 0.0)}
    for r in ("TOP", "JUNGLE", "MIDDLE", "BOTTOM"):
        casos[f"Primera sangre: la consigue {r}"] = np.where(sin_sop & D.fb_asesino.eq(r), D.fb_lado.fillna(0), 0.0)
    for r in ("TOP", "JUNGLE", "MIDDLE", "BOTTOM"):
        casos[f"Primera sangre: muere {r}"] = np.where(sin_sop & D.fb_victima.eq(r), D.fb_lado.fillna(0), 0.0)
    casos["Primera torre"] = D.ft_lado.fillna(0).to_numpy()
    for l in ("TOP", "MIDDLE", "BOTTOM"):
        casos[f"Primera torre en {l}"] = np.where(D.ft_linea.eq(l), D.ft_lado.fillna(0), 0.0)
    casos["Primer dragón"] = D.dr1.to_numpy()
    casos["Dos primeros dragones"] = D.dr2.to_numpy()
    casos["Tres primeros dragones"] = D.dr3.to_numpy()
    gr = D.grubs.to_numpy()
    casos["Vacuolarvas 2-1"] = np.where(np.abs(gr) == 1, np.sign(gr), 0.0)
    casos["Vacuolarvas 3-0"] = np.where(np.abs(gr) == 3, np.sign(gr), 0.0)
    casos["Heraldo"] = D.heraldo.to_numpy()
    for nom, e in casos.items():
        for f in oro_equivalente(g, y, np.asarray(e, float)[:, None], [nom], b_oro):
            filas.append({"grupo": "piloto", "variable": "fb/ft", **f})
    return filas


def main():
    T = pd.DataFrame(kills_y_cs() + fb_y_ft())
    T.to_csv(SALIDA, index=False)
    pd.set_option("display.width", 200)
    print(T.to_string(index=False, float_format=lambda x: f"{x:,.3f}" if abs(x) < 1 else f"{x:,.0f}"))


if __name__ == "__main__":
    main()
