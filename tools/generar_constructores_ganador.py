"""Genera src/ganador/_constructores.py copiando el texto exacto de las funciones de los scripts originales.

Los `brainstorm_*` del escalón 9 están en `historial/escalon9/` desde §9.S.79; `embeddings_partidas.py` sigue en
`scripts/analysis/`.

Cada cambio es un reemplazo de texto con assert (si el original cambió, falla en vez de copiar mal).
"""
import ast
import builtins
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SA = REPO / "scripts" / "analysis"
ESC9 = REPO / "historial" / "escalon9"


def _ruta(archivo):
    return (ESC9 if archivo.startswith("brainstorm_") else SA) / archivo


def extraer(archivo, nombre, hasta=None):
    src = _ruta(archivo).read_text(encoding="utf-8")
    t = ast.parse(src)
    fn = next(n for n in t.body if isinstance(n, (ast.FunctionDef,)) and n.name == nombre)
    lineas = src.splitlines()[fn.lineno - 1: fn.end_lineno]
    texto = "\n".join(lineas)
    if hasta:
        i = texto.index(hasta)
        texto = texto[:i].rstrip()
    return texto


def rep(texto, viejo, nuevo, n=1):
    assert texto.count(viejo) == n, (viejo[:70], texto.count(viejo))
    return texto.replace(viejo, nuevo)


partes = []

# ---------------------------------------------------------------- variables por jugador (embeddings_partidas)
f = extraer("embeddings_partidas.py", "construir")
f = rep(f, "def construir():", "def variables_partida() -> dict:")
f = rep(f, '    with sqlite3.connect(f"file:{REPO}/data/lol_las.db?mode=ro", uri=True) as con:',
        '    with sqlite3.connect(f"file:{rutas.DB}?mode=ro", uri=True) as con:')
f = rep(f, 'pd.read_csv(REPO / "data" / "splits" / "train" / "full_dataset.csv", usecols=USE)',
        'rutas.leer_partidas(usecols=USE)')
# inicio de la evaluación según el modo (16/09 en V; corte del split en el test, §9.S.80)
f = rep(f, "    mitad = np.median(t[t >= t_foto])\n    val = t >= mitad\n",
        "    val = t >= rutas.inicio_evaluacion(t, t_foto)   # 16/09 en V y el simulacro; corte del split en el test\n")
# al final escribía la caché: ahora devuelve el diccionario (y suma los jugadores por slot)
i = f.index("    np.savez_compressed(CACHE, Xj=Xj")
f = f[:i] + (
    "    return dict(Xj=Xj, cols=np.array(cols), mids=mids, t=t, val=val, y=y, tnum=tnum, parche=parche,\n"
    "                champ=df.champion_id.to_numpy().reshape(-1, 2, 5),\n"
    "                player=df.player_id.to_numpy().astype(str).reshape(-1, 2, 5),\n"
    "                **{f\"D_{k}\": np.asarray(v) for k, v in D.items()})\n")
f = rep(f, 'json.load(open(REPO / "data" / "ddragon" / "16.17.1" / "champion.json", encoding="utf-8"))',
        'json.load(open(rutas.DDRAGON_JSON, encoding="utf-8"))')
partes.append(f)

# ---------------------------------------------------------------- margen por campeón y por línea (brainstorm_05)
f = extraer("brainstorm_05_jugador.py", "construir_cache")
f = rep(f, "def construir_cache(D: dict, rehacer: bool = False) -> None:\n    if CACHE.exists() and not rehacer:\n        return\n",
        "def margen_campeon_linea(D: dict) -> dict:\n")
f = rep(f, 'pd.read_csv(REPO / "data" / "splits" / "train" / "full_dataset.csv", usecols=USE)', "rutas.leer_partidas(usecols=USE)")
f = rep(f, "    CACHE.parent.mkdir(parents=True, exist_ok=True)\n    np.savez_compressed(CACHE, MC=", "    return dict(MC=")
partes.append(f)

# ---------------------------------------------------------------- nivel histórico por emparejamiento (brainstorm_07)
f = extraer("brainstorm_07_mmr_historico.py", "construir_cache")
f = rep(f, "def construir_cache(D: dict, rehacer: bool = False) -> None:\n    if CACHE.exists() and not rehacer:\n        return\n",
        "def nivel_historico(D: dict) -> dict:\n")
f = rep(f, "foto1, t1 = gm._cargar_foto1(gm.RUTA_DB)", "foto1, t1 = cargar_foto1()")
f = rep(f, 'pl = D["B"]["player"].reshape(-1)', 'pl = D["player"].reshape(-1)')
f = rep(f, "    CACHE.parent.mkdir(parents=True, exist_ok=True)\n    np.savez_compressed(CACHE, A=", "    return dict(A=")
partes.append(f)

# ---------------------------------------------------------------- W/L de la foto 1, dinámico (g_01; solo FOTO1_DIN)
f = extraer("brainstorm_g_01_foto1_wl.py", "construir_extras", hasta="    # 3. lp_blend")
f = rep(f, "def construir_extras(D: dict) -> dict[str, np.ndarray]:", "def wl_foto1(D: dict) -> dict:")
f = rep(f, '    B, Z = D["B"], D["Z"]\n    pl = B["player"]', '    pl = D["player"]')
f = rep(f, "with sqlite3.connect(f\"file:{REPO / 'data' / 'lol_las.db'}?mode=ro&immutable=1\", uri=True) as con:",
        'with sqlite3.connect(f"file:{rutas.DB}?mode=ro&immutable=1", uri=True) as con:')
f += "\n    return dict(FOTO1_DIN=FOTO1_DIN)\n"
partes.append(f)

# ---------------------------------------------------------------- contracruces y sinergias (g_07; solo MATCHUPS)
f = extraer("brainstorm_g_07_matchups_parche_rematch.py", "construir_mpr")
f = rep(f, "def construir_mpr(D: dict) -> dict[str, np.ndarray]:", "def contracruces(D: dict) -> dict:")
f = rep(f, '_, pl_codes = np.unique(D["B"]["player"].reshape(-1), return_inverse=True)',
        '_, pl_codes = np.unique(D["player"].reshape(-1), return_inverse=True)')
f = rep(f, '    return {"MATCHUPS": MATCHUPS, "PARCHE": PARCHE, "REMATCH": REMATCH}', '    return {"MATCHUPS": MATCHUPS}')
partes.append(f)

# ---------------------------------------------------------------- winrate por línea y autofill (g_10; solo ROL_WR)
for h in ("_cargar_df_ordenado", "_ewma_causal_por_jugador", "_streak_causal_por_jugador"):
    f = extraer("brainstorm_g_10_nuevas_senales.py", h)
    if h == "_cargar_df_ordenado":
        f = rep(f, 'pd.read_csv(REPO / "data" / "splits" / "train" / "full_dataset.csv", usecols=cols)', "rutas.leer_partidas(usecols=cols)")
    partes.append(f)
f = extraer("brainstorm_g_10_nuevas_senales.py", "construir_cache")
f = rep(f, "def construir_cache(D: dict) -> dict:", "def winrate_linea(D: dict) -> dict:")
f = rep(f, "with sqlite3.connect(f\"file:{REPO / 'data' / 'lol_las.db'}?mode=ro&immutable=1\", uri=True) as con:",
        'with sqlite3.connect(f"file:{rutas.DB}?mode=ro&immutable=1", uri=True) as con:')
i = f.index("    CACHE.parent.mkdir(parents=True, exist_ok=True)")
f = f[:i] + "    return dict(ROL_WR=ROL_WR)\n"
partes.append(f)

# ---------------------------------------------------------------- Kalman-Elo, tempo, objetivos, economía (g_11)
f = extraer("brainstorm_g_11_elo_clutch_tempo.py", "construir_cache_g11")
f = rep(f, "def construir_cache_g11(D: dict, rehacer: bool = False) -> dict[str, np.ndarray]:\n"
           "    if CACHE.exists() and not rehacer:\n        z = np.load(CACHE)\n        if int(z[\"nm\"][0]) == D[\"NM\"]:\n"
           "            return {k: z[k] for k in z.files if k != \"nm\"}\n",
        "def kalman_elo(D: dict) -> dict:\n")
f = rep(f, '    print("Construyendo caché gemini_g11.npz...", flush=True)\n', "")
f = rep(f, 'players = D["B"]["player"].reshape(NM, 10).astype(str)', 'players = D["player"].reshape(NM, 10).astype(str)')
f = rep(f, 'pd.read_csv(REPO / "data" / "splits" / "train" / "full_dataset.csv", usecols=cols_csv)', "rutas.leer_partidas(usecols=cols_csv)")
f = rep(f, '        "CLUTCH_15": CLUTCH_15,\n', "")    # sub-bloque descartado (no entra al modelo final)
f = rep(f, "    np.savez_compressed(CACHE, nm=np.array([NM]), **out)\n    print(f\"Caché gemini_g11.npz lista en {time.time() - t0:.1f}s\", flush=True)\n", "")
partes.append(f)

# ---------------------------------------------------------------- composición del draft (g_06) + modelo por línea (g_09)
f = extraer("brainstorm_g_06_comp_poisson.py", "construir_composicion")
f = rep(f, "def construir_composicion(D: dict) -> tuple[np.ndarray, np.ndarray]:", "def composicion_campeones(D: dict) -> tuple[np.ndarray, np.ndarray]:")
f = rep(f, "pd.read_csv(ATRIB_CSV)", "pd.read_csv(rutas.ATRIB_CSV)")
partes.append(f)
f = extraer("brainstorm_g_09_hgb_lineas_causal.py", "cargar_target_oro15")
f = rep(f, 'pd.read_csv(\n        REPO / "data" / "splits" / "train" / "full_dataset.csv",\n', "rutas.leer_partidas(\n")
partes.append(f)
f = extraer("brainstorm_g_09_hgb_lineas_causal.py", "construir_hgb_causal")
f = rep(f, "def construir_hgb_causal(D: dict) -> np.ndarray:", "def oro_linea_hgb(D: dict, MC, ML, A, MATCHUPS, semilla: int = 42) -> dict:")
f = rep(f, "                random_state=42,\n", "                random_state=semilla,\n")   # semilla del HGB (§9.S.80)
f = rep(f, '    C = np.load(REPO / "data" / "analysis" / "brainstorm" / "margen_grupos.npz")\n'
           '    A_mmr = np.load(REPO / "data" / "analysis" / "brainstorm" / "mmr_historico.npz")["A"].astype(np.float32)\n'
           "    _, J_CAMP = construir_composicion(D)\n"
           '    mpr_path = REPO / "data" / "analysis" / "brainstorm" / "gemini_mpr.npz"\n'
           '    MT = np.load(mpr_path)["MATCHUPS"][:, :5] if mpr_path.exists() else np.zeros((NM, 5), np.float32)\n',
        "    C = {\"MC\": MC, \"ML\": ML}\n"
        "    A_mmr = A.astype(np.float32)\n"
        "    _, J_CAMP = composicion_campeones(D)\n"
        "    MT = MATCHUPS[:, :5]\n")
f = rep(f, "    return np.c_[PRED, PRED.sum(axis=1)].astype(np.float32)", "    return dict(HGB=np.c_[PRED, PRED.sum(axis=1)].astype(np.float32))")
partes.append(f)

# ---------------------------------------------------------------- margen esperado por par de campeones (g_23)
f = extraer("brainstorm_g_23_duelo_res_y_per_jug.py", "construir_cache_g23")
f = rep(f, "def construir_cache_g23(D: dict, rehacer: bool = False) -> dict[str, np.ndarray]:\n"
           "    if CACHE.exists() and not rehacer:\n        z = np.load(CACHE)\n        if int(z[\"nm\"][0]) == D[\"NM\"]:\n"
           "            return {k: z[k] for k in z.files if k != \"nm\"}\n",
        "def margen_esperado_par(D: dict) -> dict:\n")
f = rep(f, '    print("Construyendo caché gemini_g23.npz (RES_DUELO_15 + MATCHUP_EXP_MARGIN)...", flush=True)\n', "")
f = rep(f, 'pd.read_csv(\n        REPO / "data" / "splits" / "train" / "full_dataset.csv",\n', "rutas.leer_partidas(\n")
i = f.index("    CACHE.parent.mkdir(parents=True, exist_ok=True)")
f = f[:i] + '    return {"MATCHUP_EXP_MARGIN": MATCHUP_EXP_MARGIN}\n'
partes.append(f)

ENCABEZADO = '''"""Constructores de las variables previas del modelo final (escalón 9).

**Código copiado tal cual** de los scripts donde se probó cada variable (hoy en `historial/` o en
`scripts/analysis/`), con solo tres cambios: no leen ni escriben cachés (de eso se ocupa
`src/ganador/cache.py`), reciben como argumento lo que antes leían de disco, y devuelven solo las
columnas que usa el modelo final. La equivalencia con el código original está verificada
(`PROJECT_SUMMARY.md` §9.S.73). Por eso conservan los nombres internos de entonces (MATCHUPS, ROL_WR...).

No usar directamente: la entrada es `src.ganador.variables.calcular_todas`.
"""
# Generado por la reorganización del 2026-11-22. Origen de cada función:
#   variables_partida      <- scripts/analysis/embeddings_partidas.py::construir
#   margen_campeon_linea   <- brainstorm_05_jugador.py::construir_cache
#   nivel_historico        <- brainstorm_07_mmr_historico.py::construir_cache
#   wl_foto1               <- brainstorm_g_01_foto1_wl.py::construir_extras (solo FOTO1_DIN)
#   contracruces           <- brainstorm_g_07_matchups_parche_rematch.py::construir_mpr (solo MATCHUPS)
#   winrate_linea          <- brainstorm_g_10_nuevas_senales.py::construir_cache (solo ROL_WR)
#   kalman_elo             <- brainstorm_g_11_elo_clutch_tempo.py::construir_cache_g11 (sin CLUTCH_15)
#   composicion_campeones  <- brainstorm_g_06_comp_poisson.py::construir_composicion
#   oro_linea_hgb          <- brainstorm_g_09_hgb_lineas_causal.py::construir_hgb_causal
#   margen_esperado_par    <- brainstorm_g_23_duelo_res_y_per_jug.py::construir_cache_g23 (solo MATCHUP_EXP_MARGIN)
# ruff: noqa
import json
import math
import sqlite3
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.features.change import play_rate, pool_change
from src.features.elo_reverso import cargar_foto
from src.features.fingerprint import STYLE_FEATURES, add_rates
from src.features.lobby import prior_lobby_level
from src.features.margin import ORDEN, _previo, lane_diffs, lane_scale, prior_margin, prior_margin_by, prior_winrate
from src.features.padron import verificar_foto1
from src.features.rating import lp_absoluto
from src.ganador import rutas

T0 = time.time()
LANES = ["BOTTOM", "JUNGLE", "MIDDLE", "TOP", "UTILITY"]      # orden lexicográfico = orden del sort
'''


def constante(archivo, nombre):
    src = _ruta(archivo).read_text(encoding="utf-8")
    t = ast.parse(src)
    n = next(n for n in t.body if isinstance(n, ast.Assign) and any(getattr(x, "id", "") == nombre for x in n.targets))
    return "\n".join(src.splitlines()[n.lineno - 1: n.end_lineno])


consts = [constante("embeddings_partidas.py", c) for c in ("STATS", "TAGS", "CSTATS", "USE")]
assert constante("brainstorm_05_jugador.py", "STATS").split("=", 1)[1].split() == consts[0].split("=", 1)[1].split()

extra = '''

def log(*a):
    s = " ".join(str(x) for x in a)
    print(f"[{time.time() - T0:6.0f}s] {s}", flush=True)


def cargar_foto1():
    """Foto 1 del ranking (04/09) por jugador y su fecha, congelada (`verificar_foto1`)."""
    uri = f"file:{rutas.DB.resolve()}?mode=ro"
    foto1 = cargar_foto(
        uri,
        "SELECT m.player_id, p.tier, p.rank, p.league_points, p.wins, p.losses "
        "FROM players p JOIN player_id_map m ON m.puuid = p.puuid",
    )
    with sqlite3.connect(uri, uri=True) as con:
        t1 = verificar_foto1(con)
    return foto1, t1
'''
codigo = ENCABEZADO + "\n".join(consts) + extra + "\n\n" + "\n\n\n".join(partes) + "\n"
salida = REPO / "src" / "ganador" / "_constructores.py"
salida.parent.mkdir(exist_ok=True)
salida.write_text(codigo, encoding="utf-8", newline="")

# --- nombres sin definir (chequeo estático simple, por función)
t = ast.parse(codigo)
modulo = set(dir(builtins))
for n in t.body:
    if isinstance(n, (ast.Import, ast.ImportFrom)):
        modulo |= {(a.asname or a.name).split(".")[0] for a in n.names}
    elif isinstance(n, ast.Assign):
        modulo |= {x.id for x in n.targets if isinstance(x, ast.Name)}
    elif isinstance(n, ast.FunctionDef):
        modulo.add(n.name)
faltan = set()
for fn in [n for n in t.body if isinstance(n, ast.FunctionDef)]:
    locales = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
    for x in ast.walk(fn):
        if isinstance(x, ast.Name) and isinstance(x.ctx, (ast.Store, ast.Del)):
            locales.add(x.id)
        elif isinstance(x, (ast.FunctionDef, ast.Lambda)):
            locales |= {a.arg for a in x.args.args}
            if isinstance(x, ast.FunctionDef):
                locales.add(x.name)
        elif isinstance(x, (ast.comprehension,)):
            for y in ast.walk(x.target):
                if isinstance(y, ast.Name):
                    locales.add(y.id)
        elif isinstance(x, ast.ExceptHandler) and x.name:
            locales.add(x.name)
    for x in ast.walk(fn):
        if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load) and x.id not in locales and x.id not in modulo:
            faltan.add((fn.name, x.id))
print("líneas:", codigo.count("\n"), "| sin definir:", sorted(faltan) or "ninguno")
