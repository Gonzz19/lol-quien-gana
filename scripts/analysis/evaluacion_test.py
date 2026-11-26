"""Evaluación única en el test del modelo previo (escalón 9) y del modelo en vivo (etapa 1), según el protocolo.

**Se corre una sola vez** (issue #5, `docs/evaluacion_test/PROTOCOLO.md`, `PROJECT_SUMMARY.md` §9.S.80). Hace todo
en una corrida: variables en modo `test` (train ∪ test ∪ Diamond posterior al corte, todo lo ajustado con partidas
anteriores al 27/09), modelo previo y en vivo con las 5 semillas fijadas, métricas con IC por bootstrap por bloques
de días (B = 1000), ventanas cronológicas de 2.500 partidas con la pendiente del AUC, y escribe:

* `data/analysis/test_final/` — `metricas.csv`, `ventanas.csv`, `pendientes.csv`, `semillas.csv`,
  `ic_bootstrap_semillas.csv`, `resumen.json`, `chequeos.json` y `predicciones.npz`;
* `docs/evaluacion_test/LOG_EVALUACION_TEST.md` — fecha y hora, commit, hashes (match_split, CSV del split,
  protocolo), versiones de las librerías y los resultados.

Se niega a correr si el log ya existe (salvo `--excepcion "motivo"`, que queda escrito en el log como excepción al
protocolo) o si el árbol de git no está limpio, y antes de cargar el test verifica los hashes congelados en el
protocolo. Los chequeos que solo se pueden hacer con el test (partidas, regla de Master+, fechas, cobertura del
timeline) son `assert` dentro de la corrida: si uno falla, no se calcula ninguna métrica (ver el protocolo).

    python scripts/analysis/evaluacion_test.py --simulacro              # ensayo: V hace de test (no toca el test)
    python scripts/analysis/evaluacion_test.py --simulacro --semillas 42 # ensayo rápido, solo la semilla principal
    python scripts/analysis/evaluacion_test.py                          # LA evaluación (una sola vez, con OK del usuario)
    python scripts/analysis/evaluacion_test.py --sin-v                  # anexo: el modelo ajustado sin V, sobre el test

`--sin-v` (anexo pedido por el usuario el 2026-11-23, antes de ver resultados): el mismo test, pero con todo ajustado
solo con lo anterior al 16/09, como en el simulacro (modo `test_sin_v`). No reemplaza al número del protocolo; mide
cuánto aportan las ~11 días de V al ajuste. Escribe en `data/analysis/test_final_sin_v/` y
`docs/evaluacion_test/LOG_EVALUACION_TEST_SIN_V.md`; por defecto solo la semilla principal.

El simulacro escribe en `data/analysis/test_final_simulacro/` (con su propio log) y usa la caché
`data/cache/ganador_simulacro/`; el test, `data/cache/ganador_test/`. Ninguno toca `data/cache/ganador/`.
Tiempo estimado: ~45 min la semilla principal (variables ~22 + en vivo ~20) y ~30 min cada semilla extra.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.ganador import evaluacion as EV, rutas  # noqa: E402

PROTOCOLO = REPO / "docs" / "evaluacion_test" / "PROTOCOLO.md"
LOG = REPO / "docs" / "evaluacion_test" / "LOG_EVALUACION_TEST.md"
SALIDA = rutas.DATA / "analysis" / "test_final"
SALIDA_SIMULACRO = rutas.DATA / "analysis" / "test_final_simulacro"
SALIDA_SIN_V = rutas.DATA / "analysis" / "test_final_sin_v"
LOG_SIN_V = REPO / "docs" / "evaluacion_test" / "LOG_EVALUACION_TEST_SIN_V.md"
LOG_SIMULACRO = SALIDA_SIMULACRO / "LOG_SIMULACRO.md"

# --- congelado en el protocolo (§2-§5 de PROTOCOLO.md); cambiarlo exige cambiar el protocolo (y su hash)
SEMILLAS = (42, 0, 1, 2, 3)                    # 42 = la principal (la de siempre); las otras, fijadas el 23/11
SEMILLA_PRINCIPAL = 42
C_ETAPA1 = {10: 0.01, 15: 0.03, 20: 0.03}      # elegidos en A2-A4 por log loss en V (metricas.csv, §9.S.78)
B = 1000
SEMILLA_BOOT = 20260927
SEMILLAS_BOOT = (20260927, 1, 2, 3, 4)         # para medir cuánto mueve la semilla del bootstrap los IC
MIN_COBERTURA_TEST = 0.95                      # fracción de partidas del test con 10 filas válidas
MIN_VIVAS_10 = 0.90                            # fracción de partidas de evaluación con timeline al minuto 10
ARCHIVOS_HASH = {"match_split": rutas.MATCH_SPLIT, "train": rutas.TRAIN_CSV, "test": rutas.TEST_CSV,
                 "diamante_despues": rutas.DIAMANTE_DESPUES_CSV}
RANGOS = {1: "MASTER", 2: "GRANDMASTER", 3: "CHALLENGER"}


# --------------------------------------------------------------------------------------------- guardas

def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def hashes_del_protocolo(texto: str) -> dict:
    """Los hashes congelados: líneas `<!-- sha256 NOMBRE HEX -->` del protocolo."""
    return dict(re.findall(r"<!--\s*sha256\s+(\w+)\s+([0-9a-f]{64})\s*-->", texto))


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


CARPETAS_CODIGO = ("src", "scripts", "tools", "tests", "docs")


def estado_git() -> str:
    """Cambios en archivos versionados y archivos sin versionar en las carpetas de código (`data/` no cuenta:
    son los datos y las cachés, que no se versionan)."""
    cambios = git("status", "--porcelain", "--untracked-files=no")
    nuevos = git("status", "--porcelain", "--untracked-files=all", "--", *CARPETAS_CODIGO)
    return "\n".join(x for x in (cambios, "\n".join(l for l in nuevos.splitlines() if l.startswith("??"))) if x)


def problemas_previos(log: Path, excepcion: str | None, estado_git: str, protocolo: Path) -> list[str]:
    """Lo que impide correr la evaluación única (lista vacía = se puede)."""
    out = []
    if log.exists() and not excepcion:
        out.append(f"ya existe {log.relative_to(REPO) if log.is_relative_to(REPO) else log}: la evaluación se corre "
                   "una sola vez (para repetirla hace falta --excepcion \"motivo\" y el OK del usuario)")
    if estado_git:
        out.append("el árbol de git no está limpio (commitear o descartar antes):\n" + estado_git)
    if not protocolo.exists():
        out.append(f"falta el protocolo {protocolo}")
    return out


def verificar_hashes(esperados: dict, archivos: dict) -> dict:
    """Calcula el SHA-256 de cada archivo y lo compara con el del protocolo. Devuelve {nombre: (hash, ok)}."""
    out = {}
    for nombre, ruta in archivos.items():
        h = sha256(ruta)
        out[nombre] = (h, esperados.get(nombre) == h)
    return out


# --------------------------------------------------------------------------------------------- predicciones

def predecir(semilla: int, base: dict | None) -> tuple[dict, dict]:
    """Previo y en vivo con la semilla `semilla` (de los dos HGB). Devuelve (base, predicciones).

    `base` (partidas de evaluación, fechas, resultado, rango) se arma con la primera semilla y se verifica igual en
    las demás. Ningún paso usa el resultado de las partidas de evaluación para ajustar nada.
    """
    from src.ganador import modelo_en_vivo as MV, modelo_previo, variables
    from src.models.validacion_temporal import rango_lobby

    D, V = variables.calcular_todas(semilla=semilla)
    X = variables.matriz_extras(D, V)
    tr = modelo_previo.filas_entrenamiento_v(D)
    ev = np.flatnonzero(D["val"])
    dv = np.flatnonzero(D["diam_v"])
    desde = rutas.corte_test_ms() if rutas.MODO == "test_sin_v" else None   # sin V: V no se ajusta ni se mide
    if desde is not None:
        ev, dv = ev[D["t"][ev] >= desde], dv[D["t"][dv] >= desde]
    nueva = dict(mids=D["Z"]["mids"].astype(str), t=D["t"].astype(np.int64), y=D["y"].astype(np.int8),
                 ev=ev, dv=dv, tr=tr, rango=rango_lobby(D["Z"]["tnum"]), regla=D["regla_master"],
                 inicio_ajuste_excluido=int(D["t"][D["val0"]].min()))
    if base is None:
        base = nueva
    else:
        for k in ("mids", "ev", "dv", "tr"):
            assert np.array_equal(base[k], nueva[k]), f"las partidas cambian con la semilla ({k})"
    modelo = modelo_previo.ajustar(D, X, tr)
    pred = {"previo": modelo.predecir(D, X, ev), "previo_diamante": modelo.predecir(D, X, dv),
            "n_columnas": modelo.lr.coef_.shape[1]}
    MV.log(f"previo (semilla {semilla}): ajustado con {len(tr):,} partidas")
    del X, modelo
    I = MV.insumos(D, V, semilla=semilla)
    del D, V
    gc.collect()
    chunk = I["chunk"]
    for m in MV.MINUTOS:
        ok = MV.validas(I, m)
        v, d = np.flatnonzero((chunk == MV.TV) & ok), np.flatnonzero((chunk == MV.TDV) & ok)
        if desde is not None:
            v, d = v[nueva["t"][v] >= desde], d[nueva["t"][d] >= desde]
        R = MV.ajustar_predecir(I, m, np.r_[v, d], c_etapa1=(C_ETAPA1[m],))
        p = MV._sigm(R["z_fin"])
        pred[f"vivo_{m}"], pred[f"vivo_{m}_idx"] = p[:len(v)], v
        pred[f"vivo_{m}_diamante"], pred[f"vivo_{m}_diamante_idx"] = p[len(v):], d
        pred[f"L_{m}"] = MV._sigm(I["L"][v])                    # el previo hacia adelante en esas partidas
        pred[f"vivo_{m}_n_entrenamiento"] = R["n_entrenamiento"]
        MV.log(f"en vivo {m} (semilla {semilla}): {len(v):,} + {len(d):,} partidas, C = {R['C1']}")
        gc.collect()
    pred["chunk_ev"] = chunk[base["ev"]]
    del I
    gc.collect()
    return base, pred


# --------------------------------------------------------------------------------------------- parciales

def _ruta_parcial(salida: Path, semilla: int) -> Path:
    return salida / "parciales" / f"semilla_{semilla}.npz"


def guardar_parcial(salida: Path, semilla: int, firma: dict, base: dict, pred: dict) -> None:
    """Guarda las predicciones de una semilla apenas termina, para que un corte (memoria, Ctrl+C) no pierda lo hecho.

    Son predicciones, no métricas: ninguna métrica se calcula hasta tener todas las semillas. Se retoman solo si
    coinciden el commit, el modo y el hash del protocolo (`firma`), así que retomar es la misma evaluación única.
    """
    ruta = _ruta_parcial(salida, semilla)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, __firma__=json.dumps(firma),
                        **{f"base__{k}": np.asarray(v) for k, v in base.items()},
                        **{f"pred__{k}": np.asarray(v) for k, v in pred.items()})
    tmp.replace(ruta)


def cargar_parcial(salida: Path, semilla: int, firma: dict) -> tuple[dict, dict] | None:
    ruta = _ruta_parcial(salida, semilla)
    if not ruta.exists():
        return None
    with np.load(ruta, allow_pickle=False) as z:
        if json.loads(str(z["__firma__"])) != firma:
            print(f"[!] {ruta.name}: de otra corrida (commit, modo o protocolo distintos); se recalcula", flush=True)
            return None
        def _valor(a):
            return a.item() if a.ndim == 0 else a
        base = {k[len("base__"):]: _valor(z[k]) for k in z.files if k.startswith("base__")}
        pred = {k[len("pred__"):]: _valor(z[k]) for k in z.files if k.startswith("pred__")}
    return base, pred


# --------------------------------------------------------------------------------------------- chequeos

def chequeos(base: dict, pred: dict, simulacro: bool) -> dict:
    """Chequeos que solo se pueden hacer con las partidas de evaluación. Si uno falla, no se mide nada."""
    from src.ganador import modelo_en_vivo as MV
    t, ev, dv, tr, mids = base["t"], base["ev"], base["dv"], base["tr"], base["mids"]
    inicio = int(t[np.r_[ev, dv]].min())
    c = {}
    c["entrenamiento_anterior_a_la_evaluacion"] = bool(t[tr].max() < inicio)
    c["evaluacion_toda_master_regla_split"] = bool((base["rango"][ev] >= 1).all())
    c["previo_sin_nan"] = bool(np.isfinite(pred["previo"]).all())
    c["previo_alineado_con_en_vivo"] = bool((pred["chunk_ev"] == MV.TV).all())
    for m in MV.MINUTOS:
        c[f"en_vivo_{m}_sin_nan"] = bool(np.isfinite(pred[f"vivo_{m}"]).all())
    c["fraccion_vivas_10"] = float(len(pred["vivo_10_idx"]) / len(ev))
    c["vivas_10_suficientes"] = c["fraccion_vivas_10"] >= MIN_VIVAS_10
    if not simulacro:
        corte = rutas.corte_test_ms()
        ms = pd.read_csv(rutas.MATCH_SPLIT, usecols=["match_id", "split"])
        ids_test = set(ms.loc[ms["split"] == "test", "match_id"])
        ids_diam = set(ms.loc[ms["split"] == "diamante_despues", "match_id"])
        del ms
        en_test = np.isin(mids, list(ids_test))
        c["inicio_es_el_corte"] = bool(inicio >= corte and t[tr].max() < corte)
        c["evaluacion_son_las_del_test"] = bool(np.array_equal(np.flatnonzero(en_test), ev))
        c["diamante_son_las_diamante_despues"] = bool(np.isin(mids[dv], list(ids_diam)).all())
        c["cobertura_test"] = float(en_test.sum() / len(ids_test))
        c["cobertura_test_suficiente"] = c["cobertura_test"] >= MIN_COBERTURA_TEST
        if rutas.MODO == "test_sin_v":                 # el ajuste termina antes de V (16/09), no en el corte
            ini = int(base["inicio_ajuste_excluido"])
            c["ajuste_anterior_a_v"] = bool(t[tr].max() < ini and ini < corte - 7 * 86_400_000)
    return c


# --------------------------------------------------------------------------------------------- métricas

def _fila(modelo, minuto, conjunto, b, ic):
    f = {"modelo": modelo, "minuto": minuto, "conjunto": conjunto, "ic": ic, "n": b["n"], "n_bloques": b["n_grupos"],
         "B": b["B"]}
    for k in EV.NOMBRES:
        f[k], f[f"{k}_lo"], f[f"{k}_hi"] = b[k]
    return f


def medir(base: dict, preds: dict[int, dict]) -> dict:
    """Todas las tablas: métricas con IC, ventanas, pendientes, semillas e IC según la semilla del bootstrap."""
    from src.ganador import modelo_en_vivo as MV
    y, t, ev, dv = base["y"].astype(int), base["t"], base["ev"], base["dv"]
    P = preds[SEMILLA_PRINCIPAL]
    dia = EV.dia_art(t)
    filas, ventanas, pendientes = [], [], []

    def conj(modelo, minuto, nombre, idx, p, ventanas_tambien=False, partidas_tambien=False):
        filas.append(_fila(modelo, minuto, nombre, EV.bootstrap_bloques(y[idx], p, dia[idx], B, SEMILLA_BOOT), "dias"))
        if partidas_tambien:
            filas.append(_fila(modelo, minuto, nombre, EV.bootstrap_partidas(y[idx], p, B, SEMILLA_BOOT), "partidas"))
        if ventanas_tambien:
            w = EV.metricas_por_ventana(y[idx], p, t[idx], EV.TAM_VENTANA, B, SEMILLA_BOOT)
            ventanas.append(w.assign(modelo=modelo, minuto=minuto, conjunto=nombre))
            pendientes.append({"modelo": modelo, "minuto": minuto, "conjunto": nombre,
                               **EV.pendiente_auc(y[idx], p, t[idx], EV.TAM_VENTANA, B, SEMILLA_BOOT)})
        MV.log(f"métricas: {modelo} {minuto} {nombre}")

    # previo
    conj("previo", "", "evaluacion", ev, P["previo"], ventanas_tambien=True, partidas_tambien=True)
    for k, nombre in RANGOS.items():
        sel = base["rango"][ev] == k
        if sel.sum() >= 100:
            conj("previo", "", f"evaluacion {nombre}", ev[sel], P["previo"][sel])
    if len(dv) >= 100:
        conj("previo", "", "diamante (chequeo)", dv, P["previo_diamante"])
    # en vivo
    coh = P["vivo_20_idx"]
    for m in MV.MINUTOS:
        idx, p = P[f"vivo_{m}_idx"], P[f"vivo_{m}"]
        conj("en_vivo", m, "evaluacion", idx, p, ventanas_tambien=True, partidas_tambien=True)
        en = np.isin(idx, coh)
        conj("en_vivo", m, "cohorte 20", idx[en], p[en])
        conj("previo_hacia_adelante", m, "cohorte 20", idx[en], P[f"L_{m}"][en])
        d = P[f"vivo_{m}_diamante_idx"]
        if len(d) >= 100:
            conj("en_vivo", m, "diamante (chequeo)", d, P[f"vivo_{m}_diamante"])

    # semillas (métricas puntuales en el conjunto de evaluación) y correlación de las predicciones
    sem_filas, sem_resumen = [], {}
    claves = [("previo", "", "previo", ev)] + [("en_vivo", m, f"vivo_{m}", P[f"vivo_{m}_idx"]) for m in MV.MINUTOS]
    for modelo, minuto, clave, idx in claves:
        por = {}
        for s, Ps in preds.items():
            assert np.array_equal(Ps.get(f"{clave}_idx", ev), idx), "las partidas cambian con la semilla"
            por[s] = EV.metricas(y[idx], Ps[clave])
            sem_filas.append({"modelo": modelo, "minuto": minuto, "semilla": s, **por[s]})
        r = EV.resumen_semillas(por, SEMILLA_PRINCIPAL)
        if len(preds) > 1:
            r["predicciones"] = EV.correlacion_predicciones({s: Ps[clave] for s, Ps in preds.items()}, SEMILLA_PRINCIPAL)
        sem_resumen[f"{modelo}{minuto}"] = r

    # cuánto mueve la semilla del bootstrap los extremos del IC (semilla principal de los modelos)
    ic_boot = []
    for modelo, minuto, clave, idx in claves:
        for sb in SEMILLAS_BOOT:
            b = EV.bootstrap_bloques(y[idx], P[clave], dia[idx], B, sb)
            ic_boot.append({"modelo": modelo, "minuto": minuto, "semilla_bootstrap": sb,
                            **{f"{k}{s}": b[k][i] for k in ("auc", "acierto") for i, s in enumerate(("", "_lo", "_hi"))}})
        MV.log(f"IC por semilla del bootstrap: {modelo} {minuto}")
    return dict(metricas=pd.DataFrame(filas), ventanas=pd.concat(ventanas, ignore_index=True),
                pendientes=pd.DataFrame(pendientes), semillas=pd.DataFrame(sem_filas), semillas_resumen=sem_resumen,
                ic_bootstrap_semillas=pd.DataFrame(ic_boot))


# --------------------------------------------------------------------------------------------- salida

def versiones() -> dict:
    import scipy
    import sklearn
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "scipy": scipy.__version__, "plataforma": platform.platform()}


def _f(v, lo, hi, d=4):
    return f"{v:.{d}f} [{lo:.{d}f}; {hi:.{d}f}]"


def escribir_log(ruta: Path, info: dict, T: dict | None, chequeos_: dict) -> None:
    L = [f"# {'Simulacro de la evaluación (V hace de test)' if info['simulacro'] else 'Evaluación única en el test'}", "",
         f"* Fecha y hora: {info['inicio']} → {info['fin']} ({info['minutos']:.0f} min)",
         f"* Commit: `{info['commit']}`" + (" (árbol con cambios: solo admitido en el simulacro)" if info["git_sucio"] else ""),
         f"* Protocolo: `docs/evaluacion_test/PROTOCOLO.md`, SHA-256 `{info['hash_protocolo']}`",
         *[f"* SHA-256 `{n}`: `{h}` ({'coincide con el protocolo' if ok else 'NO COINCIDE'})"
           for n, (h, ok) in info["hashes"].items()],
         f"* Modo: `{info['modo']}`; regla de Master+: `{info['regla']}`; semillas: {list(info['semillas'])} "
         f"(principal {SEMILLA_PRINCIPAL}); B = {B}; ventana = {EV.TAM_VENTANA} partidas",
         f"* Versiones: {', '.join(f'{k} {v}' for k, v in info['versiones'].items())}",
         *([f"* **Excepción al protocolo:** {info['excepcion']}"] if info.get("excepcion") else []),
         "", "## Chequeos", "", *[f"* `{k}`: {v}" for k, v in chequeos_.items()], ""]
    if T is not None:
        M = T["metricas"]
        L += ["## Métricas (semilla principal, IC 95 % por bootstrap de días)", "",
              "| Modelo | Minuto | Conjunto | IC | n | días | AUC | Acierto | Siempre rojo | Ventaja | Log loss | ECE |",
              "| :--- | :---: | :--- | :---: | ---: | ---: | :--- | :--- | ---: | :--- | :--- | :--- |"]
        for _, r in M.iterrows():
            L.append(f"| {r.modelo} | {r.minuto} | {r.conjunto} | {r.ic} | {r.n:,} | {r.n_bloques if r.ic == 'dias' else ''} | "
                     f"{_f(r.auc, r.auc_lo, r.auc_hi)} | {_f(r.acierto, r.acierto_lo, r.acierto_hi)} | {r.siempre_rojo:.4f} | "
                     f"{_f(r.ventaja_rojo, r.ventaja_rojo_lo, r.ventaja_rojo_hi)} | {_f(r.log_loss, r.log_loss_lo, r.log_loss_hi)} | "
                     f"{_f(r.ece, r.ece_lo, r.ece_hi)} |")
        L += ["", "## Degradación: pendiente del AUC por semana (ventanas de 2.500 partidas)", "",
              "| Modelo | Minuto | Ventanas | Pendiente (AUC/semana) | IC 95 % |", "| :--- | :---: | ---: | ---: | :--- |"]
        for _, r in T["pendientes"].iterrows():
            L.append(f"| {r.modelo} | {r.minuto} | {r.n_ventanas} | {r.pendiente_semana:+.4f} | [{r.lo:+.4f}; {r.hi:+.4f}] |")
        L += ["", "## Semillas (conjunto de evaluación)", "",
              "| Modelo | Métrica | Principal | Media | Desvío | Mín | Máx | Corr. mín. de p | Mismo ganador (mín.) |",
              "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for clave, r in T["semillas_resumen"].items():
            pr = r.get("predicciones", {})
            cmin = f"{pr['corr_min_pares']:.5f}" if pr else ""
            gmin = f"{min(pr['mismo_ganador_que_principal'].values()):.4f}" if pr else ""
            for k in ("auc", "acierto", "log_loss", "ece"):
                s = r[k]
                L.append(f"| {clave} | {k} | {s['principal']:.4f} | {s['media']:.4f} | {s['desvio']:.5f} | {s['min']:.4f} | "
                         f"{s['max']:.4f} | {cmin if k == 'auc' else ''} | {gmin if k == 'auc' else ''} |")
        L += ["", "Tablas completas en `" + info["salida"] + "` (`metricas.csv`, `ventanas.csv`, `pendientes.csv`, "
              "`semillas.csv`, `ic_bootstrap_semillas.csv`, `resumen.json`)."]
    else:
        L += ["## Sin métricas", "", "Un chequeo falló: no se calculó ninguna métrica (protocolo, §7)."]
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # en Windows, con la salida redirigida, cp1252 no tiene "→"
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--simulacro", action="store_true", help="ensayo completo con V como si fuera el test (no lee el test)")
    ap.add_argument("--semillas", type=int, nargs="+", help="solo en el simulacro: otras semillas (la principal va siempre)")
    ap.add_argument("--sin-v", action="store_true",
                    help="anexo: mismo test con todo ajustado sin V (antes del 16/09); log y salida propios")
    ap.add_argument("--excepcion", metavar="MOTIVO",
                    help="EXCEPCIÓN al protocolo: repetir la evaluación aunque exista el log (requiere el OK del usuario; "
                         "el motivo queda en el log y el log anterior se conserva con fecha)")
    a = ap.parse_args(argv)
    if a.semillas and not (a.simulacro or a.sin_v):
        sys.exit("--semillas solo vale en el simulacro y en --sin-v: el test usa las semillas fijadas en el protocolo.")
    if a.sin_v and a.simulacro:
        sys.exit("--sin-v y --simulacro no van juntos.")
    por_defecto = (SEMILLA_PRINCIPAL,) if a.sin_v else SEMILLAS
    semillas = tuple(dict.fromkeys([SEMILLA_PRINCIPAL, *(a.semillas or por_defecto)]))
    t0 = time.time()
    inicio = datetime.now().astimezone().isoformat(timespec="seconds")

    sucio = estado_git()
    log = LOG_SIMULACRO if a.simulacro else LOG_SIN_V if a.sin_v else LOG
    problemas = problemas_previos(log, a.excepcion or a.simulacro, sucio, PROTOCOLO)
    if problemas:
        if a.simulacro:
            print("[simulacro] se ignoran:\n- " + "\n- ".join(problemas))
        else:
            sys.exit("No se corre la evaluación:\n- " + "\n- ".join(problemas))
    texto = PROTOCOLO.read_text(encoding="utf-8")
    esperados = hashes_del_protocolo(texto)
    archivos = {k: v for k, v in ARCHIVOS_HASH.items() if not (a.simulacro and k in ("test", "diamante_despues"))}
    hashes = verificar_hashes(esperados, archivos)
    malos = [k for k, (_, ok) in hashes.items() if not ok]
    if malos and not a.simulacro:
        sys.exit(f"Los hashes no coinciden con el protocolo ({', '.join(malos)}): no se carga el test.")
    if not a.simulacro:
        cab_train = pd.read_csv(rutas.TRAIN_CSV, nrows=0).columns.tolist()
        assert pd.read_csv(rutas.TEST_CSV, nrows=0).columns.tolist() == cab_train, "el CSV del test tiene otras columnas"
        if log.exists():                                            # excepción: se conserva el log anterior
            log.rename(log.with_name(f"{log.stem}_anterior_{datetime.now():%Y%m%d_%H%M%S}.md"))

    rutas.configurar("simulacro" if a.simulacro else "test_sin_v" if a.sin_v else "test")
    salida = SALIDA_SIMULACRO if a.simulacro else SALIDA_SIN_V if a.sin_v else SALIDA
    salida.mkdir(parents=True, exist_ok=True)
    info = dict(simulacro=a.simulacro, inicio=inicio, commit=git("rev-parse", "HEAD"), git_sucio=bool(sucio),
                hash_protocolo=hashlib.sha256(PROTOCOLO.read_bytes()).hexdigest(), hashes=hashes, modo=rutas.MODO,
                semillas=semillas, versiones=versiones(), excepcion=a.excepcion,
                salida=salida.relative_to(REPO).as_posix() if salida.is_relative_to(REPO) else str(salida))

    base, preds = None, {}
    firma = dict(commit=info["commit"], modo=rutas.MODO, hash_protocolo=info["hash_protocolo"])
    for s in semillas:
        guardado = cargar_parcial(salida, s, firma)
        if guardado is not None:
            b, preds[s] = guardado
            if base is None:
                base = b
            else:
                for k in ("mids", "ev", "dv", "tr"):
                    assert np.array_equal(base[k], b[k]), f"las partidas cambian con la semilla ({k})"
            print(f"[=] semilla {s}: predicciones retomadas de {salida / 'parciales'}", flush=True)
            continue
        base, preds[s] = predecir(s, base)
        guardar_parcial(salida, s, firma, base, preds[s])
    info["regla"] = base["regla"]
    ch = chequeos(base, preds[SEMILLA_PRINCIPAL], a.simulacro)
    falla = [k for k, v in ch.items() if v is False]
    if falla:
        info.update(fin=datetime.now().astimezone().isoformat(timespec="seconds"), minutos=(time.time() - t0) / 60)
        ruta = log.with_name(log.stem + f"_FALLIDA_{datetime.now():%Y%m%d_%H%M%S}.md")
        escribir_log(ruta, info, None, ch)
        sys.exit(f"Fallaron chequeos ({', '.join(falla)}): no se calculó ninguna métrica. Ver {ruta}.")

    T = medir(base, preds)
    T["metricas"].to_csv(salida / "metricas.csv", index=False)
    T["ventanas"].to_csv(salida / "ventanas.csv", index=False)
    T["pendientes"].to_csv(salida / "pendientes.csv", index=False)
    T["semillas"].to_csv(salida / "semillas.csv", index=False)
    T["ic_bootstrap_semillas"].to_csv(salida / "ic_bootstrap_semillas.csv", index=False)
    (salida / "chequeos.json").write_text(json.dumps(ch, indent=2, ensure_ascii=False), encoding="utf-8")
    info.update(fin=datetime.now().astimezone().isoformat(timespec="seconds"), minutos=(time.time() - t0) / 60)
    resumen = {k: v for k, v in info.items() if k != "hashes"} | {
        "hashes": {k: h for k, (h, _) in hashes.items()}, "semillas_resumen": T["semillas_resumen"],
        "n_evaluacion": int(len(base["ev"])), "n_diamante": int(len(base["dv"])), "n_entrenamiento": int(len(base["tr"])),
        "C_etapa1": C_ETAPA1, "n_columnas_previo": int(preds[SEMILLA_PRINCIPAL]["n_columnas"])}
    (salida / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    np.savez_compressed(salida / "predicciones.npz", mids=base["mids"], ev=base["ev"], dv=base["dv"],
                        **{f"s{s}_{k}": v for s, P in preds.items() for k, v in P.items() if isinstance(v, np.ndarray)})
    escribir_log(log, info, T, ch)
    print(f"\n[+] {log}\n[+] {salida}  ({(time.time() - t0) / 60:.0f} min)")
    if not a.simulacro:
        print("Commitear el log y las tablas tal como salieron (no se corre de nuevo).")


if __name__ == "__main__":
    main()
