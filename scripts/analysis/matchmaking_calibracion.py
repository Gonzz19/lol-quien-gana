"""Audita la calibración del escalón 9 y de su variante anterior al draft.

Ejecutar: python -u scripts/analysis/matchmaking_calibracion.py
Requiere pyarrow y las cachés ya construidas del escalón 9. Lee el split de
entrenamiento y la foto 1. Guarda la auditoría en data/analysis/matchmaking/.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from contextlib import closing, contextmanager
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.analysis.brainstorm_base import ajustar_predecir, cargar, extras_referencia
from src.features.fingerprint import add_rates
from src.features.lobby import prior_lobby_level
from src.features.margin import ORDEN, lane_diffs, lane_scale, prior_margin, prior_margin_by
from src.features.padron import leer_foto1
from src.features.rating import lp_absoluto
from src.models.validacion_temporal import bloques_por_fecha

ANALISIS = REPO / "data" / "analysis"
CACHES = ANALISIS / "brainstorm"
SALIDA = ANALISIS / "matchmaking"
TRAMOS = [0, .35, .40, .45, .50, .55, .60, .65, 1]
ROLES = ("BOTTOM", "JUNGLE", "MIDDLE", "TOP", "UTILITY")
STATS = (
    "kills_pm", "assists_pm", "damage_pm", "damage_share_pct", "kill_participation_pct",
    "deaths_pm", "damage_taken_pm", "gold_pm", "cs_pm", "gold_share_pct", "vision_pm",
    "d_gold_10", "d_gold_15", "d_xp_15", "d_cs_15",
)
XJ_HISTORIAL = ("wr_acum", "wr_v20", "logn_tot", "share_lane", "offrole",
                "log_horas_desde", "ritmo", "conc_rec", "conc_ant", "d_pool")
ROL_WR = ("wr_linea_media", "exceso_wr_linea_media", "exceso_wr_campeon_linea_media",
          "logn_linea_media", "estreno_linea_media", "caida_autofill_media", "autofill_media",
          "wr_linea_minimo", "exceso_wr_linea_minimo", "exceso_wr_campeon_linea_minimo")
TEMPO = ("primera_sangre", "primera_sangre_jungla_soporte", "primera_torre",
         "primera_torre_bot_top", "cambio_trinket", "duracion", "sesgo_temprano_tardio_jugador",
         "sesgo_campeon_linea_media", "sesgo_campeon_linea_maximo", "interaccion_jugador_campeon")
EXTRAS = (
    ("MC", tuple(f"{rol}/{s}" for rol in ROLES for s in STATS), ()),
    ("ML", tuple(f"{rol}/{s}" for rol in ROLES for s in STATS), tuple(range(75))),
    ("MMR_HIST", ("acumulado", "ultimas_10", "logn"), (0, 1, 2)),
    ("FOTO1_DIN", ("winrate", "logn", "victorias_netas", "master_0_lp"), (0, 1, 2, 3)),
    ("MATCHUPS", tuple(map(str, range(10))), ()),
    ("ROL_WR", ROL_WR, (0, 1, 3, 4, 5, 6, 7, 8)),
    ("KALMAN_OUT", ("media", "minimo", "maximo", "desvio", "media_dos_mayores", "forma",
                    "nivel_rivales", *ROLES), tuple(range(12))),
    ("KALMAN_MRG", ("media", "minimo", "maximo", "forma", "incertidumbre", "exceso_sobre_lobby",
                    "brecha_lobby_absoluta"), tuple(range(7))),
    ("TEMPO_OBJ", TEMPO, tuple(range(7))),
    ("ECON_MID", ("crecimiento_oro_media", "crecimiento_oro_maximo", "eficiencia_media",
                  "eficiencia_bot_mid"), (0, 1, 2, 3)),
    ("HGB", (*ROLES, "suma"), ()),
    ("MATCHUP_EXP_MARGIN", tuple(map(str, range(12))), ()),
)
RAZONES_EXTRAS = {
    "MC": "Margen del jugador condicionado al campeón que eligió.",
    "MATCHUPS": "Contracruces por línea y sinergias de campeones aliados.",
    "ROL_WR": "Exceso de win rate del jugador con el campeón elegido en la línea.",
    "TEMPO_OBJ": "Sesgo temprano/tardío del campeón en la línea e interacción con el jugador.",
    "HGB": "Predicción de oro @15 por línea y suma que incorporan campeón, composición y contracruces.",
    "MATCHUP_EXP_MARGIN": "Margen esperado de oro, XP y CS condicionado al par de campeones.",
}
TITULOS = {"escalon9": "Escalón 9 con draft", "sin_draft": "Solo jugadores y línea"}
N_BOOT = 10_000
EPS = 1e-12


@contextmanager
def reservar_recurso():
    SALIDA.mkdir(parents=True, exist_ok=True)
    marca = SALIDA / "RECURSO_EN_USO.txt"
    while True:
        try:
            archivo = marca.open("x", encoding="utf-8")
        except FileExistsError:
            print(f"Esperando la liberación de {marca.name}.", flush=True)
            time.sleep(15)
        else:
            break
    try:
        with archivo:
            archivo.write(f"sol-calibracion {datetime.now().astimezone().isoformat()}\n")
        yield
    finally:
        marca.unlink()


def verificar_archivos():
    requeridos = [ANALISIS / "embeddings_partidas" / "variables.npz",
                  ANALISIS / "mapa_lp_causal_v.pkl"]
    requeridos.extend(CACHES / n for n in (
        "base.npz", "margen_grupos.npz", "mmr_historico.npz", "gemini_mpr.npz",
        "gemini_nuevas_v2.npz", "gemini_g11.npz", "gemini_hgb_lineas.npz", "gemini_g23.npz",
    ))
    faltan = [str(r.relative_to(REPO)) for r in requeridos if not r.is_file()]
    if faltan:
        raise FileNotFoundError("Faltan cachés del escalón 9: " + ", ".join(faltan))


def esquema_sin_draft(D, extras):
    margen = tuple(c for s in STATS for c in (f"acum_{s}", f"v20_{s}"))
    conservar = set(margen + XJ_HISTORIAL)
    experiencia = {"logn_champ", "estreno", "wr_champ", "share_champ", "otp_nuevo"}
    columnas = list(map(str, D["cols"]))
    quitadas = {}
    for c in columnas:
        if c in experiencia:
            quitadas[c] = "Experiencia o cambio de uso del campeón elegido."
        elif c.startswith(("tag_", "info_", "st_", "perf_")):
            quitadas[c] = "Atributo estático o perfil del campeón elegido."
    if conservar | quitadas.keys() != set(columnas) or conservar & quitadas.keys():
        raise ValueError("Revisar el esquema de columnas de variables.npz.")

    base = [c for c in columnas if c in conservar]
    seleccion = []
    grupos = {}
    inicio = 0
    for clave, nombres, elegidos in EXTRAS:
        grupos[clave] = slice(inicio, inicio + len(nombres))
        seleccion.extend(inicio + i for i in elegidos)
        inicio += len(nombres)
    if inicio != extras.shape[1] or D["_ref"].shape[1] != grupos["MATCHUPS"].start:
        raise ValueError("Cambió el orden o el ancho de extras_referencia.")
    manifiesto = {
        "orden_columnas_modelo": [f"Xj/{rol}/{c}" for rol in ROLES for c in base]
        + [f"{g}/{nombres[i]}" for g, nombres, elegidos in EXTRAS for i in elegidos],
        "Xj_conservadas_por_linea": base,
        "Xj_excluidas_por_linea": quitadas,
        "draft_excluido": {"cantidad": D["DRAFT"].shape[1], "motivo": "Campeón elegido × línea."},
        "extras_conservadas": {g: {i: nombres[i] for i in elegidos}
                               for g, nombres, elegidos in EXTRAS if elegidos},
        "extras_excluidas": {
            g: {"indices_base_cero_en_bloque": [i for i in range(len(nombres)) if i not in elegidos],
                "motivo": RAZONES_EXTRAS[g]}
            for g, nombres, elegidos in EXTRAS if len(elegidos) < len(nombres)
        },
        "historial_de_campeones": (
            "Los márgenes se ajustan por el campeón de cada partida histórica. Conc_rec, conc_ant y "
            "d_pool describen la concentración del pool anterior. Todos se pueden computar antes del draft."
        ),
    }
    return columnas, base, seleccion, grupos, manifiesto


def rehacer_historial(D, corte, columnas, base):
    """Congela la escala de los márgenes y la imputación antes del bloque de Platt."""
    uso = [*ORDEN, "player_id", "win", "champion_id", "game_duration", "kills", "deaths", "assists",
           "kill_participation_pct", "damage_share_pct", "gold_share_pct",
           "total_damage_dealt_to_champions", "total_damage_taken", "gold_earned",
           "total_minions_killed", "neutral_minions_killed", "vision_score",
           "gold_10", "gold_15", "xp_15", "cs_15"]
    df = pd.read_csv(REPO / "data" / "splits" / "train" / "full_dataset.csv", usecols=uso,
                     dtype={"lane": "category", "match_id": "category", "player_id": "category"})
    df = df[df["lane"].isin(ROLES)]
    df = df[df.groupby(["match_id", "team_id"], observed=True)["lane"].transform("nunique") == 5]
    por_partida = df.groupby("match_id", observed=True)["win"]
    df = df[(por_partida.transform("size") == 10) & (por_partida.transform("sum") == 5)]
    df = df.sort_values(ORDEN).reset_index(drop=True)
    if not np.array_equal(df["match_id"].to_numpy()[::10], D["Z"]["mids"]):
        raise ValueError("El CSV de entrenamiento y la caché no tienen las mismas partidas ordenadas.")
    for c in ("lane", "match_id", "player_id"):
        df[c] = df[c].cat.codes
    df = lane_diffs(add_rates(df), ["gold_10", "gold_15", "xp_15", "cs_15"])
    antes = df["game_creation"] < corte
    escala = lane_scale(df.loc[antes, ["lane", *STATS]], list(STATS))
    nm = D["NM"]

    def diferencia(x):
        x = np.asarray(x, dtype=np.float32).reshape(nm, 2, 5, -1)
        return x[:, 0] - x[:, 1]

    xd = D["Xd"].reshape(nm, 5, -1)[..., [columnas.index(c) for c in base]]
    ml = np.empty((nm, 5, len(STATS)), dtype=np.float32)
    for i, stat in enumerate(STATS):
        margen = prior_margin(df, [stat], escala=escala)
        xd[..., [base.index(f"acum_{stat}"), base.index(f"v20_{stat}")]] = diferencia(margen)
        ml[..., i:i + 1] = diferencia(prior_margin_by(df, [stat], ["lane"], "ml", escala=escala))
    horas = (df["game_creation"] - df.groupby("player_id", observed=True)["game_creation"].shift()) / 3.6e6
    log_horas = np.log1p(horas)
    xd[..., base.index("log_horas_desde")] = diferencia(log_horas.fillna(log_horas[antes].median()))[..., 0]
    return xd.reshape(nm, -1), ml.reshape(nm, -1)


def nivel_sin_ventana(D, corte):
    """Usa la foto 1 y fija el prior del nivel histórico antes del bloque de Platt."""
    uri = (REPO / "data" / "lol_las.db").resolve().as_uri() + "?mode=ro&immutable=1"
    with closing(sqlite3.connect(uri, uri=True)) as con:
        foto = leer_foto1(con, "p.tier, p.rank, p.league_points, p.wins, p.losses")
    foto["z"] = [lp_absoluto(t, r, l) for t, r, l in zip(foto.tier, foto["rank"], foto.league_points)]
    foto = foto.assign(partidas=foto.wins + foto.losses).sort_values("partidas")
    x = foto.drop_duplicates("player_id", keep="last").set_index("player_id")["z"]
    df = pd.DataFrame({
        "match_id": np.repeat(np.arange(D["NM"]), 10),
        "game_creation": np.repeat(D["t"], 10),
        "player_id": pd.Categorical(D["B"]["player"].reshape(-1)),
    })
    # Códigos enteros reducen la memoria de las agrupaciones frente a los IDs de texto.
    df["x"] = x.reindex(df.player_id.cat.categories).to_numpy()[df.player_id.cat.codes]
    df["player_id"] = df.player_id.cat.codes
    mu = df.loc[df.game_creation < corte, "x"].mean()
    nivel = prior_lobby_level(df, "x", mu0=mu)
    a = np.column_stack((nivel.nivel_acum - mu, nivel.nivel_v10 - mu, np.log1p(nivel.n_nivel)))
    a = a.astype(np.float32).reshape(D["NM"], 2, 5, 3)
    return (a[:, 0] - a[:, 1]).mean(axis=1)


def logits(p):
    return logit(np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS))


def calibrar(y, p):
    """Ajuste de intercepto y pendiente por máxima verosimilitud, sin penalización."""
    z = logits(p).reshape(-1, 1)
    lr = LogisticRegression(C=np.inf, tol=1e-10, max_iter=1000).fit(z, y)
    a, b = float(lr.intercept_[0]), float(lr.coef_[0, 0])
    q = expit(a + b * z[:, 0])
    x = np.column_stack((np.ones(len(z)), z[:, 0]))
    cov = np.linalg.inv(x.T @ ((q * (1 - q))[:, None] * x))
    se = np.sqrt(cov.diagonal())
    return a, b, (a - 1.96 * se[0], a + 1.96 * se[0]), (b - 1.96 * se[1], b + 1.96 * se[1])


def aplicar_platt(p, intercepto, pendiente):
    return expit(intercepto + pendiente * logits(p))


def brier(y, p):
    return np.square(y - p).mean()


def platt_temporal(y, p, fechas):
    mitad = fechas >= np.median(fechas)
    a, b, *_ = calibrar(y[~mitad], p[~mitad])
    y2, p2 = y[mitad], p[mitad]
    q = aplicar_platt(p2, a, b)
    delta_ll = log_loss(y2, q) - log_loss(y2, p2)
    delta_brier = brier(y2, q) - brier(y2, p2)
    a, b, *_ = calibrar(y, p)
    return {
        "intercepto": a, "pendiente": b, "n_chequeo": len(y2), "n_preajuste": int((~mitad).sum()),
        "delta_log_loss_chequeo": delta_ll, "delta_brier_chequeo": delta_brier,
        "preferido_pre_v": bool(b > 0 and delta_ll < 0 and delta_brier < 0),
    }


def observacion(y, p, rng):
    n = len(y)
    if n == 0:
        return dict(n=0, p=np.nan, wr=np.nan, li=np.nan, ls=np.nan)
    wr = y.mean()
    # Para resultados binarios, remuestrear n partidas con reemplazo da exactamente Binomial(n, WR).
    li, ls = np.quantile(rng.binomial(n, wr, N_BOOT) / n, [.025, .975])
    return dict(n=n, p=p.mean(), wr=wr, li=li, ls=ls)


def confiabilidad(y, p, cortes, rng):
    indice = np.searchsorted(cortes[1:-1], p, side="right")
    filas = {}
    for i, (lo, hi) in enumerate(zip(cortes, cortes[1:])):
        m = indice == i
        filas[f"[{lo:.4g}, {hi:.4g}{']' if hi == cortes[-1] else ')'}"] = observacion(y[m], p[m], rng)
    return pd.DataFrame.from_dict(filas, orient="index")


def evaluar(y, p):
    rng = np.random.default_rng(260919)
    cuantiles = np.r_[0., np.unique(np.quantile(p, np.arange(1, 10) / 10)), 1.]
    dec = confiabilidad(y, p, cuantiles, rng)
    tramos = confiabilidad(y, p, TRAMOS, rng)
    a, b, ic_a, ic_b = calibrar(y, p)
    p_fav = np.maximum(p, 1 - p)
    gana_fav = np.where(p > .5, y, 1 - y)
    cerca = (p_fav >= .575) & (p_fav < .625)
    return {
        "n": len(y), "auc": roc_auc_score(y, p), "acierto": ((p > .5) == y).mean(),
        "log_loss": log_loss(y, p), "brier": brier(y, p), "intercepto": a, "pendiente": b,
        "ic_intercepto": ic_a, "ic_pendiente": ic_b,
        "gap": y.mean() - p.mean(), "deciles": dec, "tramos": tramos,
        "ece": (dec.n * (dec.p - dec.wr).abs()).sum() / len(y),
        "ece_fijo": (tramos.n * (tramos.p - tramos.wr).abs()).sum() / len(y),
        "cerca60": {lado: observacion(gana_fav[m], p_fav[m], rng) for lado, m in
                    (("Ambos", cerca), ("Azul", cerca & (p > .5)), ("Rojo", cerca & (p < .5)))},
    }


def graficar(clave, bruto, platt):
    fig, axes = plt.subplots(2, 2, sharex="col", figsize=(12, 7.6),
                             gridspec_kw={"height_ratios": [4, 1]}, layout="constrained")
    fig.suptitle(f"{TITULOS[clave]} · V Master+ (barras: IC 95 % bootstrap por partida, B = {N_BOOT:,})".replace(",", "."),
                 fontsize=14.5, fontweight="bold")
    for j, (tipo, titulo) in enumerate((("deciles", "Deciles de probabilidad"), ("tramos", "Tramos fijos"))):
        ax, ax_n = axes[:, j]
        ax.plot([0, 1], [0, 1], "--", color="#708090", label="Calibración ideal")
        for color, etiqueta, m in (("#07547e", "Probabilidad original", bruto), ("#ac4a18", "Platt", platt)):
            t = m[tipo].dropna()
            ax.errorbar(t.p, t.wr, yerr=np.maximum(0, [t.wr - t.li, t.ls - t.wr]),
                        color=color, fmt="o-", linewidth=1.5, markersize=4.5, capsize=3, label=etiqueta)
            ax_n.scatter(t.p, t.n, color=color, s=18)
        t = bruto[tipo].dropna()
        for i, (_, r) in enumerate(t.iterrows()):
            ax_n.annotate(f"{int(r.n):,}".replace(",", "."), (r.p, r.n), fontsize=7, ha="center",
                          xytext=(0, 5 + 10 * (i % 2)), textcoords="offset points", color="#07547e")
        ax_n.set_ylim(0, 1.33 * max(bruto[tipo].n.max(), platt[tipo].n.max()))
        ax.set(title=titulo, xlim=(.12, .88), ylim=(.12, .88), ylabel="Frecuencia de victoria azul")
        ax_n.set(xlabel="Probabilidad media de victoria azul", ylabel="Partidas")
        ax.grid(alpha=.20)
        ax_n.grid(alpha=.20)
        ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.savefig(SALIDA / f"calib_confiabilidad_{clave}.png", dpi=168)
    plt.close(fig)


def fecha(ms):
    return pd.Timestamp(int(ms), unit="ms", tz="UTC").tz_convert("America/Argentina/Buenos_Aires").strftime(
        "%d/%m/%Y %H:%M:%S")


def num(x, dec=4):
    return f"{x:.{dec}f}".replace(".", ",")


def pct(x):
    return f"{num(100*x, 2)} %"


def cuenta(n):
    return f"{n:,}".replace(",", ".")


def ic(limites, formato=num):
    return f"[{formato(limites[0])} a {formato(limites[1])}]"


def rango(t):
    return f"{fecha(t.min())} a {fecha(t.max())}"


def tabla_tramos(m):
    return [
        "| Intervalo de p azul | n | p media | WR azul | IC 95 % | WR − p (pp) |",
        "|---|---:|---:|---:|---:|---:|",
        *(f"| `{k}` | {cuenta(int(r.n))} | {pct(r.p)} | {pct(r.wr)} | "
          f"{ic((r.li,r.ls),pct)} | {num(100*(r.wr-r.p), 2)} |" if r.n else
          f"| `{k}` | 0 | — | — | — | — |" for k, r in m.iterrows()),
    ]


def tabla_metricas(evaluaciones):
    filas = ["| Variante y conjunto | n | AUC / acierto | Brier | Log loss | ECE deciles / fijo"
             " | Intercepto [IC 95 %] | Pendiente [IC 95 %] | WR − p (pp) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for titulo, m in evaluaciones:
        filas.append(
            f"| {titulo} | {cuenta(m['n'])} | {num(m['auc'])} / {pct(m['acierto'])} | {num(m['brier'])} | "
            f"{num(m['log_loss'])} | {pct(m['ece'])} / {pct(m['ece_fijo'])} | "
            f"{num(m['intercepto'])} {ic(m['ic_intercepto'])} | "
            f"{num(m['pendiente'])} {ic(m['ic_pendiente'])} | {num(100*m['gap'], 2)} |"
        )
    return filas


def respuesta_60(r):
    if r["li"] >= .60:
        return "Sí, el IC queda por encima del 60 %."
    if r["ls"] < .60:
        return "No, el IC queda por debajo del 60 %."
    punto = "alcanza" if r["wr"] >= .60 else "queda por debajo de"
    return f"La estimación puntual {punto} 60 %. El IC incluye 60 %."


def informe(resultados, mapas, fechas, tr_cal, cal, tr, columnas, comparacion, inicio):
    lineas = [
        "# Calibración del modelo previo para matchmaking", "",
        f"Se ajustaron las logísticas con **{cuenta(len(tr))}** partidas Master+ ({rango(tr)}). "
        f"Para ajustar Platt, otra logística se entrenó con **{cuenta(len(tr_cal))}** Master+ ({rango(tr_cal)}) "
        f"y predijo las **{cuenta(len(cal))}** Master+ del último décimo aproximado ({rango(cal)}). "
        "Se usó el cuantil temporal 90 % de todas las partidas anteriores a V, que coincide con un corte "
        "del HGB de la referencia. El corte vigente cae el 16/09 en UTC. "
        "Los parámetros de Platt se congelaron antes de evaluar ambas ventanas.",
        "",
        f"Se reprodujeron **{cuenta(columnas)} columnas** en el escalón 9 (C = 0,05, sin mapa mecánico de LP). "
        f"{comparacion} La calibración se midió sobre "
        "partidas observadas con sus alineaciones reales. Los swaps crean nuevas alineaciones y "
        "el efecto de esas intervenciones requiere validación propia.", "",
        "## Veredicto", "",
    ]
    for clave in TITULOS:
        m, q = resultados[clave]["V"]
        mapa = mapas[clave]
        candidat = "Platt" if mapa["preferido_pre_v"] else "la probabilidad original"
        delta = q["log_loss"] - m["log_loss"]
        estado = "mejoró" if delta < 0 else "empeoró"
        tipo_p = "p recalibrada" if mapa["recomendado_v_master"] else "p original"
        uso = (f"Usar la {tipo_p} como referencia después del draft." if clave == "escalon9" else
               f"Usar la {tipo_p} y medir desequilibrio con $|p_{{azul}}-0,5|$ en el estudio sobre Master+.")
        lineas.extend([
            f"- **{TITULOS[clave]}.** {uso} La regla temporal previa a V eligió {candidat}. "
            f"En V, Platt {estado} el log loss (Δ {num(delta, 5)}) y cambió el Brier en "
            f"{num(q['brier'] - m['brier'], 5)}.",
        ])
    lineas.extend([
        "- **Diamond-V.** La calibración de Master+ pierde validez al cambiar de tier. "
        "Las pendientes observadas indican sobreconfianza con p original y con el mapa Platt de Master+. "
        "El uso como probabilidad requiere una validación y un ajuste por tier.",
    ])
    lineas.extend(["", "## V Master+ y la pregunta del 60 %", "",
                   r"Se evalúa un equipo por partida con confianza $p_f=\max(p_{azul},1-p_{azul})$ "
                   "en [57,5 %, 62,5 %). Se invierte el resultado cuando el favorito es rojo.", ""])
    for clave, titulo in TITULOS.items():
        for sufijo, m in zip(("Original", "Platt"), resultados[clave]["V"]):
            r = m["cerca60"]["Ambos"]
            lineas.append(f"- **{titulo}, {sufijo.lower()}.** {cuenta(r['n'])} partidas, "
                          f"p media {pct(r['p'])}, gana {pct(r['wr'])} {ic((r['li'],r['ls']),pct)}. "
                          f"{respuesta_60(r)}")
    lineas.extend(["", "| Variante | Favorito | n | p media | Acierto | IC 95 % |",
                   "|---|---|---:|---:|---:|---:|"])
    for clave, titulo in TITULOS.items():
        for lado in ("Azul", "Rojo"):
            r = resultados[clave]["V"][0]["cerca60"][lado]
            lineas.append(f"| {titulo} | {lado} | {cuenta(r['n'])} | {pct(r['p'])} | {pct(r['wr'])}"
                          f" | {ic((r['li'], r['ls']), pct)} |")
    lineas.extend(["", *tabla_metricas((f"{titulo} · {estado}", m) for clave, titulo in TITULOS.items()
                                       for estado, m in zip(("original", "Platt"), resultados[clave]["V"])), ""])

    lineas.extend(["## Estabilidad temporal y por tier", ""])
    lineas.extend(f"- **{n}** · {rango(t)} (hora argentina)." for n, t in fechas.items())
    lineas.extend(["", "A4 Diamond-V comprende solo la parte Diamond/no Master+ de la ventana de V. "
                   "Se predice con el mismo ajuste de V Master+, así que la comparación por tier usa "
                   "el mismo modelo y el mismo calendario.", ""])
    for clave, titulo in TITULOS.items():
        rs = resultados[clave]
        lineas.extend([f"### {titulo}", "",
                      *tabla_metricas((f"{n} · {e}", m) for n in fechas if n != "V" for e, m in
                                      zip(("original", "Platt"), rs[n])), ""])
        for i, estado in enumerate(("Original", "Platt")):
            v = rs["V"][i]
            a4 = rs["A4 Diamond-V"][i]
            bloques = [rs[f"V{i_b}"][i] for i_b in (1, 2, 3)]
            ini, fin = bloques[0], bloques[-1]
            lineas.append(
                f"- **{estado}.** De V1 a V3, Δ pendiente {num(fin['pendiente'] - ini['pendiente'])}, "
                f"Δ intercepto {num(fin['intercepto'] - ini['intercepto'])}, Δ ECE por deciles "
                f"{num(100*(fin['ece'] - ini['ece']), 2)} pp y Δ (WR − p) {num(100*(fin['gap'] - ini['gap']), 2)} pp. "
                f"En Diamond-V frente a V Master+, Δ pendiente {num(a4['pendiente'] - v['pendiente'])}, "
                f"Δ intercepto {num(a4['intercepto'] - v['intercepto'])}, Δ ECE por deciles "
                f"{num(100*(a4['ece'] - v['ece']), 2)} pp."
            )
        lineas.append("")
    lineas.extend(["## Tablas de confiabilidad y gráficos", ""])
    for clave, titulo in TITULOS.items():
        lineas.extend([f"### {titulo}", "", f"![Diagrama con deciles y tramos](calib_confiabilidad_{clave}.png)", ""])
        for n, par in resultados[clave].items():
            for estado, m in zip(("Original", "Platt"), par):
                if n != "V" or estado == "Platt":
                    lineas.extend(["<details>", f"<summary>{n} · {estado} · deciles y tramos</summary>", ""])
                lineas.extend(["**Tramos fijos**", "", *tabla_tramos(m["tramos"]), "",
                               "**Deciles**", "", *tabla_tramos(m["deciles"]), ""])
                if n != "V" or estado == "Platt":
                    lineas.extend(["</details>", ""])
    lineas.extend([
        "## Variables, protocolo y aplicación del mapa", "",
        "- La variante sin draft conserva historial general, por línea, W/L de la foto 1, nivel histórico, "
        "Kalman, tempo por jugador y economía histórica. La concentración de campeones históricos y "
        "el rendimiento residualizado por los campeones de partidas previas están disponibles antes del draft.",
        "- Se excluyen el one-hot campeón × línea, los atributos del campeón elegido, su historial con "
        "el jugador, MC, MATCHUPS, HGB y MATCHUP_EXP_MARGIN. Se quitan ROL_WR[2, 9] y TEMPO_OBJ[7:10] "
        "con índices base cero. El JSON enumera todas las columnas y el orden utilizado.",
        "- El escalón 9 usa escalas, imputaciones y prior estimados antes de V (PROJECT_SUMMARY.md §9.S.68). "
        "Como el último décimo del ajuste aporta a esos agregados, la prueba de Platt con draft "
        "es descriptiva en ese décimo. Para la variante sin draft se recalculan escalas, imputación y prior "
        "antes de ese bloque. Las historias se actualizan con las partidas registradas previamente "
        "y la foto 1 aporta el tier, LP y W/L históricos.",
        "- Se eligió Platt de antemano como familia de dos parámetros, con la forma "
        r"$q=\sigma(a+b\,\mathrm{logit}(p))$. Se prefiere si mejora Brier y log loss en la segunda mitad "
        "del bloque de calibración anterior a V tras ajustarlo sobre la primera. Los parámetros finales "
        "del candidato se ajustan sobre el bloque completo anterior a V. Una pendiente positiva conserva "
        "el orden de probabilidades de azul. El mapa puede cambiar el umbral de decisión y la distancia a 0,5. "
        "V solo comprueba si se sostienen las mejoras de Brier y log loss.",
        "- Para aplicar el mapa, leer `calib_mapa.json`, tomar los campos `intercepto` y `pendiente` "
        "de la variante y usar `aplicar_platt(p_azul, intercepto, pendiente)`. El campo "
        "`recomendado_v_master` indica si la auditoría respalda su uso en V Master+. La probabilidad roja es $1-q$.",
        r"- $ECE=\sum_j(n_j/n)|\bar p_j-\bar y_j|$. Brier es la media de $(y-p)^2$ y el acierto "
        "usa $p>0,5$ para azul. Intercepto ideal 0 y pendiente ideal 1. Una pendiente menor que 1 "
        "señala sobreconfianza y una mayor que 1, subconfianza. Los IC de los parámetros usan Wald "
        "y los de WR son percentiles del bootstrap por partida condicionado al tramo (10.000 réplicas, "
        "semilla fija). Los IC tratan las partidas como unidades independientes.",
        "",
        "| Variante | Platt a | Platt b | n 1.ª / 2.ª mitad | Δ log loss interno | Δ Brier interno | Preferido pre-V |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    for clave, m in mapas.items():
        lineas.append(
            f"| {TITULOS[clave]} | {num(m['intercepto'])} | {num(m['pendiente'])} | "
            f"{cuenta(m['n_preajuste'])} / {cuenta(m['n_chequeo'])} | "
            f"{num(m['delta_log_loss_chequeo'], 5)} | {num(m['delta_brier_chequeo'], 5)} | "
            f"{'Sí' if m['preferido_pre_v'] else 'No'} |"
        )
    lineas.extend(["", f"Tiempo de corrida con el recurso reservado **{time.perf_counter() - inicio:.1f} s**.", ""])
    (SALIDA / "REPORTE_calibracion.md").write_text("\n".join(lineas), encoding="utf-8")


def main():
    verificar_archivos()
    with reservar_recurso(), threadpool_limits(limits=2):
        inicio = time.perf_counter()
        print("Cargando las cachés del escalón 9.", flush=True)
        D = cargar()
        for k in ("Xj", "J", "mapa"):
            D.pop(k)
        t, y = D["t"], D["y"]
        tr = D["fi_v"][D["master"][D["fi_v"]]]
        v = np.flatnonzero(D["val"])
        dv = np.flatnonzero(D["diam_v"] & (D["bloque"] == 3))
        te = np.r_[v, dv]
        if D["val0"][tr].any() or t[tr].max() >= t[te].min():
            raise ValueError("El ajuste incluye fechas de V o Diamond-V.")
        corte = np.quantile(t[~D["val0"]], .90)
        tr_cal, cal = tr[t[tr] < corte], tr[t[tr] >= corte]
        if not len(tr_cal) or not len(cal) or not D["post"][cal].all():
            raise ValueError("El bloque de Platt debe ser anterior a V y posterior a la foto 1.")
        extras = extras_referencia(D, escalon=9)
        columnas, base, keep, grupos, esquema = esquema_sin_draft(D, extras)

        print("Reproduciendo el escalón 9 y sus predicciones previas a V.", flush=True)
        p9, lr = ajustar_predecir(D, tr, te, extra=extras, extra_post=False,
                                   con_mapa=False, con_draft=True, devolver_modelo=True)
        p9_cal = ajustar_predecir(D, tr_cal, cal, extra=extras, extra_post=False,
                                  con_mapa=False, con_draft=True)
        archivo = CACHES / "pred_final.npz"
        comparacion = "La referencia de brainstorm_final.py no estaba disponible."
        if archivo.is_file():
            with np.load(archivo) as ref:
                if not np.array_equal(ref["idx_v"], v):
                    raise ValueError("Cambió la selección de filas de la referencia de brainstorm_final.py.")
                error = np.max(np.abs(ref["p_v"] - p9[:len(v)]))
            comparacion = f"Diferencia máxima de p frente a la referencia de brainstorm_final.py: {error:.3g}."
        n9 = lr.coef_.shape[1]
        ml_original = grupos["ML"]
        mmr_original = grupos["MMR_HIST"]
        extras = extras[:, keep]
        D.pop("_ref")

        print("Reconstruyendo los márgenes y el nivel histórico sin estadísticas de V.", flush=True)
        D["Xd"], ml = rehacer_historial(D, corte, columnas, base)
        extras[:, [keep.index(i) for i in range(ml_original.start, ml_original.stop)]] = ml
        del ml
        extras[:, [keep.index(i) for i in range(mmr_original.start, mmr_original.stop)]] = nivel_sin_ventana(D, corte)
        if len(esquema["orden_columnas_modelo"]) != D["Xd"].shape[1] + extras.shape[1]:
            raise ValueError("El manifiesto no coincide con la matriz sin draft.")

        print("Ajustando el modelo sin draft y evaluando la calibración.", flush=True)
        ps = ajustar_predecir(D, tr, te, extra=extras, extra_post=False, con_mapa=False, con_draft=False)
        ps_cal = ajustar_predecir(D, tr_cal, cal, extra=extras, extra_post=False, con_mapa=False, con_draft=False)
        bloques = bloques_por_fecha(t[v], 3)
        grupos_v = {"V": np.arange(len(v)), **{f"V{i+1}": np.flatnonzero(bloques == i) for i in range(3)},
                    "A4 Diamond-V": np.arange(len(v), len(te))}
        mapas, resultados = {}, {}
        for clave, p, p_cal in (("escalon9", p9, p9_cal), ("sin_draft", ps, ps_cal)):
            mapa = mapas[clave] = platt_temporal(y[cal], p_cal, t[cal])
            q = aplicar_platt(p, mapa["intercepto"], mapa["pendiente"])
            resultados[clave] = {k: (evaluar(y[te[g]], p[g]), evaluar(y[te[g]], q[g])) for k, g in grupos_v.items()}
            m, corr = resultados[clave]["V"]
            mapa["recomendado_v_master"] = bool(mapa["preferido_pre_v"] and
                                               corr["log_loss"] < m["log_loss"] and corr["brier"] < m["brier"])
            pd.DataFrame({"match_id": D["Z"]["mids"][v], "p_azul": p[:len(v)], "y": y[v]}).to_parquet(
                SALIDA / f"calib_v_{clave}.parquet", engine="pyarrow", index=False)
            graficar(clave, *resultados[clave]["V"])
            print(f"{TITULOS[clave]} · AUC {num(m['auc'])} / acierto {pct(m['acierto'])}"
                  f" · ECE {pct(m['ece'])} · pendiente {num(m['pendiente'])}", flush=True)
        for archivo, objeto in (("calib_mapa.json", {"metodo": "Platt", "epsilon_logit": EPS,
                                                    "bloque_calibracion": rango(t[cal]), **mapas}),
                               ("calib_columnas_sin_draft.json", esquema)):
            (SALIDA / archivo).write_text(json.dumps(objeto, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        fechas = {k: t[te[g]] for k, g in grupos_v.items()}
        informe(resultados, mapas, fechas, t[tr_cal], t[cal], t[tr], n9, comparacion, inicio)
        print(f"Informe guardado en {SALIDA} ({time.perf_counter() - inicio:.1f} s).", flush=True)


if __name__ == "__main__":
    main()
