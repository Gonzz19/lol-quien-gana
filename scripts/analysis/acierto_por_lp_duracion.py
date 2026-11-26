"""
Accuracy del modelo previo según el LP medio del lobby y según la duración de la partida (pedido del usuario,
2026-11-17).

* Predicciones del híbrido + mapa (`--modelo hibrido`) o del escalón 9 (`--modelo escalon9`, necesita sus cachés),
  entrenado solo con Master+ anterior al 16/09, sobre V Master+ y Diamond-V (a ciegas).
* Curvas suavizadas con ventana deslizante: LP ±50 cada 25 LP; duración ±1 min cada minuto. Solo se dibujan los
  puntos con ≥ 300 partidas en la ventana. Debajo, la cantidad de partidas (sin doble eje).
* LP medio del lobby: mapa de LP causal en escala absoluta (Diamond IV 0 = 0, Master 0 = 400); cortes de la foto 2
  (28/10): Grandmaster 1.132, Challenger 1.660.
* La duración se conoce recién al final: es descriptiva, el modelo no la usa.

    python scripts/analysis/acierto_por_lp_duracion.py      # ~3 min
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.brainstorm_base import ajustar_predecir, cargar, extras_referencia  # noqa: E402

AZUL, NARANJA, INK, INK_2, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#a3a29c", "#e6e5e0"
SALIDA = REPO / "data" / "analysis" / "figuras"
MIN_N = 300


def curva(x, ok, centros, ancho):
    filas = []
    for c in centros:
        m = np.abs(x - c) <= ancho
        if m.sum() >= MIN_N:
            filas.append({"x": c, "acc": 100 * ok[m].mean(), "n": int(m.sum())})
    return pd.DataFrame(filas)


def wilson_ci(p, n, z=1.959963984540054):
    p = np.asarray(p, dtype=float)
    n = np.asarray(n, dtype=float)
    denom = 1.0 + z * z / n
    centro = (p + z * z / (2.0 * n)) / denom
    semi = z * np.sqrt(np.maximum(p * (1.0 - p) / n + z * z / (4.0 * n * n), 0.0)) / denom
    return np.clip(centro - semi, 0.0, 1.0), np.clip(centro + semi, 0.0, 1.0)


def panel(ax_a, ax_n, curvas, xlabel, marcas=()):
    for nombre, (df, color) in curvas.items():
        if df.empty:
            continue
        df = df.sort_values("x")
        lo, hi = wilson_ci(df["acc"].to_numpy() / 100.0, df["n"].to_numpy())
        ax_a.plot(df.x, df.acc, color=color, lw=2, label=nombre if len(curvas) > 1 else None)
        ax_a.fill_between(df.x, 100.0 * lo, 100.0 * hi, color=color, alpha=0.16)
        ax_n.plot(df.x, df.n, color=color, lw=1.5)
    x_max = max(df.x.max() for df, _ in curvas.values() if not df.empty)
    afuera = [txt for xv, txt in marcas if xv > x_max + 50]
    marcas = [(xv, txt) for xv, txt in marcas if xv <= x_max + 50]
    if afuera:
        ax_n.set_xlim(right=x_max + 50)
        ax_a.text(0.99, 0.02, "sin lobbies con promedio de " + ", ".join(afuera) + " (menos de 300 partidas)",
                  transform=ax_a.transAxes, ha="right", va="bottom", fontsize=8, color=INK_2)
    for xv, txt in marcas:
        for a in (ax_a, ax_n):
            a.axvline(xv, color=MUTED, lw=0.8, ls=":")
        ax_a.text(xv, 1.01, txt, transform=ax_a.get_xaxis_transform(), ha="center", va="bottom", fontsize=8,
                  color=INK_2)
    ax_a.set_ylabel("Accuracy (%) · bandas: IC 95 % Wilson", color=INK_2)
    ax_n.set_ylabel("Partidas en la ventana", color=INK_2)
    ax_n.set_xlabel(xlabel, color=INK_2)
    ax_a.set_xlabel("")
    for a in (ax_a, ax_n):
        a.grid(True, color=GRID, lw=0.6)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.tick_params(colors=INK_2, labelsize=8)
    if len(curvas) > 1:
        ax_a.legend(frameon=False, fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", choices=["hibrido", "escalon9"], default="hibrido")
    args = ap.parse_args()
    D = cargar()
    NM, y = D["NM"], D["y"]
    te = np.flatnonzero(D["val"] | D["diam_v"])
    tr = D["fi_v"][D["master"][D["fi_v"]]]
    if args.modelo == "escalon9":
        p_te = ajustar_predecir(D, tr, te, extra=extras_referencia(D, escalon=9), extra_post=False, con_mapa=False)
    else:
        p_te = ajustar_predecir(D, tr, te)
    p = np.full(NM, np.nan)
    p[te] = p_te
    ok = (p > 0.5) == y
    lp = np.nanmean(D["B"]["lp_abs"].reshape(NM, 10), 1)
    dur = D["Z"]["D_dur_min"]
    sns.set_theme(style="white")
    SALIDA.mkdir(parents=True, exist_ok=True)
    nombre_modelo = "híbrido + mapa" if args.modelo == "hibrido" else "escalón 9"

    # 1) LP medio del lobby (V Master+ y Diamond-V juntos)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    c_lp = curva(lp[te], ok[te], np.arange(0, 2201, 25), 50)
    c_lp_m = curva(lp[D["val"]], ok[D["val"]], np.arange(0, 2201, 25), 50)
    c_lp_d = curva(lp[D["diam_v"]], ok[D["diam_v"]], np.arange(0, 2201, 25), 50)
    panel(a1, a2, {"V Master+": (c_lp_m, AZUL), "Diamond-V (a ciegas)": (c_lp_d, NARANJA)},
          "LP medio del lobby (escala absoluta)",
          marcas=((400, "Master"), (1132, "GM (foto 2)"), (1660, "Challenger")))
    fig.suptitle(f"Accuracy según el LP medio del lobby ({nombre_modelo}, ventana ±50 LP)", x=0.06, ha="left",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(SALIDA / f"acierto_por_lp_{args.modelo}.png", dpi=150)
    plt.close(fig)

    # 2) duración, V Master+ y Diamond-V por separado
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    centros = np.arange(15, 51, 1)
    curvas = {"V Master+": (curva(dur[D["val"]], ok[D["val"]], centros, 1), AZUL),
              "Diamond-V": (curva(dur[D["diam_v"]], ok[D["diam_v"]], centros, 1), NARANJA)}
    panel(a1, a2, curvas, "Duración de la partida (minutos)")
    fig.suptitle(f"Accuracy según la duración de la partida ({nombre_modelo}, ventana ±1 min)", x=0.06, ha="left",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(SALIDA / f"acierto_por_duracion_{args.modelo}.png", dpi=150)
    plt.close(fig)
    pd.concat([c_lp.assign(eje="lp"), c_lp_m.assign(eje="lp_master"), c_lp_d.assign(eje="lp_diamond"),
               curvas["V Master+"][0].assign(eje="dur_master"),
                          curvas["Diamond-V"][0].assign(eje="dur_diamond")]).to_csv(
        SALIDA / f"acierto_lp_duracion_{args.modelo}.csv", index=False)
    print(f"[+] {SALIDA}")


if __name__ == "__main__":
    main()
