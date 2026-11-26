"""Qué errores del emparejamiento de Riot explota el modelo final (escalón 9; PROJECT_SUMMARY.md §9.S.70).

El emparejamiento iguala el nivel medio de los equipos antes del draft. Lo que predice el modelo es, entonces,
lo que ese emparejamiento no mira. Este script lo separa por tipo y prueba tres hipótesis:

A. *Descomposición.* El logit de la logística es una suma, así que se parte por grupo de columnas (draft, nivel
   real del jugador, experiencia con el campeón, línea, actividad, estilo, modelo por línea). Se mide el peso de cada
   grupo en todas las partidas de V Master+ y en el 20 % más seguro, sin reajustar.
B. *Cola.* Con poca gente en cola el emparejamiento afloja. Se mide por cantidad de partidas simultáneas (±30 min)
   y por franja horaria (ART): dispersión del LP en el lobby, desbalance estimado y acierto.
C. *Lado.* Si el equipo de más nivel va al lado rojo y cuánto compensa la ventaja del azul.
D. *"Cola de perdedores".* Si al que viene de perder 3 seguidas le tocan peores compañeros.

    python historial/ideas_posteriores/brainstorm_44_errores_emparejamiento.py     # ~3 min
Salida: data/analysis/brainstorm/errores_*.csv
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.brainstorm_base import cargar, extras_referencia  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
OUT = REPO / "data" / "analysis" / "brainstorm"
D = cargar()
X9 = extras_referencia(D, escalon=9)
y, master, NM, t = D["y"], D["master"], D["NM"], D["t"]
cols = [str(c) for c in D["cols"]]

# ---------------------------------------------------------------- grupos de columnas
G = ["draft", "nivel real del jugador", "experiencia con el campeón", "línea (autofill)", "actividad",
     "estilo (tempo, economía)", "modelo por línea (mixto)"]


def grupo_var(c):
    if c.startswith(("tag_", "info_", "st_", "perf_")):
        return 0
    if c in ("logn_champ", "estreno", "wr_champ", "share_champ", "conc_rec", "conc_ant", "d_pool", "otp_nuevo"):
        return 2
    if c in ("share_lane", "offrole"):
        return 3
    if c in ("logn_tot", "log_horas_desde", "ritmo"):
        return 4
    return 1                                    # acum_*, v20_*, wr_acum, wr_v20


n_draft = D["DRAFT"].shape[1]
g_vars = np.tile([grupo_var(c) for c in cols], 5)              # Xd = 5 líneas × 85 variables
# extras en el orden de extras_referencia: MC 75, ML 75, nivel histórico 3, W/L foto 4, MATCHUPS 10, ROL_WR 10,
# KALMAN_OUT 12 + KALMAN_MRG 7, TEMPO_OBJ 10 + ECON_MID 4, HGB 6, margen esperado por par 12
g_ext = np.repeat([2, 3, 1, 1, 0, 3, 1, 5, 6, 0], [75, 75, 3, 4, 10, 10, 19, 14, 6, 12])
assert len(g_ext) == X9.shape[1], (len(g_ext), X9.shape[1])
grupo = np.r_[np.zeros(n_draft, int), g_vars, g_ext]

# ---------------------------------------------------------------- A. modelo final y descomposición
tr = D["fi_v"][master[D["fi_v"]]]
vi = np.flatnonzero(D["val"])
Xd = D["Xd"]
mu, sd = Xd[tr].mean(0), Xd[tr].std(0) + 1e-6
me, se = X9[tr].mean(0), X9[tr].std(0) + 1e-6
arm = lambda f: sparse.hstack([D["DRAFT"][f], sparse.csr_matrix((Xd[f] - mu) / sd),  # noqa: E731
                               sparse.csr_matrix((X9[f] - me) / se)]).tocsr()
lr = LogisticRegression(C=0.05, max_iter=4000).fit(arm(tr), y[tr])
B = arm(vi)
w = lr.coef_[0]
CON = np.column_stack([B[:, grupo == g] @ w[grupo == g] for g in range(len(G))])      # aporte de cada grupo al logit
logit = CON.sum(1) + lr.intercept_[0]
yv = y[vi]
p = 1 / (1 + np.exp(-logit))
ok = (logit > 0) == yv
print(f"Modelo final en V Master+ ({len(vi):,}): AUC {roc_auc_score(yv, logit):.4f} | acierto {ok.mean():.2%} | "
      f"azul gana {yv.mean():.2%} | p media {p.mean():.3f} | intercepto {lr.intercept_[0]:+.3f}")

seg = np.argsort(-np.abs(logit))[: int(0.2 * len(vi))]       # 20 % más seguro
es_seg = np.zeros(len(vi), bool)
es_seg[seg] = True
lc = logit - lr.intercept_[0]
sg = np.sign(lc)
fav = CON * sg[:, None]                                       # aporte a favor del favorito
dom = fav.argmax(1)
filas = []
for g, n in enumerate(G):
    filas.append({
        "grupo": n, "columnas": int((grupo == g).sum()),
        "% de la varianza del logit": np.cov(CON[:, g], lc)[0, 1] / lc.var(),
        "AUC solo": roc_auc_score(yv, CON[:, g]),
        "AUC sin él": roc_auc_score(yv, logit - CON[:, g]),
        "% del logit (todas)": fav[:, g].mean() / np.abs(lc).mean(),
        "% del logit (20 % seguro)": fav[es_seg, g].mean() / np.abs(lc[es_seg]).mean(),
        "domina (todas)": (dom == g).mean(), "domina (20 % seguro)": (dom[es_seg] == g).mean(),
        "acierto si domina": ok[dom == g].mean() if (dom == g).any() else np.nan,
    })
TA = pd.DataFrame(filas)
TA.to_csv(OUT / "errores_grupos.csv", index=False)
print("\nA. Qué explota el modelo, por tipo de error del emparejamiento (sin reajustar):")
m = TA.copy()
for c in m.columns[2:]:
    m[c] = m[c].map((lambda x: f"{x:.4f}") if c.startswith("AUC") else (lambda x: f"{x:.1%}"))
print(m.to_string(index=False))
# cuántos grupos empujan para el mismo lado en las seguras
print(f"   grupos a favor del favorito (de {len(G)}): todas {np.mean((fav > 0).sum(1)):.2f}, 20 % seguro {np.mean((fav[es_seg] > 0).sum(1)):.2f}")
print(f"   |logit| medio: todas {np.abs(lc).mean():.3f}, 20 % seguro {np.abs(lc[es_seg]).mean():.3f}; "
      f"acierto 20 % seguro {ok[es_seg].mean():.1%}")

# ---------------------------------------------------------------- B. cola: partidas simultáneas y hora
ts = np.sort(t)
V30 = 30 * 60 * 1000
simult = (np.searchsorted(ts, t + V30) - np.searchsorted(ts, t - V30))[vi]      # partidas del dataset en ±30 min
hora = ((t[vi] // 3_600_000) - 3) % 24                                           # ART = UTC−3
lp = D["B"]["lp_abs"][vi].reshape(len(vi), 10)
disp_lp = np.nanstd(lp, axis=1)
dif_lp = np.abs(np.nanmean(lp[:, :5], axis=1) - np.nanmean(lp[:, 5:], axis=1))


def tabla(clave, nombre):
    f = []
    for k, idx in pd.Series(np.arange(len(vi))).groupby(clave):
        i = idx.to_numpy()
        f.append({nombre: k, "partidas": len(i), "simultáneas (mediana)": np.median(simult[i]),
                  "disp. LP del lobby": np.nanmean(disp_lp[i]), "|ΔLP| entre equipos": np.nanmean(dif_lp[i]),
                  "|logit| medio": np.abs(lc[i]).mean(), "% en el 20 % seguro": es_seg[i].mean(),
                  "AUC": roc_auc_score(yv[i], logit[i]), "acierto": ok[i].mean()})
    return pd.DataFrame(f)


TB1 = tabla(pd.qcut(simult, 5, labels=["Q1 (menos gente)", "Q2", "Q3", "Q4", "Q5 (más gente)"]).astype(str), "simultáneas")
TB2 = tabla(pd.cut(hora, [-1, 5, 11, 17, 23], labels=["00-06", "06-12", "12-18", "18-24"]).astype(str), "franja (ART)")
TB1.to_csv(OUT / "errores_cola.csv", index=False)
TB2.to_csv(OUT / "errores_hora.csv", index=False)
print("\nB. Cola: partidas simultáneas (±30 min) y franja horaria, V Master+")
for T in (TB1, TB2):
    print(T.round(4).to_string(index=False))
rho = pd.Series(simult).corr(pd.Series(np.abs(lc)), method="spearman")
print(f"   Spearman(simultáneas, |logit|) = {rho:+.3f};  Spearman(simultáneas, disp. LP) = "
      f"{pd.Series(simult).corr(pd.Series(disp_lp), method='spearman'):+.3f}")

# ---------------------------------------------------------------- C. lado
A = np.load(OUT / "mmr_historico.npz")["A"]                 # (NM, 2, 5, 5): nivel histórico acumulado en [..., 0]
niv = (A[vi, 0, :, 0].mean(1) - A[vi, 1, :, 0].mean(1))     # azul − rojo
dlp = np.nanmean(lp[:, :5], axis=1) - np.nanmean(lp[:, 5:], axis=1)
c_niv = CON[:, 1]
print("\nC. Lado (V Master+)")
print(f"   azul gana {yv.mean():.2%}; el modelo elige azul {np.mean(logit > 0):.1%}; acierto eligiendo azul "
      f"{ok[logit > 0].mean():.1%}, eligiendo rojo {ok[logit <= 0].mean():.1%}")
for nombre, v in (("nivel histórico (azul − rojo)", niv), ("LP medio (azul − rojo)", dlp), ("aporte 'nivel real' al logit", c_niv),
                  ("aporte 'draft' al logit", CON[:, 0])):
    v = np.asarray(v, float)
    b = np.isfinite(v)
    print(f"   {nombre:32s} media {v[b].mean():+.4f} (EE {v[b].std() / np.sqrt(b.sum()):.4f}); rojo mejor en {np.mean(v[b] < 0):.1%}; "
          f"azul gana si azul mejor {yv[b][v[b] > 0].mean():.1%}, si rojo mejor {yv[b][v[b] < 0].mean():.1%}")
tl = D["tier_lobby"][vi]
for nombre, msk in (("Master (tier medio < GM)", tl < np.nanquantile(tl, 0.8)), ("20 % de lobbies más altos", tl >= np.nanquantile(tl, 0.8))):
    print(f"   {nombre:28s} n {msk.sum():,}: azul gana {yv[msk].mean():.2%}, rojo mejor en nivel {np.mean(niv[msk] < 0):.1%}, "
          f"AUC {roc_auc_score(yv[msk], logit[msk]):.4f}")

# ---------------------------------------------------------------- D. cola de perdedores
pl = D["B"]["player"].reshape(-1)
df = pd.DataFrame({"m": np.repeat(np.arange(NM), 10), "eq": np.tile(np.repeat([0, 1], 5), NM), "pl": pl,
                   "t": np.repeat(t, 10)})
df["gana"] = np.where(df.eq == 0, np.repeat(y, 10), 1 - np.repeat(y, 10))
iwr = cols.index("wr_acum")
medidas = {"nivel histórico": A[..., 0], "LP": D["B"]["lp_abs"], "winrate acumulado": D["Xj"][..., iwr],
           "margen de oro @15 acumulado": D["Xj"][..., cols.index("acum_d_gold_15")]}
for n, M in medidas.items():
    M = M.astype(float)
    s_eq = np.nansum(M, axis=2, keepdims=True)
    n_eq = np.isfinite(M).sum(axis=2, keepdims=True)
    comp = (s_eq - np.nan_to_num(M)) / np.maximum(n_eq - np.isfinite(M), 1)        # media de los 4 compañeros
    riv = (s_eq / np.maximum(n_eq, 1))[:, ::-1]                                    # media de los 5 rivales
    df[n] = (comp - riv).reshape(-1)
df = df.sort_values(["pl", "t"], kind="mergesort")
gb = df.groupby("pl").gana
r1, r2, r3 = gb.shift(1), gb.shift(2), gb.shift(3)
h = (df.t - df.groupby("pl").t.shift(3)) / 3_600_000
df["racha"] = np.select([(r1 + r2 + r3 == 0), (r1 + r2 + r3 == 3)], ["perdió las 3 anteriores", "ganó las 3 anteriores"], "mezcla")
df.loc[r3.isna(), "racha"] = "sin 3 previas"
df["misma_sesion"] = h <= 6                                                         # las 3 previas en las últimas 6 h
uso = df[np.repeat(D["post"] & master, 10)[df.index] & (df.racha != "sin 3 previas")]
print("\nD. 'Cola de perdedores': compañeros − rivales según la racha previa del jugador (Master+, desde la foto 1)")
filas = []
for etiqueta, sub in (("todas", uso), ("3 previas en las últimas 6 h", uso[uso.misma_sesion])):
    base = sub[sub.racha == "mezcla"]
    for r in ("perdió las 3 anteriores", "ganó las 3 anteriores", "mezcla"):
        s = sub[sub.racha == r]
        f = {"muestra": etiqueta, "racha": r, "filas": len(s), "gana esta partida": s.gana.mean()}
        for n in medidas:
            d = s[n].mean() - base[n].mean()
            ee = np.sqrt(s[n].var() / s[n].notna().sum() + base[n].var() / base[n].notna().sum())
            f[f"{n}: Δ vs mezcla"] = d
            f[f"{n}: z"] = d / ee if r != "mezcla" else 0.0
            f[f"{n}: sd"] = sub[n].std()
        filas.append(f)
TD = pd.DataFrame(filas)
TD.to_csv(OUT / "errores_racha.csv", index=False)
print(TD.round(4).T.to_string())
print(f"\n[+] {OUT}/errores_*.csv")
