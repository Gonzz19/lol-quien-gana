"""Qué experto del stacking aporta y cuál mete ruido (§9.S.33). Pedido del usuario, 2026-11-13.

Entrada: los mismos <nombre>_oos.npz que ganador_stacking.py. Cuatro miradas, de menos a más
dependientes de V (la validación de siempre):

1. Correlación entre los logits de los expertos (fuera de muestra): cuánto se repiten.
2. Meta evaluado bloque a bloque SIN V: el meta se ajusta con los bloques < k y predice el
   bloque k (k = 2..4, ventana que crece). Se reporta el log loss medio de esos bloques.
3. Sacar un experto a la vez ("leave-one-out"): cuánto empeora el meta sin él, en los bloques
   (mirada 2) y en V (diferencia pareada de AUC, IC95 con 200 remuestreos).
4. Estabilidad de los pesos: el meta ajustado en cada bloque por separado y en 200
   remuestreos del fuera de muestra; un peso que cambia de signo es ruido.
Además, meta con pesos >= 0 (minimiza el log loss con cotas) y selección hacia adelante con el
criterio de la mirada 2 (sin V), evaluada después en V.
Solo data/splits/train."""
import argparse, warnings
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
AN = REPO / "data" / "analysis"
ap = argparse.ArgumentParser()
ap.add_argument("--expertos", nargs="+", default=[
    f"margen={AN / 'ganador_red_jugador_margen_oos.npz'}", f"Poisson={AN / 'ganador_red_jugador_poisson_oos.npz'}",
    f"grupos={AN / 'ganador_red_jugador_grupos_oos.npz'}", f"lobby={AN / 'ganador_logistica_lobby_oos.npz'}",
    f"lobby+P={AN / 'ganador_logistica_lobby_poisson_oos.npz'}"])
ap.add_argument("--config", default="C")
args = ap.parse_args()
cfg = args.config

lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
sigm = lambda z: 1 / (1 + np.exp(-z))
Zo, Zv, ref = {}, {}, None
for item in args.expertos:
    et, ruta = item.split("=", 1)
    f = np.load(ruta)
    if ref is None:
        ref = {k: f[f"{cfg}_{k}"] for k in ("oos_idx", "oos_bloque", "oos_y", "val_y")}
    assert (f[f"{cfg}_oos_idx"] == ref["oos_idx"]).all()
    for k in ("logistica", "red_equipo", "red_escalar"):
        if f"{cfg}_oos_{k}" in f.files:
            n = f"{'log' if k == 'logistica' else 'red'} {et}"
            Zo[n], Zv[n] = lg(f[f"{cfg}_oos_{k}"]), lg(f[f"{cfg}_val_{k}"])
yo, yv, bl = ref["oos_y"], ref["val_y"], ref["oos_bloque"]
N = list(Zo)
O = np.c_[[Zo[n] for n in N]].T
V = np.c_[[Zv[n] for n in N]].T
B = np.random.default_rng(0).integers(0, len(yv), (200, len(yv)))
print(f"expertos: {N}\nfuera de muestra: bloques {sorted(set(bl))}, {len(yo)} partidas; V: {len(yv)}\n")


def ajustar(cols, sel, positivos=False):
    """Meta-modelo sobre los logits de `cols`; devuelve (pesos, constante)."""
    X, yy = O[sel][:, cols], yo[sel]
    if not positivos:
        m = LogisticRegression(C=1e4, max_iter=5000).fit(X, yy)
        return m.coef_[0], m.intercept_[0]
    def nll(w):
        z = X @ w[:-1] + w[-1]
        return np.mean(np.logaddexp(0, z) - yy * z)
    def grad(w):
        r = sigm(X @ w[:-1] + w[-1]) - yy
        return np.r_[X.T @ r, r.sum()] / len(yy)
    w0 = np.r_[np.full(len(cols), 1 / len(cols)), 0]
    w = minimize(nll, w0, jac=grad, method="L-BFGS-B", bounds=[(0, None)] * len(cols) + [(None, None)]).x
    return w[:-1], w[-1]


def bloques(cols, positivos=False):
    """Log loss y AUC medios del meta ajustado con los bloques < k, en el bloque k (k >= 2). Sin V."""
    ll, auc = [], []
    for k in sorted(set(bl))[1:]:
        w, c = ajustar(cols, bl < k, positivos)
        p = sigm(O[bl == k][:, cols] @ w + c)
        ll.append(log_loss(yo[bl == k], p)); auc.append(roc_auc_score(yo[bl == k], p))
    return np.mean(ll), np.mean(auc)


def en_v(cols, positivos=False):
    w, c = ajustar(cols, np.ones(len(yo), bool), positivos)
    return sigm(V[:, cols] @ w + c), w


def dif(p, q):
    d = np.array([roc_auc_score(yv[b], p[b]) - roc_auc_score(yv[b], q[b]) for b in B])
    return f"{roc_auc_score(yv, p) - roc_auc_score(yv, q):+.4f} [{np.quantile(d, .025):+.4f};{np.quantile(d, .975):+.4f}]"


# 1. correlación
print("=== 1. Correlación entre logits fuera de muestra ===")
C = np.corrcoef(O.T)
print(" " * 14 + "".join(f"{n[:11]:>12s}" for n in N))
for i, n in enumerate(N):
    print(f"{n:14s}" + "".join(f"{C[i, j]:>12.3f}" for j in range(len(N))))

# cada experto solo
print("\n=== Cada experto solo (calibrado con el meta de 1 entrada) ===")
print(f"{'':14s}{'log loss bloques':>18s}{'AUC bloques':>13s}{'AUC V':>8s}")
for i, n in enumerate(N):
    ll, a = bloques([i])
    print(f"{n:14s}{ll:>18.5f}{a:>13.4f}{roc_auc_score(yv, Zv[n]):>8.4f}")

todos = list(range(len(N)))
p_all, w_all = en_v(todos)
ll_all, a_all = bloques(todos)
p_pos, w_pos = en_v(todos, True)
ll_pos, a_pos = bloques(todos, True)
print(f"\n=== Todos ({len(N)}) ===")
print(f"  libre     bloques log loss {ll_all:.5f} AUC {a_all:.4f} | V AUC {roc_auc_score(yv, p_all):.4f} acc {accuracy_score(yv, p_all > .5):.4f}"
      f"  pesos {' / '.join(f'{w:.2f}' for w in w_all)}")
print(f"  pesos>=0  bloques log loss {ll_pos:.5f} AUC {a_pos:.4f} | V AUC {roc_auc_score(yv, p_pos):.4f} acc {accuracy_score(yv, p_pos > .5):.4f}"
      f"  pesos {' / '.join(f'{w:.2f}' for w in w_pos)}  vs libre {dif(p_pos, p_all)}")

# 3. sacar uno
print("\n=== 3. Sacar un experto (meta libre). Positivo = empeora sin él = aporta ===")
print(f"{'sin':14s}{'Δ log loss bloques (×1e4)':>27s}{'Δ AUC bloques':>15s}   Δ AUC en V al sacarlo (IC95)")
for i, n in enumerate(N):
    cols = [j for j in todos if j != i]
    ll, a = bloques(cols)
    p, _ = en_v(cols)
    print(f"{n:14s}{(ll - ll_all) * 1e4:>27.1f}{a_all - a:>+15.4f}   {dif(p_all, p)}")

# 4. estabilidad de pesos
print("\n=== 4. Pesos del meta libre por bloque y en 200 remuestreos ===")
print(f"{'':14s}" + "".join(f"{'bloque ' + str(k):>10s}" for k in sorted(set(bl))) + f"{'IC95 remuestreo':>22s}{'signo estable':>15s}")
porb = np.array([ajustar(todos, bl == k)[0] for k in sorted(set(bl))])
rng = np.random.default_rng(1)
bw = np.array([ajustar(todos, rng.integers(0, len(yo), len(yo)))[0] for _ in range(200)])
for i, n in enumerate(N):
    lo, hi = np.quantile(bw[:, i], [.025, .975])
    print(f"{n:14s}" + "".join(f"{w:>10.2f}" for w in porb[:, i]) + f"{f'[{lo:+.2f};{hi:+.2f}]':>22s}"
          f"{('sí' if lo > 0 or hi < 0 else 'no'):>15s}")

# selección hacia adelante con el criterio de bloques (sin V)
print("\n=== Selección hacia adelante por log loss en los bloques (sin V), meta libre y con pesos >= 0 ===")
for positivos in (False, True):
    elegidos, mejor = [], np.inf
    while True:
        cand = [(bloques(elegidos + [j], positivos)[0], j) for j in todos if j not in elegidos]
        if not cand:
            break
        ll, j = min(cand)
        if ll > mejor - 1e-5:
            break
        elegidos.append(j); mejor = ll
        p, w = en_v(elegidos, positivos)
        print(f"  {'pesos>=0' if positivos else 'libre':8s} + {N[j]:12s} bloques log loss {ll:.5f} | V AUC {roc_auc_score(yv, p):.4f}"
              f" acc {accuracy_score(yv, p > .5):.4f}  vs todos {dif(p, p_all)}  pesos {' / '.join(f'{x:.2f}' for x in w)}")
