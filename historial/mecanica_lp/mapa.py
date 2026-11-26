"""Recorrida hacia adelante vectorial con todas las mecánicas: F_i(z0) sobre una grilla de arranques.

Una sola función para las pruebas: se le pasan arrays ya armados, sin nada de pandas adentro, y por
eso corre en paralelo sin mandar los datos en cada tarea.
"""
import numpy as np

from mec import BANDAS, DIA, K

Z0 = np.arange(-600.0, 3605.0, 25.0)
KD = 18.75


def prob_doble(P, pos, frac):
    """P(doble | victoria), un valor por partida."""
    bins = np.digitize(frac, BANDAS[1:-1], right=True)
    out = np.full(len(pos), P["doble"], dtype=float)
    for r, tabla in P["td"].items():
        m = pos == r
        if m.any():
            out[m] = np.nan_to_num(tabla[bins[m]], nan=P["doble"])
    return out


def _decae(z, dias):
    """Aplica `dias` enteros de decay a cada elemento: 75 en Master+ hasta 375; 50 en Diamond hasta 275."""
    d = int(np.floor(dias + 1e-9))
    if d <= 0:
        return z
    z = z.copy()
    # Master+: cae hasta 375
    m = z >= 400
    usados = np.zeros_like(z)
    if m.any():
        pasos = np.minimum(np.ceil((z[m] - 375.0) / 75.0), d)
        z[m] = np.maximum(z[m] - 75.0 * pasos, 375.0)
        usados[m] = pasos
    # Lo que queda de días, en Diamond: hasta 275
    r = (z < 400) & (z > 275) & (usados < d)
    if r.any():
        pasos = np.minimum(np.ceil((z[r] - 275.0) / 50.0), d - usados[r])
        z[r] = np.maximum(z[r] - 50.0 * pasos, 275.0)
    return z


def adelante_v(win, tt, M, pdob, P, t1, t2, z0=None, tasa_oc=0.0):
    """F(z0): el LP en t2 para cada arranque, con banco de días por elemento de la grilla."""
    z = (Z0 if z0 is None else np.atleast_1d(np.asarray(z0, float))).copy()
    b = np.where(z >= 400, 14.0, 28.0)
    t_prev = t1
    gx, dy, q, pq, pb = P["gx"], P["dy"], P["perdon"], P["piso_quieto"], P["piso_baja"]
    for k in range(len(win) + 1):
        tk = tt[k] if k < len(win) else t2
        dias = (tk - t_prev) / DIA
        t_prev = tk
        b = b - dias
        falta = b < 0
        if falta.any():
            for d in np.unique(np.floor(-b[falta] + 1e-9)):
                i = falta & (np.floor(-b + 1e-9) == d)
                z[i] = _decae(z[i], d)
            b[falta] = 0.0
        if k == len(win):
            break
        K_ = np.where(z >= 400, K, KD)
        d = np.interp(M[k] - z, gx, dy) if M[k] == M[k] else 0.0
        if win[k]:
            z = z + (K_ + d) * (1 + pdob[k])
        else:
            dest = z - (K_ - d) * (1 - q)
            dest = np.where((z > 400.5) & (dest < 400), 400.0, dest)
            z = np.where((z >= 400) & (z <= 400.5), z - (1 - pq) * pb, dest)
        b = np.where(z >= 400, np.minimum(b + 1.0 + tasa_oc, 14.0), np.minimum(b + 7.0 * (1 + tasa_oc), 28.0))
    return z


def invertir(F, zf):
    """z0 con F(z0) ≈ zf, pendiente en la solución y ancho de arranques que caen a ±40 LP de zf."""
    F = np.maximum.accumulate(F)
    i = int(np.searchsorted(F, zf))
    if i <= 0:
        return Z0[0], 0.0, 0.0
    if i >= len(Z0):
        return Z0[-1], 0.0, 0.0
    w = (zf - F[i - 1]) / max(F[i] - F[i - 1], 1e-6)
    z = Z0[i - 1] + w * (Z0[i] - Z0[i - 1])
    pend = (F[i] - F[i - 1]) / (Z0[i] - Z0[i - 1])
    a = Z0[min(int(np.searchsorted(F, zf + 40)), len(Z0) - 1)] - Z0[min(int(np.searchsorted(F, zf - 40)), len(Z0) - 1)]
    return z, pend, a
