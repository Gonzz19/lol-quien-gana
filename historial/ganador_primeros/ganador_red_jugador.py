"""Red con codificador compartido por jugador (pendiente de §9.S.23).

Cada uno de los 10 jugadores pasa por la MISMA red phi:
    entrada = [embedding del campeón, embedding de la línea, experiencia, cobertura,
               dificultad, huella (+ indicador de huella)]
Dos variantes de salida, las dos antisimétricas por construcción (invertir los lados
invierte el logit, así que el espejado sale gratis):
    escalar: phi da un número por jugador; logit = sum(azul) - sum(rojo) + lado + lineal(mt, sy)
    equipo:  phi da un vector; h_T = suma del equipo; logit = g(h_A, h_R) - g(h_R, h_A) + ...
Matchups y sinergias (ya azul - rojo) entran lineales. Mismas configs A/B/C y misma
validación que ganador_red.py; la logística sobre las variables agregadas va de referencia,
con el AUC pareado por bootstrap contra ella.
Necesita ganador_feats_grande.pkl con la clave "jug" (ganador_features_grande.py actual).

--poisson: suma el modelo de ataque y defensa de ganador_poisson.py (§9.S.25): a phi, el
ataque y la defensa de cada jugador en cada medida; a la parte lineal y a la logística, las
diferencias azul - rojo por partida.
--oos K: además de predecir V, parte el ajuste en K bloques por fecha y predice cada bloque
k >= 1 con modelos ajustados SOLO con los bloques anteriores (ventana que crece). Esas
predicciones fuera de muestra son la entrada de ganador_stacking.py (§9.S.26)."""
import argparse, pickle, time, warnings
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--feats", default=str(REPO / "data" / "analysis" / "ganador_feats_grande.pkl"))
ap.add_argument("--configs", default="ABC", help="subconjunto de A, B, C")
ap.add_argument("--variantes", default="escalar,equipo")
ap.add_argument("--semillas", type=int, default=3)
ap.add_argument("--oculta", type=int, default=64)
ap.add_argument("--dropout", type=float, default=0.3)
ap.add_argument("--margen", help="salida de ganador_margen_duos.py: suma el margen previo por jugador y los duos (§9.S.24)")
ap.add_argument("--poisson", help="salida de ganador_poisson.py: ataque y defensa por jugador y por partida (§9.S.25)")
ap.add_argument("--grupos", help="salida de ganador_margen_grupos.py: margen por campeón, por línea y últimas 5 (§9.S.29)")
ap.add_argument("--lobby", help="salida de ganador_mmr.py: suma el nivel del lobby con la foto (§9.S.31, §9.S.32)")
ap.add_argument("--composicion", help="salida de ganador_composicion.py: atributos del campeón por jugador (entran en phi) "
                "y composición del equipo por partida (§9.S.37)")
ap.add_argument("--rivales", type=int, default=0,
                help="d > 0: término de interacción entre rivales de línea (§9.S.36): de phi salen una "
                     "fortaleza f y una debilidad w de dimensión d por jugador, y el logit suma "
                     "sum_lineas <f_azul, w_rojo> - <f_rojo, w_azul> (antisimétrico)")
ap.add_argument("--oos", type=int, default=0, help="K bloques temporales del ajuste; predice cada uno con los anteriores")
ap.add_argument("--maestro", help="salida de ganador_maestro.py: entrena la red con la etiqueta suave "
                                  "q = alfa*y + (1-alfa)*p_maestro (destilación con los snapshots @10/@15, §9.S.39); solo config C")
ap.add_argument("--alfa", type=float, default=0.5, help="peso de la etiqueta real con --maestro")
ap.add_argument("--sin-huella", action="store_true", help="no usar la huella en ninguna config")
ap.add_argument("--guardar", help="npz con las predicciones en V (promedio de semillas), para ganador_por_grupos.py; "
                                  "con --oos escribe además <nombre>_oos.npz para ganador_stacking.py")
args = ap.parse_args()
DEV = "cuda" if torch.cuda.is_available() else "cpu"

d = pickle.load(open(args.feats, "rb"))
if "jug" not in d:
    raise SystemExit("el pickle no tiene 'jug': correr de nuevo ganador_features_grande.py")
feat, jug, y, t, val, masks = d["feat"], d["jug"], d["y"], d["t"], d["val"], d["masks"]
COMUN = ["onehot", "experiencia", "cobertura", "dificultad", "matchups", "sinergias"]
HUELLA = {"A": "huella_foto", "B": "huella_40", "C": None}
if args.sin_huella:
    HUELLA = dict.fromkeys(HUELLA)

MARGEN = None
if args.margen:
    MARGEN = pickle.load(open(args.margen, "rb"))
    assert (MARGEN["mids"] == d["mids"]).all(), "el pickle de margen no corresponde a estas partidas"
    AGREGADO = np.hstack([MARGEN["nuevo"][k] for k in ("winrate previo", "margen acum", "margen v20", "duos")])
POISSON = None
if args.poisson:
    POISSON = pickle.load(open(args.poisson, "rb"))
    assert (POISSON["mids"] == d["mids"]).all(), "el pickle de Poisson no corresponde a estas partidas"
    if "jug" not in POISSON:
        raise SystemExit("el pickle de Poisson no tiene 'jug': correr de nuevo ganador_poisson.py")
    POI_PARTIDA = np.hstack(list(POISSON["poisson"].values()))
    POI_JUG = np.concatenate(list(POISSON["jug"].values()), axis=2)
GRUPOS = None
if args.grupos:
    GRUPOS = pickle.load(open(args.grupos, "rb"))
    assert (GRUPOS["mids"] == d["mids"]).all(), "el pickle de grupos no corresponde a estas partidas"
LOBBY = None
if args.lobby:
    LOBBY = pickle.load(open(args.lobby, "rb"))
    assert (LOBBY["mids"] == d["mids"]).all(), "el pickle de ganador_mmr.py no corresponde a estas partidas"
    LOBBY_JUG = LOBBY["jug"][:, :, :2]          # nivel_acum y nivel_v10 de la foto: las 2 primeras columnas
COMP = None
if args.composicion:
    COMP = pickle.load(open(args.composicion, "rb"))
    assert (COMP["mids"] == d["mids"]).all(), "el pickle de composición no corresponde a estas partidas"
yv = y[val]
guardadas, oos = {}, {}
B = np.random.default_rng(0).integers(0, val.sum(), (200, val.sum()))


def informe(nombre, p, n_fit, seg, ref=None):
    aucs = np.array([roc_auc_score(yv[b], p[b]) for b in B])
    lo, hi = np.quantile(aucs, [0.025, 0.975])
    extra = ""
    if ref is not None:
        dif = aucs - np.array([roc_auc_score(yv[b], ref[b]) for b in B])
        dlo, dhi = np.quantile(dif, [0.025, 0.975])
        extra = f"  vs logística {roc_auc_score(yv, p) - roc_auc_score(yv, ref):+.4f} [{dlo:+.4f};{dhi:+.4f}]"
    print(f"{nombre:36s} ajuste {n_fit:>7d}  acc {accuracy_score(yv, p > .5):.3f}  AUC {roc_auc_score(yv, p):.3f} "
          f"[{lo:.3f}-{hi:.3f}]  logloss {log_loss(yv, p):.4f}  ({seg:.0f}s){extra}", flush=True)


class RedJugador(nn.Module):
    def __init__(self, n_camp, d_num, d_partida, variante, oculta, drop, rivales=0):
        super().__init__()
        self.variante = variante
        self.rivales = rivales
        if rivales:
            # tipo NeuralAC / Blade-Chest: fortaleza y debilidad de cada jugador, cruzadas con el rival de línea
            self.fuerza = nn.Linear(oculta, rivales, bias=False)
            self.debil = nn.Linear(oculta, rivales, bias=False)
        self.camp = nn.Embedding(n_camp, 16)
        self.linea = nn.Embedding(5, 4)
        self.phi = nn.Sequential(nn.Linear(16 + 4 + d_num, oculta), nn.ReLU(), nn.Dropout(drop),
                                 nn.Linear(oculta, oculta), nn.ReLU(), nn.Dropout(drop))
        if variante == "escalar":
            self.salida = nn.Linear(oculta, 1, bias=False)
        else:
            self.g = nn.Sequential(nn.Linear(2 * oculta, oculta), nn.ReLU(), nn.Dropout(drop), nn.Linear(oculta, 1))
        self.partida = nn.Linear(d_partida, 1, bias=False)   # matchups y sinergias, ya azul - rojo
        self.lado = nn.Parameter(torch.zeros(1))
        self.register_buffer("lineas", torch.arange(10) % 5)

    def forward(self, camp, num, part):
        x = torch.cat([self.camp(camp), self.linea(self.lineas).expand(len(camp), -1, -1), num], dim=2)
        h = self.phi(x)                                     # (partidas, 10, oculta)
        if self.variante == "escalar":
            s = self.salida(h).squeeze(2)
            z = s[:, :5].sum(1) - s[:, 5:].sum(1)
        else:
            ha, hr = h[:, :5].sum(1), h[:, 5:].sum(1)
            z = (self.g(torch.cat([ha, hr], 1)) - self.g(torch.cat([hr, ha], 1))).squeeze(1)
        if self.rivales:
            f, w = self.fuerza(h), self.debil(h)                # (partidas, 10, d); puesto = 5 * rojo + línea
            z = z + ((f[:, :5] * w[:, 5:]).sum(2) - (f[:, 5:] * w[:, :5]).sum(2)).sum(1)
        return z + self.lado + self.partida(part).squeeze(1)


def entrenar(variante, Cm, Nm, Pm, tr, es, seed, q=None):
    """q: etiqueta para la pérdida (largo n; por defecto y). La parada temprana usa siempre y."""
    torch.manual_seed(seed)
    net = RedJugador(jug["n_campeones"], Nm.shape[2], Pm.shape[1], variante, args.oculta, args.dropout,
                     args.rivales).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    yt = torch.tensor(y if q is None else q, dtype=torch.float32, device=DEV)
    tr_t = torch.tensor(tr, device=DEV)
    best, best_state, paciencia = 1e9, None, 0
    for ep in range(100):
        net.train()
        for b in tr_t[torch.randperm(len(tr_t), device=DEV)].split(512):
            opt.zero_grad()
            nn.functional.binary_cross_entropy_with_logits(net(Cm[b], Nm[b], Pm[b]), yt[b]).backward()
            opt.step()
        ll = log_loss(y[es], predecir(net, Cm, Nm, Pm, es))
        if ll < best:
            best, best_state, paciencia = ll, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            paciencia += 1
            if paciencia >= 8:
                break
    net.load_state_dict(best_state)
    return net, ep + 1


MAESTRO = np.load(args.maestro) if args.maestro else None


def suave(tarea, idx_fit):
    """etiqueta suave para el ajuste de `tarea` ('V' o el bloque k); None sin --maestro"""
    if MAESTRO is None:
        return None
    assert len(MAESTRO["idx"]) == len(idx_fit) and (MAESTRO["idx"] == idx_fit).all(), \
        "el maestro es de otras partidas: --maestro solo sirve con la config C"
    q = y.astype(float).copy()
    pm = MAESTRO[f"p_{tarea}"]
    ok = ~np.isnan(pm)
    q[MAESTRO["idx"][ok]] = args.alfa * y[MAESTRO["idx"][ok]] + (1 - args.alfa) * pm[ok]
    return q


@torch.no_grad()
def predecir(net, Cm, Nm, Pm, idx):
    net.eval()
    i = torch.tensor(idx, device=DEV)
    return torch.cat([torch.sigmoid(net(Cm[b], Nm[b], Pm[b])) for b in i.split(8192)]).cpu().numpy()


def logistica(X, fit_idx, pred_idx, tambien_ajuste=False):
    """Con `tambien_ajuste`, devuelve además la predicción dentro de la muestra sobre `fit_idx` (Train)."""
    sc = StandardScaler(with_mean=False).fit(X[fit_idx])
    m = LogisticRegression(C=0.05, max_iter=5000).fit(sc.transform(X[fit_idx]), y[fit_idx])
    p = m.predict_proba(sc.transform(X[pred_idx]))[:, 1]
    return (p, m.predict_proba(sc.transform(X[fit_idx]))[:, 1]) if tambien_ajuste else p


def tensores(num, part, idx):
    """Parte `idx` (ajuste) en entrenamiento y parada temprana (el 15 % más nuevo) y escala con el entrenamiento."""
    corte = np.quantile(t[idx], 0.85)
    tr, es = idx[t[idx] < corte], idx[t[idx] >= corte]
    plano = num[tr].reshape(-1, num.shape[2])
    mu, sd = plano.mean(0), plano.std(0) + 1e-6
    Nm = torch.tensor((num - mu) / sd, dtype=torch.float32, device=DEV)
    Pm = torch.tensor(part / (part[tr].std(0) + 1e-9), dtype=torch.float32, device=DEV)
    return Nm, Pm, tr, es


print(f"dispositivo: {DEV}")
for cfg in args.configs:
    hu = HUELLA[cfg]
    fit = masks[cfg]
    idx, vi = np.where(fit)[0], np.where(val)[0]
    print(f"\n=== Config {cfg}: huella {hu or 'no'}{', con Poisson' if POISSON else ''}{', con márgenes por grupo' if GRUPOS else ''}{', con nivel del lobby' if LOBBY else ''}{', con composición' if COMP else ''}{f', con rivales d={args.rivales}' if args.rivales else ''} ===")
    # referencia: logística sobre las variables agregadas (igual que ganador_red.py)
    t0 = time.time()
    X = np.hstack([feat[k] for k in COMUN + ([hu] if hu else [])] + ([AGREGADO] if MARGEN else [])
                  + ([POI_PARTIDA] if POISSON else []) + (list(GRUPOS["nuevo"].values()) if GRUPOS else [])
                  + ([LOBBY["nuevo"]["nivel (foto)"]] if LOBBY else []) + ([COMP["nuevo"]["composicion"]] if COMP else []))
    p_log, p_log_fit = logistica(X, idx, vi, tambien_ajuste=True)
    informe(f"{cfg}. logística (agregada)", p_log, fit.sum(), time.time() - t0)
    guardadas[f"{cfg}_logistica"] = p_log
    ajuste = {f"{cfg}_fit_idx": idx, f"{cfg}_fit_logistica": p_log_fit}   # dentro de la muestra (Train)

    num = np.concatenate([jug["num"]] + ([jug[hu]] if hu else []) + ([MARGEN["jug"]] if MARGEN else [])
                         + ([POI_JUG] if POISSON else []) + ([GRUPOS["jug"]] if GRUPOS else []) + ([LOBBY_JUG] if LOBBY else [])
                         + ([COMP["jug"]] if COMP else []), axis=2)
    part = np.c_[feat["matchups"], feat["sinergias"]]
    if MARGEN:
        part = np.c_[part, MARGEN["nuevo"]["duos"]]
    if POISSON:
        part = np.c_[part, POI_PARTIDA]
    if COMP:
        part = np.c_[part, COMP["nuevo"]["composicion"]]
    Cm = torch.tensor(jug["campeon"].astype(np.int64), device=DEV)
    variantes = args.variantes.split(",")
    Nm, Pm, tr, es = tensores(num, part, idx)
    for variante in variantes:
        ps, ps_fit = [], []
        for seed in range(args.semillas):
            t0 = time.time()
            net, epocas = entrenar(variante, Cm, Nm, Pm, tr, es, seed, suave("V", idx))
            ps.append(predecir(net, Cm, Nm, Pm, vi))
            ps_fit.append(predecir(net, Cm, Nm, Pm, idx))
            informe(f"{cfg}. {variante}, semilla {seed} ({epocas} ép.)", ps[-1], fit.sum(), time.time() - t0, p_log)
        informe(f"{cfg}. {variante}, promedio {args.semillas} semillas", np.mean(ps, axis=0), fit.sum(), 0, p_log)
        guardadas[f"{cfg}_red_{variante}"] = np.mean(ps, axis=0)
        ajuste[f"{cfg}_fit_red_{variante}"] = np.mean(ps_fit, axis=0)
    del Nm, Pm
    torch.cuda.empty_cache()

    if args.oos:
        # ventana que crece: el bloque k se predice con modelos que solo vieron los bloques < k
        bloque = np.searchsorted(np.quantile(t[idx], np.linspace(0, 1, args.oos + 1)[1:-1]), t[idx], side="right")
        print(f"--- fuera de muestra: {args.oos} bloques por fecha dentro del ajuste ---")
        partes = {k: [] for k in ["idx", "bloque", "logistica"] + [f"red_{v}" for v in variantes]}
        for k in range(1, args.oos):
            t0 = time.time()
            antes, este = idx[bloque < k], idx[bloque == k]
            assert t[antes].max() <= t[este].min()
            partes["idx"].append(este); partes["bloque"].append(np.full(len(este), k))
            partes["logistica"].append(logistica(X, antes, este))
            Nm, Pm, tr, es = tensores(num, part, antes)
            for variante in variantes:
                ps = [predecir(entrenar(variante, Cm, Nm, Pm, tr, es, seed, suave(k, idx))[0], Cm, Nm, Pm, este)
                      for seed in range(args.semillas)]
                partes[f"red_{variante}"].append(np.mean(ps, axis=0))
            del Nm, Pm
            torch.cuda.empty_cache()
            print(f"  bloque {k}: ajuste {len(antes):>7d}, predice {len(este):>6d}  AUC logística "
                  f"{roc_auc_score(y[este], partes['logistica'][-1]):.3f}  " +
                  "  ".join(f"{v} {roc_auc_score(y[este], partes[f'red_{v}'][-1]):.3f}" for v in variantes) +
                  f"  ({time.time() - t0:.0f}s)", flush=True)
        o = {k: np.concatenate(v) for k, v in partes.items()}
        oos.update({f"{cfg}_oos_{k}": v for k, v in o.items()})
        oos.update({f"{cfg}_oos_y": y[o["idx"]], f"{cfg}_val_idx": vi, f"{cfg}_val_y": yv, f"{cfg}_val_t": t[vi]})
        oos.update({f"{cfg}_val_{k[2:]}": v for k, v in guardadas.items() if k.startswith(f"{cfg}_")})
        oos.update(ajuste)
    del X, Cm
    torch.cuda.empty_cache()

if args.guardar:
    np.savez(args.guardar, **guardadas)
    if oos:
        np.savez(Path(args.guardar).with_name(Path(args.guardar).stem + "_oos.npz"), **oos)

print("\nreferencia §9.S.23 (sin LP): logística A 0,610 / B 0,603 / C 0,597; red espejada agregada A 0,606 / B 0,601 / C 0,594")
