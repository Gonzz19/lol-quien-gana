"""Idea 15 del brainstorming (docs/BRAINSTORM_NOCHE.md): Red Wide & Deep por jugador en GPU (RTX 3070)
entrenada SOLO con Master+ (`solo Master+`), sumando MC/ML + mapa + Foto 1 W/L + pooling sum/max.

Motivación:
* En §9.S.29/36 la red compartida por jugador solo corrió sobre todos los tiers (con Diamond),
  sin el mapa de LP, sin Foto 1 W/L, con pooling solo por suma (`DeepSets` suma) y sin conexión
  lineal directa (`Wide & Deep`), por lo que sus interacciones multiplicativas quedaron limitadas.
* En la logística lineal (`Xd`), `wr_f1 × (1 / logn_f1)` (smurfs), `Master 0 LP × wr_f1`,
  `JG_offrole × estreno` y `carry (max)` vs `eslabón débil (min)` no pueden representarse.
* Con RTX 3070 CUDA, entrenar `A2..A4` y `V` con 3 semillas y early stopping estricto dentro
  de `tr` (último 15 % cronológico de `tr`, sin mirar `te`) toma ~1-2 min en total y no compite
  con la CPU.

Evalúa:
1. Referencia logística (`solo Master+ + MC/ML` = 0,6614 en V).
2. Logística (`+ MC/ML + Foto 1 W/L`).
3. Red Wide & Deep (`sum + max` por equipo, GPU, 3 semillas).
4. Ensamble causal 50/50 (o pesos por bloques A2..A4) Red GPU + Logística.
"""
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.brainstorm_base import cargar, comparar, evaluar, mostrar  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cargar_foto1_jugador(D: dict) -> np.ndarray:
    """Matriz (NM, 2, 5, 4) por jugador post-t1 (dinámico causal): wr_f1, logn_f1, netw_f1, zero_m."""
    NM = D["NM"]
    pl = D["B"]["player"].reshape(-1)
    pr = np.repeat(D["post"], 10)
    with sqlite3.connect(f"file:{REPO / 'data' / 'lol_las.db'}?mode=ro&immutable=1", uri=True) as con:
        foto = pd.read_sql_query(
            "SELECT m.player_id, p.tier, p.league_points, p.wins, p.losses "
            "FROM players p JOIN player_id_map m ON p.puuid = m.puuid",
            con,
        ).drop_duplicates("player_id").set_index("player_id")
    has_f1 = pd.Series(pl).isin(foto.index).to_numpy() & pr
    al = foto.reindex(pl)
    W0 = al["wins"].fillna(0.0).to_numpy(np.float64)
    L0 = al["losses"].fillna(0.0).to_numpy(np.float64)
    zm = (has_f1 & (al["tier"].fillna("").to_numpy() == "MASTER") & (al["league_points"].fillna(-1).to_numpy() == 0)).astype(float)

    # Suma causal post-t1 de W/L observados en partidas anteriores desde t1
    y_jug = np.repeat(D["y"], 10).astype(np.float64)
    es_azul = np.tile(np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=bool), NM)
    win_slot = np.where(es_azul, y_jug, 1.0 - y_jug)
    df_post = pd.DataFrame({
        "_pos": np.arange(NM * 10),
        "player_id": pl,
        "t": np.repeat(D["t"], 10),
        "m": np.repeat(np.arange(NM), 10),
        "post": pr,
        "win": win_slot,
    })
    sub_p = df_post[df_post["post"]].sort_values(["t", "m"], kind="mergesort").copy()
    sub_p["dw_prev"] = sub_p.groupby("player_id", sort=False)["win"].cumsum() - sub_p["win"]
    sub_p["dn_prev"] = sub_p.groupby("player_id", sort=False).cumcount().astype(np.float64)
    sub_p["dl_prev"] = sub_p["dn_prev"] - sub_p["dw_prev"]
    dw_all = np.zeros(NM * 10, dtype=np.float64)
    dl_all = np.zeros(NM * 10, dtype=np.float64)
    dw_all[sub_p["_pos"].to_numpy()] = sub_p["dw_prev"].to_numpy()
    dl_all[sub_p["_pos"].to_numpy()] = sub_p["dl_prev"].to_numpy()

    W = W0 + dw_all
    L = L0 + dl_all
    tot = W + L
    wr = np.where(has_f1, (W + 10.0 * 0.5) / (tot + 20.0) - 0.5, 0.0)
    ln = np.where(has_f1, np.log1p(tot) - 5.3, 0.0)
    nw = np.where(has_f1, W - L, 0.0)
    return np.stack([wr, ln, nw, zm], axis=-1).reshape(NM, 2, 5, 4).astype(np.float32)


class RedWideDeep(nn.Module):
    def __init__(self, n_camp: int, d_num: int, d_wide: int, h_dim: int = 96, drop: float = 0.25):
        super().__init__()
        self.camp = nn.Embedding(n_camp, 16)
        self.linea = nn.Embedding(5, 4)
        self.register_buffer("lineas", torch.arange(10) % 5)

        # Rama Wide lineal antisimétrica: captura el draft campeón x línea + features lineales
        self.camp_lin = nn.Embedding(n_camp * 5, 1)
        nn.init.zeros_(self.camp_lin.weight)
        self.wide = nn.Linear(d_wide, 1, bias=False)
        nn.init.zeros_(self.wide.weight)

        # Codificador compartido por jugador (phi) con LayerNorm para estabilidad con múltiples escalas
        d_in = 16 + 4 + d_num
        self.phi = nn.Sequential(
            nn.Linear(d_in, h_dim),
            nn.LayerNorm(h_dim),
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Linear(h_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.SiLU(),
            nn.Dropout(drop),
        )
        # Interacción de equipo antisimétrica con [sum, max] por equipo (= 2*h_dim por equipo)
        self.g = nn.Sequential(
            nn.Linear(4 * h_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Linear(h_dim, 1),
        )
        # Modulación leve de ventaja de lado Azul según nivel medio del lobby (ha_sum + hr_sum)
        self.lado_base = nn.Parameter(torch.zeros(1))
        self.lado_lob = nn.Linear(2 * h_dim, 1, bias=False)
        nn.init.zeros_(self.lado_lob.weight)

    def forward(self, camp, num, wide_x):
        bsz = len(camp)
        lin = self.lineas.expand(bsz, -1)
        cl_idx = camp * 5 + lin
        cl_w = self.camp_lin(cl_idx).squeeze(-1)
        z_wide = (cl_w[:, :5].sum(1) - cl_w[:, 5:].sum(1)) + self.wide(wide_x).squeeze(-1)

        x = torch.cat([self.camp(camp), self.linea(lin), num], dim=2)
        h = self.phi(x)  # (B, 10, h_dim)
        hb, hr = h[:, :5], h[:, 5:]
        eb = torch.cat([hb.mean(1), hb.max(1).values], dim=1)
        er = torch.cat([hr.mean(1), hr.max(1).values], dim=1)
        z_deep = (self.g(torch.cat([eb, er], dim=1)) - self.g(torch.cat([er, eb], dim=1))).squeeze(-1)
        z_lado = self.lado_base + self.lado_lob(eb + er).squeeze(-1)
        return z_wide + 0.5 * z_deep + z_lado


def preparar_datos_red(D: dict, F1: np.ndarray, MC_raw: np.ndarray, ML_raw: np.ndarray):
    NM = D["NM"]
    ids = D["Z"]["champ"].reshape(NM, 10)
    unicos = np.unique(ids)
    cmap = {int(c): i for i, c in enumerate(unicos)}
    camp_idx = np.vectorize(cmap.__getitem__, otypes=[np.int64])(ids)

    post_p = np.broadcast_to(D["post"][:, None, None, None], (NM, 2, 5, 1)).astype(np.float32)
    # Tensor numérico completo por jugador: (NM, 2, 5, d_num)
    P_jug = np.concatenate([D["Xj"], MC_raw, ML_raw, D["J"], F1, post_p], axis=-1).reshape(NM, 10, -1).astype(np.float32)

    # Rama wide (Azul - Rojo) por línea y agregada post
    d_pre = np.concatenate([D["Xj"], MC_raw, ML_raw], axis=-1)
    W_pre = (d_pre[:, 0] - d_pre[:, 1]).reshape(NM, -1).astype(np.float32)
    d_post_eq = np.concatenate([
        (D["J"][:, 0] - D["J"][:, 1]).mean(1),
        ((D["J"][:, 0] - D["J"][:, 1]).mean(1)[:, [1]] - (D["J"][:, 0] - D["J"][:, 1]).mean(1)[:, [3]]),
        (F1[:, 0] - F1[:, 1]).mean(1),
    ], axis=-1).astype(np.float32)
    return camp_idx, P_jug, W_pre, d_post_eq, len(unicos)


def entrenar_predecir_red(D, camp_idx, P_jug, W_pre, d_post_eq, n_camp, tr_raw, te, seeds=(0, 1, 2)):
    ms, t, y, post = D["master"], D["t"], D["y"], D["post"]
    tr = tr_raw[ms[tr_raw]]

    # Estandarización estrictamente con filas de entrenamiento tr (y tr[post] para variables post-t1)
    d_pre_jug = P_jug.shape[2] - 12  # últimas 12 cols: 7 J + 4 F1 + 1 post_flag
    n_tr_flat = P_jug[tr, :, :d_pre_jug].reshape(-1, d_pre_jug)
    mu_jp, sd_jp = n_tr_flat.mean(0), n_tr_flat.std(0) + 1e-6

    post_tr = tr[post[tr]]
    if len(post_tr) > 0:
        n_po_flat = P_jug[post_tr, :, d_pre_jug:-1].reshape(-1, 11)
        mu_jpo, sd_jpo = n_po_flat.mean(0), n_po_flat.std(0) + 1e-6
    else:
        mu_jpo, sd_jpo = np.zeros(11, np.float32), np.ones(11, np.float32)

    mu_wp, sd_wp = W_pre[tr].mean(0), W_pre[tr].std(0) + 1e-6
    if len(post_tr) > 0:
        mu_wpo, sd_wpo = d_post_eq[post_tr].mean(0), d_post_eq[post_tr].std(0) + 1e-6
    else:
        mu_wpo, sd_wpo = np.zeros(d_post_eq.shape[1], np.float32), np.ones(d_post_eq.shape[1], np.float32)

    def build_tensors(idx):
        p_norm = np.empty_like(P_jug[idx])
        p_norm[..., :d_pre_jug] = np.clip((P_jug[idx, :, :d_pre_jug] - mu_jp) / sd_jp, -8.0, 8.0)
        m_post = post[idx][:, None, None]
        p_norm[..., d_pre_jug:-1] = np.where(
            m_post, np.clip((P_jug[idx, :, d_pre_jug:-1] - mu_jpo) / sd_jpo, -8.0, 8.0), 0.0
        )
        p_norm[..., -1] = P_jug[idx, :, -1]

        w_pre_n = np.clip((W_pre[idx] - mu_wp) / sd_wp, -8.0, 8.0)
        w_post_n = np.where(
            post[idx][:, None], np.clip((d_post_eq[idx] - mu_wpo) / sd_wpo, -8.0, 8.0), 0.0
        )
        wide_n = np.hstack([w_pre_n, w_post_n]).astype(np.float32)
        return (
            torch.tensor(camp_idx[idx], dtype=torch.long, device=DEV),
            torch.tensor(p_norm, dtype=torch.float32, device=DEV),
            torch.tensor(wide_n, dtype=torch.float32, device=DEV),
        )

    # Split interno causal que garantiza filas post-t1 tanto en train como en early-stopping
    pre_tr = tr[~post[tr]]
    pos_tr = tr[post[tr]]
    if len(pos_tr) >= 1000:
        c_pre = np.quantile(t[pre_tr], 0.88) if len(pre_tr) > 0 else 0
        c_pos = np.quantile(t[pos_tr], 0.80)
        tr_sub = np.sort(np.concatenate([pre_tr[t[pre_tr] < c_pre], pos_tr[t[pos_tr] < c_pos]]))
        es_sub = np.sort(np.concatenate([pre_tr[t[pre_tr] >= c_pre], pos_tr[t[pos_tr] >= c_pos]]))
        es_mask = post[es_sub]
    else:
        corte = np.quantile(t[tr], 0.85)
        tr_sub = tr[t[tr] < corte]
        es_sub = tr[t[tr] >= corte]
        es_mask = np.ones(len(es_sub), dtype=bool)

    C_tr, N_tr, W_tr = build_tensors(tr_sub)
    C_es, N_es, W_es = build_tensors(es_sub)
    C_te, N_te, W_te = build_tensors(te)
    y_tr = torch.tensor(y[tr_sub], dtype=torch.float32, device=DEV)
    y_es_np = y[es_sub]

    preds = []
    for seed in seeds:
        torch.manual_seed(seed)
        net = RedWideDeep(n_camp, P_jug.shape[2], W_tr.shape[1], h_dim=96, drop=0.30).to(DEV)
        # Mayor weight decay en rama wide para imitar C=0.05 y evitar sobreajuste
        opt = torch.optim.AdamW([
            {"params": list(net.camp_lin.parameters()) + list(net.wide.parameters()), "weight_decay": 1e-1, "lr": 1e-3},
            {"params": [p for n, p in net.named_parameters() if not n.startswith(("camp_lin.", "wide."))], "weight_decay": 2e-2, "lr": 8e-4},
        ])
        best_ll, best_sd, pat = 1e9, None, 0
        n_rows = len(tr_sub)
        for _ in range(60):
            net.train()
            perm = torch.randperm(n_rows, device=DEV)
            for b in perm.split(512):
                opt.zero_grad()
                out = net(C_tr[b], N_tr[b], W_tr[b])
                loss = nn.functional.binary_cross_entropy_with_logits(out, y_tr[b])
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                p_es = []
                for i in range(0, len(es_sub), 4096):
                    p_es.append(torch.sigmoid(net(C_es[i:i+4096], N_es[i:i+4096], W_es[i:i+4096])).cpu().numpy())
                ll = log_loss(y_es_np[es_mask], np.clip(np.concatenate(p_es)[es_mask], 1e-6, 1 - 1e-6))
            if ll < best_ll - 1e-5:
                best_ll = ll
                best_sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                pat = 0
            else:
                pat += 1
                if pat >= 6:
                    break
        net.load_state_dict({k: v.to(DEV) for k, v in best_sd.items()})
        net.eval()
        with torch.no_grad():
            p_te = []
            for i in range(0, len(te), 4096):
                p_te.append(torch.sigmoid(net(C_te[i:i+4096], N_te[i:i+4096], W_te[i:i+4096])).cpu().numpy())
            preds.append(np.concatenate(p_te))
    del C_tr, N_tr, W_tr, C_es, N_es, W_es, C_te, N_te, W_te
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
    return np.mean(preds, axis=0)


def main():
    t0 = time.time()
    print(f"Dispositivo PyTorch: {DEV}")
    D = cargar()
    ms = D["master"]
    C = np.load(REPO / "data" / "analysis" / "brainstorm" / "margen_grupos.npz")
    MC_raw, ML_raw = C["MC"], C["ML"]
    MC_ML = np.c_[
        (MC_raw[:, 0] - MC_raw[:, 1]).reshape(D["NM"], -1),
        (ML_raw[:, 0] - ML_raw[:, 1]).reshape(D["NM"], -1),
    ].astype(np.float32)
    F1 = cargar_foto1_jugador(D)
    F1_eq = (F1[:, 0] - F1[:, 1]).mean(axis=1).astype(np.float32)

    from scripts.analysis.brainstorm_base import ajustar_predecir
    from scipy import sparse

    def log_mc_f1(tr_raw, te):
        tr = tr_raw[ms[tr_raw]]
        Xd, post, mapa = D["Xd"], D["post"], D["mapa"]
        Xtr_p = np.c_[Xd[tr], MC_ML[tr]].astype(np.float32)
        Xte_p = np.c_[Xd[te], MC_ML[te]].astype(np.float32)
        mu, sd = Xtr_p.mean(0), Xtr_p.std(0) + 1e-6
        post_f = np.c_[mapa, F1_eq].astype(np.float32)
        ptr = post[tr]
        me, se = post_f[tr][ptr].mean(0), post_f[tr][ptr].std(0) + 1e-6
        Mtr = sparse.hstack([
            D["DRAFT"][tr],
            sparse.csr_matrix(((Xtr_p - mu) / sd).astype(np.float32)),
            sparse.csr_matrix(np.where(ptr[:, None], (post_f[tr] - me) / se, 0.0).astype(np.float32)),
        ]).tocsr()
        Mte = sparse.hstack([
            D["DRAFT"][te],
            sparse.csr_matrix(((Xte_p - mu) / sd).astype(np.float32)),
            sparse.csr_matrix(np.where(post[te][:, None], (post_f[te] - me) / se, 0.0).astype(np.float32)),
        ]).tocsr()
        lr = LogisticRegression(C=0.05, max_iter=4000).fit(Mtr, D["y"][tr])
        return lr.predict_proba(Mte)[:, 1]

    camp_idx, P_jug, W_pre, d_post_eq, n_camp = preparar_datos_red(D, F1, MC_raw, ML_raw)

    print("Evaluando logística ref + logística(+Foto1 W/L)...")
    p_ref = evaluar(D, lambda tr, te: ajustar_predecir(D, tr[ms[tr]], te, extra=MC_ML, extra_post=False))
    p_log_f1 = evaluar(D, log_mc_f1)
    print(f"Evaluando Red Wide&Deep GPU ({time.time()-t0:.1f}s)...")
    p_red = evaluar(D, lambda tr, te: entrenar_predecir_red(D, camp_idx, P_jug, W_pre, d_post_eq, n_camp, tr, te))

    # Combinaciones simples promedio de probabilidades y promedio de logits (50/50 y 40/60)
    def blend_prob(a, b, w=0.5):
        return (w * a[0] + (1 - w) * b[0], w * a[1] + (1 - w) * b[1])

    from scipy.special import expit, logit
    def blend_logit(a, b, w=0.5):
        eps = 1e-6
        lo = w * logit(np.clip(a[0], eps, 1 - eps)) + (1 - w) * logit(np.clip(b[0], eps, 1 - eps))
        lv = w * logit(np.clip(a[1], eps, 1 - eps)) + (1 - w) * logit(np.clip(b[1], eps, 1 - eps))
        return (expit(lo), expit(lv))

    p_mix50 = blend_logit(p_red, p_log_f1, 0.5)
    p_mix35 = blend_logit(p_red, p_log_f1, 0.35)
    P = {
        "ref: solo Master+ + MC/ML": p_ref,
        "logística + MC/ML + Foto1 W/L": p_log_f1,
        "Red Wide&Deep GPU (solo Master+)": p_red,
        "Mix 50/50 logit (Red + Log_F1)": p_mix50,
        "Mix 35/65 logit (Red + Log_F1)": p_mix35,
    }
    mostrar(comparar(D, P, "ref: solo Master+ + MC/ML"))
    out = REPO / "data" / "analysis" / "brainstorm" / "pred_g05.npz"
    np.savez_compressed(
        out,
        po_ref=p_ref[0], pv_ref=p_ref[1],
        po_log_f1=p_log_f1[0], pv_log_f1=p_log_f1[1],
        po_red=p_red[0], pv_red=p_red[1],
    )
    print(f"Guardado {out} | Tiempo total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
