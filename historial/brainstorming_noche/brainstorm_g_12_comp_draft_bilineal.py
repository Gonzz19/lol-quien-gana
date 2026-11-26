"""Ideas G29-G30 del brainstorming (docs/BRAINSTORM_NOCHE.md):
1. COMP_PLUS + INFO_X_EXP:
   - Subroles finos de Meraki (SKIRMISHER, WARDEN, DIVER, VANGUARD, JUGGERNAUT, BURST, BATTLEMAGE,
     CATCHER, SPECIALIST) + crecimiento de stats al nivel 18 (DDragon `perlevel`),
   - Interacciones no aditivas de composición (todo_AD * tank/vanguard/warden enemigo, todo_AP * tank,
     divers/asesinos vs sin peel/enchanter/warden, skirmisher/marksman + enchanter, burst vs squishies,
     escalado HP/DPS/Resistencias al 18),
   - Complejidad del campeón x experiencia causal del jugador (`estreno * info_dif`, `inv_n_champ * dif`,
     autofill en Jungla/ADC específicos, autofill vs OTP rival de línea).
2. DRAFT_FM_OOF (Factorización Bilineal / FM de Draft Rank-16 en GPU, estrictamente forward-chaining):
   - Modelo puro de draft en RTX 3070 (`wide(champ, lane)` + duelo de línea `<u_bl, w_rl>` + cruce 5x5
     + sinergia intra-equipo FM Rank-16), entrenado en bloques cronológicos estrictamente anteriores
     (`< b`, alineados con A2/A3/A4 y `< 16/09` para V). Captura matchups y sinergias densas incluso
     para cruces con pocos juegos en la base.

Caché guardada en `data/analysis/brainstorm/gemini_g12.npz`.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.brainstorm_base import ajustar_predecir, cargar, comparar, evaluar, extras_referencia, mostrar  # noqa: E402

CACHE = REPO / "data" / "analysis" / "brainstorm" / "gemini_g12.npz"
EXT_GD = Path(r"G:\Mi unidad\Lol Database\data_completa\externos")
DDRAGON_JSON = REPO / "data" / "ddragon" / "16.17.1" / "champion.json"


def _cargar_atributos_ricos(champ_unicos: np.ndarray) -> tuple[np.ndarray, list[str]]:
    mk_path = EXT_GD / "meraki" / "champions.json"
    cd_path = EXT_GD / "cdragon" / "champions_tactical.json"
    mk = {int(v["id"]): v for v in json.load(open(mk_path, encoding="utf-8")).values()} if mk_path.exists() else {}
    cd = {int(k): v for k, v in json.load(open(cd_path, encoding="utf-8")).items()} if cd_path.exists() else {}
    dd_raw = json.load(open(DDRAGON_JSON, encoding="utf-8"))["data"] if DDRAGON_JSON.exists() else {}
    dd = {int(v["key"]): v for v in dd_raw.values()}

    ROLES = [
        "SKIRMISHER", "DIVER", "VANGUARD", "WARDEN", "JUGGERNAUT",
        "BURST", "BATTLEMAGE", "ARTILLERY", "CATCHER", "ENCHANTER",
        "ASSASSIN", "MARKSMAN", "TANK", "SPECIALIST",
    ]
    AP_MAP = {"kMagic": 1.0, "kPhysical": 0.0, "kMixed": 0.5}
    rows = []
    for cid in champ_unicos:
        cid = int(cid)
        m, c, d = mk.get(cid, {}), cd.get(cid, {}), dd.get(cid, {})
        roles = set(m.get("roles") or [])
        st = d.get("stats") or {}
        info = d.get("info") or {}
        ti = c.get("tacticalInfo") or {}
        ap = AP_MAP.get(ti.get("damageType"))
        if ap is None and m:
            ap = 1.0 if m.get("adaptiveType") == "MAGIC_DAMAGE" else 0.0
        if ap is None:
            ap = 0.5
        hp18 = (st.get("hp", 600.0) + 17.0 * st.get("hpperlevel", 95.0)) / 1000.0
        arm18 = (st.get("armor", 30.0) + 17.0 * st.get("armorperlevel", 4.0)) / 100.0
        mr18 = (st.get("spellblock", 30.0) + 17.0 * st.get("spellblockperlevel", 1.5)) / 100.0
        ad18 = st.get("attackdamage", 60.0) + 17.0 * st.get("attackdamageperlevel", 3.0)
        as18 = st.get("attackspeed", 0.65) * (1.0 + 17.0 * st.get("attackspeedperlevel", 2.5) / 100.0)
        dps18 = (ad18 * as18) / 200.0
        dif = float(info.get("difficulty", 5.0))
        rango = float(m.get("attackType") == "RANGED") if m else float(st.get("attackrange", 150) >= 350)
        vec = [float(r in roles) for r in ROLES] + [ap, hp18, arm18, mr18, dps18, dif, rango]
        rows.append(vec)
    names = ROLES + ["ap", "hp18", "arm18", "mr18", "dps18", "dif", "rango"]
    return np.asarray(rows, dtype=np.float32), names


class DraftBilinearFM(nn.Module):
    def __init__(self, n_camp: int, dim: int = 16):
        super().__init__()
        self.wide = nn.Embedding(n_camp * 5, 1)
        self.u = nn.Embedding(n_camp * 5, dim)
        self.w = nn.Embedding(n_camp * 5, dim)
        self.g = nn.Embedding(n_camp * 5, dim)
        nn.init.normal_(self.wide.weight, std=0.02)
        nn.init.normal_(self.u.weight, std=0.05)
        nn.init.normal_(self.w.weight, std=0.05)
        nn.init.normal_(self.g.weight, std=0.05)
        self.lado = nn.Parameter(torch.tensor(-0.10))
        self.sc = nn.Parameter(torch.tensor([0.2, 0.1, 0.1]))
        self.register_buffer("lin", torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=torch.long))
        self.dim = dim

    def forward_parts(self, camp_idx):
        bsz = len(camp_idx)
        cl = camp_idx * 5 + self.lin.unsqueeze(0).expand(bsz, -1)
        lw = self.wide(cl).squeeze(-1)
        z_wide = lw[:, :5].sum(1) - lw[:, 5:].sum(1) + self.lado

        uv = self.u(cl)
        wv = self.w(cl)
        gv = self.g(cl)
        s = self.dim ** 0.5
        ub, ur = uv[:, :5], uv[:, 5:]
        wb, wr = wv[:, :5], wv[:, 5:]
        gb, gr = gv[:, :5], gv[:, 5:]

        z_duel = ((ub * wr).sum(dim=(1, 2)) - (ur * wb).sum(dim=(1, 2))) / s
        z_cross = ((ub.sum(1) * wr.sum(1)).sum(1) - (ur.sum(1) * wb.sum(1)).sum(1)) / (5.0 * s)
        syn_b = 0.5 * ((gb.sum(1) ** 2 - (gb ** 2).sum(1)).sum(1)) / (4.0 * s)
        syn_r = 0.5 * ((gr.sum(1) ** 2 - (gr ** 2).sum(1)).sum(1)) / (4.0 * s)
        z_int = self.sc[0] * z_duel + self.sc[1] * z_cross + self.sc[2] * (syn_b - syn_r)
        return z_wide, z_duel, z_cross, (syn_b - syn_r), z_wide + z_int


def _entrenar_draft_fm(camp_idx: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray, n_camp: int, dev):
    if len(tr) < 2000 or len(te) == 0:
        return np.zeros((len(te), 4), dtype=np.float32)
    n_es = max(500, int(0.12 * len(tr)))
    tr_sub, es_sub = tr[:-n_es], tr[-n_es:]
    C_tr = torch.tensor(camp_idx[tr_sub], dtype=torch.long, device=dev)
    Y_tr = torch.tensor(y[tr_sub], dtype=torch.float32, device=dev)
    C_es = torch.tensor(camp_idx[es_sub], dtype=torch.long, device=dev)
    Y_es = torch.tensor(y[es_sub], dtype=torch.float32, device=dev)
    C_te = torch.tensor(camp_idx[te], dtype=torch.long, device=dev)

    preds = np.zeros((len(te), 4), dtype=np.float32)
    seeds = (0, 1)
    for seed in seeds:
        torch.manual_seed(seed)
        net = DraftBilinearFM(n_camp, dim=16).to(dev)
        opt = torch.optim.AdamW([
            {"params": net.wide.parameters(), "lr": 1.0e-2, "weight_decay": 6e-2},
            {"params": [p for n, p in net.named_parameters() if not n.startswith("wide.")], "lr": 3e-3, "weight_decay": 1e-1},
        ])
        best_loss, best_sd, pat = 1e9, None, 0
        n_rows = len(tr_sub)
        for _ in range(30):
            net.train()
            perm = torch.randperm(n_rows, device=dev)
            for b in perm.split(2048):
                opt.zero_grad()
                _, _, _, _, tot = net.forward_parts(C_tr[b])
                loss = nn.functional.binary_cross_entropy_with_logits(tot, Y_tr[b])
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                _, _, _, _, z_es = net.forward_parts(C_es)
                vloss = float(nn.functional.binary_cross_entropy_with_logits(z_es, Y_es))
            if vloss < best_loss - 1e-5:
                best_loss = vloss
                best_sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                pat = 0
            else:
                pat += 1
                if pat >= 4:
                    break
        if best_sd is not None:
            net.load_state_dict({k: v.to(dev) for k, v in best_sd.items()})
        net.eval()
        with torch.no_grad():
            for i in range(0, len(te), 8192):
                zw, zd, zc, zs, zt = net.forward_parts(C_te[i:i + 8192])
                out = torch.stack([zt, zt - zw, zd, zs], dim=1).cpu().numpy()
                preds[i:i + 8192] += out / len(seeds)
    return preds


def construir_cache_g12(D: dict, rehacer: bool = False) -> dict[str, np.ndarray]:
    if CACHE.exists() and not rehacer:
        z = np.load(CACHE)
        if int(z["nm"][0]) == D["NM"]:
            return {k: z[k] for k in z.files if k != "nm"}

    t0 = time.time()
    print("Construyendo caché gemini_g12.npz...", flush=True)
    NM = D["NM"]
    champ = D["Z"]["champ"].reshape(NM, 10)
    unicos, cinv = np.unique(champ, return_inverse=True)
    camp_idx = cinv.reshape(NM, 10).astype(np.int64)
    n_camp = len(unicos)

    A, names = _cargar_atributos_ricos(unicos)
    idx = {n: i for i, n in enumerate(names)}
    V = A[camp_idx]  # (NM, 10, D)

    def side_feats(E):
        g = {n: E[:, :, i] for i, n in enumerate(names)}
        skirm = g["SKIRMISHER"].sum(1)
        diver = g["DIVER"].sum(1)
        vang = g["VANGUARD"].sum(1)
        ward = g["WARDEN"].sum(1)
        jugg = g["JUGGERNAUT"].sum(1)
        burst = g["BURST"].sum(1)
        bmage = g["BATTLEMAGE"].sum(1)
        art = g["ARTILLERY"].sum(1)
        catch = g["CATCHER"].sum(1)
        ench = g["ENCHANTER"].sum(1)
        assn = g["ASSASSIN"].sum(1)
        mark = g["MARKSMAN"].sum(1)
        tank = g["TANK"].sum(1)
        spec = g["SPECIALIST"].sum(1)
        ap = g["ap"].sum(1)
        hp18 = g["hp18"].sum(1)
        arm18 = g["arm18"].sum(1)
        mr18 = g["mr18"].sum(1)
        dps18 = g["dps18"].sum(1)
        dif = g["dif"].sum(1)
        rng = g["rango"].sum(1)
        return dict(
            skirm=skirm, diver=diver, vang=vang, ward=ward, jugg=jugg,
            burst=burst, bmage=bmage, art=art, catch=catch, ench=ench,
            assn=assn, mark=mark, tank=tank, spec=spec,
            ap=ap, hp18=hp18, arm18=arm18, mr18=mr18, dps18=dps18, dif=dif, rng=rng,
            todo_ad=(ap <= 0.5).astype(np.float32),
            todo_ap=(ap >= 4.0).astype(np.float32),
            hard_tank=(vang + ward + tank),
            peel=(ward + ench),
            squishy=(mark + art + burst),
            melee_dive=(diver + assn),
        )

    B, R = side_feats(V[:, :5]), side_feats(V[:, 5:])
    comp_cols = []
    # 1. Roles finos propios (azul - rojo)
    for k in ["skirm", "diver", "vang", "ward", "jugg", "burst", "bmage", "art", "catch", "spec",
              "hp18", "arm18", "mr18", "dps18", "dif"]:
        comp_cols.append(B[k] - R[k])
    # 2. Cruces tácticos reales en Master+ (todo AD vs armadura real, asesinos vs sin peel, etc.)
    comp_cols.extend([
        B["todo_ad"] * R["hard_tank"] - R["todo_ad"] * B["hard_tank"],
        B["todo_ap"] * R["hard_tank"] - R["todo_ap"] * B["hard_tank"],
        B["melee_dive"] * (R["peel"] == 0) - R["melee_dive"] * (B["peel"] == 0),
        (B["skirm"] + B["mark"]) * B["ench"] - (R["skirm"] + R["mark"]) * R["ench"],
        B["art"] * ((R["vang"] + R["catch"]) == 0) - R["art"] * ((B["vang"] + B["catch"]) == 0),
        (B["burst"] + B["assn"]) * R["squishy"] - (R["burst"] + R["assn"]) * B["squishy"],
        (B["dps18"] * (B["hard_tank"] > 0)) - (R["dps18"] * (R["hard_tank"] > 0)),
        B["catch"] * (5.0 - R["rng"]) - R["catch"] * (5.0 - B["rng"]),
    ])
    COMP_PLUS = np.column_stack(comp_cols).astype(np.float32)

    # 3. Complejidad del campeon x experiencia causal del jugador + autofill especifico
    cols_x = {k: i for i, k in enumerate(D["cols"])}
    Xj = D["Xj"]
    est = Xj[..., cols_x["estreno"]]
    lnc = Xj[..., cols_x["logn_champ"]]
    n_c = np.expm1(lnc)
    off = Xj[..., cols_x["offrole"]]
    shl = Xj[..., cols_x["share_lane"]]
    dif_j = V[..., idx["dif"]].reshape(NM, 2, 5)

    exp_cols = [
        (est[:, 0] * dif_j[:, 0]).sum(1) - (est[:, 1] * dif_j[:, 1]).sum(1),
        (est[:, 0] * (dif_j[:, 0] >= 7)).sum(1) - (est[:, 1] * (dif_j[:, 1] >= 7)).sum(1),
        (dif_j[:, 0] / (1.0 + n_c[:, 0])).sum(1) - (dif_j[:, 1] / (1.0 + n_c[:, 1])).sum(1),
        off[:, 0, 1] - off[:, 1, 1],  # offrole JG
        off[:, 0, 0] - off[:, 1, 0],  # offrole ADC
        (off[:, 0] * (1.0 - shl[:, 0])).sum(1) - (off[:, 1] * (1.0 - shl[:, 1])).sum(1),
        (off[:, 0] * (shl[:, 1] > 0.7)).sum(1) - (off[:, 1] * (shl[:, 0] > 0.7)).sum(1),
    ]
    INFO_X_EXP = np.column_stack(exp_cols).astype(np.float32)

    # 4. DRAFT_FM_OOF estrictamente forward-chaining alineado con bloques (A1/A2/A3/A4/V)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entrenando DRAFT_FM_OOF en {dev}...", flush=True)
    y = D["y"]
    t = D["t"]
    val0 = D["val0"]
    post = D["post"]
    bl = D["bloque"]
    ms = D["master"]

    # Pre-post (< 04/09) dividido en 7 bloques cronológicos + post bloques 0, 1, 2, 3
    pre_m = ~post
    qs = np.quantile(t[pre_m], np.linspace(0, 1, 8)[1:-1])
    ch_pre = np.searchsorted(qs, t, side="right")  # 0..6

    # Etapa IDs estrictamente crecientes: 0..6 pre, 7=(A1), 8=(A2), 9=(A3), 10=(pre-16/09 de A4), 11=(t >= 16/09: V y diam_v)
    stage = np.full(NM, 11, dtype=np.int32)
    stage[pre_m] = ch_pre[pre_m]
    for b in (0, 1, 2, 3):
        stage[post & (~val0) & (bl == b)] = 7 + b
    # Nota: stage == 11 corresponde exclusivamente a t >= 16/09 (val0: V y diam_v). Solo se predice con < 16/09 (~val0).

    DRAFT_FM_OOF = np.zeros((NM, 4), dtype=np.float32)
    for st in range(1, 12):
        if st < 11:
            tr_st = np.flatnonzero((stage < st) & ms)
        else:
            tr_st = np.flatnonzero((~val0) & ms)
        if len(tr_st) > 1:
            tr_st = tr_st[np.argsort(t[tr_st], kind="mergesort")]
        te_st = np.flatnonzero(stage == st)
        DRAFT_FM_OOF[te_st] = _entrenar_draft_fm(camp_idx, y, tr_st, te_st, n_camp, dev)

    out = {"COMP_PLUS": COMP_PLUS, "INFO_X_EXP": INFO_X_EXP, "DRAFT_FM_OOF": DRAFT_FM_OOF}
    np.savez_compressed(CACHE, nm=np.array([NM]), **out)
    print(f"Caché gemini_g12.npz lista en {time.time() - t0:.1f}s", flush=True)
    return out


def main():
    t0 = time.time()
    D = cargar()
    ms = D["master"]
    G = construir_cache_g12(D)
    R6 = extras_referencia(D, escalon=6)

    def run_one(X):
        X32 = np.asarray(X, dtype=np.float32)
        return evaluar(D, lambda tr, te, M=X32: ajustar_predecir(D, tr[ms[tr]], te, extra=M, extra_post=False, con_mapa=False))

    P = {"ref: Escalón 6 (sin mapa)": run_one(R6)}
    for k in ["COMP_PLUS", "INFO_X_EXP", "DRAFT_FM_OOF"]:
        print(f"Evaluando +{k} s/Esc6 ({time.time()-t0:.1f}s)...", flush=True)
        P[f"+ {k} (s/Esc6)"] = run_one(np.c_[R6, G[k]])

    X_all = np.c_[R6, G["COMP_PLUS"], G["INFO_X_EXP"], G["DRAFT_FM_OOF"]]
    P["+ TODO g12 (s/Esc6)"] = run_one(X_all)

    mostrar(comparar(D, P, "ref: Escalón 6 (sin mapa)"))
    from sklearn.metrics import roc_auc_score
    di = np.flatnonzero(D["diam_v"])
    yd = D["y"][di]
    B = np.random.default_rng(1).integers(0, len(di), size=(300, len(di)))
    b0 = P["ref: Escalón 6 (sin mapa)"][0][di]
    print(f"\nChequeo Diamond-V ({len(di):,} partidas):")
    for nom in [k for k in P if not k.startswith("ref:")]:
        p = P[nom][0][di]
        d = np.array([roc_auc_score(yd[b], p[b]) - roc_auc_score(yd[b], b0[b]) for b in B])
        print(f"  {nom:24s} AUC {roc_auc_score(yd, p):.4f}  Δ {roc_auc_score(yd, p) - roc_auc_score(yd, b0):+.4f} "
              f"[{np.percentile(d, 2.5):+.4f}; {np.percentile(d, 97.5):+.4f}]")
    print(f"Listo en {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
