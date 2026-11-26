#!/usr/bin/env python3
"""
Compara los modelos del draft (escalón 2) sobre el test del split vigente.

Partida:  constante (tasa de Blue del train), regla (media de WR campeón-línea),
          logística antisimétrica y logística con lados separados.
Líneas:   constante, regla (oro medio @15, calibrada con una logística de una
          variable sobre el train), logística del duelo y logística del draft completo;
          por línea y con las 5 líneas juntas.

Se evalúan solo partidas completas (un jugador por equipo y línea, un único
ganador), así que todos los modelos ven las mismas partidas. Todo lo que se ajusta
(tablas de la regla, vocabulario, pesos y C) sale solo del train.

Uso:
    python scripts/analysis/evaluate_draft_logreg.py [--out data/analysis/modelos_draft.json]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.baselines.draft_logreg import (complete_match_ids, draft_matrix,  # noqa: E402
                                               duel_matrix, fit_temporal_C, lane_duels,
                                               match_target, prob_metrics)
from src.models.baselines.draft_rules import (LANES, lane_rule_scores, match_rule_scores,  # noqa: E402
                                              stats_from_participants)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

COLS = ["match_id", "team_id", "win", "champion_name", "lane", "gold_15", "game_creation"]


def fila(nombre: str, m: dict) -> str:
    f = lambda k, pct: (f"{100 * m[k]['valor']:.2f} [{100 * m[k]['ic95'][0]:.1f}; {100 * m[k]['ic95'][1]:.1f}]"
                        if pct else f"{m[k]['valor']:.4f} [{m[k]['ic95'][0]:.4f}; {m[k]['ic95'][1]:.4f}]")
    return f"  {nombre:<34} {m['n']:>8,}  {f('accuracy', True):>22}  {f('auc', False):>24}  {f('log_loss', False):>24}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Modelos del draft: regla y regresión logística.")
    ap.add_argument("--full", default=str(ROOT / "data" / "full_dataset.csv"))
    ap.add_argument("--split", default=str(ROOT / "data" / "splits" / "match_split.csv"))
    ap.add_argument("--out", default=str(ROOT / "data" / "analysis" / "modelos_draft.json"))
    args = ap.parse_args()
    t0 = time.time()

    parts = pd.read_csv(args.full, usecols=COLS)
    parts = parts[parts["lane"].isin(LANES)]
    split = pd.read_csv(args.split).set_index("match_id")["split"]
    ids = complete_match_ids(parts)
    lab = split.reindex(ids).to_numpy()
    tr_ids, te_ids = ids[lab == "train"], ids[lab == "test"]
    ids = tr_ids.append(te_ids)
    tr, te = np.arange(len(tr_ids)), np.arange(len(tr_ids), len(ids))
    creation = parts.groupby("match_id")["game_creation"].first().reindex(ids).to_numpy()
    p_tr = parts[parts["match_id"].isin(tr_ids)]
    p_te = parts[parts["match_id"].isin(te_ids)]
    stats = stats_from_participants(p_tr)
    print(f"[*] Partidas completas: train {len(tr_ids):,} | test {len(te_ids):,}", flush=True)

    out = {"partida": {}, "lineas": {}, "lineas_juntas": {}, "C": {}}

    # ---------------------------------------------------------------- Partida
    y = match_target(parts, ids)
    X_anti, vocab = draft_matrix(parts, ids, vocab=draft_matrix(parts, tr_ids)[1])
    X_lados, _ = draft_matrix(parts, ids, vocab=vocab, sides=True)
    regla = match_rule_scores(p_te, stats)[0].reindex(te_ids)
    out["partida"]["constante"] = prob_metrics(y[te], np.full(len(te), y[tr].mean()))
    out["partida"]["regla"] = prob_metrics(y[te], regla["prob"].to_numpy())
    for nombre, X in [("logistica_draft", X_anti), ("logistica_lados", X_lados)]:
        m, C, curva = fit_temporal_C(X, y, tr, creation)
        out["C"][f"partida/{nombre}"] = {"C": C, "curva_validacion": curva}
        out["partida"][nombre] = prob_metrics(y[te], m.predict_proba(X[te])[:, 1])
        if nombre == "logistica_draft":
            out["intercepto_partida"] = float(m.intercept_[0])
    print(f"[*] Partida lista ({time.time() - t0:.0f} s)", flush=True)

    # ---------------------------------------------------------------- Líneas
    juntas = {}
    reglas_tr = lane_rule_scores(p_tr, stats)[0].set_index(["match_id", "lane"])["score"]
    reglas_te = lane_rule_scores(p_te, stats)[0].set_index(["match_id", "lane"])["score"]
    for lane in LANES:
        d = lane_duels(parts, ids, lane)
        ok = d["y"].notna().to_numpy()
        tr_l, te_l = tr[ok[tr]], te[ok[te]]
        yl = np.nan_to_num(d["y"].to_numpy()).astype(int)
        X_duelo, _ = duel_matrix(d["champ_blue"], d["champ_red"],
                                 vocab=duel_matrix(d["champ_blue"].iloc[tr], d["champ_red"].iloc[tr])[1])

        s_tr = reglas_tr.reindex(pd.MultiIndex.from_arrays([ids[tr_l], [lane] * len(tr_l)])).to_numpy()
        s_te = reglas_te.reindex(pd.MultiIndex.from_arrays([ids[te_l], [lane] * len(te_l)])).to_numpy()
        cal = LogisticRegression().fit(np.nan_to_num(s_tr).reshape(-1, 1), yl[tr_l])
        preds = {"constante": np.full(len(te_l), yl[tr_l].mean()),
                 "regla": cal.predict_proba(np.nan_to_num(s_te).reshape(-1, 1))[:, 1]}
        for nombre, X in [("logistica_duelo", X_duelo), ("logistica_draft", X_anti)]:
            m, C, curva = fit_temporal_C(X, yl, tr_l, creation)
            out["C"][f"{lane}/{nombre}"] = {"C": C, "curva_validacion": curva}
            preds[nombre] = m.predict_proba(X[te_l])[:, 1]
        out["lineas"][lane] = {k: prob_metrics(yl[te_l], v) for k, v in preds.items()}
        for k, v in preds.items():
            juntas.setdefault(k, ([], []))
            juntas[k][0].append(yl[te_l]); juntas[k][1].append(v)
        print(f"[*] {lane} lista ({time.time() - t0:.0f} s)", flush=True)
    out["lineas_juntas"] = {k: prob_metrics(np.concatenate(a), np.concatenate(b)) for k, (a, b) in juntas.items()}

    # ---------------------------------------------------------------- Salida
    cab = f"  {'modelo':<34} {'n':>8}  {'accuracy %':>22}  {'AUC':>24}  {'log loss':>24}"
    for titulo, bloque in [("PARTIDA", out["partida"]), ("LÍNEAS JUNTAS", out["lineas_juntas"])]:
        print(f"\n{titulo}\n{cab}")
        for k, v in bloque.items():
            print(fila(k, v))
    for lane in LANES:
        print(f"\n{lane}\n{cab}")
        for k, v in out["lineas"][lane].items():
            print(fila(k, v))

    out["meta"] = {"train": int(len(tr_ids)), "test": int(len(te_ids)), "vocabulario": int(len(vocab)),
                   "segundos": round(time.time() - t0, 1)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[+] {args.out} ({out['meta']['segundos']} s)")


if __name__ == "__main__":
    main()
