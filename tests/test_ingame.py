"""Tests unitarios de src/models/ingame.py: causalidad, enmascaramiento y antisimetría."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.ingame import (
    ATRIB_COLS,
    GROWTH_COLS,
    InGameNet,
    TIMELINE_STATS,
    apply_residual_gold,
    build_composition_features,
    causal_champ_player_tempo,
    extract_minute_features,
    fit_residual_gold_coeffs,
    mask_finished_matches,
)


def test_mask_finished_matches_evita_frame_final():
    """Riot guarda frames[m] cuando dur <= 60*m (fin de partida): debe quedar en NaN."""
    vals = np.ones((4, 2, 5), dtype=np.float32) * 5000.0
    dur_sec = np.array([599.0, 900.0, 901.0, 1200.0])
    out15 = mask_finished_matches(vals, dur_sec, 15)
    assert np.isnan(out15[0]).all()
    assert np.isnan(out15[1]).all()   # exacto <= 900s también enmascarado
    assert np.isfinite(out15[2]).all()
    assert np.isfinite(out15[3]).all()


def test_residual_gold_separa_placas_y_no_mira_fuera_de_fit():
    rng = np.random.default_rng(7)
    NM = 40
    cs = rng.uniform(40, 120, size=(NM, 2, 5)).astype(np.float32)
    kills = rng.integers(0, 5, size=(NM, 2, 5)).astype(np.float32)
    assists = rng.integers(0, 6, size=(NM, 2, 5)).astype(np.float32)
    gold = (500.0 + 21.0 * cs + 300.0 * kills + 150.0 * assists).astype(np.float32)
    # Inyectamos +625g (placas) en una fila de evaluación fuera de fit
    gold[35, 0, 2] += 625.0
    fit_mask = np.arange(NM) < 30
    coeffs = fit_residual_gold_coeffs(gold, cs, kills, assists, fit_mask)
    resid = apply_residual_gold(gold, cs, kills, assists, coeffs)
    assert np.abs(resid[:30]).max() < 1e-2
    assert resid[35, 0, 2] == pytest.approx(625.0, abs=1e-1)
    # Si corrompemos las filas fuera de fit, los coeficientes no cambian
    gold_mut = gold.copy()
    gold_mut[30:] += 9999.0
    coeffs2 = fit_residual_gold_coeffs(gold_mut, cs, kills, assists, fit_mask)
    assert np.allclose(coeffs, coeffs2)


def test_causal_tempo_sin_fuga_temporal_ni_intrapartida():
    rng = np.random.default_rng(11)
    NM = 50
    t = np.arange(NM, dtype=np.int64) * 1000
    mids = [f"M{i:03d}" for i in range(NM)]
    champs = rng.integers(1, 10, size=(NM, 2, 5))
    players = np.array([f"P{p}" for p in rng.integers(0, 15, size=NM * 10)]).reshape(NM, 2, 5)
    dur_min = rng.uniform(18, 38, size=NM)
    y = rng.integers(0, 2, size=NM)
    g10 = rng.normal(0, 1, size=(NM, 5))
    g15 = rng.normal(0, 1, size=(NM, 5))
    g20 = rng.normal(0, 1, size=(NM, 5))

    res0 = causal_champ_player_tempo(t, mids, champs, players, dur_min, y, g10, g15, g20)

    # La primera partida debe tener valores neutros exactos (0)
    assert np.allclose(res0["cbias"][0], 0.0)
    assert np.allclose(res0["pbias"][0], 0.0)
    assert np.allclose(res0["cslope"][0], 0.0)

    # Mutar partida `cut` y todo lo futuro NO debe alterar filas <= cut
    cut = 25
    y_mut = y.copy()
    y_mut[cut:] = 1 - y_mut[cut:]
    dur_mut = dur_min.copy()
    dur_mut[cut:] = 36.0
    g15_mut = g15.copy()
    g15_mut[cut:] += 50.0
    g20_mut = g20.copy()
    g20_mut[cut:] -= 50.0
    res1 = causal_champ_player_tempo(t, mids, champs, players, dur_mut, y_mut, g10, g15_mut, g20_mut)

    for k in ("cbias", "cslope", "pbias", "pvol"):
        assert np.allclose(res0[k][: cut + 1], res1[k][: cut + 1])
    assert not np.allclose(res0["pbias"][cut + 1 :], res1["pbias"][cut + 1 :])


def _datos_sinteticos(NM=8, seed=42):
    rng = np.random.default_rng(seed)
    tl_raw = rng.normal(5.0, 1.5, size=(NM, 2, 5, 3, len(TIMELINE_STATS))).astype(np.float32)
    gold_resid = rng.normal(0.0, 0.3, size=(NM, 2, 5, 3)).astype(np.float32)
    L_sym = rng.normal(0.0, 0.4, size=NM).astype(np.float32)

    champs = rng.integers(1, 6, size=(NM, 2, 5))
    atrib_df = pd.DataFrame({
        "champion_id": [1, 2, 3, 4, 5],
        **{c: rng.uniform(0, 3, size=5) for c in ATRIB_COLS},
    })
    ddragon_df = pd.DataFrame({
        "champion_id": [1, 2, 3, 4, 5],
        **{c: rng.uniform(1, 100, size=5) for c in GROWTH_COLS},
    })
    comp = build_composition_features(champs, atrib_df, ddragon_df)
    tempo = {
        "cbias": rng.normal(0, 0.05, size=(NM, 2, 5)).astype(np.float32),
        "cslope": rng.normal(0, 0.3, size=(NM, 2, 5)).astype(np.float32),
        "pbias": rng.normal(0, 0.05, size=(NM, 2, 5)).astype(np.float32),
        "pvol": rng.uniform(0.5, 2.0, size=(NM, 2, 5)).astype(np.float32),
    }
    p_cols = ["acum_d_gold_15", "v20_d_gold_15", "wr_acum", "estreno", "offrole", "acum_deaths_pm"]
    player_pre = {
        "p_mat": rng.normal(0, 1, size=(NM, 2, 5, len(p_cols))).astype(np.float32),
        "p_cols": p_cols,
    }
    exp_lane = {f"{s}_{m}": rng.normal(0, 0.5, size=(NM, 5)).astype(np.float32) for m in (10, 15, 20) for s in ("gold", "xp", "cs", "damage")}
    exp_trans = {f"trans_gold_{m}": rng.normal(0, 0.5, size=(NM, 5)).astype(np.float32) for m in (15, 20)}
    return dict(L_sym=L_sym, tl_raw=tl_raw, gold_resid=gold_resid, exp_lane=exp_lane,
                exp_trans=exp_trans, comp=comp, tempo=tempo, player_pre=player_pre,
                champs=champs, atrib_df=atrib_df, ddragon_df=ddragon_df)


def _swap_datos(d):
    swap_idx = [1, 0]
    exp_lane_s = {k: -v for k, v in d["exp_lane"].items()}
    exp_trans_s = {k: -v for k, v in d["exp_trans"].items()}
    comp_s = build_composition_features(d["champs"][:, swap_idx], d["atrib_df"], d["ddragon_df"])
    tempo_s = {k: v[:, swap_idx] for k, v in d["tempo"].items()}
    player_s = {"p_mat": d["player_pre"]["p_mat"][:, swap_idx], "p_cols": d["player_pre"]["p_cols"]}
    return dict(
        L_sym=-d["L_sym"],
        tl_raw=d["tl_raw"][:, swap_idx],
        gold_resid=d["gold_resid"][:, swap_idx],
        exp_lane=exp_lane_s,
        exp_trans=exp_trans_s,
        comp=comp_s,
        tempo=tempo_s,
        player_pre=player_s,
    )


@pytest.mark.parametrize("minute", [10, 15, 20])
def test_extract_minute_features_antisimetria_y_sin_fuga_futura(minute):
    d = _datos_sinteticos()
    ds = _swap_datos(d)

    f1 = extract_minute_features(
        minute, d["L_sym"], d["tl_raw"], d["gold_resid"],
        exp_lane=d["exp_lane"], exp_trans=d["exp_trans"],
        comp=d["comp"], tempo=d["tempo"], player_pre=d["player_pre"],
    )
    f2 = extract_minute_features(
        minute, ds["L_sym"], ds["tl_raw"], ds["gold_resid"],
        exp_lane=ds["exp_lane"], exp_trans=ds["exp_trans"],
        comp=ds["comp"], tempo=ds["tempo"], player_pre=ds["player_pre"],
    )

    bad_anti = [n for i, n in enumerate(f1["names_anti"]) if not np.allclose(f1["X_anti"][:, i], -f2["X_anti"][:, i], atol=1e-5)]
    assert not bad_anti, f"Columnas no antisimétricas al minuto {minute}: {bad_anti}"
    bad_sym = [n for i, n in enumerate(f1["names_sym"]) if not np.allclose(f1["X_sym"][:, i], f2["X_sym"][:, i], atol=1e-5)]
    assert not bad_sym, f"Columnas no simétricas al minuto {minute}: {bad_sym}"
    assert np.allclose(f1["lane_b"], f2["lane_r"], atol=1e-5)
    assert np.allclose(f1["lane_r"], f2["lane_b"], atol=1e-5)

    # Cero fuga de minutos futuros: mutar minutos > minute no cambia nada
    if minute < 20:
        mi = (10, 15, 20).index(minute)
        tl_fut = d["tl_raw"].copy()
        tl_fut[:, :, :, mi + 1 :, :] = 9999.0
        gr_fut = d["gold_resid"].copy()
        gr_fut[:, :, :, mi + 1 :] = -9999.0
        f_mut = extract_minute_features(
            minute, d["L_sym"], tl_fut, gr_fut,
            exp_lane=d["exp_lane"], exp_trans=d["exp_trans"],
            comp=d["comp"], tempo=d["tempo"], player_pre=d["player_pre"],
        )
        assert np.allclose(f1["X_anti"], f_mut["X_anti"])
        assert np.allclose(f1["X_sym"], f_mut["X_sym"])
        assert np.allclose(f1["lane_b"], f_mut["lane_b"])


def test_ingamenet_antisimetria_exacta():
    d = _datos_sinteticos()
    f = extract_minute_features(
        15, d["L_sym"], d["tl_raw"], d["gold_resid"],
        exp_lane=d["exp_lane"], exp_trans=d["exp_trans"],
        comp=d["comp"], tempo=d["tempo"], player_pre=d["player_pre"],
    )
    torch.manual_seed(0)
    net = InGameNet(
        d_lane=f["lane_b"].shape[-1],
        d_anti=f["X_anti"].shape[1],
        d_sym=f["X_sym"].shape[1],
        d_aux_anti=3,
        d_aux_sym=1,
    )
    with torch.no_grad():
        net.b_side.fill_(-0.12)
    net.eval()

    lb = torch.tensor(f["lane_b"])
    lr = torch.tensor(f["lane_r"])
    xa = torch.tensor(f["X_anti"])
    xs = torch.tensor(f["X_sym"])

    with torch.no_grad():
        log1, aux_a1, aux_s1 = net(lb, lr, xa, xs)
        log2, aux_a2, aux_s2 = net(lr, lb, -xa, xs)
        b = net.b_side.detach()

    assert torch.allclose(log1 - b, -(log2 - b), atol=1e-5)
    assert torch.allclose(aux_a1, -aux_a2, atol=1e-5)
    assert torch.allclose(aux_s1, aux_s2, atol=1e-5)
