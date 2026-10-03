"""Recursive estimation (spec section 7) on a small synthetic panel."""
import json

import numpy as np
import pandas as pd
import pytest

from team_synergy.config import load_config
from team_synergy.model.recursive import estimate_window, run_recursive, save_outputs
from team_synergy.synthetic import make_synthetic_panel


@pytest.fixture(scope="module")
def setup():
    df, _ = make_synthetic_panel(n_teams=12, n_seasons=4, roster=30, seed=0)
    cfg = load_config()
    res = run_recursive(df, cfg, min_end=int(df["season"].min()) + 1)
    return df, cfg, res


def test_structure(setup):
    df, cfg, res = setup
    s0 = int(df["season"].min())
    ws = res["window_summary"]
    assert list(ws["window_end"]) == [s0 + 1, s0 + 2, s0 + 3]
    mr = res["metrics_recursive"]
    for T, g in mr.groupby("window_end"):
        assert (g["season"] == T).all()
        assert len(g) == (df["season"] == T).sum()
    tr = res["team_recursive"]
    assert (tr["season"] == tr["window_end"]).all() and {"tc_war", "eps"} <= set(tr.columns)
    assert {"rho", "rho_se", "at_bound", "ratio", "corr_tc_eps"} <= set(ws.columns)


def test_full_equals_estimate_window(setup):
    df, cfg, res = setup
    w = estimate_window(df, cfg)
    assert res["full"].sar.rho == pytest.approx(w.sar.rho)
    pd.testing.assert_frame_equal(res["metrics_full"], w.stints)
    pd.testing.assert_frame_equal(res["team_full"], w.teams.reset_index())
    assert len(res["player_season_full"]) == len(w.player_seasons)
    assert set(w.lam_als.index) == set(w.lam_prob.index)


def test_lambda_ranks(setup):
    _, _, res = setup
    lw = res["lambda_windows"]
    assert lw["rank"].between(0, 100).all()
    for _, g in lw.groupby("window_end"):
        assert g["rank"].max() == 100 and g["rank"].min() == 0
        top = g.loc[g["rank"].idxmax()]
        assert abs(top["lam_prob"]) == g["lam_prob"].abs().max()
    assert lw[["lam_als", "lam_prob"]].notna().all().all()


def test_save_outputs(setup, tmp_path):
    _, cfg, res = setup
    out = save_outputs(res, tmp_path, cfg, "synthetic")
    for n in ["metrics_recursive", "metrics_full", "team_recursive", "team_full",
              "player_season_full", "lambda_windows", "window_summary"]:
        assert (out / f"{n}.parquet").exists()
    assert (out / "window_summary.csv").exists() and (out / "lambda_windows.csv").exists()
    meta = json.loads((out / "run_metadata.json").read_text())
    assert meta["war_source"] == "synthetic" and meta["seed"] == cfg["seed"]
    assert meta["row_counts"]["window_summary"] == 3 and "timestamp" in meta
    assert pd.read_parquet(out / "metrics_recursive.parquet").shape[0] == len(res["metrics_recursive"])
