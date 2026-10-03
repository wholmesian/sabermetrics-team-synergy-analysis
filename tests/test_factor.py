import numpy as np
import pytest
from scipy.stats import spearmanr

from team_synergy.model.decompose import decompose, sign_convention_diagnostics
from team_synergy.model.factor_als import fit_factor_als, multi_start_report
from team_synergy.model.factor_prob import fit_factor_prob
from team_synergy.model.network import build_blocks
from team_synergy.model.residuals import player_residuals
from team_synergy.model.sar import fit_sar
from team_synergy.model.team_reg import fit_team_regression
from team_synergy.synthetic import make_synthetic_panel


def _pipeline(seed, conv="consistent"):
    df, truth = make_synthetic_panel(seed=seed)
    reg = fit_team_regression(df)
    y = player_residuals(df, reg, conv)["y"].to_numpy()
    blocks = build_blocks(df)
    sar = fit_sar(y, blocks)
    return df, truth, reg, blocks, sar


@pytest.fixture(scope="module")
def setup():
    return _pipeline(0)


def _aligned_truth(truth):
    tl = truth["lam"]
    return tl * np.sign(tl[tl.abs().idxmax()])  # same sign convention as the estimates


def test_als_conventions(setup):
    df, truth, reg, blocks, sar = setup
    w = df["weight"].to_numpy()
    r = fit_factor_als(df, sar.v, w)
    print(f"ALS iters={r.n_iter} explained={r.explained_share:.6f} "
          f"spearman_vs_truth={spearmanr(r.lam, _aligned_truth(truth)).statistic:.3f}")
    assert r.converged and r.n_iter < 5000
    assert r.explained_share > 0.99  # paper: "almost all"
    assert abs(r.lam.sum()) < 1e-10
    assert r.factors["tp"].std(ddof=0) == pytest.approx(1.0)
    assert r.lam[r.lam.abs().idxmax()] > 0
    assert len(r.g) == len(df) and r.method == "als"
    r2 = fit_factor_als(df, sar.v, w, scale="lambda_norm")
    assert np.linalg.norm(r2.lam) == pytest.approx(1.0)
    assert abs(r2.lam.sum()) < 1e-10
    # g is invariant to the scale convention
    assert np.abs(r2.g - r.g).max() < 1e-6
    # fixed point: single-cell rows reproduce v exactly
    assert np.abs(r.g - sar.v).max() < 1e-7


def test_prob_recovers_lambda(setup):
    df, truth, reg, blocks, sar = setup
    r = fit_factor_prob(df, sar.v, tol=1e-7)
    rho_s = spearmanr(r.lam, _aligned_truth(truth)).statistic
    print(f"prob iters={r.n_iter} explained={r.explained_share:.4f} spearman={rho_s:.3f}")
    assert r.converged
    assert abs(r.lam.sum()) < 1e-10
    assert r.extra["Sigma"][1, 1] == pytest.approx(1.0)       # prior Var(tp) = 1
    assert 0.5 < r.factors["tp"].std() <= 1.0                 # posterior means shrink
    assert r.lam[r.lam.abs().idxmax()] > 0
    assert rho_s > 0.8  # spec section 9 completion criterion
    assert np.abs(r.g - sar.v).max() > 1e-6  # shrinkage: g != v


def test_multi_start_report(setup):
    df, _, _, _, sar = setup
    rep = multi_start_report(df, sar.v, df["weight"].to_numpy(), n_starts=3, seed=0)
    assert rep["spearman_abs"].shape == (3, 3)
    assert np.allclose(np.diag(rep["spearman_abs"]), 1)
    assert set(rep["summary_abs"]) == {"min", "q25", "median", "q75", "max"}
    print("multi-start |lam| summary:", rep["summary_abs"], "lam:", rep["summary_lam"])


@pytest.mark.parametrize("conv", ["consistent", "paper_literal"])
def test_sign_convention_diagnostics(conv):
    df, _, reg, blocks, sar = _pipeline(0, conv)
    fit = fit_factor_als(df, sar.v, df["weight"].to_numpy())
    dec = decompose(df, blocks, sar.Phi, fit.g, beta=reg.beta)
    d = sign_convention_diagnostics(df, dec, "estimated")
    print(conv, d)
    if conv == "consistent":
        assert d["ratio"] < 1 and d["corr_tc_eps"] > 0
    else:  # reported only; the wrong sign should not look like the paper
        assert d["ratio"] > 1 or d["corr_tc_eps"] < 0
