import numpy as np
import pytest

from team_synergy.model.network import build_blocks
from team_synergy.model.residuals import player_residuals
from team_synergy.model.sar import concentrated_loglik, fit_sar
from team_synergy.model.team_reg import fit_team_regression
from team_synergy.synthetic import make_synthetic_panel


@pytest.fixture(scope="module")
def setup():
    df, truth = make_synthetic_panel(seed=0)
    return df, truth, build_blocks(df, history="past")


def test_rho_from_true_y(setup):
    df, truth, blocks = setup
    r = fit_sar(truth["y_true"], blocks)
    assert not r.at_bound
    assert r.se > 0
    assert len(r.profile) == 200
    assert r.loglik == pytest.approx(concentrated_loglik(r.rho, truth["y_true"], blocks))
    assert abs(r.rho - 0.3) < 0.12  # single draw: sampling sd ~0.04 (heavy-tailed tp*lambda)


def test_rho_recovery_averaged_over_seeds():
    """Quasi-ML rho_hat has sampling sd ~0.04 on one 10-season panel (seed 0 gives
    0.22), so unbiasedness is checked on the mean over 6 seeds, with the pipeline y
    (team_reg -> residuals) and the true y."""
    true_hat, pipe_hat = [], []
    for seed in range(6):
        df, truth = make_synthetic_panel(seed=seed)
        blocks = build_blocks(df, history="past")
        true_hat.append(fit_sar(truth["y_true"], blocks).rho)
        y = player_residuals(df, fit_team_regression(df))["y"].to_numpy()
        r = fit_sar(y, blocks)
        assert not r.at_bound
        pipe_hat.append(r.rho)
    true_hat, pipe_hat = np.array(true_hat), np.array(pipe_hat)
    assert abs(true_hat.mean() - 0.3) < 0.05
    assert abs(pipe_hat.mean() - 0.3) < 0.05
    # alpha_hat/beta_hat sampling error (e.g. seed 4: 43.6 / 1.11) adds a smooth
    # weight*team-level term to y, which loads on the near-unit eigenvector of A and
    # moves rho_hat by up to ~0.08; so only a coarse per-seed bound is meaningful.
    assert np.abs(true_hat - pipe_hat).max() < 0.1


def test_pipeline_regression_params(setup):
    df, _, _ = setup
    reg = fit_team_regression(df)
    assert abs(reg.beta - 1) < 0.15 and abs(reg.alpha - 48) < 5


def test_block_properties(setup):
    df, _, blocks = setup
    assert sum(len(b.idx) for b in blocks) == len(df)
    assert len(np.unique(np.concatenate([b.idx for b in blocks]))) == len(df)
    for b in blocks:
        assert b.eigvals.min() >= -1 - 1e-9 and b.eigvals.max() <= 1 + 1e-9
        s = b.A.sum(axis=1)
        nz = b.alpha.sum(axis=1) > 0
        assert np.allclose(s[nz], 1)
        assert np.allclose(b.alpha, b.alpha.T)


def test_full_history_dominates_past(setup):
    df, _, past = setup
    full = build_blocks(df, history="full")
    for p, f in zip(past, full):
        assert (f.franch_id, f.season) == (p.franch_id, p.season)
        assert (f.alpha >= p.alpha - 1e-12).all()


def test_kernel_option(setup):
    df, _, past = setup
    k = build_blocks(df, history="past", kernel_h=2.0)
    assert all((kb.alpha <= pb.alpha + 1e-12).all() for kb, pb in zip(k, past))
