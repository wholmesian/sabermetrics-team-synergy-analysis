import numpy as np
import pytest

from team_synergy.model.residuals import check_residual_identity, player_residuals
from team_synergy.model.team_reg import fit_team_regression
from team_synergy.synthetic import make_synthetic_panel


@pytest.fixture(scope="module")
def panel():
    return make_synthetic_panel(seed=1)[0]


@pytest.mark.parametrize("conv", ["consistent", "paper_literal"])
@pytest.mark.parametrize("mode", ["estimated", "fixed_one"])
def test_sum_y_identity(panel, conv, mode):
    reg = fit_team_regression(panel, beta_mode=mode)
    out = player_residuals(panel, reg, sign_convention=conv)
    assert check_residual_identity(out, reg, conv) < 1e-10
    s = out.groupby(["franch_id", "season"])["y"].sum()
    sign = -1 if conv == "consistent" else 1
    eps = reg.teams["eps"].reindex(s.index)
    assert np.abs(s - sign * eps / reg.beta).max() < 1e-10


def test_fixed_one_beta(panel):
    reg = fit_team_regression(panel, beta_mode="fixed_one")
    assert reg.beta == 1.0
    assert reg.alpha == pytest.approx((reg.teams["W"] - reg.teams["war_sum"]).mean())
    assert abs(reg.teams["eps"].mean()) < 1e-9


def test_estimated_ols_residual_mean_zero(panel):
    reg = fit_team_regression(panel)
    assert abs(reg.teams["eps"].mean()) < 1e-9
    assert reg.se_beta > 0


# ---- spec section 6 identities (decompose) ---------------------------------------
from team_synergy.model.decompose import (  # noqa: E402
    decompose, player_season_metrics, team_metrics)
from team_synergy.model.factor_als import fit_factor_als  # noqa: E402
from team_synergy.model.network import build_blocks  # noqa: E402
from team_synergy.model.sar import fit_sar  # noqa: E402


@pytest.fixture(scope="module")
def pipeline():
    df, _ = make_synthetic_panel(seed=1)
    reg = fit_team_regression(df)
    y = player_residuals(df, reg)["y"].to_numpy()
    blocks = build_blocks(df)
    sar = fit_sar(y, blocks)
    fit = fit_factor_als(df, sar.v, df["weight"].to_numpy())
    f = df[["player_id", "season"]].merge(fit.factors, on=["player_id", "season"], how="left")
    g_char = f["c"].to_numpy()
    g_team = fit.g - g_char
    dec = decompose(df, blocks, sar.Phi, fit.g, g_char, g_team, beta=reg.beta)
    return df, reg, y, dec, fit


def test_team_identities(pipeline):
    df, reg, _, dec, _ = pipeline
    tm = team_metrics(dec, reg.beta)
    assert tm["pc_war"].abs().max() < 1e-8
    assert (tm["war_plus"] - tm["war"]).abs().max() < 1e-8


def test_y_equals_own_plus_in(pipeline):
    df, _, y, dec, _ = pipeline
    n_obs = df.groupby(["player_id", "season"])["franch_id"].transform("size").to_numpy()
    single = n_obs == 1
    assert single.sum() > 0.5 * len(df)
    err = np.abs(y - (dec["own"] + dec["in_deg"]).to_numpy())[single]
    assert err.max() < 1e-8


def test_player_season_sums_and_char_team_split(pipeline):
    df, _, _, dec, _ = pipeline
    ps = player_season_metrics(dec)
    assert ps.index.is_unique and (ps.index.get_level_values(1) >= 0).all()
    tr = dec.groupby(["player_id", "season"]).size()
    assert (tr > 1).any()
    manual = dec.groupby(["player_id", "season"])["pc_war"].sum()
    assert np.abs(ps["pc_war"] - manual).max() < 1e-12
    assert len(ps) == len(tr)
    # a traded player-season equals the sum over its stints
    key = tr[tr > 1].index[0]
    rows = dec[(dec.player_id == key[0]) & (dec.season == key[1])]
    assert ps.loc[key, "in_deg"] == pytest.approx(rows["in_deg"].sum())
    assert np.abs(dec["pc_war"] - dec["pc_war_char"] - dec["pc_war_team"]).max() < 1e-9
