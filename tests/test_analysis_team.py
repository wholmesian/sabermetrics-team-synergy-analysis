"""Tests for the team-level paper-reproduction analyses (spec section 8) on synthetic data."""
import numpy as np
import pytest

from team_synergy.analysis import org_culture, pecota, table1, tc_persistence
from team_synergy.analysis.inputs import synthetic_inputs
from team_synergy.analysis.result import AnalysisResult

from conftest import SYN_SEASONS as N_SEASONS, SYN_TEAMS as N_TEAMS
NB = 30


@pytest.fixture(scope="module")
def inp(syn_inputs):
    return syn_inputs


def _check_keys(res):
    assert isinstance(res, AnalysisResult)
    cf = res.checks_frame()
    assert list(cf.columns) == ["item", "value", "expected", "passed"]
    assert len(cf) > 0


def test_table1(inp):
    res = table1.run(inp, n_boot=NB, seed=1)
    _check_keys(res)
    t = res.tables["table1"]
    n = N_SEASONS * N_TEAMS
    assert len(t) == 2 and (t["n_obs"] == n).all()
    assert t["n_folds"].iloc[0] == max(2, min(57, n // 10))
    assert t["r2"].iloc[1] >= t["r2"].iloc[0]
    assert (t["pseudo_r2"] <= t["r2"] + 1e-9).all()
    assert (t["se_beta"] > 0).all()
    assert len(res.tables["fig2_resid"]) == 2 * n
    assert {"spec", "x", "density"} <= set(res.tables["fig2_density"].columns)
    assert res.tables["fig2_ratio"]["ratio"].iloc[0] < 1


def test_kfold_pseudo_r2_small_n_and_seed():
    rng = np.random.default_rng(0)
    x = rng.normal(size=25)
    y = 2 * x + rng.normal(size=25)
    a = table1.kfold_pseudo_r2(x, y, seed=3)
    assert a["n_folds"] == 2
    assert a == table1.kfold_pseudo_r2(x, y, seed=3)
    assert table1.kfold_pseudo_r2(x, y, n_folds=57, seed=3)["n_folds"] == 25


def test_org_culture(inp):
    res = org_culture.run(inp)
    _check_keys(res)
    sc, sm = res.tables["fig3_scores"], res.tables["fig3_summary"]
    assert sc["score"].between(0, 100).all()
    assert set(sc["source"]) == {"prob", "als"}
    assert set(sm.columns) >= {"franch_id", "median", "q1", "q3", "min", "max", "n_windows"}
    p = sm[sm["primary"]]
    assert len(p) == N_TEAMS and p["median"].is_monotonic_decreasing
    assert (p["n_windows"] == N_SEASONS).all()
    assert res.checks[-1]["passed"] is None  # real-data-only check


def test_tc_persistence(inp):
    res = tc_persistence.run(inp, n_boot=NB, seed=2)
    _check_keys(res)
    pers = res.tables["persistence"]
    assert (pers["n_obs"] == N_TEAMS * (N_SEASONS - 1)).all()
    assert list(pers["term"]) == ["const", "tc_lag"]
    fig4 = res.tables["fig4"]
    assert len(fig4) == N_TEAMS * N_SEASONS
    assert (fig4["corner"] != "").sum() == 20 and (fig4["label"] != "").sum() == 20
    bars = res.tables["fig5_bars"]
    assert len(bars) == N_TEAMS and bars["synergy_wins"].is_monotonic_decreasing
    assert len(res.tables["persistence_resid"]) == N_TEAMS * (N_SEASONS - 1)
    full = tc_persistence.run(inp, n_boot=NB, source="full")
    assert len(full.tables["fig4"]) == N_TEAMS * N_SEASONS
    with pytest.raises(ValueError):
        tc_persistence.run(inp, n_boot=NB, source="x")


def test_pecota(inp):
    res = pecota.run(inp, n_boot=NB, seed=3)
    _check_keys(res)
    t = res.tables["table3"]
    n1 = t.loc[t["spec"].str.startswith("(1)"), "n_obs"].iloc[0]
    n2 = t.loc[t["spec"].str.startswith("(2)"), "n_obs"].iloc[0]
    # PECOTA exists for seasons >= first + 10; spec 1 also needs PECOTA_{t-1}
    n_pec = N_SEASONS - 10
    assert n2 == n_pec * N_TEAMS and n1 == (n_pec - 1) * N_TEAMS
    assert set(t["term"]) >= {"const", "pecota_lag", "team_war_lag", "tc_lag", "W_lag", "pecota", "proj_tc"}
    s = res.tables["oos_summary"]
    assert {"gain", "se", "dm_stat", "dm_pvalue", "dm_lags", "mae_pecota", "mae_model"} <= set(s.columns)
    oos = res.tables["oos"]
    assert len(oos) == s["n"].iloc[0] > 0
    assert np.isclose(s["gain"].iloc[0], (oos["abs_err_pecota"] - oos["abs_err_model"]).mean())
    assert oos["season"].min() > inp.pecota["season"].min()  # trained on earlier seasons only


def test_pecota_skipped_without_data(inp):
    import dataclasses
    res = pecota.run(dataclasses.replace(inp, pecota=None))
    assert res.tables == {} and "skipped" in res.checks[0]["value"]
