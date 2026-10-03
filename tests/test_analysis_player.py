"""Player-level paper-reproduction analyses (spec section 8; Figs. 6-12, Tables 4-5)."""
import numpy as np
import pandas as pd
import pytest

from team_synergy.analysis import intangibles, pc_persistence, player_eval, profiles, ross, salary
from team_synergy.analysis.inputs import synthetic_inputs
from team_synergy.stats.regression import ols


@pytest.fixture(scope="module")
def inp(syn_inputs):
    return syn_inputs


@pytest.fixture(scope="module")
def t4(inp):
    return pc_persistence.run(inp, n_boot=20, seed=0, max_jack=20)


def _check_ok(res):
    for c in res.checks:
        assert set(c) == {"item", "value", "expected", "passed"}
        assert c["passed"] in (True, False, None)


def test_tier_assignment_hand_case():
    t = pc_persistence.assign_tier([0.99, 1.0, 3.99, 4.0, -2.0, 8.0])
    assert list(t) == ["Scrub", "Role", "Role", "Star", "Scrub", "Star"]


def test_player_eval(inp):
    r = player_eval.run(inp, n_boot=20, seed=0)
    ps = inp.player_seasons
    assert len(r.tables["fig6"]) == len(r.tables["fig7"]) == len(ps)
    assert {"war", "war_minus", "war_plus"} <= set(r.tables["fig6"].columns)
    assert list(r.tables["fig6_thresholds"]["war"]) == [1.0, 4.0]
    f8 = r.tables["fig8"]
    last = ps["season"].max()
    n_active = ps.loc[ps["season"] == last, "player_id"].nunique()
    n_keep = int(np.ceil(n_active / 2))
    k = int(round(0.25 * n_keep))
    assert (f8["side"] == "top").sum() == (f8["side"] == "bottom").sum() == k
    top = f8[f8["side"] == "top"]["pc_war"]
    assert top.is_monotonic_decreasing and top.min() >= f8[f8["side"] == "bottom"]["pc_war"].max()
    np.testing.assert_allclose(f8["pc_war"], f8["pc_war_char"] + f8["pc_war_team"], atol=1e-9)
    assert player_eval.run(inp, n=5).tables["fig8"].groupby("side").size().eq(5).all()
    _check_ok(r)
    assert r.checks[0]["item"].startswith("corr") and r.checks[-1]["passed"] is None


def test_table4(inp, t4):
    d = pc_persistence.build_table4(inp.player_seasons)
    assert t4.tables["table4_fit"]["nobs"].iloc[0] == len(d.frame)
    r2 = t4.tables["table4_fit"].set_index("spec")["r2"]
    assert r2[2] >= r2[1]
    tab = t4.tables["table4"]
    assert set(tab["term"][tab.spec == 1]) == {"rho_Scrub", "rho_Role", "rho_Star", "d_Scrub", "d_Role", "d_Star"}
    assert {"war", "mlb_exp", "team_exp"} <= set(tab["term"][tab.spec == 2])
    assert "d_Scrub" not in set(tab["term"][tab.spec == 2])  # absorbed by position indicators
    assert (tab["ci_low"] <= tab["ci_high"]).all()
    # the sample only has players with a previous consecutive season
    keyed = set(zip(inp.player_seasons.player_id, inp.player_seasons.season))
    assert all((p, s - 1) in keyed for p, s in zip(d.frame.player_id, d.frame.season))
    # spec (1) point estimates equal the generic OLS helper
    fr = d.frame.assign(**{f"d_{k}": (d.frame.tier == k).astype(float) for k in pc_persistence.TIERS},
                        **{f"rho_{k}": np.where(d.frame.tier == k, d.frame.pc_lag, 0.0) for k in pc_persistence.TIERS})
    cols = [f"rho_{k}" for k in pc_persistence.TIERS] + [f"d_{k}" for k in pc_persistence.TIERS]
    ref = ols(fr, "pc_war", cols, const=False)
    got = tab[tab.spec == 1].set_index("term")["coef"]
    np.testing.assert_allclose(got[cols].to_numpy(), ref.params[cols].to_numpy(), atol=1e-6)
    f9, lines = t4.tables["fig9"], t4.tables["fig9_lines"]
    assert f9["career_resid"].sum() == pytest.approx(d.resid[1].sum(), abs=1e-6)
    assert lines["value"].iloc[0] < lines["value"].iloc[1]
    _check_ok(t4)


def test_xi_residual_mean_zero(inp):
    xi = pc_persistence.table4_residuals(inp)
    assert abs(xi["xi"].mean()) < 1e-6 and xi["xi"].notna().all()
    r = intangibles.run(inp)
    assert r.tables["xi"]["xi"].equals(xi["xi"])
    f = r.tables["fig11"]
    assert (f["side"] == "top").sum() == (f["side"] == "bottom").sum() >= 1
    assert f["xi"].notna().all()
    _check_ok(r)


def test_profiles_grid(inp):
    r = profiles.run(inp, n_boot=20, seed=0, max_jack=20)
    f = r.tables["fig10"]
    pos = set(inp.player_seasons["pos"].dropna())
    assert set(f["pos"]) == pos
    assert set(f["age"]) == set(range(20, 41)) and len(f) == len(pos) * 21
    assert (f["ci_low"] <= f["ci_high"]).all() and f["estimate"].notna().all()
    # average marginal prediction at an observed age equals the in-sample mean fitted value
    d = pc_persistence.build_table4(inp.player_seasons)
    p = "SP" if "SP" in pos else sorted(pos)[0]
    m = (d.frame["pos"] == p).to_numpy()
    fitted = d.designs[2].X @ d.beta[2]
    z = d.frame.loc[m, "z"].to_numpy()
    cols = d.designs[2].age_cols[p]
    th = d.beta[2][cols]
    for a in (25, 35):
        za = (a - 30) / 10
        direct = np.mean(fitted[m] - sum(th[k] * z ** (k + 1) for k in range(4)) + sum(th[k] * za ** (k + 1) for k in range(4)))
        assert f[(f.pos == p) & (f.age == a)]["estimate"].iloc[0] == pytest.approx(direct, abs=1e-8)
    _check_ok(r)


def test_salary(inp):
    assert list(salary.fa_class([0, 2, 3, 5, 6, 12])) == ["FA0", "FA0", "FA1", "FA1", "FA2", "FA2"]
    v, ncl = salary.real_salary([100.0, 100.0, 100.0], [2000, 2016, 2018], {2000: 100.0, 2016: 200.0}, 2016)
    np.testing.assert_allclose(v, [200.0, 100.0, 100.0])
    assert ncl == 1
    r = salary.run(inp, n_boot=20, seed=0, max_jack=20)
    t5 = r.tables["table5"]
    assert set(t5.loc[t5.spec == 1, "term"]) >= {"CumWAR", "CumpcWAR", "ind", "TeamExp"}
    s2 = t5[(t5.spec == 2) & t5.term.isin(["CumPcMinusXi", "CumXi"])]
    assert set(s2["fa"]) == {"FA1", "FA2"}  # FA0 interactions dropped in spec (2)
    assert r.tables["table5_fit"]["r2_total"].between(0, 1).all()
    d, _ = salary.build_salary_frame(inp)
    debut = inp.players.set_index("player_id")["debut_season"]
    assert (d["player_id"].map(debut) >= inp.player_seasons["season"].min()).all()
    assert (d["fa"] == salary.fa_class(d["n_prior"])).all()
    np.testing.assert_allclose(d["cum_pc"], d["cum_pcx"] + d["cum_xi"], atol=1e-9)
    # FE-demeaned point estimates equal linearmodels AbsorbingLS (via the ols helper) on a design without FE-collinear controls
    X, names = salary._design(d, 1)
    keys = [n for n in names if n.startswith(("Cum", "ind_", "TeamExp"))]
    df = pd.DataFrame(X, columns=names).assign(y=d["ln_sal"].to_numpy(), pid=d["player_id"].to_numpy())
    ref = ols(df, "y", keys, absorb=["pid"])
    codes, _ = pd.factorize(df["pid"])
    Xt = salary._demean(df[keys].to_numpy(), codes, codes.max() + 1)
    yt = salary._demean(df["y"].to_numpy(), codes, codes.max() + 1)
    np.testing.assert_allclose(salary._wls_dense(Xt, yt), ref.params[keys].to_numpy(), atol=1e-6)
    _check_ok(r)
    skipped = salary.run(type(inp)(**{**inp.__dict__, "salaries": None}))
    assert skipped.tables == {} and skipped.checks[0]["passed"] is None


def test_ross(inp):
    r = ross.run(inp)
    pid = r.tables["player"]["player_id"].iloc[0]
    f = r.tables["fig12"]
    n_st = (inp.stints_full["player_id"] == pid).sum()
    assert len(f) == n_st and (f["player_id"] == pid).all()
    assert f["label"].iloc[0].startswith(str(f["season"].iloc[0]) + " ")
    ps = inp.player_seasons
    other = ps["player_id"].iloc[0]
    assert (ross.run(inp, player_id=other).tables["fig12"]["player_id"] == other).all()
    _check_ok(r)
