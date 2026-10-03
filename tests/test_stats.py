"""Bootstrap (BCa), Diebold-Mariano and regression helpers (spec section 8)."""
import numpy as np
import pandas as pd
import pytest

from team_synergy.stats.bootstrap import cluster_bootstrap
from team_synergy.stats.dm_test import diebold_mariano
from team_synergy.stats.regression import ols


def _clustered(seed=0, G=40, m=5, mu=2.0):
    rng = np.random.default_rng(seed)
    cl = np.repeat(np.arange(G), m)
    x = mu + rng.normal(0, 1, G)[cl] + rng.normal(0, 1, G * m)
    return pd.DataFrame({"cl": cl, "x": x})


def test_bca_covers_mean_and_matches_percentile():
    df = _clustered()
    r = cluster_bootstrap(df, "cl", lambda d: np.array([d["x"].mean(), d["x"].std()]), n_boot=300, seed=1)
    assert r.draws.shape == (300, 2)
    assert r.ci_low[0] < 2.0 < r.ci_high[0]
    pl, ph = np.quantile(r.draws[:, 0], [0.025, 0.975])
    assert abs(r.ci_low[0] - pl) < 0.25 and abs(r.ci_high[0] - ph) < 0.25
    assert r.estimate[0] == pytest.approx(df["x"].mean())
    assert r.se[0] == pytest.approx(r.draws[:, 0].std(ddof=1))


def test_bca_grouped_jackknife_and_determinism():
    df = _clustered(G=30)
    f = lambda d: np.array([d["x"].mean()])
    a = cluster_bootstrap(df, "cl", f, n_boot=100, seed=3, max_jack=10)
    b = cluster_bootstrap(df, "cl", f, n_boot=100, seed=3, max_jack=10)
    assert np.array_equal(a.draws, b.draws) and a.ci_low[0] == b.ci_low[0]
    assert a.ci_low[0] < a.estimate[0] < a.ci_high[0]


def test_dm_identical_and_better_model():
    rng = np.random.default_rng(0)
    e = rng.normal(0, 2, 200)
    r = diebold_mariano(e, e)
    assert r["stat"] == 0.0 and r["pvalue"] == 1.0
    e2 = rng.normal(0, 1, 200)  # model 2 clearly more accurate
    r = diebold_mariano(e, e2, loss="abs")
    assert r["stat"] > 3 and r["pvalue"] < 0.01 and r["mean_diff"] > 0
    assert diebold_mariano(e2, e)["stat"] == pytest.approx(-r["stat"])
    h = diebold_mariano(e, e2, hln=True)
    assert h["stat"] < r["stat"] and h["pvalue"] < 0.01
    assert diebold_mariano(e, e2, loss="sq")["stat"] > 0


def test_ols_fe_cluster_absorb():
    rng = np.random.default_rng(0)
    n = 300
    pid = rng.integers(0, 40, n)
    x = rng.normal(size=n)
    df = pd.DataFrame({"y": 1 + 2 * x + rng.normal(size=40)[pid] + rng.normal(0, .1, n),
                       "x": x, "pid": pid, "g": pid % 3})
    r = ols(df, "y", ["x"], fe=["g"], cluster="pid")
    assert r.params["x"] == pytest.approx(2, abs=0.2) and r.nobs == n and len(r.resid) == n
    a = ols(df, "y", ["x"], absorb=["pid"], cluster="pid")
    assert a.params["x"] == pytest.approx(2, abs=0.05) and a.r2 > r.r2 - 1e-9
    assert float(np.std(a.resid)) < 0.2
