"""Team-level regression, paper eq. (1); spec section 3."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class TeamRegResult:
    """Result of the team-season regression W = alpha + beta * sum(WAR) + eps (eq. 1)."""

    alpha: float
    beta: float
    se_alpha: float
    se_beta: float
    r2: float
    teams: pd.DataFrame  # index (franch_id, season); cols W, war_sum, fitted, eps
    beta_mode: str = "estimated"
    war_col: str = "war"


def fit_team_regression(
    stints: pd.DataFrame, war_col: str = "war", beta_mode: str = "estimated"
) -> TeamRegResult:
    """Fit paper eq. (1) on team-seasons (spec section 3).

    W_nt = alpha + beta * sum_{i in n,t} WAR_it + eps_nt.

    ``beta_mode="estimated"``: OLS with standard errors clustered by franch_id
    (as in the paper's Table 1). ``"fixed_one"``: beta = 1 and
    alpha = mean(W - sum WAR) (spec section 3/10 alternative); SEs for beta are NaN
    and se_alpha is the plain standard error of the mean.
    ``war_col`` lets later code regress on WAR- instead of WAR.
    """
    if beta_mode not in ("estimated", "fixed_one"):
        raise ValueError(f"unknown beta_mode {beta_mode!r}")
    g = stints.groupby(["franch_id", "season"])
    teams = pd.DataFrame({"W": g["team_wins"].first(), "war_sum": g[war_col].sum()})
    W = teams["W"].to_numpy(float)
    x = teams["war_sum"].to_numpy(float)
    if beta_mode == "estimated":
        groups = pd.factorize(teams.index.get_level_values("franch_id"))[0]
        X = sm.add_constant(x)
        res = sm.OLS(W, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
        alpha, beta = float(res.params[0]), float(res.params[1])
        se_a, se_b = float(res.bse[0]), float(res.bse[1])
        r2 = float(res.rsquared)
    else:
        d = W - x
        alpha, beta = float(d.mean()), 1.0
        se_a = float(d.std(ddof=1) / np.sqrt(len(d)))
        se_b = float("nan")
        tot = ((W - W.mean()) ** 2).sum()
        r2 = float(1 - ((W - alpha - x) ** 2).sum() / tot)
    teams["fitted"] = alpha + beta * x
    teams["eps"] = W - teams["fitted"].to_numpy()
    return TeamRegResult(alpha, beta, se_a, se_b, r2, teams, beta_mode, war_col)
