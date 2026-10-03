"""Player productivity residuals, paper eqs. (4) and (7); spec section 3."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .team_reg import TeamRegResult

_CONVENTIONS = ("consistent", "paper_literal")


def player_residuals(
    stints: pd.DataFrame, reg: TeamRegResult, sign_convention: str = "consistent"
) -> pd.DataFrame:
    """Allocate team residuals to stints, eqs. (4) and (7) (spec section 3).

    W_hat_r = weight_r * (W_nt - alpha_hat)                       (eq. 7, weight = eta*tau)
    consistent:    y_r = war_r - W_hat_r / beta_hat   (default)
    paper_literal: y_r = W_hat_r / beta_hat - war_r

    Sign note (spec section 3, bullet 2): eq. (4) writes eps = sum W_hat - sum WAR,
    i.e. positive = over-performance. The paper's reported facts (WAR- cuts the
    residual variance ~40%, tcWAR positively correlated with the team residual,
    tp<0 & lambda>0 = positive spillover) are only jointly consistent with
    y = -eps/beta_hat, hence "consistent" is the default; "paper_literal" is kept to
    compare both. Dividing by beta_hat (spec simplification: the paper does not
    mention it) keeps the identity sum_r y_r = -/+ eps_hat/beta_hat exact.
    """
    if sign_convention not in _CONVENTIONS:
        raise ValueError(f"unknown sign_convention {sign_convention!r}")
    out = stints.copy()
    out["W_hat"] = out["weight"] * (out["team_wins"] - reg.alpha)
    scaled = out["W_hat"] / reg.beta
    war = out[reg.war_col] if reg.war_col in out else out["war"]
    out["y"] = war - scaled if sign_convention == "consistent" else scaled - war
    return out


def check_residual_identity(
    stints_with_y: pd.DataFrame, reg: TeamRegResult, sign_convention: str = "consistent"
) -> float:
    """Max abs error of sum_r y_r vs -eps_hat/beta_hat (consistent) or +eps_hat/beta_hat
    (paper_literal) over team-seasons (eq. 4 identity, spec section 3)."""
    s = stints_with_y.groupby(["franch_id", "season"])["y"].sum()
    eps = reg.teams["eps"].reindex(s.index)
    sign = -1.0 if sign_convention == "consistent" else 1.0
    return float(np.max(np.abs(s.to_numpy() - sign * eps.to_numpy() / reg.beta)))
