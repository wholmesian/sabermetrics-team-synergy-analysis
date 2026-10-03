"""Network metrics from the fitted two-factor model, paper eqs. (10)-(13); spec section 6.

For a team-season block b with w_ij = (Phi_b)_ij, Phi = (I - rho_hat A)^-1 and
g_j = c_j + tp_j * lambda_n (eq. 9 factor, eqs. 10/11 split into own vs teammate terms):

    own_i = w_ii g_i                         (eq. 10, "measurement error")
    in_i  = sum_{j != i} w_ij g_j            (eq. 11 / eq. 12, in-degree)
    out_i = g_i sum_{j != i} w_ji            (out-degree)
    WAR-_i = WAR_i - in_i   (eq. 12);  WAR+_i = WAR-_i + out_i;  pcWAR_i = out_i - in_i
    tcWAR_nt = -beta_hat * sum_i in_i        (eq. 13; the only metric in wins)

Sign convention: with y = WAR - W_hat/beta ("consistent", spec section 3) a positive
in_i means player i's WAR was inflated by teammates, hence WAR- subtracts it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .team_reg import fit_team_regression

_SUM_COLS = ["war", "own", "in_deg", "out_deg", "war_minus", "war_plus", "pc_war"]


def _degrees(blocks, Phi, g, n):
    own, ind, outd = np.zeros(n), np.zeros(n), np.zeros(n)
    for b, W in zip(blocks, Phi):
        gb = g[b.idx]
        d = np.diag(W)
        off = W - np.diag(d)
        own[b.idx] = d * gb
        ind[b.idx] = off @ gb
        outd[b.idx] = gb * off.sum(axis=0)
    return own, ind, outd


def decompose(stints, blocks, Phi, g, g_char=None, g_team=None, beta=None) -> pd.DataFrame:
    """Per-stint own / in / out degree and WAR-, WAR+, pcWAR (eqs. 10-12; spec section 6).

    ``blocks``/``Phi`` index the rows of ``stints`` positionally (as from build_blocks /
    fit_sar), so ``g`` must be in stint order.  If ``g_char`` (= c) and ``g_team``
    (= tp * lambda) are given, pcWAR is also split into pc_war_char / pc_war_team
    (spec section 6 "player/team decomposition"; exact because every metric is linear
    in g).  If ``beta`` is given, tc_war_contrib = -beta * in_deg is added (eq. 13 term).
    """
    n = len(stints)
    g = np.asarray(g, float)
    out = stints.copy()
    own, ind, outd = _degrees(blocks, Phi, g, n)
    out["g"] = g
    out["own"], out["in_deg"], out["out_deg"] = own, ind, outd
    out["war_minus"] = out["war"].to_numpy() - ind
    out["war_plus"] = out["war_minus"] + outd
    out["pc_war"] = outd - ind
    if g_char is not None and g_team is not None:
        for name, gg in (("char", g_char), ("team", g_team)):
            _, i2, o2 = _degrees(blocks, Phi, np.asarray(gg, float), n)
            out[f"pc_war_{name}"] = o2 - i2
    if beta is not None:
        out["tc_war_contrib"] = -beta * ind
    return out


def team_metrics(decomposed: pd.DataFrame, beta: float) -> pd.DataFrame:
    """Team-season metrics (eq. 13): tc_war = -beta * sum_i in_i, plus sums of
    war, war_minus, war_plus, pc_war (the last two identities: sum war_plus = sum war,
    sum pc_war = 0)."""
    g = decomposed.groupby(["franch_id", "season"])
    out = g[["war", "war_minus", "war_plus", "pc_war"]].sum()
    out["tc_war"] = -beta * g["in_deg"].sum()
    return out


def player_season_metrics(decomposed: pd.DataFrame) -> pd.DataFrame:
    """Sum stint metrics over a player's stints within a season (traded players)."""
    cols = [c for c in _SUM_COLS + ["pc_war_char", "pc_war_team"] if c in decomposed]
    return decomposed.groupby(["player_id", "season"])[cols].sum()


def sign_convention_diagnostics(stints, decomposed, reg_beta_mode="estimated") -> dict:
    """Spec section 3 checks for the sign convention (paper: WAR- cuts the residual
    variance of eq. (1) by ~40%; tcWAR positively correlated with the team residual).

    ratio: sd(resid of eq. (1) on sum WAR-) / sd(resid on sum WAR); expect < 1.
    corr_tc_eps: corr(tcWAR, eps_hat from the WAR regression); expect > 0.
    ``stints`` must contain team_wins; ``decomposed`` is the output of decompose()."""
    d = stints[["franch_id", "season", "team_wins"]].copy()
    d["war"] = decomposed["war"].to_numpy()
    d["war_minus"] = decomposed["war_minus"].to_numpy()
    r0 = fit_team_regression(d, "war", reg_beta_mode)
    r1 = fit_team_regression(d, "war_minus", reg_beta_mode)
    tc = -r0.beta * decomposed.groupby(["franch_id", "season"])["in_deg"].sum()
    eps = r0.teams["eps"].reindex(tc.index)
    return dict(
        ratio=float(r1.teams["eps"].std() / r0.teams["eps"].std()),
        corr_tc_eps=float(np.corrcoef(tc, eps)[0, 1]),
        sd_eps_war=float(r0.teams["eps"].std()), sd_eps_war_minus=float(r1.teams["eps"].std()),
        beta=r0.beta,
    )
