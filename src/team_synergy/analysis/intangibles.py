"""Intangibles: Fig. 11 (paper section 5.1; spec section 8).

xi = residuals of Table 4 spec (2) (eq. 16, pcWAR not explained by past pcWAR, age / position
profile, WAR, experience and team / manager / league effects). Active players of the last
season are ranked by career-average xi with the same filter as Fig. 8 (top half by average
appearance weight, paper footnote 18); top and bottom quartiles (``q``; ``n`` overrides).
Players need at least one season with a previous season to have a xi; others are dropped
after the weight filter.
"""
from __future__ import annotations

import numpy as np

from team_synergy.analysis.inputs import AnalysisInputs
from team_synergy.analysis.pc_persistence import find_player_id, player_names, table4_residuals
from team_synergy.analysis.player_eval import career_table, top_bottom
from team_synergy.analysis.result import AnalysisResult


def run(inputs: AnalysisInputs, n_boot: int = 500, seed: int = 0, q: float = 0.25,
        n: int | None = None, **opts) -> AnalysisResult:
    """Fig. 11 (paper section 5.1; spec section 8). ``n_boot`` / ``seed`` unused.

    Tables: ``fig11`` (top / bottom players: player_id, name, xi (career average), avg_weight,
    n_seasons, side, rank), ``xi`` (player-season xi, from Table 4 spec 2).
    """
    xi = table4_residuals(inputs)
    ps = inputs.player_seasons.merge(xi[["player_id", "season", "xi"]], on=["player_id", "season"], how="left")
    tab = career_table(ps, ["xi"])
    f11 = top_bottom(tab, "xi", q=q, n=n)
    f11.insert(1, "name", f11["player_id"].map(player_names(inputs.players)))
    real = inputs.war_source != "synthetic"
    mean_xi = float(xi["xi"].mean())
    top = f11[f11["side"] == "top"].iloc[0]
    kid = find_player_id(inputs.players, "Kevin Kiermaier") if real else None
    if not real:
        val, ok = "not checked (synthetic data)", None
    else:
        val = f"#1 = {top['name']} ({top['xi']:.3f})"
        ok = bool(top["player_id"] == kid and abs(top["xi"] - 0.1) < 0.05)
    checks = [
        {"item": "mean xi (sample residual mean)", "value": mean_xi,
         "expected": "approx 0 (OLS residuals; spec 2 has dummies spanning a constant)",
         "passed": bool(abs(mean_xi) < 1e-6)},
        {"item": "Fig. 11: #1 by career xi is Kevin Kiermaier, approx 0.1", "value": val,
         "expected": "Kevin Kiermaier at about 0.1", "passed": ok},
    ]
    return AnalysisResult("intangibles", {"fig11": f11, "xi": xi}, checks)
