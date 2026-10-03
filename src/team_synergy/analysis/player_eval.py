"""Player evaluation: Figs. 6, 7 and 8 (paper section 4.3; spec section 8).

Fig. 6: player-season WAR- and WAR+ against WAR with the FanGraphs thresholds WAR = 1 and 4.
Fig. 7: pcWAR (= WAR+ - WAR) against WAR. Fig. 8: active players of the last season ranked by
career-average pcWAR, restricted to the top half by career-average appearance weight
(paper footnote 18), top and bottom quartiles, stacked into the character and team
components (career averages).

Deviation from the spec: the paper shows the top / bottom 25% (quartiles), the spec says 35
players; quartiles are the default here (``q=0.25``), ``n`` overrides with a fixed count.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from team_synergy.analysis.inputs import AnalysisInputs
from team_synergy.analysis.pc_persistence import assign_tier, find_player_id, player_names
from team_synergy.analysis.result import AnalysisResult


def career_table(ps: pd.DataFrame, value_cols: list[str], weight_col: str = "weight") -> pd.DataFrame:
    """Career averages for players active in the last season, top half by average weight.

    Averages (NaN skipped) of ``value_cols`` and of ``weight_col`` over all of a player's
    seasons; keeps ceil(N/2) of the N active players with the largest average weight (paper
    footnote 18); rows whose first value column is NaN are then dropped. Columns: player_id,
    avg_weight, n_seasons, ``value_cols``.
    """
    last = ps["season"].max()
    g = ps.groupby("player_id").agg(avg_weight=(weight_col, "mean"), n_seasons=("season", "size"),
                                    **{c: (c, "mean") for c in value_cols})
    g = g[g.index.isin(ps.loc[ps["season"] == last, "player_id"])]
    g = g.sort_values("avg_weight", ascending=False).head(int(np.ceil(len(g) / 2)))
    return g.dropna(subset=[value_cols[0]]).reset_index()


def top_bottom(tab: pd.DataFrame, col: str, q: float = 0.25, n: int | None = None) -> pd.DataFrame:
    """Top and bottom ``q`` share (or ``n`` players) of ``tab`` by ``col``.

    Adds ``rank`` (1 = largest ``col``) and ``side`` ("top" / "bottom"); size k = n, else
    max(1, round(q * len(tab))). Returned top first (descending), then bottom (descending).
    """
    t = tab.sort_values(col, ascending=False).reset_index(drop=True)
    t["rank"] = np.arange(1, len(t) + 1)
    k = int(n) if n is not None else max(1, int(round(q * len(t))))
    k = min(k, len(t))
    return pd.concat([t.head(k).assign(side="top"), t.tail(k).assign(side="bottom")], ignore_index=True)


def run(inputs: AnalysisInputs, n_boot: int = 500, seed: int = 0, q: float = 0.25,
        n: int | None = None, **opts) -> AnalysisResult:
    """Figs. 6-8 (paper section 4.3; spec section 8). ``n_boot`` / ``seed`` unused (no inference).

    Tables: ``fig6`` (player-season war, war_minus, war_plus, diff columns, tier),
    ``fig6_thresholds`` (WAR = 1, 4), ``tier_summary`` (by WAR tier: mean WAR+/- gaps and pcWAR spread, for Figs. 6-7),
    ``fig7`` (player-season war, pc_war, pc_war_char, pc_war_team, tier), ``fig8`` (top /
    bottom players with name, career-average pc_war, pc_war_char, pc_war_team, side, rank).
    """
    ps = inputs.player_seasons
    f = ps[["player_id", "season", "war", "war_minus", "war_plus", "pc_war", "pc_war_char", "pc_war_team"]].copy()
    f["tier"] = assign_tier(f["war"])  # same 1 / 4 thresholds (FanGraphs), applied to WAR_t
    f["d_minus"] = f["war_minus"] - f["war"]
    f["d_plus"] = f["war_plus"] - f["war"]
    thr = pd.DataFrame({"war": [1.0, 4.0], "label": ["Scrub/Role", "Role/Star"]})
    summ = (f.groupby("tier").agg(n=("war", "size"), mean_d_minus=("d_minus", "mean"),
                                  mean_d_plus=("d_plus", "mean"), mean_pc=("pc_war", "mean"),
                                  p05_pc=("pc_war", lambda s: s.quantile(0.05)),
                                  p95_pc=("pc_war", lambda s: s.quantile(0.95))).reindex(["Scrub", "Role", "Star"])
            .reset_index())
    fig6 = f[["player_id", "season", "war", "war_minus", "war_plus", "d_minus", "d_plus", "tier"]]
    fig7 = f[["player_id", "season", "war", "pc_war", "pc_war_char", "pc_war_team", "tier"]]

    ct = career_table(ps, ["pc_war", "pc_war_char", "pc_war_team"])
    f8 = top_bottom(ct, "pc_war", q=q, n=n)
    f8.insert(1, "name", f8["player_id"].map(player_names(inputs.players)))

    corr = float(np.corrcoef(f["pc_war"], f["war"])[0, 1])
    hi, lo = f.loc[f["war"] >= 4, "d_plus"], f.loc[f["war"] < 1, "d_plus"]
    m_hi, m_lo = float(hi.mean()) if len(hi) else np.nan, float(lo.mean()) if len(lo) else np.nan
    mad_m, mad_p = float(f["d_minus"].abs().mean()), float(f["d_plus"].abs().mean())
    real = inputs.war_source != "synthetic"
    top = f8[f8["side"] == "top"].iloc[0] if len(f8) else None
    trout = find_player_id(inputs.players, "Mike Trout") if real else None
    if top is None or not real:
        trout_val, trout_ok = "not checked (synthetic data)" if not real else "empty", None
    else:
        trout_val = f"#1 = {top['name']} ({top['pc_war']:.3f})"
        trout_ok = bool(top["player_id"] == trout and top["pc_war"] > 0.5)
    checks = [
        {"item": "corr(pcWAR, WAR) > 0", "value": corr,
         "expected": "positive correlation (Fig. 7)", "passed": bool(corr > 0)},
        {"item": "mean(WAR+ - WAR | WAR >= 4) > 0", "value": m_hi,
         "expected": "WAR understates players with WAR > 4 (Fig. 6)", "passed": bool(m_hi > 0)},
        {"item": "mean(WAR+ - WAR | WAR < 1) < 0", "value": m_lo,
         "expected": "WAR overstates players with WAR < 1 (Fig. 6)", "passed": bool(m_lo < 0)},
        {"item": "WAR- closer to WAR than WAR+ (mean abs deviation)", "value": [mad_m, mad_p],
         "expected": "WAR- hugs the 45 degree line (Fig. 6)", "passed": bool(mad_m < mad_p)},
        {"item": "Fig. 8: #1 by career pcWAR is Mike Trout, average > 0.5", "value": trout_val,
         "expected": "Mike Trout, over one-half win on average", "passed": trout_ok},
    ]
    return AnalysisResult("player_eval", {"fig6": fig6, "fig6_thresholds": thr, "tier_summary": summ,
                                          "fig7": fig7, "fig8": f8}, checks)
