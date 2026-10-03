"""Paper section 4.2, Fig. 4 and Fig. 5 (spec section 8): team synergy and its persistence.

Fig. 4: team residual eps_hat of eq. (1) (W on sum WAR) against tcWAR for all team-seasons; the 5
most extreme team-seasons per corner are flagged (label "<season> <franch_id>"). Paper (real data):
2008 LAA and 2007 ARI upper right; 1998 SEA, 1999 KC, 2015 CIN lower left.
Persistence (section 4.2): tcWAR_nt = a + b tcWAR_n,t-1 + u_nt, 540 obs (30 teams x 18 seasons), BCa SE
clustered on team. Paper: very low first-order autocorrelation (strong mean reversion). The residuals
summed by franchise are the "team synergy wins above expected" (Fig. 5 bars; paper standouts OAK,
CHW, NYY, LAD, STL).

Choices / deviations
- ``source``: "recursive" (default; tcWAR of each season from the window ending that season, as in
  Table 3) or "full" (full-sample estimate). The paper is not explicit for Figs. 4-5.
- eps: the ``eps`` column stored in the chosen team table (for "recursive" it is the residual of the
  window-ending-t regression, consistent with how tcWAR was built); ``eps_pooled`` is the residual
  of one pooled OLS of W on sum WAR over all team-seasons and is also returned.
- Corners: z-scores of (eps, tcWAR); UR = largest z_eps + z_tc, LL = smallest, UL = largest
  z_tc - z_eps, LR = smallest (5 each; a point goes to its first corner only).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from team_synergy.analysis.inputs import AnalysisInputs
from team_synergy.analysis.result import AnalysisResult
from team_synergy.analysis.table1 import ols_stat
from team_synergy.stats.bootstrap import cluster_bootstrap

_STAND = ["OAK", "CHW", "NYY", "LAD", "STL"]
_UR = [(2008, "LAA"), (2007, "ARI")]
_LL = [(1998, "SEA"), (1999, "KC"), (2015, "CIN")]


def team_table(inputs: AnalysisInputs, source: str = "recursive") -> pd.DataFrame:
    """Team-season table (franch_id, season, W, war, tc_war, eps) for source "recursive" | "full"."""
    if source not in ("recursive", "full"):
        raise ValueError("source must be 'recursive' or 'full'")
    t = inputs.teams_recursive if source == "recursive" else inputs.teams_full
    return t[["franch_id", "season", "W", "war", "tc_war", "eps"]].dropna().reset_index(drop=True)


def lag_pairs(t: pd.DataFrame) -> pd.DataFrame:
    """Pairs (tc_war, tc_lag) for consecutive seasons within a franchise."""
    lag = t[["franch_id", "season", "tc_war"]].copy()
    lag["season"] += 1
    return t.merge(lag.rename(columns={"tc_war": "tc_lag"}), on=["franch_id", "season"])


def fit_ar1(t: pd.DataFrame, through: int | None = None) -> tuple[float, float]:
    """Pooled AR(1) tc_war_t = a + b tc_war_{t-1} on pairs with season <= ``through``. Returns (a, b)."""
    p = lag_pairs(t)
    if through is not None:
        p = p[p["season"] <= through]
    b, a = np.polyfit(p["tc_lag"], p["tc_war"], 1)
    return float(a), float(b)


def run(inputs: AnalysisInputs, n_boot: int = 500, seed: int = 0, source: str = "recursive",
        n_extreme: int = 5) -> AnalysisResult:
    """Fig. 4 data, tcWAR persistence regression and Fig. 5 bars.

    Tables: ``fig4`` (franch_id, season, W, war, eps, eps_pooled, tc_war, corner, label),
    ``persistence`` (term const/slope with estimate, se, BCa CI, plus n_obs, r2, autocorr),
    ``persistence_resid`` (franch_id, season, u), ``fig5_bars`` (franch_id, synergy_wins sorted
    descending).
    """
    t = team_table(inputs, source)
    b, a = np.polyfit(t["war"], t["W"], 1)
    t["eps_pooled"] = t["W"] - a - b * t["war"]
    z = lambda s: (s - s.mean()) / s.std(ddof=1)
    ze, zt = z(t["eps"]), z(t["tc_war"])
    score = {"UR": ze + zt, "LL": -(ze + zt), "UL": zt - ze, "LR": ze - zt}
    t["corner"] = ""
    for c, s in score.items():
        idx = s[t["corner"] == ""].nlargest(n_extreme).index
        t.loc[idx, "corner"] = c
    t["label"] = np.where(t["corner"] != "", t["season"].astype(str) + " " + t["franch_id"], "")
    fig4 = t[["franch_id", "season", "W", "war", "eps", "eps_pooled", "tc_war", "corner", "label"]].copy()

    p = lag_pairs(t)[["franch_id", "season", "tc_war", "tc_lag"]].reset_index(drop=True)
    bt = cluster_bootstrap(p, "franch_id", lambda d: ols_stat(d, "tc_war", ["tc_lag"]), n_boot=n_boot, seed=seed)
    pers = pd.DataFrame({"term": ["const", "tc_lag"], "estimate": bt.estimate[:2], "se": bt.se[:2],
                         "ci_low": bt.ci_low[:2], "ci_high": bt.ci_high[:2]})
    ac = float(np.corrcoef(p["tc_war"], p["tc_lag"])[0, 1])
    pers = pers.assign(n_obs=len(p), r2=bt.estimate[2], autocorr=ac, source=source)
    p["u"] = p["tc_war"] - bt.estimate[0] - bt.estimate[1] * p["tc_lag"]
    bars = (p.groupby("franch_id")["u"].sum().rename("synergy_wins").sort_values(ascending=False).reset_index())

    slope = float(bt.estimate[1])
    checks = [
        {"item": "persistence slope b (tcWAR on lag)", "value": f"{slope:.3f} (SE {bt.se[1]:.3f}); corr {ac:.3f}",
         "expected": "very low first-order autocorrelation (strong mean reversion)", "passed": bool(abs(slope) < 0.3)},
        {"item": "persistence sample size", "value": len(p),
         "expected": "540 = 30 teams x 18 seasons (real data)",
         "passed": None if inputs.war_source == "synthetic" else bool(len(p) == 540)},
    ]
    real = inputs.war_source != "synthetic"
    flagged = set(zip(fig4.loc[fig4.corner == "UR", "season"], fig4.loc[fig4.corner == "UR", "franch_id"]))
    flagged_ll = set(zip(fig4.loc[fig4.corner == "LL", "season"], fig4.loc[fig4.corner == "LL", "franch_id"]))
    hit_ur, hit_ll = sum(x in flagged for x in _UR), sum(x in flagged_ll for x in _LL)
    checks.append({"item": "Fig. 4 labelled corners", "value": f"UR {hit_ur}/2, LL {hit_ll}/3",
                   "expected": "2008 LAA, 2007 ARI upper right; 1998 SEA, 1999 KC, 2015 CIN lower left",
                   "passed": (hit_ur == 2 and hit_ll == 3) if real else None})
    top = list(bars.reindex(bars["synergy_wins"].abs().sort_values(ascending=False).index)["franch_id"].head(7))
    ov = sum(s in top for s in _STAND)
    checks.append({"item": "Fig. 5 standouts among top-7 |synergy wins|", "value": f"{ov}/5 ({', '.join(top)})",
                   "expected": "OAK, CHW, NYY, LAD, STL stand out", "passed": (ov >= 3) if real else None})
    return AnalysisResult("tc_persistence", {"fig4": fig4, "persistence": pers, "persistence_resid":
                          p[["franch_id", "season", "u"]], "fig5_bars": bars}, checks)
