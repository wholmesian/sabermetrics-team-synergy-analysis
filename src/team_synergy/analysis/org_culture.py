"""Paper Fig. 3 (spec section 8): organizational culture ranking from the lambda_n.

Per franchise, the distribution over the recursive windows of the rank of |lambda_n| scaled
0-100 as (rank - 1) / (N - 1) * 100 (N teams in the window, 29 for 30 teams; 100 = largest |lambda|).
Paper (Fig. 3, section 3.2/4.1): the top three organizations are STL, ARI, SF in both fWAR and bWAR.

Choices: the primary lambda is ``cfg["culture_lambda_source"]`` ("prob" -> ``lam_prob``,
"als" -> ``lam_als``); ranks are recomputed from |lam| within each window (ties: average rank), and
both sources are returned. The stored ``rank`` column is not used (it is only a cross-check).
"""
from __future__ import annotations

import pandas as pd

from team_synergy.analysis.inputs import AnalysisInputs
from team_synergy.analysis.result import AnalysisResult

_ALIASES = {"SF": {"SF", "SFG"}, "ARI": {"ARI", "AZ"}, "STL": {"STL"}}


def culture_scores(lw: pd.DataFrame, col: str) -> pd.DataFrame:
    """Per (window_end, franch_id) |lambda| rank score in [0, 100] from column ``col``."""
    d = lw[["window_end", "franch_id", col]].copy()
    r = d.groupby("window_end")[col].transform(lambda s: s.abs().rank(method="average"))
    n = d.groupby("window_end")[col].transform("size")
    d["score"] = (r - 1) / (n - 1).clip(lower=1) * 100.0
    return d.rename(columns={col: "lam"})


def run(inputs: AnalysisInputs, n_boot: int = 500, seed: int = 0) -> AnalysisResult:
    """Fig. 3 data. ``n_boot`` and ``seed`` are unused (no resampling; kept for the common signature).

    Tables: ``fig3_scores`` (long: source, window_end, franch_id, lam, score), ``fig3_summary``
    (source, franch_id, median, q1, q3, min, max, n_windows, primary; sorted by median descending
    within source, so the primary-source table is ready for a sorted box plot).
    """
    lw = inputs.lambda_windows
    primary = inputs.cfg.get("culture_lambda_source", "prob")
    scores, summ = [], []
    for src, col in (("prob", "lam_prob"), ("als", "lam_als")):
        if col not in lw.columns:
            continue
        s = culture_scores(lw, col).assign(source=src)
        scores.append(s)
        g = s.groupby("franch_id")["score"]
        t = pd.DataFrame({"median": g.median(), "q1": g.quantile(0.25), "q3": g.quantile(0.75),
                          "min": g.min(), "max": g.max(), "n_windows": g.size()}).reset_index()
        t.insert(0, "source", src)
        t["primary"] = src == primary
        summ.append(t.sort_values("median", ascending=False).reset_index(drop=True))
    scores = pd.concat(scores, ignore_index=True)
    summ = pd.concat(summ, ignore_index=True)
    ps = summ[summ["primary"]]
    top3 = list(ps["franch_id"].head(3))
    checks = [{"item": "scores within [0, 100]", "value": f"[{scores.score.min():.1f}, {scores.score.max():.1f}]",
               "expected": "rank scaled 0-100", "passed": bool(scores.score.between(0, 100).all())}]
    if inputs.war_source == "synthetic":
        ok = None
    else:
        ok = all(any(t in al for t in top3) for al in _ALIASES.values())
    checks.append({"item": f"top three by median ({primary} lambda)", "value": ", ".join(top3),
                   "expected": "STL, ARI, SF (both fWAR and bWAR)", "passed": ok})
    return AnalysisResult("org_culture", {"fig3_scores": scores, "fig3_summary": summ}, checks)
