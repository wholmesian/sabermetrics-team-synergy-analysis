"""Age profiles of pcWAR by position: Fig. 10 (paper section 5.1; spec section 8).

From Table 4 spec (2) (eq. 16): for position p and age a in 20..40, the average marginal
prediction is the mean over p's observations of the predicted pcWAR with age set to a and all
other covariates as observed. Age enters only through the position x age polynomial, so this
equals  xbar_p . beta + sum_m theta_pm (z_a^m - mean_i(p) z_i^m)  with xbar_p the mean design
row of p (exact for a linear model); it is evaluated per bootstrap draw without re-predicting.

Inference: BCa 95% CIs from the player-cluster bootstrap that refits spec (2) per draw.
Simplification: the covariate means (xbar_p, mean z^m) are held at the full-sample values in
every draw; only the coefficients are re-estimated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from team_synergy.analysis.inputs import AnalysisInputs
from team_synergy.analysis.pc_persistence import (AGE_CENTER, AGE_SCALE, build_table4,
                                                  weighted_cluster_boot)
from team_synergy.analysis.result import AnalysisResult


def run(inputs: AnalysisInputs, n_boot: int = 500, seed: int = 0, ages=range(20, 41),
        max_jack: int | None = 200, **opts) -> AnalysisResult:
    """Fig. 10. Tables: ``fig10`` (pos, age, estimate, se, ci_low, ci_high, n_obs) and
    ``slopes`` (pos, slope per year from OLS of the estimate on age, change from first to
    last age, n_obs, slope_rank = 1 for the steepest).
    """
    t = build_table4(inputs.player_seasons)
    d, ds = t.frame, t.designs[2]
    pos = ds.pos_present
    ages = np.asarray(list(ages), dtype=float)
    za = (ages - AGE_CENTER) / AGE_SCALE
    Zp = np.vstack([za ** m for m in range(1, 5)])  # (4, A)
    xbar, zbar = {}, {}
    for p in pos:
        m = (d["pos"] == p).to_numpy()
        xbar[p] = np.asarray(ds.X[m].mean(axis=0)).ravel()
        zbar[p] = np.array([(d.loc[m, "z"] ** k).mean() for k in range(1, 5)])

    def profile(beta):
        out = []
        for p in pos:
            th = beta[ds.age_cols[p]]
            out.append(xbar[p] @ beta + th @ (Zp - zbar[p][:, None]))
        return np.concatenate(out)

    bt = weighted_cluster_boot(d["player_id"], lambda w: profile(t.fit(2, w)), n_boot, seed, max_jack)
    A = len(ages)
    rows = []
    for i, p in enumerate(pos):
        sl = slice(i * A, (i + 1) * A)
        rows.append(pd.DataFrame({"pos": p, "age": ages.astype(int), "estimate": bt.estimate[sl], "se": bt.se[sl],
                                  "ci_low": bt.ci_low[sl], "ci_high": bt.ci_high[sl],
                                  "n_obs": int((d["pos"] == p).sum())}))
    f10 = pd.concat(rows, ignore_index=True)
    sl_rows = []
    for p, g in f10.groupby("pos", sort=False):
        slope = float(np.polyfit(g["age"], g["estimate"], 1)[0])
        sl_rows.append({"pos": p, "slope": slope, "change": float(g["estimate"].iloc[-1] - g["estimate"].iloc[0]),
                        "n_obs": int(g["n_obs"].iloc[0])})
    sl = pd.DataFrame(sl_rows)
    sl["slope_rank"] = sl["slope"].rank(ascending=False).astype(int)
    s = sl.set_index("pos")["slope"]
    up = float((s > 0).mean())
    half = len(s) / 2
    checks = [
        {"item": "share of positions with upward slope", "value": up,
         "expected": "most positions slope upward (Fig. 10)", "passed": bool(up > 0.5)},
    ]
    if {"SP", "RP"} <= set(s.index):
        checks.append({"item": "SP flat (|slope| below median |slope|)", "value": float(abs(s["SP"])),
                       "expected": "SP relatively flat", "passed": bool(abs(s["SP"]) < s.abs().median())})
        checks.append({"item": "RP slope > SP slope", "value": [float(s["RP"]), float(s["SP"])],
                       "expected": "RP steeper than SP", "passed": bool(s["RP"] > s["SP"])})
    if {"DH", "UT"} <= set(s.index):
        rk = sl.set_index("pos")["slope_rank"]
        checks.append({"item": "DH and UT among steepest (rank in top half)", "value": [int(rk["DH"]), int(rk["UT"])],
                       "expected": "DH and UT among the steepest", "passed": bool(rk["DH"] <= half and rk["UT"] <= half)})
    return AnalysisResult("profiles", {"fig10": f10, "slopes": sl}, checks)
