"""Paper Table 1 and Fig. 2 (spec section 8): team win regressions on sum(WAR) vs sum(WAR-).

Paper Table 1 (p. 253): eq. (1) W_nt = alpha + beta * sum_i WAR_it + eps_nt on 570 team-seasons,
specification (1) with sum(WAR) and specification (2) with the recursive sum(WAR-).
Paper fWAR: beta 0.996 -> 1.030, alpha 47.7 -> 48.2, R2 0.799 -> 0.887, pseudo R2 0.789 -> 0.873.
Fig. 2: densities of the two residual series; sd falls from about 5 to about 4 wins
(about 40 percent less unexplained variance).

Deviations / choices
- Pseudo R2 follows the table note "average from a 57-fold cross-validation". 570 / 57 = 10
  team-seasons per fold; for other sample sizes the number of folds is max(2, min(57, n // 10)) so
  the fold size stays near 10. Folds are random (seeded). PRIMARY = pooled out-of-fold
  1 - SSE_oos / SST (SST around the full-sample mean); SECONDARY = mean of per-fold R2
  (each fold's SST around its own mean; noisy with 10 obs per fold).
- Standard errors / CIs: BCa cluster bootstrap on franch_id (spec section 8). Only this second-stage
  regression is resampled; WAR- itself is held fixed (generated-regressor issue).
- Beta check ("within 2 SE of 1") is applied to specification (1), which is the paper's
  reference point; specification (2) is reported alongside.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from team_synergy.analysis.inputs import AnalysisInputs
from team_synergy.analysis.result import AnalysisResult
from team_synergy.stats.bootstrap import cluster_bootstrap

PAPER = {"alpha": (47.7, 48.2), "beta": (0.996, 1.030), "r2": (0.799, 0.887),
         "pseudo_r2": (0.789, 0.873)}


def ols_stat(df: pd.DataFrame, y: str, xs: list[str]) -> np.ndarray:
    """Fast OLS returning [const, coefs..., R2] (helper for bootstrap stat functions)."""
    X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy(float) for c in xs])
    yy = df[y].to_numpy(float)
    b, *_ = np.linalg.lstsq(X, yy, rcond=None)
    sst = ((yy - yy.mean()) ** 2).sum()
    r2 = 1 - ((yy - X @ b) ** 2).sum() / sst if sst > 0 else np.nan
    return np.append(b, r2)


def kfold_pseudo_r2(x: np.ndarray, y: np.ndarray, n_folds: int | None = None, seed: int = 0) -> dict:
    """Out-of-sample R2 of y = a + b x by k-fold cross-validation (Table 1 note).

    Returns dict(pooled, fold_mean, n_folds). ``n_folds`` default max(2, min(57, n // 10)).
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(y)
    k = n_folds if n_folds is not None else max(2, min(57, n // 10))
    k = min(k, n)
    folds = np.array_split(np.random.default_rng(seed).permutation(n), k)
    pred = np.empty(n)
    per = []
    for f in folds:
        tr = np.setdiff1d(np.arange(n), f)
        b, a = np.polyfit(x[tr], y[tr], 1)
        pred[f] = a + b * x[f]
        sst_f = ((y[f] - y[f].mean()) ** 2).sum()
        if len(f) > 1 and sst_f > 0:
            per.append(1 - ((y[f] - pred[f]) ** 2).sum() / sst_f)
    pooled = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return {"pooled": float(pooled), "fold_mean": float(np.mean(per)) if per else float("nan"), "n_folds": int(k)}


def run(inputs: AnalysisInputs, n_boot: int = 500, seed: int = 0) -> AnalysisResult:
    """Table 1 (both specs) and Fig. 2 data.

    Uses ``inputs.teams_recursive`` (W, war, war_minus; one row per team-season, WAR- from the
    window ending that season). Tables: ``table1`` (one row per spec), ``fig2_resid`` (residual
    series, long), ``fig2_density`` (Gaussian KDE on a common grid), ``fig2_ratio`` (sd ratio with
    BCa CI). Checks quote the paper (see module docstring).
    """
    t = inputs.teams_recursive[["franch_id", "season", "W", "war", "war_minus"]].dropna().reset_index(drop=True)
    n = len(t)
    specs = [("(1) sum WAR", "war"), ("(2) sum WAR-", "war_minus")]

    def stat(df):
        o = []
        for _, c in specs:
            o.extend(ols_stat(df, "W", [c]))
        r1 = _resid(df, "war").std(ddof=1)
        r2 = _resid(df, "war_minus").std(ddof=1)
        return np.array(o + [r2 / r1])

    bt = cluster_bootstrap(t, "franch_id", stat, n_boot=n_boot, seed=seed)
    rows, resid_rows = [], []
    for i, (label, c) in enumerate(specs):
        o = 3 * i
        cv = kfold_pseudo_r2(t[c], t["W"], seed=seed + i)
        rows.append({"spec": label, "alpha": bt.estimate[o], "se_alpha": bt.se[o],
                     "alpha_ci_low": bt.ci_low[o], "alpha_ci_high": bt.ci_high[o],
                     "beta": bt.estimate[o + 1], "se_beta": bt.se[o + 1],
                     "beta_ci_low": bt.ci_low[o + 1], "beta_ci_high": bt.ci_high[o + 1],
                     "r2": bt.estimate[o + 2], "pseudo_r2": cv["pooled"],
                     "pseudo_r2_fold_mean": cv["fold_mean"], "n_obs": n, "n_folds": cv["n_folds"],
                     "paper_alpha": PAPER["alpha"][i], "paper_beta": PAPER["beta"][i],
                     "paper_r2": PAPER["r2"][i], "paper_pseudo_r2": PAPER["pseudo_r2"][i]})
        resid_rows.append(t[["franch_id", "season"]].assign(spec=label, resid=_resid(t, c).to_numpy()))
    table1 = pd.DataFrame(rows)
    resid = pd.concat(resid_rows, ignore_index=True)
    grid = np.linspace(resid["resid"].min() - 3, resid["resid"].max() + 3, 400)
    dens = pd.concat([pd.DataFrame({"spec": s, "x": grid, "density": gaussian_kde(g["resid"])(grid)})
                      for s, g in resid.groupby("spec", sort=False)], ignore_index=True)
    sd = resid.groupby("spec", sort=False)["resid"].std(ddof=1)
    ratio = pd.DataFrame([{"sd_resid_war": sd.iloc[0], "sd_resid_war_minus": sd.iloc[1],
                           "ratio": bt.estimate[-1], "se": bt.se[-1], "ci_low": bt.ci_low[-1],
                           "ci_high": bt.ci_high[-1], "var_reduction": 1 - bt.estimate[-1] ** 2}])
    r1, r2 = table1.iloc[0], table1.iloc[1]
    checks = [
        {"item": "beta (spec 1) within 2 SE of 1", "value": f"{r1.beta:.3f} (SE {r1.se_beta:.3f})",
         "expected": "beta 0.996 (fWAR), approximately 1", "passed": bool(abs(r1.beta - 1) <= 2 * r1.se_beta)},
        {"item": "alpha (spec 1) < 50", "value": f"{r1.alpha:.2f}",
         "expected": "alpha 47.7, 'just less than 50'", "passed": bool(r1.alpha < 50)},
        {"item": "R2 rises from WAR to WAR-", "value": f"{r1.r2:.3f} -> {r2.r2:.3f}",
         "expected": "0.799 -> 0.887 (fWAR)", "passed": bool(r2.r2 > r1.r2)},
        {"item": "pseudo R2 rises from WAR to WAR-", "value": f"{r1.pseudo_r2:.3f} -> {r2.pseudo_r2:.3f}",
         "expected": "0.789 -> 0.873 (fWAR)", "passed": bool(r2.pseudo_r2 > r1.pseudo_r2)},
        {"item": "sd(resid WAR-) / sd(resid WAR) < 1",
         "value": f"{ratio.ratio[0]:.3f} (BCa95 {ratio.ci_low[0]:.3f}, {ratio.ci_high[0]:.3f}); "
                  f"sd {sd.iloc[0]:.2f} -> {sd.iloc[1]:.2f}",
         "expected": "sd about 5 -> about 4 wins; about 40% less unexplained variance",
         "passed": bool(ratio.ratio[0] < 1)},
    ]
    return AnalysisResult("table1", {"table1": table1, "fig2_resid": resid, "fig2_density": dens,
                                     "fig2_ratio": ratio}, checks)


def _resid(df: pd.DataFrame, c: str) -> pd.Series:
    b, a = np.polyfit(df[c].to_numpy(float), df["W"].to_numpy(float), 1)
    return df["W"] - a - b * df[c]
