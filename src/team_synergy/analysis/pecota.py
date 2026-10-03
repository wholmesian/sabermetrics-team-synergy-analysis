"""Paper Table 3 (p. 263, spec section 8): tcWAR as a predictor of wins next to PECOTA.

THE PAPER'S SPECIFICATION IS USED, NOT THE SPEC'S (the spec's "W = a + b PECOTA + c tcWAR" is wrong):
  (1) PECOTA_t = a + b PECOTA_{t-1} + c TeamWAR_{t-1} + d tcWAR_{t-1} + e, 2009-2016 (240 obs).
      TeamWAR_{t-1} = combined season t-1 WAR of the players on team n's season-t roster (players
      without a t-1 season count 0). tcWAR_{t-1} = recursive tcWAR of the window ending t-1.
      Paper fWAR: 0.537, 0.246, 0.260, const 29.1, R2 0.587.
  (2) W_t = a + b W_{t-1} + c PECOTA_t + d ProjectedTcWAR_t + e, 2008-2016 (270 obs).
      ProjectedTcWAR_t = a_ar + b_ar tcWAR_{t-1}, the one-step forecast from the pooled tcWAR AR(1)
      estimated on seasons <= t-1 (tc_persistence.fit_ar1). Paper fWAR: 0.363, 0.306, 16.65, const 54.45,
      R2 0.371.
  Out-of-sample: spec (2) re-estimated recursively on seasons <= t-1, predicting season t (2009-2016,
  240 predictions); compare with PECOTA's own error |W - PECOTA|. Paper MAE gain 1.251 (fWAR),
  1.487 (bWAR), Diebold-Mariano with absolute loss and Newey-West variance.

Choices: tcWAR series = ``teams_recursive`` (each season from its own window; real-time, no
look-ahead). Rosters from ``stints_full`` (anyone with a stint on the team that season), WAR from
``player_seasons``. Minimum training size for an OOS prediction is ``min_train`` (default 10)
observations. DM: e1 = PECOTA error, e2 = model error, so stat > 0 means the model is better
(``diebold_mariano`` convention). Gain CI/SE: BCa cluster bootstrap on franch_id of the mean
|e_PECOTA| - |e_model|. Coefficient SEs: BCa cluster bootstrap on franch_id with the projected tcWAR
held fixed (generated-regressor issue ignored). If ``inputs.pecota`` is missing/empty the run is skipped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from team_synergy.analysis.inputs import AnalysisInputs
from team_synergy.analysis.result import AnalysisResult
from team_synergy.analysis.table1 import ols_stat
from team_synergy.analysis.tc_persistence import fit_ar1
from team_synergy.stats.bootstrap import cluster_bootstrap
from team_synergy.stats.dm_test import diebold_mariano

PAPER_GAIN = {"fwar": 1.251, "bwar": 1.487}
S1 = ["pecota_lag", "team_war_lag", "tc_lag"]
S2 = ["W_lag", "pecota", "proj_tc"]


def build_panel(inputs: AnalysisInputs) -> pd.DataFrame:
    """Team-season panel with W, pecota, pecota_lag, team_war_lag, tc_lag, W_lag, proj_tc."""
    t = inputs.teams_recursive[["franch_id", "season", "W", "tc_war"]].copy()
    pec = inputs.pecota[["franch_id", "season", "pecota_w"]].rename(columns={"pecota_w": "pecota"})
    lagk = lambda d, c: d.assign(season=d["season"] + 1).rename(columns={c: c + "_lag"})
    df = t.merge(pec, on=["franch_id", "season"], how="inner")
    df = df.merge(lagk(pec, "pecota"), on=["franch_id", "season"], how="left")
    df = df.merge(lagk(t[["franch_id", "season", "W"]], "W"), on=["franch_id", "season"], how="left")
    df = df.merge(lagk(t[["franch_id", "season", "tc_war"]], "tc_war").rename(columns={"tc_war_lag": "tc_lag"}),
                  on=["franch_id", "season"], how="left")
    ros = inputs.stints_full[["player_id", "season", "franch_id"]].drop_duplicates()
    pw = inputs.player_seasons[["player_id", "season", "war"]].copy()
    pw["season"] += 1
    ros = ros.merge(pw.rename(columns={"war": "w_lag"}), on=["player_id", "season"], how="left")
    tw = ros.groupby(["franch_id", "season"])["w_lag"].sum(min_count=0).rename("team_war_lag").reset_index()
    df = df.merge(tw, on=["franch_id", "season"], how="left")
    proj = {}
    for s in sorted(df["season"].unique()):
        try:
            proj[s] = fit_ar1(t, through=s - 1)
        except Exception:  # not enough pairs yet
            proj[s] = (np.nan, np.nan)
    df["proj_tc"] = [proj[s][0] + proj[s][1] * x for s, x in zip(df["season"], df["tc_lag"])]
    return df.sort_values(["season", "franch_id"]).reset_index(drop=True)


def _coef_table(d: pd.DataFrame, y: str, xs: list[str], spec: str, n_boot: int, seed: int) -> pd.DataFrame:
    bt = cluster_bootstrap(d, "franch_id", lambda x: ols_stat(x, y, xs), n_boot=n_boot, seed=seed)
    k = len(xs) + 1
    return pd.DataFrame({"spec": spec, "term": ["const"] + xs, "estimate": bt.estimate[:k], "se": bt.se[:k],
                         "ci_low": bt.ci_low[:k], "ci_high": bt.ci_high[:k], "n_obs": len(d),
                         "r2": bt.estimate[k]})


def run(inputs: AnalysisInputs, n_boot: int = 500, seed: int = 0, min_train: int = 10) -> AnalysisResult:
    """Table 3 specs (1) and (2), recursive out-of-sample comparison with PECOTA.

    Tables: ``table3`` (spec, term, estimate, se, BCa CI, n_obs, r2), ``oos`` (per prediction),
    ``oos_summary`` (n, MAE both, gain, SE, CI, DM stat/p/lags, paper gain).
    """
    if inputs.pecota is None or len(inputs.pecota) == 0:
        return AnalysisResult("pecota", {}, [{"item": "Table 3", "value": "skipped: no PECOTA data",
                                              "expected": "PECOTA projections 2008-2016", "passed": None}])
    df = build_panel(inputs)
    d1 = df.dropna(subset=["pecota", *S1]).reset_index(drop=True)
    d2 = df.dropna(subset=["W", *S2]).reset_index(drop=True)
    tab = pd.concat([_coef_table(d1, "pecota", S1, "(1) PECOTA_t", n_boot, seed),
                     _coef_table(d2, "W", S2, "(2) W_t", n_boot, seed + 1)], ignore_index=True)

    rows = []
    for s in sorted(d2["season"].unique()):
        tr, te = d2[d2["season"] < s], d2[d2["season"] == s]
        if len(tr) < min_train or te.empty:
            continue
        X = np.column_stack([np.ones(len(tr))] + [tr[c] for c in S2])
        b, *_ = np.linalg.lstsq(X, tr["W"].to_numpy(float), rcond=None)
        pred = np.column_stack([np.ones(len(te))] + [te[c] for c in S2]) @ b
        rows.append(te[["franch_id", "season", "W", "pecota"]].assign(pred_model=pred))
    checks = []
    paper_gain = PAPER_GAIN.get(inputs.war_source)
    exp_gain = ("MAE(PECOTA) - MAE(model) = 1.251 (fWAR), 1.487 (bWAR); DM significant")
    if rows:
        oos = pd.concat(rows, ignore_index=True)
        oos["abs_err_model"] = (oos["W"] - oos["pred_model"]).abs()
        oos["abs_err_pecota"] = (oos["W"] - oos["pecota"]).abs()
        oos["abs_gain"] = oos["abs_err_pecota"] - oos["abs_err_model"]
        bt = cluster_bootstrap(oos, "franch_id", lambda x: np.array([x["abs_gain"].mean()]), n_boot=n_boot,
                               seed=seed + 2)
        dm = diebold_mariano(oos["W"] - oos["pecota"], oos["W"] - oos["pred_model"], loss="abs")
        summ = pd.DataFrame([{"n": len(oos), "mae_pecota": oos["abs_err_pecota"].mean(),
                              "mae_model": oos["abs_err_model"].mean(), "gain": bt.estimate[0], "se": bt.se[0],
                              "ci_low": bt.ci_low[0], "ci_high": bt.ci_high[0], "dm_stat": dm["stat"],
                              "dm_pvalue": dm["pvalue"], "dm_lags": dm["lags"],
                              "paper_gain": paper_gain if paper_gain is not None else np.nan}])
        checks.append({"item": "OOS gain MAE(PECOTA) - MAE(model)",
                       "value": f"{bt.estimate[0]:.3f} (SE {bt.se[0]:.3f}); DM {dm['stat']:.2f}, p {dm['pvalue']:.3f}",
                       "expected": exp_gain,
                       "passed": None if inputs.war_source == "synthetic" else bool(bt.estimate[0] > 0)})
        checks.append({"item": "OOS prediction count", "value": len(oos),
                       "expected": "240 predictions, 2009-2016",
                       "passed": None if inputs.war_source == "synthetic" else bool(len(oos) == 240)})
    else:
        oos, summ = pd.DataFrame(), pd.DataFrame()
        checks.append({"item": "OOS comparison", "value": "no season with >= min_train training obs",
                       "expected": exp_gain, "passed": None})
    synth = inputs.war_source == "synthetic"
    checks.append({"item": "spec (1) sample size", "value": len(d1), "expected": "240 (2009-2016)",
                   "passed": None if synth else bool(len(d1) == 240)})
    checks.append({"item": "spec (2) sample size", "value": len(d2), "expected": "270 (2008-2016)",
                   "passed": None if synth else bool(len(d2) == 270)})
    r = tab.groupby("spec")["r2"].first()
    checks.append({"item": "Table 3 R2 (1) / (2)", "value": f"{r.iloc[0]:.3f} / {r.iloc[1]:.3f}",
                   "expected": "0.587 / 0.371 (fWAR)", "passed": None})
    return AnalysisResult("pecota", {"table3": tab, "oos": oos, "oos_summary": summ}, checks)
