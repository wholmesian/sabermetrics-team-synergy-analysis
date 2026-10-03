"""Salary pricing of complementarity: Table 5, eq. (17) (paper section 5.2; spec section 8).

ln S_it = sum_c FA_c (theta_c + pi_c CumWAR_{t-1} + psi_c CumpcWAR_{t-1} + mu_c teamExp_{t-1})
        + sum_p pos_p (phi_p age_it + lambda_p mlbExp_{t-1} + tau_p mlbExp_{t-1}^2) + alpha_i + e_it

S = salary deflated to ``base_year`` dollars with the CPI of config/cpi.yaml (paper section
5.2). Sample: players whose careers began in the sample period (``players.debut_season`` >=
first sample season) with an observed positive salary. FA class from the number of prior
in-sample seasons: 0-2 FA0 (pre-arbitration), 3-5 FA1 (arbitration), 6+ FA2 (paper
footnote 22). Spec (2) replaces CumpcWAR by Cum(pcWAR - xi) and Cum xi (xi = Table 4 spec 2
residuals from ``pc_persistence``), dropping the FA0 interactions of these two (paper note).

Implementation notes / deviations:
* Cumulative variables run through t-1 (sum over earlier in-sample seasons). xi is undefined for
  a player's first season (no t-1 pcWAR); it is set to 0 there (and for seasons after a
  gap), so Cum(pcWAR - xi) + Cum xi = CumpcWAR exactly.
* teamExp_{t-1} = number of earlier in-sample seasons with the current (largest-weight) team.
  mlbExp_{t-1} = ``mlb_exp`` (games through the previous season), in units of 1000 games;
  age is (age - 30) / 10 (conditioning only).
* Player fixed effects are absorbed by within-player demeaning (exactly equivalent to
  dummies / AbsorbingLS); FA0 and the first position indicator are the omitted references of
  the otherwise FE-collinear indicator sets. Rank deficiency is handled by a tiny ridge.
* Salaries observed more than once per player-season are summed.
* Seasons outside the CPI table use the nearest available CPI year (synthetic data runs to
  2018; the real sample 1998-2016 is covered); the count is reported in ``meta``.
* SEs: BCa player-cluster bootstrap. Within-demeaning is invariant to the multiplicity of a
  drawn player, so the demeaned data are built once and each draw is a weighted regression.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.linalg as sla

from team_synergy.analysis.inputs import AnalysisInputs
from team_synergy.analysis.pc_persistence import (AGE_CENTER, AGE_SCALE, EXP_SCALE, POSITIONS,
                                                  table4_residuals, weighted_cluster_boot)
from team_synergy.analysis.result import AnalysisResult
from team_synergy.config import load_config

FA_CLASSES = ["FA0", "FA1", "FA2"]
PAPER_T5 = {"CumWAR": (0.10, 0.14, 0.04), "CumpcWAR": (-0.20, -0.50, 0.12),
            "CumPcMinusXi_FA1_FA2": (-0.46, 0.47), "CumXi_FA1_FA2": (-0.31, -0.35)}


def fa_class(n_prior) -> np.ndarray:
    """FA class from the number of prior in-sample seasons: 0-2 FA0, 3-5 FA1, 6+ FA2."""
    n = np.asarray(n_prior)
    return np.where(n <= 2, "FA0", np.where(n <= 5, "FA1", "FA2")).astype(object)


def real_salary(salary, season, cpi: dict, base_year: int):
    """Deflate nominal salary to ``base_year`` dollars: salary * CPI[base_year] / CPI[season].

    Seasons missing from ``cpi`` use the nearest available year. Returns (values, n_clamped).
    """
    yrs = np.array(sorted(cpi))
    vals = np.array([cpi[y] for y in yrs], dtype=float)
    s = np.asarray(season)
    near = yrs[np.abs(s[:, None] - yrs[None, :]).argmin(axis=1)]
    c = np.array([cpi[y] for y in near], dtype=float)
    return np.asarray(salary, dtype=float) * float(cpi[base_year]) / c, int((~np.isin(s, yrs)).sum())


def build_salary_frame(inputs: AnalysisInputs, cpi: dict | None = None, base_year: int | None = None):
    """Player-season frame for eq. (17) (see module docstring); returns (frame, n_cpi_clamped).

    Columns: player_id, season, pos, age, mlb_exp, ln_sal, fa (class), n_prior, cum_war,
    cum_pc, cum_pcx (Cum(pcWAR - xi)), cum_xi, team_exp_l.
    """
    if cpi is None:
        c = load_config("config/cpi.yaml")
        cpi, base_year = c["cpi_u"], int(c["base_year"] if base_year is None else base_year)
    base_year = int(base_year if base_year is not None else max(cpi))
    ps = inputs.player_seasons.sort_values(["player_id", "season"]).reset_index(drop=True)
    xi = table4_residuals(inputs)[["player_id", "season", "xi"]]
    ps = ps.merge(xi, on=["player_id", "season"], how="left")
    ps["xi0"] = ps["xi"].fillna(0.0)
    g = ps.groupby("player_id")
    ps["n_prior"] = g.cumcount()
    for out, col in [("cum_war", "war"), ("cum_pc", "pc_war"), ("cum_xi", "xi0")]:
        ps[out] = g[col].cumsum() - ps[col]
    ps["cum_pcx"] = ps["cum_pc"] - ps["cum_xi"]
    ps["team_exp_l"] = ps.groupby(["player_id", "franch_id"]).cumcount().astype(float)
    ps["fa"] = fa_class(ps["n_prior"])
    first = int(ps["season"].min())
    pl = inputs.players.drop_duplicates("player_id").set_index("player_id")["debut_season"]
    sal = inputs.salaries.groupby(["player_id", "season"], as_index=False)["salary"].sum()
    d = ps.merge(sal, on=["player_id", "season"], how="inner")
    d = d[(d["salary"] > 0) & (d["player_id"].map(pl) >= first)]
    d = d.dropna(subset=["pos", "age", "mlb_exp"]).copy()
    rs, ncl = real_salary(d["salary"], d["season"], cpi, base_year)
    d["ln_sal"] = np.log(rs)
    keep = ["player_id", "season", "pos", "age", "mlb_exp", "ln_sal", "fa", "n_prior", "cum_war", "cum_pc",
            "cum_pcx", "cum_xi", "team_exp_l"]
    return d[keep].reset_index(drop=True), ncl


def _design(d: pd.DataFrame, spec: int):
    """Raw regressor matrix and names for eq. (17) spec (1) or (2) (before demeaning)."""
    cols, names = [], []
    fa = d["fa"].to_numpy()

    def add(v, nm):
        cols.append(np.asarray(v, dtype=float))
        names.append(nm)

    for c in FA_CLASSES:
        m = (fa == c).astype(float)
        add(m * d["cum_war"], f"CumWAR_{c}")
        if spec == 1:
            add(m * d["cum_pc"], f"CumpcWAR_{c}")
        elif c != "FA0":
            add(m * d["cum_pcx"], f"CumPcMinusXi_{c}")
            add(m * d["cum_xi"], f"CumXi_{c}")
    for c in ("FA1", "FA2"):
        add((fa == c).astype(float), f"ind_{c}")
    for c in FA_CLASSES:
        add((fa == c) * d["team_exp_l"], f"TeamExp_{c}")
    z = (d["age"].to_numpy(float) - AGE_CENTER) / AGE_SCALE
    mk = d["mlb_exp"].to_numpy(float) / EXP_SCALE
    pos = [p for p in POSITIONS if (d["pos"] == p).any()]
    pos += sorted(set(d["pos"]) - set(pos))
    for i, p in enumerate(pos):
        ind = (d["pos"] == p).to_numpy(float)
        if i > 0:
            add(ind, f"pos_{p}")
        add(ind * z, f"age_{p}")
        add(ind * mk, f"mlb_{p}")
        add(ind * mk ** 2, f"mlb2_{p}")
    return np.column_stack(cols), names


def _demean(a: np.ndarray, codes: np.ndarray, n: int) -> np.ndarray:
    cnt = np.bincount(codes, minlength=n).astype(float)
    if a.ndim == 1:
        return a - (np.bincount(codes, weights=a, minlength=n) / cnt)[codes]
    mu = np.column_stack([np.bincount(codes, weights=a[:, j], minlength=n) for j in range(a.shape[1])]) / cnt[:, None]
    return a - mu[codes]


def _wls_dense(X, y, w=None):
    Xw = X if w is None else X * w[:, None]
    G = X.T @ Xw
    G[np.diag_indices_from(G)] += 1e-10 * max(np.trace(G) / G.shape[0], 1e-12)
    return sla.solve(G, Xw.T @ y, assume_a="pos")


def run(inputs: AnalysisInputs, n_boot: int = 500, seed: int = 0, max_jack: int | None = 200,
        cpi: dict | None = None, base_year: int | None = None, **opts) -> AnalysisResult:
    """Table 5 (eq. 17; paper section 5.2; spec section 8).

    Tables: ``table5`` (spec, term, fa, coef, se, ci_low, ci_high), ``table5_fit`` (spec, nobs,
    n_players, r2_within, r2_total (FE included)), ``meta`` (base_year, n_cpi_clamped).
    Without ``inputs.salaries`` returns an empty, skipped result.
    """
    if inputs.salaries is None:
        return AnalysisResult("salary", {}, [{"item": "salary skipped", "value": "inputs.salaries is None",
                                              "expected": "Table 5 needs Lahman salaries", "passed": None}])
    d, ncl = build_salary_frame(inputs, cpi, base_year)
    codes, _ = pd.factorize(d["player_id"])
    npl = int(codes.max()) + 1
    y_raw = d["ln_sal"].to_numpy(float)
    yt = _demean(y_raw, codes, npl)
    rows, fits, est = [], [], {}
    for s in (1, 2):
        Xr, names = _design(d, s)
        Xt = _demean(Xr, codes, npl)
        keep = [i for i, nm in enumerate(names) if nm.startswith(("Cum", "TeamExp", "ind_"))]
        bt = weighted_cluster_boot(d["player_id"], lambda w: _wls_dense(Xt, yt, w)[keep], n_boot, seed, max_jack)
        beta = _wls_dense(Xt, yt)
        res = yt - Xt @ beta
        fits.append({"spec": s, "nobs": len(d), "n_players": npl,
                     "r2_within": 1 - float(res @ res) / float(yt @ yt),
                     "r2_total": 1 - float(res @ res) / float(((y_raw - y_raw.mean()) ** 2).sum())})
        for j, i in enumerate(keep):
            nm = names[i]
            term, _, fa = nm.partition("_")
            rows.append({"spec": s, "term": term, "fa": fa, "coef": bt.estimate[j], "se": bt.se[j],
                         "ci_low": bt.ci_low[j], "ci_high": bt.ci_high[j]})
            est[(s, term, fa)] = bt.estimate[j]
    t5 = pd.DataFrame(rows)
    p = PAPER_T5
    war = [est[(1, "CumWAR", c)] for c in FA_CLASSES]
    pc2 = est[(1, "CumpcWAR", "FA2")]
    xi2 = est[(2, "CumXi", "FA2")]
    checks = [
        {"item": "spec1 FA2 return to CumpcWAR > 0", "value": pc2, "expected": f"paper fWAR +{p['CumpcWAR'][2]}",
         "passed": bool(pc2 > 0)},
        {"item": "spec2 FA2 return to Cum xi < 0", "value": xi2, "expected": f"paper fWAR {p['CumXi_FA1_FA2'][1]}",
         "passed": bool(xi2 < 0)},
        {"item": "spec2 FA2 return to Cum(pcWAR - xi)", "value": est[(2, "CumPcMinusXi", "FA2")],
         "expected": f"paper fWAR +{p['CumPcMinusXi_FA1_FA2'][1]}", "passed": bool(est[(2, "CumPcMinusXi", "FA2")] > 0)},
        {"item": "spec1 CumWAR return largest at FA1", "value": [round(v, 3) for v in war],
         "expected": f"paper fWAR {list(p['CumWAR'])} (FA0/FA1/FA2)", "passed": bool(np.argmax(war) == 1)},
        {"item": "spec1 CumpcWAR returns (FA0, FA1, FA2)", "value": [round(est[(1, 'CumpcWAR', c)], 3) for c in FA_CLASSES],
         "expected": f"paper fWAR {list(p['CumpcWAR'])}", "passed": None},
    ]
    meta = pd.DataFrame({"base_year": [base_year or int(load_config('config/cpi.yaml')['base_year'])],
                         "n_cpi_clamped": [ncl]})
    return AnalysisResult("salary", {"table5": t5, "table5_fit": pd.DataFrame(fits), "meta": meta}, checks)
