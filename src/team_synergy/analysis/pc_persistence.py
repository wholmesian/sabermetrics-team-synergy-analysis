"""pcWAR persistence: Table 4, eq. (16) and Fig. 9 (paper section 5.1; spec section 8).

This module is the single source of truth for the eq. (16) design matrices and fits; the
profiles (Fig. 10), intangibles (Fig. 11) and salary (Table 5) modules import from here.

Eq. (16), per player-season (traded players = sum over stints; team / manager / league from
the stint with the largest weight, see ``player_season_table``)::

    pcWAR_it = sum_k rho_k pcWAR_{i,t-1} 1[tier_{i,t-1}=k]
             + sum_p 1[pos_it=p] (gamma_p + sum_{m=1..4} theta_pm age_it^m)
             + delta' X_it + phi' Z_it + xi_it

tier_{t-1}: Scrub WAR_{t-1} < 1, Role 1 <= . < 4, Star >= 4. Sample: player-seasons whose
previous season (t-1, consecutive) is also in the data. Spec (1): tier dummies + the three
tier interactions, no constant. Spec (2): adds WAR_t, MLB experience and team experience
(games), league + team + manager fixed effects, 12 position indicators and position x age
polynomials. The Scrub dummy is absorbed by the position indicators (paper note), so only the
Role and Star dummies enter spec (2).

Implementation notes / deviations from the paper text:
* age is centered and scaled, z = (age - 30) / 10, before taking powers (conditioning only;
  fitted values and xi are invariant to this affine map, theta_pm are in z units).
* mlb_exp and team_exp enter in units of 1000 games (conditioning; coefficients are per 1000
  games).
* Fixed effects (league, team, manager) are dummies in a sparse design. The dummy sets are
  not linearly independent (managers nest in teams, leagues in teams), so the normal equations
  are solved with a tiny ridge (1e-10 x mean diagonal), which picks the minimum-norm solution
  on the null space: fitted values, xi and all non-FE coefficients are unaffected.
* R^2 is the centered R^2 (1 - SSE / sum (y - ybar)^2), also for the no-constant spec (1).
* Standard errors / CIs: BCa on a player-cluster bootstrap (``cluster_bootstrap``), refitting
  by weighted least squares with the draw's cluster multiplicities as row weights (identical
  to refitting on the concatenated resample). Factor-model estimates are held fixed.
* Fig. 9 uses the spec (1) residuals summed over a player's career vs. the mean of his
  WAR_{t-1} over the sample rows; reference lines at the ``line_q`` / 1 - ``line_q`` quantiles
  (paper: extremes of the distribution; spec default 1%).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.linalg as sla
import scipy.sparse as sp

from team_synergy.analysis.inputs import AnalysisInputs
from team_synergy.analysis.result import AnalysisResult
from team_synergy.stats.bootstrap import BootResult, cluster_bootstrap

TIERS = ["Scrub", "Role", "Star"]
POSITIONS = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH", "UT", "SP", "RP"]
AGE_CENTER, AGE_SCALE = 30.0, 10.0
EXP_SCALE = 1000.0
_RIDGE = 1e-10
# Paper Table 4 (fWAR) reference values.
PAPER_T4 = {"spec1_rho": (0.128, 0.375, 0.576), "spec2_rho": (0.044, 0.168, 0.244),
            "spec2_war": 0.094, "r2": (0.214, 0.795), "nobs": 20735, "nplayers": 4112}


def assign_tier(war_lag) -> np.ndarray:
    """Tier of WAR_{t-1} (eq. 16): Scrub < 1, Role in [1, 4), Star >= 4."""
    w = np.asarray(war_lag, dtype=float)
    out = np.where(w < 1.0, "Scrub", np.where(w < 4.0, "Role", "Star")).astype(object)
    out[np.isnan(w)] = None
    return out


def find_player_id(players: pd.DataFrame, name: str) -> str | None:
    """player_id whose ``name`` equals ``name`` exactly (first match), else None."""
    hit = players.loc[players["name"] == name, "player_id"]
    return None if hit.empty else str(hit.iloc[0])


def player_names(players: pd.DataFrame) -> pd.Series:
    """player_id -> name map (Series)."""
    return players.drop_duplicates("player_id").set_index("player_id")["name"]


def build_frame(ps: pd.DataFrame) -> pd.DataFrame:
    """Estimation sample of eq. (16): player-seasons with a t-1 player-season.

    Adds war_lag, pc_lag (previous season's WAR / pcWAR), tier (of WAR_{t-1}) and the scaled
    regressors z (centered age), mlb_k, team_k (1000 games). Rows with missing covariates
    are dropped. Index is reset (row i of the frame = row i of the design matrices).
    """
    lag = ps[["player_id", "season", "war", "pc_war"]].copy()
    lag["season"] += 1
    lag = lag.rename(columns={"war": "war_lag", "pc_war": "pc_lag"})
    d = ps.merge(lag, on=["player_id", "season"], how="inner")
    need = ["pos", "age", "mlb_exp", "team_exp", "lg", "franch_id", "manager_id", "war", "pc_war"]
    d = d.dropna(subset=[c for c in need if c in d.columns]).reset_index(drop=True)
    d["tier"] = assign_tier(d["war_lag"])
    d["z"] = (d["age"].astype(float) - AGE_CENTER) / AGE_SCALE
    d["mlb_k"] = d["mlb_exp"].astype(float) / EXP_SCALE
    d["team_k"] = d["team_exp"].astype(float) / EXP_SCALE
    return d


def _dummies(s: pd.Series, prefix: str, drop_first: bool):
    codes, uniq = pd.factorize(s, sort=True)
    n = len(s)
    X = sp.csr_matrix((np.ones(n), (np.arange(n), codes)), shape=(n, len(uniq)))
    names = [f"{prefix}_{u}" for u in uniq]
    if drop_first:
        X, names = X[:, 1:], names[1:]
    return X, names


@dataclass
class Design:
    """Sparse design matrix of one eq. (16) specification (row-aligned with the frame)."""

    X: sp.csr_matrix
    names: list[str]
    key_idx: dict[str, int]  # key regressors (rho_*, d_*, war, mlb_exp, team_exp)
    age_cols: dict[str, list[int]]  # position -> column index of z^1..z^4 (spec 2 only)
    pos_present: list[str]


def build_design(d: pd.DataFrame, spec: int) -> Design:
    """Design matrix for Table 4 spec (1) or (2) (eq. 16; see module docstring)."""
    n = len(d)
    cols, names = [], []
    tier = d["tier"].to_numpy()
    for k in TIERS:  # rho_k: pcWAR_{t-1} x tier dummy
        cols.append(sp.csr_matrix(np.where(tier == k, d["pc_lag"].to_numpy(), 0.0)[:, None]))
        names.append(f"rho_{k}")
    dum_tiers = TIERS if spec == 1 else ["Role", "Star"]  # Scrub dummy absorbed by positions in (2)
    for k in dum_tiers:
        cols.append(sp.csr_matrix((tier == k).astype(float)[:, None]))
        names.append(f"d_{k}")
    age_cols: dict[str, list[int]] = {}
    pos_present: list[str] = []
    if spec == 2:
        for nm, c in [("war", "war"), ("mlb_exp", "mlb_k"), ("team_exp", "team_k")]:
            cols.append(sp.csr_matrix(d[c].to_numpy(float)[:, None]))
            names.append(nm)
        pos_present = [p for p in POSITIONS if (d["pos"] == p).any()]
        pos_present += sorted(set(d["pos"]) - set(pos_present))
        z = d["z"].to_numpy(float)
        for p in pos_present:
            ind = (d["pos"] == p).to_numpy(float)
            cols.append(sp.csr_matrix(ind[:, None]))
            names.append(f"pos_{p}")
        for p in pos_present:
            ind = (d["pos"] == p).to_numpy(float)
            age_cols[p] = []
            for m in range(1, 5):
                age_cols[p].append(len(names))
                cols.append(sp.csr_matrix((ind * z ** m)[:, None]))
                names.append(f"age{m}_{p}")
        for c, pre in [("lg", "lg"), ("franch_id", "team"), ("manager_id", "mgr")]:
            Xc, nc = _dummies(d[c], pre, drop_first=True)
            cols.append(Xc)
            names += nc
    X = sp.hstack(cols, format="csr")
    key = {nm: i for i, nm in enumerate(names)
           if nm.startswith(("rho_", "d_")) or nm in ("war", "mlb_exp", "team_exp")}
    return Design(X, names, key, age_cols, pos_present)


def wls(X: sp.csr_matrix, y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Weighted least squares via sparse normal equations with a tiny ridge (see docstring)."""
    if w is None:
        Xw = X
        b = X.T @ y
    else:
        Xw = sp.diags(w) @ X
        b = X.T @ (w * y)
    G = (X.T @ Xw).toarray()
    G[np.diag_indices_from(G)] += _RIDGE * max(np.trace(G) / G.shape[0], 1e-12)
    return sla.solve(G, np.asarray(b).ravel(), assume_a="pos")


@dataclass
class Table4Data:
    """Eq. (16) sample, designs, y and the point-estimate fits of both specs."""

    frame: pd.DataFrame
    y: np.ndarray
    designs: dict[int, Design]
    beta: dict[int, np.ndarray]
    resid: dict[int, np.ndarray]
    r2: dict[int, float]

    def fit(self, spec: int, w: np.ndarray | None = None) -> np.ndarray:
        return wls(self.designs[spec].X, self.y, w)


def build_table4(ps: pd.DataFrame) -> Table4Data:
    """Fit Table 4 specs (1) and (2) on a player-season table (eq. 16 point estimates)."""
    d = build_frame(ps)
    if len(d) == 0:
        raise ValueError("no player-season has a consecutive previous season")
    y = d["pc_war"].to_numpy(float)
    sst = float(((y - y.mean()) ** 2).sum())
    designs, beta, resid, r2 = {}, {}, {}, {}
    for s in (1, 2):
        designs[s] = build_design(d, s)
        beta[s] = wls(designs[s].X, y)
        resid[s] = y - designs[s].X @ beta[s]
        r2[s] = 1.0 - float((resid[s] ** 2).sum()) / sst
    return Table4Data(d, y, designs, beta, resid, r2)


def table4_residuals(inputs: AnalysisInputs) -> pd.DataFrame:
    """Per player-season residuals: columns player_id, season, resid1 (spec 1), xi (spec 2).

    xi is the "intangibles" of paper section 5.1 (Fig. 11, Table 5 spec 2). Player-seasons
    without a previous season have no row (xi undefined).
    """
    t = build_table4(inputs.player_seasons)
    out = t.frame[["player_id", "season"]].copy()
    out["resid1"], out["xi"] = t.resid[1], t.resid[2]
    return out


def weighted_cluster_boot(player_ids, stat_w, n_boot: int, seed: int, max_jack: int | None = 200) -> BootResult:
    """BCa player-cluster bootstrap where ``stat_w(w)`` takes per-row multiplicity weights.

    Equivalent to ``cluster_bootstrap`` on concatenated resamples (a drawn cluster k times
    gets weight k on all its rows) but avoids rebuilding the data every draw.
    """
    pid = np.asarray(player_ids)
    n = len(pid)
    data = pd.DataFrame({"player_id": pid, "_row": np.arange(n)})
    return cluster_bootstrap(data, "player_id",
                             lambda s: stat_w(np.bincount(s["_row"].to_numpy(), minlength=n).astype(float)),
                             n_boot=n_boot, seed=seed, max_jack=max_jack)


def run(inputs: AnalysisInputs, n_boot: int = 500, seed: int = 0, max_jack: int | None = 200,
        line_q: float = 0.01, **opts) -> AnalysisResult:
    """Table 4 (eq. 16) and Fig. 9 (paper section 5.1; spec section 8).

    Tables: ``table4`` (spec, term, coef, se, ci_low, ci_high; BCa 95% on a player-cluster
    bootstrap), ``table4_fit`` (spec, r2, nobs, n_players), ``fig9`` (per player: career sum
    of spec (1) residuals vs. mean previous-season WAR), ``fig9_lines`` (lower / upper
    ``line_q`` quantile of the career sums), ``xi`` (player_id, season, resid1, xi).
    """
    t = build_table4(inputs.player_seasons)
    d = t.frame
    keys = {s: list(t.designs[s].key_idx) for s in (1, 2)}
    idx = {s: [t.designs[s].key_idx[k] for k in keys[s]] for s in (1, 2)}

    def stat(w):
        return np.concatenate([t.fit(1, w)[idx[1]], t.fit(2, w)[idx[2]]])

    bt = weighted_cluster_boot(d["player_id"], stat, n_boot, seed, max_jack)
    rows, off = [], 0
    for s in (1, 2):
        for j, k in enumerate(keys[s]):
            i = off + j
            rows.append({"spec": s, "term": k, "coef": bt.estimate[i], "se": bt.se[i],
                         "ci_low": bt.ci_low[i], "ci_high": bt.ci_high[i]})
        off += len(keys[s])
    t4 = pd.DataFrame(rows)
    npl = int(d["player_id"].nunique())
    fit = pd.DataFrame({"spec": [1, 2], "r2": [t.r2[1], t.r2[2]], "nobs": len(d), "n_players": npl})

    # Fig. 9
    f9 = (d.assign(resid1=t.resid[1]).groupby("player_id")
          .agg(career_resid=("resid1", "sum"), mean_prev_war=("war_lag", "mean"), n_seasons=("resid1", "size"))
          .reset_index())
    nm = player_names(inputs.players)
    f9.insert(1, "name", f9["player_id"].map(nm))
    lines = pd.DataFrame({"line": ["lower", "upper"], "q": [line_q, 1 - line_q],
                          "value": f9["career_resid"].quantile([line_q, 1 - line_q]).to_numpy()})
    xi = d[["player_id", "season"]].assign(resid1=t.resid[1], xi=t.resid[2])

    def c(term, spec):
        return float(t4.loc[(t4.spec == spec) & (t4.term == term), "coef"].iloc[0])

    r1 = {k: c(f"rho_{k}", 1) for k in TIERS}
    p1, p2 = PAPER_T4["spec1_rho"], PAPER_T4["spec2_rho"]
    ratio_ss = r1["Star"] / r1["Scrub"] if r1["Scrub"] != 0 else np.nan
    ratio_sr = r1["Star"] / r1["Role"] if r1["Role"] != 0 else np.nan
    real = inputs.war_source != "synthetic"
    checks = [
        {"item": "spec1 persistence Star/Scrub", "value": ratio_ss,
         "expected": f"approx 5 (paper fWAR {p1[2]}/{p1[0]} = {p1[2] / p1[0]:.1f}); accepted band 3-8",
         "passed": bool(3 <= ratio_ss <= 8)},
        {"item": "spec1 persistence Star/Role", "value": ratio_sr,
         "expected": f"approx 1.5 (paper {p1[2]}/{p1[1]} = {p1[2] / p1[1]:.2f}); accepted band 1.1-2.2",
         "passed": bool(1.1 <= ratio_sr <= 2.2)},
        {"item": "spec1 rho increasing Scrub<Role<Star", "value": [round(r1[k], 3) for k in TIERS],
         "expected": f"paper {list(p1)}", "passed": bool(r1["Scrub"] < r1["Role"] < r1["Star"])},
        {"item": "R2 rises from spec1 to spec2", "value": [round(t.r2[1], 3), round(t.r2[2], 3)],
         "expected": f"paper {PAPER_T4['r2'][0]} -> {PAPER_T4['r2'][1]}", "passed": bool(t.r2[2] > t.r2[1])},
        {"item": "spec2 rho (Scrub, Role, Star)", "value": [round(c(f"rho_{k}", 2), 3) for k in TIERS],
         "expected": f"paper {list(p2)}; WAR_t coefficient {PAPER_T4['spec2_war']}", "passed": None},
        {"item": "sample size (obs, players)", "value": [len(d), npl],
         "expected": f"paper {PAPER_T4['nobs']} obs, {PAPER_T4['nplayers']} players",
         "passed": None},
    ]
    # Fig. 9 named-player checks (real data only)
    for nme, which in [("Derek Jeter", "jeter"), ("Adrian Beltre", "beltre")]:
        pid = find_player_id(inputs.players, nme) if real else None
        row = f9[f9["player_id"] == pid] if pid else f9.iloc[0:0]
        if row.empty:
            val, ok = "not available (synthetic data or player not in sample)", None
        elif which == "jeter":
            stars = f9[f9["mean_prev_war"] >= 4]
            rk = int((stars["career_resid"] < row["career_resid"].iloc[0]).sum()) + 1
            val, ok = f"rank {rk} from bottom among {len(stars)} Stars", bool(rk == 1)
        else:
            rk = int((f9["career_resid"] > row["career_resid"].iloc[0]).sum()) + 1
            val, ok = f"rank {rk} of {len(f9)} (top)", bool(rk <= 3)
        checks.append({"item": f"Fig. 9: {nme}", "value": val,
                       "expected": "Jeter: negative extreme among Stars" if which == "jeter"
                       else "Beltre: top of the distribution", "passed": ok})
    return AnalysisResult("pc_persistence", {"table4": t4, "table4_fit": fit, "fig9": f9,
                                             "fig9_lines": lines, "xi": xi}, checks)
