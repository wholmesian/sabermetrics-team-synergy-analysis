"""Recursive estimation, spec section 7 (chains the section 3-6 pipeline per window).

For every window [start, T] the full pipeline is re-estimated: team regression
(paper eq. 1), stint residuals (eqs. 4, 7), network blocks and SAR (eq. 8), two-factor
model (eq. 9, Appendix 7.2) and the metric decomposition (eqs. 10-13).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from team_synergy.config import config_metadata
from .decompose import decompose, player_season_metrics, sign_convention_diagnostics, team_metrics
from .factor_als import fit_factor_als
from .factor_prob import fit_factor_prob
from .network import build_blocks
from .residuals import player_residuals
from .sar import SARResult, fit_sar
from .team_reg import TeamRegResult, fit_team_regression

log = logging.getLogger(__name__)

_KEYS = ["player_id", "season", "franch_id"]
_STINT_METRICS = ["war", "weight", "y", "v", "c", "tp", "g", "own", "in_deg", "out_deg",
                  "war_minus", "war_plus", "pc_war", "pc_war_char", "pc_war_team",
                  "tc_war_contrib"]
# Panel covariates carried through to the stint metrics when present (spec section 8
# second-stage regressions need them); absent columns are skipped.
_PASSTHROUGH = ["group", "pos", "role", "age", "mlb_exp", "team_exp", "manager_id", "lg"]


@dataclass
class WindowResult:
    """Everything estimated on one window (spec section 7)."""

    window_end: int
    reg: TeamRegResult
    sar: SARResult                 # light: v and Phi are dropped, scalars + profile kept
    stints: pd.DataFrame           # stint keys + metrics (one row per stint of the window)
    teams: pd.DataFrame            # team-season metrics (+ W, eps, alpha, beta)
    player_seasons: pd.DataFrame   # player-season metrics, index (player_id, season)
    lam_als: pd.Series
    lam_prob: pd.Series
    diagnostics: dict
    factor_info: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)


def estimate_window(stints: pd.DataFrame, cfg: dict, weights_cfg: dict | None = None) -> WindowResult:
    """Run the spec sections 3-6 chain on ``stints`` (one window).

    Steps: fit_team_regression(beta_mode) [eq. 1] -> player_residuals(sign_convention)
    [eqs. 4, 7] -> build_blocks(history, optional kernel) [eq. 8 / spec 4] ->
    fit_sar(bounds) -> factor model ``cfg['factor_method']`` on v with appearance weights
    (``wls_weights``) -> decompose with g = c + tp * lambda split into g_char = c and
    g_team = tp * lambda [eqs. 10-13] -> team / player-season metrics and sign diagnostics.
    The other factor method is fitted as well, for lambda only (plan A lambda is not
    identified for <= 2 observed teams per player-season; spec 5).
    ``weights_cfg`` is accepted for API symmetry (weights are already in the panel's
    ``weight`` column) and is unused.
    """
    t0 = time.perf_counter()
    stints = stints.reset_index(drop=True)
    window_end = int(stints["season"].max())
    fcfg = cfg.get("factor", {})
    fkw = dict(scale=fcfg.get("scale", "tp_var"), tol=fcfg.get("tol", 1e-9),
               max_iter=fcfg.get("max_iter", 5000))
    method = cfg.get("factor_method", "als")
    if method not in ("als", "prob"):
        raise ValueError(f"unknown factor_method {method!r}")
    weights = stints["weight"].to_numpy(float) if cfg.get("wls_weights", "appearance") == "appearance" else None
    sign = cfg.get("sign_convention", "consistent")

    reg = fit_team_regression(stints, beta_mode=cfg.get("beta_mode", "estimated"))
    y = player_residuals(stints, reg, sign_convention=sign)["y"].to_numpy()
    k = cfg.get("adjacency_kernel", {}) or {}
    blocks = build_blocks(stints, history=cfg.get("history", "past"),
                          kernel_h=k.get("h", 1.0) if k.get("enabled") else None)
    t1 = time.perf_counter()
    sar = fit_sar(y, blocks, bounds=tuple(cfg.get("sar", {}).get("bounds", (-0.999, 0.999))))
    t2 = time.perf_counter()

    fits = {}
    for m, fn in (("als", fit_factor_als), ("prob", fit_factor_prob)):
        t = time.perf_counter()
        kw = dict(fkw)
        if m == "prob":
            # Plan B's EM log-likelihood flattens early; a looser default keeps a window at a few seconds.
            kw["tol"] = fcfg.get("prob_tol", 1e-7)
        fits[m] = fn(stints, sar.v, weights, **kw)
        fits[m].extra["seconds"] = time.perf_counter() - t
    fit = fits[method]
    t3 = time.perf_counter()

    # Character/team split of g (Fig. 8). Plan A's lambda is not identified when player-seasons
    # have <= 2 observed teams, so the team part tp*lambda comes from the factor fit named by
    # `culture_lambda_source` (plan B by default); the character part is the remainder of the
    # metric g, which keeps pc_war_char + pc_war_team = pc_war exactly. Deviation from eq. (11),
    # where both parts come from one fit; with culture_lambda_source == factor_method it is eq. (11).
    split = fits[cfg.get("culture_lambda_source", "prob")]
    f = stints[["player_id", "season"]].merge(split.factors, on=["player_id", "season"], how="left")
    tp = f["tp"].to_numpy()
    g_team = tp * stints["franch_id"].map(split.lam).to_numpy()
    g_char = fit.g - g_team
    c = g_char
    dec = decompose(stints, blocks, sar.Phi, fit.g, g_char, g_team, beta=reg.beta)
    dec["y"], dec["v"], dec["c"], dec["tp"] = y, sar.v, c, tp
    tm = team_metrics(dec, reg.beta)
    tm = tm.join(reg.teams[["W", "eps"]])
    tm["alpha"], tm["beta"] = reg.alpha, reg.beta
    ps = player_season_metrics(dec)
    diag = sign_convention_diagnostics(stints, dec, cfg.get("beta_mode", "estimated"))
    t4 = time.perf_counter()

    out = dec[_KEYS + [m for m in _STINT_METRICS if m in dec]].copy()
    cov = [c for c in _PASSTHROUGH if c in stints.columns]
    if cov:  # stints and dec share row order; keys are unique per stint
        out = pd.concat([out, stints[cov].reset_index(drop=True)], axis=1)
    light = replace(sar, v=np.empty(0), Phi=[])
    info = {m: dict(n_iter=int(fits[m].n_iter), converged=bool(fits[m].converged),
                    explained_share=float(fits[m].explained_share),
                    seconds=float(fits[m].extra["seconds"])) for m in fits}
    info["method"] = method
    timings = dict(network=t1 - t0, sar=t2 - t1, factor=t3 - t2, decompose=t4 - t3, total=t4 - t0)
    return WindowResult(window_end, reg, light, out, tm, ps, fits["als"].lam, fits["prob"].lam,
                        diag, info, timings)


def _culture_rank(lam: pd.Series) -> pd.Series:
    """Rank 0-100 by |lambda| (100 = largest), (rank - 1)/(n - 1) * 100 (spec section 8 Fig. 3;
    n - 1 = 29 for the 30 franchises)."""
    r = lam.abs().rank(method="average")
    return (r - 1) / max(len(lam) - 1, 1) * 100.0


def _summary_row(w: WindowResult) -> dict:
    fi = w.factor_info
    m = fi["method"]
    return dict(
        window_end=w.window_end, alpha=w.reg.alpha, beta=w.reg.beta, rho=w.sar.rho,
        rho_se=w.sar.se, at_bound=w.sar.at_bound, sigma2=w.sar.sigma2, n_stints=len(w.stints),
        factor_method=m, factor_iter=fi[m]["n_iter"], factor_converged=fi[m]["converged"],
        explained_share=fi[m]["explained_share"], als_iter=fi["als"]["n_iter"],
        prob_iter=fi["prob"]["n_iter"], prob_converged=fi["prob"]["converged"],
        ratio=w.diagnostics["ratio"], corr_tc_eps=w.diagnostics["corr_tc_eps"],
        sd_eps_war=w.diagnostics["sd_eps_war"], sd_eps_war_minus=w.diagnostics["sd_eps_war_minus"],
        seconds=w.timings["total"])


def _window(stints, cfg, start, T):
    sub = stints[(stints["season"] >= start) & (stints["season"] <= T)]
    w = estimate_window(sub, cfg)
    log.info("window [%d, %d] rho=%.3f beta=%.3f %.1fs", start, T, w.sar.rho, w.reg.beta,
             w.timings["total"])
    return w


def run_recursive(stints: pd.DataFrame, cfg: dict, start: int | None = None, end: int | None = None,
                  min_end: int | None = None, n_jobs: int = 1) -> dict:
    """Recursive estimation over windows [start, T], T = min_end..end (spec section 7).

    Defaults: start = first season, end = last season, min_end = start (first window is
    a single season).  Returns dict with ``metrics_recursive`` (season == T rows of each
    window, with ``window_end``), ``team_recursive`` (season T team metrics incl. tc_war,
    eps), ``lambda_windows`` (lam_als, lam_prob and rank 0-100 from |lambda| of
    ``culture_lambda_source``), ``window_summary``, and the full-sample window
    (window ending at ``end``): ``metrics_full``, ``team_full``, ``player_season_full``,
    plus ``full`` (its WindowResult).  ``n_jobs`` > 1 parallelises windows with joblib
    if it is installed (otherwise sequential).
    """
    seasons = sorted(stints["season"].unique())
    start = int(seasons[0] if start is None else start)
    end = int(seasons[-1] if end is None else end)
    min_end = start if min_end is None else int(min_end)
    Ts = [T for T in seasons if min_end <= T <= end and T >= start]
    if not Ts:
        raise ValueError("no windows: check start/end/min_end against the available seasons")
    src = cfg.get("culture_lambda_source", "prob")
    if src not in ("prob", "als"):
        raise ValueError(f"unknown culture_lambda_source {src!r}")

    results = None
    if n_jobs != 1:
        try:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=n_jobs)(delayed(_window)(stints, cfg, start, T) for T in Ts)
        except ImportError:
            log.warning("joblib not installed; running windows sequentially")
    if results is None:
        results = [_window(stints, cfg, start, T) for T in Ts]

    m_rows, t_rows, l_rows, s_rows = [], [], [], []
    for T, w in zip(Ts, results):
        m = w.stints[w.stints["season"] == T].copy()
        m.insert(0, "window_end", T)
        m_rows.append(m)
        t = w.teams.reset_index()
        t = t[t["season"] == T].copy()
        t.insert(0, "window_end", T)
        t_rows.append(t)
        lam = pd.DataFrame({"lam_als": w.lam_als, "lam_prob": w.lam_prob})
        lam["rank"] = _culture_rank(lam["lam_" + src])
        lam.index.name = "franch_id"
        lam = lam.reset_index()
        lam.insert(0, "window_end", T)
        l_rows.append(lam)
        s_rows.append(_summary_row(w))
    full = results[-1]
    return dict(
        metrics_recursive=pd.concat(m_rows, ignore_index=True),
        team_recursive=pd.concat(t_rows, ignore_index=True),
        lambda_windows=pd.concat(l_rows, ignore_index=True),
        window_summary=pd.DataFrame(s_rows),
        metrics_full=full.stints.copy(),
        team_full=full.teams.reset_index(),
        player_season_full=full.player_seasons.reset_index(),
        full=full,
    )


def save_outputs(results: dict, out_dir, cfg: dict, war_source: str) -> Path:
    """Write the recursive results to ``out_dir`` (spec section 7).

    Parquet: metrics_recursive, metrics_full, team_recursive, team_full,
    player_season_full, lambda_windows, window_summary; small CSVs window_summary.csv and
    lambda_windows.csv (trackable); run_metadata.json (config, war_source, seed,
    package version, UTC timestamp, row counts).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = ["metrics_recursive", "metrics_full", "team_recursive", "team_full",
             "player_season_full", "lambda_windows", "window_summary"]
    counts = {}
    for n in names:
        results[n].to_parquet(out / f"{n}.parquet", index=False)
        counts[n] = int(len(results[n]))
    results["window_summary"].to_csv(out / "window_summary.csv", index=False)
    results["lambda_windows"].to_csv(out / "lambda_windows.csv", index=False)
    try:
        from importlib.metadata import version
        ver = version("team-synergy")
    except Exception:  # package not installed
        ver = "unknown"
    meta = dict(config=config_metadata(cfg), war_source=war_source, seed=cfg.get("seed"),
                package_version=ver, timestamp=datetime.now(timezone.utc).isoformat(),
                row_counts=counts)
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    return out
