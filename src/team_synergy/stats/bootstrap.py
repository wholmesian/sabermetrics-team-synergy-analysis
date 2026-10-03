"""Cluster bootstrap with BCa intervals (spec section 8; paper Tables 1, 3-5, Figs. 2, 10).

Only the second-stage regressions are resampled; the factor model and SAR estimates
(WAR-, pcWAR, tcWAR, ...) are held fixed. This ignores the sampling error of those
generated regressors, so the reported intervals are too narrow (generated-regressor
issue; stated in the README).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class BootResult:
    """estimate: stat on the full sample (1-D); se: bootstrap sd; ci_low/ci_high: BCa
    bounds (same shape); draws: (n_boot, k) bootstrap statistics."""

    estimate: np.ndarray
    se: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    draws: np.ndarray


def cluster_bootstrap(data: pd.DataFrame, cluster: str, stat_fn, n_boot: int = 500, seed: int = 0,
                      alpha: float = 0.05, jackknife: str = "cluster", max_jack: int | None = None) -> BootResult:
    """Cluster bootstrap with BCa intervals (Efron 1987).

    Whole clusters (values of ``data[cluster]``) are resampled with replacement;
    ``stat_fn(df) -> 1-D array`` is evaluated on each resample (several coefficients at
    once). Bias correction z0 = Phi^-1(share of draws < estimate) (ties count half);
    acceleration a = sum(d^3) / (6 sum(d^2)^1.5), d = jackknife mean - jackknife stat,
    from a leave-one-cluster-out jackknife (``jackknife="cluster"``). If the number of
    clusters exceeds ``max_jack`` the jackknife leaves out ``max_jack`` random groups of
    clusters instead (grouped jackknife; simplification to bound cost, fixed by ``seed``).
    ``jackknife="none"`` sets a = 0 (bias-corrected only). Draws on which ``stat_fn``
    fails or returns NaN are kept as NaN and ignored (nan-aware quantiles).
    Returns BootResult; CI level 1 - alpha.
    """
    rng = np.random.default_rng(seed)
    groups = {k: g for k, g in data.groupby(cluster, sort=True)}
    keys = list(groups)
    G = len(keys)
    est = np.atleast_1d(np.asarray(stat_fn(data), dtype=float))
    draws = np.full((n_boot, est.size), np.nan)
    for b in range(n_boot):
        pick = rng.integers(0, G, G)
        sample = pd.concat([groups[keys[i]] for i in pick], ignore_index=True)
        draws[b] = np.asarray(stat_fn(sample), dtype=float)

    # bias correction
    less = np.nanmean(draws < est, axis=0) + 0.5 * np.nanmean(draws == est, axis=0)
    less = np.clip(less, 1.0 / (2 * n_boot), 1 - 1.0 / (2 * n_boot))
    z0 = norm.ppf(less)

    # acceleration
    acc = np.zeros(est.size)
    if jackknife == "cluster" and G > 2:
        if max_jack is not None and G > max_jack:
            members = np.array_split(rng.permutation(G), max_jack)
        else:
            members = [np.array([i]) for i in range(G)]
        jk = []
        for m in members:
            keep = [keys[i] for i in range(G) if i not in set(m.tolist())]
            jk.append(np.asarray(stat_fn(pd.concat([groups[k] for k in keep], ignore_index=True)), dtype=float))
        jk = np.vstack(jk)
        d = np.nanmean(jk, axis=0) - jk
        den = 6.0 * np.nansum(d ** 2, axis=0) ** 1.5
        acc = np.divide(np.nansum(d ** 3, axis=0), den, out=np.zeros(est.size), where=den > 0)
    elif jackknife not in ("cluster", "none"):
        raise ValueError("jackknife must be 'cluster' or 'none'")

    zl, zu = norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)
    lo, hi = np.empty(est.size), np.empty(est.size)
    for j in range(est.size):
        col = draws[:, j]
        col = col[~np.isnan(col)]
        if col.size == 0:
            lo[j] = hi[j] = np.nan
            continue
        p = [norm.cdf(z0[j] + (z0[j] + z) / (1 - acc[j] * (z0[j] + z))) for z in (zl, zu)]
        lo[j], hi[j] = np.quantile(col, p)
    return BootResult(est, np.nanstd(draws, axis=0, ddof=1), lo, hi, draws)
