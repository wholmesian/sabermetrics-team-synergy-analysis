"""Diebold-Mariano test of equal predictive accuracy (paper Table 3; spec section 8)."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm, t as t_dist


def diebold_mariano(e1, e2, loss: str = "abs", h: int = 1, lags: int | None = None,
                    hln: bool = False) -> dict:
    """DM test on loss differential d_t = L(e1_t) - L(e2_t), L = |e| ("abs") or e^2 ("sq").

    H0: E[d] = 0. Sign: stat > 0 means model 1 has the LARGER loss (model 2 is better);
    stat < 0 means model 1 is better. Variance of mean(d): Newey-West long-run variance with
    Bartlett weights and ``lags`` (default: max(h - 1, floor(4 (T/100)^(2/9)))). Normal
    p-value (two-sided); with ``hln=True`` the Harvey-Leybourne-Newbold (1997) factor
    sqrt((T + 1 - 2h + h(h-1)/T) / T) is applied and a t(T-1) reference is used. If the
    differential is identically zero or the variance is not positive, stat = 0 and
    pvalue = 1. Returns dict(stat, pvalue, mean_diff, lags, n).
    """
    e1, e2 = np.asarray(e1, float), np.asarray(e2, float)
    ok = ~(np.isnan(e1) | np.isnan(e2))
    e1, e2 = e1[ok], e2[ok]
    if loss == "abs":
        d = np.abs(e1) - np.abs(e2)
    elif loss == "sq":
        d = e1 ** 2 - e2 ** 2
    else:
        raise ValueError("loss must be 'abs' or 'sq'")
    T = d.size
    if T < 2:
        raise ValueError("need at least 2 observations")
    L = max(h - 1, int(np.floor(4 * (T / 100) ** (2 / 9)))) if lags is None else int(lags)
    L = min(L, T - 1)
    dm = d.mean()
    x = d - dm
    lrv = x @ x / T
    for k in range(1, L + 1):
        lrv += 2 * (1 - k / (L + 1)) * (x[k:] @ x[:-k]) / T
    if not lrv > 1e-300:
        return dict(stat=0.0, pvalue=1.0, mean_diff=float(dm), lags=L, n=T)
    stat = dm / np.sqrt(lrv / T)
    if hln:
        stat *= np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
        p = 2 * t_dist.sf(abs(stat), T - 1)
    else:
        p = 2 * norm.sf(abs(stat))
    return dict(stat=float(stat), pvalue=float(p), mean_diff=float(dm), lags=L, n=T)
