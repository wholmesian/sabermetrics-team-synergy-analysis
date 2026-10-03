"""Thin regression helpers for the team and player analyses (spec section 8, eqs. 1, 16, 17)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class OLSResult:
    """params / se: pd.Series by regressor (FE coefficients excluded for absorbed FE);
    r2; resid / fitted: aligned with the estimation sample's index; nobs; model: backend."""

    params: pd.Series
    se: pd.Series
    r2: float
    resid: pd.Series
    fitted: pd.Series
    nobs: int
    model: object = None


def ols(df: pd.DataFrame, y: str, X: list[str], fe: list[str] | None = None, cluster: str | None = None,
        absorb: list[str] | None = None, const: bool = True) -> OLSResult:
    """OLS of ``y`` on ``X`` (+ constant) with optional fixed effects and clustered SEs.

    ``fe``: small FE sets (league, team, manager, ...) entered as dummies (the dummy
    coefficients stay in ``params``); ``absorb``: large FE sets (player FE in eq. 17) absorbed
    with linearmodels AbsorbingLS (no constant; r2 = linearmodels ``rsquared``, the share
    of variance explained by X and the absorbed FE together; ``fitted`` = y - resid). ``cluster``: column for cluster-robust SEs (CR1 via
    statsmodels; AbsorbingLS clustered covariance). Rows with NaN in any used column are
    dropped. Simplification: no small-sample correction beyond the backend default.
    """
    cols = [y] + list(X) + (fe or []) + (absorb or []) + ([cluster] if cluster else [])
    d = df[list(dict.fromkeys(cols))].dropna()
    Xm = d[list(X)].astype(float)
    if fe:
        Xm = pd.concat([Xm, pd.get_dummies(d[fe], drop_first=True, dtype=float)], axis=1)
    if absorb:
        from linearmodels.iv.absorbing import AbsorbingLS
        a = d[absorb].astype("category")
        mod = AbsorbingLS(d[y].astype(float), Xm, absorb=a)
        r = mod.fit(cov_type="clustered", clusters=d[cluster]) if cluster else mod.fit(cov_type="robust")
        resid = pd.Series(np.asarray(r.resids).ravel(), index=d.index)  # full residual (FE included)
        fitted = d[y].astype(float) - resid
        return OLSResult(r.params, r.std_errors, float(r.rsquared), resid, fitted, int(r.nobs), r)
    if const:
        Xm = sm.add_constant(Xm, has_constant="add")
    mod = sm.OLS(d[y].astype(float), Xm)
    r = mod.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d[cluster])[0]}) if cluster else mod.fit()
    return OLSResult(r.params, r.bse, float(r.rsquared), r.resid, r.fittedvalues, int(r.nobs), r)
