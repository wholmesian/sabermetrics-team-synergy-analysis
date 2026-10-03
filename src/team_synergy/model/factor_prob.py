"""Two-factor spatial model, plan B: probabilistic factor model EM (spec section 5 plan B).

Robustness alternative to the paper's EM-ALS (factor_als).  Not in the paper; it is the
spec's identified counterpart of eq. (9):

    f_r = (c_r, tp_r)' ~ N(mu, Sigma_f),  v_rn = c_r + tp_r * lambda_n + e_rn,
    e ~ N(0, sigma_e^2),  observed cells only.

The prior on f_r makes lambda identified from the cross-section (team-specific variance
of v and the variance of within-trade differences), unlike plan A where rows with <= 2
observed cells are fitted exactly for any lambda.

E-step: per row 2x2 posterior (P = Sigma^-1 + H'H/s2, m = P^-1(Sigma^-1 mu + H'v/s2)),
computed for all rows at once from per-row sums (bincount) of the observed cells.
M-step (ECM): mu, Sigma_f from the posterior moments; lambda_n by expected-complete-data
LS; sigma_e^2.  Constraints are exact reparametrisations of (mu, Sigma_f, lambda), so
they never change the likelihood: sum(lambda) = 0 by shifting c <- c + m*tp, and
Var(tp) = Sigma_f[tp,tp] = 1 by rescaling tp and lambda; sign: largest |lambda| > 0.

Deviations: ``weights`` (appearance weights) enter as precision multipliers
w / mean(w) on e (var = sigma_e^2 / w~); the paper/spec prescribe no weighting in plan B,
default None.  Initialisation without a seed is the deterministic SVD start of factor_als
(spec's lambda^0 = 0 is degenerate); with a seed a random centred normal.  The reported
g = posterior-mean c + tp*lambda is shrunk, so g != v; the difference is absorbed in the
own term (spec section 5).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .factor_als import FactorResult, _init_lambda, build_cells, explained_share


def _posterior(cells, lam, mu, Sig, s2, w):
    n = cells.n_rows
    h = lam[cells.ci]
    ri, v = cells.ri, cells.val
    a = np.bincount(ri, w, n)
    b = np.bincount(ri, w * h, n)
    d = np.bincount(ri, w * h * h, n)
    HtH = np.empty((n, 2, 2))
    HtH[:, 0, 0], HtH[:, 0, 1], HtH[:, 1, 0], HtH[:, 1, 1] = a, b, b, d
    Htv = np.stack([np.bincount(ri, w * v, n), np.bincount(ri, w * h * v, n)], 1)
    Si = np.linalg.inv(Sig)
    P = Si[None] + HtH / s2
    S = np.linalg.inv(P)
    m = np.einsum("nij,nj->ni", S, (Si @ mu)[None] + Htv / s2)
    return m, S, HtH, Htv


def _loglik(cells, lam, mu, Sig, s2, w, HtH, Htv):
    """Marginal log-likelihood (Woodbury / determinant lemma, per row, up to 2pi const)."""
    n = cells.n_rows
    h = lam[cells.ci]
    ri, v = cells.ri, cells.val
    cnt = np.bincount(ri, minlength=n)
    # r = v - H mu;  r'Wr, b = H'W r
    r = v - mu[0] - mu[1] * h
    rWr = np.bincount(ri, w * r * r, n)
    bvec = Htv - np.einsum("nij,j->ni", HtH, mu)
    Si = np.linalg.inv(Sig)
    P = Si[None] + HtH / s2
    # det(s2 W^-1 + H Sig H') = prod(s2/w) * det(I + Sig H'WH/s2)
    logdet = (cnt * np.log(s2) - np.bincount(ri, np.log(w), n)
              + np.linalg.slogdet(np.eye(2)[None] + Sig[None] @ HtH / s2)[1])
    sol = np.linalg.solve(P, bvec[:, :, None])[:, :, 0]
    quad = rWr / s2 - (bvec * sol).sum(1) / s2**2
    return float(-0.5 * (logdet + quad).sum())


def fit_factor_prob(stints, v, weights=None, scale="tp_var", tol=1e-9, max_iter=3000,
                    lam0=None, seed=None, return_cells=False) -> FactorResult:
    """Probabilistic two-factor EM (spec section 5 plan B; counterpart of eq. 9).

    Convergence: relative change of the marginal log-likelihood < tol (sse_path stores
    the observed-cell SSE of (v - g) per iteration, loglik_path in ``extra``).
    ``scale``: "tp_var" (Var(tp) = 1, spec) or "lambda_norm" (lambda'lambda = 1).
    """
    cells = build_cells(stints, v, None)  # unit weights for the cell values
    n, N = cells.n_rows, len(cells.cols)
    if weights is None:
        w = np.ones(len(cells.val))
    else:
        cw = cells.w
        w = np.maximum(cw / cw.mean(), 1e-12)
    lam = _init_lambda(cells, seed, lam0)
    vv = cells.val
    mu = np.array([vv.mean(), 0.0])
    Sig = np.diag([max(vv.var() / 2, 1e-6), max(vv.var() / 2, 1e-6)])
    s2 = max(vv.var() / 4, 1e-6)
    ll_path, sse_path, converged = [], [], False
    for it in range(1, max_iter + 1):
        m, S, HtH, Htv = _posterior(cells, lam, mu, Sig, s2, w)
        ll = _loglik(cells, lam, mu, Sig, s2, w, HtH, Htv)
        ll_path.append(ll)
        if it > 1 and abs(ll - ll_path[-2]) <= tol * abs(ll_path[-2]):
            converged = True
            break
        # M-step (ECM): lambda | posterior, then mu, Sigma, sigma2
        mc, mt = m[cells.ri, 0], m[cells.ri, 1]
        Sct, Stt = S[cells.ri, 0, 1], S[cells.ri, 1, 1]
        num = np.bincount(cells.ci, w * ((vv - mc) * mt - Sct), N)
        den = np.bincount(cells.ci, w * (mt**2 + Stt), N)
        lam = num / den
        Scc = S[cells.ri, 0, 0]
        res = vv - mc - mt * lam[cells.ci]
        s2 = float((w * (res**2 + Scc + 2 * lam[cells.ci] * Sct
                         + lam[cells.ci] ** 2 * Stt)).sum() / len(vv))
        mu = m.mean(0)
        dm = m - mu
        Sig = (S.sum(0) + dm.T @ dm) / n
        # exact reparametrisations: sum(lambda)=0, then scale
        k = lam.mean()
        T = np.array([[1.0, k], [0.0, 1.0]])
        lam, mu, Sig = lam - k, T @ mu, T @ Sig @ T.T
        s = np.sqrt(Sig[1, 1]) if scale == "tp_var" else np.linalg.norm(lam)
        if scale == "tp_var":
            D = np.diag([1.0, 1 / s])
            lam, mu, Sig = lam * s, D @ mu, D @ Sig @ D
        else:
            D = np.diag([1.0, s])
            lam, mu, Sig = lam / s, D @ mu, D @ Sig @ D
        gcur = m[:, 0][cells.stint_ri] + m[:, 1][cells.stint_ri] * lam[cells.stint_ci]
        sse_path.append(float(((np.asarray(v, float) - gcur) ** 2).sum()))
    # final posterior under the final parameters
    m, S, _, _ = _posterior(cells, lam, mu, Sig, s2, w)
    c, tp = m[:, 0], m[:, 1]
    if lam[np.argmax(np.abs(lam))] < 0:
        lam, tp = -lam, -tp
    g = c[cells.stint_ri] + tp[cells.stint_ri] * lam[cells.stint_ci]
    fac = cells.keys.assign(c=c, tp=tp)
    return FactorResult(pd.Series(lam, index=cells.cols, name="lam"), fac, g, it, converged,
                        sse_path, explained_share(v, g), "prob",
                        extra=dict(loglik_path=ll_path, sigma2=s2, mu=mu, Sigma=Sig))
