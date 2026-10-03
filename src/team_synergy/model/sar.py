"""First-stage spatial autoregression y = rho A y + u, paper eq. (8); spec section 4."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from .network import Block


@dataclass
class SARResult:
    rho: float
    se: float
    sigma2: float
    loglik: float
    v: np.ndarray            # (I - rho A) y in stint order
    Phi: list                # per-block (I - rho A_b)^-1
    profile: pd.DataFrame    # rho / loglik grid
    at_bound: bool = False


def _prep(y: np.ndarray, blocks: list[Block]):
    y = np.asarray(y, float)
    Ay = np.empty_like(y)
    for b in blocks:
        Ay[b.idx] = b.A @ y[b.idx]
    omega = np.concatenate([b.eigvals for b in blocks])
    return y, Ay, omega


def _ll(rho: float, y: np.ndarray, Ay: np.ndarray, omega: np.ndarray) -> float:
    v = y - rho * Ay
    return float(np.log1p(-rho * omega).sum() - 0.5 * len(y) * np.log(v @ v / len(y)))


def concentrated_loglik(rho: float, y: np.ndarray, blocks: list[Block]) -> float:
    """Concentrated log-likelihood of the panel SAR (eq. 8, spec section 4).

    l_c(rho) = sum_b sum_k log(1 - rho * omega_bk) - (N/2) log sigma2_hat(rho),
    sigma2_hat(rho) = ||(I - rho A) y||^2 / N.
    """
    y, Ay, omega = _prep(y, blocks)
    return _ll(rho, y, Ay, omega)


def fit_sar(y: np.ndarray, blocks: list[Block], bounds=(-0.999, 0.999)) -> SARResult:
    """ML estimate of rho (eq. 8, spec section 4).

    Bounded scalar maximization of l_c; SE from the central-difference second
    derivative of l_c (simplification: curvature of the concentrated, not the full,
    likelihood, so sigma2 uncertainty is ignored). ``at_bound`` flags rho_hat within
    1e-3 of a bound. Returns filtered residual v and block inverses Phi_b.
    """
    y, Ay, omega = _prep(y, blocks)
    N = len(y)
    f = lambda r: -_ll(r, y, Ay, omega)
    res = minimize_scalar(f, bounds=bounds, method="bounded", options={"xatol": 1e-8})
    rho = float(res.x)
    h = 1e-3
    lo = min(max(rho, bounds[0] + h), bounds[1] - h)  # keep stencil inside the bounds
    d2 = (_ll(lo + h, y, Ay, omega) - 2 * _ll(lo, y, Ay, omega) + _ll(lo - h, y, Ay, omega)) / h**2
    se = float(1 / np.sqrt(-d2)) if d2 < 0 else float("nan")
    v = y - rho * Ay
    grid = np.linspace(bounds[0], bounds[1], 200)
    prof = pd.DataFrame({"rho": grid, "loglik": [_ll(g, y, Ay, omega) for g in grid]})
    Phi = [np.linalg.solve(np.eye(len(b.idx)) - rho * b.A, np.eye(len(b.idx))) for b in blocks]
    at_bound = bool(min(abs(rho - bounds[0]), abs(rho - bounds[1])) < 1e-3)
    return SARResult(rho, se, float(v @ v / N), _ll(rho, y, Ay, omega), v, Phi, prof, at_bound)
