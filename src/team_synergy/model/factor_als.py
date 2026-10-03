"""Two-factor spatial model, plan A: EM-ALS (paper eq. 9, Appendix 7.2; spec section 5).

Model (eq. 9): eps_hat = Omega F Lambda, with F = [c, tp] (player-season factors) and
Lambda = [1'; lambda'] (first row a unit vector across franchises, non-unit row summing
to zero, Reis-Watson 2010 loading restrictions).  After the first-stage SAR, the
filtered residual v = (I - rho_hat A) y satisfies v = F Lambda, so the factors are
estimated on the (player-season x franchise) matrix V with most cells missing.

Deviations from the paper (all deliberate, spec section 5 / 10):
* The paper's Lambda-step (eq. 20) is a regression in Y-space,
  Lambda = (F' Omega' Omega F)^-1 F' Omega' Y, and the F-step (eq. 21) uses
  Omega^-1 Y.  Here both steps run in V-space (V = Omega^-1 Y); both give the same g
  when every row has a single observed cell.  The spec adopts the V-space version
  with appearance-weight WLS for lambda.
* Scale normalisation is Var(tp) = 1 (config ``factor.scale: tp_var``); the paper's
  Lambda Lambda' = I analogue (lambda'lambda = 1) is available as ``lambda_norm``.
* Initialisation: spec step 1 says lambda^0 = 0, but then tp = 0 and the lambda-step
  divides by zero.  Without a seed we therefore start from the leading right singular
  vector of the row-mean-centred filled matrix (deterministic); with a seed, from a
  random centred normal vector.
* The lambda-step uses observed cells only (weight 0 for filled cells), a direct ALS
  rather than the full-EM filled-cell version.
* F-step acceleration (``fstep="prox"``, default): literally refilling missing cells and
  re-solving (``fstep="em"``, spec steps 2 and 4) converges only sub-linearly (rate
  ~1 - k/N for a row with k of N cells observed; >5000 iterations on the synthetic
  panel).  ``prox`` takes the fixed point of that refill loop directly: per row,
  min_F sum_obs w (v - c - tp lam_n)^2 + delta (F - F_old)' (L L'/N) (F - F_old) with a
  tiny delta, i.e. observed cells are fitted (nearly) exactly and the unidentified
  directions stay at their previous value, as under EM.  Missing cells are then implicit
  (= F L), which is what the refill step would produce.
* Identification caveat (spec section 5): every row with <= 2 observed cells is
  fitted exactly by (c, tp), so the observed-cell SSE is ~0 for ANY lambda and lambda is
  determined only by the EM path / initial value.  Plan B (factor_prob) is the
  identified alternative.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass
class FactorResult:
    lam: pd.Series          # lambda by franch_id (sum 0)
    factors: pd.DataFrame   # player_id, season, c, tp
    g: np.ndarray           # stint order: c + tp * lam[franch]
    n_iter: int
    converged: bool
    sse_path: list
    explained_share: float  # 1 - SSE(v - g) / SS(v - mean(v)) over observed stints
    method: str
    extra: dict = field(default_factory=dict)


@dataclass
class _Cells:
    ri: np.ndarray        # player-season row of each cell
    ci: np.ndarray        # franchise column of each cell
    val: np.ndarray       # cell value (weighted mean over duplicate stints)
    w: np.ndarray         # cell weight (sum of stint weights)
    stint_ri: np.ndarray
    stint_ci: np.ndarray
    n_rows: int
    cols: np.ndarray      # franchise labels
    keys: pd.DataFrame    # player_id, season per row


def build_cells(stints: pd.DataFrame, v, weights=None) -> _Cells:
    """Map stints to the sparse (player-season x franchise) matrix V of spec section 5.

    Duplicate (player-season, franchise) stints are merged into one cell with the
    weighted mean value (never happens in the synthetic panel)."""
    v = np.asarray(v, float)
    w = np.ones(len(v)) if weights is None else np.asarray(weights, float)
    ps = stints[["player_id", "season"]].reset_index(drop=True)
    sri = ps.groupby(["player_id", "season"], sort=True).ngroup().to_numpy()
    keys = ps.drop_duplicates().assign(_k=lambda d: sri[d.index]).sort_values("_k")
    keys = keys[["player_id", "season"]].reset_index(drop=True)
    cols, sci = np.unique(stints["franch_id"].to_numpy(), return_inverse=True)
    n_cols = len(cols)
    uniq, inv = np.unique(sri * n_cols + sci, return_inverse=True)
    ws = np.bincount(inv, w)
    val = np.bincount(inv, w * v) / np.where(ws > 0, ws, 1.0)
    return _Cells(uniq // n_cols, uniq % n_cols, val, ws, sri, sci, len(keys), cols, keys)


def _init_lambda(cells: _Cells, seed, lam0) -> np.ndarray:
    n_cols = len(cells.cols)
    if lam0 is not None:
        lam = np.asarray(lam0, float).copy()
    elif seed is not None:
        lam = np.random.default_rng(seed).normal(size=n_cols)
    else:
        # leading right singular vector of row-mean-centred (zero-filled) V
        dev = _fill_rowmean(cells) - _row_means(cells)[:, None]
        lam = np.linalg.svd(dev, full_matrices=False)[2][0]
    lam = lam - lam.mean()
    return lam / np.linalg.norm(lam)


def _row_means(cells: _Cells) -> np.ndarray:
    cnt = np.bincount(cells.ri, minlength=cells.n_rows)
    return np.bincount(cells.ri, cells.val, minlength=cells.n_rows) / np.maximum(cnt, 1)


def _fill_rowmean(cells: _Cells) -> np.ndarray:
    V = np.repeat(_row_means(cells)[:, None], len(cells.cols), axis=1)
    V[cells.ri, cells.ci] = cells.val
    return V


def _normalise(c, tp, lam, scale):
    """Spec section 5 step 4: scale (Var(tp)=1 or lam'lam=1) and sign (largest |lam| > 0)."""
    if scale == "tp_var":
        s = tp.std()
    elif scale == "lambda_norm":
        s = np.linalg.norm(lam)
    else:
        raise ValueError(f"unknown scale {scale!r}")
    if s > 0:
        if scale == "tp_var":
            tp, lam = tp / s, lam * s
        else:
            tp, lam = tp * s, lam / s
    if lam[np.argmax(np.abs(lam))] < 0:
        tp, lam = -tp, -lam
    return c, tp, lam


def _prox_fstep(cells, lam, c, tp, delta):
    """Row-wise proximal LS update of F = (c, tp) on observed cells (see module docstring)."""
    n, N = cells.n_rows, len(lam)
    h = lam[cells.ci]
    w, v, ri = cells.w, cells.val, cells.ri
    # normal equations H'WH F = H'Wv per row (2x2), via bincount
    a = np.bincount(ri, w, n)
    b = np.bincount(ri, w * h, n)
    d = np.bincount(ri, w * h * h, n)
    r0 = np.bincount(ri, w * v, n)
    r1 = np.bincount(ri, w * h * v, n)
    sc = max(w.mean(), 1e-12)
    M = np.array([[1.0, lam.mean()], [lam.mean(), lam @ lam / N]]) * (delta * sc)
    A = np.empty((n, 2, 2))
    A[:, 0, 0], A[:, 0, 1], A[:, 1, 0], A[:, 1, 1] = a, b, b, d
    A = A + M
    rhs = np.stack([r0, r1], 1) + np.stack([c, tp], 1) @ M.T
    F = np.linalg.solve(A, rhs[:, :, None])[:, :, 0]
    return F[:, 0], F[:, 1]


def explained_share(v, g) -> float:
    v = np.asarray(v, float)
    return float(1 - ((v - g) ** 2).sum() / ((v - v.mean()) ** 2).sum())


def fit_factor_als(stints, v, weights=None, scale="tp_var", tol=1e-9, max_iter=5000,
                   lam0=None, seed=None, fstep="prox", delta=1e-6) -> FactorResult:
    """EM-ALS for the two-factor model (eq. 9; Appendix 7.2 eqs. 20-21; spec section 5 plan A).

    Steps (spec section 5): (1) fill missing cells with row means; (2) F-step
    F = V~ L'(L L')^-1 (eq. 21, V-space); (3) per-franchise WLS
    lam_n = sum w tp (V_n - c) / sum w tp^2 over observed cells (eq. 20 analogue, V-space),
    then re-centre sum(lam)=0 moving mean*tp into c; (4) refill only missing cells with
    F L, normalise scale and sign; (5) stop when the relative change of the observed-cell
    SSE is < tol.  ``weights`` are stint appearance weights eta*tau (None = unweighted).
    ``seed``/``lam0`` select the starting lambda (see module docstring for the
    deviation from lambda^0 = 0).  ``fstep`` is "prox" (default, fast) or "em" (literal
    spec refill); see module docstring.  Returns g = c + tp*lam[franchise] in stint order.
    """
    cells = build_cells(stints, v, weights)
    n, N = cells.n_rows, len(cells.cols)
    lam = _init_lambda(cells, seed, lam0)
    V = _fill_rowmean(cells)
    obs = np.zeros((n, N), bool)
    obs[cells.ri, cells.ci] = True
    Wm = np.zeros((n, N))
    Wm[cells.ri, cells.ci] = cells.w
    sse_path, converged = [], False
    c, tp = _row_means(cells), np.zeros(n)
    sst = float(((cells.val - cells.val.mean()) ** 2).sum())
    for it in range(1, max_iter + 1):
        if fstep == "em":
            # F-step, eq. (21): F = V L' (L L')^-1 with L = [1; lam]
            LL = np.array([[N, lam.sum()], [lam.sum(), lam @ lam]])
            rhs = np.stack([V.sum(1), V @ lam], axis=1)
            F = np.linalg.solve(LL, rhs.T).T
            c, tp = F[:, 0], F[:, 1]
        else:
            c, tp = _prox_fstep(cells, lam, c, tp, delta)
        # lambda-step: weighted LS per franchise on observed cells (eq. 20 analogue)
        num = (Wm * tp[:, None] * (V - c[:, None])).sum(0)
        den = (Wm * tp[:, None] ** 2).sum(0)
        lam = np.where(den > 0, num / np.where(den > 0, den, 1.0), lam)
        m = lam.mean()
        lam = lam - m
        c = c + m * tp
        # refill only missing cells, then normalise
        if fstep == "em":
            V = np.where(obs, V, c[:, None] + tp[:, None] * lam[None, :])
        c, tp, lam = _normalise(c, tp, lam, scale)
        sse = float(((cells.val - (c[cells.ri] + tp[cells.ri] * lam[cells.ci])) ** 2).sum())
        sse_path.append(sse)
        if it > 1:
            prev = sse_path[-2]
            # relative SSE change (spec step 5); an exactly-fitted panel has SSE ~ 0, so
            # an absolute floor relative to the total sum of squares also counts.
            if abs(prev - sse) <= tol * max(prev, 1e-12 * sst):
                converged = True
                break
    c, tp, lam = _normalise(c, tp, lam, scale)  # final exact scale/sign (g unchanged)
    g = c[cells.stint_ri] + tp[cells.stint_ri] * lam[cells.stint_ci]
    fac = cells.keys.assign(c=c, tp=tp)
    return FactorResult(pd.Series(lam, index=cells.cols, name="lam"), fac, g, it, converged,
                        sse_path, explained_share(v, g), "als")


def multi_start_report(stints, v, weights=None, n_starts=20, seed=0, **kw) -> dict:
    """Identification diagnostic of spec section 5: refit from ``n_starts`` random lambda^0
    (seeds seed..seed+n_starts-1) and report Spearman rank correlations between starts of
    |lambda| (organisation-culture strength ranking) and of lambda itself.

    Returns dict with lam (DataFrame franch x start), spearman_abs / spearman_lam
    (n_starts x n_starts matrices), summary quantiles (min, q25, median, q75, max of the
    off-diagonal entries) and n_iter / explained_share per start."""
    fits = [fit_factor_als(stints, v, weights, seed=seed + k, **kw) for k in range(n_starts)]
    lam = pd.DataFrame({k: f.lam for k, f in enumerate(fits)})

    def corr(a):
        if n_starts < 2:
            return np.ones((n_starts, n_starts))
        r = spearmanr(a).statistic
        return np.atleast_2d(r) if n_starts > 2 else np.array([[1, r], [r, 1]])

    def summ(M):
        off = M[~np.eye(len(M), dtype=bool)]
        if off.size == 0:
            return {}
        q = np.quantile(off, [0, .25, .5, .75, 1])
        return dict(zip(["min", "q25", "median", "q75", "max"], map(float, q)))

    sa, sl = corr(lam.abs().to_numpy()), corr(lam.to_numpy())
    return dict(lam=lam, spearman_abs=sa, spearman_lam=sl, summary_abs=summ(sa),
                summary_lam=summ(sl), n_iter=[f.n_iter for f in fits],
                explained_share=[f.explained_share for f in fits])
