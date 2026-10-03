"""Teammate network matrix A, paper eq. (8) and Appendix 7.2; spec section 4."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Block:
    """One team-season block of the block-diagonal network matrix."""

    franch_id: str
    season: int
    idx: np.ndarray      # positional row indices into the stints frame
    alpha: np.ndarray    # raw symmetric weights alpha_ij (zero diagonal)
    A: np.ndarray        # row-normalized D^-1 alpha
    eigvals: np.ndarray  # real eigenvalues of A, sorted ascending


def build_blocks(
    stints: pd.DataFrame, history: str = "past", kernel_h: float | None = None
) -> list[Block]:
    """Build team-season blocks of A (eq. 8, spec section 4).

    alpha_ij(n,t) = sum over seasons s (s <= t if history="past", all if "full")
    in which players i and j shared a team-season (on any team) of
    (kappa_is + kappa_js), using kappa of their stints on that shared team-season.
    Diagonal is 0. If ``kernel_h`` is given, alpha_ij is multiplied by
    exp(-|slot_i - slot_j| / h) using ``lineup_slot_mean`` (spec section 4 extension,
    not in the paper; NaN slot -> no multiplication).

    Row normalization A = D^-1 alpha. Rows with zero sum (a player with no teammate
    history in the block, only possible with a kernel or a one-man block) stay zero
    rather than being divided by 0; such rows contribute eigenvalue 0.
    Eigenvalues come from the similar symmetric matrix D^-1/2 alpha D^-1/2 via eigvalsh,
    restricted to positive-degree nodes (zeros appended for the rest).
    """
    if history not in ("past", "full"):
        raise ValueError(f"unknown history {history!r}")
    df = stints.reset_index(drop=True)
    n = len(df)
    kappa = df["kappa"].to_numpy(float)
    season = df["season"].to_numpy()
    pid = pd.factorize(df["player_id"])[0]
    ts = df.groupby(["franch_id", "season"], sort=True).ngroup().to_numpy()
    slot = df["lineup_slot_mean"].to_numpy(float) if kernel_h is not None else None

    order = np.argsort(pid, kind="stable")
    bounds = np.flatnonzero(np.diff(pid[order])) + 1
    player_rows = np.split(order, bounds)
    rows_of = {pid[g[0]]: g for g in player_rows}

    block_rows = np.split(np.argsort(ts, kind="stable"), np.flatnonzero(np.diff(np.sort(ts))) + 1)
    blocks: list[Block] = []
    for idx in block_rows:
        idx = np.sort(idx)
        r0 = idx[0]
        t = season[r0]
        m = len(idx)
        members: dict[int, tuple[list, list]] = {}
        for k, r in enumerate(idx):
            for r2 in rows_of[pid[r]]:
                if history == "past" and season[r2] > t:
                    continue
                ks, kp = members.setdefault(ts[r2], ([], []))
                ks.append(k)
                kp.append(kappa[r2])
        alpha = np.zeros((m, m))
        for ks, kp in members.values():
            if len(ks) < 2:
                continue
            ks = np.asarray(ks)
            kp = np.asarray(kp)
            alpha[np.ix_(ks, ks)] += kp[:, None] + kp[None, :]
        if kernel_h is not None:
            s = slot[idx]
            d = np.abs(s[:, None] - s[None, :])
            alpha *= np.where(np.isnan(d), 1.0, np.exp(-np.nan_to_num(d) / kernel_h))
        np.fill_diagonal(alpha, 0.0)
        deg = alpha.sum(axis=1)
        pos = deg > 0
        A = np.zeros_like(alpha)
        A[pos] = alpha[pos] / deg[pos, None]
        eig = np.zeros(m)
        if pos.any():
            sub = alpha[np.ix_(pos, pos)]
            sq = 1.0 / np.sqrt(deg[pos])
            ev = np.linalg.eigvalsh(sq[:, None] * sub * sq[None, :])
            eig = np.concatenate([ev, np.zeros(m - len(ev))])
        blocks.append(Block(df["franch_id"].iat[r0], int(t), idx, alpha, A, np.sort(eig)))
    return blocks
