"""Spec §2: Batting order (b_j), fielding position (p_j), and play intensity (kappa) weights.

Paper references (Brave, Butters & Roberts 2019):
  eq. (5)  l_it = sum_{j=1..9} b_j S_ijt     (lineup weight, S = share of starts in slot j)
  eq. (6)  d_it = sum_j p_j g_ijt            (defensive weight, g = share of appearances by position)
  eq. (7)  W_hat = eta * tau * (W - alpha_hat), with kappa / tau / eta as below
  Table 2  b_j (Tango et al. 2007) and p_j (FanGraphs / B-Ref position adjustments)
All functions are pure functions on DataFrames / arrays.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from team_synergy.schema import FIELD_POSITIONS

LEAGUE_GAMES = 162
KAPPA_PA_DENOM = 3 * LEAGUE_GAMES      # 3 PA per game
KAPPA_OUT_DENOM = 27 * LEAGUE_GAMES    # 27 outs per game


def defensive_weights(weights_cfg: Mapping, war_source: str) -> dict:
    """Position weights p_j for eq. (6), keys = 8 field positions + DH (DH = 0).

    ``weight_source: table2`` returns paper Table 2 values from config (spec §2).
    ``weight_source: derived`` computes p_j = adj_j - adj_DH from the position
    adjustment runs (``position_adj_runs_bref`` overrides when non-null and
    ``war_source == 'bwar'``) and normalizes by ``weights_cfg['normalize']``:
    ``sum`` (default, sums to 1) or ``max`` (largest weight = 1).
    """
    source = weights_cfg.get("weight_source", "table2")
    if source == "table2":
        table = weights_cfg["defensive_weights"][war_source]
        return {k: float(v) for k, v in table.items()}
    if source != "derived":
        raise ValueError(f"unknown weight_source: {source!r}")
    adj = weights_cfg["position_adj_runs"]
    if war_source == "bwar" and weights_cfg.get("position_adj_runs_bref"):
        adj = weights_cfg["position_adj_runs_bref"]
    raw = {pos: float(adj[pos]) - float(adj["DH"]) for pos in FIELD_POSITIONS}
    norm = weights_cfg.get("normalize", "sum")
    if norm == "sum":
        denom = sum(raw.values())
    elif norm == "max":
        denom = max(raw.values())
    else:
        raise ValueError(f"unknown normalize: {norm!r}")
    out = {pos: v / denom for pos, v in raw.items()}
    out["DH"] = 0.0
    return out


def lineup_weights(weights_cfg: Mapping) -> np.ndarray:
    """Batting-order weights b_1..b_9 for eq. (5) (paper Table 2, Tango et al. 2007).

    Only ``weight_source: table2`` is implemented; a derived version (average PA
    per slot from Retrosheet) is not, because Table 2 values are available.
    """
    if weights_cfg.get("weight_source", "table2") != "table2":
        raise NotImplementedError(
            "Derived b_j (Retrosheet PA per slot) is not implemented; "
            "use weight_source: table2 (paper Table 2)."
        )
    b = np.asarray(weights_cfg["batting_order"], dtype=float)
    if b.shape != (9,):
        raise ValueError("batting_order must have 9 entries")
    return b


def lineup_weight_l(
    starts_long: pd.DataFrame,
    b: Sequence[float],
    id_cols: Sequence[str] = ("season", "team_retro", "retro_id"),
) -> pd.DataFrame:
    """Eq. (5): l = sum_j b_j S_j per (season, team, player) from long-format starts.

    Args:
        starts_long: columns ``id_cols`` + ``slot`` (1..9) + ``starts``.
        b: 9 slot weights.

    Returns:
        DataFrame ``id_cols`` + [l, lineup_slot_mean, n_starts]. ``lineup_slot_mean``
        is the starts-weighted mean slot. Players absent from this table (no
        starts) should receive ``mean(b)`` (spec §1); see ``fill_l``.
    """
    b = np.asarray(b, dtype=float)
    id_cols = list(id_cols)
    df = starts_long[id_cols + ["slot", "starts"]].copy()
    df = df[df["starts"] > 0]
    df["slot"] = df["slot"].astype(int)
    if ((df["slot"] < 1) | (df["slot"] > 9)).any():
        raise ValueError("slot must be in 1..9")
    df["bw"] = df["starts"] * b[df["slot"].to_numpy() - 1]
    df["sw"] = df["starts"] * df["slot"]
    g = df.groupby(id_cols, as_index=False)[["starts", "bw", "sw"]].sum()
    g["l"] = g["bw"] / g["starts"]
    g["lineup_slot_mean"] = g["sw"] / g["starts"]
    g = g.rename(columns={"starts": "n_starts"})
    return g[id_cols + ["l", "lineup_slot_mean", "n_starts"]]


def fill_l(l: pd.Series, b: Sequence[float]) -> pd.Series:
    """Spec §1: players with no starts get l = mean(b)."""
    return l.fillna(float(np.mean(b)))


def defensive_weight_d(appearances: pd.DataFrame, p: Mapping[str, float]) -> pd.Series:
    """Eq. (6): d = sum_j p_j g_j with g_j = games at position j / games over the 8 field positions.

    ``appearances`` has Lahman columns G_c, G_1b, ..., G_rf. Players with zero
    field games (pure DH / pinch hitters) get d = 0: DH has weight 0 in Table 2,
    so their contribution is lineup-only.
    """
    cols = [f"G_{pos.lower()}" for pos in FIELD_POSITIONS]
    games = appearances[cols].fillna(0).to_numpy(dtype=float)
    total = games.sum(axis=1)
    pw = np.array([p[pos] for pos in FIELD_POSITIONS], dtype=float)
    num = games @ pw
    d = np.divide(num, total, out=np.zeros_like(num), where=total > 0)
    return pd.Series(d, index=appearances.index, name="d")


def compute_kappa(df: pd.DataFrame) -> pd.Series:
    """Spec §2 / eq. (7) play intensity kappa.

    Position players (group == 'bat'): l*PA/(3*162) + d*DOuts/(27*162).
    Pitchers (group == 'pit'):         d*POuts/(27*162).
    Requires columns group, l, d, pa, douts, pouts.
    """
    bat = df["l"].fillna(0) * df["pa"] / KAPPA_PA_DENOM + df["d"] * df["douts"] / KAPPA_OUT_DENOM
    pit = df["d"] * df["pouts"] / KAPPA_OUT_DENOM
    return pd.Series(np.where(df["group"] == "pit", pit, bat), index=df.index, name="kappa")


def compute_tau_eta(
    df: pd.DataFrame,
    eta_cfg: Mapping[str, float],
    team_cols: Sequence[str] = ("season", "franch_id"),
) -> pd.DataFrame:
    """Eq. (7): tau = kappa / sum kappa over same team-season and same group; eta by group.

    Returns DataFrame [tau, eta, weight] aligned to ``df`` with weight = eta*tau,
    so that sum(weight) = eta_bat + eta_pit = 1 per team-season. A group whose
    kappa sums to zero gets tau = 0.
    """
    keys = list(team_cols) + ["group"]
    total = df.groupby(keys)["kappa"].transform("sum")
    tau = (df["kappa"] / total).where(total > 0, 0.0)
    eta = df["group"].map({"bat": float(eta_cfg["bat"]), "pit": float(eta_cfg["pit"])})
    return pd.DataFrame({"tau": tau, "eta": eta, "weight": eta * tau}, index=df.index)
