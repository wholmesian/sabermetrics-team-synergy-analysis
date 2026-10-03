"""Single-player complementarity view: Fig. 12, the "David Ross effect" (paper section 5.2).

Season-by-season stint rows of pcWAR and intangibles xi against WAR for one player; a traded
player has several rows in a season (label "season franch"). Stint WAR and pcWAR are the
stint-level model outputs; xi (Table 4 spec 2, a player-season residual) is split across the
season's stints in proportion to the stint weights (deviation: the paper shows one value
per stint row without saying how xi is attributed). Seasons without a previous season have
no xi (NaN).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from team_synergy.analysis.inputs import AnalysisInputs
from team_synergy.analysis.pc_persistence import player_names, table4_residuals
from team_synergy.analysis.result import AnalysisResult


def default_player(inputs: AnalysisInputs) -> str:
    """Lahman id "rossda01" if present, else (synthetic) the player with the most seasons among
    those with median season WAR < 1 (ties: smallest id); falls back to the most seasons overall."""
    ps = inputs.player_seasons
    if (ps["player_id"] == "rossda01").any():
        return "rossda01"
    g = ps.groupby("player_id")["war"].agg(n="size", med="median")
    cand = g[g["med"] < 1]
    cand = cand if len(cand) else g
    return str(cand.sort_values("n", ascending=False, kind="stable").index[0])


def run(inputs: AnalysisInputs, n_boot: int = 500, seed: int = 0, player_id: str | None = None,
        **opts) -> AnalysisResult:
    """Fig. 12. Tables: ``fig12`` (player_id, season, franch_id, label, war, pc_war, xi), ``player``
    (the chosen player_id and name). ``n_boot`` / ``seed`` unused (no inference)."""
    pid = player_id or default_player(inputs)
    st = inputs.stints_full
    s = st[st["player_id"] == pid][["player_id", "season", "franch_id", "war", "pc_war", "weight"]].copy()
    if s.empty:
        raise ValueError(f"player {pid!r} not in stints_full")
    xi = table4_residuals(inputs)[["player_id", "season", "xi"]]
    s = s.merge(xi, on=["player_id", "season"], how="left")
    share = s["weight"] / s.groupby("season")["weight"].transform("sum")
    s["xi"] = s["xi"] * share
    s["label"] = s["season"].astype(str) + " " + s["franch_id"].astype(str)
    s = s.sort_values(["season", "franch_id"]).drop(columns="weight").reset_index(drop=True)
    nm = player_names(inputs.players).get(pid, pid)
    seas = s.groupby("season")["pc_war"].sum()
    share_pos = float((seas > 0).mean())
    real = inputs.war_source != "synthetic"
    checks = [{"item": "Fig. 12: share of seasons with pcWAR > 0", "value": share_pos,
               "expected": "David Ross: most seasons have positive pcWAR",
               "passed": bool(share_pos > 0.5) if real and pid == "rossda01" else None}]
    return AnalysisResult("ross", {"fig12": s, "player": pd.DataFrame({"player_id": [pid], "name": [nm]})}, checks)
