"""Shared session-scoped synthetic AnalysisInputs for the spec section 8 analysis tests.

Building the inputs runs the full recursive estimation, so it is done once per session.
Factor tolerances are loosened here only to keep the suite under a minute.
"""
import pytest

from team_synergy.analysis.inputs import synthetic_inputs
from team_synergy.config import load_config

SYN_SEASONS, SYN_TEAMS, SYN_ROSTER = 12, 10, 22


@pytest.fixture(scope="session")
def syn_inputs():
    cfg = load_config(overrides={"factor": {"tol": 1e-7, "prob_tol": 1e-5}})
    return synthetic_inputs(seed=0, n_seasons=SYN_SEASONS, n_teams=SYN_TEAMS, roster=SYN_ROSTER, cfg=cfg)
