"""Inputs shared by all spec section 8 analyses.

Bundles the estimation outputs (data/processed/<src>/), the panel covariates and the
side tables (players, salaries, PECOTA) in one object, for real or synthetic data.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from team_synergy.config import load_config

_KEYS = ["player_id", "season", "franch_id"]
_SUM_COLS = ["war", "own", "in_deg", "out_deg", "war_minus", "war_plus", "pc_war",
             "pc_war_char", "pc_war_team", "weight"]
_MAIN_COLS = ["franch_id", "manager_id", "lg", "pos", "role", "group"]


@dataclass
class AnalysisInputs:
    """Everything the section 8 analyses read.

    stints_full / stints_recursive: stint-level metrics (metrics_full / metrics_recursive;
    recursive has ``window_end``); teams_full / teams_recursive: team-season metrics
    (team_full / team_recursive); player_seasons: see ``player_season_table``;
    lambda_windows, window_summary: recursive outputs; players, salaries, pecota: side
    tables (None if absent); cfg: config dict; war_source: "fwar" | "bwar" | "synthetic".
    """

    stints_full: pd.DataFrame
    stints_recursive: pd.DataFrame
    teams_full: pd.DataFrame
    teams_recursive: pd.DataFrame
    player_seasons: pd.DataFrame
    lambda_windows: pd.DataFrame
    window_summary: pd.DataFrame
    players: pd.DataFrame
    salaries: pd.DataFrame | None
    pecota: pd.DataFrame | None
    cfg: dict
    war_source: str


def player_season_table(stints_full: pd.DataFrame, panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (player_id, season) for the player-level analyses (spec section 8, eq. 16/17).

    Metrics ``war, own, in_deg, out_deg, war_minus, war_plus, pc_war, pc_war_char,
    pc_war_team, weight`` are summed over stints (spec 8: traded players = stint sum).
    ``franch_id, manager_id, lg, pos, role, group`` and ``team_exp`` come from the stint
    with the largest ``weight`` (ties: alphabetical franch_id); ``mlb_exp`` is the minimum
    over stints (= pre-season experience) and ``age`` the first value. Covariates missing
    from ``stints_full`` are taken from ``panel`` (merged on player_id, season, franch_id);
    covariates absent from both are omitted. Extra column ``n_stints``.
    """
    st = stints_full.copy()
    if panel is not None:
        extra = [c for c in ["group", "pos", "role", "age", "mlb_exp", "team_exp", "manager_id", "lg"]
                 if c in panel.columns and c not in st.columns]
        if extra:
            st = st.merge(panel[_KEYS + extra].drop_duplicates(_KEYS), on=_KEYS, how="left")
    k = ["player_id", "season"]
    sums = st.groupby(k)[[c for c in _SUM_COLS if c in st.columns]].sum()
    main = st.sort_values(k + ["weight", "franch_id"], ascending=[True, True, False, True])
    main = main.drop_duplicates(k).set_index(k)
    cols = [c for c in _MAIN_COLS + ["team_exp"] if c in main.columns]
    out = main[cols].copy()
    out = out.join(sums)
    g = st.groupby(k)
    if "mlb_exp" in st:
        out["mlb_exp"] = g["mlb_exp"].min()
    if "age" in st:
        out["age"] = g["age"].first()
    out["n_stints"] = g.size()
    return out.reset_index()


def _read_opt(dirs, name):
    for d in dirs:
        p = Path(d) / f"{name}.parquet"
        if p.exists():
            return pd.read_parquet(p)
    return None


def load_inputs(processed_dir, war_source: str, panel: pd.DataFrame | None = None,
                cfg: dict | None = None) -> AnalysisInputs:
    """Load estimation outputs from ``processed_dir`` (= data/processed/<war_source>/).

    The panel defaults to ``panel_<war_source>.parquet`` in ``processed_dir`` or its
    parent; side tables (players, salaries, pecota .parquet, written by ``synergy build``)
    are searched in ``processed_dir`` then its parent. Missing salaries/pecota -> None.
    """
    pdir = Path(processed_dir)
    dirs = [pdir, pdir.parent]
    if panel is None:
        panel = _read_opt(dirs, f"panel_{war_source}")
    rd = lambda n: pd.read_parquet(pdir / f"{n}.parquet")
    sf = rd("metrics_full")
    players = _read_opt(dirs, "players")
    if players is None:
        raise FileNotFoundError(f"players.parquet not found in {dirs}; run `synergy build`")
    return AnalysisInputs(
        stints_full=sf, stints_recursive=rd("metrics_recursive"), teams_full=rd("team_full"),
        teams_recursive=rd("team_recursive"), player_seasons=player_season_table(sf, panel),
        lambda_windows=rd("lambda_windows"), window_summary=rd("window_summary"),
        players=players, salaries=_read_opt(dirs, "salaries"), pecota=_read_opt(dirs, "pecota"),
        cfg=cfg if cfg is not None else load_config(), war_source=war_source)


def synthetic_inputs(seed: int = 0, n_seasons: int = 19, full_only: bool = False, cfg: dict | None = None,
                     first_season: int = 1998, **kw) -> AnalysisInputs:
    """Synthetic AnalysisInputs: panel + extras (spec 9 step 1) and ``run_recursive``.

    ``kw`` goes to ``make_synthetic_panel`` (e.g. n_teams=12, roster=30 for fast tests).
    ``full_only`` estimates only the full-sample window (recursive tables then hold one
    window). Seasons are relabelled to first_season .. first_season + n_seasons - 1
    (default 1998, matching the paper sample, the CLI and config/cpi.yaml).
    """
    from team_synergy.model.recursive import run_recursive
    from team_synergy.synthetic import make_synthetic_extras, make_synthetic_panel

    panel, truth = make_synthetic_panel(n_seasons=n_seasons, seed=seed, **kw)
    ex = make_synthetic_extras(panel, truth, seed)
    # Shift all calendar fields together so ages, debuts, salaries and PECOTA stay aligned.
    off = first_season - int(panel["season"].min())
    if off:
        panel = panel.assign(season=panel["season"] + off)
        ex = dict(ex)
        ex["players"] = ex["players"].assign(birth_year=ex["players"]["birth_year"] + off,
                                             debut_season=ex["players"]["debut_season"] + off)
        for k in ("salaries", "pecota"):
            if ex.get(k) is not None:
                ex[k] = ex[k].assign(season=ex[k]["season"] + off)
    cfg = dict(cfg if cfg is not None else load_config())
    cfg["seed"] = seed
    last = int(panel["season"].max())
    res = run_recursive(panel, cfg, min_end=last if full_only else None)
    return AnalysisInputs(
        stints_full=res["metrics_full"], stints_recursive=res["metrics_recursive"],
        teams_full=res["team_full"], teams_recursive=res["team_recursive"],
        player_seasons=player_season_table(res["metrics_full"], panel),
        lambda_windows=res["lambda_windows"], window_summary=res["window_summary"],
        players=ex["players"], salaries=ex["salaries"], pecota=ex["pecota"], cfg=cfg,
        war_source="synthetic")
