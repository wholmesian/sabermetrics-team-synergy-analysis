"""Spec §1–2: Stint-level panel construction (stint = player, season, team).

Implements the data construction of Brave, Butters & Roberts (2019): eqs. (5)-(7)
weights, Appendix 7.1 (fWAR split across stints, position/age/experience/manager
covariates) and footnote 13 (drop pitchers' batting WAR and position players'
pitching WAR).
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from team_synergy.build.weights import (
    compute_kappa,
    compute_tau_eta,
    defensive_weight_d,
    defensive_weights,
    fill_l,
    lineup_weight_l,
    lineup_weights,
)
from team_synergy.schema import FIELD_POSITIONS, FRANCHISE_MAP, PANEL_COLUMNS

KEY = ["player_id", "season", "teamID"]
BAT_SUM = ["AB", "BB", "HBP", "SH", "SF"]
APP_COLS = ["G_all", "G_p", "G_dh"] + [f"G_{p.lower()}" for p in FIELD_POSITIONS]


def _num_id(s: pd.Series) -> pd.Series:
    """Normalize numeric IDs (e.g. idfg) so int/float/str forms match."""
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _team_table(teams: pd.DataFrame) -> pd.DataFrame:
    t = teams.copy()
    if "franch_id" not in t.columns:
        t["franch_id"] = t["franchID"].replace(FRANCHISE_MAP)
    keep = ["yearID", "teamID", "franch_id", "lgID", "W", "teamIDBR", "teamIDretro"]
    t = t[[c for c in keep if c in t.columns]].drop_duplicates(["yearID", "teamID"])
    return t.rename(columns={"yearID": "season"})


def _stint_base(lahman: Mapping[str, pd.DataFrame], teams: pd.DataFrame) -> pd.DataFrame:
    """One row per (player_id, season, teamID) with appearance/PA/outs aggregates."""
    app = lahman["Appearances"].rename(columns={"yearID": "season", "playerID": "player_id"})
    base = app.groupby(KEY[:2] + ["teamID"], as_index=False)[APP_COLS].sum()

    bat = lahman["Batting"].rename(columns={"yearID": "season", "playerID": "player_id"})
    bat = bat.groupby(KEY, as_index=False)[BAT_SUM].sum()
    bat["pa"] = bat[BAT_SUM].sum(axis=1)

    fld = lahman["Fielding"].rename(columns={"yearID": "season", "playerID": "player_id"})
    # Lahman `Fielding` reports outfielders only as the aggregate POS "OF"
    # (LF/CF/RF splits live in FieldingOFsplit), so DOuts = C/1B/2B/3B/SS + OF.
    # Excludes P (pitchers use POuts) and DH.
    fld = fld[fld["POS"].isin(["C", "1B", "2B", "3B", "SS", "OF"])]
    fld = fld.groupby(KEY, as_index=False)["InnOuts"].sum().rename(columns={"InnOuts": "douts"})

    pit = lahman["Pitching"].rename(columns={"yearID": "season", "playerID": "player_id"})
    pit = pit.groupby(KEY, as_index=False)[["IPouts", "G", "GS"]].sum()
    pit = pit.rename(columns={"IPouts": "pouts", "G": "pit_G", "GS": "pit_GS"})

    for part in (bat[KEY + ["pa"]], fld, pit):
        base = base.merge(part, on=KEY, how="left")
    for c in ["pa", "douts", "pouts", "pit_G", "pit_GS"]:
        base[c] = base[c].fillna(0)
    base = base.merge(teams, on=["season", "teamID"], how="left")
    return base


def _primary_pos(base: pd.DataFrame) -> pd.Series:
    """Appendix 7.1 / spec §1 primary position (pitchers -> role, handled by caller)."""
    g = base[[f"G_{p.lower()}" for p in FIELD_POSITIONS]].to_numpy(dtype=float)
    total = g.sum(axis=1)
    mx = g.max(axis=1)
    arg = np.array(FIELD_POSITIONS, dtype=object)[g.argmax(axis=1)]
    share = np.divide(mx, total, out=np.zeros_like(mx), where=total > 0)
    pos = np.where(share < 0.5, "UT", arg)  # includes total == 0 (e.g. pinch hitters)
    pos = np.where(base["G_dh"].to_numpy() > mx, "DH", pos)
    return pd.Series(pos, index=base.index, dtype=object)


def _attach_war(
    df: pd.DataFrame, war: Mapping[str, pd.DataFrame], idmap: pd.DataFrame,
    teams: pd.DataFrame, cfg: Mapping, war_source: str,
) -> pd.DataFrame:
    """Attach stint WAR. bWAR: per stint as-is. fWAR: season total split across stints."""
    idm = idmap.drop_duplicates("player_id")
    df = df.copy()
    if war_source == "bwar":
        df = df.merge(idm[["player_id", "bbref_id"]], on="player_id", how="left")
        df["_has_id"] = df["bbref_id"].notna()
        parts = []
        for grp in ("bat", "pit"):
            w = war[grp].rename(columns={"team_br": "teamIDBR", "war": "_w"})
            w = w.groupby(["bbref_id", "season", "teamIDBR"], as_index=False)["_w"].sum()
            w["group"] = grp
            parts.append(w)
        w = pd.concat(parts, ignore_index=True)
        # Footnote 13: the merge on `group` keeps batting WAR for position players only
        # and pitching WAR for pitchers only.
        df = df.merge(w, on=["bbref_id", "season", "teamIDBR", "group"], how="left")
        df["war"] = df["_w"]
    else:
        df = df.merge(idm[["player_id", "idfg"]], on="player_id", how="left")
        df["idfg"] = _num_id(df["idfg"])
        df["_has_id"] = df["idfg"].notna()
        parts = []
        for grp in ("bat", "pit"):
            w = war[grp].copy()
            w["idfg"] = _num_id(w["idfg"])
            w = w.groupby(["idfg", "season"], as_index=False)["war"].sum().rename(columns={"war": "_w"})
            w["group"] = grp
            parts.append(w)
        w = pd.concat(parts, ignore_index=True)
        df = df.merge(w, on=["idfg", "season", "group"], how="left")
        # Appendix 7.1: split the season total across stints proportionally to kappa
        # (or games). Simplification: if the proportional base is 0 for all stints,
        # fall back to equal split.
        mode = cfg.get("fwar_allocation", "kappa")
        base_w = df["kappa"] if mode == "kappa" else df["G_all"].astype(float)
        gk = [df["player_id"], df["season"], df["group"]]
        tot = base_w.groupby(gk).transform("sum")
        n = base_w.groupby(gk).transform("size")
        share = np.where(tot > 0, base_w / tot.where(tot > 0, 1), 1.0 / n)
        df["war"] = df["_w"] * share
    df["war_missing"] = df["war"].isna()
    df["war"] = df["war"].fillna(0.0)
    return df.drop(columns=["_w"])


def build_panel(
    lahman: Mapping[str, pd.DataFrame],
    lineup_starts: pd.DataFrame,
    war: Mapping[str, pd.DataFrame],
    idmap: pd.DataFrame,
    cfg: Mapping,
    weights_cfg: Mapping,
    war_source: str,
) -> pd.DataFrame:
    """Build the stint panel (spec §1-2) with schema.PANEL_COLUMNS.

    One row per (player_id, season, franch_id); Lahman stints on the same team
    are aggregated. Experience counters use every Appearances season supplied in
    ``lahman`` (left-censored at the first available year: if Lahman is loaded
    from 1998 on, 1998 experience is 0 for everyone). Seasons are then restricted
    to ``cfg['seasons']``. ``panel.attrs`` carries ``id_map_fail_rate`` and
    ``war_missing_rate`` for ``check_panel``.
    """
    if war_source not in ("fwar", "bwar"):
        raise ValueError("war_source must be 'fwar' or 'bwar'")
    s0, s1 = cfg["seasons"]["start"], cfg["seasons"]["end"]
    teams = _team_table(lahman["Teams"])
    df = _stint_base(lahman, teams)

    # --- experience (before restricting seasons, to use all history available) ---
    df = df.sort_values(["player_id", "season"]).reset_index(drop=True)
    pg = df.groupby(["player_id", "season"])["G_all"].transform("sum")
    first = ~df.duplicated(["player_id", "season"])
    yr = df.loc[first, ["player_id", "season"]].assign(g=pg[first])
    yr["mlb_exp"] = yr.groupby("player_id")["g"].cumsum() - yr["g"]
    df = df.merge(yr[["player_id", "season", "mlb_exp"]], on=["player_id", "season"], how="left")
    df = df.sort_values(["player_id", "franch_id", "season"])
    df["team_exp"] = df.groupby(["player_id", "franch_id"])["G_all"].cumsum() - df["G_all"]
    # (a player has one stint per franchise-season, so cumsum - own G = prior seasons)

    df = df[(df["season"] >= s0) & (df["season"] <= s1)].copy()

    # --- group / role / position ---
    df["group"] = np.where((df["G_all"] > 0) & (df["G_p"] >= 0.5 * df["G_all"]), "pit", "bat")
    gs_share = np.divide(df["pit_GS"], df["pit_G"], out=np.zeros(len(df)), where=df["pit_G"] > 0)
    df["role"] = np.where(df["group"] == "pit", np.where(gs_share >= 0.5, "SP", "RP"), None)
    df["pos"] = np.where(df["group"] == "pit", df["role"], _primary_pos(df))

    # Players with no PA and no outs are excluded (Appendix 7.1).
    df = df[(df["pa"] > 0) | (df["douts"] > 0) | (df["pouts"] > 0)].copy()

    # --- weights: l, d, kappa, tau, eta ---
    b = lineup_weights(weights_cfg)
    p = defensive_weights(weights_cfg, war_source)
    people = lahman["People"][["playerID", "birthYear", "retroID"]].rename(columns={"playerID": "player_id"})
    df = df.merge(people, on="player_id", how="left")
    starts = lineup_weight_l(lineup_starts, b)
    df = df.merge(
        starts[["season", "team_retro", "retro_id", "l", "lineup_slot_mean"]],
        left_on=["season", "teamIDretro", "retroID"], right_on=["season", "team_retro", "retro_id"], how="left",
    ).drop(columns=["team_retro", "retro_id"])
    df["l"] = fill_l(df["l"], b)
    df["d"] = defensive_weight_d(df, p)
    # Simplification: pitchers' l is unused by kappa (paper: d = 1 for pitchers) -> NaN.
    df.loc[df["group"] == "pit", "d"] = float(weights_cfg.get("pitcher_d", 1.0))
    df.loc[df["group"] == "pit", ["l", "lineup_slot_mean"]] = np.nan
    df["kappa"] = compute_kappa(df)
    df = pd.concat([df, compute_tau_eta(df, cfg["eta"][war_source])], axis=1)

    # --- WAR ---
    df = _attach_war(df, war, idmap, teams, cfg, war_source)

    # --- covariates ---
    df["age"] = df["season"] - df["birthYear"]
    mg = lahman["Managers"]
    mg = mg[mg["inseason"] == 1].sort_values(["yearID", "teamID", "playerID"])
    mg = mg.drop_duplicates(["yearID", "teamID"]).rename(
        columns={"yearID": "season", "playerID": "manager_id"})[["season", "teamID", "manager_id"]]
    df = df.merge(mg, on=["season", "teamID"], how="left")
    df = df.rename(columns={"W": "team_wins", "lgID": "lg"})

    attrs = {
        "id_map_fail_rate": float((~df["_has_id"]).mean()) if len(df) else 0.0,
        "war_missing_rate": float(df["war_missing"].mean()) if len(df) else 0.0,
    }
    out = df[PANEL_COLUMNS].sort_values(["season", "franch_id", "group", "player_id"]).reset_index(drop=True)
    out.attrs.update(attrs)
    return out


def check_panel(
    panel: pd.DataFrame, cfg: Mapping, strict: bool = True, n_teams: int = 30,
    tol_war: float = 0.05, tol_weight: float = 1e-9, max_id_fail: float = 0.01,
) -> dict:
    """Spec §1 integrity checks. Returns a report dict; raises AssertionError if strict and any fails.

    Checks: (1) team-seasons == n_seasons * n_teams (570 for 1998-2016, 30 teams);
    (2) per-season sum of WAR within +-5% of 1000 * n_teams/30; (3) ID-map failure
    rate (from ``panel.attrs``, if present) < 1%; (4) sum(weight) = 1 per team-season.
    """
    s0, s1 = cfg["seasons"]["start"], cfg["seasons"]["end"]
    n_seasons = s1 - s0 + 1
    n_ts = int(panel[["season", "franch_id"]].drop_duplicates().shape[0])
    target = 1000.0 * n_teams / 30.0
    war_sum = panel.groupby("season")["war"].sum()
    wdev = ((war_sum - target).abs() / target)
    wsum = panel.groupby(["season", "franch_id"])["weight"].sum()
    wsum_dev = float((wsum - 1.0).abs().max()) if len(wsum) else float("nan")
    fail = panel.attrs.get("id_map_fail_rate")
    report = {
        "n_team_seasons": n_ts,
        "expected_team_seasons": n_seasons * n_teams,
        "war_sum_by_season": war_sum.to_dict(),
        "war_sum_max_rel_dev": float(wdev.max()) if len(wdev) else float("nan"),
        "id_map_fail_rate": fail,
        "weight_sum_max_dev": wsum_dev,
    }
    problems = []
    if n_ts != n_seasons * n_teams:
        problems.append(f"team-seasons {n_ts} != expected {n_seasons * n_teams}")
    if len(wdev) and (wdev > tol_war).any():
        bad = {int(k): round(float(war_sum[k]), 1) for k in wdev[wdev > tol_war].index}
        problems.append(f"league WAR sum outside +-{tol_war:.0%} of {target:.0f}: {bad}")
    if fail is not None and fail >= max_id_fail:
        problems.append(f"ID-map failure rate {fail:.3%} >= {max_id_fail:.0%}")
    if not (wsum_dev <= tol_weight):
        problems.append(f"sum(weight) != 1 for some team-season (max dev {wsum_dev:.3g})")
    report["problems"] = problems
    report["ok"] = not problems
    if strict and problems:
        raise AssertionError("panel integrity check failed: " + "; ".join(problems))
    return report


def save_panel(panel: pd.DataFrame, path) -> Path:
    """Write the panel to parquet (creating parent dirs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(path, index=False)
    return path


def build_side_tables(lahman: Mapping[str, pd.DataFrame], panel: Optional[pd.DataFrame] = None,
                      pecota: Optional[pd.DataFrame] = None) -> dict:
    """Side tables for spec section 8 analyses (Lahman People/Salaries, optional PECOTA).

    - ``players``: player_id, name (nameFirst + " " + nameLast), birth_year, debut_season
      (debut_season from the Lahman ``debut`` date; NaN if missing);
    - ``salaries``: player_id, season, salary (Lahman Salaries, summed over stints of a
      player-season; Lahman salaries start in 1985);
    - ``pecota``: franch_id, season, pecota_w, only if ``pecota`` is given.
    ``panel`` is unused (reserved for restricting to panel players).
    """
    pe = lahman["People"].rename(columns={"playerID": "player_id", "birthYear": "birth_year"})
    name = (pe["nameFirst"].fillna("") + " " + pe["nameLast"].fillna("")).str.strip()
    deb = pd.to_datetime(pe["debut"], errors="coerce").dt.year if "debut" in pe else np.nan
    players = pd.DataFrame({"player_id": pe["player_id"], "name": name,
                            "birth_year": pe["birth_year"], "debut_season": deb})
    sal = lahman["Salaries"].rename(columns={"playerID": "player_id", "yearID": "season"})
    salaries = sal.groupby(["player_id", "season"], as_index=False)["salary"].sum()
    tables = {"players": players, "salaries": salaries}
    if pecota is not None:
        tables["pecota"] = pecota
    return tables


def save_side_tables(tables: Mapping[str, pd.DataFrame], out_dir) -> list[Path]:
    """Write players.parquet, salaries.parquet, pecota.parquet (those present) to ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in ("players", "salaries", "pecota"):
        if tables.get(name) is not None:
            p = out / f"{name}.parquet"
            tables[name].to_parquet(p, index=False)
            paths.append(p)
    return paths


def build_from_raw(cfg: Mapping, war_source: Optional[str] = None, weights_cfg: Optional[Mapping] = None,
                   return_side_tables: bool = False):
    """Load raw data via the io loaders and build the panel (not unit-tested).

    Raw files are read from ``cfg['paths']['raw']``. The ID map is built from
    Lahman People and the Chadwick register (``io.idmap``). Returns the panel, or
    ``(panel, side_tables)`` if ``return_side_tables`` (see ``build_side_tables``;
    PECOTA is included when ``data/raw/pecota/pecota.csv`` exists).
    """
    from team_synergy.config import _find_repo_root, load_weights
    from team_synergy.io import idmap as io_idmap
    from team_synergy.io import lahman as io_lahman
    from team_synergy.io import pecota as io_pecota
    from team_synergy.io import retrosheet as io_retro
    from team_synergy.io import war_bref, war_fangraphs

    war_source = war_source or cfg.get("war_source", "fwar")
    weights_cfg = weights_cfg or load_weights()
    raw = _find_repo_root() / cfg["paths"]["raw"]
    seasons = (cfg["seasons"]["start"], cfg["seasons"]["end"])
    lahman = io_lahman.load_lahman(raw)
    starts = io_retro.lineup_starts(io_retro.load_gamelogs(raw, seasons))
    if war_source == "fwar":
        war = war_fangraphs.load_fwar(raw, seasons)
    else:
        war = war_bref.load_bwar(raw, seasons)
    idmap = io_idmap.build_idmap(lahman["People"], io_idmap.load_register(raw))
    panel = build_panel(lahman, starts, war, idmap, cfg, weights_cfg, war_source)
    if not return_side_tables:
        return panel
    return panel, build_side_tables(lahman, panel, io_pecota.load_pecota(raw))
