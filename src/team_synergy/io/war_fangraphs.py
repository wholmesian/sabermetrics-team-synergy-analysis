"""Spec §1: FanGraphs WAR reader (season-level batting and pitching WAR)."""

from pathlib import Path
import logging
import pandas as pd
from . import MissingRawDataError

logger = logging.getLogger(__name__)

# FanGraphs WAR file names
FWAR_BAT_FILE = "fg_batting.csv"
FWAR_PIT_FILE = "fg_pitching.csv"


def load_fwar(raw_dir: Path, seasons: tuple[int, int] = (1998, 2016)) -> dict[str, pd.DataFrame]:
    """Load FanGraphs WAR data for batters and pitchers (season-level).

    Reads fg_batting.csv and fg_pitching.csv (season-level aggregates).
    Filters to the specified season range.

    Params:
        raw_dir: Path to data/raw/ directory
        seasons: Tuple (start_year, end_year) inclusive

    Returns:
        Dict {"bat": DataFrame[...], "pit": DataFrame[...]} with columns:
            - idfg (int): FanGraphs player ID
            - season (int): Year
            - name (str): Player name
            - war (float): Season WAR value

    Raises:
        MissingRawDataError: If either file is missing.
    """
    fgraphs_dir = raw_dir / "fangraphs"

    # Check both files exist
    bat_path = fgraphs_dir / FWAR_BAT_FILE
    pit_path = fgraphs_dir / FWAR_PIT_FILE

    missing = []
    if not bat_path.exists():
        missing.append(FWAR_BAT_FILE)
    if not pit_path.exists():
        missing.append(FWAR_PIT_FILE)

    if missing:
        msg = (
            f"Missing FanGraphs WAR files in {fgraphs_dir}:\n"
            f"  Expected: {', '.join(missing)}\n"
            f"  Format: CSV with columns IDfg, Season, Name, WAR, Team (optional)\n"
            f"  Download: Export from https://www.fangraphs.com/leaders.aspx\n"
            f"    (Batting/Pitching leaderboard, all seasons)\n"
            f"  Or use: pybaseball.batting_stats(qual=0) / pitching_stats(qual=0)\n"
            f"  Save to data/raw/fangraphs/"
        )
        raise MissingRawDataError(msg)

    start_year, end_year = seasons

    # Load batting WAR
    try:
        bat_df = pd.read_csv(bat_path)
    except Exception as e:
        raise MissingRawDataError(f"Failed to read {bat_path}: {e}")

    # Load pitching WAR
    try:
        pit_df = pd.read_csv(pit_path)
    except Exception as e:
        raise MissingRawDataError(f"Failed to read {pit_path}: {e}")

    # Standardize column names (expected: IDfg, Season, Name, WAR, Team)
    # Normalize to lowercase with underscores
    bat_df.columns = bat_df.columns.str.lower().str.replace(" ", "_")
    pit_df.columns = pit_df.columns.str.lower().str.replace(" ", "_")

    # Rename to standardized names
    bat_df = bat_df.rename(columns={
        "idfg": "idfg",
        "season": "season",
        "name": "name",
        "war": "war",
    })

    pit_df = pit_df.rename(columns={
        "idfg": "idfg",
        "season": "season",
        "name": "name",
        "war": "war",
    })

    # Check required columns
    for df, file_name in [(bat_df, FWAR_BAT_FILE), (pit_df, FWAR_PIT_FILE)]:
        required = ["idfg", "season", "name", "war"]
        missing_cols = [c for c in required if c not in df.columns]
        if missing_cols:
            raise MissingRawDataError(
                f"{file_name} missing expected columns: {missing_cols}\n"
                f"Expected columns: IDfg, Season, Name, WAR"
            )

    # Filter to season range
    bat_df = bat_df[(bat_df["season"] >= start_year) & (bat_df["season"] <= end_year)].copy()
    pit_df = pit_df[(pit_df["season"] >= start_year) & (pit_df["season"] <= end_year)].copy()

    # Ensure data types and convert WAR to numeric
    for df in [bat_df, pit_df]:
        df["idfg"] = pd.to_numeric(df["idfg"], errors="coerce").astype("Int64")
        df["season"] = df["season"].astype(int)
        df["war"] = pd.to_numeric(df["war"], errors="coerce")

    # Select and order columns
    cols = ["idfg", "season", "name", "war"]
    bat_df = bat_df[cols].reset_index(drop=True)
    pit_df = pit_df[cols].reset_index(drop=True)

    logger.info(f"Loaded fWAR: {len(bat_df)} batting, {len(pit_df)} pitching records")
    return {"bat": bat_df, "pit": pit_df}


def fetch_fwar(raw_dir: Path, seasons: tuple[int, int] = (1998, 2016)) -> None:
    """Download and save FanGraphs WAR data using pybaseball.

    Uses pybaseball.batting_stats() and pitching_stats() to fetch season-level data.
    Saves to data/raw/fangraphs/fg_batting.csv and fg_pitching.csv.

    Params:
        raw_dir: Path to data/raw/ directory
        seasons: Tuple (start_year, end_year) inclusive (passed to pybaseball)

    Raises:
        MissingRawDataError: If download fails or pybaseball is not available.
    """
    try:
        import pybaseball
    except ImportError:
        raise MissingRawDataError(
            "pybaseball not installed. Install with: pip install pybaseball\n"
            "Or manually export from:\n"
            "  https://www.fangraphs.com/leaders.aspx\n"
            f"  Save to data/raw/fangraphs/{FWAR_BAT_FILE} and {FWAR_PIT_FILE}"
        )

    fgraphs_dir = raw_dir / "fangraphs"
    fgraphs_dir.mkdir(parents=True, exist_ok=True)

    try:
        start_year, end_year = seasons

        logger.info(f"Fetching FanGraphs WAR batting data ({start_year}–{end_year})...")
        bat_df = pybaseball.batting_stats(start_season=start_year, end_season=end_year, qual=0)
        bat_df.to_csv(fgraphs_dir / FWAR_BAT_FILE, index=False)

        logger.info(f"Fetching FanGraphs WAR pitching data ({start_year}–{end_year})...")
        pit_df = pybaseball.pitching_stats(start_season=start_year, end_season=end_year, qual=0)
        pit_df.to_csv(fgraphs_dir / FWAR_PIT_FILE, index=False)

        logger.info(f"Saved fWAR data to {fgraphs_dir}/")
    except Exception as e:
        raise MissingRawDataError(
            f"Failed to fetch fWAR data: {e}\n"
            "Manually export from https://www.fangraphs.com/leaders.aspx\n"
            f"and save to {fgraphs_dir}/"
        )
