"""Spec §1: Baseball-Reference WAR reader (war_daily_bat, war_daily_pitch)."""

from pathlib import Path
import logging
import pandas as pd
from . import MissingRawDataError

logger = logging.getLogger(__name__)

# Baseball-Reference WAR file names
BWAR_BAT_FILE = "war_daily_bat.txt"
BWAR_PIT_FILE = "war_daily_pitch.txt"


def _standardize_bwar_columns(df: pd.DataFrame, file_name: str) -> pd.DataFrame:
    """Standardize bWAR column names to lowercase with underscores.

    Handles various column name formats from Baseball-Reference.
    """
    # Normalize column names
    df.columns = df.columns.str.lower()

    # Exact mapping: war_daily files also carry WAR_off, WAR_def, WAR_rep, teamRpG,
    # oppRpG, ... so substring matching would create duplicate columns.
    exact = {"player_id": "player_id", "year_id": "year_id", "team_id": "team_id",
             "stint_id": "stint_id", "war": "war"}
    rename_map = {c: exact[c] for c in df.columns if c in exact}

    df = df.rename(columns=rename_map)

    # Check all required columns exist
    required = ["player_id", "year_id", "team_id", "stint_id", "war"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise MissingRawDataError(
            f"{file_name} missing expected columns: {missing_cols}\n"
            f"Expected: player_ID, year_ID, team_ID, stint_ID, WAR"
        )

    return df


def load_bwar(raw_dir: Path, seasons: tuple[int, int] = (1998, 2016)) -> dict[str, pd.DataFrame]:
    """Load Baseball-Reference WAR data for batters and pitchers.

    Reads war_daily_bat.txt and war_daily_pitch.txt (comma-separated with header).
    Filters to the specified season range.

    Params:
        raw_dir: Path to data/raw/ directory
        seasons: Tuple (start_year, end_year) inclusive

    Returns:
        Dict {"bat": DataFrame[...], "pit": DataFrame[...]} with columns:
            - bbref_id (str): Baseball-Reference player ID
            - season (int): Year
            - team_br (str): Team code (3-letter)
            - stint (int): Stint number within the season
            - war (float): WAR value (NULL converted to NaN)
            - pitcher_flag (bool): True for pitchers, False for batters

    Raises:
        MissingRawDataError: If either file is missing.
    """
    bref_dir = raw_dir / "bref"

    # Check both files exist
    bat_path = bref_dir / BWAR_BAT_FILE
    pit_path = bref_dir / BWAR_PIT_FILE

    missing = []
    if not bat_path.exists():
        missing.append(BWAR_BAT_FILE)
    if not pit_path.exists():
        missing.append(BWAR_PIT_FILE)

    if missing:
        msg = (
            f"Missing Baseball-Reference WAR files in {bref_dir}:\n"
            f"  Expected: {', '.join(missing)}\n"
            f"  Format: Comma-separated with header\n"
            f"  Download: https://www.baseball-reference.com/data/\n"
            f"  Or use: pybaseball.bwar_bat() / pybaseball.bwar_pitch()\n"
            f"  Save to data/raw/bref/"
        )
        raise MissingRawDataError(msg)

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

    start_year, end_year = seasons

    # Process batting data
    bat_df = _standardize_bwar_columns(bat_df, BWAR_BAT_FILE)
    bat_df = bat_df[(bat_df["year_id"] >= start_year) & (bat_df["year_id"] <= end_year)].copy()
    bat_df = bat_df.rename(columns={
        "player_id": "bbref_id",
        "year_id": "season",
        "team_id": "team_br",
        "stint_id": "stint",
    })
    bat_df["war"] = pd.to_numeric(bat_df["war"], errors="coerce")
    bat_df["pitcher_flag"] = False

    # Process pitching data
    pit_df = _standardize_bwar_columns(pit_df, BWAR_PIT_FILE)
    pit_df = pit_df[(pit_df["year_id"] >= start_year) & (pit_df["year_id"] <= end_year)].copy()
    pit_df = pit_df.rename(columns={
        "player_id": "bbref_id",
        "year_id": "season",
        "team_id": "team_br",
        "stint_id": "stint",
    })
    pit_df["war"] = pd.to_numeric(pit_df["war"], errors="coerce")
    pit_df["pitcher_flag"] = True

    # Select and order columns
    cols = ["bbref_id", "season", "team_br", "stint", "war", "pitcher_flag"]
    bat_df = bat_df[cols].reset_index(drop=True)
    pit_df = pit_df[cols].reset_index(drop=True)

    logger.info(f"Loaded bWAR: {len(bat_df)} batting, {len(pit_df)} pitching records")
    return {"bat": bat_df, "pit": pit_df}


def fetch_bwar(raw_dir: Path) -> None:
    """Download and save Baseball-Reference WAR data using pybaseball.

    Uses pybaseball.bwar_bat() and pybaseball.bwar_pitch() to fetch data.
    Saves to data/raw/bref/war_daily_bat.txt and war_daily_pitch.txt.

    Params:
        raw_dir: Path to data/raw/ directory

    Raises:
        MissingRawDataError: If download fails or pybaseball is not available.
    """
    try:
        import pybaseball
    except ImportError:
        raise MissingRawDataError(
            "pybaseball not installed. Install with: pip install pybaseball\n"
            "Or manually download WAR files from:\n"
            "  https://www.baseball-reference.com/data/\n"
            "  Save to data/raw/bref/war_daily_bat.txt and war_daily_pitch.txt"
        )

    bref_dir = raw_dir / "bref"
    bref_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Fetching Baseball-Reference bWAR batting data...")
        bat_df = pybaseball.bwar_bat()
        bat_df.to_csv(bref_dir / BWAR_BAT_FILE, index=False)

        logger.info("Fetching Baseball-Reference bWAR pitching data...")
        pit_df = pybaseball.bwar_pitch()
        pit_df.to_csv(bref_dir / BWAR_PIT_FILE, index=False)

        logger.info(f"Saved bWAR data to {bref_dir}/")
    except Exception as e:
        raise MissingRawDataError(
            f"Failed to fetch bWAR data: {e}\n"
            "Manually download from https://www.baseball-reference.com/data/\n"
            f"and save to {bref_dir}/"
        )
