"""Spec §1: PECOTA pre-season win projections (optional, for Table 3)."""

from pathlib import Path
import logging
import pandas as pd
from . import MissingRawDataError

logger = logging.getLogger(__name__)

# PECOTA file name
PECOTA_FILE = "pecota.csv"


def load_pecota(raw_dir: Path) -> pd.DataFrame | None:
    """Load PECOTA pre-season win projections (optional).

    Reads data/raw/pecota/pecota.csv if present. Returns None if file does not
    exist (PECOTA is optional; analyses that need it will be skipped).

    Params:
        raw_dir: Path to data/raw/ directory

    Returns:
        DataFrame with columns:
            - franch_id (str): Canonical 30-franchise code
            - season (int): Year
            - pecota_w (float): Projected wins

        Returns None if file does not exist.
    """
    pecota_path = raw_dir / "pecota" / PECOTA_FILE

    if not pecota_path.exists():
        logger.info(f"PECOTA file not found at {pecota_path} (optional; Table 3 analyses will be skipped)")
        return None

    try:
        df = pd.read_csv(pecota_path)
    except Exception as e:
        raise MissingRawDataError(
            f"Failed to read PECOTA file {pecota_path}: {e}\n"
            f"If you don't have PECOTA, this is optional. Just skip Table 3 analyses."
        )

    # Standardize column names
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # Check required columns
    required = ["franch_id", "season", "pecota_w"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning(
            f"PECOTA file missing expected columns: {missing}\n"
            f"Expected: franch_id (canonical), season, pecota_w"
        )

    # Ensure correct data types
    df["season"] = df["season"].astype(int)
    df["pecota_w"] = pd.to_numeric(df["pecota_w"], errors="coerce")

    logger.info(f"Loaded PECOTA: {len(df)} team-season projections")
    return df[required].reset_index(drop=True)


def fetch_pecota(raw_dir: Path) -> None:
    """Note: PECOTA is proprietary (Baseball Prospectus) and cannot be auto-fetched.

    This function raises with instructions on obtaining PECOTA data.

    Params:
        raw_dir: Path to data/raw/ directory

    Raises:
        MissingRawDataError: Always; PECOTA must be manually obtained.
    """
    raise MissingRawDataError(
        "PECOTA is proprietary Baseball Prospectus data and cannot be auto-fetched.\n"
        "To use Table 3 analyses:\n"
        f"  1. Obtain PECOTA projections from https://www.baseballprospectus.com/\n"
        f"  2. Format as CSV with columns: franch_id, season, pecota_w\n"
        f"  3. Save to data/raw/pecota/pecota.csv\n"
        f"If you skip this, Table 3 analyses will be automatically omitted."
    )
