"""Spec §1: Lahman database readers (Teams, Batting, Fielding, Pitching, Appearances)."""

from pathlib import Path
import logging
import pandas as pd
from . import MissingRawDataError
from ..schema import FRANCHISE_MAP

logger = logging.getLogger(__name__)

# Expected Lahman CSV files (as documented by SABR)
LAHMAN_FILES = ["Teams", "Batting", "Fielding", "Pitching", "Appearances", "People", "Salaries", "Managers"]


def canonical_franchise(franch_id: pd.Series) -> pd.Series:
    """Apply FRANCHISE_MAP to convert historical franchise IDs to canonical 30-team codes.

    Params:
        franch_id: Series of Lahman franchID values

    Returns:
        Series with canonical franchise codes applied.
    """
    return franch_id.map(lambda x: FRANCHISE_MAP.get(x, x))


def load_lahman(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all Lahman CSV files from data/raw/lahman/.

    Reads Teams, Batting, Fielding, Pitching, Appearances, People, Salaries, Managers.
    Keeps original Lahman column names. Adds canonical franch_id to Teams frame.

    Params:
        raw_dir: Path to data/raw/ directory

    Returns:
        Dict of {table_name: DataFrame} with original Lahman columns.
        Teams frame has an added 'franch_id' column (canonical).

    Raises:
        MissingRawDataError: If any expected file is missing.
    """
    lahman_dir = raw_dir / "lahman"

    # Check all expected files exist
    missing = []
    for fname in LAHMAN_FILES:
        if not (lahman_dir / f"{fname}.csv").exists():
            missing.append(fname)

    if missing:
        msg = (
            f"Missing Lahman CSV files in {lahman_dir}:\n"
            f"  Expected: {', '.join(missing)}.csv\n"
            f"  Format: SABR/Lahman CSV (https://www.seanlahman.com/baseball-archive/statistics)\n"
            f"  Download: Baseball-Reference or SABR archive, extract to data/raw/lahman/"
        )
        raise MissingRawDataError(msg)

    # Load all tables
    tables = {}
    for fname in LAHMAN_FILES:
        fpath = lahman_dir / f"{fname}.csv"
        try:
            tables[fname] = pd.read_csv(fpath)
        except Exception as e:
            raise MissingRawDataError(
                f"Failed to read {fname}.csv: {e}"
            )

    # Add canonical franch_id to Teams
    if "Teams" in tables:
        teams = tables["Teams"]
        if "franchID" in teams.columns:
            teams["franch_id"] = canonical_franchise(teams["franchID"])
        else:
            raise MissingRawDataError(
                f"Teams.csv missing 'franchID' column. Check Lahman format."
            )

    logger.info(f"Loaded Lahman database: {len(tables)} tables from {lahman_dir}")
    return tables
