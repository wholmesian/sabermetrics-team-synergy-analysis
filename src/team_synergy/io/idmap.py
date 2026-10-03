"""Spec §1: Player ID mapping (Lahman, Chadwick register, FanGraphs, Baseball-Reference)."""

from pathlib import Path
import logging
import pandas as pd
import numpy as np
from . import MissingRawDataError

logger = logging.getLogger(__name__)

# Chadwick register file names
CHADWICK_REGISTER_FILE = "people.csv"
CHADWICK_REGISTER_PATTERN = "people-*.csv"


def load_register(raw_dir: Path) -> pd.DataFrame:
    """Load Chadwick register for ID mapping across data sources.

    Reads either a single people.csv or multiple people-*.csv files from
    data/raw/chadwick/ and concatenates them.

    Params:
        raw_dir: Path to data/raw/ directory

    Returns:
        DataFrame with columns:
            - key_bbref (str): Baseball-Reference player ID
            - key_fangraphs (Int64, nullable): FanGraphs player ID (may be NaN)
            - key_retro (str): Retrosheet player ID
            - key_mlbam (str): MLB Advanced Media ID (optional)
            - name_first (str): First name
            - name_last (str): Last name

    Raises:
        MissingRawDataError: If register file(s) not found.
    """
    chadwick_dir = raw_dir / "chadwick"

    # Try single file first
    register_path = chadwick_dir / CHADWICK_REGISTER_FILE

    if register_path.exists():
        try:
            df = pd.read_csv(register_path)
        except Exception as e:
            raise MissingRawDataError(f"Failed to read {register_path}: {e}")
    else:
        # Try split files
        split_files = sorted(chadwick_dir.glob(CHADWICK_REGISTER_PATTERN))
        if not split_files:
            msg = (
                f"Missing Chadwick register in {chadwick_dir}:\n"
                f"  Expected: {CHADWICK_REGISTER_FILE} or {CHADWICK_REGISTER_PATTERN}\n"
                f"  Format: CSV with columns key_bbref, key_fangraphs, key_retro, name_first, name_last\n"
                f"  Download: https://github.com/chadwickbureau/register/\n"
                f"  Save to data/raw/chadwick/"
            )
            raise MissingRawDataError(msg)

        try:
            dfs = [pd.read_csv(f) for f in split_files]
            df = pd.concat(dfs, ignore_index=True)
        except Exception as e:
            raise MissingRawDataError(f"Failed to read Chadwick split files: {e}")

    # Standardize column names (normalize to lowercase)
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # Ensure required columns exist
    required_cols = ["key_bbref", "key_fangraphs", "key_retro", "name_first", "name_last"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise MissingRawDataError(
            f"Chadwick register missing columns: {missing}\n"
            f"Expected: {', '.join(required_cols)}"
        )

    # Ensure key_fangraphs is nullable integer type
    df["key_fangraphs"] = pd.to_numeric(df["key_fangraphs"], errors="coerce").astype("Int64")

    logger.info(f"Loaded Chadwick register: {len(df)} players")
    return df[required_cols]


def fetch_register(raw_dir: Path) -> None:
    """Download and save Chadwick register using pybaseball.

    Uses pybaseball.chadwick_register() to fetch the register.
    Saves to data/raw/chadwick/people.csv.

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
            "Or download from:\n"
            "  https://github.com/chadwickbureau/register/\n"
            f"  Save to data/raw/chadwick/"
        )

    chadwick_dir = raw_dir / "chadwick"
    chadwick_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Fetching Chadwick register...")
        df = pybaseball.chadwick_register()
        df.to_csv(chadwick_dir / CHADWICK_REGISTER_FILE, index=False)
        logger.info(f"Saved Chadwick register to {chadwick_dir}/")
    except Exception as e:
        raise MissingRawDataError(
            f"Failed to fetch Chadwick register: {e}\n"
            "Download from https://github.com/chadwickbureau/register/\n"
            f"and save to {chadwick_dir}/"
        )


def build_idmap(
    people_lahman: pd.DataFrame,
    register: pd.DataFrame
) -> pd.DataFrame:
    """Build ID mapping table by joining Lahman People to Chadwick register.

    Joins Lahman People table (with playerID, bbrefID, retroID) to the Chadwick
    register on bbref_id, creating a unified ID mapping across all sources.

    Params:
        people_lahman: Lahman People DataFrame with columns: playerID, bbrefID, retroID
        register: Chadwick register from load_register()

    Returns:
        DataFrame with columns:
            - player_id (str): Lahman playerID (primary key)
            - bbref_id (str): Baseball-Reference ID
            - retro_id (str): Retrosheet ID
            - idfg (Int64, nullable): FanGraphs ID

    Raises:
        ValueError: If joining fails.
    """
    # Ensure required columns in Lahman data
    required_lahman = ["playerID", "bbrefID", "retroID"]
    missing = [c for c in required_lahman if c not in people_lahman.columns]
    if missing:
        raise ValueError(
            f"Lahman People missing columns: {missing}\n"
            f"Expected: {', '.join(required_lahman)}"
        )

    # Rename Lahman columns to match register
    lahman = people_lahman[required_lahman].rename(columns={
        "playerID": "player_id",
        "bbrefID": "key_bbref",
        "retroID": "key_retro",
    })

    # Join to register on key_bbref
    merged = lahman.merge(
        register,
        on="key_bbref",
        how="left",
        suffixes=("_lahman", "_register")
    )

    # Select and rename output columns
    merged = merged.rename(columns={
        "key_bbref": "bbref_id",
        "key_retro_lahman": "retro_id",  # Prefer Lahman's retroID if present
        "key_fangraphs": "idfg",
    })

    # Use Lahman retroID if available, otherwise use register
    if "key_retro_lahman" in merged.columns and "key_retro_register" in merged.columns:
        merged["retro_id"] = merged["key_retro_lahman"].fillna(merged["key_retro_register"])
        merged = merged.drop(columns=["key_retro_register"])
    elif "key_retro_register" in merged.columns:
        merged = merged.rename(columns={"key_retro_register": "retro_id"})

    # Drop extra columns
    output_cols = ["player_id", "bbref_id", "retro_id", "idfg"]
    output = merged[[c for c in output_cols if c in merged.columns]].copy()

    # Ensure idfg is nullable integer
    if "idfg" in output.columns:
        output["idfg"] = pd.to_numeric(output["idfg"], errors="coerce").astype("Int64")

    logger.info(f"Built ID map: {len(output)} players mapped")
    return output.reset_index(drop=True)


def mapping_failure_rate(idmap: pd.DataFrame) -> float:
    """Compute the fraction of players missing FanGraphs IDs.

    Params:
        idmap: DataFrame from build_idmap()

    Returns:
        Fraction of rows with NaN idfg (should be < 0.01 per spec).
    """
    if "idfg" not in idmap.columns:
        return 0.0

    missing = idmap["idfg"].isna().sum()
    rate = missing / len(idmap)
    logger.info(f"ID mapping failure rate: {rate:.2%} ({missing}/{len(idmap)})")
    return rate
