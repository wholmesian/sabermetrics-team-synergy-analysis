"""Spec §1: Retrosheet game log parser (batting order and fielding positions)."""

from pathlib import Path
import logging
import pandas as pd
import numpy as np
from . import MissingRawDataError

logger = logging.getLogger(__name__)

# Retrosheet game log format: 161 fields, header-less, comma-separated.
# Field positions (0-based indexing):
#   - Season: field 3
#   - Date: field 0 (YYYYMMDD)
#   - Game number: field 4
#   - Visiting team: field 1 (3-letter Retrosheet code)
#   - Home team: field 2 (3-letter Retrosheet code)
#   - Visiting starting lineup: fields 105-131 (9 players × 3 fields: id, name, position)
#   - Home starting lineup: fields 132-158 (9 players × 3 fields: id, name, position)
# Field numbers from Retrosheet documentation (1-based): 106–132 visitor, 133–159 home.
# Converted to 0-based: 105–131 visitor, 132–158 home.

# Visitor batting order: 9 slots, each with 3 fields (id, name, position)
VISITOR_LINEUP_START = 105  # 0-based index for first visitor slot id
VISITOR_LINEUP_SLOTS = 9
HOME_LINEUP_START = 132     # 0-based index for first home slot id
HOME_LINEUP_SLOTS = 9


def load_gamelogs(raw_dir: Path, seasons: tuple[int, int] = (1998, 2016)) -> pd.DataFrame:
    """Load Retrosheet game logs (gl{YYYY}.txt) and parse batting lineups.

    Reads game log files from data/raw/retrosheet/ for the specified season range.
    Returns a long DataFrame with one row per player appearance in the batting lineup
    (visiting or home, slots 1–9).

    Params:
        raw_dir: Path to data/raw/ directory
        seasons: Tuple (start_year, end_year) inclusive for which years to load

    Returns:
        DataFrame with columns:
            - season (int): Year of game
            - date (str): Game date (YYYYMMDD)
            - game_num (int): Game number for the day (1–2)
            - team_retro (str): 3-letter Retrosheet team code
            - side (str): "away" or "home"
            - slot (int): Batting order slot (1–9)
            - retro_id (str): Retrosheet player ID
            - def_pos (int): Fielding position (1–10, 1=pitcher)

        One row per (game, team, slot) combination with data.

    Raises:
        MissingRawDataError: If game log files are missing.
    """
    retrosheet_dir = raw_dir / "retrosheet"
    start_year, end_year = seasons

    # Check at least some game logs exist
    years_to_load = list(range(start_year, end_year + 1))
    missing_years = []
    for year in years_to_load:
        # Accept both gl{YYYY}.txt and GL{YYYY}.TXT
        if not ((retrosheet_dir / f"gl{year}.txt").exists() or
                (retrosheet_dir / f"GL{year}.TXT").exists()):
            missing_years.append(year)

    if missing_years:
        msg = (
            f"Missing Retrosheet game log files in {retrosheet_dir}:\n"
            f"  Expected: gl{{1998}}.txt through gl{{2016}}.txt (or GL{{YYYY}}.TXT)\n"
            f"  Format: Retrosheet header-less CSV, 161 fields\n"
            f"  Download: https://www.retrosheet.org/gamelogs/\n"
            f"  Extract to data/raw/retrosheet/"
        )
        raise MissingRawDataError(msg)

    rows = []
    for year in years_to_load:
        # Try both cases
        fpath = retrosheet_dir / f"gl{year}.txt"
        if not fpath.exists():
            fpath = retrosheet_dir / f"GL{year}.TXT"

        if not fpath.exists():
            continue

        try:
            # Load the game log file (no header)
            df = pd.read_csv(
                fpath,
                header=None,
                dtype=str,  # Read all as strings initially
                low_memory=False
            )
        except Exception as e:
            raise MissingRawDataError(
                f"Failed to read game log {fpath}: {e}"
            )

        # Parse each game's lineups
        for _, game in df.iterrows():
            try:
                season = int(game.iloc[3])
                date = game.iloc[0]
                game_num = int(game.iloc[4])
                visiting_team = game.iloc[1]
                home_team = game.iloc[2]

                # Parse visitor starting lineup (slots 1-9)
                for slot in range(1, VISITOR_LINEUP_SLOTS + 1):
                    field_idx = VISITOR_LINEUP_START + (slot - 1) * 3
                    retro_id = game.iloc[field_idx] if field_idx < len(game) else None
                    pos_str = game.iloc[field_idx + 2] if field_idx + 2 < len(game) else None

                    if retro_id and retro_id.strip():
                        try:
                            def_pos = int(pos_str) if pos_str else 0
                        except (ValueError, TypeError):
                            def_pos = 0

                        rows.append({
                            "season": season,
                            "date": date,
                            "game_num": game_num,
                            "team_retro": visiting_team,
                            "side": "away",
                            "slot": slot,
                            "retro_id": retro_id.strip(),
                            "def_pos": def_pos,
                        })

                # Parse home starting lineup (slots 1-9)
                for slot in range(1, HOME_LINEUP_SLOTS + 1):
                    field_idx = HOME_LINEUP_START + (slot - 1) * 3
                    retro_id = game.iloc[field_idx] if field_idx < len(game) else None
                    pos_str = game.iloc[field_idx + 2] if field_idx + 2 < len(game) else None

                    if retro_id and retro_id.strip():
                        try:
                            def_pos = int(pos_str) if pos_str else 0
                        except (ValueError, TypeError):
                            def_pos = 0

                        rows.append({
                            "season": season,
                            "date": date,
                            "game_num": game_num,
                            "team_retro": home_team,
                            "side": "home",
                            "slot": slot,
                            "retro_id": retro_id.strip(),
                            "def_pos": def_pos,
                        })
            except (IndexError, ValueError, TypeError) as e:
                # Skip malformed games
                logger.debug(f"Skipped malformed game record in {year}: {e}")
                continue

    result = pd.DataFrame(rows)
    result["season"] = result["season"].astype(int)
    result["slot"] = result["slot"].astype(int)
    result["def_pos"] = result["def_pos"].astype(int)

    logger.info(f"Loaded Retrosheet game logs: {len(result)} lineup entries for seasons {start_year}–{end_year}")
    return result


def lineup_starts(gamelogs: pd.DataFrame) -> pd.DataFrame:
    """Count starting appearances per player-season-team combination.

    Aggregates the game-level lineup data to count how many times each player
    started in each slot for each team-season.

    Params:
        gamelogs: DataFrame from load_gamelogs()

    Returns:
        DataFrame with columns:
            - season (int)
            - team_retro (str)
            - retro_id (str)
            - slot (int): Batting order slot (1–9)
            - starts (int): Number of games started in this slot
    """
    starts = gamelogs.groupby(["season", "team_retro", "retro_id", "slot"]).size().reset_index(name="starts")
    logger.info(f"Computed {len(starts)} (player, slot, team) starting combinations")
    return starts


RETROSHEET_ZIP_URL = "https://www.retrosheet.org/gamelogs/gl{year}.zip"


def fetch_gamelogs(raw_dir: Path, seasons: tuple[int, int] = (1998, 2016), force: bool = False) -> list[Path]:
    """Download Retrosheet game logs gl{YYYY}.zip and unzip to data/raw/retrosheet/gl{YYYY}.txt.

    Existing gl{YYYY}.txt files are kept unless ``force``. Raises MissingRawDataError listing the
    seasons that failed (other seasons are still saved).
    """
    import io as _io
    import zipfile
    from .download import http_get, record

    out_dir = Path(raw_dir) / "retrosheet"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved, failed = [], []
    for year in range(seasons[0], seasons[1] + 1):
        dest = out_dir / f"gl{year}.txt"
        if dest.exists() and not force:
            saved.append(dest)
            continue
        url = RETROSHEET_ZIP_URL.format(year=year)
        try:
            r = http_get(url)
            with zipfile.ZipFile(_io.BytesIO(r.content)) as zf:
                member = next(n for n in zf.namelist() if n.lower() == f"gl{year}.txt")
                dest.write_bytes(zf.read(member))
        except (MissingRawDataError, zipfile.BadZipFile, StopIteration) as e:
            failed.append(f"{year} ({e})")
            continue
        record(raw_dir, dest, url)
        saved.append(dest)
    if failed:
        raise MissingRawDataError(
            f"Failed to fetch Retrosheet game logs: {'; '.join(failed)}\n"
            f"Download gl{{YYYY}}.zip from https://www.retrosheet.org/gamelogs/ , unzip, "
            f"and save as {out_dir}/gl{{YYYY}}.txt"
        )
    return saved
