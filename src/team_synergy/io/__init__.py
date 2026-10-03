"""Spec §1: Data I/O from Lahman, Retrosheet, WAR sources, and ID mapping."""

__all__ = [
    "MissingRawDataError",
    "load_lahman",
    "canonical_franchise",
    "load_gamelogs",
    "lineup_starts",
    "load_bwar",
    "fetch_bwar",
    "load_fwar",
    "fetch_fwar",
    "load_register",
    "fetch_register",
    "build_idmap",
    "mapping_failure_rate",
    "load_pecota",
    "fetch_pecota",
    "fetch_lahman",
    "fetch_gamelogs",
]


class MissingRawDataError(FileNotFoundError):
    """Raised when expected raw data files are missing.

    Message lists expected file names, format, and where to obtain them.
    """
    pass


from .lahman import load_lahman, canonical_franchise, fetch_lahman
from .retrosheet import load_gamelogs, lineup_starts, fetch_gamelogs
from .war_bref import load_bwar, fetch_bwar
from .war_fangraphs import load_fwar, fetch_fwar
from .idmap import load_register, fetch_register, build_idmap, mapping_failure_rate
from .pecota import load_pecota, fetch_pecota
