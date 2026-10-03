"""Common return type of the spec section 8 analysis modules."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class AnalysisResult:
    """Output of one analysis ``run``.

    name: short module name; tables: tidy DataFrames ready to plot or print;
    checks: list of dicts {"item", "value", "expected", "passed"} where ``expected`` quotes the
    paper's reported result and ``passed`` is bool, or None when the check is not applicable
    (e.g. needs real data).
    """

    name: str
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    checks: list[dict] = field(default_factory=list)

    def checks_frame(self) -> pd.DataFrame:
        """Checks as a DataFrame with columns item, value, expected, passed."""
        return pd.DataFrame(self.checks, columns=["item", "value", "expected", "passed"])
