"""Orchestration of spec section 8 analyses: run_all, save_results."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from team_synergy.analysis.inputs import AnalysisInputs
from team_synergy.analysis.result import AnalysisResult

log = logging.getLogger("team_synergy")

# Spec section 8 module list in order
MODULES = ["table1", "org_culture", "tc_persistence", "pecota", "player_eval",
           "pc_persistence", "profiles", "intangibles", "salary", "ross"]


def run_all(inputs: AnalysisInputs, n_boot: int = 500, seed: int = 0, only: list[str] | None = None
            ) -> dict[str, AnalysisResult]:
    """Run all spec section 8 analyses.

    Args:
        inputs: AnalysisInputs (from load_inputs or synthetic_inputs).
        n_boot: number of bootstrap replicates for CIs.
        seed: random seed (passed to each module).
        only: restrict to named modules (e.g. ["table1", "org_culture"]).

    Returns:
        dict mapping module name to AnalysisResult. Modules with exceptions
        get an AnalysisResult with a single failed check {"item": "module error",
        "value": str(e), "expected": "", "passed": False}.
    """
    from team_synergy.analysis import (
        table1, org_culture, tc_persistence, pecota, player_eval,
        pc_persistence, profiles, intangibles, salary, ross
    )

    modules_map = {
        "table1": table1, "org_culture": org_culture, "tc_persistence": tc_persistence,
        "pecota": pecota, "player_eval": player_eval, "pc_persistence": pc_persistence,
        "profiles": profiles, "intangibles": intangibles, "salary": salary, "ross": ross,
    }

    to_run = [m for m in MODULES if only is None or m in only]
    results = {}

    for name in to_run:
        mod = modules_map[name]
        try:
            t0 = time.time()
            result = mod.run(inputs, n_boot=n_boot, seed=seed)
            elapsed = time.time() - t0
            log.info(f"{name}: {elapsed:.2f}s, {len(result.tables)} tables, {len(result.checks)} checks")
            results[name] = result
        except Exception as e:
            log.error(f"{name}: {type(e).__name__}: {e}")
            results[name] = AnalysisResult(
                name=name, tables={},
                checks=[{"item": "module error", "value": str(e), "expected": "", "passed": False}]
            )

    return results


def save_results(results: dict[str, AnalysisResult], out_dir: Path | str) -> None:
    """Save all tables and checks to CSV.

    Writes:
        outputs/<src>/tables/<module>__<table>.csv for each table,
        outputs/<src>/checks.csv concatenating all checks with a 'module' column.
    """
    out_dir = Path(out_dir)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    all_checks = []
    for name, result in results.items():
        # Save tables
        for table_name, df in result.tables.items():
            path = tables_dir / f"{name}__{table_name}.csv"
            df.to_csv(path, index=False)
            log.debug(f"wrote {path}")

        # Collect checks
        if result.checks:
            checks_df = result.checks_frame()
            checks_df.insert(0, "module", name)
            all_checks.append(checks_df)

    # Save combined checks
    if all_checks:
        checks_all = pd.concat(all_checks, ignore_index=True)
        checks_path = out_dir / "checks.csv"
        checks_all.to_csv(checks_path, index=False)
        log.info(f"checks written to {checks_path}")
