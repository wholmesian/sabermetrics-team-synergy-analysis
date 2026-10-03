"""Tests for spec section 8 runner, figures, and report."""
import pytest
from pathlib import Path

from team_synergy.analysis.runner import run_all, save_results
from team_synergy.viz.figures import save_all as save_figures
from team_synergy.analysis.report import write_report


# `syn_inputs` is the session-scoped fixture from tests/conftest.py.


def test_run_all_basic(syn_inputs):
    """Test run_all with a small module subset."""
    results = run_all(syn_inputs, n_boot=10, seed=0,
                      only=["table1", "org_culture", "tc_persistence"])

    assert "table1" in results
    assert "org_culture" in results
    assert "tc_persistence" in results

    # Check structure
    r = results["table1"]
    assert r.name == "table1"
    assert "table1" in r.tables
    assert len(r.checks) > 0


def test_run_all_with_error(syn_inputs, monkeypatch):
    """Test that module errors are caught and recorded."""
    def error_run(inputs, n_boot=500, seed=0):
        raise ValueError("test error")

    # Import and patch
    import team_synergy.analysis.table1 as t1
    monkeypatch.setattr(t1, "run", error_run)

    results = run_all(syn_inputs, n_boot=10, seed=0, only=["table1"])

    assert "table1" in results
    r = results["table1"]
    assert len(r.checks) == 1
    assert r.checks[0]["item"] == "module error"
    assert "test error" in r.checks[0]["value"]
    assert r.checks[0]["passed"] is False


def test_run_all_cheap_modules(syn_inputs):
    """Run cheap modules (table1, org_culture, tc_persistence, player_eval, ross)."""
    modules = ["table1", "org_culture", "tc_persistence", "player_eval", "ross"]
    results = run_all(syn_inputs, n_boot=10, seed=0, only=modules)

    for name in modules:
        assert name in results
        assert results[name].tables  # has some tables


def test_save_results(syn_inputs, tmp_path):
    """Test save_results creates CSV files."""
    results = run_all(syn_inputs, n_boot=10, seed=0, only=["table1"])

    save_results(results, tmp_path)

    # Check files
    tables_dir = tmp_path / "tables"
    assert tables_dir.exists()

    # Should have table1__<name>.csv files
    csv_files = list(tables_dir.glob("*.csv"))
    assert len(csv_files) > 0

    # Check checks.csv
    checks_csv = tmp_path / "checks.csv"
    assert checks_csv.exists()


def test_save_figures(syn_inputs, tmp_path):
    """Test save_figures creates PNG files."""
    from team_synergy.analysis import pc_persistence, profiles
    results = run_all(syn_inputs, n_boot=10, seed=0,
                      only=["table1", "org_culture", "tc_persistence", "player_eval", "intangibles", "ross"])
    # The bootstrap-heavy modules run with a tiny jackknife so fig9/fig10 drawing stays covered.
    results["pc_persistence"] = pc_persistence.run(syn_inputs, n_boot=5, seed=0, max_jack=10)
    results["profiles"] = profiles.run(syn_inputs, n_boot=5, seed=0, max_jack=10)

    fig_paths = save_figures(results, tmp_path)
    assert {"fig9", "fig10"} <= {Path(f).stem for f in fig_paths}

    # Should have created some figures
    assert len(fig_paths) > 0

    # Check PNG files exist
    for fig_name in fig_paths:
        path = tmp_path / fig_name
        assert path.exists(), f"figure {fig_name} not found"


def test_write_report(syn_inputs, tmp_path):
    """Test write_report creates REPORT.md."""
    results = run_all(syn_inputs, n_boot=10, seed=0,
                      only=["table1", "org_culture", "tc_persistence"])

    report_path = tmp_path / "REPORT.md"
    write_report(results, syn_inputs, report_path, fig_dir_rel="figures")

    assert report_path.exists()

    content = report_path.read_text(encoding="utf-8")
    assert "재현 보고서" in content
    assert "표 1" in content
    assert "항목" in content  # checks table header
    assert "판정" in content


def test_integration_analyze(syn_inputs, tmp_path):
    """Integration test: run_all -> save_results -> save_figures -> write_report."""
    results = run_all(syn_inputs, n_boot=10, seed=0,
                      only=["table1", "org_culture"])

    save_results(results, tmp_path)
    save_figures(results, tmp_path / "figures")
    write_report(results, syn_inputs, tmp_path / "REPORT.md")

    # Check outputs
    assert (tmp_path / "tables").exists()
    assert (tmp_path / "checks.csv").exists()
    assert (tmp_path / "figures").exists()
    assert (tmp_path / "REPORT.md").exists()

    # Check at least one table CSV
    csv_files = list((tmp_path / "tables").glob("*.csv"))
    assert len(csv_files) > 0

    # Check report content
    content = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert "✅" in content or "❌" in content or "실데이터 필요" in content
