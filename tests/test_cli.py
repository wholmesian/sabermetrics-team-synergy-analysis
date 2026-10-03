"""CLI smoke tests (spec section 9)."""
import json

from team_synergy.cli import main


def test_estimate_synthetic_full_only(tmp_path, capsys):
    rc = main(["estimate", "--war", "fwar", "--synthetic", "--seed", "0", "--full-only",
               "--seasons", "1998", "1999", "--synthetic-teams", "12", "--synthetic-roster", "30",
               "--out-dir", str(tmp_path), "--set", "sign_convention=consistent"])
    assert rc == 0
    assert (tmp_path / "metrics_full.parquet").exists()
    meta = json.loads((tmp_path / "run_metadata.json").read_text())
    assert meta["config"]["sign_convention"] == "consistent"
    out = capsys.readouterr().out
    assert "PASS" in out or "FAIL" in out
    assert "rho=" in out


def test_build_missing_raw_returns_2(tmp_path, capsys):
    raw = tmp_path / "raw"
    raw.mkdir()
    assert main(["build", "--war", "fwar", "--set", f"paths.raw={raw}"]) == 2
    assert capsys.readouterr().out.strip()


def test_analyze_synthetic(tmp_path, capsys, monkeypatch, syn_inputs):
    """Test analyze with synthetic data (quick check with small n_boot).

    The synthetic estimation itself is covered elsewhere; reuse the session fixture."""
    import team_synergy.analysis.inputs as inputs_mod
    monkeypatch.setattr(inputs_mod, "synthetic_inputs", lambda **kw: syn_inputs)
    rc = main(["analyze", "--synthetic", "--n-boot", "10", "--only", "table1,org_culture",
               "--out-dir", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "REPORT.md").exists()
    assert (tmp_path / "checks.csv").exists()
    assert (tmp_path / "tables").exists()
    assert (tmp_path / "figures").exists()


def test_analyze_missing_data(tmp_path):
    """Test analyze returns 2 when data is missing and not synthetic."""
    rc = main(["analyze", "--processed-dir", str(tmp_path)])  # empty dir: no estimation outputs
    assert rc == 2
