"""Spec §10: Tests for configuration loading."""

import pytest
from team_synergy.config import load_config, load_weights, config_metadata


def test_load_config_default():
    """Test loading default configuration."""
    cfg = load_config()

    # Basic structure
    assert isinstance(cfg, dict)
    assert "seasons" in cfg
    assert "war_source" in cfg
    assert cfg["war_source"] in ["fwar", "bwar"]
    assert "eta" in cfg
    assert cfg["seasons"]["start"] == 1998
    assert cfg["seasons"]["end"] == 2016


def test_load_config_with_overrides():
    """Test deep merging of overrides."""
    overrides = {
        "war_source": "bwar",
        "seasons": {"start": 2000},
        "new_key": "value",
    }
    cfg = load_config(overrides=overrides)

    # Check overrides applied
    assert cfg["war_source"] == "bwar"
    assert cfg["seasons"]["start"] == 2000
    assert cfg["seasons"]["end"] == 2016  # not overridden
    assert cfg["new_key"] == "value"


def test_load_weights():
    """Test loading weights configuration."""
    weights = load_weights()

    assert isinstance(weights, dict)
    assert "defensive_weights" in weights
    assert "batting_order" in weights
    assert "pitcher_d" in weights
    assert weights["weight_source"] == "table2"

    # Check defensive weights for both fwar and bwar
    assert "fwar" in weights["defensive_weights"]
    assert "bwar" in weights["defensive_weights"]
    assert "C" in weights["defensive_weights"]["fwar"]
    assert len(weights["batting_order"]) == 9


def test_config_metadata():
    """Test extracting JSON-serializable metadata."""
    cfg = load_config()
    meta = config_metadata(cfg)

    # Should be serializable
    import json
    json_str = json.dumps(meta)
    assert isinstance(json_str, str)
    assert len(json_str) > 0
