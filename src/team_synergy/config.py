"""Spec §9: Configuration loading and validation."""

import json
import os
from pathlib import Path
from typing import Any, Optional

import yaml


def _find_repo_root(start_path: Optional[str] = None) -> Path:
    """
    Walk up from start_path to find the directory containing pyproject.toml.

    Args:
        start_path: Starting directory. Defaults to current working directory.

    Returns:
        Path to the repository root.

    Raises:
        FileNotFoundError: If pyproject.toml is not found.
    """
    if start_path is None:
        start_path = os.getcwd()

    current = Path(start_path).resolve()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent

    raise FileNotFoundError("pyproject.toml not found; cannot determine repo root.")


def _resolve_path(rel_path: str, repo_root: Path) -> Path:
    """Resolve a relative path against the repo root."""
    return (repo_root / rel_path).resolve()


def _deep_merge(base: dict, overrides: dict) -> dict:
    """
    Deep merge overrides into base, returning a new dict.

    Args:
        base: Base configuration dictionary.
        overrides: Dictionary with overrides. Can be nested.

    Returns:
        Merged dictionary.
    """
    result = base.copy()
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str = "config/default.yaml", overrides: Optional[dict] = None) -> dict:
    """
    Load configuration from a YAML file and optionally merge overrides.

    Paths are resolved relative to the repository root (detected by walking up to pyproject.toml).

    Args:
        path: Relative path to config YAML (default: config/default.yaml).
        overrides: Dict of overrides. Nested dicts are deep-merged (not replaced).

    Returns:
        Merged configuration dict.

    Raises:
        FileNotFoundError: If config file or repo root not found.
    """
    repo_root = _find_repo_root()
    config_path = _resolve_path(path, repo_root)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}

    if overrides:
        config = _deep_merge(config, overrides)

    return config


def load_weights(path: str = "config/weights.yaml") -> dict:
    """
    Load weights configuration from a YAML file.

    Args:
        path: Relative path to weights YAML (default: config/weights.yaml).

    Returns:
        Weights configuration dict.

    Raises:
        FileNotFoundError: If weights file or repo root not found.
    """
    repo_root = _find_repo_root()
    weights_path = _resolve_path(path, repo_root)

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    with open(weights_path, "r") as f:
        weights = yaml.safe_load(f) or {}

    return weights


def config_metadata(cfg: dict) -> dict:
    """
    Extract a JSON-serializable copy of the configuration for output metadata.

    Removes non-serializable objects and returns a clean dict suitable for
    writing to JSON (e.g., in output tables or run logs).

    Args:
        cfg: Configuration dict.

    Returns:
        JSON-serializable copy.
    """
    # For now, just ensure the dict is JSON-serializable by testing.
    try:
        json.dumps(cfg)
        return cfg.copy()
    except (TypeError, ValueError) as e:
        # If not serializable, do a simple recursive sanitization.
        result = {}
        for k, v in cfg.items():
            if isinstance(v, dict):
                result[k] = config_metadata(v)
            elif isinstance(v, (list, tuple)):
                result[k] = [
                    config_metadata(item) if isinstance(item, dict) else item
                    for item in v
                ]
            elif isinstance(v, (str, int, float, bool, type(None))):
                result[k] = v
            # Skip non-serializable types
        return result
