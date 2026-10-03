"""Shared HTTP helpers for ``synergy fetch``: polite downloads and a MANIFEST.json writer.

Rules (spec section 0): descriptive User-Agent, >= 1 s between requests to the same host,
never work around blocks (403/Cloudflare) -- callers turn failures into manual-placement
instructions instead.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

from . import MissingRawDataError

USER_AGENT = "team-synergy-research/0.1 (academic replication of Brave et al. 2019)"
MIN_INTERVAL_S = 1.0
TIMEOUT_S = 120
MANIFEST_NAME = "MANIFEST.json"

_last_request: dict[str, float] = {}


def _throttle(host: str) -> None:
    wait = MIN_INTERVAL_S - (time.monotonic() - _last_request.get(host, -1e9))
    if wait > 0:
        time.sleep(wait)
    _last_request[host] = time.monotonic()


def http_get(url: str, **kwargs) -> requests.Response:
    """GET with the project User-Agent and per-host throttling. Raises MissingRawDataError on HTTP/network errors."""
    _throttle(urlparse(url).netloc)
    headers = {"User-Agent": USER_AGENT, **kwargs.pop("headers", {})}
    try:
        r = requests.get(url, headers=headers, timeout=TIMEOUT_S, **kwargs)
    except requests.RequestException as e:
        raise MissingRawDataError(f"request failed for {url}: {e}") from e
    if r.status_code != 200:
        raise MissingRawDataError(f"HTTP {r.status_code} for {url}")
    return r


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def record(raw_dir: Path, path: Path, source_url: str, *, downloaded_utc: str | None = None) -> None:
    """Add/replace the manifest entry of ``path`` (key = path relative to raw_dir)."""
    mpath = Path(raw_dir) / MANIFEST_NAME
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    key = str(Path(path).resolve().relative_to(Path(raw_dir).resolve()))
    manifest[key] = {
        "source_url": source_url,
        "sha256": sha256_of(path),
        "size": Path(path).stat().st_size,
        "downloaded_utc": downloaded_utc or _now(),
    }
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def has_entry(raw_dir: Path, path: Path) -> bool:
    mpath = Path(raw_dir) / MANIFEST_NAME
    if not mpath.exists():
        return False
    key = str(Path(path).resolve().relative_to(Path(raw_dir).resolve()))
    return key in json.loads(mpath.read_text())


def download_file(url: str, dest: Path, raw_dir: Path, *, force: bool = False) -> bool:
    """Download ``url`` to ``dest`` unless it exists (and not ``force``). Returns True if downloaded."""
    dest = Path(dest)
    if dest.exists() and not force:
        if not has_entry(raw_dir, dest):
            record(raw_dir, dest, url, downloaded_utc=datetime.fromtimestamp(
                dest.stat().st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        return False
    r = http_get(url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    record(raw_dir, dest, url)
    return True
