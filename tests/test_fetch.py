"""Tests for ``synergy fetch`` and the cross-checks (HTTP is mocked; no network)."""
import io
import json
import zipfile

import pandas as pd
import pytest

from team_synergy import cli
from team_synergy.io import (MissingRawDataError, download, fetch_bwar, fetch_gamelogs, fetch_lahman,
                             fetch_register, fetch_fwar)
from team_synergy.io import crosscheck


class FakeResp:
    def __init__(self, content=b"", status=200, json_data=None):
        self.content, self.status_code, self._json = content, status, json_data

    def json(self):
        return self._json


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(download, "MIN_INTERVAL_S", 0.0)


def _patch_get(monkeypatch, handler):
    calls = []

    def fake_get(url, headers=None, timeout=None, **kw):
        calls.append((url, headers))
        return handler(url)
    monkeypatch.setattr(download.requests, "get", fake_get)
    return calls


def _gl_text(year, games):
    # games: list of (visitor, home, vs, hs) -> minimal 11-column log
    return "".join(f'"{year}0401",0,"Mon","{v}","AL",1,"{h}","AL",1,{vs},{hs}\n' for v, h, vs, hs in games)


def _gl_zip(year, games):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"GL{year}.TXT", _gl_text(year, games))
    return buf.getvalue()


def test_gamelogs_download_skip_and_manifest(tmp_path, monkeypatch):
    calls = _patch_get(monkeypatch, lambda url: FakeResp(_gl_zip(int(url[-8:-4]), [("NYA", "BOS", 3, 2)])))
    fetch_gamelogs(tmp_path, (2000, 2001))
    assert (tmp_path / "retrosheet" / "gl2000.txt").exists()
    assert len(calls) == 2 and "team-synergy" in calls[0][1]["User-Agent"]
    e = json.loads((tmp_path / "MANIFEST.json").read_text())["retrosheet/gl2000.txt"]
    assert e["source_url"].endswith("gl2000.zip") and len(e["sha256"]) == 64 and e["size"] > 0
    assert e["downloaded_utc"].endswith("Z")
    fetch_gamelogs(tmp_path, (2000, 2001))
    assert len(calls) == 2  # existing files skipped
    fetch_gamelogs(tmp_path, (2000, 2000), force=True)
    assert len(calls) == 3


def test_gamelogs_partial_failure(tmp_path, monkeypatch):
    _patch_get(monkeypatch, lambda url: FakeResp(_gl_zip(2000, [("A", "B", 1, 0)])) if "2000" in url
               else FakeResp(status=404))
    with pytest.raises(MissingRawDataError, match="2001"):
        fetch_gamelogs(tmp_path, (2000, 2001))
    assert (tmp_path / "retrosheet" / "gl2000.txt").exists()


def test_register_and_bwar_download(tmp_path, monkeypatch):
    _patch_get(monkeypatch, lambda url: FakeResp(b"a,b\n1,2\n"))
    fetch_register(tmp_path)
    assert len(list((tmp_path / "chadwick").glob("people-*.csv"))) == 16
    fetch_bwar(tmp_path)
    assert (tmp_path / "bref" / "war_daily_bat.txt").exists()
    assert (tmp_path / "bref" / "war_daily_pitch.txt").exists()


def test_bwar_403_gives_manual_instructions(tmp_path, monkeypatch):
    calls = _patch_get(monkeypatch, lambda url: FakeResp(status=403))
    with pytest.raises(MissingRawDataError, match="403") as ei:
        fetch_bwar(tmp_path)
    assert "war_daily_bat.txt" in str(ei.value) and len(calls) == 1  # no retries around the block


def test_lahman_manual_and_zip_extract(tmp_path):
    with pytest.raises(MissingRawDataError, match="sabr.org"):
        fetch_lahman(tmp_path)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for n in ["Teams", "Batting", "Fielding", "Pitching", "Appearances", "People", "Salaries", "Managers"]:
            zf.writestr(f"lahman_1871-2025_csv/{n}.csv", "a\n1\n")
    (tmp_path / "lahman" / "x.zip").write_bytes(buf.getvalue())
    assert len(fetch_lahman(tmp_path)) == 8
    assert "lahman/Teams.csv" in json.loads((tmp_path / "MANIFEST.json").read_text())


def test_fwar_api(tmp_path, monkeypatch):
    payload = {"data": [{"playerid": "10155", "Name": '<a href="x">Mike Trout</a>', "Team": "<a>LAA</a>",
                         "Season": 2016, "WAR": 8.6}]}
    _patch_get(monkeypatch, lambda url: FakeResp(json_data=payload))
    fetch_fwar(tmp_path, (2016, 2016))
    df = pd.read_csv(tmp_path / "fangraphs" / "fg_batting.csv")
    assert df.loc[0, "Name"] == "Mike Trout" and df.loc[0, "IDfg"] == 10155
    _patch_get(monkeypatch, lambda url: FakeResp(status=403))
    with pytest.raises(MissingRawDataError, match="403"):
        fetch_fwar(tmp_path, (2016, 2016), force=True)


def _write_raw(tmp_path, w_nya):
    ldir = tmp_path / "lahman"
    ldir.mkdir()
    pd.DataFrame({"yearID": [2000, 2000], "teamID": ["NYA", "BOS"], "teamIDretro": ["NYA", "BOS"],
                  "franchID": ["NYY", "BOS"], "W": [w_nya, 0]}).to_csv(ldir / "Teams.csv", index=False)
    for n in ["Batting", "Fielding", "Pitching", "Appearances", "People", "Salaries", "Managers"]:
        (ldir / f"{n}.csv").write_text("a\n1\n")
    rdir = tmp_path / "retrosheet"
    rdir.mkdir()
    # NYA 2 wins, BOS 0 wins, one tie
    games = [("NYA", "BOS", 3, 2), ("NYA", "BOS", 1, 1), ("BOS", "NYA", 0, 5)]
    (rdir / "gl2000.txt").write_text(_gl_text(2000, games))


def test_check_wins(tmp_path):
    _write_raw(tmp_path, 2)
    res = crosscheck.check_wins(tmp_path, (2000, 2000))
    assert res["status"] == "ok" and res["n"] == 2
    _write_raw_teams = pd.read_csv(tmp_path / "lahman" / "Teams.csv")
    _write_raw_teams.loc[0, "W"] = 3
    _write_raw_teams.to_csv(tmp_path / "lahman" / "Teams.csv", index=False)
    res = crosscheck.check_wins(tmp_path, (2000, 2000))
    assert res["status"] == "mismatch" and res["mismatches"]["diff"].tolist() == [-1]


def test_checks_skip_when_inputs_missing(tmp_path):
    assert crosscheck.check_wins(tmp_path)["status"] == "skipped"
    assert crosscheck.check_war_totals(tmp_path)["status"] == "skipped"
    assert crosscheck.check_fangraphs_mapping(tmp_path)["status"] == "skipped"


def test_fangraphs_mapping_writes_unmapped(tmp_path):
    fdir, cdir = tmp_path / "fangraphs", tmp_path / "chadwick"
    fdir.mkdir(), cdir.mkdir()
    pd.DataFrame({"IDfg": [1, 2, 3, 4], "Season": 2000, "Name": list("abcd"), "WAR": 1.0}).to_csv(
        fdir / "fg_batting.csv", index=False)
    pd.DataFrame({"IDfg": [9], "Season": 2000, "Name": ["z"], "WAR": 1.0}).to_csv(fdir / "fg_pitching.csv", index=False)
    pd.DataFrame({"key_bbref": list("abc"), "key_fangraphs": [1, 2, 3], "key_retro": list("abc"),
                  "name_first": "x", "name_last": "y"}).to_csv(cdir / "people-0.csv", index=False)
    res = crosscheck.check_fangraphs_mapping(tmp_path, (2000, 2000))
    assert res["status"] == "fail" and res["n_unmapped"] == 2
    assert sorted(pd.read_csv(tmp_path / "unmapped_fangraphs.csv")["idfg"]) == [4, 9]


def test_cli_fetch_exit_codes(tmp_path, monkeypatch, capsys):
    def handler(url):
        if "baseball-reference" in url:
            return FakeResp(status=403)
        if "retrosheet" in url:
            return FakeResp(_gl_zip(int(url[-8:-4]), [("A", "B", 1, 0)]))
        return FakeResp(b"a,b\n1,2\n")
    _patch_get(monkeypatch, handler)
    rc = cli.main(["fetch", "--league", "mlb", "--raw-dir", str(tmp_path), "--only", "bref,chadwick"])
    out = capsys.readouterr().out
    assert rc == 2 and "[FAIL] bref" in out and "[ok]   chadwick" in out
    assert (tmp_path / "MANIFEST.json").exists()
    assert cli.main(["fetch", "--raw-dir", str(tmp_path), "--only", "chadwick"]) == 0
    assert cli.main(["fetch", "--raw-dir", str(tmp_path / "empty"), "--check-only"]) == 0
