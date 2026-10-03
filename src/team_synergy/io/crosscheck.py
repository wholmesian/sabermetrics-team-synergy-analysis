"""Pre-build cross-checks of the raw sources (run at the end of ``synergy fetch``).

(a) Lahman Teams.W vs wins tallied from Retrosheet game logs (per team-season),
(b) per-season WAR totals of bWAR and fWAR (bat + pit), flagged outside 1000 +/- 5 %,
(c) FanGraphs IDfg -> Chadwick key_fangraphs mapping failure rate (unmapped rows written to
    data/raw/unmapped_fangraphs.csv when > 1 %).

Each check is skipped (not failed) when its inputs are missing. These are diagnostics only; the
authoritative integrity checks remain those of ``synergy build``.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import MissingRawDataError, load_bwar, load_fwar, load_lahman, load_register

logger = logging.getLogger(__name__)

WAR_TARGET, WAR_TOL = 1000.0, 0.05
MAP_FAIL_MAX = 0.01


def retrosheet_wins(raw_dir: Path, seasons: tuple[int, int]) -> pd.DataFrame:
    """Wins per (season, Retrosheet team code) from gl{YYYY}.txt. Ties/no-decisions count for nobody.

    Game-log fields (0-based): 0 date, 3 visiting team, 6 home team, 9 visiting score, 10 home score.
    """
    rows = []
    rdir = Path(raw_dir) / "retrosheet"
    for y in range(seasons[0], seasons[1] + 1):
        path = next((p for p in (rdir / f"gl{y}.txt", rdir / f"GL{y}.TXT") if p.exists()), None)
        if path is None:
            continue
        gl = pd.read_csv(path, header=None, usecols=[3, 6, 9, 10], names=["v", "h", "vs", "hs"],
                         dtype={"v": str, "h": str})
        win = gl["vs"] > gl["hs"]
        hwin = gl["hs"] > gl["vs"]
        w = pd.concat([gl.loc[win, "v"], gl.loc[hwin, "h"]]).value_counts()
        rows.append(pd.DataFrame({"season": y, "team_retro": w.index, "w_retro": w.values}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["season", "team_retro", "w_retro"])


def check_wins(raw_dir: Path, seasons=(1998, 2016)) -> dict:
    """(a) Lahman Teams.W vs Retrosheet-tallied wins. Returns {'status','n','mismatches': DataFrame}."""
    try:
        teams = load_lahman(raw_dir)["Teams"]
    except MissingRawDataError:
        return {"status": "skipped", "reason": "Lahman tables missing"}
    rw = retrosheet_wins(raw_dir, seasons)
    if rw.empty:
        return {"status": "skipped", "reason": "Retrosheet game logs missing"}
    t = teams[(teams["yearID"] >= seasons[0]) & (teams["yearID"] <= seasons[1])].copy()
    code = "teamIDretro" if "teamIDretro" in t.columns else "teamID"
    t = t.rename(columns={"yearID": "season", code: "team_retro"})[["season", "team_retro", "W"]]
    got_seasons = set(rw["season"])
    t = t[t["season"].isin(got_seasons)]
    m = t.merge(rw, on=["season", "team_retro"], how="left")
    m["w_retro"] = m["w_retro"].fillna(0).astype(int)
    bad = m[m["W"] != m["w_retro"]].copy()
    bad["diff"] = bad["w_retro"] - bad["W"]
    return {"status": "ok" if bad.empty else "mismatch", "n": len(m), "mismatches": bad}


def check_war_totals(raw_dir: Path, seasons=(1998, 2016)) -> dict:
    """(b) Per-season WAR sums for bWAR and fWAR. Returns {'table': DataFrame, 'flagged': [...]}."""
    cols, skipped = {}, []
    for name, loader in (("bWAR", load_bwar), ("fWAR", load_fwar)):
        try:
            d = loader(raw_dir, seasons)
        except MissingRawDataError:
            skipped.append(name)
            continue
        cols[name] = pd.concat([d["bat"], d["pit"]]).groupby("season")["war"].sum()
    if not cols:
        return {"status": "skipped", "reason": "no WAR files", "skipped": skipped}
    table = pd.DataFrame(cols)
    lo, hi = WAR_TARGET * (1 - WAR_TOL), WAR_TARGET * (1 + WAR_TOL)
    flagged = [(int(s), c, float(v)) for c in table for s, v in table[c].items() if not lo <= v <= hi]
    return {"status": "ok" if not flagged else "flagged", "table": table, "flagged": flagged, "skipped": skipped}


def check_fangraphs_mapping(raw_dir: Path, seasons=(1998, 2016)) -> dict:
    """(c) Share of distinct fWAR IDfg without a Chadwick key_fangraphs match; writes unmapped rows if > 1 %."""
    try:
        fw = load_fwar(raw_dir, seasons)
        reg = load_register(raw_dir)
    except MissingRawDataError:
        return {"status": "skipped", "reason": "fWAR or Chadwick register missing"}
    fg = pd.concat([fw["bat"], fw["pit"]])
    fg = fg[fg["idfg"].notna()]
    known = set(reg["key_fangraphs"].dropna().astype(int))
    ids = fg["idfg"].astype(int).drop_duplicates()
    unmapped_ids = set(ids) - known
    rate = len(unmapped_ids) / max(len(ids), 1)
    out = {"status": "ok" if rate <= MAP_FAIL_MAX else "fail", "rate": rate,
           "n_ids": len(ids), "n_unmapped": len(unmapped_ids)}
    if rate > MAP_FAIL_MAX:
        rows = fg[fg["idfg"].astype(int).isin(unmapped_ids)]
        path = Path(raw_dir) / "unmapped_fangraphs.csv"
        rows.to_csv(path, index=False)
        out["path"] = str(path)
    return out


def run_checks(raw_dir: Path, seasons=(1998, 2016)) -> bool:
    """Print all three checks; return True unless a check ran and failed (skips do not fail)."""
    ok = True
    print("\n== cross-check (a): Lahman Teams.W vs Retrosheet wins ==")
    a = check_wins(raw_dir, seasons)
    if a["status"] == "skipped":
        print(f"skipped: {a['reason']}")
    else:
        print(f"{a['n']} team-seasons compared, {len(a.get('mismatches', []))} mismatches")
        if a["status"] == "mismatch":
            ok = False
            print(a["mismatches"].head(40).to_string(index=False))
    print("\n== cross-check (b): per-season WAR totals (target 1000 +/- 5%) ==")
    b = check_war_totals(raw_dir, seasons)
    if b["status"] == "skipped":
        print(f"skipped: {b['reason']}")
    else:
        print(b["table"].round(1).to_string())
        for name in b["skipped"]:
            print(f"{name}: skipped (files missing)")
        for s, c, v in b["flagged"]:
            print(f"FLAG {c} {s}: {v:.1f} outside [{WAR_TARGET*(1-WAR_TOL):.0f}, {WAR_TARGET*(1+WAR_TOL):.0f}]")
        ok &= not b["flagged"]
    print("\n== cross-check (c): fWAR IDfg -> Chadwick key_fangraphs ==")
    c = check_fangraphs_mapping(raw_dir, seasons)
    if c["status"] == "skipped":
        print(f"skipped: {c['reason']}")
    else:
        print(f"unmapped {c['n_unmapped']}/{c['n_ids']} IDfg = {c['rate']:.2%} (limit {MAP_FAIL_MAX:.0%})")
        if c["status"] == "fail":
            ok = False
            print(f"unmapped rows written to {c['path']}")
    return ok
