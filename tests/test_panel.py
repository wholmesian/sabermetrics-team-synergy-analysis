"""Tests for build/panel.py on a hand-built fixture (2 teams, 2 seasons).

Fixture follows the io output contract (Lahman original column names); io
loaders are not imported.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from team_synergy.build.panel import build_panel, check_panel
from team_synergy.build.weights import defensive_weights, lineup_weights
from team_synergy.schema import PANEL_COLUMNS

WCFG = yaml.safe_load((Path(__file__).parents[1] / "config" / "weights.yaml").read_text())
CFG = {
    "seasons": {"start": 1998, "end": 1999},
    "eta": {"fwar": {"bat": 0.57, "pit": 0.43}, "bwar": {"bat": 0.59, "pit": 0.41}},
    "fwar_allocation": "kappa",
}
POSCOLS = ["G_p", "G_c", "G_1b", "G_2b", "G_3b", "G_ss", "G_lf", "G_cf", "G_rf", "G_dh"]
IDFG = {"c1": 1, "u1": 2, "d1": 3, "sp1": 4, "rp1": 5, "tr": 6, "b1": 7, "sp2": 8, "rp2": 9, "gh": 10}


def _app(year, team, pid, g_all, **pos):
    row = {"yearID": year, "teamID": team, "playerID": pid, "G_all": g_all}
    row.update({c: 0 for c in POSCOLS})
    row.update(pos)
    return row


def make_data():
    teams = pd.DataFrame({
        "yearID": [1998, 1998, 1999, 1999], "lgID": ["AL", "NL"] * 2, "teamID": ["AAA", "BBB"] * 2,
        "franchID": ["AAA", "BBB"] * 2, "franch_id": ["AAA", "BBB"] * 2,
        "teamIDBR": ["AAA", "BBB"] * 2, "teamIDretro": ["AAA", "BBB"] * 2,
        "W": [90, 72, 85, 77], "G": 162,
    })
    app = []
    for y in (1998, 1999):
        app += [_app(y, "AAA", "c1", 100, G_c=100), _app(y, "BBB", "b1", 120, G_ss=120),
                _app(y, "AAA", "sp1", 30, G_p=30), _app(y, "AAA", "rp1", 60, G_p=60),
                _app(y, "BBB", "sp2", 30, G_p=30), _app(y, "BBB", "rp2", 60, G_p=60)]
    app += [_app(1998, "AAA", "u1", 60, G_1b=20, G_2b=20, G_3b=20),
            _app(1998, "AAA", "d1", 80, G_dh=75, G_1b=5),
            _app(1998, "AAA", "tr", 40, G_ss=40), _app(1998, "BBB", "tr", 60, G_ss=60),
            _app(1998, "AAA", "gh", 2)]
    bat = [("c1", 1998, 1, "AAA", 300, 30, 5, 0, 5), ("c1", 1999, 1, "AAA", 300, 30, 5, 0, 5),
           ("u1", 1998, 1, "AAA", 50, 5, 0, 0, 0), ("u1", 1998, 2, "AAA", 50, 5, 0, 0, 0),
           ("d1", 1998, 1, "AAA", 250, 40, 0, 0, 0),
           ("sp1", 1998, 1, "AAA", 10, 0, 0, 0, 0),
           ("tr", 1998, 1, "AAA", 100, 10, 0, 0, 0), ("tr", 1998, 2, "BBB", 200, 20, 0, 0, 0),
           ("b1", 1998, 1, "BBB", 400, 40, 0, 0, 0), ("b1", 1999, 1, "BBB", 400, 40, 0, 0, 0)]
    batting = pd.DataFrame(bat, columns=["playerID", "yearID", "stint", "teamID", "AB", "BB", "HBP", "SH", "SF"])
    fld = [("c1", "AAA", "C", 2400), ("u1", "AAA", "1B", 300), ("u1", "AAA", "2B", 300), ("u1", "AAA", "3B", 300),
           ("d1", "AAA", "1B", 100), ("tr", "AAA", "SS", 900), ("tr", "BBB", "SS", 1400),
           ("b1", "BBB", "SS", 3000), ("sp1", "AAA", "P", 600), ("c1", "AAA", "OF", 99)]
    rows = []
    for pid, team, pos, io in fld:
        for y in (1998, 1999):
            if y == 1999 and pid not in ("c1", "b1"):
                continue
            rows.append((pid, y, 1, team, pos, io, 10))
    fielding = pd.DataFrame(rows, columns=["playerID", "yearID", "stint", "teamID", "POS", "InnOuts", "G"])
    pit = []
    for y in (1998, 1999):
        pit += [("sp1", y, "AAA", 600, 30, 30), ("rp1", y, "AAA", 200, 60, 0),
                ("sp2", y, "BBB", 540, 30, 30), ("rp2", y, "BBB", 180, 60, 0)]
    pit.append(("u1", 1998, "AAA", 3, 1, 0))
    pitching = pd.DataFrame(pit, columns=["playerID", "yearID", "teamID", "IPouts", "G", "GS"]).assign(stint=1)
    pids = sorted(IDFG)
    people = pd.DataFrame({"playerID": pids, "birthYear": 1970, "bbrefID": [p + "01" for p in pids],
                           "retroID": [p + "r" for p in pids]})
    mgr = pd.DataFrame({"playerID": ["mgrA", "mgrA2", "mgrB"] * 1, "yearID": 1998,
                        "teamID": ["AAA", "AAA", "BBB"], "inseason": [1, 2, 1]})
    mgr = pd.concat([mgr, mgr.assign(yearID=1999)])
    lahman = {"Teams": teams, "Batting": batting, "Fielding": fielding, "Pitching": pitching,
              "Appearances": pd.DataFrame(app), "People": people, "Managers": mgr}
    starts = pd.DataFrame([
        (1998, "AAA", "c1r", 3, 20), (1998, "AAA", "c1r", 4, 80),
        (1998, "AAA", "tr r".replace(" ", ""), 2, 30), (1998, "BBB", "trr", 1, 50),
        (1998, "BBB", "b1r", 1, 120)], columns=["season", "team_retro", "retro_id", "slot", "starts"])
    idmap = pd.DataFrame({"player_id": pids, "bbref_id": [p + "01" for p in pids],
                          "retro_id": [p + "r" for p in pids], "idfg": [IDFG[p] for p in pids]})
    return lahman, starts, idmap


def make_fwar():
    bat, pit = [], []
    for y in (1998, 1999):
        bat += [(1, y, "c1", 4.0), (7, y, "b1", 5.0)]
        pit += [(4, y, "sp1", 3.0), (5, y, "rp1", 1.0), (8, y, "sp2", 2.0), (9, y, "rp2", 0.5)]
    bat += [(2, 1998, "u1", 1.0), (3, 1998, "d1", 0.5), (6, 1998, "tr", 3.0), (4, 1998, "sp1", -0.3)]
    cols = ["idfg", "season", "name", "war"]
    return {"bat": pd.DataFrame(bat, columns=cols), "pit": pd.DataFrame(pit, columns=cols)}


def make_bwar():
    bat = [("c101".replace("c101", "c101"), 1998, "AAA", 1, 4.0, False),
           ("u101", 1998, "AAA", 1, 1.0, False), ("sp101", 1998, "AAA", 1, 0.5, True),
           ("tr01", 1998, "AAA", 1, 1.0, False), ("tr01", 1998, "BBB", 2, 2.0, False)]
    pit = [("sp101", 1998, "AAA", 1, 2.0, True), ("u101", 1998, "AAA", 1, 0.3, False)]
    cols = ["bbref_id", "season", "team_br", "stint", "war", "pitcher_flag"]
    return {"bat": pd.DataFrame(bat, columns=cols), "pit": pd.DataFrame(pit, columns=cols)}


@pytest.fixture(scope="module")
def panel():
    lahman, starts, idmap = make_data()
    return build_panel(lahman, starts, make_fwar(), idmap, CFG, WCFG, "fwar")


def row(panel, pid, season=1998, franch="AAA"):
    r = panel[(panel.player_id == pid) & (panel.season == season) & (panel.franch_id == franch)]
    assert len(r) == 1
    return r.iloc[0]


def test_columns_and_exclusion(panel):
    assert list(panel.columns) == PANEL_COLUMNS
    assert "gh" not in set(panel.player_id)  # no PA, no outs
    assert len(panel[(panel.player_id == "tr")]) == 2
    assert row(panel, "u1").pa == 110  # two stints on the same team aggregated


def test_weight_sums_to_one(panel):
    s = panel.groupby(["season", "franch_id"])["weight"].sum()
    assert len(s) == 4
    assert s.tolist() == pytest.approx([1.0] * 4)
    tau = panel.groupby(["season", "franch_id", "group"])["tau"].sum()
    assert tau.tolist() == pytest.approx([1.0] * len(tau))


def test_position_rules(panel):
    assert row(panel, "c1").pos == "C"
    assert row(panel, "u1").pos == "UT"
    assert row(panel, "d1").pos == "DH"
    assert row(panel, "sp1").pos == "SP" and row(panel, "sp1").group == "pit"
    assert row(panel, "rp1").pos == "RP" and row(panel, "rp1").role == "RP"
    assert row(panel, "tr", franch="BBB").pos == "SS"
    assert pd.isna(row(panel, "c1").role)


def test_l_d_kappa(panel):
    b = lineup_weights(WCFG)
    p = defensive_weights(WCFG, "fwar")
    assert row(panel, "u1").l == pytest.approx(b.mean())  # no starts
    assert row(panel, "c1").l == pytest.approx(0.2 * b[2] + 0.8 * b[3])
    assert row(panel, "c1").lineup_slot_mean == pytest.approx(3.8)
    assert np.isnan(row(panel, "u1").lineup_slot_mean)
    assert row(panel, "d1").d == pytest.approx(5 / 80 * 0 + p["1B"])  # only 1B field games
    sp = row(panel, "sp1")
    assert sp.pa == 10 and sp.d == 1.0
    assert sp.kappa == pytest.approx(600 / (27 * 162))  # pitcher: batting PA ignored
    c = row(panel, "c1")
    # Lahman Fielding reports outfield play only as the aggregate "OF" row; it counts toward DOuts.
    assert c.douts == 2499
    assert c.kappa == pytest.approx(c.l * 340 / 486 + p["C"] * 2499 / 4374)


def test_fwar_allocation_preserves_totals(panel):
    tr = panel[panel.player_id == "tr"]
    assert tr.war.sum() == pytest.approx(3.0)
    assert tr.set_index("franch_id").war["AAA"] == pytest.approx(3.0 * tr.set_index("franch_id").kappa["AAA"] / tr.kappa.sum())
    # pitcher batting fWAR (-0.3) dropped; position-player totals preserved
    assert row(panel, "sp1").war == pytest.approx(3.0)
    tot = panel[panel.season == 1998].war.sum()
    assert tot == pytest.approx(4 + 1 + 0.5 + 3 + 5 + 3 + 1 + 2 + 0.5)


def test_covariates(panel):
    assert row(panel, "c1", 1999).mlb_exp == 100 and row(panel, "c1", 1999).team_exp == 100
    assert row(panel, "c1", 1998).mlb_exp == 0
    assert row(panel, "c1").age == 28
    assert row(panel, "c1").manager_id == "mgrA"
    assert row(panel, "c1").team_wins == 90 and row(panel, "c1").lg == "AL"


def test_bwar_per_stint_and_pitcher_batting_dropped():
    lahman, starts, idmap = make_data()
    pn = build_panel(lahman, starts, make_bwar(), idmap, CFG, WCFG, "bwar")
    assert row(pn, "sp1").war == 2.0  # pitching WAR only
    assert row(pn, "u1").war == 1.0   # batting WAR only (pitching 0.3 dropped)
    assert row(pn, "tr", franch="AAA").war == 1.0 and row(pn, "tr", franch="BBB").war == 2.0
    assert row(pn, "c1").eta == 0.59


def test_check_panel(panel):
    rep = check_panel(panel, CFG, strict=False, n_teams=2)
    assert rep["n_team_seasons"] == 4
    assert rep["weight_sum_max_dev"] < 1e-9
    assert rep["id_map_fail_rate"] == 0.0
    assert any("league WAR" in m for m in rep["problems"])  # fixture WAR total (~20) is not ~67
    with pytest.raises(AssertionError, match="league WAR"):
        check_panel(panel, CFG, n_teams=2)
    assert check_panel(panel, CFG, n_teams=2, tol_war=1.0)["ok"]
    with pytest.raises(AssertionError, match="team-seasons"):
        check_panel(panel, CFG, strict=True, n_teams=30, tol_war=1e9)
