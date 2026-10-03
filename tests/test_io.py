"""Tests for team_synergy.io loaders (spec §1).

Test that loaders:
  1. Check for required files and raise MissingRawDataError with helpful message
  2. Load sample data correctly
  3. Return DataFrames with expected column names and types
  4. Filter to the specified season range
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from team_synergy.io import (
    MissingRawDataError,
    load_lahman,
    load_gamelogs,
    lineup_starts,
    load_bwar,
    load_fwar,
    load_register,
    build_idmap,
    mapping_failure_rate,
    load_pecota,
)


class TestMissingRawDataError:
    """Test that MissingRawDataError is properly defined."""

    def test_error_is_file_not_found(self):
        """MissingRawDataError should be a FileNotFoundError."""
        err = MissingRawDataError("test message")
        assert isinstance(err, FileNotFoundError)
        assert str(err) == "test message"


class TestLoadLahman:
    """Test load_lahman() with fake data."""

    def test_lahman_missing_files(self, tmp_path):
        """Should raise MissingRawDataError when files are missing."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        lahman_dir = raw_dir / "lahman"
        lahman_dir.mkdir()

        with pytest.raises(MissingRawDataError) as exc_info:
            load_lahman(raw_dir)

        err_msg = str(exc_info.value)
        assert "Teams" in err_msg
        assert "data/raw/lahman" in err_msg
        assert "CSV" in err_msg

    def test_lahman_loads_valid_files(self, tmp_path):
        """Should load all Lahman tables when files exist."""
        raw_dir = tmp_path / "raw"
        lahman_dir = raw_dir / "lahman"
        lahman_dir.mkdir(parents=True)

        # Create dummy CSV files
        teams_df = pd.DataFrame({
            "franchID": ["NYY", "BOS"],
            "yearID": [2000, 2000],
            "W": [87, 85],
        })
        teams_df.to_csv(lahman_dir / "Teams.csv", index=False)

        for fname in ["Batting", "Fielding", "Pitching", "Appearances", "People", "Salaries", "Managers"]:
            pd.DataFrame({"col1": [1]}).to_csv(lahman_dir / f"{fname}.csv", index=False)

        result = load_lahman(raw_dir)

        assert isinstance(result, dict)
        assert "Teams" in result
        assert "Batting" in result
        assert "franch_id" in result["Teams"].columns
        assert result["Teams"]["franch_id"].iloc[0] == "NYY"

    def test_lahman_franchise_mapping(self, tmp_path):
        """Should apply FRANCHISE_MAP to canonical franchise codes."""
        raw_dir = tmp_path / "raw"
        lahman_dir = raw_dir / "lahman"
        lahman_dir.mkdir(parents=True)

        # Create Teams.csv with historical franchise IDs
        teams_df = pd.DataFrame({
            "franchID": ["MON", "FLA", "ANA", "TBD"],
            "yearID": [2000, 2000, 2000, 2000],
            "W": [70, 80, 90, 100],
        })
        teams_df.to_csv(lahman_dir / "Teams.csv", index=False)

        for fname in ["Batting", "Fielding", "Pitching", "Appearances", "People", "Salaries", "Managers"]:
            pd.DataFrame({"col1": [1]}).to_csv(lahman_dir / f"{fname}.csv", index=False)

        result = load_lahman(raw_dir)
        teams = result["Teams"]

        # Check mapping
        assert teams.loc[teams["franchID"] == "MON", "franch_id"].iloc[0] == "WSN"
        assert teams.loc[teams["franchID"] == "FLA", "franch_id"].iloc[0] == "MIA"
        assert teams.loc[teams["franchID"] == "ANA", "franch_id"].iloc[0] == "LAA"
        assert teams.loc[teams["franchID"] == "TBD", "franch_id"].iloc[0] == "TBR"


class TestLoadGamelogs:
    """Test load_gamelogs() and lineup_starts()."""

    def test_gamelogs_missing_files(self, tmp_path):
        """Should raise MissingRawDataError when game log files are missing."""
        raw_dir = tmp_path / "raw"
        retrosheet_dir = raw_dir / "retrosheet"
        retrosheet_dir.mkdir(parents=True)

        with pytest.raises(MissingRawDataError) as exc_info:
            load_gamelogs(raw_dir, seasons=(1998, 1999))

        err_msg = str(exc_info.value)
        assert "retrosheet" in err_msg.lower()
        assert "Retrosheet" in err_msg

    def test_gamelogs_loads_valid_file(self, tmp_path):
        """Should load game log and parse batting lineups."""
        raw_dir = tmp_path / "raw"
        retrosheet_dir = raw_dir / "retrosheet"
        retrosheet_dir.mkdir(parents=True)

        # Create a minimal 161-field game log line
        # Fields: 0=date, 1=visit_team, 2=home_team, 3=season, 4=game_num, rest=data
        fields = ["20000101", "NYA", "BOS", "2000", "1"]  # date, visit, home, season, game_num
        # Add dummy fields to reach position 105 (for visitor lineup start)
        fields.extend(["0"] * 100)
        # Visitor lineup (slots 1-9, starting at field 105 [0-based])
        for slot in range(9):
            fields.extend([f"player{slot+1}", f"name{slot+1}", "3"])  # id, name, position
        # Home lineup (slots 1-9)
        for slot in range(9):
            fields.extend([f"phome{slot+1}", f"hname{slot+1}", "3"])

        # Pad to exactly 161 fields
        while len(fields) < 161:
            fields.append("0")
        fields = fields[:161]

        # Write the game log
        game_line = ",".join(fields)
        with open(retrosheet_dir / "gl2000.txt", "w") as f:
            f.write(game_line + "\n")

        result = load_gamelogs(raw_dir, seasons=(2000, 2000))

        assert isinstance(result, pd.DataFrame)
        assert all(col in result.columns for col in ["season", "team_retro", "slot", "retro_id", "def_pos"])
        assert len(result) > 0
        assert set(result["side"].unique()).issubset({"away", "home"})
        assert result["slot"].min() >= 1
        assert result["slot"].max() <= 9

    def test_lineup_starts_aggregation(self, tmp_path):
        """Should aggregate game logs to count starting appearances."""
        raw_dir = tmp_path / "raw"
        retrosheet_dir = raw_dir / "retrosheet"
        retrosheet_dir.mkdir(parents=True)

        # Create a game log with two games
        for game_num in [1, 2]:
            fields = ["2000-01-01"] + ["NYA", "BOS"] + ["0"] * 5
            fields.extend(["0"] * 98)
            for slot in range(9):
                fields.extend(["player1", "name1", "3"])  # Same player in slot 1
            for slot in range(9):
                fields.extend(["phome1", "hname1", "3"])
            while len(fields) < 161:
                fields.append("0")
            fields = fields[:161]

            with open(retrosheet_dir / f"gl2000.txt", "a") as f:
                f.write(",".join(fields) + "\n")

        gamelogs = load_gamelogs(raw_dir, seasons=(2000, 2000))
        starts = lineup_starts(gamelogs)

        assert isinstance(starts, pd.DataFrame)
        assert all(col in starts.columns for col in ["season", "team_retro", "retro_id", "slot", "starts"])
        # Player should have 2 starts in slot 1 (one as visitor, one as home)
        assert starts["starts"].sum() > 0


class TestLoadBWAR:
    """Test load_bwar() and season filtering."""

    def test_bwar_missing_files(self, tmp_path):
        """Should raise MissingRawDataError when files are missing."""
        raw_dir = tmp_path / "raw"
        bref_dir = raw_dir / "bref"
        bref_dir.mkdir(parents=True)

        with pytest.raises(MissingRawDataError) as exc_info:
            load_bwar(raw_dir)

        err_msg = str(exc_info.value)
        assert "war_daily_bat.txt" in err_msg or "Baseball-Reference" in err_msg

    def test_bwar_loads_valid_files(self, tmp_path):
        """Should load bWAR data when files exist."""
        raw_dir = tmp_path / "raw"
        bref_dir = raw_dir / "bref"
        bref_dir.mkdir(parents=True)

        # Create dummy bWAR files
        bat_df = pd.DataFrame({
            "player_ID": ["abreubo01", "altuvej01"],
            "year_ID": [2010, 2010],
            "team_ID": ["CHA", "HOU"],
            "stint_ID": [1, 1],
            "lg_ID": ["AL", "AL"],
            "WAR": [2.5, 3.1],
        })
        pit_df = pd.DataFrame({
            "player_ID": ["verlaju01"],
            "year_ID": [2010],
            "team_ID": ["DET"],
            "stint_ID": [1],
            "lg_ID": ["AL"],
            "WAR": [5.4],
        })

        bat_df.to_csv(bref_dir / "war_daily_bat.txt", index=False)
        pit_df.to_csv(bref_dir / "war_daily_pitch.txt", index=False)

        result = load_bwar(raw_dir, seasons=(2010, 2010))

        assert isinstance(result, dict)
        assert "bat" in result and "pit" in result
        bat, pit = result["bat"], result["pit"]

        assert all(col in bat.columns for col in ["bbref_id", "season", "team_br", "stint", "war", "pitcher_flag"])
        assert all(col in pit.columns for col in ["bbref_id", "season", "team_br", "stint", "war", "pitcher_flag"])
        assert not bat["pitcher_flag"].any()
        assert pit["pitcher_flag"].all()
        assert len(bat) == 2
        assert len(pit) == 1


class TestLoadFWAR:
    """Test load_fwar()."""

    def test_fwar_missing_files(self, tmp_path):
        """Should raise MissingRawDataError when files are missing."""
        raw_dir = tmp_path / "raw"
        fg_dir = raw_dir / "fangraphs"
        fg_dir.mkdir(parents=True)

        with pytest.raises(MissingRawDataError) as exc_info:
            load_fwar(raw_dir)

        err_msg = str(exc_info.value)
        assert "FanGraphs" in err_msg or "fg_batting" in err_msg

    def test_fwar_loads_valid_files(self, tmp_path):
        """Should load fWAR data when files exist."""
        raw_dir = tmp_path / "raw"
        fg_dir = raw_dir / "fangraphs"
        fg_dir.mkdir(parents=True)

        # Create dummy FanGraphs files
        bat_df = pd.DataFrame({
            "IDfg": [123, 456],
            "Season": [2010, 2010],
            "Name": ["Jose Altuve", "Jose Abreu"],
            "WAR": [3.1, 2.5],
        })
        pit_df = pd.DataFrame({
            "IDfg": [789],
            "Season": [2010],
            "Name": ["Justin Verlander"],
            "WAR": [5.4],
        })

        bat_df.to_csv(fg_dir / "fg_batting.csv", index=False)
        pit_df.to_csv(fg_dir / "fg_pitching.csv", index=False)

        result = load_fwar(raw_dir, seasons=(2010, 2010))

        assert isinstance(result, dict)
        assert "bat" in result and "pit" in result
        bat, pit = result["bat"], result["pit"]

        assert all(col in bat.columns for col in ["idfg", "season", "name", "war"])
        assert len(bat) == 2
        assert len(pit) == 1


class TestLoadRegister:
    """Test load_register() and build_idmap()."""

    def test_register_missing_file(self, tmp_path):
        """Should raise MissingRawDataError when register file is missing."""
        raw_dir = tmp_path / "raw"
        chadwick_dir = raw_dir / "chadwick"
        chadwick_dir.mkdir(parents=True)

        with pytest.raises(MissingRawDataError) as exc_info:
            load_register(raw_dir)

        err_msg = str(exc_info.value)
        assert "Chadwick" in err_msg or "people.csv" in err_msg

    def test_register_loads_single_file(self, tmp_path):
        """Should load a single people.csv file."""
        raw_dir = tmp_path / "raw"
        chadwick_dir = raw_dir / "chadwick"
        chadwick_dir.mkdir(parents=True)

        register_df = pd.DataFrame({
            "key_bbref": ["abreubo01", "altuvej01"],
            "key_fangraphs": [123, 456],
            "key_retro": ["abreubo01", "altuvej01"],
            "key_mlbam": ["mlbam1", "mlbam2"],
            "name_first": ["Jose", "Jose"],
            "name_last": ["Abreu", "Altuve"],
        })
        register_df.to_csv(chadwick_dir / "people.csv", index=False)

        result = load_register(raw_dir)

        assert isinstance(result, pd.DataFrame)
        assert all(col in result.columns for col in ["key_bbref", "key_fangraphs", "key_retro", "name_first", "name_last"])
        assert len(result) == 2

    def test_build_idmap(self, tmp_path):
        """Should join Lahman People to Chadwick register."""
        # Create Lahman People data
        people = pd.DataFrame({
            "playerID": ["abreubo01", "altuvej01"],
            "bbrefID": ["abreubo01", "altuvej01"],
            "retroID": ["abreubo01", "altuvej01"],
        })

        # Create Chadwick register
        register = pd.DataFrame({
            "key_bbref": ["abreubo01", "altuvej01"],
            "key_fangraphs": [123, 456],
            "key_retro": ["abreubo01", "altuvej01"],
            "name_first": ["Jose", "Jose"],
            "name_last": ["Abreu", "Altuve"],
        })

        result = build_idmap(people, register)

        assert isinstance(result, pd.DataFrame)
        assert all(col in result.columns for col in ["player_id", "bbref_id", "retro_id", "idfg"])
        assert len(result) == 2
        assert result["idfg"].dtype == "Int64"  # Nullable integer

    def test_mapping_failure_rate(self):
        """Should compute fraction of missing FanGraphs IDs."""
        idmap = pd.DataFrame({
            "player_id": ["p1", "p2", "p3"],
            "bbref_id": ["b1", "b2", "b3"],
            "retro_id": ["r1", "r2", "r3"],
            "idfg": [123, None, 456],  # One missing
        })

        rate = mapping_failure_rate(idmap)
        assert rate == pytest.approx(1/3, rel=0.01)


class TestLoadPECOTA:
    """Test load_pecota()."""

    def test_pecota_missing_file_returns_none(self, tmp_path):
        """Should return None if PECOTA file is missing."""
        raw_dir = tmp_path / "raw"

        result = load_pecota(raw_dir)
        assert result is None

    def test_pecota_loads_valid_file(self, tmp_path):
        """Should load PECOTA file when it exists."""
        raw_dir = tmp_path / "raw"
        pecota_dir = raw_dir / "pecota"
        pecota_dir.mkdir(parents=True)

        pecota_df = pd.DataFrame({
            "franch_id": ["NYY", "BOS"],
            "season": [2010, 2010],
            "pecota_w": [97, 89],
        })
        pecota_df.to_csv(pecota_dir / "pecota.csv", index=False)

        result = load_pecota(raw_dir)

        assert isinstance(result, pd.DataFrame)
        assert all(col in result.columns for col in ["franch_id", "season", "pecota_w"])
        assert len(result) == 2


class TestSeasonFiltering:
    """Test that loaders correctly filter by season range."""

    def test_bwar_season_filter(self, tmp_path):
        """Should filter bWAR to specified season range."""
        raw_dir = tmp_path / "raw"
        bref_dir = raw_dir / "bref"
        bref_dir.mkdir(parents=True)

        bat_df = pd.DataFrame({
            "player_ID": ["p1", "p2", "p3"],
            "year_ID": [1998, 2000, 2016],
            "team_ID": ["NYY", "BOS", "NYY"],
            "stint_ID": [1, 1, 1],
            "lg_ID": ["AL", "AL", "AL"],
            "WAR": [2.0, 3.0, 4.0],
        })
        pit_df = pd.DataFrame({
            "player_ID": ["p"],
            "year_ID": [2000],
            "team_ID": ["NYY"],
            "stint_ID": [1],
            "lg_ID": ["AL"],
            "WAR": [5.0],
        })

        bat_df.to_csv(bref_dir / "war_daily_bat.txt", index=False)
        pit_df.to_csv(bref_dir / "war_daily_pitch.txt", index=False)

        result = load_bwar(raw_dir, seasons=(2000, 2010))

        assert result["bat"]["season"].min() == 2000
        assert result["bat"]["season"].max() == 2000
        assert 1998 not in result["bat"]["season"].values
        assert 2016 not in result["bat"]["season"].values

    def test_fwar_season_filter(self, tmp_path):
        """Should filter fWAR to specified season range."""
        raw_dir = tmp_path / "raw"
        fg_dir = raw_dir / "fangraphs"
        fg_dir.mkdir(parents=True)

        bat_df = pd.DataFrame({
            "IDfg": [1, 2, 3],
            "Season": [1998, 2010, 2016],
            "Name": ["A", "B", "C"],
            "WAR": [1.0, 2.0, 3.0],
        })
        pit_df = pd.DataFrame({
            "IDfg": [4],
            "Season": [2005],
            "Name": ["D"],
            "WAR": [4.0],
        })

        bat_df.to_csv(fg_dir / "fg_batting.csv", index=False)
        pit_df.to_csv(fg_dir / "fg_pitching.csv", index=False)

        result = load_fwar(raw_dir, seasons=(2000, 2010))

        # Should only have 2010 from batting (1998 and 2016 are outside range)
        assert result["bat"]["season"].min() == 2010
        assert result["bat"]["season"].max() == 2010
        # Pitching should have 2005
        assert result["pit"]["season"].min() == 2005
        assert result["pit"]["season"].max() == 2005


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_bwar_ignores_auxiliary_war_columns(tmp_path):
    """Real war_daily files carry WAR_off/WAR_def/WAR_rep and team* columns besides WAR."""
    from team_synergy.io.war_bref import load_bwar

    d = tmp_path / "bref"
    d.mkdir()
    header = "name_common,player_ID,year_ID,team_ID,stint_ID,lg_ID,teamRpG,oppRpG_rep,WAR_off,WAR_def,WAR_rep,WAR,pitcher\n"
    (d / "war_daily_bat.txt").write_text(header + "A B,abc01,2000,NYY,1,AL,5.1,4.9,2.0,0.5,1.8,3.2,N\n")
    (d / "war_daily_pitch.txt").write_text(header + "C D,cde01,2000,NYY,1,AL,5.1,4.9,NULL,NULL,1.0,1.5,Y\n")
    out = load_bwar(tmp_path, seasons=(1998, 2016))
    assert list(out["bat"]["war"]) == [3.2]
    assert list(out["pit"]["war"]) == [1.5]
    assert out["bat"].columns.is_unique
