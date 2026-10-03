"""AnalysisInputs, player_season_table, synthetic extras (spec sections 1, 8)."""
import numpy as np
import pandas as pd
import pytest

from team_synergy import schema
from team_synergy.analysis.inputs import player_season_table, synthetic_inputs
from team_synergy.build.panel import save_side_tables
from team_synergy.synthetic import make_synthetic_extras, make_synthetic_panel


@pytest.fixture(scope="module")
def inp(syn_inputs):
    return syn_inputs


def test_panel_columns_and_model_columns_unchanged():
    df, truth = make_synthetic_panel(n_teams=12, n_seasons=4, roster=30, seed=0)
    assert list(df.columns) == schema.PANEL_COLUMNS
    assert df["pos"].nunique() <= 12 and set(df["lg"]) == {"AL", "NL"}
    assert (df["age"].between(18, 50)).all() and (df["mlb_exp"] >= 0).all()
    pit = df["group"] == "pit"
    assert df.loc[pit, "role"].isin(["SP", "RP"]).all() and df.loc[~pit, "role"].isna().all()
    assert df.groupby("franch_id")["manager_id"].nunique().min() >= 1
    # covariates come from a separate stream: model columns do not depend on them
    assert df["war"].sum() == pytest.approx(make_synthetic_panel(n_teams=12, n_seasons=4, roster=30, seed=0)[0]["war"].sum())


def test_extras_columns():
    df, truth = make_synthetic_panel(n_teams=12, n_seasons=12, roster=30, seed=0)
    ex = make_synthetic_extras(df, truth, 0)
    assert list(ex["players"].columns) == ["player_id", "name", "birth_year", "debut_season"]
    assert ex["players"]["name"].iloc[0].startswith("Player ")
    assert list(ex["salaries"].columns) == ["player_id", "season", "salary"]
    ps = df[["player_id", "season"]].drop_duplicates()
    assert 0.75 < len(ex["salaries"]) / len(ps) < 0.95 and (ex["salaries"]["salary"] > 0).all()
    pe = ex["pecota"]
    assert list(pe.columns) == ["franch_id", "season", "pecota_w"]
    assert pe["season"].min() == df["season"].min() + 10 and pe["pecota_w"].notna().all()


def test_player_season_table_sums_and_main_stint(inp):
    st, ps = inp.stints_full, inp.player_seasons
    assert len(ps) == st.groupby(["player_id", "season"]).ngroups
    s = st.groupby(["player_id", "season"])[["war", "pc_war", "war_minus"]].sum()
    m = ps.set_index(["player_id", "season"]).loc[s.index]
    for c in s:
        np.testing.assert_allclose(m[c], s[c], atol=1e-9)
    multi = st.groupby(["player_id", "season"]).size()
    multi = multi[multi > 1].index
    assert len(multi) > 0
    top = (st.sort_values("weight", ascending=False).drop_duplicates(["player_id", "season"])
           .set_index(["player_id", "season"]).loc[multi, "franch_id"])
    assert (m.loc[multi, "franch_id"] == top).all()
    assert {"manager_id", "lg", "pos", "age", "mlb_exp", "team_exp"} <= set(ps.columns)


def test_player_season_table_from_panel_covariates():
    df, _ = make_synthetic_panel(n_teams=12, n_seasons=3, roster=30, seed=0)
    stints = df[["player_id", "season", "franch_id", "war", "weight"]].copy()
    stints["own"] = stints["in_deg"] = stints["out_deg"] = 0.0
    ps = player_season_table(stints, df)
    assert "manager_id" in ps and ps["war"].sum() == pytest.approx(df["war"].sum())


def test_inputs_tables_and_side_tables(inp, tmp_path):
    assert inp.war_source == "synthetic" and inp.salaries is not None
    assert {"window_end"} <= set(inp.stints_recursive.columns)
    assert {"manager_id", "pos"} <= set(inp.stints_full.columns)
    save_side_tables({"players": inp.players, "salaries": inp.salaries, "pecota": None}, tmp_path)
    assert (tmp_path / "players.parquet").exists() and not (tmp_path / "pecota.parquet").exists()
