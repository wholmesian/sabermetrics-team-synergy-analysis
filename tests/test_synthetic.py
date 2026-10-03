import numpy as np

from team_synergy import schema
from team_synergy.synthetic import make_synthetic_panel


def test_panel_shape_columns_weights():
    df, truth = make_synthetic_panel(n_teams=30, n_seasons=10, seed=0)
    assert set(schema.MODEL_COLUMNS + ["lineup_slot_mean"]) <= set(df.columns)
    assert df.groupby(["franch_id", "season"]).ngroups == 300
    assert len(truth["y_true"]) == len(df)
    w = df.groupby(["franch_id", "season"])["weight"].sum()
    assert np.abs(w - 1).max() < 1e-12
    assert (df["kappa"] > 0).all()


def test_traded_share_and_determinism():
    df, _ = make_synthetic_panel(seed=0, trade_frac=0.12)
    n = df.groupby(["player_id", "season"]).size()
    assert abs((n > 1).mean() - 0.12) < 0.03
    df2, _ = make_synthetic_panel(seed=0, trade_frac=0.12)
    assert df.equals(df2)
