"""Tests for build/weights.py (paper eqs. 5-7, Table 2; spec §2)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from team_synergy.build.weights import (
    compute_kappa,
    compute_tau_eta,
    defensive_weight_d,
    defensive_weights,
    fill_l,
    lineup_weight_l,
    lineup_weights,
)
from team_synergy.schema import FIELD_POSITIONS

W = yaml.safe_load((Path(__file__).parents[1] / "config" / "weights.yaml").read_text())


def test_table2_weights_sum_to_one():
    for src in ("fwar", "bwar"):
        p = defensive_weights(W, src)
        assert sum(p[k] for k in FIELD_POSITIONS) == pytest.approx(1.0, abs=2e-3)
        assert p["DH"] == 0.0
    assert lineup_weights(W).sum() == pytest.approx(1.0, abs=2e-3)


def test_derived_p_reproduces_table2_fwar():
    derived = defensive_weights({**W, "weight_source": "derived"}, "fwar")
    table2 = defensive_weights(W, "fwar")
    assert derived["C"] == pytest.approx(30 / 140)
    for pos in FIELD_POSITIONS:
        assert derived[pos] == pytest.approx(table2[pos], abs=1e-3)
    mx = defensive_weights({**W, "weight_source": "derived", "normalize": "max"}, "fwar")
    assert mx["C"] == pytest.approx(1.0)


def test_derived_b_not_implemented():
    with pytest.raises(NotImplementedError):
        lineup_weights({**W, "weight_source": "derived"})


def test_l_from_starts_and_no_start_fallback():
    b = lineup_weights(W)
    starts = pd.DataFrame({
        "season": 1998, "team_retro": "AAA", "retro_id": ["x", "x", "y"],
        "slot": [1, 2, 9], "starts": [30, 10, 5],
    })
    out = lineup_weight_l(starts, b).set_index("retro_id")
    assert out.loc["x", "l"] == pytest.approx((30 * b[0] + 10 * b[1]) / 40)
    assert out.loc["x", "lineup_slot_mean"] == pytest.approx((30 * 1 + 10 * 2) / 40)
    assert out.loc["y", "l"] == pytest.approx(b[8])
    filled = fill_l(pd.Series([np.nan, 0.1]), b)
    assert filled.iloc[0] == pytest.approx(b.mean())
    assert filled.iloc[1] == 0.1


def test_defensive_weight_d():
    p = defensive_weights(W, "fwar")
    app = pd.DataFrame({"G_c": [50, 0, 0], "G_1b": [50, 0, 0], "G_2b": 0, "G_3b": 0, "G_ss": 0,
                        "G_lf": 0, "G_cf": 0, "G_rf": 0, "G_dh": [0, 100, 0]})
    d = defensive_weight_d(app, p)
    assert d.iloc[0] == pytest.approx(0.5 * p["C"] + 0.5 * p["1B"])
    assert d.iloc[1] == 0.0  # DH only (weight 0)
    assert d.iloc[2] == 0.0  # no field games


def test_kappa_hand_example():
    df = pd.DataFrame({
        "group": ["bat", "pit"], "l": [0.2, np.nan], "d": [0.5, 1.0],
        "pa": [486, 0], "douts": [2187, 0], "pouts": [0, 2187],
    })
    k = compute_kappa(df)
    assert k.iloc[0] == pytest.approx(0.2 * 486 / 486 + 0.5 * 2187 / 4374)  # 0.45
    assert k.iloc[0] == pytest.approx(0.45)
    assert k.iloc[1] == pytest.approx(0.5)


def test_tau_sums_to_one_in_group_and_weights_to_one():
    df = pd.DataFrame({
        "season": 1998, "franch_id": ["A", "A", "A", "A", "B", "B"],
        "group": ["bat", "bat", "pit", "pit", "bat", "pit"],
        "kappa": [1.0, 3.0, 2.0, 2.0, 5.0, 1.0],
    })
    out = compute_tau_eta(df, {"bat": 0.57, "pit": 0.43})
    full = pd.concat([df, out], axis=1)
    assert full.groupby(["franch_id", "group"])["tau"].sum().tolist() == pytest.approx([1] * 4)
    assert full.groupby("franch_id")["weight"].sum().tolist() == pytest.approx([1, 1])
    assert out["tau"].iloc[0] == pytest.approx(0.25)
