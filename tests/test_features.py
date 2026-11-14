"""M1 step 2: annotation checks — normalization, no lookahead, sanity on known regimes."""
import numpy as np
import pandas as pd
import pytest

from almanac import config
from almanac.data import load_all
from almanac.features import VARS, annotate, compute_zstats, raw_features

ZCOLS = [f"z_{v}" for v in VARS]


@pytest.fixture(scope="module")
def frames():
    return load_all()


@pytest.fixture(scope="module")
def ann(frames):
    return annotate(frames)


def test_config_zstats_match_learning_period(frames):
    assert compute_zstats(raw_features(frames)) == config.ZSTATS


def test_zscores_normalized_in_learning_period(ann):
    learn = ann.loc[config.LEARN_START:config.LEARN_END, ZCOLS]
    assert np.allclose(learn.mean(), 0, atol=1e-3)
    assert np.allclose(learn.std(), 1, atol=1e-3)


def test_full_coverage_from_learning_start(ann):
    assert ann.index[0] < pd.Timestamp(config.LEARN_START)
    assert not ann.loc[config.LEARN_START:, ZCOLS + ["rsi2"]].isna().any().any()


@pytest.mark.parametrize("cut", ["2008-10-10", "2011-08-08", "2015-08-24", "2020-03-16", "2023-05-31"])
def test_no_lookahead(frames, ann, cut):
    """Annotating with data cut at `cut` must give identical values for that day."""
    truncated = {s: df.loc[:cut] for s, df in frames.items()}
    day = annotate(truncated).iloc[-1]
    assert day.name == pd.Timestamp(cut)
    full = ann.loc[cut]
    for c in ZCOLS + ["rsi2"]:
        assert day[c] == pytest.approx(full[c], rel=1e-9), c
    assert day["above200"] == full["above200"]


def test_crisis_days_look_like_crises(ann):
    for d in ["2008-10-10", "2020-03-16"]:
        row = ann.loc[d]
        assert row["z_vol"] > 2 and row["z_vix"] > 2, d
        assert row["z_drawdown"] < -1 and row["z_trend"] < -1, d
        assert row["z_credit"] < -1, d  # corporate bonds falling vs Treasuries (vs 60-day avg)
        assert not row["above200"], d


def test_detrended_vars_stay_in_bounds_after_learning(ann):
    """credit and rates are changes, not levels — they must not drift out of the learning range for years."""
    yearly = ann.loc["2013":, ["z_credit", "z_rates"]].groupby(ann.loc["2013":].index.year).mean()
    assert (yearly.abs() < 1).all().all(), yearly


def test_rsi_range(ann):
    assert ann["rsi2"].between(0, 100).all()
