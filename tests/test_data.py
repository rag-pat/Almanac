"""M1 step 1: sanity checks on the downloaded CSVs."""
import pandas as pd
import pytest

from almanac import config
from almanac.data import load_csv

YIELDS = ["^TNX", "^IRX"]                   # quoted in percent; IRX hit ~0% / slightly negative 2009-15, 2020
TRADED = ["SPY", "LQD", "IEF"]              # real ETFs with volume
PRICES = [s for s in config.SYMBOLS if s not in YIELDS]


@pytest.fixture(scope="module")
def frames():
    return {s: load_csv(s) for s in config.SYMBOLS}


def test_date_range(frames):
    for sym, df in frames.items():
        assert df.index[0] < pd.Timestamp("2005-01-10"), sym
        assert df.index[-1] > pd.Timestamp("2024-12-20"), sym
        assert df.index.is_monotonic_increasing and df.index.is_unique, sym


def test_no_missing_values(frames):
    for sym, df in frames.items():
        assert not df[["open", "high", "low", "close"]].isna().any().any(), sym
        assert (df["high"] >= df["low"]).all(), sym


def test_prices_positive(frames):
    for sym in PRICES:
        assert (frames[sym][["open", "high", "low", "close"]] > 0).all().all(), sym


def test_yields_plausible(frames):
    for sym in YIELDS:
        c = frames[sym]["close"]
        assert c.between(-1, 20).all(), (sym, c.min(), c.max())


def test_traded_have_volume(frames):
    for sym in TRADED:
        assert (frames[sym]["volume"] > 0).all(), sym


def test_aux_aligns_with_spy(frames):
    spy = frames["SPY"].index
    for sym in config.SYMBOLS:
        missing = spy.difference(frames[sym].index)
        # bond market closes a few extra days (Columbus/Veterans Day) — forward-filled in features
        assert len(missing) <= 20, (sym, len(missing))
