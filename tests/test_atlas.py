"""M1 step 3: Atlas load checks — counts match CSVs, unique index, bars_until never leaks the future."""
import pandas as pd
import pytest

from almanac import config, db
from almanac.data import bars_until, features_until, load_csv
from almanac.features import VARS

pytestmark = pytest.mark.skipif(
    not config.MONGODB_URI or "USER:PASS" in config.MONGODB_URI, reason="no Atlas URI"
)


def test_bar_counts_match_csv():
    bars = db.db()["bars"]
    for sym in config.SYMBOLS:
        assert bars.count_documents({"symbol": sym}) == len(load_csv(sym)), sym


def test_unique_index_on_symbol_date():
    idx = db.db()["bars"].index_information()
    keys = [tuple(k for k, _ in v["key"]) for v in idx.values() if v.get("unique")]
    assert ("symbol", "date") in keys


def test_duplicate_insert_rejected():
    from pymongo.errors import DuplicateKeyError
    doc = db.db()["bars"].find_one({"symbol": "SPY"}, {"_id": 0})
    with pytest.raises(DuplicateKeyError):
        db.db()["bars"].insert_one(doc)


@pytest.mark.parametrize("cut", ["2008-10-10", "2020-03-16", "2020-03-14"])  # 03-14 is a Saturday
def test_bars_until_never_returns_future(cut):
    df = bars_until(cut, n=50)
    assert len(df) == 50
    assert df.index.max() <= pd.Timestamp(cut)


def test_bars_until_matches_csv():
    df = bars_until("2020-03-16", n=5)
    csv = load_csv("SPY").loc[:"2020-03-16"].tail(5)
    assert list(df.index) == list(csv.index)
    assert df["close"].tolist() == pytest.approx(csv["close"].tolist())


def test_features_until():
    df = features_until("2020-03-16", n=1)
    assert df.index[-1] == pd.Timestamp("2020-03-16")
    assert all(f"z_{v}" in df.columns for v in VARS)
    assert df["z_vix"].iloc[-1] > 5
