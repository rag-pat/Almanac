"""M4: broker fills/costs/equity and outcome scoring."""
import pandas as pd
import pytest

from almanac import config
from almanac.broker import Broker, outcomes, record_decision, resolve_due, score
from almanac.data import load_csv

pytestmark = pytest.mark.skipif(not config.MONGODB_URI, reason="no Atlas URI")

AGENT = "pytest-broker"
SPY = load_csv("SPY")


def days_from(start, n):
    return list(SPY.loc[start:].index[:n])


@pytest.fixture(autouse=True)
def clean():
    outcomes().delete_many({"agent_id": AGENT})
    yield
    outcomes().delete_many({"agent_id": AGENT})


def test_buy_fills_next_open_with_cost_then_tracks_close():
    d = days_from("2019-01-02", 4)
    b = Broker()
    b.step(d[0]); b.order("buy")
    assert not b.invested                       # decided at close, nothing filled yet
    b.step(d[1])
    assert b.invested and b.trades[0]["date"] == d[1]
    o1, c1, c2 = SPY.loc[d[1], "open"], SPY.loc[d[1], "close"], SPY.loc[d[2], "close"]
    assert b.equity[-1][1] == pytest.approx((1 - config.COST) * c1 / o1)
    b.step(d[2])
    assert b.equity[-1][1] == pytest.approx((1 - config.COST) * c2 / o1)


def test_round_trip_pays_cost_twice_and_redundant_orders_are_free():
    d = days_from("2019-01-02", 4)
    b = Broker()
    b.step(d[0]); b.order("sell")               # sell while out: ignored
    b.step(d[1]); assert b.trades == [] and b.equity[-1][1] == 1.0
    b.order("buy"); b.step(d[2]); b.order("buy")  # buy while in: ignored
    b.order("sell"); b.step(d[3])
    o2, o3 = SPY.loc[d[2], "open"], SPY.loc[d[3], "open"]
    assert len(b.trades) == 2
    assert b.equity[-1][1] == pytest.approx((1 - config.COST) ** 2 * o3 / o2)


def test_position_days_counter():
    d = days_from("2019-01-02", 3)
    b = Broker()
    for x in d:
        b.step(x)
    assert b.position == {"in": False, "days": 3}


def test_score_rules():
    c = config.COST
    assert score("buy", 0.01) == "win" and score("buy", c / 2) == "loss"
    assert score("sell", -0.01) == "win" and score("sell", -c / 2) == "loss"
    assert score("hold", 0.05) is None


def test_outcome_resolves_after_exactly_5_trading_days_using_closes():
    d = days_from("2020-03-09", 7)
    assert record_decision(AGENT, d[0], "hold", [], {}, "x") is None   # holds not scored
    record_decision(AGENT, d[0], "buy", ["abc"], {"vix": 5.0}, "panic")
    assert resolve_due(AGENT, d[4]) == []                               # only 4 days later
    res = resolve_due(AGENT, d[6])                                      # 6 days later: resolve
    assert len(res) == 1
    r = SPY.loc[d[5], "close"] / SPY.loc[d[0], "close"] - 1             # t+5 close, not t+6
    assert res[0]["R"] == pytest.approx(r)
    assert res[0]["resolved_date"] == pd.Timestamp(d[5]).to_pydatetime()
    assert res[0]["outcome"] == score("buy", r)
    assert resolve_due(AGENT, d[6]) == []                               # not resolved twice


def test_sell_outcome_in_march_2020_crash():
    d = days_from("2020-03-02", 6)
    record_decision(AGENT, d[0], "sell", [], {}, "x")
    res = resolve_due(AGENT, d[5])
    assert res[0]["R"] < 0 and res[0]["outcome"] == "win"               # market fell -> selling won
