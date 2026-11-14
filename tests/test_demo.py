"""M6: metrics + demo window rules."""
import pytest

from almanac import config
from almanac.metrics import summarize


def test_metrics_on_known_curve():
    eq = [1.0, 1.1, 0.99, 1.2]
    m = summarize(eq, [{"action": "buy", "price": 10}, {"action": "sell", "price": 12}], invested_days=2)
    assert m["total_return"] == pytest.approx(0.2)
    assert m["max_drawdown"] == pytest.approx(0.99 / 1.1 - 1)
    assert m["hit_rate"] == 1.0 and m["trades"] == 2 and m["time_invested"] == 0.5
    assert m["sharpe"] > 0 and m["sortino"] > 0 and m["calmar"] > 0


def test_flat_curve_is_all_zero():
    m = summarize([1.0, 1.0, 1.0], [], invested_days=0)
    assert m["total_return"] == 0 and m["max_drawdown"] == 0 and m["hit_rate"] is None


@pytest.mark.skipif(not config.MONGODB_URI, reason="no Atlas URI")
def test_window_rules():
    from almanac.demo import window_days
    with pytest.raises(ValueError):
        window_days("2014-01-02", "2014-02-01")          # before demo start (notebook period)
    days = window_days("2018-01-02", "2018-12-31")
    assert len(days) == config.DEMO_MAX_DAYS             # capped at 60 trading days
    for name, (s, e) in config.PRESETS.items():
        assert 50 <= len(window_days(s, e)) <= 60, name
