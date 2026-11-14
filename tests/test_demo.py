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
    with pytest.raises(ValueError):
        window_days("2024-06-03", "2025-03-01")          # past the end of the data
    assert len(window_days("2018-01-02", "2018-12-31")) > 240   # no length cap any more


@pytest.mark.skipif(not config.MONGODB_URI, reason="no Atlas URI")
def test_simulator_charges_cost_only_when_target_changes(monkeypatch):
    """Targets: IN, IN, OUT, OUT. Fill at next open; one buy (day 1 open) and one sell (day 3 open)."""
    import almanac.demo as demo
    from almanac.data import load_csv
    days = list(load_csv("SPY").loc["2019-03-01":].index[:5])
    targets = ["IN", "IN", "OUT", "OUT", "OUT"]
    dec = {(a, d.to_pydatetime()): {"target": t, "reason": "", "cited": [], "n_lessons": 0}
           for d, t in zip(days, targets) for a in demo.AGENTS}
    monkeypatch.setattr(demo, "load", lambda s, e, seed=None: (days, {}, dec))
    sim = demo.simulate(days[0], days[-1])
    spy = load_csv("SPY")
    expected = (1 - config.COST) ** 2 * spy.loc[days[3], "open"] / spy.loc[days[1], "open"]
    rec = sim["days"]
    assert [r["agents"]["almanac"]["action"] for r in rec] == ["buy", "stay_in", "sell", "stay_out", "stay_out"]
    assert rec[-1]["agents"]["almanac"]["equity"] == pytest.approx(expected, abs=1e-5)  # streamed equity is rounded to 5 dp
    assert sim["metrics"]["almanac"]["trades"] == 2
