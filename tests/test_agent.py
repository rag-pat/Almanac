"""M3 step 2: agent prompt + decision safety."""
import re

import pytest
from bson import ObjectId

from almanac import config
from almanac.agent import SYSTEM, build_prompt, decide
from almanac.data import bars_until
from almanac.features import VARS

pytestmark = pytest.mark.skipif(not config.MONGODB_URI, reason="no Atlas URI")
needs_llm = pytest.mark.skipif(not config.CLAUDE_API_KEY, reason="no Claude key")

DAYS = ["2008-10-10", "2017-06-30", "2020-03-16", "2022-10-12"]


def lesson(text, w=3, l=1, fp=None):
    return {"_id": ObjectId(), "text": text, "wins": w, "losses": l, "fp": fp or {v: 0.0 for v in VARS}}


CRASH_LOST = lesson("With fear very high and credit stress severe, buying a very deep drawdown lost money; "
                    "the selloff kept going.", w=5, l=1, fp={v: 2.0 for v in VARS})
CALM_WON = lesson("In a calm market near recent highs with fear low, staying invested paid off.", w=4, l=0)


@pytest.mark.parametrize("day", DAYS)
@pytest.mark.parametrize("mode", ["plain", "plain_ctx", "almanac", "none"])
def test_prompt_never_leaks_date_ticker_or_price(day, mode):
    user, _ = build_prompt(day, {"in": False, "days": 3}, [CRASH_LOST, CALM_WON], mode)
    text = SYSTEM + "\n" + user
    assert not re.search(r"\b(19|20)\d{2}\b", text), "year-like number in prompt"
    assert "SPY" not in text and "S&P" not in text
    for month in ("January", "March", "October", "Jan ", "Mar ", "Oct "):
        assert month not in text
    real_close = bars_until(day, n=1)["close"].iloc[-1]
    assert f"{real_close:.1f}" not in user and f"{real_close:.2f}" not in user


def test_prompt_shows_records_short_ids_and_mode_differences():
    plain, ids = build_prompt("2020-03-16", {"in": False, "days": 0}, [CRASH_LOST, CALM_WON], "plain")
    ctx, _ = build_prompt("2020-03-16", {"in": False, "days": 0}, [CRASH_LOST, CALM_WON], "plain_ctx")
    assert "[L1] (5W/1L)" in plain and "[L2] (4W/0L)" in plain
    assert ids == {"L1": str(CRASH_LOST["_id"]), "L2": str(CALM_WON["_id"])}
    assert "learned at:" not in plain and "Today's numbers:" not in plain
    assert ctx.count("learned at:") == 2 and "Today's numbers:" in ctx
    empty, _ = build_prompt("2020-03-16", {"in": True, "days": 0}, [], "almanac")
    assert "none apply" in empty


def test_safety_rules(monkeypatch):
    import almanac.agent as agent
    replies = iter([
        {"action": "buy", "cited": ["L1", "L9"], "reason": "x " * 60},   # invented L9, long reason
        {"action": "sell", "cited": [], "reason": "already out"},        # sell while out -> hold
        {"action": "yolo", "cited": [], "reason": "bad"},                # unknown action -> hold
    ])
    monkeypatch.setattr(agent, "llm_json", lambda *a, **k: next(replies))
    out = {"in": False, "days": 1}
    d = decide("2020-03-16", out, [CRASH_LOST], "plain")
    assert d["action"] == "buy" and d["cited"] == [str(CRASH_LOST["_id"])] and len(d["reason"].split()) <= 25
    assert decide("2020-03-16", out, [CRASH_LOST], "plain")["action"] == "hold"
    assert decide("2020-03-16", out, [CRASH_LOST], "plain")["action"] == "hold"


@needs_llm
@pytest.mark.parametrize("mode", ["plain", "plain_ctx", "almanac"])
def test_live_decision_is_valid(mode):
    for day in ("2020-03-16", "2017-06-30"):
        d = decide(day, {"in": False, "days": 5}, [CRASH_LOST, CALM_WON], mode)
        assert d["raw_action"] in ("buy", "sell", "hold")
        assert set(d["cited"]) <= {str(CRASH_LOST["_id"]), str(CALM_WON["_id"])}
        assert 0 < len(d["reason"].split()) <= 25


def test_none_mode_has_no_lessons_even_if_given_some():
    user, ids = build_prompt("2020-03-16", {"in": False, "days": 0}, [CRASH_LOST, CALM_WON], "none")
    assert "No past lessons available." in user and "[L1]" not in user and ids == {}
    plain, _ = build_prompt("2020-03-16", {"in": False, "days": 0}, [CRASH_LOST], "plain")
    assert user.split("No past lessons")[0] == plain.split("Past lessons")[0]   # identical market section


@pytest.mark.parametrize("mode", ["plain", "plain_ctx", "almanac", "none"])
def test_target_prompt_is_position_independent_and_leak_free(mode):
    from almanac.agent import SYSTEM_TARGET
    user, _ = build_prompt("2020-03-16", None, [CRASH_LOST, CALM_WON], mode)
    assert "You are currently" not in user and user.rstrip().endswith("Choose your target position for tomorrow: IN or OUT.")
    text = SYSTEM_TARGET + "\n" + user
    assert not re.search(r"\b(19|20)\d{2}\b", text) and "SPY" not in text
