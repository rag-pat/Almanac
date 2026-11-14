"""M8: counterfactual flip-rate logic (LLM mocked) + endpoint validation."""
import pytest
from bson import ObjectId

from almanac import config
import almanac.counterfactual as cf

pytestmark = pytest.mark.skipif(not config.MONGODB_URI, reason="no Atlas URI")

KEY, OTHER = ObjectId(), ObjectId()
LESSONS = [{"_id": KEY, "text": "key lesson", "wins": 4, "losses": 0},
           {"_id": OTHER, "text": "other lesson", "wins": 1, "losses": 0}]


def fake_decide(date, lessons, mode, feat=None, ctx=None, fresh=False, seed=None):
    has_key = any(l["_id"] == KEY for l in lessons)
    return {"target": "OUT" if has_key else "IN", "reason": "r", "cited": []}


def test_flip_rate_when_lesson_drives_decision(monkeypatch):
    monkeypatch.setattr(cf, "_almanac_context", lambda d: (LESSONS, {}, {}))
    monkeypatch.setattr(cf, "decide_target", fake_decide)
    r = cf.counterfactual("2020-03-16", str(KEY))
    assert r["original"] == "OUT"
    assert r["flip_rate"] == 1.0 and r["baseline_agreement"] == 1.0
    assert r["without_lesson"] == ["IN"] * cf.REPEATS


def test_zero_flip_for_irrelevant_lesson(monkeypatch):
    monkeypatch.setattr(cf, "_almanac_context", lambda d: (LESSONS, {}, {}))
    monkeypatch.setattr(cf, "decide_target", fake_decide)
    assert cf.counterfactual("2020-03-16", str(OTHER))["flip_rate"] == 0.0


def test_rejects_lesson_not_in_prompt_and_non_trading_day(monkeypatch):
    monkeypatch.setattr(cf, "_almanac_context", lambda d: (LESSONS, {}, {}))
    with pytest.raises(ValueError):
        cf.counterfactual("2020-03-16", str(ObjectId()))
    monkeypatch.undo()
    with pytest.raises(ValueError):
        cf.counterfactual("2020-03-14", str(KEY))  # Saturday
