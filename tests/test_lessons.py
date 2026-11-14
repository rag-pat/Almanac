"""M2 steps 2-5: lessons schema, vector index, retrieval."""
import pytest
from pymongo.errors import WriteError

from almanac import config
from almanac.features import VARS
from almanac.lessons import coll, make_lesson

pytestmark = pytest.mark.skipif(not config.MONGODB_URI, reason="no Atlas URI")

TEST_AGENT = "pytest"


@pytest.fixture(autouse=True)
def cleanup():
    coll().delete_many({"agent_id": TEST_AGENT})
    yield
    coll().delete_many({"agent_id": TEST_AGENT})


def good(**over):
    doc = make_lesson("test lesson", [0.01] * config.EMBED_DIM, {v: 0.0 for v in VARS},
                      "buy", "win", "2010-01-04", agent_id=TEST_AGENT)
    doc.update(over)
    return doc


def test_valid_lesson_accepted():
    coll().insert_one(good())
    d = coll().find_one({"agent_id": TEST_AGENT})
    assert d["lo_vix"] == -0.5 and d["hi_vix"] == 0.5 and d["wins"] == 1 and d["losses"] == 0


@pytest.mark.parametrize("field", ["text", "embedding", "fp", "status", "learned_date", "history",
                                   "wins", "action", "outcome", "lo_vix", "hi_credit"])
def test_missing_field_rejected(field):
    doc = good()
    doc.pop(field)
    with pytest.raises(WriteError):
        coll().insert_one(doc)


@pytest.mark.parametrize("bad", [
    {"status": "maybe"},
    {"action": "hold"},                       # holds are never scored, so never lessons
    {"outcome": "draw"},
    {"embedding": [0.1] * 10},                # wrong dimension
    {"wins": -1},
    {"learned_date": "2010-01-04"},           # string, not a date
    {"fp": {"vol": 0.0}},                     # fingerprint missing variables
])
def test_bad_values_rejected(bad):
    with pytest.raises(WriteError):
        coll().insert_one(good(**bad))


def _search(qvec, before, extra=None, limit=50):
    filt = {"status": "active", "learned_date": {"$lt": before}, "agent_id": TEST_AGENT}
    filt.update(extra or {})
    return list(coll().aggregate([
        {"$vectorSearch": {"index": "lessons_vector", "path": "embedding", "queryVector": qvec,
                           "exact": True, "limit": limit, "filter": filt}},
        {"$project": {"text": 1, "score": {"$meta": "vectorSearchScore"}}},
    ]))


def _wait_for(fn, n, timeout=30):
    import time
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = fn()
        if len(r) >= n:
            return r
        time.sleep(1)
    return fn()


def test_vector_index_filters_status_and_date():
    from datetime import datetime
    v = [0.0] * config.EMBED_DIM
    v[0] = 1.0
    coll().insert_many([
        good(text="old active", embedding=v),
        good(text="old retired", embedding=v, status="retired"),
        good(text="future active", embedding=v, learned_date=datetime(2015, 1, 1)),
    ])
    res = _wait_for(lambda: _search(v, datetime(2013, 1, 1)), 1)
    assert [r["text"] for r in res] == ["old active"]
    assert res[0]["score"] > 0.99


def test_atlas_score_is_normalized_cosine():
    """Atlas cosine vectorSearchScore = (1 + cos) / 2 — retrieval must convert before comparing to SIM_RETRIEVE."""
    from datetime import datetime
    a = [0.0] * config.EMBED_DIM; a[0] = 1.0
    b = [0.0] * config.EMBED_DIM; b[1] = 1.0           # cos(a, b) = 0
    c = [0.0] * config.EMBED_DIM; c[0] = 0.5; c[1] = 0.75 ** 0.5   # cos(a, c) = 0.5
    coll().insert_many([good(text="orthogonal", embedding=b), good(text="cos0.5", embedding=c)])
    res = {r["text"]: r["score"] for r in _wait_for(lambda: _search(a, datetime(2013, 1, 1)), 2)}
    assert res["orthogonal"] == pytest.approx(0.5, abs=1e-4)
    assert res["cos0.5"] == pytest.approx(0.75, abs=1e-4)
