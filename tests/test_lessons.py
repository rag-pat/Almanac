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
