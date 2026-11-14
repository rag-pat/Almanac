"""M2 step 4-5: the Atlas gate end to end, with seeded lessons and real embeddings."""
import time
from datetime import datetime

import pytest

from almanac import config
from almanac.data import features_until
from almanac.features import VARS
from almanac.lessons import coll, make_lesson, retrieve
from almanac.llm import embed

pytestmark = pytest.mark.skipif(not (config.MONGODB_URI and config.VOYAGE_API_KEY), reason="no keys")

AGENT = "pytest-retrieval"
BEFORE = datetime(2013, 1, 1)
CRASH = "With fear very high and credit stress severe, buying a very deep drawdown lost money; the selloff kept going."
CRASH2 = "With fear very high and price far below its trend, buying after a very deep drawdown paid off on a sharp bounce."
CALM = "In a calm market near recent highs with low volatility and fear low, staying invested paid off."


def shifted(z, n, by=2.0):
    """Copy of z with the first n variables moved out of a ±0.5 box."""
    return {v: z[v] + (by if i < n else 0.0) for i, v in enumerate(VARS)}


@pytest.fixture(scope="module")
def today():
    row = features_until("2020-03-16", n=1).iloc[-1]
    return {v: float(row[f"z_{v}"]) for v in VARS}


@pytest.fixture(scope="module")
def seeded(today):
    coll().delete_many({"agent_id": AGENT})
    e_crash, e_crash2, e_calm = embed(CRASH), embed(CRASH2), embed(CALM)
    L = lambda name, emb, fp, **kw: {**make_lesson(name, emb, fp, "buy", "loss", kw.pop("d", "2010-05-06"),
                                                   agent_id=AGENT), **kw}
    docs = [
        L("crash 8of8", e_crash, today),
        L("crash 6of8", e_crash2, shifted(today, 2)),
        L("crash 5of8", e_crash, shifted(today, 3)),
        L("crash 0of8", e_crash2, shifted(today, 8)),
        L("calm 8of8", e_calm, today),                       # in range, but doesn't sound like today
        L("crash retired", e_crash, today, status="retired"),
        L("crash future", e_crash, today, d="2015-01-02"),  # learned after the window start
    ]
    coll().insert_many(docs)
    t0 = time.time()  # vector index is eventually consistent — wait until all visible docs are searchable
    while time.time() - t0 < 60:
        if len(retrieve(today, BEFORE, agent_id=AGENT)["relevant"]) >= 4:
            break
        time.sleep(1)
    yield
    coll().delete_many({"agent_id": AGENT})


def names(rows):
    return [r["text"] for r in rows]


def test_plain_gets_everything_that_sounds_like_today(today, seeded):
    r = retrieve(today, BEFORE, agent_id=AGENT)
    assert set(names(r["relevant"])) == {"crash 8of8", "crash 6of8", "crash 5of8", "crash 0of8"}
    scores = [x["score"] for x in r["relevant"]]
    assert scores == sorted(scores, reverse=True) and min(scores) >= config.ATLAS_SCORE_MIN


def test_almanac_sees_only_lessons_proven_here(today, seeded):
    """Only birth trades are evidence here. The lesson born at today's conditions is proven here
    (n_eff = 1, A = 2/3); lessons born 2+ z-units away are unproven here (n_eff < 1) and left out."""
    r = retrieve(today, BEFORE, agent_id=AGENT)
    assert names(r["almanac"]) == ["crash 8of8"]
    x = r["almanac"][0]
    assert x["cls"] == 0 and x["A"] == pytest.approx(2 / 3) and x["n_eff"] == pytest.approx(1.0)
    assert all(y["cls"] == 2 for y in r["relevant"] if y["text"] != "crash 8of8")


def test_retired_future_and_dissimilar_never_returned(today, seeded):
    r = retrieve(today, BEFORE, agent_id=AGENT)
    for bad in ("crash retired", "crash future", "calm 8of8"):
        assert bad not in names(r["relevant"]) + names(r["almanac"])


def test_coverage_counts_whole_visible_notebook(today, seeded):
    r = retrieve(today, BEFORE, agent_id=AGENT)
    assert r["n_total"] == 5                  # retired + future excluded
    assert r["n_applies"] == 2                # crash 8of8 + calm 8of8 (coverage ignores text similarity)
    assert r["n_unproven"] == 3               # 6of8, 5of8, 0of8: born too far from today
    assert r["n_failing"] == 0
    assert r["unfamiliar"] == []


def test_unfamiliar_flags_variable_outside_every_lesson(today, seeded):
    weird = {**today, "curve": today["curve"] + 10}
    r = retrieve(weird, BEFORE, agent_id=AGENT)
    assert r["unfamiliar"] == ["curve"]


def test_cap(today, seeded):
    from almanac.lessons import search
    qvec = embed(retrieve(today, BEFORE, agent_id=AGENT)["description"], "query")
    assert len(search(qvec, today, BEFORE, almanac=False, agent_id=AGENT, cap=2)) == 2
