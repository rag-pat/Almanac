"""Conditional track record (A gate): pipeline math matches the reference, and evidence dated after the
window start never affects A."""
import time
from datetime import datetime

import pytest

from almanac import config
from almanac.data import features_until
from almanac.features import VARS
from almanac.lessons import applicability_py, coll, evidence_entry, make_lesson, retrieve
from almanac.llm import embed

pytestmark = pytest.mark.skipif(not (config.MONGODB_URI and config.VOYAGE_API_KEY), reason="no keys")

AGENT = "pytest-evidence"
WINDOW = datetime(2020, 2, 10)
TEXT = "With fear very high and credit stress severe, buying a very deep drawdown lost money; the selloff kept going."


@pytest.fixture(scope="module")
def today():
    row = features_until("2020-03-16", n=1).iloc[-1]
    return {v: float(row[f"z_{v}"]) for v in VARS}


def shift(z, by, n=8):
    return {v: z[v] + (by if i < n else 0) for i, v in enumerate(VARS)}


@pytest.fixture(scope="module")
def seeded(today):
    coll().delete_many({"agent_id": AGENT})
    emb = embed(TEXT)
    past = [evidence_entry(datetime(2011, 5, 2), shift(today, 0.4), 1),
            evidence_entry(datetime(2012, 3, 1), shift(today, 1.0, 2), 0),
            evidence_entry(datetime(2013, 7, 1), shift(today, 0.2), 1)]
    future_losses = [evidence_entry(datetime(2021, 3, 1), today, 0) for _ in range(4)]
    future_wins = [evidence_entry(datetime(2021, 3, 1), today, 1) for _ in range(4)]
    docs = []
    for name, future in [("future losses", future_losses), ("future wins", future_wins), ("no future", [])]:
        d = make_lesson(name, emb, today, "sell", "win", "2010-06-01", agent_id=AGENT)
        d["evidence"] += past + future
        docs.append(d)
    coll().insert_many(docs)
    t0 = time.time()
    while time.time() - t0 < 60 and len(retrieve(today, WINDOW, agent_id=AGENT)["relevant"]) < 3:
        time.sleep(1)
    yield {d["text"]: d for d in docs}
    coll().delete_many({"agent_id": AGENT})


def test_pipeline_matches_reference_math(today, seeded):
    r = {x["text"]: x for x in retrieve(today, WINDOW, agent_id=AGENT)["relevant"]}
    for name, doc in seeded.items():
        a, n = applicability_py(doc["evidence"], today, WINDOW)
        assert r[name]["A"] == pytest.approx(a, rel=1e-9) and r[name]["n_eff"] == pytest.approx(n, rel=1e-9)


def test_evidence_after_window_start_never_affects_A(today, seeded):
    r = {x["text"]: x for x in retrieve(today, WINDOW, agent_id=AGENT)["relevant"]}
    a = {k: r[k]["A"] for k in seeded}
    assert a["future losses"] == pytest.approx(a["no future"]) == pytest.approx(a["future wins"])
    assert r["future losses"]["n_evidence"] == r["no future"]["n_evidence"] == 4   # birth + 3 past
    # sanity: once the window starts after those entries, they DO count
    later = {x["text"]: x for x in retrieve(today, datetime(2022, 1, 3), agent_id=AGENT)["relevant"]}
    assert later["future losses"]["A"] < a["no future"] < later["future wins"]["A"]


def test_almanac_ranks_by_A_and_shows_record_in_prompt(today, seeded):
    from almanac.agent import build_prompt
    r = retrieve(today, WINDOW, agent_id=AGENT)
    cls = [x["cls"] for x in r["almanac"]]
    assert cls == sorted(cls)                                     # proven, then mixed, then unproven
    proven = [x["A"] for x in r["almanac"] if x["cls"] < 2]
    assert proven == sorted(proven, reverse=True)
    user, _ = build_prompt("2020-03-16", {"in": False, "days": 0}, r["almanac"], "almanac")
    assert "in markets like today (" in user and "similar uses)" in user
    plain, _ = build_prompt("2020-03-16", {"in": False, "days": 0}, r["relevant"], "plain")
    assert "W/" in plain and "markets like today" not in plain


def test_lessons_proven_to_fail_here_are_hidden(today, seeded):
    from almanac.lessons import coll
    d = dict(seeded["no future"])
    d.pop("_id", None)
    d["text"] = "fails here"
    from almanac.lessons import evidence_entry
    d["evidence"] = [evidence_entry(datetime(2011, 1, 3), today, 0) for _ in range(4)]   # A = 1/6, n_eff = 4
    coll().insert_one(d)
    t0 = time.time()
    while time.time() - t0 < 30 and "fails here" not in [x["text"] for x in retrieve(today, WINDOW, agent_id=AGENT)["relevant"]]:
        time.sleep(1)
    r = retrieve(today, WINDOW, agent_id=AGENT)
    assert "fails here" in [x["text"] for x in r["relevant"]]
    assert "fails here" not in [x["text"] for x in r["almanac"]]
    assert r["n_failing"] >= 1
