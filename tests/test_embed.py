"""M2 step 1: day descriptions + embedding function."""
import time

import pytest

from almanac import config
from almanac.data import features_until
from almanac.features import VARS, bucket, describe
from almanac.llm import cosine, embed


def test_bucket_edges():
    assert [bucket(z) for z in (-3, -1.5, 0, 1.5, 3)] == [0, 1, 2, 3, 4]


def test_describe_prefers_zscores_over_raw():
    doc = {**{v: 999.0 for v in VARS}, **{f"z_{v}": 0.0 for v in VARS}}  # raw values huge, z all normal
    assert "very" not in describe(doc)


def test_describe_is_deterministic_and_hides_numbers():
    s = describe({f"z_{v}": 2.5 for v in VARS})
    assert s == describe({f"z_{v}": 2.5 for v in VARS})
    assert not any(ch.isdigit() for ch in s)


needs_key = pytest.mark.skipif(not config.VOYAGE_API_KEY or not config.MONGODB_URI, reason="no keys")


@needs_key
def test_crash_vs_calm_descriptions():
    crash = describe(features_until("2020-03-16", n=1).iloc[-1].to_dict())
    calm = describe(features_until("2017-06-30", n=1).iloc[-1].to_dict())
    assert "Fear very high" in crash and "Credit stress severe" in crash
    assert "Fear low" in calm


@needs_key
def test_embed_dim_cache_and_ranking():
    v = embed("Fear very high. Credit stress severe.", "query")
    assert len(v) == config.EMBED_DIM
    t = time.time()
    assert embed("Fear very high. Credit stress severe.", "query") == v
    assert time.time() - t < 0.05  # served from disk cache
    crash_lesson = embed("With fear very high and credit stress severe, buying the dip lost money.")
    calm_lesson = embed("In a calm market near recent highs with fear low, staying invested paid off.")
    assert cosine(v, crash_lesson) > cosine(v, calm_lesson)
