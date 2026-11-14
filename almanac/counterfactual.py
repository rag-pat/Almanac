"""M8 proof: does a cited lesson actually drive the almanac decision?

Rebuild the exact almanac prompt for a demo day, then ask the LLM REPEATS times (fresh, uncached)
with the lesson and REPEATS times without it. Flip rate = share of runs without the lesson whose
action differs from the original decision. The with-lesson runs show the model's baseline noise.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from almanac import config
from almanac.agent import decide
from almanac.data import _to_dt, features_until
from almanac.features import VARS
from almanac.lessons import retrieve

REPEATS = 5


def counterfactual(date, window_start, lesson_id: str, invested: bool, days_in_state: int = 0,
                   repeats: int = REPEATS) -> dict:
    row = features_until(date, n=1)
    if row.empty or row.index[-1] != _to_dt(date):
        raise ValueError(f"{date} is not a trading day")
    feat = row.iloc[-1].to_dict()
    z = {v: float(feat[f"z_{v}"]) for v in VARS}
    lessons = retrieve(z, window_start, agent_id=config.LEARN_AGENT)["almanac"]
    target = next((l for l in lessons if str(l["_id"]) == lesson_id), None)
    if target is None:
        raise ValueError("that lesson was not in the almanac agent's prompt on this day")
    without = [l for l in lessons if str(l["_id"]) != lesson_id]
    pos = {"in": invested, "days": days_in_state}

    original = decide(date, pos, lessons, "almanac", feat)  # cached: the decision shown in the demo
    with ThreadPoolExecutor(2 * repeats) as pool:
        w = [pool.submit(decide, date, pos, lessons, "almanac", feat, True) for _ in range(repeats)]
        wo = [pool.submit(decide, date, pos, without, "almanac", feat, True) for _ in range(repeats)]
        with_runs, without_runs = [f.result() for f in w], [f.result() for f in wo]

    orig = original["action"]
    flips = sum(r["action"] != orig for r in without_runs)
    stays = sum(r["action"] == orig for r in with_runs)
    return {
        "date": str(date)[:10], "lesson": {"id": lesson_id, "text": target["text"],
                                           "wins": target["wins"], "losses": target["losses"]},
        "original": orig, "original_reason": original["reason"],
        "with_lesson": [r["action"] for r in with_runs],
        "without_lesson": [r["action"] for r in without_runs],
        "without_reasons": [r["reason"] for r in without_runs],
        "flip_rate": flips / repeats,
        "baseline_agreement": stays / repeats,
        "summary": (f"Without this lesson the decision changed in {flips} of {repeats} runs "
                    f"({flips / repeats:.0%}). With it, {stays} of {repeats} runs kept {orig.upper()}. "
                    f"Without it: {dict(Counter(r['action'] for r in without_runs))}."),
    }
