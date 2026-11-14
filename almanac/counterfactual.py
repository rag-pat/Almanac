"""Proof: does a cited lesson actually drive the almanac decision?

Rebuild the exact almanac prompt for a demo day (frozen notebook: lessons and evidence before
NOTEBOOK_CUTOFF), then ask the LLM REPEATS times (fresh, uncached) with the lesson and REPEATS times
without it. Flip rate = share of runs without the lesson whose target (IN/OUT) differs from the original
decision. The with-lesson runs show the model's baseline noise.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from almanac.agent import decide_target
from almanac.daily import day_contexts
from almanac.data import _to_dt, features_until

REPEATS = 5


def _almanac_context(date):
    row = features_until(date, n=1)
    if row.empty or row.index[-1] != _to_dt(date):
        raise ValueError(f"{date} is not a trading day")
    d = row.index[-1]
    c = day_contexts([d])[d]
    return c["r"]["almanac"], c["feat"], {"closes": c["closes"], "day_number": c["day_number"]}


def counterfactual(date, lesson_id: str, repeats: int = REPEATS) -> dict:
    lessons, feat, ctx = _almanac_context(date)
    target_lesson = next((l for l in lessons if str(l["_id"]) == lesson_id), None)
    if target_lesson is None:
        raise ValueError("that lesson was not in the almanac agent's prompt on this day")
    without = [l for l in lessons if str(l["_id"]) != lesson_id]

    original = decide_target(date, lessons, "almanac", feat, ctx)  # cached: the decision shown in the demo
    with ThreadPoolExecutor(2 * repeats) as pool:
        w = [pool.submit(decide_target, date, lessons, "almanac", feat, ctx, True) for _ in range(repeats)]
        wo = [pool.submit(decide_target, date, without, "almanac", feat, ctx, True) for _ in range(repeats)]
        with_runs, without_runs = [f.result() for f in w], [f.result() for f in wo]

    orig = original["target"]
    flips = sum(r["target"] != orig for r in without_runs)
    stays = sum(r["target"] == orig for r in with_runs)
    return {
        "date": str(date)[:10],
        "lesson": {"id": lesson_id, "text": target_lesson["text"], "wins": target_lesson["wins"],
                   "losses": target_lesson["losses"], "A": target_lesson.get("A"), "n_eff": target_lesson.get("n_eff")},
        "original": orig, "original_reason": original["reason"],
        "with_lesson": [r["target"] for r in with_runs],
        "without_lesson": [r["target"] for r in without_runs],
        "without_reasons": [r["reason"] for r in without_runs],
        "flip_rate": flips / repeats,
        "baseline_agreement": stays / repeats,
        "summary": (f"Without this lesson the target changed in {flips} of {repeats} runs "
                    f"({flips / repeats:.0%}). With it, {stays} of {repeats} runs kept {orig}. "
                    f"Without it: {dict(Counter(r['target'] for r in without_runs))}."),
    }
