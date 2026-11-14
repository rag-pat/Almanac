"""Run every demo preset once so the live demo replays from cache (instant). Prints each preset's results.

Run after the learning run finishes:  python -m almanac.warm
"""
import time
from concurrent.futures import ThreadPoolExecutor

from almanac import config
from almanac.demo import run_demo


def warm_one(name, start, end):
    t0, last = time.time(), None
    for msg in run_demo(start, end):
        last = msg
    m = last["metrics"]
    line = "  ".join(f"{a}: {m[a]['total_return']:+.1%} (DD {m[a]['max_drawdown']:.1%})" for a in m)
    return f"{name:11s} {last['days']} days in {time.time() - t0:.0f}s | {line} | buy&hold {last['benchmark']['total_return']:+.1%}"


if __name__ == "__main__":
    with ThreadPoolExecutor(len(config.PRESETS)) as pool:
        futs = [pool.submit(warm_one, n, s, e) for n, (s, e) in config.PRESETS.items()]
        for f in futs:
            print(f.result(), flush=True)
