"""Per-agent diagnostics for a window: trades, % invested, cost drag, avg lessons shown, and how often the
almanac agent's lesson set changes day to day.   python -m almanac.diagnose 2016-02-10 2017-05-01"""
import sys

import numpy as np

from almanac.daily import AGENTS
from almanac.demo import simulate


def diagnose(start, end):
    sim = simulate(start, end)
    days = sim["days"]
    print(f"\n=== {start} -> {end} ({len(days)} trading days) ===")
    print(f"  {'agent':9s} {'trades':>6s} {'invested':>9s} {'cost drag':>10s} {'return':>8s} {'avg shown':>10s}")
    for a in AGENTS:
        m = sim["metrics"][a]
        shown = np.mean([d["agents"][a]["n_lessons"] for d in days])
        print(f"  {a:9s} {m['trades']:6d} {m['time_invested']:9.0%} {m['cost_drag']:10.2%} "
              f"{m['total_return']:+8.1%} {shown:10.1f}")
    print(f"  buy&hold {'':6s} {'100%':>9s} {'':10s} {sim['benchmark']['total_return']:+8.1%}")
    sets = [frozenset(d["agents"]["almanac"]["shown"]) for d in days]
    changes = sum(a != b for a, b in zip(sets, sets[1:]))
    print(f"  almanac lesson set changed on {changes} of {len(sets) - 1} day-to-day transitions "
          f"({changes / max(1, len(sets) - 1):.0%}); avg set size {np.mean([len(s) for s in sets]):.1f}, "
          f"days with 0 lessons: {sum(len(s) == 0 for s in sets)}")
    return sim


if __name__ == "__main__":
    diagnose(sys.argv[1], sys.argv[2])
