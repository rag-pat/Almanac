"""Offline evaluation sweep: all 4 agents over 12 non-overlapping ~60-trading-day windows (2016-2024).

Canonical runs use one cached sample per prompt (Haiku rejects a temperature setting, so the cache is what
makes reruns identical). Variance runs: each preset re-run with seeds 1..3 — separate cached samples at the
model's default sampling — to show mean ± std.

Run:  python -m almanac.sweep             # canonical sweep (12 windows x 4 agents, parallel)
      python -m almanac.sweep variance    # presets x 3 seeds
      python -m almanac.sweep summary
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np

from almanac import config, db
from almanac.daily import AGENTS
from almanac.demo import simulate

# 12 non-overlapping windows; the 3 presets are among them
WINDOWS = {
    "early 2016 selloff": ("2016-01-04", "2016-03-30"),
    "late 2016": ("2016-09-01", "2016-11-23"),
    "mid 2017 calm": ("2017-06-01", "2017-08-24"),
    "feb 2018 vol spike": ("2018-02-01", "2018-04-26"),
    "late 2018": config.PRESETS["late 2018"],
    "2019 trade war": ("2019-05-01", "2019-07-25"),
    "2020 crash": config.PRESETS["2020 crash"],
    "late 2020 rebound": ("2020-09-01", "2020-11-23"),
    "mid 2021": ("2021-06-01", "2021-08-24"),
    "2022 bear": config.PRESETS["2022 bear"],
    "2023 banks": ("2023-03-01", "2023-05-24"),
    "mid 2024": ("2024-06-03", "2024-08-26"),
}
SEEDS = (1, 2, 3)


def sweeps():
    return db.db()["sweeps"]


def _run_window(name, start, end, seed=None, kind="canonical"):
    t0 = time.time()
    last = simulate(start, end, seed)
    last["days"] = len(last["days"])
    docs = [{"kind": kind, "window": name, "start": start, "end": end, "seed": seed, "agent": a,
             "metrics": last["metrics"][a], "days": last["days"], "at": datetime.utcnow()} for a in AGENTS]
    docs.append({"kind": kind, "window": name, "start": start, "end": end, "seed": seed, "agent": "buy_hold",
                 "metrics": last["benchmark"], "days": last["days"], "at": datetime.utcnow()})
    sweeps().delete_many({"kind": kind, "window": name, "seed": seed})
    sweeps().insert_many(docs)
    m = last["metrics"]
    print(f"  {name:20s} {'seed ' + str(seed) if seed else '':7s} {time.time() - t0:4.0f}s  " +
          "  ".join(f"{a} {m[a]['total_return']:+.1%}" for a in AGENTS) +
          f"  buy&hold {last['benchmark']['total_return']:+.1%}", flush=True)


def run_canonical(workers: int = 6):
    print(f"sweep: {len(WINDOWS)} windows x {len(AGENTS)} agents, {workers} windows in parallel")
    with ThreadPoolExecutor(workers) as pool:
        for f in [pool.submit(_run_window, n, s, e) for n, (s, e) in WINDOWS.items()]:
            f.result()


def run_variance(workers: int = 6):
    jobs = [(n, s, e, seed) for n, (s, e) in config.PRESETS.items() for seed in SEEDS]
    print(f"variance: {len(config.PRESETS)} presets x {len(SEEDS)} seeds")
    with ThreadPoolExecutor(workers) as pool:
        for f in [pool.submit(_run_window, n, s, e, seed, "variance") for n, s, e, seed in jobs]:
            f.result()


def summary() -> dict:
    rows = list(sweeps().find({"kind": "canonical"}, {"_id": 0}))
    by = {}
    for r in rows:
        by.setdefault(r["agent"], {})[r["window"]] = r["metrics"]
    out = {"windows": len({r["window"] for r in rows}), "agents": {}}
    plain = by.get("plain", {})
    for a in list(AGENTS) + ["buy_hold"]:
        ms = by.get(a, {})
        if not ms:
            continue
        rets = [m["total_return"] for m in ms.values()]
        out["agents"][a] = {
            "mean_return": float(np.mean(rets)),
            "mean_max_drawdown": float(np.mean([m["max_drawdown"] for m in ms.values()])),
            "mean_sharpe": float(np.mean([m["sharpe"] for m in ms.values()])),
            "beat_plain": (sum(ms[w]["total_return"] > plain[w]["total_return"] for w in ms if w in plain)
                           if a != "plain" else None),
            "n": len(ms),
        }
    var = {}
    for r in sweeps().find({"kind": "variance"}, {"_id": 0}):
        var.setdefault(r["window"], {}).setdefault(r["agent"], []).append(r["metrics"]["total_return"])
    out["variance"] = {w: {a: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                               "runs": len(v)} for a, v in ags.items()} for w, ags in var.items()}
    out["windows_list"] = {n: list(se) for n, se in WINDOWS.items()}
    return out


def print_summary():
    s = summary()
    print(f"\n=== sweep summary ({s['windows']} windows) ===")
    print(f"  {'agent':10s} {'mean return':>12s} {'mean max DD':>12s} {'mean Sharpe':>12s}  beat plain")
    for a, m in s["agents"].items():
        bp = f"{m['beat_plain']} of {m['n']}" if m["beat_plain"] is not None else "-"
        print(f"  {a:10s} {m['mean_return']:>+12.2%} {m['mean_max_drawdown']:>12.2%} {m['mean_sharpe']:>12.2f}  {bp}")
    if "almanac" in s["agents"]:
        print(f"\n  almanac beat plain in {s['agents']['almanac']['beat_plain']} of {s['agents']['almanac']['n']} windows")
    for w, ags in s["variance"].items():
        print(f"  {w}: " + "  ".join(f"{a} {v['mean']:+.1%} ± {v['std']:.1%} (n={v['runs']})" for a, v in ags.items()))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "canonical"
    if cmd == "canonical":
        run_canonical()
        print_summary()
    elif cmd == "variance":
        run_variance()
        print_summary()
    elif cmd == "summary":
        print_summary()
    else:
        sys.exit(f"unknown command: {cmd}")
