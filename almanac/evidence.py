"""Evidence logs for the conditional-track-record gate.

backfill(): rebuild every lesson's evidence from the learning run, in ONE aggregation:
  birth entry   {date: learned_date, z: fp, win: 1}
  + one entry per resolved decision that cited the lesson ($lookup into outcomes)
    {date: resolved_date, z: decision-day z, win: outcome == "win"}
  -> $merge back into lessons. Idempotent.

report(): per demo preset, how the A gate behaves.

Run:  python -m almanac.evidence backfill
      python -m almanac.evidence report
"""
import sys

import numpy as np

from almanac import config, db
from almanac.data import features_until
from almanac.features import VARS
from almanac.lessons import retrieve

AGENT = config.LEARN_AGENT


def _zarr(path: str) -> list:
    return [f"{path}.{v}" for v in VARS]


def backfill(agent_id: str = AGENT) -> int:
    pipeline = [
        {"$match": {"agent_id": agent_id}},
        {"$lookup": {
            "from": "outcomes",
            "let": {"lid": "$_id"},
            "pipeline": [
                {"$match": {"$expr": {"$and": [{"$in": ["$$lid", {"$ifNull": ["$cited", []]}]},
                                               {"$eq": ["$status", "resolved"]}]}}},
                {"$project": {"resolved_date": 1, "fp": 1, "outcome": 1}},
            ],
            "as": "uses"}},
        {"$project": {"evidence": {"$concatArrays": [
            [{"date": "$learned_date", "z": _zarr("$fp"), "win": 1}],
            {"$map": {"input": {"$sortArray": {"input": "$uses", "sortBy": {"resolved_date": 1}}}, "as": "u",
                      "in": {"date": "$$u.resolved_date", "z": _zarr("$$u.fp"),
                             "win": {"$cond": [{"$eq": ["$$u.outcome", "win"]}, 1, 0]}}}},
        ]}}},
        {"$merge": {"into": "lessons", "on": "_id", "whenMatched": "merge", "whenNotMatched": "discard"}},
    ]
    db.db()["lessons"].aggregate(pipeline)
    stats = next(db.db()["lessons"].aggregate([
        {"$match": {"agent_id": agent_id}},
        {"$project": {"n": {"$size": {"$ifNull": ["$evidence", []]}}}},
        {"$group": {"_id": None, "lessons": {"$sum": 1}, "entries": {"$sum": "$n"}, "max": {"$max": "$n"}}}]))
    print(f"backfilled {stats['lessons']} lessons, {stats['entries']} evidence entries "
          f"(avg {stats['entries'] / stats['lessons']:.1f}, max {stats['max']})")
    return stats["entries"]


def report(agent_id: str = AGENT):
    print(f"\n=== A gate per preset (h={config.KERNEL_H}, A>={config.A_MIN}, n_eff>={config.NEFF_MIN}) ===")
    for name, (s, e) in config.PRESETS.items():
        f = features_until(e).loc[s:e]
        shown, a_shown, hidden_low_a, unproven, best_example = [], [], [], [], None
        for _, row in f.iterrows():
            z = {v: float(row[f"z_{v}"]) for v in VARS}
            r = retrieve(z, s, agent_id=agent_id)
            shown.append(len(r["almanac"]))
            a_shown += [x["A"] for x in r["almanac"]]
            hidden_low_a.append(sum(1 for x in r["relevant"] if x["n_eff"] >= config.NEFF_MIN and x["A"] < config.A_MIN))
            unproven.append(sum(1 for x in r["relevant"] if x["n_eff"] < config.NEFF_MIN))
            for x in r["relevant"]:  # good overall record, poor record in markets like today
                n = x["wins"] + x["losses"]
                overall = x["wins"] / n if n else 0
                if n >= 5 and overall >= 0.6 and x["n_eff"] >= config.NEFF_MIN and x["A"] < config.A_MIN:
                    gap = overall - x["A"]
                    if best_example is None or gap > best_example[0]:
                        best_example = (gap, row.name, x, overall)
        print(f"\n  {name} ({len(f)} days, window starts {s})")
        print(f"    avg lessons shown to almanac: {np.mean(shown):.1f}   (plain sees up to {config.RETRIEVE_CAP})")
        print(f"    avg A of shown lessons: {np.mean(a_shown):.2f}" if a_shown else "    avg A of shown lessons: n/a (none shown)")
        print(f"    avg relevant lessons hidden for low A: {np.mean(hidden_low_a):.1f}   "
              f"avg unproven here (n_eff<{config.NEFF_MIN:g}): {np.mean(unproven):.1f}")
        if best_example:
            _, day, x, overall = best_example
            print(f"    example ({day:%Y-%m-%d}): overall {x['wins']}W/{x['losses']}L ({overall:.0%}) but "
                  f"A today {x['A']:.0%} from {x['n_eff']:.1f} similar uses")
            print(f"      \"{x['text']}\"")
        else:
            print("    example: none found (no lesson with a good overall record and low A on these days)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "backfill":
        backfill()
    elif cmd == "report":
        report()
    else:
        sys.exit(f"unknown command: {cmd}")
