"""Evidence logs for the conditional-track-record gate.

backfill(): rebuild every lesson's evidence and W/L from the learning run.
  Each entry is {date, z, win}: decision-day z-scores, dated when the result was known, and win = whether
  the lesson's OWN recommendation (REC: buy-won -> IN, buy-lost -> OUT, sell-won -> OUT, sell-lost -> IN)
  was right over that 5-day stretch.
    birth         the trade the lesson was written from (win 1 by construction)
    rediscovery   every later trade that merged into it (recovered by replaying the learner's lesson
                  writing from the disk cache and matching the text; never calls the model)
    citation      every resolved decision that cited it ($lookup lessons -> outcomes)
  wins/losses are recounted from the evidence (so the W/L line and A describe the same history), and the
  retirement rule is re-run on the corrected counts. Idempotent.

report(): per demo preset, how the gate behaves.

Run:  python -m almanac.evidence backfill
      python -m almanac.evidence report
"""
import json
import sys
from datetime import datetime

import numpy as np
from pymongo import UpdateOne
from scipy.stats import beta

from almanac import config, db
from almanac.data import features_until
from almanac.features import VARS
from almanac.lessons import REC, evidence_entry, rec_win, retrieve

AGENT = config.LEARN_AGENT


def _cached_lesson_text(o: dict) -> str | None:
    """The lesson text the learner wrote for outcome `o`, read from the LLM disk cache (None if not cached)."""
    from almanac.learn import LESSON_SCHEMA, LESSON_SYSTEM, lesson_prompt
    from almanac.llm import _cache_path
    payload = {"model": config.LLM_MODEL, "effort": config.LLM_EFFORT, "system": LESSON_SYSTEM,
               "user": lesson_prompt(o), "schema": LESSON_SCHEMA}
    p = _cache_path("llm", payload)
    return json.loads(p.read_text())["lesson"].strip() if p.exists() else None


def config_cutoff() -> datetime:
    """Retirements from the recount are dated at the end of the learning period (evidence ends there)."""
    return datetime.fromisoformat(config.NOTEBOOK_END)


def _should_retire(w: int, l: int) -> tuple[bool, float]:
    p = float(beta.cdf(0.5, 1 + w, 1 + l))
    return (w + l >= config.RETIRE_MIN_N and p > config.RETIRE_PROB), p


def backfill(agent_id: str = AGENT) -> dict:
    L = db.db()["lessons"]
    # citations via $lookup (one aggregation), plus the fields we need from each lesson
    rows = list(L.aggregate([
        {"$match": {"agent_id": agent_id}},
        {"$lookup": {"from": "outcomes", "let": {"lid": "$_id"}, "as": "uses", "pipeline": [
            {"$match": {"$expr": {"$and": [{"$in": ["$$lid", {"$ifNull": ["$cited", []]}]},
                                           {"$eq": ["$status", "resolved"]}]}}},
            {"$project": {"resolved_date": 1, "fp": 1, "R": 1}}]}},
        {"$project": {"text": 1, "action": 1, "outcome": 1, "fp": 1, "learned_date": 1, "status": 1,
                      "wins": 1, "losses": 1, "history": 1, "uses": 1}},
    ]))
    by_text = {r["text"].strip(): r["_id"] for r in rows}
    by_variant = {h["variant"].strip(): r["_id"] for r in rows for h in r.get("history", [])
                  if h.get("event") == "widen" and h.get("variant")}

    # rediscoveries: replay each resolved outcome's lesson text from cache and match it
    merges, births, missing = {}, set(), 0
    for o in db.db()["outcomes"].find({"agent_id": agent_id, "status": "resolved"}):
        text = _cached_lesson_text(o)
        if text is None:
            missing += 1
        elif text in by_variant:
            merges.setdefault(by_variant[text], []).append(o)
        elif text in by_text:
            births.add(by_text[text])

    ops, retired_now, before_after = [], [], []
    now = datetime.utcnow()
    for r in rows:
        rec = REC[(r["action"], r["outcome"])]
        ev = [evidence_entry(r["learned_date"], r["fp"], 1)]
        ev += [evidence_entry(o["resolved_date"], o["fp"], rec_win(rec, o["R"])) for o in merges.get(r["_id"], [])]
        ev += [evidence_entry(u["resolved_date"], u["fp"], rec_win(rec, u["R"])) for u in r["uses"]]
        ev.sort(key=lambda e: e["date"])
        w = sum(e["win"] for e in ev)
        l = len(ev) - w
        retire, p = _should_retire(w, l)
        sets = {"evidence": ev, "wins": w, "losses": l, "rec": rec, "status": "retired" if retire else "active"}
        upd = {"$set": sets}
        if retire and r["status"] != "retired":
            retired_now.append((r, w, l, p))
            upd["$push"] = {"history": {"event": "retired", "date": now, "p_lose": p, "by": "evidence recount"}}
        ops.append(UpdateOne({"_id": r["_id"]}, upd))
        before_after.append((r["wins"], r["losses"], w, l))
    L.bulk_write(ops)
    if retired_now:
        db.db()["memory_events"].insert_many([
            {"type": "retire", "lesson_id": r["_id"], "date": config_cutoff(), "at": now, "wins": w, "losses": l, "p_lose": p, "text": r["text"], "by": "evidence recount"}
            for r, w, l, p in retired_now])

    n_ev = [1 + len(merges.get(r["_id"], [])) + len(r["uses"]) for r in rows]
    stats = {"lessons": len(rows), "entries": sum(n_ev), "median": float(np.median(n_ev)), "max": max(n_ev),
             "merges_matched": sum(len(v) for v in merges.values()), "births_matched": len(births),
             "cache_misses": missing, "retired": len(retired_now),
             "active": L.count_documents({"agent_id": agent_id, "status": "active"})}
    print(f"backfilled {stats['lessons']} lessons: {stats['entries']} evidence entries "
          f"(median {stats['median']:.0f}, max {stats['max']}); rediscoveries matched {stats['merges_matched']}, "
          f"births matched {stats['births_matched']}, cache misses {missing}")
    print(f"retired by the corrected records: {len(retired_now)}  (active now: {stats['active']})")
    for r, w, l, p in retired_now[:5]:
        print(f"  retired {w}W/{l}L (P(lose)={p:.2f}): {r['text'][:110]}")
    big = sorted(before_after, key=lambda t: -(t[0] + t[1]))[:3]
    print("  biggest lessons, W/L before -> after:", ", ".join(f"{a}W/{b}L -> {c}W/{d}L" for a, b, c, d in big))
    return stats


def report(agent_id: str = AGENT):
    print(f"\n=== gate per preset (h={config.KERNEL_H}, proven A>={config.A_MIN}, hide A<{config.A_FAIL}, "
          f"n_eff>={config.NEFF_MIN}, {config.GATE_SMOOTH_DAYS}-day smoothed z) ===")
    for name, (s, e) in config.PRESETS.items():
        allf = features_until(e)
        smooth = allf[[f"z_{v}" for v in VARS]].rolling(config.GATE_SMOOTH_DAYS, min_periods=1).mean()
        f = allf.loc[s:e]
        shown, a_shown, hidden_low_a, unproven, best_example = [], [], [], [], None
        for _, row in f.iterrows():
            z = {v: float(row[f"z_{v}"]) for v in VARS}
            zg = {v: float(smooth.loc[row.name, f"z_{v}"]) for v in VARS}
            r = retrieve(z, config.NOTEBOOK_CUTOFF, agent_id=agent_id, z_gate=zg)
            shown.append(len(r["almanac"]))
            a_shown += [x["A"] for x in r["almanac"]]
            hidden_low_a.append(sum(1 for x in r["relevant"] if x["cls"] == 3))
            unproven.append(sum(1 for x in r["relevant"] if x["cls"] == 2))
            for x in r["relevant"]:  # good overall record, poor record in markets like today
                n = x["wins"] + x["losses"]
                overall = x["wins"] / n if n else 0
                if n >= 5 and overall >= 0.6 and x["cls"] == 3:
                    gap = overall - x["A"]
                    if best_example is None or gap > best_example[0]:
                        best_example = (gap, row.name, x, overall)
        print(f"\n  {name} ({len(f)} days, window starts {s})")
        print(f"    avg lessons shown to almanac: {np.mean(shown):.1f}   (plain sees up to {config.RETRIEVE_CAP})")
        print(f"    avg A of shown lessons: {np.mean(a_shown):.2f}" if a_shown else "    avg A of shown lessons: n/a (none shown)")
        print(f"    avg relevant lessons hidden (proven to fail here, A<{config.A_FAIL}): {np.mean(hidden_low_a):.1f}   "
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
