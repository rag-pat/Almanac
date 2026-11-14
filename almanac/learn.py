"""M5: the learning run — the agent trades 2007-2012 weekly and writes its own notebook.

Each decision week t:
  1. resolve trades from >= 5 trading days ago -> update cited lessons, write a new lesson, merge or insert,
     retire losers  (all DB writes for one outcome in ONE transaction; logged to memory_events)
  2. retrieve lessons through the Atlas gate (range rule), decide, explore 10% of weeks, queue the order
Between decisions the broker steps every trading day (fill at next open, mark equity at close).

Run:  python -m almanac.learn                       # full run 2007-2012 (fresh)
      python -m almanac.learn 2007-01-01 2007-06-30  # short test window
Reruns are fast: every LLM / embedding call is served from the disk cache.
"""
import hashlib
import random
import sys
import time
from datetime import datetime

import numpy as np
from bson import ObjectId
from scipy.stats import beta

from almanac import config, db
from almanac.agent import decide
from almanac.broker import Broker, find_due, outcomes, record_decision
from almanac.data import _to_dt, features_until
from almanac.features import VARS, describe
from almanac.lessons import BOX, coll as lessons, make_lesson, notebook_coverage, retrieve
from almanac.llm import cosine, embed, llm_json

AGENT = config.LEARN_AGENT

LESSON_SYSTEM = (
    "You write one-sentence trading lessons from your own past trades in one US asset. "
    "Describe the market conditions using the condition words given, what you did, and how it turned out, "
    "so the lesson tells your future self what to do in similar conditions. "
    "No dates, tickers or exact numbers. Under 30 words."
)
LESSON_SCHEMA = {"type": "object", "properties": {"lesson": {"type": "string"}},
                 "required": ["lesson"], "additionalProperties": False}


def events():
    return db.db()["memory_events"]


def write_lesson(o: dict) -> str:
    moved = "rose" if o["R"] > 0 else "fell"
    if o.get("kept"):
        did = "held and stayed fully invested" if o["action"] == "buy" else "held and stayed in cash"
    else:
        did = "bought (went fully invested)" if o["action"] == "buy" else "sold (went to cash)"
    user = (f"Conditions when you decided: {o['description']}\n"
            f"Signals: RSI(2) = {o.get('rsi2', 50):.0f}; price {'above' if o.get('above200') else 'below'} "
            f"its 200-day average.\n"
            f"You {did}. Over the next week the asset {moved} {abs(o['R']) * 100:.1f}%, "
            f"so the trade was a {o['outcome'].upper()}.\nWrite the lesson.")
    return llm_json(LESSON_SYSTEM, user, LESSON_SCHEMA)["lesson"].strip()


def find_merge(emb: list[float], action: str, outcome: str):
    """Best active lesson with same action AND outcome; merge only if cosine >= SIM_MERGE."""
    best, best_cos = None, -1.0
    for les in lessons().find({"agent_id": AGENT, "status": "active", "action": action, "outcome": outcome},
                              {"embedding": 1, "fp": 1, **{f"lo_{v}": 1 for v in VARS},
                               **{f"hi_{v}": 1 for v in VARS}}):
        c = cosine(emb, les["embedding"])
        if c > best_cos:
            best, best_cos = les, c
    return best, best_cos


def should_retire(wins: int, losses: int) -> bool:
    n = wins + losses
    return n >= config.RETIRE_MIN_N and beta.cdf(0.5, 1 + wins, 1 + losses) > config.RETIRE_PROB


def apply_outcome(o: dict, upd: dict, text: str, emb: list[float], merge_into, merge_cos: float, today):
    """All writes for one resolved outcome, in one transaction."""
    won = upd["outcome"] == "win"
    fp = o["fp"]
    now = datetime.utcnow()
    sim = _to_dt(today)

    def txn(s):
        outcomes().update_one({"_id": o["_id"]}, {"$set": upd}, session=s)
        # 1. credit / blame the lessons this decision cited
        for lid in o.get("cited", []):
            les = lessons().find_one_and_update(
                {"_id": lid}, {"$inc": {"wins" if won else "losses": 1},
                               "$push": {"history": {"event": "cited", "date": sim, "outcome": upd["outcome"]}}},
                session=s, return_document=True)
            if les and les["status"] == "active" and should_retire(les["wins"], les["losses"]):
                p = float(beta.cdf(0.5, 1 + les["wins"], 1 + les["losses"]))
                lessons().update_one({"_id": lid}, {"$set": {"status": "retired"},
                                     "$push": {"history": {"event": "retired", "date": sim, "p_lose": p}}},
                                     session=s)
                events().insert_one({"type": "retire", "lesson_id": lid, "date": sim, "at": now,
                                     "wins": les["wins"], "losses": les["losses"], "p_lose": p,
                                     "text": les["text"]}, session=s)
        # 2. rediscovery -> widen, else new lesson
        if merge_into is not None:
            sets, widened = {}, []
            for v in VARS:
                lo, hi = min(merge_into[f"lo_{v}"], fp[v] - BOX), max(merge_into[f"hi_{v}"], fp[v] + BOX)
                if lo < merge_into[f"lo_{v}"] or hi > merge_into[f"hi_{v}"]:
                    widened.append(v)
                sets[f"lo_{v}"], sets[f"hi_{v}"] = lo, hi
            lessons().update_one({"_id": merge_into["_id"]},
                                 {"$set": sets, "$inc": {"wins": 1},
                                  "$push": {"history": {"event": "widen", "date": sim, "cos": merge_cos,
                                                        "vars": widened, "variant": text}}}, session=s)
            events().insert_one({"type": "widen", "lesson_id": merge_into["_id"], "date": sim, "at": now,
                                 "vars": widened, "cos": merge_cos, "variant": text}, session=s)
            return merge_into["_id"], "widen"
        doc = make_lesson(text, emb, fp, o["action"], upd["outcome"], sim, agent_id=AGENT)
        lid = lessons().insert_one(doc, session=s).inserted_id
        events().insert_one({"type": "create", "lesson_id": lid, "date": sim, "at": now, "text": text}, session=s)
        return lid, "create"

    with db.client().start_session() as s:
        return s.with_transaction(txn)


def reset():
    for c in (lessons(), outcomes(), events()):
        c.delete_many({"agent_id": AGENT} if c.name != "memory_events" else {})


def _explore(date) -> bool:
    h = int(hashlib.sha256(f"explore-{date:%Y-%m-%d}".encode()).hexdigest(), 16)
    return (h % 1000) / 1000 < config.EXPLORE_RATE


def _pct(xs):
    if not xs:
        return "n/a"
    q = np.percentile(xs, [5, 25, 50, 75, 95])
    return f"n={len(xs)}  p5={q[0]:.2f} p25={q[1]:.2f} p50={q[2]:.2f} p75={q[3]:.2f} p95={q[4]:.2f}  max={max(xs):.2f}"


def print_score_distribution(retr_cos, merge_cos, label):
    print(f"\n=== similarity distribution ({label}) ===")
    print(f"  retrieval cosine (all lessons returned, cutoff {config.SIM_RETRIEVE}): {_pct(retr_cos)}")
    print(f"  merge: best same-action/outcome cosine per new lesson (cutoff {config.SIM_MERGE}): {_pct(merge_cos)}")
    if merge_cos:
        print(f"  would merge at 0.80: {np.mean(np.array(merge_cos) >= 0.80):.0%}   at 0.85: "
              f"{np.mean(np.array(merge_cos) >= 0.85):.0%}   at 0.90: {np.mean(np.array(merge_cos) >= 0.90):.0%}\n")


def run(start=config.LEARN_START, end=config.NOTEBOOK_END, fresh=True):
    if fresh:
        reset()
    feats = features_until(end)
    days = list(feats.loc[start:end].index)
    broker = Broker()
    retr_cos, merge_cos, printed50 = [], [], False
    stats = {"decisions": 0, "explore": 0, "create": 0, "widen": 0, "retire": 0}
    t0 = time.time()

    for i, day in enumerate(days):
        broker.step(day)
        if i % config.DECIDE_EVERY:
            continue
        feat = feats.loc[day].to_dict()
        z = {v: float(feat[f"z_{v}"]) for v in VARS}

        # 1. learn from trades that just matured
        for o, upd in find_due(AGENT, day):
            o2 = {**o, **upd}
            text = write_lesson(o2)
            emb = embed(text, "document")
            cand, c = find_merge(emb, o["action"], upd["outcome"])
            if cand is not None:
                merge_cos.append(c)
            _, kind = apply_outcome(o, upd, text, emb, cand if c >= config.SIM_MERGE else None, c, day)
            stats[kind] += 1
        stats["retire"] = events().count_documents({"type": "retire"})

        # 2. decide (learning agent uses the range-filtered notebook)
        r = retrieve(z, day, agent_id=AGENT)
        retr_cos += [2 * x["score"] - 1 for x in r["relevant"]]
        d = decide(day, broker.position, r["almanac"], "almanac", feat)
        intent, cited, explore = d["raw_action"], d["cited"], _explore(day)
        if explore:
            rng = random.Random(f"{day:%Y-%m-%d}")
            intent = {"buy": "sell", "sell": "buy"}.get(intent) or rng.choice(["buy", "sell"])
            cited = []  # forced action: no lesson gets credit or blame
            stats["explore"] += 1
        # a hold keeps the current position, so it is scored like that position: in -> long, out -> cash
        kept = intent == "hold"
        scored = ("buy" if broker.invested else "sell") if kept else intent
        broker.order(intent)
        record_decision(AGENT, day, scored, [ObjectId(c) for c in cited], z, r["description"],
                        {"rsi2": float(feat["rsi2"]), "above200": bool(feat["above200"]), "explore": explore,
                         "kept": kept, "reason": d["reason"]})
        stats["decisions"] += 1

        n_active = lessons().count_documents({"agent_id": AGENT, "status": "active"})
        print(f"{day:%Y-%m-%d}  {'EXPLORE ' if explore else ''}{intent:4s} in={int(broker.invested)} "
              f"eq={broker.equity[-1][1]:.3f}  saw {len(r['almanac'])}/{len(r['relevant'])} "
              f"(in-range/relevant)  notebook={n_active} active  +{stats['create']} new "
              f"{stats['widen']} widen {stats['retire']} retired  [{time.time() - t0:.0f}s]", flush=True)
        if not printed50 and stats["create"] >= 50:
            print_score_distribution(retr_cos, merge_cos, "first ~50 lessons")
            printed50 = True

    print_score_distribution(retr_cos, merge_cos, "full run")
    total = lessons().count_documents({"agent_id": AGENT})
    active = lessons().count_documents({"agent_id": AGENT, "status": "active"})
    print(f"\n=== LEARNING RUN DONE ({start} -> {end}, {time.time() - t0:.0f}s) ===")
    print(f"  decisions: {stats['decisions']} ({stats['explore']} exploration)")
    print(f"  total lessons: {total}  ({active} active, {total - active} retired)")
    print(f"  merges (rediscoveries that widened a box): {events().count_documents({'type': 'widen'})}")
    print(f"  retirements: {events().count_documents({'type': 'retire'})}")
    print(f"  final equity: {broker.equity[-1][1]:.3f}")
    return stats


def preset_coverage():
    """Average lessons in range per demo preset (frozen notebook, lessons learned before window start)."""
    print("\n=== notebook coverage on demo presets ===")
    for name, (s, e) in config.PRESETS.items():
        f = features_until(e).loc[s:e]
        rows = [notebook_coverage({v: float(r[f"z_{v}"]) for v in VARS}, s, AGENT) for _, r in f.iterrows()]
        unf = {}
        for r in rows:
            for v in r["unfamiliar"]:
                unf[v] = unf.get(v, 0) + 1
        print(f"  {name:11s} {len(rows)} days  notebook={rows[0]['n_total']}  "
              f"avg in range (>=6/8): {np.mean([r['n_in_range'] for r in rows]):.1f}  "
              f"min {min(r['n_in_range'] for r in rows)}  max {max(r['n_in_range'] for r in rows)}  "
              f"unfamiliar days: {unf or 'none'}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "coverage":
        preset_coverage()
    else:
        run(*(args[:2] if len(args) >= 2 else ()))
        preset_coverage()
