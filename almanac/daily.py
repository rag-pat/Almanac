"""Precomputed demo decisions: every agent, every trading day 2016-2024, stored in `daily_decisions`.

Demo decisions are position-independent (each agent picks tomorrow's target, IN or OUT), so they can be
computed once per (agent, date) and replayed for any window. The notebook is frozen: only lessons learned,
and evidence dated, before NOTEBOOK_CUTOFF are visible.

Docs:  {agent: "_market", date, seed, description, variables, signals, coverage, applies: [lesson ids]}
       {agent: <name>, date, seed, target: "IN"|"OUT"|None, reason, cited: [lesson views], n_lessons}

Run:  python -m almanac.daily                 # all of 2016-2024 (resumes; skips what is stored)
      python -m almanac.daily 2020-02-10 2020-05-01
"""
import asyncio
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from pymongo import ASCENDING, UpdateOne

from almanac import config, db
from almanac.agent import DAY_OFFSET, decide_target
from almanac.data import _to_dt, bars_until, features_until
from almanac.features import VARS, describe
from almanac.lessons import applicable_lessons, retrieve
from almanac.llm import embed_many

AGENTS = ("none", "plain", "plain_ctx", "almanac")
CUTOFF = _to_dt(config.NOTEBOOK_CUTOFF)
NOTEBOOK = config.LEARN_AGENT


def coll():
    c = db.db()["daily_decisions"]
    c.create_index([("agent", ASCENDING), ("date", ASCENDING), ("seed", ASCENDING)], unique=True)
    return c


def trading_days(start, end) -> list:
    return list(features_until(end).loc[str(start)[:10]:str(end)[:10]].index)


def _lesson_view(les: dict) -> dict:
    return {"id": str(les["_id"]), "text": les["text"], "wins": les["wins"], "losses": les["losses"],
            "A": round(float(les.get("A", 0.5)), 3), "n_eff": round(float(les.get("n_eff", 0.0)), 2)}


def day_contexts(dates: list, workers: int = 16) -> dict:
    """Everything a day's prompts need, with batched embeddings and parallel Atlas retrieval."""
    dates = sorted(dates)
    feats = features_until(dates[-1])
    spy_dates = list(feats.index)  # features exist for every SPY trading day
    bars = bars_until(dates[-1], since=spy_dates[max(0, spy_dates.index(dates[0]) - 30)])
    all_bar_dates = db.db()["bars"].distinct("date", {"symbol": "SPY", "date": {"$lte": _to_dt(dates[-1])}})
    all_bar_dates.sort()
    pos = {d: i for i, d in enumerate(all_bar_dates)}

    z_by = {d: {v: float(feats.loc[d, f"z_{v}"]) for v in VARS} for d in dates}
    texts = [describe(z_by[d]) for d in dates]
    qvecs = dict(zip(dates, embed_many(texts, "query")))

    def one(d):
        z = z_by[d]
        r = retrieve(z, CUTOFF, agent_id=NOTEBOOK, qvec=qvecs[d])
        closes = list(bars.loc[:d]["close"].iloc[-20:])
        return d, {"feat": feats.loc[d].to_dict(), "z": z, "r": r, "closes": closes,
                   "day_number": DAY_OFFSET + pos[d.to_pydatetime()] + 1,
                   "applies": list(applicable_lessons(z, CUTOFF, NOTEBOOK))}

    with ThreadPoolExecutor(workers) as pool:
        return dict(pool.map(one, dates))


def _market_doc(d, c, seed) -> dict:
    r, feat = c["r"], c["feat"]
    return {"agent": "_market", "date": d.to_pydatetime(), "seed": seed, "description": r["description"],
            "variables": {v: round(c["z"][v], 3) for v in VARS},
            "signals": {"rsi2": round(float(feat["rsi2"]), 1), "above200": bool(feat["above200"])},
            "coverage": {"applies": r["n_applies"], "unproven": r["n_unproven"], "total": r["n_total"],
                         "relevant": len(r["relevant"]), "unfamiliar": r["unfamiliar"]},
            "applies": c["applies"], "at": datetime.utcnow()}


def _decide_with_retry(d, agent, c, seed, tries: int = 6) -> dict:
    seen = {"none": [], "plain": c["r"]["relevant"], "plain_ctx": c["r"]["relevant"], "almanac": c["r"]["almanac"]}
    for attempt in range(tries):
        try:
            out = decide_target(d, seen[agent], agent, c["feat"],
                                {"closes": c["closes"], "day_number": c["day_number"]}, seed=seed)
            by_id = {str(x["_id"]): x for x in seen[agent]}
            return {"agent": agent, "date": d.to_pydatetime(), "seed": seed, "target": out["target"],
                    "reason": out["reason"], "n_lessons": out["n_lessons"],
                    "cited": [_lesson_view(by_id[i]) for i in out["cited"] if i in by_id], "at": datetime.utcnow()}
        except Exception as e:  # rate limits / overload / network: back off and retry
            if attempt == tries - 1:
                raise
            time.sleep(min(60, 2 ** attempt + random.random() * 2))
            last = e  # noqa: F841


async def _decide_all(jobs, seed, concurrency, label):
    sem = asyncio.Semaphore(concurrency)
    done, t0, out = 0, time.time(), []

    async def run(d, agent, c):
        nonlocal done
        async with sem:
            doc = await asyncio.to_thread(_decide_with_retry, d, agent, c, seed)
        done += 1
        if done % 200 == 0 or done == len(jobs):
            rate = done / (time.time() - t0)
            print(f"  {label}: {done}/{len(jobs)} decisions  {rate:.1f}/s  ETA {(len(jobs) - done) / rate / 60:.1f} min",
                  flush=True)
        return doc

    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(concurrency + 8))
    for chunk_start in range(0, len(jobs), 400):  # write as we go so progress survives interruptions
        chunk = jobs[chunk_start:chunk_start + 400]
        docs = await asyncio.gather(*(run(*j) for j in chunk))
        coll().bulk_write([UpdateOne({"agent": x["agent"], "date": x["date"], "seed": seed}, {"$set": x}, upsert=True)
                           for x in docs], ordered=False)
        out += docs
    return out


def compute(dates: list, seed: int | None = None, concurrency: int = config.PRECOMPUTE_CONCURRENCY,
            label: str = "precompute") -> int:
    """Compute and store whatever is missing for these dates. Returns number of decisions computed."""
    if not dates:
        return 0
    have = {(x["agent"], x["date"]) for x in coll().find(
        {"date": {"$gte": _to_dt(min(dates)), "$lte": _to_dt(max(dates))}, "seed": seed}, {"agent": 1, "date": 1})}
    need_days = [d for d in dates if any((a, d.to_pydatetime()) not in have for a in AGENTS + ("_market",))]
    if not need_days:
        return 0
    total = 0
    for i in range(0, len(need_days), 250):  # contexts in chunks keep memory and embedding batches sane
        days = need_days[i:i + 250]
        t0 = time.time()
        ctxs = day_contexts(days)
        coll().bulk_write([UpdateOne({"agent": "_market", "date": d.to_pydatetime(), "seed": seed},
                                     {"$set": _market_doc(d, ctxs[d], seed)}, upsert=True) for d in days])
        jobs = [(d, a, ctxs[d]) for d in days for a in AGENTS if (a, d.to_pydatetime()) not in have]
        print(f"{label}: days {days[0]:%Y-%m-%d}..{days[-1]:%Y-%m-%d} contexts in {time.time() - t0:.0f}s, "
              f"{len(jobs)} decisions", flush=True)
        asyncio.run(_decide_all(jobs, seed, concurrency, label))
        total += len(jobs)
    return total


def load(start, end, seed: int | None = None) -> tuple[list, dict, dict]:
    """(trading days, market docs by date, decisions by (agent, date)) — computing anything missing first."""
    days = trading_days(start, end)
    compute(days, seed, label="on-the-fly")
    q = {"date": {"$gte": _to_dt(days[0]), "$lte": _to_dt(days[-1])}, "seed": seed}
    market, dec = {}, {}
    for x in coll().find(q, {"_id": 0}):
        if x["agent"] == "_market":
            market[x["date"]] = x
        else:
            dec[(x["agent"], x["date"])] = x
    return days, market, dec


if __name__ == "__main__":
    s, e = (sys.argv[1], sys.argv[2]) if len(sys.argv) >= 3 else (config.DEMO_MIN_START, config.DEMO_END)
    t0 = time.time()
    n = compute(trading_days(s, e))
    print(f"DONE: {n} decisions computed for {s}..{e} in {(time.time() - t0) / 60:.1f} min "
          f"(collection now has {coll().count_documents({'seed': None})} docs)")
