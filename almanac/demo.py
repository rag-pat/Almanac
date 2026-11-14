"""M6: demo engine — 3 agents trade the same window daily on the frozen notebook (read only).

  plain      all relevant lessons (text)
  plain_ctx  all relevant lessons + fingerprints, judges relevance itself
  almanac    relevant lessons that are in range (>= IN_RANGE_MIN of 8)

Each day's variables are written to `market_state`; a change stream on it turns them into
"lessons entered / left range" events for the almanac agent. run_demo() yields one dict per day.
"""
import queue
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from almanac import config, db
from almanac.agent import decide
from almanac.broker import Broker
from almanac.data import _to_dt, features_until
from almanac.features import VARS
from almanac.lessons import coll as lessons_coll, in_range_lessons, retrieve
from almanac.metrics import summarize

AGENTS = ("plain", "plain_ctx", "almanac")
NOTEBOOK = config.LEARN_AGENT


def market_state():
    return db.db()["market_state"]


def window_days(start, end) -> list:
    if _to_dt(start) < _to_dt(config.DEMO_MIN_START):
        raise ValueError(f"demo windows start on or after {config.DEMO_MIN_START}")
    days = list(features_until(end).loc[start:end].index)
    if not days:
        raise ValueError("no trading days in window")
    return days[:config.DEMO_MAX_DAYS]


class RangeWatcher(threading.Thread):
    """Change stream on market_state: for each new day, diff the set of in-range lessons vs the previous day."""

    def __init__(self, run_id: str, before):
        super().__init__(daemon=True)
        self.run_id, self.before = run_id, before
        self.out: queue.Queue = queue.Queue()
        self.ready = threading.Event()
        self.stop = threading.Event()

    def run(self):
        pipeline = [{"$match": {"operationType": "insert", "fullDocument.run_id": self.run_id}}]
        prev = None
        with market_state().watch(pipeline, max_await_time_ms=300) as stream:
            self.ready.set()
            while not self.stop.is_set():
                ch = stream.try_next()
                if ch is None:
                    continue
                doc = ch["fullDocument"]
                z = {v: doc[f"z_{v}"] for v in VARS}
                now = in_range_lessons(z, self.before, NOTEBOOK)
                events = []
                if prev is None:
                    events.append({"kind": "baseline", "text": f"{len(now)} lessons in range at the start"})
                else:
                    entered, left = set(now) - set(prev), set(prev) - set(now)
                    if entered:
                        events.append({"kind": "entered", "text": f"{len(entered)} lesson(s) entered range",
                                       "lessons": [now[i] for i in list(entered)[:3]]})
                    if left:
                        events.append({"kind": "left", "text": f"{len(left)} lesson(s) left range",
                                       "lessons": [prev[i] for i in list(left)[:3]]})
                prev = now
                self.out.put((doc["date"], events))


def _lesson_view(les: dict) -> dict:
    return {"id": str(les["_id"]), "text": les["text"], "wins": les["wins"], "losses": les["losses"],
            "score": round(2 * les.get("score", 0.5) - 1, 3), "in_range": les.get("in_range")}


def run_demo(start, end):
    days = window_days(start, end)
    before = _to_dt(days[0])
    run_id = uuid.uuid4().hex[:12]
    feats = features_until(days[-1])
    brokers = {a: Broker() for a in AGENTS}
    invested_days = {a: 0 for a in AGENTS}
    closes, prev_unfamiliar = [], []
    notebook_total = lessons_coll().count_documents({"agent_id": NOTEBOOK, "learned_date": {"$lt": before}})

    watcher = RangeWatcher(run_id, before)
    watcher.start()
    watcher.ready.wait(10)
    pool = ThreadPoolExecutor(len(AGENTS))
    try:
        for day in days:
            bars = {a: b.step(day) for a, b in brokers.items()}
            bar = bars["almanac"]
            closes.append(bar["close"])
            for a, b in brokers.items():
                invested_days[a] += b.invested
            feat = feats.loc[day].to_dict()
            z = {v: float(feat[f"z_{v}"]) for v in VARS}

            market_state().insert_one({"run_id": run_id, "date": _to_dt(day), "at": datetime.utcnow(),
                                       **{f"z_{v}": z[v] for v in VARS},
                                       "rsi2": float(feat["rsi2"]), "above200": bool(feat["above200"])})
            r = retrieve(z, before, agent_id=NOTEBOOK)
            seen = {"plain": r["relevant"], "plain_ctx": r["relevant"], "almanac": r["almanac"]}
            futures = {a: pool.submit(decide, day, brokers[a].position, seen[a], a, feat) for a in AGENTS}
            decisions = {a: f.result() for a, f in futures.items()}

            agents_out = {}
            for a in AGENTS:
                d = decisions[a]
                pos_at_decision = brokers[a].position
                brokers[a].order(d["action"])
                by_id = {str(x["_id"]): x for x in seen[a]}
                agents_out[a] = {"action": d["action"], "intent": d["raw_action"], "reason": d["reason"],
                                 "invested": brokers[a].invested, "equity": bar_eq(brokers[a]),
                                 "days_in_state": pos_at_decision["days"],
                                 "n_lessons": len(seen[a]),
                                 "cited": [_lesson_view(by_id[c]) for c in d["cited"] if c in by_id]}

            events = []
            try:
                _, events = watcher.out.get(timeout=10)
            except queue.Empty:
                events = [{"kind": "warn", "text": "change stream event not received"}]
            if r["unfamiliar"] != prev_unfamiliar and r["unfamiliar"]:
                events.append({"kind": "unfamiliar", "text": "unfamiliar conditions: " + ", ".join(r["unfamiliar"])})
            prev_unfamiliar = r["unfamiliar"]

            yield {"type": "day", "run_id": run_id, "date": f"{day:%Y-%m-%d}", "window_start": f"{days[0]:%Y-%m-%d}",
                   "ohlc": {k: bar[k] for k in ("open", "high", "low", "close")},
                   "variables": {v: round(z[v], 2) for v in VARS},
                   "signals": {"rsi2": round(float(feat["rsi2"]), 1), "above200": bool(feat["above200"])},
                   "description": r["description"],
                   "coverage": {"in_range": r["n_in_range"], "total": notebook_total,
                                "relevant": len(r["relevant"]), "unfamiliar": r["unfamiliar"]},
                   "agents": agents_out, "events": events}

        bench = [c / closes[0] for c in closes]
        yield {"type": "end", "run_id": run_id, "days": len(days),
               "metrics": {a: summarize([e for _, e in b.equity], b.trades, invested_days[a], closes)
                           for a, b in brokers.items()},
               "benchmark": summarize(bench, [{"action": "buy", "price": closes[0]}], len(days), closes)}
    finally:
        watcher.stop.set()
        pool.shutdown(wait=False)


def bar_eq(b: Broker) -> float:
    return round(b.equity[-1][1], 5)
