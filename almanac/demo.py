"""Demo engine — 4 agents on the frozen notebook, replayed from precomputed `daily_decisions`.

  none       same prompt and signals, no lessons at all (memory-free baseline)
  plain      all relevant lessons, overall W/L
  plain_ctx  all relevant lessons + fingerprints, judges relevance itself
  almanac    lessons with a good conditional track record in markets like today

Each agent's daily answer is a target (IN / OUT) for tomorrow; the simulator fills at the next open and
charges the 0.05% cost when the target changes. Any window from 2016 works (no length cap); missing days are
computed on the fly. Each day's state is written to `market_state`; a change stream on it turns the stored
"lessons that apply" sets into "now apply / stopped applying" events.
"""
import math
import threading
import time
import uuid
from datetime import datetime

from almanac import config, db
from almanac.broker import Broker
from almanac.daily import AGENTS, load, trading_days
from almanac.data import _to_dt, bars_until
from almanac.lessons import coll as lessons_coll
from almanac.metrics import summarize

NOTEBOOK = config.LEARN_AGENT


def market_state():
    return db.db()["market_state"]


def window_days(start, end) -> list:
    if _to_dt(start) < _to_dt(config.DEMO_MIN_START):
        raise ValueError(f"demo windows start on or after {config.DEMO_MIN_START}")
    if _to_dt(end) > _to_dt(config.DEMO_END):
        raise ValueError(f"data ends on {config.DEMO_END}")
    days = trading_days(start, end)
    if not days:
        raise ValueError("no trading days in window")
    return days


def simulate(start, end, seed: int | None = None) -> dict:
    """Replay stored decisions through 4 brokers. Returns per-day records + end metrics."""
    days, market, dec = load(start, end, seed)
    bars = bars_until(days[-1], since=days[0])
    brokers = {a: Broker() for a in AGENTS}
    free = {a: Broker(cost=0.0) for a in AGENTS}  # zero-cost shadow brokers -> cost drag
    invested_days = {a: 0 for a in AGENTS}
    last_target = {a: None for a in AGENTS}  # confirmation rule: act only on the same target CONFIRM_DAYS in a row
    streak = {a: 0 for a in AGENTS}
    closes, records = [], []
    for d in days:
        key = d.to_pydatetime()
        b = bars.loc[d]
        for a, br in brokers.items():
            for x in (br, free[a]):
                x.on_open(d, float(b["open"]))
                x.on_close(d, float(b["close"]))
            invested_days[a] += br.invested
        closes.append(float(b["close"]))
        agents = {}
        for a, br in brokers.items():
            x = dec.get((a, key), {})
            target, was_in = x.get("target"), br.invested
            streak[a] = streak[a] + 1 if (target is not None and target == last_target[a]) else (1 if target else 0)
            last_target[a] = target
            confirmed = target if streak[a] >= config.CONFIRM_DAYS else None
            if confirmed == "IN":
                action = "stay_in" if was_in else "buy"
            elif confirmed == "OUT":
                action = "sell" if was_in else "stay_out"
            else:  # unconfirmed or no usable answer: keep the position
                action = "stay_in" if was_in else "stay_out"
            br.order({"buy": "buy", "sell": "sell"}.get(action, "hold"))
            free[a].order({"buy": "buy", "sell": "sell"}.get(action, "hold"))
            agents[a] = {"action": action, "target": target, "confirmed": confirmed is not None,
                         "reason": x.get("reason", ""),
                         "invested": was_in, "equity": round(br.equity[-1][1], 5),
                         "n_lessons": x.get("n_lessons", 0), "cited": x.get("cited", []),
                         "shown": x.get("shown", [])}
        m = market.get(key, {})
        records.append({"date": f"{d:%Y-%m-%d}",
                        "ohlc": {k: float(b[k]) for k in ("open", "high", "low", "close")},
                        "variables": m.get("variables", {}), "signals": m.get("signals", {}),
                        "description": m.get("description", ""), "coverage": m.get("coverage", {}),
                        "applies": m.get("applies", []), "agents": agents})
    bench = [c / closes[0] for c in closes]
    metrics = {}
    for a, br in brokers.items():
        metrics[a] = summarize([e for _, e in br.equity], br.trades, invested_days[a], closes)
        metrics[a]["cost_drag"] = float(free[a].equity[-1][1] - br.equity[-1][1])  # return lost to costs
    return {"days": records,
            "metrics": metrics,
            "benchmark": summarize(bench, [{"action": "buy", "price": closes[0]}], len(days), closes)}


def diff_events(prev: set | None, now: set, texts: dict) -> list:
    if prev is None:
        return [{"kind": "baseline", "text": f"{len(now)} lessons have a good record in markets like this"}]
    out = []
    entered, left = now - prev, prev - now
    if entered:
        out.append({"kind": "entered", "text": f"{len(entered)} lesson(s) now apply",
                    "lessons": [texts.get(i, "") for i in list(entered)[:3]]})
    if left:
        out.append({"kind": "left", "text": f"{len(left)} lesson(s) stopped applying",
                    "lessons": [texts.get(i, "") for i in list(left)[:3]]})
    return out


class EventWatcher(threading.Thread):
    """Change stream on market_state: diff each day's 'applies' set against the previous day."""

    def __init__(self, run_id: str, texts: dict, n: int):
        super().__init__(daemon=True)
        self.run_id, self.texts, self.n = run_id, texts, n
        self.events: dict[str, list] = {}
        self.ready, self.done = threading.Event(), threading.Event()

    def run(self):
        prev = None
        pipeline = [{"$match": {"operationType": "insert", "fullDocument.run_id": self.run_id}}]
        try:
            with market_state().watch(pipeline, max_await_time_ms=200) as stream:
                self.ready.set()
                t0 = time.time()
                while len(self.events) < self.n and time.time() - t0 < 15:
                    ch = stream.try_next()
                    if ch is None:
                        continue
                    doc = ch["fullDocument"]
                    now = set(doc.get("applies", []))
                    self.events[doc["day"]] = diff_events(prev, now, self.texts)
                    prev = now
        finally:
            self.ready.set()
            self.done.set()


def stream_demo(start, end, seed: int | None = None):
    """Yield SSE-ready messages: start, days (chunks paced to ~STREAM_SECONDS total), end."""
    t0 = time.time()
    sim = simulate(start, end, seed)
    recs = sim["days"]
    texts = {str(l["_id"]): l["text"] for l in lessons_coll().find(
        {"agent_id": NOTEBOOK, "learned_date": {"$lt": _to_dt(config.NOTEBOOK_CUTOFF)}}, {"text": 1})}

    # events via change stream: bulk-insert this window's market_state, the watcher diffs the 'applies' sets
    run_id = uuid.uuid4().hex[:12]
    watcher = EventWatcher(run_id, texts, len(recs))
    watcher.start()
    watcher.ready.wait(10)
    market_state().insert_many([{"run_id": run_id, "day": r["date"], "date": _to_dt(r["date"]),
                                 "variables": r["variables"], "applies": r["applies"], "at": datetime.utcnow()}
                                for r in recs])
    watcher.done.wait(15)
    prev, prev_unf = None, []
    for r in recs:  # fallback for any event the stream didn't deliver in time
        now = set(r["applies"])
        r["events"] = list(watcher.events.get(r["date"]) or diff_events(prev, now, texts))
        prev = now
        unf = r["coverage"].get("unfamiliar", [])
        if unf and unf != prev_unf:
            r["events"].append({"kind": "unfamiliar", "text": "unfamiliar conditions: " + ", ".join(unf)})
        prev_unf = unf
        r.pop("applies", None)
        for name, ag in r["agents"].items():
            if name != "almanac":  # keep almanac's shown ids so the UI can mark lessons hidden from it
                ag.pop("shown", None)

    yield {"type": "start", "n_days": len(recs), "prep_seconds": round(time.time() - t0, 1),
           "from_stream": len(watcher.events)}
    frames = min(len(recs), 140)
    size = math.ceil(len(recs) / frames)
    n_frames = math.ceil(len(recs) / size)
    for i in range(0, len(recs), size):
        yield {"type": "days", "days": recs[i:i + size]}
        time.sleep(config.STREAM_SECONDS / n_frames)
    yield {"type": "end", "days": len(recs), "metrics": sim["metrics"], "benchmark": sim["benchmark"]}
