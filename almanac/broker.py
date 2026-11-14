"""M4: broker (fills, costs, equity) + outcomes (score each buy/sell after HORIZON trading days).

Timeline for one day t:  on_open(t) executes yesterday's order at t's open  ->  on_close(t) marks equity
                         ->  agent decides at t's close  ->  order(action) queues it for t+1's open.
Every price comes from bars_until(), so nothing can see past the current day.
"""
from datetime import datetime

from pymongo import ASCENDING

from almanac import config, db
from almanac.data import _to_dt, bars_until


class Broker:
    """Long-only, all-in or all-out. Equity starts at 1.0."""

    def __init__(self, cost: float = config.COST):
        self.cost = cost
        self.invested = False
        self.units = 0.0     # asset units held when invested
        self.cash = 1.0      # cash when out
        self.pending = None  # "buy" / "sell" to execute at next open
        self.days_in_state = 0
        self.trades: list[dict] = []
        self.equity: list[tuple] = []  # (date, equity)

    @property
    def position(self) -> dict:
        return {"in": self.invested, "days": self.days_in_state}

    def order(self, action: str):
        """Queue a decision made at today's close. Redundant orders are ignored (no trade, no cost)."""
        if (action == "buy" and not self.invested) or (action == "sell" and self.invested):
            self.pending = action

    def on_open(self, date, open_px: float):
        if not self.pending:
            return
        if self.pending == "buy":
            self.units = self.cash * (1 - self.cost) / open_px
            self.cash, self.invested = 0.0, True
        else:
            self.cash = self.units * open_px * (1 - self.cost)
            self.units, self.invested = 0.0, False
        self.trades.append({"date": _to_dt(date), "action": self.pending, "price": open_px})
        self.pending, self.days_in_state = None, 0

    def on_close(self, date, close_px: float) -> float:
        eq = self.units * close_px if self.invested else self.cash
        self.equity.append((_to_dt(date), eq))
        self.days_in_state += 1
        return eq

    def step(self, date) -> dict:
        """Process day `date`: fill any pending order at its open, mark equity at its close."""
        bar = bars_until(date, n=1)
        assert bar.index[-1] == _to_dt(date), f"{date} is not a trading day"
        row = bar.iloc[-1]
        self.on_open(date, float(row["open"]))
        eq = self.on_close(date, float(row["close"]))
        return {"date": _to_dt(date), "open": float(row["open"]), "high": float(row["high"]),
                "low": float(row["low"]), "close": float(row["close"]), "equity": eq}


# ---------- outcomes ----------

def outcomes():
    c = db.db()["outcomes"]
    c.create_index([("agent_id", ASCENDING), ("status", ASCENDING), ("decision_date", ASCENDING)])
    return c


def record_decision(agent_id: str, date, action: str, cited: list, fp: dict, description: str,
                    extra: dict | None = None):
    """Store a pending outcome for a buy/sell decided at `date`'s close. Holds are not scored."""
    if action not in ("buy", "sell"):
        return None
    close = float(bars_until(date, n=1)["close"].iloc[-1])
    doc = {"agent_id": agent_id, "decision_date": _to_dt(date), "action": action, "cited": cited,
           "fp": fp, "description": description, "price": close, "status": "pending",
           "created_at": datetime.utcnow(), **(extra or {})}
    return outcomes().insert_one(doc).inserted_id


def score(action: str, r: float, cost: float = config.COST) -> str | None:
    if action == "buy":
        return "win" if r > cost else "loss"
    if action == "sell":
        return "win" if r < -cost else "loss"
    return None


def find_due(agent_id: str, today, horizon: int = config.HORIZON) -> list[tuple[dict, dict]]:
    """Pending outcomes that are >= `horizon` trading days old as of `today`, with their computed result
    (not yet written). R = close(t + horizon) / close(t) - 1, using only bars dated <= today."""
    today = _to_dt(today)
    bars = db.db()["bars"]
    due = []
    for o in outcomes().find({"agent_id": agent_id, "status": "pending", "decision_date": {"$lt": today}}
                             ).sort("decision_date", ASCENDING):
        later = list(bars.find({"symbol": "SPY", "date": {"$gt": o["decision_date"], "$lte": today}},
                               {"close": 1, "date": 1}).sort("date", ASCENDING).limit(horizon))
        if len(later) < horizon:
            continue
        exit_bar = later[horizon - 1]
        r = exit_bar["close"] / o["price"] - 1
        due.append((o, {"status": "resolved", "R": r, "outcome": score(o["action"], r),
                        "resolved_date": exit_bar["date"]}))
    return due


def resolve_due(agent_id: str, today, horizon: int = config.HORIZON) -> list[dict]:
    """find_due + write the results (used when no lesson updates need to share the transaction)."""
    done = []
    for o, upd in find_due(agent_id, today, horizon):
        outcomes().update_one({"_id": o["_id"]}, {"$set": upd})
        done.append({**o, **upd})
    return done
