"""Performance metrics for an equity curve (daily closes) + the broker's trade list."""
import numpy as np

TRADING_DAYS = 252


def summarize(equity: list[float], trades: list[dict], invested_days: int, closes: list[float] | None = None) -> dict:
    eq = np.asarray(equity, dtype=float)
    if len(eq) < 2:
        return {}
    rets = np.diff(eq) / eq[:-1]
    n = len(rets)
    total = eq[-1] / eq[0] - 1
    ann = (1 + total) ** (TRADING_DAYS / n) - 1
    sd = rets.std(ddof=1) if n > 1 else 0.0
    down = rets[rets < 0]
    dsd = np.sqrt(np.mean(down ** 2)) if len(down) else 0.0
    peak = np.maximum.accumulate(eq)
    mdd = float((eq / peak - 1).min())

    # hit rate over round trips (buy fill -> sell fill; an open trade is marked to the last close)
    wins, rounds, entry = 0, 0, None
    for t in trades:
        if t["action"] == "buy":
            entry = t["price"]
        elif t["action"] == "sell" and entry is not None:
            rounds += 1
            wins += t["price"] > entry
            entry = None
    if entry is not None and closes:
        rounds += 1
        wins += closes[-1] > entry

    return {
        "total_return": float(total),
        "sharpe": float(rets.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0,
        "sortino": float(rets.mean() / dsd * np.sqrt(TRADING_DAYS)) if dsd > 0 else 0.0,
        "max_drawdown": mdd,
        "calmar": float(ann / abs(mdd)) if mdd < 0 else 0.0,
        "hit_rate": float(wins / rounds) if rounds else None,
        "trades": len(trades),
        "time_invested": float(invested_days / len(eq)),
    }
