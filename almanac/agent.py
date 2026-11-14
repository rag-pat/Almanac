"""M3: the trading agent — builds a disguised prompt, asks the LLM, returns a safe decision.

decide(date, position, lessons, mode) -> {"action", "cited", "reason", ...}
modes: "plain" (lesson text), "plain_ctx" (text + fingerprints, judge relevance yourself), "almanac" (in-range only)
"""
from almanac import db
from almanac.data import bars_until, features_until
from almanac.features import VARS, describe
from almanac.llm import LLMRefusal, llm_json

MODES = ("plain", "plain_ctx", "almanac", "none")  # none = same prompt and signals, no memory at all
DAY_OFFSET = 10_000  # "day 13412" — never looks like a year
REASON_MAX_WORDS = 25

LABELS = {"vol": "volatility", "trend": "trend", "drawdown": "drawdown", "volume": "volume",
          "vix": "fear", "rates": "yield change", "curve": "yield curve", "credit": "credit"}

SYSTEM = (
    "You trade one US asset, long only, fully in or fully out. "
    "buy = go 100% invested, sell = go to 0% (cash), hold = keep your current position. "
    "You decide at today's close; the trade fills at the next day's open; each trade costs 0.05%. "
    "You have two signals: RSI(2), where low values mean a sharp short-term dip that often bounces, "
    "and the 200-day average, where price below it means a downtrend. "
    "Past lessons are notes from your own earlier trades, each with its track record. "
    "Use a lesson only if it applies to today. Trust lessons with better records more. "
    "If no lesson applies, decide from the signals alone. Cite only lesson IDs shown. "
    f"Keep the reason under {REASON_MAX_WORDS} words."
)
# Demo agents are position-independent: they pick tomorrow's target position; the simulator charges the cost
SYSTEM_TARGET = (
    "You manage one US asset, long only. Each day, at the close, you choose your target position for tomorrow: "
    "IN = fully invested, OUT = in cash. The position changes at the next day's open, and every switch costs 0.05%. "
    "You are judged on the next week's return (about 5 trading days), not the long run, so short-term moves "
    "matter. You can change your mind on any day. "
    "You have two signals: RSI(2), where low values mean a sharp short-term dip that often bounces, "
    "and the 200-day average, where price below it means a downtrend. "
    "Past lessons are notes from your own earlier trades, each with its track record. "
    "Use a lesson only if it applies to today. Trust lessons with better records more. "
    "If no lesson applies, decide from the signals alone. Cite only lesson IDs shown. "
    f"Keep the reason under {REASON_MAX_WORDS} words."
)
CTX_NOTE = ("Each lesson shows the market numbers (z-scores) on the day it was learned. "
            "Compare them with today's numbers and judge for yourself whether each lesson applies.")


def _z_line(z: dict) -> str:
    return ", ".join(f"{LABELS[v]} {z[v]:+.1f}" for v in VARS)


def _day_number(date) -> int:
    from almanac.data import _to_dt
    return DAY_OFFSET + db.db()["bars"].count_documents({"symbol": "SPY", "date": {"$lte": _to_dt(date)}})


def build_prompt(date, position: dict | None, lessons: list[dict], mode: str, feat: dict | None = None,
                 ctx: dict | None = None):
    """Returns (user_prompt, id_map short->real). Nothing here may reveal the date, ticker or real prices.
    position=None -> position-independent prompt (asks for tomorrow's target). ctx may carry precomputed
    'closes' (last 20 closes, oldest first) and 'day_number' to avoid per-call queries."""
    assert mode in MODES, mode
    ctx = ctx or {}
    feat = feat or features_until(date, n=1).iloc[-1].to_dict()
    z = {v: float(feat[f"z_{v}"]) for v in VARS}
    closes = list(ctx["closes"]) if "closes" in ctx else list(bars_until(date, n=20)["close"])
    rebased = ", ".join(f"{c / closes[0] * 100:.1f}" for c in closes)
    day_no = ctx.get("day_number") or _day_number(date)
    if position is None:
        head = f"Asset X, day {day_no}."
    else:
        state = "IN (fully invested)" if position["in"] else "OUT (in cash)"
        head = f"Asset X, day {day_no}. You are currently {state} for {position.get('days', 0)} days."

    lines = [
        head,
        "",
        f"Last {len(closes)} closes (rebased to 100): {rebased}",
        f"Conditions: {describe(z)}",
        f"Signals: RSI(2) = {feat['rsi2']:.0f}. Price is {'ABOVE' if feat['above200'] else 'BELOW'} its 200-day average.",
    ]
    if mode == "plain_ctx":
        lines += ["", CTX_NOTE, f"Today's numbers: {_z_line(z)}"]

    id_map = {}
    if mode == "none":
        lessons = []
        lines += ["", "No past lessons available."]
    else:
        lines += ["", "Past lessons:" if lessons else "Past lessons: none apply."]
    for i, les in enumerate(lessons, 1):
        sid = f"L{i}"
        id_map[sid] = str(les["_id"])
        if mode == "almanac" and "A" in les:  # conditional record: how it did in markets like today
            line = f"[{sid}] worked {les['A']:.0%} in markets like today ({les['n_eff']:.1f} similar uses) {les['text']}"
        else:
            line = f"[{sid}] ({les['wins']}W/{les['losses']}L) {les['text']}"
        if mode == "plain_ctx":
            line += f"\n     learned at: {_z_line(les['fp'])}"
        lines.append(line)
    lines += ["", "Choose your target position for tomorrow: IN or OUT." if position is None
              else "Decide: buy, sell or hold."]
    return "\n".join(lines), id_map


def _schema(short_ids: list[str]) -> dict:
    cited = {"type": "array", "items": {"type": "string", "enum": short_ids} if short_ids else {"type": "string"}}
    return {"type": "object",
            "properties": {"action": {"type": "string", "enum": ["buy", "sell", "hold"]},
                           "cited": cited, "reason": {"type": "string"}},
            "required": ["action", "cited", "reason"], "additionalProperties": False}


def decide(date, position: dict, lessons: list[dict], mode: str, feat: dict | None = None,
           fresh: bool = False, seed: int | None = None) -> dict:
    user, id_map = build_prompt(date, position, lessons, mode, feat)
    try:
        out = llm_json(SYSTEM, user, _schema(list(id_map)), use_cache=not fresh, seed=seed)
    except (LLMRefusal, ValueError, KeyError, StopIteration) as e:  # bad/blocked output -> do nothing
        out = {"action": "hold", "cited": [], "reason": f"fallback: {type(e).__name__}"}

    action = out.get("action") if out.get("action") in ("buy", "sell", "hold") else "hold"
    if (action == "buy" and position["in"]) or (action == "sell" and not position["in"]):
        action = "hold"  # already there: no trade, no cost
    cited_short = [c for c in out.get("cited", []) if c in id_map]  # drop invented IDs
    reason = " ".join(str(out.get("reason", "")).split()[:REASON_MAX_WORDS])
    return {"action": action, "raw_action": out.get("action"), "cited": [id_map[c] for c in cited_short],
            "cited_short": cited_short, "reason": reason, "n_lessons": len(lessons), "prompt": user}


def _target_schema(short_ids: list[str]) -> dict:
    cited = {"type": "array", "items": {"type": "string", "enum": short_ids} if short_ids else {"type": "string"}}
    return {"type": "object",
            "properties": {"target": {"type": "string", "enum": ["IN", "OUT"]}, "cited": cited,
                           "reason": {"type": "string"}},
            "required": ["target", "cited", "reason"], "additionalProperties": False}


def decide_target(date, lessons: list[dict], mode: str, feat: dict | None = None, ctx: dict | None = None,
                  fresh: bool = False, seed: int | None = None) -> dict:
    """Position-independent demo decision: tomorrow's target, IN or OUT. A blocked/broken answer -> target None
    (the simulator keeps the current position). API errors (rate limits) propagate so callers can retry."""
    user, id_map = build_prompt(date, None, lessons, mode, feat, ctx)
    try:
        out = llm_json(SYSTEM_TARGET, user, _target_schema(list(id_map)), use_cache=not fresh, seed=seed)
    except (LLMRefusal, ValueError, KeyError, StopIteration) as e:
        out = {"target": None, "cited": [], "reason": f"fallback: {type(e).__name__}"}
    target = out.get("target") if out.get("target") in ("IN", "OUT") else None
    cited_short = [c for c in out.get("cited", []) if c in id_map]
    reason = " ".join(str(out.get("reason", "")).split()[:REASON_MAX_WORDS])
    return {"target": target, "cited": [id_map[c] for c in cited_short], "cited_short": cited_short,
            "reason": reason, "n_lessons": 0 if mode == "none" else len(lessons)}
