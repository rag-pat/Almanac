# Almanac

**An AI trading agent whose memory knows when it applies.**

Almanac is an LLM trading agent that writes lessons from its own trades. Each lesson stores the market conditions it was learned in, plus a valid range for each condition. MongoDB Atlas only lets the agent retrieve lessons whose ranges contain today's conditions. Ranges widen when a lesson is rediscovered in new conditions, and lessons are retired when Bayesian evidence says they lose. We compare it live against two agents with plain memory.

Built for the MongoDB **Agents in Context** hackathon, Track 2: **Memory You Can Govern**.

---

## The problem

Agent memory is usually "store notes, retrieve the ones that sound similar." For a trading agent, that fails in a predictable way. A lesson like *"buying the dip after a sharp drop paid off"* sounds just as relevant in a calm 2017 pullback as in the March 2020 crash, but it was learned in one of those and is dangerous in the other.

Almanac makes memory **governed**. The database, not the LLM, decides which lessons the agent is allowed to see, based on whether today looks like the conditions each lesson was learned in.

## How it works

```
 Yahoo daily data ──► 8 market conditions per day (z-scored) ──► Atlas: bars, features
                                                                       │
 LEARNING RUN (2007-2015, weekly)                                      ▼
   retrieve lessons through the Atlas gate ──► Claude decides buy / sell / hold
   ──► broker fills at next open (5 bps) ──► 5 trading days later: score win / loss
   ──► Claude writes a one-sentence lesson ──► merge (widen box) or insert ──► retire losers
                                                                       │  frozen notebook
 DEMO (any window from 2016, daily)                                    ▼
   plain       all lessons that sound relevant
   plain_ctx   same lessons + where each was learned; judges relevance itself
   almanac     only lessons whose learned ranges contain today's conditions (6 of 8)
   ──► live in the browser over SSE, with memory events from a change stream
```

**Market conditions.** For each trading day, using past data only: Parkinson volatility (20d), trend vs the 200-day average, drawdown from the 1-year high, volume vs the last 60 days, VIX, the 20-day change in the 10-year yield, the yield curve (10y minus 3m) and credit stress (corporate vs Treasury bonds, relative to their 60-day average). Each is converted to a z-score using means and standard deviations from 2007-2012 only, frozen as constants.

**A lesson** is one sentence the agent wrote about its own trade, plus:
- `fp`: the 8 z-scores on the decision day (its fingerprint)
- `lo_<var>` / `hi_<var>`: a valid range per variable, starting at fp ± 0.5
- `wins` / `losses`: the record of its advice. New lessons start at 1W/0L, a rediscovery adds a win, and being cited in a winning or losing decision adds a win or loss.
- `status` (active or retired), `learned_date`, `history` of every change

**The Atlas gate** is one aggregation pipeline:
1. `$vectorSearch` (exact) on the embedding of today's description, filtered to active lessons learned before the window
2. keep lessons with cosine ≥ 0.50, best first, at most 30. **This is what the plain agents get.**
3. `$addFields` counts how many of the 8 variables fall inside the lesson's `[lo, hi]` range
4. **Almanac only:** `$match` count ≥ 6

A second aggregation reports **unfamiliar conditions**: variables where today's value is outside every lesson's range.

**Learning rules.**
- **Rediscovery:** a new lesson with cosine ≥ 0.85 to an active lesson **with the same action and outcome** is merged. The old lesson's box widens to cover the new fingerprint ± 0.5.
- **Retirement:** with θ ~ Beta(1 + wins, 1 + losses), a lesson is retired when n ≥ 5 and P(θ < 0.5) > 0.9.
- **Exploration:** 10% of weeks force a different action, so the agent also learns from trades it wouldn't have chosen.
- **Holds:** a hold is scored by the position it keeps (invested means long, cash means out).

## MongoDB features used

| Feature | Where |
|---|---|
| **Atlas Vector Search** (exact ENN) with 19 filter fields | the retrieval gate (`lessons.py`) |
| **Aggregation pipeline** | range counting, the ≥ 6-of-8 rule and unfamiliar-condition detection, all server-side |
| **`$jsonSchema` validation** | the `lessons` collection rejects any lesson missing a field or with bad values |
| **Multi-document transactions** | each resolved trade updates the outcome, cited lessons, merges or inserts and retirements atomically (`learn.py`) |
| **Change streams** | `market_state` inserts are turned into "lessons entered / left range" events for the UI (`demo.py`) |
| **Atlas AI Model APIs (Voyage `voyage-4`)** | embeddings for lessons and day descriptions |
| Unique compound indexes | `bars (symbol, date)`, `features (date)` |

Collections: `bars`, `features`, `lessons`, `outcomes`, `memory_events`, `market_state`.

## No lookahead, enforced in code

- Every price read goes through `bars_until(date)`, and tests confirm it never returns later rows.
- Z-score constants come from 2007-2012 only, and a test checks the constants match.
- Features on day *t* are identical whether computed on all data or data cut at *t* (tested on 5 dates).
- Fingerprints come from the decision day. Trades fill at the next day's open.
- Demo agents only see lessons with `learned_date` before the window starts (the notebook ends in 2015, and demos start in 2016).
- The LLM never sees real dates, the ticker or real prices. It sees "Asset X, day 13826" with prices rebased to 100. A test scans prompts for years, months, "SPY" and real prices.

## Proof: counterfactual

`GET /counterfactual?date=…&start=…&lesson_id=…` rebuilds the Almanac prompt for a demo day. It asks Claude 5 times with the cited lesson and 5 times without it (fresh calls, no cache), then reports the **flip rate**: how often removing the lesson changes the decision. The with-lesson runs show the model's baseline noise. In the UI, every lesson Almanac cites has a "Test: decide again without this lesson" button.

## Results

<!-- RESULTS -->
*Filled in after the learning run and preset runs.*

## Run it

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv -r requirements.txt
cp .env.example .env   # MONGODB_URI, VOYAGE_API_KEY (Atlas AI Model APIs), CLAUDE_API_KEY

python -m almanac.data download     # 1. prices for SPY, VIX, TNX, IRX, LQD, IEF -> data/*.csv
python -m almanac.data load         # 2. bars + features -> Atlas
python -m almanac.lessons setup     # 3. lessons collection, validator, vector index
python -m almanac.learn             # 4. learning run 2007-2015 (builds the notebook)
python -m almanac.warm              # 5. optional: run the presets once so the demo replays from cache
uvicorn almanac.server:app --port 8000   # 6. open http://localhost:8000
python -m pytest tests/             # tests
```

Every LLM and embedding call is cached on disk (keyed by a hash of the input), so reruns are instant and free.

## Stack

Python 3.11, pymongo, FastAPI, pandas, numpy, scipy, yfinance. **Claude Haiku 5.5** (structured JSON output) for decisions and lesson writing. **Voyage `voyage-4`** through Atlas AI Model APIs for embeddings. MongoDB Atlas. One static HTML page with TradingView Lightweight Charts.

## Built today

Everything in this repo was built during the hackathon on 2026-11-14:

- Data pipeline: 20 years of daily data for 6 symbols, 8 condition variables, z-scoring frozen to 2007-2012, and no-lookahead tests. Credit and rates were redesigned mid-build after we found they drifted out of range for years.
- The lessons notebook in Atlas: schema validation, a vector index with 19 filter fields, and the single-pipeline retrieval gate with the 6-of-8 range rule and unfamiliar-condition detection.
- Similarity cutoffs calibrated for Voyage. This is also where we found Atlas reports cosine as (1 + cos) / 2.
- A Claude trading agent with a disguised prompt, three memory modes and safety checks (made-up lesson IDs dropped, bad output becomes hold).
- Broker and outcome scoring. The learning run writes, merges, widens and retires lessons in transactions with a full event log.
- The 3-agent demo engine with a change stream on `market_state`, SSE streaming, the front end, metrics and the counterfactual flip-rate test.
- 90+ tests.
