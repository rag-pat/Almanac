# Almanac

**An AI trading agent whose memory knows when it applies.**

Almanac is an LLM trading agent that writes lessons from its own trades. Each lesson keeps an evidence log of the market conditions it was tested in and whether it worked. MongoDB Atlas computes, inside the retrieval pipeline, how well each lesson's advice has worked **in markets like today's**, and only shows the agent lessons proven there. Lessons merge when rediscovered and are retired when Bayesian evidence says they lose. We compare it against an agent with no memory and two agents with plain memory, on any window from 2016 to 2024.

Built for the MongoDB **Agents in Context** hackathon, Track 2: **Memory You Can Govern**.

---

## The problem

Agent memory is usually "store notes, retrieve the ones that sound similar." For a trading agent, that fails in a predictable way. A lesson like *"buying the dip after a sharp drop paid off"* sounds just as relevant in a calm 2017 pullback as in the March 2020 crash, but it was learned in one of those and is dangerous in the other.

Almanac makes memory **governed**. The database, not the LLM, decides which lessons the agent is allowed to see, based on how each lesson has actually performed in conditions like today's.

## How it works

```
 Yahoo daily data ──► 8 market conditions per day (z-scored) ──► Atlas: bars, features
                                                                       │
 LEARNING RUN (2007-2015, weekly)                                      ▼
   retrieve lessons through the Atlas gate ──► Claude decides buy / sell / hold
   ──► broker fills at next open (5 bps) ──► 5 trading days later: score win / loss
   ──► Claude writes a one-sentence lesson ──► merge (widen box) or insert ──► retire losers
                                                                       │  frozen notebook
 DEMO (every trading day 2016-2024 precomputed, 4 agents)              ▼
   none        same prompt and signals, no lessons at all
   plain       all lessons that sound relevant, with overall W/L
   plain_ctx   same lessons + where each was learned; judges relevance itself
   almanac     only lessons proven in markets like today (conditional record)
   each answers "target for tomorrow: IN or OUT" ──► simulator (next-open fills, 5 bps,
   act only on the same target 2 days in a row) ──► replayed in the browser over SSE in ~7 s
```

**Market conditions.** For each trading day, using past data only: Parkinson volatility (20d), trend vs the 200-day average, drawdown from the 1-year high, volume vs the last 60 days, VIX, the 20-day change in the 10-year yield, the yield curve (10y minus 3m) and credit stress (corporate vs Treasury bonds, relative to their 60-day average). Each is converted to a z-score using means and standard deviations from 2007-2012 only, frozen as constants.

**A lesson** is one sentence the agent wrote about its own trade, plus:
- `fp`: the 8 z-scores on the decision day (its fingerprint)
- `rec`: what the lesson recommends, from the trade it was written about: "buying won" → IN, "buying lost" → OUT, "selling won" → OUT, "selling lost" → IN.
- `evidence`: `[{date, z, win}]`, one entry per time the lesson was tested: its birth trade, every later trade that rediscovered it, and every decision that cited it. Each is dated when the result was known, `z` is that decision day's 8 z-scores, and `win` says whether the lesson's **own recommendation** was right over those 5 days.
- `wins` / `losses`: the totals of that evidence, so the W/L line and the conditional record describe the same history.
- `lo_<var>` / `hi_<var>`: a box per variable, starting at fp ± 0.5 and widening on rediscovery. Boxes only drive the "unfamiliar conditions" flag.
- `status` (active or retired), `learned_date`, `history` of every change

**The Atlas gate** is one aggregation pipeline:
1. `$vectorSearch` (exact) on the embedding of today's description, filtered to active lessons learned before the window
2. keep lessons with cosine ≥ 0.50, best first, at most 30. **This is what the plain agents get.**
3. `$filter` the evidence log to entries dated **before the window start**, then compute with `$map` / `$reduce`:
   - `d_i` = Euclidean distance between today's 8 z-scores (averaged over the last 5 days, so the gate doesn't flicker) and entry *i*
   - `w_i = exp(-d_i² / 2h²)` with h = 1.5
   - **A** = (1 + Σ wᵢ·winᵢ) / (2 + Σ wᵢ), the smoothed win rate in markets like today
   - **n_eff** = Σ wᵢ, how much similar evidence there is
4. A confidence band on that win rate, from a normal approximation to the kernel-weighted Beta posterior: `lcb, ucb = A ∓ 1.28·√(A(1−A)/(n_eff+3))`. Each lesson is classed as **proven here** (90% confident it beats a coin flip: lcb > 0.5, n_eff ≥ 1), **mixed**, **unproven here** (n_eff < 1) or **proven to fail here** (ucb < 0.5). With lots of similar evidence a lesson needs to be right about 62% of the time to count as proven; with little, about 72%.
5. **Almanac only:** keep lessons proven here, ranked by A, at most 30. On days with none, Almanac decides from the signals alone, like the no-memory agent.

The Almanac prompt shows each lesson as *"[L1] proven here: worked 84% in markets like today (4.3 similar uses)"*. The plain agents see the overall W/L. A second aggregation counts the classes over the whole notebook and reports **unfamiliar conditions**: variables where today's value is outside every lesson's box.

Evidence for the citations comes from one aggregation (`$lookup` lessons → outcomes). Rediscoveries are recovered by replaying the learner's lesson writing from the LLM disk cache and matching the text (372 of 372 matched, with no model calls). New trades `$push` evidence inside the same transaction that scores them.

**Learning rules.**
- **Rediscovery:** a new lesson with cosine ≥ 0.85 to an active lesson **with the same action and outcome** is merged. The old lesson's box widens to cover the new fingerprint ± 0.5.
- **Retirement:** with θ ~ Beta(1 + wins, 1 + losses), a lesson is retired when n ≥ 5 and P(θ < 0.5) > 0.9.
- **Exploration:** 10% of weeks force a different action, so the agent also learns from trades it wouldn't have chosen.
- **Holds:** a hold is scored by the position it keeps (invested means long, cash means out).

## MongoDB features used

| Feature | Where |
|---|---|
| **Atlas Vector Search** (exact ENN) with 19 filter fields | the retrieval gate (`lessons.py`) |
| **Aggregation pipeline** (`$filter`, `$map`, `$reduce`, `$exp`) | the conditional track record (A, n_eff) computed per lesson after `$vectorSearch`, plus unfamiliar-condition detection |
| **`$lookup` + `$merge`** | evidence backfill from outcomes into lessons (`evidence.py`) |
| **`$jsonSchema` validation** | the `lessons` collection rejects any lesson missing a field or with bad values |
| **Multi-document transactions** | each resolved trade updates the outcome, cited lessons, merges or inserts and retirements atomically (`learn.py`) |
| **Change streams** | `market_state` inserts are turned into "lessons now apply / stopped applying" events for the UI (`demo.py`) |
| **Precomputed decisions** | `daily_decisions` holds every agent's answer for every trading day 2016-2024, keyed by (agent, date, seed) and tagged with the notebook version |
| **Atlas AI Model APIs (Voyage `voyage-4`)** | embeddings for lessons and day descriptions |
| Unique compound indexes | `bars (symbol, date)`, `features (date)` |

Collections: `bars`, `features`, `lessons`, `outcomes`, `memory_events`, `market_state`, `daily_decisions`, `sweeps`.

## No lookahead, enforced in code

- Every price read goes through `bars_until(date)`, and tests confirm it never returns later rows.
- Z-score constants come from 2007-2012 only, and a test checks the constants match.
- Features on day *t* are identical whether computed on all data or data cut at *t* (tested on 5 dates).
- Fingerprints come from the decision day. Trades fill at the next day's open.
- Demo agents only see lessons with `learned_date` before the window starts (the notebook ends in 2015, and demos start in 2016). Only evidence dated before the window start counts toward A, and a test checks that later evidence never changes it.
- The LLM never sees real dates, the ticker or real prices. It sees "Asset X, day 13826" with prices rebased to 100. A test scans prompts for years, months, "SPY" and real prices.
- The demo prompt states the signals as facts (the RSI(2) value, above or below the 200-day average) without saying which way they usually resolve, so any edge has to come from memory.

## Proof: counterfactual

`GET /counterfactual?date=…&lesson_id=…` rebuilds the Almanac prompt for a demo day. It asks Claude 5 times with the cited lesson and 5 times without it (fresh calls, no cache), then reports the **flip rate**: how often removing the lesson changes the decision. The with-lesson runs show the model's baseline noise. In the UI, every lesson Almanac cites has a "Test: decide again without this lesson" button.

## Results

<!-- RESULTS -->
**The notebook.** The learning run (2007-2015, 454 weekly decisions, 46 forced explorations) wrote **81 lessons** and merged **372 rediscoveries** into them. Their evidence log has **1,447 entries** (median 3 per lesson). No lesson qualifies for retirement: the weakest one with at least 5 tests is still right 55% of the time.

**12 windows of about 60 trading days, 2016-2024** (same model, prompt, signals and costs; only memory differs):

| Agent | Mean return | Mean max drawdown | Mean Sharpe | Beat plain |
|---|---|---|---|---|
| No memory | −2.06% | −8.13% | −0.11 | 7 of 12 |
| Plain memory | −3.19% | −10.07% | −0.23 | – |
| Plain + context | −2.84% | −9.48% | −0.19 | 4 of 12 |
| Almanac | −2.30% | −8.40% | −0.05 | 6 of 12 |
| Buy and hold | −1.84% | −10.47% | 0.18 | 10 of 12 |

**The three presets:**

| Window | No memory | Plain | Plain + context | Almanac | Buy and hold |
|---|---|---|---|---|---|
| Late 2018 | **−12.7%** | −19.6% | −16.8% | −22.7% | −19.2% |
| 2020 crash | **−14.3%** | −19.0% | −22.4% | −17.7% | −15.0% |
| 2022 bear | −14.2% | −17.3% | −14.6% | **−4.6%** | −11.9% |
| Feb 2019 - Feb 2020 (calm) | +5.8% | +16.5% | +19.1% | **+19.7%** | +21.1% |

Seeded reruns (3 extra samples per agent) on the presets: in the 2022 bear Almanac averages −7.5% ± 0.5 against −14% to −18% for the other agents.

**What this shows, honestly.** The trading results are mixed. Every agent holds the same single index, long only, fully in or out, so in calm uptrends all of them track buy and hold and the differences are small. Almanac's clearest win is the 2022 bear. When no lesson is proven for today's conditions, Almanac decides from the raw signals alone, so in most of the 2020 crash and late 2018 it had nothing to lean on. The individual calls are still telling. On 2020-03-05 plain memory bought on a "buying a deeply oversold dip paid off" lesson that Almanac hid as unproven in crash conditions, and the market fell 18% over the next five days. On 2022-08-26, removing one lesson from Almanac's prompt flipped its answer from OUT (4 of 5 fresh runs) to IN (5 of 5). All results come from one sample per prompt plus seeded reruns on the presets (`python -m almanac.sweep summary`).

**How we got here.** Two QA passes found the agents were following the 200-day average because the prompt told them what it meant, and that Almanac's gate had too little evidence to work (median 1 entry per lesson). We removed the signal interpretation from the prompt, rebuilt the evidence log to include rediscoveries and score each lesson's own recommendation, and restricted Almanac to lessons proven here. A first version defined "proven" as A ≥ 0.55. That let coin-flip lessons through, such as an "RSI overbought, wait for a pullback" lesson at 58% on 26 similar uses that kept pulling Almanac out of the 2019 rally. We replaced it with the 90% confidence bound. That fixed the calm-market exits (+13.0% → +19.7% in Feb 2019 - Feb 2020) and helped in the 2022 bear, but cost the crash results the weaker rule had (2020 crash +2.4% → −17.7%). The 12-window mean went from −0.29% to −2.30%. We kept the principled rule rather than tune further on these windows.

## Run it

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv -r requirements.txt
cp .env.example .env   # MONGODB_URI, VOYAGE_API_KEY (Atlas AI Model APIs), CLAUDE_API_KEY

python -m almanac.data download     # 1. prices for SPY, VIX, TNX, IRX, LQD, IEF -> data/*.csv
python -m almanac.data load         # 2. bars + features -> Atlas
python -m almanac.lessons setup     # 3. lessons collection, validator, vector index
python -m almanac.learn             # 4. learning run 2007-2015 (builds the notebook)
python -m almanac.evidence backfill # 4b. evidence logs + recounted records for the conditional track record
python -m almanac.daily             # 5. precompute every agent for every trading day 2016-2024 (~15 min)
python -m almanac.sweep canonical   # 6. 12-window evaluation (python -m almanac.sweep variance for spread)
uvicorn almanac.server:app --port 8000   # 7. open http://localhost:8000 (any window replays in ~7 s)
python -m almanac.diagnose 2016-01-04 2017-12-22   # per-agent trades, time invested, cost drag
python -m pytest tests/             # tests
```

Every LLM and embedding call is cached on disk (keyed by a hash of the input), so reruns are instant and free.

## Stack

Python 3.11, pymongo, FastAPI, pandas, numpy, scipy, yfinance. **Claude Haiku 5.5** (structured JSON output) for decisions and lesson writing. **Voyage `voyage-4`** through Atlas AI Model APIs for embeddings. MongoDB Atlas. One static HTML page with TradingView Lightweight Charts.

## Built today

Everything in this repo was built during the hackathon on 2026-11-14:

- Data pipeline: 20 years of daily data for 6 symbols, 8 condition variables, z-scoring frozen to 2007-2012, and no-lookahead tests. Credit and rates were redesigned mid-build after we found they drifted out of range for years.
- The lessons notebook in Atlas: schema validation, a vector index with 19 filter fields, and the single-pipeline retrieval gate. It started as a 6-of-8 range box and was upgraded mid-day to a conditional track record (kernel-weighted win rate in markets like today) computed inside the aggregation.
- Similarity cutoffs calibrated for Voyage. This is also where we found Atlas reports cosine as (1 + cos) / 2.
- A Claude trading agent with a disguised prompt, four memory modes and safety checks (made-up lesson IDs dropped, bad output means no change).
- Broker and outcome scoring. The learning run writes, merges, widens and retires lessons in transactions with a full event log.
- The 4-agent demo engine (no memory, plain, plain + context, Almanac): position-independent daily targets precomputed for 2016-2024, a simulator with a 2-day confirmation rule and cost-drag accounting, a change stream on `market_state`, SSE replay in about 7 seconds for any window, the front end, metrics, a 12-window sweep and the counterfactual flip-rate test.
- Two QA passes and the fixes they led to (neutral prompt, rebuilt evidence, recommendation-based scoring, proven-only gate with a confidence bound), plus a "vs buy and hold" view on the equity chart.
- 103 tests. Demo script in `DEMO.md`.
