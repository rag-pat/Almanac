# Almanac

**Agent memory that knows when it applies.**

Almanac is an LLM trading agent that writes lessons from its own trades and keeps an evidence log for each one: the market conditions every time the lesson was tested, and whether its advice was right. When the agent asks for lessons, MongoDB Atlas works out how each one has performed *in markets like today's* and only hands over the ones it can trust. The database decides what the agent remembers, not the model.

Built in one day at the MongoDB **Agents in Context** hackathon (Toronto, November 2026), Track 2: *Memory You Can Govern*.

---

## Why

Agent memory usually means "retrieve notes that sound similar." That breaks when context matters. A lesson like *"buying the dip after a sharp drop paid off"* reads just as relevant in a calm pullback as in a market crash, but it was learned in one and is dangerous in the other. Almanac attaches evidence to every memory and gates retrieval on it.

## How it works

```
daily market data ─► 8 condition variables per day (z-scored) ─► Atlas
                                                                    │
learning run (2007-2015, weekly)                                    ▼
  retrieve lessons ─► Claude decides ─► trade scored after 5 days
  ─► Claude writes a one-sentence lesson ─► merge into a similar lesson or store a new one
  ─► every outcome is added to the lessons' evidence logs (one transaction)
                                                                    │  notebook frozen
demo (any window 2016-2024, daily)                                  ▼
  4 agents, same model and prompt, different memory ─► simulator ─► browser
```

**Market conditions.** Each trading day is described by eight variables computed from past data only: volatility, trend vs the 200-day average, drawdown, volume, VIX, change in the 10-year yield, yield curve and credit stress. Each is z-scored against 2007-2012 statistics frozen as constants.

**Lessons.** A lesson is a sentence plus the conditions it was learned in, what it recommends (IN or OUT), and an evidence log `[{date, z, win}]`. Rediscovering a lesson in new conditions adds evidence; so does every decision that cites it. Lessons are retired when a Bayesian test says their advice loses.

**The gate.** One aggregation pipeline per day:

1. `$vectorSearch` (exact) for lessons similar to a plain-English description of today, filtered to active lessons learned before the window.
2. For each lesson, keep only evidence dated before the window, then compute with `$map` / `$reduce`:
   - a similarity weight for each entry, `w = exp(-d² / 2h²)`, where `d` is the distance between today's conditions and that entry's
   - **A**, the weighted win rate in markets like today, and **n_eff**, how much similar evidence exists
   - a 90% confidence band on A
3. A lesson is **proven here** when that band's lower edge is above 50%. Almanac sees only proven lessons, and the prompt shows each as *"worked 84% in markets like today (4.3 similar uses)"*.

**The agents.** All four get the same prompt, signals and costs. Each day they choose a target position for tomorrow (IN or OUT).

| Agent | Memory |
|---|---|
| No memory | none |
| Plain | the 30 most similar lessons, with overall win/loss records |
| Plain + context | the same, plus the conditions each lesson was learned in |
| **Almanac** | only lessons proven in markets like today |

The simulator fills at the next open, charges 5 bps per trade and only switches after the same target appears two days in a row. Decisions for every trading day are precomputed, so any window replays in seconds.

**Proof.** A counterfactual endpoint re-asks the model with one cited lesson removed (fresh calls, five times each way) and reports how often the decision flips.

## MongoDB features

| Feature | Used for |
|---|---|
| Atlas Vector Search (exact, 19 filter fields) | similarity retrieval with status, date and range filters |
| Aggregation pipeline (`$filter`, `$map`, `$reduce`, `$exp`) | the conditional track record and confidence band, computed per lesson in the database |
| `$lookup` | rebuilding evidence logs from trade outcomes |
| Multi-document transactions | each scored trade updates its outcome, cited lessons, merges and retirements atomically |
| Change streams | live "lessons now apply / stopped applying" events in the UI |
| `$jsonSchema` validation | lessons missing a field or with bad values are rejected |
| Atlas AI Model APIs (Voyage `voyage-4`) | embeddings for lessons and day descriptions |

## Guardrails

- Every price read goes through `bars_until(date)`; trades fill at the next day's open.
- Lessons and evidence stop at the end of 2015; demo windows start in 2016.
- The model never sees dates, the ticker or real prices ("Asset X, day 13826", prices rebased to 100).
- The prompt states signals as facts without saying which way they usually resolve.
- Tests check all of the above, plus that evidence dated after a window's start never changes a lesson's score.

## Results

Across twelve ~60-day windows from 2016 to 2024, the results are mixed. Every agent holds the same single index, long only, so in calm uptrends all four track buy-and-hold closely. Almanac's clearest win was the 2022 bear market: −4.6% (−7.5% ± 0.5 across resampled runs) against −14% to −17% for the other agents and −11.9% for buy-and-hold. The more useful signal is in individual decisions. Almanac hides lessons that sound relevant but have never been proven in the current conditions, and removing a single proven lesson can flip its call. `python -m almanac.sweep summary` prints the full table.

## Run it

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv -r requirements.txt
cp .env.example .env   # MONGODB_URI, VOYAGE_API_KEY (Atlas AI Model APIs), CLAUDE_API_KEY

python -m almanac.data download && python -m almanac.data load   # prices + features -> Atlas
python -m almanac.lessons setup                                   # collection, validator, vector index
python -m almanac.learn                                           # learning run, 2007-2015
python -m almanac.evidence backfill                               # evidence logs and records
python -m almanac.daily                                           # precompute 2016-2024 decisions
uvicorn almanac.server:app --port 8000                            # open http://localhost:8000
python -m pytest tests/
```

Every model and embedding call is cached on disk, so reruns are fast and free.

## Layout

```
almanac/
  data.py, features.py     market data, condition variables, bars_until
  lessons.py               schema, vector index, the retrieval gate
  learn.py, evidence.py    learning run, evidence logs, retirement
  agent.py, llm.py         prompts, Claude and Voyage calls, disk cache
  daily.py, demo.py        precomputed decisions, simulator, replay stream
  counterfactual.py        lesson-removal flip test
  sweep.py, metrics.py     multi-window evaluation
  server.py                FastAPI app
static/index.html          single-page UI (TradingView Lightweight Charts)
tests/                     103 tests
```

## Stack

Python 3.11, FastAPI, pymongo, pandas, numpy, scipy, yfinance. Claude Haiku 5.5 for decisions and lesson writing, Voyage `voyage-4` for embeddings, MongoDB Atlas for everything stored.
