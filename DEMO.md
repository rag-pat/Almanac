# Almanac: 2-minute demo script

Setup: `uvicorn almanac.server:app --port 8000`, open http://localhost:8000. Every trading day 2016-2024 is precomputed, so any window replays in about 7 seconds.

---

## 0:00 The problem (15 s)

> "AI agents remember things, but they don't know *when* a memory applies. Plain memory retrieves notes that *sound* similar to today. For a trading agent that's dangerous: 'buying the dip paid off' sounds just as relevant in a crash as in a calm pullback."

## 0:15 What Almanac is (20 s)

> "Our agent traded 2007 to 2015 and wrote 81 lessons from its own trades. Every lesson keeps an evidence log: the market conditions each time it was tested, and whether its advice was right. That's 1,447 entries in Atlas.
>
> When the agent asks for lessons, **MongoDB decides what it's allowed to see.** One aggregation pipeline runs vector search, then computes each lesson's track record *in markets like today's*, and only shows lessons we're 90% confident beat a coin flip here. The LLM never gets a vote."

Point at: the four agent panels. Same model, same prompt, same costs. Only memory differs: none, plain, plain + context, Almanac.

## 0:35 Moment 1: Almanac hides a lesson plain memory trusts (30 s)

Pick **2020 crash**, press Run, and let it play to early March.

> "March 5th, 2020. Plain memory goes **IN**. It cites a lesson that says *'buying a deeply oversold dip paid off'*, which has a perfect 1-and-0 record.
>
> Almanac **hides that lesson**. It's only been tested 0.3 times in conditions like this, so it's *unproven here*. Almanac stays **OUT**.
>
> The market fell **18%** over the next five days."

Point at: the strip ("X of 81 lessons proven in markets like today") and the memory events feed (lessons starting and stopping to apply, delivered by a MongoDB change stream).

## 1:05 Moment 2: Proof the memory drives the decision (25 s)

Set the window to **2022-08-15 → 2022-11-04** (the 2022 bear preset) and Run. In the Almanac panel on **2022-08-26**, click **"Test: decide again without this lesson"** on the cited lesson.

> "Jackson Hole, August 26th, 2022. Almanac goes **OUT**, citing *'buying an oversold dip was a small loss, wait for confirmation'*. It was right 77% of the time in markets like this.
>
> We re-ask Claude ten times live. With the lesson: OUT 4 times out of 5. **Remove that one lesson: IN, 5 out of 5.** The memory is what made the call. The market fell another 3% that week."

## 1:30 Moment 3: The result, honestly (20 s)

Let the 2022 run finish, then flip the equity chart to **"Vs buy and hold"**.

> "Over that bear market, Almanac lost **4.6%** while no memory, plain and plain + context lost 14 to 17%, and buy-and-hold lost 12%. Across three resampled reruns Almanac averaged **−7.5% ± 0.5**, still the best of the four.
>
> We'll be honest about the rest. Across 12 windows from 2016 to 2024, results are mixed. In calm bull markets every agent mostly tracks buy-and-hold, because staying invested is the right call. The end card shows all 12."

## 1:50 Close (10 s)

> "Almanac is memory you can govern. Lessons carry their evidence, the database enforces when they apply, every change is audited, and you can prove a memory changed a decision."

---

## If asked

- **"Why do the curves look alike?"** Every agent holds the same single index, long only, fully in or out. When invested, its line *is* the market's line. The "Vs buy and hold" toggle removes that shared shape.
- **"Isn't this overfit?"** The z-score constants come from 2007-2012, lessons and evidence stop at the end of 2015, and demos start in 2016. The LLM never sees dates, the ticker or real prices. Tests enforce all of it.
- **"Why does Almanac lose some windows?"** When no lesson is proven for today's conditions, Almanac decides from the raw signals alone. In the 2020 crash that was true on most days, which is why its full-window result there is −17.7% despite the March 5th call.
- **MongoDB pieces:** Atlas Vector Search (exact, 19 filter fields), aggregation with `$filter` / `$map` / `$reduce` for the conditional record and confidence bound, `$lookup` for evidence, multi-document transactions in the learning loop, change streams for memory events, `$jsonSchema` validation, Voyage embeddings through Atlas AI Model APIs.
