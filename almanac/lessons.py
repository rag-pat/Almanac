"""M2: the lessons notebook — schema-validated collection, vector index, retrieval.

Run:  python -m almanac.lessons setup   # create collection + validator + vector index (idempotent)
"""
import sys
import time
from datetime import datetime

from pymongo.errors import CollectionInvalid
from pymongo.operations import SearchIndexModel

from almanac import config, db
from almanac.features import VARS

COLL = "lessons"
INDEX = "lessons_vector"
BOX = config.BOX

NUM = {"bsonType": ["double", "int", "long"]}

SCHEMA = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["agent_id", "text", "embedding", "fp", "wins", "losses", "status",
                     "learned_date", "history", "action", "outcome"]
                    + [f"lo_{v}" for v in VARS] + [f"hi_{v}" for v in VARS],
        "properties": {
            "agent_id": {"bsonType": "string"},
            "text": {"bsonType": "string", "minLength": 1},
            "embedding": {"bsonType": "array", "minItems": config.EMBED_DIM, "maxItems": config.EMBED_DIM,
                          "items": NUM},
            "fp": {"bsonType": "object", "required": VARS, "properties": {v: NUM for v in VARS}},
            "wins": {"bsonType": ["int", "long"], "minimum": 0},
            "losses": {"bsonType": ["int", "long"], "minimum": 0},
            "status": {"enum": ["active", "retired"]},
            "learned_date": {"bsonType": "date"},
            "history": {"bsonType": "array"},
            "action": {"enum": ["buy", "sell"]},
            "outcome": {"enum": ["win", "loss"]},
            **{f"lo_{v}": NUM for v in VARS},
            **{f"hi_{v}": NUM for v in VARS},
        },
    }
}


def coll():
    return db.db()[COLL]


def ensure_collection():
    d = db.db()
    try:
        d.create_collection(COLL, validator=SCHEMA, validationLevel="strict", validationAction="error")
        print(f"created {COLL} with validator")
    except CollectionInvalid:
        d.command("collMod", COLL, validator=SCHEMA, validationLevel="strict", validationAction="error")
        print(f"updated validator on {COLL}")


FILTER_FIELDS = ["status", "learned_date", "agent_id"] + [f"lo_{v}" for v in VARS] + [f"hi_{v}" for v in VARS]

INDEX_DEF = {
    "fields": [{"type": "vector", "path": "embedding", "numDimensions": config.EMBED_DIM, "similarity": "cosine"}]
              + [{"type": "filter", "path": f} for f in FILTER_FIELDS]
}


def ensure_index(wait: bool = True, timeout: int = 300):
    c = coll()
    existing = {i["name"]: i for i in c.list_search_indexes()}
    if INDEX not in existing:
        c.create_search_index(SearchIndexModel(definition=INDEX_DEF, name=INDEX, type="vectorSearch"))
        print(f"creating vector index {INDEX}…")
    elif existing[INDEX].get("latestDefinition") != INDEX_DEF:
        c.update_search_index(INDEX, INDEX_DEF)
        print(f"updating vector index {INDEX}…")
    if not wait:
        return
    t0 = time.time()
    while time.time() - t0 < timeout:
        idx = next((i for i in c.list_search_indexes() if i["name"] == INDEX), None)
        if idx and idx.get("queryable") and idx.get("status") == "READY":
            print(f"vector index {INDEX} READY ({time.time() - t0:.0f}s)")
            return
        time.sleep(5)
    raise TimeoutError(f"vector index {INDEX} not ready after {timeout}s")


def make_lesson(text: str, embedding: list[float], fp: dict, action: str, outcome: str,
                learned_date, agent_id: str = "learner") -> dict:
    """Build a new lesson document with a starting box of fp ± BOX on every variable."""
    if not isinstance(learned_date, datetime):
        learned_date = datetime.fromisoformat(str(learned_date))
    doc = {
        "agent_id": agent_id,
        "text": text,
        "embedding": embedding,
        "fp": {v: float(fp[v]) for v in VARS},
        "action": action,
        "outcome": outcome,
        # record tracks the lesson's ADVICE: it was right about the trade it came from (even a "this lost" warning)
        "wins": 1,
        "losses": 0,
        "status": "active",
        "learned_date": learned_date,
        "history": [{"event": "created", "date": learned_date, "outcome": outcome}],
        # conditional track record: the birth trade is the first entry (its advice was right -> win 1)
        "evidence": [evidence_entry(learned_date, fp, 1)],
    }
    for v in VARS:
        doc[f"lo_{v}"] = float(fp[v]) - BOX
        doc[f"hi_{v}"] = float(fp[v]) + BOX
    return doc


def evidence_entry(date, z: dict, win: int) -> dict:
    """One evidence-log entry: decision-day z-scores (VARS order), 1/0 result, dated when the result was known."""
    return {"date": date, "z": [float(z[v]) for v in VARS], "win": int(win)}


# ---------- retrieval (the Atlas gate) ----------

IN_RANGE_MIN = config.IN_RANGE_MIN


def _applicability_stages(z: dict, before) -> list[dict]:
    """Aggregation stages adding A (conditional win rate in markets like today) and n_eff (similar evidence).
    Only evidence dated before `before` (the window start) counts."""
    today = [float(z[v]) for v in VARS]
    two_h2 = 2 * config.KERNEL_H ** 2
    sq_dist = {"$reduce": {"input": {"$range": [0, len(VARS)]}, "initialValue": 0,
                           "in": {"$add": ["$$value", {"$pow": [{"$subtract": [
                               {"$arrayElemAt": ["$$e.z", "$$this"]}, {"$arrayElemAt": [today, "$$this"]}]}, 2]}]}}}
    return [
        {"$addFields": {"_ev": {"$filter": {"input": {"$ifNull": ["$evidence", []]}, "as": "e",
                                            "cond": {"$lt": ["$$e.date", _to_date(before)]}}}}},
        {"$addFields": {"_w": {"$map": {"input": "$_ev", "as": "e", "in": {
            "w": {"$exp": {"$divide": [{"$multiply": [-1, sq_dist]}, two_h2]}}, "win": "$$e.win"}}}}},
        {"$addFields": {
            "n_eff": {"$reduce": {"input": "$_w", "initialValue": 0, "in": {"$add": ["$$value", "$$this.w"]}}},
            "_wwin": {"$reduce": {"input": "$_w", "initialValue": 0,
                                  "in": {"$add": ["$$value", {"$multiply": ["$$this.w", "$$this.win"]}]}}}}},
        {"$addFields": {"A": {"$divide": [{"$add": [1, "$_wwin"]}, {"$add": [2, "$n_eff"]}]},
                        "n_evidence": {"$size": "$_ev"}}},
        {"$project": {"_ev": 0, "_w": 0, "_wwin": 0}},
    ]


def applicability_py(evidence: list[dict], z: dict, before) -> tuple[float, float]:
    """Reference implementation of the pipeline math (tests + reporting)."""
    import math
    before, today = _to_date(before), [float(z[v]) for v in VARS]
    ws = [(math.exp(-sum((a - b) ** 2 for a, b in zip(e["z"], today)) / (2 * config.KERNEL_H ** 2)), e["win"])
          for e in evidence if e["date"] < before]
    n_eff = sum(w for w, _ in ws)
    return (1 + sum(w * win for w, win in ws)) / (2 + n_eff), n_eff


def _applies() -> dict:
    return {"A": {"$gte": config.A_MIN}, "n_eff": {"$gte": config.NEFF_MIN}}


def _to_date(d) -> datetime:
    return d if isinstance(d, datetime) else datetime.fromisoformat(str(d)[:10])


def _inside(v: str, zv: float) -> dict:
    """1 if today's z for variable v is inside the lesson's [lo, hi], else 0 (aggregation expression)."""
    return {"$cond": [{"$and": [{"$lte": [f"$lo_{v}", zv]}, {"$gte": [f"$hi_{v}", zv]}]}, 1, 0]}


def _in_range_count(z: dict) -> dict:
    return {"$add": [_inside(v, float(z[v])) for v in VARS]}


def _notebook_filter(before, agent_id: str) -> dict:
    return {"status": "active", "learned_date": {"$lt": _to_date(before)}, "agent_id": agent_id}


def search(qvec: list[float], z: dict, before, almanac: bool, agent_id: str = "learner",
           cap: int = config.RETRIEVE_CAP) -> list[dict]:
    """Relevant lessons for today. Plain: best similarity first. almanac=True keeps only lessons with a good
    conditional record here (A >= A_MIN, n_eff >= NEFF_MIN), ranked by A — all in one pipeline."""
    pipeline = [
        {"$vectorSearch": {"index": INDEX, "path": "embedding", "queryVector": qvec, "exact": True,
                           "limit": 1000, "filter": _notebook_filter(before, agent_id)}},
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
        {"$match": {"score": {"$gte": config.ATLAS_SCORE_MIN}}},
        *_applicability_stages(z, before),
        {"$addFields": {"in_range": _in_range_count(z)}},
    ]
    if almanac:
        pipeline += [{"$match": _applies()}, {"$sort": {"A": -1, "score": -1}}]
    else:
        pipeline += [{"$sort": {"score": -1}}]
    pipeline += [{"$limit": cap}, {"$project": {"embedding": 0, "evidence": 0}}]
    return list(coll().aggregate(pipeline))


def notebook_coverage(z: dict, before, agent_id: str = "learner") -> dict:
    """Over the WHOLE visible notebook: how many lessons apply here (A gate), how many are unproven here
    (n_eff < NEFF_MIN), and which variables are outside every lesson's box ("unfamiliar conditions")."""
    group = {"_id": None, "total": {"$sum": 1},
             "applies": {"$sum": {"$cond": [{"$and": [{"$gte": ["$A", config.A_MIN]},
                                                       {"$gte": ["$n_eff", config.NEFF_MIN]}]}, 1, 0]}},
             "unproven": {"$sum": {"$cond": [{"$lt": ["$n_eff", config.NEFF_MIN]}, 1, 0]}}}
    group.update({f"cov_{v}": {"$sum": _inside(v, float(z[v]))} for v in VARS})
    res = next(coll().aggregate([{"$match": _notebook_filter(before, agent_id)},
                                 *_applicability_stages(z, before), {"$group": group}]), None)
    if not res:
        return {"n_total": 0, "n_applies": 0, "n_unproven": 0, "unfamiliar": list(VARS)}
    return {"n_total": res["total"], "n_applies": res["applies"], "n_unproven": res["unproven"],
            "unfamiliar": [v for v in VARS if res[f"cov_{v}"] == 0]}


def applicable_lessons(z: dict, before, agent_id: str = "learner") -> dict:
    """{lesson_id: text} for every visible lesson that applies today (A gate; ignores text similarity)."""
    rows = coll().aggregate([
        {"$match": _notebook_filter(before, agent_id)},
        *_applicability_stages(z, before),
        {"$match": _applies()},
        {"$project": {"text": 1}},
    ])
    return {str(r["_id"]): r["text"] for r in rows}


def retrieve(z: dict, before, agent_id: str = "learner", qvec: list[float] | None = None) -> dict:
    """One call per day: embed today's description, return what each agent type sees + coverage for the UI.
    z: today's z-scores keyed by variable name. before: window start (only lessons learned earlier are visible).
    qvec: precomputed query embedding of describe(z) (batched precompute passes it in)."""
    from almanac.features import describe
    from almanac.llm import embed

    before = _to_date(before)
    z = {v: float(z[f"z_{v}"] if f"z_{v}" in z else z[v]) for v in VARS}
    text = describe(z)
    qvec = qvec or embed(text, "query")
    return {
        "description": text,
        "relevant": search(qvec, z, before, almanac=False, agent_id=agent_id),  # plain + plain_ctx
        "almanac": search(qvec, z, before, almanac=True, agent_id=agent_id),    # almanac (conditional record)
        **notebook_coverage(z, before, agent_id),
    }


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "setup"
    if cmd == "setup":
        ensure_collection()
        ensure_index()
    else:
        sys.exit(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()
