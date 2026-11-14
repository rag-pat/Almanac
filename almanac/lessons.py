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
    }
    for v in VARS:
        doc[f"lo_{v}"] = float(fp[v]) - BOX
        doc[f"hi_{v}"] = float(fp[v]) + BOX
    return doc


# ---------- retrieval (the Atlas gate) ----------

IN_RANGE_MIN = config.IN_RANGE_MIN


def _inside(v: str, zv: float) -> dict:
    """1 if today's z for variable v is inside the lesson's [lo, hi], else 0 (aggregation expression)."""
    return {"$cond": [{"$and": [{"$lte": [f"$lo_{v}", zv]}, {"$gte": [f"$hi_{v}", zv]}]}, 1, 0]}


def _in_range_count(z: dict) -> dict:
    return {"$add": [_inside(v, float(z[v])) for v in VARS]}


def _notebook_filter(before, agent_id: str) -> dict:
    if not isinstance(before, datetime):
        before = datetime.fromisoformat(str(before)[:10])
    return {"status": "active", "learned_date": {"$lt": before}, "agent_id": agent_id}


def search(qvec: list[float], z: dict, before, almanac: bool, agent_id: str = "learner",
           cap: int = config.RETRIEVE_CAP) -> list[dict]:
    """Relevant lessons for today, best first. almanac=True adds the >= 6-of-8 range rule — all in one pipeline."""
    pipeline = [
        {"$vectorSearch": {"index": INDEX, "path": "embedding", "queryVector": qvec, "exact": True,
                           "limit": 1000, "filter": _notebook_filter(before, agent_id)}},
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
        {"$match": {"score": {"$gte": config.ATLAS_SCORE_MIN}}},
        {"$addFields": {"in_range": _in_range_count(z)}},
    ]
    if almanac:
        pipeline.append({"$match": {"in_range": {"$gte": IN_RANGE_MIN}}})
    pipeline += [{"$sort": {"score": -1}}, {"$limit": cap}, {"$project": {"embedding": 0}}]
    return list(coll().aggregate(pipeline))


def notebook_coverage(z: dict, before, agent_id: str = "learner") -> dict:
    """Over the WHOLE visible notebook: total lessons, how many are in range (>= 6 of 8),
    and which variables are outside every lesson's range ("unfamiliar conditions")."""
    group = {"_id": None, "total": {"$sum": 1},
             "in_range": {"$sum": {"$cond": [{"$gte": [_in_range_count(z), IN_RANGE_MIN]}, 1, 0]}}}
    group.update({f"cov_{v}": {"$sum": _inside(v, float(z[v]))} for v in VARS})
    res = next(coll().aggregate([{"$match": _notebook_filter(before, agent_id)}, {"$group": group}]), None)
    if not res:
        return {"n_total": 0, "n_in_range": 0, "unfamiliar": list(VARS)}
    return {"n_total": res["total"], "n_in_range": res["in_range"],
            "unfamiliar": [v for v in VARS if res[f"cov_{v}"] == 0]}


def retrieve(z: dict, before, agent_id: str = "learner") -> dict:
    """One call per day: embed today's description, return what each agent type sees + coverage for the UI.
    z: today's z-scores keyed by variable name. before: window start (only lessons learned earlier are visible)."""
    from almanac.features import describe
    from almanac.llm import embed

    if not isinstance(before, datetime):
        before = datetime.fromisoformat(str(before))
    z = {v: float(z[f"z_{v}"] if f"z_{v}" in z else z[v]) for v in VARS}
    text = describe(z)
    qvec = embed(text, "query")
    return {
        "description": text,
        "relevant": search(qvec, z, before, almanac=False, agent_id=agent_id),  # plain + plain_ctx
        "almanac": search(qvec, z, before, almanac=True, agent_id=agent_id),    # almanac (>= 6 of 8 in range)
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
