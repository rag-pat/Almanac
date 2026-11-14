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
BOX = 0.5  # starting half-width of each range around the fingerprint

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
        "wins": 1 if outcome == "win" else 0,
        "losses": 1 if outcome == "loss" else 0,
        "status": "active",
        "learned_date": learned_date,
        "history": [{"event": "created", "date": learned_date, "outcome": outcome}],
    }
    for v in VARS:
        doc[f"lo_{v}"] = float(fp[v]) - BOX
        doc[f"hi_{v}"] = float(fp[v]) + BOX
    return doc


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "setup"
    if cmd == "setup":
        ensure_collection()
        ensure_index()
    else:
        sys.exit(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()
