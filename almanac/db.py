"""Single MongoDB client for the whole app."""
from functools import lru_cache

from pymongo import MongoClient

from almanac import config


@lru_cache(maxsize=1)
def client() -> MongoClient:
    if not config.MONGODB_URI:
        raise RuntimeError("MONGODB_URI is not set — copy .env.example to .env")
    return MongoClient(config.MONGODB_URI)


def db():
    return client()[config.MONGODB_DB]


def ping() -> bool:
    client().admin.command("ping")
    return True
