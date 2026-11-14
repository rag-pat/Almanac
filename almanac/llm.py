"""The ONLY place that talks to model providers. Every call is cached on disk by input hash.

embed(text, kind)  -> list[float]   (kind: "document" for lessons, "query" for today's description)
llm_json(...)      -> dict          (added in M3 once the provider is picked)
"""
import hashlib
import json
import time
import urllib.error
import urllib.request

import numpy as np

from almanac import config


# ---------- disk cache ----------

def _cache_path(namespace: str, payload: dict):
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    d = config.CACHE_DIR / namespace
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}.json"


def cached(namespace: str, payload: dict, fn):
    path = _cache_path(namespace, payload)
    if path.exists():
        return json.loads(path.read_text())
    result = fn()
    path.write_text(json.dumps(result))
    return result


# ---------- embeddings ----------

def _post(url: str, body: dict, key: str, retries: int = 4) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(retries):
        try:
            return json.load(urllib.request.urlopen(req, timeout=60))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"embedding call failed: HTTP {e.code} {e.read().decode()[:200]}") from e
        except urllib.error.URLError:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise


def embed(text: str, kind: str = "document") -> list[float]:
    assert kind in ("document", "query"), kind
    if not config.VOYAGE_API_KEY:
        raise RuntimeError("VOYAGE_API_KEY is not set in .env")
    payload = {"model": config.EMBED_MODEL, "input_type": kind, "text": text}

    def call():
        r = _post(config.EMBED_URL, {"input": [text], "model": config.EMBED_MODEL, "input_type": kind},
                  config.VOYAGE_API_KEY)
        vec = r["data"][0]["embedding"]
        assert len(vec) == config.EMBED_DIM, (len(vec), config.EMBED_DIM)
        return vec

    return cached("embed", payload, call)


def cosine(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
