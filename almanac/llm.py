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


# ---------- LLM (JSON out) ----------

_client = None


def _anthropic():
    global _client
    if _client is None:
        import anthropic
        if not config.CLAUDE_API_KEY:
            raise RuntimeError("CLAUDE_API_KEY is not set in .env")
        _client = anthropic.Anthropic(api_key=config.CLAUDE_API_KEY, max_retries=4)
    return _client


class LLMRefusal(RuntimeError):
    pass


def llm_json(system: str, user: str, schema: dict, effort: str | None = None, use_cache: bool = True,
             seed: int | None = None) -> dict:
    """One Claude call that must return JSON matching `schema`. Cached on disk by full input
    (use_cache=False forces a fresh call — the counterfactual uses this to measure real variation).

    Note: current Claude models reject non-default `temperature`, so determinism comes from the
    cache — the same prompt always returns the same stored answer on reruns.
    """
    effort = effort or config.LLM_EFFORT
    payload = {"model": config.LLM_MODEL, "effort": effort, "system": system, "user": user, "schema": schema}
    if seed is not None:  # a seeded variant: a separate cached sample of the same prompt (for variance runs)
        payload["seed"] = seed

    def call():
        resp = _anthropic().messages.create(
            model=config.LLM_MODEL,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        )
        if resp.stop_reason == "refusal":
            raise LLMRefusal(str(resp.stop_details))
        text = next(b.text for b in resp.content if b.type == "text")
        return json.loads(text)

    return cached("llm", payload, call) if use_cache else call()


def embed_many(texts: list[str], kind: str = "document", batch: int = 128) -> list[list[float]]:
    """Batched embed(): cached texts are read from disk, the rest go to the API in batches of `batch`."""
    out: dict[str, list[float]] = {}
    missing = []
    for t in dict.fromkeys(texts):
        p = _cache_path("embed", {"model": config.EMBED_MODEL, "input_type": kind, "text": t})
        if p.exists():
            out[t] = json.loads(p.read_text())
        else:
            missing.append(t)
    for i in range(0, len(missing), batch):
        chunk = missing[i:i + batch]
        r = _post(config.EMBED_URL, {"input": chunk, "model": config.EMBED_MODEL, "input_type": kind},
                  config.VOYAGE_API_KEY)
        for t, d in zip(chunk, sorted(r["data"], key=lambda d: d["index"])):
            out[t] = d["embedding"]
            _cache_path("embed", {"model": config.EMBED_MODEL, "input_type": kind, "text": t}).write_text(
                json.dumps(d["embedding"]))
    return [out[t] for t in texts]


def cosine(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
