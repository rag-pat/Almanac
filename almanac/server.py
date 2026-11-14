"""FastAPI app: static page, SSE demo replay from daily_decisions, sweep summary, counterfactual.

Run:  uvicorn almanac.server:app --port 8000
"""
import json

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse

from almanac import config
from almanac.demo import stream_demo, window_days

app = FastAPI(title="Almanac")
STATIC = config.ROOT / "static"


@app.on_event("startup")
def preload():
    """Load the static features table once in the background so the first /run doesn't wait for it."""
    import threading
    from almanac.data import _features_all
    threading.Thread(target=_features_all, daemon=True).start()


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/presets")
def presets():
    return config.PRESETS


@app.get("/run")
def run(start: str = Query(...), end: str = Query(...)):
    try:
        window_days(start, end)
    except ValueError as e:
        raise HTTPException(400, str(e))

    def stream():
        try:
            for msg in stream_demo(start, end):
                yield f"event: {msg['type']}\ndata: {json.dumps(msg, default=str)}\n\n"
        except Exception as e:  # surface errors to the browser instead of a silent hang
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/sweep")
def sweep_summary():
    """Offline sweep results (12 windows x 4 agents) + variance runs on the presets, for the end card."""
    from almanac.sweep import summary
    return summary()


@app.get("/counterfactual")
def counterfactual_endpoint(date: str, lesson_id: str):
    """Rerun one demo day's almanac decision with a cited lesson removed (5 fresh runs) -> flip rate."""
    from almanac.counterfactual import counterfactual
    try:
        return counterfactual(date, lesson_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
