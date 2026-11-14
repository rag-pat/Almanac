"""FastAPI app: static page + SSE demo stream (+ counterfactual in M8).

Run:  uvicorn almanac.server:app --port 8000
"""
import json

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse

from almanac import config
from almanac.demo import run_demo, window_days

app = FastAPI(title="Almanac")
STATIC = config.ROOT / "static"


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
            for msg in run_demo(start, end):
                yield f"event: {msg['type']}\ndata: {json.dumps(msg, default=str)}\n\n"
        except Exception as e:  # surface errors to the browser instead of a silent hang
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
