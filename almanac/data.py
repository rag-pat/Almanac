"""M1: market data — download from Yahoo, save CSV backups, load into Atlas, read safely.

Run:  python -m almanac.data download   # pull CSVs into data/
      python -m almanac.data load       # push CSVs into Atlas "bars"
"""
import sys
from datetime import datetime

import pandas as pd
import yfinance as yf
from pymongo import ASCENDING, DESCENDING, UpdateOne

from almanac import config, db

COLS = ["open", "high", "low", "close", "volume"]


def csv_path(symbol: str):
    return config.DATA_DIR / f"{symbol.lstrip('^')}.csv"


def download(symbol: str) -> pd.DataFrame:
    raw = yf.download(
        symbol,
        start=config.DATA_START,
        end=config.DATA_END,
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )
    if raw.empty:
        raise RuntimeError(f"No data returned for {symbol}")
    df = raw.rename(columns=str.lower)[COLS]
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    return df.dropna(subset=["close"])


def download_all() -> dict[str, pd.DataFrame]:
    config.DATA_DIR.mkdir(exist_ok=True)
    out = {}
    for sym in config.SYMBOLS:
        df = download(sym)
        df.to_csv(csv_path(sym))
        out[sym] = df
        print(f"{sym:5s} {len(df):5d} rows  {df.index[0].date()} -> {df.index[-1].date()}")
    return out


def load_csv(symbol: str) -> pd.DataFrame:
    return pd.read_csv(csv_path(symbol), index_col="date", parse_dates=True)


def load_all() -> dict[str, pd.DataFrame]:
    return {s: load_csv(s) for s in config.SYMBOLS}


# ---------- Atlas ----------

def _upsert(coll, docs, keys):
    ops = [UpdateOne({k: d[k] for k in keys}, {"$set": d}, upsert=True) for d in docs]
    for i in range(0, len(ops), 5000):
        coll.bulk_write(ops[i:i + 5000], ordered=False)


def load_to_atlas():
    """Push CSV bars and annotated features into Atlas. Safe to rerun (upserts on unique keys)."""
    from almanac.features import annotate

    bars, feats = db.db()["bars"], db.db()["features"]
    bars.create_index([("symbol", ASCENDING), ("date", ASCENDING)], unique=True)
    feats.create_index([("date", ASCENDING)], unique=True)

    frames = load_all()
    for sym, df in frames.items():
        docs = [{"symbol": sym, "date": d.to_pydatetime(), **{c: float(r[c]) for c in COLS}}
                for d, r in df.iterrows()]
        _upsert(bars, docs, ["symbol", "date"])
        print(f"bars     {sym:5s} {bars.count_documents({'symbol': sym}):5d}")

    ann = annotate(frames)
    docs = []
    for d, r in ann.iterrows():
        doc = {"date": d.to_pydatetime(), "above200": bool(r["above200"]), "rsi2": float(r["rsi2"])}
        doc.update({c: float(r[c]) for c in ann.columns if c not in ("above200", "rsi2")})
        docs.append(doc)
    _upsert(feats, docs, ["date"])
    print(f"features {feats.count_documents({}):5d} days")


def _to_dt(date) -> datetime:
    return pd.Timestamp(date).to_pydatetime()


def bars_until(date, symbol: str = "SPY", n: int | None = None) -> pd.DataFrame:
    """THE gate for every price read: rows with date <= `date` only (optionally the last n)."""
    cur = db.db()["bars"].find({"symbol": symbol, "date": {"$lte": _to_dt(date)}},
                               {"_id": 0, "symbol": 0}).sort("date", DESCENDING)
    if n:
        cur = cur.limit(n)
    df = pd.DataFrame(list(cur))
    if df.empty:
        return df
    return df.set_index("date").sort_index()


def features_until(date, n: int | None = None) -> pd.DataFrame:
    """Annotated days with date <= `date` (values already use past data only)."""
    cur = db.db()["features"].find({"date": {"$lte": _to_dt(date)}}, {"_id": 0}).sort("date", DESCENDING)
    if n:
        cur = cur.limit(n)
    df = pd.DataFrame(list(cur))
    if df.empty:
        return df
    return df.set_index("date").sort_index()


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "download"
    if cmd == "download":
        download_all()
    elif cmd == "load":
        load_to_atlas()
    else:
        sys.exit(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()
