"""M1: market data — download from Yahoo, save CSV backups, load into Atlas, read safely.

Run:  python -m almanac.data download   # pull CSVs into data/
      python -m almanac.data load       # push CSVs into Atlas "bars"
"""
import sys

import pandas as pd
import yfinance as yf

from almanac import config

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


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "download"
    if cmd == "download":
        download_all()
    else:
        sys.exit(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()
