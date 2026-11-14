"""M1 step 2: market annotation — 8 condition variables per SPY day, z-scored, plus signals.

Every value on day t uses only data up to and including day t's close.

Run:  python -m almanac.features stats   # print learning-period z-stats to paste into config
      python -m almanac.features show    # print a few annotated days
"""
import sys

import numpy as np
import pandas as pd

from almanac import config
from almanac.data import load_all

VARS = ["vol", "trend", "drawdown", "volume", "vix", "rates", "curve", "credit"]


def raw_features(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    spy = frames["SPY"]
    idx = spy.index

    def aux(sym):  # align to SPY days; bond holidays take the previous day's value (past only)
        return frames[sym]["close"].reindex(idx).ffill()

    close, vol_ = spy["close"], spy["volume"]
    ma200 = close.rolling(200).mean()
    prev60 = vol_.shift(1).rolling(60)  # compare today's volume to the 60 days BEFORE it

    out = pd.DataFrame(index=idx)
    out["vol"] = np.sqrt((np.log(spy["high"] / spy["low"]) ** 2).rolling(20).mean() / (4 * np.log(2))) * np.sqrt(252)
    out["trend"] = (close / ma200 - 1) * 100
    out["drawdown"] = (close / close.rolling(252).max() - 1) * 100
    out["volume"] = (vol_ - prev60.mean()) / prev60.std()
    out["vix"] = aux("^VIX")
    tnx = aux("^TNX")
    ratio = aux("LQD") / aux("IEF")
    out["rates"] = tnx - tnx.shift(20)                            # 20-day change in 10y yield (pct points)
    out["curve"] = tnx - aux("^IRX")                              # level: 10y minus 3m
    out["credit"] = (ratio / ratio.rolling(60).mean() - 1) * 100  # LQD/IEF vs its 60-day avg, in %
    return out


def rsi(close: pd.Series, n: int = 2) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def compute_zstats(raw: pd.DataFrame) -> dict[str, tuple[float, float]]:
    learn = raw.loc[config.LEARN_START:config.LEARN_END]
    return {v: (round(float(learn[v].mean()), 6), round(float(learn[v].std()), 6)) for v in VARS}


def annotate(frames: dict[str, pd.DataFrame], zstats=None) -> pd.DataFrame:
    zstats = zstats or config.ZSTATS
    raw = raw_features(frames)
    close = frames["SPY"]["close"]
    out = raw.copy()
    for v in VARS:
        mu, sd = zstats[v]
        out[f"z_{v}"] = (raw[v] - mu) / sd
    out["rsi2"] = rsi(close, 2)
    out["above200"] = close > close.rolling(200).mean()
    return out.dropna(subset=VARS)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    frames = load_all()
    if cmd == "stats":
        stats = compute_zstats(raw_features(frames))
        print("ZSTATS = {")
        for v, (mu, sd) in stats.items():
            print(f'    "{v}": ({mu}, {sd}),')
        print("}")
    elif cmd == "show":
        df = annotate(frames)
        cols = [f"z_{v}" for v in VARS] + ["rsi2", "above200"]
        print(f"{len(df)} annotated days, {df.index[0].date()} -> {df.index[-1].date()}")
        for d in ["2008-10-10", "2017-06-30", "2020-03-16", "2022-10-12"]:
            row = df.loc[d, cols]
            print(d, {k: (round(float(x), 2) if k != "above200" else bool(x)) for k, x in row.items()})
    else:
        sys.exit(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()
