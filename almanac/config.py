"""All settings and constants in one place."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / ".cache"

MONGODB_URI = os.getenv("MONGODB_URI", "")
MONGODB_DB = os.getenv("MONGODB_DB", "almanac")

# Data
SYMBOLS = ["SPY", "^VIX", "^TNX"]
DATA_START = "2005-01-01"
DATA_END = "2025-01-01"

# Learning period (z-score stats come from here ONLY)
LEARN_START = "2007-01-01"
LEARN_END = "2012-12-31"
DEMO_MIN_START = "2013-01-01"
DEMO_MAX_DAYS = 60

# Trading
COST = 0.0005  # 5 bps per trade
HORIZON = 5    # trading days to score an outcome
