"""Sweep window rules: 12 non-overlapping windows, presets included, each a valid demo window."""
import pytest

from almanac import config
from almanac.sweep import WINDOWS

pytestmark = pytest.mark.skipif(not config.MONGODB_URI, reason="no Atlas URI")


def test_twelve_non_overlapping_windows_including_presets():
    from almanac.demo import window_days
    assert len(WINDOWS) == 12
    for name, se in config.PRESETS.items():
        assert WINDOWS[name] == se
    spans = sorted((window_days(s, e)[0], window_days(s, e)[-1]) for s, e in WINDOWS.values())
    for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
        assert e1 < s2, "windows overlap"
    for s, e in WINDOWS.values():
        assert 50 <= len(window_days(s, e)) <= config.DEMO_MAX_DAYS
