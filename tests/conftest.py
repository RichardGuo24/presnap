"""Shared test fixtures.

make_pbp builds a synthetic pbp-shaped DataFrame with every column features.py
and labels.py touch, PLUS the post-play "danger" columns that the perturbation
test corrupts. Hermetic — no DB, no network.
"""

import polars as pl
import pytest

_DEFAULTS = {
    # keys + allowlisted (pre-snap) columns features.py reads
    "game_id": "2023_01_AAA_BBB",
    "play_id": 1,
    "posteam_type": "home",
    "spread_line": -3.0,
    "score_differential": 0,
    "game_seconds_remaining": 1800,
    "half_seconds_remaining": 900,
    "down": 1,
    "ydstogo": 10,
    "yardline_100": 75,
    "goal_to_go": 0,
    "posteam_timeouts_remaining": 3,
    "defteam_timeouts_remaining": 3,
    "qtr": 1,
    "total_line": 45.0,
    "div_game": 0,
    "week": 1,
    "season_type": "REG",
    "roof": "outdoors",
    "surface": "grass",
    "temp": 60,
    "wind": 5,
    # non-allowlisted post-play "danger" columns (features must ignore these)
    "result": 7,
    "play_type": "pass",
    "desc": "(shotgun) 11-A.B. pass short right",
    "td_team": "AAA",
    "total_home_score": 0,
    "total_away_score": 0,
}


@pytest.fixture
def make_pbp():
    def _make(rows: list[dict]) -> pl.DataFrame:
        recs = []
        for i, override in enumerate(rows):
            r = dict(_DEFAULTS)
            r["play_id"] = i + 1
            r.update(override)
            recs.append(r)
        return pl.DataFrame(recs)

    return _make
