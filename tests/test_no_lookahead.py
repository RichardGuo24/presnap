"""The mechanical no-lookahead guarantee (the project's centerpiece).

If a feature secretly read a post-play column, corrupting that column would
change the feature's output. So: corrupt EVERY non-allowlisted column and assert
the feature table is unchanged. Identical output => features depend on nothing
outside the allowlist.

Note what this proves and what it does NOT: it proves features stay INSIDE the
allowlist. It does not prove the allowlist is correct (that a listed column is
truly pre-snap) — that separate guarantee was discharged in Phase 1
(scripts/verify_prepost.py).
"""

import polars as pl

from presnap import contract, features


def _danger_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in contract.PRESNAP_COLUMNS]


def _sample(make_pbp):
    return make_pbp([
        {"posteam_type": "home", "spread_line": -3.0, "result": 10,
         "play_type": "pass", "total_home_score": 21, "down": 1},
        {"posteam_type": "away", "spread_line": 6.5, "result": -3,
         "play_type": "run", "total_away_score": 14, "down": 3},
        {"posteam_type": "home", "roof": "dome", "surface": "fieldturf",
         "result": 0, "down": 4, "score_differential": -7},
        {"posteam_type": "away", "temp": None, "wind": None,
         "result": 21, "td_team": "BBB", "down": 2},
    ])


def test_features_ignore_every_non_allowlisted_column(make_pbp):
    df = _sample(make_pbp)
    danger = _danger_columns(df)
    assert danger, "fixture must contain post-play columns to corrupt"

    base = features.build_features(df)
    nulled = df.with_columns([pl.lit(None).alias(c) for c in danger])
    shuffled = df.with_columns([pl.col(c).reverse().alias(c) for c in danger])

    assert base.equals(features.build_features(nulled)), "features changed when post-play cols nulled -> LEAK"
    assert base.equals(features.build_features(shuffled)), "features changed when post-play cols shuffled -> LEAK"


def test_output_contains_no_raw_danger_columns(make_pbp):
    out = features.build_features(_sample(make_pbp))
    assert out.columns == ["game_id", "play_id", *features.FEATURE_COLUMNS]
    for danger in ["result", "play_type", "total_home_score", "desc", "td_team"]:
        assert danger not in out.columns


def test_perturbation_actually_has_teeth(make_pbp):
    """Meta-test: prove the corruption DOES change a genuinely leaky feature, so
    a passing no-lookahead test means 'no leak', not 'corruption was a no-op'."""
    df = make_pbp([{"total_home_score": i} for i in range(5)])

    def leaky(d: pl.DataFrame) -> pl.DataFrame:
        return d.select("game_id", "play_id", pl.col("total_home_score").alias("f_leak"))

    base = leaky(df)
    shuffled = df.with_columns(pl.col("total_home_score").reverse())
    assert not base.equals(leaky(shuffled)), "corruption failed to change a leaky feature"
