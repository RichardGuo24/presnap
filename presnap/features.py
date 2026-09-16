"""Feature construction — pure, row-wise transforms over pre-snap columns.

CONTRACT OF THIS MODULE (what makes the no-lookahead guarantee mechanical):
  * build_features() reads ONLY columns in contract.PRESNAP_COLUMNS.
  * It performs deterministic, row-wise transforms only — no statistics fitted
    from the data (no medians, scalers, encoders learned here). Those belong in
    Phase 3, fit on the training split, so nothing leaks across rows or splits.

Because it only reads allowlisted columns, running it on data where every
non-allowlisted column has been corrupted produces identical output — which is
exactly what tests/test_no_lookahead.py asserts.

Everything is oriented to the POSTEAM (the team we predict for), matching the
label "did posteam win?". score_differential is already posteam-minus-defteam;
spread_line is flipped when posteam is the away team.
"""

from __future__ import annotations

import polars as pl

# The feature columns produced, for reference by tests and downstream phases.
FEATURE_COLUMNS: list[str] = [
    "f_score_diff",
    "f_game_sec",
    "f_half_sec",
    "f_down",
    "f_ydstogo",
    "f_yardline_100",
    "f_goal_to_go",
    "f_pos_timeouts",
    "f_def_timeouts",
    "f_qtr",
    "f_is_home",
    "f_posteam_spread",
    "f_total_line",
    "f_div_game",
    "f_week",
    "f_is_postseason",
    "f_is_indoor",
    "f_is_turf",
    "f_temp",
    "f_wind",
]


def build_features(df: pl.DataFrame) -> pl.DataFrame:
    """raw/view rows -> (game_id, play_id) + feature columns. Pure and row-wise."""
    # Categorical text in this data is dirty (e.g. surface "grass " with a
    # trailing space, empty-string surface). Strip whitespace before comparing,
    # and treat empty as missing — otherwise "grass " is silently labeled turf.
    ptype = pl.col("posteam_type").str.strip_chars()
    stype = pl.col("season_type").str.strip_chars()
    roof = pl.col("roof").str.strip_chars()
    surf = pl.col("surface").str.strip_chars()

    is_home = ptype == "home"
    # spread_line > 0 means the HOME team is favored (verified against `result`).
    # Orient to the posteam: positive f_posteam_spread => posteam is favored.
    posteam_spread = (
        pl.when(is_home)
        .then(pl.col("spread_line"))
        .otherwise(-pl.col("spread_line"))
    )
    # grass (incl. "grass ") -> 0; known turf -> 1; empty/missing -> null.
    is_turf = (
        pl.when(surf.is_null() | (surf == "")).then(None)
        .when(surf == "grass").then(0)
        .otherwise(1)
        .cast(pl.Int8)
    )

    return df.select(
        "game_id",
        "play_id",
        pl.col("score_differential").alias("f_score_diff"),
        pl.col("game_seconds_remaining").alias("f_game_sec"),
        pl.col("half_seconds_remaining").alias("f_half_sec"),
        pl.col("down").alias("f_down"),
        pl.col("ydstogo").alias("f_ydstogo"),
        pl.col("yardline_100").alias("f_yardline_100"),
        pl.col("goal_to_go").alias("f_goal_to_go"),
        pl.col("posteam_timeouts_remaining").alias("f_pos_timeouts"),
        pl.col("defteam_timeouts_remaining").alias("f_def_timeouts"),
        pl.col("qtr").alias("f_qtr"),
        is_home.cast(pl.Int8).alias("f_is_home"),
        posteam_spread.alias("f_posteam_spread"),
        pl.col("total_line").alias("f_total_line"),
        pl.col("div_game").alias("f_div_game"),
        pl.col("week").alias("f_week"),
        (stype == "POST").cast(pl.Int8).alias("f_is_postseason"),
        roof.is_in(["dome", "closed"]).cast(pl.Int8).alias("f_is_indoor"),
        is_turf.alias("f_is_turf"),
        pl.col("temp").alias("f_temp"),
        pl.col("wind").alias("f_wind"),
    )


def feature_matrix(feats: pl.DataFrame):
    """FEATURE_COLUMNS as a float matrix (nulls -> NaN).

    The ONE feature->matrix path, shared by training (split.to_xy) and serving
    (inference.py). Routing both through here is the structural guard against
    train/serve skew: the frontend cannot compute features differently from how
    the model was trained, because it runs this exact code.
    """
    return feats.select([pl.col(c).cast(pl.Float64) for c in FEATURE_COLUMNS]).to_numpy()
