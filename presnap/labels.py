"""The label — the ONE module allowed to read the future.

Target: did the possessing team (`posteam`) win the game?

`result` is the final home margin (home_final - away_final). Combined with
`posteam_type`:
    posteam won  <=>  (result > 0 and posteam is home)
                  or  (result < 0 and posteam is away)

Ties (result == 0) get a null label and are dropped downstream — this project
models a binary win/loss.

This is deliberately the only place that touches a post-play column, so the
"sees the future" surface is one small, auditable file.
"""

from __future__ import annotations

import polars as pl

LABEL_COLUMN = "label_posteam_won"


def build_labels(df: pl.DataFrame) -> pl.DataFrame:
    """df needs game_id, play_id, result, posteam_type. Tie rows are dropped."""
    is_home = pl.col("posteam_type") == "home"
    posteam_won = (
        pl.when(pl.col("result") > 0).then(is_home.cast(pl.Int8))
        .when(pl.col("result") < 0).then((~is_home).cast(pl.Int8))
        .otherwise(None)  # tie -> null -> dropped
    )
    return (
        df.select("game_id", "play_id", posteam_won.alias(LABEL_COLUMN))
        .drop_nulls(LABEL_COLUMN)
    )
