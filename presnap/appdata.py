"""Frontend data source — one table of per-play win probability for the app.

Two backends, chosen by env var, so deployment (Option A) and a live database
(Option B) share all app code:

    PRESNAP_APP_SOURCE=parquet  (default) -> read the committed snapshot
    PRESNAP_APP_SOURCE=db                 -> recompute live from Postgres

The snapshot is produced by scripts/export_wp.py running the db backend once.
Both paths return the SAME columns, so no screen changes between A and B.
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl

SNAPSHOT_PATH = Path("app/snapshot/wp_plays.parquet")


def _from_db() -> pl.DataFrame:
    # Imported lazily so the deployed (parquet) app needs no db/model/sklearn.
    from presnap import db, inference
    from presnap.schema import PRESNAP_VIEW, RAW_TABLE, q

    df = db.read_sql(
        f"SELECT * FROM {q(PRESNAP_VIEW)} WHERE down IS NOT NULL ORDER BY game_id, play_id"
    )
    home_wp = inference.home_wp(df)
    result = db.read_sql(f"SELECT DISTINCT game_id, result FROM {q(RAW_TABLE)}")
    return (
        df.select(
            "game_id", "play_id", "home_team", "away_team", "posteam",
            "qtr", "down", "ydstogo",
            pl.col("game_id").str.slice(0, 4).cast(pl.Int32).alias("season"),
            pl.col("score_differential").alias("score_diff"),
            ((3600 - pl.col("game_seconds_remaining")) / 60.0).alias("elapsed_min"),
        )
        .with_columns(pl.Series("home_wp", home_wp))
        .join(result, on="game_id", how="left")
    )


def load_plays() -> pl.DataFrame:
    if os.environ.get("PRESNAP_APP_SOURCE", "parquet") == "db":
        return _from_db()
    return pl.read_parquet(SNAPSHOT_PATH)


def export_snapshot(path: Path = SNAPSHOT_PATH) -> int:
    df = _from_db()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return df.height
