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

# Anchored to the repo root (this file is presnap/appdata.py) so it resolves
# regardless of the process working directory.
SNAPSHOT_PATH = Path(__file__).resolve().parent.parent / "app" / "snapshot" / "wp_plays.parquet"


def _from_db() -> pl.DataFrame:
    # Imported lazily so the deployed (parquet) app needs no db/model/sklearn.
    import numpy as np
    from scipy.stats import norm

    from presnap import db, inference
    from presnap.schema import PRESNAP_VIEW, RAW_TABLE, q

    df = db.read_sql(
        f"SELECT * FROM {q(PRESNAP_VIEW)} WHERE down IS NOT NULL ORDER BY game_id, play_id"
    )
    home_wp = inference.home_wp(df)
    # Pregame market win probability for the home team, from the closing spread.
    # Home expected margin = spread_line; NFL final-margin std is ~13.86 points.
    market_wp = norm.cdf(df["spread_line"].to_numpy() / 13.86)
    result = db.read_sql(f"SELECT DISTINCT game_id, result FROM {q(RAW_TABLE)}")
    return (
        df.select(
            "game_id", "play_id", "home_team", "away_team", "posteam",
            "week", "qtr", "down", "ydstogo",
            pl.col("game_id").str.slice(0, 4).cast(pl.Int32).alias("season"),
            pl.col("score_differential").alias("score_diff"),
            ((3600 - pl.col("game_seconds_remaining")) / 60.0).alias("elapsed_min"),
        )
        .with_columns(
            pl.Series("home_wp", home_wp),
            pl.Series("market_wp", market_wp),
        )
        .join(result, on="game_id", how="left")
    )


def snapshot_sig() -> tuple[int, int]:
    """A cheap fingerprint of the snapshot file (size, mtime).

    Streamlit Cloud hot-reloads code without clearing @st.cache_data, so pages
    pass this as the cache key — a changed parquet forces a reload instead of
    serving a stale, wrong-schema DataFrame.
    """
    s = SNAPSHOT_PATH.stat()
    return (s.st_size, int(s.st_mtime))


def load_plays() -> pl.DataFrame:
    if os.environ.get("PRESNAP_APP_SOURCE", "parquet") == "db":
        return _from_db()
    return pl.read_parquet(SNAPSHOT_PATH)


def export_snapshot(path: Path = SNAPSHOT_PATH) -> int:
    df = _from_db()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return df.height
