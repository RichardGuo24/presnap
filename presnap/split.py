"""Temporal train/val/cal/test split — the anti-leakage seam of Phase 3.

The split is BY SEASON (from contract.SPLIT_SEASONS), and seasons partition
games, so no game can land in two splits — the "no game_id in two splits"
guarantee is structural, not something we police after the fact. The ordering is
temporal (train earliest ... test latest) so we never train on the future and
test on the past.

    train 2010-2020 | val 2021 | cal 2022 | test 2023-2024

`cal` is reserved for Phase 4's calibrator; the model never trains on it.
"""

from __future__ import annotations

import polars as pl

from presnap import contract, db
from presnap.build_features import MODEL_TABLE
from presnap.features import feature_matrix
from presnap.labels import LABEL_COLUMN

SPLIT_NAMES = ("train", "val", "cal", "test")


def add_season(df: pl.DataFrame) -> pl.DataFrame:
    """Derive the season from the game_id prefix (e.g. '2021_01_ARI_TEN')."""
    return df.with_columns(
        pl.col("game_id").str.slice(0, 4).cast(pl.Int32).alias("season")
    )


def split_frame(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Partition rows into the named splits by season. Pure — testable offline."""
    df = add_season(df)
    return {
        name: df.filter(pl.col("season").is_in(list(contract.SPLIT_SEASONS[name])))
        for name in SPLIT_NAMES
    }


def load_splits() -> dict[str, pl.DataFrame]:
    return split_frame(db.read_sql(f'SELECT * FROM "{MODEL_TABLE}"'))


def to_xy(df: pl.DataFrame):
    """Feature matrix (float, nulls -> NaN) and label vector for scikit-learn."""
    return feature_matrix(df), df[LABEL_COLUMN].to_numpy()
