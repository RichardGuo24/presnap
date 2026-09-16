"""Serving path: pre-snap game state -> win probability.

Every frontend screen calls this. It reuses features.build_features and the
shared feature_matrix — it does NOT reimplement feature logic — so the numbers
the app shows are computed the same way the model was trained (no train/serve
skew).

posteam_wp() is the possessing team's win probability (what the model predicts);
home_wp() reorients it to the home team so a single game draws one continuous
curve instead of flipping on every change of possession.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import polars as pl

from presnap import features

GBM_PATH = Path("artifacts/gbm.joblib")

_model = None


def _load_model():
    global _model
    if _model is None:
        _model = joblib.load(GBM_PATH)["model"]
    return _model


def presnap_to_matrix(df: pl.DataFrame) -> np.ndarray:
    """The exact training transform: build_features -> feature_matrix."""
    return features.feature_matrix(features.build_features(df))


def posteam_wp(df: pl.DataFrame) -> np.ndarray:
    """Win probability for the team with the ball, per pre-snap row."""
    return _load_model().predict_proba(presnap_to_matrix(df))[:, 1]


def home_wp(df: pl.DataFrame) -> np.ndarray:
    """Win probability for the HOME team (continuous across possession changes)."""
    p = posteam_wp(df)
    is_home = (df["posteam_type"].str.strip_chars() == "home").to_numpy()
    return np.where(is_home, p, 1.0 - p)
