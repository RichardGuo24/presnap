"""Model definitions for PreSnap.

Two estimators, both scikit-learn:

  logistic : SimpleImputer(median) -> StandardScaler -> LogisticRegression.
             A linear baseline. Imputation/scaling live INSIDE the pipeline so
             they are fit on the training fold only (no cross-split leakage) —
             this is the deferred imputation promised in Phase 2.

  gbm      : HistGradientBoostingClassifier. Handles NaN natively (so temp/wind
             nulls need no imputation) and captures interactions like
             lead x time-remaining that a linear model can't.

Both output calibrated-*able* probabilities; Phase 4 does the actual calibration.
"""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_logistic() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )


def make_gbm() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=400,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        early_stopping=True,   # carves an internal validation set from TRAIN only
        random_state=0,
    )


MODELS = {
    "logistic": make_logistic,
    "gbm": make_gbm,
}
