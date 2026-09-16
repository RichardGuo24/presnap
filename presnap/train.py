"""Train PreSnap models and report validation Brier score (Phase 3).

    python -m presnap.train                # train all models
    python -m presnap.train --model gbm

Trains on the train split, evaluates Brier on the val split (2021), and saves
each fitted model to artifacts/<name>.joblib alongside the feature list. The test
split is NOT touched here — it is reserved for Phase 4.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import brier_score_loss

from presnap.features import FEATURE_COLUMNS
from presnap.model import MODELS
from presnap.split import load_splits, to_xy

ARTIFACT_DIR = Path("artifacts")


def train(model_names: list[str]) -> int:
    splits = load_splits()
    X_train, y_train = to_xy(splits["train"])
    X_val, y_val = to_xy(splits["val"])

    # Reference: a model that always predicts the training base rate.
    base_rate = float(y_train.mean())
    base_brier = brier_score_loss(y_val, np.full(len(y_val), base_rate))
    print(f"baseline (always predict {base_rate:.3f}): val Brier = {base_brier:.4f}")

    ARTIFACT_DIR.mkdir(exist_ok=True)
    for name in model_names:
        model = MODELS[name]()
        model.fit(X_train, y_train)
        p_val = model.predict_proba(X_val)[:, 1]
        brier = brier_score_loss(y_val, p_val)
        path = ARTIFACT_DIR / f"{name}.joblib"
        joblib.dump({"model": model, "features": FEATURE_COLUMNS}, path)
        print(f"{name:9s}: val Brier = {brier:.4f}  (saved {path})")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train PreSnap win-probability models.")
    p.add_argument("--model", choices=[*MODELS, "all"], default="all")
    args = p.parse_args(argv)
    names = list(MODELS) if args.model == "all" else [args.model]
    return train(names)


if __name__ == "__main__":
    sys.exit(main())
