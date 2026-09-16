"""Probability calibrators, fit on the CAL split only.

Each calibrator maps a model's raw win probability -> a calibrated one. They
operate on probability vectors (not the model), so they are decoupled from which
estimator produced the scores.

  platt    : a 1-D logistic fit on the log-odds of the raw probability. Best when
             miscalibration is a smooth S-curve; few parameters, hard to overfit.
  isotonic : a monotonic step function. More flexible, needs more data, can
             overfit — which is exactly why it is fit on the dedicated CAL split
             and chosen on VAL, never judged on the data it was fit on.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class PlattCalibrator:
    def fit(self, p_raw, y):
        self._lr = LogisticRegression()
        self._lr.fit(_logit(p_raw).reshape(-1, 1), np.asarray(y))
        return self

    def predict(self, p_raw) -> np.ndarray:
        return self._lr.predict_proba(_logit(p_raw).reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    def fit(self, p_raw, y):
        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._iso.fit(np.asarray(p_raw, dtype=float), np.asarray(y, dtype=float))
        return self

    def predict(self, p_raw) -> np.ndarray:
        return self._iso.predict(np.asarray(p_raw, dtype=float))


CALIBRATORS = {
    "platt": PlattCalibrator,
    "isotonic": IsotonicCalibrator,
}
