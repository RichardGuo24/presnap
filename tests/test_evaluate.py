"""Guards for the evaluation metrics and segmentation.

Hermetic — pure numpy, no model or DB.
"""

import numpy as np

from presnap import evaluate


def test_brier_of_constant_half_is_quarter():
    y = np.array([0, 1, 0, 1, 1, 0])
    assert abs(evaluate.brier(y, np.full(len(y), 0.5)) - 0.25) < 1e-12


def test_brier_constant_matches_closed_form():
    # For a constant forecast c: Brier = (1-p)*c^2 + p*(1-c)^2, p = mean(y).
    y = np.array([0, 1, 1, 0, 1, 1, 1, 0])
    c = 0.3
    p = y.mean()
    expected = (1 - p) * c**2 + p * (1 - c) ** 2
    assert abs(evaluate.brier(y, np.full(len(y), c)) - expected) < 1e-12


def test_every_margin_maps_to_exactly_one_bucket():
    sd = np.arange(-60, 61)  # every integer margin we could see
    buckets = evaluate.margin_bucket(sd)
    assert "?" not in set(buckets.tolist()), "a margin fell outside all buckets"
    assert len(buckets) == len(sd)
    assert set(buckets.tolist()) <= set(evaluate.MARGIN_ORDER)


def test_segments_cover_every_row_exactly_once():
    rng = np.random.default_rng(0)
    sd = rng.integers(-40, 41, size=1000)
    qtr = rng.integers(1, 7, size=1000)  # include double-OT (6) -> "OT"
    mseg = evaluate.margin_bucket(sd)
    qseg = evaluate.quarter_label(qtr)
    covered = 0
    for q in evaluate.QUARTER_ORDER:
        for m in evaluate.MARGIN_ORDER:
            covered += int(((qseg == q) & (mseg == m)).sum())
    assert covered == len(sd)  # no row dropped, none double-counted


def test_ece_zero_for_perfect_calibration():
    # If predictions equal outcomes, calibration error is 0.
    y = np.array([0, 0, 1, 1, 0, 1])
    assert evaluate.ece(y, y.astype(float)) == 0.0


def test_decomposition_reconstructs_brier_reasonably():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=5000)
    p = np.clip(0.5 + 0.3 * (y - 0.5) + rng.normal(0, 0.1, size=5000), 0, 1)
    rel, res, unc = evaluate.brier_decomposition(y, p)
    # rel - res + unc approximates Brier (exact up to within-bin variance)
    assert abs((rel - res + unc) - evaluate.brier(y, p)) < 0.02
