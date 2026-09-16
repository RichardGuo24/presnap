"""Evaluate + calibrate PreSnap on the untouched test split (Phase 4).

    python -m presnap.evaluate            # metrics + reliability plots
    python -m presnap.evaluate --no-plots

Protocol (keeps test pristine):
  * raw probabilities come from the GBM trained in Phase 3,
  * both calibrators are FIT on cal (2022),
  * the calibration method is CHOSEN on val (2021),
  * the winner is measured on test (2023-24) exactly once.

Reports overall Brier (+ reliability/resolution/uncertainty decomposition) and
ECE, then Brier + ECE segmented by quarter x score-margin. Writes reliability
diagrams to reports/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

from presnap.calibrate import CALIBRATORS
from presnap.split import load_splits, to_xy

REPORT_DIR = Path("reports")
GBM_PATH = Path("artifacts/gbm.joblib")
LOGISTIC_PATH = Path("artifacts/logistic.joblib")


# --------------------------------------------------------------------------- #
# pure metric helpers (imported by tests)
# --------------------------------------------------------------------------- #
def brier(y, p) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def _bin_index(p: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    return np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)


def ece(y, p, n_bins: int = 15) -> float:
    """Expected Calibration Error: sample-weighted |predicted - observed|."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    idx = _bin_index(p, n_bins)
    total = 0.0
    for k in range(n_bins):
        m = idx == k
        n = int(m.sum())
        if n:
            total += n * abs(p[m].mean() - y[m].mean())
    return float(total / len(y))


def brier_decomposition(y, p, n_bins: int = 10) -> tuple[float, float, float]:
    """Murphy decomposition: Brier ~= reliability - resolution + uncertainty."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    n_total = len(y)
    base = y.mean()
    idx = _bin_index(p, n_bins)
    rel = res = 0.0
    for k in range(n_bins):
        m = idx == k
        n = int(m.sum())
        if n:
            rel += n * (p[m].mean() - y[m].mean()) ** 2
            res += n * (y[m].mean() - base) ** 2
    return rel / n_total, res / n_total, float(base * (1 - base))


def reliability_bins(y, p, n_bins: int = 10):
    """Per-bin (mean predicted, observed frequency, count) for reliability plots."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    idx = _bin_index(p, n_bins)
    out = []
    for k in range(n_bins):
        m = idx == k
        n = int(m.sum())
        if n:
            out.append((float(p[m].mean()), float(y[m].mean()), n))
    return out


# --------------------------------------------------------------------------- #
# segmentation: quarter x score-margin (exhaustive, mutually exclusive)
# --------------------------------------------------------------------------- #
MARGIN_ORDER = ["<=-15", "-14..-8", "-7..-4", "-3..-1", "0",
                "1..3", "4..7", "8..14", ">=15"]


def margin_bucket(sd) -> np.ndarray:
    sd = np.asarray(sd)
    conds = [
        sd <= -15,
        (sd >= -14) & (sd <= -8),
        (sd >= -7) & (sd <= -4),
        (sd >= -3) & (sd <= -1),
        sd == 0,
        (sd >= 1) & (sd <= 3),
        (sd >= 4) & (sd <= 7),
        (sd >= 8) & (sd <= 14),
        sd >= 15,
    ]
    return np.select(conds, MARGIN_ORDER, default="?")


def quarter_label(q) -> np.ndarray:
    q = np.asarray(q).astype(int)
    return np.where(q >= 5, "OT", np.char.add("Q", q.astype(str)))


QUARTER_ORDER = ["Q1", "Q2", "Q3", "Q4", "OT"]


# --------------------------------------------------------------------------- #
def _probs(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def evaluate(make_plots: bool = True) -> int:
    splits = load_splits()
    _, y_train = to_xy(splits["train"])
    X_val, y_val = to_xy(splits["val"])
    X_cal, y_cal = to_xy(splits["cal"])
    X_test, y_test = to_xy(splits["test"])

    gbm = joblib.load(GBM_PATH)["model"]
    raw_val = _probs(gbm, X_val)
    raw_cal = _probs(gbm, X_cal)
    raw_test = _probs(gbm, X_test)

    # fit calibrators on CAL, choose the method on VAL
    fitted = {name: cls().fit(raw_cal, y_cal) for name, cls in CALIBRATORS.items()}
    val_options = {"raw": raw_val, **{n: c.predict(raw_val) for n, c in fitted.items()}}
    val_briers = {k: brier(y_val, v) for k, v in val_options.items()}
    chosen = min(val_briers, key=val_briers.get)
    print("method selection on val (2021):",
          {k: round(v, 4) for k, v in val_briers.items()}, "-> chosen:", chosen)

    p_test_cal = raw_test if chosen == "raw" else fitted[chosen].predict(raw_test)

    # ---- overall (test) ----
    base = float(y_train.mean())
    print(f"\nbaseline (predict {base:.3f}): test Brier = {brier(y_test, np.full(len(y_test), base)):.4f}")
    for label, p in [("raw", raw_test), (f"calibrated[{chosen}]", p_test_cal)]:
        rel, res, unc = brier_decomposition(y_test, p)
        print(f"gbm {label:18s}: Brier={brier(y_test, p):.4f}  ECE={ece(y_test, p):.4f}  "
              f"(reliability={rel:.4f} resolution={res:.4f} uncertainty={unc:.4f})")
    logistic = joblib.load(LOGISTIC_PATH)["model"]
    print(f"logistic (raw)        : test Brier = {brier(y_test, _probs(logistic, X_test)):.4f}")

    # ---- segmented: quarter x margin (on the calibrated probabilities) ----
    qseg = quarter_label(splits["test"]["f_qtr"].to_numpy())
    mseg = margin_bucket(splits["test"]["f_score_diff"].to_numpy())
    assert "?" not in set(mseg.tolist()), "a score margin fell outside all buckets"

    print("\ncalibrated ECE by quarter x margin (n in parens):")
    header = "  margin      " + "".join(f"{q:>12}" for q in QUARTER_ORDER)
    print(header)
    covered = 0
    worst = []
    for mlabel in MARGIN_ORDER:
        cells = []
        for qlabel in QUARTER_ORDER:
            m = (qseg == qlabel) & (mseg == mlabel)
            n = int(m.sum())
            covered += n
            if n >= 50:
                seg_ece = ece(y_test[m], p_test_cal[m], n_bins=10)
                cells.append(f"{seg_ece:.03f}({n})")
                worst.append((seg_ece, qlabel, mlabel, n))
            else:
                cells.append(f"-({n})" if n else "-")
        print(f"  {mlabel:<10}" + "".join(f"{c:>12}" for c in cells))
    assert covered == len(y_test), f"segment coverage {covered} != {len(y_test)}"

    worst.sort(reverse=True)
    print("\nworst-calibrated segments (ECE, n>=50):")
    for e, q, mrg, n in worst[:3]:
        print(f"  {q} {mrg}: ECE={e:.3f} (n={n})")

    if make_plots:
        from presnap import plots
        REPORT_DIR.mkdir(exist_ok=True)
        plots.reliability_overall(y_test, raw_test, p_test_cal, chosen,
                                  REPORT_DIR / "reliability_overall.png")
        plots.reliability_by_quarter(y_test, p_test_cal, qseg,
                                     REPORT_DIR / "reliability_by_quarter.png")
        print(f"\nwrote plots to {REPORT_DIR}/")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate + calibrate PreSnap.")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)
    return evaluate(make_plots=not args.no_plots)


if __name__ == "__main__":
    sys.exit(main())
