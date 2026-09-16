"""Reliability diagrams for Phase 4.

A reliability diagram plots mean predicted probability (x) against observed
frequency (y); perfect calibration lies on the y=x diagonal. Colors are the
Okabe-Ito colorblind-safe palette (blue = raw, vermillion = calibrated); the two
series also carry a legend, so identity never rests on color alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from presnap.evaluate import reliability_bins  # noqa: E402

BLUE = "#0072B2"       # raw
VERMILLION = "#D55E00"  # calibrated
GRAY = "#8A8A8A"        # diagonal reference
INK = "#2B2B2B"
GRID = "#E8E8E8"


def _style(ax) -> None:
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#CCCCCC")
    ax.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.tick_params(colors=INK, labelsize=9)


def _curve(ax, y, p, color, label, n_bins=12) -> None:
    bins = reliability_bins(y, p, n_bins)
    xs = [b[0] for b in bins]
    ys = [b[1] for b in bins]
    ax.plot(xs, ys, "-o", color=color, lw=2, ms=6, label=label, zorder=3)


def reliability_overall(y, raw, cal, chosen, path: Path) -> None:
    fig, (ax, axh) = plt.subplots(
        2, 1, figsize=(6, 6.6), height_ratios=[3, 1], sharex=True
    )
    ax.plot([0, 1], [0, 1], "--", color=GRAY, lw=1.5, zorder=1, label="perfect")
    _curve(ax, y, raw, BLUE, "raw")
    _curve(ax, y, cal, VERMILLION, f"calibrated ({chosen})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_ylabel("Observed win frequency", color=INK, fontsize=10)
    ax.set_title("Reliability — GBM (test 2023–24)", color=INK, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _style(ax)

    axh.hist(raw, bins=25, range=(0, 1), color=BLUE, alpha=0.75)
    axh.set_xlabel("Predicted win probability", color=INK, fontsize=10)
    axh.set_ylabel("count", color=INK, fontsize=9)
    _style(axh)
    axh.grid(False)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def reliability_by_quarter(y, p, qseg, path: Path) -> None:
    order = ["Q1", "Q2", "Q3", "Q4", "OT"]
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.1), sharex=True, sharey=True)
    for ax, q in zip(axes, order):
        m = qseg == q
        ax.plot([0, 1], [0, 1], "--", color=GRAY, lw=1.2)
        if int(m.sum()) > 0:
            _curve(ax, y[m], p[m], VERMILLION, None, n_bins=10)
        ax.set_title(f"{q}  (n={int(m.sum())})", fontsize=10, color=INK)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        _style(ax)
    axes[0].set_ylabel("Observed win frequency", color=INK, fontsize=10)
    fig.suptitle("Reliability by quarter — calibrated (test)",
                 color=INK, fontsize=12, x=0.02, ha="left")
    fig.supxlabel("Predicted win probability", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
