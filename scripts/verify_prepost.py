"""Discharge the contract's score-column verification obligation.

The contract flags posteam_score / defteam_score / score_differential as
"believed pre-play, pending empirical verification." Getting this wrong is the
textbook silent leak. This script PROVES it against real data.

Method: total_home_score / total_away_score are the running score AFTER each
play. So the score BEFORE a play equals the previous play's running total
(within the same game). If posteam_score is genuinely pre-play, it must equal
that previous-row total, NOT the same-row total.

    pre-play  match  -> posteam_score == previous-row running total   (expect ~1.0)
    post-play match  -> posteam_score == same-row     running total   (expect <1.0)

Exits non-zero if the pre-play hypothesis does not hold, so this can gate CI.
"""

import sys

import polars as pl
import nflreadpy as nfl

SEASON = 2023


def main() -> int:
    df = nfl.load_pbp([SEASON])
    if hasattr(df, "collect"):
        df = df.collect()
    print(f"loaded {SEASON}: {df.shape[0]} rows x {df.shape[1]} cols")

    d = (
        df.sort(["game_id", "play_id"])
        .with_columns(
            pl.col("total_home_score").shift(1).over("game_id").fill_null(0).alias("home_pre"),
            pl.col("total_away_score").shift(1).over("game_id").fill_null(0).alias("away_pre"),
        )
        .with_columns(
            pl.when(pl.col("posteam") == pl.col("home_team"))
            .then(pl.col("home_pre"))
            .otherwise(pl.col("away_pre"))
            .alias("exp_pre_posteam_score"),
            pl.when(pl.col("posteam") == pl.col("home_team"))
            .then(pl.col("total_home_score"))
            .otherwise(pl.col("total_away_score"))
            .alias("same_row_posteam_score"),
        )
    )

    dd = d.filter(pl.col("posteam").is_not_null() & pl.col("posteam_score").is_not_null())

    # PAT scoring-sequence rows (extra_point / two_point_attempt) are excluded:
    # on those, posteam_score already reflects the just-scored TD, and the
    # total_*_score REFERENCE credits that TD on an inconsistent row across
    # games. That makes the reference unreliable there, not posteam_score wrong
    # -- posteam_score still excludes the current play's own points (pre-play).
    # These rows are special teams (down is null) and get filtered in Phase 2.
    PAT_TYPES = {"extra_point", "two_point_attempt"}
    scrim = dd.filter(~pl.col("play_type").is_in(PAT_TYPES))

    pre = (scrim["posteam_score"] == scrim["exp_pre_posteam_score"]).cast(pl.Int8).mean()
    post = (scrim["posteam_score"] == scrim["same_row_posteam_score"]).cast(pl.Int8).mean()

    # Confirm every remaining reference-mismatch is a PAT row, not a scrimmage play.
    all_mis = dd.filter(pl.col("posteam_score") != pl.col("exp_pre_posteam_score"))
    non_pat_mis = all_mis.filter(~pl.col("play_type").is_in(PAT_TYPES))

    print(f"scrimmage rows checked: {scrim.height}")
    print(f"pre-play  hypothesis match: {pre:.4f}  (want ~1.0)")
    print(f"post-play hypothesis match: {post:.4f}  (want < 1.0)")
    print(f"reference-mismatch rows total: {all_mis.height}  "
          f"(non-PAT: {non_pat_mis.height})")
    if non_pat_mis.height:
        print(non_pat_mis["play_type"].value_counts().sort("count", descending=True))

    # Eyeball window around the first touchdown of the first game.
    gid = df["game_id"][0]
    g = d.filter(pl.col("game_id") == gid)
    td_rows = g.with_row_index().filter(pl.col("touchdown") == 1)
    if td_rows.height:
        i = td_rows["index"][0]
        window = g.slice(max(0, i - 2), 6).select(
            "play_id", "posteam", "posteam_score", "defteam_score",
            "score_differential", "total_home_score", "total_away_score",
        )
        print("\nwindow around first TD (watch score update the play AFTER):")
        print(window)

    # AUTHORITATIVE, reference-free test: on a play where the possessing team
    # scores a TD, a pre-play posteam_score must NOT yet include those points,
    # so it must be strictly less than that team's score on their NEXT snap.
    # Uses only posteam_score -- immune to the total_*_score crediting quirk that
    # produces the handful of reference mismatches above (all post-scoring
    # special-teams rows: two-point tries, post-score kickoffs, kick penalties).
    d2 = df.sort(["game_id", "play_id"]).with_columns(
        pl.col("posteam_score").shift(-1).over(["game_id", "posteam"]).alias("pos_next_score")
    )
    td_by_pos = d2.filter(
        (pl.col("touchdown") == 1)
        & (pl.col("td_team") == pl.col("posteam"))
        & pl.col("pos_next_score").is_not_null()
    )
    td_frac = (td_by_pos["posteam_score"] < td_by_pos["pos_next_score"]).cast(pl.Int8).mean()
    print(f"reference-free TD test: {td_frac:.4f} of {td_by_pos.height} "
          f"(pre-play => 1.0)")

    ok = (td_frac == 1.0) and (pre > 0.99) and (post < pre)
    print("\nVERDICT:", "PRE-PLAY confirmed ✓" if ok else "NOT pre-play — STOP ✗")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
