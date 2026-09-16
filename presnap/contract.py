"""PreSnap contract — the decisions every later phase imports rather than re-decides.

This module is the spine of the pipeline. The five decisions below were hoisted
here (out of phases 1/3/4) because each one, if made late, becomes a
cross-boundary dependency that silently corrupts results:

    1. Prediction grain      — one row = one pre-snap moment; key (game_id, play_id).
    2. Label definition       — target is "did `posteam` win this game".
    3. Split policy           — temporal, by season, partitioned on game_id,
                                with a dedicated calibration holdout.
    4. Segmentation columns   — qtr and score_differential survive untransformed
                                to evaluation, or phase 4 cannot segment.
    5. Pre-snap allowlist     — the frozenset of raw columns a feature may read.
                                This *is* the hard rule, encoded.

------------------------------------------------------------------------------
THE HARD RULE (from CLAUDE.md)
    No feature may use information from after the play it describes.
    Every feature must be computable from pre-snap game state alone.

TWO SEPARATE GUARANTEES — do not conflate them
    (a) Features read only allowlisted columns.
        Enforced by machinery: the `presnap_play` view (physical) + the
        perturbation test in phase 2 (behavioral). If a feature touches a
        non-allowlisted column, output changes and CI goes red.

    (b) The allowlist is *correct* — every column in it is truly pre-snap.
        NOT enforced by the perturbation test. If a post-play column were
        wrongly allowlisted, the test would leave it untouched and PASS.
        Guarantee (b) is discharged by an ingest-time pre/post check
        (see SCORE_COLUMNS_PENDING_VERIFICATION below) and by human review
        of this file. That is why the allowlist lives in one auditable place.
------------------------------------------------------------------------------
"""

from __future__ import annotations

# =============================================================================
# 1. PREDICTION GRAIN
# =============================================================================
# One row = one pre-snap moment (one play). This is the unit we predict on and
# the unit downstream joins on. Fixing it here stops later phases from silently
# regrinding the data (e.g. to drive level) and breaking every join.
PREDICTION_GRAIN: tuple[str, str] = ("game_id", "play_id")

# Row-identity columns. Not features — carried so every table can be joined back
# to the play it came from.
KEY_COLUMNS: frozenset[str] = frozenset({"game_id", "play_id"})


# =============================================================================
# 2. LABEL DEFINITION
# =============================================================================
# Target = "did the team currently on offense (`posteam`) win this game?"
#
# This is derived from the FINAL score, which is a post-play value. That is
# allowed: it is the *label*, not a feature. The label is the one thing
# permitted to see the future, and it is built in exactly one module
# (presnap/labels.py) so that "allowed to see the future" has a single,
# auditable surface.
#
# Perspective (posteam vs. home) is fixed HERE so it cannot silently flip in
# phase 3/4. A flipped label is a classic silent bug: the model learns the
# inverse relationship and Brier stays deceptively mediocre instead of
# obviously broken.
#
# `result` in the pbp data is the final home margin (home_final - away_final).
# Combined with `posteam_type` (home/away, an allowlisted pre-snap column):
#     posteam won  <=>  (result > 0) == (posteam_type == "home")
# Ties (result == 0, rare regular-season OT) are an open decision for
# labels.py: drop them or label 0.5. Flagged, not silently coerced.
LABEL_PERSPECTIVE: str = "posteam"

# Columns the label is allowed to read. Deliberately tiny. MUST be disjoint
# from PRESNAP_COLUMNS (asserted at import).
LABEL_COLUMNS: frozenset[str] = frozenset({"result"})


# =============================================================================
# 3. SPLIT POLICY
# =============================================================================
# Temporal split, BY SEASON, partitioned on game_id (never on play_id): plays
# within a game are correlated, so a random play-level split leaks. A dedicated
# calibration holdout is reserved so phase 4's calibrator never touches train or
# test. Encoding the concrete season assignment here (not in phase 3) makes this
# the single source of truth for what "test" means.
SPLIT_UNIT: str = "game_id"        # partition key — no game spans two splits
SPLIT_DIMENSION: str = "season"    # temporal ordering axis

# Default season assignment for a 2010–2024 pull (15 seasons). Configurable, but
# this is the canonical default; phase 3 imports it instead of inventing one.
SPLIT_SEASONS: dict[str, tuple[int, ...]] = {
    "train": tuple(range(2010, 2021)),  # 2010–2020
    "val":   (2021,),                   # model selection / early stopping
    "cal":   (2022,),                   # calibrator fit ONLY — never train/test
    "test":  (2023, 2024),              # touched once, at the very end
}


# =============================================================================
# 4. SEGMENTATION CARRY-THROUGH COLUMNS
# =============================================================================
# Phase 4 reports calibration segmented by quarter x score margin. Those two
# columns must reach evaluation UNTRANSFORMED (not scaled away, not dropped) or
# segmentation is impossible. Declaring them here forces phase 2 to preserve
# them. Both are also in PRESNAP_COLUMNS (asserted at import).
SEGMENT_COLUMNS: frozenset[str] = frozenset({"qtr", "score_differential"})


# =============================================================================
# 5. PRE-SNAP ALLOWLIST
# =============================================================================
# The only raw columns a feature may read. Grouped by kind for auditability.
# Column names follow the nflfastR / nflreadpy load_pbp schema.

# --- Situational game state, known before the snap ---------------------------
# These describe the pre-snap situation and cannot be post-play outcomes.
_PRESNAP_STATE: frozenset[str] = frozenset({
    "qtr",                         # quarter (1-4, 5=OT)
    "game_half",                   # Half1 / Half2 / Overtime
    "down",                        # 1-4 (null on kickoffs/etc.)
    "ydstogo",                     # yards to first down
    "goal_to_go",                  # 1 if goal-to-go situation
    "yardline_100",                # distance to opponent end zone (0-100)
    "side_of_field",               # which team's side the ball is on
    "drive",                       # drive number within the game
    "game_seconds_remaining",      # seconds left in the game
    "half_seconds_remaining",      # seconds left in the half
    "quarter_seconds_remaining",   # seconds left in the quarter
    "posteam_timeouts_remaining",  # offense timeouts left
    "defteam_timeouts_remaining",  # defense timeouts left
})

# --- Team / possession identity ----------------------------------------------
_PRESNAP_IDENTITY: frozenset[str] = frozenset({
    "posteam",       # team on offense
    "defteam",       # team on defense
    "posteam_type",  # "home" / "away" — also used to orient the label
    "home_team",
    "away_team",
})

# --- Pre-game context, fixed before kickoff ----------------------------------
# All known before the game starts, so trivially pre-snap for every play.
_PRESNAP_CONTEXT: frozenset[str] = frozenset({
    "season",
    "season_type",   # REG / POST
    "week",
    "spread_line",   # closing point spread (home perspective)
    "total_line",    # closing over/under
    "roof",          # dome / outdoors / closed / open
    "surface",       # grass / turf variants
    "temp",          # game-time temperature
    "wind",          # game-time wind
    "div_game",      # 1 if divisional matchup
    "location",      # Home / Neutral
    "stadium_id",
})

# --- Score columns: BELIEVED pre-play, PENDING empirical verification --------
# In nflfastR, posteam_score / defteam_score / score_differential are documented
# as the score at the START of the play, while total_home_score / total_away_score
# and the *_post variants are AFTER the play. Getting this backwards is the
# textbook silent leak this whole project guards against.
#
# These are in the allowlist because the pipeline needs them (score_differential
# is also a segmentation column), BUT their pre-play semantics are an INGEST-TIME
# VERIFICATION OBLIGATION, not an assumption. Phase 1 must confirm empirically —
# e.g. that within a scoring drive, score_differential updates on the play AFTER
# the score, not the scoring play itself — before these are trusted.
#
# Reminder: the phase-2 perturbation test CANNOT catch a mislabeled column here,
# because allowlisted columns are left untouched by that test. This comment is
# the guard; the ingest check is the enforcement.
SCORE_COLUMNS_PENDING_VERIFICATION: frozenset[str] = frozenset({
    "posteam_score",
    "defteam_score",
    "score_differential",
})

# The effective allowlist.
PRESNAP_COLUMNS: frozenset[str] = (
    KEY_COLUMNS
    | _PRESNAP_STATE
    | _PRESNAP_IDENTITY
    | _PRESNAP_CONTEXT
    | SCORE_COLUMNS_PENDING_VERIFICATION
)


# --- Explicitly excluded, with reasons (documentation, not enforcement) ------
# Not exhaustive — the pbp table has ~370 columns and the allowlist is a
# whitelist, so anything absent is excluded by default. These are called out
# because they are the tempting mistakes.
EXPLICITLY_EXCLUDED: dict[str, str] = {
    "wp":            "the answer — nflfastR's own win probability",
    "vegas_wp":      "the answer — spread-adjusted win probability",
    "def_wp":        "the answer, defense perspective",
    "home_wp":       "the answer, home perspective",
    "away_wp":       "the answer, away perspective",
    "wpa":           "win-probability added — post-play delta",
    "ep":            "expected points; a MODEL output, not raw pre-snap state",
    "epa":           "expected-points added — post-play delta",
    "play_type":     "run/pass/punt is the play OUTCOME, revealed at the snap",
    "yards_gained":  "post-play outcome",
    "total_home_score": "running score AFTER the play (post-play)",
    "total_away_score": "running score AFTER the play (post-play)",
    "posteam_score_post":      "post-play by name",
    "score_differential_post": "post-play by name",
    "result":        "final game margin — this is the LABEL, not a feature",
}


# =============================================================================
# IMPORT-TIME INVARIANTS — fail loudly if the contract is internally broken
# =============================================================================
# A broken contract must crash on import, not produce a subtly wrong pipeline.

def _validate_contract() -> None:
    # Grain keys must be inside the allowlist so every table can join back.
    assert KEY_COLUMNS <= PRESNAP_COLUMNS, \
        "KEY_COLUMNS must be a subset of PRESNAP_COLUMNS"

    # Segmentation columns must survive to evaluation => must be pre-snap.
    missing_seg = SEGMENT_COLUMNS - PRESNAP_COLUMNS
    assert not missing_seg, \
        f"SEGMENT_COLUMNS not in the allowlist (phase 4 can't segment): {missing_seg}"

    # The label must NOT be readable as a feature.
    leaked = LABEL_COLUMNS & PRESNAP_COLUMNS
    assert not leaked, \
        f"LABEL_COLUMNS leaked into the allowlist: {leaked}"

    # Nothing explicitly excluded may appear in the allowlist.
    contradiction = PRESNAP_COLUMNS & set(EXPLICITLY_EXCLUDED)
    assert not contradiction, \
        f"columns are both allowlisted and explicitly excluded: {contradiction}"

    # No split may share a season with another (temporal split must be clean).
    seen: dict[int, str] = {}
    for split, seasons in SPLIT_SEASONS.items():
        for yr in seasons:
            assert yr not in seen, \
                f"season {yr} assigned to both {seen[yr]!r} and {split!r}"
            seen[yr] = split


_validate_contract()
