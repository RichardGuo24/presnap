"""Guards for the phase-0 contract.

These do not touch the database or nflreadpy — they assert the contract is
internally coherent and that no known post-play column has crept into the
allowlist. Fast, run on every CI push.
"""

from presnap import contract


def test_grain_keys_are_allowlisted():
    assert contract.KEY_COLUMNS <= contract.PRESNAP_COLUMNS


def test_segmentation_columns_survive_to_evaluation():
    # Phase 4 segments by these; if they are not pre-snap they can't be carried.
    assert contract.SEGMENT_COLUMNS <= contract.PRESNAP_COLUMNS


def test_label_is_not_a_feature():
    assert not (contract.LABEL_COLUMNS & contract.PRESNAP_COLUMNS)


def test_no_known_post_play_column_is_allowlisted():
    # The dangerous ones: the answer itself, post-play deltas, outcome columns,
    # and post-play running scores. If any appears in the allowlist, fail loudly.
    forbidden = {
        "wp", "vegas_wp", "def_wp", "home_wp", "away_wp",  # the answer
        "wpa", "epa", "ep",                                # deltas / model output
        "play_type", "yards_gained", "air_yards",          # play outcomes
        "complete_pass", "touchdown", "interception",
        "fumble", "sack", "first_down",
        "total_home_score", "total_away_score",            # post-play running score
        "posteam_score_post", "defteam_score_post",
        "score_differential_post",
        "result", "total",                                 # final-game outcomes
    }
    leaked = forbidden & contract.PRESNAP_COLUMNS
    assert not leaked, f"post-play columns leaked into the allowlist: {leaked}"


def test_excluded_and_allowlisted_are_disjoint():
    assert not (contract.PRESNAP_COLUMNS & set(contract.EXPLICITLY_EXCLUDED))


def test_split_seasons_are_disjoint():
    seen: set[int] = set()
    for seasons in contract.SPLIT_SEASONS.values():
        for yr in seasons:
            assert yr not in seen, f"season {yr} assigned to two splits"
            seen.add(yr)


def test_calibration_holdout_is_separate_from_train_and_test():
    cal = set(contract.SPLIT_SEASONS["cal"])
    train = set(contract.SPLIT_SEASONS["train"])
    test = set(contract.SPLIT_SEASONS["test"])
    assert not (cal & train) and not (cal & test)


def test_score_columns_pending_verification_are_flagged_and_allowlisted():
    # They are in use (so allowlisted) but explicitly marked as needing the
    # ingest-time pre/post check. This test documents that coupling.
    assert contract.SCORE_COLUMNS_PENDING_VERIFICATION <= contract.PRESNAP_COLUMNS
    assert "score_differential" in contract.SCORE_COLUMNS_PENDING_VERIFICATION
