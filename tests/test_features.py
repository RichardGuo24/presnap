"""Unit tests for individual feature and label values on known rows."""

from presnap import features, labels


def test_spread_is_oriented_to_posteam(make_pbp):
    # spread_line > 0 => home favored. f_posteam_spread flips for away posteam.
    df = make_pbp([
        {"posteam_type": "home", "spread_line": -6.5},   # home underdog
        {"posteam_type": "away", "spread_line": -6.5},   # posteam is the favorite
        {"posteam_type": "home", "spread_line": 3.0},    # home favored
    ])
    out = features.build_features(df)
    assert out["f_posteam_spread"].to_list() == [-6.5, 6.5, 3.0]


def test_home_and_categorical_flags(make_pbp):
    df = make_pbp([
        {"posteam_type": "home", "roof": "dome", "surface": "fieldturf", "season_type": "POST"},
        {"posteam_type": "away", "roof": "outdoors", "surface": "grass", "season_type": "REG"},
        {"posteam_type": "home", "roof": "closed", "surface": "matrixturf", "season_type": "REG"},
    ])
    out = features.build_features(df)
    assert out["f_is_home"].to_list() == [1, 0, 1]
    assert out["f_is_indoor"].to_list() == [1, 0, 1]
    assert out["f_is_turf"].to_list() == [1, 0, 1]
    assert out["f_is_postseason"].to_list() == [1, 0, 0]


def test_dirty_surface_is_normalized(make_pbp):
    # Real data has "grass " (trailing space) and empty surface; neither is turf.
    df = make_pbp([
        {"surface": "grass "},    # trailing space -> grass -> 0
        {"surface": "grass"},     # -> 0
        {"surface": "fieldturf"}, # -> 1
        {"surface": ""},          # empty -> unknown -> null
    ])
    out = features.build_features(df)
    assert out["f_is_turf"].to_list() == [0, 0, 1, None]


def test_passthrough_values(make_pbp):
    df = make_pbp([
        {"score_differential": -4, "down": 3, "ydstogo": 8, "yardline_100": 42,
         "game_seconds_remaining": 615, "posteam_timeouts_remaining": 2},
    ])
    out = features.build_features(df)
    row = out.row(0, named=True)
    assert row["f_score_diff"] == -4
    assert row["f_down"] == 3
    assert row["f_ydstogo"] == 8
    assert row["f_yardline_100"] == 42
    assert row["f_game_sec"] == 615
    assert row["f_pos_timeouts"] == 2


def test_label_win_loss_and_tie(make_pbp):
    df = make_pbp([
        {"result": 10, "posteam_type": "home"},   # home won, posteam home  -> 1
        {"result": 10, "posteam_type": "away"},    # home won, posteam away  -> 0
        {"result": -10, "posteam_type": "away"},   # away won, posteam away  -> 1
        {"result": -10, "posteam_type": "home"},   # away won, posteam home  -> 0
        {"result": 0, "posteam_type": "home"},     # tie                     -> dropped
    ])
    out = labels.build_labels(df)
    assert out.height == 4  # tie dropped
    assert out[labels.LABEL_COLUMN].to_list() == [1, 0, 1, 0]
