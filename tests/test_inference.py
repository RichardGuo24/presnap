"""Guard the train/serve backbone.

The serving feature path must be byte-identical to the training feature path —
otherwise the app would show numbers the model was never trained to produce.
Hermetic: no model file or DB needed (we only check the feature transform).
"""

import numpy as np

from presnap import features, inference


def test_serving_matrix_equals_training_feature_path(make_pbp):
    df = make_pbp([
        {"posteam_type": "home", "spread_line": -3.0, "score_differential": 7},
        {"posteam_type": "away", "spread_line": 6.5, "score_differential": -3, "temp": None},
        {"posteam_type": "home", "roof": "dome", "surface": "grass ", "down": 4},
    ])
    serving = inference.presnap_to_matrix(df)
    training = features.feature_matrix(features.build_features(df))
    # equal_nan: weather features are legitimately NaN and must line up too.
    assert np.array_equal(serving, training, equal_nan=True)
