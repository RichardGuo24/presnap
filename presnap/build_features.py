"""Build the model frame: features + label, one row per pre-snap scrimmage play.

    python -m presnap.build_features

Reads features from the presnap_play VIEW (physically can't see post-play
columns) and the label from the raw pbp table (via labels.py, the one
future-allowed module). Keeps only scrimmage plays (down IS NOT NULL) and drops
tie games (null label). Writes the `model_frame` table, fully rebuilt each run.
"""

from __future__ import annotations

import sys

from presnap import db, features, labels
from presnap.labels import LABEL_COLUMN
from presnap.schema import PRESNAP_VIEW, RAW_TABLE, q

MODEL_TABLE = "model_frame"

# Pre-snap-safe row filter: down is set exactly on scrimmage plays. We CANNOT
# filter on play_type here even if we wanted to — the view doesn't expose it.
SCRIMMAGE = "down IS NOT NULL"


def build() -> int:
    feat_src = db.read_sql(f"SELECT * FROM {q(PRESNAP_VIEW)} WHERE {SCRIMMAGE}")
    label_src = db.read_sql(
        f"SELECT game_id, play_id, result, posteam_type "
        f"FROM {q(RAW_TABLE)} WHERE {SCRIMMAGE}"
    )

    feats = features.build_features(feat_src)
    labs = labels.build_labels(label_src)

    # inner join drops tie-game rows (they have no label)
    model_frame = feats.join(labs, on=["game_id", "play_id"], how="inner")

    db.write_dataframe(MODEL_TABLE, model_frame, pk=("game_id", "play_id"))

    n_pos = model_frame[LABEL_COLUMN].sum()
    print(f"wrote {MODEL_TABLE}: {model_frame.height} rows, "
          f"{len(features.FEATURE_COLUMNS)} features, "
          f"posteam win rate {n_pos / model_frame.height:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(build())
