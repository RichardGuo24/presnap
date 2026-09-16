"""Export the win-probability snapshot the deployed app reads.

    python -m scripts.export_wp    (or: .venv/bin/python scripts/export_wp.py)

Runs the model over every pre-snap scrimmage play once and writes
app/snapshot/wp_plays.parquet. Rerun after retraining, then commit the parquet.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from presnap import appdata  # noqa: E402


def main() -> int:
    n = appdata.export_snapshot()
    print(f"wrote {appdata.SNAPSHOT_PATH}: {n} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
