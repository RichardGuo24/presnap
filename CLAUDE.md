# PreSnap
NFL win probability pipeline. Python 3.12, venv at .venv, data via nflreadpy (returns Polars).

## Hard rule
No feature may use information from after the play it describes.
Every feature must be computable from pre-snap game state alone.

## Commands
Test: pytest
Run: python -m presnap.<module>