"""Physical schema for PreSnap, single-sourced from the contract.

The raw `pbp` table stores a CURATED set of columns: the pre-snap allowlist, the
label, and a few post-play columns kept ONLY for verification / spot-checks
(never for features). We deliberately do not dump all ~370 pbp columns — storing
only what the pipeline needs avoids per-season dtype-inference roulette and keeps
the table legible.

The `presnap_play` VIEW is GENERATED from contract.PRESNAP_COLUMNS — the
allowlist literally becomes the view — so the view can never drift from the
contract. This is the PHYSICAL half of the no-lookahead guarantee: feature code
reads the view, where post-play columns simply do not exist, so touching one is a
SQL error rather than a silent wrong number. (This is why there is no static
schema.sql: duplicating the allowlist in SQL would let it drift.)

Integer-valued fields are stored as integers even though nflreadpy returns many
as Float64 — ingest casts them. Real-valued betting lines stay double. All
identifiers are quoted at DDL time because "desc" is a SQL reserved word.
"""

from __future__ import annotations

from presnap import contract

RAW_TABLE = "pbp"
PRESNAP_VIEW = "presnap_play"

# column -> Postgres type
RAW_SCHEMA: dict[str, str] = {
    # --- keys ---
    "game_id": "TEXT",
    "play_id": "BIGINT",
    # --- pre-snap context (known before kickoff) ---
    "season": "INTEGER",
    "season_type": "TEXT",
    "week": "INTEGER",
    "spread_line": "DOUBLE PRECISION",
    "total_line": "DOUBLE PRECISION",
    "roof": "TEXT",
    "surface": "TEXT",
    "temp": "INTEGER",
    "wind": "INTEGER",
    "div_game": "INTEGER",
    "location": "TEXT",
    "stadium_id": "TEXT",
    # --- pre-snap situational state ---
    "qtr": "INTEGER",
    "game_half": "TEXT",
    "down": "INTEGER",
    "ydstogo": "INTEGER",
    "goal_to_go": "INTEGER",
    "yardline_100": "INTEGER",
    "side_of_field": "TEXT",
    "drive": "INTEGER",
    "game_seconds_remaining": "INTEGER",
    "half_seconds_remaining": "INTEGER",
    "quarter_seconds_remaining": "INTEGER",
    "posteam_timeouts_remaining": "INTEGER",
    "defteam_timeouts_remaining": "INTEGER",
    # --- identity / possession ---
    "posteam": "TEXT",
    "defteam": "TEXT",
    "posteam_type": "TEXT",
    "home_team": "TEXT",
    "away_team": "TEXT",
    # --- pre-snap score (empirically verified pre-play; see scripts/verify_prepost.py) ---
    "posteam_score": "INTEGER",
    "defteam_score": "INTEGER",
    "score_differential": "INTEGER",
    # --- label (future-allowed; NOT exposed by the view) ---
    "result": "INTEGER",
    # --- verification / debug only (post-play; NOT exposed by the view) ---
    "play_type": "TEXT",
    "desc": "TEXT",
    "td_team": "TEXT",
    "total_home_score": "INTEGER",
    "total_away_score": "INTEGER",
}

# Everything the contract needs must be storable.
_missing = (contract.PRESNAP_COLUMNS | contract.LABEL_COLUMNS) - set(RAW_SCHEMA)
assert not _missing, f"RAW_SCHEMA is missing contract columns: {_missing}"


def q(ident: str) -> str:
    """Quote a SQL identifier (needed for reserved words like `desc`)."""
    return '"' + ident.replace('"', '""') + '"'


def create_raw_table_sql() -> str:
    cols = ",\n  ".join(f"{q(c)} {t}" for c, t in RAW_SCHEMA.items())
    return (
        f"CREATE TABLE IF NOT EXISTS {q(RAW_TABLE)} (\n  {cols},\n"
        f"  PRIMARY KEY ({q('game_id')}, {q('play_id')})\n);"
    )


def create_presnap_view_sql() -> str:
    """The allowlist becomes the view. Post-play columns in pbp are not selected,
    so they do not exist from a feature's perspective."""
    cols = ", ".join(q(c) for c in sorted(contract.PRESNAP_COLUMNS))
    return (
        f"CREATE OR REPLACE VIEW {q(PRESNAP_VIEW)} AS\n"
        f"SELECT {cols} FROM {q(RAW_TABLE)};"
    )
