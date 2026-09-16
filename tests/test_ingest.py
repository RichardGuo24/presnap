"""Structural guards for Phase 1 ingestion.

These require a reachable Postgres with data already loaded, so they SKIP when
the DB is down (e.g. CI without a database — Phase 5 wires one in). They assert
the enforcement guarantees, not just that rows exist:
  - the view exposes EXACTLY the contract allowlist,
  - the view HIDES every post-play / label column,
  - keys are non-null,
  - loaded seasons actually have rows.
"""

import pytest

from presnap import contract, db
from presnap.schema import PRESNAP_VIEW, RAW_TABLE, q


def _ready() -> bool:
    try:
        with db.connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT to_regclass(%s), to_regclass(%s)",
                (RAW_TABLE, PRESNAP_VIEW),
            )
            table, view = cur.fetchone()
            return table is not None and view is not None
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _ready(), reason="Postgres/table/view not available (run ingest first)"
)


def _view_columns() -> set[str]:
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = %s",
            (PRESNAP_VIEW,),
        )
        return {r[0] for r in cur.fetchall()}


def test_view_exposes_exactly_the_allowlist():
    assert _view_columns() == set(contract.PRESNAP_COLUMNS)


def test_view_hides_post_play_and_label_columns():
    cols = _view_columns()
    for forbidden in [
        "total_home_score", "total_away_score", "play_type",
        "desc", "td_team", "result",
    ]:
        assert forbidden not in cols, f"{forbidden} leaked into {PRESNAP_VIEW}"


def test_keys_are_non_null():
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT count(*) FROM {q(RAW_TABLE)} "
            "WHERE game_id IS NULL OR play_id IS NULL"
        )
        assert cur.fetchone()[0] == 0


def test_loaded_seasons_have_rows():
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {q(RAW_TABLE)}")
        assert cur.fetchone()[0] > 0
