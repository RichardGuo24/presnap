"""Ingest nflreadpy play-by-play into Postgres (Phase 1).

    python -m presnap.ingest --seasons 2010:2024
    python -m presnap.ingest --seasons 2022,2023

Idempotent per season: re-running a season deletes its rows then re-inserts, so
counts never inflate. Only the curated RAW_SCHEMA columns are stored; a season
missing any of them fails loudly (schema-drift guard) instead of silently
nulling. Integer-valued columns are cast from Float64 so Postgres INTEGER columns
don't choke on "6.0".
"""

from __future__ import annotations

import argparse
import io
import sys

import nflreadpy as nfl
import polars as pl

from presnap import db
from presnap.schema import (
    RAW_SCHEMA,
    RAW_TABLE,
    PRESNAP_VIEW,
    create_presnap_view_sql,
    create_raw_table_sql,
    q,
)

_INT_TYPES = {"INTEGER", "BIGINT", "SMALLINT"}


def _cast_expr(col: str, pgtype: str) -> pl.Expr:
    e = pl.col(col)
    if pgtype in _INT_TYPES:
        return e.cast(pl.Int64)
    if pgtype == "DOUBLE PRECISION":
        return e.cast(pl.Float64)
    return e.cast(pl.Utf8)


def load_season(season: int) -> pl.DataFrame:
    df = nfl.load_pbp([season])
    if hasattr(df, "collect"):
        df = df.collect()
    cols = list(RAW_SCHEMA)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"season {season} missing expected columns {missing} "
            "(schema drift — refusing to load silently)"
        )
    return df.select([_cast_expr(c, RAW_SCHEMA[c]).alias(c) for c in cols])


def _copy_into(conn, df: pl.DataFrame) -> None:
    cols = list(RAW_SCHEMA)
    buf = io.BytesIO()
    df.select(cols).write_csv(buf, include_header=False)
    collist = ", ".join(q(c) for c in cols)
    with conn.cursor() as cur:
        with cur.copy(
            f"COPY {q(RAW_TABLE)} ({collist}) FROM STDIN WITH (FORMAT CSV)"
        ) as cp:
            cp.write(buf.getvalue())


def ingest(seasons: list[int]) -> None:
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(create_raw_table_sql())
        conn.commit()

        for season in seasons:
            df = load_season(season)
            # delete + copy in one transaction so a season is never half-loaded
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {q(RAW_TABLE)} WHERE season = %s", (season,))
            _copy_into(conn, df)
            conn.commit()
            print(f"loaded {season}: {df.height} rows")

        with conn.cursor() as cur:
            cur.execute(create_presnap_view_sql())
        conn.commit()
    print(f"done — view {PRESNAP_VIEW} rebuilt from the contract allowlist.")


def parse_seasons(spec: str) -> list[int]:
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            a, b = part.split(":")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Ingest NFL play-by-play into Postgres.")
    p.add_argument("--seasons", required=True, help="e.g. 2010:2024 or 2022,2023")
    args = p.parse_args(argv)
    ingest(parse_seasons(args.seasons))
    return 0


if __name__ == "__main__":
    sys.exit(main())
