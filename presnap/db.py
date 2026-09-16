"""Database connection for PreSnap.

One place that knows how to reach Postgres; everything else imports connect().
This is the seam that keeps the storage choice reversible — point
PRESNAP_DB_URL elsewhere (another host, or a different engine's URL) and no
caller changes.
"""

from __future__ import annotations

import io
import os

import polars as pl
import psycopg

DEFAULT_DB_URL = "postgresql://presnap:presnap@localhost:5432/presnap"


def db_url() -> str:
    return os.environ.get("PRESNAP_DB_URL", DEFAULT_DB_URL)


def connect() -> psycopg.Connection:
    return psycopg.connect(db_url())


def read_sql(sql: str, params: tuple | None = None) -> pl.DataFrame:
    """Run a query and return a Polars DataFrame."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d.name for d in cur.description]
        rows = cur.fetchall()
    if not rows:
        return pl.DataFrame(schema={c: pl.Utf8 for c in cols})
    # infer_schema_length=None scans ALL rows: a column that is null for the
    # first 100 rows (e.g. temp in early dome games) would otherwise be inferred
    # as Null and fail when a real value appears later.
    return pl.DataFrame(rows, schema=cols, orient="row", infer_schema_length=None)


def _pg_type(dt: pl.DataType) -> str:
    if dt in (pl.Int8, pl.Int16, pl.Int32, pl.UInt8, pl.UInt16, pl.UInt32):
        return "INTEGER"
    if dt in (pl.Int64, pl.UInt64):
        return "BIGINT"
    if dt in (pl.Float32, pl.Float64):
        return "DOUBLE PRECISION"
    if dt == pl.Boolean:
        return "BOOLEAN"
    return "TEXT"


def write_dataframe(table: str, df: pl.DataFrame, pk: tuple[str, ...] | None = None) -> None:
    """Replace `table` with the contents of `df` (drop + create + COPY).

    Used for fully-derived tables (features) that are cheaper to rebuild than to
    upsert. Column types are mapped from the DataFrame's Polars dtypes.
    """
    def qi(ident: str) -> str:
        return '"' + ident.replace('"', '""') + '"'

    cols = df.columns
    coldefs = ", ".join(f"{qi(c)} {_pg_type(df.schema[c])}" for c in cols)
    pk_sql = f", PRIMARY KEY ({', '.join(qi(c) for c in pk)})" if pk else ""
    buf = io.BytesIO()
    df.write_csv(buf, include_header=False)
    collist = ", ".join(qi(c) for c in cols)
    with connect() as conn, conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {qi(table)}")
        cur.execute(f"CREATE TABLE {qi(table)} ({coldefs}{pk_sql})")
        with cur.copy(f"COPY {qi(table)} ({collist}) FROM STDIN WITH (FORMAT CSV)") as cp:
            cp.write(buf.getvalue())
        conn.commit()
