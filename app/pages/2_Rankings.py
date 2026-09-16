"""A season's wildest games, ranked by total win-probability swing."""

import polars as pl
import streamlit as st

from presnap import db, inference
from presnap.schema import PRESNAP_VIEW, RAW_TABLE, q

st.title("Wildest games — by win-probability swing")
st.caption(
    "Swing = total absolute change in the home team's win probability across the "
    "game. Big swings are comebacks and collapses."
)


@st.cache_data
def list_seasons():
    return db.read_sql(
        f"SELECT DISTINCT left(game_id, 4) AS s FROM {q(RAW_TABLE)} ORDER BY s DESC"
    )["s"].to_list()


@st.cache_data
def rankings(season):
    df = db.read_sql(
        f"SELECT * FROM {q(PRESNAP_VIEW)} WHERE game_id LIKE %s AND down IS NOT NULL "
        "ORDER BY game_id, play_id",
        (season + "%",),
    )
    hwp = inference.home_wp(df)
    d = (
        df.select("game_id", "home_team", "away_team")
        .with_columns(pl.Series("hwp", hwp))
        .with_columns(pl.col("hwp").diff().abs().over("game_id").alias("dwp"))
    )
    swing = (
        d.group_by("game_id", "home_team", "away_team")
        .agg(pl.col("dwp").sum().alias("swing"))
        .sort("swing", descending=True)
    )
    res = db.read_sql(
        f"SELECT DISTINCT game_id, result FROM {q(RAW_TABLE)} WHERE game_id LIKE %s",
        (season + "%",),
    )
    return (
        swing.join(res, on="game_id", how="left")
        .with_columns(pl.col("swing").round(2))
        .head(20)
    )


season = st.selectbox("Season", list_seasons())
tbl = rankings(season).rename({
    "game_id": "Game", "home_team": "Home", "away_team": "Away",
    "swing": "WP swing", "result": "Home margin",
})
st.dataframe(tbl.to_pandas(), use_container_width=True, hide_index=True)
