"""A season's wildest games, ranked by total win-probability swing."""

import polars as pl
import streamlit as st

from presnap import appdata

st.title("Wildest games — by win-probability swing")
st.caption(
    "Swing = total absolute change in the home team's win probability across the "
    "game. Big swings are comebacks and collapses."
)


@st.cache_data
def plays():
    return appdata.load_plays()


df = plays()
seasons = sorted(df["season"].unique().to_list(), reverse=True)
season = st.selectbox("Season", seasons)

d = df.filter(pl.col("season") == season).sort("game_id", "play_id")
d = d.with_columns(pl.col("home_wp").diff().abs().over("game_id").alias("dwp"))
swing = (
    d.group_by("game_id", "home_team", "away_team")
    .agg(
        pl.col("dwp").sum().alias("swing"),
        pl.col("result").first().alias("result"),
    )
    .sort("swing", descending=True)
    .with_columns(pl.col("swing").round(2))
    .head(20)
    .rename({
        "game_id": "Game", "home_team": "Home", "away_team": "Away",
        "swing": "WP swing", "result": "Home margin",
    })
)
st.dataframe(swing.to_pandas(), use_container_width=True, hide_index=True)
