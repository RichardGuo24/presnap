"""A season's wildest games, ranked by total win-probability swing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root -> import presnap

import polars as pl  # noqa: E402
import streamlit as st  # noqa: E402

from presnap import appdata  # noqa: E402

st.title("Wildest games of the season")
st.caption("Ranked by total win-probability swing — the sum of every up-and-down "
           "move in the home team's win probability. Big swing = a chaotic, "
           "back-and-forth game; small swing = a wire-to-wire result.")


@st.cache_data
def plays():
    return appdata.load_plays()


df = plays()
seasons = sorted(df["season"].unique().to_list(), reverse=True)
season = st.selectbox("Season", seasons)

d = df.filter(pl.col("season") == season).sort("game_id", "play_id")
d = d.with_columns(pl.col("home_wp").diff().abs().over("game_id").alias("dwp"))
swing = (
    d.group_by("game_id", "week", "home_team", "away_team")
    .agg(
        pl.col("dwp").sum().alias("swing"),
        pl.col("result").first().alias("result"),
    )
    .sort("swing", descending=True)
    .head(20)
    .with_columns(
        (pl.lit("Wk ") + pl.col("week").cast(pl.Utf8) + pl.lit(" · ")
         + pl.col("away_team") + pl.lit(" @ ") + pl.col("home_team")).alias("Game"),
        pl.col("swing").round(2).alias("WP swing"),
        pl.col("result").alias("Home margin"),
    )
    .select("Game", "WP swing", "Home margin")
)
st.dataframe(swing.to_pandas(), use_container_width=True, hide_index=True)

with st.expander("What is “WP swing”?"):
    st.markdown(
        "Imagine the win-probability line jumping around during a game. **WP swing "
        "adds up every jump** (in absolute terms). A comeback where a team goes from "
        "10% → 90% racks up a huge swing; a blowout that sits at 95% the whole way "
        "barely moves. So this list surfaces the season's most dramatic finishes — "
        "open one in the **Game** tab to see its curve."
    )
