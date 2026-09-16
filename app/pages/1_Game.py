"""Pick a game, watch the home team's win-probability curve."""

import altair as alt
import polars as pl
import streamlit as st

from presnap import appdata

st.title("Game win-probability curve")


@st.cache_data
def plays():
    return appdata.load_plays()


df = plays()
seasons = sorted(df["season"].unique().to_list(), reverse=True)
season = st.selectbox("Season", seasons)
games = df.filter(pl.col("season") == season)["game_id"].unique().sort().to_list()
game_id = st.selectbox("Game", games)

g = df.filter(pl.col("game_id") == game_id).sort("play_id")
home, away, result = g["home_team"][0], g["away_team"][0], int(g["result"][0])

if result > 0:
    outcome = f"**{home}** won by {result}"
elif result < 0:
    outcome = f"**{away}** won by {-result}"
else:
    outcome = "Tie"
st.markdown(f"{away} @ {home} — final: {outcome}")

frame = g.select(
    "elapsed_min", "home_wp", "qtr", "posteam", "score_diff", "down", "ydstogo"
).to_pandas()

rule = (
    alt.Chart(frame.iloc[:1])
    .transform_calculate(y="0.5")
    .mark_rule(strokeDash=[4, 4], color="#8A8A8A")
    .encode(y="y:Q")
)
line = (
    alt.Chart(frame)
    .mark_line(color="#0072B2", interpolate="step-after")
    .encode(
        x=alt.X("elapsed_min:Q", title="Game minutes elapsed",
                scale=alt.Scale(domain=[0, 60])),
        y=alt.Y("home_wp:Q", title=f"{home} win probability",
                scale=alt.Scale(domain=[0, 1])),
        tooltip=[
            alt.Tooltip("qtr:Q", title="Quarter"),
            alt.Tooltip("posteam:N", title="Possession"),
            alt.Tooltip("score_diff:Q", title="Margin (offense)"),
            alt.Tooltip("down:Q", title="Down"),
            alt.Tooltip("ydstogo:Q", title="To go"),
            alt.Tooltip("home_wp:Q", title=f"{home} WP", format=".0%"),
        ],
    )
)
st.altair_chart(rule + line, use_container_width=True)
st.caption("Curve is the home team's win probability. Hover for the situation at each snap.")
