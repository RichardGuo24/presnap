"""Pick a game, watch the home team's win-probability curve."""

import altair as alt
import pandas as pd
import streamlit as st

from presnap import db, inference
from presnap.schema import PRESNAP_VIEW, RAW_TABLE, q

st.title("Game win-probability curve")


@st.cache_data
def list_seasons():
    return db.read_sql(
        f"SELECT DISTINCT left(game_id, 4) AS s FROM {q(RAW_TABLE)} ORDER BY s DESC"
    )["s"].to_list()


@st.cache_data
def list_games(season):
    return db.read_sql(
        f"SELECT DISTINCT game_id FROM {q(RAW_TABLE)} WHERE game_id LIKE %s ORDER BY game_id",
        (season + "%",),
    )["game_id"].to_list()


@st.cache_data
def game_data(game_id):
    df = db.read_sql(
        f"SELECT * FROM {q(PRESNAP_VIEW)} WHERE game_id = %s AND down IS NOT NULL "
        "ORDER BY play_id",
        (game_id,),
    )
    meta = db.read_sql(
        f"SELECT DISTINCT home_team, away_team, result FROM {q(RAW_TABLE)} WHERE game_id = %s",
        (game_id,),
    ).row(0, named=True)
    frame = pd.DataFrame({
        "elapsed_min": (3600 - df["game_seconds_remaining"].to_numpy()) / 60.0,
        "home_wp": inference.home_wp(df),
        "qtr": df["qtr"].to_list(),
        "posteam": df["posteam"].to_list(),
        "margin_off": df["score_differential"].to_list(),
        "down": df["down"].to_list(),
        "ydstogo": df["ydstogo"].to_list(),
    })
    return frame, meta


season = st.selectbox("Season", list_seasons())
game_id = st.selectbox("Game", list_games(season))
frame, meta = game_data(game_id)
home, away, result = meta["home_team"], meta["away_team"], int(meta["result"])

if result > 0:
    outcome = f"**{home}** won by {result}"
elif result < 0:
    outcome = f"**{away}** won by {-result}"
else:
    outcome = "Tie"
st.markdown(f"{away} @ {home} — final: {outcome}")

rule = (
    alt.Chart(pd.DataFrame({"y": [0.5]}))
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
            alt.Tooltip("margin_off:Q", title="Margin (offense)"),
            alt.Tooltip("down:Q", title="Down"),
            alt.Tooltip("ydstogo:Q", title="To go"),
            alt.Tooltip("home_wp:Q", title=f"{home} WP", format=".0%"),
        ],
    )
)
st.altair_chart(rule + line, use_container_width=True)
st.caption("Curve is the home team's win probability. Hover for the situation at each snap.")
