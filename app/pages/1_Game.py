"""Pick a game, watch the home team's win-probability curve vs. the Vegas line."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root -> import presnap

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
import streamlit as st  # noqa: E402

from presnap import appdata, webui  # noqa: E402

MODEL_GREEN = "#1A7A4C"
MARKET_AMBER = "#E0902A"
GRID_GRAY = "#9AA0A6"

webui.inject()
st.title("Game win-probability curve")


# Cache-bust key computed here (page scripts always reload; imported modules may
# not on Streamlit Cloud hot reloads), so a changed snapshot forces a reload.
_SNAP = Path(__file__).resolve().parents[1] / "snapshot" / "wp_plays.parquet"


@st.cache_data
def plays(sig):
    return appdata.load_plays()


_s = _SNAP.stat()
df = plays((_s.st_size, int(_s.st_mtime)))
seasons = sorted(df["season"].unique().to_list(), reverse=True)
season = st.selectbox("Season", seasons)

# Readable, type-to-search labels instead of codes like "2023_08_NYJ_NYG".
meta = (
    df.filter(pl.col("season") == season)
    .select("game_id", "week", "home_team", "away_team")
    .unique()
    .sort("week", "home_team")
    .to_dicts()
)
options = {f"Wk {r['week']} · {r['away_team']} @ {r['home_team']}": r["game_id"] for r in meta}
label = st.selectbox("Game (type a team to search)", list(options))
game_id = options[label]

g = df.filter(pl.col("game_id") == game_id).sort("play_id")
home, away = g["home_team"][0], g["away_team"][0]
result, market = int(g["result"][0]), float(g["market_wp"][0])

winner = (f"**{home}** won by {result}" if result > 0
          else f"**{away}** won by {-result}" if result < 0 else "Tie")
c1, c2 = st.columns(2)
c1.markdown(f"**Final:** {winner}")
c2.markdown(f"**Vegas pregame:** {home} {market:.0%} to win")

frame = g.select("elapsed_min", "home_wp", "qtr", "posteam",
                 "score_diff", "down", "ydstogo").to_pandas()

model = (
    alt.Chart(frame)
    .mark_line(color=MODEL_GREEN, interpolate="step-after")
    .encode(
        x=alt.X("elapsed_min:Q", title="Game minutes elapsed", scale=alt.Scale(domain=[0, 60])),
        y=alt.Y("home_wp:Q", title=f"{home} win probability", scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%")),
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
even = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(
    strokeDash=[3, 3], color=GRID_GRAY).encode(y="y:Q")
vegas = alt.Chart(pd.DataFrame({"y": [market]})).mark_rule(
    strokeDash=[6, 3], color=MARKET_AMBER, size=2).encode(y="y:Q")

st.altair_chart(even + vegas + model, use_container_width=True)
st.markdown(
    f"<span style='color:{MODEL_GREEN}'>●</span> Model (live)  &nbsp;&nbsp; "
    f"<span style='color:{MARKET_AMBER}'>▬</span> Vegas pregame  &nbsp;&nbsp; "
    f"<span style='color:{GRID_GRAY}'>┈</span> 50/50",
    unsafe_allow_html=True,
)

with st.expander("How do I read this?"):
    st.markdown(
        f"""
- The **blue line** is the model's estimate of the **{home}** (home team) win
  probability *before each snap*. It rises when {home} does well and dives on
  turnovers or big plays.
- The **amber line** is what **Vegas expected before kickoff** (from the betting
  spread). When the blue line climbs above it, the home team is doing *better*
  than the market predicted; below it, worse.
- **50/50** is a pure toss-up. The line ends near 100% or 0% because by the final
  snap the outcome is basically decided.
- Hover any point for the exact situation (down, distance, score, possession).
"""
    )
