"""PreSnap frontend — landing page.

    streamlit run app/Home.py

Reads a committed win-probability snapshot (app/snapshot/wp_plays.parquet), so it
needs no database or model to run. Set PRESNAP_APP_SOURCE=db to recompute live.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root -> import presnap

import streamlit as st  # noqa: E402

from presnap import webui  # noqa: E402

st.set_page_config(page_title="PreSnap", page_icon="🏈", layout="wide")
webui.inject()

st.title("🏈 PreSnap")
st.subheader("What were the odds — before every snap?")

st.markdown(
    """
For any moment in an NFL game, PreSnap estimates **each team's chance of winning**
using only what's known *before the ball is snapped* — the score, time left, down,
field position, and the betting line. Never what the play actually did.

**Pick a tab on the left:**
- **Game** — watch a game's win-probability curve rise and crash, play by play.
- **Rankings** — a season's wildest, most back-and-forth games.
- **Explained** — a 2-minute, plain-English tour of every term and the math behind it.
"""
)

c1, c2, c3 = st.columns(3)
c1.metric("Seasons", "2010–2024")
c2.metric("Plays modeled", "605,941")
c3.metric("Accuracy (Brier)", "0.151",
          help="Average squared error of the predicted probabilities. 0 is perfect, "
               "0.25 is a coin flip. Measured on 2023–24 games the model never trained on.")

st.caption("New here? The **Explained** tab defines win probability, Brier score, "
           "calibration, and “WP swing” with simple visuals.")
